"""Exact attention over the selected key blocks (``block_sparse_attn`` extension)."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch

from ..sparse.types import AttentionBranch

_EXTRA_PATH = os.environ.get("RECAP_BLOCK_SPARSE_ATTN_PATH")


def block_sparse_available() -> bool:
    """Whether ``block_sparse_attn`` can be imported."""
    try:
        if _EXTRA_PATH and _EXTRA_PATH not in sys.path:
            sys.path.insert(0, str(Path(_EXTRA_PATH)))
        from block_sparse_attn import block_sparse_attn_func  # noqa: F401

        return True
    except Exception:
        return False


def _guard_row_blockmask() -> None:
    """Pad the kernel's converted block mask by one row of -1.

    The kernel can read one entry past the end of the mask; the padding keeps that read in
    bounds.  Outputs are unchanged.
    """
    from block_sparse_attn import block_sparse_attn_interface as interface

    convert = interface.convert_blockmask_row_reverse
    if getattr(convert, "_recap_guarded", False):
        return

    def guarded(blockmask: torch.Tensor, causal: bool = False) -> torch.Tensor:
        rows = convert(blockmask, causal)
        flat = rows.reshape(-1)
        padded = rows.new_full((flat.numel() + rows.shape[-1],), -1)
        padded[: flat.numel()] = flat
        return padded[: flat.numel()].view(rows.shape)

    guarded._recap_guarded = True
    interface.convert_blockmask_row_reverse = guarded


def build_kernel_block_mask(
    video_mask: torch.Tensor,
    *,
    sequence_tokens: int,
    video_tokens: int,
    block_size: int,
) -> torch.Tensor:
    """Video block selection on the kernel's block grid; text is always selected."""
    prefix = video_mask.shape[:-2]
    video_blocks = math.ceil(video_tokens / block_size)
    total_blocks = math.ceil(sequence_tokens / block_size)
    if video_mask.shape[-2:] != (video_blocks, video_blocks):
        raise ValueError("video mask shape does not match video token count")
    out = torch.zeros(
        (*prefix, total_blocks, total_blocks),
        dtype=torch.bool,
        device=video_mask.device,
    )
    out[..., :video_blocks, :video_blocks] = video_mask
    shared_boundary_block = video_tokens % block_size != 0 and sequence_tokens > video_tokens
    dense_start = video_blocks - 1 if shared_boundary_block else video_blocks
    out[..., dense_start:, :] = True
    out[..., :, dense_start:] = True
    return out


@torch.no_grad()
def block_sparse_exact(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_mask: torch.Tensor,
) -> AttentionBranch:
    """Exact attention over the masked blocks, with its log-sum-exp.

    Inputs are ``[1, heads, sequence, dim]``.
    """
    if not block_sparse_available():
        raise RuntimeError(
            "block_sparse_attn is unavailable; install it (or set "
            "RECAP_BLOCK_SPARSE_ATTN_PATH to its source tree)"
        )
    from block_sparse_attn import block_sparse_attn_func

    _guard_row_blockmask()
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError("block-sparse backend expects [1, heads, sequence, dim]")
    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError("Q/K/V shapes differ")
    sequence = query.shape[-2]
    heads = query.shape[1]
    if kernel_mask.ndim != 3 or kernel_mask.shape[0] != heads:
        raise ValueError("kernel mask must be [heads, query blocks, key blocks]")
    expected_blocks = math.ceil(sequence / 128)
    if kernel_mask.shape[-2:] != (expected_blocks, expected_blocks):
        raise ValueError(
            f"kernel mask grid {tuple(kernel_mask.shape[-2:])} does not match "
            f"the kernel's {expected_blocks}x{expected_blocks} 128-token grid"
        )
    cu = torch.tensor([0, sequence], dtype=torch.int32, device=query.device)
    q = query[0].permute(1, 0, 2).contiguous()
    k = key[0].permute(1, 0, 2).contiguous()
    v = value[0].permute(1, 0, 2).contiguous()
    head_mask_type = torch.ones(heads, dtype=torch.int32, device=query.device)
    blockmask = kernel_mask.unsqueeze(0).contiguous()
    output, lse, _ = block_sparse_attn_func(
        q,
        k,
        v,
        cu,
        cu,
        head_mask_type=head_mask_type,
        streaming_info=None,
        base_blockmask=blockmask,
        max_seqlen_q_=sequence,
        max_seqlen_k_=sequence,
        p_dropout=0.0,
        deterministic=False,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=True,
    )
    del q, k, v, blockmask
    return AttentionBranch(
        output=output.permute(1, 0, 2).unsqueeze(0).contiguous(),
        lse=lse[0, :, :sequence].unsqueeze(0).contiguous(),
    )
