"""Token-to-block means."""

from __future__ import annotations

import torch

from .types import BlockSummary


def pool_blocks(
    tokens: torch.Tensor, block_size: int, n_video_tokens: int | None = None
) -> BlockSummary:
    """Mean of each block over the video tokens."""
    if tokens.ndim < 2:
        raise ValueError("tokens must have at least sequence and feature dimensions")
    n = tokens.shape[-2] if n_video_tokens is None else int(n_video_tokens)
    if not 0 < n <= tokens.shape[-2]:
        raise ValueError(f"invalid n_video_tokens={n}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    x = tokens[..., :n, :]
    n_blocks = (n + block_size - 1) // block_size
    pad = n_blocks * block_size - n
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    prefix = x.shape[:-2]
    x = x.reshape(*prefix, n_blocks, block_size, x.shape[-1])
    counts = torch.full((n_blocks,), block_size, dtype=torch.float32, device=x.device)
    if pad:
        counts[-1] = block_size - pad
    mean = x.sum(dim=-2, dtype=torch.float32) / counts.view(
        *([1] * len(prefix)), n_blocks, 1
    )
    return BlockSummary(mean=mean, counts=counts)
