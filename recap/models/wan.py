"""Wan2.1 T2V integration.

Self-attention sees video tokens only (text enters through cross-attention, which stays
dense).  The two guidance branches are separate transformer calls, handled as two
operator streams.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..config import RecapConfig
from ..kernels.block_sparse import block_sparse_available
from ..operator import Operator
from ..sparse.attention import SparseGeometry
from ..sparse.geometry import hilbert_permutation
from ..sparse.layout import Layout

# one transformer call per guidance branch
CALLS_PER_STEP = 2


class RecapWanAttnProcessor:
    """Wan self-attention with the attention call routed to the operator."""

    def __init__(self, layer_index: int, operator: Operator, sequence: int):
        self.layer_index = int(layer_index)
        self.operator = operator
        self.sequence = int(sequence)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None or getattr(attn, "add_k_proj", None) is not None:
            raise ValueError(
                "the RECAP Wan processor belongs on T2V self-attention (attn1) only"
            )
        if attention_mask is not None:
            raise ValueError("Wan self-attention expects no runtime attention mask")

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        if query.shape[-2] != self.sequence:
            raise ValueError(
                f"Wan token sequence {query.shape[-2]} != configured {self.sequence}; "
                "the geometry grid does not match this resolution / frame count"
            )

        if rotary_emb is not None:
            if isinstance(rotary_emb, tuple):
                # diffusers >= 0.35: real cos/sin
                freqs_cos, freqs_sin = rotary_emb
                cos = freqs_cos.transpose(1, 2)[..., 0::2]
                sin = freqs_sin.transpose(1, 2)[..., 1::2]

                def apply_rotary_emb(x: torch.Tensor) -> torch.Tensor:
                    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                    out = torch.empty_like(x)
                    out[..., 0::2] = x1 * cos - x2 * sin
                    out[..., 1::2] = x1 * sin + x2 * cos
                    return out
            else:
                # diffusers 0.34: complex frequencies
                def apply_rotary_emb(x: torch.Tensor) -> torch.Tensor:
                    dtype = torch.float32 if x.device.type == "mps" else torch.float64
                    rotated = torch.view_as_complex(x.to(dtype).unflatten(3, (-1, 2)))
                    return torch.view_as_real(rotated * rotary_emb).flatten(3, 4).type_as(x)

            query = apply_rotary_emb(query)
            key = apply_rotary_emb(key)

        attended = self.operator.attend(self.layer_index, query, key, value)

        attended = attended.transpose(1, 2).flatten(2, 3).type_as(query)
        attended = attn.to_out[0](attended)
        attended = attn.to_out[1](attended)
        return attended


def build_operator(pipe, config: RecapConfig) -> tuple[Operator, dict[str, Any]]:
    """Install one sparse state per (guidance branch, block) and return the operator."""
    if not block_sparse_available():
        raise RuntimeError(
            "the block_sparse_attn CUDA extension is required for the exact head but "
            "could not be imported; install it, or point RECAP_BLOCK_SPARSE_ATTN_PATH "
            "at its source tree"
        )
    geometry_cfg = config.geometry
    if geometry_cfg.text_tokens != 0:
        raise ValueError(
            "Wan self-attention is video-only; set geometry.text_tokens to 0"
        )
    permutation = hilbert_permutation(geometry_cfg.grid)
    layout = Layout.from_permutation(permutation, text_tokens=0, text_first=False)
    geometry = SparseGeometry.build(layout, geometry_cfg.block_size)

    blocks = pipe.transformer.blocks
    operator_cls = Operator
    if config.operator.host_banks:
        from ..host_banks import HostBankOperator

        operator_cls = HostBankOperator
    operator = operator_cls.build(
        config.operator, geometry, n_layers=len(blocks), n_streams=CALLS_PER_STEP
    )
    if config.operator.host_banks:
        operator.set_prefetch(config.operator.prefetch_banks)
        transformer_config = pipe.transformer.config
        operator.preallocate_host(
            n_layers=len(blocks),
            n_banks=int(transformer_config.num_attention_heads),
            dim=int(transformer_config.attention_head_dim),
        )
    for index, block in enumerate(blocks):
        block.attn1.set_processor(
            RecapWanAttnProcessor(index, operator, layout.video_tokens)
        )

    info = {
        "model": "wan2.1-1.3b",
        "calls_per_step": CALLS_PER_STEP,
        "layers": len(blocks),
        "video_tokens": layout.video_tokens,
        "text_tokens": 0,
        "key_blocks": geometry.n_blocks,
        "head_blocks": max(
            1, round(config.operator.head_density * geometry.n_blocks)
        ),
        "grid": list(geometry_cfg.grid),
    }
    return operator, info
