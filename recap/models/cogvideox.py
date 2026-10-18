"""CogVideoX-1.5 integration."""

from __future__ import annotations

from typing import Any

import torch

from ..config import RecapConfig
from ..kernels.block_sparse import block_sparse_available
from ..operator import Operator
from ..sparse.attention import SparseGeometry
from ..sparse.geometry import hilbert_permutation
from ..sparse.layout import Layout


# guidance branches are batched into one call
CALLS_PER_STEP = 1


class RecapCogAttnProcessor:
    """CogVideoX self-attention with the attention call routed to the operator."""

    def __init__(self, layer_index: int, operator: Operator):
        self.layer_index = int(layer_index)
        self.operator = operator

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_length = encoder_hidden_states.size(1)
        joint = torch.cat((encoder_hidden_states, hidden_states), dim=1)
        batch, sequence, _ = joint.shape
        if attention_mask is not None:
            raise ValueError(
                "the CogVideoX integration expects the standard fixed padded text "
                "sequence without a runtime attention mask"
            )
        query = attn.to_q(joint)
        key = attn.to_k(joint)
        value = attn.to_v(joint)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch, sequence, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, sequence, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, sequence, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_length:] = apply_rotary_emb(
                query[:, :, text_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_length:] = apply_rotary_emb(
                    key[:, :, text_length:], image_rotary_emb
                )

        attended = self.operator.attend(self.layer_index, query, key, value)

        attended = attended.transpose(1, 2).reshape(
            batch, sequence, attn.heads * head_dim
        )
        attended = attn.to_out[0](attended)
        attended = attn.to_out[1](attended)
        encoder_out, video_out = attended.split(
            [text_length, sequence - text_length], dim=1
        )
        return video_out, encoder_out


def build_operator(pipe, config: RecapConfig) -> tuple[Operator, dict[str, Any]]:
    """Install one sparse state per block and return the operator."""
    if not block_sparse_available():
        raise RuntimeError(
            "the block_sparse_attn CUDA extension is required for the exact head but "
            "could not be imported; install it, or point RECAP_BLOCK_SPARSE_ATTN_PATH "
            "at its source tree"
        )
    geometry_cfg = config.geometry
    permutation = hilbert_permutation(geometry_cfg.grid)
    layout = Layout.from_permutation(
        permutation, text_tokens=geometry_cfg.text_tokens, text_first=True
    )
    geometry = SparseGeometry.build(layout, geometry_cfg.block_size)

    blocks = pipe.transformer.transformer_blocks
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
            n_banks=2 * int(transformer_config.num_attention_heads),
            dim=int(transformer_config.attention_head_dim),
        )
    for index, block in enumerate(blocks):
        block.attn1.processor = RecapCogAttnProcessor(index, operator)

    info = {
        "model": "cogvideox1.5",
        "calls_per_step": CALLS_PER_STEP,
        "layers": len(blocks),
        "video_tokens": layout.video_tokens,
        "text_tokens": layout.text_tokens,
        "key_blocks": geometry.n_blocks,
        "head_blocks": max(
            1, round(config.operator.head_density * geometry.n_blocks)
        ),
        "grid": list(geometry_cfg.grid),
    }
    return operator, info
