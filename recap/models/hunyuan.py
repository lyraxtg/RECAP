"""HunyuanVideo integration.

The native sequence is [video | text].  The prompt padding is trimmed instead of masked,
which is exact, so the text length changes per prompt and the shared layout is
retargeted for each one.  HunyuanVideo is guidance-distilled: one transformer call per
step.
"""

from __future__ import annotations

from typing import Any

import torch

from ..config import RecapConfig
from ..kernels.block_sparse import block_sparse_available
from ..operator import Operator
from ..sparse.attention import SparseGeometry
from ..sparse.geometry import hilbert_permutation
from ..sparse.layout import Layout

CALLS_PER_STEP = 1

MAX_TEXT_TOKENS = 256


# -- per-prompt text length ------------------------------------------------------------


def retarget_text_tokens(layout: Layout, text_tokens: int) -> bool:
    """Set the shared layout to this prompt's text length."""
    text_tokens = int(text_tokens)
    if text_tokens < 0:
        raise ValueError("text_tokens must be non-negative")
    if text_tokens == layout.text_tokens:
        return False
    layout.text_tokens = text_tokens
    layout._gather_cache.clear()
    return True


def prompt_text_tokens(pipe, prompt: str, max_sequence_length: int = MAX_TEXT_TOKENS) -> int:
    """Unpadded prompt length, computed with the tokenizer alone."""
    from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import (
        DEFAULT_PROMPT_TEMPLATE,
    )

    template = DEFAULT_PROMPT_TEMPLATE
    rendered = template["template"].format(prompt)
    crop_start = template.get("crop_start")
    if crop_start is None:
        probe = pipe.tokenizer(
            template["template"], padding="max_length", return_tensors="pt",
            return_length=False, return_overflowing_tokens=False,
            return_attention_mask=False,
        )
        crop_start = int(probe["input_ids"].shape[-1]) - 2
    tokenized = pipe.tokenizer(
        [rendered],
        max_length=int(max_sequence_length) + int(crop_start),
        padding="max_length", truncation=True, return_tensors="pt",
        return_length=False, return_overflowing_tokens=False,
        return_attention_mask=True,
    )
    return int(tokenized.attention_mask[:, int(crop_start):].sum().item())


# -- attention processor ---------------------------------------------------------------


class RecapHunyuanAttnProcessor:
    """HunyuanVideo self-attention (double- or single-stream) routed to the operator."""

    def __init__(self, layer_index: int, operator: Operator, layout: Layout):
        self.layer_index = int(layer_index)
        self.operator = operator
        self.layout = layout

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None:
            raise ValueError(
                "RECAP Hunyuan requires the trimmed forward: padded text keys must not "
                "reach the exact block-sparse kernel, which has no mask"
            )
        if encoder_hidden_states is None:
            raise ValueError("Hunyuan joint attention requires encoder hidden states")
        text_tokens = int(encoder_hidden_states.shape[1])
        if text_tokens != self.layout.text_tokens:
            retarget_text_tokens(self.layout, text_tokens)

        joint = attn.add_q_proj is None
        projected = (
            torch.cat([hidden_states, encoder_hidden_states], dim=1)
            if joint else hidden_states
        )
        query = attn.to_q(projected)
        key = attn.to_k(projected)
        value = attn.to_v(projected)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if joint:
                # RoPE on the video tokens only
                query = torch.cat(
                    [apply_rotary_emb(query[:, :, :-text_tokens], image_rotary_emb),
                     query[:, :, -text_tokens:]], dim=2)
                key = torch.cat(
                    [apply_rotary_emb(key[:, :, :-text_tokens], image_rotary_emb),
                     key[:, :, -text_tokens:]], dim=2)
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        if not joint:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)
            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        sequence = query.shape[-2]
        expected = self.layout.video_tokens + text_tokens
        if sequence != expected:
            raise ValueError(
                f"Hunyuan joint sequence {sequence} != video {self.layout.video_tokens} "
                f"+ text {text_tokens}; the geometry grid does not match this "
                "resolution / frame count"
            )

        attended = self.operator.attend(self.layer_index, query, key, value)

        attended = attended.transpose(1, 2).flatten(2, 3).to(query.dtype)
        video_out, encoder_out = attended.split(
            [self.layout.video_tokens, text_tokens], dim=1
        )
        if getattr(attn, "to_out", None) is not None:
            video_out = attn.to_out[0](video_out)
            video_out = attn.to_out[1](video_out)
        if getattr(attn, "to_add_out", None) is not None:
            encoder_out = attn.to_add_out(encoder_out)
        return video_out, encoder_out


# -- installation ----------------------------------------------------------------------


def install_trimmed_forward(transformer) -> None:
    """Transformer forward that trims the padded text tokens instead of masking them."""
    from diffusers import HunyuanVideoTransformer3DModel
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance=None,
        attention_kwargs=None,
        return_dict=True,
    ):
        assert hidden_states.shape[0] == 1, (
            "the trimmed HunyuanVideo forward supports batch size 1 only"
        )
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        batch_size, _, num_frames, height, width = hidden_states.shape
        patch, patch_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // patch_t
        post_patch_height = height // patch
        post_patch_width = width // patch
        first_frame_num_tokens = post_patch_height * post_patch_width
        image_rotary_emb = self.rope(hidden_states)
        temb, token_replace_emb = self.time_text_embed(
            timestep, pooled_projections, guidance
        )
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, timestep, encoder_attention_mask
        )
        effective_text_length = int(encoder_attention_mask.sum(dim=1)[0])
        encoder_hidden_states = encoder_hidden_states[:, :effective_text_length]
        attention_mask = None

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )
        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            -1,
            patch_t,
            patch,
            patch,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)

    trimmed_forward = forward.__get__(
        transformer, HunyuanVideoTransformer3DModel
    )
    if hasattr(transformer, "_hf_hook") and hasattr(transformer, "_old_forward"):
        # keep the CPU-offload hook
        transformer._old_forward = trimmed_forward
    else:
        transformer.forward = trimmed_forward


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
        permutation, text_tokens=int(geometry_cfg.text_tokens), text_first=False
    )
    geometry = SparseGeometry.build(layout, geometry_cfg.block_size)

    install_trimmed_forward(pipe.transformer)
    double = list(pipe.transformer.transformer_blocks)
    single = list(pipe.transformer.single_transformer_blocks)
    blocks = double + single
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
        block.attn.set_processor(RecapHunyuanAttnProcessor(index, operator, layout))

    info = {
        "model": "hunyuan",
        "calls_per_step": CALLS_PER_STEP,
        "layers": len(blocks),
        "video_tokens": layout.video_tokens,
        "key_blocks": geometry.n_blocks,
        "head_blocks": max(
            1, round(config.operator.head_density * geometry.n_blocks)
        ),
        "grid": list(geometry_cfg.grid),
    }
    return operator, info
