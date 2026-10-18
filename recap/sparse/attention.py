"""Sparse attention for one layer: block selection and the corrected coarse tail.

``head_partition``       key blocks each query block attends to exactly (top-k over
                         block-mean scores);
``coarse_tail_context``  partition-independent statistics: block means, per-block key
                         variance in the shared covariance basis, value-key covariance;
``coarse_tail_apply``    corrected coarse attention over the omitted blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from ..kernels import closure_stats
from ..kernels.coarse_tail import coarse_tail_attention
from .closure import (
    ClosureBank,
    closure_basis,
    fold_rank,
    pooled_within_block_scatter,
    rotated_query_square,
    value_drift,
)
from .layout import Layout
from .planner import build_coarse_tail_plan
from .pooling import pool_blocks
from .router import block_proxy_mask


@dataclass
class SparseGeometry:
    """Model-level data shared by all layers."""

    layout: Layout
    block_size: int
    n_blocks: int
    block_counts: torch.Tensor  # [n_blocks] fp32, tokens per video block

    @classmethod
    def build(cls, layout: Layout, block_size: int) -> "SparseGeometry":
        n_blocks = math.ceil(layout.video_tokens / block_size)
        block_counts = torch.full((n_blocks,), float(block_size), dtype=torch.float32)
        block_counts[-1] = layout.video_tokens - block_size * (n_blocks - 1)
        return cls(
            layout=layout,
            block_size=int(block_size),
            n_blocks=int(n_blocks),
            block_counts=block_counts,
        )

    def to(self, device: torch.device) -> None:
        """Move the block counts to ``device``."""
        if self.block_counts.device != device:
            self.block_counts = self.block_counts.to(device)


class SparseAttention:
    """Per-layer state: the shared covariance basis, computed once per sample."""

    def __init__(self, geometry: SparseGeometry, config):
        self.geometry = geometry
        self.config = config
        self._basis: torch.Tensor | None = None

    def reset(self) -> None:
        """Forget the basis."""
        self._basis = None

    # -- the partition -----------------------------------------------------------------

    @torch.no_grad()
    def head_partition(
        self, query: torch.Tensor, key: torch.Tensor, scale: float
    ) -> torch.Tensor:
        """Boolean ``[..., query blocks, key blocks]`` mask of the selected blocks."""
        block_size = self.geometry.block_size
        n_video = self.geometry.layout.video_tokens
        qb = pool_blocks(query, block_size, n_video)
        kb = pool_blocks(key, block_size, n_video)
        n_blocks = qb.mean.shape[-2]
        keep = max(1, round(float(self.config.head_density) * n_blocks))
        mask = block_proxy_mask(qb.mean, kb.mean, keep, scale)
        return self._match_kernel_granularity(mask)

    def _match_kernel_granularity(self, mask: torch.Tensor) -> torch.Tensor:
        """Select a trailing partial video block in full, as query and as key."""
        out = mask.clone()
        if self.geometry.layout.video_tokens % self.geometry.block_size:
            out[..., -1, :] = True
            out[..., :, -1] = True
        return out

    # -- the moment statistics ---------------------------------------------------------

    @torch.no_grad()
    def build_closure(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_block_mean: torch.Tensor,
        block_counts: torch.Tensor,
    ) -> ClosureBank:
        """Moment statistics in the shared covariance basis.

        The basis (eigenbasis of the pooled within-block key covariance) is computed at the
        first call of a sample; the variances and the value-key covariance at every call.
        """
        geometry = self.geometry
        n_video = geometry.layout.video_tokens
        block_size = geometry.block_size
        basis = self._basis
        if basis is None:
            scatter = pooled_within_block_scatter(
                key, key_block_mean, block_counts,
                block_size=block_size, n_video=n_video,
            )
            basis = closure_basis(scatter)

        _block_mean, key_variance = closure_stats.rotated_leaf_moments_triton(
            key, key_block_mean, basis, block_counts,
            n_video=n_video, block_size=block_size,
            block_t=32, block_o=32, num_warps=4,
        )
        operator = closure_stats.pooled_cross_centered_triton_fp32(
            key, value, key_block_mean, block_counts,
            n_video=n_video, block_size=block_size,
            block_m=32, block_n=32, block_k=64, num_warps=4, num_stages=3,
        )
        return ClosureBank(
            basis=basis, key_variance=key_variance.float(), value_operator=operator
        )

    @torch.no_grad()
    def _order_basis(
        self, closure: ClosureBank, block_counts: torch.Tensor
    ) -> ClosureBank:
        """Sort the basis by pooled within-block variance, descending."""
        variance = closure.key_variance.float()
        weights = block_counts.to(variance.dtype).reshape(1, -1, 1)
        pooled = (variance * weights).sum(1) / weights.sum().clamp_min(1.0)
        order = torch.argsort(pooled, dim=-1, descending=True)  # [banks, d]
        basis = torch.gather(
            closure.basis, 2, order.unsqueeze(1).expand(-1, closure.basis.shape[1], -1)
        ).contiguous()
        key_variance = torch.gather(
            closure.key_variance, 2,
            order.unsqueeze(1).expand(-1, closure.key_variance.shape[1], -1),
        ).contiguous()
        return replace(closure, basis=basis, key_variance=key_variance)

    # -- the coarse tail ---------------------------------------------------------------

    @torch.no_grad()
    def coarse_tail_context(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
    ) -> dict:
        """Statistics of the coarse tail that do not depend on the partition."""
        config = self.config
        geometry = self.geometry
        b, h, sequence, dim = query.shape
        banks = b * h
        n_video = geometry.layout.video_tokens
        block_size = geometry.block_size
        geometry.to(query.device)

        kb = pool_blocks(key, block_size, n_video)
        vb = pool_blocks(value, block_size, n_video)
        counts = kb.counts.to(query.device)

        first_call = self._basis is None
        closure = self.build_closure(
            key.reshape(banks, sequence, dim),
            value.reshape(banks, sequence, dim),
            kb.mean.reshape(banks, -1, dim),
            counts,
        )
        if first_call:
            closure = self._order_basis(closure, counts)
            self._basis = closure.basis
        rank = int(config.closure_rank)
        if rank:
            closure = replace(
                closure, key_variance=fold_rank(closure.key_variance, rank)
            )

        query_bank = query.reshape(banks, sequence, dim)
        qsq = rotated_query_square(query_bank[:, :n_video], closure.basis)
        drift = value_drift(query_bank[:, :n_video], closure.value_operator, scale=scale)
        return {
            "kb_mean": kb.mean, "vb_mean": vb.mean, "closure": closure,
            "qsq": qsq.reshape(b, h, n_video, dim),
            "drift": drift.reshape(b, h, n_video, dim),
            "banks": banks, "sequence": sequence, "dim": dim, "n_video": n_video,
            "block_size": block_size,
        }

    @torch.no_grad()
    def coarse_tail_apply(
        self,
        query: torch.Tensor,
        context: dict,
        head_mask: torch.Tensor,
        scale: float,
        rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Corrected coarse attention over the blocks ``head_mask`` omits."""
        geometry = self.geometry
        banks, sequence, dim = context["banks"], context["sequence"], context["dim"]
        plan = build_coarse_tail_plan(
            context["kb_mean"], context["vb_mean"], head_mask,
            block_counts=geometry.block_counts,
            closure=context["closure"],
            gate_theta=self.config.closure_gate_theta,
            query_square=context["qsq"],
            value_drift=context["drift"],
        )
        branch = coarse_tail_attention(
            query, plan, scale=scale,
            query_block_size=context["block_size"],
            video_query_tokens=context["n_video"],
        )
        output = branch.output.reshape(banks, sequence, dim)[:, :rows].float()
        lse = branch.lse.reshape(banks, sequence)[:, :rows].float()
        return output, lse
