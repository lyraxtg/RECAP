"""Shared value types.

An ``AttentionBranch`` is an output normalized over a subset of keys plus its log-sum-exp,
so branches over disjoint key sets merge exactly into one softmax.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlockSummary:
    """Block means and token counts."""

    mean: torch.Tensor  # [..., blocks, dim], fp32
    counts: torch.Tensor  # [blocks], fp32


@dataclass
class AttentionBranch:
    output: torch.Tensor  # [..., rows, dim], normalized over this branch's keys
    lse: torch.Tensor  # [..., rows], log-sum-exp of this branch's logits


@dataclass(frozen=True)
class CoarseTailPlan:
    """Inputs of the coarse-tail kernel."""

    key_blocks: torch.Tensor  # [..., blocks, d]
    value_blocks: torch.Tensor  # [..., blocks, d]
    key_variance: torch.Tensor  # [..., blocks, d]
    block_counts: torch.Tensor  # [blocks]
    block_ids: torch.Tensor  # [..., query_blocks, groups]
    valid: torch.Tensor  # [..., query_blocks, groups]
    query_square: torch.Tensor  # [..., rows, d], (U^T q)^2
    value_drift: torch.Tensor  # [..., rows, d]
    gate_theta: float
