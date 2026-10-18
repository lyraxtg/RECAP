"""Input of the coarse tail: the key blocks each query block omits."""

from __future__ import annotations

import torch

from .closure import ClosureBank
from .types import CoarseTailPlan


@torch.no_grad()
def build_coarse_tail_plan(
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    block_counts: torch.Tensor,
    closure: ClosureBank,
    gate_theta: float,
    query_square: torch.Tensor,
    value_drift: torch.Tensor,
) -> CoarseTailPlan:
    """Pack the omitted key blocks of every query block into a padded list."""
    unselected = ~selected_mask
    width = int(unselected.sum(dim=-1).max().item())
    packed = torch.topk(unselected.to(torch.int8), k=max(width, 1), dim=-1, sorted=False)
    block_ids, valid = packed.indices, packed.values.bool()
    return CoarseTailPlan(
        key_blocks=key_blocks,
        value_blocks=value_blocks,
        key_variance=closure.key_variance.reshape_as(key_blocks),
        block_counts=block_counts,
        block_ids=block_ids,
        valid=valid,
        query_square=query_square,
        value_drift=value_drift,
        gate_theta=float(gate_theta),
    )
