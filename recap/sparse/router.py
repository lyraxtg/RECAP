"""Block selection: each query block attends exactly to the ``k`` key blocks with the
highest block-mean scores.  Selections are made at dense refreshes only.
"""

from __future__ import annotations

import torch


def block_proxy_mask(
    query_blocks: torch.Tensor,
    key_blocks: torch.Tensor,
    keep_blocks: int,
    scale: float,
) -> torch.Tensor:
    """Top-``keep_blocks`` key blocks per query block by block-mean dot product."""
    scores = torch.matmul(query_blocks.float(), key_blocks.float().transpose(-1, -2))
    scores.mul_(float(scale))
    k = max(1, min(int(keep_blocks), key_blocks.shape[-2]))
    indices = torch.topk(scores, k=k, dim=-1, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask
