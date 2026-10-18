"""Moment corrections of the coarse estimate.

An omitted block with mean key ``kbar``, mean value ``vbar`` and ``n`` tokens enters the
softmax with weight ``exp(s q.kbar + log n)``.  Two corrections refine it:

* mass: add ``s^2 q^T C q / 2`` to the logit, with the key covariance ``C`` of the
  block, evaluated diagonally in the shared covariance basis;
* value: add ``s C_vk q``, with one value-key covariance pooled over the blocks.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch


@contextlib.contextmanager
def eigh_on_cusolver():
    """Run ``eigh`` on cuSOLVER."""
    if not torch.cuda.is_available():
        yield
        return
    previous = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library("cusolver")
    try:
        yield
    finally:
        torch.backends.cuda.preferred_linalg_library(previous)


@dataclass(frozen=True)
class ClosureBank:
    """Moment statistics of one call."""

    basis: torch.Tensor  # [banks, d, d], orthonormal columns
    key_variance: torch.Tensor  # [banks, blocks, d], diag(U^T C_b U) per block
    value_operator: torch.Tensor  # [banks, dv, dk], pooled value-key covariance


@torch.no_grad()
def pooled_within_block_scatter(
    key: torch.Tensor,
    key_block_mean: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    block_size: int,
    n_video: int,
    chunk_blocks: int = 64,
) -> torch.Tensor:
    """Pooled within-block key covariance per bank."""
    banks, _, dim = key.shape
    n_blocks = (n_video + block_size - 1) // block_size
    device = key.device
    counts = block_counts.to(device=device, dtype=torch.float32)
    scatter = torch.zeros((banks, dim, dim), dtype=torch.float32, device=device)
    for start in range(0, n_blocks, chunk_blocks):
        stop = min(n_blocks, start + chunk_blocks)
        lo = start * block_size
        hi = min(stop * block_size, n_video)
        span = stop - start
        chunk = key[:, lo:hi].float()
        pad = span * block_size - (hi - lo)
        if pad:
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad))
        grouped = chunk.reshape(banks, span, block_size, dim)
        centred = grouped - key_block_mean[:, start:stop].float().unsqueeze(2)
        if pad:
            centred[:, -1, block_size - pad :] = 0.0
        flat = centred.reshape(banks, span * block_size, dim)
        scatter += torch.einsum("btd,bte->bde", flat, flat)
    return scatter / counts.sum()


def closure_basis(scatter: torch.Tensor) -> torch.Tensor:
    """Orthonormal eigenbasis of ``scatter``."""
    symmetric = 0.5 * (scatter + scatter.transpose(-1, -2))
    with eigh_on_cusolver():
        _, evecs = torch.linalg.eigh(symmetric.float())
    return evecs


def rotated_query_square(query: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """``(U^T q)^2`` per token."""
    return torch.bmm(query, basis.to(query.dtype)).square()


def value_drift(
    query: torch.Tensor, operator: torch.Tensor, *, scale: float
) -> torch.Tensor:
    """``s C_vk q`` per token."""
    return float(scale) * torch.bmm(query, operator.to(query.dtype).transpose(-1, -2))


def fold_rank(variance: torch.Tensor, rank: int) -> torch.Tensor:
    """Keep the leading ``rank`` variances per block and average the rest."""
    dim = variance.shape[-1]
    if rank <= 0 or rank >= dim:
        return variance
    head = variance[..., :rank]
    residual = variance[..., rank:].sum(-1, keepdim=True) / float(dim - rank)
    tail = residual.expand(*variance.shape[:-1], dim - rank)
    return torch.cat([head, tail], dim=-1).contiguous()
