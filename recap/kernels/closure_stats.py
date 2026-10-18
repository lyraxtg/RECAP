"""Triton kernels for the moment statistics of the coarse tail.

``rotated_leaf_moments_triton``        per-block key mean and variance in the shared
                                      covariance basis.
``pooled_cross_centered_triton_fp32``  pooled value-key covariance ``C_vk``.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - CPU importability
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _batched_centered_cross_fp32_kernel(
        value_ptr,
        key_ptr,
        mean_ptr,
        output_ptr,
        n_tokens: tl.constexpr,
        d_head: tl.constexpr,
        block_size: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vt: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_mb: tl.constexpr,
        stride_mn: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Accumulate ``V^T (K - Kbar_block)``."""
        bank = tl.program_id(0)
        tile_m = tl.program_id(1)
        tile_n = tl.program_id(2)
        offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for start in range(0, n_tokens, BLOCK_K):
            tokens = start + offs_k
            valid_k = tokens < n_tokens
            blocks = tokens // block_size
            left = tl.load(
                value_ptr
                + bank * stride_vb
                + tokens[None, :] * stride_vt
                + offs_m[:, None],
                mask=(offs_m[:, None] < d_head) & valid_k[None, :],
                other=0.0,
            )
            key = tl.load(
                key_ptr
                + bank * stride_kb
                + tokens[:, None] * stride_kt
                + offs_n[None, :],
                mask=valid_k[:, None] & (offs_n[None, :] < d_head),
                other=0.0,
            )
            mean = tl.load(
                mean_ptr
                + bank * stride_mb
                + blocks[:, None] * stride_mn
                + offs_n[None, :],
                mask=valid_k[:, None] & (offs_n[None, :] < d_head),
                other=0.0,
            )
            centered = key.to(tl.float32) - mean
            centered_hi = centered.to(tl.bfloat16)
            centered_lo = (
                centered - centered_hi.to(tl.float32)
            ).to(tl.bfloat16)
            acc += tl.dot(left, centered_hi)
            acc += tl.dot(left, centered_lo)
        offsets = bank * d_head * d_head + offs_m[:, None] * d_head + offs_n[None, :]
        tl.store(
            output_ptr + offsets,
            acc,
            mask=(offs_m[:, None] < d_head) & (offs_n[None, :] < d_head),
        )

    @triton.jit
    def _rotated_leaf_variance_kernel(
        key_ptr,
        basis_ptr,
        count_ptr,
        mean_out_ptr,
        variance_ptr,
        n_video: tl.constexpr,
        n_blocks: tl.constexpr,
        d_head: tl.constexpr,
        block_size: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_bb: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vn: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        bank = tl.program_id(0)
        block = tl.program_id(1)
        out_tile = tl.program_id(2)
        offs_o = out_tile * BLOCK_O + tl.arange(0, BLOCK_O)
        offs_d = tl.arange(0, BLOCK_D)
        valid_o = offs_o < d_head
        valid_d = offs_d < d_head
        count = tl.load(count_ptr + block).to(tl.int32)
        basis = tl.load(
            basis_ptr
            + bank * stride_bb
            + offs_d[:, None] * stride_bn
            + offs_o[None, :],
            mask=valid_d[:, None] & valid_o[None, :],
            other=0.0,
        )
        # fp32 basis as a sum of two bf16 terms
        basis_hi = basis.to(tl.bfloat16)
        basis_lo = (basis - basis_hi.to(tl.float32)).to(tl.bfloat16)
        sum_value = tl.zeros((BLOCK_O,), tl.float32)
        for start in range(0, block_size, BLOCK_T):
            offs_t = start + tl.arange(0, BLOCK_T)
            token = block * block_size + offs_t
            valid_t = (offs_t < count) & (token < n_video)
            key = tl.load(
                key_ptr
                + bank * stride_kb
                + token[:, None] * stride_kt
                + offs_d[None, :],
                mask=valid_t[:, None] & valid_d[None, :],
                other=0.0,
            )
            spun = tl.dot(key, basis_hi)
            spun += tl.dot(key, basis_lo)
            sum_value += tl.sum(spun.to(tl.float32), axis=0)
        rotated_mean = sum_value / count.to(tl.float32)
        sum_sq = tl.zeros((BLOCK_O,), tl.float32)
        # second pass: variance around the block mean
        for start in range(0, block_size, BLOCK_T):
            offs_t = start + tl.arange(0, BLOCK_T)
            token = block * block_size + offs_t
            valid_t = (offs_t < count) & (token < n_video)
            key = tl.load(
                key_ptr
                + bank * stride_kb
                + token[:, None] * stride_kt
                + offs_d[None, :],
                mask=valid_t[:, None] & valid_d[None, :],
                other=0.0,
            )
            spun = tl.dot(key, basis_hi)
            spun += tl.dot(key, basis_lo)
            spun = tl.where(
                valid_t[:, None], spun - rotated_mean[None, :], 0.0
            )
            sum_sq += tl.sum(spun.to(tl.float32) * spun.to(tl.float32), axis=0)
        variance = sum_sq / count.to(tl.float32)
        tl.store(
            mean_out_ptr
            + bank * stride_vb
            + block * stride_vn
            + offs_o,
            rotated_mean,
            mask=valid_o,
        )
        tl.store(
            variance_ptr
            + bank * stride_vb
            + block * stride_vn
            + offs_o,
            variance,
            mask=valid_o,
        )


@torch.no_grad()
def rotated_leaf_moments_triton(
    key: torch.Tensor,
    key_block_mean: torch.Tensor,
    basis: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    n_video: int,
    block_size: int,
    block_t: int = 32,
    block_o: int = 16,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block key mean and variance in ``basis``."""
    if not TRITON_AVAILABLE or not key.is_cuda:
        raise RuntimeError("Triton CUDA is required")
    banks, _, dim = key.shape
    n_blocks = math.ceil(n_video / block_size)
    low_basis = basis.float().contiguous()
    mean = torch.empty(
        banks, n_blocks, dim, dtype=torch.float32, device=key.device
    )
    variance = torch.empty(
        banks, n_blocks, dim, dtype=torch.float32, device=key.device
    )
    block_d = triton.next_power_of_2(dim)
    _rotated_leaf_variance_kernel[
        (banks, n_blocks, triton.cdiv(dim, block_o))
    ](
        key,
        low_basis,
        block_counts.contiguous(),
        mean,
        variance,
        n_video=n_video,
        n_blocks=n_blocks,
        d_head=dim,
        block_size=block_size,
        stride_kb=key.stride(0),
        stride_kt=key.stride(1),
        stride_bb=low_basis.stride(0),
        stride_bn=low_basis.stride(1),
        stride_vb=variance.stride(0),
        stride_vn=variance.stride(1),
        BLOCK_T=block_t,
        BLOCK_O=block_o,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=2,
    )
    return mean, variance.clamp_min_(0.0)


@torch.no_grad()
def pooled_cross_centered_triton_fp32(
    key: torch.Tensor,
    value: torch.Tensor,
    key_block_mean: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    n_video: int,
    block_size: int,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Pooled value-key covariance ``C_vk`` per bank."""
    if not TRITON_AVAILABLE or not key.is_cuda:
        raise RuntimeError("Triton CUDA is required")
    banks, _, dim = key.shape
    key_video = key[:, :n_video].contiguous()
    value_video = value[:, :n_video].contiguous()
    mean = key_block_mean.float().contiguous()
    total = torch.empty(banks, dim, dim, dtype=torch.float32, device=key.device)
    _batched_centered_cross_fp32_kernel[
        (banks, triton.cdiv(dim, block_m), triton.cdiv(dim, block_n))
    ](
        value_video,
        key_video,
        mean,
        total,
        n_tokens=n_video,
        d_head=dim,
        block_size=block_size,
        stride_vb=value_video.stride(0),
        stride_vt=value_video.stride(1),
        stride_kb=key_video.stride(0),
        stride_kt=key_video.stride(1),
        stride_mb=mean.stride(0),
        stride_mn=mean.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return total / block_counts.float().sum()

