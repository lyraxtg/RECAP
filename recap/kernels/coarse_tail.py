"""Triton kernel for the corrected coarse tail.

Each omitted key block ``b`` enters the softmax with the logit

```
s q.kbar_b + log n_b + [ s^2 (U^T q)^2 . diag(U^T C_b U) / 2 ]_gated
```

The variance correction is dropped where the block's logit spread exceeds ``theta``.  The
value correction ``s C_vk q`` does not depend on the block and is added after the
softmax, weighted by the share of mass whose correction passed the gate.
"""

from __future__ import annotations

import math

import torch

from ..sparse.types import AttentionBranch, CoarseTailPlan

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - exercised by CPU-only environments
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _coarse_tail_kernel(
        q_ptr,
        qsq_ptr,
        k_ptr,
        w_ptr,
        v_ptr,
        count_ptr,
        block_ptr,
        valid_ptr,
        out_ptr,
        lse_ptr,
        gate_ptr,
        n_rows: tl.constexpr,
        n_groups: tl.constexpr,
        n_qblocks: tl.constexpr,
        d_head: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qm: tl.constexpr,
        stride_sh: tl.constexpr,
        stride_sm: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_wh: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_bh: tl.constexpr,
        stride_bq: tl.constexpr,
        stride_bg: tl.constexpr,
        stride_vah: tl.constexpr,
        stride_vaq: tl.constexpr,
        stride_vag: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_om: tl.constexpr,
        scale: tl.constexpr,
        half_scale_sq: tl.constexpr,
        scale_sq: tl.constexpr,
        theta_sq: tl.constexpr,
        query_block_size: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        row_mask = offs_m < n_rows
        dim_mask = offs_d < d_head
        q_raw = tl.load(
            q_ptr + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :],
            mask=row_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        q_sq = tl.load(
            qsq_ptr + pid_h * stride_sh + offs_m[:, None] * stride_sm + offs_d[None, :],
            mask=row_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        qblock = (pid_m * BLOCK_M) // query_block_size
        qblock = tl.minimum(qblock, n_qblocks - 1)
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        g_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        for start_n in range(0, n_groups, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            group_mask = offs_n < n_groups
            valid = tl.load(
                valid_ptr
                + pid_h * stride_vah
                + qblock * stride_vaq
                + offs_n * stride_vag,
                mask=group_mask,
                other=0,
            ).to(tl.int1)
            block_ids = tl.load(
                block_ptr
                + pid_h * stride_bh
                + qblock * stride_bq
                + offs_n * stride_bg,
                mask=group_mask,
                other=0,
            )
            k = tl.load(
                k_ptr
                + pid_h * stride_kh
                + block_ids[:, None] * stride_kn
                + offs_d[None, :],
                mask=valid[:, None] & dim_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q_raw, tl.trans(k)) * scale
            counts = tl.load(count_ptr + block_ids, mask=valid, other=1.0)
            scores += tl.log(counts)[None, :]
            w = tl.load(
                w_ptr
                + pid_h * stride_wh
                + block_ids[:, None] * stride_wn
                + offs_d[None, :],
                mask=valid[:, None] & dim_mask[None, :],
                other=0.0,
            )
            quadratic = tl.dot(q_sq, tl.trans(w))
            gate = quadratic * scale_sq <= theta_sq
            scores += tl.where(gate, quadratic * half_scale_sq, 0.0)
            scores = tl.where(
                row_mask[:, None] & valid[None, :],
                scores,
                -float("inf"),
            )
            m_ij = tl.max(scores, axis=1)
            tile_valid = m_ij != -float("inf")
            m_new = tl.where(tile_valid, tl.maximum(m_i, m_ij), m_i)
            alpha = tl.where(tile_valid, tl.exp(m_i - m_new), 1.0)
            shifted = tl.where(
                scores == -float("inf"),
                -float("inf"),
                scores - m_new[:, None],
            )
            probability = tl.exp(shifted)
            l_i = alpha * l_i + tl.sum(probability, axis=1)
            g_i = alpha * g_i + tl.sum(tl.where(gate, probability, 0.0), axis=1)
            v = tl.load(
                v_ptr
                + pid_h * stride_vh
                + block_ids[:, None] * stride_vn
                + offs_d[None, :],
                mask=valid[:, None] & dim_mask[None, :],
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(probability.to(v.dtype), v)
            m_i = m_new
        has_tail = l_i > 0.0
        safe_l = tl.maximum(l_i, 1.0e-30)
        result = acc / safe_l[:, None]
        tl.store(
            out_ptr + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
            tl.where(has_tail[:, None], result, 0.0),
            mask=row_mask[:, None] & dim_mask[None, :],
        )
        tl.store(
            lse_ptr + pid_h * n_rows + offs_m,
            tl.where(has_tail, m_i + tl.log(l_i), -float("inf")),
            mask=row_mask,
        )
        tl.store(
            gate_ptr + pid_h * n_rows + offs_m,
            tl.where(has_tail, g_i / safe_l, 0.0),
            mask=row_mask,
        )


def _flatten_query(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    prefix = tuple(int(value) for value in x.shape[:-2])
    return x.reshape(-1, x.shape[-2], x.shape[-1]).contiguous(), prefix


@torch.no_grad()
def coarse_tail_attention(
    query: torch.Tensor,
    plan: CoarseTailPlan,
    *,
    scale: float | None = None,
    query_block_size: int = 128,
    video_query_tokens: int | None = None,
    block_m: int = 32,
    block_n: int = 32,
) -> AttentionBranch:
    """Corrected coarse attention over the key blocks listed in ``plan``."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not importable")
    if not query.is_cuda:
        raise ValueError("the coarse tail kernel requires CUDA")
    if query.shape[:-2] != plan.key_blocks.shape[:-2]:
        raise ValueError("query and plan prefixes differ")
    if query.shape[-1] != plan.key_blocks.shape[-1]:
        raise ValueError("query/key dimensions differ")
    if plan.key_blocks.shape != plan.value_blocks.shape:
        raise ValueError("the coarse tail requires equal key/value dimensions")
    if plan.key_variance.shape != plan.key_blocks.shape:
        raise ValueError("key variance and key block shapes differ")
    if query_block_size % block_m:
        raise ValueError("BLOCK_M must divide query_block_size")
    rows_total = query.shape[-2]
    rows = rows_total if video_query_tokens is None else int(video_query_tokens)
    scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else float(scale)
    q, prefix = _flatten_query(query[..., :rows, :])
    n_blocks = plan.key_blocks.shape[-2]
    groups = plan.block_ids.shape[-1]
    qblocks = plan.block_ids.shape[-2]
    if groups == 0:
        return AttentionBranch(
            output=torch.zeros_like(query),
            lse=torch.full(
                (*query.shape[:-2], rows_total),
                -torch.inf,
                dtype=torch.float32,
                device=query.device,
            ),
        )
    dim = q.shape[-1]
    qsq = plan.query_square[..., :rows, :].reshape(-1, rows, dim).to(q.dtype).contiguous()
    k = plan.key_blocks.reshape(-1, n_blocks, dim).to(q.dtype).contiguous()
    v = plan.value_blocks.reshape(-1, n_blocks, dim).to(q.dtype).contiguous()
    w = plan.key_variance.reshape(-1, n_blocks, dim).to(q.dtype).contiguous()
    counts = plan.block_counts.float().contiguous()
    block_ids = plan.block_ids.reshape(-1, qblocks, groups).contiguous()
    valid = plan.valid.reshape(-1, qblocks, groups).contiguous()
    bh = q.shape[0]
    out = torch.empty((bh, rows, dim), dtype=q.dtype, device=q.device)
    lse = torch.empty((bh, rows), dtype=torch.float32, device=q.device)
    gate_fraction = torch.empty((bh, rows), dtype=torch.float32, device=q.device)
    block_d = triton.next_power_of_2(dim)
    grid = (triton.cdiv(rows, block_m), bh)
    theta = float(plan.gate_theta)
    _coarse_tail_kernel[grid](
        q,
        qsq,
        k,
        w,
        v,
        counts,
        block_ids,
        valid,
        out,
        lse,
        gate_fraction,
        n_rows=rows,
        n_groups=groups,
        n_qblocks=qblocks,
        d_head=dim,
        stride_qh=q.stride(0),
        stride_qm=q.stride(1),
        stride_sh=qsq.stride(0),
        stride_sm=qsq.stride(1),
        stride_kh=k.stride(0),
        stride_kn=k.stride(1),
        stride_wh=w.stride(0),
        stride_wn=w.stride(1),
        stride_vh=v.stride(0),
        stride_vn=v.stride(1),
        stride_bh=block_ids.stride(0),
        stride_bq=block_ids.stride(1),
        stride_bg=block_ids.stride(2),
        stride_vah=valid.stride(0),
        stride_vaq=valid.stride(1),
        stride_vag=valid.stride(2),
        stride_oh=out.stride(0),
        stride_om=out.stride(1),
        scale=scale,
        half_scale_sq=0.5 * scale * scale,
        scale_sq=scale * scale,
        theta_sq=theta * theta,
        query_block_size=query_block_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4 if block_d <= 128 else 8,
        num_stages=3,
    )
    drift = plan.value_drift[..., :rows, :].reshape(-1, rows, dim).to(q.dtype)
    out = out + gate_fraction[..., None].to(q.dtype) * drift
    if rows < rows_total:
        full_out = torch.zeros(
            (bh, rows_total, dim), dtype=query.dtype, device=query.device
        )
        full_lse = torch.full(
            (bh, rows_total), -torch.inf, dtype=torch.float32, device=query.device
        )
        full_out[:, :rows] = out
        full_lse[:, :rows] = lse
        out, lse = full_out, full_lse
    return AttentionBranch(
        output=out.reshape(*prefix, rows_total, dim),
        lse=lse.reshape(*prefix, rows_total),
    )
