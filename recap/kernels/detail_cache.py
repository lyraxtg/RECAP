"""Triton kernels that write and apply the attention refinement.

``refresh_detail``  stores the refinement, the exact tail minus the coarse tail, for the
                    tail output and the tail log-mass.  With an existing refinement it
                    first measures the error that refinement would have caused.
``sparse_merge``    merges the head with the coarse tail and the cached refinement.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - import failure is reported by entry points
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _refresh_detail_kernel(
        full_o,
        full_l,
        head_o,
        head_l,
        coarse_o,
        coarse_l,
        permutation,
        detail,
        detail_l,
        validation_stats,
        drift_old_gl,
        drift_center,
        rows: tl.constexpr,
        sequence: tl.constexpr,
        dim: tl.constexpr,
        video_offset: tl.constexpr,
        stride_fo_b: tl.constexpr,
        stride_fo_s: tl.constexpr,
        stride_fl_b: tl.constexpr,
        stride_fl_s: tl.constexpr,
        stride_ho_b: tl.constexpr,
        stride_ho_s: tl.constexpr,
        stride_hl_b: tl.constexpr,
        stride_hl_s: tl.constexpr,
        stride_go_b: tl.constexpr,
        stride_go_s: tl.constexpr,
        stride_gl_b: tl.constexpr,
        stride_gl_s: tl.constexpr,
        stride_d_b: tl.constexpr,
        stride_d_s: tl.constexpr,
        stride_dl_b: tl.constexpr,
        stride_dl_s: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        VALIDATE_PREVIOUS: tl.constexpr,
        DRIFT_ADMIT: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        bank = tl.program_id(1)
        rr = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dd = tl.arange(0, BLOCK_D)
        row_ok = rr < rows
        dim_ok = dd < dim
        native_video = tl.load(permutation + rr, mask=row_ok, other=0)
        native_row = video_offset + native_video

        lf = tl.load(
            full_l + bank * stride_fl_b + native_row * stride_fl_s,
            mask=row_ok,
            other=0.0,
        ).to(tl.float32)
        lh = tl.load(
            head_l + bank * stride_hl_b + rr * stride_hl_s,
            mask=row_ok,
            other=-float("inf"),
        ).to(tl.float32)
        ratio = tl.exp(tl.minimum(lh - lf, 0.0))
        ratio = tl.minimum(ratio, 1.0 - 1.0e-6)
        denominator = 1.0 - ratio

        of = tl.load(
            full_o
            + bank * stride_fo_b
            + native_row[:, None] * stride_fo_s
            + dd[None, :],
            mask=row_ok[:, None] & dim_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        oh = tl.load(
            head_o
            + bank * stride_ho_b
            + rr[:, None] * stride_ho_s
            + dd[None, :],
            mask=row_ok[:, None] & dim_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        go = tl.load(
            coarse_o
            + bank * stride_go_b
            + rr[:, None] * stride_go_s
            + dd[None, :],
            mask=row_ok[:, None] & dim_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        gl = tl.load(
            coarse_l + bank * stride_gl_b + rr * stride_gl_s,
            mask=row_ok,
            other=0.0,
        ).to(tl.float32)
        if VALIDATE_PREVIOUS:
            # Error of the old refinement: rebuild the sparse output it would have
            # produced and compare it with the dense output.
            old_detail = tl.load(
                detail
                + bank * stride_d_b
                + rr[:, None] * stride_d_s
                + dd[None, :],
                mask=row_ok[:, None] & dim_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            old_dl = tl.load(
                detail_l + bank * stride_dl_b + rr * stride_dl_s,
                mask=row_ok,
                other=0.0,
            ).to(tl.float32)
            if DRIFT_ADMIT:
                # same drift weight as in sparse_merge
                ogl = tl.load(
                    drift_old_gl + bank * rows + rr, mask=row_ok, other=0.0
                ).to(tl.float32)
                cshift = tl.load(drift_center + bank).to(tl.float32)
                admit = tl.exp(-tl.abs(gl - ogl - cshift))
                corrected_l = gl + old_dl
                tail_common = tl.maximum(gl, corrected_l)
                sparse_mass = (1.0 - admit) * tl.exp(gl - tail_common)
                cache_mass = admit * tl.exp(corrected_l - tail_common)
                tail_den = tl.maximum(sparse_mass + cache_mass, 1.0e-30)
                old_tail_o = (
                    go * sparse_mass[:, None]
                    + (go + old_detail) * cache_mass[:, None]
                ) / tail_den[:, None]
                old_tail_l = tail_common + tl.log(tail_den)
            else:
                old_tail_l = gl + old_dl
                old_tail_o = go + old_detail
            common = tl.maximum(lh, old_tail_l)
            wh = tl.exp(lh - common)
            wt = tl.exp(old_tail_l - common)
            merge_den = tl.maximum(wh + wt, 1.0e-30)
            old_sparse = (
                oh * wh[:, None] + old_tail_o * wt[:, None]
            ) / merge_den[:, None]
            valid = row_ok[:, None] & dim_ok[None, :]
            err2 = tl.where(valid, (old_sparse - of) * (old_sparse - of), 0.0)
            ref2 = tl.where(valid, of * of, 0.0)
            num = tl.sum(tl.sum(err2, axis=1), axis=0)
            den = tl.sum(tl.sum(ref2, axis=1), axis=0)
            tl.atomic_add(validation_stats, num)
            tl.atomic_add(validation_stats + 1, den)
        tail = (of - ratio[:, None] * oh) / denominator[:, None]
        cached = tail - go
        cached = tl.maximum(tl.minimum(cached, 6.0e4), -6.0e4)
        cached = tl.where((ratio > 1.0 - 1.0e-3)[:, None], 0.0, cached)
        tl.store(
            detail
            + bank * stride_d_b
            + rr[:, None] * stride_d_s
            + dd[None, :],
            cached,
            mask=row_ok[:, None] & dim_ok[None, :],
        )

        exact_tail_l = lf + tl.log(denominator)
        dl = exact_tail_l - gl
        dl = tl.maximum(tl.minimum(dl, 30.0), -30.0)
        tl.store(
            detail_l + bank * stride_dl_b + rr * stride_dl_s,
            dl,
            mask=row_ok,
        )

    @triton.jit
    def _sparse_merge_kernel(
        head_o,
        head_l,
        coarse_o,
        coarse_l,
        detail,
        detail_l,
        drift_old_gl,
        drift_center,
        output,
        rows: tl.constexpr,
        sequence: tl.constexpr,
        dim: tl.constexpr,
        stride_ho_b: tl.constexpr,
        stride_ho_s: tl.constexpr,
        stride_hl_b: tl.constexpr,
        stride_hl_s: tl.constexpr,
        stride_go_b: tl.constexpr,
        stride_go_s: tl.constexpr,
        stride_gl_b: tl.constexpr,
        stride_gl_s: tl.constexpr,
        stride_d_b: tl.constexpr,
        stride_d_s: tl.constexpr,
        stride_dl_b: tl.constexpr,
        stride_dl_s: tl.constexpr,
        stride_o_b: tl.constexpr,
        stride_o_s: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        DRIFT_ADMIT: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        bank = tl.program_id(1)
        rr = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        dd = tl.arange(0, BLOCK_D)
        row_ok = rr < sequence
        video = rr < rows
        dim_ok = dd < dim

        oh = tl.load(
            head_o
            + bank * stride_ho_b
            + rr[:, None] * stride_ho_s
            + dd[None, :],
            mask=row_ok[:, None] & dim_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        lh = tl.load(
            head_l + bank * stride_hl_b + rr * stride_hl_s,
            mask=row_ok,
            other=-float("inf"),
        ).to(tl.float32)
        go = tl.load(
            coarse_o
            + bank * stride_go_b
            + rr[:, None] * stride_go_s
            + dd[None, :],
            mask=video[:, None] & dim_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        det = tl.load(
            detail
            + bank * stride_d_b
            + rr[:, None] * stride_d_s
            + dd[None, :],
            mask=video[:, None] & dim_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        gl = tl.load(
            coarse_l + bank * stride_gl_b + rr * stride_gl_s,
            mask=video,
            other=-float("inf"),
        ).to(tl.float32)
        dl = tl.load(
            detail_l + bank * stride_dl_b + rr * stride_dl_s,
            mask=video,
            other=0.0,
        ).to(tl.float32)
        if DRIFT_ADMIT:
            # Drift weight exp(-|shift|), where shift is the change of the coarse
            # log-mass since the refresh, centred per bank.
            ogl = tl.load(
                drift_old_gl + bank * rows + rr, mask=video, other=0.0
            ).to(tl.float32)
            cshift = tl.load(drift_center + bank).to(tl.float32)
            admit = tl.exp(-tl.abs(gl - ogl - cshift))
            corrected_l = gl + dl
            tail_common = tl.maximum(gl, corrected_l)
            sparse_mass = (1.0 - admit) * tl.exp(gl - tail_common)
            cache_mass = admit * tl.exp(corrected_l - tail_common)
            tail_den = tl.maximum(sparse_mass + cache_mass, 1.0e-30)
            tail_o = (
                go * sparse_mass[:, None]
                + (go + det) * cache_mass[:, None]
            ) / tail_den[:, None]
            tail_l = tail_common + tl.log(tail_den)
        else:
            tail_o = go + det
            tail_l = gl + dl
        common = tl.maximum(lh, tail_l)
        wh = tl.exp(lh - common)
        wt = tl.exp(tail_l - common)
        denominator = tl.maximum(wh + wt, 1.0e-30)
        merged = (oh * wh[:, None] + tail_o * wt[:, None]) / denominator[:, None]
        merged = tl.where(video[:, None], merged, oh)
        tl.store(
            output
            + bank * stride_o_b
            + rr[:, None] * stride_o_s
            + dd[None, :],
            merged,
            mask=row_ok[:, None] & dim_ok[None, :],
        )


@torch.no_grad()
def refresh_detail(
    full_output_native: torch.Tensor,
    full_lse_native: torch.Tensor,
    head_output_reordered: torch.Tensor,
    head_lse_reordered: torch.Tensor,
    coarse_output_reordered: torch.Tensor,
    coarse_lse_reordered: torch.Tensor,
    permutation: torch.Tensor,
    detail: torch.Tensor,
    detail_lse: torch.Tensor,
    *,
    video_offset: int,
    validation_stats: torch.Tensor | None = None,
    drift_old_lse: torch.Tensor | None = None,
    drift_center: torch.Tensor | None = None,
) -> None:
    """Write the refinement in place, optionally scoring the previous one first."""
    if not TRITON_AVAILABLE or not full_output_native.is_cuda:
        raise RuntimeError("Triton CUDA is required for fused detail refresh")
    banks, sequence, dim = full_output_native.shape
    rows = detail.shape[1]
    if detail.shape != (banks, rows, dim) or detail_lse.shape != (banks, rows):
        raise ValueError("detail-cache shape mismatch")
    if permutation.numel() < rows:
        raise ValueError("permutation does not cover cached video rows")
    if validation_stats is not None:
        if validation_stats.shape != (2,) or validation_stats.dtype != torch.float32:
            raise ValueError("validation_stats must be a float32 [2] tensor")
        if validation_stats.device != detail.device:
            raise ValueError("validation_stats must share the detail-cache device")
    _check_drift_inputs(drift_old_lse, drift_center, banks, rows)
    if drift_old_lse is not None and validation_stats is None:
        raise ValueError("the drift weight at a refresh only affects validation")
    block_m = 32
    block_d = triton.next_power_of_2(dim)
    _refresh_detail_kernel[(triton.cdiv(rows, block_m), banks)](
        full_output_native,
        full_lse_native,
        head_output_reordered,
        head_lse_reordered,
        coarse_output_reordered,
        coarse_lse_reordered,
        permutation,
        detail,
        detail_lse,
        detail if validation_stats is None else validation_stats,
        detail if drift_old_lse is None else drift_old_lse,
        detail if drift_center is None else drift_center,
        rows=rows,
        sequence=sequence,
        dim=dim,
        video_offset=int(video_offset),
        stride_fo_b=full_output_native.stride(0),
        stride_fo_s=full_output_native.stride(1),
        stride_fl_b=full_lse_native.stride(0),
        stride_fl_s=full_lse_native.stride(1),
        stride_ho_b=head_output_reordered.stride(0),
        stride_ho_s=head_output_reordered.stride(1),
        stride_hl_b=head_lse_reordered.stride(0),
        stride_hl_s=head_lse_reordered.stride(1),
        stride_go_b=coarse_output_reordered.stride(0),
        stride_go_s=coarse_output_reordered.stride(1),
        stride_gl_b=coarse_lse_reordered.stride(0),
        stride_gl_s=coarse_lse_reordered.stride(1),
        stride_d_b=detail.stride(0),
        stride_d_s=detail.stride(1),
        stride_dl_b=detail_lse.stride(0),
        stride_dl_s=detail_lse.stride(1),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        VALIDATE_PREVIOUS=validation_stats is not None,
        DRIFT_ADMIT=drift_old_lse is not None,
        num_warps=4,
        num_stages=2,
    )


def _check_drift_inputs(
    drift_old_lse: torch.Tensor | None,
    drift_center: torch.Tensor | None,
    banks: int,
    rows: int,
) -> None:
    if (drift_old_lse is None) != (drift_center is None):
        raise ValueError("drift_old_lse and drift_center must be passed together")
    if drift_old_lse is None:
        return
    if (drift_old_lse.shape != (banks, rows)
            or drift_old_lse.dtype != torch.float32
            or not drift_old_lse.is_contiguous()):
        raise ValueError("drift_old_lse must be contiguous fp32 [banks, rows]")
    if (drift_center.shape != (banks,)
            or drift_center.dtype != torch.float32
            or not drift_center.is_contiguous()):
        raise ValueError("drift_center must be contiguous fp32 [banks]")


@torch.no_grad()
def sparse_merge(
    head_output: torch.Tensor,
    head_lse: torch.Tensor,
    coarse_output: torch.Tensor,
    coarse_lse: torch.Tensor,
    detail: torch.Tensor,
    detail_lse: torch.Tensor,
    *,
    rows: int,
    drift_old_lse: torch.Tensor | None = None,
    drift_center: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge the head with the coarse tail and the drift-weighted refinement."""
    if not TRITON_AVAILABLE or not head_output.is_cuda:
        raise RuntimeError("Triton CUDA is required for fused sparse merge")
    banks, sequence, dim = head_output.shape
    _check_drift_inputs(drift_old_lse, drift_center, banks, int(rows))
    output = torch.empty_like(head_output)
    block_m = 32
    block_d = triton.next_power_of_2(dim)
    _sparse_merge_kernel[(triton.cdiv(sequence, block_m), banks)](
        head_output,
        head_lse,
        coarse_output,
        coarse_lse,
        detail,
        detail_lse,
        detail if drift_old_lse is None else drift_old_lse,
        detail if drift_center is None else drift_center,
        output,
        rows=int(rows),
        sequence=sequence,
        dim=dim,
        stride_ho_b=head_output.stride(0),
        stride_ho_s=head_output.stride(1),
        stride_hl_b=head_lse.stride(0),
        stride_hl_s=head_lse.stride(1),
        stride_go_b=coarse_output.stride(0),
        stride_go_s=coarse_output.stride(1),
        stride_gl_b=coarse_lse.stride(0),
        stride_gl_s=coarse_lse.stride(1),
        stride_d_b=detail.stride(0),
        stride_d_s=detail.stride(1),
        stride_dl_b=detail_lse.stride(0),
        stride_dl_s=detail_lse.stride(1),
        stride_o_b=output.stride(0),
        stride_o_s=output.stride(1),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        DRIFT_ADMIT=drift_old_lse is not None,
        num_warps=4,
        num_stages=2,
    )
    return output
