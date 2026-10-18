"""Attention for the EXACT, REFRESH and SPARSE steps.

For each query block, keys are split into selected blocks (the *head*), computed exactly,
and the *tail*.  The tail is ``coarse + refinement``: a coarse estimate recomputed at
every call, plus a refinement cached per layer at the last dense refresh (the *detail
bank*).

``EXACT``    dense attention.
``REFRESH``  dense attention; also stores the refinement (exact tail minus coarse tail)
             and measures the error of the previous refinement (``e_detail``).
``SPARSE``   head computed exactly, plus the coarse tail and the cached refinement
             weighted by the drift weight, merged in log-mass space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

from .config import OperatorConfig
from .kernels import detail_cache
from .kernels.block_sparse import block_sparse_exact, build_kernel_block_mask
from .sparse.attention import SparseAttention, SparseGeometry
from .sparse.layout import Layout

EXACT = "exact"
REFRESH = "refresh"
SPARSE = "sparse"


@dataclass
class DetailBank:
    """One layer's cached refinement and its partition."""

    head_kernel_mask: torch.Tensor | None = None  # [banks, blocks, blocks] bool
    partition: torch.Tensor | None = None  # [..., query blocks, key blocks] bool
    # refinement of the tail output and of the tail log-mass
    value: torch.Tensor | None = None  # [banks, rows, dim] fp16
    log_mass: torch.Tensor | None = None  # [banks, rows] fp32
    # coarse tail log-mass at the refresh, reference for the drift weight
    coarse_lse: torch.Tensor | None = None  # [banks, rows] fp32
    valid: bool = False
    refreshed_at: int = 0
    # error of the previous refinement, measured at this refresh
    pending_stats: torch.Tensor | None = None  # [2] fp32: (sum sq error, sum sq output)
    pending_age: int | None = None

    def release(self) -> None:
        self.head_kernel_mask = None
        self.partition = None
        self.value = None
        self.log_mass = None
        self.coarse_lse = None
        self.valid = False
        self.refreshed_at = 0
        self.pending_stats = None
        self.pending_age = None


class BankChunk:
    """A :class:`DetailBank` restricted to a range of heads (``bank_chunks``)."""

    __slots__ = ("bank", "lo", "hi", "total", "batch", "head0", "head1", "heads",
                 "valid", "refreshed_at", "pending_stats", "pending_age")

    #: layer-level state, published after all groups have run
    _DEFERRED = ("valid", "refreshed_at", "pending_stats", "pending_age")

    def __init__(self, bank: DetailBank, *, lo: int, hi: int, total: int,
                 batch: int, head0: int, head1: int, heads: int):
        self.bank, self.lo, self.hi, self.total = bank, lo, hi, total
        self.batch, self.head0, self.head1, self.heads = batch, head0, head1, heads
        for name in BankChunk._DEFERRED:
            object.__setattr__(self, name, getattr(bank, name))

    def __getattr__(self, name):
        return getattr(self.bank, name)

    def __setattr__(self, name, value):
        if name in BankChunk.__slots__:
            object.__setattr__(self, name, value)
        elif name in ("value", "log_mass", "coarse_lse", "partition", "head_kernel_mask"):
            self._store(name, value)
        else:
            setattr(self.bank, name, value)

    def commit(self, others: list["BankChunk"]) -> None:
        """Publish the layer-level state; validation sums add over groups."""
        self.bank.valid = self.valid
        self.bank.refreshed_at = self.refreshed_at
        self.bank.pending_age = self.pending_age
        stats = [c.pending_stats for c in others if c.pending_stats is not None]
        self.bank.pending_stats = None if not stats else (
            stats[0] if len(stats) == 1 else torch.stack(stats).sum(dim=0))

    # -- tensors -----------------------------------------------------------------------

    def _slice(self, name):
        whole = getattr(self.bank, name)
        if whole is None:
            return None
        if name == "partition":
            return whole[self.batch:self.batch + 1, self.head0:self.head1]
        return whole[self.lo:self.hi]

    def _store(self, name, value):
        if value is None:
            setattr(self.bank, name, None)
            return
        whole = getattr(self.bank, name)
        if whole is None:
            full_shape = ((self._batches(), self.heads, *value.shape[2:])
                          if name == "partition"
                          else (self.total, *value.shape[1:]))
            whole = torch.empty(full_shape, dtype=value.dtype, device=value.device)
            setattr(self.bank, name, whole)
        if name == "partition":
            whole[self.batch:self.batch + 1, self.head0:self.head1] = value
        else:
            whole[self.lo:self.hi] = value

    def _batches(self) -> int:
        return max(1, self.total // self.heads)

    @property
    def value(self):
        return self._slice("value")

    @property
    def log_mass(self):
        return self._slice("log_mass")

    @property
    def coarse_lse(self):
        return self._slice("coarse_lse")

    @property
    def partition(self):
        return self._slice("partition")

    @property
    def head_kernel_mask(self):
        return self._slice("head_kernel_mask")


def _drift_center(coarse_lse: torch.Tensor, old_lse: torch.Tensor) -> torch.Tensor:
    """Per-bank mean shift of the coarse log-mass since the refresh."""
    return (coarse_lse - old_lse).mean(dim=1).contiguous()


@dataclass
class Operator:
    """Dispatches one attention call per layer and owns the detail banks."""

    config: OperatorConfig
    layout: Layout
    #: per-layer sparse state, indexed ``[stream][layer]``
    layers: list[list[SparseAttention]]
    banks: list[list[DetailBank]] = field(default_factory=list)
    kind: str = EXACT
    step: int = 0
    stream: int = 0
    refreshed_layers: int = 0
    _chunk_states: dict = field(default_factory=dict)
    _forced_partition: torch.Tensor | None = None
    _forced_rerank: bool | None = None

    def __post_init__(self) -> None:
        if not self.layers or isinstance(self.layers[0], SparseAttention):
            raise TypeError("Operator.layers must be indexed [stream][layer]")
        self.banks = [[DetailBank() for _ in row] for row in self.layers]

    @classmethod
    def build(
        cls,
        config: OperatorConfig,
        geometry: SparseGeometry,
        *,
        n_layers: int,
        n_streams: int = 1,
    ) -> "Operator":
        """One sparse state per (stream, layer)."""
        if n_layers < 1 or n_streams < 1:
            raise ValueError("n_layers and n_streams must be positive")
        layers = [
            [SparseAttention(geometry, config) for _ in range(n_layers)]
            for _ in range(n_streams)
        ]
        return cls(config, geometry.layout, layers)

    @property
    def n_streams(self) -> int:
        return len(self.layers)

    def reset(self) -> None:
        for row in self.banks:
            for bank in row:
                bank.release()
        for row in self.layers:
            for layer in row:
                layer.reset()
        self.kind = EXACT
        self.step = 0
        self.stream = 0
        self.refreshed_layers = 0
        for state in self._chunk_states.values():
            state.reset()

    @property
    def banks_valid(self) -> bool:
        """Whether every layer holds a refinement."""
        return bool(self.banks) and all(
            bank.valid for row in self.banks for bank in row
        )

    def begin_step(self, step: int, kind: str) -> None:
        if kind not in (EXACT, REFRESH, SPARSE):
            raise ValueError(f"unknown computed call kind {kind!r}")
        self.step = int(step)
        self.kind = kind
        self.stream = 0
        self.refreshed_layers = 0

    def set_stream(self, stream: int) -> None:
        if not 0 <= stream < self.n_streams:
            raise ValueError(f"stream {stream} outside 0..{self.n_streams - 1}")
        self.stream = int(stream)

    # -- dispatch ----------------------------------------------------------------------

    @torch.no_grad()
    def attend(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """One attention call."""
        chunks = int(self.config.bank_chunks)
        if chunks > 1 and self.kind != EXACT:
            return self._attend_in_chunks(layer_index, query, key, value, chunks)
        layer = self.layers[self.stream][layer_index]
        bank = self.banks[self.stream][layer_index]
        if self.kind == SPARSE:
            return self._sparse(layer, bank, query, key, value)
        if self.kind == EXACT:
            return F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
        # A refresh also needs the log-sum-exp.
        output, lse = torch.ops.aten._scaled_dot_product_flash_attention(
            query, key, value, 0.0, False, False
        )[:2]
        self._refresh(layer, bank, query, key, value, output, lse)
        return output

    @torch.no_grad()
    def _attend_in_chunks(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        chunks: int,
    ) -> torch.Tensor:
        """Run one layer in groups of heads (``bank_chunks``)."""
        b, h, sequence, dim = query.shape
        bank = self.banks[self.stream][layer_index]
        out = lse = None
        if self.kind == REFRESH:
            out, lse = torch.ops.aten._scaled_dot_product_flash_attention(
                query, key, value, 0.0, False, False
            )[:2]
        else:
            out = torch.empty_like(query)
        # The re-selection is decided once for the whole layer.
        forced_partition = None
        layer0 = self.layers[self.stream][layer_index]
        if self.kind == REFRESH and bank.valid and bank.partition is not None:
            scale = 1.0 / math.sqrt(dim)
            ql_all = self.layout.reorder(query)
            kl_all = self.layout.reorder(key)
            forced_partition = layer0.head_partition(ql_all, kl_all, scale)
            del ql_all, kl_all
            kept = (forced_partition & bank.partition).sum()
            total = forced_partition.sum().clamp_min(1)
            self._forced_rerank = (float(kept) / float(total)
                                   < self.config.rerank_min_overlap)
        per = max(1, math.ceil(h / chunks))
        group = 0
        views: list[BankChunk] = []
        refreshed_before = self.refreshed_layers
        for bi in range(b):
            for h0 in range(0, h, per):
                h1 = min(h0 + per, h)
                state = self._chunk_states.get((self.stream, layer_index, group))
                if state is None:
                    state = SparseAttention(self.layers[self.stream][layer_index].geometry,
                                            self.config)
                    self._chunk_states[(self.stream, layer_index, group)] = state
                view = BankChunk(
                    bank, lo=bi * h + h0, hi=bi * h + h1, total=b * h,
                    batch=bi, head0=h0, head1=h1, heads=h)
                views.append(view)
                q = query[bi:bi + 1, h0:h1]
                k = key[bi:bi + 1, h0:h1]
                v = value[bi:bi + 1, h0:h1]
                self._forced_partition = (
                    None if forced_partition is None
                    else forced_partition[bi:bi + 1, h0:h1])
                if self.kind == SPARSE:
                    out[bi:bi + 1, h0:h1] = self._sparse(state, view, q, k, v)
                else:
                    self._refresh(state, view, q, k, v,
                                  out[bi:bi + 1, h0:h1], lse[bi:bi + 1, h0:h1])
                group += 1
        self._forced_partition = None
        self._forced_rerank = None
        if views:
            views[-1].commit(views)
        if self.refreshed_layers > refreshed_before:
            self.refreshed_layers = refreshed_before + 1
        return out

    # -- refresh -----------------------------------------------------------------------

    @torch.no_grad()
    def _refresh(
        self,
        layer: SparseAttention,
        bank: DetailBank,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        full_output: torch.Tensor,
        full_lse: torch.Tensor,
    ) -> None:
        """Store the refinement and measure the previous one."""
        scale = 1.0 / math.sqrt(query.shape[-1])
        b, h, sequence, dim = query.shape
        n_banks = b * h
        n_video = self.layout.video_tokens
        block_size = self.config.block_size
        rows = (n_video // block_size) * block_size

        ql = self.layout.reorder(query)
        kl = self.layout.reorder(key)
        vl = self.layout.reorder(value)

        revalidating = bank.valid
        if not revalidating:
            partition = layer.head_partition(ql, kl, scale)
            flat = partition.reshape(n_banks, partition.shape[-2], partition.shape[-1])
            bank.partition = partition
            bank.head_kernel_mask = build_kernel_block_mask(
                flat, sequence_tokens=sequence, video_tokens=n_video,
                block_size=block_size,
            )

        head = block_sparse_exact(
            ql.reshape(1, n_banks, sequence, dim),
            kl.reshape(1, n_banks, sequence, dim),
            vl.reshape(1, n_banks, sequence, dim),
            bank.head_kernel_mask,
        )
        head_output = head.output.reshape(n_banks, sequence, dim)
        head_lse = head.lse.reshape(n_banks, sequence).float()
        tail_context = layer.coarse_tail_context(ql, kl, vl, scale)
        del kl, vl
        coarse, coarse_lse = layer.coarse_tail_apply(
            ql, tail_context, bank.partition, scale, rows
        )

        if bank.value is None:
            bank.value = torch.empty(
                n_banks, rows, dim, dtype=torch.float16, device=query.device
            )
            bank.log_mass = torch.empty(
                n_banks, rows, dtype=torch.float32, device=query.device
            )

        stats = (
            torch.zeros(2, dtype=torch.float32, device=query.device)
            if revalidating
            else None
        )
        old_lse = bank.coarse_lse if revalidating else None
        center = None
        if old_lse is not None:
            # Score the old refinement with the drift weight, as sparse steps used it.
            center = _drift_center(coarse_lse, old_lse)
        detail_cache.refresh_detail(
            full_output.reshape(n_banks, sequence, dim),
            full_lse.reshape(n_banks, sequence),
            head_output,
            head_lse,
            coarse,
            coarse_lse,
            self.layout.video_permutation(query.device),
            bank.value,
            bank.log_mass,
            video_offset=self.layout.video_offset,
            validation_stats=stats,
            drift_old_lse=old_lse,
            drift_center=center,
        )
        if stats is not None:
            bank.pending_stats = stats
            bank.pending_age = self.step - bank.refreshed_at

        if revalidating:
            # Re-select the key blocks from the current attention.
            kl = None
            if self._forced_partition is not None:
                partition = self._forced_partition
            else:
                kl = self.layout.reorder(key)
                partition = layer.head_partition(ql, kl, scale)
            # Keep the old partition if the selection barely changed.
            if bank.partition is not None:
                if self._forced_rerank is None:
                    kept = (partition & bank.partition).sum()
                    total = partition.sum().clamp_min(1)
                    keep_split = (float(kept) / float(total)
                                  >= self.config.rerank_min_overlap)
                else:
                    keep_split = not self._forced_rerank
                if keep_split:
                    bank.coarse_lse = coarse_lse.clone()
                    bank.refreshed_at = self.step
                    bank.valid = True
                    self.refreshed_layers += 1
                    return
            bank.partition = partition
            bank.head_kernel_mask = build_kernel_block_mask(
                partition.reshape(n_banks, partition.shape[-2], partition.shape[-1]),
                sequence_tokens=sequence, video_tokens=n_video, block_size=block_size,
            )
            del head, head_output, head_lse, coarse, coarse_lse, old_lse, center  # noqa: F821
            if kl is None:
                kl = self.layout.reorder(key)
            vl = self.layout.reorder(value)
            head = block_sparse_exact(
                ql.reshape(1, n_banks, sequence, dim),
                kl.reshape(1, n_banks, sequence, dim),
                vl.reshape(1, n_banks, sequence, dim),
                bank.head_kernel_mask,
            )
            head_output = head.output.reshape(n_banks, sequence, dim)
            head_lse = head.lse.reshape(n_banks, sequence).float()
            coarse, coarse_lse = layer.coarse_tail_apply(
                ql, tail_context, bank.partition, scale, rows
            )
            detail_cache.refresh_detail(
                full_output.reshape(n_banks, sequence, dim),
                full_lse.reshape(n_banks, sequence),
                head_output,
                head_lse,
                coarse,
                coarse_lse,
                self.layout.video_permutation(query.device),
                bank.value,
                bank.log_mass,
                video_offset=self.layout.video_offset,
                validation_stats=None,
            )
        bank.coarse_lse = coarse_lse.clone()
        bank.refreshed_at = self.step
        bank.valid = True
        self.refreshed_layers += 1

    # -- sparse ------------------------------------------------------------------------

    @torch.no_grad()
    def _sparse(
        self,
        layer: SparseAttention,
        bank: DetailBank,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Head computed exactly, plus coarse tail and weighted cached refinement."""
        if not bank.valid or bank.head_kernel_mask is None:
            raise RuntimeError("a sparse call was requested before an exact refresh")
        scale = 1.0 / math.sqrt(query.shape[-1])
        b, h, sequence, dim = query.shape
        n_banks = b * h
        n_video = self.layout.video_tokens
        block_size = self.config.block_size
        rows = (n_video // block_size) * block_size

        ql = self.layout.reorder(query)
        kl = self.layout.reorder(key)
        vl = self.layout.reorder(value)

        head = block_sparse_exact(
            ql.reshape(1, n_banks, sequence, dim),
            kl.reshape(1, n_banks, sequence, dim),
            vl.reshape(1, n_banks, sequence, dim),
            bank.head_kernel_mask,
        )
        head_output = head.output.reshape(n_banks, sequence, dim)
        head_lse = head.lse.reshape(n_banks, sequence).float()
        tail_context = layer.coarse_tail_context(ql, kl, vl, scale)
        del kl, vl
        coarse, coarse_lse = layer.coarse_tail_apply(
            ql, tail_context, bank.partition, scale, rows
        )
        del tail_context
        old_lse = bank.coarse_lse
        center = _drift_center(coarse_lse, old_lse)
        merged = detail_cache.sparse_merge(
            head_output,
            head_lse,
            coarse,
            coarse_lse,
            bank.value,
            bank.log_mass,
            rows=rows,
            drift_old_lse=old_lse,
            drift_center=center,
        )
        return self.layout.restore(merged.reshape(b, h, sequence, dim))

    # -- the refinement error ----------------------------------------------------------

    def drain_validation(self) -> tuple[int, float] | None:
        """Relative RMS error over all layers, and the smallest refinement age."""
        stats: list[torch.Tensor] = []
        ages: list[int] = []
        for row in self.banks:
            for bank in row:
                if bank.pending_stats is not None and bank.pending_age is not None:
                    stats.append(bank.pending_stats)
                    ages.append(int(bank.pending_age))
                bank.pending_stats = None
                bank.pending_age = None
        if not stats:
            return None
        total = torch.stack(stats).sum(dim=0)
        numerator, denominator = (float(x) for x in total)
        error = math.sqrt(numerator / denominator) if denominator > 0.0 else float("inf")
        return min(ages), error

    def peak_bank_bytes(self) -> int:
        """Bytes held by the refinement."""
        total = 0
        for row in self.banks:
            for bank in row:
                for tensor in (bank.value, bank.log_mass):
                    if tensor is not None:
                        total += tensor.numel() * tensor.element_size()
        return total
