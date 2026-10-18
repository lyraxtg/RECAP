"""Attention refinement kept in host memory and streamed to the GPU layer by layer.

Outputs are identical to keeping it on the GPU.  Enabled by ``operator.host_banks``.
"""

from __future__ import annotations

import torch

from .operator import REFRESH, SPARSE, Operator

_FIELDS = ("value", "log_mass")

_SPLIT_FIELDS = ("partition", "head_kernel_mask", "coarse_lse")


class HostBankOperator(Operator):
    """An :class:`Operator` with the refinement in pinned host memory and two GPU
    staging buffers."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._host: dict = {}
        self._slots: list[dict] = []
        self._slot_ready: list = [None, None]  # write-back to host finished
        self._slot_free: list = [None, None]  # compute finished reading the slot
        self._want_prefetch = True
        self._prefetch_stream = None
        self._host_split: dict = {}
        self._split_slots: dict = {}
        self._device = None
        self._out_stream = None
        self._prefetched = None

    def set_prefetch(self, enabled: bool) -> None:
        """Overlap the next layer's transfer with this layer's compute."""
        self._want_prefetch = bool(enabled)

    # -- device slots ------------------------------------------------------------------

    def _rows(self) -> int:
        block = self.config.block_size
        return (self.layout.video_tokens // block) * block

    def _alloc_slots(self, n_banks: int, rows: int, dim: int, device) -> None:
        self._slots = []
        for _ in range(2 if self._want_prefetch else 1):
            self._slots.append({
                "value": torch.empty(
                    (n_banks, rows, dim), dtype=torch.float16, device=device),
                "log_mass": torch.empty((n_banks, rows), dtype=torch.float32, device=device),
            })
        if self._want_prefetch and self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=device)
        if self._out_stream is None:
            self._out_stream = torch.cuda.Stream(device=device)

    def _ensure_slots(self, query: torch.Tensor) -> None:
        """Allocate the staging slots if missing or of the wrong shape."""
        b, h, _, dim = query.shape
        n_banks, rows = b * h, self._rows()
        if self._slots and self._slots[0]["value"].shape == (n_banks, rows, dim):
            return
        self._alloc_slots(n_banks, rows, dim, query.device)

    @staticmethod
    def _new_host_bank(n_banks: int, rows: int, dim: int) -> dict:
        return {
            "value": torch.empty(
                (n_banks, rows, dim), dtype=torch.float16).pin_memory(),
            "log_mass": torch.empty((n_banks, rows), dtype=torch.float32).pin_memory(),
        }

    def _host_bank(self, key_id) -> dict:
        """Pinned host tensors for one (stream, layer)."""
        existing = self._host.get(key_id)
        if existing is not None:
            return existing
        slot = self._slots[0]
        n_banks, rows, dim = slot["value"].shape
        made = self._new_host_bank(n_banks, rows, dim)
        self._host[key_id] = made
        return made

    def preallocate_host(self, n_layers: int, n_banks: int, dim: int) -> int:
        """Allocate all host memory and the GPU slots; returns the host bytes."""
        rows = self._rows()
        for stream in range(self.n_streams):
            for layer in range(len(self.layers[stream])):
                if layer >= n_layers:
                    break
                key_id = (stream, layer)
                if key_id not in self._host:
                    self._host[key_id] = self._new_host_bank(n_banks, rows, dim)
        if torch.cuda.is_available() and not self._slots:
            self._alloc_slots(n_banks, rows, dim, torch.device("cuda"))
        return self.peak_bank_bytes()

    @staticmethod
    def _copy(dst: dict, src: dict, *, non_blocking: bool, fields=_FIELDS) -> None:
        for field in fields:
            if dst.get(field) is not None and src.get(field) is not None:
                dst[field].copy_(src[field], non_blocking=non_blocking)

    @staticmethod
    def _attach(bank, slot: dict) -> None:
        bank.value, bank.log_mass = slot["value"], slot["log_mass"]

    @staticmethod
    def _detach(bank) -> None:
        """Drop the device references."""
        bank.value = bank.log_mass = None

    # -- per-layer partition state -----------------------------------------------------

    def _split_in(self, bank, key_id) -> None:
        """Copy this layer's partition, kernel mask and drift reference to the GPU."""
        host = self._host_split.get(key_id)
        if not host:
            return
        slot = self._split_slots.setdefault(
            key_id[1] % max(1, len(self._slots)), {})
        for field, tensor in host.items():
            device_tensor = slot.get(field)
            if device_tensor is None or device_tensor.shape != tensor.shape \
                    or device_tensor.dtype != tensor.dtype:
                device_tensor = torch.empty_like(tensor, device=self._device)
                slot[field] = device_tensor
            device_tensor.copy_(tensor, non_blocking=True)
            setattr(bank, field, device_tensor)

    def _split_out(self, bank, key_id) -> None:
        """Copy the partition state written by a refresh back to host memory."""
        host = self._host_split.setdefault(key_id, {})
        for field in _SPLIT_FIELDS:
            tensor = getattr(bank, field, None)
            if tensor is None:
                host.pop(field, None)
                continue
            buffer = host.get(field)
            if buffer is None or buffer.shape != tensor.shape \
                    or buffer.dtype != tensor.dtype:
                buffer = torch.empty(tensor.shape, dtype=tensor.dtype).pin_memory()
                host[field] = buffer
            buffer.copy_(tensor, non_blocking=False)

    @staticmethod
    def _split_detach(bank) -> None:
        for field in _SPLIT_FIELDS:
            setattr(bank, field, None)

    # -- dispatch ----------------------------------------------------------------------

    @torch.no_grad()
    def attend(self, layer_index: int, query, key, value):
        if self.kind not in (REFRESH, SPARSE):
            return super().attend(layer_index, query, key, value)

        bank = self.banks[self.stream][layer_index]
        self._ensure_slots(query)
        index = layer_index % len(self._slots)
        slot = self._slots[index]
        key_id = (self.stream, layer_index)
        host = self._host.get(key_id)
        if host is not None and host["value"].shape != slot["value"].shape:
            self._host.pop(key_id)
            host = None

        # Wait for a pending write-back from this slot.
        pending = self._slot_ready[index]
        if pending is not None:
            torch.cuda.current_stream().wait_event(pending)
            self._slot_ready[index] = None

        self._device = query.device
        if self.kind == SPARSE:
            if host is None:
                raise RuntimeError("a sparse call was requested before an exact refresh")
            self._stage_in(key_id, slot)
        elif bank.valid and host is not None:
            # The refresh measures the old refinement.
            self._stage_in(key_id, slot)
        self._attach(bank, slot)
        self._split_in(bank, key_id)

        try:
            out = super().attend(layer_index, query, key, value)
        finally:
            if self.kind == REFRESH:
                self._stage_out(key_id, slot, index)
                self._split_out(bank, key_id)
            self._split_detach(bank)
            # The slot may be refilled by a prefetch after this point.
            released = torch.cuda.Event()
            released.record()
            self._slot_free[index] = released
            self._detach(bank)
        return out

    def _stage_in(self, key_id, slot: dict) -> None:
        host = self._host[key_id]
        if self._prefetched == key_id and self._prefetch_stream is not None:
            torch.cuda.current_stream().wait_stream(self._prefetch_stream)
            self._prefetched = None
        else:
            self._copy(slot, host, non_blocking=True)
        self._maybe_prefetch(key_id)

    def _maybe_prefetch(self, key_id) -> None:
        """Copy the next layer's refinement while this layer computes."""
        if self._prefetch_stream is None:
            return
        stream, layer = key_id
        nxt = (stream, layer + 1)
        if nxt not in self._host:
            return
        index = (layer + 1) % len(self._slots)
        if self._slot_ready[index] is not None:
            return  # still being written back
        with torch.cuda.stream(self._prefetch_stream):
            released = self._slot_free[index]
            if released is not None:
                self._prefetch_stream.wait_event(released)
            self._copy(self._slots[index], self._host[nxt], non_blocking=True)
        self._prefetched = nxt

    def _stage_out(self, key_id, slot: dict, index: int) -> None:
        """Copy the new refinement to host memory on the write-back stream."""
        host = self._host_bank(key_id)
        if self._out_stream is not None:
            written = torch.cuda.Event()
            written.record()
            with torch.cuda.stream(self._out_stream):
                self._out_stream.wait_event(written)
                self._copy(host, slot, non_blocking=True)
                event = torch.cuda.Event()
                event.record()
        else:
            self._copy(host, slot, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        self._slot_ready[index] = event

    # -- bookkeeping -------------------------------------------------------------------

    def finish(self) -> None:
        """Wait for all transfers in flight."""
        for events in (self._slot_ready, self._slot_free):
            for event in events:
                if event is not None:
                    event.synchronize()
        if self._prefetch_stream is not None:
            self._prefetch_stream.synchronize()
        if self._out_stream is not None:
            self._out_stream.synchronize()
        self._slot_ready = [None, None]
        self._slot_free = [None, None]
        self._prefetched = None

    def reset(self) -> None:
        self.finish()
        super().reset()

    def __del__(self):  # pragma: no cover - teardown path
        try:
            self.finish()
        except Exception:
            pass

    def peak_bank_bytes(self) -> int:
        """Host bytes held by the refinement."""
        return sum(
            tensor.numel() * tensor.element_size()
            for bank in self._host.values()
            for tensor in bank.values() if tensor is not None
        )
