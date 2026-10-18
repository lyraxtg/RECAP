"""Step loop: one decision per denoising step.

``EXACT``    dense transformer evaluation.
``REFRESH``  dense evaluation that also renews the attention refinement and measures the
             error of the old one.
``SPARSE``   corrected sparse evaluation.
``REUSE``    no evaluation; the prediction is extrapolated as ``o + g * h * v`` from the
             latest computed prediction ``o`` and its velocity ``v``, with a gain ``g``
             fitted online by least squares.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any

import torch

from .config import RecapConfig
from .controller import Controller
from .operator import EXACT, REFRESH, SPARSE, Operator

REUSE = "reuse"
_COMPUTE = "compute"
_SPARSE_COVER = "sparse"


def _extract(out: Any) -> torch.Tensor:
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "sample"):
        return out.sample
    return out


def _rebuild(template: Any, tensor: torch.Tensor) -> Any:
    if isinstance(template, tuple):
        return (tensor,) + template[1:]
    if hasattr(template, "sample"):
        return dataclasses.replace(template, sample=tensor)
    return tensor


class RecapSession:
    """Per-sample state, wrapped around the transformer's forward.

    Pipelines that run the guidance branches as separate calls make ``calls_per_step``
    calls per denoising step.  All of them execute the same decision.
    """

    def __init__(
        self,
        config: RecapConfig,
        operator: Operator,
        *,
        calls_per_step: int = 1,
        model_info: dict[str, Any] | None = None,
    ):
        self.config = config
        self.operator = operator
        self.calls_per_step = int(calls_per_step)
        if self.calls_per_step < 1:
            raise ValueError("calls_per_step must be positive")
        if operator.n_streams != self.calls_per_step:
            raise ValueError(
                f"operator has {operator.n_streams} streams but the pipeline makes "
                f"{self.calls_per_step} transformer calls per step"
            )
        self.model_info = dict(model_info or {})
        self.reset()

    def reset(self) -> None:
        """Call before every sample."""
        # CPU offload restores the original forward after each pipeline call.
        transformer = getattr(self, "_transformer", None)
        if transformer is not None and transformer.forward is not self._wrapped_forward:
            self._wrap(transformer)
        self.controller = Controller(self.config.schedule)
        self.operator.reset()
        self.step = 0
        self.call_index = 0
        self.last_computed: int | None = None
        self.last_refresh: int | None = None
        self.phase_active = False
        self.actions: list[dict] = []
        # prediction cache per stream: latest computed prediction and its velocity
        self.caches: list[dict[str, torch.Tensor | None] | None] = [
            None
        ] * self.calls_per_step
        self.gain = 0.0
        self._gain_num = 0.0
        self._gain_den = 0.0
        self.output_objects: list[Any] = [None] * self.calls_per_step
        self.output_dtypes: list[torch.dtype | None] = [None] * self.calls_per_step
        self.midpoint_step: int | None = None
        self.midpoint_done = False
        self._step_action: str | None = None
        self._step_gain: float = 0.0
        self._step_seconds: float = 0.0
        self._step_residuals: list[torch.Tensor] = []
        self._step_outputs: list[torch.Tensor] = []
        self._step_gain_num: float = 0.0
        self._step_gain_den: float = 0.0

    # -- decisions ---------------------------------------------------------------------

    def _decide_cover(self, t: int) -> str:
        controller = self.controller
        if self.last_computed is None:
            return _COMPUTE
        if controller.in_exact_tail(t):
            return REUSE if controller.budget_exhausted() else _COMPUTE
        if controller.mid_budget_exhausted():
            return REUSE
        if not controller.licensed:
            if t < self.config.schedule.prefix_floor:
                return _COMPUTE
            cap = self.config.schedule.prefix_cap
            if controller.prefix_overspent() or (cap and t >= cap):
                controller.license()
            else:
                return _COMPUTE
        if t - self.last_computed < controller.allowed_gap(t):
            return REUSE
        if not self.phase_active or not self.operator.banks_valid:
            return _COMPUTE
        if (
            self.last_refresh is None
            or not controller.lease_ok(t - self.last_refresh)
        ):
            # Refinement too old: refresh it.
            return _COMPUTE
        return _SPARSE_COVER

    def _terminal_sparse(self, t: int) -> bool:
        """Compute the last step sparsely if its interval exceeds the validated span."""
        schedule = self.config.schedule
        if not (
            self._terminal_unvalidated(t)
            and t == schedule.num_steps - 1
            and self.operator.banks_valid
        ):
            return False
        if self.last_refresh is not None:
            if not self.controller.lease_ok(t - self.last_refresh):
                return False
        return not self.controller.budget_exhausted()

    def _terminal_unvalidated(self, t: int) -> bool:
        if self.last_refresh is None or self.midpoint_step is None or not self.midpoint_done:
            return True
        validated = self.midpoint_step - (self.last_refresh + 1)
        return (t - self.midpoint_step) > validated

    def _action(self, t: int) -> str:
        cover = self._decide_cover(t)
        if cover != REUSE and self.controller.in_exact_tail(t):
            return EXACT
        if self._terminal_sparse(t):
            cover = _SPARSE_COVER
        if cover == REUSE:
            return REUSE
        if self.controller.licensed and not self.phase_active and cover == _COMPUTE:
            # The first computation after reuse starts is the first dense refresh.
            self.phase_active = True
        if not self.phase_active:
            return EXACT
        if cover == _SPARSE_COVER and self.operator.banks_valid:
            return SPARSE
        return REFRESH

    # -- prediction mismatch -----------------------------------------------------------

    @torch.no_grad()
    def _observe_stream(self, t: int, stream: int, output: torch.Tensor) -> None:
        """Renew the prediction cache and record the mismatch of the extrapolation."""
        cache = self.caches[stream]
        if cache is None:
            self.caches[stream] = {"o": output, "v": None}
            return
        if self.last_computed is None:
            raise RuntimeError("prediction cache exists without a previous computed step")
        h = t - int(self.last_computed)
        if h <= 0:
            raise RuntimeError(f"non-positive computed-state gap {h} at step {t}")

        delta = output - cache["o"]
        velocity = cache["v"]
        if velocity is not None:
            prediction = float(h) * velocity
            gain = self._step_gain
            residual = delta - gain * prediction if gain else delta
            denominator = float(torch.linalg.vector_norm(prediction) ** 2)
            if denominator > 0.0:
                self._step_gain_num += float((delta * prediction).sum())
                self._step_gain_den += denominator
        else:
            residual = delta
        self._step_residuals.append(residual)
        self._step_outputs.append(output)
        self.caches[stream] = {"o": output, "v": delta / float(h)}

    def _finish_observation(self, t: int) -> dict[str, Any]:
        """Relative mismatch ``m`` over all streams, and the updated gain."""
        residuals, outputs = self._step_residuals, self._step_outputs
        self._step_residuals, self._step_outputs = [], []
        if not residuals:
            return {}
        if self._step_gain_den > 0.0:
            self._gain_num += self._step_gain_num
            self._gain_den += self._step_gain_den
            self.gain = self._gain_num / max(self._gain_den, 1e-30)
        if len(residuals) == 1:
            residual_norm = torch.linalg.vector_norm(residuals[0])
            output_norm = torch.linalg.vector_norm(outputs[0])
        else:
            residual_norm = torch.sqrt(
                sum(torch.linalg.vector_norm(r) ** 2 for r in residuals)
            )
            output_norm = torch.sqrt(
                sum(torch.linalg.vector_norm(o) ** 2 for o in outputs)
            )
        misfit = float(residual_norm / output_norm.clamp_min(1e-30))
        h = t - int(self.last_computed)
        return {"h": h, "m": round(misfit, 6), "g": round(self.gain, 4)}

    # -- loop --------------------------------------------------------------------------

    def begin(self, t: int) -> str:
        action = self._action(t)
        if action != REUSE:
            self.controller.computed += 1
            self.operator.begin_step(t, action)
            if t == self.midpoint_step and action == SPARSE:
                self.midpoint_done = True
        return action

    def complete(
        self, t: int, action: str, diagnostics: dict[str, Any]
    ) -> dict[str, Any]:
        row: dict[str, Any] = {"t": t, "a": action}
        if action == REFRESH:
            row.update(self._after_refresh(t))
        row.update(diagnostics)
        if "m" in row and "h" in row:
            self.controller.observe_residual(int(row["h"]), float(row["m"]), t)
        self.last_computed = t
        return row

    def install(self, transformer) -> None:
        """Wrap ``transformer.forward`` with the step loop."""
        if getattr(transformer, "_recap_installed", False):
            raise RuntimeError("this transformer already has a RECAP step loop")
        self._transformer = transformer
        self._wrap(transformer)
        transformer._recap_installed = True

    def _wrap(self, transformer) -> None:
        inner = transformer.forward

        def forward(*args, **kwargs):
            streams = self.calls_per_step
            stream = self.call_index % streams
            t = self.call_index // streams
            if t >= self.config.schedule.num_steps:
                raise RuntimeError(
                    f"transformer call {self.call_index} is past the configured horizon "
                    f"of {self.config.schedule.num_steps} steps x {streams} calls; "
                    "check schedule.num_steps and the pipeline's guidance settings"
                )
            started = time.perf_counter()
            if stream == 0:
                self._step_action = self.begin(t)
                self._step_gain = self.gain
                self._step_gain_num = 0.0
                self._step_gain_den = 0.0
                self._step_seconds = 0.0

            action = self._step_action

            if action == REUSE:
                out = self._reuse(t, stream)
                self._step_seconds += time.perf_counter() - started
                self.call_index += 1
                if stream == streams - 1:
                    entry = {
                        "t": t, "a": REUSE, "h": t - int(self.last_computed),
                        "g": round(self.gain, 4),
                        "sec": round(self._step_seconds, 4),
                    }
                    self.actions.append(entry)
                    self.step = t + 1
                return out

            self.operator.set_stream(stream)
            out = inner(*args, **kwargs)
            if self.config.sync_step_timing:
                torch.cuda.synchronize()
            self._step_seconds += time.perf_counter() - started

            with torch.no_grad():
                self._observe_stream(t, stream, _extract(out).detach().float())
            self.output_objects[stream] = out
            self.output_dtypes[stream] = _extract(out).dtype
            self.call_index += 1
            if stream == streams - 1:
                row = self.complete(t, action, self._finish_observation(t))
                row["sec"] = round(self._step_seconds, 3)
                self.actions.append(row)
                self.step = t + 1
            return out

        transformer.forward = forward
        self._wrapped_forward = forward

    def _after_refresh(self, t: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        measured = self.operator.drain_validation()
        if measured is not None:
            age, error = measured
            passed = self.controller.observe_validation(age, error)
            record = {"t": t, "age": age, "error": round(error, 6), "passed": passed}
            self.controller.validations.append(record)
            out["validation"] = record
        self.last_refresh = t
        last_step = self.config.schedule.num_steps - 1
        self.midpoint_step = int(math.ceil((t + last_step) / 2.0))
        self.midpoint_done = False
        return out

    def _reuse(self, t: int, stream: int) -> Any:
        cache = self.caches[stream]
        if cache is None or self.last_computed is None:
            raise RuntimeError("reuse requested before any computed step")
        h = t - int(self.last_computed)
        velocity = cache["v"]
        if self.gain and velocity is not None:
            corrected = (cache["o"] + self.gain * float(h) * velocity).to(
                self.output_dtypes[stream]
            )
            return _rebuild(self.output_objects[stream], corrected)
        return self.output_objects[stream]

    # -- records -----------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for row in self.actions:
            counts[row["a"]] = counts.get(row["a"], 0) + 1
        return {
            "model": dict(self.model_info),
            "counts": counts,
            "schedule": [row["a"] for row in self.actions],
            "per_step_sec": [row.get("sec") for row in self.actions],
            "reuse_gain_final": round(self.gain, 4),
            **self.controller.state(),
        }
