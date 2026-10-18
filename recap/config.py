"""Configuration.

A config file gives the model, the preset and the operating point (``B``, ``tau_pred``,
``tau_detail``).  All other settings come from the preset's profile in
``recap/profiles/``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

PRESETS = ("normal_slow", "normal_fast", "fast")
MODELS = ("cogvideox1.5", "wan2.1-1.3b", "hunyuan")


@dataclass(frozen=True)
class OperatingPoint:
    """Budget ``B`` and the two error tolerances."""

    B: int
    tau_pred: float
    tau_detail: float

    def validate(self) -> None:
        if type(self.B) is not int or self.B < 3:
            raise ValueError("operating_point.B must be an integer of at least 3")
        for name in ("tau_pred", "tau_detail"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"operating_point.{name} must be finite and positive")


@dataclass(frozen=True)
class GeometryConfig:
    """Token layout of a model."""

    grid: tuple[int, ...]
    text_tokens: int = 226
    block_size: int = 128

    def validate(self) -> None:
        if len(self.grid) != 3 or any(int(x) <= 0 for x in self.grid):
            raise ValueError("grid must be three positive integers (T, H, W)")
        if self.text_tokens < 0:
            raise ValueError("text_tokens must be non-negative")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.block_size != 128:
            raise ValueError("the exact block-sparse kernel requires block_size=128")


@dataclass(frozen=True)
class OperatorConfig:
    """Sparse attention settings."""

    head_density: float = 0.10  # fraction of key blocks selected per query block
    closure_gate_theta: float = 2.0  # gate of the key-variance correction (nats)
    closure_rank: int = 4  # rank of the per-block key variance
    rerank_min_overlap: float = 0.95
    host_banks: bool = False
    prefetch_banks: bool = True
    bank_chunks: int = 1

    block_size: int = 128

    def validate(self) -> None:
        if not 0.0 < self.head_density < 1.0:
            raise ValueError("head_density must be in (0,1)")
        if self.closure_gate_theta <= 0:
            raise ValueError("closure_gate_theta must be positive")
        if self.closure_rank < 0:
            raise ValueError("closure_rank must be non-negative")
        if not 0.0 <= self.rerank_min_overlap <= 1.0:
            raise ValueError("rerank_min_overlap must lie in [0,1]")
        if self.bank_chunks < 1:
            raise ValueError("bank_chunks must be at least 1")


@dataclass(frozen=True)
class ScheduleConfig:
    """Scheduling settings."""

    num_steps: int = 50
    tau: float = 0.10  # tau_pred
    detail_tau_mult: float = 1.0  # tau_detail / tau_pred
    budget: int | None = None  # B
    prefix_floor: int = 0
    prefix_cap: int = 0
    tail_exact: int = 0

    def validate(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.detail_tau_mult <= 0.0:
            raise ValueError("detail_tau_mult must be positive")
        if self.budget is not None and self.budget < 3:
            raise ValueError("budget must allow at least 3 computed endpoints")
        if self.prefix_cap and self.prefix_cap < self.prefix_floor:
            raise ValueError("prefix_cap must not be below prefix_floor")
        if self.prefix_floor < 0 or self.tail_exact < 0:
            raise ValueError("prefix_floor and tail_exact must be non-negative")
        if self.budget is not None and self.tail_exact >= self.budget:
            raise ValueError("tail_exact must leave room in the budget")
        if self.tail_exact >= self.num_steps:
            raise ValueError("tail_exact must be smaller than num_steps")


@dataclass(frozen=True)
class RecapConfig:
    model: str = "cogvideox1.5"
    geometry: GeometryConfig = None  # type: ignore[assignment]
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    sync_step_timing: bool = True

    @property
    def operating_point(self) -> OperatingPoint:
        if self.schedule.budget is None:
            raise ValueError("an operating point requires a finite budget")
        return OperatingPoint(self.schedule.budget, self.schedule.tau,
                              self.schedule.tau * self.schedule.detail_tau_mult)

    def validate(self) -> "RecapConfig":
        self.geometry.validate()
        self.operator.validate()
        self.schedule.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["geometry"]["grid"] = list(self.geometry.grid)
        return payload


def _section(raw: dict[str, Any], name: str, allowed: type) -> dict[str, Any]:
    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise TypeError(f"config section {name!r} must be an object")
    known = {f.name for f in fields(allowed)}
    unknown = set(value) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in config section {name!r}: {sorted(unknown)}; "
            f"known keys are {sorted(known)}"
        )
    return dict(value)


PACKAGE_DIR = Path(__file__).resolve().parent


def config_from_dict(raw: dict[str, Any]) -> RecapConfig:
    """Build a config from the contents of a config file."""
    unknown = set(raw) - {"model", "preset", "operating_point", "example"}
    if unknown:
        raise ValueError(
            f"unknown config keys: {sorted(unknown)}; a config gives model, preset, "
            "operating_point and optionally example"
        )
    model, preset = raw.get("model"), raw.get("preset")
    if model not in MODELS or preset not in PRESETS:
        raise ValueError(f"unsupported model/preset: {model!r}/{preset!r}")
    point_raw = raw.get("operating_point")
    if not isinstance(point_raw, dict) or set(point_raw) != {"B", "tau_pred", "tau_detail"}:
        raise ValueError("operating_point must contain exactly B, tau_pred, tau_detail")
    point = OperatingPoint(**point_raw)
    point.validate()
    profile = json.loads((PACKAGE_DIR / "profiles" / f"{model}_{preset}.json").read_text())
    if point.B > profile["schedule"]["num_steps"]:
        raise ValueError("operating_point.B cannot exceed num_steps")
    profile["schedule"].update(budget=point.B, tau=point.tau_pred,
                               detail_tau_mult=point.tau_detail / point.tau_pred)
    return _from_profile(model, profile)


def _from_profile(model: str, raw: dict[str, Any]) -> RecapConfig:
    unknown = set(raw) - {"model", "geometry", "operator", "schedule", "sync_step_timing"}
    if unknown:
        raise ValueError(f"unknown profile keys: {sorted(unknown)}")
    geometry_raw = _section(raw, "geometry", GeometryConfig)
    if "grid" not in geometry_raw:
        raise ValueError("geometry.grid is required")
    geometry = GeometryConfig(
        grid=tuple(int(x) for x in geometry_raw["grid"]),
        text_tokens=int(geometry_raw.get("text_tokens", 226)),
        block_size=int(geometry_raw.get("block_size", 128)),
    )
    operator_raw = _section(raw, "operator", OperatorConfig)
    operator = OperatorConfig(
        **{k: v for k, v in operator_raw.items() if k != "block_size"},
        block_size=geometry.block_size,
    )
    schedule_raw = _section(raw, "schedule", ScheduleConfig)
    budget = schedule_raw.get("budget")
    schedule = ScheduleConfig(
        **{k: v for k, v in schedule_raw.items() if k != "budget"},
        budget=None if budget is None else int(budget),
    )
    return RecapConfig(
        model=str(model),
        geometry=geometry,
        operator=operator,
        schedule=schedule,
        sync_step_timing=bool(raw.get("sync_step_timing", True)),
    ).validate()


def load_config(path: str | Path) -> RecapConfig:
    """Read a config JSON file."""
    return config_from_dict(json.loads(Path(path).read_text()))
