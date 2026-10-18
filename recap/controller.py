"""Controller: decides when to compute and which computation to run.

It reads two errors measured during generation:

``m``         prediction mismatch at a computed step; ``m / h`` is its rate over an
              interval of ``h`` steps.
``e_detail``  refinement error measured at a dense refresh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import ScheduleConfig

EARN = 2  # growth of an interval after a successful check


@dataclass
class Controller:
    """Scheduling state for one sample."""

    config: ScheduleConfig

    streak: int = 0
    licensed: bool = False  # reuse has started
    next_gap: int | None = None  # next reuse interval
    detail_lease: int | None = None  # oldest refinement age usable by sparse steps
    lease_span: int = EARN
    rate_last: float | None = None
    rate_sum: float = 0.0
    rate_obs: int = 0
    computed: int = 0  # transformer evaluations so far

    validations: list[dict] = field(default_factory=list)

    # -- errors ------------------------------------------------------------------------

    @property
    def lease_tolerance(self) -> float:
        """tau_detail."""
        return self.config.tau * self.config.detail_tau_mult

    def observe_residual(self, h: int, m: float, t: int | None = None) -> None:
        """Update the reuse interval from the mismatch ``m`` over ``h`` steps."""
        tau = self.config.tau
        if h <= 0 or not math.isfinite(m) or m <= 0.0:
            return
        if not self.licensed:
            # Reuse starts after two consecutive one-step mismatches below tau.
            if h != 1:
                return
            self.streak = self.streak + 1 if m < tau else 0
            if self.streak >= 2 and (t is None or t >= self.config.prefix_floor):
                self.license()
            return
        rate = m / float(h)
        self.rate_last = rate
        self.rate_sum += rate
        self.rate_obs += 1
        if h >= EARN:
            # A passing interval grows by 2, a failing one stays.
            self.next_gap = h + EARN if rate < tau else h

    def observe_validation(self, age: int, error: float) -> bool:
        """Grant or revoke the refinement span from ``e_detail``."""
        passed = math.isfinite(error) and error < self.lease_tolerance
        if passed:
            self.detail_lease = age + self.lease_span
            self.lease_span += EARN
        else:
            self.detail_lease = None
            self.lease_span = EARN
        return passed

    def lease_ok(self, age: int) -> bool:
        """Whether a sparse step may use a refinement of this age."""
        return self.detail_lease is not None and age <= self.detail_lease

    # -- placement ---------------------------------------------------------------------

    def license(self) -> None:
        """Start reuse at the shortest interval."""
        self.licensed = True
        self.next_gap = EARN

    def in_exact_tail(self, t: int) -> bool:
        tail = self.config.tail_exact
        return tail > 0 and t >= self.config.num_steps - tail

    def pacing_gap(self, t: int) -> int | None:
        """Interval that spends the remaining budget over the remaining steps."""
        budget = self.config.budget
        if budget is None:
            return None
        left = budget - self.computed - self.config.tail_exact
        if left <= 0:
            return None
        base = max(1, math.ceil((self.config.num_steps - self.config.tail_exact - t) / left))
        if self.rate_obs and self.rate_last:
            # Shorter intervals when the mismatch rate is above its mean, never longer.
            mean = self.rate_sum / self.rate_obs
            base = min(base, max(1, round(base * mean / max(self.rate_last, 1e-12))))
        return base

    def allowed_gap(self, t: int) -> int:
        if self.next_gap is None:
            raise RuntimeError("allowed_gap queried before reuse was licensed")
        pace = self.pacing_gap(t)
        return self.next_gap if pace is None else min(self.next_gap, pace)

    def budget_exhausted(self) -> bool:
        budget = self.config.budget
        return budget is not None and self.computed >= budget

    def mid_budget_exhausted(self) -> bool:
        budget = self.config.budget
        return budget is not None and \
            self.computed >= budget - self.config.tail_exact

    def prefix_overspent(self) -> bool:
        budget = self.config.budget
        return budget is not None and 2 * self.computed >= budget

    def state(self) -> dict:
        return {
            "computed_endpoints": self.computed,
            "budget": self.config.budget,
            "validation_history": list(self.validations),
        }
