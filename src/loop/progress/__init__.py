"""Progress engine: raw signals, no fake-precision score.

Signals: blueprint goal metric (regression-aware), repetition detection via
per-tool fingerprints (advisory), LLM self-assessment (trace-only, never gates
exits in v1 — by design).
"""
from __future__ import annotations

from pydantic import BaseModel

from ..core.state import IterationRecord, MetricSnapshot, State


class GoalMetric:
    """Blueprint-supplied. measure() returns a snapshot; is_success() gates SUCCESS."""

    def measure(self, observation: str, state: State) -> MetricSnapshot:
        raise NotImplementedError

    def is_success(self, snapshot: MetricSnapshot) -> bool:
        return False


class ProgressReport(BaseModel):
    trend: str  # improving | flat | regressing | unknown
    repetition: bool
    no_progress_streak: int
    metric_delta: float | None = None


class ProgressEngine:
    def evaluate(self, state: State, iteration: IterationRecord) -> ProgressReport:
        prev = state.iterations[-1] if state.iterations else None

        # repetition: identical fingerprint set to the previous iteration's (non-empty)
        fps = {a.fingerprint for a in iteration.actions}
        prev_fps = {a.fingerprint for a in prev.actions} if prev else set()
        iteration.repetition = bool(fps) and fps == prev_fps

        delta: float | None = None
        if iteration.metric and prev and prev.metric:
            delta = iteration.metric.value - prev.metric.value

        if iteration.metric and iteration.metric.regression:
            trend = "regressing"
        elif delta is None:
            trend = "unknown"
        elif delta > 0:
            trend = "improving"
        elif delta < 0:
            trend = "regressing"
        else:
            trend = "flat"

        flat = delta is None or delta == 0
        flagged = iteration.repetition or iteration.plan_invalid
        if flagged and flat:
            state.no_progress_streak += 1
        else:
            state.no_progress_streak = 0

        if iteration.plan_invalid:
            state.plan_invalid_streak += 1
        else:
            state.plan_invalid_streak = 0

        return ProgressReport(
            trend=trend,
            repetition=iteration.repetition,
            no_progress_streak=state.no_progress_streak,
            metric_delta=delta,
        )
