"""Live-streamed structured trace + explain()."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from ..core.result import LoopResult
from ..core.state import State


class TraceSink:
    """Appends each event to a JSONL file (live) and forwards to an optional callback."""

    def __init__(self, path: str | Path | None = None, on_event: Callable[[dict], None] | None = None):
        self.path = Path(path) if path else None
        self.on_event = on_event

    def emit(self, kind: str, **data: Any) -> None:
        event = {"ts": time.time(), "kind": kind, **data}
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass  # user callback failures never take down the loop


class ExplainReport(BaseModel):
    status: str
    reason: str
    signals: dict[str, Any] = Field(default_factory=dict)
    decision_chain: list[str] = Field(default_factory=list)

    def render(self) -> str:
        lines = [f"# Why the loop stopped: {self.status}", "", f"**Reason:** {self.reason}", "", "## Signals"]
        lines += [f"- {k}: {v}" for k, v in self.signals.items()]
        lines += ["", "## Decision chain"]
        lines += [f"{i + 1}. {step}" for i, step in enumerate(self.decision_chain)]
        return "\n".join(lines)


def explain(state: State, result: LoopResult | None = None) -> ExplainReport:
    """Structured 'why did it stop' report; callable live or post-hoc."""
    last_metric = next((it.metric for it in reversed(state.iterations) if it.metric), None)
    signals: dict[str, Any] = {
        "iterations": len(state.iterations),
        "input_tokens": state.metrics.input_tokens,
        "output_tokens": state.metrics.output_tokens,
        "llm_calls": state.metrics.llm_calls,
        "elapsed_s": round(state.metrics.elapsed_s, 2),
        "cost_usd_estimate": state.metrics.cost_usd,
        "no_progress_streak": state.no_progress_streak,
        "plan_invalid_streak": state.plan_invalid_streak,
        "last_metric": last_metric.model_dump() if last_metric else None,
        "failed_approaches": len(state.failed_approaches),
    }
    chain: list[str] = []
    for it in state.iterations:
        bits = [f"iteration {it.index}:"]
        if it.plan_invalid:
            bits.append("plan failed validation;")
        for a in it.actions:
            bits.append(f"{a.tool}[{a.status}]")
        if it.metric:
            bits.append(f"metric={it.metric.value}")
            if it.metric.regression:
                bits.append("(REGRESSION)")
        if it.repetition:
            bits.append("(repetition flagged)")
        chain.append(" ".join(bits))
    status = result.status.value if result else "RUNNING"
    reason = result.reason if result else "loop has not exited"
    if result:
        chain.append(f"exit: {status} — {reason}")
    return ExplainReport(status=status, reason=reason, signals=signals, decision_chain=chain)
