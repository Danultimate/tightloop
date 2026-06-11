"""Composable policies.

Precedence note: hard ceilings (iterations/tokens/wall-clock) are enforced by
the engine itself before every action and before granting any approval —
policies layer on top of those guarantees, they don't replace them.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Iterable

from pydantic import BaseModel

from ..core.result import LoopStatus
from ..core.state import State
from ..pricing import DEFAULT_PRICING, estimate_cost


class DecisionKind(str, Enum):
    CONTINUE = "CONTINUE"
    STOP = "STOP"
    PAUSE = "PAUSE"  # request human approval


class Decision(BaseModel):
    kind: DecisionKind = DecisionKind.CONTINUE
    reason: str = ""
    status: LoopStatus | None = None  # for STOP


CONTINUE = Decision()


class Policy:
    def before_iteration(self, state: State) -> Decision:
        return CONTINUE

    def before_action(self, state: State, tool_name: str, args: dict[str, Any]) -> Decision:
        return CONTINUE


class NoProgress(Policy):
    """Fires after `window` consecutive flagged iterations with zero goal-metric delta
   . Streak accounting is done by the ProgressEngine; this reads it."""

    def __init__(self, window: int = 3):
        self.window = window

    def before_iteration(self, state: State) -> Decision:
        if state.no_progress_streak >= self.window:
            return Decision(
                kind=DecisionKind.STOP,
                status=LoopStatus.NO_PROGRESS,
                reason=(
                    f"{state.no_progress_streak} consecutive iterations with repeated/invalid "
                    "actions and zero goal-metric delta"
                ),
            )
        return CONTINUE


class CostLimit(Policy):
    def __init__(self, usd: float, model_id: str, pricing: dict | None = None):
        self.usd = usd
        self.model_id = model_id
        self.pricing = pricing or DEFAULT_PRICING

    def before_iteration(self, state: State) -> Decision:
        cost = estimate_cost(
            state.metrics.input_tokens, state.metrics.output_tokens, self.model_id, self.pricing
        )
        if cost is not None and cost >= self.usd:
            return Decision(
                kind=DecisionKind.STOP,
                status=LoopStatus.BUDGET_EXHAUSTED,
                reason=f"estimated cost ${cost:.2f} >= limit ${self.usd:.2f} (tokens are authoritative)",
            )
        return CONTINUE


class RequireApproval(Policy):
    """Pause for human approval when an action matches. Matcher: tool-name iterable
    or callable(tool_name, args) -> bool."""

    def __init__(self, matcher: Iterable[str] | Callable[[str, dict], bool], reason: str = "requires approval"):
        if callable(matcher):
            self._match = matcher
        else:
            names = set(matcher)
            self._match = lambda name, args: name in names
        self.reason = reason

    def before_action(self, state: State, tool_name: str, args: dict[str, Any]) -> Decision:
        if self._match(tool_name, args):
            return Decision(kind=DecisionKind.PAUSE, reason=self.reason)
        return CONTINUE
