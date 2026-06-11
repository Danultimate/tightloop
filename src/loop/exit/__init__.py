"""Declarative exit conditions. Evaluated after each iteration;
first matching exit wins. The always-on ceilings (max_iterations, token_limit,
wall-clock) are engine constructor args — these are additional conditions."""
from __future__ import annotations

from typing import Callable

from ..core.result import LoopStatus
from ..core.state import State


class ExitCondition:
    def evaluate(self, state: State) -> tuple[LoopStatus, str] | None:
        raise NotImplementedError


class _Lambda(ExitCondition):
    def __init__(self, fn: Callable[[State], tuple[LoopStatus, str] | None]):
        self.fn = fn

    def evaluate(self, state: State) -> tuple[LoopStatus, str] | None:
        return self.fn(state)


class Exit:
    @staticmethod
    def success(predicate: Callable[[State], bool], reason: str = "goal achieved") -> ExitCondition:
        return _Lambda(lambda s: (LoopStatus.SUCCESS, reason) if predicate(s) else None)

    @staticmethod
    def max_iterations(n: int = 20) -> ExitCondition:
        return _Lambda(
            lambda s: (LoopStatus.BUDGET_EXHAUSTED, f"max_iterations {n} reached")
            if len(s.iterations) >= n
            else None
        )

    @staticmethod
    def token_limit(n: int) -> ExitCondition:
        return _Lambda(
            lambda s: (LoopStatus.BUDGET_EXHAUSTED, f"token_limit {n} reached")
            if s.metrics.total_tokens >= n
            else None
        )

    @staticmethod
    def cost_limit(usd: float) -> ExitCondition:
        return _Lambda(
            lambda s: (LoopStatus.BUDGET_EXHAUSTED, f"cost_limit ${usd:.2f} reached")
            if s.metrics.cost_usd is not None and s.metrics.cost_usd >= usd
            else None
        )

    @staticmethod
    def stagnation(n: int = 3) -> ExitCondition:
        return _Lambda(
            lambda s: (LoopStatus.NO_PROGRESS, f"stagnation: {s.no_progress_streak} flat iterations")
            if s.no_progress_streak >= n
            else None
        )
