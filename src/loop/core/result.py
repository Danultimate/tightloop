"""LoopResult status matrix — every status is explicit and actionable."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from .state import Metrics


class LoopStatus(str, Enum):
    SUCCESS = "SUCCESS"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    NO_PROGRESS = "NO_PROGRESS"
    PLAN_FAILED = "PLAN_FAILED"
    APPROVAL_DENIED = "APPROVAL_DENIED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    PENDING_EXPIRED = "PENDING_EXPIRED"
    ERROR = "ERROR"


STATUS_INFO: dict[LoopStatus, tuple[bool, str]] = {
    LoopStatus.SUCCESS: (False, "done"),
    LoopStatus.BUDGET_EXHAUSTED: (True, "inspect snapshot; Loop.resume(..., extend={...}) with a larger budget"),
    LoopStatus.NO_PROGRESS: (True, "change tools, goal, or limits, then Loop.resume(...)"),
    LoopStatus.PLAN_FAILED: (True, "fix tool schemas or prompt, then Loop.resume(...)"),
    LoopStatus.APPROVAL_DENIED: (True, "adjust plan or policy, then Loop.resume(...)"),
    LoopStatus.AWAITING_APPROVAL: (True, "approve via token: Loop.resume(path, approval={'token': ..., 'approved': True})"),
    LoopStatus.PENDING_EXPIRED: (True, "resume to re-request approval"),
    LoopStatus.ERROR: (False, "inspect trace via loop.explain()"),
}


class LoopResult(BaseModel):
    status: LoopStatus
    reason: str = ""
    resumable: bool
    recommended_action: str
    iterations: int
    metrics: Metrics
    state_path: str | None = None
    approval_token: str | None = None

    @classmethod
    def make(
        cls,
        status: LoopStatus,
        reason: str,
        iterations: int,
        metrics: Metrics,
        state_path: str | None = None,
        approval_token: str | None = None,
    ) -> "LoopResult":
        resumable, action = STATUS_INFO[status]
        return cls(
            status=status,
            reason=reason,
            resumable=resumable,
            recommended_action=action,
            iterations=iterations,
            metrics=metrics,
            state_path=state_path,
            approval_token=approval_token,
        )
