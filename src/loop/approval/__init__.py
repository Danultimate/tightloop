"""Human approval checkpoints.

ApprovalRequest is frozen at type level: callbacks get a
read-only payload — action, args, reason, digests — never the full context.
Callback runner: 60s timeout, deny-on-exception, every invocation traced by the engine.
"""
from __future__ import annotations

import secrets
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict


class ApprovalDecision(str, Enum):
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    PENDING = "PENDING"  # headless: serialize state, resume by token


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    token: str
    tool: str
    args: dict[str, Any]
    reason: str
    action_hash: str
    state_version: int
    created_at: float
    ttl_s: float


def new_token() -> str:
    return secrets.token_urlsafe(8)


class ApprovalRunner:
    def request(self, req: ApprovalRequest) -> tuple[ApprovalDecision, str]:
        raise NotImplementedError


class CLIApprovalRunner(ApprovalRunner):
    """Interactive default: prompts on stdin."""

    def request(self, req: ApprovalRequest) -> tuple[ApprovalDecision, str]:
        print(f"\n[loop] approval required: {req.tool}({req.args})\nreason: {req.reason}")
        answer = input("approve? [y/N] ").strip().lower()
        if answer in ("y", "yes"):
            return ApprovalDecision.APPROVED, "approved via CLI"
        return ApprovalDecision.DENIED, "denied via CLI"


class CallbackApprovalRunner(ApprovalRunner):
    def __init__(self, fn: Callable[[ApprovalRequest], bool], timeout_s: float = 60.0):
        self.fn = fn
        self.timeout_s = timeout_s

    def request(self, req: ApprovalRequest) -> tuple[ApprovalDecision, str]:
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self.fn, req)
            try:
                approved = future.result(timeout=self.timeout_s)
            except FutureTimeout:
                return ApprovalDecision.DENIED, f"callback timed out after {self.timeout_s}s (deny-on-timeout)"
            except Exception as e:
                return ApprovalDecision.DENIED, f"callback raised {type(e).__name__}: {e} (deny-on-exception)"
            if approved:
                return ApprovalDecision.APPROVED, "approved via callback"
            return ApprovalDecision.DENIED, "denied via callback"
        finally:
            executor.shutdown(wait=False)


class HeadlessApprovalRunner(ApprovalRunner):
    """Always returns PENDING: the engine serializes state and exits AWAITING_APPROVAL;
    resume with Loop.resume(path, approval={'token': ..., 'approved': True})."""

    def __init__(self, ttl_s: float = 3600.0):
        self.ttl_s = ttl_s

    def request(self, req: ApprovalRequest) -> tuple[ApprovalDecision, str]:
        return ApprovalDecision.PENDING, f"awaiting approval, token={req.token}"
