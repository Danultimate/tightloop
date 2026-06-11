"""The Loop engine.

Per iteration: policies → observe → plan (validated, retried) → per action:
hard-ceiling check → approval gate → enforced execution → record → progress →
exits. Hard ceilings are checked before every action and before granting any
approval; provider max_tokens is clamped to the remaining budget so no single
call can overshoot. Nested Loop.run() inside a tool raises.
"""
from __future__ import annotations

import contextvars
import json
import time
from typing import Any, Callable

from ..approval import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRunner,
    CLIApprovalRunner,
    HeadlessApprovalRunner,
    new_token,
)
from ..context import ContextManager
from ..exit import ExitCondition
from ..llm import LLMClient, LLMResponse, ToolCallReq, complete_with_retry
from ..policy import DecisionKind, NoProgress, Policy
from ..pricing import DEFAULT_PRICING, check_staleness, estimate_cost
from ..progress import GoalMetric, ProgressEngine
from ..tools import Tool, ToolRegistry, ToolValidationError
from ..trace import ExplainReport, TraceSink, explain
from .result import LoopResult, LoopStatus
from .state import (
    ActionRecord,
    ArtifactDriftError,
    IterationRecord,
    MetricSnapshot,
    PendingApproval,
    SchemaChangedError,
    State,
    digest,
    excerpt,
)

_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar("loop_active", default=False)

_CONFIG_KEYS = (
    "max_iterations",
    "token_limit",
    "wall_clock_s",
    "cost_limit_usd",
    "verbatim_window",
    "max_tokens_per_call",
)


class NestedLoopError(RuntimeError):
    """Loops may not be invoked from inside a tool. Delegate sub-tasks via a tool
    that returns a result instead."""


class LoopConfigError(ValueError):
    pass


class Loop:
    def __init__(
        self,
        goal: str,
        tools: list[Tool | Callable],
        llm: LLMClient,
        *,
        observe: Callable[[State], str] | None = None,
        goal_metric: GoalMetric | None = None,
        policies: list[Policy] | None = None,
        exits: list[ExitCondition] | None = None,
        max_iterations: int = 20,
        token_limit: int = 500_000,
        wall_clock_s: float = 1800.0,
        cost_limit_usd: float | None = None,
        pricing: dict | None = None,
        pricing_staleness: str = "warn",
        approval_runner: ApprovalRunner | None = None,
        approval_ttl_s: float = 3600.0,
        summarizer: LLMClient | None = None,
        verbatim_window: int = 3,
        max_tokens_per_call: int = 4096,
        state: State | None = None,
        state_path: str | None = None,
        trace_path: str | None = None,
        on_event: Callable[[dict], None] | None = None,
        allow_schema_change: bool = False,
        allow_artifact_drift: bool = False,
        quiet: bool = False,
    ):
        self.llm = llm
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.observe_fn = observe
        self.goal_metric = goal_metric
        self.policies = list(policies) if policies is not None else [NoProgress(3)]
        self.exits = list(exits or [])
        self.max_iterations = max_iterations
        self.token_limit = token_limit
        self.wall_clock_s = wall_clock_s
        self.cost_limit_usd = cost_limit_usd
        self.pricing = pricing or DEFAULT_PRICING
        self.approval_runner = approval_runner or CLIApprovalRunner()
        self.approval_ttl_s = approval_ttl_s
        self.max_tokens_per_call = max_tokens_per_call
        self.verbatim_window = verbatim_window
        self.state_path = state_path
        self.trace = TraceSink(trace_path, on_event)
        self.context = ContextManager(verbatim_window=verbatim_window, summarizer=summarizer)
        self.progress = ProgressEngine()
        self.quiet = quiet
        self._last_result: LoopResult | None = None
        self._session_start = time.monotonic()

        self._usd_enabled = True
        if cost_limit_usd is not None:
            self._usd_enabled = check_staleness(self.pricing, pricing_staleness)

        if state is not None:
            if state.tool_schema_hash != self.registry.schema_hash:
                if not allow_schema_change:
                    raise SchemaChangedError(
                        "tool schemas changed since this state was saved; pass "
                        "allow_schema_change=True to accept the new schemas"
                    )
                state.tool_schema_hash = self.registry.schema_hash
            drift = self.context.check_artifact_drift(state)
            if drift and not allow_artifact_drift:
                raise ArtifactDriftError(
                    "stored context artifacts were produced under a different configuration: "
                    + "; ".join(drift)
                    + " — pass allow_artifact_drift=True to reuse them anyway"
                )
            self.state = state
        else:
            self.state = State(goal=goal, tool_schema_hash=self.registry.schema_hash)
        self.state.config = {k: getattr(self, k) for k in _CONFIG_KEYS}

    # ---------------------------------------------------------------- run

    def run(self) -> LoopResult:
        if _ACTIVE.get():
            raise NestedLoopError(
                "Loop.run() called inside an active loop's tool execution. Delegate "
                "sub-tasks via a tool that returns a result; nested loops are unsupported in v1."
            )
        if self.state.pending_approval:
            raise LoopConfigError(
                "this state has a pending approval; use Loop.resume(path, approval={...})"
            )
        token = _ACTIVE.set(True)
        self._session_start = time.monotonic()
        self._announce()
        try:
            return self._loop()
        finally:
            _ACTIVE.reset(token)

    def _announce(self) -> None:
        limits = (
            f"{self.max_iterations} iterations, {self.token_limit:,} tokens, "
            f"{self.wall_clock_s:.0f}s wall-clock"
            + (f", ${self.cost_limit_usd:.2f} cost" if self.cost_limit_usd is not None else "")
        )
        if not self.quiet:
            print(f"[loop] goal={self.state.goal!r} | limits: {limits}")
        self.trace.emit("loop.start", goal=self.state.goal, limits=limits)

    # ------------------------------------------------------------- ceilings

    def _elapsed(self) -> float:
        return self.state.metrics.elapsed_s + (time.monotonic() - self._session_start)

    def _remaining_tokens(self) -> int:
        return self.token_limit - self.state.metrics.total_tokens

    def _ceiling(self) -> tuple[LoopStatus, str] | None:
        m = self.state.metrics
        if len(self.state.iterations) >= self.max_iterations:
            return LoopStatus.BUDGET_EXHAUSTED, f"max_iterations ({self.max_iterations}) reached"
        if m.total_tokens >= self.token_limit:
            return LoopStatus.BUDGET_EXHAUSTED, f"token_limit ({self.token_limit:,}) reached"
        if self._elapsed() >= self.wall_clock_s:
            return LoopStatus.BUDGET_EXHAUSTED, f"wall_clock limit ({self.wall_clock_s:.0f}s) reached"
        if self.cost_limit_usd is not None and self._usd_enabled and m.cost_usd is not None:
            if m.cost_usd >= self.cost_limit_usd:
                return LoopStatus.BUDGET_EXHAUSTED, (
                    f"cost_limit (${self.cost_limit_usd:.2f}) reached — estimate; tokens authoritative"
                )
        return None

    # ----------------------------------------------------------- main loop

    def _loop(self) -> LoopResult:
        while True:
            ceiling = self._ceiling()
            if ceiling:
                return self._finish(*ceiling)

            for p in self.policies:
                d = p.before_iteration(self.state)
                if d.kind == DecisionKind.STOP:
                    self.trace.emit("policy.stop", policy=type(p).__name__, reason=d.reason)
                    return self._finish(d.status or LoopStatus.ERROR, d.reason)

            index = len(self.state.iterations)
            obs = self._observe()
            metric = self._measure(obs)
            self.trace.emit("iteration.start", index=index,
                            metric=metric.model_dump() if metric else None)

            if metric and self.goal_metric and self.goal_metric.is_success(metric):
                self._record(IterationRecord(index=index, observation=obs, metric=metric))
                return self._finish(LoopStatus.SUCCESS, "goal metric reports success")

            planned = self._plan(obs)
            if planned is None:  # 3 validation failures this iteration
                it = IterationRecord(index=index, observation=obs, plan_invalid=True, metric=metric)
                self._record(it)
                if self.state.plan_invalid_streak >= 2:
                    return self._finish(
                        LoopStatus.PLAN_FAILED,
                        "two consecutive iterations failed tool-argument validation "
                        "(check tool schemas and prompt)",
                    )
                continue

            resp, calls = planned
            actions: list[ActionRecord] = []
            for call in calls:
                ceiling = self._ceiling()
                if ceiling:
                    self.trace.emit("budget.preempt", dropped_action=call.name)
                    it = IterationRecord(index=index, observation=obs, plan_text=resp.text,
                                         actions=actions, metric=metric)
                    self._record(it)
                    return self._finish(*ceiling)

                pause = self._gate(call)
                if pause is not None:
                    outcome = self._handle_approval(call, pause, metric, resp.text, index, obs, actions)
                    if isinstance(outcome, LoopResult):
                        return outcome
                actions.append(self._execute(call))

            it = IterationRecord(
                index=index, observation=obs, plan_text=resp.text, actions=actions,
                metric=metric, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            )
            self._record(it)

            for ex in self.exits:
                hit = ex.evaluate(self.state)
                if hit:
                    return self._finish(*hit)

    # ------------------------------------------------------------- helpers

    def _observe(self) -> str:
        raw = self.observe_fn(self.state) if self.observe_fn else ""
        return excerpt(str(raw))

    def _measure(self, obs: str) -> MetricSnapshot | None:
        return self.goal_metric.measure(obs, self.state) if self.goal_metric else None

    def _plan(self, obs: str) -> tuple[LLMResponse, list[ToolCallReq]] | None:
        """One planning call; invalid tool args are fed back as structured errors,
        retry budget 2. Returns None after 3 failed validations."""
        messages = self.context.build(self.state, obs)
        for attempt in range(3):
            max_toks = max(16, min(self.max_tokens_per_call, self._remaining_tokens()))
            resp = complete_with_retry(self.llm, messages, self.registry.schemas, max_toks)
            m = self.state.metrics
            m.input_tokens += resp.input_tokens
            m.output_tokens += resp.output_tokens
            m.llm_calls += 1
            if self._usd_enabled:
                m.cost_usd = estimate_cost(m.input_tokens, m.output_tokens,
                                           self.llm.model_id, self.pricing)
            self.trace.emit("llm.call", model=resp.model_id or self.llm.model_id,
                            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                            attempt=attempt)

            errors: list[str] = []
            validated: list[ToolCallReq] = []
            for call in resp.tool_calls:
                t = self.registry.get(call.name)
                if t is None:
                    errors.append(f"unknown tool {call.name!r}")
                    continue
                try:
                    validated.append(ToolCallReq(id=call.id, name=call.name, args=t.validate(call.args)))
                except ToolValidationError as e:
                    errors.append(str(e))
            if not errors:
                return resp, validated
            self.trace.emit("plan.invalid", attempt=attempt, errors=errors)
            messages = messages + [
                {"role": "assistant", "content": resp.text or json.dumps(
                    [c.model_dump() for c in resp.tool_calls])},
                {"role": "user", "content": "Tool call rejected:\n" + "\n".join(errors)
                    + "\nRetry with valid arguments matching the tool schemas."},
            ]
        return None

    def _gate(self, call: ToolCallReq) -> str | None:
        for p in self.policies:
            d = p.before_action(self.state, call.name, call.args)
            if d.kind == DecisionKind.PAUSE:
                return d.reason
            if d.kind == DecisionKind.STOP:
                return d.reason  # treated as approval-style gate; engine pauses
        return None

    def _execute(self, call: ToolCallReq) -> ActionRecord:
        t = self.registry.get(call.name)
        result = self.registry.execute(call.name, call.args)
        record = ActionRecord(
            tool=call.name,
            args_excerpt=excerpt(json.dumps(call.args, default=str), 1024),
            status=result.status,
            result_excerpt=result.output,
            duration_s=result.duration_s,
            fingerprint=t.fingerprint(call.args),
        )
        self.trace.emit("action.executed", tool=call.name, status=result.status,
                        duration_s=round(result.duration_s, 3))
        return record

    def _action_hash(self, call: ToolCallReq) -> str:
        return digest(call.name + json.dumps(call.args, sort_keys=True, default=str)
                      + str(self.state.state_version))

    def _handle_approval(
        self,
        call: ToolCallReq,
        reason: str,
        metric: MetricSnapshot | None,
        plan_text: str,
        index: int,
        obs: str,
        actions_so_far: list[ActionRecord],
    ) -> LoopResult | None:
        """Returns None if approved (caller executes), or a LoopResult to return."""
        ttl = getattr(self.approval_runner, "ttl_s", self.approval_ttl_s)
        req = ApprovalRequest(
            token=new_token(), tool=call.name, args=call.args, reason=reason,
            action_hash=self._action_hash(call), state_version=self.state.state_version,
            created_at=time.time(), ttl_s=ttl,
        )
        self.trace.emit("approval.requested", tool=call.name, token=req.token, reason=reason)
        decision, note = self.approval_runner.request(req)
        self.trace.emit("approval.decision", token=req.token, decision=decision.value, note=note)

        if decision == ApprovalDecision.APPROVED:
            return None
        if decision == ApprovalDecision.DENIED:
            it = IterationRecord(index=index, observation=obs, plan_text=plan_text,
                                 actions=actions_so_far, metric=metric)
            self._record(it)
            return self._finish(LoopStatus.APPROVAL_DENIED, note)

        # PENDING: serialize and hand back a resume token
        if not self.state_path:
            raise LoopConfigError("headless approval requires state_path= so the loop can pause")
        if actions_so_far or plan_text:
            it = IterationRecord(index=index, observation=obs, plan_text=plan_text,
                                 actions=actions_so_far, metric=metric)
            self._record(it)
        self.state.pending_approval = PendingApproval(
            token=req.token, tool=call.name, args=call.args, reason=reason,
            action_hash=req.action_hash, state_version=req.state_version,
            created_at=req.created_at, ttl_s=req.ttl_s, plan_text=plan_text,
            precondition_metric=metric,
        )
        return self._finish(LoopStatus.AWAITING_APPROVAL,
                            f"action {call.name!r} awaits approval (token {req.token})",
                            approval_token=req.token)

    def _record(self, it: IterationRecord) -> None:
        report = self.progress.evaluate(self.state, it)
        if report.trend == "regressing" and it.plan_text:
            self.state.failed_approaches.append(
                f"iteration {it.index}: {it.plan_text[:160]} -> metric regressed"
            )
        self.state.iterations.append(it)
        self.state.state_version += 1
        created = self.context.ensure_summaries(self.state)
        self.trace.emit(
            "iteration.end", index=it.index, trend=report.trend,
            repetition=report.repetition, no_progress_streak=report.no_progress_streak,
            summarized_iterations=created,
            accounting=self.context.budget_report(self.state),
        )
        if self.state_path:
            self._sync_elapsed()
            self.state.save(self.state_path)

    def _sync_elapsed(self) -> None:
        now = time.monotonic()
        self.state.metrics.elapsed_s += now - self._session_start
        self._session_start = now

    def _finish(self, status: LoopStatus, reason: str, approval_token: str | None = None) -> LoopResult:
        self._sync_elapsed()
        if status != LoopStatus.AWAITING_APPROVAL and self.state.pending_approval:
            self.trace.emit("approval.cancelled", token=self.state.pending_approval.token,
                            reason=f"loop ended: {reason}")
            self.state.pending_approval = None
        if self.state_path:
            self.state.save(self.state_path)
        result = LoopResult.make(status, reason, len(self.state.iterations),
                                 self.state.metrics, state_path=self.state_path,
                                 approval_token=approval_token)
        self.trace.emit("loop.end", status=status.value, reason=reason,
                        iterations=result.iterations,
                        total_tokens=self.state.metrics.total_tokens)
        self._last_result = result
        return result

    # ------------------------------------------------------------- public

    def explain(self, result: LoopResult | None = None) -> ExplainReport:
        return explain(self.state, result or self._last_result)

    def budget_report(self) -> dict[str, Any]:
        return self.context.budget_report(self.state)

    # ------------------------------------------------------------- resume

    @classmethod
    def resume(
        cls,
        state_path: str,
        *,
        tools: list[Tool | Callable],
        llm: LLMClient,
        approval: dict[str, Any] | None = None,
        extend: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LoopResult:
        state = State.load(state_path)
        if extend:
            unknown = set(extend) - set(_CONFIG_KEYS)
            if unknown:
                raise LoopConfigError(f"extend has unknown keys: {sorted(unknown)}")
            state.config.update(extend)
        for key in _CONFIG_KEYS:
            if key in state.config:
                kwargs.setdefault(key, state.config[key])
        self = cls(goal=state.goal, tools=tools, llm=llm, state=state,
                   state_path=state_path, **kwargs)

        if state.pending_approval:
            outcome = self._resume_pending(approval)
            if outcome is not None:
                return outcome

        token = _ACTIVE.set(True)
        self._session_start = time.monotonic()
        self._announce()
        try:
            return self._loop()
        finally:
            _ACTIVE.reset(token)

    def _resume_pending(self, approval: dict[str, Any] | None) -> LoopResult | None:
        """Handles a pending approval on resume. Returns a LoopResult to stop with,
        or None to continue looping (approved action already executed)."""
        pa = self.state.pending_approval
        if time.time() - pa.created_at > pa.ttl_s:
            self.trace.emit("approval.expired", token=pa.token)
            self.state.pending_approval = None
            return self._finish(LoopStatus.PENDING_EXPIRED,
                                f"approval token {pa.token} expired after {pa.ttl_s:.0f}s")
        if approval is None:
            raise LoopConfigError(
                "state has a pending approval; pass approval={'token': ..., 'approved': bool}"
            )
        if approval.get("token") != pa.token:
            raise LoopConfigError("approval token does not match the pending request")
        if not approval.get("approved"):
            self.state.pending_approval = None
            return self._finish(LoopStatus.APPROVAL_DENIED, "denied via resume token")

        # Re-observe: stale precondition invalidates the approval
        obs = self._observe()
        metric = self._measure(obs)
        pre = pa.precondition_metric
        if pre is not None and metric is not None and metric.model_dump() != pre.model_dump():
            fresh = new_token()
            self.trace.emit("approval.invalidated", old_token=pa.token, new_token=fresh,
                            reason="precondition changed since approval was requested")
            self.state.pending_approval = pa.model_copy(
                update={"token": fresh, "created_at": time.time(), "precondition_metric": metric}
            )
            return self._finish(
                LoopStatus.AWAITING_APPROVAL,
                "the situation changed since this approval was requested (goal metric "
                f"differs); approval re-requested with new token {fresh}",
                approval_token=fresh,
            )

        call = ToolCallReq(name=pa.tool, args=pa.args)
        action = self._execute(call)
        it = IterationRecord(index=len(self.state.iterations), observation=obs,
                             plan_text=pa.plan_text or f"approved action {pa.tool}",
                             actions=[action], metric=metric)
        self.state.pending_approval = None
        self._record(it)
        return None
