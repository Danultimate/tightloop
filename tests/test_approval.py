import time

import pytest
from pydantic import ValidationError

from loop import (
    ApprovalRequest,
    CallbackApprovalRunner,
    HeadlessApprovalRunner,
    Loop,
    LoopStatus,
    RequireApproval,
)
from conftest import ScriptedLLM, call, make_env


def test_callback_deny_ends_loop():
    env, fix, observe, metric = make_env({"t1"})
    llm = ScriptedLLM([call("fix", name="t1")])
    loop = Loop(goal="fix", tools=[fix], llm=llm, observe=observe, goal_metric=metric,
                policies=[RequireApproval({"fix"})],
                approval_runner=CallbackApprovalRunner(lambda req: False), quiet=True)
    result = loop.run()
    assert result.status == LoopStatus.APPROVAL_DENIED
    assert result.resumable
    assert env["failing"] == {"t1"}  # the action never ran


def test_callback_exception_denies():
    env, fix, observe, metric = make_env({"t1"})

    def bad_callback(req):
        raise RuntimeError("approval service down")

    llm = ScriptedLLM([call("fix", name="t1")])
    loop = Loop(goal="fix", tools=[fix], llm=llm, observe=observe, goal_metric=metric,
                policies=[RequireApproval({"fix"})],
                approval_runner=CallbackApprovalRunner(bad_callback), quiet=True)
    result = loop.run()
    assert result.status == LoopStatus.APPROVAL_DENIED
    assert "deny-on-exception" in result.reason


def test_approval_request_is_frozen():
    req = ApprovalRequest(token="x", tool="fix", args={}, reason="r",
                          action_hash="h", state_version=0, created_at=0.0, ttl_s=10.0)
    with pytest.raises(ValidationError):
        req.tool = "other"


def test_headless_pause_and_resume_approve(tmp_path):
    env, fix, observe, metric = make_env({"t1"})
    path = str(tmp_path / "state.json")
    llm = ScriptedLLM([call("fix", name="t1")])
    loop = Loop(goal="fix", tools=[fix], llm=llm, observe=observe, goal_metric=metric,
                policies=[RequireApproval({"fix"})],
                approval_runner=HeadlessApprovalRunner(), state_path=path, quiet=True)
    result = loop.run()
    assert result.status == LoopStatus.AWAITING_APPROVAL
    assert result.resumable
    assert result.approval_token
    assert env["failing"] == {"t1"}

    result2 = Loop.resume(path, tools=[fix], llm=ScriptedLLM([call("fix", name="t1")]),
                          approval={"token": result.approval_token, "approved": True},
                          observe=observe, goal_metric=metric, quiet=True)
    assert result2.status == LoopStatus.SUCCESS
    assert env["failing"] == set()


def test_stale_precondition_invalidates_approval(tmp_path):
    env, fix, observe, metric = make_env({"t1"})
    path = str(tmp_path / "state.json")
    llm = ScriptedLLM([call("fix", name="t1")])
    loop = Loop(goal="fix", tools=[fix], llm=llm, observe=observe, goal_metric=metric,
                policies=[RequireApproval({"fix"})],
                approval_runner=HeadlessApprovalRunner(), state_path=path, quiet=True)
    result = loop.run()
    old_token = result.approval_token

    env["failing"].discard("t1")  # the world changed while approval was pending

    result2 = Loop.resume(path, tools=[fix], llm=ScriptedLLM([call("fix", name="t1")]),
                          approval={"token": old_token, "approved": True},
                          observe=observe, goal_metric=metric, quiet=True)
    assert result2.status == LoopStatus.AWAITING_APPROVAL
    assert result2.approval_token != old_token  # re-requested with a fresh token
    assert "changed" in result2.reason


def test_expired_approval(tmp_path):
    env, fix, observe, metric = make_env({"t1"})
    path = str(tmp_path / "state.json")
    llm = ScriptedLLM([call("fix", name="t1")])
    loop = Loop(goal="fix", tools=[fix], llm=llm, observe=observe, goal_metric=metric,
                policies=[RequireApproval({"fix"})],
                approval_runner=HeadlessApprovalRunner(ttl_s=0.05), state_path=path, quiet=True)
    result = loop.run()
    assert result.status == LoopStatus.AWAITING_APPROVAL

    time.sleep(0.1)
    result2 = Loop.resume(path, tools=[fix], llm=ScriptedLLM([call("fix", name="t1")]),
                          approval={"token": result.approval_token, "approved": True},
                          observe=observe, goal_metric=metric, quiet=True)
    assert result2.status == LoopStatus.PENDING_EXPIRED
    assert result2.resumable
