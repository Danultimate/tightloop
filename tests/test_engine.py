import time

from loop import Loop, LoopStatus, tool
from conftest import ScriptedLLM, call, make_env


def test_quickstart_reaches_success():
    env, fix, observe, metric = make_env({"t1", "t2"})
    llm = ScriptedLLM([call("fix", name="t1"), call("fix", name="t2")])
    loop = Loop(goal="fix tests", tools=[fix], llm=llm, observe=observe,
                goal_metric=metric, quiet=True)
    result = loop.run()
    assert result.status == LoopStatus.SUCCESS
    assert not result.resumable
    assert env["failing"] == set()
    assert result.iterations == 3  # fix t1, fix t2, success observation
    report = loop.explain(result)
    assert "SUCCESS" in report.render()


def test_budget_exhaustion_is_resumable(tmp_path):
    env, fix, observe, metric = make_env({"a", "b", "c"})
    path = str(tmp_path / "state.json")
    llm = ScriptedLLM([call("fix", name="a"), call("fix", name="b")])
    loop = Loop(goal="fix tests", tools=[fix], llm=llm, observe=observe,
                goal_metric=metric, token_limit=120, state_path=path, quiet=True)
    result = loop.run()
    assert result.status == LoopStatus.BUDGET_EXHAUSTED
    assert result.resumable
    assert "token_limit" in result.reason
    # ceilings are checked before each action: the second plan landed exactly on the
    # limit, so its action was dropped (traced) rather than overshooting the budget
    assert env["failing"] == {"b", "c"}

    # resume with an extended budget → runs to success, never a mysterious stop
    result2 = Loop.resume(path, tools=[fix],
                          llm=ScriptedLLM([call("fix", name="b"), call("fix", name="c")]),
                          extend={"token_limit": 10_000}, observe=observe,
                          goal_metric=metric, quiet=True)
    assert result2.status == LoopStatus.SUCCESS
    assert env["failing"] == set()


def test_three_invalid_plans_then_plan_failed():
    env, fix, observe, metric = make_env({"a"})
    llm = ScriptedLLM([call("fix", bogus=1)])  # missing required 'name' forever
    loop = Loop(goal="fix tests", tools=[fix], llm=llm, observe=observe,
                goal_metric=metric, quiet=True)
    result = loop.run()
    assert result.status == LoopStatus.PLAN_FAILED
    assert result.resumable
    assert result.iterations == 2
    assert all(it.plan_invalid for it in loop.state.iterations)
    # 3 validation attempts per iteration, 2 iterations
    assert loop.state.metrics.llm_calls == 6


def test_no_progress_exit_on_repetition_with_flat_metric():
    env, fix, observe, metric = make_env({"a", "b"})
    llm = ScriptedLLM([call("fix", name="zz")])  # repeats a no-op forever
    loop = Loop(goal="fix tests", tools=[fix], llm=llm, observe=observe,
                goal_metric=metric, quiet=True)
    result = loop.run()
    assert result.status == LoopStatus.NO_PROGRESS
    assert result.resumable
    # streak needs 3 consecutive flagged+flat iterations after the first
    assert result.iterations == 4


def test_nested_loop_invocation_raises():
    env, fix, observe, metric = make_env({"a"})

    @tool
    def nested() -> str:
        """Illegally starts a loop inside a tool."""
        inner = Loop(goal="inner", tools=[], llm=ScriptedLLM([call("fix", name="x")]), quiet=True)
        inner.run()
        return "should not get here"

    llm = ScriptedLLM([call("nested")])
    loop = Loop(goal="outer", tools=[nested], llm=llm, observe=observe,
                goal_metric=metric, max_iterations=1, quiet=True)
    loop.run()
    action = loop.state.iterations[0].actions[0]
    assert action.status == "error"
    assert "NestedLoopError" in action.result_excerpt


def test_tool_timeout_marks_aborted():
    @tool(timeout_s=0.2)
    def slow() -> str:
        """Sleeps past its timeout."""
        time.sleep(1.0)
        return "done"

    llm = ScriptedLLM([call("slow")])
    loop = Loop(goal="timeout", tools=[slow], llm=llm, max_iterations=1, quiet=True)
    loop.run()
    action = loop.state.iterations[0].actions[0]
    assert action.status == "aborted"
    assert "timeout" in action.result_excerpt


def test_budget_report_is_itemized():
    env, fix, observe, metric = make_env({"a"})
    llm = ScriptedLLM([call("fix", name="a")])
    loop = Loop(goal="fix tests", tools=[fix], llm=llm, observe=observe,
                goal_metric=metric, quiet=True)
    loop.run()
    report = loop.budget_report()
    for key in ("pinned_system_tokens", "summary_tokens", "verbatim_tokens",
                "total_context_tokens", "spent_input_tokens"):
        assert key in report
