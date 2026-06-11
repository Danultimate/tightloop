import typing

import pytest

from loop import (
    Loop,
    LoopStatus,
    SchemaChangedError,
    State,
    Tool,
    UnsupportedTypeError,
    run_command,
    tool,
)
from loop.pricing import PricingStalenessError, check_staleness
from conftest import ScriptedLLM, call, make_env


def test_unsupported_type_hint_fails_at_registration():
    def bad(handler: typing.Callable) -> str:
        return "x"

    with pytest.raises(UnsupportedTypeError):
        Tool(bad)


def test_missing_type_hint_fails_at_registration():
    def bad(x) -> str:
        return "x"

    with pytest.raises(UnsupportedTypeError):
        Tool(bad)


def test_fingerprint_hashes_long_content_keeps_paths_exact():
    @tool
    def edit(path: str, content: str) -> str:
        """Edit a file."""
        return "ok"

    fp1 = edit.fingerprint({"path": "a.py", "content": "x" * 500})
    fp2 = edit.fingerprint({"path": "a.py", "content": "x" * 499 + "y"})
    fp3 = edit.fingerprint({"path": "b.py", "content": "x" * 500})
    assert fp1 != fp2  # near-identical content still distinguishes
    assert fp1 != fp3  # path is exact


def test_run_command_enforces_timeout():
    res = run_command("sleep 5", timeout_s=0.3)
    assert res.timed_out


def test_state_roundtrip_and_integrity(tmp_path):
    state = State(goal="g", tool_schema_hash="abc")
    path = tmp_path / "s.json"
    state.save(path)
    loaded = State.load(path)
    assert loaded.goal == "g"

    # tamper → integrity failure
    text = path.read_text().replace('"g"', '"tampered"')
    path.write_text(text)
    with pytest.raises(Exception):
        State.load(path)


def test_schema_change_on_resume_is_explicit(tmp_path):
    env, fix, observe, metric = make_env({"a", "b", "c"})
    path = str(tmp_path / "state.json")
    llm = ScriptedLLM([call("fix", name="a")])
    Loop(goal="fix", tools=[fix], llm=llm, observe=observe, goal_metric=metric,
         token_limit=60, state_path=path, quiet=True).run()

    @tool
    def fix_v2(name: str, aggressive: bool = False) -> str:
        """Different schema."""
        return "x"

    with pytest.raises(SchemaChangedError):
        Loop.resume(path, tools=[fix_v2], llm=llm, observe=observe,
                    goal_metric=metric, quiet=True)

    # explicit override is allowed
    result = Loop.resume(path, tools=[fix_v2], llm=ScriptedLLM([call("fix_v2", name="b")]),
                         observe=observe, goal_metric=metric, quiet=True,
                         allow_schema_change=True, extend={"token_limit": 10_000},
                         max_iterations=3)
    assert result.status in (LoopStatus.BUDGET_EXHAUSTED, LoopStatus.NO_PROGRESS,
                             LoopStatus.SUCCESS)


def test_pricing_staleness_behaviors():
    old = {"as_of": "2020-01-01", "models": {}}
    assert check_staleness(old, "token-only") is False
    with pytest.raises(PricingStalenessError):
        check_staleness(old, "refuse")
    with pytest.warns(UserWarning):
        assert check_staleness(old, "warn") is True
