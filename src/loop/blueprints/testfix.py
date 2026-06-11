"""TestFixLoop blueprint: fix failing tests until all pass.

Progress tracks test IDENTITY, not counts: value = originally_failing_fixed −
newly_broken, and newly-broken tests flag `regression` even when totals improve.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.engine import Loop
from ..core.state import MetricSnapshot, State
from ..llm import LLMClient
from ..progress import GoalMetric
from ..tools import Tool, run_command

_FAILED_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)
_EXIT_RE = re.compile(r"\[exit code: (-?\d+)\]")


def parse_failing(output: str) -> set[str]:
    return {m.split(" - ")[0] for m in _FAILED_RE.findall(output)}


class PytestFailureMetric(GoalMetric):
    def measure(self, observation: str, state: State) -> MetricSnapshot:
        failing = parse_failing(observation)
        exit_match = _EXIT_RE.search(observation)
        exit_code = int(exit_match.group(1)) if exit_match else None

        baseline: set[str] | None = None
        prev: set[str] | None = None
        for it in state.iterations:
            if it.metric and "failing" in it.metric.detail:
                if baseline is None:
                    baseline = set(it.metric.detail["baseline"] or it.metric.detail["failing"])
                prev = set(it.metric.detail["failing"])
        if baseline is None:
            baseline = set(failing)

        fixed = baseline - failing
        newly_broken = failing - baseline
        regressed_vs_prev = bool(failing - prev) if prev is not None else False
        return MetricSnapshot(
            value=float(len(fixed) - len(newly_broken)),
            regression=bool(newly_broken) or regressed_vs_prev,
            detail={
                "failing": sorted(failing),
                "baseline": sorted(baseline),
                "fixed": len(fixed),
                "newly_broken": len(newly_broken),
                "exit_code": exit_code,
            },
        )

    def is_success(self, snapshot: MetricSnapshot) -> bool:
        return not snapshot.detail.get("failing") and snapshot.detail.get("exit_code") == 0


class TestFixLoop(Loop):
    __test__ = False  # not a pytest test class, despite the name

    def __init__(
        self,
        llm: LLMClient,
        repo: str = ".",
        test_cmd: str = "python -m pytest -q -rf --tb=short",
        test_timeout_s: float = 300.0,
        goal: str | None = None,
        **kwargs: Any,
    ):
        repo_path = Path(repo).resolve()

        def _run_tests() -> str:
            res = run_command(test_cmd, timeout_s=test_timeout_s, cwd=str(repo_path))
            suffix = " [timed out]" if res.timed_out else ""
            return f"{res.stdout}\n{res.stderr}\n[exit code: {res.code}]{suffix}"

        def run_tests() -> str:
            """Run the test suite and return its output."""
            return _run_tests()

        def read_file(path: str) -> str:
            """Read a file from the repository."""
            target = (repo_path / path).resolve()
            if not target.is_relative_to(repo_path):
                raise ValueError(f"path {path!r} escapes the repository")
            return target.read_text()

        def edit_file(path: str, content: str) -> str:
            """Replace the full contents of a file in the repository."""
            target = (repo_path / path).resolve()
            if not target.is_relative_to(repo_path):
                raise ValueError(f"path {path!r} escapes the repository")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            # drop stale bytecode: pyc validation uses whole-second mtime + size,
            # so a same-size edit within the same second would be masked by the cache
            cache_dir = target.parent / "__pycache__"
            if target.suffix == ".py" and cache_dir.is_dir():
                for pyc in cache_dir.glob(f"{target.stem}.*.pyc"):
                    pyc.unlink(missing_ok=True)
            return f"wrote {len(content)} chars to {path}"

        super().__init__(
            goal=goal or f"Fix failing tests in {repo_path.name} until all pass",
            tools=[
                Tool(run_tests, timeout_s=test_timeout_s + 10),
                Tool(read_file),
                Tool(edit_file),
            ],
            llm=llm,
            observe=lambda state: _run_tests(),
            goal_metric=PytestFailureMetric(),
            **kwargs,
        )
