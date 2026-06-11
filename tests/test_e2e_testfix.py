"""End-to-end: TestFixLoop fixes a real failing pytest suite in a temp repo.
The LLM is scripted; tool execution, pytest runs, and metric parsing are real."""
import sys

from loop import LoopStatus, TestFixLoop
from conftest import ScriptedLLM, call

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def test_testfix_loop_end_to_end(tmp_path):
    (tmp_path / "calc.py").write_text(BUGGY)
    (tmp_path / "test_calc.py").write_text(TEST)

    llm = ScriptedLLM([call("edit_file", path="calc.py", content=FIXED)])
    loop = TestFixLoop(
        llm=llm,
        repo=str(tmp_path),
        test_cmd=f"{sys.executable} -m pytest -q -rf --tb=short",
        quiet=True,
    )
    result = loop.run()

    assert result.status == LoopStatus.SUCCESS
    assert (tmp_path / "calc.py").read_text() == FIXED
    # baseline tracked by identity
    first = loop.state.iterations[0].metric
    assert first.detail["failing"] == ["test_calc.py::test_add"]
    report = loop.explain(result)
    assert "SUCCESS" in report.status
