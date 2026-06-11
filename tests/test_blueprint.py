from loop import State
from loop.blueprints import PytestFailureMetric
from loop.blueprints.testfix import parse_failing
from loop.core.state import IterationRecord


def test_parse_failing():
    output = (
        "..F.\n"
        "FAILED tests/test_a.py::test_one - AssertionError\n"
        "FAILED tests/test_b.py::test_two\n"
        "ERROR tests/test_c.py::test_three\n"
        "1 failed\n[exit code: 1]"
    )
    assert parse_failing(output) == {
        "tests/test_a.py::test_one",
        "tests/test_b.py::test_two",
        "tests/test_c.py::test_three",
    }


def test_metric_tracks_identity_not_counts():
    metric = PytestFailureMetric()
    state = State(goal="g")

    # baseline: a and b failing
    snap0 = metric.measure(
        "FAILED t.py::a\nFAILED t.py::b\n[exit code: 1]", state
    )
    assert snap0.value == 0.0
    assert not snap0.regression
    state.iterations.append(IterationRecord(index=0, observation="", metric=snap0))

    # a fixed, but c newly broke: totals same — must still flag regression
    snap1 = metric.measure(
        "FAILED t.py::b\nFAILED t.py::c\n[exit code: 1]", state
    )
    assert snap1.value == 0.0  # 1 fixed - 1 newly broken
    assert snap1.regression
    assert snap1.detail["newly_broken"] == 1
    state.iterations.append(IterationRecord(index=1, observation="", metric=snap1))

    # everything passes
    snap2 = metric.measure(".. 2 passed\n[exit code: 0]", state)
    assert metric.is_success(snap2)
    assert not metric.is_success(snap1)
