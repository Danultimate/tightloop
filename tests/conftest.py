from __future__ import annotations

from loop import CallableLLM, GoalMetric, LLMResponse, MetricSnapshot, ToolCallReq, tool


class ScriptedLLM(CallableLLM):
    """Returns responses in order; the last one repeats."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.model_id = "scripted"

    def complete(self, messages, tool_schemas, max_tokens):
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def call(tool_name: str, text: str = "", tokens: tuple[int, int] = (50, 10), **args) -> LLMResponse:
    return LLMResponse(
        text=text or f"{tool_name} {args}",
        tool_calls=[ToolCallReq(name=tool_name, args=args)],
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        model_id="scripted",
    )


class FailingTestsMetric(GoalMetric):
    """Counts failing names out of an 'failing: a,b' observation string."""

    def measure(self, observation: str, state) -> MetricSnapshot:
        names = set(observation.removeprefix("failing: ").split(",")) - {""}
        return MetricSnapshot(value=-float(len(names)), detail={"failing": sorted(names)})

    def is_success(self, snapshot: MetricSnapshot) -> bool:
        return not snapshot.detail["failing"]


def make_env(failing: set[str]):
    """A tiny fake repo: a mutable failing-test set, a fix tool, an observe fn, a metric."""
    env = {"failing": set(failing)}

    @tool
    def fix(name: str) -> str:
        """Fix one failing test by name."""
        env["failing"].discard(name)
        return f"fixed {name}"

    def observe(state) -> str:
        return "failing: " + ",".join(sorted(env["failing"]))

    return env, fix, observe, FailingTestsMetric()
