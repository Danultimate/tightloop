"""Context manager: pinned facts never summarized away,
failed-approaches registry always in context, version-stamped summaries
computed once and stored (deterministic resume), transparent token accounting.
"""
from __future__ import annotations

import hashlib
from typing import Any

from ..core.state import (
    ENGINE_VERSION,
    ArtifactStamp,
    ContextArtifact,
    IterationRecord,
    State,
)
from ..llm import LLMClient

SUMMARY_PROMPT = (
    "Summarize this agent-loop iteration in 3 sentences or fewer. Preserve: what was "
    "attempted, the outcome, and any error messages verbatim.\n\n{body}"
)
_PROMPT_HASH = hashlib.sha256(SUMMARY_PROMPT.encode()).hexdigest()[:16]


def _render_iteration(it: IterationRecord, result_cap: int = 2000) -> str:
    lines = [f"### Iteration {it.index}"]
    if it.observation:
        lines.append(f"Observed:\n{it.observation[:result_cap]}")
    if it.plan_text:
        lines.append(f"Planned: {it.plan_text[:600]}")
    if it.plan_invalid:
        lines.append("(plan failed tool-argument validation)")
    for a in it.actions:
        lines.append(f"Action {a.tool}({a.args_excerpt[:300]}) -> [{a.status}] {a.result_excerpt[:result_cap]}")
    if it.metric:
        lines.append(f"Metric: {it.metric.value}" + (" (REGRESSION)" if it.metric.regression else ""))
    return "\n".join(lines)


def _est_tokens(text: str) -> int:
    return len(text) // 4  # documented heuristic; itemized, not hidden


class ContextManager:
    def __init__(self, verbatim_window: int = 3, summarizer: LLMClient | None = None,
                 summary_max_tokens: int = 400):
        self.verbatim_window = verbatim_window
        self.summarizer = summarizer
        self.summary_max_tokens = summary_max_tokens

    @property
    def stamp(self) -> ArtifactStamp:
        model_id = self.summarizer.model_id if self.summarizer else "deterministic-truncate"
        return ArtifactStamp(engine_version=ENGINE_VERSION, model_id=model_id, prompt_hash=_PROMPT_HASH)

    def check_artifact_drift(self, state: State) -> list[str]:
        """Returns mismatch descriptions for artifacts produced under a different config."""
        current = self.stamp
        problems = []
        for a in state.artifacts:
            if a.kind == "summary" and a.stamp != current:
                problems.append(
                    f"summary for iteration {a.iteration}: produced by "
                    f"{a.stamp.engine_version}/{a.stamp.model_id}, current is "
                    f"{current.engine_version}/{current.model_id}"
                )
        return problems

    def ensure_summaries(self, state: State) -> list[int]:
        """Summarize iterations that just left the verbatim window. Computed once,
        stored, reused on resume — never recomputed."""
        done = {a.iteration for a in state.artifacts if a.kind == "summary"}
        cutoff = len(state.iterations) - self.verbatim_window
        created = []
        for it in state.iterations[:cutoff] if cutoff > 0 else []:
            if it.index in done:
                continue
            body = _render_iteration(it, result_cap=600)
            if self.summarizer:
                resp = self.summarizer.complete(
                    [{"role": "user", "content": SUMMARY_PROMPT.format(body=body)}],
                    [], self.summary_max_tokens,
                )
                content = resp.text
                state.metrics.input_tokens += resp.input_tokens
                state.metrics.output_tokens += resp.output_tokens
                state.metrics.llm_calls += 1
            else:
                content = body[:800]  # deterministic fallback
            state.artifacts.append(
                ContextArtifact(kind="summary", iteration=it.index, content=content, stamp=self.stamp)
            )
            created.append(it.index)
        return created

    def build(self, state: State, observation: str) -> list[dict[str, str]]:
        system_parts = [
            "You are an agent executing one step of a structured loop. Use the provided "
            "tools to make progress toward the goal. Respond with tool calls.",
            f"Goal: {state.goal}",
        ]
        if state.pinned_facts:
            system_parts.append("Key facts (pinned):\n" + "\n".join(f"- {f}" for f in state.pinned_facts))
        if state.failed_approaches:
            system_parts.append(
                "Approaches already tried that FAILED (do not repeat):\n"
                + "\n".join(f"- {f}" for f in state.failed_approaches)
            )

        user_parts = []
        summaries = [a for a in state.artifacts if a.kind == "summary"]
        if summaries:
            user_parts.append(
                "## Earlier iterations (summarized)\n"
                + "\n".join(f"- iter {a.iteration}: {a.content}" for a in summaries)
            )
        recent = state.iterations[-self.verbatim_window:]
        if recent:
            user_parts.append("## Recent iterations\n" + "\n\n".join(_render_iteration(it) for it in recent))
        user_parts.append(f"## Current observation\n{observation}\n\nDecide the next action(s).")

        return [
            {"role": "system", "content": "\n\n".join(system_parts)},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    def budget_report(self, state: State, observation: str = "") -> dict[str, Any]:
        """Itemized token accounting per section."""
        messages = self.build(state, observation)
        system, user = messages[0]["content"], messages[1]["content"]
        summaries = [a for a in state.artifacts if a.kind == "summary"]
        return {
            "pinned_system_tokens": _est_tokens(system),
            "summary_tokens": _est_tokens("\n".join(a.content for a in summaries)),
            "verbatim_tokens": _est_tokens(
                "\n".join(_render_iteration(it) for it in state.iterations[-self.verbatim_window:])
            ),
            "observation_tokens": _est_tokens(observation),
            "total_context_tokens": _est_tokens(system) + _est_tokens(user),
            "spent_input_tokens": state.metrics.input_tokens,
            "spent_output_tokens": state.metrics.output_tokens,
            "note": "estimates use a len/4 heuristic; provider-reported usage is authoritative",
        }
