"""Explicit, serializable loop state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1
ENGINE_VERSION = "0.1.0"
INLINE_CAP = 16 * 1024  # max inline chars per record; large payloads live in the trace


class StateError(Exception):
    pass


class SchemaChangedError(StateError):
    """Tool schemas changed since the state was saved (resume requires allow_schema_change)."""


class ArtifactDriftError(StateError):
    """Stored context artifacts were produced by a different engine/summarizer version."""


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def excerpt(text: str, cap: int = INLINE_CAP) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n...[truncated {len(text) - cap} chars, digest={digest(text)}]"


class ArtifactStamp(BaseModel):
    engine_version: str
    model_id: str
    prompt_hash: str


class ContextArtifact(BaseModel):
    kind: Literal["summary", "fact"]
    iteration: int | None = None
    content: str
    stamp: ArtifactStamp


class MetricSnapshot(BaseModel):
    value: float
    regression: bool = False
    detail: dict[str, Any] = Field(default_factory=dict)


class ActionRecord(BaseModel):
    tool: str
    args_excerpt: str
    status: Literal["ok", "error", "aborted"]
    result_excerpt: str
    duration_s: float
    fingerprint: str


class IterationRecord(BaseModel):
    index: int
    observation: str
    plan_text: str = ""
    actions: list[ActionRecord] = Field(default_factory=list)
    metric: MetricSnapshot | None = None
    repetition: bool = False
    plan_invalid: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class Metrics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    elapsed_s: float = 0.0
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class PendingApproval(BaseModel):
    token: str
    tool: str
    args: dict[str, Any]
    reason: str
    action_hash: str
    state_version: int
    created_at: float
    ttl_s: float
    plan_text: str = ""
    precondition_metric: MetricSnapshot | None = None


class State(BaseModel):
    schema_version: int = SCHEMA_VERSION
    goal: str
    config: dict[str, Any] = Field(default_factory=dict)
    tool_schema_hash: str = ""
    iterations: list[IterationRecord] = Field(default_factory=list)
    artifacts: list[ContextArtifact] = Field(default_factory=list)
    pinned_facts: list[str] = Field(default_factory=list)
    failed_approaches: list[str] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
    no_progress_streak: int = 0
    plan_invalid_streak: int = 0
    state_version: int = 0
    pending_approval: PendingApproval | None = None

    def save(self, path: str | Path) -> None:
        payload = self.model_dump(mode="json")
        body = json.dumps(payload, sort_keys=True)
        wrapper = {
            "integrity": hashlib.sha256(body.encode()).hexdigest(),
            "state": payload,
        }
        Path(path).write_text(json.dumps(wrapper, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "State":
        try:
            wrapper = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise StateError(f"cannot read state file {path}: {e}") from e
        payload = wrapper.get("state")
        if payload is None:
            raise StateError(f"{path} is not a Loop state file")
        body = json.dumps(payload, sort_keys=True)
        if hashlib.sha256(body.encode()).hexdigest() != wrapper.get("integrity"):
            raise StateError(f"integrity check failed for {path}")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise StateError(
                f"state schema_version {payload.get('schema_version')} != supported {SCHEMA_VERSION}"
            )
        return cls.model_validate(payload)
