"""LLMClient protocol + canonical ToolCall normalization.

Provider responses are normalized into LLMResponse/ToolCallReq at this
boundary, so the engine handles hallucinated or malformed tool calls uniformly
regardless of provider leniency.

Timeout defaults are recommended values, not SLA-backed (see README).
"""
from __future__ import annotations

import time
from typing import Any, Callable

from pydantic import BaseModel, Field

DEFAULT_TIMEOUT_S = 120.0


class ToolCallReq(BaseModel):
    id: str = ""
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCallReq] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = ""


class LLMClient:
    """Adapter protocol. Subclasses implement complete()."""

    model_id: str = "unknown"
    timeout_s: float = DEFAULT_TIMEOUT_S

    def complete(
        self,
        messages: list[dict[str, str]],
        tool_schemas: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse:
        raise NotImplementedError


class CallableLLM(LLMClient):
    """Wraps any fn(messages, tool_schemas) -> LLMResponse. Useful for tests and raw APIs."""

    def __init__(self, fn: Callable[[list[dict], list[dict]], LLMResponse], model_id: str = "callable"):
        self.fn = fn
        self.model_id = model_id

    def complete(self, messages, tool_schemas, max_tokens):
        return self.fn(messages, tool_schemas)


def complete_with_retry(
    client: LLMClient,
    messages: list[dict[str, str]],
    tool_schemas: list[dict[str, Any]],
    max_tokens: int,
) -> LLMResponse:
    """One retry with backoff on timeout/transient errors."""
    try:
        return client.complete(messages, tool_schemas, max_tokens)
    except Exception:
        time.sleep(1.0)
        return client.complete(messages, tool_schemas, max_tokens)
