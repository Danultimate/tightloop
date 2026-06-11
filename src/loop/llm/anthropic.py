"""Anthropic adapter. Requires the `anthropic` extra. Default timeout: 120s (not SLA-backed)."""
from __future__ import annotations

from typing import Any

from . import DEFAULT_TIMEOUT_S, LLMClient, LLMResponse, ToolCallReq


class AnthropicLLM(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6", timeout_s: float = DEFAULT_TIMEOUT_S, **client_kwargs: Any):
        import anthropic  # lazy: optional dependency

        self.model_id = model
        self.timeout_s = timeout_s
        self._client = anthropic.Anthropic(timeout=timeout_s, **client_kwargs)

    def complete(self, messages, tool_schemas, max_tokens):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        tools = [
            {"name": s["name"], "description": s.get("description", ""), "input_schema": s["input_schema"]}
            for s in tool_schemas
        ]
        resp = self._client.messages.create(
            model=self.model_id,
            system=system or None,
            messages=convo,
            tools=tools or None,
            max_tokens=max_tokens,
        )
        text_parts: list[str] = []
        calls: list[ToolCallReq] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                calls.append(ToolCallReq(id=block.id, name=block.name, args=args))
        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=calls,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model_id=self.model_id,
        )
