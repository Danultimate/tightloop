"""OpenAI adapter. Requires the `openai` extra. Default timeout: 120s (not SLA-backed)."""
from __future__ import annotations

import json
from typing import Any

from . import DEFAULT_TIMEOUT_S, LLMClient, LLMResponse, ToolCallReq


class OpenAILLM(LLMClient):
    def __init__(self, model: str = "gpt-4o", timeout_s: float = DEFAULT_TIMEOUT_S, **client_kwargs: Any):
        import openai  # lazy: optional dependency

        self.model_id = model
        self.timeout_s = timeout_s
        self._client = openai.OpenAI(timeout=timeout_s, **client_kwargs)

    def complete(self, messages, tool_schemas, max_tokens):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "parameters": s["input_schema"],
                },
            }
            for s in tool_schemas
        ]
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=tools or None,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0].message
        calls: list[ToolCallReq] = []
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # malformed args become an empty call; engine-side validation feeds
                # a structured error back to the model
                args = {"__malformed__": tc.function.arguments}
            if not isinstance(args, dict):
                args = {"__malformed__": tc.function.arguments}
            calls.append(ToolCallReq(id=tc.id, name=tc.function.name, args=args))
        usage = resp.usage
        return LLMResponse(
            text=choice.content or "",
            tool_calls=calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model_id=self.model_id,
        )
