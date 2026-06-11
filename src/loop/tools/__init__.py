"""Tools: type-hint schema derivation, validated execution, enforced timeouts.

- Unsupported type hints fail at REGISTRATION, never silently.
- Schemas are derived once and frozen for the loop lifetime (hash-checked on resume).
- Thread runner: timeout marks the result `aborted` (threads cannot be force-killed —
  use run_command for long/untrusted operations; it escalates SIGTERM → SIGKILL).
"""
from __future__ import annotations

import contextvars
import enum
import hashlib
import inspect
import json
import signal
import subprocess
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable, Literal, Union

from pydantic import BaseModel, ValidationError, create_model

from ..core.state import digest, excerpt

_SUPPORTED_BASES = (str, int, float, bool)


class UnsupportedTypeError(TypeError):
    pass


class ToolValidationError(Exception):
    pass


def _check_supported(annotation: Any, tool_name: str, param: str) -> None:
    if annotation in _SUPPORTED_BASES or annotation is type(None):
        return
    if isinstance(annotation, type) and issubclass(annotation, (enum.Enum, BaseModel)):
        return
    origin = typing.get_origin(annotation)
    if origin in (list, dict, Union, Literal):
        for arg in typing.get_args(annotation):
            if origin is Literal:
                continue  # literal values, not types
            _check_supported(arg, tool_name, param)
        return
    if annotation in (list, dict):
        return
    raise UnsupportedTypeError(
        f"tool {tool_name!r} parameter {param!r}: unsupported type hint {annotation!r}. "
        "Supported: str, int, float, bool, list, dict, Optional, Literal, Enum, pydantic models."
    )


def _default_normalize(args: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint normalizer: long strings (file contents) are hashed, short values
    (paths, names) kept exact — so near-identical edits still fingerprint distinctly."""
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 200:
            out[k] = f"sha:{digest(v)}"
        else:
            out[k] = v
    return out


class Tool:
    def __init__(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        timeout_s: float = 60.0,
        cleanup: Callable[[], None] | None = None,
        normalizer: Callable[[dict], dict] | None = None,
    ):
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description or (fn.__doc__ or "").strip()
        self.timeout_s = timeout_s
        self.cleanup = cleanup
        self.normalizer = normalizer or _default_normalize

        sig = inspect.signature(fn)
        # resolve string annotations (PEP 563 / `from __future__ import annotations`)
        hints = typing.get_type_hints(fn)
        fields: dict[str, Any] = {}
        for pname, p in sig.parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise UnsupportedTypeError(f"tool {self.name!r}: *args/**kwargs are not supported")
            if pname not in hints:
                raise UnsupportedTypeError(f"tool {self.name!r} parameter {pname!r}: type hint required")
            annotation = hints[pname]
            _check_supported(annotation, self.name, pname)
            default = p.default if p.default is not inspect.Parameter.empty else ...
            fields[pname] = (annotation, default)
        self.params_model = create_model(f"{self.name}_params", **fields)
        self.json_schema = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.params_model.model_json_schema(),
        }

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            model = self.params_model.model_validate(args)
        except ValidationError as e:
            raise ToolValidationError(f"invalid arguments for tool {self.name!r}: {e}") from e
        return {k: getattr(model, k) for k in self.params_model.model_fields}

    def fingerprint(self, args: dict[str, Any]) -> str:
        normalized = self.normalizer(args)
        return digest(self.name + json.dumps(normalized, sort_keys=True, default=str))


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout_s: float = 60.0,
    cleanup: Callable[[], None] | None = None,
    normalizer: Callable[[dict], dict] | None = None,
) -> Tool | Callable[[Callable], Tool]:
    def wrap(f: Callable) -> Tool:
        return Tool(f, name=name, description=description, timeout_s=timeout_s,
                    cleanup=cleanup, normalizer=normalizer)

    return wrap(fn) if fn is not None else wrap


class ToolResult(BaseModel):
    status: Literal["ok", "error", "aborted"]
    output: str
    duration_s: float


class ToolRegistry:
    """Frozen at construction; schema_hash detects drift on resume."""

    def __init__(self, tools: list[Tool | Callable]):
        self.tools: dict[str, Tool] = {}
        for t in tools:
            t = t if isinstance(t, Tool) else Tool(t)
            if t.name in self.tools:
                raise ValueError(f"duplicate tool name {t.name!r}")
            self.tools[t.name] = t
        canonical = json.dumps([t.json_schema for t in self.tools.values()], sort_keys=True)
        self.schema_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [t.json_schema for t in self.tools.values()]

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        t = self.tools[name]
        start = time.monotonic()
        # copy_context so the engine's re-entrancy guard propagates into the worker thread
        ctx = contextvars.copy_context()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(ctx.run, t.fn, **args)
            try:
                out = future.result(timeout=t.timeout_s)
                return ToolResult(status="ok", output=excerpt(str(out)),
                                  duration_s=time.monotonic() - start)
            except FutureTimeout:
                if t.cleanup:
                    try:
                        t.cleanup()
                    except Exception:
                        pass
                return ToolResult(
                    status="aborted",
                    output=f"aborted: exceeded timeout of {t.timeout_s}s",
                    duration_s=time.monotonic() - start,
                )
            except Exception as e:
                return ToolResult(status="error", output=excerpt(f"{type(e).__name__}: {e}"),
                                  duration_s=time.monotonic() - start)
        finally:
            executor.shutdown(wait=False)


class CommandResult(BaseModel):
    code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def run_command(cmd: list[str] | str, timeout_s: float = 120.0, cwd: str | None = None) -> CommandResult:
    """Subprocess runner with enforced SIGTERM → SIGKILL escalation."""
    proc = subprocess.Popen(
        cmd,
        shell=isinstance(cmd, str),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return CommandResult(code=proc.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        return CommandResult(code=proc.returncode if proc.returncode is not None else -9,
                             stdout=stdout or "", stderr=stderr or "", timed_out=True)
