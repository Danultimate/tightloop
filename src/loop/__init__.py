"""Loop — production-grade loops for AI agents.

A structured runtime for reliable, observable, governable agent loops.
"""
from .approval import (
    ApprovalDecision,
    ApprovalRequest,
    CallbackApprovalRunner,
    CLIApprovalRunner,
    HeadlessApprovalRunner,
)
from .blueprints import PytestFailureMetric, TestFixLoop
from .core.engine import Loop, LoopConfigError, NestedLoopError
from .core.result import LoopResult, LoopStatus
from .core.state import (
    ArtifactDriftError,
    MetricSnapshot,
    SchemaChangedError,
    State,
)
from .exit import Exit, ExitCondition
from .llm import CallableLLM, LLMClient, LLMResponse, ToolCallReq
from .policy import CostLimit, NoProgress, Policy, RequireApproval
from .progress import GoalMetric
from .tools import Tool, ToolRegistry, UnsupportedTypeError, run_command, tool
from .trace import explain

__version__ = "0.1.0"

__all__ = [
    "Loop", "LoopResult", "LoopStatus", "State", "MetricSnapshot",
    "Exit", "ExitCondition", "Policy", "NoProgress", "CostLimit", "RequireApproval",
    "GoalMetric", "Tool", "tool", "ToolRegistry", "run_command",
    "LLMClient", "LLMResponse", "ToolCallReq", "CallableLLM",
    "ApprovalRequest", "ApprovalDecision", "CLIApprovalRunner",
    "CallbackApprovalRunner", "HeadlessApprovalRunner",
    "TestFixLoop", "PytestFailureMetric", "explain",
    "NestedLoopError", "LoopConfigError", "SchemaChangedError",
    "ArtifactDriftError", "UnsupportedTypeError",
]
