"""Tool abstraction and registry. Arguments are validated with pydantic before any tool runs."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from forge_agent.errors import SafetyError, ToolError
from forge_agent.messages import ToolCall, ToolSpec
from forge_agent.safety import ApprovalRequest, Approver, Risk

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Outcome of a tool call, always returned to the model as text."""

    content: str
    is_error: bool = False


class Tool[ArgsT: BaseModel](ABC):
    """A capability the model can invoke. Subclasses declare their arguments as a pydantic model."""

    name: ClassVar[str]
    description: ClassVar[str]
    risk: ClassVar[Risk]
    args_model: ClassVar[type[BaseModel]]

    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return ToolSpec(name=self.name, description=self.description, parameters=schema)

    def precheck(self, args: ArgsT) -> None:  # noqa: B027 - optional hook, intentionally empty
        """Reject obviously bad input before the user is asked to approve it.

        Raise ToolError to refuse. Runs before describe() and the approval gate, so a human is
        never prompted to approve an action that would be rejected anyway.
        """

    @abstractmethod
    def describe(self, args: ArgsT) -> str:
        """One-line, human-readable summary of what this call will do (shown for approval)."""

    @abstractmethod
    def run(self, args: ArgsT) -> str:
        """Execute. Raise ToolError/SafetyError for expected failures; return text on success."""


class ToolRegistry:
    """Holds tools, advertises their schemas, and executes calls defensively."""

    def __init__(self, approver: Approver, max_output_chars: int) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        self._approver = approver
        self._max_output_chars = max_output_chars

    def register(self, tool: Tool[Any]) -> None:
        if tool.name in self._tools:
            msg = f"duplicate tool name: {tool.name}"
            raise ValueError(msg)
        self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def execute(self, call: ToolCall) -> ToolResult:
        """Run a model-requested call. Never raises for tool-level problems: the model gets the
        error text so it can correct itself, which is what small models need most."""
        tool = self._tools.get(call.name)
        if tool is None:
            known = ", ".join(self.names())
            return ToolResult(f"Unknown tool '{call.name}'. Available tools: {known}", True)

        try:
            args = tool.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResult(f"Invalid arguments for {call.name}: {_format_validation(exc)}", True)

        try:
            tool.precheck(args)
            summary = tool.describe(args)
            if tool.risk is not Risk.READ and not self._approver.approve(
                ApprovalRequest(tool=tool.name, risk=tool.risk, summary=summary)
            ):
                return ToolResult(f"Action denied by the user/policy: {summary}", True)
            output = tool.run(args)
        except (ToolError, SafetyError) as exc:
            return ToolResult(str(exc), True)
        except Exception as exc:
            log.exception("Unexpected failure in tool %s", call.name)
            return ToolResult(f"Internal error in {call.name}: {type(exc).__name__}: {exc}", True)

        return ToolResult(self._truncate(output))

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_output_chars:
            return text
        head = self._max_output_chars * 2 // 3
        tail = self._max_output_chars - head
        omitted = len(text) - head - tail
        return f"{text[:head]}\n... [{omitted} characters omitted] ...\n{text[-tail:]}"


def _format_validation(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False):
        location = ".".join(str(p) for p in err["loc"]) or "arguments"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)
