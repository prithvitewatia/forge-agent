"""Provider-neutral message and tool-call types shared by the agent, LLM clients and traces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request from the model to run a tool."""

    name: str
    arguments: dict[str, Any]

    def fingerprint(self) -> str:
        """Stable identity used to detect the model repeating itself."""
        return f"{self.name}:{json.dumps(self.arguments, sort_keys=True, default=str)}"


@dataclass(frozen=True, slots=True)
class Message:
    """One entry in the conversation history."""

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_name: str | None = None  # set on TOOL messages

    def to_trace(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in self.tool_calls],
            "tool_name": self.tool_name,
        }


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Schema advertised to the model for one tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """What the model returned for one call."""

    message: Message
    usage: Usage = field(default_factory=Usage)
    thinking: str | None = None
