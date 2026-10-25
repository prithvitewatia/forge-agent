"""Test doubles and builders shared across test modules."""

from __future__ import annotations

from collections.abc import Sequence

from forge_agent.errors import LLMError
from forge_agent.messages import AssistantTurn, Message, Role, ToolCall, ToolSpec, Usage
from forge_agent.safety import ApprovalRequest


class ScriptedLLM:
    """Replays a fixed list of turns (or errors) and records what it was sent."""

    def __init__(self, script: Sequence[AssistantTurn | LLMError]) -> None:
        self._script = list(script)
        self.calls: list[list[Message]] = []

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AssistantTurn:
        self.calls.append(list(messages))
        if not self._script:
            msg = "ScriptedLLM ran out of turns"
            raise AssertionError(msg)
        item = self._script.pop(0)
        if isinstance(item, LLMError):
            raise item
        return item


class RecordingApprover:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.requests: list[ApprovalRequest] = []

    def approve(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return self.answer


def say(text: str) -> AssistantTurn:
    return AssistantTurn(Message(Role.ASSISTANT, text), Usage(10, 5))


def call(name: str, **arguments: object) -> AssistantTurn:
    tool_call = ToolCall(name, dict(arguments))
    return AssistantTurn(Message(Role.ASSISTANT, "", (tool_call,)), Usage(10, 5))
