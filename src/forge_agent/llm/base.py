"""The only thing the agent knows about a model backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from forge_agent.messages import AssistantTurn, Message, ToolSpec


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat-with-tools contract. Implement this to add a new provider."""

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AssistantTurn:
        """Send the conversation and return the model's next turn.

        Raises:
            forge_agent.errors.LLMError: on any backend failure.
        """
        ...
