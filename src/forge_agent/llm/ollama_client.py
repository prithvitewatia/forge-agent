"""Ollama adapter. All Ollama-specific wire details live here and nowhere else."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import httpx
import ollama

from forge_agent.config import Settings
from forge_agent.errors import LLMError
from forge_agent.messages import AssistantTurn, Message, Role, ToolCall, ToolSpec, Usage

log = logging.getLogger(__name__)


class _ChatBackend(Protocol):
    """The slice of ollama.Client we use; lets tests inject a stub."""

    def chat(self, **kwargs: Any) -> Any: ...


class OllamaClient:
    """LLMClient implementation backed by a local Ollama server."""

    def __init__(self, settings: Settings, backend: _ChatBackend | None = None) -> None:
        self._settings = settings
        # Typed as Any: ollama.Client.chat has a richer, overloaded signature than _ChatBackend,
        # so the structural check cannot be satisfied without losing real keyword validation.
        self._backend: Any = backend or ollama.Client(
            host=settings.ollama_host, timeout=settings.request_timeout_s
        )

    # -- public API ---------------------------------------------------------------------------

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AssistantTurn:
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [_to_wire_message(m) for m in messages],
            "tools": [_to_wire_tool(t) for t in tools],
            "options": {
                "num_ctx": self._settings.num_ctx,
                "temperature": self._settings.temperature,
            },
        }
        if self._settings.think is not None:
            kwargs["think"] = self._settings.think

        try:
            response = self._backend.chat(**kwargs)
        except ollama.ResponseError as exc:
            raise LLMError(_describe_response_error(exc, self._settings.model)) from exc
        except (httpx.HTTPError, ConnectionError, OSError) as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self._settings.ollama_host}. "
                "Is the server running? Try `ollama serve`."
            ) from exc

        return _from_wire_response(response)


# -- wire conversion (module-level so they can be unit-tested directly) ---------------------------


def _to_wire_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _to_wire_message(message: Message) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in message.tool_calls
        ]
    if message.role is Role.TOOL and message.tool_name:
        wire["tool_name"] = message.tool_name
    return wire


def _from_wire_response(response: Any) -> AssistantTurn:
    try:
        raw = response.message
        calls = tuple(
            ToolCall(name=c.function.name, arguments=_coerce_arguments(c.function.arguments))
            for c in (raw.tool_calls or ())
        )
        message = Message(role=Role.ASSISTANT, content=raw.content or "", tool_calls=calls)
        usage = Usage(
            prompt_tokens=getattr(response, "prompt_eval_count", None) or 0,
            completion_tokens=getattr(response, "eval_count", None) or 0,
        )
        return AssistantTurn(message=message, usage=usage, thinking=getattr(raw, "thinking", None))
    except AttributeError as exc:
        raise LLMError(f"Unexpected response shape from Ollama: {exc}") from exc


def _coerce_arguments(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(arguments) if arguments else {}


def _describe_response_error(exc: ollama.ResponseError, model: str) -> str:
    status = getattr(exc, "status_code", -1)
    if status == 404:
        return f"Model '{model}' is not available locally. Run `ollama pull {model}`."
    return f"Ollama returned an error (HTTP {status}): {exc.error}"
