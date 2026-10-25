from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import ollama
import pytest

from forge_agent.config import Settings
from forge_agent.errors import LLMError
from forge_agent.llm import LLMClient, OllamaClient
from forge_agent.llm.ollama_client import _to_wire_message, _to_wire_tool
from forge_agent.messages import Message, Role, ToolCall, ToolSpec


class StubBackend:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict[str, Any] = {}

    def chat(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def ollama_response(
    content: str = "", tool_calls: list[tuple[str, dict[str, Any]]] | None = None
) -> SimpleNamespace:
    calls = [
        SimpleNamespace(function=SimpleNamespace(name=n, arguments=a)) for n, a in tool_calls or []
    ]
    message = SimpleNamespace(content=content, tool_calls=calls or None, thinking="hmm")
    return SimpleNamespace(message=message, prompt_eval_count=11, eval_count=7)


SPEC = ToolSpec("read_file", "Read a file", {"type": "object", "properties": {}})


def test_satisfies_llm_protocol(settings: Settings) -> None:
    assert isinstance(OllamaClient(settings, StubBackend()), LLMClient)


def test_request_shape(settings: Settings) -> None:
    backend = StubBackend(ollama_response("hi"))
    OllamaClient(settings, backend).chat([Message(Role.USER, "hello")], [SPEC])

    assert backend.kwargs["model"] == settings.model
    assert backend.kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert backend.kwargs["tools"][0]["function"]["name"] == "read_file"
    assert backend.kwargs["options"]["num_ctx"] == settings.num_ctx
    assert "think" not in backend.kwargs  # unset by default


def test_think_flag_is_forwarded_when_set(workspace: Any) -> None:
    settings = Settings(workspace=workspace, think=True)
    backend = StubBackend(ollama_response("hi"))
    OllamaClient(settings, backend).chat([Message(Role.USER, "x")], [])
    assert backend.kwargs["think"] is True


def test_parses_text_tool_calls_usage_and_thinking(settings: Settings) -> None:
    backend = StubBackend(ollama_response("", [("read_file", {"path": "a.py"})]))
    turn = OllamaClient(settings, backend).chat([Message(Role.USER, "x")], [SPEC])

    assert turn.message.tool_calls == (ToolCall("read_file", {"path": "a.py"}),)
    assert (turn.usage.prompt_tokens, turn.usage.completion_tokens) == (11, 7)
    assert turn.thinking == "hmm"


def test_assistant_tool_calls_and_tool_results_serialise() -> None:
    assistant = Message(Role.ASSISTANT, "", (ToolCall("read_file", {"path": "a"}),))
    tool = Message(Role.TOOL, "contents", tool_name="read_file")

    assert _to_wire_message(assistant)["tool_calls"] == [
        {"function": {"name": "read_file", "arguments": {"path": "a"}}}
    ]
    assert _to_wire_message(tool) == {
        "role": "tool",
        "content": "contents",
        "tool_name": "read_file",
    }


def test_wire_tool_format() -> None:
    assert _to_wire_tool(SPEC) == {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_missing_model_gives_actionable_error(settings: Settings) -> None:
    backend = StubBackend(error=ollama.ResponseError("model not found", 404))
    with pytest.raises(LLMError, match="ollama pull"):
        OllamaClient(settings, backend).chat([Message(Role.USER, "x")], [])


def test_other_server_error_includes_status(settings: Settings) -> None:
    backend = StubBackend(error=ollama.ResponseError("boom", 500))
    with pytest.raises(LLMError, match="500"):
        OllamaClient(settings, backend).chat([Message(Role.USER, "x")], [])


@pytest.mark.parametrize("error", [httpx.ConnectError("refused"), ConnectionError("refused")])
def test_connection_failure_suggests_starting_server(settings: Settings, error: Exception) -> None:
    with pytest.raises(LLMError, match="ollama serve"):
        OllamaClient(settings, StubBackend(error=error)).chat([Message(Role.USER, "x")], [])


def test_malformed_response_becomes_llm_error(settings: Settings) -> None:
    with pytest.raises(LLMError, match="Unexpected response"):
        OllamaClient(settings, StubBackend(SimpleNamespace())).chat([Message(Role.USER, "x")], [])
