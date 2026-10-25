"""Observer interface so the agent loop never depends on how progress is displayed or stored."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from forge_agent.messages import ToolCall

log = logging.getLogger(__name__)


class EventSink(Protocol):
    """Receives structured notifications as a run progresses."""

    def on_assistant_text(self, text: str) -> None: ...
    def on_thinking(self, text: str) -> None: ...
    def on_tool_call(self, step: int, call: ToolCall) -> None: ...
    def on_tool_result(self, step: int, call: ToolCall, content: str, is_error: bool) -> None: ...
    def on_notice(self, text: str) -> None: ...


class NullSink:
    """Discards everything. Default for library use and tests."""

    def on_assistant_text(self, text: str) -> None:
        pass

    def on_thinking(self, text: str) -> None:
        pass

    def on_tool_call(self, step: int, call: ToolCall) -> None:
        pass

    def on_tool_result(self, step: int, call: ToolCall, content: str, is_error: bool) -> None:
        pass

    def on_notice(self, text: str) -> None:
        pass


class CompositeSink:
    """Fans each event out to several sinks."""

    def __init__(self, *sinks: EventSink) -> None:
        self._sinks = sinks

    def on_assistant_text(self, text: str) -> None:
        for s in self._sinks:
            s.on_assistant_text(text)

    def on_thinking(self, text: str) -> None:
        for s in self._sinks:
            s.on_thinking(text)

    def on_tool_call(self, step: int, call: ToolCall) -> None:
        for s in self._sinks:
            s.on_tool_call(step, call)

    def on_tool_result(self, step: int, call: ToolCall, content: str, is_error: bool) -> None:
        for s in self._sinks:
            s.on_tool_result(step, call, content, is_error)

    def on_notice(self, text: str) -> None:
        for s in self._sinks:
            s.on_notice(text)


class JsonlTraceSink:
    """Appends one JSON object per event to a file, giving a replayable record of every run."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        self.path = directory / f"run-{stamp}.jsonl"

    def _write(self, kind: str, **data: Any) -> None:
        record = {"ts": datetime.now(UTC).isoformat(), "event": kind, **data}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def on_assistant_text(self, text: str) -> None:
        self._write("assistant", text=text)

    def on_thinking(self, text: str) -> None:
        self._write("thinking", text=text)

    def on_tool_call(self, step: int, call: ToolCall) -> None:
        self._write("tool_call", step=step, name=call.name, arguments=call.arguments)

    def on_tool_result(self, step: int, call: ToolCall, content: str, is_error: bool) -> None:
        self._write("tool_result", step=step, name=call.name, is_error=is_error, content=content)

    def on_notice(self, text: str) -> None:
        self._write("notice", text=text)
