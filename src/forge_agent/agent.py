"""The agent loop: model -> tool calls -> observations -> model, until done or a guard trips."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from forge_agent.config import Settings
from forge_agent.context import fit_to_budget
from forge_agent.errors import LLMError
from forge_agent.events import EventSink, NullSink
from forge_agent.llm import LLMClient
from forge_agent.messages import Message, Role, ToolCall, Usage
from forge_agent.prompts import SYSTEM_PROMPT
from forge_agent.tools import ToolRegistry

log = logging.getLogger(__name__)

_EMPTY_REPLY_NUDGE = (
    "Your last reply was empty. Either call a tool to continue, or give your final summary."
)


class RunStatus(StrEnum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    LOOP_DETECTED = "loop_detected"
    LLM_ERROR = "llm_error"


@dataclass(slots=True)
class RunResult:
    status: RunStatus
    final_message: str
    steps: int
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.COMPLETED


class Agent:
    """Runs one task to completion. Stateless between runs; build a new history each time."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        settings: Settings,
        sink: EventSink | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._settings = settings
        self._sink: EventSink = sink or NullSink()
        self._system_prompt = system_prompt

    def run(self, task: str) -> RunResult:
        history: list[Message] = [
            Message(Role.SYSTEM, self._system_prompt),
            Message(Role.USER, task),
        ]
        specs = self._tools.specs()
        prompt_tokens = completion_tokens = 0
        last_fingerprint: str | None = None
        repeat_count = 0
        nudged = False

        for step in range(1, self._settings.max_steps + 1):
            window = fit_to_budget(history, self._settings.context_budget_tokens)
            try:
                turn = self._llm.chat(window, specs)
            except LLMError as exc:
                log.error("LLM failure at step %d: %s", step, exc)
                return RunResult(
                    RunStatus.LLM_ERROR, "", step, Usage(prompt_tokens, completion_tokens), str(exc)
                )

            prompt_tokens += turn.usage.prompt_tokens
            completion_tokens += turn.usage.completion_tokens
            message = turn.message
            history.append(message)

            if turn.thinking:
                self._sink.on_thinking(turn.thinking)
            if message.content:
                self._sink.on_assistant_text(message.content)

            if not message.tool_calls:
                if not message.content.strip() and not nudged:
                    nudged = True
                    history.append(Message(Role.USER, _EMPTY_REPLY_NUDGE))
                    continue
                return RunResult(
                    RunStatus.COMPLETED,
                    message.content,
                    step,
                    Usage(prompt_tokens, completion_tokens),
                )

            for call in message.tool_calls:
                fingerprint = call.fingerprint()
                repeat_count = repeat_count + 1 if fingerprint == last_fingerprint else 1
                last_fingerprint = fingerprint
                if repeat_count >= self._settings.max_repeated_calls:
                    notice = f"Stopping: '{call.name}' was called {repeat_count} times in a row."
                    self._sink.on_notice(notice)
                    return RunResult(
                        RunStatus.LOOP_DETECTED,
                        message.content,
                        step,
                        Usage(prompt_tokens, completion_tokens),
                        notice,
                    )
                history.append(self._execute(step, call))

        notice = f"Stopped after {self._settings.max_steps} steps without finishing."
        self._sink.on_notice(notice)
        return RunResult(
            RunStatus.MAX_STEPS,
            "",
            self._settings.max_steps,
            Usage(prompt_tokens, completion_tokens),
            notice,
        )

    def _execute(self, step: int, call: ToolCall) -> Message:
        self._sink.on_tool_call(step, call)
        result = self._tools.execute(call)
        self._sink.on_tool_result(step, call, result.content, result.is_error)
        prefix = "ERROR: " if result.is_error else ""
        return Message(Role.TOOL, prefix + result.content, tool_name=call.name)
