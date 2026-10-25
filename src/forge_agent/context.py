"""History budgeting: keep the conversation inside the model's context window."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from forge_agent.messages import Message, Role

_ELIDED = "[older tool output removed to save context; re-run the tool if you need it]"
_CHARS_PER_TOKEN = 4  # crude but dependency-free; deliberately conservative for code


def estimate_tokens(messages: Sequence[Message]) -> int:
    chars = 0
    for m in messages:
        chars += len(m.content)
        chars += sum(len(c.name) + len(str(c.arguments)) for c in m.tool_calls)
    return chars // _CHARS_PER_TOKEN + len(messages) * 4


def fit_to_budget(
    messages: Sequence[Message], budget_tokens: int, *, keep_recent: int = 4
) -> list[Message]:
    """Return a copy of `messages` under budget by eliding the oldest tool outputs first.

    The system prompt, user messages, and assistant turns are never altered, and the most recent
    `keep_recent` tool results are preserved because the model is likely still reasoning about them.
    """
    result = list(messages)
    if estimate_tokens(result) <= budget_tokens:
        return result

    tool_indexes = [i for i, m in enumerate(result) if m.role is Role.TOOL]
    protected = set(tool_indexes[-keep_recent:]) if keep_recent else set()

    for i in tool_indexes:
        if estimate_tokens(result) <= budget_tokens:
            break
        if i in protected or result[i].content == _ELIDED:
            continue
        result[i] = replace(result[i], content=_ELIDED)
    return result
