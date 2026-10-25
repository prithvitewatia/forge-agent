from __future__ import annotations

from forge_agent.context import estimate_tokens, fit_to_budget
from forge_agent.messages import Message, Role


def tool_msg(text: str) -> Message:
    return Message(Role.TOOL, text, tool_name="read_file")


def test_under_budget_is_unchanged() -> None:
    history = [Message(Role.SYSTEM, "s"), Message(Role.USER, "u"), tool_msg("small")]
    assert fit_to_budget(history, 10_000) == history


def test_elides_oldest_tool_output_first_and_keeps_recent() -> None:
    big = "x" * 4000
    history = [
        Message(Role.SYSTEM, "sys"),
        Message(Role.USER, "task"),
        tool_msg(big + "OLD"),
        tool_msg(big + "MID"),
        tool_msg("recent"),
    ]
    fitted = fit_to_budget(history, budget_tokens=1200, keep_recent=1)

    assert "removed to save context" in fitted[2].content
    assert fitted[4].content == "recent"
    assert estimate_tokens(fitted) <= 1200 or "removed" in fitted[3].content


def test_never_alters_system_user_or_assistant_messages() -> None:
    long_user = "u" * 8000
    history = [
        Message(Role.SYSTEM, "s" * 4000),
        Message(Role.USER, long_user),
        Message(Role.ASSISTANT, "a" * 4000),
    ]
    assert fit_to_budget(history, budget_tokens=100) == history


def test_does_not_mutate_input() -> None:
    history = [Message(Role.USER, "t"), tool_msg("y" * 8000), tool_msg("z")]
    fit_to_budget(history, budget_tokens=100, keep_recent=1)
    assert history[1].content == "y" * 8000
