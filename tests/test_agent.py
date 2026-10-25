from __future__ import annotations

from pathlib import Path

from forge_agent.agent import Agent, RunStatus
from forge_agent.config import ApprovalMode, Settings
from forge_agent.errors import LLMError
from forge_agent.messages import Role, ToolCall
from forge_agent.safety import ModeApprover
from forge_agent.tools import ToolRegistry, build_default_registry
from tests.helpers import RecordingApprover, ScriptedLLM, call, say


class CollectingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def on_assistant_text(self, text: str) -> None:
        self.events.append(("text", text))

    def on_thinking(self, text: str) -> None:
        self.events.append(("thinking", text))

    def on_tool_call(self, step: int, call: ToolCall) -> None:
        self.events.append(("call", call.name))

    def on_tool_result(self, step: int, call: ToolCall, content: str, is_error: bool) -> None:
        self.events.append(("result", is_error))

    def on_notice(self, text: str) -> None:
        self.events.append(("notice", text))


def make_agent(llm: ScriptedLLM, registry: ToolRegistry, settings: Settings, **kw: object) -> Agent:
    return Agent(llm=llm, tools=registry, settings=settings, **kw)  # type: ignore[arg-type]


def test_completes_after_tool_use(
    registry: ToolRegistry, settings: Settings, workspace: Path
) -> None:
    llm = ScriptedLLM(
        [
            call("write_file", path="hello.py", content="print('hi')\n"),
            call("run_command", command="python hello.py"),
            say("Created hello.py and ran it; it prints hi."),
        ]
    )
    result = make_agent(llm, registry, settings).run("make hello.py")

    assert result.ok
    assert result.steps == 3
    assert "hello.py" in result.final_message
    assert (workspace / "hello.py").read_text() == "print('hi')\n"
    assert result.usage.prompt_tokens == 30


def test_tool_results_are_fed_back_to_the_model(
    registry: ToolRegistry, settings: Settings, workspace: Path
) -> None:
    (workspace / "a.txt").write_text("needle")
    llm = ScriptedLLM([call("read_file", path="a.txt"), say("done")])
    make_agent(llm, registry, settings).run("look")

    second_call_history = llm.calls[1]
    tool_messages = [m for m in second_call_history if m.role is Role.TOOL]
    assert len(tool_messages) == 1
    assert "needle" in tool_messages[0].content
    assert tool_messages[0].tool_name == "read_file"


def test_tool_error_is_visible_to_model_so_it_can_recover(
    registry: ToolRegistry, settings: Settings
) -> None:
    llm = ScriptedLLM([call("read_file", path="nope.txt"), say("It does not exist.")])
    result = make_agent(llm, registry, settings).run("read nope")

    assert result.ok
    tool_message = next(m for m in llm.calls[1] if m.role is Role.TOOL)
    assert tool_message.content.startswith("ERROR:")


def test_stops_at_max_steps(registry: ToolRegistry, settings: Settings, workspace: Path) -> None:
    # Distinct arguments each time so the repetition guard does not fire first.
    varied = ScriptedLLM(
        [call("write_file", path=f"f{i}.txt", content=str(i)) for i in range(settings.max_steps)]
    )
    result = make_agent(varied, registry, settings).run("loop forever")

    assert result.status is RunStatus.MAX_STEPS
    assert result.steps == settings.max_steps
    assert not result.ok


def test_detects_repeated_identical_calls(registry: ToolRegistry, settings: Settings) -> None:
    llm = ScriptedLLM([call("list_dir", path=".")] * 5)
    result = make_agent(llm, registry, settings).run("spin")

    assert result.status is RunStatus.LOOP_DETECTED
    assert result.steps == settings.max_repeated_calls


def test_alternating_calls_are_not_flagged_as_a_loop(
    registry: ToolRegistry, settings: Settings
) -> None:
    llm = ScriptedLLM(
        [
            call("list_dir", path="."),
            call("list_dir", path="x"),
            call("list_dir", path="."),
            say("ok"),
        ]
    )
    assert make_agent(llm, registry, settings).run("t").ok


def test_llm_failure_is_reported_not_raised(registry: ToolRegistry, settings: Settings) -> None:
    llm = ScriptedLLM([LLMError("Ollama is down")])
    result = make_agent(llm, registry, settings).run("t")

    assert result.status is RunStatus.LLM_ERROR
    assert result.error == "Ollama is down"


def test_empty_reply_is_nudged_once_then_accepted(
    registry: ToolRegistry, settings: Settings
) -> None:
    llm = ScriptedLLM([say(""), say("Here is my summary.")])
    result = make_agent(llm, registry, settings).run("t")

    assert result.ok
    assert result.final_message == "Here is my summary."
    assert llm.calls[1][-1].role is Role.USER  # the nudge


def test_second_empty_reply_ends_the_run(registry: ToolRegistry, settings: Settings) -> None:
    llm = ScriptedLLM([say(""), say("")])
    result = make_agent(llm, registry, settings).run("t")
    assert result.ok
    assert result.final_message == ""


def test_denied_action_is_reported_to_model(settings: Settings, workspace: Path) -> None:
    approver = RecordingApprover(answer=False)
    registry = build_default_registry(settings, approver)
    llm = ScriptedLLM([call("write_file", path="x.txt", content="x"), say("Could not write.")])
    result = make_agent(llm, registry, settings).run("write")

    assert result.ok
    assert not (workspace / "x.txt").exists()
    tool_message = next(m for m in llm.calls[1] if m.role is Role.TOOL)
    assert "denied" in tool_message.content


def test_deny_mode_makes_agent_read_only(workspace: Path) -> None:
    settings = Settings(workspace=workspace, approval_mode=ApprovalMode.DENY)
    registry = build_default_registry(settings, ModeApprover(ApprovalMode.DENY))
    llm = ScriptedLLM([call("write_file", path="x.txt", content="x"), say("ok")])
    make_agent(llm, registry, settings).run("t")
    assert not (workspace / "x.txt").exists()


def test_events_are_emitted_in_order(registry: ToolRegistry, settings: Settings) -> None:
    sink = CollectingSink()
    llm = ScriptedLLM([call("list_dir"), say("fin")])
    make_agent(llm, registry, settings, sink=sink).run("t")

    assert sink.events == [("call", "list_dir"), ("result", False), ("text", "fin")]


def test_system_prompt_and_task_lead_the_history(
    registry: ToolRegistry, settings: Settings
) -> None:
    llm = ScriptedLLM([say("done")])
    make_agent(llm, registry, settings, system_prompt="SYS").run("do the thing")

    first = llm.calls[0]
    assert (first[0].role, first[0].content) == (Role.SYSTEM, "SYS")
    assert (first[1].role, first[1].content) == (Role.USER, "do the thing")
