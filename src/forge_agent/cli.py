"""Command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from forge_agent import __version__
from forge_agent.agent import Agent, RunResult
from forge_agent.config import ApprovalMode, Settings
from forge_agent.doctor import CheckStatus, run_checks
from forge_agent.events import CompositeSink, EventSink, JsonlTraceSink
from forge_agent.llm import OllamaClient
from forge_agent.messages import ToolCall
from forge_agent.safety import ApprovalRequest, ModeApprover
from forge_agent.tools import build_default_registry

app = typer.Typer(
    name="forge",
    help="A local-first, safety-conscious coding agent built on Ollama.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

_MAX_PREVIEW_CHARS = 600


class RichSink:
    """Renders agent progress to the terminal."""

    def __init__(self, show_thinking: bool) -> None:
        self._show_thinking = show_thinking

    def on_assistant_text(self, text: str) -> None:
        console.print(f"[bold cyan]agent[/] {escape(text)}")

    def on_thinking(self, text: str) -> None:
        if self._show_thinking:
            console.print(f"[dim italic]{escape(text)}[/]")

    def on_tool_call(self, step: int, call: ToolCall) -> None:
        console.print(f"[yellow]step {step}[/] [bold]{call.name}[/] {escape(str(call.arguments))}")

    def on_tool_result(self, step: int, call: ToolCall, content: str, is_error: bool) -> None:
        preview = (
            content if len(content) <= _MAX_PREVIEW_CHARS else content[:_MAX_PREVIEW_CHARS] + " ..."
        )
        style = "red" if is_error else "dim"
        console.print(f"[{style}]{escape(preview)}[/]")

    def on_notice(self, text: str) -> None:
        console.print(f"[bold magenta]{escape(text)}[/]")


class ConsoleApprover:
    """Asks the human at the terminal."""

    def approve(self, request: ApprovalRequest) -> bool:
        console.print(Panel(escape(request.summary), title=f"Approve {request.risk.value}?"))
        return typer.confirm("Proceed?", default=False)


def _build_settings(**overrides: Any) -> Settings:
    try:
        return Settings(**{k: v for k, v in overrides.items() if v is not None})
    except ValidationError as exc:
        err_console.print(f"[red]Invalid configuration:[/]\n{exc}")
        raise typer.Exit(code=2) from exc


def _show_version(value: bool) -> None:
    if value:
        console.print(f"forge-agent {__version__}")
        raise typer.Exit


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """Forge: a local-first coding agent."""


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="What you want the agent to do.")],
    workspace: Annotated[
        Path | None, typer.Option(help="Project directory (default: cwd).")
    ] = None,
    model: Annotated[str | None, typer.Option(help="Ollama model tag.")] = None,
    host: Annotated[str | None, typer.Option(help="Ollama server URL.")] = None,
    approval: Annotated[
        ApprovalMode | None, typer.Option(help="ask | auto | deny (deny = read-only).")
    ] = None,
    max_steps: Annotated[int | None, typer.Option(min=1, max=200)] = None,
    think: Annotated[
        bool | None, typer.Option("--think/--no-think", help="Model thinking mode.")
    ] = None,
    show_thinking: Annotated[bool, typer.Option(help="Print the model's reasoning.")] = False,
    trace: Annotated[bool, typer.Option(help="Write a JSONL trace of the run.")] = True,
    log_level: Annotated[str | None, typer.Option(help="DEBUG, INFO, WARNING, ...")] = None,
) -> None:
    """Run the agent on a task."""
    settings = _build_settings(
        workspace=workspace,
        model=model,
        ollama_host=host,
        approval_mode=approval,
        max_steps=max_steps,
        think=think,
        log_level=log_level,
    )
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    sinks: list[EventSink] = [RichSink(show_thinking)]
    if trace:
        trace_sink = JsonlTraceSink(settings.resolved_trace_dir)
        sinks.append(trace_sink)
        console.print(f"[dim]trace: {trace_sink.path}[/]")

    approver = ModeApprover(settings.approval_mode, prompt=ConsoleApprover())
    agent = Agent(
        llm=OllamaClient(settings),
        tools=build_default_registry(settings, approver),
        settings=settings,
        sink=CompositeSink(*sinks),
    )

    console.print(
        f"[dim]model={settings.model} workspace={settings.workspace} "
        f"approval={settings.approval_mode}[/]"
    )
    result = agent.run(task)
    _report(result)
    raise typer.Exit(code=0 if result.ok else 1)


def _report(result: RunResult) -> None:
    usage = result.usage
    summary = (
        f"status={result.status.value} steps={result.steps} "
        f"tokens(prompt/completion)={usage.prompt_tokens}/{usage.completion_tokens}"
    )
    if result.ok:
        console.print(Panel(escape(result.final_message or "(no summary)"), title="Done"))
        console.print(f"[green]{summary}[/]")
    else:
        err_console.print(f"[red]{escape(result.error or 'run did not complete')}[/]")
        err_console.print(f"[red]{summary}[/]")


@app.command()
def doctor(
    model: Annotated[str | None, typer.Option(help="Ollama model tag to check.")] = None,
    host: Annotated[str | None, typer.Option(help="Ollama server URL.")] = None,
) -> None:
    """Check that Ollama, the model, and tool calling are ready."""
    settings = _build_settings(model=model, ollama_host=host)
    failed = False
    for check in run_checks(settings):
        colour = {CheckStatus.OK: "green", CheckStatus.WARN: "yellow", CheckStatus.FAIL: "red"}[
            check.status
        ]
        console.print(
            f"[{colour}]{check.status.value.upper():<4}[/] {check.name}: {escape(check.detail)}"
        )
        failed = failed or check.status is CheckStatus.FAIL
    raise typer.Exit(code=1 if failed else 0)
