"""Command execution: allowlisted executables, no shell, workspace cwd, timeout, scrubbed env."""

from __future__ import annotations

import os
import subprocess
from typing import ClassVar

from pydantic import BaseModel, Field

from forge_agent.config import Settings
from forge_agent.errors import SafetyError, ToolError
from forge_agent.safety import CommandPolicy, Risk, Workspace
from forge_agent.tools.base import Tool

# Only these variables are passed through; secrets in the parent environment never reach commands.
_ENV_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "VIRTUAL_ENV", "UV_CACHE_DIR")


class _RunArgs(BaseModel):
    command: str = Field(
        description="A single command line, e.g. 'pytest -q' or 'git diff'. No pipes, &&, or ;."
    )


class RunCommand(Tool[_RunArgs]):
    name = "run_command"
    description = (
        "Run one allowlisted command in the workspace and return its exit code and output. "
        "Shell features (pipes, redirects, &&, ;, $VAR) are not available."
    )
    risk = Risk.EXEC
    args_model: ClassVar[type[BaseModel]] = _RunArgs

    def __init__(self, workspace: Workspace, policy: CommandPolicy, settings: Settings) -> None:
        self._ws = workspace
        self._policy = policy
        self._timeout = settings.command_timeout_s

    def describe(self, args: _RunArgs) -> str:
        return f"run: {args.command}"

    def run(self, args: _RunArgs) -> str:
        argv = self._policy.parse(args.command)
        self._check_path_arguments(argv)
        try:
            completed = subprocess.run(
                argv,
                cwd=self._ws.root,
                env=_scrubbed_env(),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            msg = f"command timed out after {self._timeout:.0f}s"
            raise ToolError(msg) from exc
        except FileNotFoundError as exc:
            msg = f"executable not found on PATH: {argv[0]}"
            raise ToolError(msg) from exc

        parts = [f"exit code: {completed.returncode}"]
        if completed.stdout:
            parts.append(f"--- stdout ---\n{completed.stdout.rstrip()}")
        if completed.stderr:
            parts.append(f"--- stderr ---\n{completed.stderr.rstrip()}")
        return "\n".join(parts)

    def _check_path_arguments(self, argv: list[str]) -> None:
        """Reject arguments that point outside the workspace (e.g. `cat ../../.ssh/id_rsa`).

        This is defence in depth against mistakes, NOT a sandbox: an allowlisted interpreter such
        as `python` can still run arbitrary code. See ARCHITECTURE.md ("Threat model").
        """
        for arg in argv[1:]:
            if arg.startswith("-") and "=" not in arg:
                continue
            value = arg.split("=", 1)[1] if arg.startswith("-") else arg
            looks_like_path = value.startswith(("/", "~", ".")) or "/" in value
            if looks_like_path:
                try:
                    self._ws.resolve(value)
                except SafetyError as exc:
                    msg = f"argument '{arg}' points outside the workspace"
                    raise SafetyError(msg) from exc


def _scrubbed_env() -> dict[str, str]:
    return {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
