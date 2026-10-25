"""Command policy and human-approval gate."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from forge_agent.config import ApprovalMode, Settings
from forge_agent.errors import SafetyError


class Risk(StrEnum):
    READ = "read"
    WRITE = "write"
    EXEC = "exec"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """What a human is asked to approve."""

    tool: str
    risk: Risk
    summary: str


class Approver(Protocol):
    """Decides whether a state-changing action may proceed."""

    def approve(self, request: ApprovalRequest) -> bool: ...


class ModeApprover:
    """Applies the configured ApprovalMode, delegating to `prompt` in ASK mode."""

    def __init__(self, mode: ApprovalMode, prompt: Approver | None = None) -> None:
        self._mode = mode
        self._prompt = prompt

    def approve(self, request: ApprovalRequest) -> bool:
        if request.risk is Risk.READ:
            return True
        if self._mode is ApprovalMode.DENY:
            return False
        if self._mode is ApprovalMode.AUTO:
            return True
        # ASK: with no way to ask, fail closed.
        return self._prompt.approve(request) if self._prompt else False


# Characters that only mean something to a shell. We never use a shell, so refuse them outright:
# a model emitting them is either confused or attempting chaining/redirection.
_SHELL_METACHARACTERS = frozenset(";&|<>`$\n\r(){}")


# Options on otherwise read-only git subcommands that write files or launch external programs.
_DANGEROUS_GIT_FLAGS = (
    "--output",
    "--ext-diff",
    "--open-files-in-pager",
    "--exec",
    "--upload-pack",
)


class CommandPolicy:
    """Validates a command line against an executable allowlist. No shell is ever involved."""

    def __init__(self, settings: Settings) -> None:
        self._allowed = frozenset(settings.allowed_commands)
        self._git_subcommands = frozenset(settings.allowed_git_subcommands)

    def parse(self, command: str) -> list[str]:
        """Return argv for an allowed command, or raise SafetyError explaining why not."""
        if not command.strip():
            msg = "empty command"
            raise SafetyError(msg)

        bad = sorted(_SHELL_METACHARACTERS.intersection(command))
        if bad:
            msg = (
                f"shell operators are not supported ({' '.join(repr(c) for c in bad)}). "
                "Run one simple command at a time; use the file tools for redirection."
            )
            raise SafetyError(msg)

        try:
            argv = shlex.split(command)
        except ValueError as exc:
            msg = f"could not parse command: {exc}"
            raise SafetyError(msg) from exc

        executable = argv[0]
        if "/" in executable:
            msg = "use a bare command name, not a path"
            raise SafetyError(msg)
        if executable not in self._allowed:
            allowed = ", ".join(sorted(self._allowed))
            msg = f"command '{executable}' is not allowed. Allowed: {allowed}"
            raise SafetyError(msg)

        if executable == "git":
            self._check_git(argv)
        return argv

    def _check_git(self, argv: list[str]) -> None:
        # The subcommand must come first. Global options (-c, -C, --git-dir, --exec-path) can
        # inject config or redirect git elsewhere, so they are not accepted before it.
        sub = argv[1] if len(argv) > 1 else None
        if sub is None or sub not in self._git_subcommands:
            allowed = ", ".join(sorted(self._git_subcommands))
            msg = f"git subcommand '{sub}' is not allowed (it must come first). Allowed: {allowed}"
            raise SafetyError(msg)
        for arg in argv[2:]:
            if arg.startswith(_DANGEROUS_GIT_FLAGS):
                msg = f"git option '{arg}' is not allowed"
                raise SafetyError(msg)
