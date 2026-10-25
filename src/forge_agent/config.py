"""Runtime configuration, loaded from defaults, a .env file, or FORGE_* environment variables."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApprovalMode(StrEnum):
    """How the agent treats actions that modify state."""

    ASK = "ask"  # prompt a human before every write or command
    AUTO = "auto"  # auto-approve writes inside the workspace and allowlisted commands
    DENY = "deny"  # read-only: every write or command is refused


class Settings(BaseSettings):
    """All tunables in one validated object. Override with FORGE_<NAME> env vars."""

    model_config = SettingsConfigDict(
        env_prefix="FORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- model backend ---
    model: str = "qwen3:8b"
    ollama_host: str = "http://localhost:11434"
    num_ctx: int = Field(
        default=16384, ge=2048, description="Context window requested from Ollama."
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    think: bool | None = Field(
        default=None,
        description="Enable model 'thinking' mode. None leaves the model default untouched.",
    )
    request_timeout_s: float = Field(default=300.0, gt=0)

    # --- agent loop ---
    max_steps: int = Field(default=25, ge=1, le=200)
    max_repeated_calls: int = Field(
        default=3,
        ge=2,
        description="Abort when the same tool call repeats this many times in a row.",
    )
    context_budget_tokens: int = Field(
        default=12000,
        ge=1000,
        description="Estimated token budget for history; old tool output is elided beyond it.",
    )

    # --- tools & safety ---
    workspace: Path = Field(default_factory=Path.cwd)
    approval_mode: ApprovalMode = ApprovalMode.ASK
    command_timeout_s: float = Field(default=60.0, gt=0)
    max_tool_output_chars: int = Field(default=8000, ge=500)
    max_file_bytes: int = Field(default=512_000, ge=1024)
    allowed_commands: tuple[str, ...] = (
        "python",
        "python3",
        "pytest",
        "uv",
        "ruff",
        "mypy",
        "git",
        "ls",
        "cat",
        "grep",
        "head",
        "tail",
        "wc",
    )
    # Only these git subcommands may run; anything that rewrites history or hits the network is out.
    allowed_git_subcommands: tuple[str, ...] = (
        "status",
        "diff",
        "log",
        "show",
        "branch",
        "ls-files",
    )

    # --- observability ---
    trace_dir: Path | None = Field(
        default=None,
        description="Where to write JSONL run traces. Default: <workspace>/.forge/runs.",
    )
    log_level: str = "WARNING"

    @field_validator("workspace")
    @classmethod
    def _resolve_workspace(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if not resolved.is_dir():
            msg = f"workspace is not an existing directory: {resolved}"
            raise ValueError(msg)
        return resolved

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        upper = value.upper()
        if upper not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            msg = f"invalid log level: {value}"
            raise ValueError(msg)
        return upper

    @property
    def resolved_trace_dir(self) -> Path:
        return self.trace_dir or (self.workspace / ".forge" / "runs")
