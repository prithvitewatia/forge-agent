from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from forge_agent import __version__
from forge_agent.cli import app
from forge_agent.config import ApprovalMode, Settings
from forge_agent.doctor import CheckStatus, run_checks
from forge_agent.messages import ToolCall

runner = CliRunner()


class TestSettings:
    def test_defaults(self, workspace: Path) -> None:
        s = Settings(workspace=workspace)
        assert s.model == "qwen3:8b"
        assert s.approval_mode is ApprovalMode.ASK  # safe by default
        assert s.workspace == workspace.resolve()

    def test_env_override(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_MODEL", "llama3.1:8b")
        monkeypatch.setenv("FORGE_MAX_STEPS", "7")
        s = Settings(workspace=workspace)
        assert (s.model, s.max_steps) == ("llama3.1:8b", 7)

    def test_rejects_missing_workspace(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="workspace"):
            Settings(workspace=tmp_path / "nope")

    def test_rejects_out_of_range_values(self, workspace: Path) -> None:
        with pytest.raises(ValidationError):
            Settings(workspace=workspace, max_steps=0)

    def test_is_immutable(self, workspace: Path) -> None:
        with pytest.raises(ValidationError):
            Settings(workspace=workspace).model = "x"  # type: ignore[misc]

    def test_log_level_is_normalised_and_validated(self, workspace: Path) -> None:
        assert Settings(workspace=workspace, log_level="debug").log_level == "DEBUG"
        with pytest.raises(ValidationError):
            Settings(workspace=workspace, log_level="loud")

    def test_trace_dir_defaults_inside_workspace(self, workspace: Path) -> None:
        assert Settings(workspace=workspace).resolved_trace_dir == workspace / ".forge" / "runs"


class FakeApi:
    def __init__(
        self, models: list[str], capabilities: list[str] | None, fail: bool = False
    ) -> None:
        self._models = models
        self._capabilities = capabilities
        self._fail = fail

    def list(self) -> Any:
        if self._fail:
            raise ConnectionError("refused")
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in self._models])

    def show(self, model: str) -> Any:
        return SimpleNamespace(capabilities=self._capabilities)


def statuses(checks: list[Any]) -> dict[str, CheckStatus]:
    return {c.name: c.status for c in checks}


class TestDoctor:
    @pytest.fixture(autouse=True)
    def _no_network(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        def refuse(*_: Any, **__: Any) -> Any:
            raise httpx.ConnectError("no network in tests")

        monkeypatch.setattr(httpx, "get", refuse)

    def test_all_good(self, settings: Settings) -> None:
        api = FakeApi(["qwen3:8b"], ["completion", "tools", "thinking"])
        result = statuses(run_checks(settings, api))
        assert result["server"] is CheckStatus.OK
        assert result["model"] is CheckStatus.OK
        assert result["tool calling"] is CheckStatus.OK

    def test_server_down(self, settings: Settings) -> None:
        result = statuses(run_checks(settings, FakeApi([], None, fail=True)))
        assert result["server"] is CheckStatus.FAIL
        assert "model" not in result

    def test_model_not_pulled(self, settings: Settings) -> None:
        result = statuses(run_checks(settings, FakeApi(["llama3.1:8b"], ["tools"])))
        assert result["model"] is CheckStatus.FAIL

    def test_latest_tag_is_matched(self, workspace: Path) -> None:
        settings = Settings(workspace=workspace, model="mymodel")
        result = statuses(run_checks(settings, FakeApi(["mymodel:latest"], ["tools"])))
        assert result["model"] is CheckStatus.OK

    def test_model_without_tools_fails(self, settings: Settings) -> None:
        result = statuses(run_checks(settings, FakeApi(["qwen3:8b"], ["completion"])))
        assert result["tool calling"] is CheckStatus.FAIL

    def test_missing_capabilities_is_a_warning_not_a_failure(self, settings: Settings) -> None:
        result = statuses(run_checks(settings, FakeApi(["qwen3:8b"], None)))
        assert result["capabilities"] is CheckStatus.WARN

    def test_think_requires_thinking_capability(self, workspace: Path) -> None:
        settings = Settings(workspace=workspace, think=True)
        result = statuses(run_checks(settings, FakeApi(["qwen3:8b"], ["tools"])))
        assert result["thinking"] is CheckStatus.FAIL


class TestCli:
    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_help_lists_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "doctor" in result.output

    def test_invalid_workspace_exits_with_config_error(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["run", "task", "--workspace", str(tmp_path / "missing")])
        assert result.exit_code == 2

    def test_doctor_exit_code_reflects_failure(
        self, monkeypatch: pytest.MonkeyPatch, workspace: Path
    ) -> None:
        from forge_agent import cli
        from forge_agent.doctor import Check

        monkeypatch.setattr(
            cli, "run_checks", lambda _s: [Check("server", CheckStatus.FAIL, "down")]
        )
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1
        assert "down" in result.output

    def test_run_end_to_end_with_fake_model(
        self, monkeypatch: pytest.MonkeyPatch, workspace: Path
    ) -> None:
        from forge_agent import cli
        from tests.helpers import ScriptedLLM, call, say

        script = ScriptedLLM(
            [call("write_file", path="out.txt", content="done"), say("All finished.")]
        )
        monkeypatch.setattr(cli, "OllamaClient", lambda _settings: script)

        result = runner.invoke(
            app,
            [
                "run",
                "write out.txt",
                "--workspace",
                str(workspace),
                "--approval",
                "auto",
                "--no-trace",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (workspace / "out.txt").read_text() == "done"
        assert "All finished." in result.output

    def test_run_writes_trace_file(self, monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
        from forge_agent import cli
        from tests.helpers import ScriptedLLM, say

        monkeypatch.setattr(cli, "OllamaClient", lambda _s: ScriptedLLM([say("ok")]))
        runner.invoke(app, ["run", "t", "--workspace", str(workspace)])

        traces = list((workspace / ".forge" / "runs").glob("run-*.jsonl"))
        assert len(traces) == 1
        assert '"event": "assistant"' in traces[0].read_text()

    def test_failed_run_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, workspace: Path
    ) -> None:
        from forge_agent import cli
        from forge_agent.errors import LLMError
        from tests.helpers import ScriptedLLM

        monkeypatch.setattr(cli, "OllamaClient", lambda _s: ScriptedLLM([LLMError("down")]))
        result = runner.invoke(app, ["run", "t", "--workspace", str(workspace), "--no-trace"])
        assert result.exit_code == 1


def test_tool_call_fingerprint_is_order_insensitive() -> None:
    a = ToolCall("t", {"x": 1, "y": 2})
    b = ToolCall("t", {"y": 2, "x": 1})
    assert a.fingerprint() == b.fingerprint()
