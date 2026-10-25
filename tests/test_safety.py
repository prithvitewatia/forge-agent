from __future__ import annotations

from pathlib import Path

import pytest

from forge_agent.config import ApprovalMode, Settings
from forge_agent.errors import SafetyError
from forge_agent.safety import ApprovalRequest, CommandPolicy, ModeApprover, Risk, Workspace
from tests.helpers import RecordingApprover


class TestWorkspace:
    def test_resolves_relative_path_inside_root(self, workspace: Path) -> None:
        (workspace / "a.txt").write_text("x")
        assert Workspace(workspace).resolve("a.txt") == workspace / "a.txt"

    @pytest.mark.parametrize("bad", ["../outside.txt", "/etc/passwd", "a/../../x", "~/secret"])
    def test_rejects_escapes(self, workspace: Path, bad: str) -> None:
        with pytest.raises(SafetyError):
            Workspace(workspace).resolve(bad)

    def test_rejects_symlink_pointing_outside(self, workspace: Path, tmp_path: Path) -> None:
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        (workspace / "link").symlink_to(outside)
        with pytest.raises(SafetyError):
            Workspace(workspace).resolve("link")

    @pytest.mark.parametrize("bad", ["", "a\x00b"])
    def test_rejects_invalid_strings(self, workspace: Path, bad: str) -> None:
        with pytest.raises(SafetyError):
            Workspace(workspace).resolve(bad)

    def test_protects_git_directory_for_writes_only(self, workspace: Path) -> None:
        (workspace / ".git").mkdir()
        ws = Workspace(workspace)
        assert ws.resolve(".git/config")  # reading is fine
        with pytest.raises(SafetyError):
            ws.resolve(".git/config", for_write=True)


class TestCommandPolicy:
    @pytest.fixture
    def policy(self, settings: Settings) -> CommandPolicy:
        return CommandPolicy(settings)

    def test_accepts_allowed_command(self, policy: CommandPolicy) -> None:
        assert policy.parse("pytest -q tests/") == ["pytest", "-q", "tests/"]

    def test_respects_quoting(self, policy: CommandPolicy) -> None:
        assert policy.parse("grep 'two words' file.txt") == ["grep", "two words", "file.txt"]

    @pytest.mark.parametrize(
        "command",
        ["ls; rm -rf /", "ls && ls", "cat a | head", "echo $HOME", "ls > out", "ls `id`", "ls\nls"],
    )
    def test_rejects_shell_operators(self, policy: CommandPolicy, command: str) -> None:
        with pytest.raises(SafetyError, match="shell operators"):
            policy.parse(command)

    @pytest.mark.parametrize("command", ["rm -rf x", "curl http://x", "sudo ls", "bash -c ls"])
    def test_rejects_unlisted_executables(self, policy: CommandPolicy, command: str) -> None:
        with pytest.raises(SafetyError, match="not allowed"):
            policy.parse(command)

    def test_rejects_executable_paths(self, policy: CommandPolicy) -> None:
        with pytest.raises(SafetyError, match="bare command"):
            policy.parse("/bin/ls")

    @pytest.mark.parametrize("command", ["git status", "git diff --stat", "git log -n 3 --oneline"])
    def test_allows_read_only_git(self, policy: CommandPolicy, command: str) -> None:
        assert policy.parse(command)[0] == "git"

    @pytest.mark.parametrize(
        "command",
        [
            "git push",
            "git commit -m x",
            "git reset --hard",
            "git",
            "git -C . log",  # global options before the subcommand are refused
            "git -c core.pager=sh log",
            "git --git-dir=/elsewhere status",
        ],
    )
    def test_rejects_mutating_or_redirected_git(self, policy: CommandPolicy, command: str) -> None:
        with pytest.raises(SafetyError, match="git subcommand"):
            policy.parse(command)

    @pytest.mark.parametrize("command", ["git diff --output=x.txt", "git log --ext-diff"])
    def test_rejects_git_options_that_write_or_exec(
        self, policy: CommandPolicy, command: str
    ) -> None:
        with pytest.raises(SafetyError, match="git option"):
            policy.parse(command)

    @pytest.mark.parametrize("command", ["", "   "])
    def test_rejects_empty(self, policy: CommandPolicy, command: str) -> None:
        with pytest.raises(SafetyError, match="empty"):
            policy.parse(command)

    def test_rejects_unbalanced_quotes(self, policy: CommandPolicy) -> None:
        with pytest.raises(SafetyError, match="could not parse"):
            policy.parse("grep 'oops file")


class TestModeApprover:
    def _request(self, risk: Risk) -> ApprovalRequest:
        return ApprovalRequest(tool="t", risk=risk, summary="s")

    def test_reads_always_allowed(self) -> None:
        assert ModeApprover(ApprovalMode.DENY).approve(self._request(Risk.READ))

    def test_deny_blocks_writes_and_exec(self) -> None:
        approver = ModeApprover(ApprovalMode.DENY)
        assert not approver.approve(self._request(Risk.WRITE))
        assert not approver.approve(self._request(Risk.EXEC))

    def test_auto_allows(self) -> None:
        assert ModeApprover(ApprovalMode.AUTO).approve(self._request(Risk.WRITE))

    def test_ask_delegates_to_prompt(self) -> None:
        prompt = RecordingApprover(answer=False)
        assert not ModeApprover(ApprovalMode.ASK, prompt).approve(self._request(Risk.WRITE))
        assert len(prompt.requests) == 1

    def test_ask_without_prompt_fails_closed(self) -> None:
        assert not ModeApprover(ApprovalMode.ASK).approve(self._request(Risk.EXEC))
