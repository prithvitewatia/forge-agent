from __future__ import annotations

from pathlib import Path

import pytest

from forge_agent.config import ApprovalMode, Settings
from forge_agent.messages import ToolCall
from forge_agent.safety import ModeApprover
from forge_agent.tools import ToolRegistry, build_default_registry
from tests.helpers import RecordingApprover


def run(registry: ToolRegistry, name: str, **args: object) -> tuple[str, bool]:
    result = registry.execute(ToolCall(name, dict(args)))
    return result.content, result.is_error


class TestRegistry:
    def test_unknown_tool_lists_available(self, registry: ToolRegistry) -> None:
        content, is_error = run(registry, "nope")
        assert is_error
        assert "read_file" in content

    def test_invalid_arguments_are_reported_not_raised(self, registry: ToolRegistry) -> None:
        content, is_error = run(registry, "read_file")  # missing path
        assert is_error
        assert "path" in content

    def test_wrong_argument_type_is_reported(self, registry: ToolRegistry) -> None:
        _, is_error = run(registry, "read_file", path="a.txt", start_line="abc")
        assert is_error

    def test_specs_are_valid_json_schema_objects(self, registry: ToolRegistry) -> None:
        specs = {s.name: s for s in registry.specs()}
        assert set(specs) == {"list_dir", "read_file", "write_file", "edit_file", "run_command"}
        for spec in specs.values():
            assert spec.parameters["type"] == "object"
            assert "title" not in spec.parameters

    def test_duplicate_registration_rejected(
        self, registry: ToolRegistry, settings: Settings
    ) -> None:
        other = build_default_registry(settings, ModeApprover(ApprovalMode.AUTO))
        with pytest.raises(ValueError, match="duplicate"):
            registry.register(other._tools["read_file"])

    def test_long_output_is_truncated(self, workspace: Path, settings: Settings) -> None:
        small = Settings(
            workspace=workspace, max_tool_output_chars=500, approval_mode=ApprovalMode.AUTO
        )
        registry = build_default_registry(small, ModeApprover(ApprovalMode.AUTO))
        (workspace / "big.txt").write_text("\n".join(f"line {i}" for i in range(300)))
        content, is_error = run(registry, "read_file", path="big.txt", max_lines=300)
        assert not is_error
        assert "characters omitted" in content
        assert len(content) < 700

    def test_denied_write_does_not_touch_disk(self, workspace: Path, settings: Settings) -> None:
        approver = RecordingApprover(answer=False)
        registry = build_default_registry(settings, approver)
        content, is_error = run(registry, "write_file", path="x.txt", content="hi")
        assert is_error
        assert "denied" in content
        assert not (workspace / "x.txt").exists()
        assert len(approver.requests) == 1

    def test_reads_never_ask_for_approval(self, workspace: Path, settings: Settings) -> None:
        approver = RecordingApprover(answer=False)
        registry = build_default_registry(settings, approver)
        (workspace / "a.txt").write_text("x")
        _, is_error = run(registry, "read_file", path="a.txt")
        assert not is_error
        assert approver.requests == []


class TestFilesystemTools:
    def test_write_then_read_with_line_numbers(self, registry: ToolRegistry) -> None:
        run(registry, "write_file", path="pkg/mod.py", content="a = 1\nb = 2\n")
        content, is_error = run(registry, "read_file", path="pkg/mod.py")
        assert not is_error
        assert content.splitlines()[:2] == ["1\ta = 1", "2\tb = 2"]

    def test_read_window(self, registry: ToolRegistry, workspace: Path) -> None:
        (workspace / "f.txt").write_text("\n".join(str(i) for i in range(1, 11)))
        content, _ = run(registry, "read_file", path="f.txt", start_line=3, max_lines=2)
        assert content.startswith("3\t3\n4\t4")
        assert "showing lines 3-4 of 10" in content

    def test_read_missing_file(self, registry: ToolRegistry) -> None:
        content, is_error = run(registry, "read_file", path="missing.txt")
        assert is_error
        assert "not found" in content

    def test_read_binary_file(self, registry: ToolRegistry, workspace: Path) -> None:
        (workspace / "b.bin").write_bytes(b"\xff\xfe\x00\x80")
        content, is_error = run(registry, "read_file", path="b.bin")
        assert is_error
        assert "UTF-8" in content

    def test_read_oversized_file(self, workspace: Path) -> None:
        tiny = Settings(workspace=workspace, max_file_bytes=1024, approval_mode=ApprovalMode.AUTO)
        registry = build_default_registry(tiny, ModeApprover(ApprovalMode.AUTO))
        (workspace / "big.txt").write_text("x" * 2000)
        content, is_error = run(registry, "read_file", path="big.txt")
        assert is_error
        assert "too large" in content

    def test_path_escape_is_blocked_for_every_file_tool(self, registry: ToolRegistry) -> None:
        for name, args in [
            ("read_file", {"path": "../x"}),
            ("list_dir", {"path": "/"}),
            ("write_file", {"path": "../x", "content": "c"}),
            ("edit_file", {"path": "../x", "old_text": "a", "new_text": "b"}),
        ]:
            content, is_error = run(registry, name, **args)
            assert is_error, name
            assert "escapes" in content, name

    def test_edit_replaces_unique_match(self, registry: ToolRegistry, workspace: Path) -> None:
        (workspace / "c.py").write_text("x = 1\ny = 2\n")
        _, is_error = run(registry, "edit_file", path="c.py", old_text="y = 2", new_text="y = 3")
        assert not is_error
        assert (workspace / "c.py").read_text() == "x = 1\ny = 3\n"

    def test_edit_rejects_missing_and_ambiguous_matches(
        self, registry: ToolRegistry, workspace: Path
    ) -> None:
        (workspace / "c.py").write_text("a\na\n")
        content, is_error = run(registry, "edit_file", path="c.py", old_text="zzz", new_text="b")
        assert is_error
        assert "not found" in content
        content, is_error = run(registry, "edit_file", path="c.py", old_text="a", new_text="b")
        assert is_error
        assert "2 places" in content
        assert (workspace / "c.py").read_text() == "a\na\n"

    def test_cannot_write_into_git_dir(self, registry: ToolRegistry, workspace: Path) -> None:
        (workspace / ".git").mkdir()
        content, is_error = run(registry, "write_file", path=".git/hooks/pre-commit", content="x")
        assert is_error
        assert "protected" in content

    def test_list_dir_marks_directories_and_skips_noise(
        self, registry: ToolRegistry, workspace: Path
    ) -> None:
        (workspace / "src").mkdir()
        (workspace / "src" / "m.py").write_text("")
        (workspace / ".git").mkdir()
        (workspace / "README.md").write_text("")
        content, _ = run(registry, "list_dir")
        assert "src/" in content
        assert "README.md" in content
        assert ".git" not in content

    def test_list_dir_recursive(self, registry: ToolRegistry, workspace: Path) -> None:
        (workspace / "src").mkdir()
        (workspace / "src" / "m.py").write_text("")
        content, _ = run(registry, "list_dir", recursive=True)
        assert "src/m.py" in content

    def test_list_empty_dir(self, registry: ToolRegistry) -> None:
        content, _ = run(registry, "list_dir")
        assert content == "(empty directory)"


class TestRunCommand:
    def test_runs_allowed_command_in_workspace(
        self, registry: ToolRegistry, workspace: Path
    ) -> None:
        (workspace / "hello.txt").write_text("hi there\n")
        content, is_error = run(registry, "run_command", command="cat hello.txt")
        assert not is_error
        assert "exit code: 0" in content
        assert "hi there" in content

    def test_reports_nonzero_exit_and_stderr(self, registry: ToolRegistry) -> None:
        content, is_error = run(registry, "run_command", command="ls does-not-exist")
        assert not is_error  # a failing command is information, not a tool error
        assert "exit code:" in content
        assert "exit code: 0" not in content
        assert "stderr" in content

    def test_blocks_disallowed_command(self, registry: ToolRegistry) -> None:
        content, is_error = run(registry, "run_command", command="rm -rf .")
        assert is_error
        assert "not allowed" in content

    def test_blocks_paths_outside_workspace(self, registry: ToolRegistry) -> None:
        for command in ["cat /etc/passwd", "cat ../secret", "ls ~", "grep -r x --include=../y"]:
            content, is_error = run(registry, "run_command", command=command)
            assert is_error, command
            assert "outside the workspace" in content, command

    def test_inline_python_with_shell_characters_is_refused(self, registry: ToolRegistry) -> None:
        content, is_error = run(
            registry, "run_command", command="python -c 'import os;os.system(1)'"
        )
        assert is_error
        assert "shell operators" in content

    def test_environment_is_scrubbed(
        self, registry: ToolRegistry, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SUPER_SECRET_TOKEN", "hunter2")
        (workspace / "env.py").write_text("import os\nprint(sorted(os.environ))\n")
        content, is_error = run(registry, "run_command", command="python env.py")
        assert not is_error
        assert "exit code: 0" in content  # the script really ran, so the next check is meaningful
        assert "SUPER_SECRET_TOKEN" not in content
        assert "hunter2" not in content

    def test_timeout(self, workspace: Path) -> None:
        quick = Settings(
            workspace=workspace, command_timeout_s=0.5, approval_mode=ApprovalMode.AUTO
        )
        registry = build_default_registry(quick, ModeApprover(ApprovalMode.AUTO))
        (workspace / "slow.py").write_text("import time\ntime.sleep(5)\n")
        content, is_error = run(registry, "run_command", command="python slow.py")
        assert is_error
        assert "timed out" in content
