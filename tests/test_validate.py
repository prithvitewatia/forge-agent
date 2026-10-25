"""Guards against the ways a small model mangles files: double-escaped newlines, invalid syntax."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_agent.config import Settings
from forge_agent.messages import ToolCall
from forge_agent.tools import ToolRegistry, build_default_registry
from forge_agent.tools.validate import check_syntax, looks_double_escaped
from tests.helpers import RecordingApprover

# The exact payload Qwen3 8B produced when asked for a pyproject.toml: every newline arrived as the
# two characters backslash and n, so the file landed on disk as a single line.
INCIDENT_CONTENT = (
    '[tool.poetry]\\nname = "todo-app"\\nversion = "0.1.0"\\ndescription = ""\\n'
    'authors = ["Your Name <you@example.com>"]\\n\\n[tool.poetry.dependencies]\\n'
    'django = "^4.2"\\n'
)


def run(registry: ToolRegistry, name: str, **args: object) -> tuple[str, bool]:
    result = registry.execute(ToolCall(name, dict(args)))
    return result.content, result.is_error


class TestLooksDoubleEscaped:
    def test_recognises_the_real_incident(self) -> None:
        assert looks_double_escaped(INCIDENT_CONTENT)

    def test_ordinary_multiline_text_is_fine(self) -> None:
        assert not looks_double_escaped("a = 1\nb = 2\n")

    def test_real_newlines_win_even_if_escapes_are_also_present(self) -> None:
        assert not looks_double_escaped('print("a\\nb\\nc")\n')

    def test_a_single_escape_is_not_enough(self) -> None:
        assert not looks_double_escaped('print("a\\nb")')

    def test_empty_text_is_fine(self) -> None:
        assert not looks_double_escaped("")


class TestCheckSyntax:
    @pytest.mark.parametrize(
        ("name", "text"),
        [
            ("pyproject.toml", '[project]\nname = "x"\n'),
            ("data.json", '{"a": 1}'),
            ("mod.py", "x = 1\n"),
            ("notes.txt", "anything [ at all"),  # unknown types are never reported
            ("README.md", "# {{ broken"),
        ],
    )
    def test_valid_or_unknown_files_report_nothing(self, name: str, text: str) -> None:
        assert check_syntax(Path(name), text) is None

    @pytest.mark.parametrize(
        ("name", "text", "expected"),
        [
            ("pyproject.toml", "[project\nname = 1", "invalid TOML"),
            ("data.json", "{not json}", "invalid JSON"),
            ("mod.py", "def f(:\n    pass\n", "Python syntax error"),
        ],
    )
    def test_invalid_files_are_described(self, name: str, text: str, expected: str) -> None:
        problem = check_syntax(Path(name), text)
        assert problem is not None
        assert expected in problem

    def test_the_incident_content_is_invalid_toml(self) -> None:
        assert check_syntax(Path("pyproject.toml"), INCIDENT_CONTENT) is not None

    def test_suffix_match_is_case_insensitive(self) -> None:
        assert check_syntax(Path("DATA.JSON"), "{bad") is not None


class TestWriteFileGuards:
    def test_double_escaped_write_is_refused_and_nothing_is_written(
        self, workspace: Path, settings: Settings
    ) -> None:
        registry = build_default_registry(settings, RecordingApprover(answer=True))
        content, is_error = run(
            registry, "write_file", path="pyproject.toml", content=INCIDENT_CONTENT
        )
        assert is_error
        assert "double-escaped" in content
        assert not (workspace / "pyproject.toml").exists()

    def test_human_is_not_asked_to_approve_a_write_that_will_be_refused(
        self, settings: Settings
    ) -> None:
        approver = RecordingApprover(answer=True)
        registry = build_default_registry(settings, approver)
        run(registry, "write_file", path="pyproject.toml", content=INCIDENT_CONTENT)
        assert approver.requests == []

    def test_escape_hatch_for_a_genuine_single_line_file(
        self, workspace: Path, registry: ToolRegistry
    ) -> None:
        text = 'print("a\\nb\\nc")\n'  # ends with one real newline, so it is not flagged
        _, is_error = run(registry, "write_file", path="one.py", content=text)
        assert not is_error
        assert (workspace / "one.py").read_text() == text

    def test_refusal_message_explains_the_escape_hatch(self, registry: ToolRegistry) -> None:
        content, _ = run(registry, "write_file", path="a.txt", content="x\\ny\\nz")
        assert "real line break" in content

    def test_result_reports_line_count(self, registry: ToolRegistry) -> None:
        content, is_error = run(registry, "write_file", path="a.txt", content="one\ntwo\nthree\n")
        assert not is_error
        assert "(3 lines)" in content

    def test_result_uses_singular_for_one_line(self, registry: ToolRegistry) -> None:
        content, _ = run(registry, "write_file", path="a.txt", content="just one")
        assert "(1 line)" in content

    def test_invalid_toml_is_written_but_flagged(
        self, workspace: Path, registry: ToolRegistry
    ) -> None:
        content, is_error = run(
            registry, "write_file", path="pyproject.toml", content="[project\nname = 1\n"
        )
        assert not is_error  # advisory only: intermediate states may be legitimately invalid
        assert "WARNING" in content
        assert "invalid TOML" in content
        assert (workspace / "pyproject.toml").exists()

    def test_valid_file_has_no_warning(self, registry: ToolRegistry) -> None:
        content, _ = run(registry, "write_file", path="ok.py", content="x = 1\n")
        assert "WARNING" not in content


class TestEditFileGuards:
    def test_edit_that_breaks_python_is_flagged(
        self, workspace: Path, registry: ToolRegistry
    ) -> None:
        (workspace / "m.py").write_text("def f():\n    return 1\n")
        content, is_error = run(
            registry, "edit_file", path="m.py", old_text="def f():", new_text="def f(:"
        )
        assert not is_error
        assert "WARNING" in content
        assert "Python syntax error" in content

    def test_clean_edit_has_no_warning(self, workspace: Path, registry: ToolRegistry) -> None:
        (workspace / "m.py").write_text("x = 1\n")
        content, _ = run(registry, "edit_file", path="m.py", old_text="x = 1", new_text="x = 2")
        assert "WARNING" not in content

    def test_edits_are_not_subject_to_the_double_escape_refusal(
        self, workspace: Path, registry: ToolRegistry
    ) -> None:
        # A replacement fragment legitimately may contain escaped newlines inside a string.
        (workspace / "m.py").write_text("msg = ''\n")
        _, is_error = run(registry, "edit_file", path="m.py", old_text="''", new_text="'a\\nb\\nc'")
        assert not is_error


class TestVirtualenvHandling:
    def test_venv_is_skipped_when_listing(self, workspace: Path, registry: ToolRegistry) -> None:
        (workspace / "venv" / "lib").mkdir(parents=True)
        (workspace / "venv" / "lib" / "huge.py").write_text("")
        (workspace / "app.py").write_text("")
        content, _ = run(registry, "list_dir", recursive=True)
        assert "app.py" in content
        assert "venv" not in content
        assert "huge.py" not in content

    def test_venv_cannot_be_written_to(self, workspace: Path, registry: ToolRegistry) -> None:
        (workspace / "venv").mkdir()
        content, is_error = run(registry, "write_file", path="venv/x.py", content="x = 1\n")
        assert is_error
        assert "protected" in content
        assert not (workspace / "venv" / "x.py").exists()

    def test_venv_can_still_be_read(self, workspace: Path, registry: ToolRegistry) -> None:
        (workspace / "venv").mkdir()
        (workspace / "venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
        _, is_error = run(registry, "read_file", path="venv/pyvenv.cfg")
        assert not is_error
