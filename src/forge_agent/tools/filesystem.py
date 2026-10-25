"""File tools: list, read, write, and exact-match edit. All paths are confined to the workspace."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from forge_agent.errors import ToolError
from forge_agent.safety import Risk, Workspace
from forge_agent.tools.base import Tool
from forge_agent.tools.validate import check_syntax, double_escape_message, looks_double_escaped

_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",  # the other common virtualenv name; listing it floods the model's context
        "node_modules",
        "__pycache__",
        ".forge",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
    }
)
_MAX_LIST_ENTRIES = 300


class _ListArgs(BaseModel):
    path: str = Field(default=".", description="Directory to list, relative to the workspace.")
    recursive: bool = Field(default=False, description="Also list nested directories.")


class ListDir(Tool[_ListArgs]):
    name = "list_dir"
    description = "List files and directories. Directories end with '/'. Skips .git and venvs."
    risk = Risk.READ
    args_model: ClassVar[type[BaseModel]] = _ListArgs

    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace

    def describe(self, args: _ListArgs) -> str:
        return f"list {args.path}"

    def run(self, args: _ListArgs) -> str:
        root = self._ws.resolve(args.path)
        if not root.is_dir():
            msg = f"not a directory: {args.path}"
            raise ToolError(msg)

        entries: list[str] = []
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
            base = self._ws.display(Path(current))
            prefix = "" if base == "." else f"{base}/"
            entries.extend(f"{prefix}{d}/" for d in dirs)
            entries.extend(f"{prefix}{f}" for f in sorted(files))
            if not args.recursive or len(entries) >= _MAX_LIST_ENTRIES:
                break

        if not entries:
            return "(empty directory)"
        shown = entries[:_MAX_LIST_ENTRIES]
        suffix = f"\n... and {len(entries) - len(shown)} more" if len(entries) > len(shown) else ""
        return "\n".join(shown) + suffix


class _ReadArgs(BaseModel):
    path: str = Field(description="File to read, relative to the workspace.")
    start_line: int = Field(default=1, ge=1, description="First line to return (1-based).")
    max_lines: int = Field(default=400, ge=1, le=2000, description="Maximum lines to return.")


class ReadFile(Tool[_ReadArgs]):
    name = "read_file"
    description = "Read a UTF-8 text file. Lines are prefixed with their line number."
    risk = Risk.READ
    args_model: ClassVar[type[BaseModel]] = _ReadArgs

    def __init__(self, workspace: Workspace, max_bytes: int) -> None:
        self._ws = workspace
        self._max_bytes = max_bytes

    def describe(self, args: _ReadArgs) -> str:
        return f"read {args.path}"

    def run(self, args: _ReadArgs) -> str:
        path = self._ws.resolve(args.path)
        if not path.is_file():
            msg = f"file not found: {args.path}"
            raise ToolError(msg)
        if path.stat().st_size > self._max_bytes:
            msg = f"file too large ({path.stat().st_size} bytes, limit {self._max_bytes})"
            raise ToolError(msg)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            msg = f"not a UTF-8 text file: {args.path}"
            raise ToolError(msg) from exc

        lines = text.splitlines()
        window = lines[args.start_line - 1 : args.start_line - 1 + args.max_lines]
        if not window:
            return f"(no lines in range; file has {len(lines)} lines)"
        numbered = "\n".join(f"{i}\t{line}" for i, line in enumerate(window, start=args.start_line))
        end = args.start_line + len(window) - 1
        note = (
            f"\n[showing lines {args.start_line}-{end} of {len(lines)}]" if end < len(lines) else ""
        )
        return numbered + note


class _WriteArgs(BaseModel):
    path: str = Field(description="File to create or overwrite, relative to the workspace.")
    content: str = Field(description="Complete new contents of the file.")


class WriteFile(Tool[_WriteArgs]):
    name = "write_file"
    description = (
        "Create a file or completely overwrite it. Prefer edit_file for changing existing files."
    )
    risk = Risk.WRITE
    args_model: ClassVar[type[BaseModel]] = _WriteArgs

    def __init__(self, workspace: Workspace, max_bytes: int) -> None:
        self._ws = workspace
        self._max_bytes = max_bytes

    def precheck(self, args: _WriteArgs) -> None:
        if looks_double_escaped(args.content):
            raise ToolError(double_escape_message())

    def describe(self, args: _WriteArgs) -> str:
        verb = "overwrite" if self._ws.resolve(args.path, for_write=True).exists() else "create"
        return f"{verb} {args.path} ({len(args.content.encode())} bytes)"

    def run(self, args: _WriteArgs) -> str:
        data = args.content.encode("utf-8")
        if len(data) > self._max_bytes:
            msg = f"content too large ({len(data)} bytes, limit {self._max_bytes})"
            raise ToolError(msg)
        path = self._ws.resolve(args.path, for_write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        # Reporting the line count gives the model a cheap signal when a multi-line file arrived
        # mangled onto a single line.
        lines = len(args.content.splitlines())
        noun = "line" if lines == 1 else "lines"
        summary = f"wrote {len(data)} bytes ({lines} {noun}) to {self._ws.display(path)}"
        return _with_syntax_warning(summary, path, args.content)


class _EditArgs(BaseModel):
    path: str = Field(description="Existing file to modify, relative to the workspace.")
    old_text: str = Field(min_length=1, description="Exact text to replace. Must match once.")
    new_text: str = Field(description="Replacement text (may be empty to delete).")


class EditFile(Tool[_EditArgs]):
    name = "edit_file"
    description = (
        "Replace one exact occurrence of old_text with new_text in an existing file. "
        "Fails if old_text is missing or appears more than once; include more context to be unique."
    )
    risk = Risk.WRITE
    args_model: ClassVar[type[BaseModel]] = _EditArgs

    def __init__(self, workspace: Workspace, max_bytes: int) -> None:
        self._ws = workspace
        self._max_bytes = max_bytes

    def describe(self, args: _EditArgs) -> str:
        return f"edit {args.path} (replace {len(args.old_text)} chars with {len(args.new_text)})"

    def run(self, args: _EditArgs) -> str:
        path = self._ws.resolve(args.path, for_write=True)
        if not path.is_file():
            msg = f"file not found: {args.path}"
            raise ToolError(msg)
        text = path.read_text(encoding="utf-8")
        count = text.count(args.old_text)
        if count == 0:
            msg = "old_text was not found. Re-read the file and copy the text exactly."
            raise ToolError(msg)
        if count > 1:
            msg = f"old_text matches {count} places. Include more surrounding lines to be unique."
            raise ToolError(msg)
        updated = text.replace(args.old_text, args.new_text, 1)
        if len(updated.encode("utf-8")) > self._max_bytes:
            msg = "resulting file would exceed the size limit"
            raise ToolError(msg)
        path.write_text(updated, encoding="utf-8")
        return _with_syntax_warning(f"edited {self._ws.display(path)}", path, updated)


def _with_syntax_warning(message: str, path: Path, text: str) -> str:
    """Append an advisory when the file we just wrote does not parse, so the model can fix it."""
    problem = check_syntax(path, text)
    if problem is None:
        return message
    return f"{message}\nWARNING: the file was written but has a problem: {problem}. Fix it."
