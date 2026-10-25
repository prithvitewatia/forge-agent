"""Cheap, dependency-free checks that catch the most common ways a small model mangles a file."""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

# Literal backslash followed by "n": the two characters, not a line break.
_ESCAPED_NEWLINE = "\\n"


def looks_double_escaped(text: str) -> bool:
    """True when text has no real line break but several literal backslash-n sequences.

    Small models sometimes double-escape newlines in JSON tool arguments, so a multi-line file
    arrives as one long line. Two or more escapes and zero real line breaks is a strong signal;
    a single-line snippet with one escape (for example print("a\\nb")) is left alone.
    """
    return "\n" not in text and text.count(_ESCAPED_NEWLINE) >= 2


def double_escape_message() -> str:
    return (
        "The content looks double-escaped: it contains literal backslash-n sequences but no real "
        "line breaks, so the file would be a single line. Nothing was written. Resend the content "
        "with actual line breaks between lines. If the file really is a single line that contains "
        "backslash-n text, end the content with one real line break and resend."
    )


def check_syntax(path: Path, text: str) -> str | None:
    """Return a one-line problem description if `text` is invalid for its file type, else None.

    Unknown file types are never reported. A problem is advisory: intermediate edits may
    legitimately be invalid, so callers surface it as a warning rather than an error.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".toml":
            tomllib.loads(text)
        elif suffix == ".json":
            json.loads(text)
        elif suffix == ".py":
            ast.parse(text, filename=path.name)
    except tomllib.TOMLDecodeError as exc:
        return f"invalid TOML: {exc}"
    except json.JSONDecodeError as exc:
        return f"invalid JSON: {exc}"
    except SyntaxError as exc:
        return f"Python syntax error: {exc.msg} (line {exc.lineno})"
    return None
