"""Workspace confinement: every file the agent touches must resolve inside the workspace root."""

from __future__ import annotations

from pathlib import Path

from forge_agent.errors import SafetyError

# Directories the agent may never write into, even inside the workspace.
_PROTECTED_PARTS = frozenset({".git", ".forge", ".venv"})


class Workspace:
    """A directory jail. Resolves symlinks before checking containment."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            msg = f"workspace is not a directory: {self.root}"
            raise SafetyError(msg)

    def resolve(self, relative: str, *, for_write: bool = False) -> Path:
        """Return an absolute path inside the workspace or raise SafetyError.

        Args:
            relative: A path as supplied by the model (relative, or absolute inside the root).
            for_write: Also reject protected directories such as .git.
        """
        if not relative or "\x00" in relative:
            msg = "empty or invalid path"
            raise SafetyError(msg)

        candidate = Path(relative).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # resolve() follows symlinks, so a link pointing outside the root is caught below.
        resolved = candidate.resolve()

        if not resolved.is_relative_to(self.root):
            msg = f"path escapes the workspace: {relative}"
            raise SafetyError(msg)

        if for_write and _PROTECTED_PARTS.intersection(resolved.relative_to(self.root).parts):
            msg = f"writing to protected location is not allowed: {relative}"
            raise SafetyError(msg)
        return resolved

    def display(self, path: Path) -> str:
        """Workspace-relative form of a path, for showing to the model and the user."""
        try:
            return path.relative_to(self.root).as_posix() or "."
        except ValueError:  # pragma: no cover - callers pass resolved paths
            return str(path)
