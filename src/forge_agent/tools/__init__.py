from __future__ import annotations

from forge_agent.config import Settings
from forge_agent.safety import Approver, CommandPolicy, Workspace
from forge_agent.tools.base import Tool, ToolRegistry, ToolResult
from forge_agent.tools.filesystem import EditFile, ListDir, ReadFile, WriteFile
from forge_agent.tools.shell import RunCommand


def build_default_registry(settings: Settings, approver: Approver) -> ToolRegistry:
    """Assemble the standard toolset, wired to the workspace jail and command policy."""
    workspace = Workspace(settings.workspace)
    registry = ToolRegistry(approver=approver, max_output_chars=settings.max_tool_output_chars)
    registry.register(ListDir(workspace))
    registry.register(ReadFile(workspace, settings.max_file_bytes))
    registry.register(WriteFile(workspace, settings.max_file_bytes))
    registry.register(EditFile(workspace, settings.max_file_bytes))
    registry.register(RunCommand(workspace, CommandPolicy(settings), settings))
    return registry


__all__ = ["Tool", "ToolRegistry", "ToolResult", "build_default_registry"]
