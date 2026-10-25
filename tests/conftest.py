from __future__ import annotations

from pathlib import Path

import pytest

from forge_agent.config import ApprovalMode, Settings
from forge_agent.safety import ModeApprover
from forge_agent.tools import ToolRegistry, build_default_registry


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return Settings(workspace=workspace, approval_mode=ApprovalMode.AUTO, max_steps=6)


@pytest.fixture
def registry(settings: Settings) -> ToolRegistry:
    return build_default_registry(settings, ModeApprover(ApprovalMode.AUTO))
