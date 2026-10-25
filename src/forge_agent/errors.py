"""Exception hierarchy. Everything the library raises on purpose derives from ForgeError."""

from __future__ import annotations


class ForgeError(Exception):
    """Base class for all errors raised deliberately by forge_agent."""


class ConfigError(ForgeError):
    """Invalid or inconsistent configuration."""


class LLMError(ForgeError):
    """The model backend failed (unreachable, missing model, bad response)."""


class SafetyError(ForgeError):
    """An action was blocked by the safety layer (path escape, disallowed command)."""


class ToolError(ForgeError):
    """A tool failed in an expected, reportable way. The message is shown to the model."""
