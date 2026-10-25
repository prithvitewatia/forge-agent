"""Forge: a local-first, safety-conscious coding agent built on Ollama."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("forge-agent")
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
