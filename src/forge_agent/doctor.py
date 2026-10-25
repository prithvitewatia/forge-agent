"""Environment diagnostics: verify the Ollama server, the model, and tool-calling support."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx
import ollama

from forge_agent.config import Settings


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: CheckStatus
    detail: str


def run_checks(settings: Settings, client: Any | None = None) -> list[Check]:
    """Run all diagnostics. Never raises; problems are reported as Check entries."""
    checks: list[Check] = [_check_server_version(settings)]
    api = client or ollama.Client(host=settings.ollama_host, timeout=10.0)

    try:
        installed = {m.model for m in api.list().models if m.model}
    except Exception as exc:
        checks.append(
            Check(
                "server",
                CheckStatus.FAIL,
                f"Cannot reach Ollama at {settings.ollama_host} ({type(exc).__name__}). "
                "Start it with `ollama serve` or open the Ollama app.",
            )
        )
        return checks
    checks.append(Check("server", CheckStatus.OK, f"reachable at {settings.ollama_host}"))

    wanted = settings.model
    names = installed | {n.removesuffix(":latest") for n in installed}
    if wanted not in names:
        checks.append(
            Check(
                "model", CheckStatus.FAIL, f"'{wanted}' is not pulled. Run `ollama pull {wanted}`."
            )
        )
        return checks
    checks.append(Check("model", CheckStatus.OK, f"'{wanted}' is available"))

    checks.extend(_check_capabilities(api, wanted, settings))
    return checks


def _check_server_version(settings: Settings) -> Check:
    try:
        response = httpx.get(f"{settings.ollama_host.rstrip('/')}/api/version", timeout=5.0)
        response.raise_for_status()
        return Check("version", CheckStatus.OK, f"server version {response.json().get('version')}")
    except (httpx.HTTPError, ValueError):
        return Check("version", CheckStatus.WARN, "could not read server version")


def _check_capabilities(api: Any, model: str, settings: Settings) -> list[Check]:
    try:
        info = api.show(model)
    except Exception as exc:
        return [Check("capabilities", CheckStatus.WARN, f"could not inspect model: {exc}")]

    capabilities = list(getattr(info, "capabilities", None) or [])
    if not capabilities:
        return [
            Check(
                "capabilities",
                CheckStatus.WARN,
                "server did not report capabilities (older Ollama?); tool support is unverified",
            )
        ]

    results: list[Check] = []
    if "tools" in capabilities:
        results.append(Check("tool calling", CheckStatus.OK, "model supports tools"))
    else:
        results.append(
            Check("tool calling", CheckStatus.FAIL, f"'{model}' does not report tool support")
        )
    if settings.think and "thinking" not in capabilities:
        results.append(
            Check("thinking", CheckStatus.FAIL, f"think is enabled but '{model}' lacks 'thinking'")
        )
    return results
