from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SemantierAgent",
    "app",
    "execute",
    "run_agent_cli",
    "run_api",
    "run_gateway_cli",
    "run_runtime_cli",
]


def __getattr__(name: str) -> Any:
    if name in {"app", "execute"}:
        module = import_module("apps.combined")
        return getattr(module, name)
    if name == "SemantierAgent":
        module = import_module("agents.semantier_agent")
        return getattr(module, name)
    if name in {"run_agent_cli", "run_api", "run_gateway_cli", "run_runtime_cli"}:
        module = import_module("agents.launcher")
        return getattr(module, name)
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
