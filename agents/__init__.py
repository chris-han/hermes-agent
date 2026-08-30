"""Optional Semantier core integration seams."""

from importlib import import_module
from pathlib import Path
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


# When Hermes is checked out as Semantier's submodule, keep parent-owned
# runtime modules importable beneath the shared ``agents`` namespace without
# vendoring them into Hermes or eagerly importing runtime state.
_semantier_agents_path = Path(__file__).resolve().parents[2] / "src" / "agents"
if _semantier_agents_path.is_dir():
    __path__.append(str(_semantier_agents_path))


def __getattr__(name: str) -> Any:
    """Lazily expose the parent runtime's historical ``agents`` API."""
    if not _semantier_agents_path.is_dir():
        raise AttributeError(f"module 'agents' has no attribute {name!r}")
    if name in {"app", "execute"}:
        return getattr(import_module("agents.gateway"), name)
    if name == "SemantierAgent":
        return getattr(import_module("agents.semantier_agent"), name)
    if name in {"run_agent_cli", "run_api", "run_gateway_cli", "run_runtime_cli"}:
        return getattr(import_module("agents.launcher"), name)
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
