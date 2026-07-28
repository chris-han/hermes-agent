"""Semantier-bound governed-context bridge.

This Hermes Agent checkout is embedded in the Semantier runtime repository.
Governed context authority belongs to the parent Semantier implementation at
``src/agents/governed_context.py``. Missing parent runtime code is a packaging
error, not a standalone fallback mode.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


def _load_semantier_governed_context() -> object:
    repo_root = Path(__file__).resolve().parents[2]
    repo_agents_root = repo_root / "src" / "agents"
    repo_governed_context = repo_agents_root / "governed_context.py"
    if not repo_governed_context.exists():
        raise RuntimeError(
            "Semantier governed context implementation is required at "
            f"{repo_governed_context}"
        )
    agents_pkg = sys.modules.get("agents")
    package_path = getattr(agents_pkg, "__path__", None)
    if package_path is not None and str(repo_agents_root) not in package_path:
        package_path.append(str(repo_agents_root))
    spec = importlib.util.spec_from_file_location(
        "_semantier_repo_governed_context",
        repo_governed_context,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load Semantier governed context implementation from "
            f"{repo_governed_context}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SEMANTIER_GOVERNED_CONTEXT = _load_semantier_governed_context()

build_governed_runtime_context_prompt = getattr(
    _SEMANTIER_GOVERNED_CONTEXT,
    "build_governed_runtime_context_prompt",
)
is_analytics_query_message = getattr(
    _SEMANTIER_GOVERNED_CONTEXT,
    "is_analytics_query_message",
)
resolve_governed_activation_context = getattr(
    _SEMANTIER_GOVERNED_CONTEXT,
    "resolve_governed_activation_context",
)
resolve_user_id_for_workspace = getattr(
    _SEMANTIER_GOVERNED_CONTEXT,
    "resolve_user_id_for_workspace",
)

__all__ = [
    "build_governed_runtime_context_prompt",
    "is_analytics_query_message",
    "resolve_governed_activation_context",
    "resolve_user_id_for_workspace",
]
