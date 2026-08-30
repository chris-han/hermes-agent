"""Bridge Hermes gateway context assembly to Semantier-owned governance."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def _runtime_root() -> Path:
    configured = os.environ.get("SEMANTIER_RUNTIME_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "src" / "agents" / "governed_context.py").is_file():
            return root
    raise RuntimeError(
        "Semantier governed context is required; set SEMANTIER_RUNTIME_ROOT to the runtime checkout"
    )


def _load_semantier_governed_context() -> object:
    repo_agents_root = _runtime_root() / "src" / "agents"
    implementation = repo_agents_root / "governed_context.py"
    agents_pkg = sys.modules.get("agents")
    package_path = getattr(agents_pkg, "__path__", None)
    if package_path is not None and str(repo_agents_root) not in package_path:
        package_path.append(str(repo_agents_root))
    spec = importlib.util.spec_from_file_location(
        "_semantier_repo_governed_context",
        implementation,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Semantier governed context from {implementation}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_IMPLEMENTATION = _load_semantier_governed_context()

build_governed_runtime_context_prompt = getattr(
    _IMPLEMENTATION, "build_governed_runtime_context_prompt"
)
is_analytics_query_message = getattr(_IMPLEMENTATION, "is_analytics_query_message")
resolve_governed_activation_context = getattr(
    _IMPLEMENTATION, "resolve_governed_activation_context"
)
resolve_user_id_for_workspace = getattr(_IMPLEMENTATION, "resolve_user_id_for_workspace")

__all__ = [
    "build_governed_runtime_context_prompt",
    "is_analytics_query_message",
    "resolve_governed_activation_context",
    "resolve_user_id_for_workspace",
]
