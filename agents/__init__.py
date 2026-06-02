"""Compatibility shims for optional Semantier-owned runtime modules.

This checkout does not vendor the full Semantier `agents.*` package tree, but
some gateway paths and tests patch import targets beneath `agents.*`.
Minimal stubs live here so those imports can be patched when the Semantier
runtime extensions are absent.
"""

import importlib.util
import os
import sys
from pathlib import Path


_repo_agents_path = Path(__file__).resolve().parents[2] / "src" / "agents"
if _repo_agents_path.exists():
    __path__.append(str(_repo_agents_path))


def _load_repo_agents_module(module_name: str) -> None:
    module_path = _repo_agents_path / f"{module_name}.py"
    if not module_path.exists():
        return
    qualified_name = f"{__name__}.{module_name}"
    force_repo_module = os.getenv("SEMANTIER_USE_REPO_AGENTS") == "1"
    if qualified_name in sys.modules and not force_repo_module:
        return
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    setattr(sys.modules[__name__], module_name, module)

_repo_preload_modules: list[str] = []
if os.getenv("SEMANTIER_USE_REPO_AGENTS") == "1":
    _repo_preload_modules = [
        "auth_db",
        "gateway_identity",
        "auth_session",
        "workspace_session_logs",
        "webapi_gateway",
        "gateway",
    ]

for _repo_module in _repo_preload_modules:
    _load_repo_agents_module(_repo_module)
