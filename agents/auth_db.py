"""Thin compatibility wrapper around the canonical Semantier auth DB layer."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def _canonical_module_path() -> Path:
    configured = os.environ.get("SEMANTIER_RUNTIME_ROOT", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser() / "src/agents/auth_db.py")
    current = Path(__file__).resolve()
    candidates.extend(parent / "src/agents/auth_db.py" for parent in current.parents)
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() != current:
            return candidate
    raise ModuleNotFoundError(
        "Canonical Semantier auth_db implementation is unavailable; set "
        "SEMANTIER_RUNTIME_ROOT to the semantier-runtime checkout"
    )


def _load_upstream_module() -> ModuleType:
    module_path = _canonical_module_path()
    qualified_name = "_semantier_canonical_agents_auth_db"
    cached = sys.modules.get(qualified_name)
    if cached is not None and Path(cached.__file__).resolve() == module_path.resolve():
        return cached
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Cannot load canonical auth_db at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    source_root = str(module_path.parents[1])
    inserted = source_root not in sys.path
    if inserted:
        sys.path.insert(0, source_root)
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(qualified_name, None)
        raise
    finally:
        if inserted:
            sys.path.remove(source_root)
    return module


_UPSTREAM = _load_upstream_module()

auth_db_path = _UPSTREAM.auth_db_path
load_users = _UPSTREAM.load_users
save_users = _UPSTREAM.save_users
load_organizations = _UPSTREAM.load_organizations
save_organizations = _UPSTREAM.save_organizations
load_organization_events = _UPSTREAM.load_organization_events
save_organization_events = _UPSTREAM.save_organization_events
load_gateway_correlations = _UPSTREAM.load_gateway_correlations
save_gateway_correlations = _UPSTREAM.save_gateway_correlations
load_weixin_runtime_accounts = _UPSTREAM.load_weixin_runtime_accounts
get_weixin_runtime_account = _UPSTREAM.get_weixin_runtime_account
save_weixin_runtime_account = _UPSTREAM.save_weixin_runtime_account
update_weixin_runtime_account_state = _UPSTREAM.update_weixin_runtime_account_state
load_weixin_login_states = _UPSTREAM.load_weixin_login_states
save_weixin_login_states = _UPSTREAM.save_weixin_login_states
load_feishu_link_states = _UPSTREAM.load_feishu_link_states
save_feishu_link_states = _UPSTREAM.save_feishu_link_states


def ensure_auth_db(*, json_loader: Callable[[str], Any] | None = None) -> Path:
    _UPSTREAM.ensure_auth_db(json_loader=json_loader)
    return auth_db_path()
