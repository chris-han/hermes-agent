"""Thin compatibility wrapper around the canonical Semantier auth DB layer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def _load_upstream_module() -> ModuleType:
    current_path = Path(__file__).resolve()
    module_candidates = [
        current_path.parents[2] / "src" / "agents" / "auth_db.py",
        current_path.parents[1] / "src" / "agents" / "auth_db.py",
    ]
    module_path = next((candidate for candidate in module_candidates if candidate.exists()), module_candidates[0])
    qualified_name = "_semantier_upstream_agents_auth_db"
    module = sys.modules.get(qualified_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Missing upstream auth_db implementation at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
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
