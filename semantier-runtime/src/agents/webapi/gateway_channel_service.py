from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any
import uuid

import httpx
from fastapi import File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from agents.auth_db import (
    auth_db_path,
    is_postgresql,
    get_feishu_bot_config_for_user,
    get_user_settings_for_user,
    get_weixin_runtime_account,
    upsert_user_settings_for_user,
    save_weixin_runtime_account,
    update_weixin_runtime_account_state,
)
from agents.route_policy import (
    RouteAuthorizationClass,
    classify_authorization_class,
    forwarded_upstream_allowed,
)
from agents import company_dataset_imports
from agents.workspace_uploads import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    WorkspaceUploadError,
    save_workspace_upload,
)
from agents.workspace_session_logs import (
    get_workspace_session_log_payload,
    list_workspace_sessions,
)
from agents.auth_session import (
    RequestContext,
    VT_BROWSER_SESSION_COOKIE,
    VT_SESSION_COOKIE,
    auth_context_payload,
    browser_session_nonce,
    browser_session_nonce_hash,
    create_vt_session_token,
    decode_vt_session_token,
    request_context_from_request,
    request_user_agent_hash,
    session_grace_window_seconds,
    session_cookie_max_age,
    session_issue_time,
)
import importlib
import importlib.util
import sys
import yaml

from storage.event_store import UnifiedEventStore

def _compat_attr(name: str):
    compat_module = sys.modules.get("agents.webapi_gateway")
    if compat_module is not None and name in vars(compat_module):
        return getattr(compat_module, name)
    return globals()[name]

_LOGGER = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_HERMES_HOME = _REPO_ROOT / ".semantier-home"
_REQUEST_HEADER_EXCLUDE = frozenset({"host", "content-length", "connection"})
_RESPONSE_HEADER_EXCLUDE = frozenset({"content-length", "connection", "transfer-encoding"})
_RUNTIME_EMBED_GATEWAY_ENV = "SEMANTIER_RUNTIME_EMBED_GATEWAY"
_SEMANTIER_SANDBOX_KEY_HEADER = "X-Semantier-Sandbox-Key"
_SEMANTIER_USER_ID_HEADER = "X-Semantier-User-Id"
_SEMANTIER_WORKSPACE_ID_HEADER = "X-Semantier-Workspace-Id"
_SEMANTIER_UPLOAD_SESSION_ID_HEADER = "X-Semantier-Upload-Session-Id"


def _boundary_header_payload(
    *,
    workspace_id: str | None,
    user_id: str | None,
    session_id: str | None,
    hermes_home: Path | None,
) -> str:
    payload = {
        "workspace_id": workspace_id,
        "user_id": user_id,
        "session_id": session_id,
        "hermes_home": str(hermes_home.resolve()) if hermes_home is not None else None,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")

SUPPORTED_GATEWAYS = ("weixin", "feishu", "web_api")

GATEWAY_PLATFORM_MAP = {
    "weixin": "weixin",
    "feishu": "feishu",
    "web_api": "api_server",
}
MESSAGING_DOCS_URL = {
    "feishu": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/feishu",
    "weixin": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/weixin",
}
WEIXIN_RECONNECT_CONTRACT_VERSION = "weixin-reconnect.v1"
WEIXIN_LOGIN_STATE_VERSION = "weixin-login.v2"

_RATE_LIMIT_STATE_LOCK = threading.Lock()
_RATE_LIMIT_STATE: dict[str, tuple[float, int]] = {}
# (max_requests, window_seconds)
_RATE_LIMITS: dict[str, tuple[int, int]] = {
    "auth_public": (60, 60),
    "weixin_reconnect_user": (12, 60),
    "session_create_user": (30, 60),
    "session_create_tenant": (120, 60),
    "chat_user": (120, 60),
    "chat_tenant": (600, 60),
}



def _load_gateway_identity() -> object:
    required = [
        "attach_user_to_seeded_demo_organization",
        "authenticate_password_user",
        "count_active_organization_owners",
        "complete_user_profile",
        "consume_weixin_login_state",
        "create_feishu_link_state",
        "create_weixin_login_state",
        "delete_feishu_link_state",
        "delete_workspace_feishu_bot_config",
        "get_active_weixin_correlation_for_user",
        "get_organization_materialization_policy",
        "get_pending_role_notifications_for_user",
        "get_user_membership_for_organization",
        "get_feishu_link_state",
        "get_user_organization_context",
        "get_weixin_login_state",
        "join_organization",
        "link_feishu_open_id",
        "list_organization_events",
        "list_organization_members",
        "list_seeded_demo_organization_profiles",
        "read_workspace_feishu_bot_config",
        "resolve_seeded_demo_organization_for_industry",
        "save_workspace_feishu_bot_config",
        "set_user_sharing_enabled",
        "switch_user_organization",
        "unlink_feishu_open_id",
        "update_membership_role",
        "update_membership_status",
        "update_organization_materialization_policy",
        "update_weixin_login_state",
        "update_feishu_link_state",
        "upsert_weixin_owner",
    ]

    try:
        # Try the normal import first (may be provided by an installed package).
        mod = importlib.import_module("agents.gateway_identity")
        if all(hasattr(mod, name) for name in required):
            return mod
    except Exception:
        pass

    # Fallback: explicitly load the repo `src/agents/gateway_identity.py`.
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "src" / "agents" / "gateway_identity.py"
    spec = importlib.util.spec_from_file_location("agents.gateway_identity", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Semantier gateway_identity from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agents.gateway_identity"] = module
    spec.loader.exec_module(module)
    return module


_gateway_identity = _load_gateway_identity()

def _gateway_identity_module() -> object:
    # Keep the identity authority selected by _load_gateway_identity.
    # Later package attachment must not replace the qualified owner.
    return _gateway_identity


def __getattr_from_gateway_identity(name: str):
    return getattr(_gateway_identity_module(), name)


def attach_user_to_seeded_demo_organization(*args, **kwargs):
    return __getattr_from_gateway_identity("attach_user_to_seeded_demo_organization")(*args, **kwargs)


def authenticate_password_user(*args, **kwargs):
    return __getattr_from_gateway_identity("authenticate_password_user")(*args, **kwargs)


def count_active_organization_owners(*args, **kwargs):
    return __getattr_from_gateway_identity("count_active_organization_owners")(*args, **kwargs)


def complete_user_profile(*args, **kwargs):
    return __getattr_from_gateway_identity("complete_user_profile")(*args, **kwargs)


def consume_weixin_login_state(*args, **kwargs):
    return __getattr_from_gateway_identity("consume_weixin_login_state")(*args, **kwargs)


def create_feishu_link_state(*args, **kwargs):
    return __getattr_from_gateway_identity("create_feishu_link_state")(*args, **kwargs)


def create_weixin_login_state(*args, **kwargs):
    return __getattr_from_gateway_identity("create_weixin_login_state")(*args, **kwargs)


def delete_feishu_link_state(*args, **kwargs):
    return __getattr_from_gateway_identity("delete_feishu_link_state")(*args, **kwargs)


def delete_workspace_feishu_bot_config(*args, **kwargs):
    return __getattr_from_gateway_identity("delete_workspace_feishu_bot_config")(*args, **kwargs)


def get_active_weixin_correlation_for_user(*args, **kwargs):
    return __getattr_from_gateway_identity("get_active_weixin_correlation_for_user")(*args, **kwargs)


def get_organization_materialization_policy(*args, **kwargs):
    return __getattr_from_gateway_identity("get_organization_materialization_policy")(*args, **kwargs)


def get_pending_role_notifications_for_user(*args, **kwargs):
    return __getattr_from_gateway_identity("get_pending_role_notifications_for_user")(*args, **kwargs)


def get_user_membership_for_organization(*args, **kwargs):
    return __getattr_from_gateway_identity("get_user_membership_for_organization")(*args, **kwargs)


def get_feishu_link_state(*args, **kwargs):
    return __getattr_from_gateway_identity("get_feishu_link_state")(*args, **kwargs)


def get_user_organization_context(*args, **kwargs):
    return __getattr_from_gateway_identity("get_user_organization_context")(*args, **kwargs)


def get_weixin_login_state(*args, **kwargs):
    return __getattr_from_gateway_identity("get_weixin_login_state")(*args, **kwargs)


def get_user_record(*args, **kwargs):
    return __getattr_from_gateway_identity("get_user_record")(*args, **kwargs)


def get_user_record_by_workspace_id(*args, **kwargs):
    return __getattr_from_gateway_identity("get_user_record_by_workspace_id")(*args, **kwargs)


def join_organization(*args, **kwargs):
    return __getattr_from_gateway_identity("join_organization")(*args, **kwargs)


def link_feishu_open_id(*args, **kwargs):
    return __getattr_from_gateway_identity("link_feishu_open_id")(*args, **kwargs)


def list_organization_events(*args, **kwargs):
    return __getattr_from_gateway_identity("list_organization_events")(*args, **kwargs)


def list_organization_members(*args, **kwargs):
    return __getattr_from_gateway_identity("list_organization_members")(*args, **kwargs)


def list_seeded_demo_organization_profiles(*args, **kwargs):
    return __getattr_from_gateway_identity("list_seeded_demo_organization_profiles")(*args, **kwargs)


def read_workspace_feishu_bot_config(*args, **kwargs):
    return __getattr_from_gateway_identity("read_workspace_feishu_bot_config")(*args, **kwargs)


def resolve_seeded_demo_organization_for_industry(*args, **kwargs):
    return __getattr_from_gateway_identity("resolve_seeded_demo_organization_for_industry")(*args, **kwargs)


def save_workspace_feishu_bot_config(*args, **kwargs):
    return __getattr_from_gateway_identity("save_workspace_feishu_bot_config")(*args, **kwargs)


def set_user_sharing_enabled(*args, **kwargs):
    return __getattr_from_gateway_identity("set_user_sharing_enabled")(*args, **kwargs)


def switch_user_organization(*args, **kwargs):
    return __getattr_from_gateway_identity("switch_user_organization")(*args, **kwargs)


def unlink_feishu_open_id(*args, **kwargs):
    return __getattr_from_gateway_identity("unlink_feishu_open_id")(*args, **kwargs)


def update_membership_role(*args, **kwargs):
    return __getattr_from_gateway_identity("update_membership_role")(*args, **kwargs)


def update_membership_status(*args, **kwargs):
    return __getattr_from_gateway_identity("update_membership_status")(*args, **kwargs)


def update_organization_materialization_policy(*args, **kwargs):
    return __getattr_from_gateway_identity("update_organization_materialization_policy")(*args, **kwargs)


def update_weixin_login_state(*args, **kwargs):
    return __getattr_from_gateway_identity("update_weixin_login_state")(*args, **kwargs)


def update_feishu_link_state(*args, **kwargs):
    return __getattr_from_gateway_identity("update_feishu_link_state")(*args, **kwargs)


def upsert_weixin_owner(*args, **kwargs):
    return __getattr_from_gateway_identity("upsert_weixin_owner")(*args, **kwargs)

def _ensure_repo_hermes_agent_on_path() -> None:
    hermes_agent_root = _REPO_ROOT / "hermes-agent"
    if not hermes_agent_root.exists():
        return
    normalized_root = str(hermes_agent_root.resolve())
    sys_path = []
    for existing in os.sys.path:
        try:
            if Path(existing).resolve() == hermes_agent_root.resolve():
                continue
        except OSError:
            pass
        sys_path.append(existing)
    os.sys.path[:] = [normalized_root, *sys_path]


def _workspace_hermes_home_for_user(user: dict[str, Any]) -> str:
    workspace_id = str(user["workspace_id"])
    ensure_workspace_paths = __getattr_from_gateway_identity("ensure_workspace_paths")
    _, hermes_home = ensure_workspace_paths(workspace_id)
    return str(hermes_home)


def _find_pending_pairing_by_code(store: Any, platform: str, code: str):
    """Inspect adopted Hermes salted codes before governed owner authorization.

    Approval remains with the native store. Never approve a code merely to learn
    its owner: a foreign owner's code must leave the grant and pending row intact.
    """
    if is_postgresql():
        from agents.auth_pairing_postgres import PostgresPairingStore
        if not isinstance(store,PostgresPairingStore):
            raise RuntimeError("PostgreSQL pairing requires the governed native owner")
        return store.peek_code(platform,code)
    import secrets
    from gateway.pairing import _is_hashed_entry

    with store._lock:
        store._cleanup_expired(platform)
        if store._is_locked_out(platform):
            return None
        normalized = "".join(str(code or "").upper().split())
        for entry in store._load_json(store._pending_path(platform)).values():
            if not _is_hashed_entry(entry):
                continue
            try:
                salt = bytes.fromhex(entry["salt"])
            except ValueError:
                continue
            if secrets.compare_digest(store._hash_code(normalized, salt), entry["hash"]):
                return {"user_id": entry["user_id"], "user_name": entry.get("user_name", "")}
        # This pre-approval lookup must retain native brute-force accounting;
        # invalid probes never reach approve_code after the route returns404.
        store._record_failed_attempt(platform)
    return None


def _iter_gateway_pairing_stores(
    *,
    hermes_home: str | None = None,
    ctx: RequestContext | None = None,
):
    seen: set[str] = set()

    def _remember(store: Any):
        if store is None:
            return None
        store_dir = getattr(store, "_dir", None) or getattr(store, "_pairing_dir", None)
        key = str(Path(store_dir).resolve()) if store_dir is not None else f"object:{id(store)}"
        if key in seen:
            return None
        seen.add(key)
        return store

    try:
        from agents.hermes_embedded_gateway import get_embedded_hermes_gateway_service

        service = get_embedded_hermes_gateway_service()
        runner = getattr(service, "_runner", None)
        live_store = getattr(runner, "pairing_store", None) if runner is not None else None
        if is_postgresql() and live_store is not None:
            from agents.auth_pairing_postgres import PostgresPairingStore
            if not isinstance(live_store,PostgresPairingStore):
                raise RuntimeError("PostgreSQL pairing requires the governed native owner")
        remembered = _remember(live_store)
        if remembered is not None:
            yield remembered
    except Exception:
        pass

    _ensure_repo_hermes_agent_on_path()
    import gateway.pairing as pairing_mod

    PairingStore = pairing_mod.PairingStore

    candidate_dirs: list[Path] = []
    if hermes_home:
        home = Path(str(hermes_home)).expanduser()
        candidate_dirs.extend([home / "platforms" / "pairing", home / "pairing"])
    elif ctx is not None:
        home = Path(str(ctx.hermes_home)).expanduser()
        candidate_dirs.extend([home / "platforms" / "pairing", home / "pairing"])
    try:
        from runtime_paths import platform_runtime_root

        runtime_root = platform_runtime_root()
        candidate_dirs.extend([runtime_root / "platforms" / "pairing", runtime_root / "pairing"])
    except Exception:
        pass
    if is_postgresql():
        from agents.auth_pairing_postgres import PostgresPairingStore
        for pairing_dir in candidate_dirs:
            remembered = _remember(PostgresPairingStore(pairing_dir=pairing_dir))
            if remembered is not None:
                yield remembered
        return
    pairing_auth_db_path = auth_db_path()
    for pairing_dir in candidate_dirs:
        try:
            candidate_store = PairingStore(
                pairing_dir=pairing_dir,
                auth_db_path=pairing_auth_db_path,
            )
        except TypeError:
            original_pairing_dir = getattr(pairing_mod, "PAIRING_DIR", None)
            original_auth_db_path = os.environ.get("SEMANTIER_AUTH_DB_PATH")
            try:
                pairing_mod.PAIRING_DIR = pairing_dir
                os.environ["SEMANTIER_AUTH_DB_PATH"] = str(pairing_auth_db_path)
                candidate_store = PairingStore()
            finally:
                pairing_mod.PAIRING_DIR = original_pairing_dir
                if original_auth_db_path is None:
                    os.environ.pop("SEMANTIER_AUTH_DB_PATH", None)
                else:
                    os.environ["SEMANTIER_AUTH_DB_PATH"] = original_auth_db_path
        remembered = _remember(candidate_store)
        if remembered is not None:
            yield remembered


def _iter_weixin_pairing_stores(ctx: RequestContext):
    yield from _iter_gateway_pairing_stores(ctx=ctx)


def _load_workspace_pairing_store(hermes_home: str):
    _ensure_repo_hermes_agent_on_path()
    import gateway.pairing as pairing_mod

    pairing_dir = Path(hermes_home).expanduser() / "platforms" / "pairing"
    if is_postgresql():
        from agents.auth_pairing_postgres import PostgresPairingStore
        return PostgresPairingStore(pairing_dir=pairing_dir)
    PairingStore = pairing_mod.PairingStore
    pairing_auth_db_path = auth_db_path()
    try:
        return PairingStore(
            pairing_dir=pairing_dir,
            auth_db_path=pairing_auth_db_path,
        )
    except TypeError:
        original_pairing_dir = getattr(pairing_mod, "PAIRING_DIR", None)
        original_auth_db_path = os.environ.get("SEMANTIER_AUTH_DB_PATH")
        try:
            pairing_mod.PAIRING_DIR = pairing_dir
            os.environ["SEMANTIER_AUTH_DB_PATH"] = str(pairing_auth_db_path)
            return PairingStore()
        finally:
            pairing_mod.PAIRING_DIR = original_pairing_dir
            if original_auth_db_path is None:
                os.environ.pop("SEMANTIER_AUTH_DB_PATH", None)
            else:
                os.environ["SEMANTIER_AUTH_DB_PATH"] = original_auth_db_path


def _ensure_gateway_pairing_approved(
    *,
    hermes_home: str,
    platform: str,
    external_user_id: str,
    user_name: str = "",
) -> None:
    normalized_user_id = str(external_user_id or "").strip()
    normalized_platform = str(platform or "").strip().lower()
    if not normalized_platform:
        return
    if not normalized_user_id:
        return

    for store in _iter_gateway_pairing_stores(hermes_home=hermes_home):
        if store.is_approved(normalized_platform, normalized_user_id):
            continue

        pending_matches = [
            item
            for item in store.list_pending(normalized_platform)
            if str(item.get("user_id") or "").strip() == normalized_user_id
        ]
        approved = False
        for item in pending_matches:
            request_id = str(item.get("request_id") or "").strip()
            if not request_id:
                continue
            approve_request = getattr(store, "approve_request", None)
            result = (
                approve_request(normalized_platform, request_id)
                if callable(approve_request)
                else None
            )
            if result:
                approved = True
                break

        if not approved:
            lock = getattr(store, "_lock", None)
            approve_user = getattr(store, "_approve_user", None)
            if lock is not None and callable(approve_user):
                with lock:
                    approve_user(
                        normalized_platform,
                        normalized_user_id,
                        user_name or normalized_user_id,
                    )


def _ensure_weixin_workspace_pairing_approved(
    *,
    hermes_home: str,
    weixin_user_id: str,
    user_name: str = "",
) -> None:
    _ensure_gateway_pairing_approved(
        hermes_home=hermes_home,
        platform="weixin",
        external_user_id=weixin_user_id,
        user_name=user_name,
    )


def _ensure_feishu_workspace_pairing_approved(
    *,
    hermes_home: str,
    user_id: str,
    feishu_open_id: str,
    user_name: str = "",
) -> None:
    normalized_user_id = str(user_id or "").strip()
    normalized_open_id = str(feishu_open_id or "").strip()
    if not normalized_user_id or not normalized_open_id:
        return
    config = get_feishu_bot_config_for_user(normalized_user_id)
    if not isinstance(config, dict):
        return
    app_id = str(config.get("app_id") or "").strip()
    app_secret = str(config.get("app_secret") or "").strip()
    if not app_id or not app_secret:
        return
    _ensure_gateway_pairing_approved(
        hermes_home=hermes_home,
        platform="feishu",
        external_user_id=normalized_open_id,
        user_name=user_name or normalized_open_id,
    )


def _revoke_feishu_workspace_pairing(
    *,
    hermes_home: str,
    feishu_open_id: str,
) -> None:
    normalized_open_id = str(feishu_open_id or "").strip()
    if not normalized_open_id:
        return
    for store in _iter_gateway_pairing_stores(hermes_home=hermes_home):
        revoke = getattr(store, "revoke", None)
        if callable(revoke):
            revoke("feishu", normalized_open_id)


def _issue_auth_response(
    user: dict[str, Any],
    *,
    auth_provider: str,
    request: Request | None = None,
    state: str | None = None,
) -> JSONResponse:
    profile_completed = bool(user.get("profile_completed_at"))
    browser_nonce = (
        request.cookies.get(VT_BROWSER_SESSION_COOKIE)
        if request is not None
        else None
    )
    if not browser_nonce:
        browser_nonce = browser_session_nonce()
    token_value = create_vt_session_token(
        user_id=str(user["user_id"]),
        workspace_id=str(user["workspace_id"]),
        workspace_slug=str(user["workspace_slug"]),
        issued_at=session_issue_time(),
        browser_session_nonce_hash=browser_session_nonce_hash(browser_nonce),
        user_agent_hash=request_user_agent_hash(request),
    )
    payload: dict[str, Any] = {
        "status": "confirmed",
        "authenticated": True,
        "profile_completed": profile_completed,
        "workspace_slug": user["workspace_slug"],
        "user": {
            "user_id": user["user_id"],
            "name": user.get("name") or user["workspace_slug"],
            "workspace_slug": user["workspace_slug"],
            "password_login_name": user.get("password_login_name"),
            "weixin_user_id": user.get("weixin_user_id"),
            "auth_provider": auth_provider,
            "profile_completed": profile_completed,
        },
    }
    if state is not None:
        payload["state"] = state
    response = JSONResponse(payload)
    response.set_cookie(
        VT_BROWSER_SESSION_COOKIE,
        browser_nonce,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=session_cookie_max_age(),
    )
    response.set_cookie(
        VT_SESSION_COOKIE,
        token_value,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=session_cookie_max_age(),
    )
    return response


def _weixin_auth_response(
    user: dict[str, Any],
    *,
    request: Request | None = None,
    state: str | None = None,
) -> JSONResponse:
    return _issue_auth_response(
        user,
        auth_provider="weixin",
        request=request,
        state=state,
    )


def _read_workspace_weixin_account(ctx: RequestContext) -> dict[str, Any] | None:
    if ctx.authenticated:
        try:
            correlation = get_active_weixin_correlation_for_user(ctx.user_id)
        except Exception:
            correlation = None
        if isinstance(correlation, dict):
            account_id = str(correlation.get("linked_account_id") or "").strip()
            if account_id:
                runtime_account = get_weixin_runtime_account(account_id)
                if isinstance(runtime_account, dict):
                    normalized = dict(runtime_account)
                    normalized["account_id"] = account_id
                    return normalized

    return None


def _normalize_weixin_home_channel(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _default_weixin_home_channel(*, home_channel: Any, fallback_user_id: Any) -> str:
    normalized = _normalize_weixin_home_channel(home_channel)
    if normalized:
        return normalized
    return _normalize_weixin_home_channel(fallback_user_id)


def _sync_weixin_home_channel_env(home_channel: str) -> None:
    normalized = _normalize_weixin_home_channel(home_channel)
    if normalized:
        os.environ["WEIXIN_HOME_CHANNEL"] = normalized
    else:
        os.environ.pop("WEIXIN_HOME_CHANNEL", None)


def _runtime_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _remove_workspace_weixin_account_files(hermes_home: str) -> None:
    """Remove legacy Weixin JSON credentials and sidecars from a workspace."""
    accounts_dir = Path(hermes_home) / "weixin" / "accounts"
    if not accounts_dir.exists():
        return

    for path in sorted(accounts_dir.glob("*.json")):
        try:
            path.unlink()
        except OSError:
            pass
    try:
        accounts_dir.rmdir()
        accounts_dir.parent.rmdir()
    except OSError:
        pass


@contextmanager
def _temporary_embedded_gateway_enabled():
    """Ensure embedded gateway mode is enabled for the current operation."""
    previous = os.environ.get(_RUNTIME_EMBED_GATEWAY_ENV)
    os.environ[_RUNTIME_EMBED_GATEWAY_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_RUNTIME_EMBED_GATEWAY_ENV, None)
        else:
            os.environ[_RUNTIME_EMBED_GATEWAY_ENV] = previous


def _gateway_started(status: Any) -> bool:
    """Interpret embedded gateway startup result across mixed return shapes."""
    if status is None:
        return True
    running = getattr(status, "running", None)
    if isinstance(running, bool):
        return running
    return bool(status)


def _sync_shared_runtime_weixin_env(weixin_account: dict[str, Any]) -> None:
    """Write active workspace Weixin credentials to shared runtime ``.env``.

    Embedded gateway startup refreshes environment values from the shared
    launcher-managed ``.semantier-home/.env``. Keep that file aligned with the
    selected workspace account so reconnect/restart does not snap back to a
    stale account.
    """

    account_id = str(weixin_account.get("account_id") or "").strip()
    token = str(weixin_account.get("token") or "").strip()
    base_url = str(weixin_account.get("base_url") or "").strip()
    if not account_id or not token:
        return

    runtime_root = Path(
        os.environ.get("SEMANTIER_LOCAL_STATE_DIR") or str(_LOCAL_HERMES_HOME)
    ).expanduser().resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    env_path = runtime_root / ".env"

    lines: list[str] = []
    if env_path.exists():
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []

    updates = {
        "WEIXIN_ACCOUNT_ID": account_id,
        "WEIXIN_TOKEN": token,
    }
    if base_url:
        updates["WEIXIN_BASE_URL"] = base_url

    seen: set[str] = set()
    out_lines: list[str] = []
    for line in lines:
        replaced = False
        for key, value in updates.items():
            prefix = f"{key}="
            if line.startswith(prefix):
                out_lines.append(f"{key}={value}")
                seen.add(key)
                replaced = True
                break
        if not replaced:
            out_lines.append(line)

    for key, value in updates.items():
        if key not in seen:
            out_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")


def _ensure_shared_runtime_auth_db_env() -> None:
    """Bind ``SEMANTIER_AUTH_DB_PATH`` to the shared runtime root when absent."""
    if is_postgresql():
        from agents.auth_db import ensure_auth_db
        ensure_auth_db()
        return
    if str(os.environ.get("SEMANTIER_AUTH_DB_PATH") or "").strip():
        return
    os.environ["SEMANTIER_AUTH_DB_PATH"] = str(auth_db_path().resolve())


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _adapter_key(*, platform: str, workspace_id: str, account_id: str) -> str:
    return f"{platform}:{workspace_id}:{account_id}"


def _hash_token(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized_env_set(name: str) -> set[str]:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return set()
    return {
        item.strip()
        for item in raw.split(",")
        if item.strip()
    }


def _weixin_auth_policy_allows(
    *,
    account_id: str,
    external_user_id: str,
) -> bool:
    approved_accounts = _normalized_env_set("SEMANTIER_WEIXIN_AUTH_APPROVED_ACCOUNT_IDS")
    approved_users = _normalized_env_set("SEMANTIER_WEIXIN_AUTH_APPROVED_EXTERNAL_USER_IDS")
    if not approved_accounts and not approved_users:
        return True
    return (
        (account_id in approved_accounts if account_id else False)
        or (external_user_id in approved_users if external_user_id else False)
    )


def _cookie_browser_session_hash(request: Request) -> str | None:
    return browser_session_nonce_hash(request.cookies.get(VT_BROWSER_SESSION_COOKIE))


def _resolve_governed_weixin_account(
    ctx: RequestContext,
    workspace_account: dict[str, Any],
) -> dict[str, Any]:
    correlation = get_active_weixin_correlation_for_user(ctx.user_id)
    if not isinstance(correlation, dict):
        raise HTTPException(status_code=409, detail="No active governed Weixin correlation found")
    owner_workspace_id = str(correlation.get("owner_workspace_id") or "").strip()
    if owner_workspace_id != ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Governed Weixin correlation workspace mismatch")
    governed_account_id = str(correlation.get("linked_account_id") or "").strip()
    if not governed_account_id:
        raise HTTPException(status_code=409, detail="Governed Weixin correlation is missing linked account_id")

    workspace_account_id = str(workspace_account.get("account_id") or "").strip()
    if workspace_account_id and workspace_account_id != governed_account_id:
        raise HTTPException(
            status_code=409,
            detail="Workspace Weixin runtime cache does not match governed account linkage",
        )

    token = str(workspace_account.get("token") or "").strip()
    if not token:
        raise HTTPException(status_code=409, detail="Weixin runtime token is missing")

    normalized = dict(workspace_account)
    normalized["account_id"] = governed_account_id
    return normalized


def _request_context_from_session_metadata(meta) -> RequestContext:
    ensure_workspace_paths = __getattr_from_gateway_identity("ensure_workspace_paths")
    user = get_user_record(meta.user_id)
    if not isinstance(user, dict):
        raise HTTPException(status_code=401, detail="Authentication required")
    workspace_root, hermes_home = ensure_workspace_paths(meta.workspace_id)
    return RequestContext(
        authenticated=True,
        user_id=meta.user_id,
        workspace_id=meta.workspace_id,
        workspace_slug=meta.workspace_slug,
        workspace_root=workspace_root,
        hermes_home=hermes_home,
        auth_provider=str(user.get("auth_provider") or "").strip() or None,
        weixin_user_id=str(user.get("weixin_user_id") or "").strip() or None,
        profile_completed=bool(user.get("profile_completed_at")),
    )


def _workspace_id_from_hermes_home_path(hermes_home: Path | None) -> str | None:
    if hermes_home is None:
        return None
    resolved = hermes_home.expanduser().resolve()
    workspace_id = resolved.name.strip()
    if not workspace_id:
        return None
    try:
        from runtime_paths import workspace_runtime_home_path

        expected = workspace_runtime_home_path(workspace_id).resolve()
    except ValueError:
        expected = None
    if expected is not None and resolved == expected:
        return workspace_id
    local_state_dir = str(os.environ.get("SEMANTIER_LOCAL_STATE_DIR") or "").strip()
    if local_state_dir:
        local_state_path = Path(local_state_dir).expanduser().resolve()
        if resolved == local_state_path:
            return workspace_id
        legacy_workspaces_root = local_state_path.parent / "workspaces"
        if resolved.parent == legacy_workspaces_root:
            return workspace_id
    return None


async def _persist_gateway_reconnect_artifact(
    *,
    ctx: RequestContext,
    correlation_id: str,
    event_type: str,
    adapter_key: str,
    payload: dict[str, Any],
) -> None:
    started = time.monotonic()
    last_exc: Exception | None = None
    attempts = 0
    while attempts < 2 and (time.monotonic() - started) <= 3.0:
        attempts += 1
        store: UnifiedEventStore | None = None
        try:
            store = UnifiedEventStore()
            artifact_payload = dict(payload)
            artifact_payload.setdefault("adapter_key", adapter_key)
            artifact_payload.setdefault("correlation_id", correlation_id)
            artifact_payload.setdefault("timestamp_utc", _utc_now())
            artifact_payload.setdefault("decision_contract_version", WEIXIN_RECONNECT_CONTRACT_VERSION)
            store.append_event(
                event_id=f"{event_type}:{uuid.uuid4().hex}",
                event_type=event_type,
                payload=artifact_payload,
                correlation_id=correlation_id,
                org_id=ctx.organization_id or None,
                stream_id=f"gateway_reconnect:{ctx.workspace_id}:{adapter_key}",
            )
            return
        except Exception as exc:
            last_exc = exc
            if attempts >= 2 or (time.monotonic() - started) > 3.0:
                break
            await asyncio.sleep(0.1)
        finally:
            if store is not None:
                try:
                    store.conn.close()
                except Exception:
                    pass
    raise RuntimeError(f"failed to persist reconnect artifact: {last_exc}") from last_exc


async def _persist_weixin_auth_artifact(
    *,
    event_type: str,
    correlation_id: str,
    payload: dict[str, Any],
) -> None:
    started = time.monotonic()
    last_exc: Exception | None = None
    attempts = 0
    while attempts < 2 and (time.monotonic() - started) <= 3.0:
        attempts += 1
        store: UnifiedEventStore | None = None
        try:
            store = UnifiedEventStore()
            artifact_payload = dict(payload)
            artifact_payload.setdefault("timestamp_utc", _utc_now())
            artifact_payload.setdefault("state_contract_version", WEIXIN_LOGIN_STATE_VERSION)
            store.append_event(
                event_id=f"{event_type}:{uuid.uuid4().hex}",
                event_type=event_type,
                payload=artifact_payload,
                correlation_id=correlation_id,
                org_id=None,
                stream_id=f"weixin_auth:{correlation_id}",
            )
            return
        except Exception as exc:
            last_exc = exc
            if attempts >= 2 or (time.monotonic() - started) > 3.0:
                break
            await asyncio.sleep(0.1)
        finally:
            if store is not None:
                try:
                    store.conn.close()
                except Exception:
                    pass
    raise RuntimeError(f"failed to persist Weixin auth artifact: {last_exc}") from last_exc


def _read_feishu_bot_config(ctx: RequestContext) -> dict[str, Any] | None:
    user = auth_context_payload(ctx).get("user") or {}
    workspace_id = str(user.get("workspace_id") or "").strip()
    if not workspace_id:
        return None
    payload = read_workspace_feishu_bot_config(workspace_id)
    return payload if isinstance(payload, dict) else None


def _feishu_chat_type_from_session_key(session_key: Any) -> str:
    parts = str(session_key or "").split(":")
    if len(parts) >= 7 and parts[0] == "agent" and parts[2] == "workspace":
        offset = 4
    elif len(parts) >= 5 and parts[0] == "agent":
        offset = 2
    else:
        return ""
    if len(parts) > offset + 2 and parts[offset] == "feishu":
        value = str(parts[offset + 1] or "").strip().lower()
        return value if value in {"dm", "group", "forum", "thread"} else ""
    return ""


def _platform_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


def _timestamp_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return 0.0
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _iter_feishu_runner_session_entries() -> list[Any]:
    try:
        from agents.hermes_embedded_gateway import get_embedded_hermes_gateway_service

        service = get_embedded_hermes_gateway_service()
        runtime = getattr(service, "_shared_runtime", None)
        runner = getattr(runtime, "runner", None) if runtime is not None else None
        session_store = getattr(runner, "session_store", None) if runner is not None else None
        if session_store is None:
            return []
        ensure_loaded = getattr(session_store, "_ensure_loaded", None)
        if callable(ensure_loaded):
            ensure_loaded()
        entries = getattr(session_store, "_entries", {})
        if isinstance(entries, dict):
            return list(entries.values())
    except Exception:
        return []
    return []


def _list_feishu_live_channels(ctx: RequestContext) -> list[dict[str, Any]]:
    channels: dict[tuple[str, str], dict[str, Any]] = {}

    def remember(candidate: dict[str, Any]) -> None:
        chat_id = str(candidate.get("chat_id") or "").strip()
        if not chat_id:
            return
        thread_id = str(candidate.get("thread_id") or "").strip()
        key = (chat_id, thread_id)
        candidate["chat_id"] = chat_id
        candidate["thread_id"] = thread_id
        candidate["last_active"] = _timestamp_value(candidate.get("last_active"))
        current = channels.get(key)
        if current is None or float(candidate.get("last_active") or 0) > float(
            current.get("last_active") or 0
        ):
            channels[key] = candidate

    try:
        rows = list_workspace_sessions(ctx.hermes_home)
    except Exception:
        rows = []
    for row in rows:
        if _platform_value(row.get("platform")) != "feishu":
            continue
        workspace_owner_id = str(row.get("workspace_owner_id") or "").strip()
        if workspace_owner_id and workspace_owner_id != ctx.workspace_id:
            continue
        chat_id = str(row.get("chat_id") or "").strip()
        if not chat_id:
            continue
        chat_type = _feishu_chat_type_from_session_key(row.get("session_key"))
        if not chat_type:
            try:
                payload = get_workspace_session_log_payload(
                    ctx.hermes_home,
                    str(row.get("session_id") or ""),
                )
            except Exception:
                payload = None
            if isinstance(payload, dict):
                chat_type = _feishu_chat_type_from_session_key(
                    payload.get("session_key")
                )
        remember(
            {
                "chat_id": chat_id,
                "thread_id": str(row.get("thread_id") or "").strip(),
                "chat_type": chat_type,
                "name": str(row.get("display_name") or row.get("title") or chat_id),
                "last_active": row.get("last_active"),
                "session_id": str(row.get("session_id") or ""),
            }
        )

    for entry in _iter_feishu_runner_session_entries():
        source = getattr(entry, "origin", None)
        if (
            source is None
            or _platform_value(getattr(source, "platform", None)) != "feishu"
        ):
            continue
        workspace_owner_id = str(
            getattr(source, "workspace_owner_id", "") or ""
        ).strip()
        if workspace_owner_id and workspace_owner_id != ctx.workspace_id:
            continue
        chat_id = str(getattr(source, "chat_id", "") or "").strip()
        if not chat_id:
            continue
        remember(
            {
                "chat_id": chat_id,
                "thread_id": str(getattr(source, "thread_id", "") or "").strip(),
                "chat_type": str(getattr(source, "chat_type", "") or "").strip(),
                "name": str(
                    getattr(source, "chat_name", None)
                    or getattr(entry, "display_name", None)
                    or chat_id
                ),
                "last_active": getattr(entry, "updated_at", None),
                "session_id": str(getattr(entry, "session_id", "") or ""),
            }
        )

    return sorted(
        channels.values(),
        key=lambda item: float(item.get("last_active") or 0),
        reverse=True,
    )


def _messaging_platforms_payload(ctx: RequestContext) -> dict[str, Any]:
    user = auth_context_payload(ctx).get("user") or {}
    weixin_account = _read_workspace_weixin_account(ctx)
    feishu_account = _read_feishu_bot_config(ctx)
    weixin_config: dict[str, Any] = {}
    weixin_configured = False
    weixin_updated_at: str | None = None
    if isinstance(weixin_account, dict):
        weixin_configured = True
        weixin_updated_at = (
            str(weixin_account.get("saved_at") or "").strip() or None
        )
        runtime_session_state = str(weixin_account.get("runtime_session_state") or "").strip()
        runtime_session_error = str(weixin_account.get("runtime_session_error") or "").strip()
        runtime_session_updated_at = str(weixin_account.get("runtime_session_updated_at") or "").strip()
        weixin_config = {
            "account_id": str(weixin_account.get("account_id") or ""),
            "token": str(weixin_account.get("token") or ""),
            "base_url": str(weixin_account.get("base_url") or "https://ilinkai.weixin.qq.com"),
            "runtime_session_state": runtime_session_state,
            "runtime_session_error": runtime_session_error,
            "runtime_session_updated_at": runtime_session_updated_at,
            "dm_policy": "pairing",
            "allow_from": [],
            "group_policy": "disabled",
            "group_allow_from": [],
            "home_channel": _normalize_weixin_home_channel(weixin_account.get("home_channel")),
        }

    feishu_configured = bool(feishu_account) or bool(str(user.get("feishu_open_id") or "").strip())
    feishu_config: dict[str, Any] = {}
    feishu_updated_at: str | None = None
    if isinstance(feishu_account, dict):
        feishu_updated_at = (
            str(
                feishu_account.get("saved_at")
                or feishu_account.get("updated_at")
                or ""
            ).strip()
            or None
        )
        feishu_config = {
            "app_id": str(feishu_account.get("app_id") or ""),
            "app_secret": str(feishu_account.get("app_secret") or ""),
            "connection_mode": str(feishu_account.get("connection_mode") or "websocket"),
            "domain": str(feishu_account.get("domain") or "feishu"),
            "bot_open_id": str(feishu_account.get("bot_open_id") or ""),
            "bot_name": str(feishu_account.get("bot_name") or ""),
            "home_channel": str(feishu_account.get("home_channel") or ""),
            "live_channels": _list_feishu_live_channels(ctx),
            "scope": "organization",
            "manager_user_id": str(feishu_account.get("manager_user_id") or ""),
            "feishu_open_id": str(user.get("feishu_open_id") or "").strip(),
        }
    elif feishu_configured:
        feishu_config = {
            "feishu_open_id": str(user.get("feishu_open_id") or "").strip(),
            "connection_mode": "websocket",
            "domain": "feishu",
            "home_channel": "",
            "live_channels": _list_feishu_live_channels(ctx),
            "scope": "organization",
        }

    return {
        "owner_id": ctx.user_id,
        "workspace_slug": ctx.workspace_slug,
        "platforms": [
            {
                "platform": "feishu",
                "configured": feishu_configured,
                "docs_url": MESSAGING_DOCS_URL["feishu"],
                "updated_at": feishu_updated_at,
                "validated_at": (
                    str(feishu_account.get("validated_at") or "").strip()
                    if isinstance(feishu_account, dict)
                    else None
                ),
                "last_error": (
                    str(feishu_account.get("last_error") or "").strip()
                    if isinstance(feishu_account, dict)
                    else None
                ),
                "config": feishu_config,
            },
            {
                "platform": "weixin",
                "configured": weixin_configured,
                "docs_url": MESSAGING_DOCS_URL["weixin"],
                "updated_at": weixin_updated_at,
                "validated_at": weixin_updated_at,
                "last_error": (
                    str(weixin_account.get("runtime_session_error") or "").strip()
                    if isinstance(weixin_account, dict)
                    else None
                ),
                "config": weixin_config,
            },
        ],
    }


async def _restart_embedded_gateway_for_weixin_account(
    weixin_account: dict[str, Any],
) -> dict[str, Any]:
    account_id = str(weixin_account.get("account_id") or "").strip()
    token = str(weixin_account.get("token") or "").strip()
    if not token:
        return {
            "ok": False,
            "error": "weixin account token is missing",
        }

    base_url = str(weixin_account.get("base_url") or "").strip()
    cdn_base_url = str(weixin_account.get("cdn_base_url") or "").strip()

    os.environ["WEIXIN_TOKEN"] = token
    if account_id:
        os.environ["WEIXIN_ACCOUNT_ID"] = account_id
    if base_url:
        os.environ["WEIXIN_BASE_URL"] = base_url
    if cdn_base_url:
        os.environ["WEIXIN_CDN_BASE_URL"] = cdn_base_url

    try:
        _sync_shared_runtime_weixin_env(weixin_account)
    except Exception:
        # Best effort sync: runtime restart should still proceed.
        pass

    try:
        from agents.hermes_embedded_gateway import get_embedded_hermes_gateway_service

        service = get_embedded_hermes_gateway_service()
        with _temporary_embedded_gateway_enabled():
            await service.shutdown()
            status = await service.ensure_started()
        if not _gateway_started(status):
            return {
                "ok": False,
                "error": "embedded gateway did not start (disabled or unavailable)",
            }
        return {
            "ok": True,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"embedded gateway restart failed: {exc}",
        }


@contextmanager
def _temporary_hermes_home(hermes_home: str):
    """Temporarily bind HERMES_HOME to a workspace for embedded gateway calls."""
    target_home = str(hermes_home or "").strip()
    if not target_home:
        yield
        return
    context_token = None
    reset_active_hermes_home = None
    try:
        from hermes_constants import reset_active_hermes_home as _reset_home, set_active_hermes_home

        context_token = set_active_hermes_home(Path(target_home))
        reset_active_hermes_home = _reset_home
    except Exception:
        context_token = None
        reset_active_hermes_home = None

    # Cross-gateway env contract (HERMES_HOME + SEMANTIER_WORKSPACE_RUNS_DIR)
    # is owned by runtime_paths.bind_workspace_env so every adapter binds
    # identically and downstream file-IO tools see a consistent context.
    from runtime_paths import bind_workspace_env

    try:
        try:
            binding = bind_workspace_env(target_home)
        except ValueError:
            binding = None
        if binding is not None:
            with binding:
                yield
        elif _compat_attr("_workspace_id_from_hermes_home_path")(Path(target_home)):
            previous = {
                "HERMES_HOME": os.environ.get("HERMES_HOME"),
                "SEMANTIER_WORKSPACE_ID": os.environ.get("SEMANTIER_WORKSPACE_ID"),
                "SEMANTIER_WRITE_SAFE_ROOT": os.environ.get("SEMANTIER_WRITE_SAFE_ROOT"),
                "SEMANTIER_TERMINAL_CWD": os.environ.get("SEMANTIER_TERMINAL_CWD"),
            }
            resolved_home = Path(target_home).expanduser().resolve()
            os.environ["HERMES_HOME"] = str(resolved_home)
            os.environ["SEMANTIER_WORKSPACE_ID"] = resolved_home.name
            os.environ["SEMANTIER_WRITE_SAFE_ROOT"] = str(resolved_home)
            os.environ["SEMANTIER_TERMINAL_CWD"] = str(resolved_home)
            try:
                yield
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        else:
            raise ValueError("not a workspace runtime home")
    finally:
        if context_token is not None and reset_active_hermes_home is not None:
            try:
                reset_active_hermes_home(context_token)
            except Exception:
                pass


@contextmanager
def _compat_hermes_home_bound(target_home: Path, *, session_id: str | None = None):
    from agents.semantier_agent import _hermes_home_bound

    try:
        with _hermes_home_bound(target_home, session_id=session_id):
            yield
    except ValueError:
        resolved_home = Path(target_home).expanduser().resolve()
        previous = {
            "HERMES_HOME": os.environ.get("HERMES_HOME"),
            "SEMANTIER_WORKSPACE_ID": os.environ.get("SEMANTIER_WORKSPACE_ID"),
            "SEMANTIER_WRITE_SAFE_ROOT": os.environ.get("SEMANTIER_WRITE_SAFE_ROOT"),
            "SEMANTIER_TERMINAL_CWD": os.environ.get("SEMANTIER_TERMINAL_CWD"),
        }
        context_token = None
        reset_active_hermes_home = None
        try:
            from hermes_constants import reset_active_hermes_home as _reset_home, set_active_hermes_home

            context_token = set_active_hermes_home(resolved_home)
            reset_active_hermes_home = _reset_home
        except Exception:
            context_token = None
            reset_active_hermes_home = None
        os.environ["HERMES_HOME"] = str(resolved_home)
        os.environ["SEMANTIER_WORKSPACE_ID"] = resolved_home.name
        os.environ["SEMANTIER_WRITE_SAFE_ROOT"] = str(resolved_home)
        os.environ["SEMANTIER_TERMINAL_CWD"] = str(resolved_home)
        try:
            yield
        finally:
            if context_token is not None and reset_active_hermes_home is not None:
                try:
                    reset_active_hermes_home(context_token)
                except Exception:
                    pass
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


async def _restart_embedded_gateway_for_feishu_account(
    feishu_account: dict[str, Any],
) -> dict[str, Any]:
    app_id = str(feishu_account.get("app_id") or "").strip()
    app_secret = str(feishu_account.get("app_secret") or "").strip()
    if not app_id or not app_secret:
        return {
            "ok": False,
            "error": "feishu app_id and app_secret are required",
        }

    domain = str(feishu_account.get("domain") or "feishu").strip()
    connection_mode = str(feishu_account.get("connection_mode") or "websocket").strip()

    os.environ["FEISHU_APP_ID"] = app_id
    os.environ["FEISHU_APP_SECRET"] = app_secret
    if domain:
        os.environ["FEISHU_DOMAIN"] = domain
    if connection_mode:
        os.environ["FEISHU_CONNECTION_MODE"] = connection_mode
    owner_user_id = str(
        feishu_account.get("owner_user_id")
        or feishu_account.get("manager_user_id")
        or ""
    ).strip()
    if owner_user_id:
        try:
            get_user_record = __getattr_from_gateway_identity("get_user_record")
            user = get_user_record(owner_user_id)
        except Exception:
            user = None
        if isinstance(user, dict):
            feishu_open_id = str(user.get("feishu_open_id") or "").strip()
            if feishu_open_id:
                existing_allowed = {
                    item.strip()
                    for item in os.environ.get("FEISHU_ALLOWED_USERS", "").split(",")
                    if item.strip()
                }
                existing_allowed.add(feishu_open_id)
                os.environ["FEISHU_ALLOWED_USERS"] = ",".join(sorted(existing_allowed))

    try:
        from agents.hermes_embedded_gateway import get_embedded_hermes_gateway_service

        service = get_embedded_hermes_gateway_service()
        await service.shutdown()
        await service.ensure_started()
        return {
            "ok": True,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"embedded gateway restart failed: {exc}",
        }
