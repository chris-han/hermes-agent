from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Request
import importlib

VT_SESSION_COOKIE = "vt_session"
VT_BROWSER_SESSION_COOKIE = "vt_browser_session"
SESSION_SECRET_ENV = "SEMANTIER_SESSION_SECRET"
SESSION_COOKIE_MAX_AGE_ENV = "SEMANTIER_SESSION_MAX_AGE"
SESSION_GRACE_WINDOW_ENV = "SEMANTIER_WEIXIN_RECONNECT_GRACE_WINDOW_SECONDS"
DEFAULT_SESSION_MAX_AGE = 60 * 60 * 24 * 14
DEFAULT_SESSION_GRACE_WINDOW_SECONDS = 300
DEFAULT_DEV_SECRET = "semantier-dev-session-secret"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACES_ROOT = _REPO_ROOT / "workspaces"


@dataclass(frozen=True)
class AuthSessionPayload:
    user_id: str
    workspace_id: str
    workspace_slug: str


@dataclass(frozen=True)
class AuthSessionMetadata:
    user_id: str
    workspace_id: str
    workspace_slug: str
    issued_at: int | None = None
    browser_session_nonce_hash: str | None = None
    user_agent_hash: str | None = None


@dataclass(frozen=True)
class RequestContext:
    authenticated: bool
    user_id: str
    workspace_id: str
    workspace_slug: str
    workspace_root: Path
    hermes_home: Path
    organization_id: str | None = None
    organization_name: str | None = None
    dataset_type: str | None = None
    industry_code: str | None = None
    membership_status: str = "unassigned"
    member_role: str | None = None
    sharing_enabled: bool = False
    can_invite_members: bool = False
    can_change_settings: bool = False
    user_name: str | None = None
    login_name: str | None = None
    auth_provider: str | None = None
    weixin_user_id: str | None = None
    feishu_open_id: str | None = None
    profile_completed: bool = False
    auth_invalid_reason: str | None = None
    avatar_url: str | None = None


_AUTH_WORKSPACE_INITIALIZED: set[tuple[Path, Path]] = set()
_AUTH_WORKSPACE_BOOTSTRAP_LOCK = threading.RLock()


def _resolve_authenticated_workspace_paths(payload: AuthSessionPayload) -> tuple[Path, Path]:
    gateway = _gateway_identity_module()
    root, home = gateway.resolve_workspace_paths(payload.workspace_id)
    from agents.auth_db import is_postgresql
    if is_postgresql():
        # Hosted reads resolve the verified identity's governed paths only.
        # Signup/login and explicit operator setup own filesystem preparation;
        # a cold GET cannot repair configuration or enqueue account work.
        return root, home
    # Provisioning belongs to account/bootstrap/config operations. Repeating it
    # for every authenticated read rewrites plugin trees and SQLite config.
    stored = materialized = None
    if root.is_dir() and (home / "config.yaml").is_file():
        import yaml
        from agents.auth_db import get_workspace_config
        stored = get_workspace_config(payload.workspace_id)
        try:
            materialized = yaml.safe_load((home / 'config.yaml').read_text()) or {}
        except (OSError, yaml.YAMLError):
            materialized = None
        # Configuration changes (including newly enabled optional tools) still
        # materialize through the existing owner, once per changed config.
        if (root, home) in _AUTH_WORKSPACE_INITIALIZED and materialized is not None and (not stored or stored == materialized):
            return root, home
    # First authenticated use still applies the existing runtime registration
    # side effects. Serialize the infrequent bootstrap's process env mutations.
    with _AUTH_WORKSPACE_BOOTSTRAP_LOCK:
        if (root, home) not in _AUTH_WORKSPACE_INITIALIZED or not root.is_dir() or not (home/'config.yaml').is_file() or stored != materialized:
            paths = gateway.ensure_workspace_paths(payload.workspace_id)
            _AUTH_WORKSPACE_INITIALIZED.add(paths)
            return paths
        return root, home


def _gateway_identity_module():
    agents_pkg = importlib.import_module("agents")
    module = getattr(agents_pkg, "gateway_identity", None)
    if module is not None:
        return module
    return importlib.import_module("agents.gateway_identity")


def _parse_utc_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_weixin_runtime_session_expired(
    user_id: str,
    *,
    session_issued_at: int | None = None,
) -> bool:
    try:
        from agents.auth_db import get_weixin_runtime_account

        get_active_weixin_correlation_for_user = getattr(
            _gateway_identity_module(),
            "get_active_weixin_correlation_for_user",
        )
    except Exception:
        return False

    try:
        correlation = get_active_weixin_correlation_for_user(user_id)
    except Exception:
        return True
    if not isinstance(correlation, dict):
        return False

    account_id = str(correlation.get("linked_account_id") or "").strip()
    if not account_id:
        return False

    payload = get_weixin_runtime_account(account_id)
    if not isinstance(payload, dict):
        return False

    state = str(payload.get("runtime_session_state") or "").strip().lower()
    if state == "expired" and session_issued_at is not None:
        updated_at = _parse_utc_timestamp(payload.get("runtime_session_updated_at"))
        if updated_at is not None:
            issued_at = datetime.fromtimestamp(session_issued_at, tz=timezone.utc)
            if updated_at <= issued_at:
                return False
    return state == "expired"


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * ((4 - len(raw) % 4) % 4)
    return base64.urlsafe_b64decode(raw + padding)


def _session_secret() -> bytes:
    secret = os.environ.get(SESSION_SECRET_ENV, "").strip()
    if not secret:
        secret = DEFAULT_DEV_SECRET
    return secret.encode("utf-8")


def _looks_like_legacy_identity_label(
    candidate: str | None,
    *,
    user_id: str | None,
    workspace_slug: str | None,
    weixin_user_id: str | None,
) -> bool:
    value = (candidate or "").strip()
    if not value:
        return False
    comparisons = {
        (user_id or "").strip(),
        (workspace_slug or "").strip(),
        (weixin_user_id or "").strip(),
    }
    return value in comparisons


def session_cookie_max_age() -> int:
    value = os.environ.get(SESSION_COOKIE_MAX_AGE_ENV, "").strip()
    if not value:
        return DEFAULT_SESSION_MAX_AGE
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_SESSION_MAX_AGE
    return parsed if parsed > 0 else DEFAULT_SESSION_MAX_AGE


def _sign_payload(payload_b64: str) -> str:
    digest = hmac.new(
        _session_secret(),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(digest)


def create_vt_session_token(
    *,
    user_id: str,
    workspace_id: str,
    workspace_slug: str,
    issued_at: int | None = None,
    browser_session_nonce_hash: str | None = None,
    user_agent_hash: str | None = None,
) -> str:
    payload = {
        "user_id": user_id,
        "workspace_id": workspace_id,
        "workspace_slug": workspace_slug,
    }
    if issued_at is not None:
        payload["issued_at"] = int(issued_at)
    if browser_session_nonce_hash:
        payload["browser_session_nonce_hash"] = browser_session_nonce_hash
    if user_agent_hash:
        payload["user_agent_hash"] = user_agent_hash
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return f"{payload_b64}.{_sign_payload(payload_b64)}"


def parse_vt_session_token(token: str | None) -> AuthSessionPayload | None:
    decoded = decode_vt_session_token(token)
    if decoded is None:
        return None
    return AuthSessionPayload(
        user_id=decoded.user_id,
        workspace_id=decoded.workspace_id,
        workspace_slug=decoded.workspace_slug,
    )


def decode_vt_session_token(token: str | None) -> AuthSessionMetadata | None:
    if not token:
        return None
    raw = token.strip()
    if not raw or "." not in raw:
        return None
    payload_b64, signature_b64 = raw.split(".", 1)
    expected = _sign_payload(payload_b64)
    if not hmac.compare_digest(expected, signature_b64):
        return None
    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    user_id = str(payload.get("user_id") or "").strip()
    workspace_id = str(payload.get("workspace_id") or "").strip()
    workspace_slug = str(payload.get("workspace_slug") or "").strip()
    if not user_id or not workspace_id or not workspace_slug:
        return None
    issued_at_raw = payload.get("issued_at")
    issued_at = int(issued_at_raw) if isinstance(issued_at_raw, int) else None
    browser_session_nonce_hash = str(payload.get("browser_session_nonce_hash") or "").strip() or None
    user_agent_hash = str(payload.get("user_agent_hash") or "").strip() or None
    return AuthSessionMetadata(
        user_id=user_id,
        workspace_id=workspace_id,
        workspace_slug=workspace_slug,
        issued_at=issued_at,
        browser_session_nonce_hash=browser_session_nonce_hash,
        user_agent_hash=user_agent_hash,
    )


def session_grace_window_seconds() -> int:
    value = os.environ.get(SESSION_GRACE_WINDOW_ENV, "").strip()
    if not value:
        return DEFAULT_SESSION_GRACE_WINDOW_SECONDS
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_SESSION_GRACE_WINDOW_SECONDS
    return parsed if parsed > 0 else DEFAULT_SESSION_GRACE_WINDOW_SECONDS


def browser_session_nonce() -> str:
    return secrets.token_urlsafe(24)


def browser_session_nonce_hash(nonce: str | None) -> str | None:
    value = str(nonce or "").strip()
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def request_user_agent_hash(request: Request | None) -> str | None:
    if request is None:
        return None
    user_agent = str(request.headers.get("user-agent") or "").strip()
    if not user_agent:
        return None
    return hashlib.sha256(user_agent.encode("utf-8")).hexdigest()


def session_issue_time() -> int:
    return int(time.time())


def request_context_from_request(request: Request) -> RequestContext:
    token = request.cookies.get(VT_SESSION_COOKIE)
    payload = decode_vt_session_token(token)
    if payload is None:
        return RequestContext(
            authenticated=False,
            user_id="",
            workspace_id="public",
            workspace_slug="public",
            workspace_root=_WORKSPACES_ROOT / "public",
            hermes_home=_WORKSPACES_ROOT / "public",
            auth_invalid_reason=None,
        )
    user_name: str | None = None
    login_name: str | None = None
    auth_provider: str | None = None
    weixin_user_id: str | None = None
    feishu_open_id: str | None = None
    profile_completed = False
    organization_id: str | None = None
    organization_name: str | None = None
    membership_status = "unassigned"
    member_role: str | None = None
    sharing_enabled = False
    can_invite_members = False
    can_change_settings = False
    try:
        gateway_identity = _gateway_identity_module()
        get_user_organization_context = getattr(gateway_identity, "get_user_organization_context")
        get_user_record = getattr(gateway_identity, "get_user_record")

        record = get_user_record(payload.user_id)
        if not isinstance(record, dict):
            raise ValueError("session user record missing")

        record_workspace_id = str(record.get("workspace_id") or "").strip()
        record_workspace_slug = str(record.get("workspace_slug") or "").strip()
        if (
            not record_workspace_id
            or not record_workspace_slug
            or record_workspace_id != payload.workspace_id
            or record_workspace_slug != payload.workspace_slug
        ):
            raise ValueError("session workspace binding mismatch")

        raw_weixin_user_id = record.get("weixin_user_id")
        weixin_user_id = (
            raw_weixin_user_id.strip()
            if isinstance(raw_weixin_user_id, str) and raw_weixin_user_id.strip()
            else None
        )
        raw_name = record.get("display_name") or record.get("name")
        raw_login_name = record.get("password_login_name")
        if _looks_like_legacy_identity_label(
            raw_name if isinstance(raw_name, str) else None,
            user_id=payload.user_id,
            workspace_slug=payload.workspace_slug,
            weixin_user_id=weixin_user_id,
        ) and isinstance(raw_login_name, str) and raw_login_name.strip():
            raw_name = raw_login_name
        user_name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
        login_name = (
            raw_login_name.strip() if isinstance(raw_login_name, str) and raw_login_name.strip() else None
        )
        raw_provider = record.get("auth_provider")
        auth_provider = (
            raw_provider.strip() if isinstance(raw_provider, str) and raw_provider.strip() else None
        )
        raw_feishu_open_id = record.get("feishu_open_id")
        feishu_open_id = (
            raw_feishu_open_id.strip()
            if isinstance(raw_feishu_open_id, str) and raw_feishu_open_id.strip()
            else None
        )
        raw_profile_completed = record.get("profile_completed_at")
        profile_completed = (
            isinstance(raw_profile_completed, str) and bool(raw_profile_completed.strip())
        )
        org_context = get_user_organization_context(payload.user_id) or {}
        raw_organization_id = org_context.get("organization_id")
        organization_id = (
            raw_organization_id.strip()
            if isinstance(raw_organization_id, str) and raw_organization_id.strip()
            else None
        )
        raw_organization_name = org_context.get("organization_name")
        organization_name = (
            raw_organization_name.strip()
            if isinstance(raw_organization_name, str) and raw_organization_name.strip()
            else None
        )
        raw_membership_status = org_context.get("membership_status")
        membership_status = (
            raw_membership_status.strip()
            if isinstance(raw_membership_status, str) and raw_membership_status.strip()
            else "unassigned"
        )
        raw_member_role = org_context.get("member_role")
        member_role = (
            raw_member_role.strip()
            if isinstance(raw_member_role, str) and raw_member_role.strip()
            else None
        )
        raw_dataset_type = org_context.get("dataset_type")
        if not isinstance(raw_dataset_type, str) or not raw_dataset_type.strip():
            memberships = org_context.get("memberships")
            if isinstance(memberships, list):
                raw_dataset_type = next(
                    (
                        membership.get("dataset_type")
                        for membership in memberships
                        if isinstance(membership, dict)
                        and membership.get("organization_id") == organization_id
                        and isinstance(membership.get("dataset_type"), str)
                        and membership["dataset_type"].strip()
                    ),
                    None,
                )
        dataset_type = (
            raw_dataset_type.strip()
            if isinstance(raw_dataset_type, str) and raw_dataset_type.strip()
            else None
        )
        raw_industry_code = org_context.get("industry_code")
        industry_code = (
            raw_industry_code.strip()
            if isinstance(raw_industry_code, str) and raw_industry_code.strip()
            else None
        )
        sharing_enabled = bool(org_context.get("sharing_enabled"))
        can_invite_members = bool(org_context.get("can_invite_members"))
        can_change_settings = bool(org_context.get("can_change_settings"))

        if auth_provider == "weixin" and _is_weixin_runtime_session_expired(
            payload.user_id,
            session_issued_at=payload.issued_at,
        ):
            raise ValueError("weixin runtime session expired")
    except ValueError as exc:
        reason = (
            "weixin_session_expired"
            if str(exc).strip().lower() == "weixin runtime session expired"
            else None
        )
        return RequestContext(
            authenticated=False,
            user_id="",
            workspace_id="public",
            workspace_slug="public",
            workspace_root=_WORKSPACES_ROOT / "public",
            hermes_home=_WORKSPACES_ROOT / "public",
            auth_invalid_reason=reason,
        )
    except Exception:
        # The session token decoded successfully; enrichment failed for an
        # unexpected reason (e.g. a transient DB lock). Surface this as a
        # retryable context-unavailable state rather than treating the valid
        # session as invalid, so callers do not evict the session cookie.
        return RequestContext(
            authenticated=False,
            user_id="",
            workspace_id="public",
            workspace_slug="public",
            workspace_root=_WORKSPACES_ROOT / "public",
            hermes_home=_WORKSPACES_ROOT / "public",
            auth_invalid_reason="context_unavailable",
        )
    workspace_root, hermes_home = _resolve_authenticated_workspace_paths(payload)
    return RequestContext(
        authenticated=True,
        user_id=payload.user_id,
        workspace_id=payload.workspace_id,
        workspace_slug=payload.workspace_slug,
        workspace_root=workspace_root,
        hermes_home=hermes_home,
        organization_id=organization_id,
        organization_name=organization_name,
        dataset_type=dataset_type,
        industry_code=industry_code,
        membership_status=membership_status,
        member_role=member_role,
        sharing_enabled=sharing_enabled,
        can_invite_members=can_invite_members,
        can_change_settings=can_change_settings,
        user_name=user_name,
        login_name=login_name,
        auth_provider=auth_provider,
        weixin_user_id=weixin_user_id,
        feishu_open_id=feishu_open_id,
        profile_completed=profile_completed,
        auth_invalid_reason=None,
        avatar_url=record.get('avatar_url') if isinstance(record.get('avatar_url'), str) else None,
    )


def auth_context_payload(ctx: RequestContext) -> dict[str, Any]:
    password_login_enabled = False
    try:
        password_login_enabled = getattr(
            _gateway_identity_module(),
            "has_password_login_enabled",
        )()
    except Exception:
        password_login_enabled = False
    if not ctx.authenticated:
        return {
            "authenticated": False,
            "feishu_oauth_enabled": False,
            "password_login_enabled": password_login_enabled,
            "workspace_slug": None,
            "organization_id": None,
            "organization_name": None,
            "dataset_type": None,
            "industry_code": None,
            "membership_status": "unassigned",
            "member_role": None,
            "sharing_enabled": False,
            "can_invite_members": False,
            "can_change_settings": False,
            "profile_completed": False,
            "auth_invalid_reason": ctx.auth_invalid_reason,
            "user": None,
            "workspace": None,
            "currentWorkspaceId": None,
            "currentWorkspaceSlug": None,
            "currentWorkspaceRoot": None,
            "currentHermesHome": None,
        }

    user_payload = {
        "user_id": ctx.user_id,
        "workspace_id": ctx.workspace_id,
        "name": ctx.user_name or ctx.workspace_slug,
        "email": None,
        "avatar_url": ctx.avatar_url,
        "feishu_open_id": ctx.feishu_open_id or "",
        "workspace_slug": ctx.workspace_slug,
        "password_login_name": ctx.login_name,
        "auth_provider": ctx.auth_provider,
        "weixin_user_id": ctx.weixin_user_id,
        "organization_id": ctx.organization_id,
        "organization_name": ctx.organization_name,
        "dataset_type": ctx.dataset_type,
        "industry_code": ctx.industry_code,
        "membership_status": ctx.membership_status,
        "member_role": ctx.member_role,
        "sharing_enabled": ctx.sharing_enabled,
        "can_invite_members": ctx.can_invite_members,
        "can_change_settings": ctx.can_change_settings,
        "profile_completed": ctx.profile_completed,
    }
    workspace_payload = {
        "id": ctx.workspace_id,
        "slug": ctx.workspace_slug,
        "root": str(ctx.workspace_root),
        "hermes_home": str(ctx.hermes_home),
    }

    return {
        "authenticated": True,
        "feishu_oauth_enabled": False,
        "password_login_enabled": password_login_enabled,
        "workspace_slug": ctx.workspace_slug,
        "organization_id": ctx.organization_id,
        "organization_name": ctx.organization_name,
        "dataset_type": ctx.dataset_type,
        "industry_code": ctx.industry_code,
        "membership_status": ctx.membership_status,
        "member_role": ctx.member_role,
        "sharing_enabled": ctx.sharing_enabled,
        "can_invite_members": ctx.can_invite_members,
        "can_change_settings": ctx.can_change_settings,
        "profile_completed": ctx.profile_completed,
        "auth_invalid_reason": None,
        "user": user_payload,
        "workspace": workspace_payload,
        "currentWorkspaceId": ctx.workspace_id,
        "currentWorkspaceSlug": ctx.workspace_slug,
        "currentWorkspaceRoot": str(ctx.workspace_root),
        "currentHermesHome": str(ctx.hermes_home),
    }
