"""Read-only system diagnostic routes for the Semantier web API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from agents.auth_db import auth_db_path, get_weixin_runtime_account, is_postgresql, ensure_auth_db
from agents.gateway_identity import get_active_weixin_correlation_for_user
from agents.route_policy import classify_authorization_class
from agents.webapi.deps import require_route_authorization
from agents.webapi.errors import route_auth_class_not_configured


from agents.webapi.gateway_channel_service import get_active_weixin_correlation_for_user
from agents.webapi.organization_service import _auth_runtime_account_snapshot

router = APIRouter(tags=["webapi-gateway"])


def _auth_runtime_account_snapshot(account: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(account, dict):
        return None
    return {
        "account_id": str(account.get("account_id") or "").strip(),
        "owner_user_id": str(account.get("owner_user_id") or "").strip(),
        "owner_workspace_id": str(account.get("owner_workspace_id") or "").strip(),
        "external_user_id": str(account.get("external_user_id") or "").strip(),
        "base_url": str(account.get("base_url") or "").strip(),
        "home_channel": str(account.get("home_channel") or "").strip(),
        "runtime_session_state": str(account.get("runtime_session_state") or "").strip(),
        "runtime_session_updated_at": str(account.get("runtime_session_updated_at") or "").strip(),
        "saved_at": str(account.get("saved_at") or "").strip(),
        "token_present": bool(str(account.get("token") or "").strip()),
    }


@router.get("/system/auth/state")
async def system_auth_state(request: Request):
    authz_class = classify_authorization_class("GET", "/system/auth/state")
    if authz_class is None:
        raise route_auth_class_not_configured()
    ctx = require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="GET /system/auth/state",
    )
    user = {
        "user_id": ctx.user_id,
        "workspace_id": ctx.workspace_id,
        "workspace_slug": ctx.workspace_slug,
        "organization_id": ctx.organization_id,
        "organization_name": ctx.organization_name,
        "membership_status": ctx.membership_status,
        "member_role": ctx.member_role,
        "auth_provider": ctx.auth_provider,
        "weixin_user_id": ctx.weixin_user_id,
        "feishu_open_id": ctx.feishu_open_id,
        "profile_completed": ctx.profile_completed,
    }
    correlation = get_active_weixin_correlation_for_user(ctx.user_id)
    runtime_account = None
    if isinstance(correlation, dict):
        account_id = str(correlation.get("linked_account_id") or "").strip()
        if account_id:
            runtime_account = get_weixin_runtime_account(account_id)
    if is_postgresql():
        ensure_auth_db()
        auth_store = {"backend": "postgresql", "schema": "semantier_identity", "path": None, "exists": True}
    else:
        auth_store = {"backend": "sqlite", "path": str(auth_db_path()), "exists": auth_db_path().exists()}
    return {
        "authDb": auth_store,
        "user": user,
        "weixin": {
            "correlation": correlation,
            "runtimeAccount": _auth_runtime_account_snapshot(runtime_account),
        },
    }
