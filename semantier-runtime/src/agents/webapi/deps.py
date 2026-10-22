"""Shared FastAPI dependencies for Semantier web API route modules."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
import time

from fastapi import HTTPException, Request

from agents.auth_session import RequestContext, request_context_from_request
from agents.route_policy import RouteAuthorizationClass


_LOGGER = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RATE_LIMIT_STATE_LOCK = threading.Lock()
_RATE_LIMIT_STATE: dict[str, tuple[float, int]] = {}
_RATE_LIMITS: dict[str, tuple[int, int]] = {
    "auth_public": (60, 60),
    "weixin_reconnect_user": (12, 60),
    "session_create_user": (30, 60),
    "session_create_tenant": (120, 60),
    "chat_user": (120, 60),
    "chat_tenant": (600, 60),
}


def ensure_repo_hermes_agent_on_path() -> None:
    """Use the launcher ordering so parent-owned packages remain authoritative."""
    from agents.launcher import _ensure_repo_hermes_agent_on_path

    _ensure_repo_hermes_agent_on_path()


def request_context(request: Request) -> RequestContext:
    return request_context_from_request(request)


def optional_workspace_hermes_home(request: Request) -> Path | None:
    ctx = request_context(request)
    return ctx.hermes_home if ctx.authenticated else None


def require_authenticated_context(request: Request) -> RequestContext:
    ctx = request_context(request)
    if not ctx.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    return ctx


def request_client_scope(request: Request) -> str:
    forwarded_for = str(request.headers.get("x-forwarded-for") or "").strip()
    if forwarded_for:
        first_hop = forwarded_for.split(",", 1)[0].strip()
        if first_hop:
            return first_hop
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown-client"


def _log_tenant_authz(
    *,
    ctx: RequestContext,
    authorization_class: RouteAuthorizationClass,
    route_hint: str,
) -> None:
    _LOGGER.info(
        "tenant_authz class=%s route=%s user_id=%s workspace_id=%s organization_id=%s membership_status=%s",
        authorization_class.value,
        route_hint,
        ctx.user_id,
        ctx.workspace_id,
        str(ctx.organization_id or ""),
        ctx.membership_status,
    )


def require_route_authorization(
    request: Request,
    *,
    authorization_class: RouteAuthorizationClass,
    route_hint: str,
    target_user_id: str | None = None,
) -> RequestContext:
    ctx = require_authenticated_context(request)
    if authorization_class == RouteAuthorizationClass.SELF_ONLY:
        target = str(target_user_id or ctx.user_id).strip()
        if target and target != ctx.user_id:
            raise HTTPException(status_code=403, detail="Self-only route")
        _log_tenant_authz(ctx=ctx, authorization_class=authorization_class, route_hint=route_hint)
        return ctx

    if authorization_class == RouteAuthorizationClass.TENANT_MEMBER:
        if not ctx.organization_id:
            raise HTTPException(status_code=409, detail="No organization selected")
        _log_tenant_authz(ctx=ctx, authorization_class=authorization_class, route_hint=route_hint)
        return ctx

    if authorization_class == RouteAuthorizationClass.TENANT_ADMIN:
        if not ctx.organization_id:
            raise HTTPException(status_code=409, detail="No organization selected")
        _log_tenant_authz(ctx=ctx, authorization_class=authorization_class, route_hint=route_hint)
        return ctx

    if authorization_class == RouteAuthorizationClass.SYSTEM_OPERATOR:
        _log_tenant_authz(ctx=ctx, authorization_class=authorization_class, route_hint=route_hint)
        return ctx

    raise HTTPException(status_code=500, detail="Unsupported authorization class")


def enforce_rate_limit(
    bucket: str,
    *,
    request: Request,
    ctx: RequestContext | None = None,
    subject: str | None = None,
) -> None:
    config = _RATE_LIMITS.get(bucket)
    if config is None:
        return
    max_requests, window_seconds = config
    if max_requests <= 0 or window_seconds <= 0:
        return

    if bucket.endswith("_user"):
        actor_scope = str(ctx.user_id or "").strip() if ctx else ""
        scope = f"user={actor_scope or 'unknown-user'}"
    elif bucket.endswith("_tenant"):
        tenant_scope = str(ctx.organization_id or ctx.workspace_id or "").strip() if ctx else ""
        scope = f"tenant={tenant_scope or 'unknown-tenant'}"
    else:
        scope = request_client_scope(request)
        if subject:
            scope = f"{scope}|subject={subject}"

    state_key = f"{bucket}|{scope}"
    now = time.monotonic()
    with _RATE_LIMIT_STATE_LOCK:
        window_start, count = _RATE_LIMIT_STATE.get(state_key, (now, 0))
        if now - window_start >= float(window_seconds):
            window_start = now
            count = 0
        count += 1
        _RATE_LIMIT_STATE[state_key] = (window_start, count)

    if count > max_requests:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


def require_org_admin(ctx: RequestContext) -> None:
    if not ctx.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not ctx.organization_id:
        raise HTTPException(status_code=409, detail="No organization selected")


_request_client_scope = request_client_scope
_enforce_rate_limit = enforce_rate_limit
_log_tenant_authz = _log_tenant_authz
_require_authenticated_context = require_authenticated_context
_require_route_authorization = require_route_authorization
_require_org_admin = require_org_admin
