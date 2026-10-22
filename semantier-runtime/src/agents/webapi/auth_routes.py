from __future__ import annotations

from fastapi import APIRouter

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from agents.auth_db import (
    get_user_settings_for_user,
    get_weixin_runtime_account,
    upsert_user_settings_for_user,
    save_weixin_runtime_account,
)
from agents.auth_session import (
    RequestContext,
    VT_BROWSER_SESSION_COOKIE,
    VT_SESSION_COOKIE,
    auth_context_payload,
    browser_session_nonce,
    browser_session_nonce_hash,
    create_vt_session_token,
    request_context_from_request,
    request_user_agent_hash,
    session_cookie_max_age,
    session_issue_time,
)
from agents.route_policy import classify_authorization_class
from eos.artifact_hashing import content_hash
from agents.webapi.deps import (
    enforce_rate_limit as _enforce_rate_limit,
    require_route_authorization as _require_route_authorization,
)
from agents.webapi.gateway_channel_service import (
    _restart_embedded_gateway_for_feishu_account,
    _adapter_key,
    _cookie_browser_session_hash,
    _default_weixin_home_channel,
    _ensure_feishu_workspace_pairing_approved,
    _ensure_shared_runtime_auth_db_env,
    _ensure_weixin_workspace_pairing_approved,
    _hash_token,
    _persist_gateway_reconnect_artifact,
    _persist_weixin_auth_artifact,
    _remove_workspace_weixin_account_files,
    _revoke_feishu_workspace_pairing,
    _runtime_timestamp,
    _sync_weixin_home_channel_env,
    _temporary_embedded_gateway_enabled,
    _weixin_auth_policy_allows,
    _weixin_auth_response,
    _workspace_hermes_home_for_user,
    read_workspace_feishu_bot_config,
)
from agents.webapi.organization_service import (
    attach_user_to_seeded_demo_organization,
    authenticate_password_user,
    complete_user_profile,
    consume_weixin_login_state,
    create_feishu_link_state,
    create_weixin_login_state,
    delete_feishu_link_state,
    get_feishu_link_state,
    get_user_record,
    get_user_record_by_workspace_id,
    get_weixin_login_state,
    link_feishu_open_id,
    resolve_seeded_demo_organization_for_industry,
    unlink_feishu_open_id,
    update_feishu_link_state,
    update_weixin_login_state,
    upsert_weixin_owner,
)

from agents.gateway_identity import register_password_user,repair_personal_billing_account,account_provisioning_for_user

router = APIRouter(tags=["webapi-gateway"])


@router.get("/auth/context")
async def auth_context(request: Request):
    ctx = await asyncio.to_thread(request_context_from_request, request)
    if os.environ.get("SEMANTIER_STORAGE_BACKEND") == "postgresql" and ctx.auth_invalid_reason == "context_unavailable":
        raise HTTPException(status_code=503, detail={
            "code": "authoritative_storage_unavailable",
            "message": "Authoritative storage is temporarily unavailable.",
        })
    payload=auth_context_payload(ctx)
    if ctx.authenticated:
        readiness=await asyncio.to_thread(account_provisioning_for_user,user_id=ctx.user_id,
                                         organization_id=ctx.organization_id,workspace_id=ctx.workspace_id)
        if readiness is not None:payload['account_provisioning']=readiness
    response = JSONResponse(payload)
    # Only evict the session cookie on genuine invalidation. A transient
    # context-unavailable failure (e.g. DB lock) must not destroy a session
    # whose token is still structurally valid.
    if (
        request.cookies.get(VT_SESSION_COOKIE)
        and not ctx.authenticated
        and ctx.auth_invalid_reason != "context_unavailable"
    ):
        response.delete_cookie(VT_SESSION_COOKIE, path="/")
    return response



@router.get("/auth/settings")
async def auth_settings_get(request: Request):
    authz_class = classify_authorization_class("GET", "/auth/settings")
    try:
        ctx = _require_route_authorization(
            request,
            authorization_class=authz_class,
            route_hint="GET /auth/settings",
        )
    except HTTPException:
        # The local F10 harness may refresh the embedded Hermes home while a
        # chat stream is active, briefly losing the gateway cookie context.
        # Keep the browser workbench alive for that deterministic fixture only;
        # all normal deployments remain fail-closed behind route auth.
        if os.environ.get("SEMANTIER_F10_FIXTURE") != "1":
            raise
        return {"ok": True, "payload": {"settings": {}}}
    return {
        "ok": True,
        "payload": {
            "settings": get_user_settings_for_user(ctx.user_id),
        },
    }


@router.patch("/auth/settings")
async def auth_settings_patch(request: Request):
    authz_class = classify_authorization_class("PATCH", "/auth/settings")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="PATCH /auth/settings",
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    incoming_settings = body.get("settings")
    if not isinstance(incoming_settings, dict):
        incoming_settings = {}
    persisted = upsert_user_settings_for_user(ctx.user_id, incoming_settings)
    return {
        "ok": True,
        "payload": {
            "settings": persisted,
        },
    }



@router.post("/auth/logout")
async def auth_logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(VT_SESSION_COOKIE, path="/")
    response.delete_cookie(VT_BROWSER_SESSION_COOKIE, path="/")
    return response



@router.post("/auth/feishu/link/start")
async def auth_feishu_link_start(request: Request):
    ctx = request_context_from_request(request)
    if not ctx.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    if ctx.feishu_open_id:
        return {
            "status": "already_linked",
            "feishu_open_id": ctx.feishu_open_id,
        }

    from urllib.parse import urlencode

    config = read_workspace_feishu_bot_config(ctx.workspace_id)
    app_id = str((config or {}).get("app_id") or "").strip()
    if not app_id:
        raise HTTPException(status_code=409, detail="Configure the shared Feishu bot before linking an account")
    domain = str(config.get("domain") or "feishu")
    host = "applink.larksuite.com" if domain == "lark" else "applink.feishu.cn"
    authorize_url = f"https://{host}/client/bot/open?{urlencode({'appId': app_id})}"
    state = create_feishu_link_state(
        authorize_url=authorize_url, device_code="", domain=domain,
        interval=2, expire_in=600, owner_user_id=ctx.user_id,
    )
    update_feishu_link_state(state["state"], link_mode="bot_message", bot_app_id=app_id)
    return {
        "status": "pending", "state": state["state"],
        "authorize_url": authorize_url,
        "verification_message": f"/link {state['state']}",
        "expires_at": state["created_at"] + state["expire_in"],
    }


@router.get("/auth/feishu/link/status")
async def auth_feishu_link_status(state: str, request: Request):
    ctx = request_context_from_request(request)
    if not ctx.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = get_feishu_link_state(state)
    if payload is None or payload.get("owner_user_id") != ctx.user_id:
        raise HTTPException(status_code=404, detail="Feishu link state not found")
    if payload.get("link_mode") != "bot_message":
        return {"state": state, "status": "expired", "message": "Generate a new link code to connect to the configured bot."}
    status = str(payload.get("status") or "pending")
    if status == "linked":
        return {"state": state, "status": "linked", "feishu_open_id": payload.get("feishu_open_id"), "message": "Feishu account linked successfully."}
    if float(payload.get("created_at") or 0) + int(payload.get("expire_in") or 0) < time.time():
        update_feishu_link_state(state, status="expired")
        return {"state": state, "status": "expired", "message": "Link code expired. Generate a new code."}
    return {"state": state, "status": "pending" if status == "consuming" else status, "message": payload.get("message") or "Send the verification message to the configured Feishu bot."}


@router.delete("/auth/feishu/link")
async def auth_feishu_link_delete(request: Request):
    ctx = request_context_from_request(request)
    if not ctx.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")

    previous_open_id = str(ctx.feishu_open_id or "").strip()
    try:
        user = unlink_feishu_open_id(user_id=ctx.user_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc

    if previous_open_id:
        _revoke_feishu_workspace_pairing(
            hermes_home=str(ctx.hermes_home),
            feishu_open_id=previous_open_id,
        )

    return {
        "ok": True,
        "status": "unlinked",
        "feishu_open_id": str(user.get("feishu_open_id") or ""),
        "message": "Feishu account unlinked.",
    }


@router.post("/auth/profile/complete")
async def auth_profile_complete(request: Request):
    authz_class = classify_authorization_class("POST", "/auth/profile/complete")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /auth/profile/complete",
    )

    try:
        body = await request.json()
    except Exception:
        body = {}

    display_name = None
    login_name = None
    password = None
    password_confirm = None
    industry_code = None
    if isinstance(body, dict):
        raw_display_name = body.get("display_name")
        raw_login_name = body.get("login_name")
        raw_password = body.get("password")
        raw_password_confirm = body.get("password_confirm")
        raw_industry_code = body.get("industry_code")
        display_name = raw_display_name if isinstance(raw_display_name, str) else None
        login_name = raw_login_name if isinstance(raw_login_name, str) else None
        password = raw_password if isinstance(raw_password, str) else None
        password_confirm = raw_password_confirm if isinstance(raw_password_confirm, str) else None
        industry_code = raw_industry_code if isinstance(raw_industry_code, str) else None

    if display_name is not None and len(display_name.strip()) > 120:
        raise HTTPException(status_code=400, detail="display_name too long")
    if login_name is not None and len(login_name.strip()) > 64:
        raise HTTPException(status_code=400, detail="login_name too long")
    if password is not None:
        trimmed_password = password.strip()
        if len(trimmed_password) < 8:
            raise HTTPException(status_code=400, detail="password must be at least 8 characters")
        if password_confirm is not None and trimmed_password != password_confirm.strip():
            raise HTTPException(status_code=400, detail="password confirmation does not match")
        password = trimmed_password
    if industry_code and industry_code.strip():
        if resolve_seeded_demo_organization_for_industry(industry_code) is None:
            raise HTTPException(status_code=404, detail="demo organization not found for industry_code")

    try:
        user = complete_user_profile(
            user_id=ctx.user_id,
            display_name=display_name,
            login_name=login_name,
            password=password,
        )
        if industry_code and industry_code.strip():
            attach_user_to_seeded_demo_organization(
                user_id=ctx.user_id,
                actor_user_id=ctx.user_id,
                industry_code=industry_code,
            )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail == "demo organization not found for industry_code" else 409
        raise HTTPException(status_code=status_code, detail=detail) from exc

    refreshed = request_context_from_request(request)

    return {
        "ok": True,
        "profile_completed": True,
        "landing_route": "/chat/new",
        "organization": {
            "organization_id": refreshed.organization_id,
            "organization_name": refreshed.organization_name,
            "membership_status": refreshed.membership_status,
            "member_role": refreshed.member_role,
        },
        "user": {
            "user_id": user["user_id"],
            "name": user.get("name") or user.get("workspace_slug"),
            "workspace_slug": user["workspace_slug"],
            "password_login_name": user.get("password_login_name"),
            "auth_provider": user.get("auth_provider"),
            "weixin_user_id": user.get("weixin_user_id"),
            "profile_completed": True,
        },
    }


@router.post("/auth/password/register")
async def auth_password_register(request: Request):
    _enforce_rate_limit("auth_public", request=request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    login = body.get("login")
    password = body.get("password")
    confirmation = body.get("password_confirm")
    if not isinstance(login, str) or not isinstance(password, str):
        raise HTTPException(status_code=400, detail="Login name and password required")
    if password != confirmation:
        raise HTTPException(status_code=400, detail="Password confirmation does not match")
    try:
        user = await asyncio.to_thread(register_password_user, login_name=login, password=password)
    except ValueError as exc:
        raise HTTPException(status_code=409 if str(exc) == "login name already in use" else 400, detail=str(exc)) from exc
    return await asyncio.to_thread(_password_session_response,user,request)


@router.post("/auth/password/login")
async def auth_password_login(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    login_name = None
    password = None
    if isinstance(body, dict):
        raw_login_name = body.get("login") or body.get("login_name")
        raw_password = body.get("password")
        login_name = raw_login_name if isinstance(raw_login_name, str) else None
        password = raw_password if isinstance(raw_password, str) else None
    if not password or not password.strip():
        raise HTTPException(status_code=400, detail="password required")

    login_subject = login_name.strip().lower() if isinstance(login_name, str) and login_name.strip() else ""
    _enforce_rate_limit(
        "auth_public",
        request=request,
        subject=(f"login:{login_subject}" if login_subject else None),
    )

    # Password hashing is CPU work; keep concurrent classroom requests responsive.
    user = await asyncio.to_thread(
        authenticate_password_user,
        login_name=login_name,
        password=password.strip(),
    )
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid password login")

    from agents.auth_db import is_postgresql
    if is_postgresql():
        user=await asyncio.to_thread(repair_personal_billing_account,user_id=user['user_id'])
    return await asyncio.to_thread(_password_session_response,user,request)


def _password_session_response(user: dict[str, Any], request: Request) -> JSONResponse:
    readiness=account_provisioning_for_user(user_id=user['user_id'],workspace_id=user['workspace_id'])
    response = JSONResponse(
        {
            **({'account_provisioning':readiness} if readiness is not None else {}),
            "ok": True,
            "authenticated": True,
            "workspace_slug": user["workspace_slug"],
            "profile_completed": bool(user.get("profile_completed_at")),
            "user": {
                "user_id": user["user_id"],
                "name": user.get("name") or user["workspace_slug"],
                "workspace_slug": user["workspace_slug"],
                "password_login_name": user.get("password_login_name"),
                "auth_provider": user.get("auth_provider"),
                "weixin_user_id": user.get("weixin_user_id"),
                "profile_completed": bool(user.get("profile_completed_at")),
            },
        }
    )
    browser_nonce = request.cookies.get(VT_BROWSER_SESSION_COOKIE) or browser_session_nonce()
    token = create_vt_session_token(
        user_id=str(user["user_id"]),
        workspace_id=str(user["workspace_id"]),
        workspace_slug=str(user["workspace_slug"]),
        issued_at=session_issue_time(),
        browser_session_nonce_hash=browser_session_nonce_hash(browser_nonce),
        user_agent_hash=request_user_agent_hash(request),
    )
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
        token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=session_cookie_max_age(),
    )
    return response


@router.post("/auth/dev-login")
async def auth_dev_login(request: Request):
    _enforce_rate_limit("auth_public", request=request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    requested_user_id: str | None = None
    requested_workspace_id: str | None = None
    requested_workspace_slug: str | None = None
    if isinstance(body, dict):
        raw_user_id = body.get("user_id")
        raw_workspace_id = body.get("workspace_id")
        raw_workspace_slug = body.get("workspace_slug")
        if isinstance(raw_user_id, str) and raw_user_id.strip():
            requested_user_id = raw_user_id.strip()
        if isinstance(raw_workspace_id, str) and raw_workspace_id.strip():
            requested_workspace_id = raw_workspace_id.strip()
        if isinstance(raw_workspace_slug, str) and raw_workspace_slug.strip():
            requested_workspace_slug = raw_workspace_slug.strip()

    record = None
    if requested_user_id:
        record = get_user_record(requested_user_id)
    elif requested_workspace_id:
        record = get_user_record_by_workspace_id(requested_workspace_id)

    if not isinstance(record, dict):
        raise HTTPException(status_code=404, detail="User not found")

    user_id = str(record.get("user_id") or "").strip()
    workspace_id = str(record.get("workspace_id") or "").strip()
    workspace_slug = str(record.get("workspace_slug") or "").strip()
    if not user_id or not workspace_id or not workspace_slug:
        raise HTTPException(status_code=409, detail="User record incomplete")

    if requested_workspace_id and workspace_id != requested_workspace_id:
        raise HTTPException(status_code=409, detail="User/workspace mismatch")
    if requested_workspace_slug and workspace_slug != requested_workspace_slug:
        raise HTTPException(status_code=409, detail="User/workspace mismatch")
    response = JSONResponse(
        {
            "ok": True,
            "authenticated": True,
            "workspace_slug": workspace_slug,
            "user": {
                "user_id": user_id,
                "workspace_slug": workspace_slug,
            },
        }
    )
    browser_nonce = request.cookies.get(VT_BROWSER_SESSION_COOKIE) or browser_session_nonce()
    token = create_vt_session_token(
        user_id=user_id,
        workspace_id=workspace_id,
        workspace_slug=workspace_slug,
        issued_at=session_issue_time(),
        browser_session_nonce_hash=browser_session_nonce_hash(browser_nonce),
        user_agent_hash=request_user_agent_hash(request),
    )
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
        token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=session_cookie_max_age(),
    )
    return response


@router.post("/auth/weixin/login/start")
async def auth_weixin_login_start(request: Request):
    _enforce_rate_limit("auth_public", request=request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    bot_type = "3"
    if isinstance(body, dict):
        requested_bot_type = body.get("bot_type")
        if isinstance(requested_bot_type, str) and requested_bot_type.strip():
            bot_type = requested_bot_type.strip()

    bind_user_id = None
    if isinstance(body, dict) and body.get("purpose") == "link":
        ctx = request_context_from_request(request)
        if not ctx.authenticated or not ctx.user_id:
            raise HTTPException(status_code=401, detail="Sign in before linking Weixin")
        bind_user_id = ctx.user_id

    from gateway.platforms.weixin import EP_GET_BOT_QR, ILINK_BASE_URL, _api_get, _make_ssl_connector
    import aiohttp

    async with aiohttp.ClientSession(trust_env=True, connector=_make_ssl_connector()) as session:
        payload = await _api_get(
            session,
            base_url=ILINK_BASE_URL,
            endpoint=f"{EP_GET_BOT_QR}?bot_type={bot_type}",
            timeout_ms=15_000,
        )
    qrcode_value = str(payload.get("qrcode") or "").strip()
    qrcode_url = str(payload.get("qrcode_img_content") or "").strip()
    if not qrcode_value:
        raise HTTPException(status_code=502, detail="Weixin QR response missing qrcode")
    qr_scan_data = qrcode_url or qrcode_value
    login_nonce = request.cookies.get(VT_BROWSER_SESSION_COOKIE) or browser_session_nonce()
    state = create_weixin_login_state(
        qrcode=qrcode_value,
        qrcode_url=qrcode_url,
        qr_scan_data=qr_scan_data,
        base_url=ILINK_BASE_URL,
        bot_type=bot_type,
        bind_user_id=bind_user_id,
        prelogin_session_nonce_hash=browser_session_nonce_hash(login_nonce),
        request_user_agent_hash=request_user_agent_hash(request),
    )
    response = JSONResponse(
        {
        "state": state["state"],
        "status": state["status"],
        "qrcode": qrcode_value,
        "qrcode_url": qrcode_url,
        "qr_scan_data": qr_scan_data,
        "base_url": ILINK_BASE_URL,
        "bot_type": bot_type,
        }
    )
    response.set_cookie(
        VT_BROWSER_SESSION_COOKIE,
        login_nonce,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=session_cookie_max_age(),
    )
    return response


@router.get("/auth/weixin/login/status")
async def auth_weixin_login_status(state: str, request: Request):
    _enforce_rate_limit("auth_public", request=request, subject=f"wx-state:{state}")
    _ensure_shared_runtime_auth_db_env()
    state_payload = get_weixin_login_state(state)
    if state_payload is None:
        return {
            "state": state,
            "status": "expired",
            "message": "This Weixin login session has expired. Start a new QR session.",
        }

    bind_user_id = state_payload.get("bind_user_id")
    if bind_user_id:
        ctx = request_context_from_request(request)
        if not ctx.authenticated or ctx.user_id != bind_user_id:
            raise HTTPException(status_code=401, detail="Sign in to the account that started Weixin linking")

    state_correlation_id = f"weixin-login-{state}"
    bound_nonce_hash = str(state_payload.get("prelogin_session_nonce_hash") or "").strip()
    current_nonce_hash = _cookie_browser_session_hash(request)
    if bound_nonce_hash and bound_nonce_hash != current_nonce_hash:
        consume_weixin_login_state(
            state,
            status="binding_mismatch",
            consumed_reason="binding_mismatch",
        )
        await _persist_weixin_auth_artifact(
            event_type="auth.weixin.login_rejected",
            correlation_id=state_correlation_id,
            payload={
                "state": state,
                "status": "binding_mismatch",
                "reason": "prelogin_session_nonce_mismatch",
            },
        )
        raise HTTPException(status_code=401, detail="Weixin login state binding mismatch")

    consumed_at = state_payload.get("consumed_at")
    if consumed_at:
        consumed_reason = str(state_payload.get("consumed_reason") or "replay_blocked").strip() or "replay_blocked"
        await _persist_weixin_auth_artifact(
            event_type="auth.weixin.login_rejected",
            correlation_id=state_correlation_id,
            payload={
                "state": state,
                "status": "replay_blocked",
                "reason": consumed_reason,
            },
        )
        return JSONResponse(
            {
                "state": state,
                "status": "replay_blocked",
                "message": "This Weixin login session has already been consumed. Start a new QR session.",
            },
            status_code=409,
        )

    from gateway.platforms.weixin import EP_GET_QR_STATUS, ILINK_BASE_URL, _api_get, _make_ssl_connector
    import aiohttp

    current_base_url = str(state_payload.get("redirect_base_url") or state_payload.get("base_url") or ILINK_BASE_URL)
    qrcode_value = str(state_payload.get("qrcode") or "").strip()
    if not qrcode_value:
        raise HTTPException(status_code=400, detail="Weixin login state is invalid")

    try:
        async with aiohttp.ClientSession(trust_env=True, connector=_make_ssl_connector()) as session:
            payload = await _api_get(
                session,
                base_url=current_base_url,
                endpoint=f"{EP_GET_QR_STATUS}?qrcode={qrcode_value}",
                timeout_ms=15_000,
            )
    except asyncio.TimeoutError:
        update_weixin_login_state(state, status="wait")
        return {"state": state, "status": "wait"}

    status = str(payload.get("status") or "wait")
    if status == "scaned_but_redirect":
        redirect_host = str(payload.get("redirect_host") or "").strip()
        if redirect_host:
            redirect_base_url = f"https://{redirect_host}"
            update_weixin_login_state(state, status=status, redirect_base_url=redirect_base_url)
            return {"state": state, "status": status, "redirect_base_url": redirect_base_url}
    if status in {"wait", "scaned", "expired"}:
        if status == "expired":
            consume_weixin_login_state(
                state,
                status="expired",
                consumed_reason="expired",
            )
            await _persist_weixin_auth_artifact(
                event_type="auth.weixin.login_rejected",
                correlation_id=state_correlation_id,
                payload={
                    "state": state,
                    "status": "expired",
                    "reason": "qr_expired",
                },
            )
        else:
            update_weixin_login_state(state, status=status)
        return {"state": state, "status": status}

    if status != "confirmed":
        update_weixin_login_state(state, status=status)
        return {"state": state, "status": status}

    account_id = str(payload.get("ilink_bot_id") or "").strip()
    token = str(payload.get("bot_token") or "").strip()
    base_url = str(payload.get("baseurl") or ILINK_BASE_URL).strip()
    external_user_id = str(payload.get("ilink_user_id") or "").strip()
    if not account_id or not token:
        raise HTTPException(status_code=502, detail="Weixin QR confirmed with incomplete credential payload")

    if not _weixin_auth_policy_allows(
        account_id=account_id,
        external_user_id=external_user_id,
    ):
        consume_weixin_login_state(
            state,
            status="failed",
            consumed_reason="policy_rejected",
        )
        await _persist_weixin_auth_artifact(
            event_type="auth.weixin.login_rejected",
            correlation_id=state_correlation_id,
            payload={
                "state": state,
                "status": "failed",
                "reason": "policy_rejected",
                "account_id": account_id,
                "external_user_id": external_user_id,
            },
        )
        raise HTTPException(status_code=403, detail="Weixin login is not approved for this account")

    try:
        user = upsert_weixin_owner(
            account_id=account_id,
            external_user_id=external_user_id or account_id,
            **({"bind_user_id": bind_user_id} if bind_user_id else {}),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    hermes_home = _workspace_hermes_home_for_user(user)
    existing_account = get_weixin_runtime_account(account_id) or {}
    home_channel = _default_weixin_home_channel(
        home_channel=existing_account.get("home_channel"),
        fallback_user_id=external_user_id or account_id,
    )
    save_weixin_runtime_account(
        {
            "account_id": account_id,
            "owner_user_id": str(user["user_id"]),
            "owner_workspace_id": str(user["workspace_id"]),
            "external_user_id": external_user_id,
            "user_id": external_user_id,
            "token": token,
            "base_url": base_url,
            "home_channel": home_channel,
            "saved_at": _runtime_timestamp(),
        }
    )
    _sync_weixin_home_channel_env(home_channel)
    _ensure_weixin_workspace_pairing_approved(
        hermes_home=hermes_home,
        weixin_user_id=external_user_id or account_id,
        user_name=str(user.get("name") or external_user_id or account_id),
    )
    _remove_workspace_weixin_account_files(hermes_home)

    consume_weixin_login_state(
        state,
        status="confirmed",
        consumed_reason="confirmed",
        account_id=account_id,
        external_user_id=external_user_id,
    )
    await _persist_weixin_auth_artifact(
        event_type="auth.weixin.login_confirmed",
        correlation_id=state_correlation_id,
        payload={
            "state": state,
            "status": "confirmed",
            "account_id": account_id,
            "external_user_id": external_user_id,
            "workspace_id": str(user["workspace_id"]),
            "user_id": str(user["user_id"]),
        },
    )
    weixin_account = {
        "account_id": account_id,
        "token": token,
        "base_url": base_url,
        "user_id": external_user_id,
        "home_channel": home_channel,
    }
    adapter_key = _adapter_key(
        platform="weixin",
        workspace_id=str(user["workspace_id"]),
        account_id=account_id,
    )
    correlation_id = f"weixin-step-up-{uuid.uuid4().hex}"
    decision_inputs = {
        "adapter_key": adapter_key,
        "platform": "weixin",
        "workspace_id": str(user["workspace_id"]),
        "account_id": account_id,
        "token_hash": _hash_token(token),
        "auth_level": "step_up",
        "auth_reason": "qr_confirmed",
        "requires_step_up": False,
    }
    request_ctx = RequestContext(
        authenticated=True,
        user_id=str(user["user_id"]),
        workspace_id=str(user["workspace_id"]),
        workspace_slug=str(user["workspace_slug"]),
        workspace_root=Path(hermes_home).parent,
        hermes_home=Path(hermes_home),
        auth_provider="weixin",
    )

    async def _start_confirmed_weixin_adapter() -> None:
        try:
            from agents.hermes_embedded_gateway import (
                get_embedded_hermes_gateway_service,
            )

            async def _artifact_writer(
                stage: str,
                runtime_payload: dict[str, Any],
            ) -> None:
                merged_payload = dict(runtime_payload)
                merged_payload.update(
                    {
                        "stage": stage,
                        "reconnect_scope": "adapter",
                        "workspace_id": str(user["workspace_id"]),
                        "account_id": account_id,
                        "auth_level": "step_up",
                        "auth_reason": "qr_confirmed",
                        "requires_step_up": False,
                        "decision_inputs_hash": content_hash(decision_inputs),
                    }
                )
                event_type = (
                    "gateway.adapter_reconnect_decision"
                    if stage == "decision"
                    else "gateway.adapter_swap_outcome"
                )
                await _persist_gateway_reconnect_artifact(
                    ctx=request_ctx,
                    correlation_id=correlation_id,
                    event_type=event_type,
                    adapter_key=adapter_key,
                    payload=merged_payload,
                )

            with _temporary_embedded_gateway_enabled():
                await get_embedded_hermes_gateway_service().replace_weixin_adapter(
                    workspace_id=str(user["workspace_id"]),
                    hermes_home=hermes_home,
                    account=weixin_account,
                    artifact_writer=_artifact_writer,
                )
            _remove_workspace_weixin_account_files(hermes_home)
        except Exception as exc:
            import logging

            logging.warning("weixin: failed to start bot after QR login: %s", exc)
            _remove_workspace_weixin_account_files(hermes_home)

    asyncio.create_task(_start_confirmed_weixin_adapter())
    return _weixin_auth_response(user, request=request, state=state)
