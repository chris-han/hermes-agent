"""Password ingress shared by legacy and independent-task compositions."""

from __future__ import annotations
import asyncio
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from agents.auth_session import (
    VT_BROWSER_SESSION_COOKIE,
    VT_SESSION_COOKIE,
    browser_session_nonce,
    browser_session_nonce_hash,
    create_vt_session_token,
    request_user_agent_hash,
    session_cookie_max_age,
    session_issue_time,
)
from agents.webapi.deps import enforce_rate_limit as _enforce_rate_limit
from agents.gateway_identity import (
    register_password_user,
    repair_personal_billing_account,
    account_provisioning_for_user,
    authenticate_password_user,
)

router = APIRouter(tags=["authentication"])


@router.get("/auth/model-preferences")
async def signup_model_recommendations():
    from agents.model_gateway.default_models import public_catalog
    return public_catalog()


def _registration_observation(request, attempt_ref, *, failed=False, reason=None, status=None):
    """A received signup attempt, without login/password or claimed user identity."""
    try:
        from agents.auth_db import is_postgresql
        if not is_postgresql():return
        from agents.webapi.journey_routes import read_journey
        from agents.auth_db_postgres import connection
        from agents.operations_store import append_event
        from agents.operations_journey import utc
        from datetime import datetime,timezone
        subject=read_journey(request)
        if not subject:return
        surface='workspace';basis='browser_registration_request'
        intent=request.query_params.get('intent_ref')
        if intent:
            with connection() as c:
                device=c.execute("SELECT 1 FROM semantier_gateway.device_connections WHERE intent_ref=%s AND status='pending' AND user_id IS NULL AND expires_at>now()",(intent,)).fetchone()
            surface='windows' if device else 'unknown';basis='server_device_intent' if device else 'unavailable'
        now=utc(datetime.now(timezone.utc));name='registration_failed' if failed else 'registration_started'
        append_event(dict(event_id='registration:'+attempt_ref+':'+name,schema_version='journey-event-v1',event_name=name,producer_id='identity',producer_version='v1',surface=surface,subject_ref=subject,occurred_at=now,received_at=now,evidence_type='server_fact',receipt_ref=attempt_ref,status=status or ('failed' if failed else 'started'),clock_status='accepted',properties={'attempt_id':attempt_ref,'origin_basis':basis,**({'reason_code':reason} if reason else {})}))
    except Exception:
        import logging
        logging.getLogger(__name__).warning('Registration observation unavailable')


@router.post("/auth/password/register")
async def auth_password_register(request: Request):
    _enforce_rate_limit("auth_public", request=request)
    from uuid import uuid4
    attempt_ref=uuid4().hex
    await asyncio.to_thread(_registration_observation,request,attempt_ref)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    from agents.model_gateway.default_models import normalize_preference
    try:
        model_preference = normalize_preference(body.get("model_preference"))
    except ValueError:
        raise HTTPException(422, "Please choose a supported model") from None
    login = body.get("login")
    password = body.get("password")
    confirmation = body.get("password_confirm")
    if not isinstance(login, str) or not isinstance(password, str):
        await asyncio.to_thread(_registration_observation,request,attempt_ref,failed=True,reason='missing_fields')
        raise HTTPException(status_code=400, detail="Login name and password required")
    if password != confirmation:
        await asyncio.to_thread(_registration_observation,request,attempt_ref,failed=True,reason='confirmation_mismatch')
        raise HTTPException(
            status_code=400, detail="Password confirmation does not match"
        )
    try:
        from agents.auth_db import is_postgresql
        if is_postgresql():
            from agents.trial_invitations import register_invited
            from agents.webapi.journey_routes import read_journey
            journey = read_journey(request)
            if journey is None:
                await asyncio.to_thread(_registration_observation,request,attempt_ref,failed=True,reason='invitation_required',status='awaiting_invitation')
                raise HTTPException(403, "Approved trial invitation required")
            user = await asyncio.to_thread(register_invited, journey_ref=journey, login_name=login, password=password,device_intent=request.query_params.get("intent_ref"),model_preference=model_preference)
        else:
            user = await asyncio.to_thread(register_password_user, login_name=login, password=password, model_preference=model_preference)
    except PermissionError:
        await asyncio.to_thread(_registration_observation,request,attempt_ref,failed=True,reason='invitation_required',status='awaiting_invitation')
        raise HTTPException(403, "Approved trial invitation required") from None
    except ValueError as exc:
        await asyncio.to_thread(_registration_observation,request,attempt_ref,failed=True,reason='input_or_login_rejected')
        raise HTTPException(
            status_code=409 if str(exc) == "login name already in use" else 400,
            detail=str(exc),
        ) from exc
    return await asyncio.to_thread(_password_session_response, user, request)


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

    login_subject = (
        login_name.strip().lower()
        if isinstance(login_name, str) and login_name.strip()
        else ""
    )
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
        user = await asyncio.to_thread(
            repair_personal_billing_account, user_id=user["user_id"]
        )
    return await asyncio.to_thread(_password_session_response, user, request)


def _password_session_response(user: dict[str, Any], request: Request) -> JSONResponse:
    from agents.auth_db import is_postgresql
    if is_postgresql():
        from agents.webapi.journey_routes import link_authenticated_journey
        from agents.operations_store import capture_fact
        link_authenticated_journey(request, user)
        surface='workspace'
        intent=request.query_params.get('intent_ref')
        if intent:
            from agents.auth_db_postgres import connection
            with connection() as c:
                device=c.execute("SELECT user_id FROM semantier_gateway.device_connections WHERE intent_ref=%s AND status IN ('pending','approved','claimed','installed') AND expires_at>now()",(intent,)).fetchone()
            if device and device['user_id'] in {None,user['user_id']}:surface='windows'
        capture_fact(name='login_completed', user_id=user['user_id'], producer='identity',
                     receipt_ref=browser_session_nonce(), surface=surface)
    readiness = account_provisioning_for_user(
        user_id=user["user_id"], workspace_id=user["workspace_id"]
    )
    response = JSONResponse(
        {
            **({"account_provisioning": readiness} if readiness is not None else {}),
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
    if is_postgresql():
        from agents.webapi.journey_routes import issue_journey_cookie
        issue_journey_cookie(response,request)
    browser_nonce = (
        request.cookies.get(VT_BROWSER_SESSION_COOKIE) or browser_session_nonce()
    )
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
