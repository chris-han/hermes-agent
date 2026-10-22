from __future__ import annotations

from fastapi import APIRouter

from fastapi import HTTPException, Request

from agents.auth_session import request_context_from_request
from agents.knowledge_access_model import knowledge_tier_access_payload
from agents.route_policy import RouteAuthorizationClass, classify_authorization_class
from agents.webapi.deps import (
    _log_tenant_authz,
    require_authenticated_context as _require_authenticated_context,
    require_route_authorization as _require_route_authorization,
)
from agents.webapi.organization_service import (
    _organization_payload_for_context,
    attach_user_to_seeded_demo_organization,
    get_organization_materialization_policy,
    get_user_membership_for_organization,
    join_organization,
    list_seeded_demo_organization_profiles,
    set_user_sharing_enabled,
    switch_user_organization,
    update_membership_role,
    update_membership_status,
    update_organization_materialization_policy,
)

router = APIRouter(tags=["webapi-gateway"])

@router.get("/organizations/me")
async def organizations_me(request: Request):
    authz_class = classify_authorization_class("GET", "/organizations/me")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="GET /organizations/me",
    )
    return _organization_payload_for_context(ctx)


@router.get("/organizations/materialization-policy")
async def organizations_materialization_policy_get(request: Request):
    authz_class = classify_authorization_class("GET", "/organizations/materialization-policy")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="GET /organizations/materialization-policy",
    )
    try:
        policy = get_organization_materialization_policy(str(ctx.organization_id))
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail == "organization not found" else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return {
        "organization_id": ctx.organization_id,
        "policy": policy,
    }


@router.post("/organizations/materialization-policy")
async def organizations_materialization_policy_update(request: Request):
    authz_class = classify_authorization_class("POST", "/organizations/materialization-policy")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /organizations/materialization-policy",
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    policy_patch = body.get("policy") if isinstance(body.get("policy"), dict) else body
    try:
        update_organization_materialization_policy(
            organization_id=str(ctx.organization_id),
            policy_patch=policy_patch,
            actor_user_id=ctx.user_id,
        )
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail == "organization not found" else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    refreshed = request_context_from_request(request)
    return _organization_payload_for_context(refreshed)


@router.get("/organizations/demo-profiles")
async def organizations_demo_profiles(request: Request):
    authz_class = classify_authorization_class("GET", "/organizations/demo-profiles")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="GET /organizations/demo-profiles",
    )
    return {"profiles": list_seeded_demo_organization_profiles()}


@router.get("/organizations/knowledge-access")
async def organizations_knowledge_access(request: Request):
    authz_class = classify_authorization_class("GET", "/organizations/knowledge-access")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="GET /organizations/knowledge-access",
    )
    return knowledge_tier_access_payload(
        role=ctx.member_role,
        can_change_settings=ctx.can_change_settings,
        user_id=ctx.user_id,
        organization_id=ctx.organization_id,
        organization_name=ctx.organization_name,
        workspace_id=ctx.workspace_id,
        membership_status=ctx.membership_status,
    )


@router.post("/organizations/demo-onboarding")
async def organizations_demo_onboarding(request: Request):
    authz_class = classify_authorization_class("POST", "/organizations/demo-onboarding")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /organizations/demo-onboarding",
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    industry_code = body.get("industry_code")
    if not isinstance(industry_code, str) or not industry_code.strip():
        raise HTTPException(status_code=400, detail="industry_code required")
    try:
        attach_user_to_seeded_demo_organization(
            user_id=ctx.user_id,
            actor_user_id=ctx.user_id,
            industry_code=industry_code,
        )
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail == "demo organization not found for industry_code" else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    refreshed = request_context_from_request(request)
    return {
        "ok": True,
        "landing_route": "/chat/new",
        **_organization_payload_for_context(refreshed),
    }


@router.post("/organizations/real-company")
async def organizations_real_company(request: Request):
    authz_class = classify_authorization_class("POST", "/organizations/real-company")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /organizations/real-company",
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    display_name = body.get("company_display_name") or body.get("organization_name")
    if not isinstance(display_name, str) or not display_name.strip():
        raise HTTPException(status_code=400, detail="company_display_name required")
    organization_id = body.get("organization_id")
    industry_code = body.get("industry_code")
    local_currency = body.get("local_currency")
    fiscal_year_start_month = body.get("fiscal_year_start_month")
    try:
        fiscal_month = int(fiscal_year_start_month) if fiscal_year_start_month is not None else None
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="fiscal_year_start_month must be 1..12") from exc
    if fiscal_month is not None and not 1 <= fiscal_month <= 12:
        raise HTTPException(status_code=400, detail="fiscal_year_start_month must be 1..12")
    try:
        join_organization(
            user_id=ctx.user_id,
            actor_user_id=ctx.user_id,
            organization_id=organization_id if isinstance(organization_id, str) else None,
            organization_name=display_name,
            dataset_type="REAL",
            industry_code=industry_code if isinstance(industry_code, str) else None,
            fiscal_year_start_month=fiscal_month,
            local_currency=local_currency if isinstance(local_currency, str) else None,
            create=True,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = 409 if detail == "organization already exists" else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    refreshed = request_context_from_request(request)
    return {
        "ok": True,
        "landing_route": "/settings/data-connections?import=company-dataset",
        **_organization_payload_for_context(refreshed),
    }


@router.post("/organizations/join")
async def organizations_join(request: Request):
    authz_class = classify_authorization_class("POST", "/organizations/join")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /organizations/join",
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    organization_id = body.get("organization_id")
    organization_name = body.get("organization_name")
    create = bool(body.get("create"))
    try:
        join_organization(
            user_id=ctx.user_id,
            actor_user_id=ctx.user_id,
            organization_id=organization_id if isinstance(organization_id, str) else None,
            organization_name=organization_name if isinstance(organization_name, str) else None,
            create=create,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail == "organization not found" else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    refreshed = request_context_from_request(request)
    return _organization_payload_for_context(refreshed)


@router.post("/organizations/switch")
async def organizations_switch(request: Request):
    authz_class = classify_authorization_class("POST", "/organizations/switch")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /organizations/switch",
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    organization_id = body.get("organization_id")
    if not isinstance(organization_id, str) or not organization_id.strip():
        raise HTTPException(status_code=400, detail="organization_id required")
    try:
        switch_user_organization(
            user_id=ctx.user_id,
            organization_id=organization_id,
            actor_user_id=ctx.user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail == "membership not found" else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    refreshed = request_context_from_request(request)
    return _organization_payload_for_context(refreshed)



@router.post("/organizations/sharing")
async def organizations_sharing(request: Request):
    authz_class = classify_authorization_class("POST", "/organizations/sharing")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /organizations/sharing",
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    sharing_enabled = bool(body.get("sharing_enabled"))
    try:
        set_user_sharing_enabled(
            user_id=ctx.user_id,
            organization_id=ctx.organization_id,
            sharing_enabled=sharing_enabled,
            actor_user_id=ctx.user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    refreshed = request_context_from_request(request)
    return _organization_payload_for_context(refreshed)


@router.post("/organizations/approve")
async def organizations_approve(request: Request):
    authz_class = classify_authorization_class("POST", "/organizations/approve")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /organizations/approve",
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    target_user_id = body.get("user_id")
    if not isinstance(target_user_id, str) or not target_user_id.strip():
        raise HTTPException(status_code=400, detail="user_id required")
    try:
        update_membership_status(
            target_user_id=target_user_id,
            organization_id=str(ctx.organization_id),
            membership_status="active",
            member_role="member",
            actor_user_id=ctx.user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    refreshed = request_context_from_request(request)
    return _organization_payload_for_context(refreshed)


@router.post("/organizations/remove")
async def organizations_remove(request: Request):
    authz_class = classify_authorization_class("POST", "/organizations/remove")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /organizations/remove",
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    target_user_id = body.get("user_id")
    if not isinstance(target_user_id, str) or not target_user_id.strip():
        raise HTTPException(status_code=400, detail="user_id required")
    try:
        update_membership_status(
            target_user_id=target_user_id,
            organization_id=str(ctx.organization_id),
            membership_status="revoked",
            member_role="member",
            actor_user_id=ctx.user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    refreshed = request_context_from_request(request)
    return _organization_payload_for_context(refreshed)


@router.post("/organizations/member-role")
async def organizations_member_role(request: Request):
    ctx = _require_authenticated_context(request)
    if not ctx.organization_id:
        raise HTTPException(status_code=409, detail="No organization selected")
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    target_user_id = body.get("user_id")
    member_role = body.get("member_role")
    if not isinstance(target_user_id, str) or not target_user_id.strip():
        raise HTTPException(status_code=400, detail="user_id required")
    if not isinstance(member_role, str) or not member_role.strip():
        raise HTTPException(status_code=400, detail="member_role required")
    requested_role = member_role.strip().lower()
    if requested_role not in {"owner", "admin", "member", "governance_security_reviewer"}:
        raise HTTPException(status_code=400, detail="member_role must be owner/admin/member/governance_security_reviewer")

    _log_tenant_authz(
        ctx=ctx,
        authorization_class=RouteAuthorizationClass.TENANT_ADMIN,
        route_hint="POST /organizations/member-role",
    )

    membership = get_user_membership_for_organization(
        user_id=str(target_user_id),
        organization_id=str(ctx.organization_id),
    )
    if not isinstance(membership, dict):
        raise HTTPException(status_code=404, detail="membership not found")

    previous_role = str(membership.get("member_role") or "")

    try:
        update_membership_role(
            target_user_id=str(target_user_id),
            organization_id=str(ctx.organization_id),
            member_role=requested_role,
            actor_user_id=ctx.user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail == "membership not found" else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    refreshed = request_context_from_request(request)
    return _organization_payload_for_context(refreshed)


@router.post("/organizations/invite")
async def organizations_invite(request: Request):
    authz_class = classify_authorization_class("POST", "/organizations/invite")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /organizations/invite",
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    invitee = str(body.get("invitee") or body.get("email") or body.get("user_id") or "").strip()
    if not invitee:
        raise HTTPException(status_code=400, detail="invitee required")
    return {
        "ok": True,
        "organization_id": ctx.organization_id,
        "invitee": invitee,
        "status": "pending_manual_acceptance",
        "message": "Invitation placeholder recorded; invite tokens are not implemented in phase 1.",
    }

