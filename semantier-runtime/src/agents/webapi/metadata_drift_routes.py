from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from agents.route_policy import classify_authorization_class
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from semantier.institutional_drift_monitor.service import InstitutionalDriftService

router = APIRouter(tags=["webapi-gateway"])
_SERVICE = InstitutionalDriftService()


def _authz(request: Request, method: str, path: str):
    authz_class = classify_authorization_class(method, path)
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    return _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint=f"{method} {path}",
    )


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return body


def _organization_id(ctx) -> str:
    organization_id = str(ctx.organization_id or "").strip()
    if not organization_id:
        raise HTTPException(status_code=409, detail="No organization selected")
    return organization_id


@router.post("/api/metadata-drift/policies")
async def create_metadata_drift_policy(request: Request):
    route_path = "/api/metadata-drift/policies"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    body["organization_id"] = _organization_id(ctx)
    body["created_by"] = str(ctx.user_id or body.get("created_by") or "unknown")
    try:
        policy = _SERVICE.register_policy(body)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "policy": policy}


@router.post("/api/metadata-drift/policies/{policy_id}/certify")
async def certify_metadata_drift_policy(policy_id: str, request: Request):
    route_path = "/api/metadata-drift/policies/{policy_id}/certify"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        event = _SERVICE.certify_policy(
            {
                "policy_id": policy_id,
                "reviewer_ref": str(ctx.user_id or ""),
                "reviewer_role": str(body.get("reviewer_role") or "governance_reviewer"),
                "evidence_hash": str(body.get("evidence_hash") or ""),
            }
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="policy not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "event": event}


@router.post("/api/metadata-drift/policies/{policy_id}/activate")
async def activate_metadata_drift_policy(policy_id: str, request: Request):
    route_path = "/api/metadata-drift/policies/{policy_id}/activate"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        event = _SERVICE.activate_policy(
            {
                "policy_id": policy_id,
                "approver_ref": str(ctx.user_id or ""),
                "approver_role": str(body.get("approver_role") or "governance_reviewer"),
                "certification_event_id": str(body.get("certification_event_id") or ""),
            }
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="policy not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "event": event}


@router.post("/api/metadata-drift/baselines")
async def create_metadata_drift_baseline(request: Request):
    route_path = "/api/metadata-drift/baselines"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    body["organization_id"] = _organization_id(ctx)
    try:
        baseline = _SERVICE.create_baseline(body)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "baseline": baseline}


@router.post("/api/metadata-drift/observations")
async def observe_metadata_drift(request: Request):
    route_path = "/api/metadata-drift/observations"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    body["organization_id"] = _organization_id(ctx)
    try:
        result = _SERVICE.observe_and_classify(body)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result.get("ok", False):
        raise HTTPException(status_code=409, detail=result.get("error_code"))
    return result


@router.post("/api/metadata-drift/recovery-verifications")
async def verify_metadata_drift_recovery(request: Request):
    route_path = "/api/metadata-drift/recovery-verifications"
    _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        verification = _SERVICE.verify_recovery(body)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "verification": verification}


@router.get("/api/metadata-drift/findings/{finding_id}/audit")
async def get_metadata_drift_audit(finding_id: str, request: Request):
    route_path = "/api/metadata-drift/findings/{finding_id}/audit"
    _authz(request, "GET", route_path)
    try:
        audit = _SERVICE.audit_export(finding_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="finding not found") from exc
    return {"ok": True, "audit": audit}


@router.get("/api/metadata-drift/dashboard")
async def get_metadata_drift_dashboard(request: Request):
    route_path = "/api/metadata-drift/dashboard"
    ctx = _authz(request, "GET", route_path)
    return {"ok": True, "dashboard": _SERVICE.dashboard(_organization_id(ctx))}
