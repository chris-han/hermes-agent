from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from agents.route_policy import classify_authorization_class
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from semantier.architecture_evolution_monitor.service import (
    ArchitectureEvolutionMonitorService,
)
from semantier.architecture_evolution_monitor.store import ReviewConflictError

router = APIRouter(tags=["webapi-gateway"])
_SERVICE = ArchitectureEvolutionMonitorService()


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


@router.post("/api/architecture-monitor/evaluations")
async def evaluate_architecture_metrics(request: Request):
    route_path = "/api/architecture-monitor/evaluations"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    organization_id = str(ctx.organization_id or "").strip()
    if not organization_id:
        raise HTTPException(status_code=409, detail="No organization selected")
    body["organization_id"] = organization_id
    try:
        result = _SERVICE.evaluate_snapshot(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, **result}


@router.post("/api/architecture-monitor/scheduled-evaluations")
async def scheduled_architecture_metrics_evaluation(request: Request):
    route_path = "/api/architecture-monitor/scheduled-evaluations"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    organization_id = str(ctx.organization_id or "").strip()
    if not organization_id:
        raise HTTPException(status_code=409, detail="No organization selected")
    body["organization_id"] = organization_id
    try:
        result = _SERVICE.scheduled_evaluate_snapshot(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, **result}


@router.post("/api/architecture-monitor/schedules")
async def create_architecture_monitor_schedule(request: Request):
    route_path = "/api/architecture-monitor/schedules"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        cron_job = _SERVICE.create_schedule(
            scope_id=str(body.get("scope_id") or ctx.organization_id or "platform"),
            schedule_ref=str(body.get("schedule_ref") or ""),
            organization_id=str(ctx.organization_id or ""),
            environment=str(body.get("environment") or "production"),
            architecture_profile=str(body.get("architecture_profile") or "legal-authority-relational"),
            cron_expression=str(body.get("cron_expression") or ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "cron_job": cron_job}


@router.get("/api/architecture-monitor/schedules")
async def list_architecture_monitor_schedules(request: Request):
    route_path = "/api/architecture-monitor/schedules"
    ctx = _authz(request, "GET", route_path)
    return {
        "ok": True,
        "schedules": _SERVICE.list_schedules(str(ctx.organization_id or "platform")),
    }


@router.get("/api/architecture-monitor/reviews")
async def list_architecture_reviews(request: Request):
    route_path = "/api/architecture-monitor/reviews"
    ctx = _authz(request, "GET", route_path)
    organization_id = str(ctx.organization_id or "").strip()
    return {"ok": True, "reviews": _SERVICE.list_reviews(organization_id)}


@router.get("/api/architecture-monitor/reviews/{review_id}")
async def get_architecture_review(review_id: str, request: Request):
    route_path = "/api/architecture-monitor/reviews/{review_id}"
    ctx = _authz(request, "GET", route_path)
    organization_id = str(ctx.organization_id or "").strip()
    review = _SERVICE.get_review(organization_id, review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="review not found")
    return {"ok": True, "review": review}


@router.post("/api/architecture-monitor/reviews/{review_id}/decision")
async def decide_architecture_review(review_id: str, request: Request):
    route_path = "/api/architecture-monitor/reviews/{review_id}/decision"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        decision = _SERVICE.decide_review(
            organization_id=str(ctx.organization_id or ""),
            review_id=review_id,
            expected_version=int(body.get("expected_version")),
            expected_hash=str(body.get("expected_hash") or ""),
            to_state=str(body.get("to_state") or ""),
            idempotency_key=str(body.get("idempotency_key") or ""),
            actor_ref=str(ctx.user_id or ""),
            reason_code=str(body.get("reason_code") or ""),
            payload=dict(body.get("payload") or {}),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="review not found") from exc
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "decision": decision}


@router.post("/api/architecture-monitor/reviews/{review_id}/reopen")
async def reopen_architecture_review(review_id: str, request: Request):
    route_path = "/api/architecture-monitor/reviews/{review_id}/reopen"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        decision = _SERVICE.decide_review(
            organization_id=str(ctx.organization_id or ""),
            review_id=review_id,
            expected_version=int(body.get("expected_version")),
            expected_hash=str(body.get("expected_hash") or ""),
            to_state="OPEN",
            idempotency_key=str(body.get("idempotency_key") or ""),
            actor_ref=str(ctx.user_id or ""),
            reason_code=str(body.get("reason_code") or "REOPEN"),
            payload=dict(body.get("payload") or {}),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="review not found") from exc
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "decision": decision}


@router.post("/api/architecture-monitor/reviews/{review_id}/handoff")
async def request_architecture_handoff(review_id: str, request: Request):
    route_path = "/api/architecture-monitor/reviews/{review_id}/handoff"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    payload = dict(body.get("payload") or {})
    payload["handoff_requested"] = True
    try:
        result = _SERVICE.request_handoff(
            organization_id=str(ctx.organization_id or ""),
            review_id=review_id,
            expected_version=int(body.get("expected_version")),
            expected_hash=str(body.get("expected_hash") or ""),
            to_state=str(body.get("to_state") or "CODEX_ANALYSIS_REQUESTED"),
            idempotency_key=str(body.get("idempotency_key") or ""),
            actor_ref=str(ctx.user_id or ""),
            reason_code=str(body.get("reason_code") or "HANDOFF_REQUESTED"),
            payload=payload,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="review not found") from exc
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, **result}


@router.post("/api/architecture-monitor/reviews/{review_id}/handoff/revoke")
async def revoke_architecture_handoff(review_id: str, request: Request):
    route_path = "/api/architecture-monitor/reviews/{review_id}/handoff/revoke"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    payload = dict(body.get("payload") or {})
    payload["handoff_revoked"] = True
    try:
        result = _SERVICE.revoke_handoff(
            organization_id=str(ctx.organization_id or ""),
            review_id=review_id,
            expected_version=int(body.get("expected_version")),
            expected_hash=str(body.get("expected_hash") or ""),
            idempotency_key=str(body.get("idempotency_key") or ""),
            actor_ref=str(ctx.user_id or ""),
            reason_code=str(body.get("reason_code") or "HANDOFF_REVOKED"),
            payload=payload,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="review not found") from exc
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, **result}


@router.post("/api/architecture-monitor/reviews/{review_id}/close")
async def close_architecture_review(review_id: str, request: Request):
    route_path = "/api/architecture-monitor/reviews/{review_id}/close"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        decision = _SERVICE.decide_review(
            organization_id=str(ctx.organization_id or ""),
            review_id=review_id,
            expected_version=int(body.get("expected_version")),
            expected_hash=str(body.get("expected_hash") or ""),
            to_state="CLOSED",
            idempotency_key=str(body.get("idempotency_key") or ""),
            actor_ref=str(ctx.user_id or ""),
            reason_code=str(body.get("reason_code") or "CLOSED"),
            payload=dict(body.get("payload") or {}),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="review not found") from exc
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "decision": decision}


@router.get("/api/architecture-monitor/dashboard")
async def get_architecture_monitor_dashboard(request: Request):
    route_path = "/api/architecture-monitor/dashboard"
    ctx = _authz(request, "GET", route_path)
    return {"ok": True, "dashboard": _SERVICE.dashboard(str(ctx.organization_id or ""))}


@router.get("/api/architecture-monitor/inspector/reviews/{review_id}")
async def inspect_architecture_review(review_id: str, request: Request):
    route_path = "/api/architecture-monitor/inspector/reviews/{review_id}"
    ctx = _authz(request, "GET", route_path)
    try:
        inspection = _SERVICE.inspect_review(str(ctx.organization_id or ""), review_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="review not found") from exc
    return {"ok": True, "inspection": inspection}
