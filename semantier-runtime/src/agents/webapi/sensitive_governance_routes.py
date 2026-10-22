from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from agents.route_policy import classify_authorization_class
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from agents.webapi.sensitive_governance_service import SensitiveGovernanceService
from semantier.sensitive_governance_contracts.requests import (
    CopySegmentRequest,
    ExecuteAssessmentRequest,
    ReplayRunRequest,
    RevealSegmentRequest,
    ReviewEscalationRequest,
    SensitiveGovernanceRequest,
)

router = APIRouter(tags=["webapi-gateway"])
_SERVICE = SensitiveGovernanceService()


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


@router.post("/api/workshop/finance/assessments")
@router.post("/api/sensitive-governance/assessments")
async def create_sensitive_governance_assessment(request: Request):
    route_path = request.url.path
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    if route_path == "/api/workshop/finance/assessments" and body.get("purpose") != "workshop_finance_annotations":
        raise HTTPException(400, "WORKSHOP_FINANCE_PURPOSE_REQUIRED")
    try:
        contract_request = SensitiveGovernanceRequest.from_payload(body)
        if contract_request.purpose == "workshop_finance_annotations":
            from agents.webapi.session_routes import (
                _resolve_workspace_runtime_session_id, _session_belongs_to_workspace,
            )
            session = str(body.get("session_id") or "")
            internal = _resolve_workspace_runtime_session_id(ctx, session)
            if not internal or not _session_belongs_to_workspace(internal, ctx):
                raise HTTPException(status_code=404, detail="Session not found")
        assessment = _SERVICE.create_assessment(
            ctx=ctx,
            request=contract_request,
            session_id=str(body.get("session_id") or "").strip() or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if contract_request.purpose == "workshop_finance_annotations":
        import json
        from agents.webapi.session_routes import _workspace_session_db
        db = _workspace_session_db(ctx)
        try:
            db.append_messages_batch(internal, [
                {"role": "user", "content": str(body.get("message") or contract_request.purpose)},
                {"role": "assistant", "content": json.dumps(
                    assessment["prepared_payload"]["finance_teaching_result"], ensure_ascii=False)},
            ])
        finally:
            db.close()
    return {"ok": True, "assessment": assessment}


@router.post("/api/sensitive-governance/assessments/{assessment_id}/execute")
async def execute_sensitive_governance_assessment(assessment_id: str, request: Request):
    route_path = "/api/sensitive-governance/assessments/{assessment_id}/execute"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        contract_request = ExecuteAssessmentRequest.from_payload(body)
        run = _SERVICE.execute_assessment(
            ctx=ctx,
            assessment_id=assessment_id,
            request=contract_request,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="assessment not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "run": run}


@router.get("/api/sensitive-governance/runs/{run_id}")
async def get_sensitive_governance_run(run_id: str, request: Request):
    route_path = "/api/sensitive-governance/runs/{run_id}"
    ctx = _authz(request, "GET", route_path)
    run = _SERVICE.get_run(ctx=ctx, run_id=run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"ok": True, "run": run}


@router.post("/api/sensitive-governance/responses/{response_id}/segments/{segment_id}/reveal")
async def reveal_sensitive_governance_segment(
    response_id: str,
    segment_id: str,
    request: Request,
):
    route_path = "/api/sensitive-governance/responses/{response_id}/segments/{segment_id}/reveal"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        contract_request = RevealSegmentRequest.from_payload(body)
        decision = _SERVICE.reveal_segment(
            ctx=ctx,
            response_id=response_id,
            segment_id=segment_id,
            request=contract_request,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "decision": decision}


@router.post("/api/sensitive-governance/responses/{response_id}/segments/{segment_id}/copy")
async def copy_sensitive_governance_segment(
    response_id: str,
    segment_id: str,
    request: Request,
):
    route_path = "/api/sensitive-governance/responses/{response_id}/segments/{segment_id}/copy"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        contract_request = CopySegmentRequest.from_payload(body)
        decision = _SERVICE.copy_segment(
            ctx=ctx,
            response_id=response_id,
            segment_id=segment_id,
            request=contract_request,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "decision": decision}


@router.post("/api/sensitive-governance/escalations/{escalation_id}/review")
async def review_sensitive_governance_escalation(escalation_id: str, request: Request):
    route_path = "/api/sensitive-governance/escalations/{escalation_id}/review"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        contract_request = ReviewEscalationRequest.from_payload(body)
        review = _SERVICE.review_escalation(
            ctx=ctx,
            escalation_id=escalation_id,
            request=contract_request,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "review": review}


@router.post("/api/sensitive-governance/runs/{run_id}/replay")
async def replay_sensitive_governance_run(run_id: str, request: Request):
    route_path = "/api/sensitive-governance/runs/{run_id}/replay"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        contract_request = ReplayRunRequest.from_payload(body)
        replay = _SERVICE.replay_run(ctx=ctx, run_id=run_id, request=contract_request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "replay": replay}
