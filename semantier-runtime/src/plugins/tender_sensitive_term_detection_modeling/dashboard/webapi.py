from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from agents.webapi.deps import require_route_authorization

from . import plugin_api

router = APIRouter(tags=["tender-sensitive-term-detection-modeling"])

ROUTE_POLICY_MAP = {
    (route["method"], route["path"]): "authenticated"
    for route in plugin_api.ROUTES
}
ROUTE_AUTHZ_CLASS_MAP = {
    (route["method"], route["path"]): (
        "tenant-admin"
        if route["authorization"] in {"governed_reviewer_capability", "governed_activation_capability"}
        else "tenant-member"
    )
    for route in plugin_api.ROUTES
}


def _ctx(request: Request, method: str, path: str):
    from agents.route_policy import RouteAuthorizationClass

    authz_class = RouteAuthorizationClass(ROUTE_AUTHZ_CLASS_MAP[(method, path)])
    return require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint=f"{method} {path}",
    )


async def _body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@router.get("/api/plugins/tender-sensitive-term-detection-modeling/v1/capabilities")
async def capabilities(request: Request):
    _ctx(request, "GET", "/api/plugins/tender-sensitive-term-detection-modeling/v1/capabilities")
    return plugin_api.capabilities()


@router.get("/api/plugins/tender-sensitive-term-detection-modeling/v1/compilation-runs/{run_id}")
async def get_compilation_run(run_id: str, request: Request):
    ctx = _ctx(
        request,
        "GET",
        "/api/plugins/tender-sensitive-term-detection-modeling/v1/compilation-runs/{run_id}",
    )
    return plugin_api.get_compilation_run(ctx=ctx, run_id=run_id)


@router.get("/api/plugins/tender-sensitive-term-detection-modeling/v1/candidates")
async def list_candidates(request: Request):
    ctx = _ctx(request, "GET", "/api/plugins/tender-sensitive-term-detection-modeling/v1/candidates")
    return plugin_api.list_candidates(ctx=ctx, filters=dict(request.query_params))


@router.post("/api/plugins/tender-sensitive-term-detection-modeling/v1/candidates/{candidate_id}/review")
async def review_candidate(candidate_id: str, request: Request):
    ctx = _ctx(
        request,
        "POST",
        "/api/plugins/tender-sensitive-term-detection-modeling/v1/candidates/{candidate_id}/review",
    )
    return plugin_api.review_candidate(
        ctx=ctx,
        candidate_id=candidate_id,
        payload=await _body(request),
    )


@router.post("/api/plugins/tender-sensitive-term-detection-modeling/v1/detections")
async def create_detection(request: Request):
    ctx = _ctx(request, "POST", "/api/plugins/tender-sensitive-term-detection-modeling/v1/detections")
    return plugin_api.create_detection(ctx=ctx, payload=await _body(request))


@router.get("/api/plugins/tender-sensitive-term-detection-modeling/v1/detections/{run_id}")
async def get_detection(run_id: str, request: Request):
    ctx = _ctx(
        request,
        "GET",
        "/api/plugins/tender-sensitive-term-detection-modeling/v1/detections/{run_id}",
    )
    return plugin_api.get_detection(ctx=ctx, run_id=run_id)


@router.post("/api/plugins/tender-sensitive-term-detection-modeling/v1/reports")
async def create_report(request: Request):
    _ctx(request, "POST", "/api/plugins/tender-sensitive-term-detection-modeling/v1/reports")
    return plugin_api.create_report(await _body(request))


@router.get("/api/plugins/tender-sensitive-term-detection-modeling/v1/reports/{report_id}")
async def get_report(report_id: str, request: Request):
    ctx = _ctx(
        request,
        "GET",
        "/api/plugins/tender-sensitive-term-detection-modeling/v1/reports/{report_id}",
    )
    return plugin_api.get_report(ctx=ctx, report_id=report_id)


@router.post("/api/plugins/tender-sensitive-term-detection-modeling/v1/labeled-docx")
async def create_labeled_docx(request: Request):
    ctx = _ctx(request, "POST", "/api/plugins/tender-sensitive-term-detection-modeling/v1/labeled-docx")
    return plugin_api.create_labeled_docx(ctx=ctx, payload=await _body(request))


@router.get("/api/plugins/tender-sensitive-term-detection-modeling/v1/labeled-docx/{artifact_id}")
async def get_labeled_docx(artifact_id: str, request: Request):
    ctx = _ctx(
        request,
        "GET",
        "/api/plugins/tender-sensitive-term-detection-modeling/v1/labeled-docx/{artifact_id}",
    )
    return plugin_api.get_labeled_docx(ctx=ctx, artifact_id=artifact_id)


@router.post("/api/plugins/tender-sensitive-term-detection-modeling/v1/replay/{replay_binding_id}/verify")
async def verify_replay(replay_binding_id: str, request: Request):
    ctx = _ctx(
        request,
        "POST",
        "/api/plugins/tender-sensitive-term-detection-modeling/v1/replay/{replay_binding_id}/verify",
    )
    return plugin_api.verify_replay(ctx=ctx, replay_binding_id=replay_binding_id)

