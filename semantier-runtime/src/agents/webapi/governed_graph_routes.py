from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from agents.route_policy import classify_authorization_class
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from agents.webapi.governed_graph_service import GovernedGraphService

router = APIRouter(tags=["governed-graph"])

ROUTE_PROJECTION_RESOLVE = "/api/governed-graph/projections/resolve"
ROUTE_PROJECTION = "/api/governed-graph/projections/{projection_id}"
ROUTE_PROJECTION_RERESOLVE = "/api/governed-graph/projections/{projection_id}/resolve"
ROUTE_NODE_DETAIL = "/api/governed-graph/nodes/{node_id}"
ROUTE_ASSERTION_DETAIL = "/api/governed-graph/assertions/{assertion_id}"
ROUTE_ANCHOR_DETAIL = "/api/governed-graph/anchors/{anchor_id}"
ROUTE_SEARCH = "/api/governed-graph/search"
ROUTE_QUERY = "/api/governed-graph/query"
ROUTE_CONFLICT_COMPARE = "/api/governed-graph/conflicts/compare"
ROUTE_REPLAY = "/api/governed-graph/replay/{object_ref}"
ROUTE_EVIDENCE_EXPORTS = "/api/governed-graph/evidence-exports"
ROUTE_COMMANDS = "/api/governed-graph/commands"
ROUTE_INDEX_STATUS = "/api/governed-graph/index-status/{source_ref}"
ROUTE_INDEX_RUNS = "/api/governed-graph/index-runs"

_SERVICE = GovernedGraphService()


def _authz(request: Request, method: str, path: str):
    authz_class = classify_authorization_class(method, path)
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    return _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint=f"{method} {path}",
    )


async def _json_body(request: Request) -> dict[str, object]:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="JSON body required") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object body required")
    return body


def _handle_service_error(exc: Exception) -> None:
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise exc


@router.post(ROUTE_PROJECTION_RESOLVE)
async def resolve_governed_graph_projection(request: Request):
    ctx = _authz(request, "POST", ROUTE_PROJECTION_RESOLVE)
    body = await _json_body(request)
    try:
        return _SERVICE.resolve_projection(ctx=ctx, request=body)
    except Exception as exc:
        _handle_service_error(exc)


@router.get(ROUTE_PROJECTION)
async def get_governed_graph_projection(projection_id: str, request: Request):
    ctx = _authz(request, "GET", ROUTE_PROJECTION)
    if not projection_id:
        raise HTTPException(status_code=400, detail="projection_id required")
    try:
        return _SERVICE.get_projection(ctx=ctx, projection_id=projection_id)
    except Exception as exc:
        _handle_service_error(exc)


@router.post(ROUTE_PROJECTION_RERESOLVE)
async def reresolve_governed_graph_projection(projection_id: str, request: Request):
    ctx = _authz(request, "POST", ROUTE_PROJECTION_RERESOLVE)
    if not projection_id:
        raise HTTPException(status_code=400, detail="projection_id required")
    body = await _json_body(request)
    try:
        body.setdefault("projection_id", projection_id)
        return _SERVICE.resolve_projection(ctx=ctx, request=body)
    except Exception as exc:
        _handle_service_error(exc)


@router.get(ROUTE_NODE_DETAIL)
async def get_governed_graph_node(node_id: str, request: Request):
    ctx = _authz(request, "GET", ROUTE_NODE_DETAIL)
    if not node_id:
        raise HTTPException(status_code=400, detail="node_id required")
    try:
        return _SERVICE.get_node(ctx=ctx, node_id=node_id)
    except Exception as exc:
        _handle_service_error(exc)


@router.get(ROUTE_ASSERTION_DETAIL)
async def get_governed_graph_assertion(assertion_id: str, request: Request):
    ctx = _authz(request, "GET", ROUTE_ASSERTION_DETAIL)
    if not assertion_id:
        raise HTTPException(status_code=400, detail="assertion_id required")
    try:
        return _SERVICE.get_assertion(ctx=ctx, assertion_id=assertion_id)
    except Exception as exc:
        _handle_service_error(exc)


@router.get(ROUTE_ANCHOR_DETAIL)
async def get_governed_graph_anchor(anchor_id: str, request: Request):
    ctx = _authz(request, "GET", ROUTE_ANCHOR_DETAIL)
    if not anchor_id:
        raise HTTPException(status_code=400, detail="anchor_id required")
    try:
        return _SERVICE.get_anchor(ctx=ctx, anchor_id=anchor_id)
    except Exception as exc:
        _handle_service_error(exc)


@router.post(ROUTE_SEARCH)
async def search_governed_graph(request: Request):
    ctx = _authz(request, "POST", ROUTE_SEARCH)
    body = await _json_body(request)
    try:
        return {"ok": True, "results": _SERVICE.search(ctx=ctx, request=body)}
    except Exception as exc:
        _handle_service_error(exc)


@router.post(ROUTE_QUERY)
async def query_governed_graph(request: Request):
    ctx = _authz(request, "POST", ROUTE_QUERY)
    body = await _json_body(request)
    try:
        return _SERVICE.query(ctx=ctx, request=body)
    except Exception as exc:
        _handle_service_error(exc)


@router.post(ROUTE_CONFLICT_COMPARE)
async def compare_governed_graph_conflicts(request: Request):
    ctx = _authz(request, "POST", ROUTE_CONFLICT_COMPARE)
    body = await _json_body(request)
    try:
        return _SERVICE.compare_conflicts(ctx=ctx, request=body)
    except Exception as exc:
        _handle_service_error(exc)


@router.get(ROUTE_REPLAY)
async def get_governed_graph_replay(object_ref: str, request: Request):
    ctx = _authz(request, "GET", ROUTE_REPLAY)
    if not object_ref:
        raise HTTPException(status_code=400, detail="object_ref required")
    try:
        return _SERVICE.replay(ctx=ctx, object_ref=object_ref)
    except Exception as exc:
        _handle_service_error(exc)


@router.post(ROUTE_EVIDENCE_EXPORTS)
async def create_governed_graph_evidence_export(request: Request):
    ctx = _authz(request, "POST", ROUTE_EVIDENCE_EXPORTS)
    body = await _json_body(request)
    try:
        return _SERVICE.evidence_export(ctx=ctx, request=body)
    except Exception as exc:
        _handle_service_error(exc)


@router.post(ROUTE_COMMANDS)
async def create_governed_graph_command(request: Request):
    ctx = _authz(request, "POST", ROUTE_COMMANDS)
    body = await _json_body(request)
    try:
        return _SERVICE.command(ctx=ctx, request=body)
    except Exception as exc:
        _handle_service_error(exc)


@router.get(ROUTE_INDEX_STATUS)
async def get_governed_graph_index_status(source_ref: str, request: Request):
    ctx = _authz(request, "GET", ROUTE_INDEX_STATUS)
    if not source_ref:
        raise HTTPException(status_code=400, detail="source_ref required")
    try:
        return _SERVICE.index_status(ctx=ctx, source_ref=source_ref)
    except Exception as exc:
        _handle_service_error(exc)


@router.post(ROUTE_INDEX_RUNS)
async def create_governed_graph_index_run(request: Request):
    ctx = _authz(request, "POST", ROUTE_INDEX_RUNS)
    body = await _json_body(request)
    try:
        return _SERVICE.index_run(ctx=ctx, request=body)
    except Exception as exc:
        _handle_service_error(exc)
