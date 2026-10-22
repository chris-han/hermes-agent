"""Authenticated, read-only Phase1-001 ContextGraph browser projection."""

from __future__ import annotations

from collections import deque

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from agents.webapi.deps import require_authenticated_context
from semantier.semantica_context_graph.artifact_repository import (
    ArtifactNotFound,
    ContextGraphArtifactRepository,
)
from semantier.knowledge_feature_packs.semantica_adapter import (
    _ensure_vendored_semantica_importable,
)
from semantier.semantica_context_graph.workflow_context import browser_source_anchor
from semantier.semantica_context_graph.bridge import TenderExpertBridge, TenderExpertRequest, TenderExpertResult


router = APIRouter(tags=["contextgraph-runtime"])
ROUTE_CONTEXTGRAPH_RUNTIME = "/api/contextgraph/runtime"
ROUTE_GRAPH_NODES = "/api/graph/nodes"
ROUTE_GRAPH_EDGES = "/api/graph/edges"
ROUTE_TEMPORAL_BOUNDS = "/api/temporal/bounds"
ROUTE_GRAPH_SEARCH = "/api/graph/search"
ROUTE_GRAPH_PATH = "/api/graph/path"
ROUTE_PROVENANCE_REPORT = "/api/provenance/report"
ROUTE_RUNTIME_EXPERT = "/api/contextgraph/runtime-expert"


def _evidence_refs(properties: dict[str, object]) -> list[str]:
    raw = properties.get("evidence_refs") or properties.get("evidenceRefs") or []
    if not isinstance(raw, (list, tuple)):
        return []
    refs: list[str] = []
    for item in raw:
        value = item if isinstance(item, str) else item.get("evidence_ref") or item.get("evidenceRef") if isinstance(item, dict) else None
        if isinstance(value, str) and value.strip():
            refs.append(value.strip())
    return refs


def _evidence_ref_payloads(properties: dict[str, object]) -> list[dict[str, object]]:
    raw = properties.get("evidence_refs") or properties.get("evidenceRefs") or []
    if not isinstance(raw, (list, tuple)):
        return []
    return [
        item for item in raw
        if isinstance(item, dict) and (item.get("evidence_ref") or item.get("evidenceRef"))
    ]


def _source_anchors(properties: dict[str, object]) -> list[dict[str, object]]:
    raw = properties.get("source_anchors") or properties.get("sourceAnchors") or []
    if not isinstance(raw, (list, tuple)):
        return []
    return [browser_source_anchor(item) for item in raw if isinstance(item, dict)]


def _lineage(properties: dict[str, object], candidate: object) -> dict[str, object]:
    anchors = properties.get("source_anchors") or properties.get("sourceAnchors") or []
    source_refs = {
        str(anchor.get("sourceRef") or anchor.get("source_ref") or anchor.get("anchor_id"))
        for anchor in anchors if isinstance(anchor, dict)
        and (anchor.get("sourceRef") or anchor.get("source_ref") or anchor.get("anchor_id"))
    }
    evidence_details = _evidence_ref_payloads(properties)
    for detail in evidence_details:
        source_ref = detail.get("sourceIdentityRef") or detail.get("source_identity_ref")
        if source_ref:
            source_refs.add(str(source_ref))
    extraction_run_ref = properties.get("extraction_run_id") or properties.get("extractionRunRef")
    for detail in evidence_details:
        provenance = detail.get("provenance")
        if isinstance(provenance, dict) and provenance.get("extractionRunId"):
            extraction_run_ref = extraction_run_ref or provenance["extractionRunId"]
    return {
        "sourceIdentityRefs": sorted(source_refs),
        "extractionRunRef": extraction_run_ref,
        "candidateGraphId": candidate.artifact_id if candidate.authority_state == "candidate" else candidate.parent_artifact_id,
        "acceptedReleaseId": candidate.artifact_id if candidate.authority_state == "authoritative" else None,
        "acceptedReleaseVersion": candidate.graph_version if candidate.authority_state == "authoritative" else None,
    }


def _graph_evidence_refs(graph: object) -> list[str]:
    refs: list[str] = []
    for node in graph.nodes.values():
        properties = getattr(node, "properties", {})
        if isinstance(properties, dict):
            refs.extend(_evidence_refs(properties))
    for edge in graph.edges:
        properties = getattr(edge, "metadata", {})
        if isinstance(properties, dict):
            refs.extend(_evidence_refs(properties))
    return sorted(set(refs))
def _node_payload(node_id: str, node: object, candidate: object) -> dict[str, object]:
    properties = getattr(node, "properties", {})
    if not isinstance(properties, dict):
        properties = {}
    content = str(getattr(node, "content", "") or node_id)
    node_type = str(getattr(node, "node_type", "concept"))
    return {
        "id": node_id,
        "content": content,
        "type": node_type,
        "properties": properties,
        "evidenceRefs": _evidence_refs(properties),
        "evidenceRefDetails": _evidence_ref_payloads(properties),
        "groundingState": str(properties.get("grounding_state") or ("accepted" if candidate.authority_state == "authoritative" else "pending")),
        "lineage": _lineage(properties, candidate),
        "sourceAnchors": _source_anchors(properties),
    }


def _requested_graph_identity(request: Request) -> tuple[str, str | None]:
    query_params = getattr(request, "query_params", {})
    graph_ref = str(query_params.get("graph_ref") or query_params.get("graphRef") or "").strip()
    graph_version = str(query_params.get("graph_version") or query_params.get("graphVersion") or "").strip() or None
    accepted_release_id = str(query_params.get("accepted_release_id") or query_params.get("acceptedReleaseId") or "").strip()
    if accepted_release_id:
        if graph_ref and graph_ref != accepted_release_id:
            raise HTTPException(status_code=400, detail="graph_ref and accepted_release_id must identify the same release")
        graph_ref = accepted_release_id
    if not graph_ref:
        raise HTTPException(status_code=400, detail="graph_ref and graph_version, or accepted_release_id, are required")
    if not accepted_release_id and not graph_version:
        raise HTTPException(status_code=400, detail="graph_version is required when graph_ref is provided")
    return graph_ref, graph_version


def _load_runtime_graph(request: Request) -> tuple[object, object]:
    ctx = require_authenticated_context(request)
    organization_id = str(ctx.organization_id or "").strip()
    if not organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")
    # Artifact loading reconstructs a Semantica ContextGraph from JSON. Ensure
    # the exact clean vendor checkout is importable even when this request is
    # the first Semantica operation in a fresh gateway process.
    _ensure_vendored_semantica_importable()
    repository = ContextGraphArtifactRepository(
        ctx.workspace_root,
        organization_id=organization_id,
        workspace_id=ctx.workspace_id,
    )
    graph_ref, graph_version = _requested_graph_identity(request)
    try:
        candidate = repository.get(graph_ref)
    except ArtifactNotFound as exc:
        raise HTTPException(status_code=404, detail="ContextGraph artifact not found in authenticated workspace") from exc
    if candidate.graph_id != "tender-sensitive" or (graph_version and candidate.graph_version != graph_version):
        raise HTTPException(status_code=404, detail="ContextGraph identity does not match the requested release")
    query_params = getattr(request, "query_params", {})
    accepted_release_id = str(query_params.get("accepted_release_id") or query_params.get("acceptedReleaseId") or "").strip()
    if accepted_release_id and candidate.authority_state != "authoritative":
        raise HTTPException(status_code=404, detail="Requested identity is not an accepted graph release")
    try:
        graph = repository.load_graph(candidate.artifact_id)
    except ArtifactNotFound as exc:
        raise HTTPException(status_code=404, detail="ContextGraph artifact not found") from exc
    return candidate, graph


def _edge_payload(index: int, edge: object, candidate: object) -> dict[str, object]:
    if isinstance(edge, dict):
        properties = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
        return {
            "id": str(edge.get("edge_id") or edge.get("id") or index),
            "familyId": str(edge.get("family_id") or edge.get("edge_id") or edge.get("id") or index),
            "source": str(edge.get("source") or edge.get("source_id") or ""),
            "target": str(edge.get("target") or edge.get("target_id") or ""),
            "type": str(edge.get("type") or edge.get("edge_type") or "related_to"),
            "weight": float(edge.get("weight") or 1.0),
            "properties": properties,
            "sourceAnchors": _source_anchors(properties),
            "evidenceRefs": _evidence_refs(properties),
            "evidenceRefDetails": _evidence_ref_payloads(properties),
            "groundingState": str(properties.get("grounding_state") or ("accepted" if candidate.authority_state == "authoritative" else "pending")),
            "lineage": _lineage(properties, candidate),
        }
    properties = getattr(edge, "metadata", {}) if isinstance(getattr(edge, "metadata", {}), dict) else {}
    return {
        "id": str(getattr(edge, "edge_id", index)),
        "familyId": str(getattr(edge, "edge_id", index)),
        "source": str(getattr(edge, "source_id", "")),
        "target": str(getattr(edge, "target_id", "")),
        "type": str(getattr(edge, "edge_type", "related_to")),
        "weight": float(getattr(edge, "weight", 1.0) or 1.0),
        "properties": properties,
        "sourceAnchors": _source_anchors(properties),
        "evidenceRefs": _evidence_refs(properties),
        "evidenceRefDetails": _evidence_ref_payloads(properties),
        "groundingState": str(properties.get("grounding_state") or ("accepted" if candidate.authority_state == "authoritative" else "pending")),
        "lineage": _lineage(properties, candidate),
    }


@router.get(ROUTE_CONTEXTGRAPH_RUNTIME)
async def get_contextgraph_runtime(request: Request) -> dict[str, object]:
    candidate, graph = _load_runtime_graph(request)
    nodes = [
        _node_payload(str(node_id), node, candidate)
        for node_id, node in graph.nodes.items()
    ]
    edges: list[dict[str, object]] = []
    for index, edge in enumerate(graph.edges):
        edges.append(_edge_payload(index, edge, candidate))
    return {
        "schemaVersion": "semantier.contextgraph.browser_projection.v1",
        "artifactId": candidate.artifact_id,
        "graphRef": candidate.artifact_id,
        "graphVersion": candidate.graph_version,
        "graphHash": candidate.content_hash,
        "authorityState": candidate.authority_state,
        "candidateGraphId": candidate.artifact_id if candidate.authority_state == "candidate" else candidate.parent_artifact_id,
        "acceptedReleaseId": candidate.artifact_id if candidate.authority_state == "authoritative" else None,
        "acceptedReleaseVersion": candidate.graph_version if candidate.authority_state == "authoritative" else None,
        "runMode": candidate.run_mode,
        "semanticaCommit": candidate.semantica_commit,
        "nodes": nodes,
        "edges": edges,
        "sourceEvidenceRefs": _graph_evidence_refs(graph),
        "workflow": {
            "workflowId": "phase1-001",
            "phasePlanId": "phase1-001",
            "stage": "BUILD",
            "stageState": "server_evidence_required",
            "activeProfileId": "tender-runtime-expert",
            "graphRef": candidate.artifact_id,
            "graphVersion": candidate.graph_version,
            "graphHash": candidate.content_hash,
            "authorityState": candidate.authority_state,
            "runMode": candidate.run_mode if candidate.run_mode in {"evaluation_baseline", "authoritative"} else "evaluation_baseline",
            "groundingManifestRef": candidate.grounding_manifest_hash,
            "heldoutManifestRef": candidate.evaluation_manifest_hash,
            "v0RunRef": None,
            "v1RunRef": None,
            "learningDecision": None,
            "graphConstructionCertification": "not_certified",
            "reasoningCertification": "not_certified",
            "governanceStatus": "not_admitted",
        },
    }


@router.get(ROUTE_GRAPH_NODES)
async def get_graph_nodes(
    request: Request,
    limit: int = Query(default=1000, ge=1, le=1000),
    cursor: int = Query(default=0, ge=0),
) -> dict[str, object]:
    candidate, graph = _load_runtime_graph(request)
    nodes = [_node_payload(str(node_id), node, candidate) for node_id, node in graph.nodes.items()]
    page = nodes[cursor : cursor + limit]
    return {"nodes": page, "total": len(nodes), "skip": cursor, "limit": limit, "next_cursor": cursor + limit if cursor + limit < len(nodes) else None}


@router.get(ROUTE_GRAPH_EDGES)
async def get_graph_edges(
    request: Request,
    limit: int = Query(default=1000, ge=1, le=1000),
    cursor: int = Query(default=0, ge=0),
) -> dict[str, object]:
    candidate, graph = _load_runtime_graph(request)
    edges = [_edge_payload(index, edge, candidate) for index, edge in enumerate(graph.edges)]
    page = edges[cursor : cursor + limit]
    return {"edges": page, "total": len(edges), "skip": cursor, "limit": limit, "next_cursor": cursor + limit if cursor + limit < len(edges) else None}


@router.get(ROUTE_TEMPORAL_BOUNDS)
async def get_temporal_bounds(request: Request) -> dict[str, None]:
    _load_runtime_graph(request)
    return {"min": None, "max": None}


@router.post(ROUTE_GRAPH_SEARCH)
async def search_graph(request: Request) -> dict[str, object]:
    candidate, graph = _load_runtime_graph(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    query = str(body.get("query") or "").casefold() if isinstance(body, dict) else ""
    limit = min(max(int(body.get("limit", 20)), 1), 50) if isinstance(body, dict) else 20
    results = []
    for node_id, node in graph.nodes.items():
        content = str(getattr(node, "content", "") or "")
        node_type = str(getattr(node, "node_type", "") or "")
        properties = getattr(node, "properties", {})
        haystack = f"{content} {node_type} {properties}".casefold()
        if query and query not in haystack:
            continue
        results.append(
            {
                "node": {
                    "id": str(node_id),
                    "type": node_type,
                    "content": content,
                    "properties": properties if isinstance(properties, dict) else {},
                },
                "score": 1.0 if query and query in content.casefold() else 0.5,
            }
        )
    return {"results": results[:limit]}


@router.get(ROUTE_GRAPH_PATH)
async def get_graph_path(
    request: Request,
    source: str = Query(..., min_length=1),
    target: str = Query(..., min_length=1),
) -> dict[str, object]:
    candidate, graph = _load_runtime_graph(request)
    if source not in graph.nodes or target not in graph.nodes:
        raise HTTPException(status_code=404, detail="path endpoint is not present in pinned graph")
    queue = deque([source])
    parents: dict[str, tuple[str, object] | None] = {source: None}
    while queue and target not in parents:
        current = queue.popleft()
        for edge in graph.edges:
            neighbor = (
                edge.target_id if edge.source_id == current
                else edge.source_id if edge.target_id == current
                else None
            )
            if neighbor is not None and neighbor not in parents:
                parents[neighbor] = (current, edge)
                queue.append(neighbor)
    if target not in parents:
        return {"path": [], "edge_ids": [], "total_weight": 0, "hop_count": 0, "distance_band": "distant"}
    path = [target]
    edge_ids: list[str] = []
    current = target
    while parents[current] is not None:
        parent, edge = parents[current]
        path.append(parent)
        edge_ids.append(str(getattr(edge, "edge_id", "")))
        current = parent
    path.reverse()
    edge_ids.reverse()
    hops = len(edge_ids)
    return {
        "path": path,
        "edge_ids": edge_ids,
        "total_weight": float(hops),
        "hop_count": hops,
        "distance_band": "direct" if hops <= 1 else "near" if hops <= 3 else "mid-range",
    }


@router.get(ROUTE_PROVENANCE_REPORT)
async def get_provenance_report(
    request: Request,
    node_id: str = Query(..., min_length=1),
    format: str = Query(default="json", pattern="^(json|markdown)$"),
) -> Response:
    candidate, graph = _load_runtime_graph(request)
    node = graph.nodes.get(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="node is not present in pinned graph")
    payload = _node_payload(node_id, node, candidate)
    if format == "markdown":
        anchors = payload["sourceAnchors"]
        text = f"# Provenance: {payload['label']}\n\n- Node: `{node_id}`\n- Type: `{payload['nodeType']}`\n\n## Source anchors\n"
        text += "\n".join(f"- `{anchor.get('sourceRef')}` · `{anchor.get('locator')}`" for anchor in anchors)
        return Response(content=text + "\n", media_type="text/markdown")
    return JSONResponse(content={"node": payload, "sourceAnchors": payload["sourceAnchors"]})


@router.post(ROUTE_RUNTIME_EXPERT)
async def runtime_expert(request: Request) -> dict[str, object]:
    candidate, graph = _load_runtime_graph(request)
    ctx = require_authenticated_context(request)
    body = await request.json()
    if not isinstance(body, dict) or any(body.get(key) != value for key, value in {
        "graphRef": candidate.artifact_id, "graphVersion": candidate.graph_version, "graphHash": candidate.content_hash,
    }.items()):
        raise HTTPException(status_code=409, detail="stale graph binding")
    selected = [str(value) for value in body.get("selectedNodeIds", []) if isinstance(value, str)]
    if any(node_id not in graph.nodes for node_id in selected):
        raise HTTPException(status_code=409, detail="stale node selection")
    request_pin = TenderExpertRequest(
        organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id), parent_session_id=str(body.get("sessionId") or "graph-explorer"),
        graph_ref=candidate.artifact_id, graph_version=candidate.graph_version, graph_hash=candidate.content_hash,
        semantica_commit=candidate.semantica_commit, message=str(body.get("message") or "Explain the selected tender risk."),
        selected_node_ids=tuple(selected), request_id=str(body.get("requestId") or "runtime-expert-mvl"),
    )
    async def deterministic_handler(pin: TenderExpertRequest) -> TenderExpertResult:
        return TenderExpertResult(status="completed", request_id=pin.request_id, specialist_session_id=pin.specialist_session_id,
                                  message="This is a candidate, non-authoritative graph explanation grounded in pinned source evidence.",
                                  graph_refs=({"graphRef": pin.graph_ref, "graphVersion": pin.graph_version, "graphHash": pin.graph_hash},))
    result = await TenderExpertBridge().run(request_pin, deterministic_handler, retry_once=False)
    return {"status": result.status, "requestId": result.request_id, "message": result.message,
            "graphRefs": list(result.graph_refs), "candidate": True, "authorityState": "candidate"}
