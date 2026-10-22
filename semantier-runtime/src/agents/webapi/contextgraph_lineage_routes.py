"""ContextGraph lineage routes (Plan B M1, extended M5+).

Five authenticated routes expose the lineage projection service to the
Lineage tab:

- ``GET /api/contextgraph/lineages``
- ``GET /api/contextgraph/workspace-domains``
- ``GET /api/contextgraph/lineage-graph``
- ``GET /api/contextgraph/lineages/{lineage_root_artifact_id}/versions``
- ``DELETE /api/contextgraph/lineages/{lineage_root_artifact_id}``
- ``POST /api/contextgraph/lineages/batch-delete``

M5+ change: the URL path parameter ``lineage_root_artifact_id`` is now
interpreted as ``lineage_id`` (= source_id). One source upload begins
one lineage; the parameter name is preserved for API stability but its
semantic is "lineage identifier = source identifier". Read queries
also accept a ``lineage_id`` query parameter alias.

Read paths serve the lineage tab's list + DAG. The DELETE path
routes through Plan A's source-retirement orchestrator (D3).
"""

from __future__ import annotations

from storage import hosted_postgres as hosted


import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agents.route_policy import classify_authorization_class
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from runtime_paths import sqlite_db_path
from semantier.knowledge_source_retirement import (
    AuthoritativeBlock,
    InvalidStateTransition,
    SourceRetirementState,
    TerminalStateReached,
)
from semantier.lineage_projection_service import (
    LINEAGE_STATE_IN_PROGRESS,
    LINEAGE_STATE_LIVE,
    LINEAGE_STATE_ORPHAN,
    LineageProjectionError,
    LineageProjectionService,
    UnknownLineage,
)
from semantier.workspace_retirement_stager import RepresentationMissing

router = APIRouter(tags=["webapi-gateway"])


class LineageRetirementPreviewRequest(BaseModel):
    domain_id: str
    lineage_ids: list[str]
    retirement_mode: Literal["ordinary", "governed"]


class LineageRetirementConfirmation(BaseModel):
    lineage_id: str
    operation_id: str
    inventory_hash: str


class LineageBatchDeleteRequest(BaseModel):
    domain_id: str
    retirement_mode: Literal["ordinary", "governed"]
    reason: str | None = None
    lineages: list[LineageRetirementConfirmation]


def _authz(request: Request, method: str, path: str):
    authz_class = classify_authorization_class(method, path)
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    return _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint=f"{method} {path}",
    )


def _service_for(ctx, domain_id: str) -> LineageProjectionService:
    eos_path = Path(
        os.environ.get(
            "SEMANTIER_EOS_DB_PATH",
            str(sqlite_db_path("eos.db", ".semantier/eos.db")),
        )
    )
    return LineageProjectionService(
        workspace_root=ctx.workspace_root,
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id),
        domain_id=domain_id,
        eos_path=eos_path,
    )


def _resolve_lineage_id(params) -> str | None:
    """Accept either ``lineage_id`` or ``lineage_root_artifact_id``.

    M5+: the canonical lineage identifier is source_id. Old callers
    still pass ``lineage_root_artifact_id``; the projection treats both
    as the same string.
    """
    return (
        params.get("lineage_id")
        or params.get("lineage_root_artifact_id")
    )


# ---- routes ---------------------------------------------------------------


@router.get("/api/contextgraph/workspace-domains")
async def list_workspace_domains(request: Request) -> dict:
    """Return context-graph domains present in the active workspace."""
    route_path = "/api/contextgraph/workspace-domains"
    ctx = _authz(request, "GET", route_path)
    workspace_root = Path(ctx.workspace_root).resolve()
    manifests_root = workspace_root / ".semantier" / "context-graphs"
    domains: set[str] = set()
    if manifests_root.is_dir():
        domains.update(path.name for path in manifests_root.iterdir() if path.is_dir())
    if not domains:
        domains.add("default")
    return {"ok": True, "domains": sorted(domains)}


@router.get("/api/contextgraph/lineages")
async def list_lineages(request: Request) -> dict:
    """Return every lineage root in the active domain.

    Query: ``domain_id`` (required), ``filter`` (one of
    ``all|in_progress|live|orphan``; default ``all``).

    M5+ exposes the ``in_progress`` filter so the lineage tab can
    surface source-only lineages (uploaded, no graph artifact yet) as
    a workspace health stat.
    """
    route_path = "/api/contextgraph/lineages"
    ctx = _authz(request, "GET", route_path)
    params = request.query_params
    domain_id = params.get("domain_id")
    if not domain_id:
        raise HTTPException(status_code=400, detail="domain_id is required")
    flt = params.get("filter") or "all"
    service = _service_for(ctx, domain_id)
    try:
        rows = service.list_lineages(filter=flt)
    except LineageProjectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "domain_id": domain_id,
        "filter": flt,
        "lineages": rows,
        "states": {
            "in_progress": LINEAGE_STATE_IN_PROGRESS,
            "live": LINEAGE_STATE_LIVE,
            "orphan": LINEAGE_STATE_ORPHAN,
        },
    }


@router.get("/api/contextgraph/lineage-graph")
async def get_lineage_graph(request: Request) -> dict:
    """Return descendants (nodes + edges) of a lineage.

    Query: ``lineage_id`` or ``lineage_root_artifact_id`` (required),
    ``domain_id`` (required), ``graph_version`` (optional).

    M5+ aliases ``lineage_id`` for the legacy ``lineage_root_artifact_id``
    parameter; both are interpreted as the lineage identifier (= source_id).
    For in_progress lineages, returns a single-node DAG with the source row.
    """
    route_path = "/api/contextgraph/lineage-graph"
    ctx = _authz(request, "GET", route_path)
    params = request.query_params
    lineage_root = _resolve_lineage_id(params)
    domain_id = params.get("domain_id")
    if not lineage_root or not domain_id:
        raise HTTPException(
            status_code=400,
            detail="lineage_id and domain_id are required",
        )
    graph_version = params.get("graph_version")
    service = _service_for(ctx, domain_id)
    try:
        graph = service.get_lineage_graph(
            lineage_id=lineage_root,
            graph_version=graph_version,
        )
    except UnknownLineage as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, **graph}


@router.get("/api/contextgraph/lineages/{lineage_root_artifact_id}/versions")
async def get_lineage_versions(
    lineage_root_artifact_id: str, request: Request
) -> dict:
    """Return distinct ``graph_version`` rows for a lineage.

    Query: ``domain_id`` (required).

    M5+: the URL path parameter is now interpreted as ``lineage_id``
    (= source_id). For in_progress lineages, returns an empty list.
    """
    route_path = "/api/contextgraph/lineages/{lineage_root_artifact_id}/versions"
    ctx = _authz(request, "GET", route_path)
    params = request.query_params
    domain_id = params.get("domain_id")
    if not domain_id:
        raise HTTPException(status_code=400, detail="domain_id is required")
    service = _service_for(ctx, domain_id)
    try:
        versions = service.list_lineage_versions(
            lineage_id=lineage_root_artifact_id,
        )
    except UnknownLineage as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "ok": True,
        "lineage_id": lineage_root_artifact_id,
        "lineage_root_artifact_id": lineage_root_artifact_id,
        "domain_id": domain_id,
        "versions": versions,
    }


def _eos_path() -> Path:
    return Path(
        os.environ.get(
            "SEMANTIER_EOS_DB_PATH",
            str(sqlite_db_path("eos.db", ".semantier/eos.db")),
        )
    )


def _normalize_lineage_ids(values: list[str]) -> list[str]:
    normalized = [str(value).strip() for value in values]
    if not normalized or any(not value for value in normalized):
        raise HTTPException(status_code=400, detail="lineage_ids must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise HTTPException(status_code=400, detail="lineage_ids must be unique")
    return normalized


def _runtime_for(ctx, *, domain_id: str):
    from semantier.knowledge_source_retirement import SourceRetirementOrchestrator
    from semantier.lineage_cascade_inventory import (
        EosLineageInventoryCollector,
        GraphArtifactInventoryCollector,
        GraphSchemaInventoryCollector,
        LineageCascadeInventoryService,
    )
    from semantier.lineage_cascade_retirement import (
        LineageCascadeRetirementService,
    )
    from semantier.semantica_context_graph.artifact_repository import (
        ContextGraphArtifactRepository,
    )
    from semantier.workspace_retirement_stager import WorkspaceRetirementStager
    from storage.graph_schema_discovery_store import GraphSchemaDiscoveryStore
    from storage.human_grounding_store import HumanGroundingStore
    from storage.knowledge_extraction_store import KnowledgeExtractionStore

    workspace_root = Path(ctx.workspace_root).resolve()
    extraction_store = KnowledgeExtractionStore(_eos_path())
    HumanGroundingStore(_eos_path(), graph_backend=None)
    domain_ids = {domain_id}
    manifests_root = workspace_root / ".semantier" / "context-graphs"
    if manifests_root.is_dir():
        domain_ids.update(
            path.parent.name for path in manifests_root.glob("*/manifest.sqlite3")
        )
    repositories = tuple(
        ContextGraphArtifactRepository(
            workspace_root,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id),
            domain_id=item,
        )
        for item in sorted(domain_ids)
    )
    schema_store = GraphSchemaDiscoveryStore(
        workspace_root, workspace_id=str(ctx.workspace_id)
    )
    inventory_service = LineageCascadeInventoryService(
        workspace_root=workspace_root,
        workspace_id=str(ctx.workspace_id),
        organization_id=str(ctx.organization_id),
        actor_id=str(ctx.user_id),
        collectors=(
            EosLineageInventoryCollector(
                store=extraction_store,
                workspace_root=workspace_root,
                organization_id=str(ctx.organization_id),
                workspace_id=str(ctx.workspace_id),
            ),
            *(GraphArtifactInventoryCollector(repository=repo) for repo in repositories),
            GraphSchemaInventoryCollector(store=schema_store),
        ),
    )
    cascade = LineageCascadeRetirementService(
        inventory_service=inventory_service,
        extraction_store=extraction_store,
        artifact_repositories=repositories,
        schema_stores=(schema_store,),
        actor_id=str(ctx.user_id),
    )
    stager = WorkspaceRetirementStager(
        workspace_root,
        workspace_id=str(ctx.workspace_id),
        allow_missing=True,
    )
    orchestrator = SourceRetirementOrchestrator(
        extraction_store=extraction_store,
        file_stager=stager,
    )
    return extraction_store, inventory_service, cascade, stager, orchestrator


def _require_lineage_owner(store, *, lineage_id: str, ctx) -> None:
    row = store._conn.execute(
        hosted.sql_query("""SELECT source_id FROM knowledge_builder_sources
        WHERE COALESCE(lineage_id, source_id) = ?
          AND organization_id = ? AND workspace_id = ?""", 'SELECT source_id FROM knowledge_builder_sources\n        WHERE COALESCE(lineage_id, source_id) = %s\n          AND organization_id = %s AND workspace_id = %s'),
        (lineage_id, str(ctx.organization_id), str(ctx.workspace_id)),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"lineage not found: {lineage_id}")


def _preview_one(
    *,
    lineage_id: str,
    retirement_mode: Literal["ordinary", "governed"],
    domain_id: str,
    ctx,
) -> dict:
    store, inventory_service, _cascade, _stager, orchestrator = _runtime_for(
        ctx, domain_id=domain_id
    )
    _require_lineage_owner(store, lineage_id=lineage_id, ctx=ctx)
    operation_id = f"op_{uuid4().hex}"
    inventory = inventory_service.preview(
        operation_id=operation_id,
        lineage_id=lineage_id,
        retirement_mode=retirement_mode,
        escalate_authoritative=True,
    )
    orchestrator.begin(
        operation_id=operation_id,
        source_id=lineage_id,
        actor_id=str(ctx.user_id),
        retirement_mode=inventory.retirement_mode,
        domain_id=domain_id,
        inventory_hash=inventory.inventory_hash,
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id),
    )
    authoritative_count = len(
        {
            entry.member_id
            for entry in inventory.entries
            if entry.authority_state == "authoritative"
        }
    )
    return {
        "lineage_id": lineage_id,
        "operation_id": operation_id,
        "inventory_hash": inventory.inventory_hash,
        "cascade_scope": inventory.category_counts,
        "authoritative_count": authoritative_count,
        "requires_reason": inventory.retirement_mode == "governed",
    }


@router.post("/api/contextgraph/lineages/retirement-preview")
async def preview_lineage_retirement(
    payload: LineageRetirementPreviewRequest,
    request: Request,
) -> dict:
    route_path = "/api/contextgraph/lineages/retirement-preview"
    ctx = _authz(request, "POST", route_path)
    domain_id = payload.domain_id.strip()
    if not domain_id:
        raise HTTPException(status_code=400, detail="domain_id is required")
    lineage_ids = _normalize_lineage_ids(payload.lineage_ids)
    previews = [
        _preview_one(
            lineage_id=lineage_id,
            retirement_mode=payload.retirement_mode,
            domain_id=domain_id,
            ctx=ctx,
        )
        for lineage_id in lineage_ids
    ]
    if payload.retirement_mode == "ordinary" and any(
        item["authoritative_count"] for item in previews
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "requires_governed_retirement",
                "message": "authoritative materializations require governed retirement",
                "lineages": previews,
            },
        )
    return {"ok": True, "domain_id": domain_id, "lineages": previews}


class _InventoryBoundStager:
    def __init__(self, stager, inventory) -> None:
        self.stager = stager
        self.inventory = inventory

    def stage_many(self, **_kwargs) -> str:
        return self.stager.stage_inventory(self.inventory)

    def restore(self, **kwargs):
        return self.stager.restore(**kwargs)

    def unlink(self, **kwargs):
        return self.stager.unlink(**kwargs)


def _operation_row(store, operation_id: str):
    return store._conn.execute(
        hosted.sql_query("SELECT * FROM source_retirement_operations WHERE operation_id = ?", 'SELECT * FROM source_retirement_operations WHERE operation_id = %s'),
        (operation_id,),
    ).fetchone()


def _retire_confirmation(
    *,
    confirmation: LineageRetirementConfirmation,
    retirement_mode: Literal["ordinary", "governed"],
    reason: str | None,
    domain_id: str,
    ctx,
) -> dict:
    from semantier.lineage_cascade_inventory import InventoryChanged
    from semantier.lineage_cascade_retirement import (
        GovernedReasonRequired,
        RequiresGovernedRetirement,
    )

    store, inventory_service, cascade, stager, orchestrator = _runtime_for(
        ctx, domain_id=domain_id
    )
    operation = _operation_row(store, confirmation.operation_id)
    if operation is None:
        raise HTTPException(status_code=404, detail={"code": "operation_not_found"})
    expected = (
        confirmation.lineage_id,
        str(ctx.user_id),
        retirement_mode,
        domain_id,
        str(ctx.organization_id),
        str(ctx.workspace_id),
    )
    actual = (
        operation["source_id"],
        operation["actor_id"],
        operation["retirement_mode"],
        operation["domain_id"],
        operation["organization_id"],
        operation["workspace_id"],
    )
    if actual != expected:
        raise HTTPException(status_code=409, detail={"code": "scope_mismatch"})
    inventory = inventory_service.load(confirmation.operation_id)
    if (
        confirmation.inventory_hash != inventory.inventory_hash
        or operation["inventory_hash"] != inventory.inventory_hash
    ):
        raise HTTPException(status_code=409, detail={"code": "inventory_changed"})
    try:
        if operation["state"] == SourceRetirementState.REQUESTED.value:
            inventory_service.assert_current(inventory)
        else:
            inventory_service.assert_scope_membership(inventory)
        bound_stager = _InventoryBoundStager(stager, inventory)

        def artifact_stage() -> None:
            cascade.tombstone_artifacts(inventory, governed_reason=reason)
            cascade.assert_retired(inventory)

        current = orchestrator.resume(
            operation_id=confirmation.operation_id,
            file_stager=bound_stager,
            metadata_stage=lambda: cascade.tombstone_metadata(inventory),
            artifact_stage=artifact_stage,
        )
    except InventoryChanged as exc:
        raise HTTPException(status_code=409, detail={"code": "inventory_changed"}) from exc
    except RequiresGovernedRetirement as exc:
        raise HTTPException(
            status_code=409, detail={"code": "requires_governed_retirement"}
        ) from exc
    except GovernedReasonRequired as exc:
        raise HTTPException(status_code=400, detail={"code": "reason_required"}) from exc
    if current is not SourceRetirementState.COMPLETED:
        row = _operation_row(store, confirmation.operation_id)
        raise HTTPException(
            status_code=409,
            detail={
                "code": "retirement_failed_retryable",
                "message": str(row["failure_reason"] or "retirement did not complete"),
            },
        )
    return {
        "lineage_id": confirmation.lineage_id,
        "operation_id": confirmation.operation_id,
        "inventory_hash": confirmation.inventory_hash,
        "state": current.value,
    }


@router.delete("/api/contextgraph/lineages/{lineage_root_artifact_id}")
async def delete_lineage(
    lineage_root_artifact_id: str, request: Request
) -> dict:
    """Cascade-tombstone one lineage through the governed orchestrator."""
    route_path = "/api/contextgraph/lineages/{lineage_root_artifact_id}"
    ctx = _authz(request, "DELETE", route_path)
    domain_id = request.query_params.get("domain_id")
    if not domain_id:
        raise HTTPException(status_code=400, detail="domain_id is required")
    preview = _preview_one(
        lineage_id=lineage_root_artifact_id,
        retirement_mode="ordinary",
        domain_id=domain_id,
        ctx=ctx,
    )
    if preview["authoritative_count"]:
        raise HTTPException(
            status_code=409,
            detail={"code": "requires_governed_retirement", **preview},
        )
    deleted = _retire_confirmation(
        confirmation=LineageRetirementConfirmation(
            lineage_id=lineage_root_artifact_id,
            operation_id=preview["operation_id"],
            inventory_hash=preview["inventory_hash"],
        ),
        retirement_mode="ordinary",
        reason=None,
        domain_id=domain_id,
        ctx=ctx,
    )
    return {"ok": True, **deleted}


@router.post("/api/contextgraph/lineages/batch-delete")
async def batch_delete_lineages(
    payload: LineageBatchDeleteRequest,
    request: Request,
) -> dict:
    """Retire selected lineages in one request with per-lineage outcomes."""
    route_path = "/api/contextgraph/lineages/batch-delete"
    ctx = _authz(request, "POST", route_path)
    domain_id = payload.domain_id.strip()
    if not domain_id:
        raise HTTPException(status_code=400, detail="domain_id is required")
    if payload.retirement_mode == "governed" and not str(payload.reason or "").strip():
        raise HTTPException(status_code=400, detail={"code": "reason_required"})
    lineage_ids = _normalize_lineage_ids(
        [confirmation.lineage_id for confirmation in payload.lineages]
    )
    deleted: list[dict] = []
    skipped: list[dict] = []
    failed: list[dict] = []
    for confirmation in payload.lineages:
        lineage_id = confirmation.lineage_id
        try:
            deleted.append(
                _retire_confirmation(
                    confirmation=confirmation,
                    retirement_mode=payload.retirement_mode,
                    reason=payload.reason,
                    ctx=ctx,
                    domain_id=domain_id,
                )
            )
        except HTTPException as exc:
            detail = exc.detail
            if isinstance(detail, dict):
                code = str(detail.get("code") or f"http_{exc.status_code}")
                message = str(detail.get("message") or code)
            else:
                code = f"http_{exc.status_code}"
                message = str(detail)
            outcome = {
                "lineage_id": lineage_id,
                "code": code,
                "message": message,
                "status": exc.status_code,
            }
            if code == "representation_missing":
                skipped.append(outcome)
            else:
                failed.append(outcome)
        except Exception:
            failed.append(
                {
                    "lineage_id": lineage_id,
                    "code": "internal_error",
                    "message": "lineage retirement failed",
                    "status": 500,
                }
            )

    return {
        "ok": not skipped and not failed,
        "requested_count": len(lineage_ids),
        "deleted": deleted,
        "skipped": skipped,
        "failed": failed,
    }


__all__ = [
    "LineageBatchDeleteRequest",
    "LineageRetirementConfirmation",
    "LineageRetirementPreviewRequest",
    "batch_delete_lineages",
    "preview_lineage_retirement",
    "router",
]
