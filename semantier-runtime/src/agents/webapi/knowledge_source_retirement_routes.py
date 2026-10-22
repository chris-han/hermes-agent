"""Knowledge source retirement routes (Plan A M4).

These four routes expose the durable source-retirement saga as gateway
endpoints, completing the cascade-deletion surface required by Plan A.

Routes
------
- ``POST /api/knowledge/source/{source_id}/retire``
  Begin a new retirement operation and drive it forward through the saga
  until ``completed`` or a terminal state.
- ``GET  /api/knowledge/source/{source_id}/retirement``
  Return the most recent retirement operation for a source. Used by the
  Lineage tab (Plan B) to render per-source retirement state.
- ``POST /api/knowledge/source/{source_id}/retirement/{operation_id}/resume``
  Resume an interrupted operation forward through the saga.
- ``GET  /api/knowledge/source/{source_id}/retirement/preview``
  Return cascade scope and severity tier for the confirm dialog. The
  preview always includes the authoritative-descendant count, which the
  UI uses to escalate the confirmation prompt to a typed-confirm modal.

Error mapping
-------------
- :class:`AuthoritativeBlock` -> ``409 requires_governed_retirement``
  (Plan A D5: typed-confirm UX cannot override an authority boundary.)
- :class:`TerminalStateReached` -> ``409 terminal_state``
- :class:`InvalidStateTransition` -> ``422 invalid_state_transition``
"""

from __future__ import annotations

from storage import hosted_postgres as hosted


import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from agents.route_policy import classify_authorization_class
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from contracts.evidence_location import SourceIdentity
from runtime_paths import sqlite_db_path
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import (
    adapt_docx_bytes,
    persist_canonical_source_ir,
)
from semantier.knowledge_source_retirement import (
    AuthoritativeBlock,
    InvalidStateTransition,
    SourceRetirementOrchestrator,
    SourceRetirementState,
    TerminalStateReached,
)
from semantier.workspace_retirement_stager import (
    RepresentationMissing,
    WorkspaceRetirementStager,
)
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.knowledge_extraction_store import KnowledgeExtractionStore

router = APIRouter(tags=["webapi-gateway"])


# ---- shared helpers -------------------------------------------------------


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
    except Exception as exc:
        raise HTTPException(status_code=400, detail="JSON body required") from exc
    return body if isinstance(body, dict) else {}


def _eos_path() -> Path:
    return Path(
        os.environ.get(
            "SEMANTIER_EOS_DB_PATH",
            str(sqlite_db_path("eos.db", ".semantier/eos.db")),
        )
    )


def _extraction_store() -> KnowledgeExtractionStore:
    return KnowledgeExtractionStore(_eos_path())


def _stager_for(ctx) -> WorkspaceRetirementStager:
    """Workspace-bound retirement stager (Law 6).

    The trash root lives under ``<workspace_root>/.semantier-retirement/trash``;
    no global scratch path is permitted.
    """
    return WorkspaceRetirementStager(
        Path(ctx.workspace_root),
        workspace_id=str(ctx.workspace_id),
    )


def _governed_file(workspace_root: str | Path, relative_path: str) -> Path:
    root = Path(workspace_root).resolve()
    candidate = (root / relative_path).resolve()
    if Path(relative_path).is_absolute() or not candidate.is_relative_to(root):
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_path", "message": "Path is outside the workspace."},
        )
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _orchestrator(
    ctx,
    *,
    extraction_store: KnowledgeExtractionStore,
    file_stager: WorkspaceRetirementStager,
) -> SourceRetirementOrchestrator:
    def _complete_source_artifact_stage() -> None:
        """Acknowledge the source-family artifact stage.

        This route owns uploaded files and normalized representations. Their
        lifecycle rows are tombstoned by the orchestrator's metadata stage and
        their files are already held by ``WorkspaceRetirementStager``. Graph
        artifacts use the governed lineage retirement route instead.
        """

    return SourceRetirementOrchestrator(
        extraction_store=extraction_store,
        file_stager=file_stager,
        artifact_stage=_complete_source_artifact_stage,
    )


def _retryable_operation_id(
    extraction_store: KnowledgeExtractionStore,
    source_id: str,
) -> str | None:
    """Return the newest recoverable saga without reusing preview operations."""
    for operation in extraction_store.list_source_retirement(source_id):
        if operation["state"] == SourceRetirementState.FAILED_RETRYABLE.value:
            return str(operation["operation_id"])
    return None


# ---- retirement stage chain ----------------------------------------------


_FORWARD_CHAIN = (
    SourceRetirementState.FILE_STAGED,
    SourceRetirementState.METADATA_TOMBSTONED,
    SourceRetirementState.ARTIFACTS_TOMBSTONED,
    SourceRetirementState.COMPLETED,
)


def _serialise_operation(row: dict) -> dict:
    return {
        "operation_id": row["operation_id"],
        "source_id": row["source_id"],
        "actor_id": row.get("actor_id"),
        "state": row["state"],
        "staged_path": row.get("staged_path"),
        "original_path": row.get("original_path"),
        "failure_reason": row.get("failure_reason"),
        "failure_stage": row.get("failure_stage"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _compute_preview(conn, source_id: str) -> dict:
    """Plan A M4: cascade scope, severity tier, authoritative count.

    Counts every cascade component that lives in the eos.db canonical
    store. The authoritative-descendant count drives whether the UI
    escalates the confirm modal to a typed-confirm flow (tier-3) and
    whether the saga will short-circuit to ``failed/blocked_authoritative``
    on ``FILE_STAGED``.

    The preview is defensive: tables from other stores (e.g. the manifest
    graph_artifacts table) are not queried here. Cross-store aggregation
    belongs to Plan B (workspace lineage surface).
    """
    def _count(table: str) -> int:
        if hosted.is_postgres(conn) and conn.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_attribute "
            "WHERE attrelid=to_regclass(%s) AND attname='source_id' AND NOT attisdropped)",
            (table,),
        ).fetchone()[0] is False:
            return 0
        try:
            row = conn.execute(
                hosted.sql_query(f"SELECT COUNT(*) AS c FROM {table} WHERE source_id = ?", f'SELECT COUNT(*) AS c FROM {table} WHERE source_id = %s'),
                (source_id,),
            ).fetchone()
            return int(row["c"] or 0)
        except Exception:
            if hosted.is_postgres(conn):
                raise
            # Table not present in this DB; cross-store aggregation is
            # out of scope for Plan A.
            return 0

    extraction_runs = _count("knowledge_builder_extraction_runs")
    canonical_expressions = _count("contextgraph_canonical_expressions")
    reference_concepts = _count("contextgraph_reference_concepts")
    graph_artifacts = _count("graph_artifacts")
    normalized_rows = conn.execute(
        hosted.sql_query("SELECT representation.representation_id, representation.artifact_ref, "
        "representation.artifact_path FROM source_representation_edges representation "
        "JOIN source_representation_lifecycle_events lifecycle "
        "ON lifecycle.event_id = (SELECT latest.event_id "
        "FROM source_representation_lifecycle_events latest "
        "WHERE latest.representation_id = representation.representation_id "
        "ORDER BY latest.created_at DESC, latest.rowid DESC LIMIT 1) "
        "WHERE representation.source_id = ? AND lifecycle.next_state = 'active' "
        "ORDER BY representation.version", "SELECT representation.representation_id, representation.artifact_ref, representation.artifact_path FROM source_representation_edges representation JOIN source_representation_lifecycle_events lifecycle ON lifecycle.event_id = (SELECT latest.event_id FROM source_representation_lifecycle_events latest WHERE latest.representation_id = representation.representation_id ORDER BY latest.created_at DESC, latest.event_sequence DESC LIMIT 1) WHERE representation.source_id = %s AND lifecycle.next_state = 'active' ORDER BY representation.version"),
        (source_id,),
    ).fetchall()
    normalized_artifacts = len(normalized_rows)

    authoritative = conn.execute(
        hosted.sql_query("SELECT COUNT(*) AS c FROM contextgraph_artifact_origins"
        " WHERE source_id = ? AND authority_state = 'authoritative'", "SELECT COUNT(*) AS c FROM contextgraph_artifact_origins WHERE source_id = %s AND authority_state = 'authoritative'"),
        (source_id,),
    ).fetchone()
    authoritative_count = int(authoritative["c"] or 0)

    cascade_total = (
        extraction_runs
        + canonical_expressions
        + reference_concepts
        + graph_artifacts
        + normalized_artifacts
    )
    if authoritative_count > 0:
        severity = "tier-3-locked"
    elif cascade_total >= 50:
        severity = "tier-2-medium"
    else:
        severity = "tier-1-light"

    return {
        "source_id": source_id,
        "cascade_scope": {
            "extraction_runs": extraction_runs,
            "reference_concepts": reference_concepts,
            "canonical_expressions": canonical_expressions,
            "graph_artifacts": graph_artifacts,
            "normalized_artifacts": normalized_artifacts,
        },
        "normalized_representations": [
            {
                "representation_id": row["representation_id"],
                "artifact_ref": row["artifact_ref"],
                "display_label": Path(row["artifact_path"]).name,
            }
            for row in normalized_rows
        ],
        "authoritative_descendants": authoritative_count,
        "severity_tier": severity,
    }


# ---- routes ---------------------------------------------------------------


@router.post("/api/knowledge/sources")
async def register_knowledge_source(request: Request) -> dict:
    """Create the canonical workspace-bound lineage root before normalization."""
    route_path = "/api/knowledge/sources"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    source_upload_ref = str(body.get("source_upload_ref") or "").strip()
    original_path = str(body.get("original_path") or "").strip()
    if not source_upload_ref or not original_path:
        raise HTTPException(status_code=400, detail="source_upload_ref and original_path are required")
    original = _governed_file(ctx.workspace_root, original_path)
    if not original.is_file():
        raise HTTPException(
            status_code=409,
            detail={"code": "representation_missing", "message": "Original upload is missing."},
        )

    extraction = _extraction_store()
    existing = extraction._conn.execute(
        hosted.sql_query("SELECT source_id FROM knowledge_builder_sources WHERE organization_id = ? "
        "AND workspace_id = ? AND source_ref = ? AND source_text = ? "
        "ORDER BY created_at LIMIT 1", 'SELECT source_id FROM knowledge_builder_sources WHERE organization_id = %s AND workspace_id = %s AND source_ref = %s AND source_text = %s ORDER BY created_at LIMIT 1'),
        (str(ctx.organization_id), str(ctx.workspace_id), source_upload_ref, original_path),
    ).fetchone()
    if existing is not None:
        return {"ok": True, "source_id": existing["source_id"], "created": False}
    store = KnowledgeBuilderStore(_eos_path())
    row = store.create_source(
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id),
        source_kind="file",
        source_ref=source_upload_ref,
        source_text=original_path,
        source_metadata={
            "original_path": original_path,
            "original_content_hash": _sha256(original),
        },
    )
    return {"ok": True, "source_id": row["source_id"], "created": True}


@router.post("/api/knowledge/source-representations")
async def register_source_representation(request: Request) -> dict:
    """Attach a durable normalized child to an existing canonical source."""
    route_path = "/api/knowledge/source-representations"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    source_id = str(body.get("source_id") or "").strip()
    artifact_ref = str(body.get("artifact_ref") or "").strip()
    artifact_path = str(body.get("artifact_path") or "").strip()
    if not source_id or not artifact_ref or not artifact_path:
        raise HTTPException(status_code=400, detail="source_id, artifact_ref, and artifact_path are required")

    extraction = _extraction_store()
    source = extraction._conn.execute(
        hosted.sql_query("SELECT source_id, source_text, content_hash, created_at FROM knowledge_builder_sources "
        "WHERE source_id = ? AND organization_id = ? AND workspace_id = ?", 'SELECT source_id, source_text, content_hash, created_at FROM knowledge_builder_sources WHERE source_id = %s AND organization_id = %s AND workspace_id = %s'),
        (source_id, str(ctx.organization_id), str(ctx.workspace_id)),
    ).fetchone()
    if source is None:
        raise HTTPException(status_code=404, detail={
            "code": "source_not_found", "message": "Source is not governed by this workspace.",
            "source_id": source_id,
        })
    original_path = str(source["source_text"])
    original = _governed_file(ctx.workspace_root, original_path)
    artifact = _governed_file(ctx.workspace_root, artifact_path)
    if not original.is_file() or not artifact.is_file():
        raise HTTPException(status_code=409, detail={
            "code": "representation_missing",
            "message": "Original or normalized representation is missing.",
            "source_id": source_id,
        })
    edge = extraction.record_source_representation(
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id),
        source_id=source_id,
        lineage_id=source_id,
        relation_kind="normalized_from",
        original_path=original_path,
        original_content_hash=_sha256(original),
        artifact_ref=artifact_ref,
        artifact_path=artifact_path,
        artifact_content_hash=_sha256(artifact),
    )
    canonical_ir_ref = None
    source_identity_ref = None
    if original.suffix.lower() == ".docx":
        source_identity_ref = f"governed-source:{source_id}"
        canonical_ir = adapt_docx_bytes(
            original.read_bytes(), source_identity_ref=source_identity_ref
        )
        canonical_ir_ref = persist_canonical_source_ir(
            canonical_ir,
            ctx.hermes_home,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id),
        )
        registry = KnowledgeBuilderStore(_eos_path(), workspace_root=ctx.workspace_root)
        registry.bind_canonical_ir_ref(
            source_id=source_id,
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id),
            canonical_ir_ref=canonical_ir_ref,
            expected_content_hash=str(source["content_hash"]),
        )
        created_at = datetime.fromisoformat(
            str(source["created_at"]).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        registry.register_source_identity(
            source_identity=SourceIdentity(
                schemaVersion="semantier.source_identity.v1",
                sourceIdentityRef=source_identity_ref,
                organizationId=str(ctx.organization_id),
                workspaceId=str(ctx.workspace_id),
                sourceHash=f"sha256:{_sha256(original)}",
                mediaType=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                originalName=original.name,
                artifactRef=original_path,
                sourceVersion=None,
                supersedesSourceIdentityRef=None,
                createdAt=created_at,
            ),
            lineage_id=source_id,
        )
    return {
        "ok": True,
        "source_id": source_id,
        "canonical_ir_ref": canonical_ir_ref,
        "source_identity_ref": source_identity_ref,
        "representation": {**edge, "state": "active"},
    }


@router.get("/api/knowledge/source-representations")
async def list_source_representations(request: Request) -> dict:
    """Project canonical roots and their active normalized children."""
    route_path = "/api/knowledge/source-representations"
    ctx = _authz(request, "GET", route_path)
    extraction = _extraction_store()
    source_rows = extraction._conn.execute(
        hosted.sql_query("SELECT source_id, source_ref, source_text, source_metadata_json FROM knowledge_builder_sources "
        "WHERE organization_id = ? AND workspace_id = ? ORDER BY created_at, source_id", 'SELECT source_id, source_ref, source_text, source_metadata_json FROM knowledge_builder_sources WHERE organization_id = %s AND workspace_id = %s ORDER BY created_at, source_id'),
        (str(ctx.organization_id), str(ctx.workspace_id)),
    ).fetchall()
    sources = []
    registry = KnowledgeBuilderStore(_eos_path(), workspace_root=ctx.workspace_root)
    try:
        for source in source_rows:
            children = extraction.list_active_source_representations(
                source["source_id"],
                organization_id=str(ctx.organization_id),
                workspace_id=str(ctx.workspace_id),
            )
            metadata = json.loads(source["source_metadata_json"] or "{}")
            identities = []
            if metadata.get("canonical_ir_ref"):
                identities = [identity for identity in registry.list_source_identities(
                    organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id),
                    lineage_id=source["source_id"],
                ) if identity.artifact_ref == source["source_text"]
                    and identity.source_hash.removeprefix("sha256:") == str(metadata.get("original_content_hash") or "").removeprefix("sha256:")]
            retirement_operations = extraction.list_source_retirement(source["source_id"])
            sources.append({
                "source_id": source["source_id"],
                "retirement_state": retirement_operations[0]["state"] if retirement_operations else None,
                "source_identity_ref": identities[0].source_identity_ref if len(identities) == 1 else None,
                "canonical_ir_ref": metadata.get("canonical_ir_ref"),
                "source_upload_ref": source["source_ref"],
                "original_path": source["source_text"],
                "original_name": identities[0].original_name if len(identities) == 1 else metadata.get("original_name"),
                "media_type": identities[0].media_type if len(identities) == 1 else None,
                "normalized_representations": [{
                    "representation_id": child["representation_id"],
                    "relation_kind": child["relation_kind"],
                    "version": child["version"],
                    "artifact_ref": child["artifact_ref"],
                    "artifact_path": child["artifact_path"],
                    "display_label": Path(child["artifact_path"]).name,
                    "state": "active",
                } for child in children],
            })
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="Governed source identity projection is invalid") from exc
    finally:
        registry.close()
    return {"ok": True, "sources": sources}


@router.post("/api/knowledge/source/{source_id}/retire")
async def retire_knowledge_source(source_id: str, request: Request) -> dict:
    """Begin and drive a retirement saga to completion.

    Body (optional): ``{"targetState": "completed"}`` controls how far
    forward the saga is driven. The default is ``"completed"`` (full
    cascade). ``"requested"`` opens the operation without driving
    forward.
    """
    route_path = "/api/knowledge/source/{source_id}/retire"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    target = body.get("targetState") or body.get("target_state") or "completed"

    extraction = _extraction_store()
    stager = _stager_for(ctx)
    orch = _orchestrator(
        ctx, extraction_store=extraction, file_stager=stager
    )
    try:
        target_state = SourceRetirementState(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"unknown target_state: {target}") from exc

    retryable_operation_id = (
        _retryable_operation_id(extraction, source_id)
        if target_state is SourceRetirementState.COMPLETED
        else None
    )
    operation_id = retryable_operation_id or orch.begin(
        source_id=source_id,
        actor_id=str(ctx.user_id),
    )

    if target_state is SourceRetirementState.REQUESTED:
        operation = extraction.list_source_retirement(source_id)[0]
        return {
            "ok": True,
            "operation_id": operation_id,
            "state": operation["state"],
            "source_id": source_id,
        }

    try:
        if retryable_operation_id is not None:
            reached = orch.resume(operation_id=operation_id, file_stager=stager)
            if reached is not SourceRetirementState.COMPLETED:
                operation = next(
                    row
                    for row in extraction.list_source_retirement(source_id)
                    if row["operation_id"] == operation_id
                )
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "retirement_failed_retryable",
                        "message": str(
                            operation.get("failure_reason")
                            or "retirement did not complete"
                        ),
                        "operation_id": operation_id,
                    },
                )
            operation = next(
                row
                for row in extraction.list_source_retirement(source_id)
                if row["operation_id"] == operation_id
            )
            return {
                "ok": True,
                "operation_id": operation_id,
                "state": operation["state"],
                "source_id": source_id,
            }
        current = orch._current_state(operation_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    reached_state = current.value
    try:
        for stage in _FORWARD_CHAIN:
            reached = orch.advance(
                operation_id=operation_id,
                target_state=stage,
                file_stager=stager,
            )
            reached_state = reached.value
            if reached is SourceRetirementState.FAILED_RETRYABLE:
                operation = extraction.list_source_retirement(source_id)[0]
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "retirement_failed_retryable",
                        "message": str(
                            operation.get("failure_reason")
                            or "retirement did not complete"
                        ),
                        "operation_id": operation_id,
                    },
                )
            if reached_state == target_state.value:
                break
    except AuthoritativeBlock as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "requires_governed_retirement",
                "message": str(exc),
                "operation_id": operation_id,
            },
        ) from exc
    except RepresentationMissing as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "representation_missing",
                "message": str(exc),
                "source_id": source_id,
            },
        ) from exc
    except InvalidStateTransition as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_state_transition", "message": str(exc)},
        ) from exc
    except TerminalStateReached as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "terminal_state", "message": str(exc)},
        ) from exc

    operation = extraction.list_source_retirement(source_id)[0]
    return {
        "ok": True,
        "operation_id": operation_id,
        "state": operation["state"],
        "source_id": source_id,
    }


@router.get("/api/knowledge/source/{source_id}/retirement")
async def list_knowledge_source_retirement(source_id: str, request: Request) -> dict:
    """Return the most recent retirement operation for a source."""
    route_path = "/api/knowledge/source/{source_id}/retirement"
    _authz(request, "GET", route_path)
    extraction = _extraction_store()
    rows = extraction.list_source_retirement(source_id)
    return {
        "ok": True,
        "source_id": source_id,
        "operations": [_serialise_operation(row) for row in rows],
    }


@router.post("/api/knowledge/source/{source_id}/retirement/{operation_id}/resume")
async def resume_knowledge_source_retirement(
    source_id: str, operation_id: str, request: Request
) -> dict:
    """Resume an interrupted operation forward through the saga."""
    route_path = "/api/knowledge/source/{source_id}/retirement/{operation_id}/resume"
    ctx = _authz(request, "POST", route_path)
    extraction = _extraction_store()
    stager = _stager_for(ctx)
    orch = _orchestrator(
        ctx, extraction_store=extraction, file_stager=stager
    )
    try:
        final = orch.resume(operation_id=operation_id, file_stager=stager)
    except AuthoritativeBlock as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "requires_governed_retirement",
                "message": str(exc),
                "operation_id": operation_id,
            },
        ) from exc
    except RepresentationMissing as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "representation_missing",
                "message": str(exc),
                "source_id": source_id,
            },
        ) from exc
    except (InvalidStateTransition, TerminalStateReached) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": type(exc).__name__, "message": str(exc)},
        ) from exc

    return {
        "ok": True,
        "operation_id": operation_id,
        "state": final.value,
        "source_id": source_id,
    }


@router.get("/api/knowledge/source/{source_id}/retirement/preview")
async def preview_knowledge_source_retirement(source_id: str, request: Request) -> dict:
    """Return cascade scope, severity tier, and authoritative count."""
    route_path = "/api/knowledge/source/{source_id}/retirement/preview"
    _authz(request, "GET", route_path)
    extraction = _extraction_store()
    return {
        "ok": True,
        **_compute_preview(extraction._conn, source_id),
    }


__all__ = ["router"]
