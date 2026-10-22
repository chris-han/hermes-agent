"""Authenticated documentary review; no endpoint runs an inspection or activates law."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import require_route_authorization
from contracts.ontology_building import ArtifactPinV1
from contracts.tender_review import TenderReviewRecordCommandV1, TenderReviewReportCommandV1, TenderCandidateReportCommandV1
from services.tender_review_service import TenderReviewService
from services.runtime_errors import RuntimeContextError

router = APIRouter(tags=["tender-review"])


def _query(request, allowed, required=()):
    values = request.query_params
    if set(values) - set(allowed) or any(len(values.getlist(key)) != 1 for key in values) or any(not values.get(key) for key in required):
        raise HTTPException(400, "INVALID_QUERY")
    return dict(values)


@contextmanager
def _owner(request, *, cursor_ref=None, native=False, adoption=False):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.workspace_artifact_service import WorkspaceArtifactService
    from services.workspace_artifact_preparation_service import WorkspaceArtifactPreparationService, KnowledgeDesignSourcePreparationAdapter
    from storage.knowledge_extraction_store import KnowledgeExtractionStore
    from storage.workbench_projection_cursor_store import WorkbenchProjectionCursorStore
    ctx = require_route_authorization(request, authorization_class=RouteAuthorizationClass.SELF_ONLY,
                                       route_hint=f"{request.method} {request.url.path}")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(401, "AUTHORITY_CONTEXT_REQUIRED")
    owners = None
    extraction = None
    journey = None
    try:
        owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
        cursors = WorkbenchProjectionCursorStore(build_store=owners.build_store, authority_resolver=owners.authority_resolver,
                                                 session_resolver=owners.session_resolver, session_ref=owners.session_ref)
        artifacts = WorkspaceArtifactService(build_store=owners.build_store, authority_resolver=owners.authority_resolver,
                                             max_bytes=int(os.environ.get("SEMANTIER_WORKSPACE_ARTIFACT_MAX_BYTES", "20971520")),
                                             cursor_store=cursors, cursor_ref=cursor_ref)
        extraction = KnowledgeExtractionStore(owners.source_registry._db_path)
        preparation = WorkspaceArtifactPreparationService(artifact_service=artifacts,
            design_adapter=KnowledgeDesignSourcePreparationAdapter(artifact_service=artifacts,
                knowledge_store=owners.source_registry, extraction_store=extraction))
        inspection = None
        if native:
            from services.tender_journey_service import TenderJourneyService
            from agents.webapi.tender_document_review_routes import _runtime_semantic_inspection_service
            journey = TenderJourneyService(ctx.workspace_root)
            inspection = _runtime_semantic_inspection_service(ctx, read_only=True)
        service = TenderReviewService(artifact_service=artifacts, preparation_service=preparation, session_ref=owners.session_ref,
                                       journey_service=journey, inspection_service=inspection)
        if adoption:
            from services.tender_owner_adoption_service import TenderOwnerAdoptionService
            from storage.graph_intent_brief_store import GraphIntentBriefStore
            service._adoption_service = TenderOwnerAdoptionService(
                store=owners.build_store, authority_resolver=owners.authority_resolver,
                intent_store=GraphIntentBriefStore(ctx.workspace_root, organization_id=str(ctx.organization_id),
                    workspace_id=str(ctx.workspace_id), source_registry=owners.source_registry))
            service._adoption_actor = owners.actor_context
        yield ctx, service
    except HTTPException:
        raise
    except (RuntimeContextError, PermissionError) as exc:
        raise HTTPException(403, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, "DOCUMENTARY_ARTIFACT_NOT_FOUND") from exc
    except ValidationError as exc:
        raise HTTPException(400, "INVALID_DOCUMENTARY_REQUEST") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    finally:
        if journey is not None:
            journey.close()
        if extraction is not None:
            extraction._conn.close()
        if owners is not None:
            owners.build_store.close()
            owners.source_registry.close()


def _pin(kind, ref, digest):
    try:
        return ArtifactPinV1(artifact_kind=kind, artifact_ref=ref, artifact_hash=digest)
    except ValidationError as exc:
        raise HTTPException(400, "INVALID_DOCUMENTARY_PIN") from exc


async def _command(request, model):
    from agents.webapi.ontology_workbench_context_routes import _unique_json_object
    _query(request, ())
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 262144:
            raise HTTPException(400, "REQUEST_TOO_LARGE")
    try:
        return model.model_validate(json.loads(body, object_pairs_hook=_unique_json_object))
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(400, "INVALID_DOCUMENTARY_REQUEST") from exc


@router.get("/api/tender-review/context")
def context(request: Request):
    query = _query(request, ("cursor_ref", "journey_ref", "episode_ref"))
    if "episode_ref" in query and "journey_ref" not in query:
        raise HTTPException(400, "PAIRED_EPISODE_SELECTION_REQUIRED")
    with _owner(request, cursor_ref=query.get("cursor_ref"), native="journey_ref" in query, adoption=True) as (ctx, service):
        # Reuse the existing artifact owner for initial cursor genesis.
        service._artifacts.list(ctx=ctx)
        journey = None
        episode_ref = query.get("episode_ref")
        if query.get("journey_ref"):
            head = service._journeys._store.get_session_journey(organization_id=str(ctx.organization_id),
                workspace_id=str(ctx.workspace_id), actor_ref="user:" + str(ctx.user_id), session_ref=service._session)
            if head is None or head.journey_id != query["journey_ref"]:
                raise ValueError("EPISODE_SESSION_MISMATCH")
            journey = service._journeys.project(head).model_dump(mode="json")
            if episode_ref is None and head.current_phase == "REVIEW_RESULTS":
                episode_ref = head.latest_transition_id
        result = service.context(ctx=ctx, cursor_ref=query.get("cursor_ref"),
            journey_ref=query.get("journey_ref") if episode_ref else None, episode_ref=episode_ref)
        if journey is not None:
            result["native_journey"] = journey
        result["owner_adoptions"] = _adoption_entries(service)
        try:
            service._adoption_service.authority.require_current_owner(actor_context=service._adoption_actor)
            result["owner_adoption_allowed"] = True
        except (RuntimeContextError, PermissionError):
            result["owner_adoption_allowed"] = False
        result["requested_session_ref"] = request.headers["X-Workbench-Session-Ref"]
        return result


@router.post("/api/tender-review/records", status_code=201)
async def save_record(request: Request):
    command = await _command(request, TenderReviewRecordCommandV1)
    with _owner(request, cursor_ref=command.cursor_ref, native=command.subject.kind == "GOVERNED_FINDING") as (ctx, service):
        return service.save_record(ctx=ctx, command=command).model_dump(mode="json")


@router.get("/api/tender-review/records")
def records(request: Request):
    query = _query(request, ("target_artifact_ref", "target_artifact_hash", "limit", "continuation"), ("target_artifact_ref", "target_artifact_hash"))
    try:
        limit = int(query.get("limit", "100"))
        if not 1 <= limit <= 100:
            raise ValueError()
    except ValueError as exc:
        raise HTTPException(400, "INVALID_LIST_LIMIT") from exc
    pin = _pin("workspace_artifact", query["target_artifact_ref"], query["target_artifact_hash"])
    with _owner(request) as (ctx, service):
        return service.list_records(ctx=ctx, target_pin=pin, limit=limit, continuation=query.get("continuation"))


@router.get("/api/tender-review/records/{record_ref}")
def record(record_ref: str, request: Request):
    query = _query(request, ("hash",), ("hash",))
    pin = _pin("tender_review_record", record_ref, query["hash"])
    with _owner(request) as (ctx, service):
        return service.get_record(ctx=ctx, pin=pin).model_dump(mode="json")


@router.post("/api/tender-review/reports", status_code=201)
async def save_report(request: Request):
    command = await _command(request, TenderReviewReportCommandV1)
    with _owner(request, native=command.episode_pin is not None) as (ctx, service):
        return service.save_report(ctx=ctx, command=command).model_dump(mode="json")


@router.post("/api/tender-review/candidate-reports", status_code=201)
async def save_candidate_report(request: Request):
    # Deliberately async/direct: native SIGALRM cannot run in FastAPI's worker pool.
    command = await _command(request, TenderCandidateReportCommandV1)
    with _owner(request, cursor_ref=command.cursor_ref, native=command.episode_pin is not None) as (ctx, service):
        return service.save_report(ctx=ctx, command=command).model_dump(mode="json")


@router.get("/api/tender-review/reports/{report_ref}")
def report(report_ref: str, request: Request):
    query = _query(request, ("hash",), ("hash",))
    pin = _pin("tender_review_report", report_ref, query["hash"])
    with _owner(request) as (ctx, service):
        return service.get_report(ctx=ctx, pin=pin).model_dump(mode="json")


@router.get("/api/tender-review/reports/{report_ref}/export")
def export(report_ref: str, request: Request):
    query = _query(request, ("hash",), ("hash",))
    pin = _pin("tender_review_report", report_ref, query["hash"])
    with _owner(request) as (ctx, service):
        raw = service.export_report(ctx=ctx, pin=pin)
        filename = "tender-review-" + pin.artifact_hash.removeprefix("sha256:")[:16] + ".json"
        return Response(raw, media_type="application/json", headers={"Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store"})


@router.post("/api/tender-review/docx", status_code=201)
async def publish_docx(request: Request):
    pin = await _command(request, ArtifactPinV1)
    with _owner(request) as (ctx, service):
        return service.export_native_docx(ctx=ctx, report_pin=pin).model_dump(mode="json")


@router.get("/api/tender-review/docx/{derivative_ref}/download")
def download_docx(derivative_ref: str, request: Request):
    query = _query(request, ("hash",), ("hash",))
    pin = _pin("tender_review_docx_derivative", derivative_ref, query["hash"])
    with _owner(request) as (ctx, service):
        derivative, raw = service.download_native_docx(ctx=ctx, pin=pin)
        filename = "tender-review-" + derivative.content_hash[7:23] + ".docx"
        return Response(raw,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store"})


class _AdoptCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposal_pin: ArtifactPinV1
    request_id: str = Field(min_length=1, max_length=200)


class _RevokeCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    adoption_pin: ArtifactPinV1
    expected_head_pin: ArtifactPinV1
    request_id: str = Field(min_length=1, max_length=200)


class _ProposeCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_pin: ArtifactPinV1
    plan_pin: ArtifactPinV1
    rule_ref: str = Field(min_length=1)
    canonical_pattern_ref: str = Field(min_length=1)
    context_pin: ArtifactPinV1
    ontology_pin: ArtifactPinV1


def _adoption_entries(service):
    return [{key: value.model_dump(mode="json", by_alias=True) if hasattr(value, "model_dump") else value
             for key, value in entry.items()}
            for entry in service._adoption_service.list(actor_context=service._adoption_actor)]


@router.post("/api/tender-review/owner-adoption-proposals", status_code=201)
async def propose_owner_adoption(request: Request):
    command = await _command(request, _ProposeCommand)
    with _owner(request, adoption=True) as (_ctx, service):
        return service._adoption_service.propose(actor_context=service._adoption_actor,
            **{key: getattr(command, key) for key in type(command).model_fields}).model_dump(mode="json", by_alias=True)


@router.post("/api/tender-review/owner-adoptions", status_code=201)
async def adopt_owner_bundle(request: Request):
    command = await _command(request, _AdoptCommand)
    with _owner(request, adoption=True) as (_ctx, service):
        return service._adoption_service.adopt(actor_context=service._adoption_actor,
            proposal_pin=command.proposal_pin, request_id=command.request_id,
            recorded_at=datetime.now(timezone.utc)).model_dump(mode="json", by_alias=True)


@router.post("/api/tender-review/owner-adoptions/revoke", status_code=201)
async def revoke_owner_bundle(request: Request):
    command = await _command(request, _RevokeCommand)
    with _owner(request, adoption=True) as (_ctx, service):
        return service._adoption_service.revoke(actor_context=service._adoption_actor,
            adoption_pin=command.adoption_pin, expected_head_pin=command.expected_head_pin,
            request_id=command.request_id, recorded_at=datetime.now(timezone.utc)).model_dump(mode="json", by_alias=True)
