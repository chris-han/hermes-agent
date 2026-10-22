"""Private governed owner composition for runtime advisory trials."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import ValidationError

from agents.auth_session import RequestContext
from runtime_paths import workspace_session_segment
from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import (
    require_authenticated_context,
    require_route_authorization,
)
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_workbench_context import (
    FocusEvidenceActionEnvelopeV6,
    RuntimeReportCommandV1,
    RuntimeExportCommandV1,
    RuntimeCorrectionCommandV1,
    WorkbenchActionEnvelopeV6,
    WorkbenchSelectionRequestV1,
    WorkbenchActionReceiptV6,
    CompletedWorkbenchActionResultV6,
    WorkbenchRepresentationV2,
    serialize_unified_workbench_context_v3,
    WorkbenchArtifactReadRequestV2,
    WorkbenchArtifactProjectionV2,
    TemporalActionEnvelopeV1,
    TemporalActionResultV1,
)
from agents.workspace_uploads import (
    WorkspaceUploadError,
    read_temporal_workspace_upload,
)
from gateway.execution_boundary import BoundaryPathRejected
from services.runtime_temporal_source_service import RuntimeTemporalSourceService
from services.runtime_document_evaluation_service import (
    RuntimeDocumentEvaluationService,
)
from storage.runtime_semantic_evaluation_store import RuntimeSemanticEvaluationStore
from services.runtime_context_proposal_provider import (
    RuntimeContextProposalProviderRegistry,
)
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from services.runtime_errors import RuntimeContextError
from services.runtime_session_service import (
    GovernedRuntimeSessionResolver,
    runtime_context_dto_from_request_context,
)
from services.workbench_context_resolver import WorkbenchContextResolver
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path

SUPPORTED_WORKBENCH_ACTIONS_V6 = frozenset({
    "START_RUNTIME",
    "FOCUS_EVIDENCE",
    "CREATE_REPORT",
    "EXPORT_REPORT",
    "SUBMIT_RUNTIME_CORRECTION",
    "SUBMIT_SEMANTIC_GRAPH_EDIT",
    "REVIEW_RUNTIME_CORRECTION",
    "EVALUATE_RUNTIME_REPAIR_PAIR",
    "REQUEST_RUNTIME_LEARNING_FUTURE_RUN",
})


_PROVIDER_REGISTRY = (
    RuntimeContextProposalProviderRegistry.compose_from_packaged_resources()
)
_ACTION_PATH = "/api/workbench/actions"
_ACTION_VERSION_V6 = "workbench-action.v6"
_ARTIFACT_READ_PATH = "/api/workbench-contexts/artifact"

router = APIRouter()


def _workspace_artifact_error(code: str, status: int):
    from contracts.ontology_workbench_context import WorkspaceArtifactErrorV1

    result = WorkspaceArtifactErrorV1(
        schema_version="workspace_artifact_error.v1", code=code,
        retryable=status == 503 or code.startswith("STALE_"),
        current_collection_pin=None, current_collection_revision=None,
        current_preparation_pin=None, current_preparation_revision=None,
        current_selection_pin=None, current_selection_revision=None,
        current_binding_set_pin=None, current_binding_set_revision=None,
        current_cursor_pin=None, current_material_projection_hash=None,
        current_material_context_revision=None, detail=code,
    )
    return JSONResponse(status_code=status, content=result.model_dump(mode="json"), headers={"Cache-Control": "no-store"})


@router.get("/api/workbench/artifacts")
@router.post("/api/workbench/artifacts")
@router.post("/api/workbench/artifacts/select")
@router.post("/api/workbench/artifacts/prepare")
@router.put("/api/workbench/artifact-bindings/{lifecycle}/{role}")
@router.delete("/api/workbench/artifact-bindings/{lifecycle}/{role}")
async def workspace_artifacts(request: Request):
    import os
    import sqlite3
    from storage.hosted_postgres import integrity_errors
    from starlette.datastructures import UploadFile
    from contracts import ontology_workbench_context as wire
    from services.workspace_artifact_service import WorkspaceArtifactService
    from services.workspace_artifact_preparation_service import WorkspaceArtifactPreparationService, KnowledgeDesignSourcePreparationAdapter
    from services.lifecycle_artifact_binding_service import LifecycleArtifactBindingService
    from storage.knowledge_extraction_store import KnowledgeExtractionStore
    from storage.workbench_projection_cursor_store import WorkbenchProjectionCursorStore

    form = None
    try:
        ctx = await run_in_threadpool(require_route_authorization, request, authorization_class=RouteAuthorizationClass.SELF_ONLY, route_hint=request.url.path)
        sessions = request.headers.getlist("X-Workbench-Session-Ref")
        if len(sessions) != 1 or not sessions[0].strip():
            return _workspace_artifact_error("AUTHORITY_CONTEXT_REQUIRED", 401)
        max_bytes = int(os.environ.get("SEMANTIER_WORKSPACE_ARTIFACT_MAX_BYTES", "20971520"))
        if max_bytes <= 0:
            raise OSError("invalid upload policy")
        cursor_values = request.query_params.getlist("cursor_ref")
        if len(cursor_values) > 1 or set(request.query_params) - {"cursor_ref"}:
            return _workspace_artifact_error("ARTIFACT_PIN_MISMATCH", 400)
        command = None
        stream = None
        if request.method != "GET":
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > max_bytes + 65536:
                    return _workspace_artifact_error("INVALID_BYTE_LENGTH", 400)
                body.extend(chunk)
            request._body = bytes(body)
            if request.method == "POST" and request.url.path == "/api/workbench/artifacts":
                form = await request.form(max_files=1, max_fields=1, max_part_size=max_bytes + 65536)
                if len(form.multi_items()) != 2 or set(form) != {"command", "file"} or not isinstance(form["file"], UploadFile) or not isinstance(form["command"], str):
                    return _workspace_artifact_error("INVALID_MEDIA_TYPE", 400)
                command = wire.UploadWorkspaceArtifactCommandV1.model_validate(json.loads(form["command"], object_pairs_hook=_unique_json_object))
                stream = form["file"].file
            else:
                model = (wire.ClearLifecycleArtifactBindingCommandV1 if request.method == "DELETE" else wire.BindLifecycleArtifactCommandV1 if request.method == "PUT" else wire.SelectWorkspaceArtifactCommandV1 if request.url.path.endswith("/select") else wire.PrepareWorkspaceArtifactCommandV1)
                command = model.model_validate(json.loads(body, object_pairs_hook=_unique_json_object))
            if request.path_params and (command.lifecycle, command.role) != (request.path_params["lifecycle"], request.path_params["role"]):
                return _workspace_artifact_error("INVALID_LIFECYCLE_ROLE", 400)
        cursor_ref = cursor_values[0] if cursor_values else getattr(command, "cursor_ref", None)
        def execute_in_worker():
            owners = extraction = None
            try:
                owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web", authorization_class="WORKSPACE_LOCAL")
                cursors = WorkbenchProjectionCursorStore(build_store=owners.build_store, authority_resolver=owners.authority_resolver, session_resolver=owners.session_resolver, session_ref=owners.session_ref)
                artifacts = WorkspaceArtifactService(build_store=owners.build_store, authority_resolver=owners.authority_resolver, max_bytes=max_bytes, cursor_store=cursors, cursor_ref=cursor_ref)
                if request.method == "GET":
                    result = artifacts.list(ctx=ctx)
                elif isinstance(command, wire.UploadWorkspaceArtifactCommandV1):
                    result = artifacts.upload(ctx=ctx, command=command, byte_stream=stream)
                elif isinstance(command, wire.SelectWorkspaceArtifactCommandV1):
                    result = artifacts.select(ctx=ctx, command=command)
                else:
                    knowledge = owners.source_registry
                    extraction = KnowledgeExtractionStore(knowledge._db_path)
                    preparation = WorkspaceArtifactPreparationService(artifact_service=artifacts, design_adapter=KnowledgeDesignSourcePreparationAdapter(artifact_service=artifacts, knowledge_store=knowledge, extraction_store=extraction))
                    if isinstance(command, wire.PrepareWorkspaceArtifactCommandV1):
                        result = preparation.prepare(ctx=ctx, command=command)
                        if result.outcome == "UNSUPPORTED":
                            return _workspace_artifact_error("TARGET_UNSUPPORTED" if command.purpose == "RUNTIME_TARGET" else "UNSUPPORTED_SOURCE_PREPARATION", 422)
                        if result.outcome == "FAILED":
                            return _workspace_artifact_error("SOURCE_PREPARATION_FAILED", 409)
                    else:
                        bindings = LifecycleArtifactBindingService(artifact_service=artifacts, preparation_service=preparation)
                        result = bindings.clear(ctx=ctx, command=command) if request.method == "DELETE" else bindings.bind(ctx=ctx, command=command)
                return JSONResponse(content=result.model_dump(mode="json"), headers={"Cache-Control": "no-store"})
            finally:
                if extraction is not None:
                    extraction._conn.close()
                if owners is not None:
                    owners.build_store.close()
                    owners.source_registry.close()
        return await run_in_threadpool(execute_in_worker)

    except HTTPException as exc:
        return _workspace_artifact_error("AUTHORITY_CONTEXT_REQUIRED" if exc.status_code == 401 else "FORBIDDEN", 401 if exc.status_code == 401 else 403)
    except RuntimeContextError:
        return _workspace_artifact_error("FORBIDDEN", 403)
    except ValidationError:
        return _workspace_artifact_error("ARTIFACT_PIN_MISMATCH", 400)
    except ValueError as exc:
        code = str(exc)
        allowed = wire.WorkspaceArtifactErrorV1.model_fields["code"].annotation.__args__
        if code not in allowed:
            code = "ARTIFACT_PIN_MISMATCH"
        status = 422 if code in {"TARGET_UNSUPPORTED", "UNSUPPORTED_SOURCE_PREPARATION"} else 404 if code in {"ARTIFACT_NOT_FOUND", "BINDING_NOT_FOUND"} else 400 if code in {"INVALID_MEDIA_TYPE", "INVALID_BYTE_LENGTH", "INVALID_LIFECYCLE_ROLE", "ARTIFACT_HASH_MISMATCH"} else 409
        return _workspace_artifact_error(code, status)
    except integrity_errors():
        return _workspace_artifact_error("PERSISTENCE_CONFLICT", 409)
    except (OSError, sqlite3.OperationalError):
        return _workspace_artifact_error("TEMPORARY_STORAGE_FAILURE", 503)
    finally:
        if form is not None:
            await form.close()



@dataclass(frozen=True)
class _WorkbenchOwners:
    actor_context: RuntimeContextDTO
    session_ref: str
    build_store: OntologyBuildStore
    source_registry: KnowledgeBuilderStore
    authority_resolver: RuntimeAuthorityResolver
    session_resolver: GovernedRuntimeSessionResolver


def _temporal_upload_boundary(*, ctx: RequestContext, session_ref: str):
    from agents.webapi.session_routes import _session_execution_boundary

    segment = workspace_session_segment(str(ctx.workspace_id), session_ref)
    root = ctx.workspace_root.resolve()
    uploads = root / "sessions" / segment / "uploads"
    if uploads.resolve() != uploads:
        raise RuntimeContextError("FORBIDDEN_SCOPE")
    boundary = _session_execution_boundary(ctx, session_id=session_ref)
    if boundary is None or boundary.paths.uploads_root != uploads:
        raise RuntimeContextError("FORBIDDEN_SCOPE")
    return boundary, segment


def _workbench_owner_context(
    *,
    ctx: RequestContext,
    session_locator: str,
    channel: str | None,
    authorization_class: str = "TENANT_MEMBER",
) -> _WorkbenchOwners:
    if not isinstance(ctx, RequestContext) or not ctx.authenticated or not ctx.user_id:
        raise RuntimeContextError("FORBIDDEN_ACTOR")
    if not ctx.organization_id or not ctx.workspace_id:
        raise RuntimeContextError("MISSING_WORKSPACE")

    source_registry = KnowledgeBuilderStore(workspace_root=ctx.workspace_root)
    build_store = None
    try:
        build_store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
        authority_resolver = RuntimeAuthorityResolver(
            graph_intent_store=None,
            ontology_build_store=build_store,
            runtime_context_store=None,
            authority_store=None,
            provider_registry=_PROVIDER_REGISTRY,
        )
        seed = RuntimeContextDTO(
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id),
            actor_id=str(ctx.user_id),
        )
        authority_resolver.require_current_actor(actor_context=seed, authorization_class=authorization_class)
        session_resolver = GovernedRuntimeSessionResolver(
            authority_resolver=authority_resolver
        )
        canonical = session_resolver.resolve(
            actor_context=seed,
            session_ref=session_locator,
            authorization_class=authorization_class,
        )
        actor = runtime_context_dto_from_request_context(
            ctx=ctx,
            session_id=canonical,
            channel=channel,
        )
        return _WorkbenchOwners(
            actor_context=actor,
            session_ref=canonical,
            build_store=build_store,
            source_registry=source_registry,
            authority_resolver=authority_resolver,
            session_resolver=session_resolver,
        )
    except BaseException:
        try:
            if build_store is not None:
                build_store.close()
        finally:
            source_registry.close()
        raise


def dispatch_temporal_action(
    *, ctx: RequestContext, envelope: TemporalActionEnvelopeV1
):
    context = _workbench_owner_context(
        ctx=ctx, session_locator=envelope.session_ref, channel="web"
    )
    try:
        source = RuntimeTemporalSourceService(
            build_store=context.build_store,
            authority_resolver=context.authority_resolver,
            session_resolver=context.session_resolver,
        )
        payload = envelope.payload
        state = None
        if payload.command == "CAPTURE_RUNTIME_TEMPORAL_SOURCE":
            boundary, _segment = _temporal_upload_boundary(
                ctx=ctx, session_ref=context.session_ref
            )
            content = read_temporal_workspace_upload(
                boundary=boundary,
                upload_ref=payload.upload_ref,
                upload_hash=payload.upload_hash,
            )
            output = source.capture_notebook(
                actor_context=context.actor_context,
                content=content,
                request_id=payload.request_id,
            )
        elif payload.command == "IMPORT_RUNTIME_TEMPORAL_SOURCE":
            output = source.import_source(
                actor_context=context.actor_context,
                source=payload.source,
                source_cell_index=payload.source_cell_index,
                request_id=payload.request_id,
            )
        else:
            evaluations = RuntimeSemanticEvaluationStore(
                build_store=context.build_store,
                authority_resolver=context.authority_resolver,
                session_resolver=context.session_resolver,
            )
            service = RuntimeDocumentEvaluationService(
                store=evaluations,
                source_service=source,
                authority_resolver=context.authority_resolver,
                session_resolver=context.session_resolver,
            )
            output = service.resume(
                actor_context=context.actor_context, run=payload.run
            )
            state = output
        return TemporalActionResultV1.model_validate(
            dict(
                schema_version="temporal_action_result.v1",
                command=payload.command,
                output=output,
                state=state,
            )
        )
    finally:
        try:
            context.build_store.close()
        finally:
            context.source_registry.close()


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


async def _temporal_action(request: Request):
    try:
        ctx = require_route_authorization(
            request,
            authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
            route_hint=_ACTION_PATH,
        )
    except HTTPException as exc:
        if exc.status_code in {403, 409}:
            raise HTTPException(403, "FORBIDDEN_SCOPE") from None
        raise
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > 1_500_000:
            raise HTTPException(413, "RESOURCE_LIMIT_EXCEEDED")
        data.extend(chunk)
    try:
        envelope = TemporalActionEnvelopeV1.model_validate(
            json.loads(data, object_pairs_hook=_unique_json_object)
        )
    except (ValueError, TypeError, RecursionError):
        raise HTTPException(422, "MALFORMED_CONTRACT") from None
    try:
        result = dispatch_temporal_action(ctx=ctx, envelope=envelope)
    except WorkspaceUploadError as exc:
        raise HTTPException(exc.status_code, exc.code) from None
    except BoundaryPathRejected:
        raise HTTPException(404, "NOT_FOUND") from None
    except RuntimeContextError as exc:
        if exc.code in {
            "FORBIDDEN_ACTOR",
            "FORBIDDEN_SESSION",
            "SESSION_NOT_FOUND",
            "FORBIDDEN_SCOPE",
            "MISSING_WORKSPACE",
        }:
            raise HTTPException(403, "FORBIDDEN_SCOPE") from None
        if exc.code in {"NOT_FOUND", "OBJECT_NOT_FOUND", "HASH_MISMATCH"}:
            raise HTTPException(404, "NOT_FOUND") from None
        if exc.code in {"TEMPORAL_SOURCE_LIMIT_EXCEEDED", "RESOURCE_LIMIT_EXCEEDED"}:
            raise HTTPException(413, "RESOURCE_LIMIT_EXCEEDED") from None
        if exc.code == "TEMPORAL_SOURCE_INVALID":
            raise HTTPException(422, "INVALID_SOURCE") from None
        raise HTTPException(409, "INVALID_STATE") from None
    except Exception:
        raise HTTPException(500, "INTERNAL_ERROR") from None
    return JSONResponse(
        content=result.model_dump(mode="json"),
        headers={
            "Workbench-Action-Version": "temporal-context.v1",
            "Cache-Control": "no-store",
        },
    )


def read_temporal_workbench_artifact(
    *, ctx: RequestContext, request: WorkbenchArtifactReadRequestV2
):
    context = _workbench_owner_context(
        ctx=ctx, session_locator=request.session_ref, channel="web"
    )
    try:
        source = RuntimeTemporalSourceService(
            build_store=context.build_store,
            authority_resolver=context.authority_resolver,
            session_resolver=context.session_resolver,
        )
        evaluations = RuntimeSemanticEvaluationStore(
            build_store=context.build_store,
            authority_resolver=context.authority_resolver,
            session_resolver=context.session_resolver,
        )
        resolver = WorkbenchContextResolver(
            store=context.build_store,
            source_service=source,
            evaluation_store=evaluations,
            authority_resolver=context.authority_resolver,
        )
        return resolver.read_temporal_artifact(
            actor_context=context.actor_context,
            pin=ArtifactPinV1.model_validate(
                request.model_dump(
                    include={"artifact_kind", "artifact_ref", "artifact_hash"}
                )
            ),
        )
    finally:
        try:
            context.build_store.close()
        finally:
            context.source_registry.close()


@router.get(_ARTIFACT_READ_PATH, name="read_workbench_artifact")
def workbench_artifact(request: Request):
    require_authenticated_context(request)
    if request.headers.getlist("Accept-Workbench-Context-Version") != ["artifact-read.v2"]:
        raise HTTPException(status_code=422, detail="MALFORMED_CONTRACT")

    items = request.query_params.multi_items()
    expected = {"session_ref", "artifact_kind", "artifact_ref", "artifact_hash"}
    if len(items) != len(expected) or {key for key, _ in items} != expected:
        raise HTTPException(status_code=422, detail="MALFORMED_CONTRACT")
    try:
        parsed = WorkbenchArtifactReadRequestV2.model_validate(dict(items))
    except ValidationError:
        raise HTTPException(
            status_code=422,
            detail="MALFORMED_CONTRACT",
        ) from None

    try:
        ctx = require_route_authorization(
            request,
            authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
            route_hint=_ARTIFACT_READ_PATH,
        )
    except HTTPException as exc:
        if exc.status_code in {403, 409}:
            raise HTTPException(status_code=403, detail="FORBIDDEN_SCOPE") from None
        raise
    try:
        result = read_temporal_workbench_artifact(ctx=ctx, request=parsed)
    except RuntimeContextError as exc:
        if exc.code in {"NOT_FOUND", "OBJECT_NOT_FOUND", "HASH_MISMATCH"}:
            raise HTTPException(404, "NOT_FOUND") from None
        if exc.code in {
            "FORBIDDEN_SCOPE",
            "FORBIDDEN_SESSION",
            "FORBIDDEN_ACTOR",
            "SESSION_NOT_FOUND",
            "MISSING_WORKSPACE",
        }:
            raise HTTPException(403, "FORBIDDEN_SCOPE") from None
        raise HTTPException(409, "INTEGRITY_UNPROVABLE") from None
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="INTERNAL_ERROR",
        ) from None
    return JSONResponse(
        content=result.model_dump(mode="json"),
        headers={
            "Workbench-Context-Version": "artifact-read.v2",
            "Cache-Control": "no-store",
        },
    )


@router.post("/api/workbench-contexts/resolve", name="resolve_workbench_context_v3")
async def resolve_workbench_context_v3_route(request: Request):
    from services.workspace_artifact_service import WorkspaceArtifactService
    from storage.workbench_projection_cursor_store import WorkbenchProjectionCursorStore

    try:
        ctx = require_route_authorization(
            request,
            authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
            route_hint="/api/workbench-contexts/resolve",
        )
        sessions = request.headers.getlist("X-Workbench-Session-Ref")
        if len(sessions) != 1 or not sessions[0].strip():
            raise HTTPException(401, "AUTHORITY_CONTEXT_REQUIRED")
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > 256_000:
                raise HTTPException(413, "RESOURCE_LIMIT_EXCEEDED")
            data.extend(chunk)
        selection = WorkbenchSelectionRequestV1.model_validate(
            json.loads(data, object_pairs_hook=_unique_json_object)
        )
        owners = _workbench_owner_context(
            ctx=ctx, session_locator=sessions[0], channel="web"
        )
        extraction = None
        try:
            cursors = WorkbenchProjectionCursorStore(
                build_store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                session_resolver=owners.session_resolver,
                session_ref=owners.session_ref,
            )
            current_head = cursors.get(ctx=ctx, cursor_ref=selection.cursor_ref)
            artifacts = WorkspaceArtifactService(
                build_store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                max_bytes=20 * 1024 * 1024,
                cursor_store=cursors,
                cursor_ref=selection.cursor_ref,
            )
            resolver_kwargs = dict(
                store=owners.build_store,
                authority_resolver=owners.authority_resolver,
            )
            if (
                selection.evidence_focus is not None
                or current_head.evidence_focus_pin is not None
            ):
                from services.workspace_artifact_preparation_service import (
                    KnowledgeDesignSourcePreparationAdapter,
                    WorkspaceArtifactPreparationService,
                )
                from storage.knowledge_extraction_store import KnowledgeExtractionStore

                knowledge = owners.source_registry
                extraction = KnowledgeExtractionStore(
                    knowledge._db_path
                )
                resolver_kwargs["preparation_service"] = WorkspaceArtifactPreparationService(
                    artifact_service=artifacts,
                    design_adapter=KnowledgeDesignSourcePreparationAdapter(
                        artifact_service=artifacts,
                        knowledge_store=knowledge,
                        extraction_store=extraction,
                    ),
                )
            if selection.lifecycle == "RUNTIME" and selection.runtime_context_pin is not None:
                from services.runtime_temporal_source_service import RuntimeTemporalSourceService
                from storage.runtime_semantic_evaluation_store import RuntimeSemanticEvaluationStore

                runtime_source_service = RuntimeTemporalSourceService(
                    build_store=owners.build_store,
                    authority_resolver=owners.authority_resolver,
                    session_resolver=owners.session_resolver,
                )
                runtime_evaluation_store = RuntimeSemanticEvaluationStore(
                    build_store=owners.build_store,
                    authority_resolver=owners.authority_resolver,
                    session_resolver=owners.session_resolver,
                )
                resolver_kwargs.update(
                    source_service=runtime_source_service,
                    evaluation_store=runtime_evaluation_store,
                )
            resolver = WorkbenchContextResolver(**resolver_kwargs)
            result = resolver.resolve_workbench_context_v3(
                ctx=ctx,
                cursor_store=cursors,
                artifact_service=artifacts,
                selection=selection,
            )
            return JSONResponse(
                content=serialize_unified_workbench_context_v3(result),
                headers={"Workbench-Context-Version": "unified-workbench.v3", "Cache-Control": "no-store"},
            )
        finally:
            if extraction is not None:
                extraction._conn.close()
            owners.build_store.close()
            owners.source_registry.close()
    except HTTPException:
        raise
    except (ValidationError, ValueError, TypeError, RecursionError) as exc:
        code = str(exc)
        if code.startswith("STALE_") or code in {"CURSOR_PIN_MISMATCH", "CURSOR_FACT_MISMATCH", "CURSOR_NOT_FOUND"}:
            raise HTTPException(409, code) from None
        raise HTTPException(422, "MALFORMED_CONTRACT") from None
    except RuntimeContextError as exc:
        if exc.code in {"FORBIDDEN_ACTOR", "FORBIDDEN_SESSION", "SESSION_NOT_FOUND", "FORBIDDEN_SCOPE", "MISSING_WORKSPACE"}:
            raise HTTPException(403, "FORBIDDEN_SCOPE") from None
        raise HTTPException(404, "NOT_FOUND") from None


async def _workbench_action_v6(request: Request):
    from services.workspace_artifact_service import WorkspaceArtifactService
    from services.workspace_artifact_preparation_service import (
        KnowledgeDesignSourcePreparationAdapter,
        WorkspaceArtifactPreparationService,
    )
    from storage.knowledge_extraction_store import KnowledgeExtractionStore
    from storage.workbench_projection_cursor_store import WorkbenchProjectionCursorStore

    ctx = require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
        route_hint=_ACTION_PATH,
    )
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > 1_500_000:
            raise HTTPException(413, "RESOURCE_LIMIT_EXCEEDED")
        data.extend(chunk)
    try:
        envelope = WorkbenchActionEnvelopeV6.model_validate(
            json.loads(data, object_pairs_hook=_unique_json_object)
        )
    except (ValueError, TypeError, RecursionError, ValidationError):
        raise HTTPException(422, "MALFORMED_CONTRACT") from None
    if envelope.command not in SUPPORTED_WORKBENCH_ACTIONS_V6:
        raise HTTPException(422, "UNSUPPORTED_ACTION")

    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(401, "AUTHORITY_CONTEXT_REQUIRED")
    owners = None
    extraction = None
    try:
        owners = _workbench_owner_context(
            ctx=ctx, session_locator=sessions[0], channel="web"
        )
        cursors = WorkbenchProjectionCursorStore(
            build_store=owners.build_store,
            authority_resolver=owners.authority_resolver,
            session_resolver=owners.session_resolver,
            session_ref=owners.session_ref,
        )
        head = cursors.get(ctx=ctx, cursor_ref=envelope.cursor_ref)
        if (
            head.revision != envelope.expected_material_context_revision
            or head.projection_basis_hash != envelope.expected_projection_hash
        ):
            # An exact completed pair retry still revalidates its retained basis.
            from eos.artifact_hashing import content_hash
            from services.change_candidate_service import ChangeCandidateService

            retry = None
            if envelope.command == "EVALUATE_RUNTIME_REPAIR_PAIR":
                retry = ChangeCandidateService(store=owners.build_store)._request_candidate(
                    scope=(ctx.organization_id,ctx.workspace_id),actor_ref=ctx.user_id,
                    operation="ASSESS_CHANGE_CANDIDATE",request_id=envelope.request_id,
                    request_hash="sha256:" + content_hash(envelope.model_dump(mode="json")))
            if envelope.command == "REVIEW_RUNTIME_CORRECTION":
                from services.candidate_semantic_review import CandidateSemanticReviewService
                retry = CandidateSemanticReviewService(store=owners.build_store).prior(
                    ctx=ctx, request_id=envelope.idempotency_key,
                    envelope_hash="sha256:" + content_hash(envelope.model_dump(mode="json")),
                )
            if retry is None:
                raise ValueError("STALE_MATERIAL_CONTEXT")

        def runtime_stack():
            runtime_source_service = RuntimeTemporalSourceService(
                build_store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                session_resolver=owners.session_resolver,
            )
            runtime_evaluation_store = RuntimeSemanticEvaluationStore(
                build_store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                session_resolver=owners.session_resolver,
            )
            runtime_evaluation_service = RuntimeDocumentEvaluationService(
                store=runtime_evaluation_store,
                source_service=runtime_source_service,
                authority_resolver=owners.authority_resolver,
                session_resolver=owners.session_resolver,
            )
            resolver = WorkbenchContextResolver(
                store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                source_service=runtime_source_service,
                evaluation_store=runtime_evaluation_store,
            )
            artifacts = WorkspaceArtifactService(
                build_store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                max_bytes=20 * 1024 * 1024,
                cursor_store=cursors,
                cursor_ref=head.cursor_ref,
            )
            return None, None, runtime_source_service, runtime_evaluation_store, runtime_evaluation_service, resolver, artifacts

        if envelope.command == "START_RUNTIME":
            from contracts.semantic_kernel import (
                ContextEdge,
                ContextNode,
                ContextWorld,
                FormationInput,
                IntentSnapshot,
                ReasonerInvocation,
                ReasoningTrace,
            )
            from contracts.runtime_semantic_evaluation import TemporalQueryV1
            from eos.artifact_hashing import content_hash
            from services.context_graph_kernel_service import ContextGraphKernelService

            (
                _runtime_context_store,
                _runtime_context_service,
                runtime_source_service,
                runtime_evaluation_store,
                runtime_evaluation_service,
                resolver,
                artifacts,
            ) = runtime_stack()
            current_projection = resolver.resolve_workbench_context_v3(
                ctx=ctx,
                cursor_store=cursors,
                artifact_service=artifacts,
                selection=WorkbenchSelectionRequestV1(
                    cursor_ref=head.cursor_ref,
                    expected_material_context_revision=head.revision,
                    expected_projection_hash=head.projection_basis_hash,
                    design_snapshot_pin=head.design_snapshot_pin,
                    lifecycle="RUNTIME",
                    representation=WorkbenchRepresentationV2(view="evaluate", lens=None),
                ),
            )
            start = current_projection.runtime.start if current_projection.runtime is not None else None
            payload = envelope.payload
            if (
                start is None
                or start.source_graph_pin != payload.source_graph_pin
                or start.ontology_pin != payload.ontology_pin
                or start.evidence_pins != payload.evidence_pins
                or start.target != payload.target
                or start.selected_entity_ids != payload.selected_entity_ids
                or start.preparation_pin != payload.preparation_pin
            ):
                raise ValueError("STALE_MATERIAL_CONTEXT")

            target_graph = runtime_source_service.get_graph(
                actor_context=owners.actor_context,
                pin=payload.source_graph_pin,
            )
            if (
                target_graph.ontology != payload.ontology_pin
                or target_graph.evidence != payload.evidence_pins
                or tuple(sorted(row.id for row in target_graph.entities))
                != payload.selected_entity_ids
            ):
                raise ValueError("HASH_MISMATCH")

            now = datetime.now(timezone.utc)
            intent = IntentSnapshot.create(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                goal="Evaluate the governed temporal target at the selected WORLD TIME",
                competency_questions=(
                    "Which governed temporal relationships are valid at the selected WORLD TIME?",
                ),
                hard_negatives=(
                    "Do not use relationships outside their exact validity interval.",
                ),
                material_dimensions=("world_time", "validity_interval"),
                source_roles=("runtime_temporal_target",),
                required_semantics=("temporal_condition",),
                unresolved_questions=(),
                captured_at=now,
            )
            formation_input = FormationInput.create(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                intent=intent,
                ontology=payload.ontology_pin,
                evidence=payload.evidence_pins,
                policy_pins=(payload.preparation_pin,),
                predecessor_context=None,
                captured_at=now,
            )
            world = ContextWorld(
                source_graph=payload.source_graph_pin,
                mode="point",
                at_time=payload.world_time,
                start_time=None,
                end_time=None,
                reasoning_version="semantica-temporal:v1",
                governance_time=now,
            )

            def verify_external_pin(pin: ArtifactPinV1) -> None:
                if pin == payload.preparation_pin:
                    raw = owners.build_store.get_raw(
                        organization_id=ctx.organization_id,
                        workspace_id=ctx.workspace_id,
                        artifact_ref=pin.artifact_ref,
                    )
                    if (
                        raw is None
                        or raw["schema_version"] != "workspace_artifact_preparation.v1"
                        or raw["artifact_hash"] != pin.artifact_hash
                    ):
                        raise RuntimeContextError("HASH_MISMATCH")
                    return
                owners.authority_resolver.resolve_pins(
                    actor_context=owners.actor_context,
                    pins=(pin,),
                )

            kernel = ContextGraphKernelService(
                store=owners.build_store,
                verify_external_pin=verify_external_pin,
                verify_authorization=lambda basis: owners.authority_resolver.verify_authorization_basis(
                    actor_context=owners.actor_context,
                    basis=basis,
                ),
            )
            proposed = kernel.form_context(
                formation_input=formation_input,
                world=world,
                nodes=tuple(
                    ContextNode(
                        node_id=row.id,
                        node_type="TemporalEntity",
                        label=row.id,
                    )
                    for row in target_graph.entities
                ),
                edges=tuple(
                    ContextEdge(
                        edge_id=row.id,
                        source_id=row.source,
                        target_id=row.target,
                        relation_type="TEMPORAL_RELATION",
                    )
                    for row in target_graph.relationships
                ),
                assertions=(),
                selected_evidence=payload.evidence_pins,
                excluded_evidence=(),
                conflicts=(),
                formation_policy=payload.preparation_pin,
                formation_policy_version="workspace-runtime-target-preparation.v1",
                applied_change_candidate=None,
                recorded_at=now,
                persist=False,
            )
            authorization = owners.authority_resolver.current_actor_authorization_basis(
                actor_context=owners.actor_context,
                operation="ADMIT_CONTEXT",
                subject=proposed.pin(),
                authorization_class="TENANT_MEMBER",
            )
            admission_reasoning = ReasoningTrace(
                strategy="governed_context_admission",
                reasoning_version="context-admission:v1",
                invocations=(
                    ReasonerInvocation(
                        invocation_id=f"context-admission:{envelope.request_id}",
                        strategy="governed_context_admission",
                        executable_engine=ArtifactPinV1(
                            artifact_kind="authority_verifier",
                            artifact_ref=authorization.resolver_ref,
                            artifact_hash=authorization.resolver_hash,
                        ),
                        rules=(),
                        inputs=(
                            proposed.pin(),
                            payload.preparation_pin,
                            payload.source_graph_pin,
                        ),
                        outputs=("ADMITTED",),
                        proof_steps=(
                            "current tenant membership verified",
                            "runtime target preparation and source graph pins verified",
                        ),
                        evidence=payload.evidence_pins,
                    ),
                ),
                derived_assertion_ids=(),
                diagnostics=(),
            )
            request_hash = "sha256:" + content_hash(envelope.model_dump(mode="json"))
            _admission_decision, admitted = kernel.admit_new_context_with_request(
                proposed=proposed,
                authorization=authorization,
                reasoning=admission_reasoning,
                evidence=payload.evidence_pins,
                actor_id=ctx.user_id,
                reasoner_identity=authorization.resolver_ref,
                recorded_at=now,
                request_id=envelope.request_id,
                request_hash=request_hash,
                operation="START_RUNTIME",
            )
            query = TemporalQueryV1.create(
                schema_version="runtime_temporal_query.v1",
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                operation="point",
                source_graph=payload.source_graph_pin,
                at_time=payload.world_time,
                start_time=None,
                end_time=None,
                source_id=None,
                target_id=None,
                pattern_type=None,
                min_frequency=None,
                query_text="",
                include_boundaries=True,
            )
            provider_request = runtime_evaluation_service.prepare_temporal_request(
                actor_context=owners.actor_context,
                context=admitted.pin(),
                target=payload.target,
                query=query,
            )
            run = runtime_evaluation_service.evaluate(
                actor_context=owners.actor_context,
                context=admitted.pin(),
                target=payload.target,
                provider_request=provider_request.artifact_pin(),
                business_effective_at=payload.world_time,
                request_id=f"{envelope.request_id}:evaluation",
                parent_run=None,
            )
            context_projection = resolver.resolve_workbench_context_v3(
                ctx=ctx,
                cursor_store=cursors,
                artifact_service=artifacts,
                selection=WorkbenchSelectionRequestV1(
                    cursor_ref=head.cursor_ref,
                    expected_material_context_revision=head.revision,
                    expected_projection_hash=head.projection_basis_hash,
                    design_snapshot_pin=head.design_snapshot_pin,
                    runtime_context_pin=admitted.pin(),
                    runtime_run_pin=run.artifact_pin(),
                    lifecycle="RUNTIME",
                    representation=WorkbenchRepresentationV2(view="evaluate", lens=None),
                ),
            )
            receipt = WorkbenchActionReceiptV6(
                receipt_kind="RUNTIME_EVALUATION",
                receipt_pin=run.artifact_pin(),
                receipt_payload=None,
                operation_status="COMPLETE",
                request_id=envelope.request_id,
                request_hash=request_hash,
                idempotency_key=envelope.idempotency_key,
                recorded_at=run.created_at,
            )
            result = CompletedWorkbenchActionResultV6(
                command="START_RUNTIME",
                result_pin=run.artifact_pin(),
                receipt=receipt,
                context_projection=context_projection,
                recorded_at=run.created_at,
            )
            return JSONResponse(
                content={
                    **result.model_dump(mode="json", exclude={"context_projection"}),
                    "context_projection": serialize_unified_workbench_context_v3(context_projection),
                },
                headers={
                    "Workbench-Action-Version": _ACTION_VERSION_V6,
                    "Cache-Control": "no-store",
                },
            )

        def refresh_learning(*, context_pin: ArtifactPinV1, run_pin: ArtifactPinV1 | None = None, report_pin: ArtifactPinV1 | None = None, learning_candidate_pin: ArtifactPinV1 | None = None):
            stack = runtime_stack()
            resolver, artifacts = stack[-2], stack[-1]
            return resolver.resolve_workbench_context_v3(
                ctx=ctx,
                cursor_store=cursors,
                artifact_service=artifacts,
                selection=WorkbenchSelectionRequestV1(
                    cursor_ref=head.cursor_ref,
                    expected_material_context_revision=head.revision,
                    expected_projection_hash=head.projection_basis_hash,
                    design_snapshot_pin=head.design_snapshot_pin,
                    runtime_context_pin=context_pin,
                    runtime_run_pin=run_pin,
                    runtime_report_pin=report_pin,
                    learning_candidate_pin=learning_candidate_pin,
                    lifecycle="RUNTIME",
                    representation=WorkbenchRepresentationV2(view="evaluate", lens=None),
                ),
            )

        def learning_response(*, command: str, result_pin: ArtifactPinV1, receipt_pin: ArtifactPinV1, receipt_kind: str, recorded_at: datetime, context_projection, request_hash: str, candidate_review=None):
            receipt = WorkbenchActionReceiptV6(
                receipt_kind=receipt_kind,
                receipt_pin=receipt_pin,
                receipt_payload=None,
                candidate_review=candidate_review,
                operation_status="COMPLETE",
                request_id=envelope.request_id,
                request_hash=request_hash,
                idempotency_key=envelope.idempotency_key,
                recorded_at=recorded_at,
            )
            result = CompletedWorkbenchActionResultV6(
                command=command,
                result_pin=result_pin,
                receipt=receipt,
                context_projection=context_projection,
                recorded_at=recorded_at,
            )
            return JSONResponse(
                content={
                    **result.model_dump(mode="json", exclude={"context_projection"}),
                    "context_projection": serialize_unified_workbench_context_v3(context_projection),
                },
                headers={"Workbench-Action-Version": _ACTION_VERSION_V6, "Cache-Control": "no-store"},
            )

        if envelope.command == "SUBMIT_SEMANTIC_GRAPH_EDIT":
            from contracts.semantic_kernel import ContextGraph
            from eos.artifact_hashing import content_hash
            from services.change_candidate_service import ChangeCandidateService

            command = envelope.payload
            if command.graph_pin.artifact_kind != "context_graph":
                raise ValueError("NOT_RUN_CONTRACT_MISMATCH")
            graph = owners.authority_resolver.resolve_pins(
                actor_context=owners.actor_context, pins=(command.graph_pin,),
            )[0]
            if not isinstance(graph, ContextGraph):
                raise ValueError("PIN_MISMATCH")
            # Compare the observed immutable context with the current governed
            # target preparation, not browser graph data or renderer state.
            stack = runtime_stack()
            resolver, artifacts = stack[-2], stack[-1]
            current = resolver.resolve_workbench_context_v3(
                ctx=ctx, cursor_store=cursors, artifact_service=artifacts,
                selection=WorkbenchSelectionRequestV1(
                    cursor_ref=head.cursor_ref, expected_material_context_revision=head.revision,
                    expected_projection_hash=head.projection_basis_hash,
                    design_snapshot_pin=head.design_snapshot_pin, lifecycle="RUNTIME",
                    representation=WorkbenchRepresentationV2(view="evaluate", lens=None),
                ),
            )
            start = current.runtime.start if current.runtime is not None else None
            if start is None or graph.world.source_graph != start.source_graph_pin or graph.ontology != start.ontology_pin or graph.formation.formation_policy != start.preparation_pin:
                raise ValueError("STALE_SEMANTIC_BASIS")
            # Material reconstruction inputs must remain exactly resolvable.
            # Embedded intent/formation values are already hash-bound in graph.
            owners.authority_resolver.resolve_pins(
                actor_context=owners.actor_context,
                pins=(graph.ontology, graph.world.source_graph, *graph.formation.source_inputs),
            )
            from contracts.ontology_workbench_context import WorkspaceArtifactPreparationV1
            preparation = owners.build_store.get(
                organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                artifact_ref=graph.formation.formation_policy.artifact_ref,
                contract_type=WorkspaceArtifactPreparationV1,
            )
            if preparation is None or preparation.preparation_hash != graph.formation.formation_policy.artifact_hash:
                raise ValueError("REPLAY_BASIS_NOT_CLOSED")
            request_hash = "sha256:" + content_hash(envelope.model_dump(mode="json"))
            def verify_material_basis():
                current_head = cursors.get(ctx=ctx, cursor_ref=envelope.cursor_ref)
                if current_head != head:
                    raise ValueError("STALE_MATERIAL_CONTEXT")
            candidate = ChangeCandidateService(store=owners.build_store).propose_semantic_graph_edit(
                organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                command=command, current_graph_pin=graph.pin(), actor_ref=ctx.user_id,
                request_id=envelope.idempotency_key, request_hash=request_hash,
                verify_material_basis=verify_material_basis,
                recorded_at=datetime.now(timezone.utc),
            )
            projection = refresh_learning(context_pin=graph.pin(), learning_candidate_pin=candidate.pin())
            digest = content_hash({"request_hash": request_hash, "candidate": candidate.pin().model_dump(mode="json")})
            return learning_response(
                command=envelope.command, result_pin=candidate.pin(),
                receipt_pin=ArtifactPinV1(artifact_kind="change_candidate_receipt", artifact_ref=f"change-candidate-receipt:{digest}", artifact_hash=f"sha256:{digest}"),
                receipt_kind="CHANGE_CANDIDATE", recorded_at=candidate.recorded_at,
                context_projection=projection, request_hash=request_hash,
            )

        if envelope.command == "SUBMIT_RUNTIME_CORRECTION":
            from contracts.semantic_kernel import (
                ChangeDelta,
                ComparisonExpectation,
                ContextGraph,
                ContextGraphOrigin,
                DecisionRecord,
                EvaluationBasis,
                ProtectedInput,
                WorkspaceScope,
            )
            from eos.artifact_hashing import content_hash
            from services.change_candidate_service import ChangeCandidateService
            from storage.runtime_inspection_report_store import RuntimeInspectionReportStore

            command = RuntimeCorrectionCommandV1.model_validate(envelope.payload.model_dump(mode="json"))
            request_hash = "sha256:" + content_hash(envelope.model_dump(mode="json"))
            scope = WorkspaceScope(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
            )
            report = RuntimeInspectionReportStore(owners.build_store).get(
                scope=scope,
                report_pin=command.report_pin,
            )
            if command.result_pin not in report.summary.ordered_result_pins:
                raise ValueError("PIN_MISMATCH")
            decision = owners.build_store.get(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_ref=command.result_pin.artifact_ref,
                contract_type=DecisionRecord,
            )
            if decision is None or decision.pin() != command.result_pin:
                raise ValueError("PIN_MISMATCH")
            context = owners.build_store.get(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_ref=decision.context.artifact_ref,
                contract_type=ContextGraph,
            )
            if (
                context is None
                or context.pin() != decision.context
                or report.subject.context_pin != context.pin()
            ):
                raise ValueError("PIN_MISMATCH")
            available_evidence = {
                (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
                for pin in context.formation.source_inputs
            }
            if any(
                (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) not in available_evidence
                for pin in command.evidence_pins
            ):
                raise ValueError("PIN_MISMATCH")
            if command.correction_kind == "CONFIRMED":
                raise ValueError("NO_CHANGE_REQUIRED")
            if command.correction_kind == "EXCEPTION_CORRECTION":
                target, failure_class, semantic_graph_change = "ONTOLOGY", "STRUCTURE", True
            elif command.correction_kind == "WRONG_REASON":
                target, failure_class, semantic_graph_change = "REASONING", "STEERING", False
            else:
                target, failure_class, semantic_graph_change = "CONTEXT_FORMATION", "STEERING", False

            def content_pin(kind: str, prefix: str, value: object) -> ArtifactPinV1:
                digest = content_hash(value)
                return ArtifactPinV1(
                    artifact_kind=kind,
                    artifact_ref=f"{prefix}:{digest}",
                    artifact_hash=f"sha256:{digest}",
                )

            evaluator = report.subject.evaluator_pin or content_pin(
                "evaluator",
                "runtime-correction-evaluator",
                {"protocol": "runtime-correction-evaluator.v1"},
            )
            obligation_suite = content_pin(
                "obligation_suite",
                "runtime-correction-obligation-suite",
                {"protocol": "matched-plus-independent-holdout.v1", "target": target},
            )
            comparison_policy = content_pin(
                "comparison_policy",
                "runtime-correction-comparison-policy",
                {"policy": "strict-bounded-improvement.v1", "target": target},
            )
            holdout_context = None
            if command.independent_holdout_context_pin is not None:
                holdout_context = owners.build_store.get(
                    organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id,
                    artifact_ref=command.independent_holdout_context_pin.artifact_ref,
                    contract_type=ContextGraph,
                )
            if command.independent_holdout_context_pin is not None and (
                holdout_context is None
                or holdout_context.pin() != command.independent_holdout_context_pin
                or holdout_context.state != "ADMITTED"
                or holdout_context.pin() == context.pin()
                or holdout_context.ontology != context.ontology
                or holdout_context.intent != context.intent
                or holdout_context.formation.source_inputs != context.formation.source_inputs
                or holdout_context.formation.formation_policy != context.formation.formation_policy
            ):
                raise ValueError("INVALID_INDEPENDENT_HOLDOUT")
            protected_inputs = (
                ProtectedInput(slot_id="context", artifact=context.pin()),
                ProtectedInput(slot_id="decision", artifact=decision.pin()),
                *((
                    ProtectedInput(
                        slot_id="independent_holdout_context",
                        artifact=holdout_context.pin(),
                    ),
                ) if holdout_context is not None else ()),
                *tuple(
                    ProtectedInput(slot_id=f"evidence:{index}", artifact=pin)
                    for index, pin in enumerate(command.evidence_pins)
                ),
            )
            comparison_expectation = None
            if command.correction_kind == "FALSE_NEGATIVE":
                comparison_expectation = ComparisonExpectation(
                    matched_baseline_result="NO_MATCH",
                    matched_successor_result="MATCH",
                    holdout_policy="UNCHANGED",
                )
            elif command.correction_kind == "FALSE_POSITIVE":
                comparison_expectation = ComparisonExpectation(
                    matched_baseline_result="MATCH",
                    matched_successor_result="NO_MATCH",
                    holdout_policy="UNCHANGED",
                )
            basis = EvaluationBasis(
                evaluator=evaluator,
                obligation_suite=obligation_suite,
                comparison_policy=comparison_policy,
                protected_inputs=protected_inputs,
                runtime_pins=(report.subject.run_pin,),
                provider_pins=(
                    ()
                    if report.subject.provider_identity_pin is None
                    else (report.subject.provider_identity_pin,)
                ),
                config_pins=(),
                candidate_delta_bounds=f"Only {target} may change; evaluator/protected inputs are frozen",
                comparison_expectation=comparison_expectation,
            )
            candidate = ChangeCandidateService(store=owners.build_store).propose_with_request(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                origin=ContextGraphOrigin(
                    context=context.pin(),
                    evidence=command.evidence_pins,
                    decision=decision.pin(),
                ),
                failure_class=failure_class,
                target=target,
                hypothesis=command.rationale,
                evaluation_basis=basis,
                delta=ChangeDelta(
                    description=f"{command.correction_kind}: {command.rationale}",
                    claims_semantic_graph_change=semantic_graph_change,
                    basis_refs=(decision.pin(), report.report_pin),
                    structural_diff=None,
                ),
                matched_cases=(context.pin(),),
                independent_holdout_cases=(
                    () if holdout_context is None else (holdout_context.pin(),)
                ),
                recorded_at=datetime.now(timezone.utc),
                actor_ref=ctx.user_id,
                request_id=envelope.request_id,
                request_hash=request_hash,
            )
            projection = refresh_learning(
                context_pin=context.pin(),
                run_pin=report.subject.run_pin,
                report_pin=report.report_pin,
                learning_candidate_pin=candidate.pin(),
            )
            receipt_pin = content_pin(
                "change_candidate_receipt",
                "change-candidate-receipt",
                {
                    "request_hash": request_hash,
                    "candidate": candidate.pin().model_dump(mode="json"),
                },
            )
            return learning_response(
                command="SUBMIT_RUNTIME_CORRECTION",
                result_pin=candidate.pin(),
                receipt_pin=receipt_pin,
                receipt_kind="CHANGE_CANDIDATE",
                recorded_at=candidate.recorded_at,
                context_projection=projection,
                request_hash=request_hash,
            )

        if envelope.command == "REVIEW_RUNTIME_CORRECTION":
            from contracts.semantic_kernel import ContextGraphOrigin, ChangeCandidate, ContextGraph
            from eos.artifact_hashing import content_hash
            from services.change_candidate_service import ChangeCandidateService

            payload = envelope.payload
            candidate = ChangeCandidateService(store=owners.build_store)._load(
                scope=(ctx.organization_id, ctx.workspace_id), pin=payload.change_candidate_pin,
                model=ChangeCandidate,
            )
            if candidate.stage != "PROPOSAL":
                raise ValueError("PROPOSAL_REQUIRED")
            if not isinstance(candidate.origin, ContextGraphOrigin):
                raise ValueError("PRE_CONTEXT_REVIEW_HAS_NO_RUNTIME_PROJECTION")
            request_hash = "sha256:" + content_hash(envelope.model_dump(mode="json"))
            protocol_hash = "sha256:" + content_hash({"protocol": "semantic-graph-edit.v1"})
            if candidate.evaluation_basis.evaluator.artifact_hash == protocol_hash:
                from contracts.candidate_review import CandidateReviewMaterialV1
                from services.candidate_semantic_review import CandidateSemanticReviewService
                service = CandidateSemanticReviewService(store=owners.build_store)
                prior = service.prior(ctx=ctx, request_id=envelope.idempotency_key, envelope_hash=request_hash)
                projection = refresh_learning(context_pin=candidate.origin.context, learning_candidate_pin=candidate.pin())
                if prior is None:
                    graph = service._load((ctx.organization_id, ctx.workspace_id), candidate.origin.context,
                        ContextGraph)
                    basis = CandidateReviewMaterialV1.from_head(cursors.get(ctx=ctx, cursor_ref=envelope.cursor_ref), preparation_pin=graph.formation.formation_policy)
                    review_projection = service.compile(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                        proposal_pin=candidate.pin(), material_basis=basis)
                    predicates = service.evaluate(projection=review_projection)
                    evaluated_at = datetime.now(timezone.utc)
                else:
                    review_projection = prior.candidate_review.projection
                    predicates = prior.candidate_review.predicates
                    evaluated_at = prior.candidate_review.evaluated_at
                review = service.publish(ctx=ctx, cursors=cursors, projection=review_projection, predicates=predicates,
                    human_outcome=payload.outcome, human_rationale=payload.rationale,
                    request_id=envelope.idempotency_key, envelope_hash=request_hash, evaluated_at=evaluated_at)
                return learning_response(command="REVIEW_RUNTIME_CORRECTION", result_pin=candidate.pin(),
                    receipt_pin=review.decision_pin, receipt_kind="CANDIDATE_REVIEW", recorded_at=review.result.checked_at,
                    context_projection=projection, request_hash=request_hash, candidate_review=review)
            receipt_hash = content_hash(
                {
                    "request_hash": request_hash,
                    "candidate": candidate.pin().model_dump(mode="json"),
                    "outcome": payload.outcome,
                    "semantic_subtype": payload.semantic_subtype,
                    "rationale": payload.rationale,
                }
            )
            projection = refresh_learning(
                context_pin=candidate.origin.context,
                learning_candidate_pin=candidate.pin(),
            )
            return learning_response(
                command="REVIEW_RUNTIME_CORRECTION",
                result_pin=candidate.pin(),
                receipt_pin=ArtifactPinV1(
                    artifact_kind="change_candidate_review_receipt",
                    artifact_ref=f"change-candidate-review:{receipt_hash}",
                    artifact_hash=f"sha256:{receipt_hash}",
                ),
                receipt_kind="CANDIDATE_REVIEW",
                recorded_at=candidate.recorded_at,
                context_projection=projection,
                request_hash=request_hash,
            )

        if envelope.command == "EVALUATE_RUNTIME_REPAIR_PAIR":
            from contracts.semantic_kernel import ContextGraphOrigin
            from eos.artifact_hashing import content_hash
            from services.change_candidate_comparison_service import (
                ChangeCandidateComparisonService,
            )
            from services.change_candidate_service import ChangeCandidateService

            payload = envelope.payload
            service = ChangeCandidateService(store=owners.build_store)
            proposal = service.get(pin=payload.change_candidate_pin)
            if not isinstance(proposal.origin, ContextGraphOrigin):
                raise ValueError("PRE_CONTEXT_PAIR_HAS_NO_RUNTIME_PROJECTION")
            request_hash = "sha256:" + content_hash(envelope.model_dump(mode="json"))
            compared_at = datetime.now(timezone.utc)
            from services.runtime_context_reasoning_evaluation_service import RuntimeContextReasoningEvaluationService
            from services.runtime_semantica_reasoning_service import RuntimeSemanticaReasoningService
            from services.working_context_projection_service import WorkingContextProjectionService

            evaluation_store = RuntimeSemanticEvaluationStore(
                build_store=owners.build_store, authority_resolver=owners.authority_resolver,
                session_resolver=owners.session_resolver)
            context_evaluator = RuntimeContextReasoningEvaluationService(
                store=evaluation_store,
                reasoning_service=RuntimeSemanticaReasoningService(
                    build_store=owners.build_store, authority_resolver=owners.authority_resolver),
                projection_service=WorkingContextProjectionService(
                    build_store=owners.build_store, authority_resolver=owners.authority_resolver,
                    runtime_actor_id="agent:semantica-runtime"))
            comparison = ChangeCandidateComparisonService(
                store=owners.build_store, context_evaluator=context_evaluator,
                actor_context=owners.actor_context
            ).compare(
                proposal_pin=proposal.pin(),
                matched_baseline_decision_pin=payload.matched_baseline_decision_pin,
                matched_successor_decision_pin=payload.matched_successor_decision_pin,
                holdout_baseline_decision_pin=payload.holdout_baseline_decision_pin,
                holdout_successor_decision_pin=payload.holdout_successor_decision_pin,
                successor_input_pin=payload.successor_input_pin,
                actor_id=ctx.user_id,
                persist=False,
            )
            assessed = service.evaluate_pair_with_request(
                proposal=proposal.pin(),
                matched_execution_attempt=payload.matched_execution_attempt_pin,
                matched_metric_refs=(),
                matched_status="INCONCLUSIVE",
                holdout_execution_attempt=payload.holdout_execution_attempt_pin,
                holdout_metric_refs=(),
                holdout_status="INCONCLUSIVE",
                comparison_decision=comparison.pin(),
                comparison_record=comparison,
                context_evaluator=context_evaluator,
                actor_context=owners.actor_context,
                successor_input=payload.successor_input_pin,
                recorded_at=compared_at,
                actor_ref=ctx.user_id,
                request_id=envelope.request_id,
                request_hash=request_hash,
            )
            receipt_hash = content_hash(
                {
                    "request_hash": request_hash,
                    "assessment": assessed.pin().model_dump(mode="json"),
                }
            )
            projection = refresh_learning(
                context_pin=proposal.origin.context,
                learning_candidate_pin=assessed.pin(),
            )
            return learning_response(
                command="EVALUATE_RUNTIME_REPAIR_PAIR",
                result_pin=assessed.pin(),
                receipt_pin=ArtifactPinV1(
                    artifact_kind="change_candidate_evaluation_receipt",
                    artifact_ref=f"change-candidate-evaluation:{receipt_hash}",
                    artifact_hash=f"sha256:{receipt_hash}",
                ),
                receipt_kind="CANDIDATE_EVALUATION",
                recorded_at=assessed.recorded_at,
                context_projection=projection,
                request_hash=request_hash,
            )

        if envelope.command == "REQUEST_RUNTIME_LEARNING_FUTURE_RUN":
            from contracts.semantic_kernel import ChangeCandidate, ContextGraph
            from eos.artifact_hashing import content_hash
            from services.change_candidate_service import ChangeCandidateService

            payload = envelope.payload
            candidate = ChangeCandidateService(store=owners.build_store).get(
                pin=payload.change_candidate_pin
            )
            if (
                candidate.stage != "ASSESSMENT"
                or candidate.assessment is None
                or candidate.assessment.disposition != "ACCEPTED_FOR_SUCCESSOR_INPUT"
            ):
                raise ValueError("CHANGE_CANDIDATE_NOT_ACCEPTED_FOR_SUCCESSOR")
            context = owners.build_store.get(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_ref=payload.context_pin.artifact_ref,
                contract_type=ContextGraph,
            )
            if context is None or context.pin() != payload.context_pin or context.state != "ADMITTED":
                raise ValueError("PIN_MISMATCH")
            if candidate.target != "CONTEXT_FORMATION":
                raise ValueError("FUTURE_RUN_TARGET_NOT_IMPLEMENTED")
            if context.formation.applied_change_candidate != candidate.pin():
                raise ValueError("CHANGE_CANDIDATE_NOT_CONSUMED_BY_CONTEXT")
            stack = runtime_stack()
            evaluation_store, evaluation_service = stack[3], stack[4]
            run = evaluation_service.evaluate(
                actor_context=owners.actor_context,
                context=context.pin(),
                target=payload.target,
                provider_request=payload.provider_request_pin,
                business_effective_at=payload.business_effective_at,
                request_id=envelope.request_id,
            )
            state = evaluation_store.get_current_state(
                actor_context=owners.actor_context,
                run=run.artifact_pin(),
            )
            if len(state.decisions) != 1:
                raise ValueError("SEMANTIC_DECISION_REQUIRED")
            request_hash = "sha256:" + content_hash(envelope.model_dump(mode="json"))
            receipt_hash = content_hash(
                {
                    "request_hash": request_hash,
                    "candidate": candidate.pin().model_dump(mode="json"),
                    "context": context.pin().model_dump(mode="json"),
                    "run": run.artifact_pin().model_dump(mode="json"),
                    "decision": state.decisions[0].model_dump(mode="json"),
                }
            )
            projection = refresh_learning(
                context_pin=context.pin(),
                run_pin=run.artifact_pin(),
                learning_candidate_pin=candidate.pin(),
            )
            return learning_response(
                command="REQUEST_RUNTIME_LEARNING_FUTURE_RUN",
                result_pin=state.decisions[0],
                receipt_pin=ArtifactPinV1(
                    artifact_kind="future_run_receipt",
                    artifact_ref=f"future-run:{receipt_hash}",
                    artifact_hash=f"sha256:{receipt_hash}",
                ),
                receipt_kind="FUTURE_RUN",
                recorded_at=state.recorded_at,
                context_projection=projection,
                request_hash=request_hash,
            )

        if envelope.command == "CREATE_REPORT":
            from contracts.semantic_kernel import WorkspaceScope
            from eos.artifact_hashing import content_hash
            from services.runtime_inspection_report_service import RuntimeInspectionReportService
            from storage.runtime_inspection_report_store import RuntimeInspectionReportStore

            runtime_source_service = RuntimeTemporalSourceService(
                build_store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                session_resolver=owners.session_resolver,
            )
            runtime_evaluation_store = RuntimeSemanticEvaluationStore(
                build_store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                session_resolver=owners.session_resolver,
            )
            resolver = WorkbenchContextResolver(
                store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                source_service=runtime_source_service,
                evaluation_store=runtime_evaluation_store,
            )
            report_command = RuntimeReportCommandV1.model_validate(
                envelope.payload.model_dump(mode="json")
            )
            report_inputs = resolver.resolve_temporal_report_inputs(
                ctx=ctx,
                cursor_store=cursors,
                command=report_command,
            )
            report, replay = RuntimeInspectionReportService.project(
                inputs=report_inputs,
                projector_pin=report_command.projector_pin,
                renderer_pin=report_command.renderer_pin,
                template_pin=report_command.template_pin,
                environment_pin=report_command.environment_pin,
                predecessor_pin=report_command.predecessor_report_pin,
            )
            request_hash = "sha256:" + content_hash(envelope.model_dump(mode="json"))
            report_store = RuntimeInspectionReportStore(owners.build_store)
            report_receipt = report_store.put_report_with_replay(
                scope=WorkspaceScope(
                    organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id,
                ),
                report=report,
                replay_inputs=replay,
                expected_predecessor=report_command.predecessor_report_pin,
                request_id=envelope.request_id,
                request_hash=request_hash,
                actor_ref=ctx.user_id,
            )
            artifacts = WorkspaceArtifactService(
                build_store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                max_bytes=20 * 1024 * 1024,
                cursor_store=cursors,
                cursor_ref=head.cursor_ref,
            )
            context_projection = resolver.resolve_workbench_context_v3(
                ctx=ctx,
                cursor_store=cursors,
                artifact_service=artifacts,
                selection=WorkbenchSelectionRequestV1(
                    cursor_ref=head.cursor_ref,
                    expected_material_context_revision=head.revision,
                    expected_projection_hash=head.projection_basis_hash,
                    design_snapshot_pin=head.design_snapshot_pin,
                    runtime_context_pin=report_inputs.context_pin,
                    runtime_run_pin=report_inputs.run_pin,
                    runtime_report_pin=report.report_pin,
                    lifecycle="RUNTIME",
                    representation=WorkbenchRepresentationV2(view="evaluate", lens=None),
                ),
            )
            receipt = WorkbenchActionReceiptV6(
                receipt_kind="RUNTIME_REPORT",
                receipt_pin=ArtifactPinV1(
                    artifact_kind="runtime_report_receipt",
                    artifact_ref=report_receipt.receipt_ref,
                    artifact_hash=report_receipt.receipt_hash,
                ),
                receipt_payload=None,
                operation_status="COMPLETE",
                request_id=envelope.request_id,
                request_hash=request_hash,
                idempotency_key=envelope.idempotency_key,
                recorded_at=report_receipt.recorded_at,
            )
            result = CompletedWorkbenchActionResultV6(
                command="CREATE_REPORT",
                result_pin=report.report_pin,
                receipt=receipt,
                context_projection=context_projection,
                recorded_at=report_receipt.recorded_at,
            )
            return JSONResponse(
                content={
                    **result.model_dump(mode="json", exclude={"context_projection"}),
                    "context_projection": serialize_unified_workbench_context_v3(context_projection),
                },
                headers={"Workbench-Action-Version": _ACTION_VERSION_V6, "Cache-Control": "no-store"},
            )

        if envelope.command == "EXPORT_REPORT":
            from contracts.semantic_kernel import WorkspaceScope
            from eos.artifact_hashing import content_hash
            from services.runtime_report_export_service import (
                RuntimeReportExportService,
                server_runtime_export_profile,
            )
            from storage.runtime_inspection_report_store import RuntimeInspectionReportStore

            export_command = RuntimeExportCommandV1.model_validate(
                envelope.payload.model_dump(mode="json")
            )
            scope = WorkspaceScope(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
            )
            report_store = RuntimeInspectionReportStore(owners.build_store)
            report = report_store.get(scope=scope, report_pin=export_command.report_pin)
            if export_command.replay_inputs_pin != report.replay_access.replay_pin:
                raise ValueError("PIN_MISMATCH")
            profile = server_runtime_export_profile(
                report=report,
                format=export_command.format,
                access_projection=export_command.access_projection,
            )
            profile_pin = ArtifactPinV1(
                artifact_kind="pinned_render_profile",
                artifact_ref=profile.profile_ref,
                artifact_hash=profile.profile_hash,
            )
            if export_command.render_profile_pin != profile_pin:
                raise ValueError("PIN_MISMATCH")
            request_hash = "sha256:" + content_hash(envelope.model_dump(mode="json"))
            export_receipt = RuntimeReportExportService.publish(
                build_store=owners.build_store,
                workspace_root=ctx.workspace_root,
                scope=scope,
                report_pin=export_command.report_pin,
                replay_inputs_pin=export_command.replay_inputs_pin,
                render_profile=profile,
                access_projection=export_command.access_projection,
                request_id=envelope.request_id,
                request_hash=request_hash,
                actor_ref=ctx.user_id,
                expected_export_head_pin=export_command.expected_export_head_pin,
            )
            if export_receipt.published_artifact is None:
                raise ValueError("INCOMPLETE_CLOSURE")
            runtime_source_service = RuntimeTemporalSourceService(
                build_store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                session_resolver=owners.session_resolver,
            )
            runtime_evaluation_store = RuntimeSemanticEvaluationStore(
                build_store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                session_resolver=owners.session_resolver,
            )
            resolver = WorkbenchContextResolver(
                store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                source_service=runtime_source_service,
                evaluation_store=runtime_evaluation_store,
            )
            artifacts = WorkspaceArtifactService(
                build_store=owners.build_store,
                authority_resolver=owners.authority_resolver,
                max_bytes=20 * 1024 * 1024,
                cursor_store=cursors,
                cursor_ref=head.cursor_ref,
            )
            context_projection = resolver.resolve_workbench_context_v3(
                ctx=ctx,
                cursor_store=cursors,
                artifact_service=artifacts,
                selection=WorkbenchSelectionRequestV1(
                    cursor_ref=head.cursor_ref,
                    expected_material_context_revision=head.revision,
                    expected_projection_hash=head.projection_basis_hash,
                    design_snapshot_pin=head.design_snapshot_pin,
                    runtime_context_pin=report.subject.context_pin,
                    runtime_run_pin=report.subject.run_pin,
                    runtime_report_pin=report.report_pin,
                    lifecycle="RUNTIME",
                    representation=WorkbenchRepresentationV2(view="evaluate", lens=None),
                ),
            )
            receipt = WorkbenchActionReceiptV6(
                receipt_kind="RUNTIME_EXPORT",
                receipt_pin=ArtifactPinV1(
                    artifact_kind="runtime_report_export_receipt",
                    artifact_ref=export_receipt.receipt_ref,
                    artifact_hash=export_receipt.receipt_hash,
                ),
                receipt_payload=None,
                operation_status="COMPLETE",
                request_id=envelope.request_id,
                request_hash=request_hash,
                idempotency_key=envelope.idempotency_key,
                recorded_at=export_receipt.recorded_at,
            )
            result = CompletedWorkbenchActionResultV6(
                command="EXPORT_REPORT",
                result_pin=export_receipt.published_artifact,
                receipt=receipt,
                context_projection=context_projection,
                recorded_at=export_receipt.recorded_at,
            )
            return JSONResponse(
                content={
                    **result.model_dump(mode="json", exclude={"context_projection"}),
                    "context_projection": serialize_unified_workbench_context_v3(context_projection),
                },
                headers={"Workbench-Action-Version": _ACTION_VERSION_V6, "Cache-Control": "no-store"},
            )

        if head.design_snapshot_pin is None:
            raise ValueError("DESIGN_SNAPSHOT_REQUIRED")
        snapshot = cursors._design_snapshot(ctx, head.design_snapshot_pin)
        _selection, design, _runtime = cursors._facts(
            cursors._scope(ctx),
            head.selection_pin,
            head.design_binding_set_pin,
            head.runtime_binding_set_pin,
        )
        expected_ir = ArtifactPinV1(
            artifact_kind="canonical_source_ir",
            artifact_ref=snapshot.canonical_source_ir_ref,
            artifact_hash=snapshot.canonical_source_ir_hash,
        )
        matches = [
            entry
            for entry in design.entries
            if entry.state == "ACTIVE"
            and entry.source_identity_pin is not None
            and entry.source_identity_pin.artifact_ref == snapshot.source_identity_ref
            and entry.canonical_source_ir_pin == expected_ir
            and entry.preparation_pin is not None
        ]
        if len(matches) != 1:
            raise ValueError("SOURCE_NOT_BOUND")
        entry = matches[0]
        artifacts = WorkspaceArtifactService(
            build_store=owners.build_store,
            authority_resolver=owners.authority_resolver,
            max_bytes=20 * 1024 * 1024,
            cursor_store=cursors,
            cursor_ref=head.cursor_ref,
        )
        knowledge = owners.source_registry
        extraction = KnowledgeExtractionStore(
            knowledge._db_path
        )
        preparation = WorkspaceArtifactPreparationService(
            artifact_service=artifacts,
            design_adapter=KnowledgeDesignSourcePreparationAdapter(
                artifact_service=artifacts,
                knowledge_store=knowledge,
                extraction_store=extraction,
            ),
        )
        resolver = WorkbenchContextResolver(
            store=owners.build_store,
            authority_resolver=owners.authority_resolver,
            preparation_service=preparation,
        )
        acknowledgement = resolver.focus_design_evidence(
            ctx=ctx,
            cursor_store=cursors,
            command=FocusEvidenceActionEnvelopeV6.model_validate(envelope.model_dump(mode="json")),
            snapshot_pin=head.design_snapshot_pin,
            artifact_pin=entry.artifact_pin,
            preparation_pin=entry.preparation_pin,
        )
        context_projection = resolver.resolve_workbench_context_v3(
            ctx=ctx,
            cursor_store=cursors,
            artifact_service=artifacts,
            selection=WorkbenchSelectionRequestV1(
                cursor_ref=envelope.cursor_ref,
                expected_material_context_revision=acknowledgement.material_context_revision,
                expected_projection_hash=acknowledgement.projection_hash,
                design_snapshot_pin=head.design_snapshot_pin,
                lifecycle="DESIGN",
                representation=WorkbenchRepresentationV2(view="evidence", lens=None),
            ),
        )
        receipt = WorkbenchActionReceiptV6(
            receipt_kind="PRESENTATION_ACKNOWLEDGEMENT",
            receipt_pin=ArtifactPinV1(
                artifact_kind="workbench_action_acknowledgement",
                artifact_ref=acknowledgement.receipt_ref,
                artifact_hash=acknowledgement.receipt_hash,
            ),
            receipt_payload=acknowledgement,
            operation_status="COMPLETE",
            request_id=acknowledgement.request_id,
            request_hash=acknowledgement.request_hash,
            idempotency_key=acknowledgement.idempotency_key,
            recorded_at=acknowledgement.recorded_at,
        )
        result = CompletedWorkbenchActionResultV6(
            command="FOCUS_EVIDENCE",
            result_pin=acknowledgement.result_pin,
            receipt=receipt,
            context_projection=context_projection,
            recorded_at=acknowledgement.recorded_at,
        )
        return JSONResponse(
            content={
                **result.model_dump(mode="json", exclude={"context_projection"}),
                "context_projection": serialize_unified_workbench_context_v3(context_projection),
            },
            headers={"Workbench-Action-Version": _ACTION_VERSION_V6, "Cache-Control": "no-store"},
        )
    except PermissionError:
        raise HTTPException(403, "FORBIDDEN_SCOPE") from None
    except RuntimeContextError as exc:
        if exc.code in {"FORBIDDEN_ACTOR", "FORBIDDEN_SESSION", "SESSION_NOT_FOUND", "FORBIDDEN_SCOPE", "MISSING_WORKSPACE"}:
            raise HTTPException(403, "FORBIDDEN_SCOPE") from None
        if exc.code in {"OBJECT_NOT_FOUND", "NOT_FOUND", "UNRESOLVABLE_REF"}:
            raise HTTPException(404, exc.code) from None
        raise HTTPException(409, exc.code) from None
    except ValueError as exc:
        code = str(exc)
        if code in {"UNKNOWN_OBJECT", "EVIDENCE_OUT_OF_CLOSURE", "NO_CHANGE_REQUIRED", "REPLAY_BASIS_NOT_CLOSED", "NOT_RUN_CONTRACT_MISMATCH"}:
            raise HTTPException(422, code) from None
        if code in {"SOURCE_NOT_BOUND", "DESIGN_SNAPSHOT_REQUIRED"}:
            raise HTTPException(409, code) from None
        if code.startswith("STALE_") or code in {"REQUEST_CONFLICT", "CURSOR_PIN_MISMATCH", "CURSOR_FACT_MISMATCH", "ARTIFACT_PIN_MISMATCH", "PIN_MISMATCH"}:
            raise HTTPException(409, code) from None
        raise HTTPException(422, "MALFORMED_CONTRACT") from None
    finally:
        if extraction is not None:
            extraction._conn.close()
        if owners is not None:
            owners.build_store.close()
            owners.source_registry.close()


@router.post(_ACTION_PATH, name="workbench_actions")
async def workbench_actions(request: Request):
    require_authenticated_context(request)
    versions = request.headers.getlist("Accept-Workbench-Action-Version")
    if versions == [_ACTION_VERSION_V6]:
        return await _workbench_action_v6(request)
    if versions == ["temporal-context.v1"]:
        return await _temporal_action(request)
    raise HTTPException(status_code=422, detail="NEGOTIATION_REQUIRED")
