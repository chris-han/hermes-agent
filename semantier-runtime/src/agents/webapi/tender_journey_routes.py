"""Authenticated server projection for Tender journey continuity."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import require_route_authorization
from contracts.tender_journey import (
    TenderRuntimeCommandCompletionV1,
    TenderJourneyAdvanceRequestV1,
    TenderJourneyPrepareAuthorityRequestV1,
    TenderJourneyProduceNormativeSchemaRequestV1,
    TenderJourneyProduceNormativeSemanticsRequestV1,
    TenderJourneyReviewCompetencyRequestV1,
    TenderJourneyReviewValidationCasesRequestV1,
    TenderJourneyReviewCoverageRequestV1,
    TenderJourneyBlockRequestV1,
    TenderJourneyProjectionV1,
    TenderJourneySelectPolicyRequestV1,
    TenderJourneySelectRuntimeDocumentRequestV1,
    TenderJourneyRuntimeCommandV1,
    TenderJourneyBuildContextRequestV1,
    TenderJourneyDiscoverIntentRequestV1,
    TenderJourneySelectCertificationRequestV1,
    TenderJourneyCertifyContextRequestV1,
    TenderJourneyInspectionEpisodeV1,
    TenderJourneyReviewIntentRequestV1,
)
from contracts.tender_decision_basis import TenderDecisionBasisV1
from services.tender_journey_service import TenderJourneyService
from runtime_paths import sqlite_db_path


router = APIRouter(tags=["tender-journey"])


class _CreateJourneyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _LifecycleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)


def _context(request: Request, route_hint: str):
    ctx = require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
        route_hint=route_hint,
    )
    if not ctx.organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")
    return ctx


def _service(ctx) -> TenderJourneyService:
    return TenderJourneyService(ctx.workspace_root)


def _conflict(exc: ValueError, service: TenderJourneyService, ctx, journey_id: str) -> HTTPException:
    if str(exc) != "STALE_JOURNEY_REVISION":
        return HTTPException(status_code=409, detail=str(exc))
    head = service.get(journey_id=journey_id, organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id))
    return HTTPException(status_code=409, detail={
        "code": "STALE_JOURNEY_REVISION",
        "recovery": {
            "journey": service.project(head).model_dump(mode="json"),
            "suggested_action": "REFRESH_AND_REVIEW_CURRENT_JOURNEY",
        },
    })


@router.post("/api/tender-journeys", response_model=TenderJourneyProjectionV1)
def create_tender_journey(request: Request, body: _CreateJourneyRequest):
    ctx = _context(request, "POST /api/tender-journeys")
    service = _service(ctx)
    try:
        head = service.create(
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id),
            actor_ref=f"user:{ctx.user_id}",
        )
        return service.project(head)
    finally:
        service.close()


@router.get("/api/tender-journeys", response_model=list[TenderJourneyProjectionV1])
def list_tender_journeys(request: Request):
    ctx = _context(request, "GET /api/tender-journeys")
    service = _service(ctx)
    try:
        return [
            service.project(head)
            for head in service.list_active(
                organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id)
            )
        ]
    finally:
        service.close()


@router.get("/api/tender-journeys/preflight")
def get_tender_graph_preflight(request: Request):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from agents.tender_journey_tool_api import TenderJourneyToolAPI
    from services.tender_graph_preflight import tender_graph_preflight
    from services.runtime_errors import RuntimeContextError

    ctx = _context(request, "GET /api/tender-journeys/preflight")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            return tender_graph_preflight(ctx=ctx, artifacts=TenderJourneyToolAPI._artifacts(owners))
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/tender-journeys/{journey_id}/select-session", response_model=TenderJourneyProjectionV1)
def select_tender_journey_session(journey_id: str, request: Request, body: _CreateJourneyRequest):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.runtime_errors import RuntimeContextError

    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/select-session")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            head, _ = service.resolve_for_session(
                organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                actor_ref=f"user:{ctx.user_id}", session_ref=owners.session_ref, journey_id=journey_id,
            )
            return service.project(head)
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (KeyError, PermissionError) as exc:
            raise HTTPException(status_code=404, detail="Tender journey not found") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


@router.get("/api/tender-journeys/{journey_id}", response_model=TenderJourneyProjectionV1)
def get_tender_journey(journey_id: str, request: Request):
    ctx = _context(request, "GET /api/tender-journeys/{journey_id}")
    service = _service(ctx)
    try:
        try:
            head = service.get(
                journey_id=journey_id,
                organization_id=str(ctx.organization_id),
                workspace_id=str(ctx.workspace_id),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Tender journey not found") from exc
        return service.project(head)
    finally:
        service.close()


@router.get("/api/tender-journeys/{journey_id}/episodes/{transition_id}", response_model=TenderJourneyInspectionEpisodeV1)
def get_tender_journey_episode(journey_id: str, transition_id: str, request: Request):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from agents.webapi.tender_document_review_routes import _runtime_semantic_inspection_service
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "GET /api/tender-journeys/{journey_id}/episodes/{transition_id}")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            return service.resolve_inspection_episode(ctx=ctx, journey_id=journey_id, transition_id=transition_id,
                inspection_service=_runtime_semantic_inspection_service(ctx, read_only=True),
                authority_resolver=owners.authority_resolver)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="JOURNEY_EPISODE_NOT_FOUND") from exc
        except (RuntimeContextError, PermissionError) as exc:
            raise HTTPException(status_code=403, detail="FORBIDDEN") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


@router.get(
    "/api/tender-journeys/{journey_id}/episodes/{transition_id}/decision-basis",
    response_model=TenderDecisionBasisV1,
)
def get_tender_journey_decision_basis(
    journey_id: str,
    transition_id: str,
    finding_ref: str,
    finding_hash: str,
    request: Request,
):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from agents.webapi.tender_document_review_routes import _runtime_semantic_inspection_service
    from services.runtime_errors import RuntimeContextError
    from services.tender_decision_basis_service import TenderDecisionBasisService
    from contracts.canonical_source_ir import CanonicalSourceDocument
    from hashlib import sha256
    import json
    import re

    ctx = _context(
        request,
        "GET /api/tender-journeys/{journey_id}/episodes/{transition_id}/decision-basis",
    )
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    params = request.query_params
    if (
        params.getlist("finding_ref") != [finding_ref]
        or params.getlist("finding_hash") != [finding_hash]
        or not finding_ref.strip()
        or re.fullmatch(r"sha256:[0-9a-f]{64}", finding_hash) is None
    ):
        raise HTTPException(status_code=400, detail="INVALID_DECISION_BASIS_QUERY")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            inspection_reader = _runtime_semantic_inspection_service(ctx, read_only=True)
            episode = service.resolve_inspection_episode(
                ctx=ctx, journey_id=journey_id, transition_id=transition_id,
                inspection_service=inspection_reader,
                authority_resolver=owners.authority_resolver,
            )
            source_names: dict[str, str] = {}
            source_documents: dict[str, CanonicalSourceDocument] = {}
            selected = next(
                (item for item in episode.inspection_result.governed_findings if item.finding_id == finding_ref),
                None,
            )
            if selected is not None:
                for frozen in selected.evidence_refs:
                    source_ref = str(frozen.get("sourceIdentityRef", ""))
                    if not source_ref or source_ref in source_names:
                        continue
                    try:
                        _, identity = owners.source_registry.resolve_source_identity_lineage(
                            source_identity_ref=source_ref,
                            organization_id=str(ctx.organization_id),
                            workspace_id=str(ctx.workspace_id),
                        )
                    except ValueError:
                        continue
                    source_names[source_ref] = identity.original_name
            for region in episode.inspection_context.source_regions:
                pin = region.source_binding.canonical_source_ir
                match = re.fullmatch(
                    r"artifacts/canonical-source-ir/([0-9a-f]{64})/canonical_source_ir\.v1\.json",
                    pin.artifact_ref,
                )
                if match is None or pin.artifact_hash != f"sha256:{match.group(1)}":
                    continue
                path = ctx.workspace_root / pin.artifact_ref
                try:
                    if path.resolve().is_relative_to(ctx.workspace_root.resolve()) is False:
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    digest = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    if digest != match.group(1):
                        continue
                    document = CanonicalSourceDocument.model_validate(payload)
                    if document.source_identity_ref != region.source_binding.source_identity_ref:
                        continue
                    source_documents[pin.artifact_ref] = document
                except (OSError, ValueError):
                    continue
            # Optional present-day candidate context; never a saved decision input.
            from eos.legal_corpus_store import LegalCorpusStore
            from agents.webapi.tender_document_review_routes import _eos_path
            import sqlite3

            candidates = ()
            candidate_context_status = "UNAVAILABLE"
            if selected is not None:
                try:
                    candidates = LegalCorpusStore(_eos_path()).read_finding_candidate_trace(
                        organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id),
                        finding_ref=selected.finding_id, finding_hash=selected.finding_hash,
                        pattern_ref=selected.canonical_pattern_ref, pattern_hash=selected.canonical_pattern_hash,
                    )
                    candidate_context_status = "AVAILABLE" if candidates else "UNAVAILABLE"
                except sqlite3.Error:
                    candidate_context_status = "ERROR"
                    # Optional corpus absence must not hide the saved Finding.
                    candidates = ()
            return TenderDecisionBasisService().project(
                episode=episode, finding_ref=finding_ref, finding_hash=finding_hash,
                source_names=source_names,
                provision_candidates=candidates,
                candidate_context_status=candidate_context_status,
                source_documents=source_documents,
                recorded_content=inspection_reader.resolve_basis_content(
                    organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id), finding=selected,
                ) if selected is not None else {},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="JOURNEY_EPISODE_NOT_FOUND") from exc
        except (RuntimeContextError, PermissionError) as exc:
            raise HTTPException(status_code=403, detail="FORBIDDEN") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


def _lifecycle(journey_id: str, request: Request, body: _LifecycleRequest, operation: str):
    route = f"POST /api/tender-journeys/{{journey_id}}/{operation}"
    ctx = _context(request, route)
    service = _service(ctx)
    try:
        method = getattr(service, operation)
        try:
            head = method(
                journey_id=journey_id,
                organization_id=str(ctx.organization_id),
                workspace_id=str(ctx.workspace_id),
                expected_revision=body.expected_revision, actor_ref=f"user:{ctx.user_id}", surface="WORKBENCH",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Tender journey not found") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc
        return service.project(head)
    finally:
        service.close()


@router.get("/api/tender-journeys/{journey_id}/runtime-context-commands/{command_id}",
            response_model=TenderRuntimeCommandCompletionV1)
def confirm_tender_runtime_context_command(journey_id: str, command_id: str, request: Request,
                                          expected_revision: int, idempotency_key: str):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "GET /api/tender-journeys/{journey_id}/runtime-context-commands/{command_id}")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    if expected_revision < 0 or not idempotency_key.strip():
        raise HTTPException(status_code=422, detail="INVALID_COMMAND_IDENTITY")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            return service.confirm_runtime_context_command(ctx=ctx, journey_id=journey_id,
                session_ref=sessions[0], command_id=command_id,
                expected_revision=expected_revision, idempotency_key=idempotency_key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="JOURNEY_NOT_FOUND") from exc
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail="FORBIDDEN") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/tender-journeys/{journey_id}/select-policy", response_model=TenderJourneyProjectionV1)
def select_tender_journey_policy(journey_id: str, request: Request, body: TenderJourneySelectPolicyRequestV1):
    return _select_prepared_artifact(journey_id, request, body, action="select-policy")


@router.post("/api/tender-journeys/{journey_id}/select-document", response_model=TenderJourneyProjectionV1)
def select_tender_journey_document(journey_id: str, request: Request, body: TenderJourneySelectRuntimeDocumentRequestV1):
    return _select_prepared_artifact(journey_id, request, body, action="select-document")


@router.post("/api/tender-journeys/{journey_id}/form-runtime-context", response_model=TenderJourneyProjectionV1)
def form_tender_journey_runtime_context(journey_id: str, request: Request, body: TenderJourneyRuntimeCommandV1):
    return _select_prepared_artifact(journey_id, request, body, action="form-runtime-context")


def _select_prepared_artifact(journey_id, request, body, *, action):
    import os
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.workspace_artifact_service import WorkspaceArtifactService
    from services.workspace_artifact_preparation_service import (
        WorkspaceArtifactPreparationService, KnowledgeDesignSourcePreparationAdapter,
    )
    from services.runtime_errors import RuntimeContextError
    from storage.knowledge_extraction_store import KnowledgeExtractionStore

    ctx = _context(request, f"POST /api/tender-journeys/{{journey_id}}/{action}")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            extraction = KnowledgeExtractionStore(owners.source_registry._db_path)
            stack.callback(extraction._conn.close)
            max_bytes = int(os.environ.get("SEMANTIER_WORKSPACE_ARTIFACT_MAX_BYTES", "20971520"))
            if max_bytes <= 0:
                raise ValueError("INVALID_BYTE_LENGTH")
            artifacts = WorkspaceArtifactService(build_store=owners.build_store, authority_resolver=owners.authority_resolver, max_bytes=max_bytes)
            preparation = WorkspaceArtifactPreparationService(
                artifact_service=artifacts,
                design_adapter=KnowledgeDesignSourcePreparationAdapter(
                    artifact_service=artifacts, knowledge_store=owners.source_registry, extraction_store=extraction,
                ),
            )
            if action == "form-runtime-context":
                from agents.webapi.tender_document_review_routes import _runtime_semantic_inspection_service
                head = service.form_runtime_context(
                    ctx=ctx, journey_id=journey_id, expected_revision=body.expected_revision,
                    preparation_service=preparation, inspection_service=_runtime_semantic_inspection_service(ctx),
                    authority_resolver=owners.authority_resolver, command_id=body.command_id,
                    idempotency_key=body.idempotency_key, surface="WORKBENCH",
                )
            else:
                select = service.select_runtime_document if action == "select-document" else service.select_policy
                head = select(
                    ctx=ctx, journey_id=journey_id, expected_revision=body.expected_revision,
                    artifact_pin=body.artifact_pin, preparation_pin=body.preparation_pin,
                    preparation_service=preparation, command_id=body.command_id,
                    idempotency_key=body.idempotency_key, surface="WORKBENCH",
                )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="JOURNEY_NOT_FOUND") from exc
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail="FORBIDDEN") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc
        return service.project(head)


@router.post("/api/tender-journeys/{journey_id}/discover-intent", response_model=TenderJourneyProjectionV1)
def discover_tender_journey_intent(journey_id: str, request: Request, body: TenderJourneyDiscoverIntentRequestV1):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from agents.corpus_graph_schema_capability_factory import build_corpus_graph_schema_capability
    from services.runtime_errors import RuntimeContextError
    from contracts.ai_grounding import StructuredGenerationError

    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/discover-intent")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            completed = service.replay_intent_discovery(
                ctx=ctx, journey_id=journey_id, expected_revision=body.expected_revision,
                command_id=body.command_id, idempotency_key=body.idempotency_key, surface="WORKBENCH",
            )
            if completed is not None:
                return service.project(completed)
            capability = build_corpus_graph_schema_capability(
                workspace_root=ctx.workspace_root, workspace_id=ctx.workspace_id,
                user_id=ctx.user_id, organization_id=ctx.organization_id, source_registry=owners.source_registry,
            )
            head = service.discover_intent(
                ctx=ctx, journey_id=journey_id, expected_revision=body.expected_revision,
                capability=capability, authority_resolver=owners.authority_resolver,
                command_id=body.command_id, idempotency_key=body.idempotency_key, surface="WORKBENCH",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="JOURNEY_OR_SOURCE_NOT_FOUND") from exc
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail="FORBIDDEN") from exc
        except StructuredGenerationError as exc:
            raise HTTPException(status_code=409, detail="GRAPH_INTENT_PROVIDER_CONFIGURATION_UNAVAILABLE") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc
        return service.project(head)


@router.post("/api/tender-journeys/{journey_id}/build-context", response_model=TenderJourneyProjectionV1)
def build_tender_journey_context(journey_id: str, request: Request, body: TenderJourneyBuildContextRequestV1):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.runtime_errors import RuntimeContextError
    from storage.graph_intent_brief_store import GraphIntentBriefStore

    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/build-context")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            intents = GraphIntentBriefStore(ctx.workspace_root, organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id, source_registry=owners.source_registry)
            head = service.build_context(
                ctx=ctx, journey_id=journey_id, expected_revision=body.expected_revision,
                graph_intent_pin=body.graph_intent_pin, intent_store=intents,
                build_store=owners.build_store, authority_resolver=owners.authority_resolver,
                command_id=body.command_id, idempotency_key=body.idempotency_key, surface="WORKBENCH",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="JOURNEY_OR_INTENT_NOT_FOUND") from exc
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail="FORBIDDEN") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc
        return service.project(head)


@router.post("/api/tender-journeys/{journey_id}/certify-context", response_model=TenderJourneyProjectionV1)
def certify_tender_journey_context(journey_id: str, request: Request, body: TenderJourneyCertifyContextRequestV1):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.runtime_errors import RuntimeContextError

    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/certify-context")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            head = service.certify_context(
                ctx=ctx, journey_id=journey_id, expected_revision=body.expected_revision,
                competency_pin=body.competency_pin, authority_pin=body.authority_pin,
                execution_plan_pin=body.execution_plan_pin, schema_adequacy_pin=body.schema_adequacy_pin,
                representative_evaluation_pin=body.representative_evaluation_pin, paired_suite_pin=body.paired_suite_pin,
                build_store=owners.build_store, authority_resolver=owners.authority_resolver,
                command_id=body.command_id, idempotency_key=body.idempotency_key, surface="WORKBENCH",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="JOURNEY_NOT_FOUND") from exc
        except (RuntimeContextError, PermissionError) as exc:
            raise HTTPException(status_code=403, detail="FORBIDDEN") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc
        return service.project(head)


@router.post("/api/tender-journeys/{journey_id}/select-certification", response_model=TenderJourneyProjectionV1)
def select_tender_journey_certification(journey_id: str, request: Request, body: TenderJourneySelectCertificationRequestV1):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.runtime_errors import RuntimeContextError

    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/select-certification")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            head = service.select_certification(
                ctx=ctx, journey_id=journey_id, expected_revision=body.expected_revision,
                receipt_pin=body.receipt_pin, build_store=owners.build_store, authority_resolver=owners.authority_resolver,
                command_id=body.command_id, idempotency_key=body.idempotency_key, surface="WORKBENCH",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="JOURNEY_NOT_FOUND") from exc
        except (RuntimeContextError, PermissionError) as exc:
            raise HTTPException(status_code=403, detail="FORBIDDEN") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc
        return service.project(head)


@router.post("/api/tender-journeys/{journey_id}/inspect", response_model=TenderJourneyProjectionV1)
async def inspect_tender_journey_document(journey_id: str, request: Request, body: TenderJourneyRuntimeCommandV1):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from agents.webapi.tender_document_review_routes import _runtime_semantic_inspection_service
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/inspect")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            head = await service.inspect_document(
                ctx=ctx, journey_id=journey_id, expected_revision=body.expected_revision,
                inspection_service=_runtime_semantic_inspection_service(ctx), authority_resolver=owners.authority_resolver,
                command_id=body.command_id, idempotency_key=body.idempotency_key, surface="WORKBENCH",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="JOURNEY_NOT_FOUND") from exc
        except (RuntimeContextError, PermissionError) as exc:
            raise HTTPException(status_code=403, detail="FORBIDDEN") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc
        return service.project(head)


@router.post("/api/tender-journeys/{journey_id}/next-document", response_model=TenderJourneyProjectionV1)
def next_tender_journey_document(journey_id: str, request: Request, body: TenderJourneyRuntimeCommandV1):
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/next-document")
    service = _service(ctx)
    try:
        head = service.next_document(
            journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            expected_revision=body.expected_revision, actor_ref=f"user:{ctx.user_id}", surface="WORKBENCH",
            command_id=body.command_id, idempotency_key=body.idempotency_key,
        )
        return service.project(head)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="JOURNEY_NOT_FOUND") from exc
    except ValueError as exc:
        raise _conflict(exc, service, ctx, journey_id) from exc
    finally:
        service.close()


@router.post("/api/tender-journeys/{journey_id}/review-intent", response_model=TenderJourneyProjectionV1)
def review_tender_journey_intent(journey_id: str, request: Request, body: TenderJourneyReviewIntentRequestV1):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from agents.corpus_graph_schema_capability_factory import build_corpus_graph_schema_capability
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/review-intent")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            capability = build_corpus_graph_schema_capability(workspace_root=ctx.workspace_root,
                organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, user_id=ctx.user_id,
                source_registry=owners.source_registry, graph_intent_inference_enabled=False)
            head = service.review_intent(ctx=ctx, journey_id=journey_id, capability=capability,
                authority_resolver=owners.authority_resolver, surface="WORKBENCH",
                **{name: getattr(body, name) for name in type(body).model_fields if name != "surface"})
            return service.project(head)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="JOURNEY_NOT_FOUND") from exc
        except (RuntimeContextError, PermissionError) as exc:
            raise HTTPException(status_code=403, detail="FORBIDDEN") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


@router.post("/api/tender-journeys/{journey_id}/advance", response_model=TenderJourneyProjectionV1)
def advance_tender_journey(journey_id: str, request: Request, body: TenderJourneyAdvanceRequestV1):
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/advance")
    if body.to_phase == "POLICY_READY":
        raise HTTPException(status_code=409, detail="GOVERNED_POLICY_SELECTION_REQUIRED")
    if body.to_phase in {"BUILDING_CONTEXT", "REVIEW_INTENT", "CERTIFYING_CONTEXT"}:
        raise HTTPException(status_code=409, detail="GOVERNED_CONTEXT_BUILD_REQUIRED")
    if body.to_phase == "RUNTIME_BINDING_REQUIRED":
        raise HTTPException(status_code=409, detail="GOVERNED_CERTIFICATION_SELECTION_REQUIRED")
    if body.to_phase == "NEEDS_RUNTIME_DOCUMENT":
        raise HTTPException(status_code=409, detail="JOURNEY_NEXT_DOCUMENT_COMMAND_REQUIRED")
    if body.to_phase == "REVIEW_RESULTS":
        raise HTTPException(status_code=409, detail="GOVERNED_RUNTIME_INSPECTION_REQUIRED")
    if body.to_phase == "INSPECTING":
        raise HTTPException(status_code=409, detail="GOVERNED_RUNTIME_CONTEXT_FORMATION_REQUIRED")
    if body.to_phase == "FORMING_RUNTIME_CONTEXT":
        raise HTTPException(status_code=409, detail="GOVERNED_RUNTIME_DOCUMENT_SELECTION_REQUIRED")
    if body.to_phase == "CONTEXT_READY":
        raise HTTPException(status_code=409, detail="GOVERNED_RUNTIME_BINDING_REQUIRED")
    service = _service(ctx)
    try:
        try:
            head = service.advance(
                journey_id=journey_id,
                organization_id=str(ctx.organization_id),
                workspace_id=str(ctx.workspace_id),
                expected_revision=body.expected_revision,
                to_phase=body.to_phase,
                cause_code=body.cause_code,
                actor_ref=f"user:{ctx.user_id}",
                surface="WORKBENCH",
                command_id=body.command_id,
                idempotency_key=body.idempotency_key,
                operation_receipt=body.operation_receipt,
                artifact_bindings=body.artifact_bindings,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Tender journey not found") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc
        return service.project(head)
    finally:
        service.close()


@router.post("/api/tender-journeys/{journey_id}/block", response_model=TenderJourneyProjectionV1)
def block_tender_journey(journey_id: str, request: Request, body: TenderJourneyBlockRequestV1):
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/block")
    if body.artifact_bindings:
        raise HTTPException(status_code=409, detail="SERVER_OWNED_JOURNEY_BINDINGS_REQUIRED")
    service = _service(ctx)
    try:
        try:
            service.get(journey_id=journey_id, organization_id=str(ctx.organization_id),
                workspace_id=str(ctx.workspace_id))
            raise HTTPException(status_code=409, detail="SERVER_OWNED_JOURNEY_BLOCKER_REQUIRED")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Tender journey not found") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc
    finally:
        service.close()


@router.post("/api/tender-journeys/{journey_id}/bind-runtime", response_model=TenderJourneyProjectionV1)
def bind_tender_journey_runtime(journey_id: str, request: Request, body: _LifecycleRequest):
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/bind-runtime")
    service = _service(ctx)
    try:
        try:
            head = service.bind_current_runtime_context(
                journey_id=journey_id,
                organization_id=str(ctx.organization_id),
                workspace_id=str(ctx.workspace_id),
                expected_revision=body.expected_revision,
                actor_ref=f"user:{ctx.user_id}",
                surface="WORKBENCH",
                command_id=f"bind-runtime:{journey_id}:{body.expected_revision}",
                idempotency_key=f"bind-runtime:{journey_id}:{body.expected_revision}",
                runtime_db_path=sqlite_db_path("eos.db", ".semantier/eos.db"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Tender journey not found") from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc
        return service.project(head)
    finally:
        service.close()


@router.post("/api/tender-journeys/{journey_id}/pause", response_model=TenderJourneyProjectionV1)
def pause_tender_journey(journey_id: str, request: Request, body: _LifecycleRequest):
    return _lifecycle(journey_id, request, body, "pause")


@router.post("/api/tender-journeys/{journey_id}/resume", response_model=TenderJourneyProjectionV1)
def resume_tender_journey(journey_id: str, request: Request, body: _LifecycleRequest):
    return _lifecycle(journey_id, request, body, "resume")


@router.post("/api/tender-journeys/{journey_id}/cancel", response_model=TenderJourneyProjectionV1)
def cancel_tender_journey(journey_id: str, request: Request, body: _LifecycleRequest):
    return _lifecycle(journey_id, request, body, "cancel")


@router.post("/api/tender-journeys/{journey_id}/prepare-authority", response_model=TenderJourneyProjectionV1)
def prepare_tender_journey_authority(journey_id: str, request: Request, body: TenderJourneyPrepareAuthorityRequestV1):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/prepare-authority")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            fields = {name:getattr(body, name) for name in type(body).model_fields if name != "surface"}
            head = service.prepare_authority_dependencies(ctx=ctx, journey_id=journey_id, **fields,
                surface="WORKBENCH", build_store=owners.build_store, authority_resolver=owners.authority_resolver)
            return service.project(head)
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


@router.post("/api/tender-journeys/{journey_id}/produce-normative-schema", response_model=TenderJourneyProjectionV1)
async def produce_tender_journey_normative_schema(
    journey_id: str, request: Request, body: TenderJourneyProduceNormativeSchemaRequestV1,
):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from agents.corpus_graph_schema_capability_factory import build_corpus_graph_schema_capability
    from agents.normative_schema_production_factory import build_normative_schema_producer
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/produce-normative-schema")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            capability = build_corpus_graph_schema_capability(workspace_root=ctx.workspace_root,
                organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                user_id=ctx.user_id, source_registry=owners.source_registry,
                graph_intent_inference_enabled=False)
            fields = {name: getattr(body, name) for name in type(body).model_fields if name != "surface"}
            head = await service.produce_normative_schema(ctx=ctx, journey_id=journey_id, **fields,
                surface="WORKBENCH", capability=capability, producer=build_normative_schema_producer(request_context=ctx),
                build_store=owners.build_store, authority_resolver=owners.authority_resolver)
            return service.project(head)
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


@router.post("/api/tender-journeys/{journey_id}/produce-normative-semantics", response_model=TenderJourneyProjectionV1)
async def produce_tender_journey_normative_semantics(
    journey_id: str, request: Request, body: TenderJourneyProduceNormativeSemanticsRequestV1,
):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from agents.normative_schema_production_factory import build_normative_semantic_producer
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/produce-normative-semantics")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            preparation = None
            if body.source_role == "RUNTIME_TARGET":
                import os
                from services.workspace_artifact_service import WorkspaceArtifactService
                from services.workspace_artifact_preparation_service import WorkspaceArtifactPreparationService, KnowledgeDesignSourcePreparationAdapter
                from storage.knowledge_extraction_store import KnowledgeExtractionStore
                extraction = KnowledgeExtractionStore(owners.source_registry._db_path)
                stack.callback(extraction._conn.close)
                max_bytes = int(os.environ.get("SEMANTIER_WORKSPACE_ARTIFACT_MAX_BYTES", "20971520"))
                if max_bytes <= 0:
                    raise ValueError("INVALID_BYTE_LENGTH")
                artifacts = WorkspaceArtifactService(build_store=owners.build_store, authority_resolver=owners.authority_resolver, max_bytes=max_bytes)
                preparation = WorkspaceArtifactPreparationService(artifact_service=artifacts,
                    design_adapter=KnowledgeDesignSourcePreparationAdapter(artifact_service=artifacts,
                        knowledge_store=owners.source_registry, extraction_store=extraction))
            fields = {name: getattr(body, name) for name in type(body).model_fields if name != "surface"}
            head = await service.produce_normative_semantics(ctx=ctx, journey_id=journey_id, **fields,
                surface="WORKBENCH", producer=build_normative_semantic_producer(request_context=ctx),
                build_store=owners.build_store, authority_resolver=owners.authority_resolver,source_registry=owners.source_registry,
                preparation_service=preparation)
            return service.project(head)
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


@router.post("/api/tender-journeys/{journey_id}/propose-validation-cases", response_model=TenderJourneyProjectionV1)
async def propose_tender_journey_validation_cases(
    journey_id: str, request: Request, body: TenderJourneyRuntimeCommandV1,
):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from agents.normative_schema_production_factory import build_source_validation_case_producer
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/propose-validation-cases")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            fields = {name: getattr(body, name) for name in type(body).model_fields if name != "surface"}
            head = await service.propose_validation_cases(ctx=ctx, journey_id=journey_id, **fields,
                surface="WORKBENCH", producer=build_source_validation_case_producer(request_context=ctx),
                build_store=owners.build_store, authority_resolver=owners.authority_resolver)
            return service.project(head)
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


@router.post("/api/tender-journeys/{journey_id}/review-validation-cases", response_model=TenderJourneyProjectionV1)
def review_tender_journey_validation_cases(
    journey_id: str, request: Request, body: TenderJourneyReviewValidationCasesRequestV1,
):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/review-validation-cases")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            fields = {name: getattr(body, name) for name in type(body).model_fields if name != "surface"}
            head = service.review_validation_cases(ctx=ctx, journey_id=journey_id, **fields,
                surface="WORKBENCH",
                build_store=owners.build_store, authority_resolver=owners.authority_resolver)
            return service.project(head)
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


@router.post("/api/tender-journeys/{journey_id}/validate-source-semantics", response_model=TenderJourneyProjectionV1)
def validate_tender_journey_source_semantics(
    journey_id: str, request: Request, body: TenderJourneyRuntimeCommandV1,
):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/validate-source-semantics")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            fields = {name: getattr(body, name) for name in type(body).model_fields if name != "surface"}
            head = service.validate_source_semantics(ctx=ctx, journey_id=journey_id, **fields,
                surface="WORKBENCH",
                build_store=owners.build_store, authority_resolver=owners.authority_resolver)
            return service.project(head)
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


@router.post("/api/tender-journeys/{journey_id}/propose-competency", response_model=TenderJourneyProjectionV1)
def propose_tender_journey_competency(journey_id: str, request: Request, body: TenderJourneyRuntimeCommandV1):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/propose-competency")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            from storage.graph_intent_brief_store import GraphIntentBriefStore
            intents = GraphIntentBriefStore(ctx.workspace_root, organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id, source_registry=owners.source_registry)
            fields = {name:getattr(body, name) for name in type(body).model_fields if name != "surface"}
            head = service.propose_competency(ctx=ctx, journey_id=journey_id, intent_store=intents, **fields,
                surface="WORKBENCH", build_store=owners.build_store, authority_resolver=owners.authority_resolver)
            return service.project(head)
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


@router.post("/api/tender-journeys/{journey_id}/review-competency", response_model=TenderJourneyProjectionV1)
def review_tender_journey_competency(journey_id: str, request: Request, body: TenderJourneyReviewCompetencyRequestV1):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/review-competency")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            fields = {name:getattr(body, name) for name in type(body).model_fields if name != "surface"}
            head = service.review_competency(ctx=ctx, journey_id=journey_id, **fields,
                surface="WORKBENCH", build_store=owners.build_store, authority_resolver=owners.authority_resolver)
            return service.project(head)
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


@router.post("/api/tender-journeys/{journey_id}/review-source-coverage", response_model=TenderJourneyProjectionV1)
def review_tender_journey_source_coverage(journey_id: str, request: Request, body: TenderJourneyReviewCoverageRequestV1):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, "POST /api/tender-journeys/{journey_id}/review-source-coverage")
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            fields = {name:getattr(body, name) for name in type(body).model_fields if name != "surface"}
            head = service.review_source_coverage(ctx=ctx, journey_id=journey_id, **fields,
                surface="WORKBENCH", build_store=owners.build_store, authority_resolver=owners.authority_resolver)
            return service.project(head)
        except RuntimeContextError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise _conflict(exc, service, ctx, journey_id) from exc


__all__ = ["router"]


from contracts.requirements_reconciliation import AssessRequirementsV1, ReviewRequirementV1


def _requirements_route(journey_id, request, route_hint, body=None, source_role=None):
    from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
    from services.runtime_errors import RuntimeContextError
    ctx = _context(request, route_hint)
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(status_code=401, detail="AUTHORITY_CONTEXT_REQUIRED")
    with ExitStack() as stack:
        service = _service(ctx)
        stack.callback(service.close)
        try:
            owners = _workbench_owner_context(ctx=ctx, session_locator=sessions[0], channel="web")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            head, _ = service.resolve_for_session(organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,actor_ref="user:"+ctx.user_id,
                session_ref=owners.session_ref,journey_id=journey_id)
            common = dict(ctx=ctx,journey_id=journey_id,build_store=owners.build_store,authority_resolver=owners.authority_resolver)
            if body is None:
                review = service.read_requirements(**common,source_role=source_role)
            else:
                head, review = service.requirements_command(**common,command=body,surface="WORKBENCH",source_registry=owners.source_registry)
            return {"journey":service.project(head).model_dump(mode="json"),
                "requirements_review":review.model_dump(mode="json") if review else None}
        except KeyError as exc:
            raise HTTPException(status_code=404,detail="JOURNEY_NOT_FOUND") from exc
        except (RuntimeContextError,PermissionError) as exc:
            raise HTTPException(status_code=403,detail="FORBIDDEN") from exc
        except ValueError as exc:
            raise _conflict(exc,service,ctx,journey_id) from exc


@router.get("/api/tender-journeys/{journey_id}/requirements")
def read_tender_requirements(journey_id: str, request: Request, source_role: Literal["DESIGN_SOURCE", "ACTIVE_RULE_BASIS", "RUNTIME_TARGET", "PREPARED_SENSITIVITY"] | None = None):
    from agents.tender_journey_tool_api import JourneyReadRequirementsRequest
    role = JourneyReadRequirementsRequest(source_role=source_role).source_role
    return _requirements_route(journey_id,request,"GET /api/tender-journeys/{journey_id}/requirements",source_role=role)


@router.post("/api/tender-journeys/{journey_id}/assess-requirements")
def assess_tender_requirements(journey_id: str, request: Request, body: AssessRequirementsV1):
    return _requirements_route(journey_id,request,"POST /api/tender-journeys/{journey_id}/assess-requirements",body)


@router.post("/api/tender-journeys/{journey_id}/review-requirement")
def review_tender_requirement(journey_id: str, request: Request, body: ReviewRequirementV1):
    return _requirements_route(journey_id,request,"POST /api/tender-journeys/{journey_id}/review-requirement",body)
