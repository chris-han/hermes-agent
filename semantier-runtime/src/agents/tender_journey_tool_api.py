"""Authenticated Chat composition over the shared Tender journey owners."""
from __future__ import annotations

import asyncio
from contextlib import ExitStack
from typing import Any, Literal
from urllib.parse import quote, unquote, urlencode

from pydantic import BaseModel, ConfigDict, Field

from agents.auth_session import RequestContext
from agents.governed_tool_context import resolve_authenticated_tool_context
from contracts.tender_journey import (
    TenderJourneyBuildContextRequestV1,
    TenderJourneyPrepareAuthorityRequestV1,
    TenderJourneyProduceNormativeSchemaRequestV1,
    TenderJourneyReviewCompetencyRequestV1,
    TenderJourneyReviewValidationCasesRequestV1,
    TenderJourneyReviewCoverageRequestV1,
    TenderJourneyCertifyContextRequestV1,
    TenderJourneyDiscoverIntentRequestV1,
    TenderJourneyReviewIntentRequestV1,
    TenderJourneyRuntimeCommandV1,
    TenderJourneySelectCertificationRequestV1,
    TenderJourneySelectPolicyRequestV1,
    TenderJourneySelectRuntimeDocumentRequestV1,
)
from services.tender_journey_service import TenderJourneyService
from contracts.ontology_building import ArtifactPinV1


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JourneyResolveRequest(_Closed):
    journey_id: str | None = Field(default=None, min_length=1)


class JourneyCreateRequest(_Closed):
    command_id: str = Field(min_length=1)


class JourneyLifecycleRequest(_Closed):
    expected_revision: int = Field(ge=0)


class JourneyReadEpisodeRequest(_Closed):
    transition_id: str = Field(min_length=1)


class JourneyImportUploadRequest(_Closed):
    request_id: str = Field(min_length=1)
    expected_collection_revision: int = Field(ge=0)
    upload_ref: str = Field(min_length=1, description="Canonical percent-encoded session-relative upload reference; copy exactly from list_uploads")
    upload_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class JourneyListUploadsRequest(_Closed):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class JourneyPrepareArtifactRequest(_Closed):
    request_id: str = Field(min_length=1)
    artifact_pin: ArtifactPinV1
    purpose: Literal["DESIGN_SOURCE", "RUNTIME_TARGET"]
    expected_preparation_pin: ArtifactPinV1 | None = None
    expected_preparation_revision: int | None = Field(default=None, ge=0)


_ARTIFACT_ACTIONS = ("preflight", "list_uploads", "list_artifacts", "import_upload", "prepare_artifact")


_ACTION_MODELS = {
    "preflight": _Closed,
    "propose_competency": TenderJourneyRuntimeCommandV1,
    "produce_normative_schema": TenderJourneyProduceNormativeSchemaRequestV1,
    "produce_normative_semantics": TenderJourneyRuntimeCommandV1,
    "propose_validation_cases": TenderJourneyRuntimeCommandV1,
    "review_validation_cases": TenderJourneyReviewValidationCasesRequestV1,
    "validate_source_semantics": TenderJourneyRuntimeCommandV1,
    "review_competency": TenderJourneyReviewCompetencyRequestV1,
    "review_source_coverage": TenderJourneyReviewCoverageRequestV1,
    "prepare_authority_dependencies": TenderJourneyPrepareAuthorityRequestV1,
    "list_uploads": JourneyListUploadsRequest,
    "list_artifacts": _Closed,
    "import_upload": JourneyImportUploadRequest,
    "prepare_artifact": JourneyPrepareArtifactRequest,
    "create": JourneyCreateRequest,
    "pause": JourneyLifecycleRequest,
    "resume": JourneyLifecycleRequest,
    "cancel": JourneyLifecycleRequest,
    "select_policy": TenderJourneySelectPolicyRequestV1,
    "select_document": TenderJourneySelectRuntimeDocumentRequestV1,
    "discover_intent": TenderJourneyDiscoverIntentRequestV1,
    "build_context": TenderJourneyBuildContextRequestV1,
    "certify_context": TenderJourneyCertifyContextRequestV1,
    "select_certification": TenderJourneySelectCertificationRequestV1,
    "bind_runtime": TenderJourneyRuntimeCommandV1,
    "form_runtime_context": TenderJourneyRuntimeCommandV1,
    "inspect": TenderJourneyRuntimeCommandV1,
    "next_document": TenderJourneyRuntimeCommandV1,
    "read_episode": JourneyReadEpisodeRequest,
    "read_intent": _Closed,
    "review_intent": TenderJourneyReviewIntentRequestV1,
}


class JourneyActionRequest(_Closed):
    action: Literal[
        "preflight", "list_uploads", "list_artifacts", "import_upload", "prepare_artifact",
        "review_source_coverage", "review_competency", "propose_competency", "prepare_authority_dependencies", "create", "pause", "resume", "cancel", "select_policy", "select_document",
        "discover_intent", "build_context", "produce_normative_schema", "produce_normative_semantics", "propose_validation_cases", "review_validation_cases", "validate_source_semantics", "certify_context", "select_certification",
        "bind_runtime", "form_runtime_context", "inspect", "next_document", "read_episode", "read_intent", "review_intent",
    ]
    journey_id: str | None = Field(default=None, min_length=1)
    arguments: dict[str, Any]


def _schemas(actions) -> dict[str, Any]:
    result = {}
    for action in actions:
        model = _ACTION_MODELS[action]
        schema = model.model_json_schema()
        schema["properties"].pop("surface", None)
        schema["properties"].pop("schema_version", None)
        result[action] = schema
    return result


def _episode_url(journey_id: str, transition_id: str, session_ref: str) -> str:
    return "/ontology-workbench2?" + urlencode({"journeyRef": journey_id,
        "episodeRef": transition_id, "episodeSessionRef": session_ref, "view": "evidence"})


class TenderJourneyToolAPI:
    def __init__(self, *, context: RequestContext, session_ref: str):
        if not isinstance(context, RequestContext) or not context.authenticated or not context.organization_id:
            raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
        if not session_ref:
            raise ValueError("AUTHENTICATED_SESSION_REQUIRED")
        self._context = context
        self._session_ref = session_ref

    async def invoke(self, *, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Synchronous owners may invoke async providers via asyncio.run(). Keep
        # their entire SQLite lifetime on one worker, preserving turn context.
        return await asyncio.to_thread(self._invoke, operation=operation, payload=payload)

    def _invoke(self, *, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        # The same production owner factory verifies DB-backed actor and session
        # membership for both Web and Chat. Caller payloads cannot supply either.
        from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context

        if operation == "resolve":
            request = JourneyResolveRequest.model_validate(payload)
            body = None
        elif operation == "action":
            request = JourneyActionRequest.model_validate(payload)
            if "surface" in request.arguments or "schema_version" in request.arguments:
                raise ValueError("SERVER_OWNED_JOURNEY_FIELDS")
            if request.action == "create" and request.journey_id is not None:
                raise ValueError("CREATE_JOURNEY_ID_IS_SERVER_OWNED")
            body = _ACTION_MODELS[request.action].model_validate(request.arguments)
        else:
            raise ValueError("JOURNEY_TOOL_OPERATION_UNSUPPORTED")
        ctx = self._context
        with ExitStack() as stack:
            owners = _workbench_owner_context(ctx=ctx, session_locator=self._session_ref, channel="chat")
            stack.callback(owners.source_registry.close)
            stack.callback(owners.build_store.close)
            if operation == "action" and request.action in _ARTIFACT_ACTIONS:
                result = self._artifact_action(ctx, owners, stack, request.action, body)
                return {**result, "action_schemas": _schemas((*_ARTIFACT_ACTIONS, "create"))}
            service = TenderJourneyService(ctx.workspace_root)
            stack.callback(service.close)
            scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
            binding = dict(**scope, actor_ref=f"user:{ctx.user_id}", session_ref=owners.session_ref)
            if operation == "action" and request.action == "create":
                head = service.create_for_session(**binding, command_id=body.command_id)
            else:
                head, choices = service.resolve_for_session(**binding, journey_id=request.journey_id)
                if head is None:
                    return {
                        "status": "CHOICE_REQUIRED" if choices else "NO_JOURNEY",
                        "journeys": [service.project(item).model_dump(mode="json") for item in choices],
                        "action_schemas": _schemas((*_ARTIFACT_ACTIONS, "create")),
                    }
                if operation == "action":
                    if request.action in {"read_intent", "review_intent"}:
                        from agents.corpus_graph_schema_capability_factory import build_corpus_graph_schema_capability
                        capability = build_corpus_graph_schema_capability(workspace_root=ctx.workspace_root,
                            **scope, user_id=ctx.user_id, source_registry=owners.source_registry,
                            graph_intent_inference_enabled=False)
                        if request.action == "read_intent":
                            pins = [item.pin for item in head.artifact_bindings if item.role == "graph_intent"]
                            if len(pins) != 1:
                                raise ValueError("JOURNEY_GRAPH_INTENT_REQUIRED")
                            return {"graph_intent": capability.get_graph_intent(pins[0].artifact_ref,
                                graph_intent_hash=pins[0].artifact_hash)}
                    if request.action == "read_episode":
                        from agents.webapi.tender_document_review_routes import _runtime_semantic_inspection_service
                        episode = service.resolve_inspection_episode(ctx=ctx, journey_id=head.journey_id,
                            transition_id=body.transition_id, authority_resolver=owners.authority_resolver,
                            inspection_service=_runtime_semantic_inspection_service(ctx, read_only=True))
                        return {"episode": episode.model_dump(mode="json", by_alias=True),
                            "workbench_url": _episode_url(head.journey_id, body.transition_id, owners.session_ref)}
                    fields = {name: getattr(body, name) for name in type(body).model_fields
                              if name not in {"schema_version", "surface"}}
                    common = dict(journey_id=head.journey_id, surface="CHAT", **fields)
                    action = request.action
                    if action == "review_intent":
                        head = service.review_intent(ctx=ctx, **common, capability=capability,
                                                     authority_resolver=owners.authority_resolver)
                    elif action in {"pause", "resume", "cancel", "next_document"}:
                        method = getattr(service, action)
                        head = method(**scope, actor_ref=f"user:{ctx.user_id}", **common)
                    elif action == "bind_runtime":
                        from runtime_paths import sqlite_db_path
                        head = service.bind_current_runtime_context(
                            **scope, actor_ref=f"user:{ctx.user_id}", **common,
                            runtime_db_path=sqlite_db_path("eos.db", ".semantier/eos.db"),
                        )
                    elif action in {"review_validation_cases", "validate_source_semantics", "review_source_coverage", "review_competency", "prepare_authority_dependencies", "certify_context", "select_certification"}:
                        head = getattr(service, action)(ctx=ctx, **common, build_store=owners.build_store,
                                                       authority_resolver=owners.authority_resolver)
                    elif action in {"build_context", "propose_competency"}:
                        from storage.graph_intent_brief_store import GraphIntentBriefStore
                        intents = GraphIntentBriefStore(ctx.workspace_root, **scope, source_registry=owners.source_registry)
                        head = getattr(service, action)(ctx=ctx, **common, intent_store=intents,
                                                     build_store=owners.build_store, authority_resolver=owners.authority_resolver)
                    elif action == "discover_intent":
                        from agents.corpus_graph_schema_capability_factory import build_corpus_graph_schema_capability
                        head = service.replay_intent_discovery(ctx=ctx, **common)
                        if head is None:
                            capability = build_corpus_graph_schema_capability(
                                workspace_root=ctx.workspace_root, **scope, user_id=ctx.user_id,
                                source_registry=owners.source_registry,
                            )
                            head = service.discover_intent(ctx=ctx, **common, capability=capability,
                                                           authority_resolver=owners.authority_resolver)
                    elif action == "produce_normative_schema":
                        from agents.corpus_graph_schema_capability_factory import build_corpus_graph_schema_capability
                        from agents.normative_schema_production_factory import build_normative_schema_producer
                        capability = build_corpus_graph_schema_capability(workspace_root=ctx.workspace_root,
                            **scope, user_id=ctx.user_id, source_registry=owners.source_registry,
                            graph_intent_inference_enabled=False)
                        head = asyncio.run(service.produce_normative_schema(ctx=ctx, **common,
                            capability=capability, producer=build_normative_schema_producer(),
                            build_store=owners.build_store, authority_resolver=owners.authority_resolver))
                    elif action == "produce_normative_semantics":
                        from agents.normative_schema_production_factory import build_normative_semantic_producer
                        head = asyncio.run(service.produce_normative_semantics(ctx=ctx, **common,
                            producer=build_normative_semantic_producer(), build_store=owners.build_store,
                            authority_resolver=owners.authority_resolver))
                    elif action == "propose_validation_cases":
                        from agents.normative_schema_production_factory import build_source_validation_case_producer
                        head = asyncio.run(service.propose_validation_cases(ctx=ctx, **common,
                            producer=build_source_validation_case_producer(), build_store=owners.build_store,
                            authority_resolver=owners.authority_resolver))
                    elif action == "inspect":
                        from agents.webapi.tender_document_review_routes import _runtime_semantic_inspection_service
                        head = asyncio.run(service.inspect_document(
                            ctx=ctx, **common, authority_resolver=owners.authority_resolver,
                            inspection_service=_runtime_semantic_inspection_service(ctx),
                        ))
                    else:
                        preparation = self._preparation(ctx, owners, stack)
                        if action == "form_runtime_context":
                            from agents.webapi.tender_document_review_routes import _runtime_semantic_inspection_service
                            head = service.form_runtime_context(
                                ctx=ctx, **common, preparation_service=preparation,
                                authority_resolver=owners.authority_resolver,
                                inspection_service=_runtime_semantic_inspection_service(ctx),
                            )
                        else:
                            method = service.select_policy if action == "select_policy" else service.select_runtime_document
                            head = method(ctx=ctx, **common, preparation_service=preparation)
            history = service.history(**scope, journey_id=head.journey_id)
            transition = next((item for item in history if item.transition_id == head.latest_transition_id), None)
            projection = service.project(head)
            suggested = {
                "SELECT_OR_UPLOAD_POLICY": ("select_policy",),
                "BUILD_CONTEXT": ("discover_intent", "build_context"),
                "RETRY_GRAPH_INTENT_DISCOVERY": ("discover_intent",),
                "CONTINUE_CONTEXT_BUILD": ("discover_intent", "build_context"),
                "REVIEW_MATERIAL_GRAPH_INTENT": ("read_intent", "review_intent", "build_context"),
                "CONTINUE_CERTIFICATION": ("propose_competency", "review_competency", "produce_normative_schema", "produce_normative_semantics", "propose_validation_cases", "review_validation_cases", "validate_source_semantics", "review_source_coverage", "prepare_authority_dependencies", "certify_context", "select_certification"),
                "COMPLETE_GOVERNED_RUNTIME_BINDING": ("bind_runtime",),
                "SELECT_RUNTIME_DOCUMENT": ("select_document",),
                "FORM_RUNTIME_CONTEXT": ("form_runtime_context",),
                "CONTINUE_INSPECTION": ("inspect",),
                "RETRY_INSPECTION": ("inspect",),
                "REVIEW_OR_INSPECT_NEXT_DOCUMENT": ("next_document",),
                "RESUME_JOURNEY": ("resume",),
            }.get(projection.next_action_code, ())
            lifecycle = ("pause", "cancel") if head.lifecycle_status == "ACTIVE" else ("cancel",) if head.lifecycle_status == "PAUSED" else ()
            build = next((item.pin for item in head.artifact_bindings if item.role == "context_build"), None)
            workbench_url = "/ontology-workbench2?" + urlencode({"journeyRef": head.journey_id,
                **({"buildRef": build.artifact_ref} if build is not None else {})})
            result_pin = next((item.pin for item in head.artifact_bindings if item.role == "inspection_result"), None)
            completed = next((item for item in reversed(history) if item.successor_revision <= head.revision
                and item.cause_code == "RUNTIME_INSPECTION_COMPLETED" and item.to_phase == "REVIEW_RESULTS"
                and item.operation_receipt is not None and item.operation_receipt.pin == result_pin), None)
            if completed is not None:
                workbench_url = _episode_url(head.journey_id, completed.transition_id, owners.session_ref)
            competency = None
            competency_pins = [item.pin for item in head.artifact_bindings if item.role == "competency_contract"]
            if competency_pins:
                from contracts.ontology_building import CompetencyContractV1
                if len(competency_pins) != 1:
                    raise ValueError("COMPETENCY_PIN_MISMATCH")
                competency = owners.build_store.get(**scope, artifact_ref=competency_pins[0].artifact_ref,
                    contract_type=CompetencyContractV1)
                if competency is None or competency.competency_contract_hash != competency_pins[0].artifact_hash:
                    raise ValueError("COMPETENCY_PIN_MISMATCH")
            normative = adequacy = None
            schema_pins = [item.pin for item in head.artifact_bindings if item.role == "normative_schema"]
            if schema_pins:
                from services.normative_schema_production_service import NormativeSchemaProductionService
                if len(schema_pins) != 1:
                    raise ValueError("NORMATIVE_SCHEMA_PIN_MISMATCH")
                normative, adequacy = NormativeSchemaProductionService.read_persisted(
                    store=owners.build_store, **scope, schema_pin=schema_pins[0])
            semantics = None
            semantic_pins = [item.pin for item in head.artifact_bindings if item.role == "normative_semantics"]
            if semantic_pins:
                from services.normative_semantic_production_service import NormativeSemanticProductionService
                if len(semantic_pins) != 1:
                    raise ValueError("NORMATIVE_SEMANTIC_CANDIDATE_PIN_MISMATCH")
                semantics = NormativeSemanticProductionService.read_persisted(
                    store=owners.build_store, **scope, candidate_pin=semantic_pins[0])
            validation_cases = None
            case_pins = [item.pin for item in head.artifact_bindings if item.role == "validation_case_proposal"]
            if case_pins:
                from services.source_validation_case_production_service import SourceValidationCaseProductionService
                if len(case_pins) != 1:
                    raise ValueError("VALIDATION_CASE_PROPOSAL_PIN_MISMATCH")
                _, validation_cases = SourceValidationCaseProductionService.read_persisted(
                    store=owners.build_store, **scope, proposal_pin=case_pins[0])
            validation_review, reviewed_validation, review_status = service.resolve_validation_review(
                head=head, build_store=owners.build_store)
            validation_execution, validation_run, validation_status = service.resolve_source_validation(
                head=head, build_store=owners.build_store)
            return {
                "status": "RESOLVED", "journey": projection.model_dump(mode="json"),
                "certification_competency": competency.model_dump(mode="json") if competency is not None else None,
                "normative_schema": normative.model_dump(mode="json") if normative is not None else None,
                "schema_adequacy": adequacy.model_dump(mode="json") if adequacy is not None else None,
                "normative_semantics": semantics.model_dump(mode="json") if semantics is not None else None,
                "validation_case_proposal": validation_cases.model_dump(mode="json") if validation_cases is not None else None,
                "validation_case_review": validation_review.model_dump(mode="json") if validation_review is not None else None,
                "validation_review_status": review_status,
                "source_validation_status": validation_status,
                "source_validation_execution": validation_execution.model_dump(mode="json", exclude={"code_units"}) if validation_execution is not None else None,
                "quality_evaluation": validation_run.model_dump(mode="json") if validation_run is not None else None,
                "validation_contract": reviewed_validation.model_dump(mode="json") if reviewed_validation is not None else None,
                "workbench_url": workbench_url,
                "inspection_episodes": [{"transition_id": item.transition_id,
                    "revision": item.successor_revision,
                    "workbench_url": _episode_url(head.journey_id, item.transition_id, owners.session_ref),
                    "result_pin": item.operation_receipt.pin.model_dump(mode="json")}
                    for item in history if item.successor_revision <= head.revision
                    and item.cause_code == "RUNTIME_INSPECTION_COMPLETED"
                    and item.to_phase == "REVIEW_RESULTS" and item.operation_receipt is not None
                    and item.operation_receipt.role == "inspection_result"],
                "latest_transition": transition.model_dump(mode="json") if transition else None,
                "action_schemas": _schemas((*suggested, *lifecycle, *_ARTIFACT_ACTIONS, "read_intent", "read_episode", "create")),
            }

    @staticmethod
    def _artifacts(owners):
        import os
        from services.workspace_artifact_service import WorkspaceArtifactService
        from storage.workbench_projection_cursor_store import WorkbenchProjectionCursorStore

        max_bytes = int(os.environ.get("SEMANTIER_WORKSPACE_ARTIFACT_MAX_BYTES", "20971520"))
        if max_bytes <= 0:
            raise ValueError("INVALID_BYTE_LENGTH")
        cursors = WorkbenchProjectionCursorStore(build_store=owners.build_store,
            authority_resolver=owners.authority_resolver, session_resolver=owners.session_resolver,
            session_ref=owners.session_ref)
        return WorkspaceArtifactService(build_store=owners.build_store, authority_resolver=owners.authority_resolver,
                                        max_bytes=max_bytes, cursor_store=cursors)

    @staticmethod
    def _preparation(ctx, owners, stack):
        from services.workspace_artifact_preparation_service import WorkspaceArtifactPreparationService, KnowledgeDesignSourcePreparationAdapter
        from storage.knowledge_extraction_store import KnowledgeExtractionStore

        extraction = KnowledgeExtractionStore(owners.source_registry._db_path)
        stack.callback(extraction._conn.close)
        artifacts = TenderJourneyToolAPI._artifacts(owners)
        return WorkspaceArtifactPreparationService(artifact_service=artifacts, design_adapter=KnowledgeDesignSourcePreparationAdapter(
            artifact_service=artifacts, knowledge_store=owners.source_registry, extraction_store=extraction,
        ))

    @staticmethod
    def _artifact_action(ctx, owners, stack, action, body):
        import io
        from pathlib import PurePosixPath
        from contracts.ontology_workbench_context import UploadWorkspaceArtifactCommandV1, PrepareWorkspaceArtifactCommandV1
        from agents.webapi.ontology_workbench_context_routes import _temporal_upload_boundary
        from agents.workspace_uploads import WorkspaceUploadError, list_document_workspace_uploads, read_document_workspace_upload

        artifacts = TenderJourneyToolAPI._artifacts(owners)
        from services.tender_graph_preflight import tender_graph_preflight, require_tender_graph_preflight
        if action == "preflight":
            return {"preflight": tender_graph_preflight(ctx=ctx, artifacts=artifacts)}
        if action == "import_upload" or (action == "prepare_artifact" and body.purpose == "DESIGN_SOURCE"):
            require_tender_graph_preflight(ctx=ctx, artifacts=artifacts)
        if action == "list_artifacts":
            return {"artifacts": artifacts.list(ctx=ctx).model_dump(mode="json")}
        if action == "prepare_artifact":
            preparation = TenderJourneyToolAPI._preparation(ctx, owners, stack)
            command = PrepareWorkspaceArtifactCommandV1(
                schema_version="prepare_workspace_artifact_command.v1", ingress="CHAT",
                idempotency_key=body.request_id, **body.model_dump(),
            )
            return {"preparation": preparation.prepare(ctx=ctx, command=command).model_dump(mode="json")}
        boundary, _ = _temporal_upload_boundary(ctx=ctx, session_ref=owners.session_ref)
        if action == "list_uploads":
            result = list_document_workspace_uploads(boundary=boundary, max_bytes=artifacts._max_bytes,
                                                     offset=body.offset, limit=body.limit)
            for upload in result["uploads"]:
                upload["upload_ref"] = quote(upload["upload_ref"], safe="/")
            return result
        try:
            upload_ref = unquote(body.upload_ref, encoding="utf-8", errors="strict")
            if quote(upload_ref, safe="/") != body.upload_ref:
                raise ValueError("Noncanonical upload reference")
        except ValueError as exc:
            raise WorkspaceUploadError("Upload unavailable", code="NOT_FOUND") from exc
        content = read_document_workspace_upload(boundary=boundary, upload_ref=upload_ref,
                                                 upload_hash=body.upload_hash, max_bytes=artifacts._max_bytes)
        path = PurePosixPath(upload_ref)
        media_type = {".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                      ".pdf": "application/pdf", ".md": "text/markdown", ".txt": "text/plain"}[path.suffix.lower()]
        command = UploadWorkspaceArtifactCommandV1(
            schema_version="upload_workspace_artifact_command.v1", request_id=body.request_id,
            idempotency_key=body.request_id, ingress="CHAT", expected_collection_revision=body.expected_collection_revision,
            original_name=path.name, declared_media_type=media_type, byte_length=len(content), byte_hash=body.upload_hash,
        )
        return {"artifact_receipt": artifacts.upload(ctx=ctx, command=command, byte_stream=io.BytesIO(content)).model_dump(mode="json")}


def authenticated_tool_api() -> TenderJourneyToolAPI:
    from gateway.session_context import get_session_env

    context = resolve_authenticated_tool_context()
    if context is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    return TenderJourneyToolAPI(context=context, session_ref=get_session_env("HERMES_SESSION_ID", ""))
