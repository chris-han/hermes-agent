from __future__ import annotations

from dataclasses import asdict
from typing import Any

from contracts.knowledge_classification import (
    ParseRunCreateDTO,
    RuntimeContextDTO,
    canonical_content_hash,
)
from services.knowledge_parsing_evidence_service import (
    InMemoryKnowledgeParsingEvidenceService,
)
from agents.governed_tool_context import resolve_authenticated_tool_context

from .canonicalizer import canonicalize_structured_records, canonicalize_unstructured_document
from .parser_registry import require_supported_profile

_DEFAULT_SERVICE = InMemoryKnowledgeParsingEvidenceService()


def _runtime_context(args: dict[str, Any], ctx: Any) -> RuntimeContextDTO:
    org_id = str(getattr(ctx, "organization_id", None) or args.get("organization_id") or "").strip()
    if not org_id:
        raise ValueError("AUTHENTICATED_ORGANIZATION_CONTEXT_REQUIRED")
    return RuntimeContextDTO(
        organization_id=org_id,
        workspace_id=getattr(ctx, "workspace_id", None) or args.get("workspace_id"),
        actor_id=str(getattr(ctx, "user_id", None) or args.get("actor_id") or ""),
        actor_role=str(getattr(ctx, "member_role", None) or args.get("actor_role") or ""),
    )


def parse_knowledge_source(*, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    ctx = kwargs.get("ctx") or kwargs.get("request_context") or resolve_authenticated_tool_context()
    if ctx is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    context = _runtime_context(args, ctx)
    source_kind = str(args.get("source_kind") or args.get("sourceKind") or "").strip()
    profile = require_supported_profile(str(args.get("parser_profile_version") or args.get("parserProfileVersion") or "").strip())
    source_ref = str(args.get("source_ref") or args.get("sourceRef") or source_kind or "source").strip()
    source_hash = str(args.get("source_hash") or args.get("sourceHash") or "").strip()
    normalized_artifact_ref = args.get("normalized_artifact_ref") or args.get("normalizedArtifactRef")
    normalized_artifact_hash = args.get("normalized_artifact_hash") or args.get("normalizedArtifactHash")

    if source_kind == "structured_records":
        records = args.get("records")
        if not isinstance(records, list):
            raise ValueError("STRUCTURED_RECORDS_REQUIRED")
        source_hash = source_hash or canonical_content_hash(records)
        canonical_input = canonicalize_structured_records(
            records=[item for item in records if isinstance(item, dict)],
            context=context,
            source_ref=source_ref,
            source_hash=source_hash,
            parser_profile_version=profile,
            normalized_artifact_ref=normalized_artifact_ref,
            normalized_artifact_hash=normalized_artifact_hash,
        )
    elif source_kind in {"docx", "pdf", "html", "email", "text", "controlled_document"}:
        canonical_document = args.get("canonical_document") or args.get("canonicalDocument")
        if canonical_document is not None and not isinstance(canonical_document, dict):
            raise ValueError("CANONICAL_DOCUMENT_MUST_BE_OBJECT")
        document_text = str(args.get("document_text") or args.get("documentText") or "")
        if not document_text and canonical_document is None:
            raise ValueError("NORMALIZED_DOCUMENT_PAYLOAD_REQUIRED")
        source_hash = source_hash or canonical_content_hash(canonical_document if canonical_document is not None else document_text)
        canonical_input = canonicalize_unstructured_document(
            document_text=document_text,
            canonical_document=canonical_document,
            context=context,
            source_kind=source_kind,
            source_ref=source_ref,
            source_hash=source_hash,
            parser_profile_version=profile,
            normalized_artifact_ref=normalized_artifact_ref,
            normalized_artifact_hash=normalized_artifact_hash,
        )
    else:
        raise ValueError("SOURCE_KIND_UNSUPPORTED")

    evidence_service = kwargs.get("parsing_evidence_service") or _DEFAULT_SERVICE
    run = evidence_service.create_parse_run_idempotent(
        ParseRunCreateDTO(
            source_kind=source_kind,
            source_ref=source_ref,
            source_hash=source_hash,
            parser_profile_version=profile,
            normalized_artifact_ref=normalized_artifact_ref,
            normalized_artifact_hash=normalized_artifact_hash,
            idempotency_key=str(
                args.get("idempotency_key")
                or args.get("idempotencyKey")
                or canonical_content_hash({"context": asdict(context), "input": canonical_input.content_hash})
            ),
            metadata={"input_id": canonical_input.input_id},
        ),
        context,
    )
    evidence_service.persist_canonical_records(run.parse_run_id, list(canonical_input.records), context)
    evidence_service.persist_canonical_segments(run.parse_run_id, list(canonical_input.segments), context)
    evidence_service.persist_entities_and_relations(
        run.parse_run_id,
        list(canonical_input.entities),
        list(canonical_input.relations),
        context,
    )
    finalized = evidence_service.finalize_parse_run(run.parse_run_id, canonical_input, context)
    return {
        "status": "ok",
        "parse_run": asdict(finalized),
        "canonical_input": asdict(canonical_input),
    }


def get_canonical_classification_input(*, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    ctx = kwargs.get("ctx") or kwargs.get("request_context") or resolve_authenticated_tool_context()
    if ctx is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    context = _runtime_context(args, ctx)
    input_id = str(args.get("input_id") or args.get("inputId") or "").strip()
    if not input_id:
        raise ValueError("input_id is required")
    evidence_service = kwargs.get("parsing_evidence_service") or _DEFAULT_SERVICE
    return {
        "status": "ok",
        "canonical_input": asdict(evidence_service.get_canonical_classification_input(input_id, context)),
    }


def get_knowledge_parse_run(*, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    ctx = kwargs.get("ctx") or kwargs.get("request_context") or resolve_authenticated_tool_context()
    if ctx is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    context = _runtime_context(args, ctx)
    run_id = str(args.get("run_id") or args.get("runId") or "").strip()
    if not run_id:
        raise ValueError("run_id is required")
    evidence_service = kwargs.get("parsing_evidence_service") or _DEFAULT_SERVICE
    return {
        "status": "ok",
        "parse_evidence": asdict(evidence_service.get_parse_evidence(run_id, context)),
    }


def verify_knowledge_parse_replay(*, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    ctx = kwargs.get("ctx") or kwargs.get("request_context") or resolve_authenticated_tool_context()
    if ctx is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    context = _runtime_context(args, ctx)
    input_id = str(args.get("input_id") or args.get("inputId") or "").strip()
    if not input_id:
        raise ValueError("input_id is required")
    evidence_service = kwargs.get("parsing_evidence_service") or _DEFAULT_SERVICE
    return {
        "status": "ok",
        "verified": evidence_service.verify_parse_artifact_hashes(input_id, context),
    }
