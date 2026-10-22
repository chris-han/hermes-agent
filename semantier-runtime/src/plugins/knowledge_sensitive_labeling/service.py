from __future__ import annotations

from dataclasses import asdict
from typing import Any

from contracts.knowledge_classification import (
    CanonicalClassificationInputV1,
    CanonicalEntityV1,
    CanonicalFieldValueV1,
    CanonicalRecordV1,
    CanonicalRelationV1,
    CanonicalSegmentV1,
    ClassificationRunCreateDTO,
    RuntimeContextDTO,
    canonical_content_hash,
)
from services.knowledge_classification_evidence_service import (
    InMemoryKnowledgeClassificationEvidenceService,
)
from agents.governed_tool_context import resolve_authenticated_tool_context

from . import classifier, schemas
from .derivative_output import derivative_manifest
from .report import build_redaction_safe_report

_DEFAULT_SERVICE = InMemoryKnowledgeClassificationEvidenceService()


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


def classify_knowledge_input(*, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    ctx = kwargs.get("ctx") or kwargs.get("request_context") or resolve_authenticated_tool_context()
    if ctx is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    context = _runtime_context(args, ctx)
    input_payload = args.get("canonical_input") or args.get("canonicalInput")
    if isinstance(input_payload, dict):
        input_payload = _canonical_input_from_dict(input_payload)
    if not isinstance(input_payload, CanonicalClassificationInputV1):
        raise ValueError("canonical_input must be CanonicalClassificationInputV1")
    if input_payload.organization_id != context.organization_id:
        raise ValueError("CANONICAL_INPUT_SCOPE_MISMATCH")
    if (input_payload.workspace_id or None) != (context.workspace_id or None):
        raise ValueError("CANONICAL_INPUT_SCOPE_MISMATCH")
    active_rules = args.get("active_rules") or args.get("activeRules") or []
    if not isinstance(active_rules, list):
        raise ValueError("active_rules must be a list")
    snapshot_id = str(args.get("activation_set_snapshot_id") or args.get("activationSetSnapshotId") or "")
    snapshot_hash = str(args.get("resolved_activation_set_hash") or args.get("resolvedActivationSetHash") or "")
    rule_set_hash = str(
        args.get("rule_version_set_hash")
        or args.get("ruleVersionSetHash")
        or canonical_content_hash(
            [
                {
                    "rule_version_id": rule.get("rule_version_id"),
                    "content_hash": rule.get("content_hash"),
                }
                for rule in active_rules
                if isinstance(rule, dict)
            ]
        )
    )
    if not snapshot_id or not snapshot_hash:
        raise ValueError("AUTHORITY_RESOLVER_SNAPSHOT_REQUIRED")
    evidence_service = kwargs.get("classification_evidence_service") or _DEFAULT_SERVICE
    run = evidence_service.create_classification_run_idempotent(
        ClassificationRunCreateDTO(
            input_id=input_payload.input_id,
            source_hash=input_payload.source_hash,
            activation_set_snapshot_id=snapshot_id,
            resolved_activation_set_hash=snapshot_hash,
            rule_version_set_hash=rule_set_hash,
            classifier_profile_version=str(args.get("classifier_profile_version") or schemas.CLASSIFIER_PROFILE_VERSION),
            ontology_version=str(args.get("ontology_version") or schemas.ONTOLOGY_VERSION),
            idempotency_key=str(
                args.get("idempotency_key")
                or args.get("idempotencyKey")
                or canonical_content_hash(
                    {
                        "input": input_payload.content_hash,
                        "snapshot": snapshot_hash,
                        "rule_version_set_hash": rule_set_hash,
                    }
                )
            ),
            metadata={"parser_profile_version": input_payload.parser_profile_version},
        ),
        context,
    )
    result = classifier.classify_canonical_input(
        canonical_input=input_payload,
        active_rules=[item for item in active_rules if isinstance(item, dict)],
        resolver_evidence_ref=snapshot_id,
    )
    evidence_service.persist_occurrence_findings(run.run_id, result.occurrences, context)
    evidence_service.persist_field_findings(run.run_id, result.fields, context)
    evidence_service.persist_entity_findings(run.run_id, [], context)
    evidence_service.persist_relation_findings(run.run_id, [], context)
    evidence_service.persist_runtime_conflicts(run.run_id, result.runtime_conflicts, context)
    finalized = evidence_service.finalize_classification_run(run.run_id, "finalized", context)
    evidence = evidence_service.get_classification_evidence(finalized.run_id, context)
    return {
        "status": "ok",
        "classification": asdict(finalized),
        "evidence": asdict(evidence),
    }


def get_classification_evidence(*, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    ctx = kwargs.get("ctx") or kwargs.get("request_context") or resolve_authenticated_tool_context()
    if ctx is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    context = _runtime_context(args, ctx)
    run_id = str(args.get("run_id") or args.get("runId") or "").strip()
    if not run_id:
        raise ValueError("run_id is required")
    evidence_service = kwargs.get("classification_evidence_service") or _DEFAULT_SERVICE
    return {
        "status": "ok",
        "evidence": asdict(evidence_service.get_classification_evidence(run_id, context)),
    }


def verify_knowledge_labeling_replay(*, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    result = get_classification_evidence(args=args, **kwargs)
    evidence = result["evidence"]
    return {
        "status": "ok",
        "verified": bool(evidence["summary"]["evidence_set_hash"]),
        "evidence_set_hash": evidence["summary"]["evidence_set_hash"],
    }


def generate_knowledge_labeling_report(*, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    ctx = kwargs.get("ctx") or kwargs.get("request_context") or resolve_authenticated_tool_context()
    if ctx is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    context = _runtime_context(args, ctx)
    run_id = str(args.get("run_id") or args.get("runId") or "").strip()
    if not run_id:
        raise ValueError("run_id is required")
    evidence_service = kwargs.get("classification_evidence_service") or _DEFAULT_SERVICE
    evidence = evidence_service.get_classification_evidence(run_id, context)
    return {
        "status": "ok",
        "report": build_redaction_safe_report(evidence),
    }


def generate_knowledge_labeling_derivative(*, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    ctx = kwargs.get("ctx") or kwargs.get("request_context") or resolve_authenticated_tool_context()
    if ctx is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    context = _runtime_context(args, ctx)
    run_id = str(args.get("run_id") or args.get("runId") or "").strip()
    if not run_id:
        raise ValueError("run_id is required")
    evidence_service = kwargs.get("classification_evidence_service") or _DEFAULT_SERVICE
    evidence = evidence_service.get_classification_evidence(run_id, context)
    return {
        "status": "ok",
        "derivative": derivative_manifest(evidence),
    }


def _canonical_input_from_dict(payload: dict[str, Any]) -> CanonicalClassificationInputV1:
    records = tuple(
        CanonicalRecordV1(
            **{
                **record,
                "field_values": tuple(
                    CanonicalFieldValueV1(**field)
                    for field in record.get("field_values", ())
                ),
            }
        )
        for record in payload.get("records", ())
    )
    segments = tuple(
        CanonicalSegmentV1(**segment)
        for segment in payload.get("segments", ())
    )
    entities = tuple(
        CanonicalEntityV1(**entity)
        for entity in payload.get("entities", ())
    )
    relations = tuple(
        CanonicalRelationV1(**relation)
        for relation in payload.get("relations", ())
    )
    return CanonicalClassificationInputV1(
        **{
            **payload,
            "records": records,
            "segments": segments,
            "entities": entities,
            "relations": relations,
        }
    )
