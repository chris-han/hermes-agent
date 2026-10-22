from __future__ import annotations

from typing import Any

from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.runtime_feedback_store import RuntimeFeedbackStore
from storage.tender_detection_run_store import TenderDetectionRunStore
from services.contextgraph_observed_expression_projection import (
    ContextGraphObservedExpressionProjection,
)

_FEEDBACK_TO_DELTA = {
    "false_positive": "not_same_as",
    "false_negative": "allowed_context_for",
    "ambiguity": "split_cluster",
    "conflict": "prohibited_context_for",
    "remediation_rejected": "new_control_candidate",
    "user_edited_remediation": "new_control_candidate",
    "missing_control": "new_control_candidate",
    "weak_explanation": "allowed_context_for",
}


def convert_tender_feedback_to_candidate_delta(
    *,
    organization_id: str,
    workspace_id: str | None,
    run_id: str,
    finding_id: str | None,
    feedback_type: str,
    user_disposition: dict[str, Any],
    escalation_outcome: str | None,
    reviewer_notes: str | None,
    edited_remediation: str | None,
    created_by: str,
    source_span_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tender_store = TenderDetectionRunStore()
    run = tender_store.get_run(run_id)
    if run is None or run["organization_id"] != organization_id or run.get("workspace_id") != workspace_id:
        raise ValueError("tender detection run not found")
    finding = _find_finding(run, finding_id)
    source_span = _source_span(run, finding, source_span_override)
    feedback_event = RuntimeFeedbackStore().create_feedback_event(
        organization_id=organization_id,
        workspace_id=workspace_id,
        source_kind="tender_document_review",
        source_run_id=run_id,
        source_finding_id=finding_id,
        feedback_type=feedback_type,
        user_disposition={
            **user_disposition,
            "edited_remediation": edited_remediation,
        },
        escalation_outcome=escalation_outcome or "not_escalated",
        feedback_evidence={
            "reviewer_notes": reviewer_notes,
            "resolved_activation_set_hash": run["resolved_activation_set_hash"],
            "activation_set_snapshot_id": run["activation_set_snapshot_id"],
        },
        affected_term=(finding or {}).get("matched_text"),
        affected_control_ref=(finding or {}).get("triggered_rule_version_id"),
        affected_relation_ref=(finding or {}).get("resolver_evidence_ref"),
        source_span=source_span,
        created_by=created_by,
    )
    delta_kind = _FEEDBACK_TO_DELTA[feedback_type]
    discovery = KnowledgeBuilderStore().create_discovery_run(
        organization_id=organization_id,
        workspace_id=workspace_id,
        source_kind="text",
        source_ref=f"runtime-feedback:{feedback_event['feedback_event_id']}",
        source_text=_delta_source_text(feedback_type, source_span, edited_remediation),
        source_metadata={
            "source_kind": "runtime_feedback",
            "feedback_event_id": feedback_event["feedback_event_id"],
            "source_run_id": run_id,
            "source_finding_id": finding_id,
            "delta_kind": delta_kind,
            "runtime_authority_mutated": False,
        },
        model_provenance={
            "model": "deterministic-runtime-feedback-converter",
            "version": "runtime_feedback_to_candidate.v1",
        },
        procedure_provenance={
            "procedure": "runtime_exception_to_candidate_delta",
            "version": "runtime_feedback_to_candidate.v1",
        },
    )
    discovery_source = KnowledgeBuilderStore().get_source(discovery["source_id"]) or {}
    candidate_delta = RuntimeFeedbackStore().create_candidate_delta(
        feedback_event_id=feedback_event["feedback_event_id"],
        discovery_run_id=discovery["discovery_run_id"],
        delta_kind=delta_kind,
        candidate_graph_ref={
            "discovery_run_id": discovery["discovery_run_id"],
            "source_id": discovery["source_id"],
            "source_ref": discovery_source.get("source_ref"),
            "source_hash": discovery_source.get("source_hash"),
            "graph_materialized": False,
        },
        evaluation_routing={
            "route_to": "knowledge_builder_evaluation_lab",
            "required_before_promotion": True,
            "feedback_type": feedback_type,
        },
    )
    return {
        "feedbackEvent": feedback_event,
        "candidateDelta": candidate_delta,
        "discoveryRun": discovery,
    }


def project_accepted_tender_feedback_to_observed_expression(
    *,
    run_id: str,
    finding_id: str,
    human_learning_event_id: str,
    reference_concept_version_ref: str,
    accepted_graph_release_id: str,
    semantic_relation_decision_hash: str,
    semantic_relation: str | None = None,
) -> dict[str, Any]:
    return ContextGraphObservedExpressionProjection().project_accepted_finding(
        run_id=run_id,
        finding_id=finding_id,
        human_learning_event_id=human_learning_event_id,
        reference_concept_version_ref=reference_concept_version_ref,
        accepted_graph_release_id=accepted_graph_release_id,
        semantic_relation_decision_hash=semantic_relation_decision_hash,
        semantic_relation=semantic_relation,
    )


def _find_finding(run: dict[str, Any], finding_id: str | None) -> dict[str, Any] | None:
    if not finding_id:
        return None
    for finding in run["findings"]:
        if finding["finding_id"] == finding_id:
            return finding
    raise ValueError("finding not found for tender detection run")


def _source_span(
    run: dict[str, Any],
    finding: dict[str, Any] | None,
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if finding is None:
        text = str(run.get("document_text") or "")
        if not override:
            raise ValueError("sourceSpan is required for run-level feedback")
        span_text = str(override.get("text") or "")
        start = override.get("startOffset", override.get("start_offset"))
        end = override.get("endOffset", override.get("end_offset"))
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
            raise ValueError("sourceSpan offsets are out of range")
        if text[start:end] != span_text:
            raise ValueError("sourceSpan text does not match document_text")
        return {
            "text": span_text,
            "locator": "document",
            "start_offset": start,
            "end_offset": end,
            "source_document_hash": run["source_document_hash"],
        }
    return {
        "text": finding["matched_text"],
        "locator": finding["clause_anchor"],
        "source_anchor_refs": finding["source_anchor_refs"],
        "source_document_hash": run["source_document_hash"],
    }


def _delta_source_text(
    feedback_type: str,
    source_span: dict[str, Any],
    edited_remediation: str | None,
) -> str:
    parts = [
        f"Runtime feedback type: {feedback_type}.",
        f"Source span: {source_span.get('text') or ''}.",
    ]
    if edited_remediation:
        parts.append(f"Edited remediation: {edited_remediation}.")
    return " ".join(parts)
