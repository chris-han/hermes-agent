from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any
from types import SimpleNamespace
from uuid import uuid4

from contracts.tender_sensitive_term_detection_modeling_core import (
    CandidateFindingDraftDTO,
    ComposedFramePatternV01,
    FramePatternV01,
)
from contracts.knowledge_classification import (
    CanonicalClassificationInputV1,
    CanonicalSegmentV1,
    canonical_content_hash,
)
from plugins.knowledge_sensitive_labeling.classifier import classify_canonical_input
from plugins.tender_sensitive_term_detection_modeling.semantic_assertions import (
    AssertionState,
    classify_semantic_assertion,
)
from plugins.tender_sensitive_term_detection_modeling.frame_semantics import (
    accepted_composed_frame_patterns_from_rules,
    accepted_frame_patterns_from_rules,
    extract_constraint_frame,
    extract_composed_constraint_frames,
    match_composed_frames,
    match_constraint_frame,
)
from services.tender_finding_decision_service import TenderFindingDecisionService


@dataclass(frozen=True)
class TenderDetectionDomainResult:
    findings: list[dict[str, Any]]
    runtime_conflicts: list[dict[str, Any]]


def detect_tender_occurrences(
    *,
    document_text: str,
    canonical_document: dict[str, Any] | None,
    active_rules: list[dict[str, Any]],
    resolver_evidence_ref: str,
    precedent_observed_expressions: list[dict[str, Any]] | None = None,
    precedent_model_ref: str | None = None,
    accepted_frame_patterns: tuple[FramePatternV01, ...] = (),
    accepted_composed_frame_patterns: tuple[ComposedFramePatternV01, ...] = (),
) -> TenderDetectionDomainResult:
    canonical_input = _canonical_input_for_tender(
        document_text=document_text,
        canonical_document=canonical_document,
    )
    governed_frame_patterns = accepted_frame_patterns_from_rules(active_rules)
    governed_composed_patterns = accepted_composed_frame_patterns_from_rules(
        active_rules
    )
    accepted_frame_patterns = (*accepted_frame_patterns, *governed_frame_patterns)
    accepted_composed_frame_patterns = (
        *accepted_composed_frame_patterns,
        *governed_composed_patterns,
    )
    generic_result = classify_canonical_input(
        canonical_input=canonical_input,
        active_rules=active_rules,
        resolver_evidence_ref=resolver_evidence_ref,
    )
    occurrences = _with_literal_fallbacks(
        occurrences=list(generic_result.occurrences),
        canonical_input=canonical_input,
        active_rules=active_rules,
    )
    occurrences = _with_precedent_fallbacks(
        occurrences=occurrences,
        canonical_input=canonical_input,
        active_rules=active_rules,
        precedent_observed_expressions=precedent_observed_expressions or [],
        precedent_model_ref=precedent_model_ref,
    )
    findings = _tender_findings_from_generic_occurrences(
        generic_occurrences=occurrences,
        active_rules=active_rules,
        resolver_evidence_ref=resolver_evidence_ref,
        canonical_document=canonical_document,
        document_text=document_text,
    )
    findings.extend(
        _frame_findings(
            canonical_input=canonical_input,
            accepted_frame_patterns=accepted_frame_patterns,
            resolver_evidence_ref=resolver_evidence_ref,
        )
    )
    findings.extend(
        _composed_frame_findings(
            canonical_input=canonical_input,
            accepted_patterns=accepted_composed_frame_patterns,
            resolver_evidence_ref=resolver_evidence_ref,
        )
    )
    runtime_conflicts = _classify_runtime_conflicts(findings)
    if runtime_conflicts:
        _mark_conflicted_findings(findings, runtime_conflicts)
    return TenderDetectionDomainResult(
        findings=findings,
        runtime_conflicts=runtime_conflicts,
    )


def _frame_findings(
    *,
    canonical_input: CanonicalClassificationInputV1,
    accepted_frame_patterns: tuple[FramePatternV01, ...],
    resolver_evidence_ref: str,
) -> list[dict[str, Any]]:
    """Project successful accepted-frame matches into the existing finding path."""

    findings: list[dict[str, Any]] = []
    decision_service = TenderFindingDecisionService()
    for pattern in accepted_frame_patterns:
        if not isinstance(pattern, FramePatternV01) or pattern.lifecycle_state != "accepted":
            continue
        for segment in canonical_input.segments:
            observed = extract_constraint_frame(
                segment.text,
                evidence_ref=segment.source_anchor_ref,
            )
            if observed is None:
                continue
            frame_match = match_constraint_frame(observed, pattern)
            decision = decision_service.decide_frame_match(
                frame_match=frame_match,
                observed_frame=observed,
            )
            if not decision.structural_match:
                continue
            derivation = {
                "derivation_shape": frame_match.derivation_shape,
                "observed_frame": asdict(observed),
                "canonical_frame": asdict(pattern.frame),
                "matched_pattern_ref": pattern.pattern_id,
                "slot_diagnostics": [
                    asdict(item) for item in frame_match.slot_diagnostics
                ],
                "mechanism_path": [],
            }
            finding_id = f"tdf_{uuid4().hex}"
            findings.append(
                {
                    "schema_version": "contextgraph_candidate_finding.v2",
                    "finding_id": finding_id,
                    "clause_id": "document",
                    "clause_anchor": segment.source_anchor_ref,
                    "issue_type": "structural_constraint_match",
                    "matched_text": segment.text,
                    "normalized_match": segment.text.strip().lower(),
                    "triggered_rule_version_id": pattern.pattern_id,
                    "triggered_rule_version_refs": [
                        {
                            "rule_version_id": pattern.pattern_id,
                            "content_hash": pattern.pattern_hash,
                        }
                    ],
                    "winning_rule_version_ref": pattern.pattern_id,
                    "conflicting_rule_version_refs": [],
                    "conflict_resolution_ref": None,
                    "source_anchor_refs": list(pattern.source_evidence_refs),
                    "judgment_basis": decision.rationale,
                    "severity": "medium",
                    "confidence": 1.0,
                    "suggested_replacement": None,
                    "escalation_flag": False,
                    "resolver_evidence_ref": resolver_evidence_ref,
                    "target_anchor_ref": segment.source_anchor_ref,
                    "target_element_kind": "text_segment",
                    "target_element_id": segment.segment_id,
                    "start_offset": 0,
                    "end_offset": len(segment.text),
                    "occurrence_index": 0,
                    "target_element_hash": segment.local_text_hash,
                    "matched_sensitive_concept_id": pattern.pattern_id,
                    "observed_expression": segment.text,
                    "detection_method": "semantic",
                    "semantic_detection_method": "frame",
                    "semantic_relation": frame_match.semantic_relation,
                    "evidence_text": segment.text,
                    "precedent_refs": [],
                    "decision_status": decision.status,
                    "grounding_status": "ungrounded",
                    "semantic_profile_id": frame_match.semantic_profile_id,
                    "source_graph_version": pattern.graph_release_ref,
                    "derivation_shape": frame_match.derivation_shape,
                    "frame_derivation": derivation,
                    "decision_context": asdict(decision),
                }
            )
    return findings


def _composed_frame_findings(
    *,
    canonical_input: CanonicalClassificationInputV1,
    accepted_patterns: tuple[ComposedFramePatternV01, ...],
    resolver_evidence_ref: str,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    decision_service = TenderFindingDecisionService()
    for pattern in accepted_patterns:
        if (
            not isinstance(pattern, ComposedFramePatternV01)
            or pattern.lifecycle_state != "accepted"
        ):
            continue
        for segment in canonical_input.segments:
            observed = extract_composed_constraint_frames(
                segment.text, evidence_ref=segment.source_anchor_ref
            )
            if len(observed) < 2:
                continue
            frame_match = match_composed_frames(observed, pattern)
            decision = decision_service.decide_frame_match(
                frame_match=frame_match,
                observed_frame=observed[0],
            )
            if not decision.structural_match:
                continue
            finding_id = f"tdf_{uuid4().hex}"
            findings.append(
                {
                    "schema_version": "contextgraph_candidate_finding.v2",
                    "finding_id": finding_id,
                    "clause_id": "document",
                    "clause_anchor": segment.source_anchor_ref,
                    "issue_type": "composed_structural_constraint_match",
                    "matched_text": segment.text,
                    "normalized_match": segment.text.strip().lower(),
                    "triggered_rule_version_id": pattern.pattern_id,
                    "triggered_rule_version_refs": [
                        {
                            "rule_version_id": pattern.pattern_id,
                            "content_hash": pattern.pattern_hash,
                        }
                    ],
                    "winning_rule_version_ref": pattern.pattern_id,
                    "conflicting_rule_version_refs": [],
                    "conflict_resolution_ref": None,
                    "source_anchor_refs": list(pattern.source_evidence_refs),
                    "judgment_basis": decision.rationale,
                    "severity": "medium",
                    "confidence": 1.0,
                    "suggested_replacement": None,
                    "escalation_flag": False,
                    "resolver_evidence_ref": resolver_evidence_ref,
                    "target_anchor_ref": segment.source_anchor_ref,
                    "target_element_kind": "text_segment",
                    "target_element_id": segment.segment_id,
                    "start_offset": 0,
                    "end_offset": len(segment.text),
                    "occurrence_index": 0,
                    "target_element_hash": segment.local_text_hash,
                    "matched_sensitive_concept_id": pattern.pattern_id,
                    "observed_expression": segment.text,
                    "detection_method": "semantic",
                    "semantic_detection_method": "frame",
                    "semantic_relation": frame_match.semantic_relation,
                    "evidence_text": segment.text,
                    "precedent_refs": [],
                    "decision_status": decision.status,
                    "grounding_status": "ungrounded",
                    "semantic_profile_id": frame_match.semantic_profile_id,
                    "source_graph_version": pattern.graph_release_ref,
                    "derivation_shape": frame_match.derivation_shape,
                    "frame_derivation": {
                        "derivation_shape": frame_match.derivation_shape,
                        "observed_frames": [asdict(item) for item in observed],
                        "canonical_frames": [
                            asdict(item.frame) for item in pattern.constituents
                        ],
                        "matched_pattern_ref": pattern.pattern_id,
                        "constituent_matches": [
                            asdict(item) for item in frame_match.constituent_matches
                        ],
                        "mechanism_path": [],
                    },
                    "decision_context": asdict(decision),
                }
            )
    return findings


def _canonical_input_for_tender(
    *,
    document_text: str,
    canonical_document: dict[str, Any] | None,
) -> CanonicalClassificationInputV1:
    segments: list[CanonicalSegmentV1] = []
    if canonical_document is not None:
        if canonical_document.get("schema_version") != "canonical_document.v2":
            raise ValueError("canonical_document.v2 is required for fileRef occurrence labeling")
        for index, element in enumerate(canonical_document.get("elements") or [], start=1):
            text = str(element.get("text") or "")
            if not text:
                continue
            segments.append(
                CanonicalSegmentV1(
                    segment_id=f"tender_segment:{index}",
                    segment_kind=str(element.get("kind") or "document_element"),
                    text=text,
                    normalized_text=text.strip().lower(),
                    source_anchor_ref=str(element.get("anchor_ref") or element.get("id") or f"document:element:{index}"),
                    parent_segment_id=None,
                    order_index=index,
                    local_text_hash=str(element.get("local_text_hash") or canonical_content_hash(text)),
                    layout_metadata={
                        key: value
                        for key, value in element.items()
                        if key not in {"text", "anchor_ref", "local_text_hash"}
                    },
                )
            )
    else:
        segments.append(
            CanonicalSegmentV1(
                segment_id="tender_segment:1",
                segment_kind="plain_text",
                text=document_text,
                normalized_text=document_text.strip().lower(),
                source_anchor_ref="text:document",
                parent_segment_id=None,
                order_index=1,
                local_text_hash=canonical_content_hash(document_text),
                layout_metadata={"legacy_anchor_ref": "text:document"},
            )
        )
    source_payload = canonical_document if canonical_document is not None else document_text
    return CanonicalClassificationInputV1.build(
        input_id="kci_tender_compat_" + canonical_content_hash(source_payload).removeprefix("sha256:")[:16],
        organization_id="tender_compat",
        workspace_id=None,
        source_kind="controlled_document" if canonical_document is not None else "text",
        source_ref="tender_sensitive_term_detection_modeling.compat",
        source_hash=canonical_content_hash(source_payload),
        normalized_artifact_ref=None,
        normalized_artifact_hash=None,
        parser_profile_version="tender_sensitive_term_detection_modeling.compat.v1",
        schema_version="canonical_classification_input.v1",
        segments=tuple(segments),
        lineage_manifest={
            "compatibility_plugin": "tender_sensitive_term_detection_modeling",
            "classifier": "knowledge_sensitive_labeling",
        },
    )


def _tender_findings_from_generic_occurrences(
    *,
    generic_occurrences: list[Any],
    active_rules: list[dict[str, Any]],
    resolver_evidence_ref: str,
    canonical_document: dict[str, Any] | None,
    document_text: str,
) -> list[dict[str, Any]]:
    rules_by_id = {str(rule["rule_version_id"]): rule for rule in active_rules}
    # Occurrences stay countable upstream; a repeated lexical term in the same
    # semantic anchor is one assertion candidate, not multiple findings.
    grouped: dict[tuple[str, str], list[Any]] = {}
    for occurrence in generic_occurrences:
        grouped.setdefault(
            (
                occurrence.source_anchor_ref,
                occurrence.matched_text,
            ),
            [],
        ).append(occurrence)
    findings: list[dict[str, Any]] = []
    for items in grouped.values():
        matched_rule_ids = sorted(
            {item.classification_id for item in items if item.classification_id in rules_by_id}
        )
        rules = [rules_by_id[rule_id] for rule_id in matched_rule_ids]
        if not rules:
            continue
        first = items[0]
        element = _element_for_anchor(
            canonical_document=canonical_document,
            anchor_ref=first.source_anchor_ref,
            fallback_hash=first.local_text_hash,
            document_text=document_text,
        )
        scope = rules[0].get("applicability_scope") or {}
        decision = classify_semantic_assertion(
            clause=str(element.get("text") or document_text),
            term=first.matched_text,
            issue_type=str(scope.get("issue_type") or "compliance_violation"),
        )
        if decision.state == AssertionState.NOT_ASSERTED:
            continue
        findings.append(
            _finding_for_match(
                term=first.matched_text,
                rules=rules,
                resolver_evidence_ref=resolver_evidence_ref,
                occurrence={
                    "element": element,
                    "start_offset": first.start_offset,
                    "end_offset": first.end_offset,
                    "occurrence_index": first.occurrence_index,
                },
                semantic_decision=decision,
                detection_method=str(getattr(first, "detection_method", "exact")),
                semantic_relation=str(getattr(first, "semantic_relation", "exact")),
                precedent_refs=list(getattr(first, "precedent_refs", [])),
            )
        )
    return findings


def _element_for_anchor(
    *,
    canonical_document: dict[str, Any] | None,
    anchor_ref: str,
    fallback_hash: str,
    document_text: str,
) -> dict[str, Any]:
    if canonical_document is not None:
        for element in canonical_document.get("elements") or []:
            if str(element.get("anchor_ref") or element.get("id") or "") == anchor_ref:
                return element
    return {
        "anchor_ref": anchor_ref,
        "kind": "plain_text",
        "id": "document",
        "local_text_hash": fallback_hash,
        "text": document_text,
    }


def _with_literal_fallbacks(
    *,
    occurrences: list[Any],
    canonical_input: CanonicalClassificationInputV1,
    active_rules: list[dict[str, Any]],
) -> list[Any]:
    existing = {
        (item.classification_id, item.source_anchor_ref, item.start_offset, item.end_offset)
        for item in occurrences
    }
    for rule in active_rules:
        rule_id = str(rule["rule_version_id"])
        scope = rule.get("applicability_scope") or {}
        for term in scope.get("match_terms") or []:
            term = str(term)
            for segment in canonical_input.segments:
                start = segment.text.find(term)
                occurrence_index = 0
                while start >= 0:
                    end = start + len(term)
                    key = (rule_id, segment.source_anchor_ref, start, end)
                    if key not in existing:
                        occurrences.append(
                            SimpleNamespace(
                                classification_id=rule_id,
                                source_anchor_ref=segment.source_anchor_ref,
                                start_offset=start,
                                end_offset=end,
                                matched_text=term,
                                local_text_hash=segment.local_text_hash,
                                occurrence_index=occurrence_index,
                            )
                        )
                        existing.add(key)
                    occurrence_index += 1
                    start = segment.text.find(term, end)
    return occurrences


def _with_precedent_fallbacks(
    *,
    occurrences: list[Any],
    canonical_input: CanonicalClassificationInputV1,
    active_rules: list[dict[str, Any]],
    precedent_observed_expressions: list[dict[str, Any]],
    precedent_model_ref: str | None,
) -> list[Any]:
    if not precedent_observed_expressions:
        return occurrences
    rules_by_concept = {
        str(rule.get("source_graph_rule_id") or rule.get("rule_version_id")): str(rule["rule_version_id"])
        for rule in active_rules
    }
    existing = {
        (item.classification_id, item.source_anchor_ref, item.start_offset, item.end_offset)
        for item in occurrences
    }
    for expression in precedent_observed_expressions:
        if expression.get("grounding_state") != "human_accepted":
            continue
        rule_id = rules_by_concept.get(str(expression.get("concept_id") or ""))
        text = str(expression.get("normalized_text") or "").strip()
        if not rule_id or not text:
            continue
        for segment in canonical_input.segments:
            haystack = segment.text.casefold()
            start = haystack.find(text)
            occurrence_index = 0
            while start >= 0:
                end = start + len(text)
                key = (rule_id, segment.source_anchor_ref, start, end)
                if key not in existing:
                    occurrences.append(
                        SimpleNamespace(
                            classification_id=rule_id,
                            source_anchor_ref=segment.source_anchor_ref,
                            start_offset=start,
                            end_offset=end,
                            matched_text=segment.text[start:end],
                            local_text_hash=segment.local_text_hash,
                            occurrence_index=occurrence_index,
                            detection_method="precedent",
                            semantic_relation=str(expression.get("semantic_relation") or "paraphrase"),
                            precedent_refs=[
                                {
                                    "observed_expression_id": str(expression.get("expression_id")),
                                    "latest_observation_id": str(expression.get("latest_observation_id")),
                                    "precedent_model_ref": precedent_model_ref,
                                }
                            ],
                        )
                    )
                    existing.add(key)
                occurrence_index += 1
                start = haystack.find(text, end)
    return occurrences


def _finding_for_match(
    *,
    term: str,
    rules: list[dict[str, Any]],
    resolver_evidence_ref: str,
    occurrence: dict[str, Any],
    semantic_decision: Any | None = None,
    detection_method: str = "exact",
    semantic_relation: str = "exact",
    precedent_refs: list[Any] | None = None,
) -> dict[str, Any]:
    rule = rules[0]
    same_span_conflict = len(rules) > 1
    rule_ids = sorted(str(item["rule_version_id"]) for item in rules)
    triggered_refs = [
        {
            "rule_version_id": item["rule_version_id"],
            "content_hash": item["content_hash"],
        }
        for item in rules
    ]
    scope = rule.get("applicability_scope") or {}
    issue_type = (
        "conflict_escalation"
        if same_span_conflict
        else (
            str(scope.get("issue_type") or "compliance_violation")
            if isinstance(scope, dict)
            else "compliance_violation"
        )
    )
    return _finding(
        term=term,
        rule=rule,
        issue_type=issue_type,
        triggered_rule_version_id=",".join(rule_ids) if same_span_conflict else rule_ids[0],
        resolver_evidence_ref=resolver_evidence_ref,
        escalation_flag=same_span_conflict,
        occurrence=occurrence,
        triggered_rule_version_refs=triggered_refs,
        winning_rule_version_ref=None if same_span_conflict else rule_ids[0],
        conflicting_rule_version_refs=triggered_refs if same_span_conflict else [],
        conflict_resolution_ref="conflict_escalation" if same_span_conflict else None,
        semantic_decision=semantic_decision,
        detection_method=detection_method,
        semantic_relation=semantic_relation,
        precedent_refs=precedent_refs or [],
    )


def _finding(
    *,
    term: str,
    rule: dict[str, Any],
    issue_type: str,
    triggered_rule_version_id: str,
    resolver_evidence_ref: str,
    escalation_flag: bool,
    occurrence: dict[str, Any],
    triggered_rule_version_refs: list[dict[str, Any]],
    winning_rule_version_ref: str | None,
    conflicting_rule_version_refs: list[dict[str, Any]],
    conflict_resolution_ref: str | None,
    semantic_decision: Any | None = None,
    detection_method: str = "exact",
    semantic_relation: str = "exact",
    precedent_refs: list[Any] | None = None,
) -> dict[str, Any]:
    element = occurrence.get("element") or {}
    target_anchor_ref = element.get("anchor_ref") or element.get("id")
    target_element_kind = element.get("kind")
    target_element_id = element.get("id")
    target_element_hash = element.get("local_text_hash")
    start_offset = occurrence.get("start_offset")
    end_offset = occurrence.get("end_offset")
    occurrence_index = occurrence.get("occurrence_index")
    finding_id = f"tdf_{uuid4().hex}"
    decision_state = getattr(semantic_decision, "state", AssertionState.SUPPORTED)
    semantic_relation = semantic_relation or str(occurrence.get("semantic_relation") or "exact")
    concept_id = str(rule.get("source_graph_rule_id") or rule.get("rule_version_id"))
    evidence_text = str(element.get("text") or term)
    decision_service = TenderFindingDecisionService()
    decision_context = decision_service.context_from_evidence(
        finding_id=finding_id,
        concept_id=concept_id,
        evidence_text=evidence_text,
        semantic_relation=semantic_relation,
        semantic_reason=getattr(semantic_decision, "reason_code", "explicit_issue_assertion"),
        supported=decision_state == AssertionState.SUPPORTED,
    )
    decision_status = decision_service.decide(decision_context, escalation=escalation_flag)
    draft = CandidateFindingDraftDTO(
        finding_id=finding_id,
        tender_document_id="draft_unpersisted",
        source_anchor_refs=tuple(str(ref) for ref in (rule.get("source_anchor_refs") or [])),
        matched_sensitive_concept_id=concept_id,
        observed_expression=term,
        detection_method=detection_method,  # type: ignore[arg-type]
        semantic_relation=semantic_relation,
        evidence_text=evidence_text,
        confidence=float(rule.get("confidence") or 1.0),
        precedent_refs=tuple(precedent_refs or ()),
        decision_status=decision_status,
        grounding_status="ungrounded",
    )
    return {
        "schema_version": draft.schema_version,
        "finding_id": finding_id,
        "clause_id": "document",
        "clause_anchor": target_anchor_ref or f"text:contains:{term}",
        "issue_type": issue_type,
        "matched_text": term,
        "normalized_match": term.strip().lower(),
        "triggered_rule_version_id": triggered_rule_version_id,
        "triggered_rule_version_refs": triggered_rule_version_refs,
        "winning_rule_version_ref": winning_rule_version_ref,
        "conflicting_rule_version_refs": conflicting_rule_version_refs,
        "conflict_resolution_ref": conflict_resolution_ref,
        "source_anchor_refs": list(draft.source_anchor_refs),
        "judgment_basis": str(rule.get("extracted_rationale") or "governed rule matched tender text"),
        "severity": str(rule.get("severity") or "medium"),
        "confidence": draft.confidence,
        "suggested_replacement": _suggested_replacement(term=term, rule=rule),
        "escalation_flag": escalation_flag,
        "resolver_evidence_ref": resolver_evidence_ref,
        "target_anchor_ref": target_anchor_ref,
        "target_element_kind": target_element_kind,
        "target_element_id": target_element_id,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "occurrence_index": occurrence_index,
        "target_element_hash": target_element_hash,
        "matched_sensitive_concept_id": draft.matched_sensitive_concept_id,
        "observed_expression": draft.observed_expression,
        "detection_method": draft.detection_method,
        "semantic_relation": draft.semantic_relation,
        "evidence_text": draft.evidence_text,
        "precedent_refs": list(draft.precedent_refs),
        "decision_status": draft.decision_status,
        "grounding_status": draft.grounding_status,
        "decision_context": decision_context.to_dict(),
    }


def _suggested_replacement(*, term: str, rule: dict[str, Any]) -> str | None:
    scope = rule.get("applicability_scope") or {}
    if isinstance(scope, dict) and scope.get("suggested_replacement"):
        return str(scope["suggested_replacement"])
    if term:
        return f"AI-assisted recommendation: remove or revise '{term}' to satisfy the governed rule."
    return None


def _classify_runtime_conflicts(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    anchored = [
        finding
        for finding in findings
        if finding.get("target_anchor_ref")
        and finding.get("start_offset") is not None
        and finding.get("end_offset") is not None
    ]
    for finding in anchored:
        refs = finding.get("triggered_rule_version_refs") or []
        if len(refs) > 1:
            conflicts.append(
                _runtime_conflict(
                    conflict_type="same_span_multi_rule",
                    findings=[finding],
                    target_anchor_ref=str(finding["target_anchor_ref"]),
                    start_offset=int(finding["start_offset"]),
                    end_offset=int(finding["end_offset"]),
                )
            )
    for index, left in enumerate(anchored):
        for right in anchored[index + 1 :]:
            if left.get("target_anchor_ref") != right.get("target_anchor_ref"):
                continue
            left_start = int(left["start_offset"])
            left_end = int(left["end_offset"])
            right_start = int(right["start_offset"])
            right_end = int(right["end_offset"])
            if left_start == right_start and left_end == right_end:
                continue
            if left_start < right_end and right_start < left_end:
                conflicts.append(
                    _runtime_conflict(
                        conflict_type="overlapping_span_multi_rule",
                        findings=[left, right],
                        target_anchor_ref=str(left["target_anchor_ref"]),
                        start_offset=min(left_start, right_start),
                        end_offset=max(left_end, right_end),
                    )
                )
    return conflicts


def _runtime_conflict(
    *,
    conflict_type: str,
    findings: list[dict[str, Any]],
    target_anchor_ref: str,
    start_offset: int,
    end_offset: int,
) -> dict[str, Any]:
    rule_refs: dict[str, str] = {}
    finding_ids: list[str] = []
    for finding in findings:
        finding_ids.append(str(finding["finding_id"]))
        refs = finding.get("triggered_rule_version_refs") or []
        if refs:
            for ref in refs:
                rule_refs[str(ref["rule_version_id"])] = str(ref["content_hash"])
        else:
            rule_refs[str(finding["triggered_rule_version_id"])] = ""
    conflict_key = "|".join(
        [
            conflict_type,
            target_anchor_ref,
            str(start_offset),
            str(end_offset),
            ",".join(sorted(rule_refs)),
        ]
    )
    conflict_id = "tdrc_" + sha256(conflict_key.encode("utf-8")).hexdigest()[:24]
    return {
        "runtime_conflict_id": conflict_id,
        "conflict_type": conflict_type,
        "target_anchor_ref": target_anchor_ref,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "finding_ids": sorted(set(finding_ids)),
        "rule_version_refs": [
            {"rule_version_id": rule_id, "content_hash": content_hash}
            for rule_id, content_hash in sorted(rule_refs.items())
        ],
        "status": "non_executable_requires_governed_resolution",
    }


def _mark_conflicted_findings(
    findings: list[dict[str, Any]],
    runtime_conflicts: list[dict[str, Any]],
) -> None:
    by_id = {str(finding["finding_id"]): finding for finding in findings}
    for conflict in runtime_conflicts:
        conflict_ref = str(conflict["runtime_conflict_id"])
        conflict_rule_refs = conflict["rule_version_refs"]
        for finding_id in conflict.get("finding_ids") or []:
            finding = by_id.get(str(finding_id))
            if finding is None:
                continue
            finding["escalation_flag"] = True
            finding["conflict_resolution_ref"] = conflict_ref
            finding["winning_rule_version_ref"] = None
            finding["conflicting_rule_version_refs"] = conflict_rule_refs
