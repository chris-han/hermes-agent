from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contracts.knowledge_classification import (
    CanonicalClassificationInputV1,
    FieldFindingV1,
    OccurrenceFindingV1,
    canonical_content_hash,
)

from .conflict_detector import detect_runtime_conflicts


@dataclass(frozen=True)
class ClassificationDomainResult:
    occurrences: list[OccurrenceFindingV1]
    fields: list[FieldFindingV1]
    runtime_conflicts: list[Any]


def classify_canonical_input(
    *,
    canonical_input: CanonicalClassificationInputV1,
    active_rules: list[dict[str, Any]],
    resolver_evidence_ref: str,
) -> ClassificationDomainResult:
    occurrences = _classify_segments(
        canonical_input=canonical_input,
        active_rules=active_rules,
        resolver_evidence_ref=resolver_evidence_ref,
    )
    fields = _classify_fields(
        canonical_input=canonical_input,
        active_rules=active_rules,
        resolver_evidence_ref=resolver_evidence_ref,
    )
    conflicts = detect_runtime_conflicts(
        source_hash=canonical_input.source_hash,
        occurrences=occurrences,
        fields=fields,
    )
    return ClassificationDomainResult(
        occurrences=occurrences,
        fields=fields,
        runtime_conflicts=conflicts,
    )


def _rules_by_term(active_rules: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    term_rules: dict[str, list[dict[str, Any]]] = {}
    for rule in active_rules:
        scope = rule.get("applicability_scope") or {}
        terms = scope.get("match_terms") if isinstance(scope, dict) else None
        for term in [str(item).strip() for item in (terms or []) if str(item).strip()]:
            term_rules.setdefault(term, []).append(rule)
    return term_rules


def _classify_segments(
    *,
    canonical_input: CanonicalClassificationInputV1,
    active_rules: list[dict[str, Any]],
    resolver_evidence_ref: str,
) -> list[OccurrenceFindingV1]:
    findings: list[OccurrenceFindingV1] = []
    occurrence_counts: dict[tuple[str, str], int] = {}
    for segment in canonical_input.segments:
        for term, rules in _rules_by_term(active_rules).items():
            start = segment.text.find(term)
            while start >= 0:
                end = start + len(term)
                for rule in rules:
                    classification_id = str(rule["rule_version_id"])
                    key = (segment.source_anchor_ref, term)
                    occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
                    payload = {
                        "source_hash": canonical_input.source_hash,
                        "source_anchor_ref": segment.source_anchor_ref,
                        "start_offset": start,
                        "end_offset": end,
                        "classification_id": classification_id,
                    }
                    finding_hash = canonical_content_hash(payload)
                    findings.append(
                        OccurrenceFindingV1(
                            finding_id="kof_" + finding_hash.removeprefix("sha256:")[:24],
                            source_hash=canonical_input.source_hash,
                            source_anchor_ref=segment.source_anchor_ref,
                            start_offset=start,
                            end_offset=end,
                            occurrence_index=occurrence_counts[key],
                            classification_id=classification_id,
                            matched_text=term,
                            local_text_hash=segment.local_text_hash,
                            content_hash=canonical_content_hash(payload),
                            metadata={
                                "resolver_evidence_ref": resolver_evidence_ref,
                                "rule_content_hash": rule.get("content_hash"),
                                "source_anchor_refs": rule.get("source_anchor_refs") or [],
                                "issue_type": (rule.get("applicability_scope") or {}).get("issue_type")
                                if isinstance(rule.get("applicability_scope"), dict)
                                else None,
                            },
                        )
                    )
                start = segment.text.find(term, end)
    return findings


def _classify_fields(
    *,
    canonical_input: CanonicalClassificationInputV1,
    active_rules: list[dict[str, Any]],
    resolver_evidence_ref: str,
) -> list[FieldFindingV1]:
    findings: list[FieldFindingV1] = []
    for record in canonical_input.records:
        for field in record.field_values:
            for term, rules in _rules_by_term(active_rules).items():
                if term not in field.normalized_value:
                    continue
                for rule in rules:
                    classification_id = str(rule["rule_version_id"])
                    payload = {
                        "source_hash": canonical_input.source_hash,
                        "record_id": record.record_id,
                        "field_path": field.field_path,
                        "value_hash": field.value_hash,
                        "classification_id": classification_id,
                    }
                    finding_hash = canonical_content_hash(payload)
                    findings.append(
                        FieldFindingV1(
                            finding_id="kff_" + finding_hash.removeprefix("sha256:")[:24],
                            source_hash=canonical_input.source_hash,
                            record_id=record.record_id,
                            field_path=field.field_path,
                            value_hash=field.value_hash,
                            classification_id=classification_id,
                            matched_text=term,
                            content_hash=canonical_content_hash(payload),
                            metadata={
                                "resolver_evidence_ref": resolver_evidence_ref,
                                "rule_content_hash": rule.get("content_hash"),
                                "source_anchor_refs": rule.get("source_anchor_refs") or [],
                                "source_anchor_ref": field.source_anchor_ref,
                                "issue_type": (rule.get("applicability_scope") or {}).get("issue_type")
                                if isinstance(rule.get("applicability_scope"), dict)
                                else None,
                            },
                        )
                    )
    return findings
