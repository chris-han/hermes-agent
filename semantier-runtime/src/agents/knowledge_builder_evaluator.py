from __future__ import annotations

from typing import Any

EVALUATION_CASE_TYPES = {
    "positive",
    "negative",
    "ambiguous",
    "known_false_positive",
    "known_false_negative",
    "escalation_required",
}

EVALUATION_OUTCOMES = {"match", "exclude", "ambiguous", "escalate"}


def evaluate_example(
    *,
    example: dict[str, Any],
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = str(example.get("input_text") or "").lower()
    outcome = _classify_text(text)
    expected = str(example.get("expected_outcome") or "")
    matched_path = []
    excluded_path = []
    if outcome == "match":
        matched_path = _candidate_path(graph)
    if outcome == "exclude":
        excluded_path = [
            {
                "reason_code": "KNOWN_FALSE_FRIEND",
                "explanation": "Identifier-only wording is excluded from supplier exclusivity.",
            }
        ]
    error_labels = [] if outcome == expected else [_error_label(expected, outcome)]
    return {
        "actual_outcome": outcome,
        "matched_path": matched_path,
        "excluded_path": excluded_path,
        "ai_assisted_rating": "pass" if outcome == expected else "fail",
        "explanation_acceptance": "accepted" if not error_labels else "needs_review",
        "error_labels": error_labels,
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(results)
    if total == 0:
        return {
            "example_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "ambiguity_escalation_rate": 0.0,
            "source_anchor_coverage": 0.0,
            "explanation_acceptance_rate": 0.0,
            "remediation_acceptance_rate": 0.0,
        }
    tp = sum(1 for row in results if row["expected_outcome"] == "match" and row["actual_outcome"] == "match")
    fp = sum(1 for row in results if row["expected_outcome"] != "match" and row["actual_outcome"] == "match")
    fn = sum(1 for row in results if row["expected_outcome"] == "match" and row["actual_outcome"] != "match")
    expected_positive = tp + fn
    actual_positive = tp + fp
    ambiguous_or_escalated = sum(
        1 for row in results if row["actual_outcome"] in {"ambiguous", "escalate"}
    )
    anchored = sum(1 for row in results if row["matched_path"] or row["excluded_path"])
    explanations_accepted = sum(
        1 for row in results if row["explanation_acceptance"] == "accepted"
    )
    remediations_accepted = sum(
        1 for row in results if row["ai_assisted_rating"] == "pass"
    )
    return {
        "example_count": total,
        "precision": _ratio(tp, actual_positive),
        "recall": _ratio(tp, expected_positive),
        "false_positive_rate": _ratio(fp, total),
        "false_negative_rate": _ratio(fn, total),
        "ambiguity_escalation_rate": _ratio(ambiguous_or_escalated, total),
        "source_anchor_coverage": _ratio(anchored, total),
        "explanation_acceptance_rate": _ratio(explanations_accepted, total),
        "remediation_acceptance_rate": _ratio(remediations_accepted, total),
    }


def anchored_occurrence_identity(
    *,
    target_document_hash: str,
    target_anchor_ref: str,
    start_offset: int,
    end_offset: int,
    issue_type: str,
) -> str:
    return "|".join(
        [
            str(target_document_hash),
            str(target_anchor_ref),
            str(start_offset),
            str(end_offset),
            str(issue_type),
        ]
    )


def compute_anchored_occurrence_metrics(
    *,
    target_document_hash: str,
    findings: list[dict[str, Any]],
    gold_labels: list[dict[str, Any]],
    dispositions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gold_positive = {
        anchored_occurrence_identity(
            target_document_hash=target_document_hash,
            target_anchor_ref=str(label["target_anchor_ref"]),
            start_offset=int(label["start_offset"]),
            end_offset=int(label["end_offset"]),
            issue_type=str(label["issue_type"]),
        )
        for label in gold_labels
        if bool(label.get("expected_present", True))
    }
    actual_positive = {
        anchored_occurrence_identity(
            target_document_hash=target_document_hash,
            target_anchor_ref=str(finding.get("target_anchor_ref") or finding["clause_anchor"]),
            start_offset=int(finding.get("start_offset") or 0),
            end_offset=int(finding.get("end_offset") or 0),
            issue_type=str(finding["issue_type"]),
        )
        for finding in findings
    }
    true_positive = gold_positive & actual_positive
    false_positive = actual_positive - gold_positive
    false_negative = gold_positive - actual_positive
    dispositions = dispositions or []
    accepted = sum(
        1
        for disposition in dispositions
        if disposition.get("disposition") in {"accepted", "edited"}
    )
    return {
        "metric_schema_version": "anchored_occurrence_metrics.v1",
        "matching_identity": "target_document_hash|target_anchor_ref|start_offset|end_offset|issue_type",
        "matching_tolerance": "exact_anchor_offsets_issue_type",
        "gold_label_count": len(gold_positive),
        "actual_finding_count": len(actual_positive),
        "true_positive_count": len(true_positive),
        "false_positive_count": len(false_positive),
        "false_negative_count": len(false_negative),
        "capture_rate": _ratio(len(true_positive), len(gold_positive)),
        "precision": _ratio(len(true_positive), len(actual_positive)),
        "recall": _ratio(len(true_positive), len(gold_positive)),
        "type_i_error_rate": _ratio(len(false_positive), len(actual_positive)),
        "type_ii_error_rate": _ratio(len(false_negative), len(gold_positive)),
        "source_anchor_coverage": _ratio(
            sum(1 for finding in findings if finding.get("source_anchor_refs")),
            len(findings),
        ),
        "reviewer_acceptance_rate": _ratio(accepted, len(dispositions)),
        "false_positive_identities": sorted(false_positive),
        "false_negative_identities": sorted(false_negative),
    }


def _classify_text(text: str) -> str:
    if "唯一编号" in text or "account_id" in text:
        return "exclude"
    if "bank_account_number" in text or ("bank account" in text and "platform account" in text):
        return "ambiguous"
    if "escalat" in text or "人工复核" in text or "requires review" in text:
        return "escalate"
    if any(term in text for term in ("唯一授权", "唯一供应商", "独家", "指定供应商", "exclusive", "sole supplier")):
        return "match"
    if "account" in text:
        return "ambiguous"
    return "exclude"


def _candidate_path(graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not graph:
        return []
    relations = graph.get("relations") or []
    if relations:
        relation = relations[0]
        return [
            {
                "candidate_id": relation.get("relation_id"),
                "relation_type": relation.get("relation_type"),
                "source_anchor_refs": relation.get("source_anchor_refs") or [],
            }
        ]
    nodes = graph.get("nodes") or []
    if nodes:
        node = nodes[0]
        return [
            {
                "candidate_id": node.get("node_id"),
                "label": node.get("label"),
                "source_anchor_refs": node.get("source_anchor_refs") or [],
            }
        ]
    return []


def _error_label(expected: str, actual: str) -> str:
    if expected == "exclude" and actual == "match":
        return "false_positive"
    if expected == "match" and actual == "exclude":
        return "false_negative"
    if actual == "ambiguous":
        return "unexpected_ambiguity"
    if actual == "escalate":
        return "unexpected_escalation"
    return "outcome_mismatch"


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
