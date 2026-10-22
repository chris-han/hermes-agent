"""Deterministic tender Phase 1 benchmark loader and baseline runner."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

from contracts.knowledge_feature_pack import content_hash
from plugins.tender_sensitive_term_detection_modeling.detector import detect_tender_occurrences


FAMILY_ISSUE_TYPES = {
    1: "enterprise_scale_restriction", 2: "operating_history_threshold",
    3: "capital_scale_threshold", 4: "financial_threshold", 5: "geographic_discrimination",
    6: "brand_designation", 7: "intellectual_property_requirement",
    8: "past_performance_requirement", 9: "personnel_experience_threshold",
    10: "unnecessary_certification_requirement", 11: "exclusionary_constraint",
    12: "supplier_identity_restriction", 13: "biased_evaluation_criterion",
    14: "unbalanced_contract_risk", 15: "forced_lowest_price_award",
}


_VOLATILE_FINDING_IDENTITY_FIELDS = frozenset({"finding_id", "finding_ids"})


def stable_finding_hash_material(value: Any) -> Any:
    """Remove run-local occurrence identity before hashing benchmark findings."""
    if isinstance(value, dict):
        return {
            key: stable_finding_hash_material(item)
            for key, item in value.items()
            if key not in _VOLATILE_FINDING_IDENTITY_FIELDS
        }
    if isinstance(value, list):
        return [stable_finding_hash_material(item) for item in value]
    return value


def load_benchmark_cases(fixture_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path_value in fixture_paths:
        path = Path(path_value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            cases.append({**case, "fixture_path": path.as_posix()})
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark case IDs must be unique")
    return cases


def build_family_coverage_matrix(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "family_id": 0,
            "family": "",
            "positive": 0,
            "hard_negative": 0,
            "contextual": 0,
            "cross_document": 0,
        }
    )
    for case in cases:
        family_id = int(case["family_id"])
        row = counts[family_id]
        row["family_id"] = family_id
        row["family"] = str(case["family"])
        kind = str(case["case_kind"])
        if kind == "positive":
            row["positive"] += 1
        else:
            row["hard_negative"] += 1
        if kind in {"contextual_hard_negative", "cross_document_hard_negative"}:
            row["contextual"] += 1
        if kind == "cross_document_hard_negative":
            row["cross_document"] += 1
    return [counts[family_id] for family_id in sorted(counts)]


def validate_all_family_coverage(matrix: list[dict[str, Any]]) -> None:
    if [row["family_id"] for row in matrix] != list(range(1, 16)):
        raise ValueError("benchmark must cover all 15 canonical POC families")
    missing = [
        row["family_id"]
        for row in matrix
        if row["positive"] < 1 or row["hard_negative"] < 1
    ]
    if missing:
        raise ValueError(f"families missing positive or hard-negative coverage: {missing}")


def run_deterministic_baseline(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for case in cases:
        terms = list(case.get("terms") or [case["match_term"]])
        rule = {
            "rule_version_id": f"baseline::{case['case_id']}",
            "content_hash": f"sha256:{case['case_id']}",
            "rule_family": str(case["family"]),
            "rule_state": "ACTIVATED",
            "governed_rule_text": " / ".join(terms),
            "extracted_rationale": str(case.get("negative_reason") or case["family"]),
            "severity": "high",
            "source_anchor_refs": [
                str(case.get("source_anchor_ref") or f"fixture:{case['case_id']}")
            ],
            "applicability_scope": {
                "match_terms": [str(terms[0])],
                "issue_type": str(
                    case.get("issue_type") or FAMILY_ISSUE_TYPES[int(case["family_id"])]
                ),
            },
        }
        detection = detect_tender_occurrences(
            document_text=str(case["text"]),
            canonical_document=None,
            active_rules=[rule],
            resolver_evidence_ref="tender-sensitive-v1-baseline",
        )
        actual = len(detection.findings)
        expected = int(case["expected_finding_count"])
        results.append(
            {
                "case_id": case["case_id"],
                "family_id": int(case["family_id"]),
                "case_kind": case["case_kind"],
                "expected_finding_count": expected,
                "actual_finding_count": actual,
                "passed": actual == expected,
                "finding_hash": content_hash(
                    stable_finding_hash_material(detection.findings)
                ),
            }
        )
    ordered = sorted(results, key=lambda item: item["case_id"])
    return {
        "runner_version": "tender_sensitive_deterministic_baseline.v1",
        "case_manifest_hash": content_hash(
            [
                {
                    "case_id": result["case_id"],
                    "expected_finding_count": result["expected_finding_count"],
                }
                for result in ordered
            ]
        ),
        "result_hash": content_hash(ordered),
        "passed": sum(result["passed"] for result in ordered),
        "failed": sum(not result["passed"] for result in ordered),
        "results": ordered,
    }


def build_regression_cases_from_accepted_errors(
    events: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases = []
    for event in events:
        label = event.get("human_label") or {}
        process_labels = label.get("process_labels") or {}
        if label.get("decision") not in {"edit", "reject"} and "missing" not in process_labels.values():
            continue
        cases.append({
            "regression_case_id": f"regression::{event['event_id']}",
            "source_event_id": event["event_id"],
            "source_document_hash": event["source_document_hash"],
            "source_anchor_refs": list(event["source_anchor_refs"]),
            "graph_version": event["graph_version"],
            "graph_delta_ref": event.get("graph_delta_ref"),
            "human_decision": label.get("decision"),
            "process_labels": process_labels,
            "expected_verdict": event.get("downstream_outcome", {}).get("expected_verdict"),
        })
    return sorted(cases, key=lambda item: item["regression_case_id"])


def compare_parallel_proposals(
    *, deterministic_assertions: list[dict[str, Any]], llm_assertions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Persistable bake-off view; neither proposal stream is accepted authority."""
    return {
        "deterministic": deterministic_assertions,
        "llm": llm_assertions,
        "deterministic_hash": content_hash(deterministic_assertions),
        "llm_hash": content_hash(llm_assertions),
        "authority": "proposal_only",
    }
