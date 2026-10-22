"""Heuristic review pass for active or imported T5 candidates."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def audit_t5_candidate_jurisdiction(candidate: dict) -> Dict[str, Any]:
    semantic_tier = candidate.get("semantic_tier")
    content_yaml = str(candidate.get("content_yaml") or "")
    scoped_owner = candidate.get("scoped_owner")
    if semantic_tier == "T5" and not scoped_owner:
        return {"status": "REVIEW_REQUIRED", "reason": "missing_scope_metadata"}
    if semantic_tier == "T5" and "approval" in content_yaml.lower():
        return {"status": "REVIEW_REQUIRED", "reason": "potential_t4_jurisdiction_drift"}
    return {"status": "OK", "reason": "scoped_preference"}


def mark_legacy_t5_candidates_for_review(candidates: Iterable[dict]) -> list[dict]:
    results = []
    for candidate in candidates:
        audit = audit_t5_candidate_jurisdiction(candidate)
        results.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "review_status": audit["status"],
                "jurisdiction_reason": audit["reason"],
                "imported_legacy": 1,
            }
        )
    return results
