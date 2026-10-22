"""Deterministic promotion routing classifier."""

from __future__ import annotations

from typing import Any, Dict


def classify_proposal_jurisdiction(
    *,
    authority_domain: str,
    scope: str,
    affects_approval_logic: bool,
    affects_projection_meaning: bool,
    source_kind: str,
) -> Dict[str, Any]:
    if affects_approval_logic and scope == "organization":
        return {
            "route_family": "SEMANTIER_ESCALATED",
            "target_tier": "T4",
            "fast_track_allowed": False,
            "reason": "higher_tier_governance_review_required",
            "authority_domain": authority_domain,
            "source_kind": source_kind,
        }
    if affects_projection_meaning:
        return {
            "route_family": "SEMANTIER_GOVERNED_T5",
            "target_tier": "T5",
            "fast_track_allowed": True,
            "reason": "governed_t5_candidate",
            "authority_domain": authority_domain,
            "source_kind": source_kind,
        }
    return {
        "route_family": "HERMES_PROCEDURAL",
        "target_tier": "T6",
        "fast_track_allowed": False,
        "reason": "procedural_scope_only",
        "authority_domain": authority_domain,
        "source_kind": source_kind,
    }
