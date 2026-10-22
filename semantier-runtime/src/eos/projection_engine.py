"""Projection engine boundary module (v6 Slice 5)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from eos.policy_projection import project_rea_to_ledger
from eos.projection_context_store import load_snapshot

_CONTEXT_ONLY_RULE_KEYS = {"binding_rules", "rule_candidates", "policy_rules", "executable_rules"}


def _contains_context_only_rule_material(projection_context: Dict[str, Any]) -> bool:
    if not projection_context.get("context_only"):
        return False
    return any(projection_context.get(key) for key in _CONTEXT_ONLY_RULE_KEYS)


def run_projection(
    *,
    rea_id: str,
    rea_claim: Dict[str, Any],
    active_coa: Dict[str, Any],
    projection_context_hash: str,
    projection_context_ref: Optional[str] = None,
    semantic_claims: Optional[list[dict]] = None,
    projection_context: Optional[Dict[str, Any]] = None,
    pin_versions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if projection_context is None and projection_context_ref:
        snapshot = load_snapshot(projection_context_ref)
        if snapshot is None:
            return {
                "status": "FAIL_CLOSED",
                "reason": "projection_context_snapshot_missing",
                "projections": [],
            }
        projection_context = snapshot.get("selected_context") or {}

    if projection_context and _contains_context_only_rule_material(projection_context):
        return {
            "status": "FAIL_CLOSED",
            "reason": "context_only_rule_evaluation_blocked",
            "projections": [],
        }

    return project_rea_to_ledger(
        rea_id=rea_id,
        rea_claim=rea_claim,
        active_coa=active_coa,
        projection_context_hash=projection_context_hash,
        semantic_claims=semantic_claims,
        projection_context=projection_context,
        pin_versions=pin_versions,
    )
