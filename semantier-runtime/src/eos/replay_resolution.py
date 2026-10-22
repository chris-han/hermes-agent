"""Replay support for v5 resolution results."""

from __future__ import annotations

from typing import Any, Dict

from eos.db import get_justification, get_latest_validation
from eos.resolution_engine import run_resolution
from eos.validation_service import validate_tiered
from eos.replay_binding_store import verify_replay_binding


def replay_resolution(justification_id: str) -> Dict[str, Any]:
    """Recompute validation + resolution for a committed justification."""
    binding_verification = verify_replay_binding(justification_id)
    if binding_verification.get("status") == "FAIL_CLOSED":
        return {
            "status": "FAIL_CLOSED",
            "reason": binding_verification.get("reason"),
            "binding_verification": binding_verification,
        }

    jst_row = get_justification(justification_id)
    if not jst_row:
        return {"status": "NOT_FOUND", "reason": "justification_not_found"}

    validation_row = get_latest_validation(justification_id)
    if not validation_row:
        return {"status": "NOT_FOUND", "reason": "validation_not_found"}

    justification = jst_row["justification"]
    artifacts = (justification.get("governance_artifacts") or {})
    pins = justification.get("pins") or {}
    policy_version = pins.get("C_v", "rb_v1")

    validation = validate_tiered(
        justification,
        policy_version=policy_version,
        O_tag_v=artifacts.get("O_tag_v"),
        K_v=artifacts.get("K_v"),
        Pi_v=artifacts.get("Pi_v"),
        ConstraintBundle_v=artifacts.get("ConstraintBundle_v"),
        PrecedenceGraph_v=artifacts.get("PrecedenceGraph_v"),
        ResolutionRules_v=artifacts.get("ResolutionRules_v"),
    )
    replay_resolution_result = run_resolution(
        claims=justification.get("claims", []),
        constraint_results=validation.get("constraint_results", []),
        precedence_graph=artifacts.get("PrecedenceGraph_v") or {},
        resolution_rules=artifacts.get("ResolutionRules_v") or {},
        scope="execution",
        pins=pins,
    )

    stored_resolution = (validation_row.get("result") or {}).get("resolution_result")
    return {
        "status": "OK",
        "validation_status": validation.get("status"),
        "stored_resolution_result": stored_resolution,
        "replayed_resolution_result": replay_resolution_result,
        "resolution_match": stored_resolution == replay_resolution_result,
    }
