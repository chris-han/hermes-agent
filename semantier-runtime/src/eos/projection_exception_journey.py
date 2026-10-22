"""v7.7 projection exception journey orchestration.

This module composes v7.5/v7.6 primitives into an end-to-end flow:
- create projection exception with candidate-only suggestions
- record append-only exception status events
- create/approve/activate governance records
- publish governed COA/Pi versions
- perform governed reprojection and mark exception resolved in event stream
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.coa_onboarding_store import activate_onboarding_candidates
from eos.db import get_conn, now_iso
from eos.governed_reprojection import create_governed_reprojection
from eos.projection_exception_governance import (
    activate_projection_exception_governance,
    approve_projection_exception_governance,
    create_projection_exception_governance,
    get_latest_projection_exception_governance,
    get_projection_exception_governance,
)
from eos.projection_exception_store import create_projection_exception
from storage.coa_store import (
    activate_coa_change_proposal,
    activate_projection_rule_proposal,
    approve_coa_change_proposal,
    approve_projection_rule_proposal,
    create_coa_change_proposal,
    create_projection_rule_proposal,
)


EXCEPTION_TYPES = {
    "NO_MATCHING_COA_RULE",
    "AMBIGUOUS_CLASSIFICATION",
    "CONFLICTING_CLAIMS",
    "INSUFFICIENT_EVIDENCE",
    "POLICY_CONSTRAINT_BLOCK",
    "TAX_TREATMENT_UNCLEAR",
    "MANAGEMENT_VIEW_ONLY",
    "NOT_PROJECTABLE_UNDER_CURRENT_SCOPE",
}

ALLOWED_ACTIONS = {
    "MAP_TO_EXISTING_COA_NODE",
    "ADD_CUSTOM_ACCOUNT",
    "ADD_PROJECTION_RULE",
    "SPLIT_PROJECTION",
    "REQUEST_MORE_EVIDENCE",
    "ESCALATE_TO_ADVISOR",
    "MARK_NOT_PROJECTABLE",
}


def _normalize_candidate_claims(candidate_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for claim in candidate_claims:
        row = dict(claim)
        row["status"] = "CANDIDATE_ONLY"
        normalized.append(row)
    return normalized


def append_projection_exception_status_event(
    *,
    projection_exception_id: str,
    status: str,
    reason_code: Optional[str] = None,
    event_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    event = {
        "projection_exception_id": projection_exception_id,
        "status": status,
        "reason_code": reason_code,
        "event_payload": event_payload or {},
    }
    row = {
        "projection_exception_status_event_id": f"pexevt_{uuid4().hex}",
        "projection_exception_id": projection_exception_id,
        "status": status,
        "reason_code": reason_code,
        "event_json": canonical_json(event),
        "content_hash": content_hash(event),
        "created_at": now_iso(),
    }
    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO projection_exception_status_events
              (projection_exception_status_event_id, projection_exception_id, status,
               reason_code, event_json, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO projection_exception_status_events
              (projection_exception_status_event_id, projection_exception_id, status,
               reason_code, event_json, content_hash, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """),
            (
                row["projection_exception_status_event_id"],
                row["projection_exception_id"],
                row["status"],
                row["reason_code"],
                row["event_json"],
                row["content_hash"],
                row["created_at"],
            ),
        )
    row["event"] = event
    return row


def get_latest_projection_exception_status(projection_exception_id: str) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT status
            FROM projection_exception_status_events
            WHERE projection_exception_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """, """
            SELECT status
            FROM projection_exception_status_events
            WHERE projection_exception_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """),
            (projection_exception_id,),
        ).fetchone()
    if row is not None:
        return row["status"]

    with get_conn() as conn:
        exc = conn.execute(
            hosted.sql_query("SELECT status FROM projection_exceptions WHERE projection_exception_id = ?", 'SELECT status FROM projection_exceptions WHERE projection_exception_id = %s'),
            (projection_exception_id,),
        ).fetchone()
    if exc is None:
        return None
    return exc["status"]


def create_projection_exception_for_unmatched_event(
    *,
    rea_id: str,
    justification_id: str,
    projection_context_ref: Optional[str],
    projection_context_hash: Optional[str],
    coa_version: str,
    projection_bundle_version: str,
    constraint_bundle_version: Optional[str],
    reason_code: str,
    reason: str,
    candidate_claims: List[Dict[str, Any]],
    required_action: str = "GOVERNANCE_REVIEW",
) -> Dict[str, Any]:
    if reason_code not in EXCEPTION_TYPES:
        raise ValueError(f"unsupported_exception_type:{reason_code}")

    normalized_claims = _normalize_candidate_claims(candidate_claims)
    payload = {
        "reason_code": reason_code,
        "reason": reason,
        "candidate_claims": normalized_claims,
        "required_action": required_action,
    }

    exc = create_projection_exception(
        rea_id=rea_id,
        justification_id=justification_id,
        projection_context_ref=projection_context_ref,
        projection_context_hash=projection_context_hash,
        coa_version=coa_version,
        projection_bundle_version=projection_bundle_version,
        constraint_bundle_version=constraint_bundle_version,
        status="PROJECTION_EXCEPTION",
        exception_payload=payload,
    )

    append_projection_exception_status_event(
        projection_exception_id=exc["projection_exception_id"],
        status="PROJECTION_EXCEPTION",
        reason_code=reason_code,
        event_payload={"required_action": required_action},
    )

    return exc


def create_governance_record_for_exception(
    *,
    projection_exception_id: str,
    allowed_actions: List[str],
    action_type: str,
    justification: Dict[str, Any],
    proposed_by: Optional[str] = None,
    proposed_changes: Optional[Dict[str, Any]] = None,
    replay_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action_set = set(allowed_actions)
    unknown = action_set - ALLOWED_ACTIONS
    if unknown:
        raise ValueError(f"unsupported_allowed_actions:{sorted(unknown)}")
    if action_type not in action_set:
        raise ValueError(f"selected_action_not_allowed:{action_type}")

    allowed_actions_payload = [{"action": action} for action in allowed_actions]
    selected_action = {
        "action": action_type,
    }

    return create_projection_exception_governance(
        projection_exception_id=projection_exception_id,
        allowed_actions=allowed_actions_payload,
        selected_action=selected_action,
        justification=json.dumps(justification, sort_keys=True),
        validation_results={
            "selected_action": action_type,
            "justification": justification,
            "proposed_changes": proposed_changes or {},
        },
        replay_policy=replay_policy,
        proposed_by=proposed_by,
    )


def approve_and_activate_governance_record(
    governance_id: str,
    *,
    approved_by: str,
    activated_by: str,
) -> Dict[str, Dict[str, Any]]:
    approved = approve_projection_exception_governance(
        governance_id,
        approved_by=approved_by,
    )
    activated = activate_projection_exception_governance(
        approved["governance_id"],
        activated_by=activated_by,
    )
    return {
        "approved": approved,
        "activated": activated,
    }


def publish_governed_versions_from_action(
    governance_id: str,
    *,
    org_id: str,
    actor: str,
) -> Dict[str, Any]:
    governance = get_projection_exception_governance(governance_id)
    if governance is None:
        raise ValueError(f"projection_exception_governance_not_found:{governance_id}")
    if governance["status"] != "ACTIVATED":
        latest = get_latest_projection_exception_governance(
            governance["projection_exception_id"]
        )
        if latest is None or latest["status"] != "ACTIVATED":
            raise ValueError("projection_exception_governance_requires_activation")
        governance = latest

    selected_action = governance["governance"].get("selected_action", {})
    action = selected_action.get("action")
    validation_results = governance["governance"].get("validation_results", {})
    proposed_changes = validation_results.get("proposed_changes", {})

    result: Dict[str, Any] = {
        "governance_id": governance_id,
        "action": action,
        "resulting_version_pins": {},
    }

    if action == "ADD_CUSTOM_ACCOUNT":
        coa_change = proposed_changes.get("coa_change", {})
        projection_rule_change = proposed_changes.get("projection_rule_change", {})

        coa_proposal = create_coa_change_proposal(
            base_coa_version_id=None,
            proposed_accounts=[
                {
                    "account_code": coa_change.get("account_code"),
                    "account_name": coa_change.get("account_name"),
                    "account_class": coa_change.get("account_class", "expense"),
                    "semantic_tier": coa_change.get("semantic_tier", "T3"),
                }
            ],
            justification=coa_change.get("reason", "Governed ADD_CUSTOM_ACCOUNT action"),
            proposed_by=actor,
        )
        approve_coa_change_proposal(coa_proposal["coa_change_proposal_id"], approved_by=actor)
        coa_v = activate_coa_change_proposal(coa_proposal["coa_change_proposal_id"], org_id=org_id)

        pi_proposal = create_projection_rule_proposal(
            base_projection_bundle_id=None,
            proposed_rules=[projection_rule_change] if projection_rule_change else [],
            justification="Governed projection rule update",
            proposed_by=actor,
        )
        approve_projection_rule_proposal(
            pi_proposal["projection_rule_proposal_id"],
            approved_by=actor,
        )
        pi_v = activate_projection_rule_proposal(
            pi_proposal["projection_rule_proposal_id"],
            org_id=org_id,
        )

        result["coa_version"] = coa_v
        result["projection_bundle_version"] = pi_v
        result["resulting_version_pins"] = {
            "COA_v": coa_v["coa_version_id"],
            "Pi_v": pi_v["projection_bundle_id"],
        }
        return result

    if action == "ADD_PROJECTION_RULE":
        projection_rule_change = proposed_changes.get("projection_rule_change", {})
        pi_proposal = create_projection_rule_proposal(
            base_projection_bundle_id=None,
            proposed_rules=[projection_rule_change] if projection_rule_change else [],
            justification="Governed projection rule update",
            proposed_by=actor,
        )
        approve_projection_rule_proposal(
            pi_proposal["projection_rule_proposal_id"],
            approved_by=actor,
        )
        pi_v = activate_projection_rule_proposal(
            pi_proposal["projection_rule_proposal_id"],
            org_id=org_id,
        )
        result["projection_bundle_version"] = pi_v
        result["resulting_version_pins"] = {"Pi_v": pi_v["projection_bundle_id"]}
        return result

    if action in {"MAP_TO_EXISTING_COA_NODE", "SPLIT_PROJECTION", "REQUEST_MORE_EVIDENCE", "ESCALATE_TO_ADVISOR", "MARK_NOT_PROJECTABLE"}:
        return result

    raise ValueError(f"unsupported_governance_action:{action}")


def run_governed_reprojection_and_resolve_exception(
    *,
    projection_exception_id: str,
    governance_id: str,
    rea_id: str,
    new_coa_code: str,
    amount: float,
    currency: str,
    new_projection_bundle_version: Optional[str] = None,
    new_coa_version: Optional[str] = None,
    new_projection_context_ref: Optional[str] = None,
    new_projection_context_hash: Optional[str] = None,
) -> Dict[str, Any]:
    reprojection = create_governed_reprojection(
        projection_exception_id=projection_exception_id,
        governance_id=governance_id,
        rea_id=rea_id,
        new_coa_code=new_coa_code,
        amount=amount,
        currency=currency,
        new_projection_bundle_version=new_projection_bundle_version,
        new_coa_version=new_coa_version,
        new_projection_context_ref=new_projection_context_ref,
        new_projection_context_hash=new_projection_context_hash,
    )

    status_event = append_projection_exception_status_event(
        projection_exception_id=projection_exception_id,
        status="RESOLVED_BY_GOVERNED_REPROJECTION",
        reason_code="GOVERNED_REPROJECTION_APPLIED",
        event_payload={
            "governed_reprojection_id": reprojection["governed_reprojection_id"],
            "new_projection_result_ref": reprojection["new_projection_result_ref"],
        },
    )

    return {
        "governed_reprojection": reprojection,
        "status_event": status_event,
    }


def activate_from_onboarding_candidates(
    *,
    org_id: str,
    coa_taxonomy_candidate_id: str,
    projection_bundle_candidate_id: str,
    constraint_bundle_candidate_id: str,
    activated_by: str,
) -> Dict[str, Any]:
    """Convenience wrapper used by journey tests and runtime entry points."""
    return activate_onboarding_candidates(
        org_id=org_id,
        coa_taxonomy_candidate_id=coa_taxonomy_candidate_id,
        projection_bundle_candidate_id=projection_bundle_candidate_id,
        constraint_bundle_candidate_id=constraint_bundle_candidate_id,
        activated_by=activated_by,
    )
