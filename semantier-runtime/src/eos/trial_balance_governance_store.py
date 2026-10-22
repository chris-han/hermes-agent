"""Trial Balance Governance Decision Store — TrialBalanceGovernanceDecision_t.

Architecture ref: docs/canonical/architecture.md §22.9.14

Governance decisions are append-only versioned artifacts.  They must not
mutate the original validation artifact.

Decision enum:
    ACCEPT_VALIDATION_RESULT
    REJECT_VALIDATION_RESULT
    WAIVE_CHECK_WITH_REASON
    ESCALATE_TO_ACCOUNTANT
    ESCALATE_TO_EXTERNAL_AUDITOR
    REQUEST_MORE_EVIDENCE
    REPROJECT_UNDER_APPROVED_BUNDLE
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso

DECISION_ENUM = {
    "ACCEPT_VALIDATION_RESULT",
    "REJECT_VALIDATION_RESULT",
    "WAIVE_CHECK_WITH_REASON",
    "ESCALATE_TO_ACCOUNTANT",
    "ESCALATE_TO_EXTERNAL_AUDITOR",
    "REQUEST_MORE_EVIDENCE",
    "REPROJECT_UNDER_APPROVED_BUNDLE",
}


def evaluate_filing_readiness(*, safeguard: Dict[str, Any]) -> Dict[str, Any]:
    if safeguard.get("status") != "OK":
        return {"status": "BLOCKED", "reason": safeguard.get("reason")}
    return {"status": "OK", "reason": "no_blocker"}


def create_trial_balance_governance_decision(
    *,
    validation_id: str,
    check_id: str,
    decision_enum: str,
    actor: str,
    justification_ref: Optional[str] = None,
    evidence_refs: Optional[List[str]] = None,
    approved_by: Optional[str] = None,
    approved_at: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Create and persist a TrialBalanceGovernanceDecision_t.

    Parameters
    ----------
    validation_id:
        The validation run this decision governs.
    check_id:
        The specific check within that validation run.
    decision_enum:
        One of the seven governance decision values.
    actor:
        Who made the decision: ``"user:<id>"``, ``"agent"``, or ``"system"``.
    justification_ref:
        Optional reference to a TrialBalanceJustification_t ID.
    evidence_refs:
        Additional artifact IDs supporting this decision.
    approved_by:
        Accountant or auditor who approved (required for WAIVE and ESCALATE).
    approved_at:
        ISO timestamp of approval.
    notes:
        Free-text notes (stored but not used for semantic authority).

    Returns
    -------
    dict
        TrialBalanceGovernanceDecision_t with content_hash.

    Raises
    ------
    ValueError
        If ``decision_enum`` is not a recognised value.
    """
    if decision_enum not in DECISION_ENUM:
        raise ValueError(f"Invalid decision_enum: {decision_enum!r}. Must be one of {sorted(DECISION_ENUM)}")

    ev = evidence_refs or []
    payload = {
        "validation_id": validation_id,
        "check_id": check_id,
        "decision_enum": decision_enum,
        "actor": actor,
        "justification_ref": justification_ref,
        "evidence_refs": ev,
        "approved_by": approved_by,
        "approved_at": approved_at,
        "notes": notes,
    }

    decision_id = f"tbgd_{uuid4().hex}"
    row = {
        "decision_id": decision_id,
        "validation_id": validation_id,
        "check_id": check_id,
        "decision_enum": decision_enum,
        "actor": actor,
        "justification_ref": justification_ref,
        "evidence_refs_json": json.dumps(ev),
        "approved_by": approved_by,
        "approved_at": approved_at,
        "notes": notes,
        "content_hash": content_hash(payload),
        "created_at": now_iso(),
    }

    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO trial_balance_governance_decisions
              (decision_id, validation_id, check_id, decision_enum, actor,
               justification_ref, evidence_refs_json, approved_by, approved_at,
               notes, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO trial_balance_governance_decisions
              (decision_id, validation_id, check_id, decision_enum, actor,
               justification_ref, evidence_refs_json, approved_by, approved_at,
               notes, content_hash, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                row["decision_id"],
                row["validation_id"],
                row["check_id"],
                row["decision_enum"],
                row["actor"],
                row["justification_ref"],
                row["evidence_refs_json"],
                row["approved_by"],
                row["approved_at"],
                row["notes"],
                row["content_hash"],
                row["created_at"],
            ),
        )

    return {
        "decision_id": decision_id,
        **payload,
        "content_hash": row["content_hash"],
        "created_at": row["created_at"],
    }


def get_governance_decision(decision_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a TrialBalanceGovernanceDecision_t by ID."""
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM trial_balance_governance_decisions WHERE decision_id = ?", 'SELECT * FROM trial_balance_governance_decisions WHERE decision_id = %s'),
            (decision_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["evidence_refs"] = json.loads(d["evidence_refs_json"])
    return d


def list_governance_decisions_for_validation(validation_id: str) -> List[Dict[str, Any]]:
    """List all governance decisions for a validation run (append-only; never deleted)."""
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("SELECT * FROM trial_balance_governance_decisions WHERE validation_id = ? ORDER BY created_at", 'SELECT * FROM trial_balance_governance_decisions WHERE validation_id = %s ORDER BY created_at'),
            (validation_id,),
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["evidence_refs"] = json.loads(d["evidence_refs_json"])
        results.append(d)
    return results
