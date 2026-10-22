"""Trial Balance Justification Store — TrialBalanceJustification_t.

Architecture ref: docs/canonical/architecture.md §22.9.13

Every failed, escalated, or waived validation check must produce a
justification object.  Justifications are append-only governance artifacts
bound to the validation result and replayable under the same semantic versions.

A waiver is NOT deletion — it is an append-only record.
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso

_VALID_DECISIONS = {"accept", "reject", "escalate", "waive"}


def create_trial_balance_justification(
    *,
    validation_id: str,
    check_id: str,
    claim: str,
    evidence_refs: List[str],
    reasoning: Dict[str, Any],
    decision_status: str,
    actor: str,
    approved_by: Optional[str] = None,
    approved_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Create and persist a TrialBalanceJustification_t.

    Parameters
    ----------
    validation_id:
        The validation run this justification is bound to.
    check_id:
        The specific check within that validation run.
    claim:
        Human-readable claim being made to justify the decision.
    evidence_refs:
        List of artifact IDs (rea_event_id, projection_result_id, etc.)
        supporting this justification.
    reasoning:
        Dict containing at minimum ``rule_applied``, ``observed_value``,
        ``expected_value``, and optionally ``exception_basis``.
    decision_status:
        One of: ``"accept"``, ``"reject"``, ``"escalate"``, ``"waive"``.
    actor:
        Who made the decision: ``"user"``, ``"agent"``, or ``"system"``.
    approved_by:
        Human approver for escalated/waived decisions (optional).
    approved_at:
        ISO timestamp of approval (optional).

    Returns
    -------
    dict
        TrialBalanceJustification_t with content_hash.

    Raises
    ------
    ValueError
        If ``decision_status`` is not a valid value.
    """
    if decision_status not in _VALID_DECISIONS:
        raise ValueError(f"Invalid decision_status: {decision_status!r}. Must be one of {_VALID_DECISIONS}")

    payload = {
        "validation_id": validation_id,
        "check_id": check_id,
        "claim": claim,
        "evidence_refs": evidence_refs,
        "reasoning": reasoning,
        "decision": {
            "status": decision_status,
            "actor": actor,
            "approved_by": approved_by,
            "approved_at": approved_at,
        },
    }

    justification_id = f"tbj_{uuid4().hex}"
    row = {
        "justification_id": justification_id,
        "validation_id": validation_id,
        "check_id": check_id,
        "claim": claim,
        "evidence_refs_json": json.dumps(evidence_refs),
        "reasoning_json": canonical_json(reasoning),
        "decision_status": decision_status,
        "actor": actor,
        "approved_by": approved_by,
        "approved_at": approved_at,
        "content_hash": content_hash(payload),
        "created_at": now_iso(),
    }

    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO trial_balance_justifications
              (justification_id, validation_id, check_id, claim, evidence_refs_json,
               reasoning_json, decision_status, actor, approved_by, approved_at,
               content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO trial_balance_justifications
              (justification_id, validation_id, check_id, claim, evidence_refs_json,
               reasoning_json, decision_status, actor, approved_by, approved_at,
               content_hash, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                row["justification_id"],
                row["validation_id"],
                row["check_id"],
                row["claim"],
                row["evidence_refs_json"],
                row["reasoning_json"],
                row["decision_status"],
                row["actor"],
                row["approved_by"],
                row["approved_at"],
                row["content_hash"],
                row["created_at"],
            ),
        )

    return {
        "justification_id": justification_id,
        **payload,
        "content_hash": row["content_hash"],
        "created_at": row["created_at"],
    }


def get_trial_balance_justification(justification_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a TrialBalanceJustification_t by ID."""
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM trial_balance_justifications WHERE justification_id = ?", 'SELECT * FROM trial_balance_justifications WHERE justification_id = %s'),
            (justification_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["evidence_refs"] = json.loads(d["evidence_refs_json"])
    d["reasoning"] = json.loads(d["reasoning_json"])
    return d


def list_justifications_for_validation(validation_id: str) -> List[Dict[str, Any]]:
    """List all justifications for a given validation run (append-only; never deleted)."""
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("SELECT * FROM trial_balance_justifications WHERE validation_id = ? ORDER BY created_at", 'SELECT * FROM trial_balance_justifications WHERE validation_id = %s ORDER BY created_at'),
            (validation_id,),
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["evidence_refs"] = json.loads(d["evidence_refs_json"])
        d["reasoning"] = json.loads(d["reasoning_json"])
        results.append(d)
    return results
