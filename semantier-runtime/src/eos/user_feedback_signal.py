"""User feedback signal builder — UserFeedbackSignal_t.

Architecture ref: docs/canonical/architecture.md §22.9.12

Generates structured, audience-targeted feedback from validation artifacts.
Feedback must be derived exclusively from recorded validation artifacts and
evidence refs.  Live LLMs must NOT be called.

The feedback surface answers:
- Was the REA fact saved?
- Was the projection generated?
- Can the projection be trusted?
- What validation check failed?
- Why does the failure matter?
- What evidence supports the warning?
- What action can the user take?
- What happens if the user ignores it?
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, List
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso

# Severitiex that require user feedback (info is informational only)
_MATERIAL_SEVERITIES = {"warning", "high", "critical"}

_RECOMMENDED_ACTIONS_BY_TYPE: Dict[str, List[str]] = {
    "direction": [
        "provide_missing_evidence",
        "approve_exception",
        "correct_projection_rule",
        "ignore_with_waiver",
    ],
    "variance": [
        "provide_missing_evidence",
        "approve_exception",
        "request_accountant_review",
        "re-run_projection",
        "ignore_with_waiver",
    ],
    "reconciliation": [
        "provide_missing_evidence",
        "correct_projection_rule",
        "request_accountant_review",
        "approve_exception",
    ],
    "period_close": [
        "re-run_projection",
        "request_accountant_review",
        "correct_projection_rule",
        "approve_exception",
    ],
}

_TITLES_BY_SEVERITY: Dict[str, str] = {
    "warning": "Projection needs review",
    "high": "Projection blocked — accountant review required",
    "critical": "Critical validation failure — close/export blocked",
}

_SUMMARIES_BY_TYPE: Dict[str, str] = {
    "direction": "An account has an unexpected balance direction compared to the normal balance defined in the chart of accounts.",
    "variance": "A material account movement was detected without supporting evidence.",
    "reconciliation": "A required accounting relationship does not balance. This may indicate a missing event, projection error, or incomplete period close.",
    "period_close": "An income-statement account still carries a balance after period close. The period may be incompletely closed.",
}


def build_user_feedback_signals(
    validation_result: Dict[str, Any],
    audience: str = "accountant",
    created_by: str = "system",
) -> List[Dict[str, Any]]:
    """Generate UserFeedbackSignal_t artifacts from a TrialBalanceValidationResult_t.

    Parameters
    ----------
    validation_result:
        TrialBalanceValidationResult_t dict as returned by
        ``run_trial_balance_validation``.
    audience:
        Target audience: ``"owner"``, ``"accountant"``, ``"auditor"``, or
        ``"admin"``.
    created_by:
        Actor that triggered signal generation.

    Returns
    -------
    list[dict]
        One UserFeedbackSignal_t per material failed check.  Empty list if no
        material failures exist.

    Notes
    -----
    This function MUST NOT call live LLMs.  All content is derived from the
    validation result's ``checks`` list.
    """
    if audience not in ("owner", "accountant", "auditor", "admin"):
        raise ValueError(f"Invalid audience: {audience!r}")

    checks = validation_result.get("checks", [])
    validation_id = validation_result.get("validation_id", "")
    org_id = validation_result.get("org_id", "")
    projection_status = validation_result.get("resulting_trust_state", "PROJECTION_CANDIDATE")

    signals: List[Dict[str, Any]] = []

    for check in checks:
        if check.get("status") != "failed":
            continue
        severity = check.get("severity", "info")
        if severity not in _MATERIAL_SEVERITIES:
            continue

        check_id = check.get("check_id", "")
        check_type = check.get("check_type", "")
        affected_accounts = check.get("affected_accounts", [])
        evidence_refs = check.get("evidence_refs", [])

        title = _TITLES_BY_SEVERITY.get(severity, "Projection review required")
        summary = _SUMMARIES_BY_TYPE.get(check_type, "A validation check failed.")
        recommended_actions = _RECOMMENDED_ACTIONS_BY_TYPE.get(check_type, ["request_accountant_review"])

        feedback_payload = {
            "validation_id": validation_id,
            "check_id": check_id,
            "audience": audience,
            "severity": severity,
            "title": title,
            "plain_language_summary": summary,
            "technical_reason": str(check.get("observed_value", {})),
            "fact_status": "rea_committed",
            "projection_status": projection_status,
            "affected_accounts": affected_accounts,
            "evidence_refs": evidence_refs,
            "recommended_actions": recommended_actions,
            "user_decision_required": severity in ("high", "critical"),
        }

        feedback_id = f"tbf_{uuid4().hex}"
        row = {
            "feedback_id": feedback_id,
            "validation_id": validation_id,
            "check_id": check_id,
            "org_id": org_id,
            "audience": audience,
            "severity": severity,
            "feedback_json": canonical_json(feedback_payload),
            "content_hash": content_hash(feedback_payload),
            "created_at": now_iso(),
        }

        with get_conn() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO trial_balance_user_feedback_signals
                  (feedback_id, validation_id, check_id, org_id, audience,
                   severity, feedback_json, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO trial_balance_user_feedback_signals
                  (feedback_id, validation_id, check_id, org_id, audience,
                   severity, feedback_json, content_hash, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    row["feedback_id"],
                    row["validation_id"],
                    row["check_id"],
                    row["org_id"],
                    row["audience"],
                    row["severity"],
                    row["feedback_json"],
                    row["content_hash"],
                    row["created_at"],
                ),
            )

        signals.append({
            "feedback_id": feedback_id,
            **feedback_payload,
            "content_hash": row["content_hash"],
            "created_at": row["created_at"],
        })

    return signals


def get_feedback_signals_for_validation(validation_id: str) -> List[Dict[str, Any]]:
    """Retrieve all UserFeedbackSignal_t artifacts for a validation run."""
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("SELECT * FROM trial_balance_user_feedback_signals WHERE validation_id = ? ORDER BY created_at", 'SELECT * FROM trial_balance_user_feedback_signals WHERE validation_id = %s ORDER BY created_at'),
            (validation_id,),
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["feedback"] = json.loads(d["feedback_json"])
        results.append(d)
    return results
