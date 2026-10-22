"""Trial balance validation engine — TrialBalanceValidationContract_v executor.

Architecture ref: docs/canonical/architecture.md §22.9.7, §22.9.16

Orchestrates the four sub-validators and produces the canonical
TrialBalanceValidationResult_t artifact.

Gate behavior (§22.9.16):
- ``critical`` failure → PROJECTION_BLOCKED
- ``high`` failure → PROJECTION_REQUIRES_GOVERNANCE
- ``warning`` failure → PROJECTION_WARNING
- all passed → PROJECTION_VALIDATED

REA facts are never touched by this module.  The trust state applies to the
derived TrialBalanceView_t, not to REA events.
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso
from eos.direction_validator import run_direction_validation
from eos.period_close_validator import run_period_close_validation
from eos.reconciliation_validator import run_reconciliation_validation
from eos.trial_balance_view import ProjectionTrustState
from eos.variance_validator import run_variance_validation

# Severity ordering (highest = worst)
_SEVERITY_ORDER = {"critical": 4, "high": 3, "warning": 2, "info": 1}


def _worst_severity(checks: List[Dict[str, Any]]) -> str:
    worst = "info"
    for c in checks:
        if c.get("status") == "failed":
            sev = c.get("severity", "info")
            if _SEVERITY_ORDER.get(sev, 0) > _SEVERITY_ORDER.get(worst, 0):
                worst = sev
    return worst


def _determine_trust_state(checks: List[Dict[str, Any]]) -> str:
    """Map aggregate check results to a ProjectionTrustState."""
    worst = _worst_severity(checks)
    if worst == "critical":
        return ProjectionTrustState.PROJECTION_BLOCKED
    if worst == "high":
        return ProjectionTrustState.PROJECTION_REQUIRES_GOVERNANCE
    if worst == "warning":
        return ProjectionTrustState.PROJECTION_WARNING
    return ProjectionTrustState.PROJECTION_VALIDATED


def _determine_status(trust_state: str) -> str:
    if trust_state == ProjectionTrustState.PROJECTION_BLOCKED:
        return "failed"
    if trust_state == ProjectionTrustState.PROJECTION_REQUIRES_GOVERNANCE:
        return "requires_governance"
    if trust_state == ProjectionTrustState.PROJECTION_WARNING:
        return "warning"
    return "passed"


def run_trial_balance_validation(
    *,
    trial_balance_view: Dict[str, Any],
    pins: Dict[str, Any],
    coa_entries: Optional[Dict[str, Dict[str, Any]]] = None,
    prior_period_accounts: Optional[List[Dict[str, Any]]] = None,
    variance_evidence_refs: Optional[Dict[str, List[str]]] = None,
    period_close_state: str = "open",
    retained_earnings_account_code: Optional[str] = None,
    distributions_amount: float = 0.0,
    created_by: str = "system",
) -> Dict[str, Any]:
    """Execute TrialBalanceValidationContract_v over a TrialBalanceView_t.

    Parameters
    ----------
    trial_balance_view:
        TrialBalanceView_t dict as returned by ``build_trial_balance_view``.
    pins:
        Explicit semantic version pins for this validation run.  Must include
        at minimum ``ontology_version``, ``coa_version``, and
        ``validation_contract_version``.
    coa_entries:
        Mapping from account_code to COA entry dict.  Used by direction
        validator.  If omitted, direction checks are skipped.
    prior_period_accounts:
        Prior-period account balance rows for variance comparison.  If omitted,
        variance checks are skipped.
    variance_evidence_refs:
        Per-account evidence refs that explain large movements.
    period_close_state:
        ``"open"`` or ``"closed"``.
    retained_earnings_account_code:
        Account code of the retained earnings account.
    distributions_amount:
        Total declared distributions for the period.
    created_by:
        Actor that triggered the validation run.

    Returns
    -------
    dict
        TrialBalanceValidationResult_t with content_hash, status, and
        resulting_trust_state.
    """
    accounts = trial_balance_view.get("accounts", [])
    org_id = trial_balance_view.get("org_id", "")
    period_id = trial_balance_view.get("period_id", "")
    view_id = trial_balance_view.get("trial_balance_view_id", "")

    all_checks: List[Dict[str, Any]] = []

    # --- Direction validation ---
    if coa_entries:
        direction_checks = run_direction_validation(accounts, coa_entries, pins)
        all_checks.extend(direction_checks)

    # --- Variance validation ---
    variance_checks = run_variance_validation(
        accounts,
        prior_period_accounts,
        variance_evidence_refs,
        pins,
    )
    all_checks.extend(variance_checks)

    # --- Reconciliation validation ---
    reconciliation_checks = run_reconciliation_validation(
        accounts,
        period_close_state,
        retained_earnings_account_code,
        distributions_amount,
        pins,
    )
    all_checks.extend(reconciliation_checks)

    # --- Period close validation ---
    period_close_checks = run_period_close_validation(accounts, period_close_state, pins)
    all_checks.extend(period_close_checks)

    # --- Aggregate ---
    trust_state = _determine_trust_state(all_checks)
    status = _determine_status(trust_state)

    validation_id = f"tbvr_{uuid4().hex}"

    payload = {
        "org_id": org_id,
        "period_id": period_id,
        "trial_balance_view_id": view_id,
        "status": status,
        "resulting_trust_state": trust_state,
        "pins": pins,
        "checks": all_checks,
    }

    row = {
        "validation_id": validation_id,
        "org_id": org_id,
        "period_id": period_id,
        "trial_balance_view_id": view_id,
        "status": status,
        "resulting_trust_state": trust_state,
        "pins_json": canonical_json(pins),
        "checks_json": json.dumps(all_checks),
        "content_hash": content_hash(payload),
        "created_at": now_iso(),
        "created_by": created_by,
    }

    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO trial_balance_validation_results
              (validation_id, org_id, period_id, trial_balance_view_id, status,
               resulting_trust_state, pins_json, checks_json, content_hash,
               created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO trial_balance_validation_results
              (validation_id, org_id, period_id, trial_balance_view_id, status,
               resulting_trust_state, pins_json, checks_json, content_hash,
               created_at, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                row["validation_id"],
                row["org_id"],
                row["period_id"],
                row["trial_balance_view_id"],
                row["status"],
                row["resulting_trust_state"],
                row["pins_json"],
                row["checks_json"],
                row["content_hash"],
                row["created_at"],
                row["created_by"],
            ),
        )

    return {
        "validation_id": validation_id,
        **payload,
        "content_hash": row["content_hash"],
        "created_at": row["created_at"],
        "created_by": row["created_by"],
    }


def get_trial_balance_validation_result(validation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a persisted TrialBalanceValidationResult_t by ID."""
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM trial_balance_validation_results WHERE validation_id = ?", 'SELECT * FROM trial_balance_validation_results WHERE validation_id = %s'),
            (validation_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["pins"] = json.loads(result["pins_json"])
    result["checks"] = json.loads(result["checks_json"])
    return result
