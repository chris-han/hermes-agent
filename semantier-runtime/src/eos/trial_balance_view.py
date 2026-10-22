"""TrialBalanceView_t builder — derived view over projection results.

Architecture ref: docs/canonical/architecture.md §22.9.3, §22.9.4, §22.9.6

Doctrine:
- TrialBalanceView_t is derived from stored projection results and pinned COA.
- It is NOT the source of financial truth.
- Initial trust state is always PROJECTION_CANDIDATE.
- Must not query live memory, live LLMs, or unversioned rules.
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso


# ---------------------------------------------------------------------------
# Projection trust state constants (architecture §22.9.4)
# ---------------------------------------------------------------------------

class ProjectionTrustState:
    REA_COMMITTED = "REA_COMMITTED"
    PROJECTION_CANDIDATE = "PROJECTION_CANDIDATE"
    PROJECTION_VALIDATED = "PROJECTION_VALIDATED"
    PROJECTION_WARNING = "PROJECTION_WARNING"
    PROJECTION_BLOCKED = "PROJECTION_BLOCKED"
    PROJECTION_REQUIRES_GOVERNANCE = "PROJECTION_REQUIRES_GOVERNANCE"
    PROJECTION_TRUSTED = "PROJECTION_TRUSTED"
    PROJECTION_EXPORTED = "PROJECTION_EXPORTED"


# Trust states that permit close, export, or trusted representation
_TRUSTED_STATES = {
    ProjectionTrustState.PROJECTION_TRUSTED,
    ProjectionTrustState.PROJECTION_EXPORTED,
}

# Trust states that allow a provisional (non-trusted) projection to exist
_PROVISIONAL_STATES = {
    ProjectionTrustState.PROJECTION_CANDIDATE,
    ProjectionTrustState.PROJECTION_VALIDATED,
    ProjectionTrustState.PROJECTION_WARNING,
}

# Trust states that block close/export until resolved or governed
_BLOCKED_STATES = {
    ProjectionTrustState.PROJECTION_BLOCKED,
    ProjectionTrustState.PROJECTION_REQUIRES_GOVERNANCE,
}


def can_trust_projection(trust_state: str) -> bool:
    """Return True only when the projection may be used as a trusted financial representation."""
    return trust_state in _TRUSTED_STATES


def can_close_period(trust_state: str) -> bool:
    """Return True when the trust state permits period close."""
    return trust_state in _TRUSTED_STATES


def can_export(trust_state: str) -> bool:
    """Return True when the trust state permits inclusion in audit/export evidence."""
    return trust_state == ProjectionTrustState.PROJECTION_TRUSTED or trust_state == ProjectionTrustState.PROJECTION_EXPORTED


def is_blocked(trust_state: str) -> bool:
    """Return True when the trust state blocks close and export."""
    return trust_state in _BLOCKED_STATES


# ---------------------------------------------------------------------------
# TrialBalanceView_t builder
# ---------------------------------------------------------------------------

def build_trial_balance_view(
    *,
    org_id: str,
    period_id: str,
    source_projection_result_ids: List[str],
    coa_version: str,
    projection_bundle_version: str,
    constraint_bundle_version: str,
    accounts: Optional[List[Dict[str, Any]]] = None,
    created_by: str = "system",
) -> Dict[str, Any]:
    """Derive and persist a TrialBalanceView_t.

    Parameters
    ----------
    org_id:
        Organisation identifier.
    period_id:
        Accounting period identifier (e.g. "2024-Q1").
    source_projection_result_ids:
        List of committed projection result IDs that feed this view.
    coa_version:
        Pinned COA version (COA_v).
    projection_bundle_version:
        Pinned projection bundle version (Π_v).
    constraint_bundle_version:
        Pinned constraint bundle version (ConstraintBundle_v).
    accounts:
        Pre-built account balance rows.  If omitted, the builder returns an
        empty accounts list — callers must populate from projection results.
    created_by:
        Actor that triggered the build ("system", "user:<id>", "agent").

    Returns
    -------
    dict
        TrialBalanceView_t with ``content_hash`` and initial trust state
        ``PROJECTION_CANDIDATE``.

    Raises
    ------
    ValueError
        If any required pin is missing.
    """
    if not coa_version:
        raise ValueError("coa_version is required for TrialBalanceView_t")
    if not projection_bundle_version:
        raise ValueError("projection_bundle_version is required for TrialBalanceView_t")

    pins = {
        "coa_version": coa_version,
        "projection_bundle_version": projection_bundle_version,
        "constraint_bundle_version": constraint_bundle_version,
    }

    account_rows = accounts or []

    payload = {
        "org_id": org_id,
        "period_id": period_id,
        "trust_state": ProjectionTrustState.PROJECTION_CANDIDATE,
        "pins": pins,
        "source_projection_result_ids": source_projection_result_ids,
        "accounts": account_rows,
    }

    view_id = f"tbv_{uuid4().hex}"
    row = {
        "trial_balance_view_id": view_id,
        "org_id": org_id,
        "period_id": period_id,
        "trust_state": ProjectionTrustState.PROJECTION_CANDIDATE,
        "pins_json": canonical_json(pins),
        "source_projection_result_ids_json": json.dumps(source_projection_result_ids),
        "accounts_json": json.dumps(account_rows),
        "content_hash": content_hash(payload),
        "created_at": now_iso(),
        "created_by": created_by,
    }

    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO trial_balance_views
              (trial_balance_view_id, org_id, period_id, trust_state, pins_json,
               source_projection_result_ids_json, accounts_json, content_hash,
               created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO trial_balance_views
              (trial_balance_view_id, org_id, period_id, trust_state, pins_json,
               source_projection_result_ids_json, accounts_json, content_hash,
               created_at, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                row["trial_balance_view_id"],
                row["org_id"],
                row["period_id"],
                row["trust_state"],
                row["pins_json"],
                row["source_projection_result_ids_json"],
                row["accounts_json"],
                row["content_hash"],
                row["created_at"],
                row["created_by"],
            ),
        )

    return {
        "trial_balance_view_id": view_id,
        **payload,
        "content_hash": row["content_hash"],
        "created_at": row["created_at"],
        "created_by": row["created_by"],
    }


def get_trial_balance_view(trial_balance_view_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a persisted TrialBalanceView_t by ID."""
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM trial_balance_views WHERE trial_balance_view_id = ?", 'SELECT * FROM trial_balance_views WHERE trial_balance_view_id = %s'),
            (trial_balance_view_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["pins"] = json.loads(result["pins_json"])
    result["source_projection_result_ids"] = json.loads(result["source_projection_result_ids_json"])
    result["accounts"] = json.loads(result["accounts_json"])
    return result
