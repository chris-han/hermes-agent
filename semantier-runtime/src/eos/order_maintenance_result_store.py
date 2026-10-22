"""OrderMaintenanceResult_t store.

Architecture ref: docs/canonical/architecture.md §22.9.15

Append-only store for OrderMaintenanceResult_t artifacts emitted by
SemanticTierAuthorityContract_t runner.
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso

_VALID_STATUSES = {
    "order_created",
    "order_preserved",
    "order_repaired",
    "order_escalated",
    "order_failed",
}


def store_order_maintenance_result(
    *,
    org_id: str,
    period_id: str,
    verification_contract_id: str,
    status: str,
    trust_state: str,
    authority_trace: List[Dict[str, Any]],
    failed_assertions: List[Dict[str, Any]],
    result_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist an OrderMaintenanceResult_t.

    Parameters
    ----------
    org_id:
        Organisation identifier.
    period_id:
        Accounting period identifier.
    verification_contract_id:
        The SemanticTierAuthorityContract_t contract ID this result belongs to.
    status:
        One of: ``order_created``, ``order_preserved``, ``order_repaired``,
        ``order_escalated``, ``order_failed``.
    trust_state:
        The effective ProjectionTrustState at the time of the contract run.
    authority_trace:
        List of dicts recording each tier's application status.
    failed_assertions:
        List of assertion failure dicts.
    result_payload:
        Optional extra payload dict to store alongside the structured fields.

    Returns
    -------
    dict
        OrderMaintenanceResult_t with content_hash.

    Raises
    ------
    ValueError
        If ``status`` is not a valid value.
    """
    if status not in _VALID_STATUSES:
        raise ValueError(f"Invalid status: {status!r}. Must be one of {sorted(_VALID_STATUSES)}")

    payload = {
        "org_id": org_id,
        "period_id": period_id,
        "verification_contract_id": verification_contract_id,
        "status": status,
        "trust_state": trust_state,
        "authority_trace": authority_trace,
        "failed_assertions": failed_assertions,
        **(result_payload or {}),
    }

    result_id = f"omr_{uuid4().hex}"
    row = {
        "result_id": result_id,
        "org_id": org_id,
        "period_id": period_id,
        "verification_contract_id": verification_contract_id,
        "status": status,
        "trust_state": trust_state,
        "authority_trace_json": json.dumps(authority_trace),
        "failed_assertions_json": json.dumps(failed_assertions),
        "result_json": canonical_json(payload),
        "content_hash": content_hash(payload),
        "created_at": now_iso(),
    }

    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO order_maintenance_results
              (result_id, org_id, period_id, verification_contract_id, status,
               trust_state, authority_trace_json, failed_assertions_json,
               result_json, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO order_maintenance_results
              (result_id, org_id, period_id, verification_contract_id, status,
               trust_state, authority_trace_json, failed_assertions_json,
               result_json, content_hash, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                row["result_id"],
                row["org_id"],
                row["period_id"],
                row["verification_contract_id"],
                row["status"],
                row["trust_state"],
                row["authority_trace_json"],
                row["failed_assertions_json"],
                row["result_json"],
                row["content_hash"],
                row["created_at"],
            ),
        )

    return {
        "result_id": result_id,
        **payload,
        "content_hash": row["content_hash"],
        "created_at": row["created_at"],
    }


def get_order_maintenance_result(result_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve an OrderMaintenanceResult_t by ID."""
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM order_maintenance_results WHERE result_id = ?", 'SELECT * FROM order_maintenance_results WHERE result_id = %s'),
            (result_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["authority_trace"] = json.loads(d["authority_trace_json"])
    d["failed_assertions"] = json.loads(d["failed_assertions_json"])
    return d


def list_order_maintenance_results_for_period(org_id: str, period_id: str) -> List[Dict[str, Any]]:
    """List all OrderMaintenanceResult_t records for an org/period (oldest first)."""
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("SELECT * FROM order_maintenance_results WHERE org_id = ? AND period_id = ? ORDER BY created_at", 'SELECT * FROM order_maintenance_results WHERE org_id = %s AND period_id = %s ORDER BY created_at'),
            (org_id, period_id),
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["authority_trace"] = json.loads(d["authority_trace_json"])
        d["failed_assertions"] = json.loads(d["failed_assertions_json"])
        results.append(d)
    return results
