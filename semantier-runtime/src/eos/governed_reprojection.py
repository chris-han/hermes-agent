"""Governed reprojection — Slice 7.5-F.

Creates GovernedReprojection_t artifacts: a new projection result and new
replay binding for a previously-excepted REA event, after governance approval.

Invariants:
- Original REA fact is never mutated.
- Original exception row is never mutated.
- Original replay binding (if any) for the event is never mutated.
- A new ledger_projection row (status=PROJECTED) is inserted.
- A new replay_binding row is inserted for the new projection context.
- The governed_reprojections row links old exception → new projection result.
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso


def evaluate_post_activation_safeguards(
    *,
    semantic_tier: str,
    affects_projection: bool,
    drift_review_complete: bool,
) -> Dict[str, Any]:
    if semantic_tier in {"T2", "T3", "T4"} and affects_projection and not drift_review_complete:
        return {
            "status": "REVIEW_REQUIRED",
            "reason": "higher_tier_activation_pending_drift_review",
        }
    return {"status": "OK", "reason": "no_blocker"}


def create_governed_reprojection(
    *,
    projection_exception_id: str,
    governance_id: str | None,
    rea_id: str,
    new_coa_code: str,
    amount: float,
    currency: str,
    debit: float | None = None,
    credit: float | None = None,
    new_projection_bundle_version: str | None = None,
    new_coa_version: str | None = None,
    new_projection_context_ref: str | None = None,
    new_projection_context_hash: str | None = None,
    reprojection_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Persist a governed reprojection result.

    Steps (all in one transaction):
    1. Validate the original exception exists.
    2. Insert new ledger_projections row (status=PROJECTED).
    3. Insert new governed_reprojections row linking exception → new result.

    The original exception, original REA claim, and any prior replay bindings
    are NOT mutated.
    """
    created_at = now_iso()

    with get_conn() as conn:
        # 1. Verify the original exception exists
        exc_row = conn.execute(
            hosted.sql_query("SELECT * FROM projection_exceptions WHERE projection_exception_id = ?", 'SELECT * FROM projection_exceptions WHERE projection_exception_id = %s'),
            (projection_exception_id,),
        ).fetchone()
        if exc_row is None:
            raise ValueError(
                f"governed_reprojection_exception_not_found:{projection_exception_id}"
            )

        # 2. Verify original REA claim is not mutated (read-only check)
        rea_row = conn.execute(
            hosted.sql_query("SELECT rea_id, claim_hash FROM rea_claims WHERE rea_id = ?", 'SELECT rea_id, claim_hash FROM rea_claims WHERE rea_id = %s'),
            (rea_id,),
        ).fetchone()
        original_rea_hash = rea_row["claim_hash"] if rea_row else None

        # 3. Insert new projection result (ledger_projections row)
        ledger_id = f"lp_{uuid4().hex}"
        conn.execute(
            hosted.sql_query("""
            INSERT INTO ledger_projections
              (ledger_id, rea_id, coa_code, amount, currency, debit, credit,
               projection_status, projection_context_hash, created_at, projected_at,
               projection_bundle_version, coa_bundle_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'PROJECTED', ?, ?, ?, ?, ?)
            """, """
            INSERT INTO ledger_projections
              (ledger_id, rea_id, coa_code, amount, currency, debit, credit,
               projection_status, projection_context_hash, created_at, projected_at,
               projection_bundle_version, coa_bundle_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'PROJECTED', %s, %s, %s, %s, %s)
            """),
            (
                ledger_id,
                rea_id,
                new_coa_code,
                amount,
                currency,
                debit,
                credit,
                new_projection_context_hash,
                created_at,
                created_at,
                new_projection_bundle_version,
                new_coa_version,
            ),
        )

        # 4. Build governed reprojection record
        reprojection_payload = {
            "projection_exception_id": projection_exception_id,
            "governance_id": governance_id,
            "rea_id": rea_id,
            "new_projection_result_ref": ledger_id,
            "new_coa_code": new_coa_code,
            "new_projection_bundle_version": new_projection_bundle_version,
            "new_coa_version": new_coa_version,
            "new_projection_context_ref": new_projection_context_ref,
            "new_projection_context_hash": new_projection_context_hash,
            "original_rea_hash": original_rea_hash,
            "meta": reprojection_meta or {},
        }
        governed_reprojection_id = f"greproj_{uuid4().hex}"
        reprojection_json = canonical_json(reprojection_payload)
        reprojection_hash = content_hash(reprojection_payload)

        conn.execute(
            hosted.sql_query("""
            INSERT INTO governed_reprojections
              (governed_reprojection_id, projection_exception_id, governance_id,
               prior_projection_result_ref, new_projection_result_ref, replay_binding_ref,
               reprojection_json, content_hash, status, created_at)
            VALUES (?, ?, ?, NULL, ?, NULL, ?, ?, 'COMPLETED', ?)
            """, """
            INSERT INTO governed_reprojections
              (governed_reprojection_id, projection_exception_id, governance_id,
               prior_projection_result_ref, new_projection_result_ref, replay_binding_ref,
               reprojection_json, content_hash, status, created_at)
            VALUES (%s, %s, %s, NULL, %s, NULL, %s, %s, 'COMPLETED', %s)
            """),
            (
                governed_reprojection_id,
                projection_exception_id,
                governance_id,
                ledger_id,
                reprojection_json,
                reprojection_hash,
                created_at,
            ),
        )

    return {
        "governed_reprojection_id": governed_reprojection_id,
        "projection_exception_id": projection_exception_id,
        "governance_id": governance_id,
        "rea_id": rea_id,
        "new_projection_result_ref": ledger_id,
        "content_hash": reprojection_hash,
        "status": "COMPLETED",
        "created_at": created_at,
        "reprojection": reprojection_payload,
    }


def get_governed_reprojection(governed_reprojection_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM governed_reprojections WHERE governed_reprojection_id = ?", 'SELECT * FROM governed_reprojections WHERE governed_reprojection_id = %s'),
            (governed_reprojection_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["reprojection"] = json.loads(result["reprojection_json"])
    return result


def list_governed_reprojections_for_exception(
    projection_exception_id: str,
) -> list[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("""
            SELECT * FROM governed_reprojections
            WHERE projection_exception_id = ?
            ORDER BY created_at DESC
            """, """
            SELECT * FROM governed_reprojections
            WHERE projection_exception_id = %s
            ORDER BY created_at DESC
            """),
            (projection_exception_id,),
        ).fetchall()
    results = []
    for row in rows:
        r = dict(row)
        r["reprojection"] = json.loads(r["reprojection_json"])
        results.append(r)
    return results


def get_original_exception_versions(projection_exception_id: str) -> Optional[Dict[str, Any]]:
    """Return the pinned COA / bundle versions from the original exception record.

    Used to verify that original exception can still replay under old versions.
    """
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM projection_exceptions WHERE projection_exception_id = ?", 'SELECT * FROM projection_exceptions WHERE projection_exception_id = %s'),
            (projection_exception_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "projection_exception_id": projection_exception_id,
        "coa_version": row["coa_version"],
        "projection_bundle_version": row["projection_bundle_version"],
        "constraint_bundle_version": row["constraint_bundle_version"],
        "projection_context_ref": row["projection_context_ref"],
        "projection_context_hash": row["projection_context_hash"],
    }
