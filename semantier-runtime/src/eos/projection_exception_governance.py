"""Append-only ProjectionExceptionGovernance_t persistence helpers."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso


def _load_row(governance_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM projection_exception_governance WHERE governance_id = ?", 'SELECT * FROM projection_exception_governance WHERE governance_id = %s'),
            (governance_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["governance"] = json.loads(result["governance_json"])
    return result


def _insert_governance_record(
    *,
    projection_exception_id: str,
    governance_payload: Dict[str, Any],
    status: str,
    approved_at: str | None = None,
    activated_at: str | None = None,
) -> Dict[str, Any]:
    governance_id = f"pegov_{uuid4().hex}"
    created_at = now_iso()
    governance_json = canonical_json(governance_payload)
    row = {
        "governance_id": governance_id,
        "projection_exception_id": projection_exception_id,
        "governance_json": governance_json,
        "content_hash": content_hash(governance_payload),
        "status": status,
        "created_at": created_at,
        "approved_at": approved_at,
        "activated_at": activated_at,
        "governance": governance_payload,
    }
    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO projection_exception_governance
              (governance_id, projection_exception_id, governance_json, content_hash,
               status, created_at, approved_at, activated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO projection_exception_governance
              (governance_id, projection_exception_id, governance_json, content_hash,
               status, created_at, approved_at, activated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                governance_id,
                projection_exception_id,
                governance_json,
                row["content_hash"],
                status,
                created_at,
                approved_at,
                activated_at,
            ),
        )
    return row


def create_projection_exception_governance(
    *,
    projection_exception_id: str,
    allowed_actions: List[Dict[str, Any]],
    selected_action: Dict[str, Any],
    justification: str,
    validation_results: Optional[Dict[str, Any]] = None,
    replay_policy: Optional[Dict[str, Any]] = None,
    proposed_by: Optional[str] = None,
) -> Dict[str, Any]:
    if not justification.strip():
        raise ValueError("projection_exception_governance_requires_justification")
    governance_payload = {
        "projection_exception_id": projection_exception_id,
        "allowed_actions": allowed_actions,
        "selected_action": selected_action,
        "rejected_actions": [
            action for action in allowed_actions if action != selected_action
        ],
        "justification": justification,
        "validation_results": validation_results or {},
        "approval": None,
        "activation": None,
        "replay_policy": replay_policy or {"mode": "reproject_after_activation"},
        "proposed_by": proposed_by,
    }
    return _insert_governance_record(
        projection_exception_id=projection_exception_id,
        governance_payload=governance_payload,
        status="PENDING_APPROVAL",
    )


def approve_projection_exception_governance(
    governance_id: str,
    *,
    approved_by: str,
) -> Dict[str, Any]:
    current = _load_row(governance_id)
    if current is None:
        raise ValueError(f"projection_exception_governance_not_found:{governance_id}")
    if current["status"] != "PENDING_APPROVAL":
        raise ValueError(
            f"projection_exception_governance_not_pending:{current['status']}"
        )
    approved_at = now_iso()
    governance_payload = dict(current["governance"])
    governance_payload["approval"] = {
        "approved_by": approved_by,
        "approved_at": approved_at,
    }
    return _insert_governance_record(
        projection_exception_id=current["projection_exception_id"],
        governance_payload=governance_payload,
        status="APPROVED",
        approved_at=approved_at,
    )


def activate_projection_exception_governance(
    governance_id: str,
    *,
    activated_by: str,
) -> Dict[str, Any]:
    current = _load_row(governance_id)
    if current is None:
        raise ValueError(f"projection_exception_governance_not_found:{governance_id}")
    if current["status"] != "APPROVED":
        raise ValueError("projection_exception_governance_requires_approval")
    activated_at = now_iso()
    governance_payload = dict(current["governance"])
    governance_payload["activation"] = {
        "activated_by": activated_by,
        "activated_at": activated_at,
    }
    return _insert_governance_record(
        projection_exception_id=current["projection_exception_id"],
        governance_payload=governance_payload,
        status="ACTIVATED",
        approved_at=current["approved_at"],
        activated_at=activated_at,
    )


def get_projection_exception_governance(governance_id: str) -> Optional[Dict[str, Any]]:
    return _load_row(governance_id)


def get_latest_projection_exception_governance(
    projection_exception_id: str,
) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM projection_exception_governance
            WHERE projection_exception_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """, """
            SELECT * FROM projection_exception_governance
            WHERE projection_exception_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """),
            (projection_exception_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["governance"] = json.loads(result["governance_json"])
    return result
