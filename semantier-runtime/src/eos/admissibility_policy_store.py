"""Governed policy store for retrieval admissibility decisions."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Dict, Optional

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso


DEFAULT_POLICY = {
    "policy_version": "policy_v1",
    "rules": [
        {"id": "active_source_only", "reject_reason": "REJECT_NOT_IN_ACTIVE_K_V"},
        {"id": "approved_source_only", "reject_reason": "REJECT_NOT_APPROVED"},
        {"id": "domain_match", "reject_reason": "REJECT_DOMAIN_MISMATCH"},
        {"id": "governance_state", "reject_reason": "REJECT_GOVERNANCE_STATE"},
    ],
}


def ensure_default_admissibility_policy() -> Dict[str, Any]:
    existing = get_active_admissibility_policy()
    if existing is not None:
        return existing
    return create_admissibility_policy(DEFAULT_POLICY, status="ACTIVE")


def create_admissibility_policy(policy: Dict[str, Any], *, status: str = "DRAFT") -> Dict[str, Any]:
    policy_version = str(policy["policy_version"])
    created_at = now_iso()
    policy_json = canonical_json(policy)
    row = {
        "policy_version": policy_version,
        "policy_json": policy_json,
        "content_hash": content_hash(policy),
        "status": status,
        "created_at": created_at,
        "activated_at": created_at if status == "ACTIVE" else None,
    }
    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT OR REPLACE INTO adapter_admissibility_policies (
              policy_version, policy_json, content_hash, status, created_at, activated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO adapter_admissibility_policies (
              policy_version, policy_json, content_hash, status, created_at, activated_at
            ) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (policy_version) DO UPDATE SET policy_json=excluded.policy_json, content_hash=excluded.content_hash, status=excluded.status, created_at=excluded.created_at, activated_at=excluded.activated_at"""),
            (
                row["policy_version"],
                row["policy_json"],
                row["content_hash"],
                row["status"],
                row["created_at"],
                row["activated_at"],
            ),
        )
    return get_admissibility_policy(policy_version) or row


def get_admissibility_policy(policy_version: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM adapter_admissibility_policies WHERE policy_version = ?", 'SELECT * FROM adapter_admissibility_policies WHERE policy_version = %s'),
            (policy_version,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["policy"] = json.loads(result.pop("policy_json"))
    return result


def get_active_admissibility_policy() -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT * FROM adapter_admissibility_policies
            WHERE status = 'ACTIVE'
            ORDER BY activated_at DESC, created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["policy"] = json.loads(result.pop("policy_json"))
    return result
