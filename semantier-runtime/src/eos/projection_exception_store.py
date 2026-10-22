"""First-class ProjectionException_t persistence."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
import re
from typing import Any, Dict, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso


def normalize_projection_exception_payload(exception_payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(exception_payload)
    reason = normalized.get("reason")
    if isinstance(reason, str):
        match = re.fullmatch(
            r"no_coa_code_for_resource_type_(?P<resource_type>[^:]+):scope=(?P<scope>.+)",
            reason,
        )
        if match:
            normalized["reason"] = f"resource_type_not_in_chart:{match.group('resource_type')}"
            normalized["raw_reason"] = reason
            normalized["projection_scope"] = match.group("scope")
    return normalized


def create_projection_exception(
    *,
    rea_id: str,
    justification_id: str,
    projection_context_ref: str | None,
    projection_context_hash: str | None,
    coa_version: str | None,
    projection_bundle_version: str | None,
    constraint_bundle_version: str | None,
    status: str,
    exception_payload: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_payload = normalize_projection_exception_payload(exception_payload)
    row = {
        "projection_exception_id": f"pexc_{uuid4().hex}",
        "rea_id": rea_id,
        "justification_id": justification_id,
        "projection_context_ref": projection_context_ref,
        "projection_context_hash": projection_context_hash,
        "coa_version": coa_version,
        "projection_bundle_version": projection_bundle_version,
        "constraint_bundle_version": constraint_bundle_version,
        "status": status,
        "exception_json": canonical_json(normalized_payload),
        "content_hash": content_hash(normalized_payload),
        "created_at": now_iso(),
        "resolved_at": None,
    }
    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO projection_exceptions
              (projection_exception_id, rea_id, justification_id, projection_context_ref,
               projection_context_hash, coa_version, projection_bundle_version,
               constraint_bundle_version, status, exception_json, content_hash,
               created_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO projection_exceptions
              (projection_exception_id, rea_id, justification_id, projection_context_ref,
               projection_context_hash, coa_version, projection_bundle_version,
               constraint_bundle_version, status, exception_json, content_hash,
               created_at, resolved_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                row["projection_exception_id"],
                row["rea_id"],
                row["justification_id"],
                row["projection_context_ref"],
                row["projection_context_hash"],
                row["coa_version"],
                row["projection_bundle_version"],
                row["constraint_bundle_version"],
                row["status"],
                row["exception_json"],
                row["content_hash"],
                row["created_at"],
                row["resolved_at"],
            ),
        )
    return {**row, "exception": normalized_payload}


def get_projection_exception(projection_exception_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM projection_exceptions WHERE projection_exception_id = ?", 'SELECT * FROM projection_exceptions WHERE projection_exception_id = %s'),
            (projection_exception_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["exception"] = json.loads(result["exception_json"])
    return result


def get_projection_exception_for_justification(justification_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM projection_exceptions
            WHERE justification_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """, """
            SELECT * FROM projection_exceptions
            WHERE justification_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """),
            (justification_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["exception"] = json.loads(result["exception_json"])
    return result
