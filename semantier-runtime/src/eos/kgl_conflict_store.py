"""Append-only precedence and conflict review persistence."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
import sqlite3
from typing import Any, Dict, Optional
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso


def record_conflict_review(
    *,
    artifact_id: str,
    payload: dict,
    created_by: str,
    conn: sqlite3.Connection | None = None,
) -> Dict[str, Any]:
    row = {
        "conflict_id": f"kglc_{uuid4().hex}",
        "artifact_id": artifact_id,
        "higher_tier_artifact_ids_json": canonical_json(payload.get("higher_tier_artifact_ids", [])),
        "same_domain_artifact_ids_json": canonical_json(payload.get("same_domain_artifact_ids", [])),
        "decision": str(payload["decision"]),
        "review_notes": payload.get("review_notes"),
        "created_at": now_iso(),
        "created_by": created_by,
    }
    hash_payload = {
        "artifact_id": artifact_id,
        "higher_tier_artifact_ids": payload.get("higher_tier_artifact_ids", []),
        "same_domain_artifact_ids": payload.get("same_domain_artifact_ids", []),
        "decision": row["decision"],
        "review_notes": row["review_notes"],
        "created_by": created_by,
    }
    row["content_hash"] = content_hash(hash_payload)
    if conn is None:
        with get_conn() as managed_conn:
            managed_conn.execute(
                hosted.sql_query("""
                INSERT INTO kgl_conflict_records (
                  conflict_id, artifact_id, higher_tier_artifact_ids_json,
                  same_domain_artifact_ids_json, decision, review_notes,
                  content_hash, created_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO kgl_conflict_records (
                  conflict_id, artifact_id, higher_tier_artifact_ids_json,
                  same_domain_artifact_ids_json, decision, review_notes,
                  content_hash, created_at, created_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    row["conflict_id"],
                    row["artifact_id"],
                    row["higher_tier_artifact_ids_json"],
                    row["same_domain_artifact_ids_json"],
                    row["decision"],
                    row["review_notes"],
                    row["content_hash"],
                    row["created_at"],
                    row["created_by"],
                ),
            )
    else:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO kgl_conflict_records (
              conflict_id, artifact_id, higher_tier_artifact_ids_json,
              same_domain_artifact_ids_json, decision, review_notes,
              content_hash, created_at, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO kgl_conflict_records (
              conflict_id, artifact_id, higher_tier_artifact_ids_json,
              same_domain_artifact_ids_json, decision, review_notes,
              content_hash, created_at, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                row["conflict_id"],
                row["artifact_id"],
                row["higher_tier_artifact_ids_json"],
                row["same_domain_artifact_ids_json"],
                row["decision"],
                row["review_notes"],
                row["content_hash"],
                row["created_at"],
                row["created_by"],
            ),
        )
    return get_conflict_review(row["conflict_id"]) or row


def get_conflict_review(conflict_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM kgl_conflict_records WHERE conflict_id = ?", 'SELECT * FROM kgl_conflict_records WHERE conflict_id = %s'),
            (conflict_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["higher_tier_artifact_ids"] = json.loads(result.pop("higher_tier_artifact_ids_json") or "[]")
    result["same_domain_artifact_ids"] = json.loads(result.pop("same_domain_artifact_ids_json") or "[]")
    return result


def list_conflict_reviews_for_artifact(artifact_id: str) -> list[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("""
            SELECT * FROM kgl_conflict_records
            WHERE artifact_id = ?
            ORDER BY created_at ASC, conflict_id ASC
            """, """
            SELECT * FROM kgl_conflict_records
            WHERE artifact_id = %s
            ORDER BY created_at ASC, conflict_id ASC
            """),
            (artifact_id,),
        ).fetchall()
    return [get_conflict_review(str(row["conflict_id"])) for row in rows if row]
