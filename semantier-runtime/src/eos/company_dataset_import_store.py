from __future__ import annotations

from storage import hosted_postgres as hosted

import json
import sqlite3
import uuid
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

from eos.db import get_conn, now_iso


IMPORT_STATUSES = {
    "uploaded",
    "validating",
    "validation_failed",
    "staged",
    "promoting",
    "promotion_failed",
    "promoted",
    "superseded",
}

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "uploaded": {"validating"},
    "validating": {"validation_failed", "staged"},
    "validation_failed": {"validating"},
    "staged": {"promoting"},
    "promoting": {"promoted", "promotion_failed"},
    "promotion_failed": {"promoting"},
    "promoted": {"superseded"},
    "superseded": set(),
}


class CompanyDatasetImportStoreError(RuntimeError):
    pass


class InvalidImportStateTransition(CompanyDatasetImportStoreError):
    pass


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _retention_until(status: str, created_at: str | None = None) -> str | None:
    if status not in {"uploaded", "validation_failed", "staged"}:
        return None
    base = datetime.now(timezone.utc)
    if created_at:
        try:
            parsed = datetime.fromisoformat(created_at)
            base = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return (base.astimezone(timezone.utc) + timedelta(days=30)).isoformat()


def ensure_schema() -> None:
    with get_conn() as conn:
        if hosted.is_postgres(conn):
            hosted.require_tables(conn, 'company_dataset_imports', 'company_dataset_import_files', 'company_dataset_import_validation_runs', 'organization_dataset_versions')
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS company_dataset_imports (
              import_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              actor_user_id TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              status TEXT NOT NULL,
              parser_profile_schema_version TEXT NOT NULL,
              parser_profile_hash TEXT NOT NULL,
              parser_profile_json TEXT NOT NULL,
              validation_result_json TEXT,
              promotion_result_json TEXT,
              error_category TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              retention_until TEXT,
              UNIQUE (organization_id, idempotency_key)
            );

            CREATE INDEX IF NOT EXISTS idx_company_dataset_imports_org
              ON company_dataset_imports(organization_id, updated_at);

            CREATE TABLE IF NOT EXISTS company_dataset_import_files (
              file_id TEXT PRIMARY KEY,
              import_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              file_role TEXT NOT NULL,
              original_filename TEXT,
              content_type TEXT,
              byte_size INTEGER NOT NULL,
              artifact_path TEXT NOT NULL,
              source_file_hash TEXT,
              normalized_file_hash TEXT,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_company_dataset_import_files_import
              ON company_dataset_import_files(import_id, file_role);

            CREATE TABLE IF NOT EXISTS company_dataset_import_validation_runs (
              validation_run_id TEXT PRIMARY KEY,
              import_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              parser_profile_schema_version TEXT NOT NULL,
              parser_profile_hash TEXT NOT NULL,
              validation_result_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE (organization_id, idempotency_key)
            );

            CREATE TABLE IF NOT EXISTS organization_dataset_versions (
              dataset_version_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              import_id TEXT NOT NULL,
              artifact_hash_manifest_json TEXT NOT NULL,
              activated_at TEXT,
              activated_by TEXT,
              superseded_by TEXT,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            """
        )


def _row_to_import(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "import_id": row["import_id"],
        "organization_id": row["organization_id"],
        "actor_user_id": row["actor_user_id"],
        "idempotency_key": row["idempotency_key"],
        "status": row["status"],
        "parser_profile_schema_version": row["parser_profile_schema_version"],
        "parser_profile_hash": row["parser_profile_hash"],
        "parser_profile": _json_loads(row["parser_profile_json"], {}),
        "validation_result": _json_loads(row["validation_result_json"], None),
        "promotion_result": _json_loads(row["promotion_result_json"], None),
        "error_category": row["error_category"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "retention_until": row["retention_until"],
    }


def _row_to_file(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "file_id": row["file_id"],
        "import_id": row["import_id"],
        "organization_id": row["organization_id"],
        "file_role": row["file_role"],
        "original_filename": row["original_filename"],
        "content_type": row["content_type"],
        "byte_size": row["byte_size"],
        "artifact_path": row["artifact_path"],
        "source_file_hash": row["source_file_hash"],
        "normalized_file_hash": row["normalized_file_hash"],
        "created_at": row["created_at"],
    }


def _hydrate_import(conn: sqlite3.Connection, row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = _row_to_import(row)
    file_rows = conn.execute(
        hosted.sql_query("""
        SELECT * FROM company_dataset_import_files
        WHERE import_id = ?
        ORDER BY created_at, file_id
        """, """
        SELECT * FROM company_dataset_import_files
        WHERE import_id = %s
        ORDER BY created_at, file_id
        """),
        (payload["import_id"],),
    ).fetchall()
    payload["files"] = [_row_to_file(file_row) for file_row in file_rows]
    return payload



def get_import_for_idempotency(*, organization_id: str, idempotency_key: str) -> dict[str, Any] | None:
    """Read the existing org-owned request before any upload bytes are written."""
    ensure_schema()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM company_dataset_imports WHERE organization_id=? AND idempotency_key=?", 'SELECT * FROM company_dataset_imports WHERE organization_id=%s AND idempotency_key=%s'),
            (organization_id, idempotency_key),
        ).fetchone()
        return _hydrate_import(conn, row)

def create_import(
    *,
    organization_id: str,
    actor_user_id: str,
    idempotency_key: str,
    parser_profile_schema_version: str,
    parser_profile_hash: str,
    parser_profile: dict[str, Any],
    source_files: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    ensure_schema()
    if not organization_id.strip():
        raise ValueError("organization_id required")
    if not actor_user_id.strip():
        raise ValueError("actor_user_id required")
    if not idempotency_key.strip():
        raise ValueError("idempotency_key required")
    now = now_iso()
    import_id = f"import_{uuid.uuid4().hex}"
    with get_conn() as conn:
        existing = conn.execute(
            hosted.sql_query("""
            SELECT * FROM company_dataset_imports
            WHERE organization_id = ? AND idempotency_key = ?
            """, """
            SELECT * FROM company_dataset_imports
            WHERE organization_id = %s AND idempotency_key = %s
            """),
            (organization_id, idempotency_key),
        ).fetchone()
        if existing is not None:
            return _hydrate_import(conn, existing) or {}

        conn.execute(
            hosted.sql_query("""
            INSERT INTO company_dataset_imports (
              import_id, organization_id, actor_user_id, idempotency_key, status,
              parser_profile_schema_version, parser_profile_hash, parser_profile_json,
              created_at, updated_at, retention_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO company_dataset_imports (
              import_id, organization_id, actor_user_id, idempotency_key, status,
              parser_profile_schema_version, parser_profile_hash, parser_profile_json,
              created_at, updated_at, retention_until
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                import_id,
                organization_id,
                actor_user_id,
                idempotency_key,
                "uploaded",
                parser_profile_schema_version,
                parser_profile_hash,
                _json_dumps(parser_profile),
                now,
                now,
                _retention_until("uploaded", now),
            ),
        )
        for source in source_files:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO company_dataset_import_files (
                  file_id, import_id, organization_id, file_role, original_filename,
                  content_type, byte_size, artifact_path, source_file_hash,
                  normalized_file_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO company_dataset_import_files (
                  file_id, import_id, organization_id, file_role, original_filename,
                  content_type, byte_size, artifact_path, source_file_hash,
                  normalized_file_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    str(source.get("file_id") or f"file_{uuid.uuid4().hex}"),
                    import_id,
                    organization_id,
                    "source",
                    str(source.get("original_filename") or ""),
                    str(source.get("content_type") or ""),
                    int(source.get("byte_size") or 0),
                    str(source.get("artifact_path") or ""),
                    str(source.get("source_file_hash") or ""),
                    None,
                    now,
                ),
            )
        conn.commit()
        row = conn.execute(
            hosted.sql_query("SELECT * FROM company_dataset_imports WHERE import_id = ?", 'SELECT * FROM company_dataset_imports WHERE import_id = %s'),
            (import_id,),
        ).fetchone()
        return _hydrate_import(conn, row) or {}


def get_import(*, organization_id: str, import_id: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM company_dataset_imports
            WHERE organization_id = ? AND import_id = ?
            """, """
            SELECT * FROM company_dataset_imports
            WHERE organization_id = %s AND import_id = %s
            """),
            (organization_id, import_id),
        ).fetchone()
        return _hydrate_import(conn, row)


def list_imports(*, organization_id: str, limit: int = 20) -> list[dict[str, Any]]:
    ensure_schema()
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("""
            SELECT * FROM company_dataset_imports
            WHERE organization_id = ?
            ORDER BY updated_at DESC, import_id DESC
            LIMIT ?
            """, """
            SELECT * FROM company_dataset_imports
            WHERE organization_id = %s
            ORDER BY updated_at DESC, import_id DESC
            LIMIT %s
            """),
            (organization_id, int(limit)),
        ).fetchall()
        return [_hydrate_import(conn, row) or {} for row in rows]


def transition_import_status(
    *,
    organization_id: str,
    import_id: str,
    from_statuses: Iterable[str],
    to_status: str,
    validation_result: dict[str, Any] | None = None,
    promotion_result: dict[str, Any] | None = None,
    error_category: str | None = None,
) -> dict[str, Any]:
    ensure_schema()
    allowed_from = set(from_statuses)
    if to_status not in IMPORT_STATUSES:
        raise ValueError("unknown import status")
    for status in allowed_from:
        if to_status not in ALLOWED_TRANSITIONS.get(status, set()):
            raise InvalidImportStateTransition(f"invalid transition {status} -> {to_status}")
    now = now_iso()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM company_dataset_imports
            WHERE organization_id = ? AND import_id = ?
            """, """
            SELECT * FROM company_dataset_imports
            WHERE organization_id = %s AND import_id = %s
            """),
            (organization_id, import_id),
        ).fetchone()
        if row is None:
            raise KeyError(import_id)
        current_status = str(row["status"])
        if current_status not in allowed_from:
            raise InvalidImportStateTransition(f"invalid current status {current_status}")
        cursor = conn.execute(
            hosted.sql_query("""
            UPDATE company_dataset_imports
            SET status = ?,
                validation_result_json = COALESCE(?, validation_result_json),
                promotion_result_json = COALESCE(?, promotion_result_json),
                error_category = ?,
                updated_at = ?,
                retention_until = ?
            WHERE organization_id = ? AND import_id = ? AND status = ?
            """, """
            UPDATE company_dataset_imports
            SET status = %s,
                validation_result_json = COALESCE(%s, validation_result_json),
                promotion_result_json = COALESCE(%s, promotion_result_json),
                error_category = %s,
                updated_at = %s,
                retention_until = %s
            WHERE organization_id = %s AND import_id = %s AND status = %s
            """),
            (
                to_status,
                _json_dumps(validation_result) if validation_result is not None else None,
                _json_dumps(promotion_result) if promotion_result is not None else None,
                error_category,
                now,
                _retention_until(to_status, row["created_at"]),
                organization_id,
                import_id,
                current_status,
            ),
        )
        if cursor.rowcount < 1:
            raise InvalidImportStateTransition("state transition compare-and-swap failed")
        conn.commit()
        updated = conn.execute(
            hosted.sql_query("SELECT * FROM company_dataset_imports WHERE organization_id = ? AND import_id = ?", 'SELECT * FROM company_dataset_imports WHERE organization_id = %s AND import_id = %s'),
            (organization_id, import_id),
        ).fetchone()
        return _hydrate_import(conn, updated) or {}


def record_validation_run(
    *,
    organization_id: str,
    import_id: str,
    idempotency_key: str,
    parser_profile_schema_version: str,
    parser_profile_hash: str,
    validation_result: dict[str, Any],
) -> dict[str, Any]:
    ensure_schema()
    if not idempotency_key.strip():
        raise ValueError("idempotency_key required")
    now = now_iso()
    with get_conn() as conn:
        existing = conn.execute(
            hosted.sql_query("""
            SELECT validation_result_json FROM company_dataset_import_validation_runs
            WHERE organization_id = ? AND idempotency_key = ?
            """, """
            SELECT validation_result_json FROM company_dataset_import_validation_runs
            WHERE organization_id = %s AND idempotency_key = %s
            """),
            (organization_id, idempotency_key),
        ).fetchone()
        if existing is not None:
            return _json_loads(existing["validation_result_json"], {})
        conn.execute(
            hosted.sql_query("""
            INSERT INTO company_dataset_import_validation_runs (
              validation_run_id, import_id, organization_id, idempotency_key,
              parser_profile_schema_version, parser_profile_hash,
              validation_result_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO company_dataset_import_validation_runs (
              validation_run_id, import_id, organization_id, idempotency_key,
              parser_profile_schema_version, parser_profile_hash,
              validation_result_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                f"validation_{uuid.uuid4().hex}",
                import_id,
                organization_id,
                idempotency_key,
                parser_profile_schema_version,
                parser_profile_hash,
                _json_dumps(validation_result),
                now,
            ),
        )
        conn.commit()
        return validation_result


def get_validation_run_result(*, organization_id: str, idempotency_key: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT validation_result_json FROM company_dataset_import_validation_runs
            WHERE organization_id = ? AND idempotency_key = ?
            """, """
            SELECT validation_result_json FROM company_dataset_import_validation_runs
            WHERE organization_id = %s AND idempotency_key = %s
            """),
            (organization_id, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        return _json_loads(row["validation_result_json"], {})


def create_dataset_version(
    *,
    organization_id: str,
    import_id: str,
    activated_by: str,
    artifact_hash_manifest: dict[str, Any],
) -> dict[str, Any]:
    ensure_schema()
    now = now_iso()
    dataset_version_id = f"dsv_{uuid.uuid4().hex}"
    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO organization_dataset_versions (
              dataset_version_id, organization_id, import_id,
              artifact_hash_manifest_json, activated_at, activated_by,
              superseded_by, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO organization_dataset_versions (
              dataset_version_id, organization_id, import_id,
              artifact_hash_manifest_json, activated_at, activated_by,
              superseded_by, status, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                dataset_version_id,
                organization_id,
                import_id,
                _json_dumps(artifact_hash_manifest),
                now,
                activated_by,
                None,
                "active",
                now,
            ),
        )
        conn.commit()
    return {
        "dataset_version_id": dataset_version_id,
        "organization_id": organization_id,
        "import_id": import_id,
        "activated_at": now,
        "activated_by": activated_by,
        "status": "active",
        "artifact_hash_manifest": artifact_hash_manifest,
    }


def get_dataset_version(*, organization_id: str, dataset_version_id: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM organization_dataset_versions
            WHERE organization_id = ? AND dataset_version_id = ?
            """, """
            SELECT * FROM organization_dataset_versions
            WHERE organization_id = %s AND dataset_version_id = %s
            """),
            (organization_id, dataset_version_id),
        ).fetchone()
        if row is None:
            return None
        return {
            "dataset_version_id": row["dataset_version_id"],
            "organization_id": row["organization_id"],
            "import_id": row["import_id"],
            "artifact_hash_manifest": _json_loads(row["artifact_hash_manifest_json"], {}),
            "activated_at": row["activated_at"],
            "activated_by": row["activated_by"],
            "superseded_by": row["superseded_by"],
            "status": row["status"],
            "created_at": row["created_at"],
        }


def get_active_dataset_version(*, organization_id: str) -> dict[str, Any] | None:
    ensure_schema()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM organization_dataset_versions
            WHERE organization_id = ? AND status = 'active'
            ORDER BY activated_at DESC, created_at DESC, dataset_version_id DESC
            LIMIT 1
            """, """
            SELECT * FROM organization_dataset_versions
            WHERE organization_id = %s AND status = 'active'
            ORDER BY activated_at DESC, created_at DESC, dataset_version_id DESC
            LIMIT 1
            """),
            (organization_id,),
        ).fetchone()
        if row is None:
            return None
        return get_dataset_version(
            organization_id=organization_id,
            dataset_version_id=row["dataset_version_id"],
        )
