from __future__ import annotations

from storage import hosted_postgres as hosted

import hashlib
import json
import sqlite3
import uuid
from typing import Any

from eos.db import get_conn, now_iso


TERMINAL_RUN_STATUSES = {"completed", "failed", "blocked"}


def ensure_schema() -> None:
    with get_conn() as conn:
        if hosted.is_postgres(conn):
            hosted.require_tables(conn, 'real_company_setup_runs', 'real_company_setup_events', 'real_company_source_rows', 'real_company_admission_exceptions')
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS real_company_setup_runs (
              setup_run_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              dataset_version_id TEXT NOT NULL,
              requested_by TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              status TEXT NOT NULL,
              current_stage TEXT,
              blocking_reason TEXT,
              lock_owner TEXT,
              lock_acquired_at TEXT,
              lock_released_at TEXT,
              started_at TEXT NOT NULL,
              completed_at TEXT,
              updated_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              UNIQUE (organization_id, dataset_version_id, idempotency_key)
            );

            CREATE INDEX IF NOT EXISTS idx_real_company_setup_runs_lock
              ON real_company_setup_runs(organization_id, dataset_version_id, status);

            CREATE TABLE IF NOT EXISTS real_company_setup_events (
              event_id TEXT PRIMARY KEY,
              setup_run_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              dataset_version_id TEXT NOT NULL,
              stage TEXT NOT NULL,
              event_type TEXT NOT NULL,
              event_status TEXT NOT NULL,
              details_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_real_company_setup_events_scope
              ON real_company_setup_events(organization_id, dataset_version_id, stage, created_at);

            CREATE TABLE IF NOT EXISTS real_company_source_rows (
              source_row_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              dataset_version_id TEXT NOT NULL,
              import_id TEXT,
              file_id TEXT,
              source_row_hash TEXT NOT NULL,
              source_file_hash TEXT,
              parser_profile_hash TEXT,
              field_mapping_profile_hash TEXT,
              classification_status TEXT,
              row_type TEXT,
              row_json TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              UNIQUE (organization_id, dataset_version_id, source_row_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_real_company_source_rows_scope
              ON real_company_source_rows(organization_id, dataset_version_id, classification_status);

            CREATE TABLE IF NOT EXISTS real_company_admission_exceptions (
              exception_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              dataset_version_id TEXT NOT NULL,
              source_row_hash TEXT NOT NULL,
              admission_profile_hash TEXT NOT NULL,
              reason_code TEXT NOT NULL,
              details_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              UNIQUE (organization_id, dataset_version_id, source_row_hash, admission_profile_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_real_company_admission_exceptions_scope
              ON real_company_admission_exceptions(organization_id, dataset_version_id);
            """
        )


def create_or_get_setup_run(
    *,
    organization_id: str,
    dataset_version_id: str,
    requested_by: str,
    idempotency_key: str,
    stage: str = "reconcile_promoted_imports",
) -> dict[str, Any]:
    ensure_schema()
    if not organization_id.strip():
        raise ValueError("organization_id required")
    if not dataset_version_id.strip():
        raise ValueError("dataset_version_id required")
    if not requested_by.strip():
        raise ValueError("requested_by required")
    if not idempotency_key.strip():
        raise ValueError("idempotency_key required")
    now = now_iso()
    setup_run_id = f"setup_run_{uuid.uuid4().hex}"
    with get_conn() as conn:
        existing = conn.execute(
            hosted.sql_query("""
            SELECT * FROM real_company_setup_runs
            WHERE organization_id = ? AND dataset_version_id = ? AND idempotency_key = ?
            """, """
            SELECT * FROM real_company_setup_runs
            WHERE organization_id = %s AND dataset_version_id = %s AND idempotency_key = %s
            """),
            (organization_id, dataset_version_id, idempotency_key),
        ).fetchone()
        if existing is not None:
            return _row_to_run(existing)

        locked = conn.execute(
            hosted.sql_query("""
            SELECT * FROM real_company_setup_runs
            WHERE organization_id = ? AND dataset_version_id = ?
              AND status NOT IN ('completed', 'failed', 'blocked')
            ORDER BY started_at DESC, setup_run_id DESC
            LIMIT 1
            """, """
            SELECT * FROM real_company_setup_runs
            WHERE organization_id = %s AND dataset_version_id = %s
              AND status NOT IN ('completed', 'failed', 'blocked')
            ORDER BY started_at DESC, setup_run_id DESC
            LIMIT 1
            """),
            (organization_id, dataset_version_id),
        ).fetchone()
        if locked is not None:
            return {
                "ok": False,
                "error": "SETUP_RUN_LOCKED",
                "setup_run_id": locked["setup_run_id"],
                "current_stage": locked["current_stage"],
                "blocking_reasons": ["SETUP_RUN_LOCKED"],
            }

        record = {
            "setup_run_id": setup_run_id,
            "organization_id": organization_id,
            "dataset_version_id": dataset_version_id,
            "requested_by": requested_by,
            "idempotency_key": idempotency_key,
            "status": "running",
            "current_stage": stage,
            "blocking_reason": None,
            "lock_owner": setup_run_id,
            "lock_acquired_at": now,
            "lock_released_at": None,
            "started_at": now,
            "completed_at": None,
            "updated_at": now,
        }
        content_hash = _hash_payload(record)
        conn.execute(
            hosted.sql_query("""
            INSERT INTO real_company_setup_runs (
              setup_run_id, organization_id, dataset_version_id, requested_by,
              idempotency_key, status, current_stage, blocking_reason, lock_owner,
              lock_acquired_at, lock_released_at, started_at, completed_at,
              updated_at, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO real_company_setup_runs (
              setup_run_id, organization_id, dataset_version_id, requested_by,
              idempotency_key, status, current_stage, blocking_reason, lock_owner,
              lock_acquired_at, lock_released_at, started_at, completed_at,
              updated_at, content_hash
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                setup_run_id,
                organization_id,
                dataset_version_id,
                requested_by,
                idempotency_key,
                "running",
                stage,
                None,
                setup_run_id,
                now,
                None,
                now,
                None,
                now,
                content_hash,
            ),
        )
        append_setup_event(
            setup_run_id=setup_run_id,
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            stage=stage,
            event_type="run_started",
            event_status="running",
            details={"idempotency_key": idempotency_key},
            conn=conn,
        )
        return {**record, "content_hash": content_hash}


def transition_setup_run(
    *,
    setup_run_id: str,
    status: str,
    current_stage: str | None = None,
    blocking_reason: str | None = None,
) -> dict[str, Any]:
    ensure_schema()
    now = now_iso()
    lock_released_at = now if status in TERMINAL_RUN_STATUSES else None
    completed_at = now if status in TERMINAL_RUN_STATUSES else None
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM real_company_setup_runs WHERE setup_run_id = ?", 'SELECT * FROM real_company_setup_runs WHERE setup_run_id = %s'),
            (setup_run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(setup_run_id)
        payload = _row_to_run(row)
        payload.update(
            {
                "status": status,
                "current_stage": current_stage or payload.get("current_stage"),
                "blocking_reason": blocking_reason,
                "lock_released_at": lock_released_at,
                "completed_at": completed_at,
                "updated_at": now,
            }
        )
        content_hash = _hash_payload({k: v for k, v in payload.items() if k != "content_hash"})
        conn.execute(
            hosted.sql_query("""
            UPDATE real_company_setup_runs
            SET status = ?, current_stage = ?, blocking_reason = ?,
                lock_released_at = ?, completed_at = ?, updated_at = ?, content_hash = ?
            WHERE setup_run_id = ?
            """, """
            UPDATE real_company_setup_runs
            SET status = %s, current_stage = %s, blocking_reason = %s,
                lock_released_at = %s, completed_at = %s, updated_at = %s, content_hash = %s
            WHERE setup_run_id = %s
            """),
            (
                status,
                payload["current_stage"],
                blocking_reason,
                lock_released_at,
                completed_at,
                now,
                content_hash,
                setup_run_id,
            ),
        )
        return {**payload, "content_hash": content_hash}


def append_setup_event(
    *,
    setup_run_id: str,
    organization_id: str,
    dataset_version_id: str,
    stage: str,
    event_type: str,
    event_status: str,
    details: dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    if conn is None:
        ensure_schema()
    event_id = f"setup_event_{uuid.uuid4().hex}"
    created_at = now_iso()
    details_json = _json_dumps(details or {})
    payload = {
        "event_id": event_id,
        "setup_run_id": setup_run_id,
        "organization_id": organization_id,
        "dataset_version_id": dataset_version_id,
        "stage": stage,
        "event_type": event_type,
        "event_status": event_status,
        "details_json": details_json,
        "created_at": created_at,
    }
    content_hash = _hash_payload(payload)

    def _insert(active_conn: sqlite3.Connection) -> None:
        active_conn.execute(
            hosted.sql_query("""
            INSERT INTO real_company_setup_events (
              event_id, setup_run_id, organization_id, dataset_version_id, stage,
              event_type, event_status, details_json, created_at, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO real_company_setup_events (
              event_id, setup_run_id, organization_id, dataset_version_id, stage,
              event_type, event_status, details_json, created_at, content_hash
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                event_id,
                setup_run_id,
                organization_id,
                dataset_version_id,
                stage,
                event_type,
                event_status,
                details_json,
                created_at,
                content_hash,
            ),
        )

    if conn is not None:
        _insert(conn)
    else:
        with get_conn() as active_conn:
            _insert(active_conn)
    return {**payload, "details": details or {}, "content_hash": content_hash}


def upsert_source_row(
    *,
    organization_id: str,
    dataset_version_id: str,
    source_row_hash: str,
    row: dict[str, Any],
    import_id: str | None = None,
    file_id: str | None = None,
    source_file_hash: str | None = None,
    parser_profile_hash: str | None = None,
    field_mapping_profile_hash: str | None = None,
    classification_status: str | None = "classified",
    row_type: str | None = "unknown",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_schema()
    now = now_iso()
    row_json = _json_dumps(row)
    metadata_json = _json_dumps(metadata or {})
    payload = {
        "organization_id": organization_id,
        "dataset_version_id": dataset_version_id,
        "source_row_hash": source_row_hash,
        "row_json": row_json,
        "metadata_json": metadata_json,
        "classification_status": classification_status,
        "row_type": row_type,
    }
    content_hash = _hash_payload(payload)
    source_row_id = f"source_row_{uuid.uuid4().hex}"
    with get_conn() as conn:
        existing = conn.execute(
            hosted.sql_query("""
            SELECT * FROM real_company_source_rows
            WHERE organization_id = ? AND dataset_version_id = ? AND source_row_hash = ?
            """, """
            SELECT * FROM real_company_source_rows
            WHERE organization_id = %s AND dataset_version_id = %s AND source_row_hash = %s
            """),
            (organization_id, dataset_version_id, source_row_hash),
        ).fetchone()
        if existing is not None:
            return _row_to_source_row(existing)
        conn.execute(
            hosted.sql_query("""
            INSERT INTO real_company_source_rows (
              source_row_id, organization_id, dataset_version_id, import_id, file_id,
              source_row_hash, source_file_hash, parser_profile_hash,
              field_mapping_profile_hash, classification_status, row_type, row_json,
              metadata_json, created_at, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO real_company_source_rows (
              source_row_id, organization_id, dataset_version_id, import_id, file_id,
              source_row_hash, source_file_hash, parser_profile_hash,
              field_mapping_profile_hash, classification_status, row_type, row_json,
              metadata_json, created_at, content_hash
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                source_row_id,
                organization_id,
                dataset_version_id,
                import_id,
                file_id,
                source_row_hash,
                source_file_hash,
                parser_profile_hash,
                field_mapping_profile_hash,
                classification_status,
                row_type,
                row_json,
                metadata_json,
                now,
                content_hash,
            ),
        )
    return {
        "source_row_id": source_row_id,
        **payload,
        "created_at": now,
        "content_hash": content_hash,
    }


def list_source_rows(
    *,
    organization_id: str,
    dataset_version_id: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    ensure_schema()
    sql = hosted.sql_query("""
        SELECT * FROM real_company_source_rows
        WHERE organization_id = ? AND dataset_version_id = ?
        ORDER BY created_at, source_row_id
    """, """
        SELECT * FROM real_company_source_rows
        WHERE organization_id = %s AND dataset_version_id = %s
        ORDER BY created_at, source_row_id
    """)
    params: list[Any] = [organization_id, dataset_version_id]
    if limit is not None:
        sql += hosted.sql_query(" LIMIT ?", ' LIMIT %s')
        params.append(int(limit))
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_source_row(row) for row in rows]


def get_source_row(
    *,
    organization_id: str,
    dataset_version_id: str,
    source_row_hash: str,
) -> dict[str, Any] | None:
    ensure_schema()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM real_company_source_rows
            WHERE organization_id = ? AND dataset_version_id = ? AND source_row_hash = ?
            """, """
            SELECT * FROM real_company_source_rows
            WHERE organization_id = %s AND dataset_version_id = %s AND source_row_hash = %s
            """),
            (organization_id, dataset_version_id, source_row_hash),
        ).fetchone()
        return _row_to_source_row(row) if row is not None else None


def record_admission_exception(
    *,
    organization_id: str,
    dataset_version_id: str,
    source_row_hash: str,
    admission_profile_hash: str,
    reason_code: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_schema()
    now = now_iso()
    exception_id = f"admission_exception_{uuid.uuid4().hex}"
    details_json = _json_dumps(details or {})
    payload = {
        "organization_id": organization_id,
        "dataset_version_id": dataset_version_id,
        "source_row_hash": source_row_hash,
        "admission_profile_hash": admission_profile_hash,
        "reason_code": reason_code,
        "details_json": details_json,
    }
    content_hash = _hash_payload(payload)
    with get_conn() as conn:
        existing = conn.execute(
            hosted.sql_query("""
            SELECT * FROM real_company_admission_exceptions
            WHERE organization_id = ? AND dataset_version_id = ?
              AND source_row_hash = ? AND admission_profile_hash = ?
            """, """
            SELECT * FROM real_company_admission_exceptions
            WHERE organization_id = %s AND dataset_version_id = %s
              AND source_row_hash = %s AND admission_profile_hash = %s
            """),
            (organization_id, dataset_version_id, source_row_hash, admission_profile_hash),
        ).fetchone()
        if existing is not None:
            return _row_to_admission_exception(existing)
        conn.execute(
            hosted.sql_query("""
            INSERT INTO real_company_admission_exceptions (
              exception_id, organization_id, dataset_version_id, source_row_hash,
              admission_profile_hash, reason_code, details_json, created_at, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO real_company_admission_exceptions (
              exception_id, organization_id, dataset_version_id, source_row_hash,
              admission_profile_hash, reason_code, details_json, created_at, content_hash
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                exception_id,
                organization_id,
                dataset_version_id,
                source_row_hash,
                admission_profile_hash,
                reason_code,
                details_json,
                now,
                content_hash,
            ),
        )
    return {
        "exception_id": exception_id,
        **payload,
        "details": details or {},
        "created_at": now,
        "content_hash": content_hash,
    }


def list_admission_exceptions(
    *,
    organization_id: str,
    dataset_version_id: str,
) -> list[dict[str, Any]]:
    ensure_schema()
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("""
            SELECT * FROM real_company_admission_exceptions
            WHERE organization_id = ? AND dataset_version_id = ?
            ORDER BY created_at, exception_id
            """, """
            SELECT * FROM real_company_admission_exceptions
            WHERE organization_id = %s AND dataset_version_id = %s
            ORDER BY created_at, exception_id
            """),
            (organization_id, dataset_version_id),
        ).fetchall()
        return [_row_to_admission_exception(row) for row in rows]


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _row_to_source_row(row: sqlite3.Row) -> dict[str, Any]:
    payload = {key: row[key] for key in row.keys()}
    payload["row"] = json.loads(payload.pop("row_json") or "{}")
    payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
    return payload


def _row_to_admission_exception(row: sqlite3.Row) -> dict[str, Any]:
    payload = {key: row[key] for key in row.keys()}
    payload["details"] = json.loads(payload.pop("details_json") or "{}")
    return payload


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()
