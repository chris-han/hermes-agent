from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _db_path() -> Path:
    runtime_home = (
        Path(os.environ.get("SEMANTIER_LOCAL_STATE_DIR") or ".semantier-home")
        .expanduser()
        .resolve()
    )
    runtime_home.mkdir(parents=True, exist_ok=True)
    return runtime_home / "state.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS semantier_cron_jobs (
            scope_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY(scope_id, job_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS semantier_cron_scopes (
            scope_id TEXT PRIMARY KEY,
            migrated_legacy_at TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_semantier_cron_jobs_scope_updated
        ON semantier_cron_jobs(scope_id, updated_at)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS semantier_cron_outputs (
            scope_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            run_at TEXT NOT NULL,
            output_text TEXT NOT NULL,
            PRIMARY KEY(scope_id, job_id, run_at)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_semantier_cron_outputs_latest
        ON semantier_cron_outputs(scope_id, job_id, run_at DESC)
        """
    )
    conn.commit()


def legacy_import_marked(scope_id: str) -> bool:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT migrated_legacy_at FROM semantier_cron_scopes WHERE scope_id = ?",
            (scope_id,),
        ).fetchone()
        return row is not None and bool(row["migrated_legacy_at"])
    finally:
        conn.close()


def mark_legacy_imported(scope_id: str) -> None:
    now = utc_now_iso()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO semantier_cron_scopes(scope_id, migrated_legacy_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(scope_id) DO UPDATE SET
                migrated_legacy_at = excluded.migrated_legacy_at,
                updated_at = excluded.updated_at
            """,
            (scope_id, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def list_jobs(scope_id: str) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT payload_json FROM semantier_cron_jobs
            WHERE scope_id = ?
            ORDER BY created_at ASC, job_id ASC
            """,
            (scope_id,),
        ).fetchall()
        jobs: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(str(row["payload_json"] or "{}"))
            if isinstance(payload, dict):
                jobs.append(payload)
        return jobs
    finally:
        conn.close()


def replace_jobs(scope_id: str, jobs: list[dict[str, Any]]) -> None:
    now = utc_now_iso()
    conn = _connect()
    try:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM semantier_cron_jobs WHERE scope_id = ?", (scope_id,))
        for job in jobs:
            if not isinstance(job, dict):
                continue
            job_id = str(job.get("id") or "").strip()
            if not job_id:
                continue
            created_at = str(job.get("created_at") or now)
            updated_at = str(job.get("updated_at") or now)
            conn.execute(
                """
                INSERT INTO semantier_cron_jobs(
                    scope_id, job_id, created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    scope_id,
                    job_id,
                    created_at,
                    updated_at,
                    json.dumps(job, ensure_ascii=False, sort_keys=True),
                ),
            )
        conn.execute(
            """
            INSERT INTO semantier_cron_scopes(scope_id, updated_at)
            VALUES (?, ?)
            ON CONFLICT(scope_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (scope_id, now),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_output(scope_id: str, job_id: str, output_text: str, run_at: str | None = None) -> None:
    job_id = str(job_id or "").strip()
    if not job_id:
        return
    run_at = run_at or utc_now_iso()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO semantier_cron_outputs(scope_id, job_id, run_at, output_text)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(scope_id, job_id, run_at) DO UPDATE SET
                output_text = excluded.output_text
            """,
            (scope_id, job_id, run_at, str(output_text or "")),
        )
        conn.commit()
    finally:
        conn.close()


def latest_output(scope_id: str, job_id: str) -> str | None:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT output_text FROM semantier_cron_outputs
            WHERE scope_id = ? AND job_id = ?
            ORDER BY run_at DESC
            LIMIT 1
            """,
            (scope_id, str(job_id or "").strip()),
        ).fetchone()
        if row is None:
            return None
        return str(row["output_text"] or "")
    finally:
        conn.close()


def list_outputs(scope_id: str, job_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
    job_id = str(job_id or "").strip()
    try:
        normalized_limit = max(1, min(int(limit), 100))
    except Exception:
        normalized_limit = 10
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT run_at, output_text FROM semantier_cron_outputs
            WHERE scope_id = ? AND job_id = ?
            ORDER BY run_at DESC
            LIMIT ?
            """,
            (scope_id, job_id, normalized_limit),
        ).fetchall()
        return [
            {
                "filename": f"{row['run_at']}.md",
                "timestamp": str(row["run_at"]),
                "content": str(row["output_text"] or ""),
                "size": len(str(row["output_text"] or "").encode("utf-8")),
            }
            for row in rows
        ]
    finally:
        conn.close()


def delete_outputs(scope_id: str, job_id: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "DELETE FROM semantier_cron_outputs WHERE scope_id = ? AND job_id = ?",
            (scope_id, str(job_id or "").strip()),
        )
        conn.commit()
    finally:
        conn.close()