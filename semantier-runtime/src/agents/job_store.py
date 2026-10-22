from __future__ import annotations

import json
import os
import re
import sqlite3
from storage import hosted_postgres as hosted
from datetime import datetime, timezone
from pathlib import Path
import uuid
from typing import Any


class JobNotFoundError(KeyError):
    pass


_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _db_path() -> Path:
    runtime_home = (
        Path(os.environ.get("SEMANTIER_LOCAL_STATE_DIR") or ".semantier-home")
        .expanduser()
        .resolve()
    )
    if not hosted.enabled():
        runtime_home.mkdir(parents=True, exist_ok=True)
    return runtime_home / "state.db"


def _connect() -> sqlite3.Connection:
    if hosted.enabled():
        conn = hosted.connect_authority('execution', _db_path())
        _ensure_schema(conn)
        return conn
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    if hosted.is_postgres(conn):
        hosted.require_tables(conn, 'semantier_jobs')
        return
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS semantier_jobs (
            job_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            organization_id TEXT,
            state TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY(user_id, workspace_id, job_id)
        )
        """
    )
    pk_columns = [
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(semantier_jobs)").fetchall()
        if int(row["pk"] or 0) > 0
    ]
    if pk_columns == ["job_id"]:
        conn.execute("ALTER TABLE semantier_jobs RENAME TO semantier_jobs_legacy")
        conn.execute(
            """
            CREATE TABLE semantier_jobs (
                job_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                organization_id TEXT,
                state TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(user_id, workspace_id, job_id)
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO semantier_jobs(
                job_id, user_id, workspace_id, organization_id,
                state, enabled, created_at, updated_at, payload_json
            )
            SELECT
                job_id, user_id, workspace_id, organization_id,
                state, enabled, created_at, updated_at, payload_json
            FROM semantier_jobs_legacy
            """
        )
        conn.execute("DROP TABLE semantier_jobs_legacy")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_semantier_jobs_owner
        ON semantier_jobs(user_id, workspace_id, updated_at)
        """
    )
    conn.commit()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_utc_iso(value: Any, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _normalize_job_timestamps(job: dict[str, Any], *, fallback: str) -> None:
    for key in (
        "created_at",
        "updated_at",
        "next_run_at",
        "last_run_at",
        "paused_at",
        "triggered_at",
    ):
        if job.get(key) is not None:
            job[key] = _coerce_utc_iso(job.get(key), fallback)
    schedule = job.get("schedule")
    if isinstance(schedule, dict) and schedule.get("run_at") is not None:
        schedule["run_at"] = _coerce_utc_iso(schedule.get("run_at"), fallback)


def normalize_job_schedule(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        expr = str(
            value.get("expr") or value.get("display") or "* * * * *"
        ).strip()
        return {
            "kind": str(value.get("kind") or "cron").strip() or "cron",
            "expr": expr or "* * * * *",
            "display": str(value.get("display") or expr or "* * * * *").strip(),
        }
    expr = str(value or "").strip() or "* * * * *"
    return {"kind": "cron", "expr": expr, "display": expr}


def _job_payload_from_row(row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(str(row["payload_json"] or "{}"))
    if not isinstance(payload, dict):
        payload = {}
    payload["id"] = str(row["job_id"])
    payload["enabled"] = bool(row["enabled"])
    payload["state"] = str(row["state"])
    payload["created_at"] = str(row["created_at"])
    payload["updated_at"] = str(row["updated_at"])
    payload.setdefault("last_run_success", None)
    return payload


def list_jobs(
    *,
    user_id: str,
    workspace_id: str,
    include_disabled: bool = True,
) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        sql = hosted.sql_query("""
            SELECT * FROM semantier_jobs
            WHERE user_id = ? AND workspace_id = ?
        """, """
            SELECT * FROM semantier_jobs
            WHERE user_id = %s AND workspace_id = %s
        """)
        params: list[Any] = [user_id, workspace_id]
        if not include_disabled:
            sql += " AND enabled = 1"
        sql += " ORDER BY updated_at DESC, created_at DESC"
        rows = conn.execute(sql, params).fetchall()
        return [_job_payload_from_row(row) for row in rows]
    finally:
        conn.close()


def get_job(*, user_id: str, workspace_id: str, job_id: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM semantier_jobs
            WHERE job_id = ? AND user_id = ? AND workspace_id = ?
            """, """
            SELECT * FROM semantier_jobs
            WHERE job_id = %s AND user_id = %s AND workspace_id = %s
            """),
            (job_id, user_id, workspace_id),
        ).fetchone()
        return _job_payload_from_row(row) if row is not None else None
    finally:
        conn.close()


def create_job(
    *,
    user_id: str,
    workspace_id: str,
    organization_id: str | None = None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    now = utc_now_iso()
    requested_job_id = str(payload.get("id") or "").strip()
    job_id = (
        requested_job_id if _JOB_ID_RE.match(requested_job_id) else uuid.uuid4().hex[:12]
    )
    existing = get_job(user_id=user_id, workspace_id=workspace_id, job_id=job_id)
    schedule = normalize_job_schedule(payload.get("schedule"))
    name = str(payload.get("name") or "scheduled-job").strip() or "scheduled-job"
    prompt = str(payload.get("prompt") or "").strip()
    enabled = _coerce_enabled(payload.get("enabled", True))
    state = str(payload.get("state") or ("active" if enabled else "paused")).strip()
    if not state:
        state = "active" if enabled else "paused"
    created_at = _coerce_utc_iso(
        payload.get("created_at"),
        str(existing.get("created_at")) if existing else now,
    )
    updated_at = _coerce_utc_iso(payload.get("updated_at"), now)
    job: dict[str, Any] = {
        "id": job_id,
        "name": name,
        "prompt": prompt,
        "schedule": schedule,
        "schedule_display": schedule["display"],
        "enabled": enabled,
        "state": state,
        "next_run_at": None,
        "last_run_at": None,
        "last_run_success": None,
        "created_at": created_at,
        "updated_at": updated_at,
        "skills": (
            payload.get("skills") if isinstance(payload.get("skills"), list) else []
        ),
        "deliver": payload.get("deliver") or [],
    }
    for key, value in payload.items():
        if key in {"id", "user_id", "workspace_id", "organization_id"}:
            continue
        if key not in job or job[key] in (None, "", []):
            job[key] = value
    repeat = payload.get("repeat")
    if isinstance(repeat, int):
        job["repeat"] = {"times": repeat, "completed": 0}
    _normalize_job_timestamps(job, fallback=now)
    conn = _connect()
    try:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO semantier_jobs(
                job_id, user_id, workspace_id, organization_id,
                state, enabled, created_at, updated_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, workspace_id, job_id) DO UPDATE SET
                organization_id = excluded.organization_id,
                state = excluded.state,
                enabled = excluded.enabled,
                updated_at = excluded.updated_at,
                payload_json = excluded.payload_json
            """, """
            INSERT INTO semantier_jobs(
                job_id, user_id, workspace_id, organization_id,
                state, enabled, created_at, updated_at, payload_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(user_id, workspace_id, job_id) DO UPDATE SET
                organization_id = excluded.organization_id,
                state = excluded.state,
                enabled = excluded.enabled,
                updated_at = excluded.updated_at,
                payload_json = excluded.payload_json
            """),
            (
                job_id,
                user_id,
                workspace_id,
                organization_id,
                job["state"],
                1 if job["enabled"] else 0,
                job["created_at"],
                job["updated_at"],
                json.dumps(job, ensure_ascii=False, sort_keys=True),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return job


def update_job(
    *,
    user_id: str,
    workspace_id: str,
    job_id: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    current = get_job(user_id=user_id, workspace_id=workspace_id, job_id=job_id)
    if current is None:
        raise JobNotFoundError(job_id)
    job = dict(current)
    job.update(updates)
    if "schedule" in updates:
        schedule = normalize_job_schedule(updates.get("schedule"))
        job["schedule"] = schedule
        job["schedule_display"] = schedule["display"]
    enabled = _coerce_enabled(job.get("enabled", True))
    state = str(job.get("state") or ("active" if enabled else "paused"))
    now = utc_now_iso()
    job["id"] = job_id
    job["enabled"] = enabled
    job["state"] = state
    job["updated_at"] = now
    _normalize_job_timestamps(job, fallback=now)
    conn = _connect()
    try:
        conn.execute(
            hosted.sql_query("""
            UPDATE semantier_jobs
            SET state = ?, enabled = ?, updated_at = ?, payload_json = ?
            WHERE job_id = ? AND user_id = ? AND workspace_id = ?
            """, """
            UPDATE semantier_jobs
            SET state = %s, enabled = %s, updated_at = %s, payload_json = %s
            WHERE job_id = %s AND user_id = %s AND workspace_id = %s
            """),
            (
                state,
                1 if enabled else 0,
                now,
                json.dumps(job, ensure_ascii=False, sort_keys=True),
                job_id,
                user_id,
                workspace_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return job


def delete_job(*, user_id: str, workspace_id: str, job_id: str) -> None:
    conn = _connect()
    try:
        cur = conn.execute(
            hosted.sql_query("""
            DELETE FROM semantier_jobs
            WHERE job_id = ? AND user_id = ? AND workspace_id = ?
            """, """
            DELETE FROM semantier_jobs
            WHERE job_id = %s AND user_id = %s AND workspace_id = %s
            """),
            (job_id, user_id, workspace_id),
        )
        conn.commit()
        if cur.rowcount < 1:
            raise JobNotFoundError(job_id)
    finally:
        conn.close()
