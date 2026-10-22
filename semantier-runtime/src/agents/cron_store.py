from __future__ import annotations

import json
import hashlib
import os
import sqlite3
from storage import hosted_postgres as hosted
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _db_path() -> Path:
    runtime_home = (
        Path(os.environ.get("SEMANTIER_LOCAL_STATE_DIR") or ".semantier-home")
        .expanduser()
        .resolve()
    )
    if not hosted.enabled():
        runtime_home.mkdir(parents=True, exist_ok=True)
    return runtime_home / "state.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        hosted.require_tables(conn, 'semantier_cron_jobs', 'semantier_cron_outputs', 'semantier_cron_scopes')
        return
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
            hosted.sql_query("SELECT migrated_legacy_at FROM semantier_cron_scopes WHERE scope_id = ?", 'SELECT migrated_legacy_at FROM semantier_cron_scopes WHERE scope_id = %s'),
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
            hosted.sql_query("""
            INSERT INTO semantier_cron_scopes(scope_id, migrated_legacy_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(scope_id) DO UPDATE SET
                migrated_legacy_at = excluded.migrated_legacy_at,
                updated_at = excluded.updated_at
            """, """
            INSERT INTO semantier_cron_scopes(scope_id, migrated_legacy_at, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT(scope_id) DO UPDATE SET
                migrated_legacy_at = excluded.migrated_legacy_at,
                updated_at = excluded.updated_at
            """),
            (scope_id, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def list_jobs(scope_id: str, *, include_retired: bool = False) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            hosted.sql_query("""
            SELECT payload_json FROM semantier_cron_jobs
            WHERE scope_id = ?
            ORDER BY created_at ASC, job_id ASC
            """, """
            SELECT payload_json FROM semantier_cron_jobs
            WHERE scope_id = %s
            ORDER BY created_at ASC, job_id ASC
            """),
            (scope_id,),
        ).fetchall()
        jobs: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(str(row["payload_json"] or "{}"))
            if isinstance(payload, dict) and (include_retired or not payload.get("_creation_retired_at")):
                jobs.append(payload)
        return jobs
    finally:
        conn.close()


def _utc_timestamp(value: Any) -> str:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    # Explicit normalization of legacy naive timestamps at the persistence boundary.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalized_job(job: dict[str, Any], now: str) -> dict[str, Any]:
    payload = dict(job)
    for key in ("created_at", "updated_at", "next_run_at", "last_run_at", "paused_at",
                "manual_run_at", "quota_hold_until", "_scheduled_instant"):
        if payload.get(key):
            payload[key] = _utc_timestamp(payload[key])
    schedule = payload.get("schedule")
    if isinstance(schedule, dict) and schedule.get("run_at"):
        payload["schedule"] = {**schedule, "run_at": _utc_timestamp(schedule["run_at"])}
    # Only native cron record fields are timestamps; business metadata is opaque.
    for field, keys in (
        ("run_claim", ("at",)),
        ("fire_claim", ("at",)),
        ("last_fire_error", ("at",)),
        ("pending_slot", ("at", "scheduled_at")),
        ("last_dispatch", ("scheduled_at", "dispatched_at")),
        ("monitor_state", ("last_changed_at",)),
    ):
        record = payload.get(field)
        if isinstance(record, dict):
            normalized = dict(record)
            for key in keys:
                if normalized.get(key):
                    normalized[key] = _utc_timestamp(normalized[key])
            payload[field] = normalized
    payload.setdefault("created_at", now)
    payload.setdefault("updated_at", now)
    return payload


class CronStorageConflict(RuntimeError):
    pass


def jobs_snapshot(jobs: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(job["id"]): hashlib.sha256(json.dumps(job, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        for job in jobs if isinstance(job, dict) and job.get("id")
    }


def save_jobs(scope_id: str, jobs: list[dict[str, Any]], *, removed_ids=None, replace: bool = False, expected_snapshot=None) -> list[dict[str, Any]]:
    """Persist under one SQLite write transaction, preserving unrelated concurrent creates."""
    now = utc_now_iso()
    conn = _connect()
    try:
        hosted.begin_write(conn)
        current = {
            row["job_id"]: json.loads(row["payload_json"])
            for row in conn.execute(hosted.sql_query("SELECT job_id, payload_json FROM semantier_cron_jobs WHERE scope_id = ?", 'SELECT job_id, payload_json FROM semantier_cron_jobs WHERE scope_id = %s'), (scope_id,))
        }
        if expected_snapshot is not None:
            actual = jobs_snapshot([job for job in current.values() if not job.get("_creation_retired_at")])
            if any(actual.get(job_id) != digest for job_id, digest in expected_snapshot.items()):
                raise CronStorageConflict("Cron jobs changed since they were loaded; reload before saving")
            if replace and actual != expected_snapshot:
                raise CronStorageConflict("Cron jobs changed since they were loaded; reload before replacing")
            for job in jobs:
                job_id = str(job.get("id") or "") if isinstance(job, dict) else ""
                if job_id in actual and job_id not in expected_snapshot:
                    raise CronStorageConflict("Cron job already exists outside the loaded snapshot")
        payloads = ({job_id: job for job_id, job in current.items() if job.get("_creation_retired_at")}
                    if replace else dict(current))
        removed = {str(job_id) for job_id in removed_ids or ()}
        if replace:
            incoming_ids = {str(job.get("id")) for job in jobs if isinstance(job, dict)}
            removed.update(set(current) - incoming_ids)
        for job_id in removed:
            previous = current.get(job_id) or {}
            identity = previous.get("creation_identity") or {}
            if previous.get("_creation_retired_at"):
                payloads[job_id] = previous
            elif identity.get("key"):
                payloads[job_id] = {"id": job_id, "creation_identity": identity,
                    "created_at": previous.get("created_at") or now, "updated_at": now,
                    "_creation_retired_at": previous.get("_creation_retired_at") or now,
                    "enabled": False}
            else:
                payloads.pop(job_id, None)
        for job in jobs:
            if isinstance(job, dict) and str(job.get("id") or "").strip():
                job_id = str(job["id"])
                previous = current.get(job_id)
                if job.get("_creation_retired_at") or (previous or {}).get("_creation_retired_at"):
                    raise CronStorageConflict("Retired cron creation identity cannot be restored")
                if previous and job.get("creation_identity") != previous.get("creation_identity") and (job.get("creation_identity") or previous.get("creation_identity")):
                    raise CronStorageConflict("Cron creation identity is immutable")
                normalized = _normalized_job(job, now)
                meaningful = {key: value for key, value in normalized.items() if key != "updated_at"}
                prior_meaningful = {key: value for key, value in (previous or {}).items() if key != "updated_at"}
                normalized["updated_at"] = (
                    previous["updated_at"] if previous and meaningful == prior_meaningful else now
                )
                payloads[job_id] = normalized
        identities = {}
        for job_id, payload in payloads.items():
            identity = payload.get("creation_identity") or {}
            key = identity.get("key") if isinstance(identity, dict) else None
            if key:
                if key in identities and identities[key] != job_id:
                    raise CronStorageConflict("Cron creation identity already belongs to another job")
                identities[key] = job_id
        conn.execute(hosted.sql_query("DELETE FROM semantier_cron_jobs WHERE scope_id = ?", 'DELETE FROM semantier_cron_jobs WHERE scope_id = %s'), (scope_id,))
        for job_id, job in payloads.items():
            conn.execute(
                hosted.sql_query("INSERT INTO semantier_cron_jobs(scope_id, job_id, created_at, updated_at, payload_json) VALUES (?, ?, ?, ?, ?)", 'INSERT INTO semantier_cron_jobs(scope_id, job_id, created_at, updated_at, payload_json) VALUES (%s, %s, %s, %s, %s)'),
                (scope_id, job_id, job.get("created_at") or now, job.get("updated_at") or now, json.dumps(job, ensure_ascii=False, sort_keys=True)),
            )
        conn.execute(
            hosted.sql_query("INSERT INTO semantier_cron_scopes(scope_id, updated_at) VALUES (?, ?) ON CONFLICT(scope_id) DO UPDATE SET updated_at = excluded.updated_at", 'INSERT INTO semantier_cron_scopes(scope_id, updated_at) VALUES (%s, %s) ON CONFLICT(scope_id) DO UPDATE SET updated_at = excluded.updated_at'),
            (scope_id, now),
        )
        conn.commit()
        return [job for job in payloads.values() if not job.get("_creation_retired_at")]
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def replace_jobs(scope_id: str, jobs: list[dict[str, Any]], *, expected_jobs=None) -> None:
    expected = jobs_snapshot(expected_jobs) if expected_jobs is not None else None
    save_jobs(scope_id, jobs, replace=True, expected_snapshot=expected)


def save_output(scope_id: str, job_id: str, output_text: str, run_at: str | None = None) -> None:
    job_id = str(job_id or "").strip()
    if not job_id:
        return
    run_at = _utc_timestamp(run_at) if run_at is not None else utc_now_iso()
    conn = _connect()
    try:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO semantier_cron_outputs(scope_id, job_id, run_at, output_text)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(scope_id, job_id, run_at) DO UPDATE SET
                output_text = excluded.output_text
            """, """
            INSERT INTO semantier_cron_outputs(scope_id, job_id, run_at, output_text)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT(scope_id, job_id, run_at) DO UPDATE SET
                output_text = excluded.output_text
            """),
            (scope_id, job_id, run_at, str(output_text or "")),
        )
        conn.commit()
    finally:
        conn.close()


def latest_output(scope_id: str, job_id: str) -> str | None:
    outputs = list_outputs(scope_id, job_id, limit=1)
    return outputs[0]["content"] if outputs else None


def list_outputs(scope_id: str, job_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
    job_id = str(job_id or "").strip()
    try:
        normalized_limit = max(1, min(int(limit), 100))
    except Exception:
        normalized_limit = 10
    conn = _connect()
    try:
        rows = conn.execute(
            hosted.sql_query("""
            SELECT run_at, output_text FROM semantier_cron_outputs
            WHERE scope_id = ? AND job_id = ?
            """, """
            SELECT run_at, output_text FROM semantier_cron_outputs
            WHERE scope_id = %s AND job_id = %s
            """),
            (scope_id, job_id),
        ).fetchall()
        # Existing version keys can contain naive/offset ISO values. Normalize for
        # comparison and presentation without rewriting their persisted identities.
        ordered = sorted(
            rows,
            key=lambda row: (datetime.fromisoformat(_utc_timestamp(row["run_at"]).replace("Z", "+00:00")), str(row["run_at"])),
            reverse=True,
        )[:normalized_limit]
        return [
            {
                "filename": f"{row['run_at']}.md",
                "timestamp": _utc_timestamp(row["run_at"]),
                "content": str(row["output_text"] or ""),
                "size": len(str(row["output_text"] or "").encode("utf-8")),
            }
            for row in ordered
        ]
    finally:
        conn.close()


def delete_outputs(scope_id: str, job_id: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            hosted.sql_query("DELETE FROM semantier_cron_outputs WHERE scope_id = ? AND job_id = ?", 'DELETE FROM semantier_cron_outputs WHERE scope_id = %s AND job_id = %s'),
            (scope_id, str(job_id or "").strip()),
        )
        conn.commit()
    finally:
        conn.close()
