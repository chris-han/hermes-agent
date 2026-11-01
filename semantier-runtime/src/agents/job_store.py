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
    if payload.get('kind') == 'managed_work.v1':
        raise ValueError('managed work must be created by its owner')
    now = utc_now_iso()
    requested_job_id = str(payload.get("id") or "").strip()
    job_id = (
        requested_job_id if _JOB_ID_RE.match(requested_job_id) else uuid.uuid4().hex[:12]
    )
    existing = get_job(user_id=user_id, workspace_id=workspace_id, job_id=job_id)
    if existing and existing.get('kind') == 'managed_work.v1':
        raise ValueError('managed work cannot be overwritten by legacy jobs')
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
        cur = conn.execute(
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
            WHERE COALESCE(json_extract(semantier_jobs.payload_json, '$.kind'),'') != 'managed_work.v1'
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
            WHERE COALESCE(semantier_jobs.payload_json::jsonb->>'kind','') != 'managed_work.v1'
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
        if cur.rowcount != 1:
            raise ValueError('managed work cannot be overwritten by legacy jobs')
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
    if current.get('kind') == 'managed_work.v1' or updates.get('kind') == 'managed_work.v1':
        raise ValueError('managed work cannot be updated by legacy jobs')
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
        cur = conn.execute(
            hosted.sql_query("""
            UPDATE semantier_jobs
            SET state = ?, enabled = ?, updated_at = ?, payload_json = ?
            WHERE job_id = ? AND user_id = ? AND workspace_id = ?
              AND COALESCE(json_extract(payload_json, '$.kind'),'') != 'managed_work.v1'
            """, """
            UPDATE semantier_jobs
            SET state = %s, enabled = %s, updated_at = %s, payload_json = %s
            WHERE job_id = %s AND user_id = %s AND workspace_id = %s
              AND COALESCE(payload_json::jsonb->>'kind','') != 'managed_work.v1'
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
        if cur.rowcount != 1:
            raise ValueError('managed work cannot be updated by legacy jobs')
        if hosted.enabled():
            from agents.operations_task_events import capture_task
            capture_task(conn,user_id=user_id,job=job)
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
              AND COALESCE(json_extract(payload_json, '$.kind'),'') != 'managed_work.v1'
            """, """
            DELETE FROM semantier_jobs
            WHERE job_id = %s AND user_id = %s AND workspace_id = %s
              AND COALESCE(payload_json::jsonb->>'kind','') != 'managed_work.v1'
            """),
            (job_id, user_id, workspace_id),
        )
        conn.commit()
        if cur.rowcount < 1:
            raise JobNotFoundError(job_id)
    finally:
        conn.close()


def claim_independent_task(*, context, task_id, request_hash, payload):
    """Atomic, durable at-most-once dispatch identity in the existing job owner.

    Replays return the retained task, including uncertain/interrupted outcomes.
    They never schedule a second process. Task jobs are disabled for cron.
    """
    if not hosted.enabled():
        raise RuntimeError('independent tasks require PostgreSQL')
    if not _JOB_ID_RE.fullmatch(task_id):
        raise ValueError('invalid task request identity')
    now = utc_now_iso()
    job = {**payload, 'kind': 'independent_task', 'request_hash': request_hash,
           'id': task_id, 'state': 'queued', 'enabled': False,
           'created_at': now, 'updated_at': now, 'events': []}
    with _connect() as conn:
        row = conn.execute('''INSERT INTO semantier_jobs
            (job_id,user_id,workspace_id,organization_id,state,enabled,created_at,updated_at,payload_json)
            VALUES(%s,%s,%s,%s,'queued',0,%s,%s,%s)
            ON CONFLICT(user_id,workspace_id,job_id) DO NOTHING RETURNING job_id''',
            (task_id, context.user_id, context.workspace_id, context.organization_id, now, now,
             json.dumps(job, sort_keys=True))).fetchone()
        current = conn.execute('SELECT * FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND job_id=%s',
            (context.user_id, context.workspace_id, task_id)).fetchone()
        retained = _job_payload_from_row(current)
        if retained.get('kind') != 'independent_task' or retained.get('request_hash') != request_hash:
            raise ValueError('request identity already belongs to different input')
        if row is not None:
            from agents.operations_task_events import capture_task
            capture_task(conn,user_id=context.user_id,job=retained)
    return retained, row is not None


def interrupt_independent_tasks():
    """Cold-start refusal: a lost process is uncertain, never automatically rerun."""
    with _connect() as c:
        rows = c.execute("SELECT user_id,workspace_id,job_id,payload_json FROM semantier_jobs WHERE state IN ('queued','running') AND enabled=0").fetchall()
        for row in rows:
            payload = json.loads(row['payload_json'])
            if payload.get('kind') != 'independent_task':
                continue
            payload.update(state='interrupted', error='executor_lost_outcome_uncertain', updated_at=utc_now_iso())
            c.execute("UPDATE semantier_jobs SET state='interrupted',updated_at=%s,payload_json=%s WHERE user_id=%s AND workspace_id=%s AND job_id=%s", (payload['updated_at'], json.dumps(payload, sort_keys=True), row['user_id'], row['workspace_id'], row['job_id']))


def claim_managed_work(*, context, work_id, request_hash, title, work_input, schedule=None, kind="managed_work.v1"):
    """Insert an immutable owner-scoped management identity without upsert."""
    if not hosted.enabled():
        raise RuntimeError('managed work requires PostgreSQL')
    if not _JOB_ID_RE.fullmatch(work_id):
        raise ValueError('invalid work identity')
    now = utc_now_iso()
    if kind not in {'managed_work.v1', 'managed_schedule.v1'}: raise ValueError('unsupported managed job kind')
    payload = {'id':work_id, 'kind':kind, 'request_hash':request_hash,
               'title':title, 'input':work_input, 'responsibility_id':None,
               'created_at':now, 'updated_at':now, 'enabled':False, 'state':'setup',
               'accountable_owner':{'user_id':context.user_id},
               'responsible_executor':{'kind':'current_owner','user_id':context.user_id}}
    if schedule is not None:
        payload['schedule'] = schedule
    with _connect() as c:
        row = c.execute('''INSERT INTO semantier_jobs(job_id,user_id,workspace_id,organization_id,state,enabled,created_at,updated_at,payload_json)
            VALUES(%s,%s,%s,%s,'setup',0,%s,%s,%s)
            ON CONFLICT(user_id,workspace_id,job_id) DO NOTHING RETURNING job_id''',
            (work_id, context.user_id, context.workspace_id, context.organization_id, now, now, json.dumps(payload,sort_keys=True))).fetchone()
        current = c.execute('SELECT * FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND job_id=%s',
            (context.user_id,context.workspace_id,work_id)).fetchone()
        if current['organization_id'] != context.organization_id:
            raise PermissionError('work organization mismatch')
        retained = _job_payload_from_row(current)
        if retained.get('kind') != kind or retained.get('request_hash') != request_hash:
            raise ValueError('work request identity conflict')
    return retained, row is not None


def append_managed_comment(*, context, work_id, request_id, content, expected_version):
    """Append a scoped, idempotent comment without mutating lifecycle."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND organization_id=%s AND job_id=%s FOR UPDATE",
            (context.user_id, context.workspace_id, context.organization_id, work_id)).fetchone()
        if row is None:
            raise KeyError(work_id)
        data = _job_payload_from_row(row)
        if data.get('kind') != 'managed_work.v1':
            raise KeyError(work_id)
        comments = data.setdefault('comments', [])
        prior = next((item for item in comments if item['request_id'] == request_id), None)
        if prior is not None:
            if prior['content'] != content or prior['version'] != expected_version + 1:
                raise ValueError('comment identity conflict')
            return prior
        if expected_version != len(comments):
            raise ValueError('comment version conflict')
        if len(comments) >= 256:
            raise ValueError('comment limit reached')
        item = {'request_id':request_id, 'content':content, 'author_id':context.user_id,
                'created_at':utc_now_iso(), 'version':len(comments)+1}
        comments.append(item)
        conn.execute("UPDATE semantier_jobs SET payload_json=%s,updated_at=%s WHERE user_id=%s AND workspace_id=%s AND organization_id=%s AND job_id=%s",
            (json.dumps(data,sort_keys=True,ensure_ascii=False),utc_now_iso(),context.user_id,context.workspace_id,context.organization_id,work_id))
        return item


def append_managed_reference(*, context, work_id, request_id, ref, expected_version):
    """Retain one exact already-authorized work reference, not a capability."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND organization_id=%s AND job_id=%s FOR UPDATE",
            (context.user_id, context.workspace_id, context.organization_id, work_id)).fetchone()
        if row is None:
            raise KeyError(work_id)
        data = _job_payload_from_row(row)
        if data.get('kind') != 'managed_work.v1':
            raise KeyError(work_id)
        refs = data.setdefault('references', [])
        prior = next((item for item in refs if item['request_id'] == request_id), None)
        if prior is not None:
            if prior['ref'] != ref or prior['version'] != expected_version + 1:
                raise ValueError('reference identity conflict')
            return prior
        if expected_version != len(refs):
            raise ValueError('reference version conflict')
        if len(refs) >= 256:
            raise ValueError('reference limit reached')
        target = conn.execute("SELECT payload_json FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND organization_id=%s AND job_id=%s",
            (context.user_id, context.workspace_id, context.organization_id, ref)).fetchone()
        if target is None or json.loads(target['payload_json']).get('kind') != 'managed_work.v1':
            raise KeyError(ref)
        item = {'request_id':request_id, 'kind':'work', 'ref':ref, 'author_id':context.user_id,
                'created_at':utc_now_iso(), 'version':len(refs)+1}
        refs.append(item)
        conn.execute("UPDATE semantier_jobs SET payload_json=%s,updated_at=%s WHERE user_id=%s AND workspace_id=%s AND organization_id=%s AND job_id=%s",
            (json.dumps(data,sort_keys=True,ensure_ascii=False),utc_now_iso(),context.user_id,context.workspace_id,context.organization_id,work_id))
        return item


def list_managed_work(*, context, limit, cursor):
    """Bounded keyset query, with no native jobs in the projection."""
    if not hosted.enabled():
        raise RuntimeError('managed work requires PostgreSQL')
    with _connect() as c:
        rows = c.execute('''SELECT * FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s
            AND organization_id=%s AND payload_json::jsonb->>'kind'='managed_work.v1'
            AND job_id>%s ORDER BY job_id LIMIT %s''',
            (context.user_id,context.workspace_id,context.organization_id,cursor,limit+1)).fetchall()
    return [_job_payload_from_row(row) for row in rows]


def store_managed_result(conn, *, context, work_id, result, result_hash, attempt_hash):
    """Commit result with the durable owner's current-claim transaction."""
    row=conn.execute('''SELECT payload_json,organization_id FROM semantier_jobs
        WHERE user_id=%s AND workspace_id=%s AND job_id=%s FOR UPDATE''',
        (context.user_id,context.workspace_id,work_id)).fetchone()
    if not row or row['organization_id']!=context.organization_id:
        raise KeyError(work_id)
    data=json.loads(row['payload_json'])
    if data.get('kind')!='managed_work.v1' or 'result_hash' in data:
        raise ValueError('work result conflict')
    data.update(result=result,result_hash=result_hash,result_attempt_hash=attempt_hash)
    conn.execute('''UPDATE semantier_jobs SET payload_json=%s,updated_at=%s
        WHERE user_id=%s AND workspace_id=%s AND job_id=%s''',
        (json.dumps(data,sort_keys=True),utc_now_iso(),context.user_id,context.workspace_id,work_id))


def bind_managed_work(*, context, work_id, responsibility_id):
    """Bind only the exact managed identity, preserving its immutable request."""
    with _connect() as c:
        row=c.execute("SELECT * FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND organization_id=%s AND job_id=%s FOR UPDATE",(context.user_id,context.workspace_id,context.organization_id,work_id)).fetchone()
        if row is None: raise KeyError(work_id)
        data=_job_payload_from_row(row)
        if data.get('kind')!='managed_work.v1' or data.get('responsibility_id') not in (None,responsibility_id):
            raise ValueError('managed binding conflict')
        data['responsibility_id']=responsibility_id
        c.execute("UPDATE semantier_jobs SET payload_json=%s,updated_at=%s WHERE user_id=%s AND workspace_id=%s AND job_id=%s",(json.dumps(data,sort_keys=True),utc_now_iso(),context.user_id,context.workspace_id,work_id))
    return get_job(user_id=context.user_id,workspace_id=context.workspace_id,job_id=work_id)


def reserve_managed_execution(*, context, work_id, run_id, claim, skill=None):
    """Record one launch reservation on the scoped managed-work owner."""
    with _connect() as conn:
        row = conn.execute('''SELECT payload_json, organization_id FROM semantier_jobs
            WHERE user_id=%s AND workspace_id=%s AND job_id=%s FOR UPDATE''',
            (context.user_id, context.workspace_id, work_id)).fetchone()
        if not row or row['organization_id'] != context.organization_id:
            raise KeyError(work_id)
        data = json.loads(row['payload_json'])
        if data.get('kind') != 'managed_work.v1' or data.get('responsibility_id') != claim.responsibility_id:
            raise ValueError('managed execution binding conflict')
        if 'execution_reservation' in data:
            return False
        data['execution_reservation'] = {
            'run_id': run_id, 'responsibility_id': claim.responsibility_id,
            'lease_epoch': claim.lease_epoch, 'checkpoint_version': claim.checkpoint_version,
            'state': 'reserved', 'reserved_at': utc_now_iso(),
        }
        if skill is not None:
            data['execution_reservation']['skill'] = skill
        conn.execute('''UPDATE semantier_jobs SET payload_json=%s,updated_at=%s
            WHERE user_id=%s AND workspace_id=%s AND job_id=%s''',
            (json.dumps(data, sort_keys=True), utc_now_iso(), context.user_id, context.workspace_id, work_id))
        return True


def store_managed_execution_result(conn, *, context, work_id, claim, run_id, result, result_hash):
    """Bind retained execution outcome to its reservation in the checkpoint transaction."""
    row = conn.execute('''SELECT payload_json,organization_id FROM semantier_jobs
        WHERE user_id=%s AND workspace_id=%s AND job_id=%s FOR UPDATE''',
        (context.user_id, context.workspace_id, work_id)).fetchone()
    if not row or row['organization_id'] != context.organization_id:
        raise KeyError(work_id)
    data = json.loads(row['payload_json'])
    reservation = data.get('execution_reservation') or {}
    if (data.get('kind') != 'managed_work.v1' or 'result_hash' in data or
        reservation.get('state') != 'reserved' or reservation.get('run_id') != run_id or
        reservation.get('responsibility_id') != claim.responsibility_id or
        reservation.get('lease_epoch') != claim.lease_epoch or
        reservation.get('checkpoint_version') != claim.checkpoint_version):
        from agents.durable_responsibility import StaleResponsibilityClaim
        raise StaleResponsibilityClaim('execution reservation is stale')
    reservation['state'] = 'completed'
    data.update(result=result, result_hash=result_hash, result_attempt_hash=claim.owner_id)
    conn.execute('''UPDATE semantier_jobs SET payload_json=%s,updated_at=%s
        WHERE user_id=%s AND workspace_id=%s AND job_id=%s''',
        (json.dumps(data, sort_keys=True), utc_now_iso(), context.user_id, context.workspace_id, work_id))


def mark_managed_execution_stopped(*, context, work_id, run_id, epoch, process_status, reaped):
    """Record process reaping without turning uncertain effects into a result."""
    with _connect() as conn:
        row = conn.execute('''SELECT payload_json,organization_id FROM semantier_jobs
            WHERE user_id=%s AND workspace_id=%s AND job_id=%s FOR UPDATE''',
            (context.user_id, context.workspace_id, work_id)).fetchone()
        if not row or row['organization_id'] != context.organization_id:
            raise KeyError(work_id)
        data = json.loads(row['payload_json'])
        reservation = data.get('execution_reservation') or {}
        if reservation.get('run_id') != run_id or reservation.get('lease_epoch') != epoch:
            return False
        if reservation.get('state') == 'completed':
            return True
        reservation.update(state='stopped' if reaped else 'uncertain', process_status=process_status,
                           effects='uncertain', stopped_at=utc_now_iso())
        conn.execute('''UPDATE semantier_jobs SET payload_json=%s,updated_at=%s
            WHERE user_id=%s AND workspace_id=%s AND job_id=%s''',
            (json.dumps(data, sort_keys=True), utc_now_iso(), context.user_id, context.workspace_id, work_id))
        return True


def record_managed_execution_monitoring(*, context, work_id, run_id, epoch, monitoring):
    """Attach optional post-commit monitoring without changing the terminal result."""
    with _connect() as conn:
        row = conn.execute('''SELECT payload_json,organization_id FROM semantier_jobs
            WHERE user_id=%s AND workspace_id=%s AND job_id=%s FOR UPDATE''',
            (context.user_id, context.workspace_id, work_id)).fetchone()
        if not row or row['organization_id'] != context.organization_id:
            raise KeyError(work_id)
        data = json.loads(row['payload_json'])
        reservation = data.get('execution_reservation') or {}
        if reservation.get('state') != 'completed' or reservation.get('run_id') != run_id or reservation.get('lease_epoch') != epoch:
            return False
        reservation['monitoring'] = monitoring
        conn.execute('''UPDATE semantier_jobs SET payload_json=%s,updated_at=%s
            WHERE user_id=%s AND workspace_id=%s AND job_id=%s''',
            (json.dumps(data, sort_keys=True), utc_now_iso(), context.user_id, context.workspace_id, work_id))
        return True


def update_managed_schedule(*, context, work_id, schedule):
    """Schedule owner holds the scoped work lock; retain concurrent annotations."""
    with _connect() as c:
        row = c.execute('SELECT payload_json FROM semantier_jobs WHERE organization_id=%s AND workspace_id=%s AND user_id=%s AND job_id=%s FOR UPDATE', (context.organization_id, context.workspace_id, context.user_id, work_id)).fetchone()
        if row is None: raise KeyError(work_id)
        data = json.loads(row['payload_json'])
        if data.get('kind') not in {'managed_work.v1', 'managed_schedule.v1'}: raise KeyError(work_id)
        data['schedule'] = schedule
        c.execute('UPDATE semantier_jobs SET payload_json=%s,updated_at=%s WHERE organization_id=%s AND workspace_id=%s AND user_id=%s AND job_id=%s', (json.dumps(data,sort_keys=True),utc_now_iso(),context.organization_id,context.workspace_id,context.user_id,work_id))


def scheduled_work_rows(*, context=None, phases=None, pending_actions=False, unarmed=False, limit=32, kind="managed_work.v1"):
    if type(limit) is not int or not 1 <= limit <= 100: raise ValueError('bounded schedule query required')
    if kind not in {"managed_work.v1", "managed_schedule.v1"}: raise ValueError("unsupported schedule kind")
    query="SELECT * FROM semantier_jobs WHERE payload_json::jsonb->>'kind'=%s AND payload_json::jsonb ? 'schedule'"
    args=[kind]
    if context:
        query+=' AND organization_id=%s AND workspace_id=%s AND user_id=%s'
        args += [context.organization_id,context.workspace_id,context.user_id]
    if phases:
        query+=" AND (payload_json::jsonb->'schedule'->>'phase'=ANY(%s)";args.append(list(phases))
        if pending_actions:query+=" OR payload_json::jsonb->'schedule' ? 'pending_action'"
        query+=")"
        if not pending_actions:query+=" AND NOT (payload_json::jsonb ? 'execution_reservation')"
    if kind == 'managed_schedule.v1' and context is None:
        # Stopped/unknown occurrences must not monopolize the bounded recovery batch.
        query+=""" AND (payload_json::jsonb->'schedule' ? 'pending_action'
            OR NOT EXISTS (SELECT 1 FROM semantier_jobs child
                WHERE child.user_id=semantier_jobs.user_id AND child.workspace_id=semantier_jobs.workspace_id
                AND child.job_id=semantier_jobs.payload_json::jsonb->'schedule'->>'current_work_id')
            OR ((payload_json::jsonb->'schedule'->>'paused'='false'
                 OR (payload_json::jsonb->'schedule'->'repeat'->>'completed')::integer
                    < jsonb_array_length(payload_json::jsonb->'schedule'->'occurrences'))
                AND EXISTS (SELECT 1 FROM semantier_jobs child
                    WHERE child.user_id=semantier_jobs.user_id AND child.workspace_id=semantier_jobs.workspace_id
                    AND child.job_id=semantier_jobs.payload_json::jsonb->'schedule'->>'current_work_id'
                    AND child.payload_json::jsonb ? 'result_hash')))"""
    if unarmed:
        query+=" AND payload_json::jsonb->'schedule'->>'paused'='false' AND EXISTS (SELECT 1 FROM semantier_timer_occurrences t WHERE t.timer_id=payload_json::jsonb->'schedule'->>'timer_id' AND t.provider_ref IS NULL AND t.state='SCHEDULED')"
    query+=' ORDER BY updated_at,job_id LIMIT %s';args.append(limit)
    with _connect() as c: return [dict(row) for row in c.execute(query,args).fetchall()]


def create_conversation_record(*, context, conversation_id, title):
    """Insert a fresh core conversation into the existing scoped job owner."""
    now=utc_now_iso()
    payload={'id':conversation_id,'kind':'core_conversation.v1','title':title,'messages':[],'turns':[],
             'created_at':now,'updated_at':now,'enabled':False,'state':'open'}
    with _connect() as conn:
        added=conn.execute('''INSERT INTO semantier_jobs(job_id,user_id,workspace_id,organization_id,state,enabled,created_at,updated_at,payload_json)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(user_id,workspace_id,job_id) DO NOTHING''',
            (conversation_id,context.user_id,context.workspace_id,context.organization_id,'open',0,now,now,json.dumps(payload,sort_keys=True)))
        if added.rowcount!=1:
            raise ValueError('conversation identity already belongs to an existing record')
    return payload


def claim_core_channel_record(*, context, payload):
    """Strict insertion of a scoped channel association, never a legacy job upsert."""
    if payload.get('kind')!='core_feishu_channel.v1' or not _JOB_ID_RE.fullmatch(payload['id']):
        raise ValueError('unsupported channel record')
    now=utc_now_iso();payload={**payload,'created_at':now,'updated_at':now,'enabled':False,'state':'open'}
    with _connect() as conn:
        conn.execute('''INSERT INTO semantier_jobs(job_id,user_id,workspace_id,organization_id,state,enabled,created_at,updated_at,payload_json)
            VALUES(%s,%s,%s,%s,'open',0,%s,%s,%s) ON CONFLICT(user_id,workspace_id,job_id) DO NOTHING''',
            (payload['id'],context.user_id,context.workspace_id,context.organization_id,now,now,json.dumps(payload,sort_keys=True)))
        row=conn.execute('SELECT organization_id,payload_json FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND job_id=%s',(context.user_id,context.workspace_id,payload['id'])).fetchone()
        retained=json.loads(row['payload_json'])
        if row['organization_id']!=context.organization_id or retained.get('kind')!=payload['kind'] or retained['association']!=payload['association']:
            raise ValueError('channel identity conflict')
        return retained
