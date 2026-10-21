"""Profile-local durable handoff for cron delivery through live gateway adapters.

A restart-safe cron worker executes outside the gateway cgroup.  It cannot own
relay/E2EE adapter objects, so it queues the final send here.  A gateway claims
each row at most once.  If that gateway dies after claiming, the outcome is
marked unknown and never retried: losing a delivery is safer than duplicating a
possibly-completed send.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager, closing
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from agent.redact import redact_sensitive_text
from cron.executions import _owner_is_live, _process_start_time
from hermes_state_backend import postgres_enabled, resolve_postgres_scope
from hermes_state_sql import connect_postgres, native_write_transaction, NativeRow

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

DELIVERY_DB: Optional[Path] = None
_PROCESS_ID = uuid.uuid4().hex
_lock = threading.RLock()
_ACTIVE_DELIVERIES: set[str] = set()
_TERMINAL = ("delivered", "failed", "unknown", "suppressed")
MAX_TERMINAL_DELIVERIES = 1000
DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS = 300.0


def _prune_terminal_unlocked(conn: sqlite3.Connection) -> None:
    """Redact terminal payloads and retain only bounded outcome metadata."""
    conn.execute(
        _query(
            conn,
            """UPDATE deliveries SET job_json='{}', content=''
           WHERE status IN ('delivered','failed','unknown','suppressed')
             AND (job_json != '{}' OR content != '')""",
            "UPDATE {deliveries} SET job_json='{{}}', content=''\n           WHERE status IN ('delivered','failed','unknown','suppressed')\n             AND (job_json != '{{}}' OR content != '')",
        )
    )
    keep = max(0, int(MAX_TERMINAL_DELIVERIES))
    terminal_count = int(
        conn.execute(
            _query(
                conn,
                "SELECT COUNT(*) FROM deliveries "
                "WHERE status IN ('delivered','failed','unknown','suppressed')",
                "SELECT COUNT(*) FROM {deliveries} WHERE status IN ('delivered','failed','unknown','suppressed')",
            )
        ).fetchone()[0]
    )
    excess = terminal_count - keep
    if excess > 0:
        conn.execute(
            _query(
                conn,
                """INSERT OR IGNORE INTO delivery_tombstones
               (execution_id, terminal_status, finished_at)
               SELECT execution_id, status, finished_at FROM deliveries
               WHERE status IN ('delivered','failed','unknown','suppressed')
               ORDER BY julianday(finished_at), finished_at,
                        julianday(created_at), created_at, execution_id
               LIMIT ?""",
                "INSERT INTO {tombstones}\n               (execution_id, terminal_status, finished_at)\n               SELECT execution_id, status, finished_at FROM {deliveries}\n               WHERE status IN ('delivered','failed','unknown','suppressed')\n               ORDER BY finished_at::timestamptz NULLS FIRST, finished_at NULLS FIRST,\n                        created_at::timestamptz, created_at, execution_id\n               LIMIT %s ON CONFLICT DO NOTHING",
            ),
            (excess,),
        )
        conn.execute(
            _query(
                conn,
                """DELETE FROM deliveries WHERE execution_id IN (
                 SELECT execution_id FROM deliveries
                 WHERE status IN ('delivered','failed','unknown','suppressed')
                 ORDER BY julianday(finished_at), finished_at,
                          julianday(created_at), created_at, execution_id
                 LIMIT ?
               )""",
                "DELETE FROM {deliveries} WHERE execution_id IN (\n                 SELECT execution_id FROM {deliveries}\n                 WHERE status IN ('delivered','failed','unknown','suppressed')\n                 ORDER BY finished_at::timestamptz NULLS FIRST, finished_at NULLS FIRST,\n                          created_at::timestamptz, created_at, execution_id\n                 LIMIT %s\n               )",
            ),
            (excess,),
        )


def queue_path(home: Optional[Path] = None) -> Path:
    """The queue file of ``home`` (the active home when None); a test override wins."""
    if DELIVERY_DB is not None:
        return DELIVERY_DB
    root = Path(home) if home is not None else get_hermes_home()
    return root.resolve() / "cron" / "deliveries.db"


def _path() -> Path:
    return queue_path()


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_cli.sqlite_util import add_column_if_missing

    # SQLite cannot widen a CHECK in place. Preserve all old rows atomically,
    # including claimed sends, while admitting a distinct never-sent disposition.
    for table in ("deliveries", "delivery_tombstones"):
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if row and "'suppressed'" not in row[0]:
            conn.execute("SAVEPOINT notification_disposition")
            try:
                conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
                conn.execute(row[0].replace("'unknown'", "'unknown','suppressed'"))
                conn.execute(f"INSERT INTO {table} SELECT * FROM {table}_old")
                conn.execute(f"DROP TABLE {table}_old")
                conn.execute("RELEASE notification_disposition")
            except BaseException:
                conn.execute("ROLLBACK TO notification_disposition")
                conn.execute("RELEASE notification_disposition")
                raise

    conn.execute(
        """CREATE TABLE IF NOT EXISTS deliveries (
             execution_id TEXT PRIMARY KEY,
             job_json TEXT NOT NULL,
             content TEXT NOT NULL,
             for_failure INTEGER NOT NULL DEFAULT 0,
             status TEXT NOT NULL CHECK(status IN
               ('pending','delivering','delivered','failed','unknown','suppressed')),
             owner_process_id TEXT,
             owner_pid INTEGER,
             owner_started_at INTEGER,
             created_at TEXT NOT NULL,
             finished_at TEXT,
             error TEXT
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_tombstones (
             execution_id TEXT PRIMARY KEY,
             terminal_status TEXT NOT NULL CHECK(terminal_status IN
               ('delivered','failed','unknown','suppressed')),
             finished_at TEXT
           )"""
    )
    add_column_if_missing(
        conn,
        "deliveries",
        "for_failure",
        "for_failure INTEGER NOT NULL DEFAULT 0",
    )


def _query(conn, sqlite, postgresql):
    if not postgres_enabled():
        return sqlite
    from psycopg import sql

    return sql.SQL(postgresql).format(
        deliveries=sql.Identifier(conn.hermes_scope.schema, "deliveries"),
        tombstones=sql.Identifier(conn.hermes_scope.schema, "delivery_tombstones"),
    )


def _begin_write(conn):
    if not postgres_enabled():
        conn.execute("BEGIN IMMEDIATE")


def _content(value):
    return value.encode("utf-8") if postgres_enabled() else value


def _active_key(execution_id):
    return (
        (resolve_postgres_scope(_path().parent.parent).schema, execution_id)
        if postgres_enabled()
        else execution_id
    )


def _now_iso():
    value = _hermes_now()
    if postgres_enabled():
        from datetime import UTC

        value = value.astimezone(UTC)
    return value.isoformat()


def _delivery_row_factory(cursor):
    names = tuple(column.name for column in cursor.description or ())
    return lambda values: NativeRow(
        names,
        tuple(
            bytes(value).decode("utf-8")
            if name == "content" and value is not None
            else value
            for name, value in zip(names, values)
        ),
    )


def _connect(*, read_only=False):
    if postgres_enabled():
        conn = connect_postgres(_path().parent.parent, read_only=read_only)
        conn.row_factory = _delivery_row_factory
        return conn
    from hermes_cli.sqlite_util import open_db

    path = _path()
    conn = open_db(
        path,
        db_label="cron/deliveries.db",
        synchronous_full=True,
        initialize=_initialize_schema,
    )
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return conn


@contextmanager
def _transaction(*, read_only=False):
    if postgres_enabled():
        with _lock, closing(_connect(read_only=read_only)) as conn:
            with conn.transaction() if read_only else native_write_transaction(conn):
                yield conn
        return
    from hermes_cli.sqlite_util import transaction

    with _lock, transaction(_connect()) as conn:
        yield conn


def enqueue(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
) -> dict:
    """Persist one idempotent delivery request before the worker waits."""
    with _transaction() as conn:
        # Serialize the tombstone check and insert with retention in other
        # processes, which can move a terminal delivery into the tombstone table.
        _begin_write(conn)
        tombstone = conn.execute(
            _query(
                conn,
                "SELECT terminal_status, finished_at FROM delivery_tombstones "
                "WHERE execution_id=?",
                "SELECT terminal_status, finished_at FROM {tombstones} WHERE execution_id=%s",
            ),
            (str(execution_id),),
        ).fetchone()
        if tombstone is not None:
            return {
                "execution_id": str(execution_id),
                "status": tombstone["terminal_status"],
                "finished_at": tombstone["finished_at"],
            }
        conn.execute(
            _query(
                conn,
                """INSERT OR IGNORE INTO deliveries
               (execution_id, job_json, content, for_failure, status, created_at)
               VALUES (?, ?, ?, ?, 'pending', ?)""",
                "INSERT INTO {deliveries}\n               (execution_id, job_json, content, for_failure, status, created_at)\n               VALUES (%s, %s, %s, %s, 'pending', %s) ON CONFLICT DO NOTHING",
            ),
            (
                str(execution_id),
                json.dumps(job, ensure_ascii=False, sort_keys=True),
                _content(str(content)),
                int(bool(for_failure)),
                _now_iso(),
            ),
        )
        row = conn.execute(
            _query(
                conn,
                "SELECT * FROM deliveries WHERE execution_id=?",
                "SELECT * FROM {deliveries} WHERE execution_id=%s",
            ),
            (str(execution_id),),
        ).fetchone()
    return dict(row)


def get_status(execution_id: str) -> Optional[dict]:
    with _transaction(read_only=True) as conn:
        row = conn.execute(
            _query(
                conn,
                "SELECT * FROM deliveries WHERE execution_id=?",
                "SELECT * FROM {deliveries} WHERE execution_id=%s",
            ),
            (str(execution_id),),
        ).fetchone()
        if row is not None:
            return dict(row)
        tombstone = conn.execute(
            _query(
                conn,
                "SELECT execution_id, terminal_status, finished_at "
                "FROM delivery_tombstones WHERE execution_id=?",
                "SELECT execution_id, terminal_status, finished_at FROM {tombstones} WHERE execution_id=%s",
            ),
            (str(execution_id),),
        ).fetchone()
    if tombstone is None:
        return None
    return {
        "execution_id": tombstone["execution_id"],
        "status": tombstone["terminal_status"],
        "finished_at": tombstone["finished_at"],
        "error": None,
    }


def claim_next() -> Optional[dict]:
    """Atomically claim one pending send before touching the transport."""
    pid = os.getpid()
    started = _process_start_time(pid)
    with _transaction() as conn:
        # created_at carries a DST-varying offset: order by instant, not text.
        row = conn.execute(
            _query(
                conn,
                "SELECT execution_id FROM deliveries WHERE status='pending' "
                "ORDER BY julianday(created_at), created_at, execution_id LIMIT 1",
                "SELECT execution_id FROM {deliveries} WHERE status='pending' ORDER BY created_at::timestamptz, created_at, execution_id LIMIT 1",
            )
        ).fetchone()
        if row is None:
            return None
        cur = conn.execute(
            _query(
                conn,
                """UPDATE deliveries SET status='delivering', owner_process_id=?,
               owner_pid=?, owner_started_at=?
               WHERE execution_id=? AND status='pending'""",
                "UPDATE {deliveries} SET status='delivering', owner_process_id=%s,\n               owner_pid=%s, owner_started_at=%s,owner_host=current_setting('hermes.owner_host',true)\n               WHERE execution_id=%s AND status='pending'",
            ),
            (_PROCESS_ID, pid, started, row["execution_id"]),
        )
        if cur.rowcount != 1:
            return None
        claimed = conn.execute(
            _query(
                conn,
                "SELECT * FROM deliveries WHERE execution_id=?",
                "SELECT * FROM {deliveries} WHERE execution_id=%s",
            ),
            (row["execution_id"],),
        ).fetchone()
        _ACTIVE_DELIVERIES.add(_active_key(row["execution_id"]))
    result = dict(claimed)
    result["job"] = json.loads(result.pop("job_json"))
    return result


def _finish(
    execution_id: str, *, error: Optional[str], suppressed: bool = False
) -> bool:
    status = "failed" if error else "suppressed" if suppressed else "delivered"
    safe_error = (
        redact_sensitive_text(str(error), force=True, redact_url_credentials=True)
        if error
        else None
    )
    with _transaction() as conn:
        cur = conn.execute(
            _query(
                conn,
                """UPDATE deliveries SET status=?, finished_at=?, error=?
               WHERE execution_id=? AND status='delivering'
                 AND owner_process_id=? AND owner_pid=?""",
                "UPDATE {deliveries} SET status=%s, finished_at=%s, error=%s\n               WHERE execution_id=%s AND status='delivering'\n                 AND owner_process_id=%s AND owner_pid=%s AND owner_host=current_setting('hermes.owner_host',true)",
            ),
            (
                status,
                _now_iso(),
                safe_error,
                execution_id,
                _PROCESS_ID,
                os.getpid(),
            ),
        )
        _prune_terminal_unlocked(conn)
    return cur.rowcount == 1


def recover_abandoned() -> int:
    """Fence dead delivery owners as unknown; never replay uncertain sends.

    The firer settled long ago against a ``queued`` row, so each fenced row's outcome is written
    back here (job ``delivery_failed`` + error, ledger ``failed``) instead of leaving the run at
    ``delivery_queued`` forever."""
    fenced: list[tuple[Optional[str], str]] = []
    with _transaction() as conn:
        rows = conn.execute(
            _query(
                conn,
                "SELECT execution_id, job_json, owner_process_id, owner_pid, owner_started_at "
                "FROM deliveries WHERE status='delivering'",
                "SELECT execution_id, job_json, owner_process_id, owner_pid, owner_started_at FROM {deliveries} WHERE status='delivering' AND owner_host=current_setting('hermes.owner_host',true)",
            )
        ).fetchall()
        for row in rows:
            same_process = row["owner_process_id"] == _PROCESS_ID
            if same_process:
                with _lock:
                    if _active_key(row["execution_id"]) in _ACTIVE_DELIVERIES:
                        continue
            elif _owner_is_live(int(row["owner_pid"]), row["owner_started_at"]):
                continue
            error = (
                "Gateway finished delivery but could not persist its outcome; "
                "send was not retried."
                if same_process
                else "Gateway exited during delivery; send outcome is unknown and was not retried."
            )
            cur = conn.execute(
                _query(
                    conn,
                    """UPDATE deliveries SET status='unknown', finished_at=?, error=?
                   WHERE execution_id=? AND status='delivering'""",
                    "UPDATE {deliveries} SET status='unknown', finished_at=%s, error=%s\n                   WHERE execution_id=%s AND status='delivering'",
                ),
                (
                    _now_iso(),
                    error,
                    row["execution_id"],
                ),
            )
            if cur.rowcount:
                # Read the job id now: pruning below redacts a terminal row's job_json.
                fenced.append(
                    (json.loads(row["job_json"] or "{}").get("id"), row["execution_id"])
                )
        _prune_terminal_unlocked(conn)
    from cron.delivery_outcome import settle_quietly

    for job_id, execution_id in fenced:
        settle_quietly(job_id, execution_id)
    return len(fenced)


def drain(send: Callable[[dict, str, bool], Optional[str]], *, limit: int = 20) -> int:
    """Deliver pending rows through *send*, terminalizing every claimed row."""
    recover_abandoned()
    processed = 0
    for _ in range(max(0, limit)):
        row = claim_next()
        if row is None:
            break
        with _lock:
            _ACTIVE_DELIVERIES.add(_active_key(row["execution_id"]))
        try:
            try:
                error = send(row["job"], row["content"], bool(row["for_failure"]))
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            if _finish(
                row["execution_id"],
                error=error,
                suppressed=bool(row["job"].get("_notification_all_targets_suppressed")),
            ):
                from cron.delivery_outcome import settle_quietly

                settle_quietly(row["job"].get("id"), row["execution_id"])
        finally:
            with _lock:
                _ACTIVE_DELIVERIES.discard(_active_key(row["execution_id"]))
        processed += 1
    return processed


def _terminalize_wait_timeout(execution_id: str) -> str:
    """Fence a delivery whose worker can no longer wait for confirmation.

    A row still ``pending`` was provably never attempted, so it is left queued
    for whichever gateway comes up next (a restart that includes an update can
    easily exceed the worker's wait budget).  That is a deferral, not a
    failure: report success so the job is not recorded ``delivery_failed`` for
    a message the drain will still send.  Only a row caught mid-send is
    uncertain and gets fenced ``unknown``.
    """
    now = _now_iso()
    uncertain_error = (
        "timed out while gateway delivery was in progress; outcome is unknown and "
        "was not retried"
    )
    with _transaction() as conn:
        row = conn.execute(
            _query(
                conn,
                "SELECT status FROM deliveries WHERE execution_id=?",
                "SELECT status FROM {deliveries} WHERE execution_id=%s",
            ),
            (str(execution_id),),
        ).fetchone()
        if row is not None and row["status"] == "pending":
            logger.warning(
                "Cron delivery %s: no live gateway within the wait budget; "
                "left queued for the next gateway",
                execution_id,
            )
            return ""
        conn.execute(
            _query(
                conn,
                """UPDATE deliveries SET status='unknown', finished_at=?, error=?
               WHERE execution_id=? AND status='delivering'""",
                "UPDATE {deliveries} SET status='unknown', finished_at=%s, error=%s\n               WHERE execution_id=%s AND status='delivering'",
            ),
            (now, uncertain_error, str(execution_id)),
        )
        row = conn.execute(
            _query(
                conn,
                "SELECT status, error FROM deliveries WHERE execution_id=?",
                "SELECT status, error FROM {deliveries} WHERE execution_id=%s",
            ),
            (str(execution_id),),
        ).fetchone()
        _prune_terminal_unlocked(conn)
    if row is None:
        return "timed out waiting for live gateway delivery"
    if row["status"] in {"delivered", "suppressed"}:
        return ""
    return str(row["error"] or f"delivery {row['status']}")


def enqueue_and_wait(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Queue delivery and wait for a gateway's terminal at-most-once outcome."""
    queued = enqueue(execution_id, job, content, for_failure=for_failure)
    if queued["status"] in _TERMINAL:
        return (
            None
            if queued["status"] in {"delivered", "suppressed"}
            else str(queued.get("error") or f"delivery {queued['status']}")
        )
    wait_timeout = (
        DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS if timeout is None else max(0.0, timeout)
    )
    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        row = get_status(execution_id)
        if row and row["status"] in _TERMINAL:
            return (
                None
                if row["status"] in {"delivered", "suppressed"}
                else str(row.get("error") or f"delivery {row['status']}")
            )
        time.sleep(1.0)
    return _terminalize_wait_timeout(execution_id) or None
