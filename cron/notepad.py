"""Per-job durable KV notepad (cursors, watermarks) carried across cron wake-ups; profile-local
SQLite next to the executions ledger (same connection/pragma pattern as ``cron/executions.py``).

Caps are a documented contract: ``MAX_VALUE_BYTES`` (16 KB per value, UTF-8) and
``MAX_JOB_TOTAL_BYTES`` (64 KB per job, key+value). Oversized writes raise ``ValueError`` and leave
the store untouched — the notepad is prompt-injected each run. Write path is the CLI
(``hermes cron notepad <job_id> set ...``) via the terminal tool; no model tool is added.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager, closing
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_state_backend import postgres_enabled
from hermes_state_sql import connect_postgres, native_write_transaction, NativeRow

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

# Optional test override. Production resolves the path at transaction time so multiplexed profile
# ticks (set_hermes_home_override) cannot leak one profile's notepad rows into the import-time home
# — and remove_job's clear_notepad cannot wipe the wrong profile's DB.
# Same pattern as cron/executions.py. See #86519.
NOTEPAD_FILE: Optional[Path] = None
MAX_VALUE_BYTES = 16 * 1024
MAX_KEY_CHARS = 128
MAX_JOB_TOTAL_BYTES = 64 * 1024
_lock = threading.RLock()


def _current_notepad_file() -> Path:
    return NOTEPAD_FILE or (get_hermes_home().resolve() / "cron" / "notepad.db")


def _query(conn, sqlite, postgresql):
    if not postgres_enabled():
        return sqlite
    from psycopg import sql

    return sql.SQL(postgresql).format(
        notes=sql.Identifier(conn.hermes_scope.schema, "cron_notepad")
    )


def _value(value):
    return (
        value.encode("utf-8")
        if postgres_enabled() and isinstance(value, str)
        else value
    )


def _note_row_factory(cursor):
    names = tuple(column.name for column in cursor.description or ())
    return lambda values: NativeRow(
        names,
        tuple(
            bytes(value).decode("utf-8")
            if name in {"job_id", "key", "value"} and value is not None
            else value
            for name, value in zip(names, values)
        ),
    )


def _connect(*, read_only=False):
    if postgres_enabled():
        conn = connect_postgres(
            _current_notepad_file().parent.parent, read_only=read_only
        )
        conn.row_factory = _note_row_factory
        return conn
    from cron.jobs import _ensure_cron_dir
    from hermes_cli.sqlite_util import open_db

    path = _current_notepad_file()
    _ensure_cron_dir(path.parent)
    return open_db(path, db_label="cron/notepad.db", initialize=_initialize_schema)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cron_notepad (
             job_id TEXT NOT NULL,
             key TEXT NOT NULL,
             value TEXT NOT NULL,
             updated_at TEXT NOT NULL,
             PRIMARY KEY (job_id, key)
           )"""
    )


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


def _validate(job_id: str, key: str, value: str) -> None:
    if not str(job_id):
        raise ValueError("job_id must be non-empty")
    if not key:
        raise ValueError("key must be non-empty")
    if len(key) > MAX_KEY_CHARS:
        raise ValueError(f"key too long (max {MAX_KEY_CHARS} characters)")
    if len(value.encode("utf-8")) > MAX_VALUE_BYTES:
        raise ValueError(f"value too large (max {MAX_VALUE_BYTES} bytes per key)")


def set_note(job_id: str, key: str, value: str) -> dict[str, Any]:
    """Upsert one key. Raises ValueError when a size cap would be exceeded."""
    job_id, key, value = str(job_id), str(key), str(value)
    _validate(job_id, key, value)
    now = _hermes_now()
    if postgres_enabled():
        from datetime import UTC

        now = now.astimezone(UTC)
    now = now.isoformat()
    with _transaction() as conn:
        row = conn.execute(
            _query(
                conn,
                """SELECT COALESCE(SUM(LENGTH(CAST(key AS BLOB))
                 + LENGTH(CAST(value AS BLOB))), 0)
               FROM cron_notepad WHERE job_id=? AND key<>?""",
                "SELECT COALESCE(SUM(octet_length(key)+octet_length(value)),0) FROM {notes} WHERE job_id=%s AND key<>%s",
            ),
            (_value(job_id), _value(key)),
        ).fetchone()
        other_bytes = int(row[0])
        entry_bytes = len(key.encode("utf-8")) + len(value.encode("utf-8"))
        if other_bytes + entry_bytes > MAX_JOB_TOTAL_BYTES:
            raise ValueError(
                f"notepad full: job '{job_id}' would exceed "
                f"{MAX_JOB_TOTAL_BYTES} bytes total; delete unused keys first"
            )
        conn.execute(
            _query(
                conn,
                """INSERT INTO cron_notepad (job_id, key, value, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(job_id, key)
               DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                "INSERT INTO {notes}(job_id,key,value,updated_at) VALUES(%s,%s,%s,%s) ON CONFLICT(job_id,key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
            ),
            (_value(job_id), _value(key), _value(value), now),
        )
    return {"job_id": job_id, "key": key, "value": value, "updated_at": now}


def get_note(job_id: str, key: str) -> Optional[str]:
    with _transaction(read_only=True) as conn:
        row = conn.execute(
            _query(
                conn,
                "SELECT value FROM cron_notepad WHERE job_id=? AND key=?",
                "SELECT value FROM {notes} WHERE job_id=%s AND key=%s",
            ),
            (_value(str(job_id)), _value(str(key))),
        ).fetchone()
    return None if row is None else row["value"]


def delete_note(job_id: str, key: str) -> bool:
    with _transaction() as conn:
        cur = conn.execute(
            _query(
                conn,
                "DELETE FROM cron_notepad WHERE job_id=? AND key=?",
                "DELETE FROM {notes} WHERE job_id=%s AND key=%s",
            ),
            (_value(str(job_id)), _value(str(key))),
        )
    return cur.rowcount > 0


def list_notes(job_id: str) -> list[dict[str, Any]]:
    """All entries for one job, sorted by key."""
    with _transaction(read_only=True) as conn:
        rows = conn.execute(
            _query(
                conn,
                "SELECT job_id, key, value, updated_at FROM cron_notepad "
                "WHERE job_id=? ORDER BY key",
                "SELECT job_id,key,value,updated_at FROM {notes} WHERE job_id=%s ORDER BY key",
            ),
            (_value(str(job_id)),),
        ).fetchall()
    return [dict(row) for row in rows]


def clear_notepad(job_id: str) -> int:
    """Delete every key for one job (called from ``cron.jobs.remove_job``). Returns row count;
    no-ops without creating the DB when no notepad file exists yet."""
    if not postgres_enabled() and not _current_notepad_file().exists():
        return 0
    with _transaction() as conn:
        cur = conn.execute(
            _query(
                conn,
                "DELETE FROM cron_notepad WHERE job_id=?",
                "DELETE FROM {notes} WHERE job_id=%s",
            ),
            (_value(str(job_id)),),
        )
    return cur.rowcount


def render_notepad_section(job_id: str) -> str:
    """Render a job's notepad as a prompt section. An empty notepad MUST return '' so jobs that
    never use the feature get a byte-identical prompt (prompt-cache + drift safety)."""
    try:
        notes = list_notes(job_id)
    except Exception:
        if postgres_enabled():
            raise
        return ""
    if not notes:
        return ""
    lines = [f"- {note['key']}: {note['value']}" for note in notes]
    return (
        "## Job notepad (persistent across runs)\n"
        "This durable scratchpad survives between scheduled runs of this "
        "job. Update it via the CLI, e.g.:\n"
        f"`hermes cron notepad {job_id} set <key> <value>` "
        f"(also: get/delete/list; `hermes cron notepad {job_id} delete "
        "<key>` removes an entry).\n\n" + "\n".join(lines) + "\n\n"
    )
