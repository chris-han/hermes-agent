"""Durable bounded policy projection for hosted Group Chat preparation.

The append-only room log remains the user-visible source of truth. This module materializes only the state
needed to choose and reconstruct the next active discussion, so a busy room does not replay its complete
history every poll.
"""

from __future__ import annotations
from hermes_state_sql import statement as _storage_statement, parameter_marker as _parameter_marker, inserted_id as _inserted_id, in_transaction as _in_transaction

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from gateway import hosted_rooms
from gateway.hosted_room_task_input import validate_task_input
from gateway.hosted_rooms_common import DbPath, compact_json, fenced_update


MAX_ACTIVE_POLICY_EVENTS = 64
MAX_THREAD_TRANSCRIPT_EVENTS = 24
_TRANSCRIPT_SCHEMA_VERSION = 4
MAX_TRANSCRIPT_POLICY_EVENTS = MAX_THREAD_TRANSCRIPT_EVENTS * (MAX_ACTIVE_POLICY_EVENTS + 2)
_TERMINAL_KINDS = frozenset({"turn.settled", "turn.failed", "turn.cancelled", "turn.deferred"})

_SCHEMA_DDL = (
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_cursors (
        room_id TEXT PRIMARY KEY, through_seq INTEGER NOT NULL DEFAULT 0,
        stopped_through_seq INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_threads (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, discussion_event_id TEXT NOT NULL,
        latest_user_seq INTEGER NOT NULL, completed INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(room_id, thread_id))""",
    """CREATE INDEX IF NOT EXISTS idx_hosted_room_policy_pending
        ON hosted_room_policy_threads(room_id, completed, latest_user_seq, thread_id)""",
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_events (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, discussion_event_id TEXT NOT NULL,
        seq INTEGER NOT NULL, event_json TEXT NOT NULL, PRIMARY KEY(room_id, seq))""",
    """CREATE INDEX IF NOT EXISTS idx_hosted_room_policy_events_active
        ON hosted_room_policy_events(room_id, discussion_event_id, seq)""",
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_watermarks (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, member_id TEXT NOT NULL,
        seen_through_seq INTEGER NOT NULL, PRIMARY KEY(room_id, thread_id, member_id))""",
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_publications (
        room_id TEXT NOT NULL, task_id TEXT NOT NULL, kind TEXT NOT NULL,
        execution_generation INTEGER NOT NULL DEFAULT 0, seq INTEGER NOT NULL,
        PRIMARY KEY(room_id, task_id, kind, execution_generation))""",
    # Transcript stores only references into the already bounded room log, so
    # prompt payloads are never duplicated outside room byte limits.
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_transcript (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, seq INTEGER NOT NULL,
        kind TEXT NOT NULL, settled_seq INTEGER, PRIMARY KEY(room_id, thread_id, seq))""",
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_transcript_state (
        room_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL)""",
    """CREATE INDEX IF NOT EXISTS idx_hosted_room_user_thread
       ON hosted_room_events(room_id, json_extract(payload_json, '$.thread_id'), seq)
       WHERE kind='message.user'""",
)

_ROOM_EVENT_COLUMNS = hosted_rooms._EVENT_COLUMNS
_DELETE_ACTIVE_EVENTS_SQL = (_storage_statement(sqlite="DELETE FROM hosted_room_policy_events WHERE room_id=? AND discussion_event_id=?",postgresql='DELETE FROM hosted_room_policy_events WHERE room_id=%s AND discussion_event_id=%s'))
_TRANSCRIPT_EVENTS_SQL = _storage_statement(sqlite=f"""WITH transcript_events(seq) AS (
        SELECT seq FROM hosted_room_policy_transcript WHERE room_id=? AND thread_id=?
        UNION ALL
        SELECT settled_seq FROM hosted_room_policy_transcript
         WHERE room_id=? AND thread_id=? AND settled_seq IS NOT NULL)
    SELECT {", ".join("events." + column for column in _ROOM_EVENT_COLUMNS.split(", "))}
    FROM transcript_events
    JOIN hosted_room_events AS events ON events.room_id=? AND events.seq=transcript_events.seq
    ORDER BY events.seq LIMIT ?""",postgresql=f"WITH transcript_events(seq) AS (\n        SELECT seq FROM hosted_room_policy_transcript WHERE room_id=%s AND thread_id=%s\n        UNION ALL\n        SELECT settled_seq FROM hosted_room_policy_transcript\n         WHERE room_id=%s AND thread_id=%s AND settled_seq IS NOT NULL)\n    SELECT {', '.join(('events.' + column for column in _ROOM_EVENT_COLUMNS.split(', ')))}\n    FROM transcript_events\n    JOIN hosted_room_events AS events ON events.room_id=%s AND events.seq=transcript_events.seq\n    ORDER BY events.seq LIMIT %s")


@dataclass(frozen=True)
class PolicySnapshot:
    """Bounded active policy input at one durable room-log cursor."""
    through_seq: int
    stopped_through_seq: int
    events: tuple[dict[str, Any], ...]
    watermarks: Mapping[tuple[str, str], int]


_event_from_room_row = hosted_rooms._event_from_row


def _text(mapping: Mapping[str, Any], key: str) -> str:
    return str(mapping.get(key) or "")


def _require_room(conn: sqlite3.Connection, room_id: str) -> None:
    if conn.execute(_storage_statement(sqlite="SELECT 1 FROM hosted_rooms WHERE room_id=?",postgresql='SELECT 1 FROM hosted_rooms WHERE room_id=%s'), (room_id,)).fetchone() is None:
        raise hosted_rooms.RoomNotFoundError("hosted room not found")


def _settled_message(
    conn: sqlite3.Connection, room_id: str, discussion_event_id: str, message_event_id: Any) -> dict[str, Any] | None:
    """Read the committed message even after its active discussion was compacted."""
    row = conn.execute(
        _storage_statement(sqlite=f"SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=? AND event_id=?",postgresql=f'SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=%s AND event_id=%s'),
        (room_id, message_event_id)).fetchone()
    if row is None:
        return None
    event = _event_from_room_row(row)
    if event["kind"] != "message.member" or _text(event["payload"], "discussion_event_id") != discussion_event_id:
        return None
    return event


class HostedRoomPolicyCheckpoint:
    """Incrementally index room policy without compacting visible history."""
    def __init__(self, db_path: DbPath) -> None:
        self.db_path = Path(db_path)
        from hermes_state_backend import postgres_enabled
        if postgres_enabled():
            from gateway.hosted_rooms_common import open_sqlite,table_exists
            with open_sqlite(self.db_path) as conn:
                if not table_exists(conn,"hosted_room_policy_cursors"):
                    raise RuntimeError("Provisioned hosted-room policy schema required")
            return
        with self._transaction() as conn:
            for ddl in _SCHEMA_DDL:
                conn.execute(ddl)

    def _connect(self) -> sqlite3.Connection:
        from hermes_state_backend import postgres_enabled
        if postgres_enabled():
            from gateway.hosted_rooms_common import open_sqlite
            return open_sqlite(self.db_path)
        # Late import: a gateway that outlives an on-disk upgrade has the OLD sqlite_util cached.
        from hermes_cli.sqlite_util import open_db

        return open_db(self.db_path, db_label="shared-state.db (room policy checkpoint)", busy_timeout_ms=10_000)

    def _transaction(self):
        from hermes_state_backend import postgres_enabled
        if postgres_enabled():
            from gateway.hosted_rooms_common import transaction
            return transaction(lambda _:self._connect(),self.db_path,immediate=True)
        from hermes_cli.sqlite_util import transaction
        return transaction(self._connect())

    @staticmethod
    def _store_active_event(
        conn: sqlite3.Connection, *, event: Mapping[str, Any], thread_id: str, discussion_event_id: str) -> None:
        conn.execute(_storage_statement(sqlite="""INSERT OR IGNORE INTO hosted_room_policy_events(
                   room_id, thread_id, discussion_event_id, seq, event_json
               ) VALUES (?, ?, ?, ?, ?)""",postgresql='INSERT INTO hosted_room_policy_events(\n                   room_id, thread_id, discussion_event_id, seq, event_json\n               ) VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING'),
            (event["room_id"], thread_id, discussion_event_id, int(event["seq"]), compact_json(dict(event))))

    @staticmethod
    def _store_transcript_event(
        conn: sqlite3.Connection, *, event: Mapping[str, Any], thread_id: str, settled_seq: int | None = None) -> None:
        conn.execute(_storage_statement(sqlite="""INSERT INTO hosted_room_policy_transcript(
                   room_id, thread_id, seq, kind, settled_seq
               ) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(room_id, thread_id, seq) DO UPDATE SET
                   settled_seq=COALESCE(excluded.settled_seq, hosted_room_policy_transcript.settled_seq)""",postgresql='INSERT INTO hosted_room_policy_transcript(\n                   room_id, thread_id, seq, kind, settled_seq\n               ) VALUES (%s, %s, %s, %s, %s)\n               ON CONFLICT(room_id, thread_id, seq) DO UPDATE SET\n                   settled_seq=COALESCE(excluded.settled_seq, hosted_room_policy_transcript.settled_seq)'),
            (event["room_id"], thread_id, int(event["seq"]), str(event["kind"]), settled_seq))
        if event["kind"] in {"message.user", "message.member"}:
            cutoff = conn.execute(_storage_statement(sqlite="""SELECT seq FROM hosted_room_policy_transcript
                   WHERE room_id=? AND thread_id=? AND kind IN ('message.user', 'message.member')
                   ORDER BY seq DESC LIMIT 1 OFFSET ?""",postgresql="SELECT seq FROM hosted_room_policy_transcript\n                   WHERE room_id=%s AND thread_id=%s AND kind IN ('message.user', 'message.member')\n                   ORDER BY seq DESC LIMIT 1 OFFSET %s"),
                (event["room_id"], thread_id, MAX_THREAD_TRANSCRIPT_EVENTS - 1)).fetchone()
            if cutoff is not None:
                conn.execute(
                    _storage_statement(sqlite="DELETE FROM hosted_room_policy_transcript WHERE room_id=? AND thread_id=? AND seq<?",postgresql='DELETE FROM hosted_room_policy_transcript WHERE room_id=%s AND thread_id=%s AND seq<%s'),
                    (event["room_id"], thread_id, int(cutoff["seq"])))

    def _discussion_events(
        self, conn: sqlite3.Connection, *, room_id: str, thread_id: str, discussion_event_id: str, bound_error: str
    ) -> list[dict[str, Any]]:
        """Merge the thread transcript with the active projection, ordered by seq."""
        active_rows = conn.execute(_storage_statement(sqlite="""SELECT event_json FROM hosted_room_policy_events
               WHERE room_id=? AND discussion_event_id=? ORDER BY seq LIMIT ?""",postgresql='SELECT event_json FROM hosted_room_policy_events\n               WHERE room_id=%s AND discussion_event_id=%s ORDER BY seq LIMIT %s'),
            (room_id, discussion_event_id, MAX_ACTIVE_POLICY_EVENTS + 1)).fetchall()
        if len(active_rows) > MAX_ACTIVE_POLICY_EVENTS:
            raise RuntimeError(bound_error)
        rows = conn.execute(_TRANSCRIPT_EVENTS_SQL,
                            (room_id, thread_id, room_id, thread_id, room_id, MAX_TRANSCRIPT_POLICY_EVENTS + 1)).fetchall()
        if len(rows) > MAX_TRANSCRIPT_POLICY_EVENTS:
            raise RuntimeError("thread policy transcript exceeded its bound")
        events_by_seq = {
            int(event["seq"]): event
            for event in (*map(_event_from_room_row, rows), *(json.loads(row["event_json"]) for row in active_rows))}
        return [events_by_seq[seq] for seq in sorted(events_by_seq)]

    # -- per-kind projection handlers (dispatched by _apply_event) -----------

    def _apply_user_message(
        self, conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        room_id = str(event["room_id"])
        thread_id, event_id = _text(payload, "thread_id"), _text(event, "event_id")
        if not thread_id or not event_id:
            return
        conn.execute(_storage_statement(sqlite="""INSERT INTO hosted_room_policy_threads(
                   room_id, thread_id, discussion_event_id, latest_user_seq, completed
               ) VALUES (?, ?, ?, ?, 0)
               ON CONFLICT(room_id, thread_id) DO UPDATE SET
                   discussion_event_id=excluded.discussion_event_id,
                   latest_user_seq=excluded.latest_user_seq, completed=0""",postgresql='INSERT INTO hosted_room_policy_threads(\n                   room_id, thread_id, discussion_event_id, latest_user_seq, completed\n               ) VALUES (%s, %s, %s, %s, 0)\n               ON CONFLICT(room_id, thread_id) DO UPDATE SET\n                   discussion_event_id=excluded.discussion_event_id,\n                   latest_user_seq=excluded.latest_user_seq, completed=0'),
            (room_id, thread_id, event_id, int(event["seq"])))
        self._store_active_event(conn, event=event, thread_id=thread_id, discussion_event_id=event_id)
        self._store_transcript_event(conn, event=event, thread_id=thread_id)

    def _apply_discussion_event(
        self, conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        """Index member messages and terminal turn outcomes of a known discussion."""
        room_id, seq, kind = str(event["room_id"]), int(event["seq"]), _text(event, "kind")
        thread_id, discussion_event_id = _text(payload, "thread_id"), _text(payload, "discussion_event_id")
        if conn.execute(
            _storage_statement(sqlite="SELECT 1 FROM hosted_room_policy_events WHERE room_id=? AND discussion_event_id=? LIMIT 1",postgresql='SELECT 1 FROM hosted_room_policy_events WHERE room_id=%s AND discussion_event_id=%s LIMIT 1'),
            (room_id, discussion_event_id)).fetchone() is not None:
            self._store_active_event(conn, event=event, thread_id=thread_id, discussion_event_id=discussion_event_id)
        # Late outcomes still need publication receipts and transcript commits,
        # but must not resurrect a completed discussion's active projection.
        if kind not in _TERMINAL_KINDS:
            return
        task_id = _text(payload, "task_id")
        execution_generation = int(payload.get("execution_generation") or 0) if kind == "turn.deferred" else 0
        if task_id:
            conn.execute(_storage_statement(sqlite="""INSERT OR IGNORE INTO hosted_room_policy_publications(
                       room_id, task_id, kind, execution_generation, seq
                   ) VALUES (?, ?, ?, ?, ?)""",postgresql='INSERT INTO hosted_room_policy_publications(\n                       room_id, task_id, kind, execution_generation, seq\n                   ) VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING'),
                (room_id, task_id, kind, execution_generation, seq))
        member_id = _text(payload, "member_id")
        seen_through_seq = int(payload.get("seen_through_seq") or 0)
        if kind == "turn.settled" and payload.get("message_event_id"):
            committed = _settled_message(conn, room_id, discussion_event_id, payload["message_event_id"])
            if committed is not None:
                seen_through_seq = max(seen_through_seq, int(committed["seq"]))
                self._store_transcript_event(conn, event=committed, thread_id=thread_id, settled_seq=seq)
        else:
            # Non-visible receipts still supply historical reconstruction watermarks.
            self._store_transcript_event(conn, event=event, thread_id=thread_id)
        if member_id and seen_through_seq > 0:
            conn.execute(_storage_statement(sqlite="""INSERT INTO hosted_room_policy_watermarks(
                       room_id, thread_id, member_id, seen_through_seq
                   ) VALUES (?, ?, ?, ?)
                   ON CONFLICT(room_id, thread_id, member_id) DO UPDATE SET
                       seen_through_seq=MAX(hosted_room_policy_watermarks.seen_through_seq, excluded.seen_through_seq)""",postgresql='INSERT INTO hosted_room_policy_watermarks(\n                       room_id, thread_id, member_id, seen_through_seq\n                   ) VALUES (%s, %s, %s, %s)\n                   ON CONFLICT(room_id, thread_id, member_id) DO UPDATE SET\n                       seen_through_seq=MAX(hosted_room_policy_watermarks.seen_through_seq, excluded.seen_through_seq)'),
                (room_id, thread_id, member_id, seen_through_seq))

    def _apply_room_activity(
        self, conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        room_id, thread_id = str(event["room_id"]), _text(payload, "thread_id")
        conn.execute(_DELETE_ACTIVE_EVENTS_SQL, (room_id, _text(payload, "discussion_event_id")))
        conn.execute(_storage_statement(sqlite="DELETE FROM hosted_room_policy_threads WHERE room_id=? AND thread_id=?",postgresql='DELETE FROM hosted_room_policy_threads WHERE room_id=%s AND thread_id=%s'), (room_id, thread_id))

    def _apply_stop_requested(
        self, conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        conn.execute(_storage_statement(sqlite="""UPDATE hosted_room_policy_cursors
               SET stopped_through_seq=MAX(stopped_through_seq, ?) WHERE room_id=?""",postgresql='UPDATE hosted_room_policy_cursors\n               SET stopped_through_seq=MAX(stopped_through_seq, %s) WHERE room_id=%s'),
            (int(event["seq"]), str(event["room_id"])))

    _APPLY_BY_KIND: dict[str, Callable[..., None]] = {
        "message.user": _apply_user_message, "message.member": _apply_discussion_event,
        **dict.fromkeys(_TERMINAL_KINDS, _apply_discussion_event), "room.activity": _apply_room_activity,
        "room.stop_requested": _apply_stop_requested}

    def _apply_event(self, conn: sqlite3.Connection, event: Mapping[str, Any]) -> None:
        handler = self._APPLY_BY_KIND.get(_text(event, "kind"))
        if handler is not None:
            payload = event.get("payload")
            handler(self, conn, event, payload if isinstance(payload, Mapping) else {})

    def _ensure_cursor_and_transcript(self, conn: sqlite3.Connection, room_id: str) -> int:
        """Create the cursor and migrate derived policy caches by bounded replay."""
        _require_room(conn, room_id)
        conn.execute(_storage_statement(sqlite="""INSERT OR IGNORE INTO hosted_room_policy_cursors(
                   room_id, through_seq, stopped_through_seq, updated_at
               ) VALUES (?, 0, 0, 0)""",postgresql='INSERT INTO hosted_room_policy_cursors(\n                   room_id, through_seq, stopped_through_seq, updated_at\n               ) VALUES (%s, 0, 0, 0) ON CONFLICT DO NOTHING'), (room_id,))
        cursor = int(
            conn.execute(_storage_statement(sqlite="SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=?",postgresql='SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=%s'), (room_id,)).fetchone()[
                "through_seq"])
        transcript_state = conn.execute(
            _storage_statement(sqlite="SELECT schema_version FROM hosted_room_policy_transcript_state WHERE room_id=?",postgresql='SELECT schema_version FROM hosted_room_policy_transcript_state WHERE room_id=%s'), (room_id,)).fetchone()
        if transcript_state is None or int(transcript_state["schema_version"]) < _TRANSCRIPT_SCHEMA_VERSION:
            # Rebuild derived history without rewriting admissions or the room log.
            for table in ("hosted_room_policy_events", "hosted_room_policy_threads",
                          "hosted_room_policy_watermarks", "hosted_room_policy_transcript"):
                conn.execute(_storage_statement(sqlite=f"DELETE FROM {table} WHERE room_id=?",postgresql=f'DELETE FROM {table} WHERE room_id=%s'), (room_id,))
            conn.execute(_storage_statement(sqlite="UPDATE hosted_room_policy_cursors SET through_seq=0, stopped_through_seq=0 WHERE room_id=?",postgresql='UPDATE hosted_room_policy_cursors SET through_seq=0, stopped_through_seq=0 WHERE room_id=%s'),
                         (room_id,))
            cursor = 0
            conn.execute(_storage_statement(sqlite="""INSERT INTO hosted_room_policy_transcript_state(room_id, schema_version)
                   VALUES (?, ?)
                   ON CONFLICT(room_id) DO UPDATE SET schema_version=excluded.schema_version""",postgresql='INSERT INTO hosted_room_policy_transcript_state(room_id, schema_version)\n                   VALUES (%s, %s)\n                   ON CONFLICT(room_id) DO UPDATE SET schema_version=excluded.schema_version'),
                (room_id, _TRANSCRIPT_SCHEMA_VERSION))
        return cursor

    def sync(self, *, room_id: str, latest_seq: int) -> int:
        """Materialize each unseen event exactly once by durable cursor."""
        with self._transaction() as conn:
            from hermes_state_sql import is_postgres
            if not is_postgres(conn):
                conn.execute("BEGIN IMMEDIATE")
            cursor = self._ensure_cursor_and_transcript(conn, room_id)
        if cursor > latest_seq:
            with self._connect() as conn:
                room = conn.execute(_storage_statement(sqlite="SELECT next_seq FROM hosted_rooms WHERE room_id=?",postgresql='SELECT next_seq FROM hosted_rooms WHERE room_id=%s'), (room_id,)).fetchone()
                if room is None:
                    raise hosted_rooms.RoomNotFoundError("hosted room not found")
                if cursor >= int(room["next_seq"]):
                    raise RuntimeError("room policy cursor is ahead of the durable log")
        while cursor < latest_seq:
            page = hosted_rooms.read_events(
                self.db_path, room_id=room_id, since_seq=cursor, limit=hosted_rooms.MAX_LOG_LIMIT)
            rows = [event for event in page.get("events", []) if isinstance(event, Mapping)]
            next_cursor = int(page.get("cursor") or cursor)
            if not rows or next_cursor <= cursor:
                raise RuntimeError("hosted room policy cursor did not advance")
            with self._transaction() as conn:
                from hermes_state_sql import is_postgres
                if not is_postgres(conn):
                    conn.execute("BEGIN IMMEDIATE")
                _require_room(conn, room_id)
                current = conn.execute(_storage_statement(sqlite="SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=?",postgresql='SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=%s'),
                                       (room_id,)).fetchone()
                if current is None:
                    raise RuntimeError("room policy cursor disappeared during replay")
                if int(current["through_seq"]) != cursor:
                    # Another worker applied the page: reload before any projection effects.
                    cursor = int(current["through_seq"])
                    continue
                for event in rows:
                    self._apply_event(conn, event)
                fenced_update(
                    conn, _storage_statement(sqlite="UPDATE hosted_room_policy_cursors SET through_seq=?, updated_at=? WHERE room_id=?",postgresql='UPDATE hosted_room_policy_cursors SET through_seq=%s, updated_at=%s WHERE room_id=%s'),
                    (next_cursor, float(rows[-1].get("created_at") or 0), room_id),
                    RuntimeError("room policy cursor disappeared during replay"))
            cursor = next_cursor
        return cursor

    def snapshot(self, *, room_id: str, latest_seq: int) -> PolicySnapshot:
        """Return only the oldest active discussion and its watermark set."""
        self.sync(room_id=room_id, latest_seq=latest_seq)
        with self._transaction() as conn:
            cursor = conn.execute(
                _storage_statement(sqlite="SELECT through_seq, stopped_through_seq FROM hosted_room_policy_cursors WHERE room_id=?",postgresql='SELECT through_seq, stopped_through_seq FROM hosted_room_policy_cursors WHERE room_id=%s'), (room_id,)).fetchone()
            if cursor is None:
                raise hosted_rooms.RoomNotFoundError("hosted room checkpoint not found")
            through_seq = int(cursor["through_seq"])
            stopped_through_seq = int(cursor["stopped_through_seq"])
            thread = conn.execute(_storage_statement(sqlite="""SELECT thread_id, discussion_event_id FROM hosted_room_policy_threads
                   WHERE room_id=? AND completed=0 AND latest_user_seq>?
                   ORDER BY latest_user_seq, thread_id LIMIT 1""",postgresql='SELECT thread_id, discussion_event_id FROM hosted_room_policy_threads\n                   WHERE room_id=%s AND completed=0 AND latest_user_seq>%s\n                   ORDER BY latest_user_seq, thread_id LIMIT 1'), (room_id, stopped_through_seq)).fetchone()
            if thread is None:
                return PolicySnapshot(
                    through_seq=through_seq, stopped_through_seq=stopped_through_seq, events=(), watermarks={})
            thread_id = str(thread["thread_id"])
            events = self._discussion_events(
                conn, room_id=room_id, thread_id=thread_id, discussion_event_id=str(thread["discussion_event_id"]),
                bound_error="active room policy projection exceeded its bound")
            watermark_rows = conn.execute(_storage_statement(sqlite="""SELECT member_id, seen_through_seq FROM hosted_room_policy_watermarks
                   WHERE room_id=? AND thread_id=?""",postgresql='SELECT member_id, seen_through_seq FROM hosted_room_policy_watermarks\n                   WHERE room_id=%s AND thread_id=%s'), (room_id, thread_id)).fetchall()
        return PolicySnapshot(
            through_seq=through_seq, stopped_through_seq=stopped_through_seq, events=tuple(events),
            watermarks={(thread_id, str(row["member_id"])): int(row["seen_through_seq"]) for row in watermark_rows})

    def publication_exists(self, *, room_id: str, task_id: str, status: str, execution_generation: int) -> bool:
        """Return whether one exact driver outcome is already in the room log."""
        sql, params = (
            (_storage_statement(sqlite="""SELECT 1 FROM hosted_room_policy_publications
                     WHERE room_id=? AND task_id=? AND kind=? AND execution_generation=?""",postgresql='SELECT 1 FROM hosted_room_policy_publications\n                     WHERE room_id=%s AND task_id=%s AND kind=%s AND execution_generation=%s'),
             (room_id, task_id, f"turn.{status}", execution_generation))
            if status == "deferred" else
            (_storage_statement(sqlite="""SELECT 1 FROM hosted_room_policy_publications
                     WHERE room_id=? AND task_id=? AND kind IN ('turn.settled', 'turn.failed', 'turn.cancelled')""",postgresql="SELECT 1 FROM hosted_room_policy_publications\n                     WHERE room_id=%s AND task_id=%s AND kind IN ('turn.settled', 'turn.failed', 'turn.cancelled')"),
             (room_id, task_id)))
        with self._transaction() as conn:
            return conn.execute(sql, params).fetchone() is not None

    def events_for_task(self, *, room_id: str, source_event_seq: int,
                        input_context: Mapping[str, Any] | None = None,
                        task_id: str | None = None) -> list[dict[str, Any]]:
        """Load one bounded discussion projection for terminal reconstruction."""
        with self._transaction() as conn:
            published = self._published_task_events(conn, room_id=room_id, task_id=task_id)
            if input_context is not None:
                context = validate_task_input(input_context)
                seqs = sorted({source_event_seq, *context["event_seqs"]})
                placeholders = ",".join(_parameter_marker() for _ in seqs)
                rows = conn.execute(_storage_statement(sqlite=f"SELECT * FROM hosted_room_events WHERE room_id=? AND seq IN ({placeholders}) ORDER BY seq",postgresql=f'SELECT * FROM hosted_room_events WHERE room_id=%s AND seq IN ({placeholders}) ORDER BY seq'),
                                    (room_id, *seqs)).fetchall()
                if len(rows) != len(seqs):
                    raise RuntimeError("admitted task input event is missing")
                events = {int(row["seq"]): _event_from_room_row(row) for row in rows}
                source_event = events[source_event_seq]
                thread_id = _text(source_event["payload"], "thread_id")
                if source_event["kind"] != "message.user" or not thread_id:
                    raise RuntimeError("admitted task source is not a user discussion")
                # Supersession remains authoritative after the display projection compacts.
                latest = conn.execute(_storage_statement(sqlite="""SELECT * FROM hosted_room_events
                    WHERE room_id=? AND kind='message.user' AND json_extract(payload_json, '$.thread_id')=?
                    ORDER BY seq DESC LIMIT 1""",postgresql="SELECT * FROM hosted_room_events\n                    WHERE room_id=%s AND kind='message.user' AND (payload_json::jsonb->>'thread_id')=%s\n                    ORDER BY seq DESC LIMIT 1"), (room_id, thread_id)).fetchone()
                if latest is not None:
                    events[int(latest["seq"])] = _event_from_room_row(latest)
                for event in published:
                    events[int(event["seq"])] = event
                return [events[seq] for seq in sorted(events)]
            row = conn.execute(
                _storage_statement(sqlite=f"SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=? AND seq=?",postgresql=f'SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=%s AND seq=%s'),
                (room_id, source_event_seq)).fetchone()
            if row is None or row["kind"] != "message.user":
                return published
            source = _event_from_room_row(row)
            events = self._discussion_events(
                conn, room_id=room_id, thread_id=_text(source["payload"], "thread_id"),
                discussion_event_id=str(source["event_id"]),
                bound_error="task policy projection exceeded its bound")
            # The source can age out of BOTH bounded projections while a
            # deferred task remains retryable. Its frozen prompt lives in the task.
            by_seq = {event["seq"]: event for event in (*events, *published)}
            by_seq[source_event_seq] = source
            return [by_seq[seq] for seq in sorted(by_seq)]

    def compact_completed(self, *, room_id: str) -> None:
        """Drop any completed projections left by an interrupted sync."""
        with self._transaction() as conn:
            for row in conn.execute(
                _storage_statement(sqlite="SELECT discussion_event_id FROM hosted_room_policy_threads WHERE room_id=? AND completed=1",postgresql='SELECT discussion_event_id FROM hosted_room_policy_threads WHERE room_id=%s AND completed=1'), (room_id,)
            ).fetchall():
                conn.execute(_DELETE_ACTIVE_EVENTS_SQL, (room_id, str(row["discussion_event_id"])))
            conn.execute(_storage_statement(sqlite="DELETE FROM hosted_room_policy_threads WHERE room_id=? AND completed=1",postgresql='DELETE FROM hosted_room_policy_threads WHERE room_id=%s AND completed=1'), (room_id,))


    def _published_task_events(
        self,
        conn: sqlite3.Connection,
        *,
        room_id: str,
        task_id: str | None,
    ) -> list[dict[str, Any]]:
        if not task_id:
            return []
        digest = task_id.removeprefix("dtask:")
        rows = conn.execute(
            _storage_statement(sqlite="""SELECT * FROM hosted_room_events
               WHERE room_id=? AND event_id IN (?, ?) ORDER BY seq""",postgresql='SELECT * FROM hosted_room_events\n               WHERE room_id=%s AND event_id IN (%s, %s) ORDER BY seq'),
            (room_id, f"dmessage:{digest}", f"dterminal:{digest}"),
        ).fetchall()
        events = [_event_from_room_row(row) for row in rows]
        if any(event["payload"].get("task_id") != task_id for event in events):
            raise RuntimeError("published task receipt identity changed")
        return events