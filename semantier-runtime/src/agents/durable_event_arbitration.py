from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from storage import hosted_postgres as hosted
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Generic, TypeVar

from agents.durable_protocol_versions import (
    CURRENT_SCHEMA_VERSION,
    CURRENT_WORKFLOW_CONTRACT_VERSION,
    assert_row_supported,
    ensure_version_columns,
)


class DurableEventError(RuntimeError):
    pass


class StaleWorkflowVersion(DurableEventError):
    pass


class InvalidVersionTransition(DurableEventError):
    pass


@dataclass(frozen=True)
class WorkflowVersion:
    version: int
    terminal: bool = False


@dataclass(frozen=True)
class AcceptedEvent:
    workflow_type: str
    workflow_id: str
    event_id: str
    event_type: str
    dedupe_key: str
    observed_at: str
    accepted_sequence: int
    prior_state_version: int
    resulting_state_version: int
    transition_kind: str
    duplicate: bool = False


T = TypeVar("T")


@dataclass(frozen=True)
class ArbitrationResult(Generic[T]):
    event: AcceptedEvent
    value: T | None = None


StateLoader = Callable[[sqlite3.Connection], WorkflowVersion]
Transition = Callable[[sqlite3.Connection, WorkflowVersion], tuple[WorkflowVersion, T]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class DurableEventInbox:
    """Transactional event admission around an existing domain state store."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if not hosted.enabled():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if hosted.enabled():
            return hosted.connect_authority('execution', self.db_path, autocommit=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_schema(self) -> None:
        if hosted.enabled():
            with self._connect() as conn:
                hosted.require_tables(conn, 'semantier_durable_events')
            return
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantier_durable_events (
                    workflow_type TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    accepted_sequence INTEGER NOT NULL,
                    prior_state_version INTEGER NOT NULL,
                    resulting_state_version INTEGER NOT NULL,
                    transition_kind TEXT NOT NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    workflow_contract_version INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(workflow_type, workflow_id, event_id),
                    UNIQUE(workflow_type, workflow_id, dedupe_key),
                    UNIQUE(workflow_type, workflow_id, accepted_sequence)
                )
                """
            )
            ensure_version_columns(conn, "event")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantier_durable_events_replay
                ON semantier_durable_events(
                    workflow_type, workflow_id, accepted_sequence
                )
                """
            )

    @staticmethod
    def _row_to_event(row: sqlite3.Row, *, duplicate: bool = False) -> AcceptedEvent:
        assert_row_supported("event", row)
        return AcceptedEvent(
            workflow_type=str(row["workflow_type"]),
            workflow_id=str(row["workflow_id"]),
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            dedupe_key=str(row["dedupe_key"]),
            observed_at=str(row["observed_at"]),
            accepted_sequence=int(row["accepted_sequence"]),
            prior_state_version=int(row["prior_state_version"]),
            resulting_state_version=int(row["resulting_state_version"]),
            transition_kind=str(row["transition_kind"]),
            duplicate=duplicate,
        )

    def accept(
        self,
        *,
        workflow_type: str,
        workflow_id: str,
        event_id: str,
        event_type: str,
        dedupe_key: str,
        expected_state_version: int,
        load_state: StateLoader,
        transition: Transition[T],
        observed_at: str | None = None,
        connection=None,
    ) -> ArbitrationResult[T]:
        if expected_state_version < 0:
            raise ValueError("expected_state_version must be non-negative")

        owns_connection = connection is None
        conn = self._connect() if owns_connection else connection
        if not owns_connection:
            if not hosted.is_postgres(conn) or not hosted.in_transaction(conn):
                raise ValueError('joined inbox requires an active transaction')
            if conn.execute('SELECT current_schema()').fetchone()[0] != hosted.mapped_store('execution', self.db_path)['schema']:
                raise ValueError('joined inbox authority namespace mismatch')
        try:
            with (nullcontext() if owns_connection else conn.transaction()):
                hosted.begin_write(conn)
                duplicate_row = conn.execute(
                    hosted.sql_query("""
                    SELECT * FROM semantier_durable_events
                    WHERE workflow_type=? AND workflow_id=? AND dedupe_key=?
                    """, """
                    SELECT * FROM semantier_durable_events
                    WHERE workflow_type=%s AND workflow_id=%s AND dedupe_key=%s
                    """),
                    (workflow_type, workflow_id, dedupe_key),
                ).fetchone()
                if duplicate_row is not None:
                    if owns_connection:
                        conn.commit()
                    return ArbitrationResult(
                        event=self._row_to_event(duplicate_row, duplicate=True),
                        value=None,
                    )

                current = load_state(conn)
                if current.version != expected_state_version:
                    raise StaleWorkflowVersion(
                        f"expected state version {expected_state_version}, "
                        f"found {current.version} for {workflow_type}:{workflow_id}"
                    )

                resulting, value = transition(conn, current)
                persisted = load_state(conn)
                if persisted != resulting:
                    raise InvalidVersionTransition(
                        "transition result does not match persisted domain state: "
                        f"reported={resulting} persisted={persisted}"
                    )
                if resulting.version < current.version:
                    raise InvalidVersionTransition(
                        f"state version regressed {current.version}->{resulting.version}"
                    )
                if resulting.version > current.version + 1:
                    raise InvalidVersionTransition(
                        f"state version skipped {current.version}->{resulting.version}"
                    )
                if resulting.version == current.version:
                    transition_kind = "TERMINAL_OBSERVATION" if current.terminal else "NOOP"
                else:
                    transition_kind = "TRANSITION"

                next_sequence = int(
                    conn.execute(
                        hosted.sql_query("""
                        SELECT COALESCE(MAX(accepted_sequence), 0) + 1
                        FROM semantier_durable_events
                        WHERE workflow_type=? AND workflow_id=?
                        """, """
                        SELECT COALESCE(MAX(accepted_sequence), 0) + 1
                        FROM semantier_durable_events
                        WHERE workflow_type=%s AND workflow_id=%s
                        """),
                        (workflow_type, workflow_id),
                    ).fetchone()[0]
                )
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO semantier_durable_events(
                        workflow_type, workflow_id, event_id, event_type,
                        dedupe_key, observed_at, accepted_sequence,
                        prior_state_version, resulting_state_version, transition_kind,
                        schema_version, workflow_contract_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO semantier_durable_events(
                        workflow_type, workflow_id, event_id, event_type,
                        dedupe_key, observed_at, accepted_sequence,
                        prior_state_version, resulting_state_version, transition_kind,
                        schema_version, workflow_contract_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """),
                    (
                        workflow_type, workflow_id, event_id, event_type, dedupe_key,
                        observed_at or _utc_now_iso(), next_sequence, current.version,
                        resulting.version, transition_kind,
                        CURRENT_SCHEMA_VERSION, CURRENT_WORKFLOW_CONTRACT_VERSION,
                    ),
                )
                row = conn.execute(
                    hosted.sql_query("""
                    SELECT * FROM semantier_durable_events
                    WHERE workflow_type=? AND workflow_id=? AND event_id=?
                    """, """
                    SELECT * FROM semantier_durable_events
                    WHERE workflow_type=%s AND workflow_id=%s AND event_id=%s
                    """),
                    (workflow_type, workflow_id, event_id),
                ).fetchone()
                if owns_connection:
                    conn.commit()
                assert row is not None
                return ArbitrationResult(event=self._row_to_event(row), value=value)
        except Exception:
            if owns_connection:
                conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()

    def list_events(self, *, workflow_type: str, workflow_id: str) -> list[AcceptedEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT * FROM semantier_durable_events
                WHERE workflow_type=? AND workflow_id=?
                ORDER BY accepted_sequence ASC
                """, """
                SELECT * FROM semantier_durable_events
                WHERE workflow_type=%s AND workflow_id=%s
                ORDER BY accepted_sequence ASC
                """),
                (workflow_type, workflow_id),
            ).fetchall()
            return [self._row_to_event(row) for row in rows]
