from __future__ import annotations

import hashlib
import sqlite3
from storage import hosted_postgres as hosted
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agents.durable_protocol_versions import (
    CURRENT_SCHEMA_VERSION,
    CURRENT_WORKFLOW_CONTRACT_VERSION,
    assert_row_supported,
    assert_supported_versions,
    ensure_version_columns,
)


SCHEDULED = "SCHEDULED"
ACCEPTED = "ACCEPTED"
CANCELLED = "CANCELLED"


class TimerOccurrenceError(RuntimeError):
    pass


class TimerIdentityConflict(TimerOccurrenceError):
    pass


@dataclass(frozen=True)
class TimerOccurrence:
    timer_id: str
    workflow_type: str
    workflow_id: str
    purpose: str
    semantic_round: int
    scheduled_at: str
    event_id: str
    dedupe_key: str
    state: str
    provider_ref: str | None


class OneShotTimerProvider(Protocol):
    def arm(self, *, timer_id: str, scheduled_at: str) -> str | None:
        ...


def derive_timer_id(
    *,
    workflow_type: str,
    workflow_id: str,
    purpose: str,
    semantic_round: int,
    scheduled_at: str,
) -> str:
    if not workflow_type.strip():
        raise ValueError("workflow_type is required")
    if not workflow_id.strip():
        raise ValueError("workflow_id is required")
    if not purpose.strip():
        raise ValueError("purpose is required")
    if semantic_round < 1:
        raise ValueError("semantic_round must be >= 1")
    if not scheduled_at.strip():
        raise ValueError("scheduled_at is required")

    raw = "|".join(
        [workflow_type, workflow_id, purpose, str(semantic_round), scheduled_at]
    )
    return f"timer_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}"


class TimerOccurrenceStore:
    """Durable semantic timer registry.

    This object is not a scheduler. It owns timer identity, durable registration,
    provider correlation, stable delivery identity, and terminal acceptance.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if not hosted.enabled():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self.ensure_schema(conn)

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

    @staticmethod
    def ensure_schema(conn: sqlite3.Connection) -> None:
        if hosted.is_postgres(conn):
            hosted.require_tables(conn, 'semantier_timer_occurrences')
            return
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantier_timer_occurrences (
                timer_id TEXT PRIMARY KEY,
                workflow_type TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                purpose TEXT NOT NULL,
                semantic_round INTEGER NOT NULL,
                scheduled_at TEXT NOT NULL,
                event_id TEXT NOT NULL UNIQUE,
                dedupe_key TEXT NOT NULL UNIQUE,
                state TEXT NOT NULL,
                provider_ref TEXT,
                schema_version INTEGER NOT NULL DEFAULT 1,
                workflow_contract_version INTEGER NOT NULL DEFAULT 1,
                CHECK (state IN ('SCHEDULED','ACCEPTED','CANCELLED')),
                UNIQUE(
                    workflow_type,
                    workflow_id,
                    purpose,
                    semantic_round,
                    scheduled_at
                )
            )
            """
        )
        ensure_version_columns(conn, "timer")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_semantier_timer_occurrences_pending
            ON semantier_timer_occurrences(state, scheduled_at)
            """
        )

    @staticmethod
    def _row_to_occurrence(row) -> TimerOccurrence:
        if isinstance(row, sqlite3.Row):
            assert_row_supported("timer", row)
            values = {
                "timer_id": row["timer_id"],
                "workflow_type": row["workflow_type"],
                "workflow_id": row["workflow_id"],
                "purpose": row["purpose"],
                "semantic_round": row["semantic_round"],
                "scheduled_at": row["scheduled_at"],
                "event_id": row["event_id"],
                "dedupe_key": row["dedupe_key"],
                "state": row["state"],
                "provider_ref": row["provider_ref"],
            }
        else:
            assert_supported_versions(
                "timer",
                {
                    "schema_version": int(row[10]),
                    "workflow_contract_version": int(row[11]),
                },
            )
            values = {
                "timer_id": row[0],
                "workflow_type": row[1],
                "workflow_id": row[2],
                "purpose": row[3],
                "semantic_round": row[4],
                "scheduled_at": row[5],
                "event_id": row[6],
                "dedupe_key": row[7],
                "state": row[8],
                "provider_ref": row[9],
            }
        return TimerOccurrence(
            timer_id=str(values["timer_id"]),
            workflow_type=str(values["workflow_type"]),
            workflow_id=str(values["workflow_id"]),
            purpose=str(values["purpose"]),
            semantic_round=int(values["semantic_round"]),
            scheduled_at=str(values["scheduled_at"]),
            event_id=str(values["event_id"]),
            dedupe_key=str(values["dedupe_key"]),
            state=str(values["state"]),
            provider_ref=(
                str(values["provider_ref"])
                if values["provider_ref"] is not None
                else None
            ),
        )

    @staticmethod
    def schedule(
        conn: sqlite3.Connection,
        *,
        workflow_type: str,
        workflow_id: str,
        purpose: str,
        semantic_round: int,
        scheduled_at: str,
    ) -> TimerOccurrence:
        TimerOccurrenceStore.ensure_schema(conn)
        timer_id = derive_timer_id(
            workflow_type=workflow_type,
            workflow_id=workflow_id,
            purpose=purpose,
            semantic_round=semantic_round,
            scheduled_at=scheduled_at,
        )
        event_id = f"timer_event:{timer_id}"
        dedupe_key = f"timer:{timer_id}"
        conn.execute(
            hosted.sql_query("""
            INSERT OR IGNORE INTO semantier_timer_occurrences(
                timer_id, workflow_type, workflow_id, purpose,
                semantic_round, scheduled_at, event_id, dedupe_key, state,
                schema_version, workflow_contract_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'SCHEDULED', ?, ?)
            """, """
            INSERT INTO semantier_timer_occurrences(
                timer_id, workflow_type, workflow_id, purpose,
                semantic_round, scheduled_at, event_id, dedupe_key, state,
                schema_version, workflow_contract_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'SCHEDULED', %s, %s) ON CONFLICT DO NOTHING"""),
            (
                timer_id,
                workflow_type,
                workflow_id,
                purpose,
                semantic_round,
                scheduled_at,
                event_id,
                dedupe_key,
                CURRENT_SCHEMA_VERSION,
                CURRENT_WORKFLOW_CONTRACT_VERSION,
            ),
        )
        row = conn.execute(
            hosted.sql_query("SELECT * FROM semantier_timer_occurrences WHERE timer_id=?", 'SELECT * FROM semantier_timer_occurrences WHERE timer_id=%s'),
            (timer_id,),
        ).fetchone()
        if row is None:
            raise TimerOccurrenceError("timer occurrence was not persisted")
        occurrence = TimerOccurrenceStore._row_to_occurrence(row)
        if (
            occurrence.workflow_type != workflow_type
            or occurrence.workflow_id != workflow_id
            or occurrence.purpose != purpose
            or occurrence.semantic_round != semantic_round
            or occurrence.scheduled_at != scheduled_at
        ):
            raise TimerIdentityConflict("timer identity collision")
        return occurrence

    def register(
        self,
        *,
        workflow_type: str,
        workflow_id: str,
        purpose: str,
        semantic_round: int,
        scheduled_at: str,
    ) -> TimerOccurrence:
        with self._connect() as conn:
            return self.schedule(
                conn,
                workflow_type=workflow_type,
                workflow_id=workflow_id,
                purpose=purpose,
                semantic_round=semantic_round,
                scheduled_at=scheduled_at,
            )

    def get(self, timer_id: str) -> TimerOccurrence | None:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT * FROM semantier_timer_occurrences WHERE timer_id=?", 'SELECT * FROM semantier_timer_occurrences WHERE timer_id=%s'),
                (timer_id,),
            ).fetchone()
            return self._row_to_occurrence(row) if row else None

    def arm(self, *, timer_id: str, provider: OneShotTimerProvider) -> TimerOccurrence:
        occurrence = self.get(timer_id)
        if occurrence is None:
            raise KeyError(timer_id)
        if occurrence.state != SCHEDULED:
            return occurrence

        provider_ref = provider.arm(
            timer_id=occurrence.timer_id,
            scheduled_at=occurrence.scheduled_at,
        )
        normalized_ref = str(provider_ref or "").strip() or None
        with self._connect() as conn:
            cur = conn.execute(
                hosted.sql_query("""
                UPDATE semantier_timer_occurrences
                SET provider_ref=?
                WHERE timer_id=?
                  AND state='SCHEDULED'
                  AND (
                    provider_ref IS NULL
                    OR provider_ref=?
                  )
                """, """
                UPDATE semantier_timer_occurrences
                SET provider_ref=%s
                WHERE timer_id=%s
                  AND state='SCHEDULED'
                  AND (
                    provider_ref IS NULL
                    OR provider_ref=%s
                  )
                """),
                (normalized_ref, timer_id, normalized_ref),
            )
            if cur.rowcount != 1:
                row = conn.execute(
                    hosted.sql_query("SELECT provider_ref FROM semantier_timer_occurrences WHERE timer_id=?", 'SELECT provider_ref FROM semantier_timer_occurrences WHERE timer_id=%s'),
                    (timer_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(timer_id)
                if (row["provider_ref"] or None) != normalized_ref:
                    raise TimerIdentityConflict(
                        "provider attempted to change timer correlation"
                    )
        armed = self.get(timer_id)
        assert armed is not None
        return armed

    def delivery_event(self, *, timer_id: str) -> TimerOccurrence:
        occurrence = self.get(timer_id)
        if occurrence is None:
            raise KeyError(timer_id)
        return occurrence

    def mark_accepted(self, *, timer_id: str) -> TimerOccurrence:
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                UPDATE semantier_timer_occurrences
                SET state='ACCEPTED'
                WHERE timer_id=? AND state='SCHEDULED'
                """, """
                UPDATE semantier_timer_occurrences
                SET state='ACCEPTED'
                WHERE timer_id=%s AND state='SCHEDULED'
                """),
                (timer_id,),
            )
        occurrence = self.get(timer_id)
        if occurrence is None:
            raise KeyError(timer_id)
        return occurrence

    def cancel(self, *, timer_id: str) -> TimerOccurrence:
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                UPDATE semantier_timer_occurrences
                SET state='CANCELLED'
                WHERE timer_id=? AND state='SCHEDULED'
                """, """
                UPDATE semantier_timer_occurrences
                SET state='CANCELLED'
                WHERE timer_id=%s AND state='SCHEDULED'
                """),
                (timer_id,),
            )
        occurrence = self.get(timer_id)
        if occurrence is None:
            raise KeyError(timer_id)
        return occurrence

    def pending(self) -> list[TimerOccurrence]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM semantier_timer_occurrences
                WHERE state='SCHEDULED'
                ORDER BY scheduled_at, timer_id
                """
            ).fetchall()
            return [self._row_to_occurrence(row) for row in rows]
