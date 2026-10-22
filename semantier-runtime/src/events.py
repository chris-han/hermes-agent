import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled, connect_authority, require_tables, sql_query, begin_write


class ExecutionEventType(str, Enum):
    VALIDATION_PASSED = "VALIDATION_PASSED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    EXECUTION_INTENT_CREATED = "EXECUTION_INTENT_CREATED"
    EXECUTION_DUPLICATE = "EXECUTION_DUPLICATE"
    EXECUTION_COMMITTED = "EXECUTION_COMMITTED"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionEventStore:
    def __init__(self, path: str | None = None):
        raw_path = path or os.getenv(
            "SEMANTIER_EVENT_STORE",
            str(sqlite_db_path("execution_events.db", ".semantier/execution_events.db")),
        )
        self._path = Path(raw_path)
        if enabled():
            self._conn = connect_authority("standalone",self._path)
            require_tables(self._conn,"execution_events")
            self._conn.commit()
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                correlation_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_execution_events_correlation_id
            ON execution_events (correlation_id)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_execution_events_occurred_at
            ON execution_events (occurred_at)
            """
        )
        self._conn.commit()

    def append_event(
        self,
        event_type: ExecutionEventType,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type.value,
            "occurred_at": utc_now(),
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "payload": payload,
        }

        self._conn.execute(
            sql_query("""
            INSERT INTO execution_events (
                event_id,
                event_type,
                occurred_at,
                correlation_id,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?)
            """, """
            INSERT INTO execution_events (
                event_id,
                event_type,
                occurred_at,
                correlation_id,
                payload_json
            )
            VALUES (%s, %s, %s, %s, %s)
            """),
            (
                event["event_id"],
                event["event_type"],
                event["occurred_at"],
                event["correlation_id"],
                json.dumps(payload, sort_keys=True),
            ),
        )
        self._conn.commit()
        return event

    def list_events(self) -> list[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT event_id, event_type, occurred_at, correlation_id, payload_json
            FROM execution_events
            ORDER BY occurred_at ASC
            """
        ).fetchall()

        return [
            {
                "event_id": row[0],
                "event_type": row[1],
                "occurred_at": row[2],
                "correlation_id": row[3],
                "payload": json.loads(row[4]),
            }
            for row in rows
        ]

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
