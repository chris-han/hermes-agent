import sqlite3
import os
import json
from pathlib import Path

from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled, connect_authority, require_tables, sql_query, begin_write


class DomainEventStore:
    def __init__(self, path: str | None = None):
        raw_path = path or os.getenv(
            "SEMANTIER_DOMAIN_STORE",
            str(sqlite_db_path("domain.db", ".semantier/domain.db"))
        )
        self._path = Path(raw_path)
        if enabled():
            self._conn = connect_authority("standalone",self._path)
            require_tables(self._conn,"domain_events")
            self._conn.commit()
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS domain_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                payload_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def append(self, event_type: str, payload: dict):
        self._conn.execute(
            sql_query("INSERT INTO domain_events (event_type, payload_json) VALUES (?, ?)", "INSERT INTO domain_events (event_type, payload_json) VALUES (%s, %s)"),
            (event_type, json.dumps(payload, sort_keys=True)),
        )
        self._conn.commit()

    def exists_for_key(self, key: str) -> bool:
        cur = self._conn.execute(
            sql_query("SELECT 1 FROM domain_events WHERE payload_json LIKE ?", "SELECT 1 FROM domain_events WHERE payload_json LIKE %s"),
            (f'%{key}%',),
        )
        return cur.fetchone() is not None
