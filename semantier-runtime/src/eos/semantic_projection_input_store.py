from storage import hosted_postgres as hosted
import sqlite3
import json
from datetime import datetime, timezone


class SemanticProjectionInputStore:

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_schema()

    def _init_schema(self):
        if hosted.is_postgres(self.conn):
            hosted.require_tables(self.conn, 'semantic_projection_inputs')
            return
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_projection_inputs (
            org_id TEXT NOT NULL,
            input_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            input_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (org_id, input_id, version)
        )
        """)

    def append(self, org_id, input_id, input_json):
        cursor = self.conn.execute(
            hosted.sql_query("SELECT MAX(version) FROM semantic_projection_inputs WHERE org_id=? AND input_id=?", 'SELECT MAX(version) FROM semantic_projection_inputs WHERE org_id=%s AND input_id=%s'),
            (org_id, input_id)
        )
        row = cursor.fetchone()
        next_v = (row[0] or 0) + 1

        self.conn.execute(hosted.sql_query("""
        INSERT INTO semantic_projection_inputs
        (org_id, input_id, version, input_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """, """
        INSERT INTO semantic_projection_inputs
        (org_id, input_id, version, input_json, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """), (
            org_id,
            input_id,
            next_v,
            json.dumps(input_json),
            datetime.now(timezone.utc).isoformat()
        ))

        self.conn.commit()
        return next_v

    def get(self, org_id, input_id, version):
        cursor = self.conn.execute(hosted.sql_query("""
        SELECT input_json FROM semantic_projection_inputs
        WHERE org_id=? AND input_id=? AND version=?
        """, """
        SELECT input_json FROM semantic_projection_inputs
        WHERE org_id=%s AND input_id=%s AND version=%s
        """), (org_id, input_id, version))

        row = cursor.fetchone()
        return json.loads(row[0]) if row else None
