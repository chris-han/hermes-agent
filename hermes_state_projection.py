"""Storage-owned session lineage and branch metadata reads."""
import sqlite3


class SessionProjectionMixin:
    def list_session_records(self):
        return [dict(row) for row in self._read_all('SELECT * FROM sessions ORDER BY started_at,id')]

    def get_branch_records(self):
        with self._read_ctx() as conn:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='semantier_session_branches'").fetchone():
                return []
            return [dict(row) for row in conn.execute('SELECT * FROM semantier_session_branches ORDER BY child_session_id')]
