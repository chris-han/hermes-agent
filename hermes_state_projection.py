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

    def create_client_session(self,session_id,*,source,model=None,model_config=None,system_prompt=None,title=None):
        import json
        import time
        def create(conn):
            if conn.execute('SELECT id FROM sessions WHERE id=?',(session_id,)).fetchone():
                return None,'exists'
            conn.execute('INSERT INTO sessions(id,source,model,model_config,system_prompt,started_at) VALUES(?,?,?,?,?,?)',
                (session_id,source,model,json.dumps(model_config) if model_config else None,system_prompt,time.time()))
            if title is not None:
                clean=self.sanitize_title(str(title))
                if clean:
                    try:self._resolve_title_conflict(conn,session_id,clean)
                    except ValueError as exc:
                        conn.execute('DELETE FROM sessions WHERE id=?',(session_id,));return None,f'title:{exc}'
                conn.execute('UPDATE sessions SET title=? WHERE id=?',(clean,session_id))
            row=conn.execute('SELECT * FROM sessions WHERE id=?',(session_id,)).fetchone()
            return dict(row),None
        return self._execute_write(create)
