"""Core-owned documentary wiki intents, attempts and sticky task bindings.

These operational rows confer no knowledge admission or activation authority.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from storage.governed_file_io import exclusive_file_lock
import hashlib
import json
from pathlib import Path

from eos.db import get_conn
from storage import hosted_postgres as hosted


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


class WikiMutationStore:
    def __init__(self):
        with self.connection() as conn:
            if hosted.is_postgres(conn):
                hosted.require_tables(conn, 'wiki_task_bindings', 'wiki_mutation_operations', 'wiki_mutation_attempts')
                return
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS wiki_task_bindings (
                  workspace TEXT NOT NULL, session TEXT NOT NULL, actor TEXT NOT NULL,
                  selected_at TEXT NOT NULL, PRIMARY KEY(workspace,session));
                CREATE TABLE IF NOT EXISTS wiki_mutation_operations (
                  workspace TEXT NOT NULL, operation_id TEXT NOT NULL, payload TEXT NOT NULL,
                  PRIMARY KEY(workspace,operation_id));
                CREATE TABLE IF NOT EXISTS wiki_mutation_attempts (
                  sequence INTEGER PRIMARY KEY AUTOINCREMENT, workspace TEXT NOT NULL,
                  operation_id TEXT NOT NULL, produced_at TEXT NOT NULL, state TEXT NOT NULL);
            ''')

    @contextmanager
    def connection(self):
        conn = get_conn()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def workspace_lock(self, workspace):
        # Reuse the governed database owner root; never put locks/intents in an
        # actor-writable session root. flock survives threads/processes/restart.
        with self.connection() as conn:
            if hosted.is_postgres(conn):
                # Hold a dedicated session advisory lock across the service's
                # multiple owner transactions. Process/connection loss releases it.
                conn.commit()
                conn.autocommit = True
                conn.execute("SELECT pg_advisory_lock(hashtextextended(current_schema() || ':wiki-workspace:' || %s,0))", (workspace,))
                try:
                    yield
                finally:
                    conn.execute("SELECT pg_advisory_unlock(hashtextextended(current_schema() || ':wiki-workspace:' || %s,0))", (workspace,))
                return
            database = Path(conn.execute('PRAGMA database_list').fetchone()[2])
        path = database.parent / ('wiki-' + hashlib.sha256(workspace.encode()).hexdigest() + '.lock')
        with exclusive_file_lock(path):
            yield

    def bind(self, workspace, session, actor):
        with self.connection() as conn:
            conn.execute(hosted.sql_query('INSERT OR IGNORE INTO wiki_task_bindings VALUES (?,?,?,?)', 'INSERT INTO wiki_task_bindings VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING'), (workspace,session,actor,utc()))
            row=conn.execute(hosted.sql_query('SELECT actor FROM wiki_task_bindings WHERE workspace=? AND session=?', 'SELECT actor FROM wiki_task_bindings WHERE workspace=%s AND session=%s'),(workspace,session)).fetchone()
            if row[0] != actor: raise PermissionError('wiki task binding actor conflict')

    def binding(self, workspace, session):
        with self.connection() as conn:
            row=conn.execute(hosted.sql_query('SELECT actor FROM wiki_task_bindings WHERE workspace=? AND session=?', 'SELECT actor FROM wiki_task_bindings WHERE workspace=%s AND session=%s'),(workspace,session)).fetchone()
            return row[0] if row else None

    def get(self, workspace, operation_id):
        with self.connection() as conn:
            row=conn.execute(hosted.sql_query('SELECT payload FROM wiki_mutation_operations WHERE workspace=? AND operation_id=?', 'SELECT payload FROM wiki_mutation_operations WHERE workspace=%s AND operation_id=%s'),(workspace,operation_id)).fetchone()
            return json.loads(row[0]) if row else None

    def save(self, workspace, record):
        with self.connection() as conn:
            conn.execute(hosted.sql_query('INSERT INTO wiki_mutation_operations VALUES (?,?,?) ON CONFLICT(workspace,operation_id) DO UPDATE SET payload=excluded.payload', 'INSERT INTO wiki_mutation_operations VALUES (%s,%s,%s) ON CONFLICT(workspace,operation_id) DO UPDATE SET payload=excluded.payload'), (workspace,record['operation_id'],json.dumps(record,sort_keys=True)))
            conn.execute(hosted.sql_query('INSERT INTO wiki_mutation_attempts(workspace,operation_id,produced_at,state) VALUES (?,?,?,?)', 'INSERT INTO wiki_mutation_attempts(workspace,operation_id,produced_at,state) VALUES (%s,%s,%s,%s)'),(workspace,record['operation_id'],utc(),record['assurance']['state']))

    def attempt(self, workspace, operation_id, state):
        with self.connection() as conn:
            conn.execute(hosted.sql_query('INSERT INTO wiki_mutation_attempts(workspace,operation_id,produced_at,state) VALUES (?,?,?,?)', 'INSERT INTO wiki_mutation_attempts(workspace,operation_id,produced_at,state) VALUES (%s,%s,%s,%s)'),(workspace,operation_id,utc(),state))

    def pending(self, workspace):
        with self.connection() as conn:
            rows=conn.execute(hosted.sql_query('SELECT payload FROM wiki_mutation_operations WHERE workspace=?', 'SELECT payload FROM wiki_mutation_operations WHERE workspace=%s'),(workspace,)).fetchall()
        return [json.loads(row[0]) for row in rows if json.loads(row[0])['assurance']['state'] in ('INTENT','PENDING')]
