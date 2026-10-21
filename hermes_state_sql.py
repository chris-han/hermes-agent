"""Explicit SQL dialect choices for native owners. Never rewrites SQL at runtime."""
from contextlib import contextmanager
from pathlib import Path

from hermes_state_backend import postgres_enabled,resolve_postgres_scope


def statement(*,sqlite,postgresql):
    return postgresql if postgres_enabled() else sqlite


def parameter_marker():return '%s' if postgres_enabled() else '?'


def is_postgres(conn):
    return getattr(conn,'hermes_scope',None) is not None


class NativeRow:
    """Native row protocol: named and positional access over PostgreSQL values."""
    def __init__(self,names,values):self._names,self._values=names,values
    def get(self,key,default=None):
        return self[key] if key in self._names else default
    def values(self):return self._values
    def keys(self):return self._names
    def __getitem__(self,key):return self._values[self._names.index(key)] if isinstance(key,str) else self._values[key]
    def __iter__(self):return iter(self._values)
    def __len__(self):return len(self._values)


def _row_factory(cursor):
    names=tuple(column.name for column in cursor.description) if cursor.description else ()
    return lambda values:NativeRow(names,values)


def connect_postgres(home,*,read_only=False):
    import os
    import psycopg
    from psycopg import sql
    scope=resolve_postgres_scope(home)
    dsn=os.environ.get('HERMES_DATABASE_URL')
    if not dsn:raise ValueError('HERMES_DATABASE_URL required')
    conn=psycopg.connect(dsn,autocommit=True,row_factory=_row_factory)
    try:
        conn.execute(sql.SQL('SET search_path TO {},pg_catalog').format(sql.Identifier(scope.schema)))
        row=conn.execute('SELECT tenant_id,workspace_id,source_store_id FROM scope_binding').fetchone()
        if tuple(row)!=scope.identity:raise ValueError('Provisioned native namespace required')
        import socket
        conn.execute("SELECT set_config('hermes.owner_host',%s,false)",(socket.gethostname(),))
        conn.execute("SET timezone='UTC'")
        if read_only:conn.execute('SET default_transaction_read_only=on')
        conn.hermes_scope=scope
        conn.hermes_home=Path(home).resolve()
        conn.hermes_read_only=read_only
        return conn
    except BaseException:conn.close();raise


def inserted_id(cursor):
    return cursor.fetchone()[0] if is_postgres(cursor.connection) else cursor.lastrowid


def in_transaction(conn):
    if is_postgres(conn):
        from psycopg.pq import TransactionStatus
        return conn.info.transaction_status!=TransactionStatus.IDLE
    return conn.in_transaction


@contextmanager
def native_write_transaction(conn,*,allow_nested=False):
    if conn.hermes_read_only:raise PermissionError('Read-only native store')
    if in_transaction(conn) and not allow_nested:raise RuntimeError('Nested native write transaction requires explicit composition')
    with conn.transaction():
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(conn.hermes_scope.schema+'\x1fcanonical-native-writer',))
        yield conn


def integrity_errors():
    import sqlite3
    if postgres_enabled():
        import psycopg
        return (sqlite3.IntegrityError,psycopg.IntegrityError)
    return (sqlite3.IntegrityError,)
