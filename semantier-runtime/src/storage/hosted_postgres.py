"""Explicit hosted PostgreSQL selection and real transaction primitives.

SQL remains owned by each domain module. This module does not translate SQL or
emulate a SQLite connection. Missing mappings/schemas are hard failures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any


class HostedStorageUnavailable(RuntimeError):
    pass


class AuthorityRow(tuple):
    """Read-only name/position access for existing domain value projections."""
    def __new__(cls, values, columns):
        value = super().__new__(cls, values)
        value._columns = tuple(columns)
        return value

    def keys(self):
        return self._columns

    def __getitem__(self, key):
        if isinstance(key, str):
            key = self._columns.index(key)
        return super().__getitem__(key)


def _row_factory(cursor):
    columns = tuple(column.name for column in cursor.description or ())
    return lambda values: AuthorityRow(values, columns)


def enabled() -> bool:
    backend = os.environ.get('SEMANTIER_STORAGE_BACKEND', 'sqlite').lower()
    if backend not in {'sqlite', 'postgresql'}:
        raise HostedStorageUnavailable('unsupported explicit storage backend')
    return backend == 'postgresql'


def database_url() -> str:
    value = os.environ.get('SEMANTIER_DATABASE_URL', '')
    if not value:
        raise HostedStorageUnavailable('hosted PostgreSQL database URL is required')
    return value


def sql_query(sqlite: str, postgres: str) -> str:
    """Choose the two explicit, owner-supplied SQL statements; never rewrite."""
    return postgres if enabled() else sqlite


def is_postgres(conn) -> bool:
    return hasattr(conn, 'pgconn')


def integrity_errors() -> tuple[type[Exception], ...]:
    import sqlite3
    if enabled():
        import psycopg
        return (sqlite3.IntegrityError, psycopg.IntegrityError)
    return (sqlite3.IntegrityError,)


def mapped_store(owner: str, path: str | Path) -> dict[str, Any]:
    try:
        mapping = json.loads(os.environ.get('SEMANTIER_AUTHORITY_STORE_MAP', '{}'))
    except (ValueError, TypeError) as error:
        raise HostedStorageUnavailable('invalid authority store mapping') from error
    target = str(Path(path).expanduser().resolve())
    entry = mapping.get(target)
    if entry is None and owner == 'standalone':
        from storage.hosted_auxiliary_paths import resolve_auxiliary_scope
        try:
            scope = resolve_auxiliary_scope(Path(target))
            entry = {'owner': owner, 'schema': scope.schema,
                     'tenant_id': scope.tenant_id, 'workspace_id': scope.workspace_id,
                     'source_store_id': scope.source_store_id}
        except (ValueError, RuntimeError, KeyError) as error:
            raise HostedStorageUnavailable('missing authority store mapping') from error
    if entry is None and owner == 'execution':
        # Native home scope is supplied by governed runtime configuration.
        from hermes_state_backend import resolve_postgres_scope
        try:
            scope = resolve_postgres_scope(Path(target).parent)
            entry = {'owner': owner, 'schema': scope.schema}
        except (ValueError, RuntimeError, KeyError) as error:
            raise HostedStorageUnavailable('missing authority store mapping') from error
    if not isinstance(entry, dict) or entry.get('owner') != owner:
        raise HostedStorageUnavailable('missing authority store mapping')
    schema = entry.get('schema', '')
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,62}', schema):
        raise HostedStorageUnavailable('invalid mapped authority schema')
    return entry


def connect_authority(owner: str, path: str | Path, *, autocommit: bool = False):
    if not enabled():
        raise HostedStorageUnavailable('PostgreSQL authority requested in local mode')
    schema = mapped_store(owner, path)['schema']
    import psycopg
    try:
        return psycopg.connect(database_url(), autocommit=autocommit,
            row_factory=_row_factory,
            options=f'-c search_path={schema},pg_catalog -c timezone=UTC')
    except psycopg.Error:
        raise HostedStorageUnavailable('hosted PostgreSQL authority unavailable') from None


def require_tables(conn, *tables: str) -> None:
    rows = conn.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname=current_schema()")
    present = {row[0] for row in rows}
    missing = set(tables) - present
    if missing:
        raise HostedStorageUnavailable('required authority schema not ready: ' + ','.join(sorted(missing)))


def begin_write(conn) -> None:
    if is_postgres(conn):
        if not in_transaction(conn):
            conn.execute('BEGIN')
        # Preserve the original same-store serialized mutation/fence boundary.
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(current_schema()), hashtext('authority-write'))")
    else:
        conn.execute('BEGIN IMMEDIATE')


def in_transaction(conn) -> bool:
    if is_postgres(conn):
        from psycopg.pq import TransactionStatus
        return conn.info.transaction_status != TransactionStatus.IDLE
    return conn.in_transaction


def execute_script(conn, script: str) -> None:
    if is_postgres(conn):
        conn.execute(script)
    else:
        conn.executescript(script)


def execution_schema_sql() -> str:
    return '\n'.join(Path(__file__).with_name(name).read_text() for name in
                     ('hosted_execution.sql', 'hosted_meeting.sql'))


def require_meeting_ready(conn) -> None:
    asset = Path(__file__).with_name('hosted_meeting.sql').read_text()
    require_tables(conn, *re.findall(r'CREATE TABLE IF NOT EXISTS (\w+)', asset))
    present = {row[0] for row in conn.execute(
        'SELECT t.tgname FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid '
        'JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace '
        "WHERE n.nspname=current_schema() AND NOT t.tgisinternal AND t.tgenabled<>'D'")}
    if not set(re.findall(r'CREATE TRIGGER (\w+)', asset)) <= present:
        raise HostedStorageUnavailable('meeting workflow history constraints are not ready')


def migrate_execution(dsn: str, schema: str, *, app_role: str | None = None):
    import psycopg
    from psycopg import sql
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,62}', schema):
        raise ValueError('invalid schema')
    with psycopg.connect(dsn) as conn:
        conn.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('execution-migrate:' + schema,))
        conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(schema)))
        conn.execute(sql.SQL('SET LOCAL search_path TO {},pg_catalog').format(sql.Identifier(schema)))
        conn.execute(execution_schema_sql())
        if app_role:
            owned = sorted(set(re.findall(r'CREATE TABLE IF NOT EXISTS (\w+)', execution_schema_sql())))
            writable = [name for name in owned if name != 'semantier_execution_imports']
            role = sql.Identifier(app_role)
            conn.execute(sql.SQL('GRANT USAGE ON SCHEMA {} TO {}').format(sql.Identifier(schema), sql.Identifier(app_role)))
            conn.execute(sql.SQL('GRANT SELECT ON {} TO {}').format(sql.SQL(',').join(map(sql.Identifier, owned)), role))
            conn.execute(sql.SQL('GRANT INSERT,UPDATE,DELETE ON {} TO {}').format(sql.SQL(',').join(map(sql.Identifier, writable)), role))
            sequences = conn.execute("SELECT ns.nspname,s.relname FROM pg_catalog.pg_class s "
                "JOIN pg_catalog.pg_namespace ns ON ns.oid=s.relnamespace "
                "JOIN pg_catalog.pg_depend d ON d.objid=s.oid AND d.deptype IN ('a','i') "
                "JOIN pg_catalog.pg_class t ON t.oid=d.refobjid "
                "WHERE s.relkind='S' AND ns.nspname=%s AND t.relname=ANY(%s)", (schema, owned)).fetchall()
            if sequences:
                conn.execute(sql.SQL('GRANT USAGE,SELECT ON SEQUENCE {} TO {}').format(
                    sql.SQL(',').join(sql.Identifier(namespace, name) for namespace, name in sequences), role))
            conn.execute(sql.SQL('REVOKE INSERT,UPDATE,DELETE ON semantier_execution_imports FROM {}').format(sql.Identifier(app_role)))


def require_ready(dsn: str | None = None):
    """Validate the complete hosted migration projection without runtime writes."""
    from storage.hosted_readiness import require_ready as validate
    return validate(dsn)


def import_execution(source: str | Path, dsn: str, schema: str, source_store_id: str):
    """Import only execution-owned tables from a frozen shared native snapshot."""
    import hashlib
    import sqlite3
    from datetime import datetime, timezone
    import psycopg
    from psycopg import sql
    if not source_store_id or not re.fullmatch(r'[a-z][a-z0-9_]{0,62}', schema):
        raise ValueError('explicit execution namespace and source identity required')
    ordered = re.findall(r'CREATE TABLE IF NOT EXISTS (\w+)', execution_schema_sql())
    owned = set(ordered)
    owned.remove('semantier_execution_imports')
    source = Path(source).resolve(strict=True)
    def fingerprint(columns, rows):
        def cell(value):
            return {'bytes_hex': bytes(value).hex()} if isinstance(value, (bytes, bytearray, memoryview)) else value
        values = sorted(json.dumps([cell(v) for v in row], ensure_ascii=False, separators=(',', ':'), allow_nan=False) for row in rows)
        return hashlib.sha256(json.dumps([columns, values], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    snapshot = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)
    try:
        snapshot.execute('PRAGMA query_only=ON')
        snapshot.execute('BEGIN')
        source_tables = {row[0] for row in snapshot.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        source_schema = snapshot.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name').fetchall()
        sequence_highwater = dict(snapshot.execute('SELECT name,seq FROM sqlite_sequence')) if 'sqlite_sequence' in source_tables else {}
        data = {}
        hashes = {}
        for table in ordered:
            if table not in source_tables or table not in owned:
                continue
            columns = [row[1] for row in snapshot.execute(f'PRAGMA table_info("{table}")')]
            rows = snapshot.execute(f'SELECT * FROM "{table}"').fetchall()
            data[table] = (columns, rows)
            hashes[table] = fingerprint(columns, rows)
        digest = hashlib.sha256(json.dumps({'tables': hashes, 'schema': source_schema,
                                           'sequences': sequence_highwater},
                                          sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with psycopg.connect(dsn) as conn:
            conn.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('execution-import:' + schema,))
            conn.execute(sql.SQL('SET LOCAL search_path TO {},pg_catalog').format(sql.Identifier(schema)))
            require_tables(conn, *owned, 'semantier_execution_imports')
            prior = conn.execute('SELECT source_store_id,source_fingerprint FROM semantier_execution_imports').fetchone()
            if prior:
                if prior != (source_store_id, digest):
                    raise ValueError('execution namespace bound to different source or snapshot')
                return {'source_store_id': source_store_id, 'source_fingerprint': digest, 'idempotent': True}
            for table in sorted(owned):
                if conn.execute(sql.SQL('SELECT 1 FROM {} LIMIT 1').format(sql.Identifier(table))).fetchone():
                    raise ValueError('execution target collision: ' + table)
            for table, (columns, rows) in data.items():
                actual = {row[0] for row in conn.execute('SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=%s', (table,))}
                if not set(columns) <= actual:
                    raise ValueError('unmigrated execution source columns: ' + table)
                statement = sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(sql.Identifier(table),
                    sql.SQL(',').join(map(sql.Identifier, columns)), sql.SQL(',').join(sql.Placeholder() for _ in columns))
                if rows:
                    with conn.cursor() as cursor:
                        cursor.executemany(statement, rows)
                target_rows = conn.execute(sql.SQL('SELECT {} FROM {}').format(sql.SQL(',').join(map(sql.Identifier, columns)), sql.Identifier(table))).fetchall()
                if fingerprint(columns, target_rows) != hashes[table]:
                    raise ValueError('execution payload verification failed: ' + table)
                for column in columns:
                    sequence = conn.execute('SELECT pg_get_serial_sequence(%s,%s)', (schema + '.' + table, column)).fetchone()[0]
                    if sequence:
                        maximum = conn.execute(sql.SQL('SELECT max({}) FROM {}').format(sql.Identifier(column), sql.Identifier(table))).fetchone()[0]
                        highwater = max(maximum or 0, sequence_highwater.get(table, 0))
                        conn.execute('SELECT setval(%s,%s,%s)', (sequence, highwater or 1, highwater > 0))
            conn.execute('INSERT INTO semantier_execution_imports VALUES (%s,%s,%s)', (source_store_id, digest, datetime.now(timezone.utc).isoformat()))
        return {'source_store_id': source_store_id, 'source_fingerprint': digest, 'idempotent': False,
                'tables': len(data), 'rows': sum(len(rows) for _, rows in data.values()),
                'other_owner_tables': sorted(source_tables - owned - {'sqlite_sequence'})}
    finally:
        snapshot.close()
