"""Explicit EOS PostgreSQL migrations and read-only SQLite import.

Runtime uses real psycopg connections through eos.db. Schema/data migration is
operator-invoked only; an application constructor never imports a database.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
from datetime import datetime, timezone

import psycopg
from psycopg import sql

from storage import hosted_postgres as hosted

SCHEMA_VERSION = 1
DERIVED_FTS_TABLES = frozenset({
    'graph_node_search_config', 'graph_node_search_content', 'graph_node_search_data',
    'graph_node_search_docsize', 'graph_node_search_idx',
})


def schema_assets():
    """Dependency-ordered, owner-authored migration assets; never runtime DDL."""
    storage = Path(__file__).parents[1] / 'storage'
    return (Path(__file__).with_name('postgresql.sql'),
            storage / 'hosted_eos_extensions.sql',
            storage / 'hosted_eos_workflow.sql',
            storage / 'knowledge_extraction_postgres.sql',
            storage / 'human_grounding_postgres.sql',
            storage / 'legacy_eos_events_postgres.sql')


def schema_contract_hash():
    return hashlib.sha256(b'\0'.join(path.read_bytes() for path in schema_assets()) +
                          b'\0' + Path(__file__).with_name('postgresql_triggers.sql').read_bytes()).hexdigest()


def migrate_eos(dsn: str, schema: str, *, app_role: str | None = None) -> None:
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,62}', schema):
        raise ValueError('invalid EOS schema')
    with psycopg.connect(dsn) as conn:
        conn.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('eos-migrate:' + schema,))
        conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(schema)))
        conn.execute(sql.SQL('SET LOCAL search_path TO {},pg_catalog').format(sql.Identifier(schema)))
        for asset in schema_assets():
            conn.execute(asset.read_text())
        conn.execute(Path(__file__).with_name('postgresql_triggers.sql').read_text())
        conn.execute('INSERT INTO eos_postgres_schema_contract VALUES(1,%s) ON CONFLICT(singleton) DO UPDATE SET asset_sha256=excluded.asset_sha256', (schema_contract_hash(),))
        if app_role:
            role = sql.Identifier(app_role)
            conn.execute(sql.SQL('GRANT USAGE ON SCHEMA {} TO {}').format(sql.Identifier(schema), role))
            conn.execute(sql.SQL('GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA {} TO {}').format(sql.Identifier(schema), role))
            conn.execute(sql.SQL('GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}').format(sql.Identifier(schema), role))
            conn.execute(sql.SQL('REVOKE INSERT,UPDATE,DELETE ON semantier_schema_versions,eos_sqlite_imports,eos_postgres_schema_contract,benchmark_read_model_schema_metadata FROM {}').format(role))


def require_ready(conn) -> None:
    tables = re.findall(r'CREATE TABLE IF NOT EXISTS (\w+)', '\n'.join(path.read_text() for path in schema_assets()))
    hosted.require_tables(conn, *tables)
    contract = conn.execute('SELECT asset_sha256 FROM eos_postgres_schema_contract WHERE singleton=1').fetchone()
    if contract is None or contract[0] != schema_contract_hash():
        raise hosted.HostedStorageUnavailable('required EOS migration contract is absent or stale')
    versions = dict(conn.execute('SELECT schema_name,schema_version FROM semantier_schema_versions'))
    if versions.get('eos_core') != 2 or versions.get('rule_governance') != 3:
        raise hosted.HostedStorageUnavailable('required EOS schema versions are not ready')
    triggers = {row[0] for row in conn.execute(
        'SELECT t.tgname FROM pg_catalog.pg_trigger t '
        'JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid '
        'JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace '
        "WHERE n.nspname=current_schema() AND NOT t.tgisinternal AND t.tgenabled<>'D'")}
    expected = set(re.findall(r'CREATE TRIGGER (\w+)',
                             Path(__file__).with_name('postgresql_triggers.sql').read_text()))
    if not expected <= triggers:
        raise hosted.HostedStorageUnavailable('required EOS history constraints are not ready')


def _value(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {'bytes_hex': bytes(value).hex()}
    return value


def _table_fingerprint(columns, rows):
    canonical = [json.dumps([_value(value) for value in row], ensure_ascii=False,
                            separators=(',', ':'), allow_nan=False) for row in rows]
    payload = json.dumps({'columns': columns, 'rows': sorted(canonical)},
                         ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def import_sqlite(source: str | Path, dsn: str, schema: str, source_store_id: str):
    """Import a frozen source atomically; identical reruns never overwrite writes."""
    if not source_store_id or not re.fullmatch(r'[a-z][a-z0-9_]{0,62}', schema):
        raise ValueError('explicit EOS source identity and schema required')
    source = Path(source).resolve(strict=True)
    snapshot = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)
    snapshot.execute('PRAGMA query_only=ON')
    snapshot.execute('BEGIN')
    try:
        tables = sorted(name for (name,) in snapshot.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                        if name not in DERIVED_FTS_TABLES)
        source_schema = snapshot.execute(
            'SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name').fetchall()
        sequence_highwater = dict(snapshot.execute('SELECT name,seq FROM sqlite_sequence')) if any(
            row[1] == 'sqlite_sequence' for row in source_schema) else {}
        data = {}
        fingerprints = {}
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            columns = [row[1] for row in snapshot.execute(f'PRAGMA table_info({quoted})')]
            order_column = {'tender_graph_releases': 'release_sequence',
                            'source_lifecycle_events': 'event_sequence',
                            'source_representation_lifecycle_events': 'event_sequence'}.get(table)
            if order_column:
                # The source API orders releases by hidden SQLite rowid. Keep
                # that ordering explicitly; it is not part of release hashes.
                if order_column in columns:
                    raise ValueError('unexpected preexisting SQLite ordering column: ' + table)
                columns.append(order_column)
                rows = snapshot.execute(f'SELECT *,rowid FROM {quoted} ORDER BY rowid').fetchall()
            else:
                rows = snapshot.execute(f'SELECT * FROM {quoted}').fetchall()
            data[table] = (columns, rows)
            fingerprints[table] = _table_fingerprint(columns, rows)
        source_hash = hashlib.sha256(json.dumps(
            {'tables': fingerprints, 'schema': source_schema, 'sequences': sequence_highwater},
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with psycopg.connect(dsn) as conn:
            conn.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('eos-import:' + schema,))
            conn.execute(sql.SQL('SET LOCAL search_path TO {},pg_catalog').format(sql.Identifier(schema)))
            require_ready(conn)
            prior = conn.execute('SELECT source_store_id,source_fingerprint FROM eos_sqlite_imports').fetchone()
            if prior:
                if prior != (source_store_id, source_hash):
                    raise ValueError('EOS namespace already bound to a different source or snapshot')
                return {'source_store_id': source_store_id, 'source_fingerprint': source_hash, 'idempotent': True}
            present = {row[0] for row in conn.execute('SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname=current_schema()')}
            unknown = set(tables) - present
            if unknown:
                raise ValueError('unmigrated EOS source tables: ' + ','.join(sorted(unknown)))
            for table, (columns, _) in data.items():
                target_columns = {row[0] for row in conn.execute(
                    'SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=%s', (table,))}
                if not set(columns) <= target_columns:
                    raise ValueError('unmigrated EOS source columns: ' + table)
                count = conn.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(table))).fetchone()[0]
                if count and table not in {'semantier_schema_versions', 'benchmark_read_model_schema_metadata'}:
                    raise ValueError('EOS import requires an empty target: ' + table)
            conn.execute('DELETE FROM semantier_schema_versions')
            if 'benchmark_read_model_schema_metadata' in tables:
                conn.execute('DELETE FROM benchmark_read_model_schema_metadata')
            # EOS foreign keys are created in dependency order; import in that same order.
            dependencies = {table: {row[0] for row in conn.execute(
                "SELECT ccu.table_name FROM information_schema.table_constraints tc "
                "JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name=ccu.constraint_name "
                "AND tc.constraint_schema=ccu.constraint_schema WHERE tc.constraint_type='FOREIGN KEY' "
                "AND tc.table_schema=current_schema() AND tc.table_name=%s", (table,))} - {table}
                for table in tables}
            dependencies = {table: refs & set(tables) for table, refs in dependencies.items()}
            imported = set()
            while dependencies:
                ready = sorted(table for table, refs in dependencies.items() if not refs - imported)
                if not ready:
                    raise ValueError('EOS import dependency is unresolved')
                for table in ready:
                    columns, rows = data[table]
                    if rows:
                        statement = sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(
                            sql.Identifier(table), sql.SQL(',').join(map(sql.Identifier, columns)),
                            sql.SQL(',').join(sql.Placeholder() for _ in columns))
                        with conn.cursor() as cursor:
                            cursor.executemany(statement, rows)
                    target_rows = conn.execute(sql.SQL('SELECT {} FROM {}').format(
                        sql.SQL(',').join(map(sql.Identifier, columns)), sql.Identifier(table))).fetchall()
                    if _table_fingerprint(columns, target_rows) != fingerprints[table]:
                        raise ValueError('EOS import payload verification failed: ' + table)
                    # Explicitly imported identity IDs must not collide with the next append.
                    for column in columns:
                        sequence = conn.execute('SELECT pg_get_serial_sequence(%s,%s)', (schema + '.' + table, column)).fetchone()[0]
                        if sequence:
                            maximum = conn.execute(sql.SQL('SELECT max({}) FROM {}').format(sql.Identifier(column), sql.Identifier(table))).fetchone()[0]
                            highwater = max(maximum or 0, sequence_highwater.get(table, 0))
                            conn.execute('SELECT setval(%s,%s,%s)', (sequence, highwater or 1, highwater > 0))
                    imported.add(table)
                    del dependencies[table]
            require_ready(conn)
            conn.execute('INSERT INTO eos_sqlite_imports VALUES (%s,%s,%s)',
                         (source_store_id, source_hash, datetime.now(timezone.utc).isoformat()))
        return {'source_store_id': source_store_id, 'source_fingerprint': source_hash,
                'tables': len(tables), 'rows': sum(len(rows) for _, rows in data.values()), 'idempotent': False}
    finally:
        snapshot.close()
