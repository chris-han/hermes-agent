"""Read a bounded EOS analytics projection from governed PostgreSQL authority.

Rows come from one read-only repeatable-read transaction. This is an export
boundary for DuckDB analytics, never a connection adapter or replay authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from storage import hosted_postgres as hosted


ANALYTICS_TABLES = {
    'rea_claims': ('rea_id', 'claim_hash'),
    'journal_voucher_projections': ('jvp_id', 'content_hash'),
    'general_ledger_views': ('view_id', 'content_hash'),
    'financial_statement_packages': ('package_id', 'content_hash'),
    'tax_filing_packages': ('filing_id', 'content_hash'),
    'accounting_archive_packages': ('archive_id', 'content_hash'),
    'coa_projection_bundles': ('bundle_version', 'content_hash'),
    'real_company_admission_exceptions': ('exception_id', 'content_hash'),
}
SNAPSHOT_CONTRACT = 'eos-analytics-eight-text-tables-v1'


@dataclass(frozen=True)
class AnalyticsTable:
    columns: tuple[str, ...]
    rows: tuple[tuple[str | None, ...], ...]


@dataclass(frozen=True)
class AnalyticsSnapshot:
    tables: dict[str, AnalyticsTable]
    provenance: dict


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_analytics_snapshot(path: str | Path) -> AnalyticsSnapshot:
    """Resolve the configured owner mapping; callers cannot select a schema."""
    from psycopg import sql
    schema = hosted.mapped_store('eos', path)['schema']
    tables = {}
    fingerprints = {}
    with hosted.connect_authority('eos', path) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        for table, (id_column, hash_column) in ANALYTICS_TABLES.items():
            cursor = conn.execute(sql.SQL('SELECT * FROM {} ORDER BY {}').format(
                sql.Identifier(schema, table), sql.Identifier(id_column)))
            columns = tuple(column.name for column in cursor.description)
            # These eight governed assets currently contain only text columns.
            # Refuse schema expansion rather than silently changing export types.
            if any(column.type_code not in (25, 1043) for column in cursor.description):
                raise ValueError('unsupported EOS analytics source column type')
            rows = tuple(tuple(row) for row in cursor)
            if any(value is not None and not isinstance(value, str) for row in rows for value in row):
                raise ValueError('unsupported EOS analytics source value')
            tables[table] = AnalyticsTable(columns, rows)
            id_index, hash_index = columns.index(id_column), columns.index(hash_column)
            digest = hashlib.sha256()
            for row in rows:
                digest.update(str(row[id_index]).encode())
                digest.update(b'\0')
                digest.update(str(row[hash_index]).encode())
                digest.update(b'\n')
            fingerprints[table] = {
                'rows': len(rows), 'columns': list(columns),
                'column_types': ['text'] * len(columns),
                'content_hash': digest.hexdigest(),
                'canonical_sha256': _digest([columns, rows]),
            }
    return AnalyticsSnapshot(tables, {
        'source_kind': 'postgresql', 'schema': schema,
        'snapshot_contract': SNAPSHOT_CONTRACT,
        'analytics_tables': fingerprints,
    })


def load_duckdb_snapshot(conn, snapshot: AnalyticsSnapshot) -> None:
    """Materialize the fixed text projection into an actual DuckDB connection."""
    def identifier(value):
        return '"' + value.replace('"', '""') + '"'
    conn.execute("ATTACH ':memory:' AS eos")
    for name, table in snapshot.tables.items():
        columns = ', '.join(identifier(column) + ' VARCHAR' for column in table.columns)
        target = 'eos.' + identifier(name)
        conn.execute(f'CREATE TABLE {target} ({columns})')
        if table.rows:
            placeholders = ','.join('?' for _ in table.columns)
            conn.executemany(f'INSERT INTO {target} VALUES ({placeholders})', table.rows)
