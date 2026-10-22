"""Read-only admission checks for the manifest-derived hosted store projection.

The operator derives requirements from the complete frozen migration census and
successful owner receipts. This projection never imports, creates, or repairs a
store; runtime credentials can only validate the corresponding protected facts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from storage import hosted_postgres as pg

OWNERS = frozenset({'identity', 'eos', 'execution', 'hermes', 'standalone'})
HEX = re.compile(r'[0-9a-f]{64}')
SCHEMA = re.compile(r'[a-z][a-z0-9_]{0,62}')


def _refuse(message):
    raise pg.HostedStorageUnavailable(message)


def requirements():
    try:
        value = json.loads(os.environ.get('SEMANTIER_HOSTED_STORE_REQUIREMENTS', '{}'))
        if (not isinstance(value, dict) or value.get('format_version') != 1
                or value.get('complete') is not True
                or value.get('provenance') not in {'fresh', 'migrated'}
                or not HEX.fullmatch(value.get('manifest_sha256', ''))
                or not isinstance(value.get('entries'), list) or not value['entries']):
            _refuse('required hosted authority manifest is incomplete')
        entries = value['entries']
        if {entry.get('owner') for entry in entries} != OWNERS:
            _refuse('required hosted authority owner census is incomplete')
        seen = set()
        for entry in entries:
            schema = entry.get('schema', '')
            if (not SCHEMA.fullmatch(schema) or schema in {'public', 'information_schema'}
                    or schema.startswith('pg_')):
                _refuse('invalid required authority schema')
            if not isinstance(entry.get('target_path'), str) or not Path(entry['target_path']).is_absolute():
                _refuse('required authority path must be absolute')
            if entry['owner'] in {'hermes', 'execution'} and (
                    not isinstance(entry.get('target_home'), str) or not Path(entry['target_home']).is_absolute()):
                _refuse('required native home must be absolute')
            key = (entry['owner'], schema, entry.get('source_store_id'), entry.get('family'))
            if key in seen:
                _refuse('duplicate required authority destination')
            seen.add(key)
            if value['provenance'] == 'migrated' and (
                    not entry.get('source_store_id') or not HEX.fullmatch(entry.get('source_digest', ''))):
                _refuse('required hosted import receipt is incomplete')
        return value
    except (ValueError, TypeError, KeyError, AttributeError):
        _refuse('invalid required hosted authority manifest')


def _mapping(name):
    try:
        value = json.loads(os.environ.get(name, '{}'))
        if not isinstance(value, dict) or not value:
            _refuse('required hosted scope mapping is incomplete')
        if any(not isinstance(key, str) or not isinstance(entry, dict) for key, entry in value.items()):
            _refuse('invalid required hosted scope mapping')
        return value
    except (ValueError, TypeError):
        _refuse('invalid required hosted scope mapping')


def _tables(asset):
    return set(re.findall(r'CREATE TABLE IF NOT EXISTS (\w+)', asset))


def _immutable_metadata(conn, schema, names):
    for name in names:
        row = conn.execute('SELECT has_table_privilege(current_user,%s,%s)',
                           (schema + '.' + name, 'INSERT,UPDATE,DELETE,TRUNCATE')).fetchone()
        if row[0]:
            _refuse('application can mutate protected hosted metadata: ' + schema + '.' + name)
    if conn.execute('SELECT has_schema_privilege(current_user,%s,%s)', (schema, 'CREATE')).fetchone()[0]:
        _refuse('application can create hosted authority schema objects: ' + schema)


def _native_scope(conn, scope, schema):
    from hermes_state_postgresql import SessionScope, require_scope_ready
    try:
        selected = SessionScope(**scope)
    except (TypeError, ValueError):
        _refuse('invalid required native authority scope')
    if selected.schema != schema:
        _refuse('required native authority namespace differs from scope')
    require_scope_ready(conn, selected)
    # Every hosted home needs the joined execution and finite auxiliary families,
    # even when the frozen source for that home contains no rows in that family.
    pg.require_tables(conn, *_tables(pg.execution_schema_sql()))
    pg.require_meeting_ready(conn)
    from storage.hosted_auxiliary_paths import auxiliary_schema_sql
    pg.require_tables(conn, *_tables(auxiliary_schema_sql()))
    _immutable_metadata(conn, schema, ('schema_version', 'scope_binding', 'migration_receipts',
                                     'hosted_room_legacy_imports', 'semantier_execution_imports', 'standalone_imports', 'benchmark_read_model_schema_metadata'))


def _owner(conn, entry):
    owner, schema = entry['owner'], entry['schema']
    if owner == 'identity':
        if schema != 'semantier_identity':
            _refuse('required identity authority namespace differs')
        from agents.auth_db_postgres import _DDL, VERSION
        names = set(re.findall(r'CREATE TABLE IF NOT EXISTS semantier_identity\.(\w+)', _DDL))
        pg.require_tables(conn, *names, 'workspace_runtime_bindings', 'workspace_provisioning_contract', 'workspace_provisioning_contract_history')
        if conn.execute('SELECT max(version) FROM schema_version').fetchone()[0] != VERSION:
            _refuse('required identity schema version differs')
        _immutable_metadata(conn, schema, ('schema_version', 'imports', 'workspace_runtime_bindings', 'workspace_provisioning_contract', 'workspace_provisioning_contract_history'))
        if not conn.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_trigger "
            "WHERE tgrelid='semantier_identity.workspace_provisioning_contract_history'::regclass "
            "AND tgname='workspace_contract_history_immutable' AND tgenabled IN ('O','A'))"
        ).fetchone()[0]:
            _refuse('workspace provisioning history append-only guard is absent')
    elif owner == 'eos':
        from eos.postgres import require_ready
        require_ready(conn)
        _immutable_metadata(conn, schema, ('semantier_schema_versions', 'eos_sqlite_imports', 'eos_postgres_schema_contract', 'benchmark_read_model_schema_metadata'))
    elif owner in {'hermes', 'execution'}:
        _native_scope(conn, entry.get('scope', {}), schema)
    else:
        from storage.hosted_auxiliary_paths import auxiliary_schema_sql
        pg.require_tables(conn, *_tables(auxiliary_schema_sql()))
        _immutable_metadata(conn, schema, ('standalone_imports', 'benchmark_read_model_schema_metadata'))


def _import_receipt(conn, entry):
    # Explicit per-owner metadata contracts; no fallback source or inferred ID.
    owner = entry['owner']
    table, digest = {
        'identity': ('imports', 'source_fingerprint'),
        'eos': ('eos_sqlite_imports', 'source_fingerprint'),
        'execution': ('semantier_execution_imports', 'source_fingerprint'),
        'standalone': ('standalone_imports', 'source_fingerprint'),
        'hermes': ('migration_receipts', 'source_digest'),
    }[owner]
    if owner == 'hermes':
        row = conn.execute('SELECT receipt_json FROM migration_receipts WHERE source_digest=%s',
                           (entry['source_digest'],)).fetchone()
        if row is None:
            _refuse('required native import receipt is absent')
        receipt = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        scope = entry['scope']
        if (receipt.get('family') != entry.get('family', 'sessions')
                or receipt.get('scope') != [scope[k] for k in ('tenant_id', 'workspace_id', 'source_store_id')]):
            _refuse('required native import receipt scope or family differs')
    else:
        # table/column names are the fixed owner enum above, never config SQL.
        row = conn.execute(f'SELECT {digest} FROM {table} WHERE source_store_id=%s',
                           (entry['source_store_id'],)).fetchone()
        if row is None or row[0] != entry['source_digest']:
            _refuse('required hosted import receipt is absent or stale: ' + owner)


def require_ready(dsn=None):
    if not pg.enabled():
        _refuse('hosted PostgreSQL backend must be selected')
    projection = requirements()
    if os.environ.get('HERMES_STATE_BACKEND') != 'postgresql':
        _refuse('all required hosted PostgreSQL backends must be selected')
    authority = _mapping('SEMANTIER_AUTHORITY_STORE_MAP')
    homes = _mapping('HERMES_STATE_SCOPE_MAP')
    entries = projection['entries']
    for path, configured in authority.items():
        if not any(entry.get('target_path') == path and entry['owner'] == configured.get('owner')
                   and entry['schema'] == configured.get('schema') for entry in entries):
            _refuse('configured authority mapping differs from required destination')
    for home, scope in homes.items():
        if not any(entry.get('target_home') == home and entry.get('scope') == scope
                   and entry['owner'] in {'hermes', 'execution'} for entry in entries):
            _refuse('configured native home differs from required destination')
    for entry in entries:
        if entry['owner'] in {'hermes', 'execution'}:
            if homes.get(entry.get('target_home')) != entry.get('scope'):
                _refuse('required native home is absent from runtime mapping')
        else:
            mapped = authority.get(entry.get('target_path'))
            if not mapped or (mapped.get('owner'), mapped.get('schema')) != (entry['owner'], entry['schema']):
                _refuse('required authority destination is absent from runtime mapping')
    import psycopg
    from psycopg import sql
    try:
        with psycopg.connect(dsn or pg.database_url(), connect_timeout=5) as conn:
            conn.execute('SET TRANSACTION READ ONLY')
            app_role = conn.execute('SELECT current_user').fetchone()[0]
            if os.environ.get('HERMES_DATABASE_URL') != (dsn or pg.database_url()):
                _refuse('native and core hosted database connections differ')
            checked = set()
            for entry in entries:
                conn.execute(sql.SQL('SET LOCAL search_path TO {},pg_catalog').format(sql.Identifier(entry['schema'])))
                key = (entry['owner'], entry['schema'])
                if key not in checked:
                    _owner(conn, entry)
                    checked.add(key)
                if projection['provenance'] == 'migrated':
                    _import_receipt(conn, entry)
            from agents.workspace_storage_provisioning import expected_workspace_provisioning_contract
            root = str(Path(os.environ['SEMANTIER_LOCAL_STATE_DIR']).parent / 'workspaces')
            if os.environ.get('SEMANTIER_WORKSPACES_ROOT', root) != root:
                _refuse('workspace root differs from canonical runtime resolver')
            expected = expected_workspace_provisioning_contract(app_role=app_role, workspace_root=root)
            actual = conn.execute('SELECT template_hash,workspace_root,app_role FROM semantier_identity.workspace_provisioning_contract WHERE singleton=1').fetchone()
            if actual is None or tuple(actual) != tuple(expected[k] for k in ('template_hash', 'workspace_root', 'app_role')):
                _refuse('workspace provisioning contract is absent or stale; explicit migration required')
            # Freshly allocated signup homes have no legacy snapshot. Their only
            # admissible binding is the migration-installed allocation registry.
            rows = conn.execute('SELECT b.user_id,b.workspace_id,b.tenant_id,b.source_store_id,b.schema_name,b.hermes_home,u.payload_json FROM semantier_identity.workspace_runtime_bindings b LEFT JOIN semantier_identity.users u USING(user_id)').fetchall()
            from agents.workspace_storage_provisioning import registered_workspace_scope
            for row in rows:
                selected = registered_workspace_scope(row, root)
                tenant, workspace, source = selected.identity
                schema = selected.schema
                conn.execute(sql.SQL('SET LOCAL search_path TO {},pg_catalog').format(sql.Identifier(schema)))
                _native_scope(conn, {'tenant_id': tenant, 'workspace_id': workspace, 'source_store_id': source}, schema)
    except psycopg.Error:
        _refuse('required hosted authority is unavailable or application privileges are incomplete')
    except (KeyError, ValueError, TypeError):
        _refuse('required hosted authority contract is invalid')
    return {'readiness': 'PASS', 'manifest_sha256': projection['manifest_sha256'], 'destinations': len(entries)}
