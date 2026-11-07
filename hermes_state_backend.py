"""Explicit native storage selection. Desktop defaults to SQLite.

Hosted source-home mappings are supplied by the governed deployment manifest,
including source-to-container-home relocation. A pathname never implies a tenant.
"""
import json
import os
from pathlib import Path


def postgres_enabled():
    backend=os.environ.get('HERMES_STATE_BACKEND','sqlite').strip().lower()
    if backend not in ('sqlite','postgresql'): raise ValueError('Unsupported HERMES_STATE_BACKEND')
    return backend=='postgresql'


def session_store_key(db):
    """Stable owning-store key: a native scope tuple or the actual SQLite path."""
    identity = getattr(db, 'storage_identity', None)
    if identity is not None:
        if (not isinstance(identity, tuple) or len(identity) != 4
                or identity[0] != 'postgresql'
                or any(not isinstance(part, str) or not part for part in identity)):
            raise ValueError('Invalid native PostgreSQL storage identity')
        return identity
    path = getattr(db, 'db_path', None)
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError('Session store has no supported storage identity')
    return Path(path).expanduser().resolve()


def session_store_home(db):
    """Actual profile home; database locators never supply a hosted filesystem path."""
    key = session_store_key(db)
    if isinstance(key, tuple):
        home = getattr(db, 'home', None)
        if not isinstance(home, (str, os.PathLike)):
            raise ValueError('Native PostgreSQL session store has no profile home')
        return Path(home).expanduser().resolve()
    return key.parent


def resolve_postgres_scope(home):
    """Resolve a frozen migration mapping or the core-governed signup registry.

    The registry is read only here. Resource allocation belongs to the explicit
    authenticated signup transaction; opening a native store never provisions it.
    """
    from hermes_state_postgresql import SessionScope
    mappings=json.loads(os.environ.get('HERMES_STATE_SCOPE_MAP','{}'))
    if not isinstance(mappings,dict):
        raise ValueError('Explicit governed PostgreSQL session home mapping required')
    key=str(Path(home).expanduser().resolve())
    if key in mappings:
        item=mappings[key]
        if not isinstance(item,dict) or set(item)!= {'tenant_id','workspace_id','source_store_id'}:
            raise ValueError('Invalid governed PostgreSQL session home mapping')
        return SessionScope(**item)
    dsn=os.environ.get('HERMES_DATABASE_URL')
    if dsn:
        import psycopg
        try:
            with psycopg.connect(dsn) as conn:
                conn.execute('SET TRANSACTION READ ONLY')
                row=conn.execute("""SELECT b.tenant_id,b.workspace_id,b.source_store_id,b.schema_name
                    FROM semantier_identity.workspace_runtime_bindings b
                    JOIN semantier_identity.users u ON u.user_id=b.user_id
                     AND u.payload_json::jsonb->>'user_id'=u.user_id
                     AND u.payload_json::jsonb->>'workspace_id'=b.workspace_id
                    WHERE b.hermes_home=%s""",(key,)).fetchone()
                if row is None and conn.execute(
                    "SELECT to_regclass('semantier_identity.winman_profile_bindings') IS NOT NULL"
                ).fetchone()[0]:
                    row=conn.execute("""SELECT b.tenant_id,b.workspace_id,b.source_store_id,b.schema_name,b.stage_device,b.stage_inode
                        FROM semantier_identity.winman_profile_bindings b
                        JOIN semantier_identity.workspace_runtime_bindings w
                          ON w.user_id=b.user_id AND w.workspace_id=b.workspace_id
                        JOIN semantier_identity.users u ON u.user_id=b.user_id
                        WHERE b.hermes_home=%s AND b.materialized AND b.bot_id ~ '^bot_[a-f0-9]{32}$'
                          AND b.hermes_home=w.hermes_home || '/profiles/winman-' || substr(b.bot_id,5)
                          AND b.tenant_id=w.tenant_id
                          AND b.source_store_id='winman-native-' || b.user_id
                          AND u.payload_json::jsonb->>'user_id'=b.user_id
                          AND u.payload_json::jsonb->>'workspace_id'=b.workspace_id
                          AND u.payload_json::jsonb->>'organization_id'=b.organization_id
                          AND u.payload_json::jsonb->>'membership_status'='active'
                          AND EXISTS (SELECT 1 FROM jsonb_array_elements(u.payload_json::jsonb->'organization_memberships') m
                                      WHERE m->>'organization_id'=b.organization_id
                                        AND m->>'membership_status'='active')""",(key,)).fetchone()
        except psycopg.errors.UndefinedTable:
            row=None
        if row:
            scope=SessionScope(*row[:3])
            if scope.schema!=row[3]:
                raise ValueError('Governed PostgreSQL namespace binding conflict')
            if len(row) > 4:
                try:
                    profile=Path(key)
                    stat=profile.stat()
                    if not profile.is_dir() or (stat.st_dev,stat.st_ino)!=(row[4],row[5]):
                        raise ValueError('Governed Winman profile identity conflict')
                except OSError as exc:
                    raise ValueError('Governed Winman profile identity conflict') from exc
            return scope
    raise ValueError('Explicit governed PostgreSQL session home mapping required')


def open_session_db(db_path=None,*,home=None,read_only=False):
    if postgres_enabled():
        from hermes_state_postgresql import PostgreSQLSessionDB
        from hermes_constants import get_hermes_home
        if db_path is not None and Path(db_path).name!='state.db':
            raise ValueError('PostgreSQL state store must resolve an explicit source home')
        resolved_home=Path(home) if home is not None else (Path(db_path).parent if db_path is not None else get_hermes_home())
        dsn=os.environ.get('HERMES_DATABASE_URL')
        if not dsn: raise ValueError('HERMES_DATABASE_URL required in PostgreSQL mode')
        return PostgreSQLSessionDB(dsn,resolve_postgres_scope(resolved_home),home=resolved_home,read_only=read_only)
    from hermes_state import SessionDB
    return SessionDB(db_path=db_path,read_only=read_only)


def dispatch_cron_operation(module,name,sqlite_fn):
    from functools import wraps
    @wraps(sqlite_fn)
    def selected(*args,**kwargs):
        if postgres_enabled():
            from hermes_state_postgresql_cron import dispatch_operation
            return dispatch_operation(module,name,*args,**kwargs)
        return sqlite_fn(*args,**kwargs)
    return selected
