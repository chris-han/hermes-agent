"""Native PostgreSQL identity owner. No SQLite SQL or runtime import fallback.

Opaque legacy payloads remain intact. Snapshot compatibility writes only changed
rows with a compare-and-swap fence; an outer identity mutation transaction keeps
membership, channel and audit mutations together across existing API calls.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

SCHEMA = 'semantier_identity'
VERSION = 1
_current = ContextVar('semantier_identity_connection', default=None)


class IdentityConflict(ValueError):
    """The row changed after it was read; the caller must reload before retrying."""


class SnapshotDict(dict):
    def __init__(self, value, *, original=None, baseline=None):
        super().__init__(value)
        self.original = deepcopy(value if original is None else original)
        self.baseline = deepcopy(value if baseline is None else baseline)


class SnapshotList(list):
    def __init__(self, values, *, keys=None, original=None, identities=None):
        super().__init__(values)
        self.keys = keys or []
        self.original = deepcopy(dict(zip(self.keys, values)) if original is None else original)
        self.identities = identities or {id(v): k for k,v in zip(self.keys, values)}


def retain_snapshot(source, values, *, normalized=False):
    if isinstance(source, SnapshotDict):
        return SnapshotDict(values, original=source.original, baseline=values if normalized else source.baseline)
    if isinstance(source, SnapshotList):
        return SnapshotList(values, original=source.original, identities=source.identities)
    return values


def _dsn():
    value = os.environ.get('SEMANTIER_DATABASE_URL', '').strip()
    if not value:
        raise RuntimeError('PostgreSQL identity requires SEMANTIER_DATABASE_URL')
    return value


@contextmanager
def connection():
    current = _current.get()
    if current is not None:
        yield current
        return
    with psycopg.connect(_dsn(), row_factory=dict_row, connect_timeout=5) as c:
        c.execute("SET LOCAL TIME ZONE 'UTC'")
        yield c


@contextmanager
def write_transaction():
    if _current.get() is not None:
        yield
        return
    with connection() as c:
        # Existing mutations can change several memberships, channels and events.
        # Serialize their read/decide/write boundary across processes, not just threads.
        c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('semantier-identity-mutation',))
        token = _current.set(c)
        try:
            yield
        finally:
            _current.reset(token)


# Explicit owner schema: relational keys/metadata preserve the deployed auth.db
# contract. JSON text retains exact password/channel payloads without translation.
_DDL = '''
CREATE SCHEMA IF NOT EXISTS semantier_identity;
CREATE TABLE IF NOT EXISTS semantier_identity.schema_version(version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS semantier_identity.users(user_id text PRIMARY KEY,payload_json text NOT NULL,updated_at timestamptz);
CREATE UNIQUE INDEX IF NOT EXISTS identity_password_login_unique ON semantier_identity.users(lower(payload_json::jsonb->>'password_login_name')) WHERE nullif(payload_json::jsonb->>'password_login_name','') IS NOT NULL;
CREATE TABLE IF NOT EXISTS semantier_identity.organizations(organization_id text PRIMARY KEY,payload_json text NOT NULL,updated_at timestamptz);
CREATE TABLE IF NOT EXISTS semantier_identity.organization_events(event_id text PRIMARY KEY,organization_id text,created_at timestamptz,payload_json text NOT NULL);
CREATE INDEX IF NOT EXISTS identity_org_events ON semantier_identity.organization_events(organization_id,created_at,event_id);
CREATE TABLE IF NOT EXISTS semantier_identity.gateway_correlations(correlation_key text PRIMARY KEY,platform text,owner_user_id text,status text,updated_at timestamptz,payload_json text NOT NULL);
CREATE TABLE IF NOT EXISTS semantier_identity.weixin_runtime_accounts(account_id text PRIMARY KEY,owner_user_id text,owner_workspace_id text,external_user_id text,runtime_session_state text,runtime_session_updated_at timestamptz,saved_at timestamptz,payload_json text NOT NULL);
CREATE TABLE IF NOT EXISTS semantier_identity.weixin_sync_state(account_id text PRIMARY KEY,get_updates_buf text NOT NULL,updated_at timestamptz NOT NULL);
CREATE TABLE IF NOT EXISTS semantier_identity.weixin_context_tokens(account_id text NOT NULL,user_id text NOT NULL,context_token text NOT NULL,updated_at timestamptz NOT NULL,PRIMARY KEY(account_id,user_id));
CREATE TABLE IF NOT EXISTS semantier_identity.gateway_pairing_pending(scope text NOT NULL,platform text NOT NULL,code text NOT NULL,user_id text NOT NULL,user_name text,created_at timestamptz NOT NULL,PRIMARY KEY(scope,platform,code));
CREATE TABLE IF NOT EXISTS semantier_identity.gateway_pairing_approved(scope text NOT NULL,platform text NOT NULL,user_id text NOT NULL,user_name text,approved_at timestamptz NOT NULL,PRIMARY KEY(scope,platform,user_id));
CREATE TABLE IF NOT EXISTS semantier_identity.gateway_pairing_limits(scope text NOT NULL,key text NOT NULL,value double precision NOT NULL,deadline_at timestamptz,PRIMARY KEY(scope,key));
ALTER TABLE semantier_identity.gateway_pairing_limits ADD COLUMN IF NOT EXISTS deadline_at timestamptz;
CREATE TABLE IF NOT EXISTS semantier_identity.weixin_login_states(state text PRIMARY KEY,status text,created_at timestamptz,updated_at timestamptz,payload_json text NOT NULL);
CREATE TABLE IF NOT EXISTS semantier_identity.feishu_link_states(state text PRIMARY KEY,status text,created_at timestamptz,updated_at timestamptz,payload_json text NOT NULL);
CREATE TABLE IF NOT EXISTS semantier_identity.feishu_bot_configs(owner_user_id text PRIMARY KEY,owner_workspace_id text,app_id text,domain text,saved_at timestamptz,updated_at timestamptz,payload_json text NOT NULL);
CREATE TABLE IF NOT EXISTS semantier_identity.user_settings(user_id text PRIMARY KEY,settings_json text NOT NULL,updated_at timestamptz NOT NULL);
CREATE TABLE IF NOT EXISTS semantier_identity.workspace_configs(workspace_id text PRIMARY KEY,config_json text NOT NULL,updated_at timestamptz NOT NULL);
CREATE TABLE IF NOT EXISTS semantier_identity.imports(source_path text PRIMARY KEY,source_fingerprint text NOT NULL,imported_at timestamptz NOT NULL DEFAULT now(),table_counts jsonb NOT NULL,source_store_id text);
ALTER TABLE semantier_identity.imports ADD COLUMN IF NOT EXISTS source_store_id text;
ALTER TABLE semantier_identity.imports ADD COLUMN IF NOT EXISTS receipt_json jsonb;
CREATE TABLE IF NOT EXISTS semantier_identity.gateway_projection_intents(user_id text PRIMARY KEY REFERENCES semantier_identity.users,organization_id text NOT NULL REFERENCES semantier_identity.organizations,account_id text NOT NULL,proxy_user_id text UNIQUE NOT NULL,proxy_team_id text NOT NULL,proxy_key_alias text UNIQUE NOT NULL,proxy_key_hash text UNIQUE,status text NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','complete')),attempts integer NOT NULL DEFAULT 0,last_error text,updated_at timestamptz NOT NULL DEFAULT now());
'''

# Exact table metadata is shared by the narrowly scoped importer and row owner.
_TABLES = {
    'users': ('user_id', 'payload_json', 'updated_at'),
    'organizations': ('organization_id', 'payload_json', 'updated_at'),
    'organization_events': ('event_id', 'organization_id', 'created_at', 'payload_json'),
    'gateway_correlations': ('correlation_key', 'platform', 'owner_user_id', 'status', 'updated_at', 'payload_json'),
    'weixin_runtime_accounts': ('account_id', 'owner_user_id', 'owner_workspace_id', 'external_user_id', 'runtime_session_state', 'runtime_session_updated_at', 'saved_at', 'payload_json'),
    'weixin_sync_state': ('account_id', 'get_updates_buf', 'updated_at'),
    'weixin_context_tokens': ('account_id', 'user_id', 'context_token', 'updated_at'),
    'gateway_pairing_pending': ('scope', 'platform', 'code', 'user_id', 'user_name', 'created_at'),
    'gateway_pairing_approved': ('scope', 'platform', 'user_id', 'user_name', 'approved_at'),
    'gateway_pairing_limits': ('scope', 'key', 'value'),
    'weixin_login_states': ('state', 'status', 'created_at', 'updated_at', 'payload_json'),
    'feishu_link_states': ('state', 'status', 'created_at', 'updated_at', 'payload_json'),
    'feishu_bot_configs': ('owner_user_id', 'owner_workspace_id', 'app_id', 'domain', 'saved_at', 'updated_at', 'payload_json'),
    'user_settings': ('user_id', 'settings_json', 'updated_at'),
    'workspace_configs': ('workspace_id', 'config_json', 'updated_at'),
}
_KEYS = {'weixin_context_tokens': ('account_id', 'user_id'), 'gateway_pairing_pending': ('scope','platform','code'), 'gateway_pairing_approved': ('scope','platform','user_id'), 'gateway_pairing_limits': ('scope','key')}


def migrate():
    with connection() as c:
        c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('semantier-identity-migrate',))
        c.execute(_DDL)
        c.execute('INSERT INTO semantier_identity.schema_version VALUES(%s,now()) ON CONFLICT DO NOTHING', (VERSION,))
    return {'schema': SCHEMA, 'version': VERSION}


def ensure_auth_db(*, json_loader=None):
    # Legacy JSON is deliberately ignored in hosted mode: import is an explicit
    # operator action, and an outage/missing schema never creates local authority.
    with connection() as c:
        row = c.execute('SELECT max(version) AS version FROM semantier_identity.schema_version').fetchone()
        if row['version'] != VERSION:
            raise RuntimeError('unsupported PostgreSQL identity schema; explicit migration required')
        if c.execute("SELECT to_regprocedure('semantier_identity.provision_workspace_storage(text,text)') IS NOT NULL AS ready").fetchone()['ready'] is not True:
            raise RuntimeError('workspace storage provisioning migration required')
        c.execute('SELECT 1 FROM semantier_identity.workspace_runtime_bindings LIMIT 0')
        for name in _TABLES:
            c.execute(sql.SQL('SELECT 1 FROM {}.{} LIMIT 0').format(sql.Identifier(SCHEMA), sql.Identifier(name)))


def _stamp(value):
    if value is None or value == '':
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    # Explicit import normalization for legacy naive UTC values.
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _payload_row(table, key, value):
    row = {k: value.get(k) for k in _TABLES[table]}
    row[_TABLES[table][0]] = key
    row['payload_json'] = _json(value)
    for name in _TABLES[table]:
        if name.endswith('_at') or name == 'runtime_session_updated_at':
            row[name] = _stamp(row[name])
    return row


def _insert(c, table, row, *, upsert=False):
    columns = _TABLES[table]
    query = sql.SQL('INSERT INTO {}.{} ({}) VALUES ({})').format(sql.Identifier(SCHEMA), sql.Identifier(table), sql.SQL(',').join(map(sql.Identifier,columns)), sql.SQL(',').join(sql.Placeholder() for _ in columns))
    if upsert:
        keys = _KEYS.get(table, columns[:1])
        query += sql.SQL(' ON CONFLICT ({}) DO UPDATE SET {}').format(sql.SQL(',').join(map(sql.Identifier,keys)), sql.SQL(',').join(sql.SQL('{}=excluded.{}').format(sql.Identifier(k),sql.Identifier(k)) for k in columns if k not in keys))
    c.execute(query, tuple(row.get(k) for k in columns))


def _load_map(table):
    key = _TABLES[table][0]
    with connection() as c:
        rows = c.execute(sql.SQL('SELECT {},payload_json FROM {}.{} ORDER BY {}').format(sql.Identifier(key),sql.Identifier(SCHEMA),sql.Identifier(table),sql.Identifier(key))).fetchall()
    return SnapshotDict({r[key]: json.loads(r['payload_json']) for r in rows})


def _save_map(table, values):
    original = getattr(values, 'original', {})
    baseline = getattr(values, 'baseline', original)
    key_column = _TABLES[table][0]
    with connection() as c:
        for key in sorted(set(original) | set(values)):
            before, after = original.get(key), values.get(key)
            if baseline.get(key) == after:
                continue
            if before is None:
                _insert(c,table,_payload_row(table,key,after),upsert=not isinstance(values,SnapshotDict))
                continue
            predicate = sql.SQL('{}=%s AND payload_json::jsonb=%s::jsonb').format(sql.Identifier(key_column))
            if after is None:
                result = c.execute(sql.SQL('DELETE FROM {}.{} WHERE ').format(sql.Identifier(SCHEMA),sql.Identifier(table)) + predicate, (key,_json(before)))
            else:
                row = _payload_row(table,key,after)
                cols = [k for k in _TABLES[table] if k != key_column]
                query = sql.SQL('UPDATE {}.{} SET {} WHERE ').format(sql.Identifier(SCHEMA),sql.Identifier(table),sql.SQL(',').join(sql.SQL('{}=%s').format(sql.Identifier(k)) for k in cols)) + predicate
                result = c.execute(query,tuple(row[k] for k in cols)+(key,_json(before)))
            if result.rowcount != 1:
                raise IdentityConflict('identity row changed; reload before retrying')
            membership_fields=('organization_memberships','organization_id','membership_status','member_role')
            if table=='users' and after is not None and any(before.get(field)!=after.get(field) for field in membership_fields):
                from agents.model_gateway.account_sync_policy import enqueue_membership_change
                enqueue_membership_change(c,key)


def load_users(): return _load_map('users')
def save_users(users): return _save_map('users',users)
def load_organizations(): return _load_map('organizations')
def save_organizations(organizations): return _save_map('organizations',organizations)
def load_weixin_login_states(): return _load_map('weixin_login_states')
def save_weixin_login_states(states): return _save_map('weixin_login_states',states)
def load_feishu_link_states(): return _load_map('feishu_link_states')
def save_feishu_link_states(states): return _save_map('feishu_link_states',states)


def _load_list(table):
    values = _load_map(table)
    return SnapshotList(list(values.values()), keys=list(values), original=values.original)


def _save_list(table, values):
    identities = getattr(values,'identities',{})
    mapped = {}
    for value in values:
        key = value.get('event_id') if table == 'organization_events' else identities.get(id(value))
        key = key or uuid4().hex
        mapped[key] = value
    if isinstance(values,SnapshotList):
        mapped = SnapshotDict(mapped,original=values.original,baseline=values.original)
    _save_map(table,mapped)


def load_organization_events():
    values = _load_list('organization_events')
    values.sort(key=lambda e: (e.get('created_at') or '',e.get('event_id') or ''))
    return values


def save_organization_events(events):
    # Events are append-only even when an old caller submits a retained list.
    with connection() as c:
        for event in events:
            key=event['event_id']
            existing=c.execute('SELECT payload_json FROM semantier_identity.organization_events WHERE event_id=%s',(key,)).fetchone()
            if existing:
                if json.loads(existing['payload_json']) != event:
                    raise IdentityConflict('organization event history is immutable')
            else:
                _insert(c,'organization_events',_payload_row('organization_events',key,event))


def load_gateway_correlations(): return _load_list('gateway_correlations')
def save_gateway_correlations(correlations): return _save_list('gateway_correlations',correlations)


def insert_password_user(user):
    try:
        with connection() as c:
            _insert(c,'users',_payload_row('users',user['user_id'],user))
    except psycopg.errors.UniqueViolation as exc:
        raise ValueError('login name already in use') from exc


def _one_payload(table, key):
    values = _load_map(table)
    return values.get(key)


def load_weixin_runtime_accounts(): return list(_load_map('weixin_runtime_accounts').values())
def get_weixin_runtime_account(account_id): return _one_payload('weixin_runtime_accounts',account_id.strip())


def save_weixin_runtime_account(account):
    from agents.auth_db import _weixin_runtime_account_row
    row=_weixin_runtime_account_row(account)
    if row is None: raise ValueError('weixin runtime account requires account_id')
    with connection() as c:
        _insert(c,'weixin_runtime_accounts',_payload_row('weixin_runtime_accounts',row[0],json.loads(row[-1])),upsert=True)


def delete_weixin_runtime_accounts_for_user(user_id):
    with connection() as c: c.execute('DELETE FROM semantier_identity.weixin_runtime_accounts WHERE owner_user_id=%s',(user_id,))


def update_weixin_runtime_account_state(account_id, *, runtime_session_state, runtime_session_updated_at, runtime_session_error=None):
    with write_transaction():
        account=get_weixin_runtime_account(account_id)
        if account is None: return None
        account.update(runtime_session_state=runtime_session_state,runtime_session_updated_at=runtime_session_updated_at)
        if runtime_session_error: account['runtime_session_error']=runtime_session_error
        else: account.pop('runtime_session_error',None)
        save_weixin_runtime_account(account)
        return account


def save_feishu_bot_config(config):
    from agents.auth_db import _feishu_bot_config_row
    row=_feishu_bot_config_row(config)
    if row is None: raise ValueError('feishu bot config requires owner_user_id')
    with connection() as c:
        _insert(c,'feishu_bot_configs',_payload_row('feishu_bot_configs',row[0],json.loads(row[-1])),upsert=True)


def get_feishu_bot_config_for_user(owner_user_id): return _one_payload('feishu_bot_configs',owner_user_id.strip())
def list_feishu_bot_configs(): return list(_load_map('feishu_bot_configs').values())


def get_feishu_bot_config_for_workspace(owner_workspace_id):
    with connection() as c:
        row=c.execute('SELECT payload_json FROM semantier_identity.feishu_bot_configs WHERE owner_workspace_id=%s ORDER BY updated_at DESC,owner_user_id LIMIT 1',(owner_workspace_id.strip(),)).fetchone()
        return json.loads(row['payload_json']) if row else None


def delete_feishu_bot_config_for_user(owner_user_id):
    with connection() as c: c.execute('DELETE FROM semantier_identity.feishu_bot_configs WHERE owner_user_id=%s',(owner_user_id,))


def _config(table, key, column):
    with connection() as c:
        row=c.execute(sql.SQL('SELECT {} FROM {}.{} WHERE {}=%s').format(sql.Identifier(column),sql.Identifier(SCHEMA),sql.Identifier(table),sql.Identifier(_TABLES[table][0])),(key,)).fetchone()
        return json.loads(row[column]) if row else {}


def get_user_settings_for_user(user_id): return _config('user_settings',user_id.strip(),'settings_json')
def get_workspace_config(workspace_id): return _config('workspace_configs',workspace_id.strip(),'config_json')


def upsert_user_settings_for_user(user_id,updates):
    if not user_id.strip(): raise ValueError('user_id required')
    with connection() as c:
        row=c.execute('''INSERT INTO semantier_identity.user_settings VALUES(%s,%s,now()) ON CONFLICT(user_id) DO UPDATE SET settings_json=(semantier_identity.user_settings.settings_json::jsonb || excluded.settings_json::jsonb)::text,updated_at=now() RETURNING settings_json''',(user_id.strip(),_json(updates))).fetchone()
        return json.loads(row['settings_json'])


def save_workspace_config(workspace_id,config):
    if not workspace_id.strip() or not isinstance(config,dict): raise ValueError('workspace config must be a JSON object with workspace_id')
    with connection() as c:
        _insert(c,'workspace_configs',{'workspace_id':workspace_id,'config_json':_json(config),'updated_at':datetime.now(timezone.utc)},upsert=True)
    return config


def claim_feishu_message_link(state,app_id,now):
    with connection() as c:
        row=c.execute('''UPDATE semantier_identity.feishu_link_states SET status='consuming',payload_json=jsonb_set(payload_json::jsonb,'{status}','"consuming"')::text WHERE state=%s AND status='pending' AND payload_json::jsonb->>'link_mode'='bot_message' AND payload_json::jsonb->>'bot_app_id'=%s AND created_at + ((payload_json::jsonb->>'expire_in')::double precision * interval '1 second') > %s RETURNING payload_json''',(state,app_id,_stamp(now))).fetchone()
        return json.loads(row['payload_json']) if row else None


def provision_personal_signup(user, *, insert_user=True):
    """Create or repair only this canonical user's private personal account.

    Every local identity/access/desired/outbox mutation joins one real PG
    transaction. Existing business selections, revoked membership/access,
    financial history and passwords are never replaced by login repair.
    """
    from agents.gateway_identity import _resolve_current_membership
    from agents.model_gateway.account_sync_store import seed_personal_sync
    uid=user['user_id']; org='personal-'+uid
    billing=os.environ.get('SEMANTIER_MODEL_GATEWAY_BILLING_SCHEMA','semantier_billing')
    with write_transaction(), connection() as c:
        if insert_user:
            insert_password_user(user)
        row=c.execute('SELECT payload_json FROM semantier_identity.users WHERE user_id=%s FOR UPDATE',(uid,)).fetchone()
        if row is None:raise ValueError('Canonical signup user required')
        canonical=json.loads(row['payload_json'])
        user.clear();user.update(canonical)
        membership=_resolve_current_membership(user)
        selected=membership.get('organization_id')
        if selected and (selected!=org or membership.get('membership_status')!='active' or membership.get('member_role')!='owner'):
            return user
        organization=c.execute('SELECT payload_json FROM semantier_identity.organizations WHERE organization_id=%s FOR UPDATE',(org,)).fetchone()
        if organization and json.loads(organization['payload_json']).get('created_by')!=uid:
            raise IdentityConflict('Personal organization ownership conflict')
        now=datetime.now(timezone.utc).isoformat()
        name=str(user.get('name') or user.get('workspace_slug') or uid)+"'s workspace"
        if organization is None:
            payload={'organization_id':org,'kind':'personal','display_name':name,'created_by':uid,
                     'created_at':now,'updated_at':now,'allow_self_join':False,'dataset_type':'REAL'}
            _insert(c,'organizations',_payload_row('organizations',org,payload))
        if not selected:
            personal={'organization_id':org,'organization_name':name,'membership_status':'active',
                      'member_role':'owner','sharing_enabled':False,'joined_at':now,'updated_at':now}
            user.update(organization_id=org,default_organization_id=org,organization_name=name,
                        membership_status='active',member_role='owner',organization_memberships=[personal],updated_at=now)
            c.execute('UPDATE semantier_identity.users SET payload_json=%s,updated_at=%s WHERE user_id=%s',(_json(user),_stamp(now),uid))
        retained=c.execute('SELECT organization_id,account_id FROM semantier_identity.gateway_projection_intents WHERE user_id=%s FOR UPDATE',(uid,)).fetchone()
        if retained and retained['organization_id']!=org:
            raise IdentityConflict('Personal account binding conflict')
        aid=retained['account_id'] if retained else 'account-'+uid
        account=c.execute(sql.SQL("SELECT funding FROM {}.accounts WHERE organization_id=%s AND account_id=%s AND currency='CNY' FOR UPDATE").format(sql.Identifier(billing)),(org,aid)).fetchone()
        if account and account['funding']!='SEMANTIER_CREDITS':
            raise IdentityConflict('Personal account funding conflict')
        if account is None:
            other=c.execute(sql.SQL('SELECT 1 FROM {}.accounts WHERE organization_id=%s LIMIT 1').format(sql.Identifier(billing)),(org,)).fetchone()
            if other:
                # Existing financial identities without a trusted default are
                # operator/selector work, never a reason to invent an account.
                return user
            c.execute(sql.SQL("INSERT INTO {}.accounts(organization_id,account_id,currency,funding,budget_limit) VALUES(%s,%s,'CNY','SEMANTIER_CREDITS',0)").format(sql.Identifier(billing)),(org,aid))
            event={'currency':'CNY','funding':'SEMANTIER_CREDITS','budget_limit':'0','wallet':'0'}
            c.execute(sql.SQL("INSERT INTO {}.financial_policy_events(event_id,organization_id,actor_user_id,operation,target,payload) VALUES(%s,%s,%s,'account.create',%s,%s::jsonb)").format(sql.Identifier(billing)),(uuid4().hex,org,'server-provisioning',aid,_json(event)))
        permissions=['billing:manage','billing:read','model:invoke','usage:read_org','usage:read_self']
        c.execute('INSERT INTO semantier_gateway.grants VALUES(%s,%s,%s::jsonb) ON CONFLICT DO NOTHING',(org,uid,_json(permissions)))
        if os.environ.get('SEMANTIER_COMMERCE_ENVIRONMENT')=='DEVELOPMENT' and os.environ.get('SEMANTIER_ENVIRONMENT')=='development':
            c.execute(sql.SQL("INSERT INTO {}.commerce_account_bindings VALUES(%s,%s,'CNY','local-development','DEVELOPMENT') ON CONFLICT DO NOTHING").format(sql.Identifier(billing)),(org,aid))
        # Retain the v1 allocation proof/history; it is no longer a wake-up for
        # the superseded user-level native credential projector.
        c.execute('''INSERT INTO semantier_identity.gateway_projection_intents(user_id,organization_id,account_id,proxy_user_id,proxy_team_id,proxy_key_alias)
            VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',(uid,org,aid,'semantier-user-'+uid,'semantier-team-'+org,'semantier-key-'+uid))
        seed_personal_sync(c,user_id=uid,organization_id=org,account_id=aid,
                           currency='CNY',funding='SEMANTIER_CREDITS',billing_schema=billing)
        from agents.workspace_storage_provisioning import allocate_signup_storage
        allocate_signup_storage(c,user)
    return user


def import_sqlite(path, *, source_store_id=None):
    """Explicit read-only snapshot import. No source change or implicit merging."""
    path=Path(path).resolve()
    if not path.is_file(): raise ValueError('identity import source missing')
    source_store_id=str(path) if source_store_id is None else source_store_id
    if not isinstance(source_store_id,str) or not source_store_id:
        raise ValueError('explicit identity source_store_id required')
    source=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    source.row_factory=sqlite3.Row
    try:
        source.execute('BEGIN')
        names={r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        unknown=names-set(_TABLES)-{'sqlite_sequence'}
        if unknown: raise ValueError('unsupported identity source tables: '+','.join(sorted(unknown)))
        tables={name:[dict(r) for r in source.execute('SELECT * FROM "'+name+'"')] for name in sorted(names & set(_TABLES))}
        for name, rows in tables.items():
            for row in rows:
                if set(row)!=set(_TABLES[name]): raise ValueError('identity source schema drift: '+name)
        canonical={name:sorted(rows,key=lambda r:_json(r)) for name,rows in tables.items()}
        fingerprint=hashlib.sha256(_json(canonical).encode()).hexdigest()
    finally:
        source.close()
    counts={k:len(v) for k,v in tables.items()}
    def table_fingerprint(name,rows):
        columns=_TABLES[name]
        def scalar(value):
            return value.astimezone(timezone.utc).isoformat() if isinstance(value,datetime) else value
        values=sorted(_json([scalar(row.get(key)) for key in columns]) for row in rows)
        return hashlib.sha256(_json({'columns':columns,'rows':values}).encode()).hexdigest()
    receipt={'source_store_id':source_store_id,'source_fingerprint':fingerprint,'table_counts':counts,
             'source_table_fingerprints':{name:table_fingerprint(name,rows) for name,rows in tables.items()},
             'normalized_source_table_fingerprints':{},'target_table_fingerprints':{},'metadata_normalizations':[]}
    with write_transaction(),connection() as c:
        prior=c.execute('SELECT source_fingerprint,coalesce(source_store_id,source_path) AS source_store_id,receipt_json FROM semantier_identity.imports').fetchall()
        if prior:
            if len(prior)!=1 or prior[0]['source_store_id']!=source_store_id or prior[0]['source_fingerprint']!=fingerprint:
                raise ValueError('identity import source changed or namespace bound to another source')
            return {**(prior[0]['receipt_json'] or {'source_store_id':source_store_id,'source_fingerprint':fingerprint,'table_counts':counts}),'status':'already_imported'}
        for name,rows in tables.items():
            if c.execute(sql.SQL('SELECT 1 FROM {}.{} LIMIT 1').format(sql.Identifier(SCHEMA),sql.Identifier(name))).fetchone():
                raise ValueError('identity import target collision: '+name)
            for row in rows:
                raw_row_hash=table_fingerprint(name,[row])
                for key,value in row.items():
                    if key.endswith('_at') or key=='runtime_session_updated_at':
                        normalized=_stamp(value)
                        if value is not None and value!='' and str(value)!=normalized.isoformat():
                            receipt['metadata_normalizations'].append({'table':name,'column':key,'source_row_sha256':raw_row_hash,'basis':'scalar_timestamp_to_utc_timestamptz'})
                        row[key]=normalized
                _insert(c,name,row)
            actual=c.execute(sql.SQL('SELECT {} FROM {}.{}').format(sql.SQL(',').join(map(sql.Identifier,_TABLES[name])),sql.Identifier(SCHEMA),sql.Identifier(name))).fetchall()
            expected_hash=table_fingerprint(name,rows);actual_hash=table_fingerprint(name,actual)
            if expected_hash!=actual_hash:raise ValueError('identity target canonical fingerprint mismatch: '+name)
            receipt['normalized_source_table_fingerprints'][name]=expected_hash
            receipt['target_table_fingerprints'][name]=actual_hash
        c.execute('INSERT INTO semantier_identity.imports(source_path,source_fingerprint,table_counts,source_store_id,receipt_json) VALUES(%s,%s,%s::jsonb,%s,%s::jsonb)',(str(path),fingerprint,_json(counts),source_store_id,_json(receipt)))
    return {**receipt,'status':'imported'}


def get_weixin_sync_buf(account_id):
    with connection() as c:
        row=c.execute('SELECT get_updates_buf FROM semantier_identity.weixin_sync_state WHERE account_id=%s',(account_id,)).fetchone()
        return row['get_updates_buf'] if row else ''


def save_weixin_sync_buf(account_id,sync_buf):
    with connection() as c:
        c.execute('INSERT INTO semantier_identity.weixin_sync_state VALUES(%s,%s,now()) ON CONFLICT(account_id) DO UPDATE SET get_updates_buf=excluded.get_updates_buf,updated_at=now()',(account_id,sync_buf))


def get_weixin_context_token(account_id,user_id):
    with connection() as c:
        row=c.execute('SELECT context_token FROM semantier_identity.weixin_context_tokens WHERE account_id=%s AND user_id=%s',(account_id,user_id)).fetchone()
        return row['context_token'] if row else None


def save_weixin_context_token(account_id,user_id,context_token):
    with connection() as c:
        c.execute('INSERT INTO semantier_identity.weixin_context_tokens VALUES(%s,%s,%s,now()) ON CONFLICT(account_id,user_id) DO UPDATE SET context_token=excluded.context_token,updated_at=now()',(account_id,user_id,context_token))


def delete_weixin_context_token(account_id,user_id):
    with connection() as c:
        c.execute('DELETE FROM semantier_identity.weixin_context_tokens WHERE account_id=%s AND user_id=%s',(account_id,user_id))
