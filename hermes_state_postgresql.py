"""Native PostgreSQL sessions. SQL is PostgreSQL-owned; no SQLite connection facade.

The migration owner provisions a schema per governed tenant/workspace/source-store
identity. Native IDs and every reference retain their original values in that schema.
Runtime roles can only use an already provisioned namespace.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import psycopg
from psycopg import sql
from psycopg.rows import dict_row


@dataclass(frozen=True)
class SessionScope:
    tenant_id: str
    workspace_id: str
    source_store_id: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in self.identity):
            raise ValueError('Explicit governed session scope required')

    @property
    def identity(self):
        return (self.tenant_id, self.workspace_id, self.source_store_id)

    @property
    def schema(self):
        digest = hashlib.sha256(json.dumps(self.identity, separators=(',', ':')).encode()).hexdigest()
        return 'hermes_' + digest[:40]


def postgres_scope_schema_sql():
    """Pinned native DDL for the migration owner and guarded signup allocator.

    Execute only with a transaction-local search_path pointing at the validated
    SessionScope.schema. Runtime store constructors never execute this asset.
    """
    from hermes_state_postgresql_schema import SCHEMA_SQL
    from hermes_state_postgresql_kanban_schema import SCHEMA_SQL as KANBAN_SCHEMA_SQL
    from hermes_state_postgresql_rooms_schema import SCHEMA_SQL as ROOMS_SCHEMA_SQL
    return SCHEMA_SQL + "\n" + KANBAN_SCHEMA_SQL + "\n" + ROOMS_SCHEMA_SQL + "\n" + r"""
CREATE TABLE IF NOT EXISTS scope_binding (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1), tenant_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL, source_store_id TEXT NOT NULL,
    native_schema_version INTEGER NOT NULL DEFAULT 1 CHECK(native_schema_version=1));
ALTER TABLE scope_binding ADD COLUMN IF NOT EXISTS native_schema_version INTEGER NOT NULL DEFAULT 1 CHECK(native_schema_version=1);
CREATE TABLE IF NOT EXISTS migration_receipts (
    source_digest TEXT PRIMARY KEY, receipt_json TEXT NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS semantier_session_branches (
    child_session_id TEXT PRIMARY KEY,
    source_session_id TEXT NOT NULL,
    branch_point_message_id TEXT NOT NULL, branch_point_sequence INTEGER NOT NULL,
    title TEXT NOT NULL, idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL);
ALTER TABLE semantier_session_branches DROP CONSTRAINT IF EXISTS semantier_session_branches_child_session_id_fkey;
ALTER TABLE semantier_session_branches DROP CONSTRAINT IF EXISTS semantier_session_branches_source_session_id_fkey;
CREATE INDEX IF NOT EXISTS messages_session_id_pg ON messages(session_id,id);
CREATE INDEX IF NOT EXISTS sessions_parent_pg ON sessions(parent_session_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique ON sessions(title) WHERE title IS NOT NULL;
"""


def postgres_owned_tables():
    """Fixed native migration asset ownership, independent from co-located core owners."""
    import re
    return tuple(dict.fromkeys(re.findall(r'CREATE TABLE IF NOT EXISTS ([a-z][a-z0-9_]*)',postgres_scope_schema_sql(),re.IGNORECASE)))


def require_scope_ready(conn, scope=None):
    """Validate native migration state using catalog reads only; never provision here.

    The caller owns the actual psycopg transaction. With no explicit scope, the
    current search_path must already name the governed native namespace.
    """
    import re
    def values(row):
        return tuple(row.values()) if hasattr(row,'values') else tuple(row)
    namespace=scope.schema if scope is not None else values(conn.execute('SELECT current_schema()').fetchone())[0]
    if not isinstance(namespace,str) or not re.fullmatch(r'hermes_[0-9a-f]{40}',namespace):
        raise ValueError('Governed native schema required for readiness')
    tables={values(row)[0] for row in conn.execute("SELECT c.relname FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s AND c.relkind IN ('r','p')",(namespace,))}
    missing=set(postgres_owned_tables())-tables
    if missing:raise ValueError('Native tables missing: '+','.join(sorted(missing)))
    rows=conn.execute(sql.SQL('SELECT tenant_id,workspace_id,source_store_id,native_schema_version FROM {}').format(sql.Identifier(namespace,'scope_binding'))).fetchall()
    if len(rows)!=1:raise ValueError('Native scope binding missing or ambiguous')
    binding=values(rows[0]);actual=SessionScope(*binding[:3])
    if actual.schema!=namespace or (scope is not None and actual!=scope) or binding[3]!=1:
        raise ValueError('Native schema identity/version mismatch')
    expected=set(re.findall(r'CREATE TRIGGER ([a-z][a-z0-9_]*)',postgres_scope_schema_sql(),re.IGNORECASE))
    enabled={values(row)[0] for row in conn.execute("SELECT t.tgname FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s AND NOT t.tgisinternal AND t.tgenabled IN ('O','A')",(namespace,))}
    missing=expected-enabled
    if missing:raise ValueError('Native history guards missing or disabled: '+','.join(sorted(missing)))
    expected_indexes={name:bool(unique) for unique,name in re.findall(r'CREATE (UNIQUE )?INDEX IF NOT EXISTS ([a-z][a-z0-9_]*)',postgres_scope_schema_sql(),re.IGNORECASE)}
    present_indexes={values(row)[0]:values(row)[1:] for row in conn.execute("""SELECT index_class.relname,i.indisvalid,i.indisready,i.indisunique
        FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class index_class ON index_class.oid=i.indexrelid
        JOIN pg_catalog.pg_namespace n ON n.oid=index_class.relnamespace WHERE n.nspname=%s""",(namespace,))}
    missing=[name for name,unique in expected_indexes.items() if name not in present_indexes or not all(present_indexes[name][:2]) or (unique and not present_indexes[name][2])]
    if missing:raise ValueError('Native indexes missing or invalid: '+','.join(sorted(missing)))
    return {'schema':namespace,'native_schema_version':1,'tables':len(postgres_owned_tables()),'enabled_triggers':len(expected),'indexes':len(expected_indexes)}


def provision_scope(dsn, scope: SessionScope, *, app_role: str):
    """Migration-only DDL; the app role never owns schemas or tables."""
    with psycopg.connect(dsn) as conn:
        conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(scope.schema)))
        conn.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(sql.Identifier(scope.schema)))
        conn.execute(postgres_scope_schema_sql())
        conn.execute('INSERT INTO scope_binding VALUES (1,%s,%s,%s) ON CONFLICT DO NOTHING', scope.identity)
        binding = conn.execute('SELECT tenant_id,workspace_id,source_store_id FROM scope_binding').fetchone()
        if tuple(binding) != scope.identity:
            raise ValueError('Session namespace identity conflict')
        conn.execute(sql.SQL('GRANT USAGE ON SCHEMA {} TO {}').format(sql.Identifier(scope.schema),sql.Identifier(app_role)))
        protected={'scope_binding','migration_receipts','schema_version','hosted_room_legacy_imports'}
        owned=postgres_owned_tables()
        for table in owned:
            privileges=sql.SQL('SELECT' if table in protected else 'SELECT,INSERT,UPDATE,DELETE')
            conn.execute(sql.SQL('GRANT {} ON TABLE {} TO {}').format(privileges,sql.Identifier(scope.schema,table),sql.Identifier(app_role)))
            if table in protected:
                conn.execute(sql.SQL('REVOKE INSERT,UPDATE,DELETE ON TABLE {} FROM {}').format(sql.Identifier(scope.schema,table),sql.Identifier(app_role)))
        # Only sequences belonging to the native-owned tables; never re-grant a core owner's metadata.
        sequences=conn.execute("""SELECT DISTINCT sequence_ns.nspname,sequence.relname
            FROM pg_catalog.pg_class sequence
            JOIN pg_catalog.pg_namespace sequence_ns ON sequence_ns.oid=sequence.relnamespace
            JOIN pg_catalog.pg_depend d ON d.objid=sequence.oid AND d.deptype IN ('a','i')
            JOIN pg_catalog.pg_class owner_table ON owner_table.oid=d.refobjid
            JOIN pg_catalog.pg_namespace owner_ns ON owner_ns.oid=owner_table.relnamespace
            WHERE sequence.relkind='S' AND owner_ns.nspname=%s AND owner_table.relname=ANY(%s)""",(scope.schema,list(owned))).fetchall()
        for namespace,name in sequences:
            conn.execute(sql.SQL('GRANT USAGE,SELECT ON SEQUENCE {} TO {}').format(sql.Identifier(namespace,name),sql.Identifier(app_role)))



from hermes_state_postgresql_portability import PostgreSQLPortabilityMixin


from hermes_state_postgresql_search import PostgreSQLSearchMixin
from hermes_state_postgresql_usage import PostgreSQLUsageMixin
from hermes_state_postgresql_runtime import PostgreSQLRuntimeMixin
from hermes_state_rewind import SessionRewindMixin
from hermes_state_postgresql_compaction import PostgreSQLCompactionMixin
from hermes_state_postgresql_retirement import PostgreSQLRetirementMixin
from hermes_state_postgresql_titles import PostgreSQLTitlesMixin
from hermes_state_postgresql_mutations import PostgreSQLMutationsMixin


class PostgreSQLSessionDB(PostgreSQLMutationsMixin,PostgreSQLTitlesMixin,PostgreSQLRuntimeMixin,PostgreSQLPortabilityMixin,PostgreSQLSearchMixin,PostgreSQLUsageMixin,PostgreSQLCompactionMixin,PostgreSQLRetirementMixin,SessionRewindMixin):
    """Native SessionDB API used by the embedded API and Semantier session readers.

    Each operation owns a real PG transaction. External durable owners may call
    *_on_conn methods with their own psycopg connection; table names remain qualified.
    """
    backend = 'postgresql'
    _IMPORT_MAX_SESSIONS, _IMPORT_MAX_MESSAGES_PER_SESSION, _IMPORT_MAX_TOTAL_MESSAGES = 500, 10000, 50000
    _IMPORT_MAX_SESSION_BYTES, _IMPORT_MAX_TOTAL_BYTES = 5 * 1024 * 1024, 25 * 1024 * 1024

    def __init__(self, dsn, scope: SessionScope, *, home: Path, read_only=False):
        self._dsn, self.scope, self.home, self.read_only = dsn, scope, Path(home).resolve(), read_only
        self.storage_identity = ('postgresql', *scope.identity)
        self._lock = threading.RLock()
        self._closed = False
        with self._transaction(read_only=True) as conn:
            row = conn.execute(sql.SQL('SELECT tenant_id,workspace_id,source_store_id FROM {}').format(self.table('scope_binding'))).fetchone()
            if row is None or tuple(row.values()) != scope.identity:
                raise ValueError('Provisioned governed session scope required')

    def table(self, name):
        return sql.Identifier(self.scope.schema, name)

    @contextmanager
    def _transaction(self, *, read_only=False, patience_s=None):
        if self._closed:
            raise RuntimeError('Session store closed')
        if self.read_only and not read_only:
            raise PermissionError('Read-only session store')
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            conn.hermes_scope=self.scope
            conn.hermes_home=self.home
            conn.hermes_read_only=self.read_only or read_only
            if read_only:
                conn.execute('SET TRANSACTION READ ONLY')
            if patience_s is not None:
                conn.execute("SELECT set_config('lock_timeout',%s,true)",(str(max(1,int(float(patience_s)*1000)))+'ms',))
            if not read_only:
                conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(self.scope.schema+'\x1fcanonical-native-writer',))
            conn.execute(sql.SQL('SET LOCAL search_path TO {},pg_catalog').format(sql.Identifier(self.scope.schema)))
            yield conn

    def _all(self, query, params=()):
        with self._transaction(read_only=True) as conn:
            return conn.execute(query, params).fetchall()

    def _one(self, query, params=()):
        with self._transaction(read_only=True) as conn:
            return conn.execute(query, params).fetchone()

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _serialize(self, value):
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict,list)) else value

    def _insert(self, conn, table, values, *, conflict=False, returning=None):
        columns = list(values)
        query = sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(self.table(table),
            sql.SQL(',').join(map(sql.Identifier,columns)), sql.SQL(',').join(sql.Placeholder() for _ in columns))
        if conflict:
            query += sql.SQL(' ON CONFLICT DO NOTHING')
        if returning:
            query += sql.SQL(' RETURNING {}').format(sql.Identifier(returning))
        return conn.execute(query, [values[c] for c in columns])

    def _update(self, conn, table, values, key, value):
        query = sql.SQL('UPDATE {} SET {} WHERE {}=%s').format(self.table(table),
            sql.SQL(',').join(sql.SQL('{}=%s').format(sql.Identifier(c)) for c in values),sql.Identifier(key))
        return conn.execute(query,[*values.values(),value]).rowcount

    def _session_on_conn(self, conn, session_id, *, lock=False):
        row=conn.execute(sql.SQL('SELECT * FROM {} WHERE id=%s{}').format(self.table('sessions'),
            sql.SQL(' FOR UPDATE' if lock else '')), (session_id,)).fetchone()
        return None if row is None else {key:row[key] for key in row.keys()}

    def _create_on_conn(self,conn,session_id,source,**kwargs):
        from hermes_state_postgresql_fields import SESSION_FIELDS
        from hermes_state_mutation_retirement import RETIRED_PREFIX
        from hermes_state_runtime import RuntimeStoreError
        from hermes_state_sessions import _parse_model_config
        unknown=set(kwargs)-SESSION_FIELDS
        if unknown:raise TypeError('Unknown session fields: '+','.join(sorted(unknown)))
        old=self._session_on_conn(conn,session_id,lock=True)
        if old is None and conn.execute(sql.SQL('SELECT 1 FROM {} WHERE key=%s').format(self.table('state_meta')),(RETIRED_PREFIX+session_id,)).fetchone():
            raise RuntimeStoreError('not_found')
        values={'id':session_id,'source':source,'created_source':source,'started_at':time.time()}
        values.update({k:self._serialize(v) for k,v in kwargs.items()})
        if not (values.get('profile_name') or '').strip():values['profile_name']=self._own_profile_name()
        if 'system_prompt' in values:
            values['system_prompt_hash']=self._store_pg_system_prompt(conn,values.pop('system_prompt'))
        if not old:self._insert(conn,'sessions',values)
        else:
            updates={k:v for k,v in values.items() if k not in ('id','source','started_at') and old.get(k) is None}
            if old['source']=='unknown':updates['source']=source or 'unknown'
            old_config=_parse_model_config(old['model_config'])
            if values.get('model_config') is not None and set(old_config)=={'_reset_from'}:
                config=_parse_model_config(values['model_config']);config['_reset_from']=old_config['_reset_from'];updates['model_config']=json.dumps(config)
            if old['system_prompt_hash'] is None and values.get('system_prompt_hash') is not None:updates['system_prompt']=None
            if updates:self._update(conn,'sessions',updates,'id',session_id)
        child=self._session_on_conn(conn,session_id)
        parent=self._session_on_conn(conn,child['parent_session_id']) if child and child['parent_session_id'] else None
        if parent:
            inherited={k:parent[k] for k in ('cwd','git_repo_root','git_branch') if child[k] is None}
            def namespace(key):
                pos=key.find(':',6)
                return key[:pos+1] if pos>=0 else key[:6]
            if child['profile_name'] is None and (parent['session_key'] is None or child['session_key'] is None or namespace(parent['session_key'])==namespace(child['session_key'])):
                inherited['profile_name']=parent['profile_name']
            config=_parse_model_config(child['model_config'])
            if parent['end_reason']=='compression' and all(config.get(k)!=child['parent_session_id'] for k in ('_delegate_from','_branched_from')):
                inherited.update({k:parent[k] for k in ('user_id','session_key','chat_id','chat_type','thread_id','display_name','origin_json','transport_profile') if child[k] is None})
            if inherited:self._update(conn,'sessions',inherited,'id',session_id)
        return session_id

    def create_session(self, session_id, source, **kwargs):
        with self._transaction() as conn:
            return self._create_on_conn(conn,session_id,source,**kwargs)

    def ensure_session(self, session_id, source='unknown', model=None, **kwargs):
        return self.create_session(session_id,source,model=model,**kwargs)

    def get_session(self,session_id):
        from hermes_state import SessionDB
        row=self._one(sql.SQL('SELECT s.*,COALESCE(p.prompt,s.system_prompt) AS _system_prompt_resolved,COALESCE(t.prompt,s.tool_names) AS _tool_names_resolved FROM {} s LEFT JOIN {} p ON s.system_prompt_hash=p.hash LEFT JOIN {} t ON s.tool_names=t.hash WHERE s.id=%s').format(self.table('sessions'),self.table('system_prompts'),self.table('system_prompts')),(session_id,))
        return SessionDB._session_row_dict(row) if row else None

    def list_session_records(self):
        return self._all(sql.SQL('SELECT * FROM {} ORDER BY started_at,id').format(self.table('sessions')))

    def get_branch_records(self):
        return self._all(sql.SQL('SELECT * FROM {} ORDER BY child_session_id').format(self.table('semantier_session_branches')))

    def resolve_session_id(self, value):
        if self.get_session(value):
            return value
        rows = self._all(sql.SQL('SELECT id FROM {} WHERE starts_with(id,%s) LIMIT 2').format(self.table('sessions')),(value,))
        return rows[0]['id'] if len(rows)==1 else None

    def search_sessions(self, source=None, limit=20, offset=0, workspace_key=None):
        clauses, params = [], []
        if source:
            clauses.append(sql.SQL('s.source=ANY(%s)')); params.append([source] if isinstance(source,str) else list(source))
        if workspace_key:
            clauses.append(sql.SQL('(s.git_repo_root=%s OR s.cwd=%s OR starts_with(s.cwd,%s))'))
            params.extend([workspace_key,workspace_key,workspace_key.rstrip('/')+'/'])
        where = sql.SQL(' WHERE ')+sql.SQL(' AND ').join(clauses) if clauses else sql.SQL('')
        return self._all(sql.SQL('SELECT s.*, GREATEST(s.started_at,COALESCE(s.last_activity_at,0),COALESCE((SELECT MAX(m.timestamp) FROM {} m WHERE m.session_id=s.id),0)) AS last_active FROM {} s{} ORDER BY last_active DESC,s.started_at DESC,s.id DESC LIMIT %s OFFSET %s').format(self.table('messages'),self.table('sessions'),where),[*params,limit,offset])

    def session_count(self, source=None, **kwargs):
        if kwargs:
            raise NotImplementedError('PG session_count filters require explicit support')
        query=sql.SQL('SELECT COUNT(*) AS n FROM {}').format(self.table('sessions'))
        if source: query+=sql.SQL(' WHERE source=%s')
        return self._one(query,(source,) if source else ())['n']

    def _set_session(self, session_id, **values):
        with self._transaction() as conn:
            return self._update(conn,'sessions',{k:self._serialize(v) for k,v in values.items()},'id',session_id)

    def end_session(self,session_id,end_reason):
        with self._transaction() as conn:
            conn.execute(sql.SQL('UPDATE {} SET ended_at=%s,end_reason=%s WHERE id=%s AND ended_at IS NULL').format(self.table('sessions')),(time.time(),end_reason,session_id))

    def reopen_session(self, session_id):
        self._set_session(session_id,ended_at=None,end_reason=None)

    def update_session_meta(self, session_id, **kwargs):
        return self._set_session(session_id,**kwargs)

    def update_system_prompt(self,session_id,system_prompt):
        with self._transaction() as conn:
            digest=self._store_pg_system_prompt(conn,system_prompt)
            self._update(conn,'sessions',{'system_prompt_hash':digest,'system_prompt':None},'id',session_id)
            self._delete_pg_unused_prompts(conn)

    def update_session_tool_names(self,session_id,pin):
        with self._transaction() as conn:
            digest=self._store_pg_system_prompt(conn,json.dumps(pin) if pin is not None else None)
            self._update(conn,'sessions',{'tool_names':digest},'id',session_id)
            self._delete_pg_unused_prompts(conn)

    def _delete_pg_unused_prompts(self,conn):
        conn.execute(sql.SQL('DELETE FROM {} p WHERE NOT EXISTS(SELECT 1 FROM {} s WHERE s.system_prompt_hash=p.hash OR s.tool_names=p.hash)').format(self.table('system_prompts'),self.table('sessions')))

    def get_meta(self,key):
        row=self._one(sql.SQL('SELECT value FROM {} WHERE key=%s').format(self.table('state_meta')),(key,))
        return row['value'] if row else None

    def set_meta(self,key,value,*,cursor=None):
        query=sql.SQL('INSERT INTO {}(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=excluded.value').format(self.table('state_meta'))
        if cursor is not None: cursor.execute(query,(key,value)); return
        with self._transaction() as conn: conn.execute(query,(key,value))

    def list_meta_prefix(self,prefix):
        return [(r['key'],r['value']) for r in self._all(sql.SQL('SELECT key,value FROM {} WHERE starts_with(key,%s) ORDER BY key').format(self.table('state_meta')),(prefix,))]

    def _lease_key_on_conn(self,conn,session_id):
        current=self._session_on_conn(conn,session_id)
        seen={session_id}
        while current and current.get('parent_session_id'):
            cfg=json.loads(current.get('model_config') or '{}')
            parent_id=current['parent_session_id']
            if parent_id in seen or self._is_explicit_fork_child_row(current,include_reset=True):
                break
            parent=self._session_on_conn(conn,parent_id)
            if not parent or parent['end_reason']!='compression': break
            seen.add(parent_id); current=parent
        return current['id'] if current else session_id

    def _lock_conversation(self,conn,session_id):
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(self.scope.schema+'\x1fcanonical-native-writer',))
        key=self._lease_key_on_conn(conn,session_id)
        # Transaction advisory locks serialize absent-row lease creation as well as mutations.
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',
                     (self.scope.schema+'\x1f'+key,))
        return key

    def _guard_on_conn(self,conn,session_id,*,turn_lease_holder=None,compression_lock_holder=None,destructive=False):
        from hermes_state_errors import SessionTurnLeaseLostError,CompressionSessionBusyError,SessionActiveWriteGuardError
        key=self._lock_conversation(conn,session_id)
        row=self._session_on_conn(conn,session_id,lock=True)
        if not row: raise KeyError('SESSION_NOT_FOUND')
        if row['ended_at'] is not None: raise SessionActiveWriteGuardError('Closed session')
        lease=conn.execute(sql.SQL('SELECT * FROM {} WHERE conversation_id=%s FOR UPDATE').format(self.table('session_turn_leases')),(key,)).fetchone()
        now=time.time()
        if turn_lease_holder:
            if not lease or lease['holder']!=turn_lease_holder: raise SessionTurnLeaseLostError('Session lease lost')
            if lease['expires_at']<=now:
                self._update(conn,'session_turn_leases',{'expires_at':now+300},'conversation_id',key)
        elif lease and lease['expires_at']>now:
            raise SessionTurnLeaseLostError('Session has an active turn lease')
        if destructive:
            lock=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s FOR UPDATE').format(self.table('compression_locks')),(session_id,)).fetchone()
            if lock and lock['expires_at']>now and lock['holder']!=compression_lock_holder:
                raise CompressionSessionBusyError('Session compression in progress')
        return row

    def _message_values(self,session_id,message):
        from hermes_state_postgresql_fields import MESSAGE_FIELDS
        from hermes_state_messages import SessionMessagesMixin,_coerce_timestamp
        from agent.message_metadata import stamp_message_uid
        from hermes_state_identity import _absorbed_uids_json,_tool_call_uids_json,_tool_call_uid_or_none
        stamp_message_uid(message)
        values={k:self._serialize(v) for k,v in message.items() if k in MESSAGE_FIELDS and k not in ('id','display_identity')}
        values.update(session_id=session_id,timestamp=_coerce_timestamp(message.get('timestamp'),time.time()),
                      message_uid=message['message_uid'])
        content=message.get('content')
        # PostgreSQL text cannot contain NUL, including the native structured-content sentinel.
        # Use a declared reversible BYTEA sidecar for exact native text/structured payloads.
        from hermes_state import SessionDB
        encoded=SessionDB._encode_content(content)
        if isinstance(encoded,str) and '\x00' in encoded:
            values['content']=None; values['content_bytes']=encoded.encode('utf-8')
        else: values['content']=encoded
        values['display_metadata']=SessionMessagesMixin._encode_display_metadata(message.get('display_metadata'))
        values['absorbed_message_uids']=_absorbed_uids_json(message)
        values['tool_call_uids']=_tool_call_uids_json(message)
        values['tool_call_uid']=_tool_call_uid_or_none(message)
        return values

    def _decode_message(self,row):
        from hermes_state_messages import SessionMessagesMixin
        result=dict(row)
        raw=result.pop('content_bytes',None)
        if raw is not None: result['content']=bytes(raw).decode('utf-8')
        from hermes_state import SessionDB
        result['content']=SessionDB._decode_content(result.get('content'))
        for field in ('tool_calls','display_metadata'):
            if result.get(field): result[field]=json.loads(result[field])
        result.pop('display_identity',None); result.pop('display_order',None)
        if not result.pop('_compressed_summary',0): pass
        else: result['_compressed_summary']=True
        return result

    def _append_on_conn(self,conn,session_id,messages):
        from hermes_state_postgresql_transcript import append_on_conn
        return append_on_conn(self,conn,session_id,messages)

    def _recount_on_conn(self,conn,session_id):
        conn.execute(sql.SQL('UPDATE {} SET message_count=(SELECT COUNT(*) FROM {} WHERE session_id=%s AND active=1),tool_call_count=(SELECT COALESCE(SUM(CASE WHEN tool_calls IS NOT NULL THEN jsonb_array_length(tool_calls::jsonb) ELSE 0 END),0) FROM {} WHERE session_id=%s AND active=1) WHERE id=%s').format(self.table('sessions'),self.table('messages'),self.table('messages')),(session_id,session_id,session_id))

    def append_message(self,session_id,role,content=None,tool_name=None,tool_calls=None,tool_call_id=None,**kwargs):
        message=dict(role=role,content=content,tool_name=tool_name,tool_calls=tool_calls,tool_call_id=tool_call_id)
        turn=kwargs.pop('turn_lease_holder',None); compression=kwargs.pop('compression_lock_holder',None)
        kwargs.pop('turn_lease_ttl_seconds',None); message.update(kwargs)
        with self._transaction() as conn:
            self._guard_on_conn(conn,session_id,turn_lease_holder=turn,compression_lock_holder=compression)
            return self._append_on_conn(conn,session_id,[message])[0]

    def append_internal_result(self,session_id,content,*,result_id,result_hash):
        """Project host-committed bytes once under the native writer transaction."""
        if not result_id or not result_hash:
            raise ValueError('Internal result identity required')
        with self._transaction() as conn:
            self._lock_conversation(conn,session_id)
            old=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s AND platform_message_id=%s').format(self.table('messages')),(session_id,result_id)).fetchone()
            if old:
                decoded=self._decode_message(old)
                if decoded['content']!=content or (decoded.get('display_metadata') or {}).get('internal_result_hash')!=result_hash:
                    raise ValueError('INTERNAL_RESULT_IDENTITY_CONFLICT')
                return old['id']
            self._guard_on_conn(conn,session_id)
            message={'role':'assistant','content':content,'platform_message_id':result_id,
                     'display_metadata':{'internal_result_hash':result_hash}}
            return self._append_on_conn(conn,session_id,[message])[0]

    def append_messages_batch(self,session_id,messages,compression_lock_holder=None,turn_lease_holder=None,
                              chunk_rows=None,turn_lease_ttl_seconds=300):
        if not messages: return 0
        # All rows share the transaction even when a caller supplies a chunk hint.
        annotations=[dict(message) for message in messages]
        try:
            with self._transaction() as conn:
                self._guard_on_conn(conn,session_id,turn_lease_holder=turn_lease_holder,compression_lock_holder=compression_lock_holder)
                return len(self._append_on_conn(conn,session_id,messages))
        except BaseException:
            for message,previous in zip(messages,annotations): message.clear(); message.update(previous)
            raise

    def get_messages(self,session_id,include_inactive=False,include_compacted=False,limit=None,offset=0,
                     latest=False,after_id=None,include_ancestors=False):
        if after_id is not None and (latest or offset or include_compacted or include_ancestors):
            raise ValueError('after_id incompatible with requested paging')
        ids=self._resume_lineage_ids(session_id) if include_ancestors else [session_id]
        if include_compacted or len(ids)>1:
            from hermes_state import SessionDB
            raw=self._conversation_rows(ids,include_inactive=include_inactive,include_compacted=include_compacted)
            codec=object.__new__(SessionDB)
            raw=codec._dedupe_display_generations(raw)
            raw=raw[::-1][offset:][:limit][::-1] if latest else raw[offset:][:limit]
            return [self._decode_message(row) for row in raw]
        query=sql.SQL('SELECT * FROM {} WHERE session_id=ANY(%s)').format(self.table('messages'))
        params=[ids]
        if not include_inactive:
            query+=sql.SQL(' AND (active=1 OR compacted=1)' if include_compacted else ' AND active=1')
        if after_id is not None: query+=sql.SQL(' AND id>%s'); params.append(after_id)
        query+=sql.SQL(' ORDER BY id DESC' if latest else ' ORDER BY id')
        query+=sql.SQL(' LIMIT %s OFFSET %s'); params.extend([limit,offset])
        rows=self._all(query,params)
        if latest: rows.reverse()
        return [self._decode_message(row) for row in rows]

    def get_active_message_ids(self,session_id):
        return [row['id'] for row in self._all(sql.SQL('SELECT id FROM {} WHERE session_id=%s AND active=1 ORDER BY id').format(self.table('messages')),(session_id,))]

    def get_active_message_watermark(self,session_id):
        ids=self.get_active_message_ids(session_id)
        return ids[-1] if ids else 0

    def message_count(self,session_id=None):
        query=sql.SQL('SELECT COUNT(*) AS n FROM {}').format(self.table('messages'))
        if session_id: query+=sql.SQL(' WHERE session_id=%s')
        return self._one(query,(session_id,) if session_id else ())['n']

    def has_platform_message_id(self,session_id,platform_message_id):
        return bool(self._one(sql.SQL('SELECT id FROM {} WHERE session_id=%s AND platform_message_id=%s LIMIT 1').format(self.table('messages')),(session_id,platform_message_id)))

    def latest_conversation_role(self,session_id):
        rows=self.get_messages(session_id,limit=1,latest=True)
        return rows[0]['role'] if rows else None

    def latest_message_row_id(self,session_id,*,role='user',offset=0,**kwargs):
        row=self._one(sql.SQL('SELECT id FROM {} WHERE session_id=%s AND role=%s AND active=1 ORDER BY id DESC LIMIT 1 OFFSET %s').format(self.table('messages')),(session_id,role,offset))
        return row['id'] if row else None

    def get_message_role(self,session_id,row_id):
        row=self._one(sql.SQL('SELECT role FROM {} WHERE session_id=%s AND id=%s').format(self.table('messages')),(session_id,row_id))
        return row['role'] if row else None

    def _resume_lineage_ids(self,session_id):
        rows={row['id']:row for row in self.list_session_records()}
        current=rows.get(session_id);chain=[session_id];seen={session_id}
        while current and current.get('parent_session_id'):
            parent=rows.get(current['parent_session_id'])
            if self._is_explicit_fork_child_row(current,include_reset=True) or not parent or parent['end_reason']!='compression' or parent['id'] in seen:break
            seen.add(parent['id']);chain.append(parent['id']);current=parent
        return list(reversed(chain))

    def get_compression_chain(self,session_id):
        from hermes_state_sessions import _parse_model_config
        rows=self.list_session_records();by_id={r['id']:r for r in rows}
        recency={r['session_id']:r['last_message'] for r in self._all(sql.SQL('SELECT session_id,MAX(timestamp) AS last_message FROM {} GROUP BY session_id').format(self.table('messages')))}
        chain=[session_id] if session_id else [];seen=set(chain)
        for _ in range(1000):
            parent=by_id.get(chain[-1]) if chain else None
            if not parent or parent['end_reason']!='compression':break
            children=[r for r in rows if r['parent_session_id']==parent['id'] and r['source']!='tool' and all(_parse_model_config(r['model_config']).get(k) is None for k in ('_branched_from','_delegate_from','_reset_from')) and r['id'] not in seen]
            if not children:break
            children.sort(key=lambda r:r['id'],reverse=True)
            children.sort(key=lambda r:(0 if r['end_reason']=='compression' else 1 if r['ended_at'] is None else 2,-max(r['started_at'] or 0,r['last_activity_at'] or 0,recency.get(r['id']) or 0),-(r['started_at'] or 0)))
            best=children[0]
            chain.append(best['id']);seen.add(best['id'])
        return chain

    def get_compression_tip(self,session_id):
        chain=self.get_compression_chain(session_id)
        return chain[-1] if chain else session_id

    def get_compression_lineage(self,session_id):
        rows={row['id']:row for row in self.list_session_records()};session=rows.get(session_id)
        if not session:return []
        if self._is_explicit_fork_child_row(session):return [session_id]
        root=rows[self._resume_lineage_ids(session_id)[0]];lineage=[root['id']];seen=set(lineage);current=root
        while current['end_reason']=='compression':
            children=sorted((r for r in rows.values() if r['parent_session_id']==current['id']),key=lambda r:r['started_at'])
            child=next((r for r in children if not self._is_explicit_fork_child_row(r,include_reset=True)),None)
            if not child or child['id'] in seen:break
            lineage.append(child['id']);seen.add(child['id']);current=child
        return lineage if session_id in lineage else [session_id]

    def resolve_resume_session_id(self,session_id):
        return self.get_compression_tip(session_id) or session_id

    def _conversation_rows(self,session_ids,*,include_inactive=False,include_compacted=False):
        from hermes_state_postgresql_transcript import native_row
        query=sql.SQL('SELECT * FROM {} WHERE session_id=ANY(%s)').format(self.table('messages'))
        if not include_inactive:
            query+=sql.SQL(' AND (active=1 OR compacted=1)' if include_compacted else ' AND active=1')
        query+=sql.SQL(' ORDER BY id')
        return [native_row(row) for row in self._all(query,(session_ids,))]

    def get_messages_as_conversation(self,session_id,include_ancestors=False,include_inactive=False,
                                    repair_alternation=False,include_row_ids=False,include_compacted=False):
        from hermes_state import SessionDB
        codec=object.__new__(SessionDB)
        ids=self._resume_lineage_ids(session_id) if include_ancestors else [session_id]
        rows=self._conversation_rows(ids,include_inactive=include_inactive,include_compacted=include_compacted)
        if include_compacted: rows=codec._dedupe_display_generations(rows)
        return codec._rows_to_conversation(rows,session_id=session_id,include_ancestors=include_ancestors,
            repair_alternation=repair_alternation,include_row_ids=include_row_ids,include_summary_markers=repair_alternation)

    def get_resume_conversations(self,session_id):
        from hermes_state import SessionDB
        codec=object.__new__(SessionDB)
        rows=self._conversation_rows(self._resume_lineage_ids(session_id),include_compacted=True)
        model=codec._rows_to_conversation([r for r in rows if r['session_id']==session_id and r['active'] and r['role']!='session_meta'],session_id=session_id,include_ancestors=False,repair_alternation=True,include_row_ids=True,include_summary_markers=True)
        display=codec._rows_to_conversation(codec._dedupe_display_generations(rows),session_id=session_id,include_ancestors=True,repair_alternation=False,include_row_ids=True)
        return model,display

    def get_resume_message_count(self,session_id,*,tip_only=False):
        ids=[session_id] if tip_only else self._resume_lineage_ids(session_id)
        clause=sql.SQL(' AND active=1' if tip_only else ' AND (active=1 OR compacted=1)')
        return self._one(sql.SQL('SELECT COUNT(*) AS n FROM {} WHERE session_id=ANY(%s)').format(self.table('messages'))+clause,(ids,))['n']

    def assert_resume_safe(self,session_id,max_messages=None,*,tip_only=False):
        from hermes_state import SessionResumeTooLargeError,resolved_max_resume_messages
        limit=resolved_max_resume_messages() if max_messages is None else max_messages
        count=self.get_resume_message_count(session_id,tip_only=tip_only)
        if limit and count>limit: raise SessionResumeTooLargeError(count,limit)
        return count

    def _claim_lease(self,table,key_column,key,holder,ttl_seconds,*,session_id,patience_s=None):
        if not key or not holder:return False
        from hermes_state_postgresql_leases import claim_lease_row
        now=time.time()
        with self._transaction(patience_s=patience_s) as conn:
            self._lock_conversation(conn,session_id)
            return claim_lease_row(conn,table,key_column,key,holder,now,now+max(.1,float(ttl_seconds)))[0]

    def try_acquire_session_turn_lease(self,session_id,holder,*,ttl_seconds=300,patience_s=None):
        with self._transaction(read_only=True) as conn: key=self._lease_key_on_conn(conn,session_id)
        return self._claim_lease('session_turn_leases','conversation_id',key,holder,ttl_seconds,session_id=session_id,patience_s=patience_s)

    def try_acquire_compression_lock(self,session_id,holder,ttl_seconds=300):
        return self._claim_lease('compression_locks','session_id',session_id,holder,ttl_seconds,session_id=session_id)

    def _refresh_lease(self,table,key_column,key,holder,ttl_seconds,*,patience_s=None):
        with self._transaction(patience_s=patience_s) as conn:
            return bool(conn.execute(sql.SQL('UPDATE {} SET expires_at=%s WHERE {}=%s AND holder=%s').format(self.table(table),sql.Identifier(key_column)),(time.time()+max(.1,float(ttl_seconds)),key,holder)).rowcount)

    def refresh_session_turn_lease(self,session_id,holder,*,ttl_seconds=300,patience_s=None):
        with self._transaction(read_only=True) as conn: key=self._lease_key_on_conn(conn,session_id)
        return self._refresh_lease('session_turn_leases','conversation_id',key,holder,ttl_seconds,patience_s=patience_s)

    def refresh_compression_lock(self,session_id,holder,ttl_seconds=300):
        return self._refresh_lease('compression_locks','session_id',session_id,holder,ttl_seconds)

    def release_session_turn_lease(self,session_id,holder):
        with self._transaction() as conn:
            key=self._lease_key_on_conn(conn,session_id)
            conn.execute(sql.SQL('DELETE FROM {} WHERE conversation_id=%s AND holder=%s').format(self.table('session_turn_leases')),(key,holder))

    def release_compression_lock(self,session_id,holder):
        with self._transaction() as conn:
            conn.execute(sql.SQL('DELETE FROM {} WHERE session_id=%s AND holder=%s').format(self.table('compression_locks')),(session_id,holder))

    def session_turn_lease_expires_at(self,session_id,holder):
        with self._transaction(read_only=True) as conn:
            key=self._lease_key_on_conn(conn,session_id)
            row=conn.execute(sql.SQL('SELECT expires_at FROM {} WHERE conversation_id=%s AND holder=%s').format(self.table('session_turn_leases')),(key,holder)).fetchone()
            return row['expires_at'] if row else None

    def get_compression_lock_holder(self,session_id):
        row=self._one(sql.SQL('SELECT holder FROM {} WHERE session_id=%s AND expires_at>%s').format(self.table('compression_locks')),(session_id,time.time()))
        return row['holder'] if row else None

    def branch_session(self,source_session_id,child_session_id,*,title,branch_point_message_id,
                       branch_point_sequence,idempotency_key,request_hash):
        expected=dict(child_session_id=child_session_id,source_session_id=source_session_id,
            branch_point_message_id=str(branch_point_message_id),branch_point_sequence=branch_point_sequence,
            title=title,idempotency_key=idempotency_key,request_hash=request_hash)
        with self._transaction() as conn:
            source=self._session_on_conn(conn,source_session_id,lock=True)
            if not source: raise KeyError('SOURCE_SESSION_NOT_FOUND')
            existing=conn.execute(sql.SQL('SELECT * FROM {} WHERE child_session_id=%s').format(self.table('semantier_session_branches')),(child_session_id,)).fetchone()
            idempotent=bool(existing)
            if existing:
                if existing!=expected: raise ValueError('IDEMPOTENCY_CONFLICT')
            else:
                if self._session_on_conn(conn,child_session_id): raise ValueError('IDEMPOTENCY_CONFLICT')
                rows=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s AND active=1 ORDER BY id').format(self.table('messages')),(source_session_id,)).fetchall()
                if branch_point_sequence<1 or branch_point_sequence>len(rows): raise KeyError('BRANCH_POINT_NOT_FOUND')
                boundary=rows[branch_point_sequence-1]
                if str(boundary['id'])!=str(branch_point_message_id) or boundary['role']!='assistant': raise KeyError('BRANCH_POINT_NOT_FOUND')
                cfg=json.loads(source.get('model_config') or '{}'); cfg['_branched_from']=source_session_id
                self._create_on_conn(conn,child_session_id,source['source'],parent_session_id=source_session_id,
                    model=source['model'],model_config=cfg,system_prompt=source['system_prompt'],user_id=source['user_id'],display_name=title)
                for row in rows[:branch_point_sequence]:
                    values=dict(row); values.pop('id'); values['session_id']=child_session_id
                    # A fork is a new occurrence, retaining all original payload bytes.
                    values['message_uid']=str(uuid.uuid4()); self._insert(conn,'messages',values)
                self._recount_on_conn(conn,child_session_id); self._insert(conn,'semantier_session_branches',expected)
            count=conn.execute(sql.SQL('SELECT COUNT(*) AS n FROM {} WHERE session_id=%s AND active=1').format(self.table('messages')),(child_session_id,)).fetchone()['n']
        return dict(id=child_session_id,session_id=child_session_id,parent_session_id=source_session_id,
            lineage_root_id=source.get('parent_session_id') or source_session_id,
            branch_point_message_id=str(branch_point_message_id),branch_point_sequence=branch_point_sequence,
            message_count=count,idempotent=idempotent)

    def rewind_to_message(self,session_id,target_message_id,*,preserve_compaction_handoff=False,
                          expected_active_ids=None,expected_target_content=None):
        with self._transaction() as conn:
            self._guard_on_conn(conn,session_id,destructive=True)
            rows=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s AND active=1 ORDER BY id').format(self.table('messages')),(session_id,)).fetchall()
            ids=[r['id'] for r in rows]; target=next((r for r in rows if r['id']==target_message_id),None)
            if not target or target['role']!='user': raise ValueError('Rewind target must be an active user message')
            if expected_active_ids is not None and list(expected_active_ids)!=ids: raise ValueError('REWIND_CONFLICT')
            replacement=None
            if preserve_compaction_handoff or expected_target_content is not None:
                from hermes_state import SessionDB
                from hermes_state_postgresql_transcript import native_row
                replacement=object.__new__(SessionDB)._split_rewind_target(native_row(target),expected_target_content,preserve_compaction_handoff)
            removed=[i for i in ids if i>=target_message_id]
            conn.execute(sql.SQL('UPDATE {} SET active=0,compacted=0 WHERE session_id=%s AND id=ANY(%s)').format(self.table('messages')),(session_id,removed))
            conn.execute(sql.SQL('UPDATE {} SET rewind_count=rewind_count+1 WHERE id=%s').format(self.table('sessions')),(session_id,))
            replacement_id=self._append_on_conn(conn,session_id,[replacement])[0] if replacement is not None else None
            self._recount_on_conn(conn,session_id)
        return dict(**({'replacement_message_id':replacement_id,'replacement_message_uid':replacement.get('message_uid')} if preserve_compaction_handoff else {}),rewound_count=len(removed),target_message=self._decode_message(target),new_head_id=max([i for i in ids if i<target_message_id],default=None))

    def assert_api_continuation_origin_on_conn(self,conn,*,session_id,holder,routing_scope,session_key):
        from psycopg.pq import TransactionStatus
        if conn.info.transaction_status!=TransactionStatus.INTRANS:
            raise ValueError('Active native/domain transaction required')
        current=conn.execute('SELECT current_schema()').fetchone()
        namespace=current['current_schema'] if hasattr(current,'keys') else current[0]
        if namespace!=self.scope.schema:
            raise ValueError('Native/domain transaction scope mismatch')
        self._lock_conversation(conn,session_id)
        key=self._lease_key_on_conn(conn,session_id)
        lease=conn.execute(sql.SQL('SELECT holder,expires_at FROM {} WHERE conversation_id=%s FOR UPDATE').format(self.table('session_turn_leases')),(key,)).fetchone()
        if not lease or lease['holder']!=holder or lease['expires_at']<=time.time(): raise ValueError('API_STALE_OWNER')
        route=conn.execute(sql.SQL('SELECT entry_json FROM {} WHERE scope=%s AND session_key=%s').format(self.table('gateway_routing')),(routing_scope,session_key)).fetchone()
        row=self._session_on_conn(conn,session_id,lock=True)
        if not route or json.loads(route['entry_json'])['session_id']!=session_id or not row or row['ended_at'] is not None:
            raise ValueError('API_STALE_ORIGIN')

    def save_gateway_routing_entry(self,session_key,entry_json,*,scope=''):
        with self._transaction() as conn:
            conn.execute(sql.SQL('INSERT INTO {}(scope,session_key,entry_json,updated_at) VALUES(%s,%s,%s,%s) ON CONFLICT(scope,session_key) DO UPDATE SET entry_json=excluded.entry_json,updated_at=excluded.updated_at').format(self.table('gateway_routing')),(scope,session_key,entry_json,time.time()))

    def load_gateway_routing_entries(self,*,scope=''):
        return {row['session_key']:row['entry_json'] for row in self._all(sql.SQL('SELECT session_key,entry_json FROM {} WHERE scope=%s').format(self.table('gateway_routing')),(scope,))}

    def replace_gateway_routing_entries(self,entries,*,scope=''):
        with self._transaction() as conn:
            conn.execute(sql.SQL('DELETE FROM {} WHERE scope=%s').format(self.table('gateway_routing')),(scope,))
            for key,value in entries.items(): self._insert(conn,'gateway_routing',dict(scope=scope,session_key=key,entry_json=value,updated_at=time.time()))

    def record_gateway_session_peer(self,session_id,*,source,user_id=None,session_key=None,chat_id=None,
            chat_type=None,thread_id=None,display_name=None,origin_json=None,include_compression_ancestors=False,transport_profile=None):
        values=dict(source=source,user_id=user_id,session_key=session_key,chat_id=chat_id,chat_type=chat_type,thread_id=thread_id)
        values.update({k:v for k,v in dict(display_name=display_name,origin_json=origin_json,transport_profile=transport_profile).items() if v is not None})
        with self._transaction() as conn:
            self._create_on_conn(conn,session_id,source)
            ids=self._resume_lineage_ids(session_id) if include_compression_ancestors else [session_id]
            for sid in ids: self._update(conn,'sessions',values,'id',sid)

    def get_session_title(self,session_id):
        row=self.get_session(session_id)
        return row['title'] if row else None

    def get_session_title_source(self,session_id):
        row=self.get_session(session_id)
        return row['title_source'] if row else None

    def sanitize_title(self,title):
        from hermes_state import SessionDB
        return SessionDB.sanitize_title(title)

    def get_session_by_title(self,title):
        row=self._one(sql.SQL('SELECT id FROM {} WHERE title=%s').format(self.table('sessions')),(title,))
        return self.get_session(row['id']) if row else None

    def update_session_runtime_lock(self,session_id,*,model=None,provider=None,model_options=None,route_source=None,confirmed=False):
        lock=dict(provider=provider or '',model=model or '',model_options=model_options or {},route_source=route_source or '',confirmed=bool(confirmed),updated_at=time.time())
        self._write_model_config_patch(session_id,{'browser_model_lock':lock},
            'UPDATE sessions SET model_config=%s,model=COALESCE(%s,model) WHERE id=%s',
            lambda merged:(merged,model,session_id))

    def unarchive_recoverable_session(self,session_id):
        return bool(self._set_session(session_id,archived=0,auto_archived=0))

    def set_session_archived(self,session_id,archived):
        return bool(self._set_session(session_id,archived=int(archived),auto_archived=0))

    def set_session_hidden(self,session_id,hidden):
        return bool(self._set_session(session_id,hidden=int(hidden)))

    def set_session_pinned(self,session_id,pinned):
        return bool(self._set_session(session_id,pinned=int(pinned)))

    def touch_session_activity(self,session_id,**kwargs):
        self._set_session(session_id,last_activity_at=time.time(),**kwargs)

    def create_client_session(self,session_id,*,source,model=None,model_config=None,system_prompt=None,title=None):
        with self._transaction() as conn:
            self._lock_conversation(conn,session_id)
            if self._session_on_conn(conn,session_id):return None,'exists'
            clean=self.sanitize_title(str(title)) if title is not None else None
            if clean:
                self._lock_conversation(conn,'title:'+clean)
                try:self._resolve_title_conflict(conn,session_id,clean)
                except ValueError as exc:return None,f'title:{exc}'
            self._create_on_conn(conn,session_id,source,model=model,model_config=model_config,system_prompt=system_prompt,title=clean)
            return self._session_on_conn(conn,session_id),None

    def list_sessions_rich(self,source=None,sources=None,exclude_sources=None,cwd_prefix=None,limit=20,offset=0,
            include_children=False,min_message_count=0,project_compression_tips=True,order_by_last_active=False,
            include_archived=False,archived_only=False,id_query=None,search_query=None,compact_rows=False,
            include_pinned=False,session_key=None,include_hidden=False,include_subagents=False):
        where,params=[],[]
        if source or sources:where.append(sql.SQL('s.source=ANY(%s)'));params.append([source] if source else list(sources))
        if exclude_sources:where.append(sql.SQL('s.source<>ALL(%s)'));params.append(list(exclude_sources))
        if session_key:where.append(sql.SQL('s.session_key=%s'));params.append(session_key)
        if cwd_prefix:where.append(sql.SQL('(s.cwd=%s OR starts_with(s.cwd,%s) OR starts_with(s.cwd,%s))'));params.extend([cwd_prefix,cwd_prefix.rstrip('/\\')+'/',cwd_prefix.rstrip('/\\')+'\\'])
        if min_message_count:where.append(sql.SQL('s.message_count>=%s'));params.append(min_message_count)
        if archived_only:where.append(sql.SQL('s.archived=1'))
        elif not include_archived:where.append(sql.SQL('(s.archived=0 OR s.pinned=1)' if include_pinned else 's.archived=0'))
        if not include_hidden and not archived_only:where.append(sql.SQL('s.hidden=0'))
        if id_query:where.append(sql.SQL('strpos(lower(s.id),lower(%s))>0'));params.append(id_query)
        if search_query:where.append(sql.SQL("(strpos(lower(COALESCE(s.title,'')),lower(%s))>0 OR strpos(lower(COALESCE(s.display_name,'')),lower(%s))>0 OR strpos(lower(s.id),lower(%s))>0)"));params.extend([search_query]*3)
        if not include_children:
            where.append(sql.SQL("(s.parent_session_id IS NULL OR COALESCE(s.model_config::jsonb->>'_branched_from','')<>'' OR COALESCE(s.model_config::jsonb->>'_reset_from','')<>'' OR EXISTS(SELECT 1 FROM {} p WHERE p.id=s.parent_session_id AND p.end_reason='branched'))").format(self.table('sessions')))
            if not include_subagents:where.append(sql.SQL("COALESCE(s.model_config::jsonb->>'_delegate_from','')=''"))
        clauses=sql.SQL(' WHERE ')+sql.SQL(' AND ').join(where) if where else sql.SQL('')
        active=sql.SQL('GREATEST(s.started_at,COALESCE(s.last_activity_at,0),COALESCE((SELECT MAX(m.timestamp) FROM {} m WHERE m.session_id=s.id),0))').format(self.table('messages'))
        query=sql.SQL("SELECT s.*,{} AS last_active,(SELECT COALESCE(m.content,'') FROM {} m WHERE m.session_id=s.id AND m.role='user' ORDER BY m.id DESC LIMIT 1) AS preview FROM {} s{}").format(active,self.table('messages'),self.table('sessions'),clauses)
        query+=sql.SQL(' ORDER BY last_active DESC,s.started_at DESC,s.id DESC' if order_by_last_active else ' ORDER BY s.started_at DESC,s.id DESC')
        rows=self._all(query,[*params])
        # A compression continuation is represented by its tip, keeping one visible conversation.
        if project_compression_tips:
            output=[];seen=set()
            for row in rows:
                tip=self.get_compression_tip(row['id']) or row['id']
                if tip in seen:continue
                seen.add(tip);selected=next((r for r in rows if r['id']==tip),None) or self.get_session(tip)
                if selected:output.append({**selected,'last_active':row['last_active'],'preview':row['preview']})
            rows=output
        window=rows[offset:offset+limit]
        if include_pinned:
            ids={row['id'] for row in window};window.extend(row for row in rows if row['pinned'] and row['id'] not in ids)
        if compact_rows:
            for row in window:
                for key in ('model_config','system_prompt','tool_names'):row.pop(key,None)
        return window
