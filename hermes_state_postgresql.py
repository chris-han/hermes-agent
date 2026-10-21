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


def provision_scope(dsn, scope: SessionScope, *, app_role: str):
    """Migration-only DDL; the app role never owns schemas or tables."""
    from hermes_state_postgresql_schema import SCHEMA_SQL
    ddl = SCHEMA_SQL
    with psycopg.connect(dsn) as conn:
        conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(scope.schema)))
        conn.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(sql.Identifier(scope.schema)))
        conn.execute(ddl)
        conn.execute('''CREATE TABLE IF NOT EXISTS scope_binding (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1), tenant_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL, source_store_id TEXT NOT NULL)''')
        conn.execute('''INSERT INTO scope_binding VALUES (1,%s,%s,%s) ON CONFLICT DO NOTHING''', scope.identity)
        binding = conn.execute('SELECT tenant_id,workspace_id,source_store_id FROM scope_binding').fetchone()
        if tuple(binding) != scope.identity:
            raise ValueError('Session namespace identity conflict')
        conn.execute('''CREATE TABLE IF NOT EXISTS migration_receipts (
            source_digest TEXT PRIMARY KEY, receipt_json TEXT NOT NULL,
            imported_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS semantier_session_branches (
            child_session_id TEXT PRIMARY KEY REFERENCES sessions(id), source_session_id TEXT NOT NULL REFERENCES sessions(id),
            branch_point_message_id TEXT NOT NULL, branch_point_sequence INTEGER NOT NULL,
            title TEXT NOT NULL, idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL)''')
        conn.execute('CREATE INDEX IF NOT EXISTS messages_session_id_pg ON messages(session_id,id)')
        conn.execute('CREATE INDEX IF NOT EXISTS sessions_parent_pg ON sessions(parent_session_id)')
        conn.execute(sql.SQL('GRANT USAGE ON SCHEMA {} TO {}').format(sql.Identifier(scope.schema),sql.Identifier(app_role)))
        conn.execute(sql.SQL('GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA {} TO {}').format(sql.Identifier(scope.schema),sql.Identifier(app_role)))
        conn.execute(sql.SQL('GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}').format(sql.Identifier(scope.schema),sql.Identifier(app_role)))
        # Runtime cannot change the source namespace identity or migration receipts.
        conn.execute(sql.SQL('REVOKE INSERT,UPDATE,DELETE ON scope_binding,migration_receipts FROM {}').format(sql.Identifier(app_role)))


from hermes_state_postgresql_portability import PostgreSQLPortabilityMixin


from hermes_state_postgresql_search import PostgreSQLSearchMixin
from hermes_state_postgresql_usage import PostgreSQLUsageMixin


class PostgreSQLSessionDB(PostgreSQLPortabilityMixin,PostgreSQLSearchMixin,PostgreSQLUsageMixin):
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
    def _transaction(self, *, read_only=False):
        if self._closed:
            raise RuntimeError('Session store closed')
        if self.read_only and not read_only:
            raise PermissionError('Read-only session store')
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            if read_only:
                conn.execute('SET TRANSACTION READ ONLY')
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
        return conn.execute(sql.SQL('SELECT * FROM {} WHERE id=%s{}').format(self.table('sessions'),
            sql.SQL(' FOR UPDATE' if lock else '')), (session_id,)).fetchone()

    def _create_on_conn(self, conn, session_id, source, **kwargs):
        # Closed field set comes from the native schema owner, not caller SQL.
        from hermes_state_postgresql_fields import SESSION_FIELDS
        unknown = set(kwargs) - SESSION_FIELDS
        if unknown:
            raise TypeError('Unknown session fields: ' + ','.join(sorted(unknown)))
        values = {'id':session_id,'source':source,'created_source':source,'started_at':time.time()}
        values.update({k:self._serialize(v) for k,v in kwargs.items()})
        self._insert(conn,'sessions',values,conflict=True)
        return session_id

    def create_session(self, session_id, source, **kwargs):
        with self._transaction() as conn:
            return self._create_on_conn(conn,session_id,source,**kwargs)

    def ensure_session(self, session_id, source='unknown', model=None, **kwargs):
        return self.create_session(session_id,source,model=model,**kwargs)

    def get_session(self, session_id):
        row = self._one(sql.SQL('SELECT s.*,p.prompt AS _system_prompt_resolved FROM {} s LEFT JOIN {} p ON s.system_prompt_hash=p.hash WHERE s.id=%s').format(self.table('sessions'),self.table('system_prompts')),(session_id,))
        if row:
            row['_system_prompt_resolved'] = row['_system_prompt_resolved'] or row.get('system_prompt')
        return row

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

    def end_session(self, session_id, end_reason):
        self._set_session(session_id,ended_at=time.time(),end_reason=end_reason)

    def reopen_session(self, session_id):
        self._set_session(session_id,ended_at=None,end_reason=None)

    def update_session_meta(self, session_id, **kwargs):
        return self._set_session(session_id,**kwargs)

    def update_system_prompt(self, session_id, system_prompt):
        self._set_session(session_id,system_prompt=system_prompt,system_prompt_hash=None)

    def update_session_tool_names(self, session_id, pin):
        self._set_session(session_id,tool_names=pin)

    def patch_session_model_config(self, session_id, patch):
        with self._transaction() as conn:
            row=self._session_on_conn(conn,session_id,lock=True)
            if not row: return
            cfg=json.loads(row['model_config'] or '{}')
            for key,value in patch.items():
                if value is None: cfg.pop(key,None)
                else: cfg[key]=value
            self._update(conn,'sessions',{'model_config':json.dumps(cfg)},'id',session_id)

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
            if parent_id in seen or current['source'] in ('tool','kanban','oneshot') or any(cfg.get(k) for k in ('_branched_from','_reset_from','_delegate_from')):
                break
            parent=self._session_on_conn(conn,parent_id)
            if not parent or parent['end_reason']!='compression': break
            seen.add(parent_id); current=parent
        return current['id'] if current else session_id

    def _lock_conversation(self,conn,session_id):
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
        current=rows.get(session_id); chain=[session_id]; seen={session_id}
        while current and current.get('parent_session_id') and len(chain)<100:
            cfg=json.loads(current.get('model_config') or '{}'); parent=rows.get(current['parent_session_id'])
            if any(cfg.get(k) for k in ('_branched_from','_reset_from','_delegate_from')) or current['source'] in ('tool','kanban','oneshot') or not parent or parent['end_reason']!='compression' or parent['id'] in seen: break
            seen.add(parent['id']); chain.append(parent['id']); current=parent
        return list(reversed(chain))

    def get_compression_chain(self,session_id):
        chain=[session_id]; seen={session_id}
        rows=self.list_session_records()
        while len(chain)<100:
            parent=next((r for r in rows if r['id']==chain[-1]),None)
            if not parent or parent['end_reason']!='compression': break
            children=[r for r in rows if r['parent_session_id']==parent['id'] and self._resume_lineage_ids(r['id'])[0]!=r['id'] and r['id'] not in seen]
            if not children: break
            children.sort(key=lambda r:(r['started_at'],r['id']),reverse=True)
            chain.append(children[0]['id']); seen.add(children[0]['id'])
        return chain

    def get_compression_tip(self,session_id):
        if not self.get_session(session_id): return None
        return self.get_compression_chain(session_id)[-1]

    def get_compression_lineage(self,session_id):
        if not self.get_session(session_id): return []
        return self.get_compression_chain(self._resume_lineage_ids(session_id)[0])

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
        return len(self.get_messages(session_id,include_compacted=not tip_only,include_ancestors=not tip_only))

    def assert_resume_safe(self,session_id,max_messages=None,*,tip_only=False):
        from hermes_state import SessionResumeTooLargeError,resolved_max_resume_messages
        limit=resolved_max_resume_messages() if max_messages is None else max_messages
        count=self.get_resume_message_count(session_id,tip_only=tip_only)
        if limit and count>limit: raise SessionResumeTooLargeError(count,limit)
        return count

    def _claim_lease(self,table,key_column,key,holder,ttl_seconds,*,session_id):
        if not key or not holder: return False
        now=time.time()
        with self._transaction() as conn:
            self._lock_conversation(conn,session_id)
            query=sql.SQL('INSERT INTO {}({},holder,acquired_at,expires_at) VALUES(%s,%s,%s,%s) ON CONFLICT({}) DO UPDATE SET holder=excluded.holder,acquired_at=excluded.acquired_at,expires_at=excluded.expires_at WHERE {}.holder=excluded.holder OR {}.expires_at<=%s RETURNING holder').format(self.table(table),sql.Identifier(key_column),sql.Identifier(key_column),self.table(table),self.table(table))
            return bool(conn.execute(query,(key,holder,now,now+max(.1,float(ttl_seconds)),now)).fetchone())

    def try_acquire_session_turn_lease(self,session_id,holder,*,ttl_seconds=300,patience_s=None):
        with self._transaction(read_only=True) as conn: key=self._lease_key_on_conn(conn,session_id)
        return self._claim_lease('session_turn_leases','conversation_id',key,holder,ttl_seconds,session_id=session_id)

    def try_acquire_compression_lock(self,session_id,holder,ttl_seconds=300):
        return self._claim_lease('compression_locks','session_id',session_id,holder,ttl_seconds,session_id=session_id)

    def _refresh_lease(self,table,key_column,key,holder,ttl_seconds):
        with self._transaction() as conn:
            return bool(conn.execute(sql.SQL('UPDATE {} SET expires_at=%s WHERE {}=%s AND holder=%s').format(self.table(table),sql.Identifier(key_column)),(time.time()+max(.1,float(ttl_seconds)),key,holder)).rowcount)

    def refresh_session_turn_lease(self,session_id,holder,*,ttl_seconds=300,patience_s=None):
        with self._transaction(read_only=True) as conn: key=self._lease_key_on_conn(conn,session_id)
        return self._refresh_lease('session_turn_leases','conversation_id',key,holder,ttl_seconds)

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

    def publish_compression_child(self,*,parent_session_id,child_session_id,source,messages,model=None,
            model_config=None,system_prompt=None,cwd=None,profile_name=None,compression_lock_holder=None,
            require_compression_lease=True,require_lease_refresh=False,lease_ttl_seconds=300,
            watermark=None,watermark_ceiling=None):
        from hermes_state_errors import CompressionSessionBusyError,CompressionSessionClosedError
        with self._transaction() as conn:
            self._lock_conversation(conn,parent_session_id)
            parent=self._session_on_conn(conn,parent_session_id,lock=True)
            if not parent: raise KeyError('COMPRESSION_PARENT_NOT_FOUND')
            lock=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s FOR UPDATE').format(self.table('compression_locks')),(parent_session_id,)).fetchone()
            if require_lease_refresh and lock and lock['holder']==compression_lock_holder:
                lock['expires_at']=time.time()+lease_ttl_seconds
                self._update(conn,'compression_locks',{'expires_at':lock['expires_at']},'session_id',parent_session_id)
            if require_compression_lease and (not lock or lock['holder']!=compression_lock_holder or lock['expires_at']<=time.time()):
                raise CompressionSessionBusyError('Compression lease lost before publication')
            if parent['ended_at'] is not None: raise CompressionSessionClosedError(parent_session_id)
            kwargs={k:parent[k] for k in ('user_id','session_key','chat_id','chat_type','thread_id','display_name','origin_json','profile_name','tool_names','git_branch','git_repo_root','archived','auto_archived','pinned')}
            kwargs.update(model=model,model_config=model_config,system_prompt=system_prompt,cwd=cwd or parent['cwd'],parent_session_id=parent_session_id)
            if profile_name is not None: kwargs['profile_name']=profile_name
            if self._session_on_conn(conn,child_session_id): raise ValueError('COMPRESSION_CHILD_EXISTS')
            self._create_on_conn(conn,child_session_id,source,**kwargs)
            self._append_on_conn(conn,child_session_id,messages)
            if watermark is not None:
                tail=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s AND active=1 AND id>%s AND (%s::bigint IS NULL OR id<=%s) ORDER BY id').format(self.table('messages')),(parent_session_id,watermark,watermark_ceiling,watermark_ceiling)).fetchall()
                for row in tail:
                    values=dict(row); values.pop('id'); values['session_id']=child_session_id
                    self._insert(conn,'messages',values)
            self._recount_on_conn(conn,child_session_id)
            self._update(conn,'sessions',{'ended_at':time.time(),'end_reason':'compression'},'id',parent_session_id)
            conn.execute(sql.SQL('DELETE FROM {} WHERE session_id=%s AND holder=%s').format(self.table('compression_locks')),(parent_session_id,compression_lock_holder))

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

    def archive_and_compact(self,session_id,compacted_messages,*,watermark=None,compression_lock_holder=None,**kwargs):
        with self._transaction() as conn:
            self._guard_on_conn(conn,session_id,compression_lock_holder=compression_lock_holder,destructive=True)
            query=sql.SQL('UPDATE {} SET active=0,compacted=1 WHERE session_id=%s AND active=1').format(self.table('messages'))
            params=[session_id]
            if watermark is not None: query+=sql.SQL(' AND id<=%s'); params.append(watermark)
            conn.execute(query,params); self._append_on_conn(conn,session_id,compacted_messages)

    def replace_messages(self,session_id,messages,active_only=False,**kwargs):
        with self._transaction() as conn:
            self._guard_on_conn(conn,session_id,destructive=True)
            conn.execute(sql.SQL('UPDATE {} SET active=0,compacted=0 WHERE session_id=%s AND active=1').format(self.table('messages')),(session_id,))
            self._append_on_conn(conn,session_id,messages)

    def delete_session(self,session_id,sessions_dir=None,**kwargs):
        with self._transaction() as conn:
            self._lock_conversation(conn,session_id)
            # Retained children are detached explicitly; no FK disables or history renumbering.
            conn.execute(sql.SQL('UPDATE {} SET parent_session_id=NULL WHERE parent_session_id=%s').format(self.table('sessions')),(session_id,))
            for table,column in [('semantier_session_branches','child_session_id'),('semantier_session_branches','source_session_id'),('messages','session_id'),('session_model_usage','session_id'),('compression_locks','session_id'),('session_turn_leases','conversation_id')]:
                conn.execute(sql.SQL('DELETE FROM {} WHERE {}=%s').format(self.table(table),sql.Identifier(column)),(session_id,))
            return bool(conn.execute(sql.SQL('DELETE FROM {} WHERE id=%s').format(self.table('sessions')),(session_id,)).rowcount)

    def assert_api_continuation_origin_on_conn(self,conn,*,session_id,holder,routing_scope,session_key):
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

    def set_session_title(self,session_id,title):
        return bool(self._set_session(session_id,title=self.sanitize_title(title),title_source='user'))

    def set_auto_title(self,session_id,title,*,source):
        with self._transaction() as conn:
            row=self._session_on_conn(conn,session_id,lock=True)
            ranks={None:-1,'derived':0,'llm':1,'user':2}
            if not row or ranks.get(row['title_source'],-1)>=ranks.get(source,-1): return False
            return bool(self._update(conn,'sessions',{'title':self.sanitize_title(title),'title_source':source},'id',session_id))

    def get_session_by_title(self,title):
        return self._one(sql.SQL('SELECT * FROM {} WHERE title=%s ORDER BY started_at DESC LIMIT 1').format(self.table('sessions')),(title,))

    def get_next_title_in_lineage(self,base_title):
        base_title=self.sanitize_title(base_title) or ''; title=base_title; n=2
        while self.get_session_by_title(title): title=f'{base_title} ({n})'; n+=1
        return title

    def update_session_runtime_lock(self,session_id,*,model=None,provider=None,model_options=None,route_source=None,confirmed=False):
        self.patch_session_model_config(session_id,{'browser_model_lock':dict(provider=provider or '',model=model or '',model_options=model_options or {},route_source=route_source or '',confirmed=bool(confirmed),updated_at=time.time())})
        if model is not None: self._set_session(session_id,model=model)

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
