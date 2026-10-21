"""Native PG snapshot conservation and bounded history portability."""
import hashlib
import json
from pathlib import Path
import sqlite3

from psycopg import sql

from hermes_state_portability import SessionPortabilityMixin, _export_timings

NATIVE_TABLES = ('schema_version','system_prompts','sessions','messages','session_model_usage',
    'runtime_epoch','session_admissions','worker_executions','worker_receipts','state_meta',
    'gateway_routing','gateway_hygiene_state','conversation_generations','gateway_heartbeats',
    'compression_locks','session_turn_leases','async_delegations','semantier_session_branches')


SNAPSHOT_FAMILIES = {'sessions': NATIVE_TABLES, 'api_runs': ('run_idempotency',),
    'api_responses': ('responses','conversations','response_keys','response_admissions','response_output_order'),
    'cron': ('executions','cron_incidents'), 'hosted_rooms_marker': ('hosted_room_legacy_imports',)}


def _json_value(value):
    if isinstance(value,(bytes,bytearray,memoryview)): return {'bytes':bytes(value).hex()}
    raise TypeError(type(value).__name__)


class PostgreSQLPortabilityMixin(SessionPortabilityMixin):
    def import_sqlite_snapshot(self,source,*,family='sessions'):
        """One transaction imports a consistent read-only backup, retaining all native keys.

        Unknown tables are returned to the migration owner for their explicit adapters;
        they never become silently discarded native state. A changed snapshot cannot
        merge into an occupied namespace. Exact reruns compare content fingerprints.
        """
        selected=SNAPSHOT_FAMILIES[family]
        path=Path(source).resolve()
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as source_conn:
            source_conn.row_factory=sqlite3.Row
            source_conn.execute('BEGIN')
            names={r[0] for r in source_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            sequences={r[0]:r[1] for r in source_conn.execute('SELECT name,seq FROM sqlite_sequence')} if 'sqlite_sequence' in names else {}
            sequences={name:value for name,value in sequences.items() if name in selected}
            tables={name:sorted([dict(row) for row in source_conn.execute('SELECT * FROM "'+name+'"')],key=lambda row:json.dumps(row,default=_json_value,sort_keys=True)) for name in selected if name in names}
        canonical=json.dumps({'tables':tables,'sequences':sequences},default=_json_value,sort_keys=True,separators=(',',':'),ensure_ascii=True)
        digest=hashlib.sha256(canonical.encode()).hexdigest()
        receipt={'source_digest':digest,'scope':list(self.scope.identity),'family':family,'sequences':sequences,'tables':{name:len(rows) for name,rows in tables.items()},
                 'unowned_tables':sorted(name for name in names if name not in selected and not name.startswith(('sqlite_','messages_fts')))}
        with self._transaction() as conn:
            self._lock_conversation(conn,'__migration__')
            old=conn.execute(sql.SQL('SELECT receipt_json FROM {} WHERE source_digest=%s').format(self.table('migration_receipts')),(digest,)).fetchone()
            if old: return {**json.loads(old['receipt_json']),'idempotent':True}
            for name in tables:
                if conn.execute(sql.SQL('SELECT 1 FROM {} LIMIT 1').format(self.table(name))).fetchone():
                    raise ValueError('Migration table is occupied by a different snapshot')
            for name,rows in tables.items():
                for row in rows:
                    values=dict(row)
                    if name=='messages' and isinstance(values.get('content'),str) and '\x00' in values['content']:
                        values['content_bytes']=values['content'].encode('utf-8'); values['content']=None
                    self._insert(conn,name,values)
            for name,column in [('messages','id'),('session_admissions','seq')]:
                if name not in selected: continue
                qualified=self.scope.schema+'.'+name
                maximum=conn.execute(sql.SQL('SELECT COALESCE(MAX({}),0) AS n FROM {}').format(sql.Identifier(column),self.table(name))).fetchone()['n']
                maximum=max(maximum,sequences.get(name,0))
                conn.execute('SELECT setval(pg_get_serial_sequence(%s,%s),%s,%s)',(qualified,column,max(1,maximum),maximum>0))
            self._insert(conn,'migration_receipts',{'source_digest':digest,'receipt_json':json.dumps(receipt,sort_keys=True)})
        return {**receipt,'idempotent':False}

    def export_session(self,session_id,include_compacted=False,include_inactive=False):
        session=self.get_session(session_id)
        if not session: return None
        messages=self.get_messages(session_id,include_compacted=include_compacted,include_inactive=include_inactive)
        return {**session,'messages':messages,'timings':_export_timings(messages,session_id)}

    def export_all(self,source=None,include_compacted=False,include_inactive=False):
        return [self.export_session(row['id'],include_compacted,include_inactive) for row in self.search_sessions(source=source,limit=100000)]

    def assert_export_safe(self,session_id,max_messages=None):
        from hermes_state import SessionExportTooLargeError,resolved_max_export_messages
        limit=resolved_max_export_messages() if max_messages is None else max_messages
        count=self.message_count(session_id)
        if limit and count>limit: raise SessionExportTooLargeError(session_id,count,limit)
        return count

    def assert_exports_safe(self,session_ids,max_messages=None):
        for session_id in session_ids: self.assert_export_safe(session_id,max_messages)

    def import_sessions(self,sessions):
        if not isinstance(sessions,list): raise ValueError('sessions must be a list')
        if len(sessions)>self._IMPORT_MAX_SESSIONS: raise ValueError('sessions exceeds import limit')
        normalized,errors=self._validate_import_payload(sessions)
        if errors: return dict(ok=False,imported=0,skipped=0,detached=0,errors=errors)
        from hermes_state_postgresql_fields import SESSION_FIELDS
        imported,skipped,parents=[],[],[]
        with self._transaction() as conn:
            self._lock_conversation(conn,'__history_import__')
            for item in normalized:
                raw=item['session']; sid=raw['id']
                if self._session_on_conn(conn,sid): skipped.append(sid); continue
                values={k:v for k,v in raw.items() if k in SESSION_FIELDS and k not in ('id','source','parent_session_id','message_count','tool_call_count') and not k.startswith(('last_activity_','handoff_'))}
                values.update(ended_at=raw.get('ended_at'),end_reason=raw.get('end_reason'))
                self._create_on_conn(conn,sid,raw.get('source') or 'imported',**values)
                # History import starts no live ownership; active flags remain the exported flags.
                self._append_on_conn(conn,sid,[dict(message) for message in item['messages']])
                if raw.get('parent_session_id'): parents.append((sid,raw['parent_session_id']))
                imported.append(sid)
            detached=0
            for sid,parent in parents:
                if self._session_on_conn(conn,parent): self._update(conn,'sessions',{'parent_session_id':parent},'id',sid)
                else: detached+=1
        return dict(ok=True,imported=len(imported),skipped=len(skipped),detached=detached,imported_ids=imported,skipped_ids=skipped,errors=[])
