"""Native PG snapshot conservation and bounded history portability."""
import hashlib
import json
from pathlib import Path
import sqlite3

from psycopg import sql

from hermes_state_portability import SessionPortabilityMixin, _export_timings
from hermes_state_postgresql_rooms_schema import TABLES as ROOM_TABLES

NATIVE_TABLES = ('schema_version','system_prompts','sessions','messages','session_model_usage',
    'runtime_epoch','session_admissions','worker_executions','worker_receipts','state_meta',
    'gateway_routing','gateway_hygiene_state','conversation_generations','gateway_heartbeats',
    'compression_locks','session_turn_leases','async_delegations','semantier_session_branches','delivery_obligations')


SNAPSHOT_FAMILIES = {'hosted_rooms': ROOM_TABLES, 'sessions': NATIVE_TABLES, 'api_runs': ('run_idempotency',),
    'api_responses': ('responses','conversations','response_keys','response_admissions','response_output_order'),
    'cron': ('executions','cron_incidents'), 'cron_notes': ('cron_notepad',), 'cron_delivery': ('deliveries','delivery_tombstones'), 'kanban': ('tasks','task_links','task_comments','task_events','task_runs','task_attachments','kanban_notify_subs'), 'hosted_rooms_marker': ('hosted_room_legacy_imports',)}


def _json_value(value):
    if isinstance(value,(bytes,bytearray,memoryview)): return {'bytes':bytes(value).hex()}
    raise TypeError(type(value).__name__)


class PostgreSQLPortabilityMixin(SessionPortabilityMixin):
    _SQLITE_INT_MIN=-(2**63)
    _SQLITE_INT_MAX=2**63-1

    def import_sqlite_snapshot(self,source,*,family='sessions'):
        """One transaction imports a consistent read-only backup, retaining all native keys.

        Unknown tables are returned to the migration owner for their explicit adapters;
        they never become silently discarded native state. A changed snapshot cannot
        merge into an occupied namespace. Exact reruns compare content fingerprints.
        """
        selected=SNAPSHOT_FAMILIES[family]
        with self._transaction(read_only=True) as guard_conn:
            permitted=guard_conn.execute("SELECT has_table_privilege(current_user,%s,'INSERT') AS permitted",(self.scope.schema+'.migration_receipts',)).fetchone()['permitted']
            if not permitted:raise PermissionError('Snapshot import requires the explicit migration owner')
        path=Path(source).resolve()
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as source_conn:
            source_conn.row_factory=sqlite3.Row
            source_conn.execute('BEGIN')
            names={r[0] for r in source_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            sequences={r[0]:r[1] for r in source_conn.execute('SELECT name,seq FROM sqlite_sequence')} if 'sqlite_sequence' in names else {}
            sequences={name:value for name,value in sequences.items() if name in selected}
            tables={name:sorted([dict(row) for row in source_conn.execute(('SELECT rowid AS native_row_order,* FROM ' if name=='session_model_usage' else 'SELECT * FROM ' )+'"'+name+'"')],key=lambda row:json.dumps(row,default=_json_value,sort_keys=True)) for name in selected if name in names}
        canonical=json.dumps({'contract':'native-postgresql-import-v1','family':family,'tables':tables,'sequences':sequences},default=_json_value,sort_keys=True,separators=(',',':'),ensure_ascii=True)
        digest=hashlib.sha256(canonical.encode()).hexdigest()
        receipt={'import_contract':'native-postgresql-import-v1','source_digest':digest,'scope':list(self.scope.identity),'family':family,'sequences':sequences,'tables':{name:len(rows) for name,rows in tables.items()},
                 'unowned_tables':sorted(name for name in names if name not in selected and not name.startswith(('sqlite_','messages_fts')))}
        with self._transaction() as conn:
            self._lock_conversation(conn,'__migration__')
            old=conn.execute(sql.SQL('SELECT receipt_json FROM {} WHERE source_digest=%s').format(self.table('migration_receipts')),(digest,)).fetchone()
            if old: return {**json.loads(old['receipt_json']),'idempotent':True}
            for name in tables:
                if conn.execute(sql.SQL('SELECT 1 FROM {} LIMIT 1').format(self.table(name))).fetchone():
                    raise ValueError('Migration table is occupied by a different snapshot')
            if family=='hosted_rooms':
                for name in tables:
                    conn.execute(sql.SQL('ALTER TABLE {} DISABLE TRIGGER USER').format(self.table(name)))
            for name,rows in tables.items():
                for row in rows:
                    values=dict(row)
                    if name=='cron_notepad':
                        for column in ('job_id','key','value'):
                            if isinstance(values.get(column),str):values[column]=values[column].encode('utf-8')
                    if name=='deliveries' and isinstance(values.get('content'),str):values['content']=values['content'].encode('utf-8')
                    if name=='messages' and isinstance(values.get('content'),str) and '\x00' in values['content']:
                        values['content_bytes']=values['content'].encode('utf-8'); values['content']=None
                    self._insert(conn,name,values)
            if family=='hosted_rooms':
                for name in tables:
                    conn.execute(sql.SQL('ALTER TABLE {} ENABLE TRIGGER USER').format(self.table(name)))
            for name,column in [('hosted_room_work_records_invalid','evidence_id'),('messages','id'),('session_model_usage','native_row_order'),('session_admissions','seq'),('task_comments','id'),('task_events','id'),('task_runs','id'),('task_attachments','id')]:
                if name not in selected: continue
                qualified=self.scope.schema+'.'+name
                maximum=conn.execute(sql.SQL('SELECT COALESCE(MAX({}),0) AS n FROM {}').format(sql.Identifier(column),self.table(name))).fetchone()['n']
                maximum=max(maximum,sequences.get(name,0))
                conn.execute('SELECT setval(pg_get_serial_sequence(%s,%s),%s,%s)',(qualified,column,max(1,maximum),maximum>0))
            fingerprints={}
            for name,source_rows in tables.items():
                actual=conn.execute(sql.SQL('SELECT * FROM {}').format(self.table(name))).fetchall()
                if len(actual)!=len(source_rows):raise ValueError('Imported native row count mismatch')
                if not source_rows:continue
                columns=tuple(source_rows[0])
                projected=[]
                for row in actual:
                    values={column:row[column] for column in columns}
                    if name in ('cron_notepad','deliveries'):
                        for column in (('job_id','key','value') if name=='cron_notepad' else ('content',)):
                            if column in columns and all(isinstance(item.get(column),str) for item in source_rows):
                                values[column]=bytes(values[column]).decode('utf-8')
                    if name=='messages' and row.get('content_bytes') is not None:
                        values['content']=bytes(row['content_bytes']).decode('utf-8')
                    # Kanban deliberately accepts both legacy integer and boot-qualified text fingerprints.
                    if 'worker_started_at' in values:
                        source_types={type(item.get('worker_started_at')) for item in source_rows if item.get('worker_started_at') is not None}
                        if source_types=={int} and values['worker_started_at'] is not None:values['worker_started_at']=int(values['worker_started_at'])
                    projected.append(values)
                normalize=lambda rows:json.dumps(sorted(rows,key=lambda value:json.dumps(value,default=_json_value,sort_keys=True)),default=_json_value,sort_keys=True,separators=(',',':'),ensure_ascii=True)
                source_hash=hashlib.sha256(normalize(source_rows).encode()).hexdigest()
                target_hash=hashlib.sha256(normalize(projected).encode()).hexdigest()
                if source_hash!=target_hash:raise ValueError('Imported native payload fingerprint mismatch for '+name)
                fingerprints[name]=source_hash
            receipt['table_fingerprints']=fingerprints
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
        imported,skipped,parents=[],[],[]
        with self._transaction() as conn:
            self._lock_conversation(conn,'__history_import__')
            for item in normalized:
                raw=item['session'];sid=raw['id']
                if self._session_on_conn(conn,sid):skipped.append(sid);continue
                self._import_session_row(conn,raw,item['messages'],sid)
                if raw.get('parent_session_id'):parents.append((sid,raw['parent_session_id']))
                imported.append(sid)
            detached=self._attach_import_parents(conn,parents)
        return dict(ok=True,imported=len(imported),skipped=len(skipped),detached=detached,imported_ids=imported,skipped_ids=skipped,errors=[])

    def _import_session_row(self,conn,raw,messages,sid):
        """Import one normalized history row on the caller's native transaction."""
        from hermes_state_portability import _IMPORT_FLAG_COLS,_IMPORT_PASSTHROUGH_COLS,_IMPORT_FLOAT_COLS,_IMPORT_INT_COLS,_IMPORT_MESSAGE_JSON_FIELDS
        from hermes_cli.timefmt import coerce_epoch
        from hermes_state_postgresql_transcript import insert_fresh_on_conn
        from utils import safe_json_loads
        import time
        started=coerce_epoch(raw.get('started_at'),session_id=sid,field='started_at')
        values=dict(id=sid,source=str(raw.get('source') or 'import'),
            system_prompt_hash=self._store_pg_system_prompt(conn,raw.get('system_prompt')),
            started_at=time.time() if started is None else started)
        values.update({col:1 if self._coerce_or(raw.get(col),int,0) else 0 for col in _IMPORT_FLAG_COLS})
        values.update({col:raw.get(col) for col in _IMPORT_PASSTHROUGH_COLS})
        values.update({col:self._import_float_or_finite(raw.get(col),col) for col in _IMPORT_FLOAT_COLS})
        values.update({col:self._import_int_or_bounded(raw.get(col),col,0) for col in _IMPORT_INT_COLS})
        values['auto_archived'] &= values['archived']
        self._insert(conn,'sessions',values)
        messages=[{**msg,**{key:safe_json_loads(msg.get(key),default=msg.get(key)) if isinstance(msg.get(key),str) else msg.get(key) for key in _IMPORT_MESSAGE_JSON_FIELDS}} for msg in messages]
        ids=insert_fresh_on_conn(self,conn,sid,messages,prune_checkpoints=False)
        for row_id,message in zip(ids,messages):
            if not self._coerce_or(message.get('active'),int,1):
                self._update(conn,'messages',dict(active=0,compacted=1 if self._coerce_or(message.get('compacted'),int,0) else 0),'id',row_id)
        live=[message for message in messages if self._coerce_or(message.get('active'),int,1)]
        self._prune_pg_shadowed_checkpoints(conn,sid,live)
        self._reconcile_pg_display_orders(conn,sid)
        self._recount_on_conn(conn,sid)

    def _store_pg_system_prompt(self,conn,value):
        if value is None:return None
        digest=hashlib.sha256(value.encode('utf-8')).hexdigest()
        self._insert(conn,'system_prompts',dict(hash=digest,prompt=value),conflict=True)
        return digest

    @staticmethod
    def _attach_import_parents(conn, parent_updates: list[tuple]) -> int:
        """Re-attach imported children whose parent exists (in the store or the same payload)
        without creating a cycle; returns the detached count. Only the closing edge of a
        cycle is dropped, so later entries can still attach to the now-root session."""
        parent_by_child = dict(parent_updates)

        def _would_create_cycle(session_id: str, parent_id: str) -> bool:
            seen = {session_id}
            current = parent_id
            while current:
                if current in seen:
                    return True
                seen.add(current)
                if current in parent_by_child:
                    current = parent_by_child[current]
                    continue
                row = conn.execute(
                    "SELECT parent_session_id FROM sessions WHERE id = %s LIMIT 1", (current,),
                ).fetchone()
                if row is None:
                    return False
                current = row["parent_session_id"]
            return False

        detached = 0
        for session_id, parent_id in parent_updates:
            parent_exists = conn.execute("SELECT 1 FROM sessions WHERE id = %s LIMIT 1", (parent_id,)).fetchone()
            if parent_exists and not _would_create_cycle(session_id, parent_id):
                conn.execute("UPDATE sessions SET parent_session_id = %s WHERE id = %s", (parent_id, session_id))
            else:
                parent_by_child.pop(session_id, None)
                detached += 1
        return detached
