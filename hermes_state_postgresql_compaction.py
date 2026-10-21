"""Native PostgreSQL in-place compaction with canonical held-row and concurrent-tail rules."""
import json
import time
from psycopg import sql
from hermes_state import SessionDB as NativeSessionDB
from hermes_state_postgresql_coverage import PostgreSQLCoverageMixin
from hermes_state_postgresql_transcript import native_row,insert_fresh_on_conn
from hermes_state_messages import _parse_tool_calls,_scrub_surrogates
from hermes_cli.timefmt import coerce_epoch
from hermes_state_sessions import _parse_model_config


class PostgreSQLCompactionMixin(PostgreSQLCoverageMixin):
    _row_identity=staticmethod(NativeSessionDB._row_identity)
    _encode_content=staticmethod(NativeSessionDB._encode_content)
    _decode_content=staticmethod(NativeSessionDB._decode_content)
    _loaded_view_content=staticmethod(NativeSessionDB._loaded_view_content)

    def _clone_pg_rows(self,conn,ids):
        if not ids:return
        rows=conn.execute(sql.SQL('SELECT * FROM {} WHERE id=ANY(%s) ORDER BY id').format(self.table('messages')),(ids,)).fetchall()
        for row in rows:
            values={k:v for k,v in dict(row).items() if k not in ('id','display_order')}
            values.update(active=1,compacted=0)
            self._insert(conn,'messages',values)

    def _carry_pg_timestamps(self,conn,session_id,messages):
        # Canonical carry only consumes active donors, in durable order.
        donors={}
        for raw in conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s AND active=1 ORDER BY id').format(self.table('messages')),(session_id,)):
            row=native_row(raw)
            key=(row['role'],row['content'],row['tool_call_id'],row['tool_calls'],row['tool_name'])
            donors.setdefault(key,[]).append(row['timestamp'])
        for message in messages:
            if coerce_epoch(message.get('timestamp'),field='message timestamp') is not None:continue
            calls=_parse_tool_calls(message.get('tool_calls'))
            key=(message.get('role','unknown'),self._encode_content(message.get('content')),message.get('tool_call_id'),json.dumps(calls) if calls else None,_scrub_surrogates(message.get('tool_name')))
            if donors.get(key):message['timestamp']=donors[key].pop(0)

    def _reconcile_pg_display_orders(self,conn,session_id):
        codec=object.__new__(NativeSessionDB)
        first={}
        rows=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s AND (active=1 OR compacted=1) ORDER BY id').format(self.table('messages')),(session_id,)).fetchall()
        for raw in rows:
            row=native_row(raw)
            identity=codec._display_identity(codec._display_dedupe_key(row))
            order=first.setdefault(identity,row['id'])
            if row['display_order']!=order or row['display_identity']!=identity:
                self._update(conn,'messages',{'display_order':order,'display_identity':identity},'id',row['id'])

    def archive_and_compact(self,session_id,compacted_messages,model_config_patch=None,watermark=None,
            lock_holder=None,tail_count=0,carried_messages=None,covered_ids=None,unresolved_held=None,
            *,compression_lock_holder=None):
        if lock_holder is not None and compression_lock_holder is not None and lock_holder!=compression_lock_holder:
            raise ValueError('Conflicting compression owner')
        holder=lock_holder or compression_lock_holder
        return self._execute_transcript_write(lambda conn:self._archive_and_compact_on_conn(
            conn,session_id,compacted_messages,model_config_patch=model_config_patch,watermark=watermark,
            lock_holder=holder,tail_count=tail_count,carried_messages=carried_messages,
            covered_ids=covered_ids,unresolved_held=unresolved_held),compacted_messages)

    def _archive_and_compact_on_conn(self,conn,session_id,compacted_messages,model_config_patch=None,
            watermark=None,lock_holder=None,tail_count=0,carried_messages=None,covered_ids=None,unresolved_held=None):
        holder=lock_holder
        self._lock_conversation(conn,session_id)
        if holder is not None:
            from hermes_state import SessionCompressionInProgressError
            lock=conn.execute(sql.SQL('SELECT holder,expires_at FROM {} WHERE session_id=%s').format(self.table('compression_locks')),(session_id,)).fetchone()
            if not lock or lock['holder']!=holder or lock['expires_at']<=time.time():
                raise SessionCompressionInProgressError('Compression lease lost before commit')
        row=self._session_on_conn(conn,session_id,lock=True)
        patched=None
        if model_config_patch is not None:
            cfg=_parse_model_config(row['model_config'])
            for key,value in model_config_patch.items():
                if value is None:cfg.pop(key,None)
                else:cfg[key]=value
            patched=json.dumps(cfg) if cfg else None
        proved=self._proved_coverage(conn,session_id,covered_ids,unresolved_held,watermark)
        active=conn.execute(sql.SQL('SELECT id FROM {} WHERE session_id=%s AND active=1 ORDER BY id').format(self.table('messages')),(session_id,)).fetchall()
        active_ids=[r['id'] for r in active]
        carried=self._resolve_carried_row_ids(conn,session_id,carried_messages or [])
        if proved is not None:
            covered=set(proved[0])|set(carried)
            tail=[rid for rid in active_ids if rid not in covered]
            summarized=[rid for rid in active_ids if rid in covered]
            rewound=carried+(self._tail_originals(summarized,tail_count,proved[1]) if tail_count>0 else [])+tail
        else:
            tail=[] if watermark is None else [rid for rid in active_ids if rid>int(watermark)]
            bound=[rid for rid in active_ids if watermark is None or rid<=int(watermark)]
            width=int(tail_count)+self._uncounted_merged_rows(compacted_messages[-int(tail_count):]) if tail_count>0 else 0
            rewound=carried+(bound[-width:] if width else [])+tail
        if rewound:
            conn.execute(sql.SQL('UPDATE {} SET active=0,compacted=0 WHERE session_id=%s AND id=ANY(%s)').format(self.table('messages')),(session_id,list(dict.fromkeys(rewound))))
        conn.execute(sql.SQL('UPDATE {} SET active=0,compacted=1 WHERE session_id=%s AND active=1').format(self.table('messages')),(session_id,))
        self._carry_pg_timestamps(conn,session_id,compacted_messages)
        insert_fresh_on_conn(self,conn,session_id,compacted_messages,prune_checkpoints=False)
        self._clone_pg_rows(conn,tail)
        self._prune_pg_shadowed_checkpoints(conn,session_id,compacted_messages)
        self._reconcile_pg_display_orders(conn,session_id)
        self._restamp_pg_messages(conn,session_id,compacted_messages)
        self._recount_on_conn(conn,session_id)
        if model_config_patch is not None:self._update(conn,'sessions',{'model_config':patched},'id',session_id)
        return conn.execute(sql.SQL('SELECT message_count FROM {} WHERE id=%s').format(self.table('sessions')),(session_id,)).fetchone()['message_count']
