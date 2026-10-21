"""Canonical transcript mutation and compression publication with PostgreSQL statements."""
import json
import time
from psycopg import sql
from hermes_state import SessionDB
from hermes_state_common import is_automatic_end_reason, _BOUNDARY_END_REASONS
from hermes_state_sessions import _parse_model_config
from hermes_state_messages import _parse_tool_calls, _tool_calls_count
from hermes_state_postgresql_transcript import native_row,insert_fresh_on_conn


class PostgreSQLMutationsMixin:
    _ROW_STATE_KEYS=SessionDB._ROW_STATE_KEYS
    _execute_transcript_write=SessionDB._execute_transcript_write
    _stamp_kept_live_prefix=SessionDB._stamp_kept_live_prefix
    _reasoning_json_text=staticmethod(SessionDB._reasoning_json_text)
    _session_row_dict=staticmethod(SessionDB._session_row_dict)

    def _store_system_prompt(self,conn,prompt):
        return self._store_pg_system_prompt(conn,prompt)

    def _delete_unreferenced_system_prompts(self,conn):
        self._delete_pg_unused_prompts(conn)

    def _session_turn_lease_key_on_conn(self,conn,session_id):
        return self._lease_key_on_conn(conn,session_id)

    def _check_transcript_write_guards(self,conn,session_id,compression_lock_holder,
            turn_lease_holder=None,turn_lease_ttl_seconds=300.0,reject_active_turn_lease=False,
            reject_active_compression_lock=False,allow_closed_compression_parent=False):
        from hermes_state_errors import CompressionSessionClosedError,SessionTurnLeaseLostError
        from hermes_state import SessionCompressionInProgressError
        from hermes_state_postgresql_leases import stale_lease
        self._lock_conversation(conn,session_id)
        now=time.time()
        if reject_active_compression_lock:
            lock=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s FOR UPDATE').format(self.table('compression_locks')),(session_id,)).fetchone()
            if lock:
                if stale_lease(lock,now):
                    conn.execute(sql.SQL('DELETE FROM {} WHERE session_id=%s AND holder=%s').format(self.table('compression_locks')),(session_id,lock['holder']))
                elif lock['holder']!=compression_lock_holder:
                    raise SessionCompressionInProgressError(f'Session {session_id!r} is being compressed by another writer')
        if turn_lease_holder or reject_active_turn_lease:
            key=self._lease_key_on_conn(conn,session_id)
            lease=conn.execute(sql.SQL('SELECT * FROM {} WHERE conversation_id=%s FOR UPDATE').format(self.table('session_turn_leases')),(key,)).fetchone()
            if turn_lease_holder:
                if not lease or lease['holder']!=turn_lease_holder:
                    raise SessionTurnLeaseLostError(f'Session turn lease lost; refusing transcript write for {session_id!r}')
                if lease['expires_at']<=now:
                    conn.execute(sql.SQL('UPDATE {} SET expires_at=%s WHERE conversation_id=%s AND holder=%s').format(self.table('session_turn_leases')),(now+max(.1,float(turn_lease_ttl_seconds)),key,turn_lease_holder))
            elif lease:
                if not stale_lease(lease,now):
                    raise SessionTurnLeaseLostError(f'Session has an active turn lease; refusing transcript mutation for {session_id!r}')
                conn.execute(sql.SQL('DELETE FROM {} WHERE conversation_id=%s AND holder=%s').format(self.table('session_turn_leases')),(key,lease['holder']))
        row=self._session_on_conn(conn,session_id)
        if row and row['ended_at'] is not None and row['end_reason']=='compression' and not allow_closed_compression_parent:
            raise CompressionSessionClosedError(session_id)

    def _append_messages_in_transaction(self,conn,session_id,messages,*,compression_lock_holder=None,
            turn_lease_holder=None,turn_lease_ttl_seconds=300.0):
        self._check_transcript_write_guards(conn,session_id,compression_lock_holder,
            turn_lease_holder=turn_lease_holder,turn_lease_ttl_seconds=turn_lease_ttl_seconds)
        return len(self._append_on_conn(conn,session_id,messages))

    def _prune_pg_shadowed_checkpoints(self,conn,session_id,messages):
        from agent.context_compressor import _newest_checkpoint_carrier,_is_checkpoint_item
        carrier=_newest_checkpoint_carrier(messages,'codex_reasoning_items')
        if carrier<0 or not isinstance(messages[carrier].get('_row_id'),int):return
        row_id=messages[carrier]['_row_id']
        rows=conn.execute(sql.SQL('SELECT id,codex_reasoning_items FROM {} WHERE session_id=%s AND active=1 AND role=\'assistant\' AND id<%s AND strpos(codex_reasoning_items,\'"compaction"\')>0').format(self.table('messages')),(session_id,row_id)).fetchall()
        for row in rows:
            try:items=json.loads(row['codex_reasoning_items'])
            except (ValueError,TypeError):continue
            if isinstance(items,list) and any(_is_checkpoint_item(item) for item in items):
                kept=[item for item in items if not _is_checkpoint_item(item)]
                self._update(conn,'messages',{'codex_reasoning_items':self._reasoning_json_text(kept)},'id',row['id'])

    def _restamp_pg_messages(self,conn,session_id,messages):
        from agent.message_metadata import DB_ROW_SNAPSHOT
        from agent.transcript_repair import transcript_row_snapshot
        for message in messages:
            row_id=message.get('_row_id')
            if isinstance(row_id,int):
                row=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s AND id=%s').format(self.table('messages')),(session_id,row_id)).fetchone()
                if row:message[DB_ROW_SNAPSHOT]=transcript_row_snapshot(native_row(row))

    def _insert_message_rows(self,conn,session_id,messages,*,prune_checkpoints=True):
        ids=insert_fresh_on_conn(self,conn,session_id,messages,prune_checkpoints=prune_checkpoints)
        return len(ids),sum(_tool_calls_count(_parse_tool_calls(m.get('tool_calls'))) for m in messages)

    def replace_messages(self,session_id,messages,active_only=False,archive_dropped=False,reject_active_turn_lease=False):
        def write(conn):
            self._lock_conversation(conn,session_id)
            if reject_active_turn_lease:
                self._guard_on_conn(conn,session_id,destructive=True)
            else:
                row=self._session_on_conn(conn,session_id)
                if row and row['ended_at'] is not None and row['end_reason']=='compression':
                    from hermes_state_errors import CompressionSessionClosedError
                    raise CompressionSessionClosedError(session_id)
            kept=0
            if archive_dropped:
                raw=conn.execute(sql.SQL('SELECT id,role,content,tool_call_id,tool_calls,message_uid,content_bytes FROM {} WHERE session_id=%s AND active=1 ORDER BY id LIMIT %s').format(self.table('messages')),(session_id,len(messages)+1)).fetchall()
                live=[]
                for item in raw:
                    row=native_row(item)
                    live.append(tuple(row[k] for k in ('id','role','content','tool_call_id','tool_calls','message_uid')))
                kept=self._stamp_kept_live_prefix(live,messages)
                if kept<len(live):
                    conn.execute(sql.SQL('UPDATE {} SET active=0 WHERE session_id=%s AND active=1 AND id>=%s').format(self.table('messages')),(session_id,live[kept][0]))
            else:
                conn.execute(sql.SQL('DELETE FROM {} WHERE session_id=%s{}').format(self.table('messages'),sql.SQL(' AND active=1' if active_only else '')),(session_id,))
            self._insert_message_rows(conn,session_id,messages[kept:])
            self._recount_on_conn(conn,session_id)
        return self._execute_transcript_write(write,messages)

    def _publish_child_session_row(self,conn,parent,*,parent_session_id,child_session_id,source,
            model,model_config,system_prompt,cwd,profile_name):
        from hermes_state_compression import _CONVERSATION_MODEL_CONFIG_KEYS
        carried={key:value for key,value in _parse_model_config(parent['model_config']).items() if key in _CONVERSATION_MODEL_CONFIG_KEYS}
        config={**carried,**(model_config or {})}
        continuation=not any(k in config for k in ('_branched_from','_reset_from'))
        values=dict(id=child_session_id,source=source,model=model,model_config=json.dumps(config) if config else None,
            system_prompt_hash=self._store_pg_system_prompt(conn,system_prompt),tool_names=parent['tool_names'],
            parent_session_id=parent_session_id,cwd=cwd or parent['cwd'],git_branch=parent['git_branch'],git_repo_root=parent['git_repo_root'],
            profile_name=profile_name or parent['profile_name'] or self._own_profile_name(),
            pinned=int(parent['pinned'] or 0) if continuation else 0,started_at=time.time(),
            archived=(parent['archived'] or 0) if continuation else 0,auto_archived=(parent['auto_archived'] or 0) if continuation else 0)
        values.update({key:parent[key] for key in ('user_id','session_key','chat_id','chat_type','thread_id','display_name','origin_json')})
        self._insert(conn,'sessions',values)

    def _lineage_ids_on_conn(self,conn,session_id):
        statement=sql.SQL('''WITH RECURSIVE ancestors(id) AS (
            SELECT %s::text UNION SELECT parent.id FROM ancestors a JOIN {} child ON child.id=a.id
            JOIN {} parent ON parent.id=child.parent_session_id WHERE parent.end_reason='compression'),
        descendants(id) AS (SELECT %s::text UNION SELECT child.id FROM descendants d JOIN {} parent ON parent.id=d.id
            JOIN {} child ON child.parent_session_id=parent.id WHERE parent.end_reason='compression')
        SELECT id FROM ancestors UNION SELECT id FROM descendants''').format(*(self.table('sessions') for _ in range(4)))
        return [row['id'] for row in conn.execute(statement,(session_id,session_id))]

    def _unarchive_pg_auto_lineage(self,conn,session_id):
        ids=self._lineage_ids_on_conn(conn,session_id)
        if conn.execute(sql.SQL('SELECT 1 FROM {} WHERE id=ANY(%s) AND archived<>0 AND COALESCE(auto_archived,0)=0 LIMIT 1').format(self.table('sessions')),(ids,)).fetchone():return False
        return bool(conn.execute(sql.SQL('UPDATE {} SET archived=0,auto_archived=0 WHERE id=ANY(%s) AND archived<>0').format(self.table('sessions')),(ids,)).rowcount)

    def _continuation_children_on_conn(self,conn,parent_session_id,*,live=False):
        rows=conn.execute(sql.SQL('SELECT * FROM {} WHERE parent_session_id=%s{} ORDER BY started_at,id').format(self.table('sessions'),sql.SQL(' AND ended_at IS NULL' if live else '')),(parent_session_id,)).fetchall()
        return [row for row in rows if not self._is_explicit_fork_child_row(dict(row),include_reset=True)]

    def find_live_compression_child(self,parent_session_id):
        if not parent_session_id:return None
        with self._transaction(read_only=True) as conn:
            parent=self._session_on_conn(conn,parent_session_id)
            if not parent or parent['ended_at'] is None or parent['end_reason']!='compression':return None
            rows=self._continuation_children_on_conn(conn,parent_session_id,live=True)
            session_id=rows[0]['id'] if len(rows)==1 else None
        return self.get_session(session_id) if session_id else None

    def reopen_orphaned_compression_session(self,session_id):
        if not session_id:return False
        with self._transaction() as conn:
            self._lock_conversation(conn,session_id)
            parent=self._session_on_conn(conn,session_id,lock=True)
            if not parent or parent['ended_at'] is None or parent['end_reason']!='compression':return False
            if self._continuation_children_on_conn(conn,session_id):return False
            lock=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s FOR UPDATE').format(self.table('compression_locks')),(session_id,)).fetchone()
            if lock:
                if lock['expires_at'] is None or lock['expires_at']>=time.time():return False
                removed=conn.execute(sql.SQL('DELETE FROM {} WHERE session_id=%s AND holder=%s AND expires_at=%s').format(self.table('compression_locks')),(session_id,lock['holder'],lock['expires_at']))
                if removed.rowcount!=1:return False
            return bool(conn.execute(sql.SQL("UPDATE {} SET ended_at=NULL,end_reason=NULL WHERE id=%s AND ended_at IS NOT NULL AND end_reason='compression'").format(self.table('sessions')),(session_id,)).rowcount)

    def reopen_if_explicitly_closed(self,session_id,*,provenance,patience_s=None):
        if not session_id:return None
        with self._transaction() as conn:
            self._lock_conversation(conn,session_id)
            row=self._session_on_conn(conn,session_id,lock=True)
            if not row or row['ended_at'] is None:return None
            reason=row['end_reason']
            if is_automatic_end_reason(reason) or reason=='compression' or reason in _BOUNDARY_END_REASONS:return None
            if self._continuation_children_on_conn(conn,session_id):return None
            conn.execute(sql.SQL('UPDATE {} SET ended_at=NULL,end_reason=NULL WHERE id=%s AND ended_at=%s AND end_reason=%s').format(self.table('sessions')),(session_id,row['ended_at'],reason))
            return str(reason)

    def _preferred_continuation_on_conn(self,conn,parent_session_id):
        rows=conn.execute(sql.SQL('SELECT * FROM {} WHERE parent_session_id=%s').format(self.table('sessions')),(parent_session_id,)).fetchall()
        rows=[row for row in rows if row['source']!='tool' and all(_parse_model_config(row['model_config']).get(k) is None for k in ('_branched_from','_delegate_from','_reset_from'))]
        if not rows:return None
        last={row['session_id']:row['last_message'] for row in conn.execute(sql.SQL('SELECT session_id,max(timestamp) AS last_message FROM {} WHERE session_id=ANY(%s) GROUP BY session_id').format(self.table('messages')),([r['id'] for r in rows],))}
        rows.sort(key=lambda r:r['id'],reverse=True)
        rows.sort(key=lambda r:(0 if r['end_reason']=='compression' else 1 if r['ended_at'] is None else 2,
            -max(r['started_at'] or 0,r['last_activity_at'] or 0,last.get(r['id']) or 0),-(r['started_at'] or 0)))
        return rows[0]

    def _validate_pg_local_lineage(self,conn,receipt):
        from hermes_state_runtime import RuntimeStoreError
        from hermes_state_keys import local_identity
        from hermes_state_local_migration import LEGACY_PREFIX
        if 'legacy_session_id' not in receipt and receipt['session_id']!=local_identity(receipt['profile_id'],receipt['principal_id'],receipt['request_id']):raise RuntimeStoreError('storage_unavailable')
        root=receipt.get('legacy_session_id',receipt['session_id'])
        if 'legacy_session_id' in receipt:
            if root!=receipt['session_id']:raise RuntimeStoreError('storage_unavailable')
            row=conn.execute(sql.SQL('SELECT value FROM {} WHERE key=%s').format(self.table('state_meta')),(LEGACY_PREFIX+root,)).fetchone()
            expected={k:receipt[k] for k in ('session_id','profile_id','principal_id','legacy_session_id')}
            if not row or json.loads(row['value'])!=expected:raise RuntimeStoreError('storage_unavailable')
        lineage=receipt.get('lineage',[root])
        if not isinstance(lineage,list) or not lineage or len(set(lineage))!=len(lineage) or lineage[0]!=root or lineage[-1]!=receipt['entry']['session_id']:raise RuntimeStoreError('storage_unavailable')
        for parent_id,child_id in zip(lineage,lineage[1:]):
            parent=self._session_on_conn(conn,parent_id);child=self._session_on_conn(conn,child_id)
            if not parent or not child or child['parent_session_id']!=parent_id:raise RuntimeStoreError('storage_unavailable')
            if parent['end_reason']=='session_reset':valid=json.loads(child['model_config'] or '{}').get('_reset_from')==parent_id
            else:
                tip=self._preferred_continuation_on_conn(conn,parent_id)
                valid=parent['end_reason']=='compression' and tip is not None and tip['id']==child_id
            if not valid:raise RuntimeStoreError('storage_unavailable')
        tip_parent=self._session_on_conn(conn,lineage[-1])
        if tip_parent and tip_parent['end_reason']=='compression' and self._preferred_continuation_on_conn(conn,lineage[-1]) is not None:raise RuntimeStoreError('storage_unavailable')
        return lineage[-1]

    def _advance_pg_local_target(self,conn,parent_session_id,child_session_id,*,entry=None):
        from hermes_state_local import POLICY_PREFIX
        from hermes_state_local_migration import LEGACY_PREFIX
        from hermes_state_runtime import RuntimeStoreError,_json
        parent=self._session_on_conn(conn,parent_session_id)
        logical_id=parent['chat_id'] if parent else None
        if not logical_id:return
        key=POLICY_PREFIX+logical_id
        raw=conn.execute(sql.SQL('SELECT value FROM {} WHERE key=%s').format(self.table('state_meta')),(key,)).fetchone()
        if raw is None:
            legacy=conn.execute(sql.SQL('SELECT 1 FROM {} WHERE key=%s').format(self.table('state_meta')),(LEGACY_PREFIX+logical_id,)).fetchone()
            if logical_id.startswith('local-') or legacy:raise RuntimeStoreError('storage_unavailable')
            return
        try:
            receipt=json.loads(raw['value'])
            if receipt['session_id']!=logical_id or receipt['entry']['session_id']!=parent_session_id or receipt['principal_id']!=parent['user_id'] or receipt['route']!=parent['session_key']:raise RuntimeStoreError('storage_unavailable')
            route=conn.execute(sql.SQL("SELECT entry_json FROM {} WHERE scope='' AND session_key=%s").format(self.table('gateway_routing')),(receipt['route'],)).fetchone()
            routed=json.loads(route['entry_json']) if route else None
            if routed is None or routed['session_id']!=parent_session_id:raise RuntimeStoreError('admission_conflict')
            candidate=dict(entry or routed,session_id=child_session_id)
            receipt['entry']=candidate;receipt['lineage']=[*receipt.get('lineage',[logical_id]),child_session_id]
            self._validate_pg_local_lineage(conn,receipt)
        except (KeyError,TypeError,ValueError) as exc:
            if isinstance(exc,RuntimeStoreError):raise
            raise RuntimeStoreError('storage_unavailable') from exc
        conn.execute(sql.SQL("UPDATE {} SET entry_json=%s WHERE scope='' AND session_key=%s").format(self.table('gateway_routing')),(_json(candidate),receipt['route']))
        self._update(conn,'state_meta',{'value':_json(receipt)},'key',key)
        conn.execute(sql.SQL('UPDATE {} SET runtime_revision=runtime_revision+1 WHERE id=%s').format(self.table('sessions')),(logical_id,))

    def publish_compression_child(self,*,parent_session_id,child_session_id,source,messages,model=None,
            model_config=None,system_prompt=None,cwd=None,profile_name=None,compression_lock_holder=None,
            require_compression_lease=True,require_lease_refresh=False,lease_ttl_seconds=300,watermark=None,watermark_ceiling=None):
        return self._execute_transcript_write(lambda conn:self._publish_compression_child_on_conn(conn,
            parent_session_id=parent_session_id,child_session_id=child_session_id,source=source,messages=messages,
            model=model,model_config=model_config,system_prompt=system_prompt,cwd=cwd,profile_name=profile_name,
            compression_lock_holder=compression_lock_holder,require_compression_lease=require_compression_lease,
            require_lease_refresh=require_lease_refresh,lease_ttl_seconds=lease_ttl_seconds,watermark=watermark,watermark_ceiling=watermark_ceiling),messages)

    def _publish_compression_child_on_conn(self,conn,*,parent_session_id,child_session_id,source,messages,model=None,
            model_config=None,system_prompt=None,cwd=None,profile_name=None,compression_lock_holder=None,
            require_compression_lease=True,require_lease_refresh=False,lease_ttl_seconds=300,watermark=None,watermark_ceiling=None):
        from hermes_state_errors import CompressionSessionBusyError
        self._lock_conversation(conn,parent_session_id)
        if require_lease_refresh and compression_lock_holder:
            conn.execute(sql.SQL('UPDATE {} SET expires_at=%s WHERE session_id=%s AND holder=%s').format(self.table('compression_locks')),(time.time()+lease_ttl_seconds,parent_session_id,compression_lock_holder))
        lock=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s FOR UPDATE').format(self.table('compression_locks')),(parent_session_id,)).fetchone()
        if require_compression_lease and (not lock or not compression_lock_holder or lock['holder']!=compression_lock_holder or lock['expires_at']<=time.time()):raise CompressionSessionBusyError('Compression lease lost before publication: '+parent_session_id)
        parent=self._session_on_conn(conn,parent_session_id,lock=True)
        if not parent:raise RuntimeError('Compression parent not found: '+parent_session_id)
        if parent['ended_at'] is not None:
            if not is_automatic_end_reason(parent['end_reason']):raise RuntimeError('Compression parent already ended: '+parent_session_id)
            self._update(conn,'sessions',{'ended_at':None,'end_reason':None},'id',parent_session_id)
        if not messages:raise RuntimeError('Compression child handoff must not be empty')
        self._publish_child_session_row(conn,parent,parent_session_id=parent_session_id,child_session_id=child_session_id,
            source=source,model=model,model_config=model_config,system_prompt=system_prompt,cwd=cwd,profile_name=profile_name)
        self._carry_pg_timestamps(conn,parent_session_id,messages)
        self._insert_message_rows(conn,child_session_id,messages)
        if watermark is not None:
            rows=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s AND active=1 AND id>%s AND (%s::bigint IS NULL OR id<=%s) ORDER BY id').format(self.table('messages')),(parent_session_id,int(watermark),watermark_ceiling,watermark_ceiling)).fetchall()
            for raw in rows:
                values={k:v for k,v in dict(raw).items() if k not in ('id','active','compacted','display_order')}
                values.update(session_id=child_session_id,active=1,compacted=0)
                self._insert(conn,'messages',values)
        self._reconcile_pg_display_orders(conn,child_session_id)
        self._recount_on_conn(conn,child_session_id)
        updated=conn.execute(sql.SQL("UPDATE {} SET ended_at=%s,end_reason='compression' WHERE id=%s AND ended_at IS NULL").format(self.table('sessions')),(time.time(),parent_session_id))
        if updated.rowcount!=1:raise RuntimeError('Compression parent changed during publication: '+parent_session_id)
        self._advance_pg_local_target(conn,parent_session_id,child_session_id)
        if parent['archived']:self._unarchive_pg_auto_lineage(conn,child_session_id)
