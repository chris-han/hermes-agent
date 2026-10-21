"""PostgreSQL deletion with native terminal receipts, privacy retirement, and route fences."""
import json
import time
from psycopg import sql
from hermes_state_runtime import RuntimeStoreError,admission_fingerprint,_json
from hermes_state_mutation_retirement import RETIRED_PREFIX,entry_session_id,_MUTATION_TRANSCRIPT_FIELDS
from hermes_state_local import POLICY_PREFIX,API_BINDING_PREFIX,API_DECLARED_PREFIX,_policy_target


def _meta(store,conn,key):
    row=conn.execute(sql.SQL('SELECT value FROM {} WHERE key=%s').format(store.table('state_meta')),(key,)).fetchone()
    return row['value'] if row else None


def _put_meta(store,conn,key,value,*,replace=False):
    if replace:
        conn.execute(sql.SQL('INSERT INTO {}(key,value) VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=excluded.value').format(store.table('state_meta')),(key,value))
    else:store._insert(conn,'state_meta',dict(key=key,value=value))


def owned_lineage_ids(store,conn,session_id):
    row=store._session_on_conn(conn,session_id)
    owner=row['chat_id'] if row else None
    def lineage(key):
        raw=_meta(store,conn,POLICY_PREFIX+key)
        try:return json.loads(raw).get('lineage') if raw else None
        except (ValueError,TypeError,AttributeError):return None
    members=lineage(owner) if owner and owner!=session_id else None
    owner=owner if isinstance(members,list) and session_id in members else session_id
    members=lineage(owner)
    return list(dict.fromkeys([session_id,*([owner,*(s for s in members if isinstance(s,str))] if isinstance(members,list) else [])]))


def delegate_ids(rows,parents):
    from hermes_state_sessions import _parse_model_config
    seen=set(parents);frontier=set(parents)
    while frontier:
        next_ids=set()
        for row in rows:
            marker=_parse_model_config(row['model_config']).get('_delegate_from')
            if row['id'] not in seen and marker is not None and ((isinstance(marker,str) and marker in frontier) or row['parent_session_id'] in frontier):
                next_ids.add(row['id'])
        seen.update(next_ids);frontier=next_ids
    return sorted(seen-set(parents))


def _retire_terminal(store,conn,ids):
    from hermes_state_terminal import ADMISSION_PREFIX,WORKER_PREFIX,RESULT_PREFIX,identity_key
    from hermes_state_media import PREFIX as MEDIA_PREFIX
    from gateway.session_ingress_media import admission_media_references,hosted_document_references
    import hashlib
    for sid in ids:
        admissions=conn.execute(sql.SQL('SELECT * FROM {} WHERE target_session_id=%s').format(store.table('session_admissions')),(sid,)).fetchall()
        workers=conn.execute(sql.SQL('SELECT * FROM {} WHERE session_id=%s').format(store.table('worker_executions')),(sid,)).fetchall()
        states={r['status'] for r in [*admissions,*workers]}-{'terminal'}
        if states:raise RuntimeStoreError('unknown_execution' if 'unknown' in states else 'session_busy')
        for raw in admissions:
            row=dict(raw);payload=json.loads(row['payload_json'])
            refs=[*admission_media_references(payload),*hosted_document_references(row['request_id'],payload),*payload.get('api_turn_v1',{}).get('media',[])]
            for ref in refs:
                conn.execute(sql.SQL('INSERT INTO {}(key,value) VALUES(%s,%s) ON CONFLICT DO NOTHING').format(store.table('state_meta')),(MEDIA_PREFIX+hashlib.sha256(ref['path'].encode()).hexdigest(),json.dumps(ref)))
            row.update(payload_json='{}',lineage_json='[]',retired_at=time.time())
            result_key=RESULT_PREFIX+row['admission_id'];saved=_meta(store,conn,result_key)
            if saved is not None:
                value=json.loads(saved);result=value.get('result')
                if isinstance(result,dict):value['result']={**{k:v for k,v in result.items() if k not in ('messages','last_reasoning','tools','_turn_user_row')},'messages':[]}
                _put_meta(store,conn,result_key,_json(value),replace=True)
            _put_meta(store,conn,ADMISSION_PREFIX+row['admission_id'],_json(row))
            _put_meta(store,conn,identity_key(row['principal_id'],sid,row['request_id']),json.dumps(row['admission_id']))
        for raw in workers:
            row=dict(raw)
            receipts=[dict(r) for r in conn.execute(sql.SQL('SELECT sequence,payload_digest,result_json FROM {} WHERE execution_id=%s ORDER BY sequence').format(store.table('worker_receipts')),(row['execution_id'],))]
            closing=admission_fingerprint(canonical_target=sid,payload={'operation':'execution.finish','payload':{}})
            for receipt in receipts:
                if receipt['payload_digest']!=closing:receipt['result_json']=None
            row.update(receipts=receipts,retired_at=time.time())
            _put_meta(store,conn,WORKER_PREFIX+row['execution_id'],_json(row))
            conn.execute(sql.SQL('DELETE FROM {} WHERE execution_id=%s').format(store.table('worker_receipts')),(row['execution_id'],))
        conn.execute(sql.SQL('DELETE FROM {} WHERE session_id=%s').format(store.table('worker_executions')),(sid,))
        conn.execute(sql.SQL('DELETE FROM {} WHERE target_session_id=%s').format(store.table('session_admissions')),(sid,))


def retire_sessions(store,conn,ids):
    _retire_terminal(store,conn,ids)
    targets=set(ids)
    rows=conn.execute(sql.SQL('SELECT key,value FROM {} WHERE starts_with(key,%s)').format(store.table('state_meta')),('gateway.mutation.v1.',)).fetchall()
    for row in rows:
        try:receipt=json.loads(row['value'])
        except (ValueError,TypeError):continue
        result=receipt.get('result') if isinstance(receipt,dict) else None
        if not isinstance(result,dict):continue
        message=result.get('target_message')
        refs=[result.get('session_id'),message.get('session_id') if isinstance(message,dict) else None,result.get('target_session_id'),result.get('previous_target_session_id')]
        if any(isinstance(ref,str) and ref in targets for ref in refs) and any(k in result for k in _MUTATION_TRANSCRIPT_FIELDS):
            receipt['result']={k:None if k in _MUTATION_TRANSCRIPT_FIELDS else v for k,v in result.items()}
            _put_meta(store,conn,row['key'],_json(receipt),replace=True)
    for sid in ids:
        _put_meta(store,conn,RETIRED_PREFIX+sid,json.dumps({'retired_at':time.time()}),replace=True)
        conn.execute(sql.SQL('DELETE FROM {} WHERE key=%s').format(store.table('state_meta')),('gateway.api.settings.v1.'+sid,))
    for row in conn.execute(sql.SQL('SELECT scope,session_key,entry_json FROM {}').format(store.table('gateway_routing'))).fetchall():
        if entry_session_id(row['entry_json']) in targets:
            conn.execute(sql.SQL('DELETE FROM {} WHERE scope=%s AND session_key=%s').format(store.table('gateway_routing')),(row['scope'],row['session_key']))
    rows=conn.execute(sql.SQL('SELECT key,value FROM {} WHERE starts_with(key,%s) OR starts_with(key,%s)').format(store.table('state_meta')),(POLICY_PREFIX,API_DECLARED_PREFIX)).fetchall()
    gone=[row['key'] for row in rows if (_policy_target(row['value']) in targets if row['key'].startswith(POLICY_PREFIX) else row['value'] in targets)]
    gone.extend(API_BINDING_PREFIX+sid for sid in ids)
    conn.execute(sql.SQL('DELETE FROM {} WHERE key=ANY(%s)').format(store.table('state_meta')),(gone,))


class PostgreSQLRetirementMixin:
    def get_session_delete_targets(self,session_id):
        with self._transaction(read_only=True) as conn:
            if not self._session_on_conn(conn,session_id):return []
            rows=conn.execute(sql.SQL('SELECT * FROM {}').format(self.table('sessions'))).fetchall()
            scope=owned_lineage_ids(self,conn,session_id)
            return [session_id,*sorted(set(scope)-{session_id}),*delegate_ids(rows,scope)]

    def delete_session(self,session_id,sessions_dir=None,expected_delete_ids=None,expected_display_messages=None,
            exclude_active_write_guards=False,include_compression_chain=False):
        if include_compression_chain:
            return self.delete_sessions([session_id],sessions_dir=sessions_dir,exclude_active_write_guards=exclude_active_write_guards,include_compression_chain=True)>0
        with self._transaction() as conn:
            self._lock_conversation(conn,session_id)
            if not self._session_on_conn(conn,session_id):return False
            rows=conn.execute(sql.SQL('SELECT * FROM {}').format(self.table('sessions'))).fetchall()
            scope=owned_lineage_ids(self,conn,session_id)
            ids=list(dict.fromkeys([*scope,*delegate_ids(rows,scope)]))
            if exclude_active_write_guards and self._guarded_pg_ids(conn,ids):
                from hermes_state_errors import SessionActiveWriteGuardError
                raise SessionActiveWriteGuardError('Session or delegate child has an active turn lease or compression lock')
            if expected_delete_ids is not None and set(expected_delete_ids)!=set(ids):return False
            if expected_display_messages is not None and any(self.get_messages(sid,include_compacted=True)!=expected for sid,expected in expected_display_messages.items()):return False
            self._delete_pg_ids(conn,ids)
        from hermes_state import SessionDB
        for sid in ids:SessionDB._remove_session_files(sessions_dir,sid)
        return True

    def _guarded_pg_ids(self,conn,ids):
        now=time.time()
        for sid in ids:
            key=self._lease_key_on_conn(conn,sid)
            if conn.execute(sql.SQL('SELECT 1 FROM {} WHERE conversation_id=%s AND expires_at>%s UNION ALL SELECT 1 FROM {} WHERE session_id=%s AND expires_at>%s LIMIT 1').format(self.table('session_turn_leases'),self.table('compression_locks')),(key,now,sid,now)).fetchone():return True
        return False

    def _delete_pg_ids(self,conn,ids):
        retire_sessions(self,conn,ids)
        conn.execute(sql.SQL('UPDATE {} SET parent_session_id=NULL WHERE parent_session_id=ANY(%s)').format(self.table('sessions')),(ids,))
        conn.execute(sql.SQL('DELETE FROM {} WHERE child_session_id=ANY(%s) OR source_session_id=ANY(%s)').format(self.table('semantier_session_branches')),(ids,ids))
        conn.execute(sql.SQL('DELETE FROM {} WHERE session_id=ANY(%s)').format(self.table('messages')),(ids,))
        conn.execute(sql.SQL('DELETE FROM {} WHERE id=ANY(%s)').format(self.table('sessions')),(ids,))
        conn.execute(sql.SQL('DELETE FROM {} p WHERE NOT EXISTS(SELECT 1 FROM {} s WHERE s.system_prompt_hash=p.hash OR s.tool_names=p.hash)').format(self.table('system_prompts'),self.table('sessions')))

    def _expand_pg_compression_lineage(self,conn,ids,rows):
        from hermes_state_sessions import _parse_model_config
        seen=set(ids)
        by_id={row['id']:row for row in rows}
        for _ in range(1000):
            before=len(seen)
            for row in rows:
                parent=by_id.get(row['parent_session_id'])
                config=_parse_model_config(row['model_config'])
                if parent and parent['end_reason']=='compression' and row['source']!='tool' and all(config.get(key) is None for key in ('_branched_from','_delegate_from','_reset_from')):
                    if row['id'] in seen or parent['id'] in seen:seen.update((row['id'],parent['id']))
            if len(seen)==before:break
        for sid in tuple(seen):seen.update(owned_lineage_ids(self,conn,sid))
        return sorted(seen)

    def delete_sessions(self,session_ids,sessions_dir=None,exclude_active_write_guards=False,skipped_ids=None,include_compression_chain=False):
        unique={sid for sid in session_ids or () if isinstance(sid,str) and sid}
        if not unique:return 0
        with self._transaction() as conn:
            rows=conn.execute(sql.SQL('SELECT * FROM {}').format(self.table('sessions'))).fetchall()
            existing=[row['id'] for row in rows if row['id'] in unique]
            if not existing:return 0
            def expand(ids):
                return self._expand_pg_compression_lineage(conn,ids,rows) if include_compression_chain else list(dict.fromkeys(x for sid in ids for x in owned_lineage_ids(self,conn,sid)))
            if exclude_active_write_guards:
                active=[]
                for sid in existing:
                    members=expand([sid]);guard_pool=[*members,*delegate_ids(rows,members)]
                    if self._guarded_pg_ids(conn,guard_pool):active.append(sid)
                if skipped_ids is not None:skipped_ids.extend(sorted(active))
                existing=[sid for sid in existing if sid not in active]
                if not existing:return 0
            expanded=expand(existing)
            removed=list(dict.fromkeys([*expanded,*delegate_ids(rows,expanded)]))
            self._delete_pg_ids(conn,removed)
        from hermes_state import SessionDB
        for sid in removed:SessionDB._remove_session_files(sessions_dir,sid)
        return len(existing)
