"""Native hosted retention; PostgreSQL owns vacuuming, never SQLite file maintenance."""
from contextlib import contextmanager
import json
import logging
import time

from psycopg import sql
from hermes_state_common import EPOCH_MIN,EPOCH_MAX
from hermes_state_sessions import _parse_model_config
from hermes_startup_watchdog import report_startup_progress

logger=logging.getLogger('hermes_state')


def _ancestry(ids,parents):
    seen=set(ids);pending=list(ids)
    while pending:
        parent=parents.get(pending.pop())
        if parent and parent not in seen:seen.add(parent);pending.append(parent)
    return seen


def _select(rows,parents,older_than_days,source,filters,*,ended=True,whole=False):
    from hermes_state_maintenance import _PRUNE_FILTER_NAMES
    unknown=set(filters)-_PRUNE_FILTER_NAMES
    if unknown:raise TypeError('Unsupported maintenance filter: '+min(unknown))
    filters=dict(filters)
    if older_than_days is not None and filters.get('last_active_before') is None and filters.get('started_before') is None:
        if older_than_days<0:raise ValueError('older_than_days must be >= 0')
        filters['last_active_before']=time.time()-older_than_days*86400
    pinned={row['id'] for row in rows if row['pinned']}
    while True:
        tail=pinned|{child for child,parent in parents.items() if parent in pinned}
        if tail==pinned:break
        pinned=tail
    selected=[]
    for row in rows:
        if (row['ended_at'] is not None)!=ended:continue
        if source and row['source']!=source:continue
        if not filters.get('include_pinned',False) and row['id'] in pinned:continue
        if isinstance(filters.get('archived'),bool) and bool(row['archived'])!=filters['archived']:continue
        if filters.get('lineage_tips_only') and row['end_reason']=='compression':continue
        if _matches(row,filters):selected.append(row)
    if whole:
        kept=_ancestry({row['id'] for row in rows}-{row['id'] for row in selected},parents)
        selected=[row for row in selected if row['id'] not in kept]
    return sorted(selected,key=lambda row:(row['last_active'] is not None,row['last_active'] or 0,row['started_at'] or 0))


def _matches(row,filters):
    # Finite native filter surface: values are data, never incoming SQL.
    for name,column in [('last_active','last_active'),('started','started_at')]:
        for suffix,less in [('before',True),('after',False)]:
            bound=filters.get(name+'_'+suffix)
            if bound is not None and (row[column] is None or (row[column]>=bound if less else row[column]<bound)):return False
    cutoff=filters.get('last_active_before')
    if cutoff is not None and row['end_reason']=='startup_orphan_reap' and (row['ended_at'] is None or row['ended_at']>=cutoff):return False
    for key in ('source','end_reason','user_id','chat_id','chat_type'):
        if filters.get(key) and row[key]!=filters[key]:return False
    for key,column in [('title_like','title'),('model_like','model'),('branch_like','git_branch')]:
        if filters.get(key) and filters[key].lower() not in (row[column] or '').lower():return False
    if filters.get('provider') and (row['billing_provider'] or '').lower()!=filters['provider'].lower():return False
    prefix=filters.get('cwd_prefix')
    if prefix and not (row['cwd']==prefix or (row['cwd'] or '').startswith((prefix.rstrip('/\\')+'/',prefix.rstrip('/\\')+'\\'))):return False
    quantities={'messages':row['message_count'],'tokens':(row['input_tokens'] or 0)+(row['output_tokens'] or 0),'cost':row['actual_cost_usd'] if row['actual_cost_usd'] is not None else row['estimated_cost_usd'] or 0,'tool_calls':row['tool_call_count'] or 0}
    for name,value in quantities.items():
        if filters.get('min_'+name) is not None and (value is None or value<filters['min_'+name]):return False
        if filters.get('max_'+name) is not None and (value is None or value>filters['max_'+name]):return False
    return True


class PostgreSQLMaintenanceMixin:
    @contextmanager
    def _maintenance_admission(self):
        # Session-level admission spans the short independently committed owner operations.
        with self._transaction(read_only=True) as conn:
            key=self.scope.schema+'\x1fnative-auto-maintenance'
            acquired=conn.execute('SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS acquired',(key,)).fetchone()['acquired']
            try:yield acquired
            finally:
                if acquired:conn.execute('SELECT pg_advisory_unlock(hashtextextended(%s,0))',(key,))

    def _maintenance_snapshot(self,conn):
        statement=sql.SQL('''SELECT s.*,COALESCE((SELECT MAX(v) FROM (VALUES(s.last_activity_at),
            ((SELECT MAX(m.timestamp) FROM {} m WHERE m.session_id=s.id AND m.timestamp BETWEEN %s AND %s))) AS activity(v)
            WHERE v BETWEEN %s AND %s),CASE WHEN s.started_at BETWEEN %s AND %s THEN s.started_at END) AS last_active
            FROM {} s''').format(self.table('messages'),self.table('sessions'))
        rows=conn.execute(statement,(EPOCH_MIN,EPOCH_MAX,EPOCH_MIN,EPOCH_MAX,EPOCH_MIN,EPOCH_MAX)).fetchall()
        by_id={row['id']:row for row in rows};parents={}
        for row in rows:
            parent=by_id.get(row['parent_session_id']);config=_parse_model_config(row['model_config'])
            if parent and parent['end_reason']=='compression' and row['source']!='tool' and all(config.get(key)!=parent['id'] for key in ('_branched_from','_delegate_from','_reset_from')):parents[row['id']]=parent['id']
        return rows,parents

    def list_prune_candidates(self,older_than_days=None,source=None,*,whole_lineages=False,**filters):
        with self._transaction(read_only=True) as conn:
            rows,parents=self._maintenance_snapshot(conn)
            return [{key:row[key] for key in ('id','source','title','model','started_at','last_active','ended_at','message_count','archived')} for row in _select(rows,parents,older_than_days,source,filters,whole=whole_lineages)]

    def count_prune_matches(self,older_than_days=None,source=None,*,pinned_only=False,**filters):
        with self._transaction(read_only=True) as conn:
            rows,parents=self._maintenance_snapshot(conn)
            selected=_select(rows,parents,older_than_days,source,{**filters,**({'include_pinned':True} if pinned_only else {})})
            return sum(bool(row['pinned']) for row in selected) if pinned_only else len(selected)

    def count_open_prune_matches(self,older_than_days=None,source=None,**filters):
        with self._transaction(read_only=True) as conn:
            rows,parents=self._maintenance_snapshot(conn)
            return len(_select(rows,parents,older_than_days,source,filters,ended=False))

    def archive_stale_sessions(self,idle_days,*,exclude_pinned=True):
        if idle_days is None or idle_days<0:return 0
        cutoff=time.time()-float(idle_days)*86400
        with self._transaction() as conn:
            rows,parents=self._maintenance_snapshot(conn)
            pinned={row['id'] for row in rows if row['pinned']}
            tips=[row for row in rows if not row['archived'] and row['end_reason']!='compression' and row['last_active'] is not None and row['last_active']<cutoff and not(row['hidden'] and row['title']==self.CANONICAL_BOT_CHAT_TITLE)]
            selected=[row for row in tips if not exclude_pinned or not(_ancestry([row['id']],parents)&pinned)]
            ids=list({sid for row in selected for sid in self._lineage_ids_on_conn(conn,row['id'])})
            conn.execute(sql.SQL('UPDATE {} SET auto_archived=CASE WHEN archived=0 THEN 1 ELSE auto_archived END,archived=1 WHERE id=ANY(%s)').format(self.table('sessions')),(ids,))
            return len(selected)

    def archive_sessions(self,older_than_days=None,source=None,**filters):
        filters.setdefault('archived',False);filters['lineage_tips_only']=True
        with self._transaction() as conn:
            rows,parents=self._maintenance_snapshot(conn)
            selected=_select(rows,parents,older_than_days,source,filters)
            ids=list({sid for row in selected for sid in self._lineage_ids_on_conn(conn,row['id'])})
            conn.execute(sql.SQL('UPDATE {} SET archived=1,auto_archived=0 WHERE id=ANY(%s)').format(self.table('sessions')),(ids,))
            return len(selected)

    def maybe_auto_archive(self,idle_days=3,min_interval_hours=24,exclude_pinned=True):
        from hermes_state_sessions import SessionSessionsMixin
        try:
            with self._maintenance_admission() as acquired:
                if not acquired:return {'skipped':True,'archived':0}
                return SessionSessionsMixin.maybe_auto_archive(self,idle_days,min_interval_hours,exclude_pinned)
        except Exception as exc:
            logger.warning('PostgreSQL auto-archive unavailable: %s',type(exc).__name__)
            return {'skipped':False,'archived':0,'error':type(exc).__name__}

    def prune_sessions(self,older_than_days=90,source=None,sessions_dir=None,exclude_active_write_guards=False,deleted_ids=None,**filters):
        with self._transaction() as conn:
            rows,parents=self._maintenance_snapshot(conn)
            selected=_select(rows,parents,older_than_days,source,filters,whole=True)
            busy={row['id'] for row in conn.execute(sql.SQL("SELECT target_session_id AS id FROM {} WHERE status<>'terminal' UNION SELECT session_id AS id FROM {} WHERE status<>'terminal'").format(self.table('session_admissions'),self.table('worker_executions')))}
            ids={row['id'] for row in selected if row['id'] not in busy and (not exclude_active_write_guards or not self._guarded_pg_ids(conn,[row['id']]))}
            kept=_ancestry({row['id'] for row in rows}-ids,parents);removed=sorted(ids-kept)
            if removed:self._delete_pg_ids(conn,removed)
            self._prune_pg_tombstones(conn)
        from hermes_state import SessionDB
        from hermes_state_media import collect_retired_media
        collect_retired_media(self)
        for sid in removed:SessionDB._remove_session_files(sessions_dir,sid)
        if deleted_ids is not None:deleted_ids.extend(removed)
        return len(removed)

    def _prune_pg_tombstones(self,conn):
        from hermes_state_terminal import ADMISSION_PREFIX,RESULT_PREFIX,WORKER_PREFIX,identity_key
        from hermes_state_mutation_retirement import RETIRED_PREFIX,RETRY_HORIZON_S
        now=time.time();prefixes=(ADMISSION_PREFIX,WORKER_PREFIX,RETIRED_PREFIX)
        rows=conn.execute(sql.SQL('SELECT key,value FROM {} WHERE starts_with(key,%s) OR starts_with(key,%s) OR starts_with(key,%s)').format(self.table('state_meta')),prefixes).fetchall();gone=[]
        for row in rows:
            try:value=json.loads(row['value'])
            except (ValueError,TypeError):continue
            if not isinstance(value,dict):continue
            if value.get('retired_at') is None:
                value['retired_at']=now;self.set_meta(row['key'],json.dumps(value),cursor=conn);continue
            if not isinstance(value['retired_at'],(int,float)) or value['retired_at']>=now-RETRY_HORIZON_S:continue
            gone.append(row['key'])
            if row['key'].startswith(ADMISSION_PREFIX):gone.extend((RESULT_PREFIX+value['admission_id'],identity_key(value['principal_id'],value['target_session_id'],value['request_id'])))
        conn.execute(sql.SQL('DELETE FROM {} WHERE key=ANY(%s)').format(self.table('state_meta')),(gone,))

    def sweep_orphaned_sessions(self,*,max_idle_seconds,sources=('tui','desktop','subagent'),exclude_ids=(),exclude_pinned=False,heartbeat_staleness_seconds=None,heartbeat_ownership_grace_seconds=None,respect_gateway_heartbeats=True):
        if max_idle_seconds<=0 or not any(sources):return []
        now=time.time();cutoff=now-max_idle_seconds
        staleness=heartbeat_staleness_seconds if heartbeat_staleness_seconds and heartbeat_staleness_seconds>0 else max_idle_seconds*2
        grace=heartbeat_ownership_grace_seconds if heartbeat_ownership_grace_seconds is not None and heartbeat_ownership_grace_seconds>=0 else staleness
        with self._transaction() as conn:
            rows,parents=self._maintenance_snapshot(conn)
            pinned={row['id'] for row in rows if row['pinned']}
            heartbeats=conn.execute(sql.SQL('SELECT started_at,last_heartbeat FROM {} WHERE last_heartbeat>=%s').format(self.table('gateway_heartbeats')),(now-staleness,)).fetchall() if respect_gateway_heartbeats else []
            selected=[row for row in rows if row['ended_at'] is None and row['source'] in sources and row['id'] not in exclude_ids and row['started_at']<cutoff and row['last_active'] is not None and row['last_active']<cutoff]
            ids=[row['id'] for row in selected if not(exclude_pinned and _ancestry([row['id']],parents)&pinned) and not any(h['started_at']<=row['started_at']+grace for h in heartbeats) and not self._guarded_pg_ids(conn,[row['id']])]
            conn.execute(sql.SQL("UPDATE {} SET ended_at=%s,end_reason='startup_orphan_reap' WHERE id=ANY(%s) AND ended_at IS NULL").format(self.table('sessions')),(now,ids))
            return ids

    def maybe_auto_prune_and_vacuum(self,retention_days=90,min_interval_hours=24,vacuum=True,sessions_dir=None,min_vacuum_interval_days=30,min_vacuum_freelist_ratio=0.1):
        from hermes_state import SessionDB
        from hermes_state_maintenance import _seconds_since
        result={'skipped':False,'pruned':0,'closed':0,'vacuumed':False}
        if retention_days is None or retention_days<0:return {**result,'skipped':True}
        try:
            with self._maintenance_admission() as acquired:
                since=_seconds_since(time.time(),self.get_meta('last_auto_prune')) if acquired else 0
                if not acquired or (since is not None and since<min_interval_hours*3600):return {**result,'skipped':True}
                now=time.time();report_startup_progress(900.0,phase='state_db_auto_prune')
                result['pruned']=self.prune_sessions(older_than_days=retention_days,sessions_dir=sessions_dir,exclude_active_write_guards=True)
                report_startup_progress(900.0,phase='state_db_auto_sweep')
                result['closed']=len(self.sweep_orphaned_sessions(max_idle_seconds=float(retention_days)*86400,sources=SessionDB._AUTO_PRUNE_STALE_OPEN_SOURCES,exclude_pinned=True,respect_gateway_heartbeats=False))
                self.set_meta('last_auto_prune',str(now))
        except Exception as exc:
            logger.warning('PostgreSQL auto-prune unavailable: %s',type(exc).__name__);result['error']=type(exc).__name__
        return result
