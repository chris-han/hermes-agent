"""Native PG cron ledger. No schedule dispatch occurs in storage or migration.

Foreign-host and imported live ownership is indeterminate until a declared recovery
owner settles it. A local PID cannot prove that a remote PG writer has died.
"""
from contextlib import contextmanager
from datetime import datetime,UTC
import os
from pathlib import Path
import socket
import time
import uuid
from psycopg import sql

from hermes_state_backend import open_session_db


def _store(module):
    from hermes_constants import get_hermes_home
    override=getattr(module,'EXECUTIONS_FILE',None)
    home=Path(override).parent.parent if override else get_hermes_home()
    return open_session_db(home=home)


def _now():return datetime.now(UTC).isoformat()


@contextmanager
def _transaction(module):
    with _store(module) as store,store._transaction() as conn:
        store._lock_conversation(conn,'__cron_ledger__')
        yield store,conn


def _fetch(store,conn,execution_id):
    return conn.execute(sql.SQL('SELECT * FROM {} WHERE id=%s FOR UPDATE').format(store.table('executions')),(str(execution_id),)).fetchone()


def _local_owner(module,row):
    return row['owner_host']==socket.gethostname() and row['process_id']==module._PROCESS_ID and row['pid']==os.getpid()


def _live(module,row):
    if row.get('owner_host')!=socket.gethostname():return True
    return module._row_owner_live(row)


def _prune(module,store,conn):
    conn.execute(sql.SQL("DELETE FROM {} WHERE id IN(SELECT id FROM {} WHERE status IN('completed','failed','unknown') ORDER BY finished_at::timestamptz DESC,finished_at DESC,claimed_at::timestamptz DESC,claimed_at DESC,id DESC OFFSET %s)").format(store.table('executions'),store.table('executions')),(max(0,int(module.MAX_TERMINAL_EXECUTIONS)),))


def create_execution(module,job_id,*,source,scheduled_instant=None):
    from cron.occurrences import scheduled_instant as canonical
    values=dict(id=uuid.uuid4().hex,job_id=str(job_id),source=str(source),process_id=module._PROCESS_ID,pid=os.getpid(),process_started_at=module._process_start_time(os.getpid()),status='claimed',claimed_at=_now(),scheduled_instant=canonical(scheduled_instant),owner_host=socket.gethostname())
    with _transaction(module) as (store,conn):
        store._insert(conn,'executions',values);record=_fetch(store,conn,values['id'])
    module._emit_execution_state(record);return record


def _transition(module,execution_id,kind,**kwargs):
    with _transaction(module) as (store,conn):
        row=_fetch(store,conn,execution_id)
        if not row:return None
        local=_local_owner(module,row)
        inflight=row['status'] in ('claimed','running')
        changes={}
        if kind=='running':
            if not local or row['status']!='claimed' or row['handoff_pending']:return None
            changes=dict(status='running',started_at=_now(),handoff_pending=0,handoff_started_at=None)
        elif kind=='handoff':
            if not local or row['status']!='claimed':return None
            changes=dict(handoff_pending=1,handoff_started_at=time.time())
        elif kind=='adopt':
            if row['status']!='claimed' or not row['handoff_pending']:return None
            changes=dict(process_id=module._PROCESS_ID,pid=os.getpid(),process_started_at=module._process_start_time(os.getpid()),owner_host=socket.gethostname(),status='running',started_at=_now(),handoff_pending=0,handoff_started_at=None)
        elif kind in ('finish','give_up'):
            if not local or not inflight:return None
            success=kwargs.get('success')
            changes=dict(status=('completed' if success else 'failed') if kind=='finish' else 'unknown',finished_at=_now(),error=(None if success else kwargs.get('error') or 'unknown failure') if kind=='finish' else kwargs['reason'],handoff_pending=0,handoff_started_at=None)
            if kind=='finish':changes['delivery_outcome']=kwargs.get('delivery_outcome')
        elif kind=='receipt':
            if row['job_id']!=str(kwargs['job_id']) or row['status'] not in ('claimed','running','unknown') or _live(module,row):return None
            changes=dict(status='completed' if kwargs['success'] else 'failed',finished_at=_now(),error=None if kwargs['success'] else kwargs.get('error') or 'unknown failure',handoff_pending=0,handoff_started_at=None,delivery_outcome=kwargs.get('delivery_outcome'))
        elif kind=='dead':
            if local or not inflight or row.get('owner_host')!=socket.gethostname() or module._owner_is_live(int(row['pid']),row['process_started_at']):return None
            if row['handoff_pending'] and row['handoff_started_at'] is not None and time.time()-row['handoff_started_at']<module.HANDOFF_ADOPTION_GRACE_SECONDS:return None
            changes=dict(status='unknown',finished_at=_now(),error=kwargs['reason'],handoff_pending=0,handoff_started_at=None)
        else:raise ValueError(kind)
        store._update(conn,'executions',changes,'id',row['id']);record=_fetch(store,conn,row['id'])
        if record['status'] in ('completed','failed','unknown'):_prune(module,store,conn)
    module._emit_execution_state(record,delivery_outcome=kwargs.get('delivery_outcome'))
    if kind in ('finish','receipt'):module.record_cron_finish(record,kwargs.get('delivery_outcome'))
    return record


def touch_execution_progress(module,execution_id):
    with _transaction(module) as (store,conn):
        row=_fetch(store,conn,execution_id)
        if not row or not _local_owner(module,row) or row['status'] not in ('claimed','running'):return False
        store._update(conn,'executions',{'progress_at':_now()},'id',execution_id);return True


def set_execution_occurrence(module,execution_id,instant):
    from cron.occurrences import scheduled_instant
    with _transaction(module) as (store,conn):
        row=_fetch(store,conn,execution_id)
        if not row or not _local_owner(module,row) or row['status']!='claimed' or row['handoff_pending']:raise RuntimeError('Cron occurrence could not be bound before dispatch')
        store._update(conn,'executions',{'scheduled_instant':scheduled_instant(instant)},'id',execution_id)


def mark_execution_running(module,execution_id):return _transition(module,execution_id,'running')
def mark_execution_handoff_pending(module,execution_id):return _transition(module,execution_id,'handoff')
def adopt_claimed_execution(module,execution_id):return _transition(module,execution_id,'adopt')
def finish_execution(module,execution_id,*,success,error=None,delivery_outcome=None):return _transition(module,execution_id,'finish',success=success,error=error,delivery_outcome=delivery_outcome)
def give_up_execution(module,execution_id,*,reason):return _transition(module,execution_id,'give_up',reason=reason) is not None
def terminalize_dead_owner(module,execution_id,*,reason):return _transition(module,execution_id,'dead',reason=reason) is not None
def recover_receipted_execution(module,execution_id,job_id,*,success,error=None,delivery_outcome=None):return _transition(module,execution_id,'receipt',job_id=job_id,success=success,error=error,delivery_outcome=delivery_outcome)


def execution_owner_live(module,execution_id,job_id):
    with _transaction(module) as (store,conn):
        row=_fetch(store,conn,execution_id)
        return bool(row and row['job_id']==job_id and row['status'] in ('claimed','running','unknown') and _live(module,row))


def settle_delivery_outcome(module,execution_id,outcome):
    with _transaction(module) as (store,conn):
        return conn.execute(sql.SQL("UPDATE {} SET delivery_outcome=%s WHERE id=%s AND delivery_outcome='queued' AND status IN('completed','failed','unknown')").format(store.table('executions')),(outcome,execution_id)).rowcount==1


def recover_interrupted_executions(module):
    changed=0
    with _transaction(module) as (store,conn):
        rows=conn.execute(sql.SQL("SELECT * FROM {} WHERE status IN('claimed','running') FOR UPDATE").format(store.table('executions'))).fetchall()
        for row in rows:
            if row.get('owner_host')!=socket.gethostname() or _local_owner(module,row):continue
            reason=module._OWNER_GONE_REASON
            if module._owner_is_live(int(row['pid']),row['process_started_at']):
                bound=module._live_owner_stale_after_seconds()
                if bound is None or module._stale_age_seconds(row['claimed_at'],row['progress_at'])<=bound:continue
                reason=module._OWNER_WEDGED_REASON
            if row['handoff_pending'] and row['handoff_started_at'] is not None and time.time()-row['handoff_started_at']<module.HANDOFF_ADOPTION_GRACE_SECONDS:continue
            store._update(conn,'executions',dict(status='unknown',finished_at=_now(),error=reason,handoff_pending=0,handoff_started_at=None),'id',row['id']);changed+=1
        if changed:_prune(module,store,conn)
    return changed


def list_executions(module,*,job_id=None,limit=50,before_claimed_at=None):
    where,params=[],[]
    if job_id is not None:where.append(sql.SQL('job_id=%s'));params.append(str(job_id))
    if before_claimed_at is not None:where.append(sql.SQL('(claimed_at::timestamptz,claimed_at)<(%s::timestamptz,%s)'));params.extend([str(before_claimed_at)]*2)
    clause=sql.SQL(' WHERE ')+sql.SQL(' AND ').join(where) if where else sql.SQL('')
    with _store(module) as store:
        return store._all(sql.SQL('SELECT * FROM {}{} ORDER BY claimed_at::timestamptz DESC,claimed_at DESC,id DESC LIMIT %s').format(store.table('executions'),clause),[*params,max(1,min(int(limit),500))])


def get_execution(module,execution_id):
    with _store(module) as store:return store._one(sql.SQL('SELECT * FROM {} WHERE id=%s').format(store.table('executions')),(str(execution_id),))


def latest_executions(module,job_ids):
    clean=list(dict.fromkeys(str(x) for x in job_ids if x))
    if not clean:return {}
    with _store(module) as store:
        rows=store._all(sql.SQL('SELECT DISTINCT ON(job_id) * FROM {} WHERE job_id=ANY(%s) ORDER BY job_id,claimed_at::timestamptz DESC,claimed_at DESC,id DESC').format(store.table('executions')),(clean,))
        return {row['job_id']:row for row in rows}


EXECUTION_METHODS=('create_execution','touch_execution_progress','set_execution_occurrence','mark_execution_running',
 'mark_execution_handoff_pending','adopt_claimed_execution','finish_execution','give_up_execution','terminalize_dead_owner',
 'recover_receipted_execution','execution_owner_live','settle_delivery_outcome','recover_interrupted_executions',
 'list_executions','get_execution','latest_executions')


def dispatch_operation(module,name,*args,**kwargs):
    return globals()[name](module,*args,**kwargs)


def upsert_incident(module,job_id,error,*,job_name=None,failure_type=None,output_file=None):
    job_id=str(job_id or '');sig=module._error_signature(job_id,error)
    identity=module._incident_id(job_id,sig);now=_now()
    with _transaction(module) as (store,conn):
        row=conn.execute(sql.SQL('SELECT state FROM {} WHERE id=%s FOR UPDATE').format(store.table('cron_incidents')),(identity,)).fetchone()
        if row:
            changes=dict(last_seen_at=now,error=module._redact_error(error),output_file=str(output_file) if output_file is not None else None)
            if row['state']=='resolved':changes.update(state='detected',alerted_at=None,closed_at=None)
            store._update(conn,'cron_incidents',changes,'id',identity)
            return identity,row['state']=='resolved'
        store._insert(conn,'cron_incidents',dict(id=identity,job_id=job_id,error_sig=sig,state='detected',failure_type=failure_type or module._classify_failure_type(error),first_seen_at=now,last_seen_at=now,error=module._redact_error(error),output_file=str(output_file) if output_file is not None else None))
        return identity,True


def set_incident_state(module,incident_id,state):
    if state not in module.INCIDENT_STATES:return False
    with _transaction(module) as (store,conn):
        row=conn.execute(sql.SQL('SELECT state FROM {} WHERE id=%s FOR UPDATE').format(store.table('cron_incidents')),(incident_id,)).fetchone()
        if row is None or row['state']=='closed':return False
        changes={'state':state}
        if state=='alerted':changes['alerted_at']=_now()
        elif state==row['state']:return False
        elif state=='closed':changes.update(closed_at=_now(),acked_at=_now())
        store._update(conn,'cron_incidents',changes,'id',incident_id);return True


def close_incidents_for_recovered_job(module,job_id):
    with _transaction(module) as (store,conn):
        return conn.execute(sql.SQL("UPDATE {} SET state='resolved',closed_at=%s WHERE job_id=%s AND state IN('detected','alerted')").format(store.table('cron_incidents')),(_now(),str(job_id or ''))).rowcount


def list_incidents(module,state=None):
    if state is not None and state not in module.INCIDENT_STATES:return []
    with _store(module) as store:
        where=sql.SQL(' WHERE state=%s') if state is not None else sql.SQL('')
        return store._all(sql.SQL('SELECT * FROM {}{} ORDER BY last_seen_at::timestamptz DESC,last_seen_at DESC,id DESC').format(store.table('cron_incidents'),where),(state,) if state is not None else ())


def get_incident(module,incident_id):
    with _store(module) as store:return store._one(sql.SQL('SELECT * FROM {} WHERE id=%s').format(store.table('cron_incidents')),(incident_id,))


def count_incidents(module,state=None):
    if state is not None and state not in module.INCIDENT_STATES:return 0
    with _store(module) as store:
        where=sql.SQL(' WHERE state=%s') if state is not None else sql.SQL('')
        return store._one(sql.SQL('SELECT COUNT(*) AS n FROM {}{}').format(store.table('cron_incidents'),where),(state,) if state is not None else ())['n']
