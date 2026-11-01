"""Owner-scoped managed work, with durable responsibility as lifecycle owner."""
from contextlib import contextmanager
import hashlib
import json
import re
import secrets
from datetime import datetime, timezone
from agents import job_store
from agents.durable_responsibility import DurableResponsibilityStore, ResponsibilityClaim, DurableResponsibilityError, StaleResponsibilityClaim, TERMINAL

_KIND='managed_work.v1'
_ID = re.compile(r'^[A-Za-z0-9_.:-]{1,128}$')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


@contextmanager
def lock(context, work_id):
    """Session lock serializes recoverable handoffs; owner transactions remain separate."""
    c=job_store._connect()
    key='managed-work:'+context.organization_id+':'+context.workspace_id+':'+context.user_id+':'+work_id
    try:
        c.execute('SELECT pg_advisory_lock(hashtextextended(%s,0))',(key,))
        yield
    finally:
        c.close()  # Session closure releases the lock even after a process cut.


class CoreWorkService:
    def __init__(self):
        self.responsibilities=DurableResponsibilityStore(job_store._db_path())

    def _cut(self, stage):
        pass  # Test-only process cut point; production performs no side effect.

    def _load(self, context, work_id):
        row=job_store.get_job(user_id=context.user_id,workspace_id=context.workspace_id,job_id=work_id)
        if row is None or row.get('kind')!=_KIND:
            raise KeyError(work_id)
        with job_store._connect() as c:
            scope=c.execute('SELECT organization_id FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND job_id=%s',(context.user_id,context.workspace_id,work_id)).fetchone()
        if scope is None or scope['organization_id']!=context.organization_id:
            raise KeyError(work_id)
        return row

    def _bind(self, context, row):
        expected=self.responsibilities.derive_id(responsibility_kind=_KIND,workspace_id=context.workspace_id,domain_ref=row['id'],workflow_id=_KIND,workflow_instance_id=context.organization_id+':'+context.user_id+':'+row['id'])
        if row.get('responsibility_id') not in (None,expected):
            raise RuntimeError('managed responsibility binding corrupted')
        if row.get('responsibility_id'):
            responsibility=self.responsibilities.get(expected)
            if responsibility is None:
                raise RuntimeError('bound managed responsibility missing')
            if (responsibility.responsibility_kind != _KIND or responsibility.workspace_id != context.workspace_id or responsibility.domain_ref != row['id'] or responsibility.workflow_id != _KIND or responsibility.workflow_instance_id != context.organization_id+':'+context.user_id+':'+row['id']):
                raise RuntimeError('managed responsibility scope mismatch')
        else:
            responsibility=self.responsibilities.ensure(responsibility_kind=_KIND,workspace_id=context.workspace_id,domain_ref=row['id'],workflow_id=_KIND,workflow_instance_id=context.organization_id+':'+context.user_id+':'+row['id'])
            self._cut('after_responsibility_ensure')
            row=job_store.bind_managed_work(context=context,work_id=row['id'],responsibility_id=responsibility.responsibility_id)
        return row,responsibility

    def create(self, context, data):
        if not isinstance(data,dict) or set(data)!={'request_id','title','input'}:
            raise ValueError('expected request_id, title and input')
        ident=data['request_id']; title=data['title']
        if not isinstance(ident,str) or not _ID.fullmatch(ident) or not isinstance(title,str) or not title.strip() or len(title)>200 or not isinstance(data['input'],dict):
            raise ValueError('invalid work input')
        text=canonical({'title':title,'input':data['input']})
        if len(text.encode())>32768: raise ValueError('work input too large')
        digest=hashlib.sha256(text.encode()).hexdigest()
        with lock(context,ident):
            row,created=job_store.claim_managed_work(context=context,work_id=ident,request_hash=digest,title=title,work_input=data['input'])
            self._cut('after_identity_commit')
            row,r=self._bind(context,row)
            return self.project(row,r),created

    def project(self,row,r):
        status={'WAIT_TIMER':'scheduled','READY':'ready','RUNNING':'running','TERMINAL':('completed' if 'result_hash' in row else 'cancelled')}.get(r.lifecycle_state)
        if status is None: raise RuntimeError('unsupported managed lifecycle')
        if status != 'completed' and 'result_hash' in row:
            raise RuntimeError('managed result without terminal responsibility')
        history=self.responsibilities.events(r.responsibility_id)
        reservation = row.get('execution_reservation')
        if reservation:
            for event in history:
                if event.get('event_type') == 'RESPONSIBILITY_CLAIMED' and event.get('payload', {}).get('lease_epoch') == reservation.get('lease_epoch'):
                    event['payload'].pop('owner_id', None)
        value={'kind':'managed', 'id':row['id'],'title':row['title'],'input':row['input'],'status':status,'checkpoint_version':r.checkpoint_version,'created_at':row['created_at'],'history':history, 'comments':row.get('comments', []), 'references':row.get('references', []), 'accountable_owner':row.get('accountable_owner'), 'responsible_executor':row.get('responsible_executor')}
        if row.get('schedule'):
            value['schedule'] = {k:v for k,v in row['schedule'].items() if k not in {'scope','requests','pending_action'}}
            if status not in {'completed','cancelled'} and not reservation:
                value['status'] = 'paused' if row['schedule'].get('paused') else 'scheduled'
        if reservation:
            value['execution'] = {key: reservation[key] for key in ('run_id', 'lease_epoch', 'state') if key in reservation}
            if 'monitoring' in reservation:
                value['execution']['monitoring'] = reservation['monitoring']
            if reservation.get('state') != 'completed':
                value['execution']['effects'] = 'uncertain'
        if status=='completed':
            if hashlib.sha256(canonical(row.get('result')).encode()).hexdigest()!=row['result_hash'] or not history or history[-1]['event_type']!='RESPONSIBILITY_CHECKPOINT_COMMITTED':
                raise RuntimeError('managed result integrity failure')
            value['result']=row['result']
        if status=='cancelled' and (not history or history[-1]['event_type']!='RESPONSIBILITY_TERMINATED' or history[-1]['payload'].get('reason')!='managed_work_cancelled'):
            raise RuntimeError('managed terminal reason missing')
        return value

    def get(self,context,work_id):
        if not _ID.fullmatch(work_id): raise KeyError(work_id)
        with lock(context,work_id):
            row,r=self._bind(context,self._load(context,work_id))
            return self.project(row,r)

    def list(self,context,limit,cursor):
        if type(limit) is not int or not 1<=limit<=100 or not isinstance(cursor,str) or (cursor and not _ID.fullmatch(cursor)):
            raise ValueError('invalid pagination')
        rows=job_store.list_managed_work(context=context,limit=limit,cursor=cursor)
        items=[self.get(context,row['id']) for row in rows[:limit]]
        return {'items':items,'next_cursor':rows[limit-1]['id'] if len(rows)>limit else None}

    def comment(self, context, work_id, data):
        if (not isinstance(data, dict) or set(data) != {'request_id', 'content', 'expected_version'}
            or not isinstance(data['request_id'], str) or not _ID.fullmatch(data['request_id'])
            or not isinstance(data['content'], str) or not data['content'].strip()
            or len(data['content'].encode()) > 8192 or type(data['expected_version']) is not int
            or data['expected_version'] < 0):
            raise ValueError('invalid comment')
        with lock(context, work_id):
            self._load(context, work_id)
            return job_store.append_managed_comment(context=context, work_id=work_id,
                request_id=data['request_id'], content=data['content'], expected_version=data['expected_version'])

    def reference(self, context, work_id, data):
        if (not isinstance(data, dict) or set(data) != {'request_id', 'kind', 'ref', 'expected_version'}
            or data['kind'] != 'work' or not isinstance(data['request_id'], str)
            or not _ID.fullmatch(data['request_id']) or not isinstance(data['ref'], str)
            or not _ID.fullmatch(data['ref']) or type(data['expected_version']) is not int
            or data['expected_version'] < 0):
            raise ValueError('invalid reference')
        with lock(context, work_id):
            self._load(context, work_id)
            return job_store.append_managed_reference(context=context, work_id=work_id,
                request_id=data['request_id'], ref=data['ref'], expected_version=data['expected_version'])

    def _validate_series(self, context, row, *, reserving=False):
        series_id=row.get('schedule',{}).get('series_id')
        if not series_id:return
        parent=job_store.get_job(user_id=context.user_id,workspace_id=context.workspace_id,job_id=series_id)
        s=(parent or {}).get('schedule',{})
        pending=s.get('pending_action') or {}
        if (not parent or parent.get('kind')!='managed_schedule.v1'
            or s.get('scope',{}).get('organization_id')!=context.organization_id
            or s.get('current_work_id')!=row['id'] or s.get('phase')!='SERIES_ACTIVE'
            or pending.get('data',{}).get('action')=='cancel'
            or (reserving and (s.get('paused') or pending))):
            raise DurableResponsibilityError('recurring schedule changed; refresh')

    def reserve_execution(self, context, work_id, *, run_id, schedule_version=None, skill=None):
        """Reserve at most one native dispatch; lost reservations never redispatch."""
        if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', run_id):
            raise ValueError('invalid execution identity')
        with lock(context, work_id):
            row = self._load(context, work_id)
            schedule = row.get('schedule')
            self._validate_series(context,row,reserving=True)
            if schedule and (schedule_version != schedule['version'] or schedule['phase'] != 'WOKEN' or schedule.get('paused') or schedule.get('pending_action')):
                raise DurableResponsibilityError('use the current scheduled-work action')
            if 'execution_reservation' in row:
                return None
            row, responsibility = self._bind(context, row)
            owner = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
            claim = self.responsibilities.claim(responsibility_id=responsibility.responsibility_id, owner_id=owner, lease_seconds=180)
            self._cut('after_execution_claim')
            if not job_store.reserve_managed_execution(context=context, work_id=work_id, run_id=run_id, claim=claim, skill=skill):
                raise RuntimeError('execution reservation conflict')
            self._cut('after_execution_reservation')
            return claim

    def validate_execution(self, context, work_id, *, claim, run_id):
        """Validate exact scoped reservation and live fence; caller holds work lock."""
        row, responsibility = self._bind(context, self._load(context, work_id))
        reservation = row.get('execution_reservation') or {}
        self._validate_series(context,row)
        if (reservation.get('state') != 'reserved' or reservation.get('run_id') != run_id or
            reservation.get('responsibility_id') != claim.responsibility_id or
            reservation.get('lease_epoch') != claim.lease_epoch or
            reservation.get('checkpoint_version') != claim.checkpoint_version or
            responsibility.responsibility_id != claim.responsibility_id or
            not responsibility.lease_expires_at or
            datetime.fromisoformat(responsibility.lease_expires_at.replace('Z', '+00:00')) <= datetime.now(timezone.utc)):
            raise StaleResponsibilityClaim('execution reservation is stale')
        self.responsibilities.assert_claim(claim)
        return row

    def finish_execution(self, context, work_id, *, claim, run_id, result):
        """Atomically retain a native result and terminal checkpoint under its claim."""
        if not isinstance(result, dict):
            raise ValueError('execution result must be an object')
        text = canonical(result)
        if len(text.encode()) > 3000000:
            raise ValueError('execution result too large')
        digest = hashlib.sha256(text.encode()).hexdigest()
        with lock(context, work_id):
            row, responsibility = self._bind(context, self._load(context, work_id))
            self._validate_series(context,row)
            if responsibility.responsibility_id != claim.responsibility_id:
                raise StaleResponsibilityClaim('execution scope mismatch')
            def retain(conn):
                job_store.store_managed_execution_result(conn, context=context, work_id=work_id,
                    claim=claim, run_id=run_id, result=result, result_hash=digest)
                self._cut('after_execution_result_write')
            responsibility = self.responsibilities.commit(claim, continuation_disposition=TERMINAL,
                lifecycle_state=TERMINAL, transaction_hook=retain, require_unexpired=True)
            self._cut('after_execution_result_commit')
            return self.project(self._load(context, work_id), responsibility)

    def claim(self,context,work_id):
        with lock(context,work_id):
            row,r=self._bind(context,self._load(context,work_id))
            if row.get('schedule') or row['input'].get('conversation_id'): raise DurableResponsibilityError('use the owning schedule or conversation actions')
            token=secrets.token_urlsafe(32)
            owner=hashlib.sha256(token.encode()).hexdigest()
            claim=self.responsibilities.claim(responsibility_id=r.responsibility_id,owner_id=owner,lease_seconds=180)
            return {'id':row['id'],'attempt':token,'epoch':claim.lease_epoch,'checkpoint_version':claim.checkpoint_version,'lease_expires_at':claim.lease_expires_at}

    def complete(self,context,work_id,data):
        if not isinstance(data,dict) or set(data)!={'attempt','result'} or not isinstance(data['attempt'],str) or not data['attempt'] or not isinstance(data['result'],dict):
            raise ValueError('expected attempt and result')
        text=canonical(data['result'])
        if len(text.encode())>32768:raise ValueError('result too large')
        result_hash=hashlib.sha256(text.encode()).hexdigest()
        owner=hashlib.sha256(data['attempt'].encode()).hexdigest()
        with lock(context,work_id):
            row,r=self._bind(context,self._load(context,work_id))
            if row.get('schedule') or row['input'].get('conversation_id'): raise DurableResponsibilityError('work requires its native execution result owner')
            if r.lifecycle_state==TERMINAL:
                if row.get('result_hash')==result_hash and row.get('result_attempt_hash')==owner:
                    return self.project(row,r)
                raise DurableResponsibilityError('terminal work conflict')
            if r.lifecycle_state!='RUNNING' or r.lease_owner!=owner or not r.lease_expires_at or datetime.fromisoformat(r.lease_expires_at.replace('Z','+00:00'))<=datetime.now(timezone.utc):
                raise StaleResponsibilityClaim('stale or foreign work attempt')
            claim=ResponsibilityClaim(r.responsibility_id,owner,r.lease_epoch,r.lease_expires_at,r.checkpoint_version)
            def record(conn):
                job_store.store_managed_result(conn,context=context,work_id=work_id,result=data['result'],result_hash=result_hash,attempt_hash=owner)
                self._cut('after_result_write')
            r=self.responsibilities.commit(claim,continuation_disposition=TERMINAL,lifecycle_state=TERMINAL,transaction_hook=record,require_unexpired=True)
            self._cut('after_result_commit')
            return self.project(self._load(context,work_id),r)

    def cancel(self,context,work_id):
        with lock(context,work_id):
            row,r=self._bind(context,self._load(context,work_id))
            if row.get('schedule'): raise DurableResponsibilityError('use scheduled-work cancellation')
            if r.lifecycle_state==TERMINAL:
                if 'result_hash' in row: raise DurableResponsibilityError('completed work cannot be cancelled')
                return self.project(row,r)
            r=self.responsibilities.terminate(responsibility_id=r.responsibility_id,reason='managed_work_cancelled',expected_lifecycle_state=r.lifecycle_state,expected_checkpoint_version=r.checkpoint_version,cancel_active_claim=True)
            return self.project(row,r)
