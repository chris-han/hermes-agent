"""Once and finite recurrence scheduling over existing jobs, timers and work."""
from types import SimpleNamespace
from datetime import timedelta
from uuid import uuid5, NAMESPACE_URL
import hashlib
import json
from agents import job_store
from agents.durable_responsibility import TIMER, WAIT_TIMER, DETERMINISTIC_CONTINUE, DurableResponsibilityError
from agents.timer_occurrence import TimerOccurrenceStore
from contracts.meeting_delivery import instant, utc
from services.core_work_service import CoreWorkService, canonical, lock, _ID
from services.core_one_shot_timer_service import CoreOneShotTimerProvider, CoreOneShotDueDriver

KIND = 'managed_work.v1'
PURPOSE = 'managed_once_execute'


def live_context(scope):
    from agents.auth_session import background_identity_context
    from agents.task_api import validate_task_context
    return validate_task_context(background_identity_context(**scope))


class CoreScheduledWorkService:
    def __init__(self, executor):
        self.core = CoreWorkService()
        self.responsibilities = self.core.responsibilities
        self.timers = TimerOccurrenceStore(job_store._db_path())
        self.executor = executor
        self.provider = CoreOneShotTimerProvider(self.timers, self._validate_arm)
        self.driver = CoreOneShotDueDriver(self.timers, self.accept_timer, workflow_type=KIND)

    def _cut(self, stage):
        pass

    def _save(self, ctx, ident, s):
        job_store.update_managed_schedule(context=ctx, work_id=ident, schedule=s)

    def _load(self, ctx, ident):
        row, r = self.core._bind(ctx, self.core._load(ctx, ident))
        if not row.get('schedule'): raise KeyError(ident)
        return row, r, row['schedule']

    def create(self, ctx, data):
        if isinstance(data, dict) and 'recurrence' in data:
            return self._create_series(ctx, data)
        return self._create_once(ctx, data)

    def _create_once(self, ctx, data, *, series=None):
        if not isinstance(data, dict) or set(data) != {'request_id','title','input','scheduled_at'}:
            raise ValueError('once work requires request_id, title, input and scheduled_at')
        ident, title, work = data['request_id'], data['title'], data['input']
        due = instant(data['scheduled_at']).isoformat()
        if (not isinstance(ident,str) or not _ID.fullmatch(ident) or not isinstance(title,str)
            or not title.strip() or len(title)>200 or not isinstance(work,dict) or set(work)!={'instruction'}
            or not isinstance(work['instruction'],str) or not work['instruction'].strip()
            or len(work['instruction'].encode())>32768): raise ValueError('invalid one-shot work')
        digest=hashlib.sha256(canonical({'title':title,'input':work,'scheduled_at':due}).encode()).hexdigest()
        if series is not None:
            digest=hashlib.sha256(canonical({'request':digest,'series':series}).encode()).hexdigest()
        scope={k:getattr(ctx,k) for k in ('user_id','workspace_id','workspace_slug','organization_id')}
        schedule={'kind':'once','version':1,'revision':1,'scheduled_at':due,'phase':'ARM_PENDING',
                  'checkpoint':0,'paused':False,'scope':scope,'requests':[]}
        if series is not None: schedule['series_id']=series
        with lock(ctx,ident):
            prior=job_store.get_job(user_id=ctx.user_id,workspace_id=ctx.workspace_id,job_id=ident)
            if prior is None and series is None and instant(due)<=instant(utc()): raise ValueError('choose a future instant')
            row,created=job_store.claim_managed_work(context=ctx,work_id=ident,request_hash=digest,title=title,work_input=work,schedule=schedule)
            self._cut('after_schedule_identity')
            row,r=self.core._bind(ctx,row)
            self._repair(ctx,ident)
            row,r,s=self._load(ctx,ident)
            return self.core.project(row,r),created

    def _repair(self, ctx, ident):
        row,r,s=self._load(ctx,ident)
        if s.get('pending_action'):
            self._apply_action(ctx,ident,row,r,s)
            row,r,s=self._load(ctx,ident)
        if s['phase']=='CANCEL_PENDING':
            if r.lifecycle_state!='TERMINAL':
                self.responsibilities.terminate(responsibility_id=r.responsibility_id,reason='managed_work_cancelled',
                    expected_lifecycle_state=r.lifecycle_state,expected_checkpoint_version=r.checkpoint_version,cancel_active_claim=True)
            if s.get('timer_id'): self.timers.cancel(timer_id=s['timer_id'])
            s['phase']='CANCELLED';self._save(ctx,ident,s);return
        if s['phase']!='ARM_PENDING' or row.get('execution_reservation'): return
        with self.timers._connect() as c:
            timer=self.timers.schedule(c,workflow_type=KIND,workflow_id=r.responsibility_id,
                purpose=PURPOSE,semantic_round=s['revision'],scheduled_at=s['scheduled_at'])
            binding=canonical({'scope':s['scope'],'work_id':ident,'revision':s['revision']})
            c.execute("UPDATE semantier_timer_occurrences SET core_binding_json=%s,core_binding_hash=%s,responsibility_id=%s,expected_checkpoint_version=%s WHERE timer_id=%s AND core_binding_json IS NULL",
                (binding,hashlib.sha256(binding.encode()).hexdigest(),r.responsibility_id,s['checkpoint']+1,timer.timer_id))
        self._cut('after_timer_registration')
        s['timer_id']=timer.timer_id;self._save(ctx,ident,s)
        waits=[w for w in self.responsibilities.pending_waits(r.responsibility_id) if w.kind==TIMER and w.source_ref==timer.timer_id and w.correlation_key==r.responsibility_id and w.expected_checkpoint_version==s['checkpoint']+1]
        if not waits:
            if r.checkpoint_version!=s['checkpoint']: raise DurableResponsibilityError('original schedule checkpoint changed')
            claim=self.responsibilities.claim(responsibility_id=r.responsibility_id,owner_id='schedule-registration',lease_seconds=2)
            self._cut('after_schedule_claim')
            wait=self.responsibilities.register_wait(claim,kind=TIMER,source_ref=timer.timer_id,
                correlation_key=r.responsibility_id,lifecycle_state=WAIT_TIMER,continuation_disposition=DETERMINISTIC_CONTINUE,deadline_at=s['scheduled_at'])
        else: wait=waits[0]
        self._cut('after_schedule_wait')
        s.update(wait_id=wait.wait_id,expected_checkpoint_version=wait.expected_checkpoint_version,phase='WAITING')
        self._save(ctx,ident,s)
        if not s['paused']:self.timers.arm(timer_id=timer.timer_id,provider=self.provider)

    def _timer_row(self, ident):
        with self.timers._connect() as c:
            row=c.execute('SELECT * FROM semantier_timer_occurrences WHERE timer_id=%s',(ident,)).fetchone()
        if not row or row['workflow_type']!=KIND or row['purpose']!=PURPOSE: raise KeyError(ident)
        row=dict(row);raw=row['core_binding_json']
        if not raw or hashlib.sha256(raw.encode()).hexdigest()!=row['core_binding_hash']: raise PermissionError('timer binding corrupted')
        return row,json.loads(raw)

    def _validate_arm(self, timer_id, scheduled_at):
        t,b=self._timer_row(timer_id);ctx=SimpleNamespace(**b['scope'])
        row,r,s=self._load(ctx,b['work_id'])
        if s['timer_id']!=timer_id or s['revision']!=b['revision'] or s['scheduled_at']!=scheduled_at or s['phase']!='WAITING' or s['paused'] or s.get('pending_action'):
            raise DurableResponsibilityError('obsolete timer')

    def _wake(self, ctx, ident, s, *, now=None):
        t,b=self._timer_row(s['timer_id'])
        result=self.responsibilities.accept_wake(responsibility_id=t['responsibility_id'],kind=TIMER,
            source_ref=t['timer_id'],correlation_key=t['workflow_id'],dedupe_key=t['dedupe_key'],
            expected_checkpoint_version=t['expected_checkpoint_version'],observed_at=now or utc())
        self._cut('after_schedule_wake')
        # A duplicate returns its original accepted outcome through the persisted receipt.
        with self.timers._connect() as c:
            prior=c.execute('SELECT accepted FROM semantier_responsibility_wakes WHERE dedupe_key=%s',(t['dedupe_key'],)).fetchone()
        if not prior or not prior['accepted']: raise DurableResponsibilityError('schedule wait replaced')
        self.timers.mark_accepted(timer_id=t['timer_id'])
        s['phase']='WOKEN';self._save(ctx,ident,s)
        return result

    def accept_timer(self,timer_id,*,now=None):
        t,b=self._timer_row(timer_id);ctx=SimpleNamespace(**b['scope']);ident=b['work_id']
        with lock(ctx,ident):
            row,r,s=self._load(ctx,ident)
            if (s.get('timer_id')!=timer_id or s['revision']!=b['revision'] or s['phase'] not in {'WAITING','WOKEN'} or s['paused'] or s.get('pending_action') or r.lifecycle_state=='TERMINAL' or row.get('execution_reservation')):return False
            if instant(now or utc())<instant(s['scheduled_at']):return False
            self._wake(ctx,ident,s,now=now);return True

    def get(self,ctx,ident):
        row=job_store.get_job(user_id=ctx.user_id,workspace_id=ctx.workspace_id,job_id=ident)
        if row and row.get('kind')=='managed_schedule.v1':
            with lock(ctx,ident):
                row,s=self._series_load(ctx,ident)
                return self._series_project(ctx,row,s)
        with lock(ctx,ident):
            self._repair(ctx,ident);row,r,s=self._load(ctx,ident)
            return self.core.project(row,r)

    def list(self,ctx):
        rows=job_store.scheduled_work_rows(context=ctx,limit=100)
        rows=[row for row in rows if not json.loads(row['payload_json'])['schedule'].get('series_id')]
        rows+=job_store.scheduled_work_rows(context=ctx,limit=100,kind='managed_schedule.v1')
        return {'items':[self.get(ctx,row['job_id']) for row in rows]}

    def action(self,ctx,ident,data):
        row=job_store.get_job(user_id=ctx.user_id,workspace_id=ctx.workspace_id,job_id=ident)
        if row and row.get('kind')=='managed_schedule.v1': return self._series_action(ctx,ident,data)
        if row and row.get('schedule',{}).get('series_id'):
            raise DurableResponsibilityError('use the recurring schedule action')
        return self._action_once(ctx,ident,data)

    def _action_once(self,ctx,ident,data):
        required={'request_id','expected_version','action'}
        if (not isinstance(data,dict) or not required<=set(data) or set(data)-required-{'scheduled_at'}
            or not isinstance(data['request_id'],str) or not _ID.fullmatch(data['request_id'])
            or type(data['expected_version']) is not int):raise ValueError('invalid schedule action')
        action=data['action']
        if action not in {'pause','resume','reschedule','run_now','cancel'} or ('scheduled_at' in data)!=(action=='reschedule'):
            raise ValueError('unsupported one-shot action')
        digest=hashlib.sha256(canonical(data).encode()).hexdigest()
        with lock(ctx,ident):
            self._repair(ctx,ident);row,r,s=self._load(ctx,ident)
            old=next((x for x in s['requests'] if x['request_id']==data['request_id']),None)
            if old:
                if old['hash']!=digest:raise DurableResponsibilityError('schedule request conflict')
                return self.core.project(row,r)
            if s['version']!=data['expected_version']:raise DurableResponsibilityError('schedule version changed; refresh')
            if r.lifecycle_state=='TERMINAL' or (row.get('execution_reservation') and action!='cancel'):
                raise DurableResponsibilityError('reserved or terminal work cannot be scheduled again')
            if len(s['requests'])>=100:raise ValueError('schedule action limit reached')
            if action=='reschedule' and instant(data['scheduled_at'])<=instant(utc()):raise ValueError('choose a future instant')
            s['pending_action']={'data':data,'hash':digest}
            self._save(ctx,ident,s);self._cut('after_schedule_action_intent')
            self._repair(ctx,ident);row,r,s=self._load(ctx,ident);result=self.core.project(row,r)
        if action=='cancel' and row.get('execution_reservation'):
            result['stop_confirmed']=self.executor.stop(ctx,ident)
        return result

    def _apply_action(self,ctx,ident,row,r,s):
        pending=s['pending_action'];data=pending['data'];action=data['action']
        if action=='reschedule':
            due=instant(data['scheduled_at']).isoformat()
            if s['phase']=='WAITING':self._wake(ctx,ident,s)
            self.timers.cancel(timer_id=s['timer_id'])
            r=self.responsibilities.get(r.responsibility_id)
            if r.lifecycle_state!='READY':raise DurableResponsibilityError('schedule continuation already claimed')
            s.update(revision=s['revision']+1,scheduled_at=due,phase='ARM_PENDING',checkpoint=r.checkpoint_version)
            s.pop('timer_id',None);s.pop('wait_id',None)
        elif action=='pause':
            s['paused']=True
            with self.timers._connect() as c:
                c.execute("UPDATE semantier_timer_occurrences SET provider_ref=NULL WHERE timer_id=%s AND state='SCHEDULED'",(s['timer_id'],))
        elif action=='resume':s['paused']=False
        elif action=='run_now':
            s['paused']=False
            if s['phase']=='WAITING':self._wake(ctx,ident,s)
        else:s['phase']='CANCEL_PENDING'
        s['version']+=1;s['requests'].append({'request_id':data['request_id'],'hash':pending['hash'],'action':action,'at':utc(),'version':s['version']})
        del s['pending_action'];s.pop('dispatch_error',None)
        self._save(ctx,ident,s);self._cut('after_schedule_action')
        if s['phase']=='WAITING' and not s['paused']:
            self.timers.arm(timer_id=s['timer_id'],provider=self.provider)

    def recover(self):
        for row in job_store.scheduled_work_rows(kind='managed_schedule.v1',phases=('SERIES_ACTIVE',),limit=32):
            data=json.loads(row['payload_json']);ctx=SimpleNamespace(**data['schedule']['scope'])
            try:
                with lock(ctx,row['job_id']): self._series_repair(ctx,row['job_id'])
            except DurableResponsibilityError: continue
        for row in job_store.scheduled_work_rows(phases=('ARM_PENDING','CANCEL_PENDING'),pending_actions=True,limit=32):
            s=json.loads(row['payload_json'])['schedule'];ctx=SimpleNamespace(**s['scope'])
            try:
                with lock(ctx,row['job_id']):self._repair(ctx,row['job_id'])
            except DurableResponsibilityError:continue  # registration lease can expire; no native reservation is adopted.
        # Recover a wait-committed/arm-not-yet-persisted cut through the same provider.
        for row in job_store.scheduled_work_rows(phases=('WAITING',),unarmed=True,limit=32):
            s=json.loads(row['payload_json'])['schedule']
            ctx=SimpleNamespace(**s['scope'])
            with lock(ctx,row['job_id']):
                current,r,s=self._load(ctx,row['job_id'])
                if s['phase']=='WAITING' and not s['paused'] and not s.get('pending_action'):
                    self.timers.arm(timer_id=s['timer_id'],provider=self.provider)

    def dispatch_pending(self):
        for row in job_store.scheduled_work_rows(phases=('WOKEN',),limit=32):
            s=json.loads(row['payload_json'])['schedule']
            if s['paused'] or s.get('pending_action'):continue
            try:
                ctx=live_context(s['scope'])
                if s.get('series_id'):
                    with lock(ctx,s['series_id']):
                        root,rs=self._series_load(ctx,s['series_id'])
                        if rs['phase']!='SERIES_ACTIVE' or rs['paused'] or rs.get('pending_action') or rs['current_work_id']!=row['job_id']:continue
                        self.executor.submit(ctx,row['job_id'],resolve_context=lambda scope=s['scope']:live_context(scope),schedule_version=s['version'])
                else:
                    self.executor.submit(ctx,row['job_id'],resolve_context=lambda scope=s['scope']:live_context(scope),schedule_version=s['version'])
            except Exception:
                # No invented authority or automatic replay of a reserved execution.
                # Retain the failure on the same schedule for truthful UI recovery.
                ctx=SimpleNamespace(**s['scope'])
                with lock(ctx,row['job_id']):
                    current,r,s=self._load(ctx,row['job_id'])
                    if not current.get('execution_reservation'):
                        s['dispatch_error']='Current execution authority or service unavailable'
                        self._save(ctx,row['job_id'],s)

    def _series_load(self,ctx,ident):
        row=job_store.get_job(user_id=ctx.user_id,workspace_id=ctx.workspace_id,job_id=ident)
        if not row or row.get('kind')!='managed_schedule.v1': raise KeyError(ident)
        s=row['schedule']
        if any(s['scope'][k]!=getattr(ctx,k) for k in ('user_id','workspace_id','organization_id')): raise KeyError(ident)
        return row,s

    def _create_series(self,ctx,data):
        if set(data)!={'request_id','title','input','scheduled_at','recurrence'}:
            raise ValueError('unsupported recurring work fields')
        recurrence=data['recurrence']
        if (not isinstance(recurrence,dict) or set(recurrence)!={'interval_seconds','times','timezone'}
            or type(recurrence['interval_seconds']) is not int or not 60<=recurrence['interval_seconds']<=31536000
            or type(recurrence['times']) is not int or not 2<=recurrence['times']<=100 or recurrence['timezone']!='UTC'):
            raise ValueError('recurrence requires UTC interval_seconds 60..31536000 and times 2..100')
        ident,title,work=data['request_id'],data['title'],data['input']
        due=instant(data['scheduled_at']).isoformat()
        if (not isinstance(ident,str) or not _ID.fullmatch(ident) or not isinstance(title,str)
            or not title.strip() or len(title)>200 or not isinstance(work,dict) or set(work)!={'instruction'}
            or not isinstance(work['instruction'],str) or not work['instruction'].strip()
            or len(work['instruction'].encode())>32768): raise ValueError('invalid recurring work')
        digest=hashlib.sha256(canonical({**data,'scheduled_at':due}).encode()).hexdigest()
        scope={k:getattr(ctx,k) for k in ('user_id','workspace_id','workspace_slug','organization_id')}
        with lock(ctx,ident):
            prior=job_store.get_job(user_id=ctx.user_id,workspace_id=ctx.workspace_id,job_id=ident)
            if prior is None and instant(due)<=instant(utc()): raise ValueError('choose a future instant')
            child=self._occurrence_id(ctx,ident,1,1)
            s={'kind':'interval','version':1,'revision':1,'scheduled_at':due,'anchor':due,
               'timezone':'UTC','interval_seconds':recurrence['interval_seconds'],
               'repeat':{'times':recurrence['times'],'completed':0},'phase':'SERIES_ACTIVE',
               'paused':False,'scope':scope,'requests':[],'occurrences':[child],
               'current_work_id':child,'skipped_slots':0}
            row,created=job_store.claim_managed_work(context=ctx,work_id=ident,request_hash=digest,
                title=title,work_input=work,schedule=s,kind='managed_schedule.v1')
            self._cut('after_series_identity')
            self._series_repair(ctx,ident)
            row,s=self._series_load(ctx,ident)
            return self._series_project(ctx,row,s),created

    @staticmethod
    def _occurrence_id(ctx,ident,revision,ordinal):
        return 'occ-'+uuid5(NAMESPACE_URL,canonical([ctx.organization_id,ctx.workspace_id,ctx.user_id,ident,revision,ordinal])).hex

    def _series_project(self,ctx,row,s):
        items=[]
        for child in s['occurrences']:
            value=job_store.get_job(user_id=ctx.user_id,workspace_id=ctx.workspace_id,job_id=child)
            if value:
                value,r=self.core._bind(ctx,self.core._load(ctx,child))
                items.append(self.core.project(value,r))
        current=items[-1] if items and items[-1]['id']==s['current_work_id'] else None
        result=dict(current or {'title':row['title'],'status':'scheduled','history':[]})
        result.update(id=row['id'],kind='managed_schedule.v1',occurrences=items,
            schedule={k:v for k,v in s.items() if k not in {'scope','requests','pending_action','occurrences'}})
        if s['phase']=='SERIES_COMPLETE':result['status']='completed'
        elif s['phase']=='SERIES_CANCELLED':result['status']='cancelled'
        elif s['paused']:result['status']='paused'
        elif current and current['status']=='completed':result['status']='scheduled'
        if current and current.get('execution',{}).get('effects')=='uncertain':
            result['schedule']['continuation']='blocked_until_current_outcome_known'
        return result

    def _series_repair(self,ctx,ident):
        row,s=self._series_load(ctx,ident)
        if s.get('pending_action'):
            self._apply_series_action(ctx,ident,row,s)
            row,s=self._series_load(ctx,ident)
        if s['phase']!='SERIES_ACTIVE': return
        child=s['current_work_id']
        current=job_store.get_job(user_id=ctx.user_id,workspace_id=ctx.workspace_id,job_id=child)
        if current is None:
            self._cut('before_series_child')
            self._create_once(ctx,{'request_id':child,'title':row['title'],'input':row['input'],
                'scheduled_at':s['scheduled_at']},series=ident)
            self._cut('after_series_child')
            current=self.core._load(ctx,child)
        else:
            with lock(ctx,child):self._repair(ctx,child)
            current=self.core._load(ctx,child)
        if current.get('result_hash'):
            # Core projection verifies result hash and terminal history before a successor exists.
            current,r=self.core._bind(ctx,current);self.core.project(current,r)
            s['repeat']['completed']=len(s['occurrences'])
            if s['repeat']['completed']==s['repeat']['times']:
                s['phase']='SERIES_COMPLETE';s['version']+=1;self._save(ctx,ident,s);return
            if s['paused']:
                self._save(ctx,ident,s);return
            interval=s['interval_seconds'];anchor=instant(s['anchor']);now=instant(utc())
            previous=instant(s['scheduled_at'])
            next_due=max(anchor,previous+timedelta(seconds=interval))
            if next_due<=now:
                skipped=int((now-next_due).total_seconds()//interval)+1
                s['skipped_slots']+=skipped;next_due+=timedelta(seconds=skipped*interval)
            ordinal=len(s['occurrences'])+1
            child=self._occurrence_id(ctx,ident,s['revision'],ordinal)
            s.update(current_work_id=child,scheduled_at=next_due.isoformat(),version=s['version']+1)
            s['occurrences'].append(child)
            self._save(ctx,ident,s);self._cut('after_series_next_intent')
            self._series_repair(ctx,ident)

    def _series_action(self,ctx,ident,data):
        required={'request_id','expected_version','action'}
        if (not isinstance(data,dict) or not required<=set(data) or set(data)-required-{'scheduled_at'}
            or not isinstance(data['request_id'],str) or not _ID.fullmatch(data['request_id'])
            or type(data['expected_version']) is not int):raise ValueError('invalid schedule action')
        action=data['action']
        if action not in {'pause','resume','reschedule','run_now','cancel'} or ('scheduled_at' in data)!=(action=='reschedule'):
            raise ValueError('unsupported recurring action')
        digest=hashlib.sha256(canonical(data).encode()).hexdigest()
        with lock(ctx,ident):
            self._series_repair(ctx,ident);row,s=self._series_load(ctx,ident)
            prior=next((x for x in s['requests'] if x['request_id']==data['request_id']),None)
            if prior:
                if prior['hash']!=digest:raise DurableResponsibilityError('schedule request conflict')
                return self._series_project(ctx,row,s)
            if data['expected_version']!=s['version']:raise DurableResponsibilityError('schedule version changed; refresh')
            if s['phase']!='SERIES_ACTIVE':raise DurableResponsibilityError('terminal series cannot run again')
            if len(s['requests'])>=100:raise ValueError('schedule action limit reached')
            child=self.core._load(ctx,s['current_work_id'])
            if action in {'reschedule','run_now'} and child.get('execution_reservation'):
                raise DurableResponsibilityError('reserved occurrence cannot be scheduled again')
            if action=='reschedule' and instant(data['scheduled_at'])<=instant(utc()):raise ValueError('choose a future instant')
            s['pending_action']={'data':data,'hash':digest,'child':s['current_work_id'],
                                 'child_version':child['schedule']['version']}
            self._save(ctx,ident,s);self._cut('after_series_action_intent')
            self._series_repair(ctx,ident);row,s=self._series_load(ctx,ident)
            return self._series_project(ctx,row,s)

    def _apply_series_action(self,ctx,ident,row,s):
        pending=s['pending_action'];data=pending['data'];action=data['action']
        child=self.core._load(ctx,pending['child'])
        # Pause/resume changes future eligibility while an already reserved occurrence may finish.
        should_apply=not child.get('execution_reservation') or action=='cancel'
        if child.get('result_hash'):should_apply=False
        if should_apply:
            child_data={'request_id':'series-action-'+hashlib.sha256(canonical([ident,data['request_id']]).encode()).hexdigest(),
                        'expected_version':pending['child_version'],'action':action}
            if action=='reschedule':child_data['scheduled_at']=data['scheduled_at']
            self._action_once(ctx,pending['child'],child_data)
        self._cut('after_series_child_action')
        if action=='cancel':s['phase']='SERIES_CANCELLED'
        elif action=='pause':s['paused']=True
        elif action in {'resume','run_now'}:s['paused']=False
        elif action=='reschedule':
            s['revision']+=1;s['anchor']=instant(data['scheduled_at']).isoformat();s['scheduled_at']=s['anchor']
        s['version']+=1
        s['requests'].append({'request_id':data['request_id'],'hash':pending['hash'],'action':action,'at':utc(),'version':s['version']})
        del s['pending_action'];self._save(ctx,ident,s)
        self._cut('after_series_action')
