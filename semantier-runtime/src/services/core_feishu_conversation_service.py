"""One signed text DM channel over existing conversations, jobs, inbox and outbox."""
import hashlib
import json
import logging
import os
from uuid import uuid4
from agents import job_store
from agents.durable_event_arbitration import DurableEventInbox,WorkflowVersion
from agents.effect_outbox import AtomicEffectOutbox,READY,DISPATCHING,COMPLETED,EFFECT_UNKNOWN,FAILED_SAFE
from agents.feishu_ingress_identity import resolve_feishu_ingress_owner
from contracts.meeting_delivery import bot_generation
from services.core_conversation_service import CoreConversationService
from services.core_scheduled_work_service import live_context
from services.core_work_service import canonical,lock
from services.core_feishu_transport import CoreFeishuTransport,KnownNotSent

KIND='core_feishu_channel.v1'
TERMINAL={'COMPLETED','CANCELLED','EXECUTION_UNCERTAIN',EFFECT_UNKNOWN,FAILED_SAFE}
log=logging.getLogger(__name__)


def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


class CoreFeishuConversationService:
    def __init__(self,executor,*,transport_factory=None):
        self.executor=executor
        self.conversations=CoreConversationService()
        self.inbox=DurableEventInbox(job_store._db_path())
        self.outbox=AtomicEffectOutbox(job_store._db_path())
        self.transport_factory=transport_factory or (lambda bot:CoreFeishuTransport(bot))

    def _cut(self,stage):
        pass

    def configuration(self,ctx):
        from agents import auth_db
        ctx=live_context({k:getattr(ctx,k) for k in ('user_id','workspace_id','workspace_slug','organization_id')})
        if ctx.member_role not in {'owner','admin'}:raise PermissionError('workspace administrator required')
        bot=auth_db.get_feishu_bot_config_for_workspace(ctx.workspace_id) or {}
        if bot and bot.get('owner_user_id')!=ctx.user_id:raise PermissionError('current bot owner required')
        configured=bool(bot.get('core_channel_id') and bot.get('connection_mode')=='core_webhook')
        account=bot.get('core_channel_id') if configured else None
        return {'configured':configured,'account_id':account,'mode':'core_webhook',
                'app_id':bot.get('app_id',''),'domain':bot.get('domain','feishu'),
                'bot_name':bot.get('bot_name',''),'validated_at':bot.get('validated_at'),
                'label':bot.get('label',''),'revision':bot.get('configuration_revision',0),
                'events_path':'/api/channels/feishu/'+account+'/events' if account else None,
                'capabilities':{'text_dm':True,'group':False,'attachments':False,'native_qr':False,'credential_rotation':False}}

    def configure(self,ctx,body,*,probe=None):
        from agents import auth_db,gateway_identity
        self.configuration(ctx)
        keys={'app_id','app_secret','domain','tenant_key','encrypt_key','verification_token'}
        if not isinstance(body,dict) or set(body)!=keys|{'label','expected_revision'}:raise ValueError('Exact scoped webhook configuration required')
        if any(not isinstance(body[k],str) or not body[k].strip() or len(body[k])>2048 for k in keys):raise ValueError('Bounded nonempty configuration required')
        if not isinstance(body['label'],str) or len(body['label'].strip())>120 or type(body['expected_revision']) is not int or body['expected_revision']<0:
            raise ValueError('Bounded label and current revision required')
        label=body['label'].strip();expected=body['expected_revision']
        bot={k:body[k].strip() for k in keys}
        if bot['domain'] not in {'feishu','lark'}:raise ValueError('Feishu or Lark domain required')
        current=auth_db.get_feishu_bot_config_for_workspace(ctx.workspace_id) or {}
        if current.get('core_channel_id') and any(current.get(k)!=bot[k] for k in keys):
            raise ValueError('Changing active channel credentials is unavailable; exact revalidation is supported')
        revision=current.get('configuration_revision',0)
        if expected!=revision and not (expected+1==revision and label==current.get('label','')):
            raise ValueError('Configuration revision changed; refresh before revalidation')
        if probe is None:
            from services.core_feishu_transport import probe_bot
            probe=probe_bot
        info=probe(bot)  # No authority lock held over the external SDK call.
        if not info or not str(info.get('bot_open_id','')).startswith('ou_'):raise PermissionError('Current SDK bot validation required')
        with gateway_identity.channel_identity_transaction():
            self.configuration(ctx)  # Current membership/grants revalidated after network.
            current=auth_db.get_feishu_bot_config_for_workspace(ctx.workspace_id) or {}
            if current.get('core_channel_id'):
                if any(current.get(k)!=bot[k] for k in keys) or current.get('bot_open_id')!=info['bot_open_id']:
                    raise ValueError('Active channel configuration changed; refresh current state')
                revision=current.get('configuration_revision',0)
                if label!=current.get('label',''):
                    if expected!=revision:raise ValueError('Configuration revision changed; refresh before revalidation')
                    auth_db.save_feishu_bot_config({**current,'label':label,'configuration_revision':revision+1,'configuration_updated_at':job_store.utc_now_iso()})
                return self.configuration(ctx)
            if expected!=0:raise ValueError('Configuration revision changed; refresh before revalidation')
            stamp=job_store.utc_now_iso()
            auth_db.save_feishu_bot_config({**bot,**info,'label':label,'configuration_revision':1,'configuration_updated_at':stamp,'owner_user_id':ctx.user_id,'manager_user_id':ctx.user_id,
                'owner_workspace_id':ctx.workspace_id,'validated_at':stamp,'updated_at':stamp,'saved_at':stamp,'last_error':None})
            self.enable(ctx)
        return self.configuration(ctx)

    def enable(self,ctx):
        from agents import auth_db
        ctx=live_context({k:getattr(ctx,k) for k in ('user_id','workspace_id','workspace_slug','organization_id')})
        if ctx.member_role not in {'owner','admin'}:raise PermissionError('workspace administrator required')
        bot=auth_db.get_feishu_bot_config_for_workspace(ctx.workspace_id) or {}
        if (bot.get('owner_user_id')!=ctx.user_id or not bot.get('validated_at') or bot.get('last_error')
            or not all(bot.get(k) for k in ('app_id','app_secret','bot_open_id','tenant_key','encrypt_key','verification_token'))):
            raise PermissionError('current validated bot configuration required')
        account='feishu_'+digest([ctx.organization_id,ctx.workspace_id,bot['app_id'],bot['tenant_key']])[:48]
        scope={k:getattr(ctx,k) for k in ('user_id','workspace_id','workspace_slug','organization_id')}
        if bot.get('core_channel_id')!=account or bot.get('connection_mode')!='core_webhook':
            auth_db.save_feishu_bot_config({**bot,'connection_mode':'core_webhook','core_channel_id':account,'core_channel_scope':scope,'updated_at':job_store.utc_now_iso()})
        return {'account_id':account,'mode':'core_webhook','events_path':'/api/channels/feishu/'+account+'/events'}

    def _account(self,account):
        from agents import auth_db
        matches=[b for b in auth_db.list_feishu_bot_configs() if b.get('core_channel_id')==account and b.get('connection_mode')=='core_webhook']
        if len(matches)!=1:raise PermissionError('core channel account unavailable')
        bot=matches[0];scope=bot['core_channel_scope'];ctx=live_context(scope)
        if (bot.get('owner_user_id')!=ctx.user_id or bot.get('owner_workspace_id')!=ctx.workspace_id
            or not bot.get('validated_at') or bot.get('last_error')):raise PermissionError('channel account authority changed')
        return ctx,bot

    def _authority(self,a):
        ctx,bot=self._account(a['account_id'])
        from agents.feishu_ingress_identity import FeishuIngressOwnerResolutionError
        try:owner=resolve_feishu_ingress_owner(open_id=a['sender_id'])
        except FeishuIngressOwnerResolutionError as exc:raise PermissionError('unbound channel sender') from exc
        if (owner.owner_user_id!=ctx.user_id or owner.owner_workspace_id!=ctx.workspace_id
            or owner.open_id!=a['sender_id'] or a['scope']!={k:getattr(ctx,k) for k in a['scope']}
            or a['bot_generation']!=bot_generation(bot)):
            raise PermissionError('current exact channel sender/account binding required')
        return ctx,bot

    def receive(self,account,body,headers):
        ctx,bot=self._account(account)
        def accept(event):
            if event['text'].startswith('/link '):
                _,current=self._account(account)
                if bot_generation(current)!=bot_generation(bot):raise PermissionError('channel configuration changed during link proof')
                from agents.gateway_identity import consume_feishu_message_link
                consume_feishu_message_link(state=event['text'][6:].strip(),app_id=bot['app_id'],open_id=event['sender_id'])
                return {'linked':True}
            return self._accept(ctx,account,bot,event)
        return CoreFeishuTransport(bot).receive(body,headers,accept)

    def _load(self,ctx,ident):
        row=job_store.get_job(user_id=ctx.user_id,workspace_id=ctx.workspace_id,job_id=ident)
        if not row or row.get('kind')!=KIND:raise KeyError(ident)
        if row['association']['scope']!={k:getattr(ctx,k) for k in row['association']['scope']}:raise PermissionError('channel scope mismatch')
        with job_store._connect() as conn:
            scope=conn.execute('SELECT organization_id FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND job_id=%s',(ctx.user_id,ctx.workspace_id,ident)).fetchone()
        if scope['organization_id']!=ctx.organization_id:raise PermissionError('channel organization mismatch')
        return row

    def _save(self,ctx,row):
        return job_store.update_job(user_id=ctx.user_id,workspace_id=ctx.workspace_id,job_id=row['id'],updates=row)

    def _accept(self,ctx,account,bot,event):
        if not event['text'].strip() or len(event['text'].encode())>16384:raise ValueError('bounded nonempty text required')
        scope={k:getattr(ctx,k) for k in ('user_id','workspace_id','workspace_slug','organization_id')}
        a={'account_id':account,'bot_generation':bot_generation(bot),'scope':scope,'sender_id':event['sender_id'],'chat_id':event['chat_id']}
        self._authority(a)
        ident='channel_'+digest([account,scope,event['sender_id'],event['chat_id']])[:48]
        association={**a,'conversation_id':'conversation_'+digest(ident)[:48]}
        with lock(ctx,'channel-'+ident):
            row=job_store.claim_core_channel_record(context=ctx,payload={'id':ident,'kind':KIND,'association':association,'version':0,'events':[],'last_created_at':None})
            prior=next((e for e in row['events'] if e['message_id']==event['message_id']),None)
            content_hash=digest({k:v for k,v in event.items() if k!='event_id'})
            if prior:
                if prior['content_hash']!=content_hash:raise ValueError('incoming message identity conflict')
                return self.get(ctx,ident)
            if any(e['event_id']==event['event_id'] for e in row['events']):raise ValueError('event identity already used')
            if row['last_created_at'] is not None and event['created_at']<row['last_created_at']:raise ValueError('late older message refused')
            if len(row['events'])>=256:raise ValueError('channel queue limit reached')
            def load(conn):
                current=conn.execute('SELECT payload_json FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND job_id=%s',(ctx.user_id,ctx.workspace_id,ident)).fetchone()
                return WorkflowVersion(json.loads(current['payload_json'])['version'])
            def transition(conn,current):
                row['events'].append({**event,'content_hash':content_hash,'turn_id':'channel_'+digest(event['message_id'])[:48],'phase':'QUEUED','sequence':current.version+1})
                row['last_created_at']=event['created_at'];row['version']=current.version+1
                conn.execute('UPDATE semantier_jobs SET payload_json=%s WHERE user_id=%s AND workspace_id=%s AND job_id=%s',(canonical(row),ctx.user_id,ctx.workspace_id,ident))
                return WorkflowVersion(row['version']),None
            self.inbox.accept(workflow_type=KIND,workflow_id=ident,event_id=event['event_id'],event_type='text_dm',dedupe_key=event['message_id'],expected_state_version=row['version'],load_state=load,transition=transition)
            self._cut('after_inbound_commit')
            return self.get(ctx,ident)

    def get(self,ctx,ident):
        row=self._load(ctx,ident)
        return {'channel_id':ident,'conversation_id':row['association']['conversation_id'],'sender_id':row['association']['sender_id'],'chat_id':row['association']['chat_id'],'events':row['events']}

    def cancel(self,ctx,ident,message_id):
        with lock(ctx,'channel-'+ident):
            row=self._load(ctx,ident);self._authority(row['association'])
            event=next(e for e in row['events'] if e['message_id']==message_id)
            event['cancel_requested_at']=job_store.utc_now_iso()
            if event['phase'] not in TERMINAL:
                if event.get('effect_id'):
                    effect=self.outbox.get(event['effect_id'])
                    if effect.state==READY:
                        attempt=uuid4().hex
                        if self.outbox.claim_dispatch(effect_id=effect.effect_id,attempt_id=attempt):
                            self.outbox.mark_failed_safe(effect_id=effect.effect_id,attempt_id=attempt,failure_detail='cancelled before send')
                        event['phase']='CANCELLED'
                    else:
                        if effect.state==DISPATCHING:self.outbox.mark_effect_unknown(effect_id=effect.effect_id,attempt_id=effect.dispatch_attempt_id,failure_detail='abandoned dispatch observed during cancellation')
                        event['phase']=effect.state if effect.state!=DISPATCHING else EFFECT_UNKNOWN
                else:
                    try:
                        current=self.conversations.get(ctx,row['association']['conversation_id'])
                        turn=next(t for t in current['turns'] if t['id']==event['turn_id'])
                        if turn['state']!='completed':self.conversations.cancel(ctx,row['association']['conversation_id'],event['turn_id'],self.executor)
                    except (KeyError,StopIteration):pass
                    event['phase']='CANCELLED'
            self._save(ctx,row)
        return self.get(ctx,ident)

    def tick(self,*,limit=32):
        from agents import auth_db
        active=[]
        for bot in auth_db.list_feishu_bot_configs():
            if bot.get('connection_mode')!='core_webhook' or not bot.get('core_channel_id'):continue
            try:
                _,current=self._account(bot['core_channel_id'])
                active.append(bot['core_channel_id']+':'+bot_generation(current))
            except PermissionError:continue
        if not active:return 0
        with job_store._connect() as conn:
            rows=conn.execute("SELECT payload_json FROM semantier_jobs WHERE payload_json::jsonb->>'kind'=%s AND (payload_json::jsonb->'association'->>'account_id')||':'||(payload_json::jsonb->'association'->>'bot_generation')=ANY(%s) AND EXISTS (SELECT 1 FROM jsonb_array_elements(payload_json::jsonb->'events') AS e WHERE e->>'phase' NOT IN ('COMPLETED','CANCELLED','EXECUTION_UNCERTAIN','EFFECT_UNKNOWN','FAILED_SAFE')) ORDER BY updated_at LIMIT %s",(KIND,active,limit)).fetchall()
        for item in rows:
            row=json.loads(item['payload_json'])
            try:
                ctx,_=self._authority(row['association'])
                with lock(ctx,'channel-'+row['id']):self._advance(ctx,self._load(ctx,row['id']))
            except (PermissionError,ValueError,KeyError):
                log.warning('Core Feishu channel cannot advance: %s',row['id'])
        return len(rows)

    def _advance(self,ctx,row):
        a=row['association'];ctx,bot=self._authority(a)
        event=next((e for e in row['events'] if e['phase'] not in TERMINAL),None)
        if event is None:return
        cid=a['conversation_id']
        if event['phase'] in {'QUEUED','EXECUTING'}:
            self.conversations.create(ctx,{'request_id':cid,'title':'Feishu conversation'})
            current=self.conversations.accept(ctx,cid,{'request_id':event['turn_id'],'content':event['text']})
            turn=next(t for t in current['turns'] if t['id']==event['turn_id'])
            if turn['state']=='accepted':
                work_id=self.conversations.prepare(ctx,cid,event['turn_id']);event['work_id']=work_id
                self._cut('before_execution_submit')
                self.executor.submit(ctx,work_id,resolve_context=lambda:self._authority(a)[0])
                self._cut('after_execution_submit')
            current=self.conversations.get(ctx,cid);turn=next(t for t in current['turns'] if t['id']==event['turn_id'])
            event['work_id']=turn['work_id']
            if turn['state'] in {'cancelled','uncertain','failed'}:
                event['phase']='CANCELLED' if turn['state']=='cancelled' else 'EXECUTION_UNCERTAIN';self._save(ctx,row);return
            if turn['state']!='completed':event['phase']='EXECUTING';self._save(ctx,row);return
            answer=next(m for m in current['messages'] if m['id']==event['turn_id']+':assistant')
            payload={'channel_id':row['id'],'conversation_id':cid,'turn_id':turn['id'],'work_id':turn['work_id'],
                     'run_id':answer['run_id'],'result_hash':turn['result_hash'],'account_id':a['account_id'],
                     'bot_generation':a['bot_generation'],'sender_id':a['sender_id'],'chat_id':a['chat_id'],
                     'reply_to':event['message_id'],'text':answer['content'],'purpose':'channel_chat_reply'}
            with self.outbox._connect() as conn:
                effect=self.outbox.enqueue_ready(conn,workflow_id=row['id'],effect_type='core_feishu_chat_reply',semantic_round=event['sequence'],canonical_payload=canonical(payload).encode())
            self._cut('after_reply_intent')
            event.update(effect_id=effect.effect_id,phase='DELIVERY_PENDING',run_id=answer['run_id'],result_hash=turn['result_hash'])
            self._save(ctx,row)
        effect=self.outbox.get(event['effect_id'])
        if effect.state==DISPATCHING:
            # Advisory lock is held over send; only an abandoned owner can leave this state here.
            self.outbox.mark_effect_unknown(effect_id=effect.effect_id,attempt_id=effect.dispatch_attempt_id,failure_detail='dispatch owner lost; no resend')
            event['phase']=EFFECT_UNKNOWN;self._save(ctx,row);return
        if effect.state!=READY:
            event['phase']=effect.state
            if effect.receipt_ref:event['receipt_ref']=effect.receipt_ref
            self._save(ctx,row);return
        payload=json.loads(effect.canonical_payload)
        self._authority(a)
        if (payload['result_hash']!=event['result_hash'] or payload['bot_generation']!=bot_generation(bot)
            or payload['reply_to']!=event['message_id'] or payload['chat_id']!=a['chat_id']):raise PermissionError('reply association changed')
        attempt=uuid4().hex
        if not self.outbox.claim_dispatch(effect_id=effect.effect_id,attempt_id=attempt):return
        self._cut('before_send')
        try:
            _,bot=self._authority(a)
            receipt=self.transport_factory(bot).reply(payload,effect.idempotency_key)
            self.outbox.pin_provider_correlation(effect_id=effect.effect_id,correlation_ref=receipt)
            self._cut('after_send')
            self.outbox.mark_completed(effect_id=effect.effect_id,attempt_id=attempt,receipt_ref=receipt)
            event.update(phase=COMPLETED,receipt_ref=receipt)
        except KnownNotSent as exc:
            self.outbox.mark_failed_safe(effect_id=effect.effect_id,attempt_id=attempt,failure_detail=type(exc).__name__);event['phase']=FAILED_SAFE
        except Exception as exc:
            self.outbox.mark_effect_unknown(effect_id=effect.effect_id,attempt_id=attempt,failure_detail=type(exc).__name__);event['phase']=EFFECT_UNKNOWN
        self._save(ctx,row)
