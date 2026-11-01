"""Core conversation identities over existing scoped PostgreSQL jobs/work owners."""
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime,timezone
from pathlib import Path
from agents import job_store
from services.core_work_service import CoreWorkService, lock

_KIND='core_conversation.v1'
_ID=re.compile(r'[A-Za-z0-9_-]{1,100}')



class ArtifactReadLimitExceeded(ValueError):
    pass

def retained_stream_artifact(name, encoded):
    """Describe retained bytes without interpreting them as files or measured usage."""
    import base64
    raw=base64.b64decode(encoded,validate=True)
    try:
        raw.decode('utf-8',errors='strict')
        textual=b'\0' not in raw
    except UnicodeDecodeError:
        textual=False
    bounded=len(raw)<=1048576
    return ({'id':name,'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw),
        'media_type':'text/plain' if textual else 'application/octet-stream',
        'encoding':'utf-8' if textual else 'binary',
        'availability':'AVAILABLE' if bounded else 'TOO_LARGE'},raw if bounded else None)

class CoreConversationService:
    def __init__(self):
        self.work=CoreWorkService()

    def _cut(self, stage):
        pass

    def _load(self, ctx, ident):
        if not _ID.fullmatch(ident):raise KeyError(ident)
        row=job_store.get_job(user_id=ctx.user_id,workspace_id=ctx.workspace_id,job_id=ident)
        if not row or row.get('kind')!=_KIND:raise KeyError(ident)
        with job_store._connect() as c:
            scope=c.execute('SELECT organization_id FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND job_id=%s',
                (ctx.user_id,ctx.workspace_id,ident)).fetchone()
        if not scope or scope['organization_id']!=ctx.organization_id:raise KeyError(ident)
        return row

    def _save(self,ctx,row):
        return job_store.update_job(user_id=ctx.user_id,workspace_id=ctx.workspace_id,job_id=row['id'],updates=row)

    def create(self,ctx,data):
        if set(data)!={'request_id','title'} or not _ID.fullmatch(data['request_id']) or not isinstance(data['title'],str) or not 1<=len(data['title'])<=200:
            raise ValueError('request_id and title required')
        ident=data['request_id']
        with lock(ctx,'conversation-'+ident):
            try:
                row=self._load(ctx,ident)
                if row['title']!=data['title']:raise ValueError('creation identity conflict')
                return self._project(row)
            except KeyError:pass
            # Existing store, explicit kind: no scheduler or native-history adoption.
            row=job_store.create_conversation_record(context=ctx,conversation_id=ident,title=data['title'])
            return self._project(row)

    def _sync(self,ctx,row):
        if not row['turns']:return row
        turn=row['turns'][-1]
        if turn['state'] in ('completed','cancelled','uncertain','failed'):return row
        try:work=self.work.get(ctx,turn['work_id'])
        except KeyError:return row
        execution=work.get('execution',{})
        if execution and execution.get('state')=='reserved':
            responsibility=self.work.responsibilities.get(self.work._load(ctx,turn['work_id'])['responsibility_id'])
            if responsibility.lease_expires_at and datetime.fromisoformat(responsibility.lease_expires_at.replace('Z','+00:00'))<=datetime.now(timezone.utc):
                execution={**execution,'state':'uncertain'}
        state='completed' if work['status']=='completed' else 'cancelled' if work['status']=='cancelled' else 'uncertain' if execution.get('state') in ('uncertain','stopped') else 'running' if execution else 'accepted'
        if state=='completed':
            result=work.get('result',{})
            if not execution.get('run_id') or result.get('run_id') != execution['run_id']:
                raise RuntimeError('execution result run association mismatch')
            native=result.get('native_result',{});text=native.get('text')
            if not isinstance(text,str):raise RuntimeError('retained assistant result missing text')
            if not any(m['id']==turn['id']+':assistant' for m in row['messages']):
                row['messages'].append({'id':turn['id']+':assistant','role':'assistant','content':text,
                    'turn_id':turn['id'],'work_id':turn['work_id'],'run_id':result['run_id'],'created_at':job_store.utc_now_iso()})
            turn['result_hash']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
        if turn['state']!=state:
            turn['state']=state;self._cut('before_projection_commit');return self._save(ctx,row)
        return row

    @staticmethod
    def _project(row):
        return {k:row[k] for k in ('id','title','messages','turns','created_at','updated_at')}

    def get(self,ctx,ident):
        with lock(ctx,'conversation-'+ident):
            value=self._project(self._sync(ctx,self._load(ctx,ident)))
            if value['turns'] and value['turns'][-1]['state']!='completed':
                work=job_store.get_job(user_id=ctx.user_id,workspace_id=ctx.workspace_id,job_id=value['turns'][-1]['work_id'])
                value['partial_text']=''.join(e.get('text','') for e in (work or {}).get('progress_events',[]) if e.get('type')=='text')
            return value

    def list(self,ctx):
        with job_store._connect() as conn:
            rows=conn.execute("SELECT job_id FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND organization_id=%s AND payload_json::jsonb->>'kind'=%s ORDER BY updated_at DESC LIMIT 100",(ctx.user_id,ctx.workspace_id,ctx.organization_id,_KIND)).fetchall()
        return {'items':[self.get(ctx,r['job_id']) for r in rows]}

    def _history_channel(self, ctx, ident):
        with job_store._connect() as conn:
            rows=conn.execute("SELECT payload_json FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND organization_id=%s AND payload_json::jsonb->>'kind'='core_feishu_channel.v1' AND payload_json::jsonb->'association'->>'conversation_id'=%s LIMIT 2",(ctx.user_id,ctx.workspace_id,ctx.organization_id,ident)).fetchall()
        if len(rows)>1:raise RuntimeError('ambiguous conversation channel association')
        if not rows:return None
        row=rows[0]['payload_json']
        if isinstance(row,str):row=json.loads(row)
        if row['association']['scope']!={k:getattr(ctx,k) for k in ('user_id','workspace_id','workspace_slug','organization_id')}:
            raise RuntimeError('conversation channel scope mismatch')
        return row

    def _history_turn(self, ctx, conversation, turn, channel):
        from agents.effect_outbox import AtomicEffectOutbox, derive_effect_identity
        from services.core_work_service import canonical
        delivery={'state':'NOT_ASSOCIATED','receipt_ref':None}
        try:work=self.work.get(ctx,turn['work_id'])
        except KeyError:
            if turn['state'] not in ('accepted','cancelled'):raise RuntimeError('associated work unavailable')
            work=None
        if work and (work.get('input',{}).get('conversation_id')!=conversation['id'] or work.get('input',{}).get('turn_id')!=turn['id']):
            raise RuntimeError('execution conversation association mismatch')
        result=(work or {}).get('result')
        if result and (not (work or {}).get('execution',{}).get('run_id') or result.get('run_id') != work['execution']['run_id']):
            raise RuntimeError('execution result run association mismatch')
        if result and turn.get('result_hash')!=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest():
            raise RuntimeError('conversation result evidence mismatch')
        execution={'work_id':turn['work_id'],'status':(work or {}).get('status','not_started'),**(work or {}).get('execution',{})}
        if channel:
            events=[e for e in channel['events'] if e.get('turn_id')==turn['id']]
            if len(events)>1:raise RuntimeError('ambiguous turn delivery association')
            delivery={'state':'NOT_ASSOCIATED','receipt_ref':None,'channel_id':channel['id']}
            if events:
                event=events[0]
                delivery['state']=event['phase']
                if event.get('effect_id'):
                    effect=AtomicEffectOutbox(job_store._db_path()).get(event['effect_id'])
                    if effect is None:raise RuntimeError('delivery intent unavailable')
                    payload=json.loads(effect.canonical_payload)
                    expected={'channel_id':channel['id'],'conversation_id':conversation['id'],'turn_id':turn['id'],
                        'work_id':turn['work_id'],'run_id':execution.get('run_id'),'result_hash':turn.get('result_hash'),
                        'reply_to':event['message_id'],'purpose':'channel_chat_reply','text':(result or {}).get('native_result',{}).get('text'),
                        **{k:channel['association'][k] for k in ('account_id','bot_generation','sender_id','chat_id')}}
                    identity=derive_effect_identity(workflow_id=channel['id'],effect_type='core_feishu_chat_reply',semantic_round=event['sequence'],canonical_payload=effect.canonical_payload)
                    if effect.request_hash!=identity.request_hash or effect.idempotency_key!=identity.idempotency_key or effect.effect_id!=identity.effect_id or effect.workflow_id!=channel['id'] or effect.effect_type!='core_feishu_chat_reply' or effect.semantic_round!=event['sequence'] or any(payload.get(k)!=v for k,v in expected.items()):
                        raise RuntimeError('delivery evidence association mismatch')
                    delivery.update(state=effect.state,effect_id=effect.effect_id,
                        receipt_ref=effect.receipt_ref if effect.state=='COMPLETED' else None)
                    if effect.state=='COMPLETED' and not effect.receipt_ref:raise RuntimeError('completed delivery receipt unavailable')
        value={'id':turn['id'],'state':turn['state'],'execution':execution,'delivery':delivery,
            'evidence':{'conversation_result_hash':turn.get('result_hash'),
                'execution_result_hash':hashlib.sha256(canonical(result).encode()).hexdigest() if result else None},
            'result':None}
        if result:
            value['result']={'text':result.get('native_result',{}).get('text'),
                **{k:result[k] for k in ('returncode','gaps','monitoring') if k in result}}
        return value

    def history(self,ctx,ident=None,turn_id=None):
        """Read existing owners; receipt completion never follows execution success."""
        if ident is None:
            items=[]
            for item in self.list(ctx)['items']:
                channel=self._history_channel(ctx,item['id'])
                items.append({k:item[k] for k in ('id','title','created_at','updated_at')} | {
                    'source':'Feishu DM' if channel else 'Core Chat',
                    'state':item['turns'][-1]['state'] if item['turns'] else 'empty',
                    'message_count':len(item['messages'])})
            return {'items':items}
        item=self.get(ctx,ident);channel=self._history_channel(ctx,ident)
        if turn_id is not None:
            turn=next((t for t in item['turns'] if t['id']==turn_id),None)
            if turn is None:raise KeyError(turn_id)
            return self._history_turn(ctx,item,turn,channel)
        return {k:item[k] for k in ('id','title','messages','created_at','updated_at')} | {
            'source':'Feishu DM' if channel else 'Core Chat','partial_text':item.get('partial_text',''),
            'turns':[self._history_turn(ctx,item,t,channel) for t in item['turns']]}

    def _skill_observations(self, ctx, work_id, run_id, epoch):
        """Read only evidence bound to this reservation; never infer invocation."""
        from agents.task_service import evidence_owner, read_skill
        row = self.work._load(ctx, work_id)
        reservation = row.get('execution_reservation') or {}
        if reservation.get('run_id') != run_id or reservation.get('lease_epoch') != epoch:
            raise RuntimeError('inspection execution reservation changed')
        skill = {'state': 'UNAVAILABLE', 'meaning': 'prepared_not_invocation'}
        bound = reservation.get('skill')
        if bound is not None:
            try:
                skill_id, body = read_skill(ctx, bound['pin'])
            except KeyError as exc:
                raise RuntimeError('prepared skill evidence unavailable') from exc
            digest = hashlib.sha256(body).hexdigest()
            if skill_id != bound['skill_id'] or digest != bound['sha256']:
                raise RuntimeError('prepared skill evidence mismatch')
            import codecs
            preview = codecs.getincrementaldecoder('utf-8')('strict').decode(body[:65536], final=len(body) <= 65536)
            skill = {'state': 'AVAILABLE', 'meaning': 'prepared_not_invocation',
                'skill_id': skill_id, 'sha256': digest, 'size_bytes': len(body),
                'content': preview, 'truncated': len(body) > 65536, 'run_id': run_id,
                'pin': bound['pin']}
        observations = {'state': 'UNAVAILABLE', 'history_coverage': 'not_established', 'items': []}
        monitoring = reservation.get('monitoring') or {}
        if monitoring.get('status') == 'retained':
            store = None
            try:
                store, service = evidence_owner()
                snapshot = service.public_snapshot(context=ctx, session_id=ctx.workspace_id + ':connector:' + run_id)
                rows = [r for r in snapshot['observations'] if r['run_id'] == run_id and r['profile_id'] == 'work-' + run_id]
                items = []
                for r in rows[:100]:
                    payload = r.get('payload') or {}
                    event = payload.get('payload') or {}
                    items.append({'sequence': r['sequence'], 'kind': r['event_kind'],
                        'source': r['source_kind'], 'captured_at': r['captured_at'],
                        'artifact_hash': r['artifact_hash'], 'hook': payload.get('hook'),
                        'tool_name': event.get('tool_name'), 'status': event.get('status')})
                observations = {'state': 'AVAILABLE' if rows else 'UNAVAILABLE',
                    'history_coverage': snapshot['history_coverage'], 'items': items,
                    'gaps': snapshot['gaps'], 'conflicts': snapshot['conflicts'],
                    'truncated': len(rows) > 100, 'snapshot_hash': snapshot['snapshot_hash']}
            except Exception:
                pass  # Optional monitoring cannot change result availability.
            finally:
                if store: store.close()
        return skill, observations

    def inspection(self,ctx,ident,turn_id):
        """Exact retained evidence only; no native session or path lookup."""
        import base64
        value=self.history(ctx,ident,turn_id)
        try:work=self.work.get(ctx,value['execution']['work_id'])
        except KeyError:
            if value['execution']['status']!='not_started':raise
            work={}
        result=work.get('result')
        if result and hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()!=value['evidence']['conversation_result_hash']:
            raise RuntimeError('inspection result changed during read')
        logs={'source':'retained_managed_result','state':'AVAILABLE' if result else 'UNAVAILABLE','streams':[]}
        for stream in ('stdout','stderr'):
            encoded=(result or {}).get(stream+'_base64')
            if encoded is None:continue
            raw=base64.b64decode(encoded,validate=True)
            logs['streams'].append({'name':stream,'sha256':hashlib.sha256(raw).hexdigest(),
                'size_bytes':len(raw),'truncated':len(raw)>65536,'text':raw[:65536].decode('utf-8',errors='replace')})
        artifact=None
        if result:
            content=json.dumps({'conversation_id':ident,'turn_id':turn_id,**value},sort_keys=True,ensure_ascii=True).encode()
            artifact={'id':'result','media_type':'application/json','size_bytes':len(content),
                'sha256':hashlib.sha256(content).hexdigest(),
                'href':f'/api/conversation-history/{ident}/turns/{turn_id}/artifacts/result'}
        artifacts=[]
        prefix=f'/api/conversation-history/{ident}/turns/{turn_id}/artifacts/'
        if artifact:
            artifacts.append({**artifact,'availability':'AVAILABLE','encoding':'utf-8'})
        for stream in ('stdout','stderr'):
            encoded=(result or {}).get(stream+'_base64')
            if encoded is not None:
                descriptor,_=retained_stream_artifact(stream,encoded)
                artifacts.append({**descriptor,'href':prefix+stream})
        artifacts=[{**a,'source':'retained_managed_result','work_id':value['execution']['work_id'],
            'run_id':value['execution'].get('run_id')} for a in artifacts]
        run_id=value['execution'].get('run_id')
        skill,observations=self._skill_observations(ctx,value['execution']['work_id'],run_id,value['execution'].get('lease_epoch')) if run_id else ({'state':'UNAVAILABLE','meaning':'prepared_not_invocation'}, {'state':'UNAVAILABLE','history_coverage':'not_established','items':[]})
        return {'conversation_id':ident,'turn_id':turn_id,**value,'logs':logs,'artifact':artifact,'artifacts':artifacts,'skill':skill,'observations':observations}

    def result_artifact(self,ctx,ident,turn_id):
        value=self.inspection(ctx,ident,turn_id)
        if value['artifact'] is None:raise KeyError('result')
        projected={k:v for k,v in value.items() if k not in ('logs','artifact','artifacts','skill','observations')}
        return json.dumps(projected,sort_keys=True,ensure_ascii=True).encode()

    def artifact_bytes(self,ctx,ident,turn_id,artifact_id):
        if artifact_id not in ('result','stdout','stderr'):raise KeyError('artifact')
        if artifact_id=='result':return self.result_artifact(ctx,ident,turn_id),'application/json'
        value=self.inspection(ctx,ident,turn_id)
        descriptor=next((a for a in value['artifacts'] if a['id']==artifact_id),None)
        if descriptor is None:raise KeyError('artifact')
        # Re-read authoritative evidence; refuse a mixed snapshot rather than serve new bytes under old identity.
        work=self.work.get(ctx,value['execution']['work_id'])
        result=work.get('result') or {}
        if hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()!=value['evidence']['conversation_result_hash']:
            raise RuntimeError('artifact result changed during read')
        actual,raw=retained_stream_artifact(artifact_id,result[artifact_id+'_base64'])
        if actual['sha256']!=descriptor['sha256']:raise RuntimeError('artifact changed')
        if raw is None:raise ArtifactReadLimitExceeded('artifact exceeds 1 MiB read limit')
        return raw,actual['media_type']

    def accept(self,ctx,ident,data):
        if set(data)!={'request_id','content'} or not _ID.fullmatch(data['request_id']) or not isinstance(data['content'],str) or not data['content'].strip() or len(data['content'].encode())>16384:
            raise ValueError('bounded message and request_id required')
        with lock(ctx,'conversation-'+ident):
            row=self._sync(ctx,self._load(ctx,ident))
            prior=next((t for t in row['turns'] if t['id']==data['request_id']),None)
            digest=hashlib.sha256(data['content'].encode()).hexdigest()
            if prior:
                if prior['message_hash']!=digest:raise ValueError('message identity conflict')
                return self._project(row)
            if row['turns'] and row['turns'][-1]['state'] not in ('completed','cancelled','uncertain','failed'):
                raise ValueError('another turn is pending')
            context=[{'role':m['role'],'content':m['content']} for m in row['messages'] if m.get('role') in ('user','assistant') and (m['role']=='assistant' or any(t['id']==m['turn_id'] and t['state']=='completed' for t in row['turns']))]
            text=Path(__file__).parents[1]/'prompts/chat/core-conversation-turn.md'
            prefix,rest=text.read_text().split('{{conversation}}',1)
            middle,suffix=rest.split('{{message}}',1)
            instruction=prefix+json.dumps(context,ensure_ascii=False)+middle+json.dumps(data['content'],ensure_ascii=False)+suffix
            if len(instruction.encode())>32768:raise ValueError('conversation context limit reached; create a new conversation')
            work_id='chat_'+hashlib.sha256((ident+':'+data['request_id']).encode()).hexdigest()[:48]
            row['messages'].append({'id':data['request_id']+':user','role':'user','content':data['content'],'turn_id':data['request_id'],'work_id':work_id,'created_at':job_store.utc_now_iso()})
            row['turns'].append({'id':data['request_id'],'state':'accepted','work_id':work_id,'message_hash':digest,'instruction':instruction})
            row=self._save(ctx,row);self._cut('after_message_commit')
            return self._project(row)

    def prepare(self,ctx,ident,turn_id):
        with lock(ctx,'conversation-'+ident):
            row=self._load(ctx,ident);turn=next(t for t in row['turns'] if t['id']==turn_id)
            self.work.create(ctx,{'request_id':turn['work_id'],'title':row['title'],'input':{'instruction':turn['instruction'],'conversation_id':ident,'turn_id':turn_id}})
            self._cut('after_work_association');return turn['work_id']

    def cancel(self,ctx,ident,turn_id,execution):
        with lock(ctx,'conversation-'+ident):
            row=self._load(ctx,ident);turn=next(t for t in row['turns'] if t['id']==turn_id)
            try:self.work.cancel(ctx,turn['work_id'])
            except KeyError:
                turn['state']='cancelled';self._save(ctx,row)
            else:execution.stop(ctx,turn['work_id'])
        return self.get(ctx,ident)
