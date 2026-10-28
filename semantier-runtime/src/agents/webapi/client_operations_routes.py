"""Authenticated release administration; independent public distribution."""
from datetime import datetime,timedelta,timezone
import json
from uuid import uuid4
from fastapi import APIRouter,HTTPException,Request,Query
from typing import Literal
from fastapi.routing import APIRoute
from pydantic import ValidationError
from contracts.client_operations import ReleaseCandidate,Promotion,ChannelControl,EventBatch,Diagnostic,Classification,Identifier,Strict,PublicationIntent,PublicationRetry,TargetBindingAdmission
from agents.client_operations.artifact_io import OperationError
from agents.webapi.deps import require_authenticated_context
from agents.model_gateway.auth import csrf


class Errors(APIRoute):
    def get_route_handler(self):
        handler=super().get_route_handler()
        async def guarded(request):
            try:return await handler(request)
            except OperationError as e:
                raise HTTPException(e.status,{'code':e.code,'message':e.code.replace('_',' '),'retryable':e.status in {429,503},'correlation_id':uuid4().hex,'mutation_outcome':e.mutation_outcome}) from e
            except HTTPException as e:
                raise HTTPException(e.status_code,{'code':{401:'authentication_required',403:'scope_denied',409:'context_changed'}.get(e.status_code,'request_refused'),'message':'Client operation unavailable','retryable':e.status_code in {429,503},'correlation_id':uuid4().hex}) from e
            except ValidationError as e:
                raise HTTPException(422,{'code':'schema_mismatch','message':'Invalid client operations request','retryable':False,'correlation_id':uuid4().hex}) from e
        return guarded


router=APIRouter(prefix='/api/client-operations',tags=['client-operations'],route_class=Errors)


def service(request):
    owner=getattr(request.app.state,'client_operations',None)
    if not owner:raise OperationError(503,'client_operations_unavailable')
    return owner


def browser(request,capability=None):
    owner=service(request);ctx=require_authenticated_context(request)
    if request.headers.get('authorization'):raise OperationError(403,'browser_session_required')
    if request.method not in {'GET','HEAD'}:csrf(request)
    permissions=owner.store.permissions(ctx.user_id,owner.channel)
    if capability and capability not in permissions:raise OperationError(403,'explicit_platform_grant_required')
    return owner,ctx,permissions


async def customer(request):
    """Feedback-only capability or current browser identity; never model keys."""
    if request.headers.get('authorization','').startswith('Bearer '):
        from types import SimpleNamespace
        owner,principal=capability(request,'diagnostic')
        from agents.gateway_identity import get_user_record
        if not get_user_record(principal['user_id']):raise OperationError(401,'feedback_identity_unavailable')
        return owner,SimpleNamespace(user_id=principal['user_id']),()
    return browser(request)


def capability(request,kind):
    owner=service(request);header=request.headers.get('authorization','')
    principal=owner.store.principal(header[7:],kind) if header.startswith('Bearer ') else None
    if not principal or principal['channel']!=owner.channel:raise OperationError(401,'invalid_scoped_capability')
    return owner,principal


async def body(request,model,limit=5*1024*1024):
    raw=bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw)>limit:raise OperationError(413,'payload_limit')
    try:return model.model_validate_json(raw)
    except (ValueError,ValidationError) as e:raise OperationError(422,'schema_mismatch',mutation_outcome='not_started') from e


def envelope(owner,data,permissions=()):
    return {'schema_version':'1.0.0','authorized_scope':{'environment':owner.environment,'channel':owner.channel,'platform':'win32','arch':'x64'},'fetched_at':datetime.now(timezone.utc).isoformat(),'allowed_actions':sorted(permissions),**data}


@router.get('/context')
def context(request:Request):
    owner,ctx,permissions=browser(request)
    return envelope(owner,{'user_id':ctx.user_id},permissions)


@router.get('/releases')
def releases(request:Request,q:str=Query('',max_length=128),qualification:Literal['all','verified','rejected']='all',published:Literal['all','published','unpublished']='all',sort:Literal['newest','oldest']='newest',offset:int=Query(0,ge=0,le=10000)):
    owner,ctx,permissions=browser(request,'release_read')
    with owner.store.connection() as c:
        rows=c.execute("SELECT r.release_id,r.body,r.qualification,r.received_at FROM releases r LEFT JOIN delivery_heads ch ON ch.channel=%s AND ch.target_id=%s AND ch.release_id=r.release_id WHERE (%s='all' OR r.qualification=%s) AND (%s='all' OR (%s='published' AND ch.release_id IS NOT NULL) OR (%s='unpublished' AND ch.release_id IS NULL)) AND (%s='' OR strpos(lower(r.body#>>'{manifest,request,version}'),lower(%s))>0 OR strpos(lower(r.build_id),lower(%s))>0) ORDER BY CASE WHEN %s='oldest' THEN r.received_at END ASC,r.received_at DESC,r.release_id LIMIT 101 OFFSET %s",(owner.channel,owner.primary_target_id if hasattr(owner,'primary_target_id') else getattr(getattr(owner,'publisher',None),'target_id','channel1'),qualification,qualification,published,published,published,q,q,q,sort,offset)).fetchall()
    return envelope(owner,{'items':rows[:100],'coverage':'retained_candidates','limit':100,'offset':offset,'next_offset':offset+100 if len(rows)>100 else None},permissions)


@router.get('/releases/{release_id}')
def release(release_id:str,request:Request):
    owner,ctx,permissions=browser(request,'release_read')
    with owner.store.connection() as c:
        row=c.execute('SELECT * FROM releases WHERE release_id=%s',(release_id,)).fetchone()
        if not row:raise OperationError(404,'release_unavailable')
        audit=c.execute('SELECT * FROM audit WHERE object_id=%s ORDER BY created_at',(release_id,)).fetchall()
    return envelope(owner,{'release':row,'audit':audit},permissions)


@router.post('/releases')
async def ingest(request:Request):
    owner,principal=capability(request,'ci')
    candidate=await body(request,ReleaseCandidate,1024*1024)
    try:admitted=owner.publisher.admit(candidate,principal)
    except OperationError as error:
        if error.status<500:owner.publisher.record_rejection(candidate,error.code,principal['principal_id'])
        raise
    return envelope(owner,{'release':admitted})


@router.get('/channels/{channel}')
def channel(channel:str,request:Request):
    owner,ctx,permissions=browser(request,'release_read')
    if channel!=owner.channel:raise OperationError(403,'channel_scope_denied')
    with owner.store.connection() as c:row=c.execute('SELECT ch.*,h.release_id AS release_id,h.record AS record FROM channels ch LEFT JOIN delivery_heads h ON h.channel=ch.channel AND h.target_id=%s WHERE ch.channel=%s',(owner.primary_target_id if hasattr(owner,'primary_target_id') else owner.publisher.target_id,channel)).fetchone()
    return envelope(owner,{'channel':row,'coverage':'known_empty' if row is None else 'current'},permissions)


@router.post('/channels/{channel}/promotions')
async def promote(channel:str,request:Request):
    owner,ctx,permissions=browser(request,'release_manage')
    if channel!=owner.channel:raise OperationError(403,'channel_scope_denied')
    if 'publish' not in owner.store.target_permissions(ctx.user_id,owner.channel).get(owner.publisher.target_id,set()):raise OperationError(403,'delivery_target_scope_denied')
    data=await body(request,Promotion,8192)
    result = owner.publications.promote_legacy(ctx.user_id,data) if hasattr(owner,'publications') else owner.publisher.promote(ctx.user_id,data)
    return envelope(owner,{'operation':result},permissions)


@router.post('/channels/{channel}/pause')
@router.post('/channels/{channel}/resume')
async def control(channel:str,request:Request):
    owner,ctx,permissions=browser(request,'release_manage')
    if channel!=owner.channel:raise OperationError(403,'channel_scope_denied')
    data=await body(request,ChannelControl,8192);paused=request.url.path.endswith('/pause')
    with owner.store.connection() as c:
        if not c.execute('SELECT pg_try_advisory_xact_lock(hashtext(%s)) AS acquired',(owner.store.schema+':'+channel,)).fetchone()['acquired']:
            raise OperationError(409,'publication_in_progress')
        from contracts.client_operations import digest
        intent={**data.model_dump(),'control':'pause' if paused else 'resume'}
        previous=c.execute('SELECT * FROM operations WHERE user_id=%s AND idempotency_key=%s',(ctx.user_id,data.idempotency_key)).fetchone()
        if previous:
            if previous['request_hash']!=digest(intent):raise OperationError(409,'operation_idempotency_conflict')
            return envelope(owner,{'channel':previous['body']['result']},permissions)
        if c.execute("SELECT 1 FROM operations WHERE channel=%s AND status='preparing'",(channel,)).fetchone():raise OperationError(409,'publication_in_progress')
        if c.execute('SELECT 1 FROM publication_targets WHERE channel=%s AND reserved LIMIT 1',(channel,)).fetchone():raise OperationError(409,'publication_in_progress')
        row=c.execute('UPDATE channels SET paused=%s,revision=revision+1 WHERE channel=%s AND revision=%s RETURNING *',(paused,channel,data.expected_revision)).fetchone()
        if not row:raise OperationError(409,'channel_revision_conflict')
        c.execute('INSERT INTO operations(operation_id,user_id,channel,idempotency_key,request_hash,body,status) VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s)',(uuid4().hex,ctx.user_id,channel,data.idempotency_key,digest(intent),json.dumps({**intent,'result':row},default=str),'published'))
        owner.store.audit(c,ctx.user_id,'pause' if paused else 'resume',channel,{'expected_revision':data.expected_revision,'actual_revision':row['revision'],'idempotency_key':data.idempotency_key})
    return envelope(owner,{'channel':row},permissions)


@router.get('/operations/{operation_id}')
def operation(operation_id:str,request:Request):
    owner,ctx,permissions=browser(request,'release_read')
    with owner.store.connection() as c:row=c.execute('SELECT * FROM operations WHERE (operation_id=%s OR (idempotency_key=%s AND user_id=%s)) AND channel=%s',(operation_id,operation_id,ctx.user_id,owner.channel)).fetchone()
    if not row:raise OperationError(404,'operation_unavailable')
    if 'publication_intent' in row['body'] and hasattr(owner,'publications'):
        try:
            publication=owner.publications.lookup(ctx.user_id,row['body']['idempotency_key'])
            row={**row,'publication_id':publication['publication_id'],'status':owner.publications.legacy_status(publication)}
        except OperationError as error:
            if error.status!=404:raise
            row={**row,'status':'preparing'}
    return envelope(owner,{'operation':row},permissions)


@router.get('/delivery-targets')
def delivery_targets(request:Request,release_id:str=Query(...,min_length=1,max_length=128)):
    owner,ctx,permissions=browser(request,'release_read')
    return envelope(owner,owner.targets.catalog(ctx.user_id,release_id),permissions)


@router.post('/delivery-targets/bindings')
async def target_binding(request:Request):
    owner,principal=capability(request,'ci')
    data=await body(request,TargetBindingAdmission,2*1024*1024)
    return envelope(owner,{'binding':owner.targets.admit(data,principal)})


@router.post('/publications')
async def publication_submit(request:Request):
    owner,ctx,permissions=browser(request,'release_manage')
    data=await body(request,PublicationIntent,32768)
    return envelope(owner,{'publication':owner.publications.submit(ctx.user_id,data)},permissions)


@router.get('/publications')
def publications(request:Request,release_id:str|None=None,status:Literal['active','all']='all',offset:int=Query(0,ge=0,le=10000)):
    owner,ctx,permissions=browser(request,'release_read')
    return envelope(owner,owner.publications.list(ctx.user_id,release_id,status,offset),permissions)


@router.get('/publications/{publication_id}')
def publication_snapshot(publication_id:str,request:Request):
    owner,ctx,permissions=browser(request,'release_read')
    return envelope(owner,{'publication':owner.publications.snapshot(ctx.user_id,publication_id)},permissions)


@router.get('/publication-requests/{key}')
def publication_lookup(key:str,request:Request):
    owner,ctx,permissions=browser(request,'release_read')
    return envelope(owner,{'publication':owner.publications.lookup(ctx.user_id,key)},permissions)


@router.post('/publications/{publication_id}/reconcile')
def publication_reconcile(publication_id:str,request:Request):
    owner,ctx,permissions=browser(request,'release_manage')
    return envelope(owner,{'publication':owner.publications.reconcile(ctx.user_id,publication_id)},permissions)


@router.post('/publications/{publication_id}/retry')
async def publication_retry(publication_id:str,request:Request):
    owner,ctx,permissions=browser(request,'release_manage')
    data=await body(request,PublicationRetry,32768)
    return envelope(owner,{'publication':owner.publications.retry(ctx.user_id,publication_id,data)},permissions)


def reporting_window(days,start_at,end_at):
    if start_at is None and end_at is None:
        if days not in {1,7,30}:raise OperationError(422,'unsupported_window')
        end=datetime.now(timezone.utc)
        return end-timedelta(days=days),end
    if start_at is None or end_at is None:raise OperationError(422,'incomplete_window')
    if start_at.tzinfo is None or end_at.tzinfo is None:raise OperationError(422,'timezone_required')
    start,end=start_at.astimezone(timezone.utc),end_at.astimezone(timezone.utc)
    if start>=end:raise OperationError(422,'invalid_window_order')
    return start,end


@router.get('/updates')
def updates(request:Request,release_id:str|None=None,days:int=7,start_at:datetime|None=None,end_at:datetime|None=None):
    owner,ctx,permissions=browser(request,'release_read')
    start,end=reporting_window(days,start_at,end_at)
    return envelope(owner,owner.feedback.attempts(release_id,days,start,end),permissions)


@router.get('/updates/{update_id}')
def update(update_id:str,request:Request):
    owner,ctx,permissions=browser(request,'release_read')
    with owner.store.connection() as c:
        rows=c.execute('SELECT * FROM events WHERE update_id=%s ORDER BY received_at',(update_id,)).fetchall()
        diagnostics=c.execute('SELECT diagnostic_id FROM diagnostics WHERE update_id=%s',(update_id,)).fetchall() if 'diagnostic_read' in permissions else []
    if not rows:raise OperationError(404,'update_unavailable')
    return envelope(owner,{'events':rows,'diagnostics':diagnostics,'source':'client_reported'},permissions)


@router.get('/distribution')
def distribution(request:Request,release_id:str|None=None,days:int=7,start_at:datetime|None=None,end_at:datetime|None=None):
    owner,ctx,permissions=browser(request,'release_read')
    start,end=reporting_window(days,start_at,end_at)
    with owner.store.connection() as c:
        rows=c.execute('SELECT body FROM releases WHERE (%s::text IS NULL OR release_id=%s)',(release_id,release_id)).fetchall()
    if release_id and not rows:raise OperationError(404,'release_unavailable')
    # Health probes and arbitrary URL requests are never labeled downloads.
    keys=[a['key'] for row in rows for p in row['body']['manifest']['packages'] for a in [p['artifact']]+([p['distributionArchive']] if p.get('distributionArchive') else [])]
    from agents.client_operations.distribution import summary
    return envelope(owner,summary(owner.store,keys,start,end),permissions)


class Enrollment(Strict):
    installation_ref: Identifier
    consent_version: str


@router.post('/installations')
async def enroll(request:Request):
    owner,ctx,permissions=browser(request);data=await body(request,Enrollment,8192)
    return envelope(owner,owner.feedback.enroll(ctx.user_id,data.installation_ref,data.consent_version))


@router.delete('/installations/{installation_ref}/reporting')
async def withdraw(installation_ref:str,request:Request):
    if request.headers.get('authorization','').startswith('Bearer sco_'):
        from types import SimpleNamespace
        owner,principal=capability(request,'reporting')
        if principal['installation_ref']!=installation_ref:raise OperationError(403,'installation_scope_mismatch')
        ctx=SimpleNamespace(user_id=principal['user_id'])
    else:
        owner,ctx,permissions=await customer(request)
    return envelope(owner,owner.feedback.withdraw(ctx.user_id,installation_ref))


@router.post('/events')
async def events(request:Request):
    owner,principal=capability(request,'reporting')
    from agents.gateway_identity import get_user_record
    if not get_user_record(principal['user_id']):raise OperationError(401,'feedback_identity_unavailable')
    batch=await body(request,EventBatch,1024*1024)
    return envelope(owner,owner.feedback.events(principal,batch))


@router.post('/diagnostics')
async def submit(request:Request):
    owner,ctx,permissions=await customer(request);data=await body(request,Diagnostic)
    return envelope(owner,{'receipt':owner.feedback.submit(ctx.user_id,data)})


@router.get('/diagnostics/receipts')
async def receipt(request:Request,submission_key:str):
    owner,ctx,permissions=await customer(request)
    with owner.store.connection() as c:row=c.execute('SELECT * FROM diagnostics WHERE user_id=%s AND submission_key=%s',(ctx.user_id,submission_key)).fetchone()
    if not row:raise OperationError(404,'receipt_unavailable')
    return envelope(owner,{'receipt':owner.feedback.receipt(row)})


@router.get('/diagnostics')
def diagnostics(request:Request,release_id:str|None=None,status:Literal['active','all','new','triaged','resolved']='active',category:Literal['all','unclassified','startup','update','provider','ui','other']='all',severity:Literal['all','unassessed','low','medium','high']='all',q:str=Query('',max_length=128),offset:int=Query(0,ge=0,le=10000)):
    owner,ctx,permissions=browser(request,'diagnostic_read')
    with owner.store.connection() as c:rows=c.execute("SELECT diagnostic_id,release_id,update_id,metadata,classification,revision,received_at,expires_at,deleted_at FROM diagnostics WHERE (%s::text IS NULL OR release_id=%s) AND (%s='all' OR (%s='active' AND classification->>'status' IN ('new','triaged')) OR classification->>'status'=%s) AND (%s='all' OR classification->>'category'=%s) AND (%s='all' OR classification->>'severity'=%s) AND (%s='' OR strpos(lower(diagnostic_id),lower(%s))>0 OR strpos(lower(coalesce(metadata->>'error_code','')),lower(%s))>0) ORDER BY received_at DESC,diagnostic_id LIMIT 101 OFFSET %s",(release_id or None,release_id or None,status,status,status,category,category,severity,severity,q,q,q,offset)).fetchall()
    return envelope(owner,{'items':rows[:100],'coverage':'retained_reports','limit':100,'offset':offset,'next_offset':offset+100 if len(rows)>100 else None},permissions)


@router.get('/diagnostics/{diagnostic_id}')
def diagnostic(diagnostic_id:str,request:Request):
    owner,ctx,permissions=browser(request,'diagnostic_read')
    with owner.store.connection() as c:
        row=c.execute('SELECT *,CASE WHEN expires_at>now() AND deleted_at IS NULL THEN payload ELSE NULL END AS payload FROM diagnostics WHERE diagnostic_id=%s',(diagnostic_id,)).fetchone()
        if not row:raise OperationError(404,'diagnostic_unavailable')
        audit=c.execute('SELECT * FROM audit WHERE object_id=%s ORDER BY created_at',(diagnostic_id,)).fetchall()
    return envelope(owner,{'diagnostic':row,'audit':audit},permissions)


@router.patch('/diagnostics/{diagnostic_id}')
async def classify(diagnostic_id:str,request:Request):
    owner,ctx,permissions=browser(request,'diagnostic_triage');data=await body(request,Classification,16384)
    return envelope(owner,{'diagnostic':owner.feedback.classify(ctx.user_id,diagnostic_id,data)},permissions)


@router.delete('/diagnostics/{diagnostic_id}/payload')
async def erase(diagnostic_id:str,request:Request):
    owner,ctx,permissions=await customer(request)
    return envelope(owner,{'receipt':owner.feedback.erase(ctx.user_id,diagnostic_id)})


@router.post('/authorizations')
async def authorization_start(request:Request):
    from contracts.client_operations import FeedbackAuthorization
    from agents.webapi.deps import enforce_rate_limit
    enforce_rate_limit('auth_public',request=request)
    owner=service(request);data=await body(request,FeedbackAuthorization,8192)
    return envelope(owner,owner.authorization.start(data))


@router.get('/authorizations/{intent_ref}')
def authorization_view(intent_ref:str,request:Request):
    owner,ctx,permissions=browser(request)
    return envelope(owner,{**owner.authorization.view(intent_ref),'actor_user_id':ctx.user_id})


@router.post('/authorizations/{intent_ref}/decision')
async def authorization_decide(intent_ref:str,request:Request):
    from contracts.client_operations import FeedbackDecision
    owner,ctx,permissions=browser(request);data=await body(request,FeedbackDecision,4096)
    return envelope(owner,{**owner.authorization.decide(intent_ref,ctx.user_id,data.decision),'actor_user_id':ctx.user_id})


@router.post('/authorizations/{intent_ref}/poll')
@router.post('/authorizations/{intent_ref}/cancel')
async def authorization_possession(intent_ref:str,request:Request):
    from contracts.client_operations import FeedbackProof
    owner=service(request);data=await body(request,FeedbackProof,4096)
    action=owner.authorization.cancel if request.url.path.endswith('/cancel') else owner.authorization.poll
    return envelope(owner,action(intent_ref,data.proof))
