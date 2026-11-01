"""Bounded first-party observation ingress; server identity/facts are never client fields."""
from datetime import datetime, timezone
import asyncio
import hashlib
import hmac
import json
import time
from typing import Literal
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from agents.auth_session import _session_secret
from agents.webapi.deps import enforce_rate_limit, request_context
from agents.operations_journey import canonical_event
from agents.operations_store import append_event, capture_fact

router=APIRouter(tags=['journey-observations'])
COOKIE='semantier_journey'
MAX_AGE=30*86400


def read_journey(request):
    token=request.cookies.get(COOKIE,'')
    try:
        value,signature=token.rsplit('.',1)
        jid,issued=value.split(':')
        if len(jid)!=32 or not all(c in '0123456789abcdef' for c in jid):return None
        if not 0<=time.time()-int(issued)<=MAX_AGE:return None
        expected=hmac.new(_session_secret(),value.encode(),hashlib.sha256).hexdigest()
        return 'anon:'+jid if hmac.compare_digest(expected,signature) else None
    except (ValueError,TypeError):return None


def response(value):return JSONResponse(value,headers={'Cache-Control':'no-store','Vary':'Cookie'})


def issue_journey_cookie(out, request, subject=None):
    value=(subject.removeprefix('anon:') if subject else uuid4().hex)+':'+str(int(time.time()))
    token=value+'.'+hmac.new(_session_secret(),value.encode(),hashlib.sha256).hexdigest()
    out.set_cookie(COOKIE,token,httponly=True,secure=request.url.scheme=='https',samesite='lax',max_age=MAX_AGE,path='/')


@router.post('/api/journey/session')
async def journey_session(request:Request):
    enforce_rate_limit('auth_public',request=request)
    current=read_journey(request)
    out=response({'status':'ready'})
    if not current:
        issue_journey_cookie(out,request)
    return out


@router.post('/api/journey/observations/{surface}')
async def observe(request:Request,surface:Literal['landing','workspace','windows']):
    enforce_rate_limit('auth_public',request=request)
    raw=bytearray()
    async for part in request.stream():
        raw.extend(part)
        if len(raw)>4096:raise HTTPException(413,'Observation too large')
    subject=read_journey(request)
    if subject is None:raise HTTPException(409,'Journey session required')
    ctx=await asyncio.to_thread(request_context,request)
    if ctx.authenticated:await asyncio.to_thread(link_authenticated_journey,request,{'user_id':ctx.user_id})
    try:
        event=canonical_event(json.loads(raw),surface=surface,subject_ref=subject,now=datetime.now(timezone.utc))
    except (ValueError,TypeError):
        from agents.operations_store import record_rejected_observation
        await asyncio.to_thread(record_rejected_observation,surface)
        raise HTTPException(422,'Unsupported observation') from None
    from agents.operations_store import append_observation
    try:outcome=await asyncio.to_thread(append_observation,event,verified_user_id=ctx.user_id if ctx.authenticated else None)
    except ValueError:raise HTTPException(409,'Event identity collision') from None
    return response({'status':outcome})


def link_authenticated_journey(request,user):
    journey=read_journey(request)
    if journey:
        capture_fact(name='identity_linked',user_id=user['user_id'],producer='identity',receipt_ref=journey+':'+user['user_id'],linked_subject_ref=journey)


@router.post('/api/trial/application')
async def apply_trial(request:Request):
    enforce_rate_limit('auth_public',request=request)
    subject=read_journey(request)
    if subject is None:raise HTTPException(409,'Journey session required')
    from agents.trial_invitations import apply
    import asyncio
    return response(await asyncio.to_thread(apply,subject))


@router.get('/api/trial/application')
async def read_trial(request:Request):
    enforce_rate_limit('auth_public',request=request)
    subject=read_journey(request)
    if subject is None:return response({'status':'unknown'})
    from agents.trial_invitations import current
    import asyncio
    return response(await asyncio.to_thread(current,subject))


@router.post('/api/journey/handoff')
async def mint_handoff(request:Request):
    enforce_rate_limit('auth_public',request=request)
    subject=read_journey(request)
    if not subject:raise HTTPException(409,'Journey session required')
    from agents.auth_db_postgres import connection
    import secrets
    token=secrets.token_urlsafe(32)
    with connection() as c:
        count=c.execute("SELECT count(*) AS n FROM semantier_operations.handoffs WHERE journey_ref=%s AND expires_at>now()",(subject,)).fetchone()['n']
        if count>=8:raise HTTPException(429,'Handoff limit reached')
        c.execute("INSERT INTO semantier_operations.handoffs VALUES(%s,%s,now()+interval '2 minutes',NULL)",(hashlib.sha256(token.encode()).hexdigest(),subject))
    return response({'handoff_ref':token,'expires_in':120})


@router.post('/api/journey/handoff/claim')
async def claim_handoff(request:Request):
    enforce_rate_limit('auth_public',request=request)
    raw=bytearray()
    async for part in request.stream():
        raw.extend(part)
        if len(raw)>512:raise HTTPException(413,'Handoff too large')
    try:
        body=json.loads(raw)
        if set(body)!={'handoff_ref'} or not isinstance(body['handoff_ref'],str) or len(body['handoff_ref'])!=43:raise ValueError()
    except (ValueError,TypeError):raise HTTPException(422,'Bounded handoff reference required') from None
    from agents.auth_db_postgres import connection
    with connection() as c:
        row=c.execute('SELECT * FROM semantier_operations.handoffs WHERE ref_hash=%s AND expires_at>now() FOR UPDATE',(hashlib.sha256(body['handoff_ref'].encode()).hexdigest(),)).fetchone()
        if row is None or row['claimed_at'] and read_journey(request)!=row['journey_ref']:raise HTTPException(409,'Handoff unavailable')
        c.execute('UPDATE semantier_operations.handoffs SET claimed_at=coalesce(claimed_at,now()) WHERE ref_hash=%s',(row['ref_hash'],))
    out=response({'status':'linked_anonymous_journey'})
    issue_journey_cookie(out,request,row['journey_ref'])
    return out
