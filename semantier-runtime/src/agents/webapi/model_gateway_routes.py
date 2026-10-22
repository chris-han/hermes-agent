from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Request
from agents.model_gateway.auth import actor_for

router=APIRouter(tags=['model-gateway'])


def gateway(request):
    service=getattr(request.app.state,'model_gateway',None)
    if not service or not service.enabled:
        raise HTTPException(503,'model gateway disabled')
    return service


async def legacy_when_disabled(request,upstream_path):
    service=getattr(request.app.state,'model_gateway',None)
    if service and service.enabled:
        return None
    if request.headers.get('authorization','').startswith('Bearer smg_'):
        raise HTTPException(503,'model gateway disabled')
    from agents.webapi.embedded_proxy_routes import proxy_v1
    return await proxy_v1(request,upstream_path)


@router.get('/v1/models')
async def model_gateway_models(request: Request):
    legacy=await legacy_when_disabled(request,'models')
    if legacy is not None:
        return legacy
    service=gateway(request)
    actor=await actor_for(request,service,'model:invoke')
    return {'object':'list','data':[{'id':d.alias,'object':'model','owned_by':'semantier'}
        for d in await service.store.deployments(actor.organization_id) if d.enabled]}


@router.post('/v1/chat/completions')
async def model_gateway_chat(request: Request):
    legacy=await legacy_when_disabled(request,'chat/completions')
    if legacy is not None:
        return legacy
    service=gateway(request)
    actor=await actor_for(request,service,'model:invoke')
    wire=bytearray()
    async for chunk in request.stream():
        wire.extend(chunk)
        if len(wire)>1048576:
            raise HTTPException(413,'chat request exceeds 1 MiB')
    try:
        body=json.loads(wire)
    except ValueError as exc:
        raise HTTPException(400,'JSON required') from exc
    correlation={key:request.headers.get(header) for key,header in [('session_id','x-semantier-session-id'),('run_id','x-semantier-run-id'),('purpose','x-semantier-purpose')] if request.headers.get(header)}
    if any(len(value)>200 for value in correlation.values()):
        raise HTTPException(422,'correlation exceeds bound')
    return await service.chat(actor,body,request.headers.get('idempotency-key'),correlation)


@router.post('/api/model-gateway/credentials')
async def model_gateway_issue_credential(request: Request):
    service=gateway(request)
    actor=await actor_for(request,service,'model:invoke',browser_only=True)
    body=await request.json()
    if not isinstance(body,dict) or set(body)-{'permissions','expires_in_seconds'}:
        raise HTTPException(422,'unsupported credential fields')
    seconds=body.get('expires_in_seconds',3600)
    permissions=body.get('permissions',['model:invoke'])
    if isinstance(seconds,bool) or not isinstance(seconds,int) or not 60<=seconds<=86400*30:
        raise HTTPException(422,'expiry must be 60..2592000 seconds')
    if not isinstance(permissions,list) or not all(isinstance(p,str) for p in permissions) or not set(permissions)<={'model:invoke','usage:read_self','billing:read'}:
        raise HTTPException(403,'unsupported SDK scope')
    try:
        result=await service.store.issue(organization_id=actor.organization_id,user_id=actor.user_id,
            workspace_id=actor.workspace_id,permissions=permissions,expires_at=datetime.now(timezone.utc)+timedelta(seconds=seconds))
    except (ValueError,PermissionError) as exc:
        raise HTTPException(403,str(exc)) from exc
    await service.store.audit(actor.organization_id,actor.user_id,'credential.issue',result['token_id'])
    return result


@router.delete('/api/model-gateway/credentials/{token_id}')
async def model_gateway_revoke_credential(token_id: str,request: Request):
    service=gateway(request)
    actor=await actor_for(request,service,'model:invoke',browser_only=True)
    if not await service.store.revoke(actor.organization_id,actor.user_id,token_id):
        raise HTTPException(404,'credential not found')
    await service.store.audit(actor.organization_id,actor.user_id,'credential.revoke',token_id)
    return {'revoked':True}


@router.get('/api/model-gateway/invocations/{invocation_id}')
async def model_gateway_invocation(invocation_id: str,request: Request):
    service=gateway(request)
    actor=await actor_for(request,service,{'usage:read_self','usage:read_org'})
    try:
        result=await service.billing.get_invocation(actor.organization_id,invocation_id)
    except ValueError as exc:
        raise HTTPException(404,'invocation not found') from exc
    if not result or (result.get('attribution',{}).get('actor_user_id') != actor.user_id and 'usage:read_org' not in actor.permissions):
        raise HTTPException(404,'invocation not found')
    return result


@router.get('/api/model-gateway/invocations/{invocation_id}/evidence')
async def model_gateway_invocation_evidence(invocation_id: str,request: Request):
    service=gateway(request)
    actor=await actor_for(request,service,{'usage:read_self','usage:read_org'})
    try:
        invocation=await service.billing.get_invocation(actor.organization_id,invocation_id)
    except ValueError as exc:
        raise HTTPException(404,'invocation not found') from exc
    if invocation.get('attribution',{}).get('actor_user_id')!=actor.user_id and 'usage:read_org' not in actor.permissions:
        raise HTTPException(404,'invocation not found')
    bundle=await service.billing.export_invocation(actor.organization_id,invocation_id)
    return {'schema_version':'semantier.invocation-evidence.v1','invocation_id':invocation_id,
            'bundle':bundle,'hold_reference':{'invocation_id':invocation_id,'active':invocation.get('hold_active'),
                'amount':{'amount':invocation['max_cost'],'currency':invocation['currency']},
                'settlement_state':invocation['settlement_state']},
            'links':{'invocation':'/api/model-gateway/invocations/'+invocation_id,
                     'usage':'/api/billing/usage','admin_usage':'/api/admin/model-gateway/usage'}}
