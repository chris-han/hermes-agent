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
        for d in await service.authorized_deployments(actor)]}


@router.get('/api/cloud-provider/catalog')
async def cloud_provider_catalog(request: Request):
    service=gateway(request)
    actor=await actor_for(request,service,'model:invoke')
    from agents.auth_db import get_workspace_config
    from agents.model_gateway.selection import assignment_projection,supported_tasks
    from starlette.concurrency import run_in_threadpool
    payload=await service.catalog(actor,account_id=request.query_params.get('account_id'))
    config=await run_in_threadpool(get_workspace_config,actor.workspace_id)
    return {**payload,'organization_id':actor.organization_id,'assignments':assignment_projection(config,actor.organization_id),'supported_purposes':supported_tasks()}


@router.get('/api/cloud-provider/model-info')
async def cloud_provider_model_info(request: Request, model: str):
    service=gateway(request)
    actor=await actor_for(request,service,'model:invoke')
    return await service.catalog(actor,model,account_id=request.query_params.get('account_id'))


@router.get('/api/model/options')
async def cloud_model_options(request: Request, refresh: bool = False):
    from agents.webapi.deps import require_authenticated_context
    from runtime_paths import bind_workspace_env
    from starlette.concurrency import run_in_threadpool
    ctx=require_authenticated_context(request)
    def options():
        from hermes_cli.inventory import build_model_options_payload,load_picker_context
        with bind_workspace_env(ctx.hermes_home):
            return build_model_options_payload(load_picker_context(),refresh=refresh)
    return await run_in_threadpool(options)


@router.patch('/api/config')
async def hosted_configuration_patch(request: Request):
    from agents.webapi.deps import require_route_authorization
    from agents.route_policy import RouteAuthorizationClass
    from agents.model_gateway.auth import csrf
    from agents.model_gateway.selection import patch_configuration
    ctx=require_route_authorization(request,authorization_class=RouteAuthorizationClass.TENANT_MEMBER,route_hint='PATCH /api/config')
    csrf(request)
    return await patch_configuration(ctx,await request.json())


@router.post('/api/model/set')
async def cloud_model_set(request: Request):
    service=gateway(request)
    actor=await actor_for(request,service,'model:invoke',browser_only=True)
    from agents.webapi.deps import require_authenticated_context
    from agents.model_gateway.selection import set_assignment
    return await set_assignment(service,actor,require_authenticated_context(request),await request.json())


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
    from contracts.model_gateway import TASK_HEADERS
    correlation={key:request.headers[header] for key,header in TASK_HEADERS.items() if request.headers.get(header)}
    return await service.chat(actor,body,request.headers.get('idempotency-key'),correlation,
        account_id=request.headers.get('x-semantier-account-id'),
        expected_profile_revision=request.headers.get('x-semantier-capability-revision'))



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
    if result and actor.account_id is not None and (result['account_id'],result['currency']) != (actor.account_id,actor.currency):
        raise HTTPException(404,'invocation not found')
    scopes={(r['account_id'],r['currency']) for r in await service.store.account_rows(actor,service.billing.schema)}
    if result and (result['account_id'],result['currency']) not in scopes:raise HTTPException(404,'invocation not found')
    if not result or (result.get('attribution',{}).get('actor_user_id') != actor.user_id and 'usage:read_org' not in actor.permissions):
        raise HTTPException(404,'invocation not found')
    lineage=await service.billing.reexecution_lineage(actor.organization_id,invocation_id)
    result = {**result,'reexecution':lineage} if lineage else result
    if 'billing:manage' not in actor.permissions:
        from billing.visibility import customer_projection
        result = customer_projection(result)
    return result


@router.get('/api/model-gateway/invocations/{invocation_id}/evidence')
async def model_gateway_invocation_evidence(invocation_id: str,request: Request):
    service=gateway(request)
    actor=await actor_for(request,service,{'usage:read_self','usage:read_org'})
    try:
        invocation=await service.billing.get_invocation(actor.organization_id,invocation_id)
    except ValueError as exc:
        raise HTTPException(404,'invocation not found') from exc
    if actor.account_id is not None and (invocation['account_id'],invocation['currency']) != (actor.account_id,actor.currency):
        raise HTTPException(404,'invocation not found')
    if invocation.get('attribution',{}).get('actor_user_id')!=actor.user_id and 'usage:read_org' not in actor.permissions:
        raise HTTPException(404,'invocation not found')
    scopes={(r['account_id'],r['currency']) for r in await service.store.account_rows(actor,service.billing.schema)}
    if (invocation['account_id'],invocation['currency']) not in scopes:raise HTTPException(404,'invocation not found')
    bundle=await service.billing.export_invocation(actor.organization_id,invocation_id)
    if 'billing:manage' not in actor.permissions:
        from billing.visibility import customer_projection
        bundle = customer_projection(bundle)
    lineage=await service.billing.reexecution_lineage(actor.organization_id,invocation_id)
    quality = {'status': 'UNAVAILABLE', 'reason': 'qualified_quality_reference_unavailable', 'observations': []}
    attribution = invocation.get('attribution', {})
    session_id = attribution.get('session_id', '')
    if (actor.workspace_id and attribution.get('workspace_id') == actor.workspace_id
            and attribution.get('actor_user_id') == actor.user_id
            and session_id.startswith(actor.workspace_id + ':')):
        import asyncio
        from agents.auth_session import RequestContext
        from runtime_paths import workspace_root_path, workspace_hermes_home_path
        from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
        from services.learning_trace_evidence_service import LearningTraceEvidenceService
        def read_feedback():
            context = RequestContext(True, actor.user_id, actor.workspace_id, actor.workspace_id,
                workspace_root_path(actor.workspace_id), workspace_hermes_home_path(actor.workspace_id),
                organization_id=actor.organization_id, membership_status='active')
            store = OntologyBuildStore(ontology_build_store_path(context.workspace_root))
            try:
                return [row.model_dump(mode='json') for row in
                    LearningTraceEvidenceService(store=store).routing_feedback(context=context, session_id=session_id)
                    if row.invocation_id == invocation_id]
            finally:
                store.close()
        quality['observations'] = await asyncio.to_thread(read_feedback)
        # Observations, including synthetic protocol labels, are never a quality qualification.
    return {'quality_evidence': quality, **({'reexecution':lineage} if lineage else {}),'schema_version':'semantier.invocation-evidence.v1','invocation_id':invocation_id,
            'bundle':bundle,'hold_reference':{'invocation_id':invocation_id,'active':invocation.get('hold_active'),
                'amount':{'amount':invocation['max_cost'],'currency':invocation['currency']},
                'settlement_state':invocation['settlement_state']},
            'links':{'invocation':'/api/model-gateway/invocations/'+invocation_id,
                     'usage':'/api/billing/usage','admin_usage':'/api/admin/model-gateway/usage'}}


@router.post('/api/model-gateway/invocations/{invocation_id}/reexecutions')
async def model_gateway_reexecute(invocation_id: str, request: Request):
    service=gateway(request)
    actor=await actor_for(request,service,'model:invoke')
    wire=bytearray()
    async for chunk in request.stream():
        wire.extend(chunk)
        if len(wire)>1050624: raise HTTPException(413,'reexecution command exceeds bound')
    try: command=json.loads(wire)
    except ValueError as exc: raise HTTPException(400,'JSON required') from exc
    return await service.reexecute(actor,invocation_id,command,request.headers.get('idempotency-key'))
