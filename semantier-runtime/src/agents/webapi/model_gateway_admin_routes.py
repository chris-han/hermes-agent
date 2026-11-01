from fastapi import APIRouter,Request
from agents.model_gateway.auth import actor_for
from agents.model_gateway.admin import AdminFacade,provider_credential_prefix
from agents.webapi.model_gateway_routes import gateway

router=APIRouter(prefix='/api/admin/model-gateway',tags=['model-gateway-admin'])


@router.get('/models')
async def model_gateway_admin_models(request: Request):
    service=gateway(request); actor=await actor_for(request,service,'provider_credentials:manage',browser_only=True)
    return await AdminFacade(service).models(actor)


@router.post('/models')
async def model_gateway_admin_model_create(request: Request):
    service=gateway(request); actor=await actor_for(request,service,'provider_credentials:manage',browser_only=True)
    return await AdminFacade(service).mutate_model(actor,await request.json())


@router.delete('/models/{model_id}')
async def model_gateway_admin_model_delete(model_id: str,request: Request):
    service=gateway(request); actor=await actor_for(request,service,'provider_credentials:manage',browser_only=True)
    return await AdminFacade(service).mutate_model(actor,{'id':model_id},delete=True)


@router.get('/keys')
async def model_gateway_admin_keys(request: Request):
    service=gateway(request); actor=await actor_for(request,service,'provider_credentials:manage',browser_only=True)
    return await AdminFacade(service).keys(actor)


@router.post('/keys')
async def model_gateway_admin_key_create(request: Request):
    service=gateway(request); actor=await actor_for(request,service,'provider_credentials:manage',browser_only=True)
    return await AdminFacade(service).create_key(actor,await request.json())


@router.get('/usage')
async def model_gateway_admin_usage(request: Request):
    service=gateway(request); actor=await actor_for(request,service,'usage:read_org',browser_only=True)
    rows=await service.billing.list_usage(actor.organization_id)
    for row in rows:
        row['evidence_url']='/api/model-gateway/invocations/'+row['invocation_id']+'/evidence'
    return {'data':rows,'native_spend_basis':'operational_estimate',
            'financial_authority':'semantier'}


@router.get('/context')
async def model_gateway_admin_context(request: Request):
    service=gateway(request)
    actor=await actor_for(request,service,'provider_credentials:manage',browser_only=True)
    from agents.gateway_identity import get_user_organization_context
    current=get_user_organization_context(actor.user_id) or {}
    return {'user_id':actor.user_id,'organization_id':actor.organization_id,
            'organization':{'id':actor.organization_id,'display_name':current.get('organization_name')},
            'permissions':sorted(actor.permissions),
            'provider_credential_prefix':'os.environ/'+provider_credential_prefix(actor.organization_id),
            'allowed_key_models':[d.proxy_model for d in await service.store.deployments(actor.organization_id) if d.enabled]}


@router.post('/models/{model_id}/update')
async def model_gateway_admin_model_update(model_id: str,request: Request):
    service=gateway(request); actor=await actor_for(request,service,'provider_credentials:manage',browser_only=True)
    body=await request.json()
    from fastapi import HTTPException
    if not isinstance(body,dict) or body.get('model_info',{}).get('id',model_id)!=model_id:
        raise HTTPException(422,'model identity mismatch')
    body['model_info']={'id':model_id}
    return await AdminFacade(service).mutate_model(actor,body)


@router.post('/models/{model_id}/block')
async def model_gateway_admin_model_block(model_id: str,request: Request):
    service=gateway(request); actor=await actor_for(request,service,'provider_credentials:manage',browser_only=True)
    return await AdminFacade(service).set_model_blocked(actor,model_id,True)


@router.post('/models/{model_id}/unblock')
async def model_gateway_admin_model_unblock(model_id: str,request: Request):
    service=gateway(request); actor=await actor_for(request,service,'provider_credentials:manage',browser_only=True)
    return await AdminFacade(service).set_model_blocked(actor,model_id,False)


@router.get('/provider-fields')
async def model_gateway_admin_provider_fields(request: Request):
    service=gateway(request); await actor_for(request,service,'provider_credentials:manage',browser_only=True)
    data=await service.proxy.operation('GET','/public/providers/fields')
    return [row for row in data if row.get('litellm_provider') in {'azure','bedrock','volcengine','dashscope'}]


@router.get('/model-cost-map')
async def model_gateway_admin_model_cost_map(request: Request):
    service=gateway(request); await actor_for(request,service,'provider_credentials:manage',browser_only=True)
    data=await service.proxy.operation('GET','/public/litellm_model_cost_map',params={'catalog_only':'true'})
    return {name:row for name,row in data.items() if isinstance(row,dict) and row.get('litellm_provider') in {'azure','bedrock','volcengine','dashscope'}}


# Price source custody and native consumption are separate from customer tariffs.
import httpx
import re
from functools import wraps
from contracts.billing import BillingError
from fastapi import HTTPException


def pricing_errors(handler):
    @wraps(handler)
    async def invoke(*args, **kwargs):
        try:
            return await handler(*args, **kwargs)
        except BillingError as exc:
            raise HTTPException(404 if exc.code=='not_found' else 409, exc.code) from None
        except (ValueError, TypeError) as exc:
            reason=str(exc) if re.fullmatch('[A-Z_]{1,80}',str(exc)) else 'price source or projection validation failed'
            raise HTTPException(422, reason) from None
        except httpx.HTTPError:
            raise HTTPException(503, 'native pricing service unavailable') from None
    return invoke
async def pricing_actor(request, *, operator=False):
    from agents.model_gateway.pricing import PricingAdmin
    service=gateway(request)
    actor=await actor_for(request,service,'billing:manage',browser_only=True)
    admin=PricingAdmin(service)
    if operator:await admin.require_operator(actor)
    return admin,actor


@router.get('/pricing/providers')
@pricing_errors
async def pricing_providers(request: Request):
    admin,actor=await pricing_actor(request)
    return await admin.providers(actor)


@router.post('/pricing/collections')
@pricing_errors
async def pricing_collect(request: Request):
    admin,actor=await pricing_actor(request)
    return await admin.collect(actor,await request.json())


@router.get('/pricing/collections')
@pricing_errors
async def pricing_collections(request: Request):
    admin,actor=await pricing_actor(request)
    limit=int(request.query_params.get('limit','50')); offset=int(request.query_params.get('offset','0'))
    provider=request.query_params.get('provider_id'); source=request.query_params.get('source_ref')
    return {'data':await admin.store.runs(actor.organization_id,limit=limit,offset=offset,provider_id=provider,source_ref=source),'capabilities':await admin.capabilities(actor),'limit':limit,'offset':offset}


@router.get('/pricing/collections/{run_id}')
@pricing_errors
async def pricing_collection(run_id: str,request: Request):
    admin,actor=await pricing_actor(request)
    return await admin.store.run(actor.organization_id,run_id)


@router.get('/pricing/catalogs/{version_id}')
@pricing_errors
async def pricing_catalog(version_id: str,request: Request):
    admin,actor=await pricing_actor(request)
    return await admin.store.snapshot(actor.organization_id,version_id)


@router.post('/pricing/sync')
@pricing_errors
async def pricing_sync(request: Request):
    admin,actor=await pricing_actor(request,operator=True)
    return await admin.sync(actor,await request.json())


@router.get('/pricing/source')
@pricing_errors
async def pricing_source(request: Request):
    admin,actor=await pricing_actor(request)
    return await admin.source(actor,provider_id=request.query_params.get('provider_id'),source_ref=request.query_params.get('source_ref'))


@router.get('/pricing/reload-status')
@pricing_errors
async def pricing_reload_status(request: Request):
    admin,actor=await pricing_actor(request,operator=True)
    return await admin.service.proxy.operation('GET','/schedule/model_cost_map_reload/status')


@router.post('/pricing/reload')
@pricing_errors
async def pricing_reload(request: Request):
    admin,actor=await pricing_actor(request,operator=True)
    return await admin.reload(actor)


@router.post('/pricing/reload-schedule')
@pricing_errors
async def pricing_schedule(request: Request):
    from fastapi import HTTPException
    admin,actor=await pricing_actor(request,operator=True)
    body=await request.json()
    if not isinstance(body,dict) or set(body)!={'hours'} or type(body['hours']) is not int or not 1<=body['hours']<=168:
        raise HTTPException(422,'hours must be an integer in1..168')
    return await admin.service.proxy.operation('POST','/schedule/model_cost_map_reload',params={'hours':body['hours']})


@router.delete('/pricing/reload-schedule')
@pricing_errors
async def pricing_cancel_schedule(request: Request):
    admin,actor=await pricing_actor(request,operator=True)
    return await admin.service.proxy.operation('DELETE','/schedule/model_cost_map_reload')
