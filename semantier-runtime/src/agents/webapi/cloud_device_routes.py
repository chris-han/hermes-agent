"""Cloud counterpart of the native device protocol; secrets never enter browser reads."""
import json
import os
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from contracts.cloud_device import Wire, DeviceStart, DeviceProof, DeviceContinue, DeviceApprove, DeviceClaim, DeviceAck, CONTRACT
from agents.model_gateway.auth import actor_for
from agents.model_gateway.device_connections import DeviceConnections, origin
from agents.webapi.model_gateway_routes import gateway
from agents.webapi.billing_commerce_routes import permitted_account_rows

router = APIRouter(prefix='/api/cloud-provider/device', tags=['cloud-device'])


def owner(request):
    service = gateway(request)
    ring = getattr(getattr(service, 'account_sync_store', None), 'keyring', None)
    return service, DeviceConnections(service.store, ring)


async def body(request, model):
    wire = bytearray()
    async for chunk in request.stream():
        wire.extend(chunk)
        if len(wire)>8192: raise HTTPException(413, 'device_body_too_large')
    try: return model.model_validate(json.loads(wire))
    except (ValueError, TypeError): raise HTTPException(422, 'invalid_device_request') from None


def result(value):
    return JSONResponse(value, headers={'Cache-Control':'no-store', 'Pragma':'no-cache', 'Referrer-Policy':'no-referrer'})


async def browser_actor(request):
    return await actor_for(request, gateway(request), 'model:invoke', browser_only=True)


async def select_account(service, actor, aid=None):
    rows = await permitted_account_rows(service, actor)
    if aid is not None: rows = [r for r in rows if r['account_id']==aid]
    if not rows: raise HTTPException(403, 'account_unavailable')
    if len(rows)!=1: raise HTTPException(409, 'account_selection_required')
    return rows[0]


@router.post('/start')
async def start(request: Request):
    from agents.webapi.deps import enforce_rate_limit
    enforce_rate_limit('auth_public', request=request)
    _, devices = owner(request)
    command = await body(request, DeviceStart)
    portal = origin(os.environ.get('SEMANTIER_BILLING_PORTAL_URL',''))
    return result(await devices.start(command, portal))


# Fixed paths precede /{intent_ref}.
@router.post('/revoke')
async def revoke(request: Request):
    service, _ = owner(request)
    actor = await actor_for(request, service, 'model:invoke')
    await body(request, Wire)
    if not actor.token_id or actor.account_id is None: raise HTTPException(403, 'device_credential_required')
    await service.store.revoke(actor.organization_id, actor.user_id, actor.token_id)
    return result({'contract_version':CONTRACT, 'status':'revoked', 'token_id':actor.token_id})


@router.post('/renew')
async def renew(request: Request):
    service, _ = owner(request)
    actor = await actor_for(request, service, 'model:invoke')
    await body(request, Wire)
    if not actor.token_id or actor.account_id is None: raise HTTPException(403, 'device_credential_required')
    return result({'contract_version':CONTRACT, 'status':'reauthorization_required', 'start_path':'/api/cloud-provider/device/start'})


@router.get('/readiness')
async def readiness(request: Request):
    service, _ = owner(request)
    actor = await actor_for(request, service, {'model:invoke','billing:read'})
    if not actor.token_id or actor.account_id is None: raise HTTPException(403, 'device_credential_required')
    from billing.commerce import configured
    commercial = await configured(service.billing).state(actor.organization_id, actor.account_id)
    import asyncio
    from agents.gateway_identity import account_provisioning_for_user
    provisioning = await asyncio.to_thread(account_provisioning_for_user, user_id=actor.user_id,
        organization_id=actor.organization_id, workspace_id=actor.workspace_id, account_id=actor.account_id, currency=actor.currency)
    # Metadata/catalog calls have no inference side effect. Absence is unavailable,
    # never a fabricated model, settled cost or readiness claim.
    catalog = {'status':'unavailable', 'data':[]}
    try:
        if service.proxy is not None:
            models = await service.catalog(actor)
            catalog = {'status':'available', 'data':models['data']}
    except HTTPException:
        pass
    async with await service.store._connection() as c:
        installed = await (await c.execute("SELECT intent_ref,status,installed_generation FROM semantier_gateway.device_connections WHERE token_id=%s", (actor.token_id,))).fetchone()
    return result({'contract_version':CONTRACT, 'account_id':actor.account_id, 'currency':actor.currency,
        'payment_entitlement_allowance':commercial, 'account_provisioning':provisioning,
        'catalog':catalog, 'device':installed,
        'as_of':datetime.now(timezone.utc).isoformat(), 'renewal':'reauthorization_required'})


@router.get('/{intent_ref}')
async def browser_read(request: Request, intent_ref: str):
    service, devices = owner(request)
    actor = await browser_actor(request)
    value = await devices.browser(intent_ref, actor)
    value['data']['accounts'] = await permitted_account_rows(service, actor)
    return result(value)


@router.post('/{intent_ref}/continue')
async def browser_continue(request: Request, intent_ref: str):
    service, devices = owner(request)
    actor = await browser_actor(request)
    command = await body(request, DeviceContinue)
    account = await select_account(service, actor, command.account_id)
    continuation = command.model_dump(exclude={'contract_version'})
    continuation['account_id'] = account['account_id']
    if command.order_id:
        from billing.commerce import configured
        from contracts.billing import BillingError
        try: await configured(service.billing).get_order(actor.organization_id, account['account_id'], command.order_id)
        except (ValueError, BillingError): raise HTTPException(403, 'order_unavailable') from None
    return result(await devices.browser(intent_ref, actor, continuation=continuation, account=account))


@router.post('/{intent_ref}/approve')
async def approve(request: Request, intent_ref: str):
    service, devices = owner(request)
    actor = await browser_actor(request)
    command = await body(request, DeviceApprove)
    account = await select_account(service, actor, command.account_id)
    return result(await devices.browser(intent_ref, actor, account=account, decision=command.decision))


@router.post('/{intent_ref}/poll')
async def poll(request: Request, intent_ref: str):
    _, devices = owner(request)
    command = await body(request, DeviceProof)
    return result(await devices.poll(intent_ref, command.proof))


@router.post('/{intent_ref}/claim')
async def claim(request: Request, intent_ref: str):
    _, devices = owner(request)
    command = await body(request, DeviceClaim)
    inference = origin(os.environ.get('SEMANTIER_CLOUD_PUBLIC_ORIGIN',''))
    # Billing reads use the authenticated backend API origin, not the portal UI.
    return result(await devices.claim(intent_ref, command, inference, inference))


@router.post('/{intent_ref}/ack')
async def ack(request: Request, intent_ref: str):
    _, devices = owner(request)
    command = await body(request, DeviceAck)
    return result(await devices.claim(intent_ref, command, '', '', ack=True))


@router.post('/{intent_ref}/cancel')
async def cancel(request: Request, intent_ref: str):
    _, devices = owner(request)
    command = await body(request, DeviceProof)
    return result(await devices.cancel(intent_ref, command.proof))
