"""Browser-only commerce commands, separately authorized native read projection."""
import os
from urllib.parse import urlsplit
from fastapi import APIRouter, HTTPException, Request
from agents.model_gateway.auth import actor_for
from agents.webapi.model_gateway_routes import gateway
from billing.commerce import configured
from contracts.billing import BillingError

router=APIRouter(prefix='/api/billing/commerce',tags=['billing-commerce'])

def commerce(request):
    service=gateway(request)
    if not service or not service.billing: raise HTTPException(503,'billing unavailable')
    return service,configured(service.billing)

async def permitted_account_rows(service,actor,*,manage=False):
    return await service.store.account_rows(actor,service.billing.schema,manage=manage)


async def select_account_row(service,actor,aid=None,currency=None,*,manage=False):
    rows=await permitted_account_rows(service,actor,manage=manage)
    if aid is not None:rows=[r for r in rows if r['account_id']==aid]
    if currency is not None:rows=[r for r in rows if r['currency']==currency]
    if not rows:raise HTTPException(403,'billing account unavailable')
    if len(rows)!=1:raise HTTPException(409,'account_selection_required')
    return rows[0]


async def actor_accounts(request,*,manage=False):
    service,c=commerce(request)
    actor=await actor_for(request,service,'billing:manage' if manage else {'billing:read','billing:manage'},browser_only=manage)
    return service,c,actor,await permitted_account_rows(service,actor,manage=manage)

async def scoped(request,aid=None,*,manage=False):
    service,c,actor,rows=await actor_accounts(request,manage=manage)
    row=await select_account_row(service,actor,aid,request.query_params.get('currency'),manage=manage)
    return c,actor,row['account_id']

async def body(request,fields,optional=()):
    try: value=await request.json()
    except ValueError as exc: raise HTTPException(422,'JSON object required') from exc
    if not isinstance(value,dict) or not set(fields)<=set(value) or set(value)-set(fields)-set(optional): raise HTTPException(422,'explicit supported fields required')
    if any(not isinstance(value[f],str) or not 0<len(value[f])<=512 for f in value if f!='confirmed'):
        raise HTTPException(422,'bounded string fields required')
    return value

async def run(awaitable):
    try:return await awaitable
    except BillingError as exc: raise HTTPException(404,'billing object unavailable') from exc
    except ValueError as exc: raise HTTPException(409,str(exc)) from exc

@router.get('/catalog')
async def catalog(request:Request):
    # Public pricing remains available without gateway, session or onboarding.
    return configured(None).catalog()

@router.get('/accounts')
async def accounts(request:Request):
    _,_,_,rows=await actor_accounts(request)
    return {'data':rows}

@router.post('/quotes')
async def quotes(request:Request):
    b=await body(request,{'account_id','product_id'})
    c,a,aid=await scoped(request,b['account_id'],manage=True)
    return await run(c.quote(a.organization_id,aid,b['product_id']))

@router.post('/orders')
async def create_order(request:Request):
    b=await body(request,{'account_id','quote_id','idempotency_key','confirmed'})
    c,a,aid=await scoped(request,b['account_id'],manage=True)
    return await run(c.order(a.organization_id,aid,b['quote_id'],b['idempotency_key'],b['confirmed']))

@router.get('/orders')
async def list_orders(request:Request,account_id:str|None=None):
    c,a,aid=await scoped(request,account_id)
    return {'data':await run(c.orders(a.organization_id,aid))}

@router.get('/orders/{order_id}')
async def get_order(request:Request,order_id:str,account_id:str|None=None):
    c,a,aid=await scoped(request,account_id)
    return await run(c.get_order(a.organization_id,aid,order_id))

@router.post('/orders/{order_id}/test-payment')
async def test_payment(request:Request,order_id:str):
    b=await body(request,{'account_id','method'})
    c,a,aid=await scoped(request,b['account_id'],manage=True)
    return await run(c.test_payment(a.organization_id,aid,order_id,b['method']))

@router.post('/orders/{order_id}/payment-attempts')
async def start_payment(request:Request,order_id:str):
    b=await body(request,{'account_id','method'},optional={'language'})
    if b.get('language','en') not in {'en','zh'}:raise HTTPException(422,'unsupported language')
    c,a,aid=await scoped(request,b['account_id'],manage=True)
    return await run(c.start_payment(a.organization_id,aid,order_id,b['method'],language=b.get('language','en')))

@router.post('/orders/{order_id}/payment-attempts/{attempt_id}/simulate')
async def simulate_payment(request:Request,order_id:str,attempt_id:str):
    b=await body(request,{'account_id','outcome'})
    c,a,aid=await scoped(request,b['account_id'],manage=True)
    return await run(c.resolve_payment(a.organization_id,aid,order_id,attempt_id,b['outcome']))

@router.get('/state')
async def state(request:Request,account_id:str|None=None):
    c,a,aid=await scoped(request,account_id)
    result=await run(c.state(a.organization_id,aid))
    from agents.gateway_identity import account_provisioning_for_user
    import asyncio
    readiness=await asyncio.to_thread(account_provisioning_for_user,user_id=a.user_id,
        organization_id=a.organization_id,workspace_id=a.workspace_id,account_id=aid)
    if readiness is not None:result['account_provisioning']=readiness
    portal=os.getenv('SEMANTIER_BILLING_PORTAL_URL','')
    p=urlsplit(portal)
    valid=(p.scheme=='https' or (p.scheme=='http' and p.hostname in {'localhost','127.0.0.1'})) and bool(p.netloc) and not p.username and not p.password and p.path in {'','/'} and not p.query and not p.fragment
    result['portal']={'url':portal.rstrip('/') if valid else None}
    permitted=await permitted_account_rows(gateway(request),a)
    can_manage=any(row['account_id']==aid and row['can_manage'] for row in permitted)
    result['capabilities']={'can_read_usage':bool({'usage:read_self','usage:read_org'}&a.permissions),'can_manage_billing':can_manage,'can_open_portal':bool(valid),'can_charge':False,'can_change_plan':False,'can_auto_top_up':False}
    return result


@router.post('/notifications/{method}')
async def channel_notification(request:Request,method:str):
    # Channel signatures authenticate this public edge; no browser return path.
    _,c=commerce(request)
    payload=bytearray()
    async for part in request.stream():
        payload.extend(part)
        if len(payload)>65536:raise HTTPException(413,'notification too large')
    await run(c.notify(method,dict(request.headers),bytes(payload)))
    from fastapi.responses import PlainTextResponse, JSONResponse
    return PlainTextResponse('success') if method=='alipay' else JSONResponse({'code':'SUCCESS','message':'成功'})

@router.post('/orders/{order_id}/payment-attempts/{attempt_id}/query')
async def query_channel(request:Request,order_id:str,attempt_id:str):
    b=await body(request,{'account_id'})
    c,a,aid=await scoped(request,b['account_id'],manage=True)
    return await run(c.query_channel_payment(a.organization_id,aid,order_id,attempt_id))

@router.post('/orders/{order_id}/payment-attempts/{attempt_id}/close')
async def close_channel(request:Request,order_id:str,attempt_id:str):
    b=await body(request,{'account_id'})
    c,a,aid=await scoped(request,b['account_id'],manage=True)
    return await run(c.query_channel_payment(a.organization_id,aid,order_id,attempt_id,close=True))
