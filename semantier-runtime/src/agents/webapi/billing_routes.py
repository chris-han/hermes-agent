"""Currency-safe read surfaces. Commercial mutations have no MVL authority."""
import json
from datetime import datetime, timezone
from fastapi import APIRouter,HTTPException,Request
from agents.model_gateway.auth import actor_for
from agents.webapi.model_gateway_routes import gateway
from agents.webapi.billing_commerce_routes import permitted_account_rows

router=APIRouter(prefix='/api/billing',tags=['billing'])


async def account(request,permission='billing:read',*,browser_only=False):
    service=gateway(request); actor=await actor_for(request,service,permission,browser_only=browser_only)
    rows=await permitted_account_rows(service,actor,manage=isinstance(permission,str) and permission.endswith(':manage'))
    if len(rows)!=1:
        raise HTTPException(409,'billing account unavailable or ambiguous')
    aid,currency,funding=(rows[0][key] for key in ('account_id','currency','funding'))
    return service,actor,aid,currency,funding


@router.get('/state')
async def billing_state(request: Request):
    service,actor,aid,currency,funding=await account(request)
    summary=await service.billing.account_summary(actor.organization_id,aid,currency)
    result={'schema_version':'semantier.billing.v1','backend':'semantier',
        'organization':{'id':actor.organization_id},'billing_account_id':aid,
        'mode':'shadow' if funding=='CUSTOMER_CLOUD' else 'enforced','funding_source':funding,
        'balances':summary.get('balances',[]),'account':summary,
        'capabilities':{'can_read_usage':bool({'usage:read_self','usage:read_org'}&actor.permissions),
                        'can_charge':False,'can_change_plan':False,'can_auto_top_up':False},
        'usage':{'known_cost':{'amount':summary['budget']['spent'],'currency':currency},
                 'status':'PARTIAL' if summary.get('unresolved_attempts') else 'RATED',
                 'unresolved_attempts':summary.get('unresolved_attempts'),'period':'lifetime'},
        'as_of':datetime.now(timezone.utc).isoformat()}
    import asyncio
    from agents.gateway_identity import account_provisioning_for_user
    readiness=await asyncio.to_thread(account_provisioning_for_user,user_id=actor.user_id,
        organization_id=actor.organization_id,workspace_id=actor.workspace_id,account_id=aid,currency=currency)
    if readiness is not None:result['account_provisioning']=readiness
    return result


@router.get('/subscription')
async def billing_subscription(request: Request):
    service,actor,aid,currency,funding=await account(request)
    return {'schema_version':'semantier.billing.v1','backend':'semantier','billing_account_id':aid,
            'status':'unavailable','reason':'commercial subscription integration unavailable',
            'capabilities':{'can_change_plan':False,'can_charge':False}}


@router.get('/usage')
async def billing_usage(request: Request):
    service=gateway(request)
    actor=await actor_for(request,service,{'usage:read_self','usage:read_org'})
    rows=await service.billing.list_usage(actor.organization_id)
    if 'usage:read_org' not in actor.permissions:
        rows=[r for r in rows if r.get('attribution',{}).get('actor_user_id')==actor.user_id]
    for row in rows:
        row['evidence_url']='/api/model-gateway/invocations/'+row['invocation_id']+'/evidence'
    return {'schema_version':'semantier.billing.v1','backend':'semantier','data':rows}


@router.get('/comparison')
async def billing_comparison(request: Request):
    service,actor,aid,currency,funding=await account(request)
    allowed={'provider','raw_usage'}
    if set(request.query_params)-allowed:
        raise HTTPException(422,'unsupported comparison fields')
    if len(request.query_params.get('raw_usage',''))>16384:
        raise HTTPException(413,'comparison usage exceeds bound')
    provider=request.query_params.get('provider')
    try:
        raw=json.loads(request.query_params.get('raw_usage','{}'))
        if not isinstance(raw,dict) or not raw:
            raise ValueError('explicit raw usage scenario required')
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    bindings=[d for d in await service.store.deployments(actor.organization_id)
              if d.enabled and d.provider==provider and d.account_id==aid]
    if not bindings:
        raise HTTPException(404,'no authorized provider price scope')
    data=[]
    try:
        for d in bindings:
            if not d.customer_price_version:
                data.append({'deployment_id':d.deployment_id,'status':'UNPRICED'})
            else:
                comparison=await service.billing.compare([d.customer_price_version],provider,raw,d.fx_version)
                data.append({'deployment_id':d.deployment_id,'comparison':comparison})
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    return {'schema_version':'semantier.billing.v1','backend':'semantier','data':data}


@router.post('/charges')
async def billing_charges(request: Request):
    service=gateway(request); await actor_for(request,service,'billing:manage')
    raise HTTPException(503,'commercial charge integration unavailable')


@router.post('/budget')
async def billing_budget_update(request: Request):
    service,actor,aid,currency,funding=await account(request,'budgets:manage',browser_only=True)
    body=await request.json()
    if not isinstance(body,dict) or set(body)!={'limit'} or not isinstance(body['limit'],dict) or set(body['limit'])!={'amount','currency'}:
        raise HTTPException(422,'explicit limit money required')
    money=body['limit']
    if money['currency']!=currency or not isinstance(money['amount'],str):
        raise HTTPException(422,'budget must use the bound account native currency and decimal string')
    try:
        return await service.billing.set_account_budget(actor.organization_id,aid,currency,money['amount'],actor.user_id)
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.get('/prices')
async def billing_prices(request: Request):
    service=gateway(request)
    actor=await actor_for(request,service,{'billing:read','billing:manage'},browser_only=True)
    # The operator candidate catalog remains v1-compatible. Customer UI requests
    # an explicit account and receives only its enabled deployment bindings.
    if 'account_id' in request.query_params:
        aid=request.query_params['account_id']
        allowed=[r for r in await permitted_account_rows(service,actor) if r['account_id']==aid]
        if len(allowed)!=1:
            raise HTTPException(403,'billing account unavailable')
        scope=allowed[0]
        rows=[];known=set()
        for d in await service.store.deployments(actor.organization_id):
            if (d.enabled and d.account_id==aid and d.currency==scope['currency']
                and d.funding==scope['funding'] and d.customer_price_version
                and d.customer_price_version not in known):
                price=await service.billing.get_price(d.customer_price_version)
                rows.append({'version_id':price.version_id,'kind':'customer','payload':price.to_dict(),
                             'activation':'BOUND_TO_DEPLOYMENT'})
                known.add(price.version_id)
        return {'schema_version':'semantier.billing.v1','backend':'semantier',
                'organization':{'id':actor.organization_id},'billing_account_id':aid,'data':rows}
    rows=await service.billing.list_customer_prices(actor.organization_id)
    known={row['version_id'] for row in rows}
    for d in await service.store.deployments(actor.organization_id):
        if d.enabled and d.customer_price_version and d.customer_price_version not in known:
            price=await service.billing.get_price(d.customer_price_version)
            rows.append({'version_id':price.version_id,'kind':'customer','payload':price.to_dict(),'activation':'BOUND_TO_DEPLOYMENT'})
            known.add(price.version_id)
    return {'schema_version':'semantier.billing.v1','backend':'semantier','data':rows}


@router.post('/prices')
async def billing_price_create(request: Request):
    service=gateway(request)
    actor=await actor_for(request,service,'billing:manage',browser_only=True)
    body=await request.json()
    allowed={'currency','rates','min_input','max_input','unit','provider','model','region','tier'}
    if not isinstance(body,dict) or set(body)-allowed or not {'currency','rates'}<=set(body):
        raise HTTPException(422,'unsupported price fields')
    rates=body['rates']
    if not isinstance(rates,dict) or not rates or set(rates)-{'uncached_input','cache_read_input','cache_write_input','output_total'} or not all(isinstance(v,str) for v in rates.values()):
        raise HTTPException(422,'rates must be explicit native-currency decimal strings')
    for field in ('min_input','max_input','unit'):
        value=body.get(field)
        if field in body and ((value is None and field!='max_input') or (value is not None and (isinstance(value,bool) or not isinstance(value,int)))):
            raise HTTPException(422,'price boundaries and unit must be integers')
    if body.get('unit',1000000)!=1000000:
        raise HTTPException(422,'this price editor supports explicit per-million token rates')
    for field in ('provider','model','region','tier'):
        if field in body and (not isinstance(body[field],str) or not body[field].isascii()):
            raise HTTPException(422,'ASCII price scope required')
    from billing.rating import PriceVersion
    from uuid import uuid4
    try:
        price=PriceVersion(version_id='customer_'+uuid4().hex,**body,
                           provenance='operator-authored:'+actor.user_id+':'+datetime.now(timezone.utc).isoformat())
        return await service.billing.create_customer_price(actor.organization_id,price,actor.user_id)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
