"""Currency-safe read surfaces. Commercial mutations have no MVL authority."""
import json
from datetime import datetime, timezone
from fastapi import APIRouter,HTTPException,Request
from agents.model_gateway.auth import actor_for
from agents.webapi.model_gateway_routes import gateway
from agents.webapi.billing_commerce_routes import permitted_account_rows,select_account_row

router=APIRouter(prefix='/api/billing',tags=['billing'])


async def account(request,permission='billing:read',*,browser_only=False):
    service=gateway(request); actor=await actor_for(request,service,permission,browser_only=browser_only)
    row=await select_account_row(service,actor,request.query_params.get('account_id'),request.query_params.get('currency'),manage=isinstance(permission,str) and permission.endswith(':manage'))
    aid,currency,funding=(row[key] for key in ('account_id','currency','funding'))
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
    if funding == 'CUSTOMER_CLOUD' and 'billing:manage' not in actor.permissions:
        # Shadow budget tracks supplier spending, not a customer tariff.
        result['account'] = {**summary, 'budget': {'status': 'UNAVAILABLE'}}
        result['usage']['known_cost']['amount'] = None
        result['usage']['status'] = 'UNAVAILABLE'
        result['usage']['reason'] = 'supplier_basis_requires_management'
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
    allowed=await permitted_account_rows(service,actor)
    aid=request.query_params.get('account_id');currency=request.query_params.get('currency')
    if aid is not None:allowed=[r for r in allowed if r['account_id']==aid]
    if currency is not None:allowed=[r for r in allowed if r['currency']==currency]
    if not allowed:raise HTTPException(403,'billing account unavailable')
    query=dict(request.query_params)
    try:
        limit=int(query.pop('limit','100'));offset=int(query.pop('offset','0'))
        if not 1<=limit<=500 or offset<0:raise ValueError('invalid pagination')
        query.pop('account_id',None);query.pop('currency',None)
        rows=await service.billing.list_usage(actor.organization_id,limit=limit+1,offset=offset,
            accounts=[(r['account_id'],r['currency']) for r in allowed],
            actor_user_id=None if 'usage:read_org' in actor.permissions else actor.user_id,filters=query)
    except (ValueError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc
    more=len(rows)>limit;rows=rows[:limit]
    if 'billing:manage' not in actor.permissions:
        from billing.visibility import customer_projection
        rows = customer_projection(rows)
    for row in rows:
        row['evidence_url']='/api/model-gateway/invocations/'+row['invocation_id']+'/evidence'
    return {'schema_version':'semantier.billing.v1','backend':'semantier','data':rows,
        'page':{'limit':limit,'offset':offset,'has_more':more,'next_offset':offset+limit if more else None},
        'completeness':'PAGE_ONLY' if more or offset else 'SCOPED_COMPLETE',
        'tags_overlap':True,'source':'semantier_usage_observations'}



@router.get('/comparison')
async def billing_comparison(request: Request):
    service,actor,aid,currency,funding=await account(request)
    if request.query_params.get('kind'):
        return await optimization_comparison(request, service, actor, aid, currency)
    allowed={'provider','raw_usage','account_id','currency'}
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


async def optimization_comparison(request, service, actor, aid, currency):
    from billing.optimization import price_comparison
    from contracts.billing import BillingError
    query = dict(request.query_params)
    allowed = {'kind', 'basis', 'account_id', 'currency', 'baseline', 'candidate', 'raw_usage',
               'from', 'to', 'root_run_id', 'purpose', 'catalog_version', 'supplier_assumptions'}
    if set(query) - allowed:
        raise HTTPException(422, 'unsupported comparison fields')
    kind, basis = query.get('kind'), query.get('basis', 'CUSTOMER')
    if kind not in {'same_usage_estimate', 'what_if', 'observed_results'} or basis not in {'CUSTOMER', 'SUPPLIER'}:
        raise HTTPException(422, 'unsupported comparison kind or basis')
    if basis == 'SUPPLIER' and 'billing:manage' not in actor.permissions:
        raise HTTPException(403, 'supplier basis requires billing management')
    if not {'usage:read_self', 'usage:read_org'} & actor.permissions:
        raise HTTPException(403, 'usage permission required')
    deployments = [d for d in await service.authorized_deployments(actor)
                   if d.account_id == aid and d.currency == currency]
    options = [{'id': d.deployment_id, 'label': d.alias, 'revision': d.revision} for d in deployments]
    result = {'schema_version': 'semantier.optimization.v1', 'kind': kind, 'basis': basis,
        'scope': {'organization_id': actor.organization_id, 'account_id': aid, 'currency': currency,
                  'filters': {k: query[k] for k in ('from', 'to', 'root_run_id', 'purpose') if query.get(k)}},
        'capabilities': {'supplier_basis': 'billing:manage' in actor.permissions,
            'save_draft': False, 'publish': False, 'reason': 'policy_publication_owner_unavailable'},
        'candidates': options, 'data': [], 'quality': 'UNMEASURED', 'business_benefit': 'UNMEASURED',
        'status': 'UNAVAILABLE', 'reason': None, 'coverage': {'covered': 0, 'required': 0},
        'actual_with_without': {'status': 'UNAVAILABLE', 'reason': 'qualified_observed_cohorts_unavailable'}}
    if 'billing:manage' in actor.permissions:
        result['supplier_catalogs'] = await service.billing.supplier_catalog_options()
    catalog = None
    assumptions = None
    if query.get('catalog_version') or query.get('supplier_assumptions'):
        if basis != 'SUPPLIER':
            raise HTTPException(422, 'catalog scenarios require supplier basis')
        try:
            encoded = query.get('supplier_assumptions', '')
            if len(encoded) > 8192:
                raise ValueError('supplier assumptions exceed bound')
            assumptions = json.loads(encoded)
            if not isinstance(assumptions, dict):
                raise ValueError('explicit supplier assumptions required')
            catalog = await service.billing.get_supplier_catalog(query['catalog_version'])
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, 'pinned catalog and explicit supplier assumptions required') from exc
    if kind == 'observed_results':
        result['reason'] = 'qualified_observed_cohorts_unavailable'
        return result
    by_id = {d.deployment_id: d for d in deployments}
    baseline, candidate = by_id.get(query.get('baseline')), by_id.get(query.get('candidate'))
    if not baseline or not candidate:
        if query.get('baseline') or query.get('candidate'):
            raise HTTPException(403, 'comparison deployment outside authorized scope')
        result['reason'] = 'select_current_baseline_and_candidate'
        return result
    price_kind = 'supplier' if basis == 'SUPPLIER' else 'customer'
    async def price(d):
        version = d.supplier_price_version if basis == 'SUPPLIER' else d.customer_price_version
        if not version:
            return None
        try:
            return await service.billing.get_price(version, price_kind)
        except BillingError:
            return None
    old, new = await price(baseline), await price(candidate)
    valuation_at = datetime.now(timezone.utc).isoformat()
    result['valuation_at'] = valuation_at
    if kind == 'what_if':
        encoded = query.get('raw_usage', '')
        if len(encoded) > 16384:
            raise HTTPException(413, 'scenario exceeds bound')
        try:
            raw = json.loads(encoded)
            if not isinstance(raw, dict) or not raw:
                raise ValueError('explicit scenario required')
        except ValueError as exc:
            raise HTTPException(422, 'invalid scenario') from exc
        rows = [{'invocation_id': None, 'usage': {'raw_usage': raw},
                 'attribution': {'provider': baseline.provider,
                     'normalizer_version': 'native-usage.v1',
                     'inclusion_rules_version': 'provider-inclusion.v1'}}]
        result['usage_basis'] = 'USER_ASSUMED_TOTALS_V1'
        result['deployment_normalizer_version'] = baseline.normalizer_version
    else:
        rows = await service.billing.list_usage(actor.organization_id, limit=501, accounts=[(aid, currency)],
            actor_user_id=None if 'usage:read_org' in actor.permissions else actor.user_id,
            filters=result['scope']['filters'])
        result['completeness'] = 'PARTIAL_LIMIT' if len(rows) > 500 else 'SCOPED_COMPLETE'
        rows = rows[:500]
    for row in rows:
        usage = row.get('usage') or {}
        raw = usage.get('raw_usage')
        if not isinstance(raw, dict):
            item = {'status': 'UNAVAILABLE', 'amount': None, 'difference': None, 'reason': 'RAW_USAGE_UNAVAILABLE'}
        elif kind != 'what_if' and (row.get('attribution') or {}).get('provider') != baseline.provider:
            item = {'status': 'UNAVAILABLE', 'amount': None, 'difference': None, 'reason': 'HISTORICAL_PROVIDER_BINDING_CHANGED'}
        elif kind != 'what_if' and row.get('deployment_id') != baseline.deployment_id:
            item = {'status': 'UNAVAILABLE', 'amount': None, 'difference': None, 'reason': 'DIFFERENT_BASELINE_DEPLOYMENT'}
        else:
            attribution = row.get('attribution') or {}
            retained = usage.get('normalized') or {}
            legacy_v2 = (attribution.get('usage_normalization_policy') == 'dashscope-openai-details.v2'
                and baseline.provider in ('dashscope', 'alibaba') and 'completion_tokens_details' in raw)
            item = price_comparison(valuation_at=valuation_at, raw_usage=raw, provider=baseline.provider, baseline=old, candidate=new,
                kind=kind, basis=basis, normalizer_version=retained.get('normalizer_version') or attribution.get('normalizer_version') or ('native-usage.v2' if legacy_v2 else 'native-usage.v1'),
                inclusion_rules_version=retained.get('inclusion_rules_version') or attribution.get('inclusion_rules_version') or ('provider-inclusion.v2' if legacy_v2 else 'provider-inclusion.v1'))
            if catalog is not None:
                from billing.optimization import catalog_comparison
                item = catalog_comparison(catalog=catalog, raw_usage=raw, provider=baseline.provider,
                    normalizer_version=retained.get('normalizer_version') or attribution.get('normalizer_version') or ('native-usage.v2' if legacy_v2 else 'native-usage.v1'),
                    inclusion_rules_version=retained.get('inclusion_rules_version') or attribution.get('inclusion_rules_version') or ('provider-inclusion.v2' if legacy_v2 else 'provider-inclusion.v1'),
                    kind=kind, assumptions=assumptions)
        result['data'].append({'invocation_id': row['invocation_id'], **item})
    if catalog is not None:
        result['catalog_version'] = catalog.version_id
        result['comparison_currency'] = assumptions.get('currency')
    result['coverage'] = {'covered': sum(r['amount'] is not None for r in result['data']), 'required': len(rows)}
    result['status'] = 'ESTIMATE' if result['coverage']['covered'] else 'UNAVAILABLE'
    result['reason'] = None if result['coverage']['covered'] else 'no_priced_usage_in_scope'
    return result


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


async def task_scope_actor(service,actor,account_id,currency,device_token_id=None):
    if device_token_id is None:return actor
    from dataclasses import replace
    from contracts.model_gateway import canonical_hash
    async with await service.store._connection() as c:
        device=await (await c.execute('''SELECT d.context FROM semantier_gateway.device_connections d JOIN semantier_gateway.tokens t USING(token_id)
            WHERE t.token_id=%s AND t.user_id=%s AND t.organization_id=%s AND t.workspace_id=%s AND t.account_id=%s AND t.currency=%s
              AND t.revoked_at IS NULL AND t.expires_at>now()''',(device_token_id,actor.user_id,actor.organization_id,actor.workspace_id,account_id,currency))).fetchone()
    if not device:raise HTTPException(403,'device task scope unavailable')
    return replace(actor,client_namespace=canonical_hash([device['context']['runtime_ref'],device['context']['profile_ref']]))


@router.get('/task-policy')
async def billing_task_policy(request: Request):
    service,actor,aid,currency,_=await account(request)
    actor=await task_scope_actor(service,actor,aid,currency,request.query_params.get('device_token_id'))
    from contracts.model_gateway import task_context
    try:
        context=task_context(actor,aid,currency,{'root_run_id':request.query_params.get('root_run_id','')})
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    policy=await service.billing.task_policy(actor.organization_id,aid,currency,context['scope_id'])
    return {'schema_version':'semantier.task-finops-policy.v1','unit':'tokens','window':'root_lifetime','inclusion':'input_total+output_total','policy':policy,'scope_id':context['scope_id']}


@router.post('/task-policy')
async def billing_task_policy_update(request: Request):
    service,actor,aid,currency,_=await account(request,'budgets:manage',browser_only=True)
    command=await request.json()
    fields={'task','token_limit','allocation_target','expected_policy_event_id','device_token_id'}
    if not isinstance(command,dict) or set(command)-fields or not {'task','token_limit','allocation_target','expected_policy_event_id'}<=set(command):
        raise HTTPException(422,'explicit task, token limit, allocation and expected revision required')
    actor=await task_scope_actor(service,actor,aid,currency,command.get('device_token_id'))
    from contracts.model_gateway import task_context
    try:
        context=task_context(actor,aid,currency,command['task'])
        return await service.billing.set_task_policy(actor.organization_id,aid,currency,context,token_limit=command['token_limit'],allocation_target=command['allocation_target'],actor_user_id=actor.user_id,expected_policy_event_id=command['expected_policy_event_id'])
    except ValueError as exc:raise HTTPException(409 if getattr(exc,'code',None) else 422,str(exc)) from exc
