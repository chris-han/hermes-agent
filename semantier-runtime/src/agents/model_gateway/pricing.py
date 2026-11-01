"""Finite supplier-price administration, independent of inference and customer rates."""
import asyncio
import math
from fastapi import HTTPException
import httpx
from billing.supplier_pricing_adapters import adapter_for, registered_adapters
from billing.supplier_pricing_store import SupplierPricingStore
from billing.supplier_pricing_composition import compose_projection
from billing.supplier_pricing_projection import digest


def native_source_matches(projection, loaded, observed):
    etag=observed.get('etag')
    return (loaded.get('source_revision') == projection['native_revision'] == observed.get('source_revision')
        and observed.get('source') == 'remote' and not observed.get('fallback_reason')
        and isinstance(etag,str) and etag.strip('"') == projection['sha256'])


def native_projection_loaded(projection, loaded, observed, *, reconciliation=None):
    if not native_source_matches(projection, loaded, observed):
        return False
    if not isinstance(reconciliation, dict):
        return False
    registration_models = reconciliation.get('registration_models')
    registration_aliases = reconciliation.get('registration_aliases')
    observed_aliases = reconciliation.get('observed_aliases')
    required_ids = reconciliation.get('required_registration_ids')
    expected_models = reconciliation.get('expected_models')
    if (not isinstance(registration_models, dict)
        or not isinstance(required_ids, (set, frozenset, list, tuple))
        or not all(isinstance(item, str) and item for item in required_ids)
        or set(registration_models) != set(required_ids)
        or not all(isinstance(model, str) and model for model in registration_models.values())
        or not isinstance(registration_aliases, dict) or set(registration_aliases) != set(required_ids)
        or not all(isinstance(alias, str) and alias for alias in registration_aliases.values())
        or not isinstance(observed_aliases, dict) or observed_aliases != registration_aliases
        or not isinstance(expected_models, (set, frozenset, list, tuple))
        or not all(isinstance(model, str) and model for model in expected_models)
        or set(expected_models) != set(registration_models.values())):
        return False
    from billing.supplier_pricing_reconciliation import reconcile_native_workers, reconcile_native_capabilities
    result = reconcile_native_workers(
        reconciliation.get('base'), reconciliation.get('registrations'),
        reconciliation.get('observations'), reconciliation.get('expected_workers'),
        reconciliation.get('owned_overrides'), expected_base_sha256=projection['sha256'],
        required_registration_ids=reconciliation.get('required_registration_ids'),
        expected_epochs=reconciliation.get('expected_epochs'))
    observations = reconciliation.get('observations')
    return (result.get('verified') is True and bool(result.get('workers'))
        and all(row.get('effective_sha256') == observed.get('observed_map_sha256')
                for row in result['workers'].values())
        and all(reconcile_native_capabilities(reconciliation.get('expected_models'),
                                             reconciliation.get('capabilities'), entry['map'])['verified']
                for entry in observations.values()))


def summarize_native_overrides(native):
    rows = native.get('data') if isinstance(native, dict) else None
    if not isinstance(rows, list):
        return [], 'UNKNOWN'
    result = []
    known = True
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('model_info'), dict):
            known = False
            continue
        info = row['model_info']
        fields = info.get('pricing_overrides')
        if (not isinstance(info.get('id'), str) or not info['id']
            or not isinstance(fields, list) or not all(isinstance(field, str) and field for field in fields)):
            known = False
            continue
        if fields:
            result.append({'id': info.get('id'), 'model': row.get('model_name'), 'fields': sorted(set(fields))})
    return result, 'KNOWN' if known else 'UNKNOWN'


def native_registration_price_provenance(native):
    """Fingerprint explicit native registration prices, excluding credential fields.

    This identifies one observed inventory snapshot; it does not prove worker
    convergence, historical revision order, or effective-map reconciliation.
    """
    rows = native.get('data') if isinstance(native, dict) else None
    unknown = {'state': 'UNKNOWN', 'revision': None, 'count': None}
    if not isinstance(rows, list):
        return unknown
    inventory = {}
    for row in rows:
        if not isinstance(row, dict):
            return unknown
        info, params = row.get('model_info'), row.get('litellm_params')
        if not isinstance(info, dict) or not isinstance(params, dict):
            return unknown
        key, alias, model, fields = info.get('id'), row.get('model_name'), params.get('model'), info.get('pricing_overrides')
        if (not all(isinstance(value, str) and value for value in (key, alias, model))
            or key in inventory or not isinstance(fields, list)
            or not all(isinstance(field, str) and field and ('cost' in field.lower() or 'pric' in field.lower()) for field in fields)
            or len(fields) != len(set(fields))):
            return unknown
        prices = {}
        for field in fields:
            value = params.get(field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or info.get(field) != value):
                return unknown
            prices[field] = value
        inventory[key] = {'alias': alias, 'physical_model': model, 'explicit_prices': prices}
    return {'state': 'KNOWN', 'revision': digest(inventory), 'count': len(inventory)}


class PricingAdmin:
    def __init__(self, service, *, adapters=None):
        self.service=service
        self.adapters=registered_adapters() if adapters is None else adapters
        self.store=SupplierPricingStore(service.billing)

    async def native_router_bindings(self, native):
        rows = native.get('data') if isinstance(native, dict) else None
        if not isinstance(rows, list):
            return None
        expected = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('model_info'), dict) or not isinstance(row.get('litellm_params'), dict):
                return None
            key, alias, model = row['model_info'].get('id'), row.get('model_name'), row['litellm_params'].get('model')
            if not all(isinstance(value, str) and value for value in (key, alias, model)) or key in expected:
                return None
            expected[key] = (alias, model)
        async def lookup(key):
            return await self.service.proxy.operation('GET', '/model/info', params={'litellm_model_id': key})
        try:
            results = await asyncio.wait_for(asyncio.gather(*(lookup(key) for key in expected)), timeout=2)
        except (httpx.HTTPError, HTTPException, TimeoutError, TypeError, ValueError):
            return None
        bindings = {}
        for key, result in zip(expected, results):
            data = result.get('data') if isinstance(result, dict) else None
            if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
                return None
            row = data[0]
            info, params = row.get('model_info'), row.get('litellm_params')
            if (not isinstance(info, dict) or not isinstance(params, dict) or info.get('id') != key
                or (row.get('model_name'), params.get('model')) != expected[key]):
                return None
            bindings[key] = row['model_name']
        return bindings

    async def native_observation(self):
        async def identity():
            try:
                value = await asyncio.wait_for(self.service.proxy.operation('GET', '/semantier/runtime/identity'), timeout=2)
                if (isinstance(value, dict) and value.get('listener_owned') is True
                    and isinstance(value.get('epoch'), str) and len(value['epoch']) == 64
                    and all(char in '0123456789abcdef' for char in value['epoch'])):
                    return value['epoch']
            except (httpx.HTTPError, HTTPException, TimeoutError, TypeError, ValueError):
                pass
            return None

        async def metadata_snapshot(epoch):
            try:
                value = await asyncio.wait_for(self.service.proxy.operation('GET', '/semantier/runtime/registrations'), timeout=2)
                if (isinstance(value, dict) and value.get('listener_owned') is True
                    and value.get('epoch') == epoch and isinstance(value.get('registrations'), list)
                    and isinstance(value.get('durable_registrations'), list)
                    and value.get('replay_order') == ['router', 'durable']
                    and isinstance(value.get('native_rule_sha256'), dict)
                    and set(value['native_rule_sha256']) == {'router', 'registration', 'router_types', 'cost_types'}
                    and all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                            for v in value['native_rule_sha256'].values())):
                    return digest({key: value[key] for key in ('registrations', 'durable_registrations', 'replay_order', 'native_rule_sha256')})
            except (httpx.HTTPError, HTTPException, TimeoutError, TypeError, ValueError):
                pass
            return None

        enabled = getattr(self.service, 'pricing_runtime_identity_enabled', False)
        before = await identity() if enabled else None
        metadata_before = await metadata_snapshot(before) if before else None
        observed=await self.service.proxy.operation('GET','/model/cost_map/source')
        bindings = None
        registration_before = None
        if before:
            try:
                native = await asyncio.wait_for(self.service.proxy.operation('GET', '/model/info'), timeout=2)
                registration_before = native_registration_price_provenance(native)
                bindings = await self.native_router_bindings(native)
            except (httpx.HTTPError, HTTPException, TimeoutError, TypeError, ValueError):
                pass
        try:
            actual=await asyncio.wait_for(self.service.proxy.operation('GET','/public/litellm_model_cost_map'),timeout=2)
            observed['observed_map_sha256']=digest(actual)
        except (httpx.HTTPError,HTTPException,TimeoutError,TypeError,ValueError):
            observed['observed_map_sha256']=None
        registration_stable = False
        if before and registration_before and registration_before['state'] == 'KNOWN':
            try:
                native_after = await asyncio.wait_for(self.service.proxy.operation('GET', '/model/info'), timeout=2)
                registration_after = native_registration_price_provenance(native_after)
                registration_stable = registration_after == registration_before
            except (httpx.HTTPError, HTTPException, TimeoutError, TypeError, ValueError):
                pass
        metadata_after = await metadata_snapshot(before) if before else None
        after = await identity() if enabled else None
        observed['worker_observation'] = (
            {'epoch': before, 'current': True, 'router_bindings': bindings if registration_stable else None,
             'registration_price_snapshot_stable': registration_stable,
             'registration_metadata_snapshot_stable': bool(metadata_before and metadata_before == metadata_after),
             'registration_metadata_sha256': metadata_before if metadata_before and metadata_before == metadata_after else None}
            if before and before == after and observed['observed_map_sha256'] else None)
        return observed

    async def native_reconciliation(self, projection, observed):
        """Acquire a fresh native replay and actual map; fail closed on missing facts."""
        from agents.model_gateway.native_expected_evidence import registration_overlays
        worker = observed.get('worker_observation')
        if not isinstance(worker, dict) or not worker.get('registration_metadata_snapshot_stable') or not worker.get('registration_price_snapshot_stable'):
            return None
        base = projection.get('map')
        if not isinstance(base, dict) or digest(base) != projection.get('sha256'):
            return None
        try:
            native = await asyncio.wait_for(self.service.proxy.operation('POST', '/semantier/runtime/expected', body={'base': base}), timeout=50)
            if (not isinstance(native, dict) or native.get('listener_owned') is not True
                or native.get('epoch') != worker.get('epoch') or not isinstance(native.get('snapshot'), dict)
                or digest(native['snapshot']) != worker.get('registration_metadata_sha256')):
                return None
            snapshot, expected = native['snapshot'], native.get('expected_map')
            rows = snapshot['registrations']
            models = {row['id']: row['model'] for row in rows}
            aliases = {row['id']: row['alias'] for row in rows}
            if worker.get('router_bindings') != aliases: return None
            physical = set(models.values())
            if not physical: return None
            async def capability(model):
                value = await self.service.proxy.operation('GET', '/utils/supported_openai_params', params={'model': model})
                return value['supported_openai_params']
            values = await asyncio.gather(*(capability(model) for model in sorted(physical)))
            capabilities = dict(zip(sorted(physical), values))
            overlays = registration_overlays(base, snapshot, expected, capabilities)
            if overlays is None: return None
            actual = await self.service.proxy.operation('GET', '/public/litellm_model_cost_map')
            after = await self.service.proxy.operation('GET', '/semantier/runtime/registrations')
            identity = await self.service.proxy.operation('GET', '/semantier/runtime/identity')
            if (identity.get('listener_owned') is not True or identity.get('epoch') != worker['epoch']
                or after.get('epoch') != worker['epoch'] or after.get('listener_owned') is not True
                or digest({key: after[key] for key in ('registrations', 'durable_registrations', 'replay_order', 'native_rule_sha256')}) != digest(snapshot)
                or digest(actual) != observed.get('observed_map_sha256')):
                return None
            changes, claims = overlays
            return {'base': base, 'registrations': changes, 'owned_overrides': claims,
                'required_registration_ids': list(models), 'registration_models': models,
                'registration_aliases': aliases, 'observed_aliases': worker['router_bindings'],
                'expected_models': list(physical), 'capabilities': capabilities,
                'expected_workers': ['listener'], 'expected_epochs': {'listener': worker['epoch']},
                'observations': {'listener': {'current': True, 'epoch': worker['epoch'], 'map': actual}}}
        except (httpx.HTTPError, HTTPException, TimeoutError, KeyError, TypeError, ValueError):
            return None

    async def capabilities(self, actor):
        operator=await self.store.operator(actor.user_id)
        available=self.service.proxy is not None
        return {'collect':True,'view':True,'sync':operator and available,
            'reason':None if operator and available else 'NATIVE_SERVICE_UNAVAILABLE' if operator else 'PLATFORM_OPERATOR_REQUIRED'}

    async def require_operator(self, actor):
        if actor.token_id is not None or not await self.store.operator(actor.user_id):
            raise HTTPException(403,'platform pricing operator grant required')
        if self.service.proxy is None:
            raise HTTPException(503,'native pricing service unavailable')

    async def collect(self, actor, body):
        if not isinstance(body, dict):
            raise HTTPException(422, 'collection body required')
        legacy = set(body) == {'idempotency_key', 'models'}
        scoped = set(body) == {'idempotency_key', 'models', 'provider_id', 'source_ref'}
        if not (legacy or scoped):
            raise HTTPException(422, 'supplier source and selection required')
        try:
            adapter = adapter_for(body.get('provider_id', 'alibaba'), body.get('source_ref', 'alibaba-model-studio-cn-beijing'), adapters=self.adapters)
        except ValueError:
            raise HTTPException(422, 'unsupported supplier source') from None
        if (not isinstance(body['idempotency_key'],str) or not 1<=len(body['idempotency_key'])<=128
                or not isinstance(body['models'],list) or not body['models']
                or not all(isinstance(model,str) for model in body['models'])
                or len(body['models'])!=len(set(body['models']))
                or not set(body['models'])<=set(adapter.describe()['sources'][0]['models'])):
            raise HTTPException(422,'fixed model selection and idempotency key required')
        source=adapter.describe()['sources'][0]
        row,created=await self.store.begin(actor.organization_id,body['idempotency_key'],body['models'],provider_id=adapter.provider_id,source_ref=adapter.source_ref,partition=source['partition'])
        if not created:return row
        try:
            base=getattr(self.service,'pricing_api_base',None)
            if not base:
                routes=[d for d in await self.service.authorized_deployments(actor) if d.provider=='dashscope']
                if routes and self.service.proxy is not None:
                    verified=await self.service.proxy.verify_binding(routes[0])
                    base=verified['physical_binding']['api_base']
            async with self.service.billing.connection() as connection:
                locked=await (await connection.execute('SELECT pg_try_advisory_xact_lock(hashtext(%s)) AS locked',('supplier-source:'+base,))).fetchone()
                if not locked['locked']:
                    return await self.store.finish(actor.organization_id,row['run_id'],failure='SOURCE_PARTITION_BUSY')
                snapshot=await adapter.collect(base,body['models'],client=getattr(self.service,'pricing_http_client',None),api_key=getattr(self.service,'pricing_api_key',None))
            return await self.store.finish(actor.organization_id,row['run_id'],snapshot)
        except asyncio.CancelledError:
            # Durable COLLECTING lease becomes INTERRUPTED. Never automatically
            # replay the external GETs after process loss/cancellation.
            raise
        except (ValueError,httpx.HTTPError,HTTPException,KeyError):
            return await self.store.finish(actor.organization_id,row['run_id'],failure='SOURCE_CONFIGURATION_OR_CAPTURE_FAILED')

    async def providers(self, actor):
        capabilities = await self.capabilities(actor)
        return {'data': [adapter.describe() for adapter in self.adapters], 'capabilities': capabilities}

    async def source(self, actor, *, provider_id=None, source_ref=None):
        if (provider_id is None) != (source_ref is None):
            raise ValueError('INVALID_SOURCE_SCOPE')
        if provider_id is not None:
            adapter_for(provider_id,source_ref,adapters=self.adapters)
            collected=await self.store.latest_snapshot(actor.organization_id,provider_id,source_ref)
        else:
            runs=await self.store.runs(actor.organization_id)
            collected=next((run['snapshot_version'] for run in runs if run['snapshot_version']),None)
        operator=await self.store.operator(actor.user_id)
        result={'collected_version':collected,
            'capabilities':await self.capabilities(actor), 'worker_convergence':'UNKNOWN',
            'deployment_overrides':[], 'deployment_overrides_state':'UNKNOWN', 'effective':None, 'selected':None}
        if operator and self.service.proxy is not None:
            async with self.service.billing.connection() as c:
                rows=await (await c.execute('SELECT payload FROM fx_quotes ORDER BY quote_id LIMIT 50')).fetchall()
                result['fx_quotes']=[r['payload'] for r in rows]
            if getattr(self.service,'pricing_require_reference_fx',False):
                from billing.rating import FxQuote
                from billing.supplier_reference_fx import require_reference
                eligible=[]
                for q in result['fx_quotes']:
                    try:
                        evidence=await require_reference(self.service.billing,FxQuote(**q))
                        eligible.append({**q,'quote_date':evidence['quote_date']})
                    except ValueError:continue
                result['fx_quotes']=eligible
                result['fx_basis']='ECB reference conversion, not settlement; unavailable/stale quotes cannot synchronize'
                if not getattr(self.service,'pricing_reference_fx_enabled',True):
                    result['fx_quotes']=[]
                    result['fx_basis']='Reference FX publication paused; CNY remains available; existing native version retained'

            try:
                result['effective']=await asyncio.wait_for(self.native_observation(),timeout=2)
            except (httpx.HTTPError,HTTPException,TimeoutError):
                result['native_status']='UNAVAILABLE'
            active=await self.store.active()
            if active:
                result['selected']={**{k:active[k] for k in ('projection_version','effective_revision','state')},
                    'excluded':active['payload'].get('excluded',{}),'coverage':active['payload'].get('coverage'),
                    'snapshot_version':active['payload'].get('snapshot_version'),
                    'selected_revision':active['payload'].get('native_revision'),
                    'map_sha256':active['payload'].get('sha256')}
                observed = result.get('effective')
                if isinstance(observed, dict):
                    reconciliation = await self.native_reconciliation(active['payload'], observed)
                    if reconciliation is not None:
                        configured_url = getattr(self.service, 'pricing_cost_map_url', None)
                        verified = (bool(configured_url) and observed.get('url') == configured_url
                            and not observed.get('is_env_forced')
                            and native_projection_loaded(active['payload'], {'source_revision': observed.get('source_revision')}, observed, reconciliation=reconciliation))
                        result['worker_convergence'] = 'VERIFIED' if verified else 'MISMATCH'
                if result['worker_convergence'] != 'VERIFIED' and result['selected']['state'] == 'EFFECTIVE_ONE_WORKER':
                    result['selected']['state'] = 'OBSERVATION_UNKNOWN' if result['worker_convergence'] == 'UNKNOWN' else 'SYNC_FAILED'
            try:
                native=await asyncio.wait_for(self.service.proxy.operation('GET','/model/info'),timeout=2)
            except (httpx.HTTPError,HTTPException,TimeoutError):
                native={}
                result['native_status']='UNAVAILABLE'
            result['deployment_overrides'],result['deployment_overrides_state']=summarize_native_overrides(native)
            result['registration_price_provenance']=native_registration_price_provenance(native)
        return result

    async def sync(self, actor, body):
        await self.require_operator(actor)
        if not isinstance(body,dict) or set(body)!={'run_id','fx_quote_id'}:
            raise HTTPException(422,'run and exact FX quote required')
        # The lock covers selection and native observation across backend workers.
        async with self.service.billing.connection() as c:
            await c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('supplier-pricing-global-sync',))
            run=await self.store.run(actor.organization_id,body['run_id'])
            snapshot=await self.store.snapshot(actor.organization_id,run['snapshot_version'])
            fx=await self.service.billing.get_fx(body['fx_quote_id'])
            if getattr(self.service,'pricing_require_reference_fx',False):
                if not getattr(self.service,'pricing_reference_fx_enabled',True):
                    raise HTTPException(503,'reference FX publication unavailable; CNY retained')
                from billing.supplier_reference_fx import require_reference
                await require_reference(self.service.billing,fx)
            active=await self.store.active()
            prior=active['payload'] if active else {}
            base=await self.store.base_map(prior['base_sha256']) if prior else await self.service.proxy.operation('GET','/public/litellm_model_cost_map',params={'catalog_only':'true'})
            adapter=adapter_for(run['provider_id'],run['source_ref'],adapters=self.adapters)
            projection=adapter.project(snapshot,base,fx)
            projection=compose_projection(base,prior,adapter.provider_id+':'+adapter.source_ref,projection,excluded_keys=adapter.excluded_native_keys(projection))
            version=await self.store.put_projection(actor.organization_id,body['run_id'],projection,base_map=base)
            # Refuse changing a shared source silently. Native must already be
            # configured to the authenticated backend URL by its runtime owner.
            source=await self.service.proxy.operation('GET','/model/cost_map/source')
            expected=getattr(self.service,'pricing_cost_map_url',None)
            if not expected or source.get('url')!=expected or source.get('is_env_forced'):
                raise HTTPException(409,'native cost-map source must bind the authenticated backend URL')
            await self.store.select(actor.organization_id,version)
            observed = {}
            try:
                loaded=await self.service.proxy.operation('POST','/reload/model_cost_map')
                observed=await self.native_observation()
                reconciliation=await self.native_reconciliation(projection, observed)
                success=(native_projection_loaded(projection,loaded,observed,reconciliation=reconciliation)
                         and observed.get('url')==expected and not observed.get('is_env_forced'))
            except (HTTPException,httpx.HTTPError):
                success=False
                reconciliation={}
                loaded={'status':'SYNC_FAILED','reason':'NATIVE_RELOAD_FAILED'}
            unknown=(reconciliation is None and native_source_matches(projection,loaded,observed)
                     and observed.get('url') == expected and not observed.get('is_env_forced'))
            await self.store.effective(version,projection['native_revision'],success,unknown=unknown)
            return {'projection_version':version,'selected_revision':projection['native_revision'],
                'effective':success,'native':loaded,'excluded':projection['excluded'],
                'worker_convergence':'VERIFIED' if success else 'UNKNOWN','coverage':projection['coverage']}

    async def reload(self, actor):
        await self.require_operator(actor)
        async with self.service.billing.connection() as c:
            await c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('supplier-pricing-global-sync',))
            active=await self.store.active()
            try:
                loaded=await self.service.proxy.operation('POST','/reload/model_cost_map')
                if not active:
                    return loaded
                observed=await self.native_observation()
                reconciliation=await self.native_reconciliation(active['payload'], observed)
                expected=getattr(self.service,'pricing_cost_map_url',None)
                success=(bool(expected) and observed.get('url')==expected
                         and not observed.get('is_env_forced')
                         and native_projection_loaded(active['payload'],loaded,observed,reconciliation=reconciliation))
            except (HTTPException,httpx.HTTPError):
                if active:
                    await self.store.effective(active['projection_version'],active['payload']['native_revision'],False)
                raise
            unknown=(reconciliation is None and native_source_matches(active['payload'],loaded,observed)
                     and observed.get('url') == expected and not observed.get('is_env_forced'))
            await self.store.effective(active['projection_version'],active['payload']['native_revision'],success,unknown=unknown)
            return {**loaded,'effective':success}
