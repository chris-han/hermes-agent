"""Policy admission and one physical-send lifecycle, independent of settlement state."""
from __future__ import annotations
import json
import asyncio
from dataclasses import replace
import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contracts.model_gateway import canonical_hash, validate_chat
from billing.native_evidence import final_native_observation


class GatewayService:
    def __init__(self, *, store, billing, proxy, enabled=False, account_sync_store=None):
        self.store,self.billing,self.proxy=store,billing,proxy
        self.enabled=enabled
        self.account_sync_store=account_sync_store

    async def credential_for(self, actor, deployment, *, current_revision=None):
        if self.account_sync_store is None or self.proxy is None:
            raise HTTPException(503,'inference access preparation unavailable')
        return await self.account_sync_store.selected_credential(actor,deployment,current_revision=current_revision)

    async def resolve(self, actor, alias):
        if self.proxy is None:
            raise HTTPException(503,'inference access preparation unavailable')
        routes=[d for d in await self.store.deployments(actor.organization_id) if d.enabled and d.alias == alias]
        if not routes:
            raise HTTPException(404,'no allowed deployment')
        # Multiple bindings for one alias require an explicit server policy; never silent fallback.
        if len(routes) != 1:
            raise HTTPException(409,'ambiguous route policy')
        native=await self.proxy.verify_binding(routes[0])
        return replace(routes[0],provider_model_id=native['litellm_params']['model']),native['physical_binding']

    async def finish(self, actor, deployment, admitted):
        try:
            observations=await self.billing.read_raw_observations(actor.organization_id,admitted['attempt_id'])
            receipt=final_native_observation(observations,deployment.provider)
            if not receipt or not receipt.get('raw_usage'):
                await self.billing.mark_uncertain(actor.organization_id,admitted['invocation_id'])
                return
            if receipt.get('provider') != deployment.provider:
                raise ValueError('capture provider mismatch')
            if receipt.get('attempt_id',admitted['attempt_id']) != admitted['attempt_id']:
                raise ValueError('capture attempt mismatch')
            await self.billing.meter(organization_id=actor.organization_id,attempt_id=admitted['attempt_id'],
                event_id=receipt.get('event_id',admitted['attempt_id']),provider=receipt['provider'],
                raw_usage=receipt['raw_usage'],customer_price_version=deployment.customer_price_version,
                supplier_price_version=deployment.supplier_price_version,fx_version=deployment.fx_version)
            await self.billing.settle(actor.organization_id,admitted['invocation_id'])
        except Exception:
            await self.billing.mark_uncertain(actor.organization_id,admitted['invocation_id'])

    async def chat(self, actor, body, idempotency_key, correlation=None):
        try:
            body=validate_chat(body)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        if not idempotency_key or len(idempotency_key)>200:
            raise HTTPException(400,'Idempotency-Key required (max 200 characters)')
        d,physical_binding=await self.resolve(actor,body['model'])
        for capability in ('tools','stream'):
            if body.get(capability) and capability not in d.capabilities:
                raise HTTPException(422,'unsupported capability: '+capability)
        credential=await self.credential_for(actor,d)
        # The account and funding are server-bound and included in the immutable request receipt.
        try:
            admitted=await self.billing.admit(organization_id=actor.organization_id,account_id=d.account_id,
                operation='chat.completions',idempotency_key=idempotency_key,
                request_hash=canonical_hash(body),deployment_id=d.deployment_id,funding=d.funding,
                currency=d.currency,max_cost=d.max_cost,attribution={'actor_user_id':actor.user_id,
                'native_credential_generation':credential.generation,
                'native_desired_revision':credential.desired_revision,
                'native_target_instance_id':credential.target_instance_id,
                'workspace_id':actor.workspace_id,**{k:v for k,v in (correlation or {}).items() if k in {'session_id','run_id','purpose'}},'deployment_revision':d.revision,
                'config_fingerprint':d.config_fingerprint,'region':d.region,'protocol':d.protocol,
                'provider':d.provider,'customer_price_version':d.customer_price_version,
                'supplier_price_version':d.supplier_price_version,'fx_version':d.fx_version,
                'physical_binding':physical_binding,'provider_model_id':d.provider_model_id,'model_version':d.model_version,'version_status':d.version_status})
        except ValueError as exc:
            code=getattr(exc,'code','admission_rejected')
            raise HTTPException(409 if code=='idempotency_conflict' else 402,code) from exc
        if not admitted['created']:
            if admitted.get('attribution',{}).get('actor_user_id') != actor.user_id:
                raise HTTPException(409,'idempotency key already used')
            return JSONResponse({'object':'semantier.invocation','invocation':admitted},status_code=200)
        # A revoke/policy change after admission must release a definitely unsent
        # hold. A management credential is never an alternative data-plane key.
        try:
            await self.billing.mark_dispatching(actor.organization_id,admitted['invocation_id'])
            credential=await self.credential_for(actor,d,current_revision=credential.desired_revision)
        except BaseException:
            await self.billing.fail_before_send(actor.organization_id,admitted['invocation_id'])
            raise
        try:
            response=await self.proxy.chat({**body,'model':d.proxy_model},admitted['invocation_id'],admitted['attempt_id'],
                credential=credential,actor=actor,deployment=d)
        except (httpx.ConnectError,httpx.ConnectTimeout,httpx.PoolTimeout) as exc:
            await self.billing.fail_before_send(actor.organization_id,admitted['invocation_id'])
            raise HTTPException(502,'private Proxy connection unavailable before send') from exc
        except BaseException:
            await self.billing.mark_uncertain(actor.organization_id,admitted['invocation_id'])
            raise
        headers={'x-semantier-invocation-id':admitted['invocation_id'],'x-semantier-attempt-id':admitted['attempt_id']}
        if response.status_code >= 400:
            await response.aclose()
            await self.billing.mark_uncertain(actor.organization_id,admitted['invocation_id'])
            raise HTTPException(502,'private Proxy invocation rejected')
        if body.get('stream'):
            async def chunks():
                complete=False
                try:
                    async with asyncio.timeout(60):
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    complete=True
                finally:
                    await response.aclose()
                    if complete:
                        await self.finish(actor,d,admitted)
                    else:
                        await self.billing.mark_uncertain(actor.organization_id,admitted['invocation_id'])
            return StreamingResponse(chunks(),media_type='text/event-stream',headers=headers)
        try:
            raw=await response.aread()
            result=json.loads(raw)
        except BaseException:
            await self.billing.mark_uncertain(actor.organization_id,admitted['invocation_id'])
            raise
        finally:
            await response.aclose()
        await self.finish(actor,d,admitted)
        return JSONResponse(result,headers=headers)
