"""Policy admission and one physical-send lifecycle, independent of settlement state."""
from __future__ import annotations
import json
import asyncio
import logging
import anyio
from dataclasses import replace
import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contracts.model_gateway import canonical_hash, validate_chat, task_context
from billing.native_evidence import final_native_observation


logger = logging.getLogger(__name__)


class GatewayService:
    def __init__(self, *, store, billing, proxy, enabled=False, account_sync_store=None):
        self.store,self.billing,self.proxy=store,billing,proxy
        self.enabled=enabled
        self.account_sync_store=account_sync_store

    async def authorized_deployments(self,actor):
        scopes={(r['account_id'],r['currency'],r['funding']) for r in await self.store.account_rows(actor,self.billing.schema)}
        return [d for d in await self.store.deployments(actor.organization_id) if d.enabled and (d.account_id,d.currency,d.funding) in scopes]

    async def catalog(self, actor, model=None, *, account_id=None):
        from agents.model_gateway.capabilities import project_profile
        allowed=await self.authorized_deployments(actor)
        if account_id is not None and not any(d.account_id==account_id for d in allowed):
            raise HTTPException(403,'account catalog unavailable')
        routes = [d for d in allowed if (model is None or d.alias == model) and (account_id is None or d.account_id==account_id)]
        if model is not None and len(routes) != 1:
            raise HTTPException(404 if not routes else 409, 'no unique allowed deployment')
        if not routes:
            return {'object':'list', 'data':[]}
        if self.proxy is None:raise HTTPException(503,'Semantier Cloud Provider metadata unavailable')
        try:
            async with asyncio.timeout(5):
                snapshot = await self.proxy.operation('GET','/model/info')
                profiles = [project_profile(d, await self.proxy.verify_binding(d,snapshot=snapshot)) for d in routes]
        except (TimeoutError, httpx.RequestError):
            raise HTTPException(503,'Semantier Cloud Provider metadata unavailable; refresh') from None
        return profiles[0] if model is not None else {'object':'list','data':profiles}

    async def credential_for(self, actor, deployment, *, current_revision=None):
        if self.account_sync_store is None or self.proxy is None:
            raise HTTPException(503,'inference access preparation unavailable')
        return await self.account_sync_store.selected_credential(actor,deployment,current_revision=current_revision)

    async def _actor_for_context(self, ctx):
        """Internal producer entry; identity comes from the authenticated request owner.

        Recheck current membership and grants exactly as the HTTP facade does.
        Neither prompt data nor platform provider credentials establish an actor.
        """
        from agents.auth_session import RequestContext
        from agents.model_gateway.auth import GatewayActor, current_membership

        if not self.enabled:
            raise HTTPException(503, 'model gateway disabled')
        if not isinstance(ctx, RequestContext) or not ctx.authenticated:
            raise HTTPException(401, 'authenticated structured context required')
        if not ctx.organization_id or not current_membership(ctx.user_id, ctx.organization_id, ctx.workspace_id):
            raise HTTPException(403, 'current active membership required')
        permissions = await self.store.permissions(ctx.organization_id, ctx.user_id)
        if 'model:invoke' not in permissions:
            raise HTTPException(403, 'explicit model:invoke permission required')
        return GatewayActor(ctx.user_id, ctx.organization_id, ctx.workspace_id, frozenset(permissions))

    async def chat_for_context(self, ctx, body, idempotency_key, correlation=None, *, account_id=None, expected_profile_revision=None):
        actor = await self._actor_for_context(ctx)
        if len(json.dumps(body, ensure_ascii=False, allow_nan=False).encode()) > 1048576:
            raise HTTPException(413, 'chat request exceeds 1 MiB')
        if body.get('stream'):
            raise HTTPException(422, 'structured completion must be non-streaming')
        return await self.chat(actor, body, idempotency_key, correlation, account_id=account_id, expected_profile_revision=expected_profile_revision)

    async def routed_chat_for_context(self, ctx, body, idempotency_key, correlation=None, **policy):
        """Explicit private producer seam; no browser policy or customer activation."""
        from agents.model_gateway.routing import choose_only
        actor = await self._actor_for_context(ctx)
        chosen, decision = await choose_only(self, actor, body, **policy)
        return await self.chat(actor, {**body, 'model': chosen.alias}, idempotency_key,
            correlation, account_id=chosen.account_id, _routing_decision=decision)

    async def reexecute_for_context(self, ctx, predecessor_id, command, idempotency_key):
        actor = await self._actor_for_context(ctx)
        return await self.reexecute(actor, predecessor_id, command, idempotency_key)

    @staticmethod
    def _admission_error(exc):
        code=getattr(exc,'code','admission_rejected')
        status=404 if code=='not_found' else 409 if code in {'idempotency_conflict','predecessor_active','predecessor_hash_mismatch','request_relation_mismatch','task_policy_missing'} else 402
        return HTTPException(status,code)

    async def reexecute(self, actor, predecessor_id, command, idempotency_key):
        from contracts.model_gateway import ModelReexecutionCommandV1
        from agents.model_gateway.auth import current_membership
        if not self.enabled: raise HTTPException(503,'model gateway disabled')
        if not current_membership(actor.user_id,actor.organization_id,actor.workspace_id) or 'model:invoke' not in actor.permissions or 'model:invoke' not in await self.store.permissions(actor.organization_id,actor.user_id):
            raise HTTPException(403,'current model:invoke authority required')
        try:
            command=ModelReexecutionCommandV1.model_validate(command.model_dump() if isinstance(command,ModelReexecutionCommandV1) else command)
            if not isinstance(idempotency_key,str) or not 1<=len(idempotency_key)<=200 or not idempotency_key.isascii() or any(not 33<=ord(c)<=126 for c in idempotency_key):
                raise ValueError('bounded ASCII Idempotency-Key required')
        except ValueError as exc:
            raise HTTPException(422,'invalid reexecution command') from exc
        try:
            parent=await self.billing.reexecution_predecessor(actor.organization_id,predecessor_id,actor.user_id,actor.workspace_id)
            if getattr(actor,'account_id',None) is not None and (parent['account_id'],parent['currency']) != (actor.account_id,actor.currency):
                raise HTTPException(403,'credential account mismatch')
            if self.account_sync_store is None: raise HTTPException(503,'account authority unavailable')
            await self.account_sync_store.require_reexecution_access(actor,parent['account_id'],parent['currency'])
            recovered=await self.billing.recover_reexecution(actor.organization_id,parent['account_id'],actor.user_id,actor.workspace_id,idempotency_key,command.command_hash(predecessor_id,actor.user_id,actor.workspace_id))
        except ValueError as exc: raise self._admission_error(exc) from exc
        if recovered:
            return await self._reexecution_status(actor,recovered)
        return await self.chat(actor,command.request,idempotency_key,account_id=parent['account_id'],_reexecution=(predecessor_id,command,parent),
            expected_profile_revision=command.expected_profile_revision)

    async def _reexecution_status(self, actor, invocation):
        lineage=await self.billing.reexecution_lineage(actor.organization_id,invocation['invocation_id'])
        return JSONResponse({'object':'semantier.invocation','invocation':invocation,
            'reexecution':lineage,'result_availability':'NOT_RETAINED_BY_GATEWAY'})

    async def resolve(self, actor, alias, *, with_profile=False, account_id=None):
        if self.proxy is None:
            raise HTTPException(503,'inference access preparation unavailable')
        routes=[d for d in await self.authorized_deployments(actor) if d.alias==alias and (account_id is None or d.account_id==account_id)]
        if not routes:raise HTTPException(404,'no allowed deployment')
        if len(routes)!=1:raise HTTPException(409,'account_selection_required')
        try:
            native=await self.proxy.verify_binding(routes[0])
        except httpx.RequestError:
            raise HTTPException(503,'inference metadata unavailable; refresh before a new request') from None
        resolved = replace(routes[0],provider_model_id=native['litellm_params']['model'])
        if with_profile:
            from agents.model_gateway.capabilities import project_profile
            return resolved, native['physical_binding'], project_profile(resolved,native)
        return resolved,native['physical_binding']

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
            logger.warning('gateway_finalization_failed category=FINALIZATION_ERROR invocation_id=%s attempt_id=%s',
                           admitted['invocation_id'],admitted['attempt_id'])
            await self.billing.mark_uncertain(actor.organization_id,admitted['invocation_id'])

    async def chat(self, actor, body, idempotency_key, correlation=None, *, _reexecution=None, expected_profile_revision=None, account_id=None, _routing_decision=None):
        try:
            body=validate_chat(body)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        if not idempotency_key or len(idempotency_key)>200:
            raise HTTPException(400,'Idempotency-Key required (max 200 characters)')
        if account_id is not None:
            d,physical_binding,profile=await self.resolve(actor,body['model'],with_profile=True,account_id=account_id)
        else:
            d,physical_binding,profile=await self.resolve(actor,body['model'],with_profile=True)
        if _routing_decision is not None:
            from agents.model_gateway.routing import validate_choice
            await validate_choice(self, actor, d, _routing_decision, body)
        try: context=task_context(actor,d.account_id,d.currency,correlation or {})
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc
        from agents.model_gateway.capabilities import validate_profile_request
        validate_profile_request(body,profile,expected_profile_revision)
        if 'enable_thinking' in body and d.provider != 'dashscope':
            raise HTTPException(422, 'enable_thinking is unsupported by this deployment')
        for capability in ('tools','stream'):
            if body.get(capability) and capability not in d.capabilities:
                raise HTTPException(422,'unsupported capability: '+capability)
        from agents.model_gateway.capabilities import task_token_reservation_bound
        token_bound=task_token_reservation_bound(body,profile)
        credential=await self.credential_for(actor,d)
        # The account and funding are server-bound and included in the immutable request receipt.
        try:
            admission=dict(organization_id=actor.organization_id,account_id=d.account_id,
                deployment_id=d.deployment_id,funding=d.funding,
                currency=d.currency,max_cost=d.max_cost,attribution={'actor_user_id':actor.user_id,
                **({'routing_decision': _routing_decision} if _routing_decision is not None else {}),
                'native_credential_generation':credential.generation,
                'native_desired_revision':credential.desired_revision,
                'native_target_instance_id':credential.target_instance_id,
                'requested_alias':d.alias,'workspace_id':actor.workspace_id,**{k:v for k,v in (correlation or {}).items() if k in {'session_id','run_id','purpose'}},
                **({'task_context':context} if context else {}),
                'task_root_explicit':bool((correlation or {}).get('root_run_id')),
                **({'token_reservation_bound':token_bound} if token_bound is not None else {}),'deployment_revision':d.revision,'capability_revision':profile['revision'],
                'config_fingerprint':d.config_fingerprint,'region':d.region,'protocol':d.protocol,
                'provider':d.provider,'customer_price_version':d.customer_price_version,
                'supplier_price_version':d.supplier_price_version,'fx_version':d.fx_version,
                **({'normalizer_version':d.normalizer_version,'inclusion_rules_version':d.inclusion_rules_version}
                   if d.normalizer_version is not None else {}),
                'physical_binding':physical_binding,'provider_model_id':d.provider_model_id,'model_version':d.model_version,'version_status':d.version_status})
            if _reexecution is None:
                admitted=await self.billing.admit(**admission,operation='chat.completions',idempotency_key=idempotency_key,request_hash=canonical_hash({'request':body,'task_context':context}) if context and set(correlation or {})-{'session_id','run_id','purpose'} else canonical_hash(body))
            else:
                predecessor,command,parent=_reexecution
                admitted=await self.billing.admit_reexecution(**admission,actor_user_id=actor.user_id,workspace_id=actor.workspace_id,predecessor_id=predecessor,command_id=idempotency_key,command=command)
        except ValueError as exc:
            raise self._admission_error(exc) from exc
        if not admitted['created']:
            if admitted.get('attribution',{}).get('actor_user_id') != actor.user_id:
                raise HTTPException(409,'idempotency key already used')
            if _reexecution is not None: return await self._reexecution_status(actor,admitted)
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
            response=await self.proxy.chat({**body,'model':d.proxy_model,'semantier_task_context':admitted['attribution'].get('task_context',{})},admitted['invocation_id'],admitted['attempt_id'],
                credential=credential,actor=actor,deployment=d)
        except (httpx.ConnectError,httpx.ConnectTimeout,httpx.PoolTimeout) as exc:
            await self.billing.fail_before_send(actor.organization_id,admitted['invocation_id'])
            raise HTTPException(502,'private Proxy connection unavailable before send') from exc
        except BaseException:
            await self.billing.mark_uncertain(actor.organization_id,admitted['invocation_id'])
            raise
        headers={'x-semantier-invocation-id':admitted['invocation_id'],'x-semantier-attempt-id':admitted['attempt_id']}
        if _reexecution is not None:
            headers['x-semantier-predecessor-id']=_reexecution[0]
        if response.status_code >= 400:
            # Server-owned correlation only: Proxy bodies/headers may contain secrets.
            logger.warning('private Proxy invocation rejected: status=%d invocation_id=%s attempt_id=%s',
                           response.status_code, admitted['invocation_id'], admitted['attempt_id'])
            await response.aclose()
            await self.billing.mark_uncertain(actor.organization_id,admitted['invocation_id'])
            raise HTTPException(502,'private Proxy invocation rejected')
        if body.get('stream'):
            async def chunks():
                try:
                    async with asyncio.timeout(60):
                        async for chunk in response.aiter_bytes():
                            yield chunk
                finally:
                    # SDKs normally close after [DONE]. ASGI disconnect cancellation
                    # must not interrupt native-receipt-based financial finalization.
                    # finish keeps incomplete/conflicting native evidence held.
                    with anyio.CancelScope(shield=True):
                        try:
                            await response.aclose()
                        finally:
                            await self.finish(actor,d,admitted)
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
