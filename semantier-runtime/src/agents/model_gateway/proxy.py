"""Finite private HTTP transport. Provider SDKs and credentials live in Proxy."""
from __future__ import annotations
import httpx
import os
from urllib.parse import urlsplit
from fastapi import HTTPException
from contracts.model_gateway import native_fingerprint
from agents.model_gateway.physical_binding import extract_physical_binding


class ProxyClient:
    def __init__(self, base_url: str, service_key: str, *, timeout=10):
        trusted_internal = (base_url == 'http://litellm:4000'
                            and os.environ.get('SEMANTIER_STORAGE_BACKEND') == 'postgresql'
                            and os.environ.get('SEMANTIER_MODEL_GATEWAY_INTERNAL_PROXY_ENABLED') == '1')
        if not trusted_internal and not base_url.startswith(('http://127.0.0.1:', 'http://localhost:', 'https://')):
            raise ValueError('private Proxy URL required')
        parsed=urlsplit(base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('private Proxy URL must not embed credentials or parameters')
        self.base_url=base_url.rstrip('/')
        self._key=service_key
        self._client=httpx.AsyncClient(timeout=httpx.Timeout(timeout,connect=5),
                                       transport=httpx.AsyncHTTPTransport(retries=0,trust_env=False),follow_redirects=False,trust_env=False)

    async def close(self):
        await self._client.aclose()


    async def verify_binding(self,deployment, *, snapshot=None):
        data=snapshot if snapshot is not None else await self.operation('GET','/model/info')
        rows=data.get('data',[]) if isinstance(data,dict) else []
        candidates=[r for r in rows if r.get('model_name')==deployment.proxy_model]
        if len(candidates)!=1 or str(candidates[0].get('model_info',{}).get('id','')) != deployment.native_model_id:
            raise HTTPException(409,'native deployment snapshot missing or ambiguous')
        if candidates[0].get('model_info',{}).get('blocked') or candidates[0].get('litellm_params',{}).get('blocked'):
            raise HTTPException(403,'native deployment is blocked')
        if native_fingerprint(candidates[0]) != deployment.config_fingerprint:
            raise HTTPException(409,'native deployment snapshot changed; rebind required')
        model=candidates[0].get('litellm_params',{}).get('model')
        if not isinstance(model,str) or not model or (deployment.provider_model_id and deployment.provider_model_id!=model):
            raise HTTPException(409,'native provider model identity mismatch')
        try:
            physical=extract_physical_binding(candidates[0]['litellm_params'],deployment.provider)
        except ValueError as exc:
            raise HTTPException(409,'unsupported native physical binding: '+str(exc)) from exc
        return {**candidates[0],'physical_binding':physical}

    async def chat(self, body, invocation_id, attempt_id, *, credential=None, actor=None, deployment=None):
        from agents.model_gateway.credential_store import ScopedCredential
        import hashlib
        if not isinstance(credential,ScopedCredential) or actor is None or deployment is None:
            raise HTTPException(503,'verified scoped native credential required')
        if (credential.user_id!=actor.user_id or credential.organization_id!=actor.organization_id
            or credential.organization_id!=deployment.organization_id or credential.account_id!=deployment.account_id
            or credential.currency!=deployment.currency or body.get('model')!=deployment.proxy_model
            or deployment.proxy_model not in credential.models):
            raise HTTPException(403,'native credential scope mismatch')
        if hashlib.sha256(credential.secret.encode()).hexdigest()!=credential.native_key_hash:
            raise HTTPException(503,'verified scoped native credential required')
        body=dict(body)
        context=body.pop('semantier_task_context',{})
        metadata={}
        if context.get('schema_version')=='semantier.task-consumption.v1':
            metadata={'tags':['semantier.task:'+context['scope_id']] if context.get('scope_id') else [],
                'spend_logs_metadata':{k:context[k] for k in ('task_id','root_run_id','parent_run_id','run_id','session_id','purpose','business_object_ref','namespace','provenance') if k in context}}
        body={**body,'num_retries':0,'fallbacks':[], 'metadata': {
            **metadata,
            **credential.metadata,'semantier_invocation_id':invocation_id,'semantier_attempt_id':attempt_id}}
        req=self._client.build_request('POST',self.base_url+'/v1/chat/completions',json=body,
                                       headers={'Authorization':'Bearer '+credential.secret},
                                       # Native owner caps provider wait at 120s. Keep
                                       # management calls short, with no retry.
                                       timeout=httpx.Timeout(125, connect=5))
        return await self._client.send(req,stream=True)

    async def management_get(self, path, params):
        response=await self._client.get(self.base_url+path,params=params,headers={"Authorization":"Bearer "+self._key})
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise HTTPException(502, "private Proxy identity lookup rejected")
        return response.json()

    async def operation(self, method, path, body=None, params=None):
        response=await self._client.request(method,self.base_url+path,json=body,params=params,
                                            headers={'Authorization':'Bearer '+self._key})
        if response.status_code >= 400:
            raise HTTPException(502, 'private Proxy operation rejected')
        return response.json()

    async def native_operation(self,method,path,body=None,params=None):
        """Finite control API; failures expose codes, never secret-bearing bodies."""
        from agents.model_gateway.native_account_adapter import NativeAccountError
        allowed={'/user/info','/user/new','/team/info','/team/new','/team/update','/team/member_add','/team/member_delete','/team/block','/team/unblock','/key/info','/key/generate','/key/list','/key/block'}
        if method not in {'GET','POST'} or path not in allowed:raise NativeAccountError('INVALID_POLICY')
        try:
            response=await self._client.request(method,self.base_url+path,json=body,params=params,headers={'Authorization':'Bearer '+self._key})
        except httpx.TimeoutException:raise NativeAccountError('NATIVE_TIMEOUT',retryable=True) from None
        except httpx.RequestError:raise NativeAccountError('NATIVE_UNAVAILABLE',retryable=True) from None
        if response.status_code==404 and method=='GET':return None
        if response.status_code>=400:
            if response.status_code in {401,403}:raise NativeAccountError('BLOCKED_DEPENDENCY')
            if response.status_code==429:
                try:delay=max(0,min(900,float(response.headers.get('Retry-After','0'))))
                except ValueError:
                    from email.utils import parsedate_to_datetime
                    from datetime import datetime,timezone
                    try:delay=max(0,min(900,(parsedate_to_datetime(response.headers['Retry-After'])-datetime.now(timezone.utc)).total_seconds()))
                    except (KeyError,TypeError,ValueError):delay=None
                raise NativeAccountError('NATIVE_RATE_LIMIT',retryable=True,retry_after=delay)
            raise NativeAccountError('NATIVE_UNAVAILABLE' if response.status_code>=500 else 'INVALID_POLICY',retryable=response.status_code>=500)
        try:return response.json()
        except ValueError:raise NativeAccountError('NATIVE_READBACK_INVALID',retryable=True) from None
