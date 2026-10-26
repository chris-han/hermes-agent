"""Finite native management operations, scoped object references and actor audit."""
import hashlib
import re
from urllib.parse import quote, urlsplit
from contracts.model_gateway import canonical_hash
from fastapi import HTTPException


def redact(value):
    if isinstance(value,list):
        return [redact(v) for v in value]
    if isinstance(value,dict):
        return {k:('[REDACTED]' if (k.lower() in {'key','access_key','aws_access_key_id','aws_secret_access_key'} or any(s in k.lower() for s in ('api_key','token','secret','password','authorization','credential'))) else redact(v)) for k,v in value.items()}
    return value


def provider_credential_prefix(organization_id):
    return 'SEMANTIER_PROVIDER_'+hashlib.sha256(organization_id.encode()).hexdigest()[:12].upper()+'_'


class AdminFacade:
    def __init__(self, service):
        self.service=service

    async def models(self, actor):
        data=await self.service.proxy.operation('GET','/model/info')
        rows=data.get('data',[]) if isinstance(data,dict) else []
        visible=[]
        for row in rows:
            mid=str(row.get('model_info',{}).get('id',''))
            if await self.service.store.owns(actor.organization_id,'model',mid):
                visible.append(row)
        return {'data':redact(visible)}

    async def mutate_model(self, actor, body, *, delete=False):
        if not isinstance(body,dict):
            raise HTTPException(422,'object required')
        if delete:
            if set(body)!={'id'} or not await self.service.store.owns(actor.organization_id,'model',body['id']):
                raise HTTPException(404,'model not found')
            await self.service.store.audit(actor.organization_id,actor.user_id,'model.delete',body['id'])
            result=await self.service.proxy.operation('POST','/model/delete',body)
            await self.service.store.invalidate(actor.organization_id,body['id'])
            return redact(result)
        if set(body)-{'model_name','litellm_params','model_info'}:
            raise HTTPException(422,'unsupported native model fields')
        params=body.get('litellm_params',{})
        if not isinstance(params,dict) or set(params)-{'model','api_key','api_base','api_version','aws_region_name','aws_access_key_id','aws_secret_access_key','aws_session_token','timeout','num_retries','max_retries','fallbacks'}:
            raise HTTPException(422,'unsupported provider configuration')
        if 'api_base' in params:
            endpoint=urlsplit(params['api_base']) if isinstance(params['api_base'],str) else None
            if not endpoint or endpoint.scheme not in {'http','https'} or not endpoint.netloc or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
                raise HTTPException(422,'provider endpoint must not embed credentials or query parameters')
        info=body.get('model_info',{})
        if not isinstance(info,dict) or set(info)-{'id'}:
            raise HTTPException(422,'unsupported model metadata')
        timeout = params.get('timeout', 10.0)
        if type(timeout) not in (int, float) or not 1 <= timeout <= 120:
            raise HTTPException(422, 'provider timeout must be between 1 and 120 seconds')
        controlled={'timeout':timeout,'num_retries':0,'max_retries':0,'fallbacks':[]}
        if any(key in params and params[key]!=value for key,value in controlled.items()):
            raise HTTPException(422,'native retry/fallback policy cannot be changed')
        if any(not isinstance(value,str) for key,value in params.items() if key not in controlled):
            raise HTTPException(422,'provider configuration values must be strings')
        if not isinstance(body.get('model_name'),str) or not body['model_name'].isascii():
            raise HTTPException(422,'ASCII model name required')
        mid=info.get('id')
        if mid is not None and not isinstance(mid,str):
            raise HTTPException(422,'native model identity must be a string')
        if mid and not await self.service.store.owns(actor.organization_id,'model',mid):
            raise HTTPException(404,'model not found')
        secret_fields={'api_key','aws_access_key_id','aws_secret_access_key','aws_session_token'}
        if not mid and any(params.get(key)=='[REDACTED]' for key in secret_fields):
            raise HTTPException(422,'new native deployment requires explicit credentials')
        if mid:
            params={key:value for key,value in params.items() if key not in secret_fields or value!='[REDACTED]'}
        prefix='os.environ/'+provider_credential_prefix(actor.organization_id)
        for key in secret_fields:
            if key in params and (not isinstance(params[key],str) or not params[key].startswith(prefix) or not re.fullmatch(r'[A-Z][A-Z0-9_]*',params[key][len(prefix):])):
                raise HTTPException(422,'provider credentials require organization-scoped private server environment references')
        if not body.get('model_name') or not params.get('model'):
            raise HTTPException(422,'native model binding required')
        namespace='smg_'+hashlib.sha256(actor.organization_id.encode()).hexdigest()[:12]+'/'
        name=body['model_name']
        if not name.startswith(namespace):
            name=namespace+name
        body={**body,'model_name':name}
        await self.service.store.audit(actor.organization_id,actor.user_id,'model.update' if mid else 'model.create',mid or body['model_name'],canonical_hash(body))
        body={**body,'litellm_params':{**params,**controlled}}
        result=await self.service.proxy.operation('PATCH' if mid else 'POST',
            '/model/'+quote(mid,safe='')+'/update' if mid else '/model/new',body)
        new_id=(result.get('model_info') or {}).get('id') or result.get('id') or mid
        if not new_id:
            raise HTTPException(502,'native model identity missing')
        await self.service.store.register_object(actor.organization_id,'model',str(new_id))
        if mid:
            await self.service.store.invalidate(actor.organization_id,mid)
        return redact(result)

    async def keys(self,actor):
        data=await self.service.proxy.operation('GET','/key/list',params={'return_full_object':'true'})
        rows=data.get('keys',[]) if isinstance(data,dict) else []
        visible=[]
        for row in rows:
            if isinstance(row,dict) and await self.service.store.owns(actor.organization_id,'key',hashlib.sha256(str(row.get('token') or row.get('key') or '').encode()).hexdigest()):
                visible.append(self.key_view(actor,{**row,'semantier_key_id':hashlib.sha256(str(row.get('token') or row.get('key') or '').encode()).hexdigest()}))
        return {'keys':redact(visible)}

    @staticmethod
    def key_view(actor,row):
        prefix='smg_'+hashlib.sha256(actor.organization_id.encode()).hexdigest()[:12]+'/'
        alias=row.get('key_alias')
        if isinstance(alias,str) and alias.startswith(prefix):
            return {**row,'key_alias':alias[len(prefix):]}
        return row

    async def create_key(self,actor,body):
        if not isinstance(body,dict) or set(body)-{'key_alias','models','duration'}:
            raise HTTPException(422,'unsupported key fields; native financial controls unavailable')
        alias=body.get('key_alias')
        if alias is not None:
            if not isinstance(alias,str) or not alias:
                raise HTTPException(422,'nonempty key alias required')
            prefix='smg_'+hashlib.sha256(actor.organization_id.encode()).hexdigest()[:12]+'/'
            body={**body,'key_alias':alias if alias.startswith(prefix) else prefix+alias}
        models=body.get('models')
        if not isinstance(models,list) or not models or not all(isinstance(model,str) for model in models):
            raise HTTPException(422,'explicit model scope required')
        allowed={d.proxy_model for d in await self.service.store.deployments(actor.organization_id) if d.enabled}
        if not set(models)<=allowed:
            raise HTTPException(403,'foreign model scope')
        await self.service.store.audit(actor.organization_id,actor.user_id,'key.create',str(body.get('key_alias','')))
        result=await self.service.proxy.operation('POST','/key/generate',{**body,'metadata':{'semantier_organization_id':actor.organization_id}})
        # Native operational keys never become client model credentials: all calls use Semantier admission.
        token=result.get('token') or result.get('key')
        if token:
            await self.service.store.register_object(actor.organization_id,'key',hashlib.sha256(str(token).encode()).hexdigest())
        return redact(self.key_view(actor,{**result,'semantier_key_id':hashlib.sha256(str(token).encode()).hexdigest() if token else None}))


    async def set_model_blocked(self,actor,model_id,blocked):
        if not await self.service.store.owns(actor.organization_id,'model',model_id):
            raise HTTPException(404,'model not found')
        operation='model.block' if blocked else 'model.unblock'
        await self.service.store.audit(actor.organization_id,actor.user_id,operation,model_id)
        result=await self.service.proxy.operation('POST','/model/block' if blocked else '/model/unblock',{'model_id':model_id})
        # Unblocking the registry never reactivates an invalidated policy snapshot.
        await self.service.store.invalidate(actor.organization_id,model_id)
        return redact(result)
