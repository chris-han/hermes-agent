"""Finite server provisioning from explicit stdin JSON; no credit issuance or host files."""
from __future__ import annotations
from decimal import Decimal
import hashlib
from contracts.model_gateway import Deployment,native_fingerprint
from agents.model_gateway.runtime import stores


def member(body):
    from agents.gateway_identity import get_user_record,get_user_membership_for_organization
    org,user=body.get('organization_id'),body.get('user_id')
    if not isinstance(org,str) or not isinstance(user,str):
        raise ValueError('existing organization_id and target user_id required')
    record=get_user_record(user)
    membership=get_user_membership_for_organization(user_id=user,organization_id=org)
    if not record or not record.get('workspace_id') or not membership or membership.get('membership_status')!='active':
        raise ValueError('current existing identity and active membership required')
    return org,user,record['workspace_id']


async def provision(config,operation,body):
    if not isinstance(body,dict):
        raise ValueError('JSON object required')
    fields={'account-create':{'organization_id','user_id','account_id','currency','funding','budget_limit'},
            'grant':{'organization_id','user_id','permissions'},
            'account-access':{'organization_id','user_id','account_id','currency','role','status'},
            'sync-retry':{'organization_id','user_id','account_id','currency'},
            'bind':{'organization_id','user_id','deployment'}}
    if operation not in fields or set(body)!=fields[operation]:
        raise ValueError('exact finite provisioning fields required')
    org,user,workspace=member(body)
    store,billing=stores(config)
    if operation=='account-create':
        from agents.model_gateway.account_sync_store import grant_account_access
        async with billing.connection() as connection:
            await billing.create_zero_credit_account(org,body['account_id'],body['currency'],body['funding'],body['budget_limit'],connection=connection)
            await grant_account_access(connection,user_id=user,organization_id=org,account_id=body['account_id'],
                currency=body['currency'],role='owner',billing_schema=config.billing_schema)
        result=await billing.account_summary(org,body['account_id'],body['currency'])
    elif operation=='account-access':
        from agents.model_gateway.account_sync_store import grant_account_access
        async with billing.connection() as connection:
            result=await grant_account_access(connection,user_id=user,organization_id=org,account_id=body['account_id'],
                currency=body['currency'],role=body['role'],status=body['status'],billing_schema=config.billing_schema)
    elif operation=='sync-retry':
        from agents.model_gateway.account_sync_store import AccountSyncStore
        sync=AccountSyncStore(config.dsn,billing_schema=config.billing_schema,
            target_instance_id=config.native_target_instance_id,environment_namespace=config.native_environment_namespace)
        await sync.require_ready()
        result=await sync.retry(org,body['account_id'],body['currency'])
    elif operation=='grant':
        permissions=body['permissions']
        if not isinstance(permissions,list) or not all(isinstance(p,str) for p in permissions):
            raise ValueError('explicit permission list required')
        await store.grant(org,user,permissions)
        result={'target_user_id':user,'workspace_id':workspace,'permissions':sorted(permissions)}
    else:
        data=body['deployment']
        if not isinstance(data,dict) or set(data)-{'deployment_id','alias','native_model_id','revision','account_id','funding','currency','max_cost','provider','region','protocol','credential_binding_id','customer_price_version','supplier_price_version','fx_version','capabilities','model_version','version_status','tokens_per_minute','capability_profile','normalizer_version','inclusion_rules_version'}:
            raise ValueError('unsupported deployment fields')
        from agents.model_gateway.proxy import ProxyClient
        proxy=ProxyClient(config.proxy_url,config.proxy_key)
        try:
            rows=(await proxy.operation('GET','/model/info')).get('data',[])
        finally:
            await proxy.close()
        native=[r for r in rows if r.get('model_info',{}).get('id')==data.get('native_model_id')]
        if len(native)!=1:
            raise ValueError('exact native deployment missing or ambiguous')
        row=native[0]
        namespace='smg_'+hashlib.sha256(org.encode()).hexdigest()[:12]+'/'
        if not isinstance(row.get('model_name'),str) or not row['model_name'].startswith(namespace) or sum(r.get('model_name')==row['model_name'] for r in rows)!=1:
            raise ValueError('unique organization-scoped native model name required')
        prefix={'azure':'azure/','bedrock':'bedrock/','ark':'volcengine/','dashscope':'dashscope/'}.get(data.get('provider'))
        model=row.get('litellm_params',{}).get('model','')
        if not prefix or not isinstance(model,str) or not model.startswith(prefix):
            raise ValueError('native provider model does not match declared provider')
        if not isinstance(data.get('max_cost'),str):
            raise ValueError('max_cost must be an explicit decimal string')
        native_version=row.get('model_info',{}).get('model_version') or row.get('litellm_params',{}).get('model_version')
        if data.get('version_status')=='PINNED' and data.get('model_version')!=native_version:
            raise ValueError('pinned model version must match the native registry')
        data={**data,'organization_id':org,'proxy_model':row['model_name'],'config_fingerprint':native_fingerprint(row),
              'provider_model_id':model,'max_cost':Decimal(data['max_cost'])}
        if 'capabilities' in data:
            data['capabilities']=tuple(data['capabilities'])
        deployment=Deployment(**data)
        account=await billing.account_summary(org,deployment.account_id,deployment.currency)
        if account['funding_source']!=deployment.funding:
            raise ValueError('account funding mismatch')
        if deployment.customer_price_version:
            price=await billing.get_price(deployment.customer_price_version)
            if price.currency!=deployment.currency:
                raise ValueError('customer tariff currency mismatch')
        await store.bind(deployment)
        await store.register_object(org,'model',deployment.native_model_id)
        result={'deployment_id':deployment.deployment_id,'revision':deployment.revision,'native_model_id':deployment.native_model_id,
                'proxy_model':deployment.proxy_model,'config_fingerprint':deployment.config_fingerprint,'credential_view':'MASKED_NATIVE_REGISTRY'}
    await store.audit(org,'server-provisioning',operation,str(result.get('deployment_id') or body.get('account_id') or user),result.get('revision'))
    return {'source':'server-provisioning','organization_id':org,'result':result}
