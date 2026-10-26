"""Hosted Cloud assignments at the existing profile/configuration and token owners."""
from __future__ import annotations
from fastapi import HTTPException


def validate_assignment(body, profile, tasks):
    if (not isinstance(body, dict) or set(body)-{'scope','provider','model','task','account_id','revision'}
            or body.get('scope') not in {'main','auxiliary'} or body.get('provider')!='semantier-cloud'
            or body.get('model')!=profile['id'] or body.get('account_id')!=profile['account_id']):
        raise HTTPException(422,'scoped Cloud assignment required')
    if body.get('revision')!=profile['revision']:
        raise HTTPException(409,'cloud_profile_revision_changed; refresh before saving')
    task=body.get('task','')
    if body['scope']=='main' and task or body['scope']=='auxiliary' and task not in tasks:
        raise HTTPException(422,'supported auxiliary purpose required')
    required={'vision':'vision','contextgraph_ai_ground':'structured_output'}.get(task)
    if required and profile['capabilities'].get(required) is not True:
        raise HTTPException(422,'unsupported or unknown capability: '+required)
    if profile['protocol'] not in {'chat','chat_completions','openai_chat_completions'}:
        raise HTTPException(422,'unsupported Cloud protocol')
    if not profile.get('context_window') or not profile.get('max_output_tokens'):
        raise HTTPException(422,'Cloud context/output limits unknown')
    return body['scope'],task


def apply_assignment(profile, *, scope, task, base_url, key_env, organization_id=''):
    from hermes_cli.web_server_config import _apply_model_assignment_sync
    return _apply_model_assignment_sync(scope,'semantier-cloud',profile['id'],task,'',
        cloud_profile=profile, cloud_connection={'base_url':base_url,'key_env':key_env,
            'organization_id':organization_id,'account_id':profile['account_id']})


def supported_tasks():
    # These production seams carry the authenticated task/account binding. Native CLI-only
    # slots remain available to explicit custom providers, but are not Cloud assignments.
    from agents.hermes_current_agent_llm import HermesCurrentAgentStructuredLlm
    return ('compression','title_generation','vision',HermesCurrentAgentStructuredLlm._AUXILIARY_TASK)


def assignment_projection(config, organization_id):
    def project(value, *, main=False):
        if (not isinstance(value,dict) or value.get('organization_id')!=organization_id
                or (main and value.get('provider')!='semantier-cloud')):
            return None
        return {'model':value.get('default',value.get('model','')), 'account_id':value.get('account_id'),
            'revision':value.get('capability_revision'), 'provider':'semantier-cloud'}
    aux=config.get('auxiliary') or {}
    return {'main':project(config.get('model'),main=True), 'auxiliary':{task:project(value) for task,value in aux.items()}}


async def set_assignment(service, actor, ctx, body):
    import asyncio
    import os
    from datetime import datetime,timedelta,timezone
    from hashlib import sha256
    from agents.model_gateway.device_connections import origin
    from runtime_paths import bind_workspace_env
    if not isinstance(body,dict) or not isinstance(body.get('model'),str) or not body['model']:
        raise HTTPException(422,'Cloud model required')
    profile=await service.catalog(actor,body['model'],account_id=body.get('account_id'))
    scope,task=validate_assignment(body,profile,supported_tasks())
    # This is the same server-owned origin used by the existing Cloud connection owner.
    try:
        base=origin(os.environ.get('SEMANTIER_CLOUD_PUBLIC_ORIGIN',''))+'/v1'
    except ValueError:
        raise HTTPException(503,'Cloud connection provisioning unavailable') from None
    def existing_secret():
        from hermes_cli.config import read_user_config_raw,get_env_value
        with bind_workspace_env(ctx.hermes_home):
            cfg=read_user_config_raw()
            record=cfg.get('model') if scope=='main' else (cfg.get('auxiliary') or {}).get(task)
            if not isinstance(record,dict) or record.get('organization_id')!=actor.organization_id or record.get('account_id')!=profile['account_id']:
                return (None,None)
            key=record.get('key_env','')
            return (get_env_value(key),key) if key else (None,None)
    secret,existing_key_env=await asyncio.to_thread(existing_secret)
    token=await service.store.token(secret) if secret else None
    valid=bool(token and (token['user_id'],token['organization_id'],token['workspace_id'],token.get('account_id'),token.get('currency'))==
        (actor.user_id,actor.organization_id,actor.workspace_id,profile['account_id'],profile['currency']) and 'model:invoke' in token['permissions'])
    issued=None
    if not valid:
        issued=await service.store.issue(organization_id=actor.organization_id,user_id=actor.user_id,
            workspace_id=actor.workspace_id,permissions=['model:invoke'],account_id=profile['account_id'],currency=profile['currency'],
            expires_at=datetime.now(timezone.utc)+timedelta(days=30))
        secret=issued['secret']
        key_env='HERMES_CUSTOM_SEMANTIER_CLOUD_'+sha256(issued['token_id'].encode()).hexdigest()[:16].upper()+'_API_KEY'
    else:
        key_env=existing_key_env
    def persist():
        from agents.launcher import _workspace_bootstrap_lock
        from hermes_cli.config import _CONFIG_LOCK,require_readable_config_before_write,save_config,save_env_value_secure
        from hermes_cli.credential_lifecycle import remove_provider_env_credential
        from agents.webapi.system_skill_service import persist_runtime_config_to_workspace_db
        with _workspace_bootstrap_lock(ctx.hermes_home),bind_workspace_env(ctx.hermes_home),_CONFIG_LOCK:
            before=require_readable_config_before_write()
            try:
                if issued:save_env_value_secure(key_env,secret)
                result=apply_assignment(profile,scope=scope,task=task,base_url=base,key_env=key_env,organization_id=actor.organization_id)
                persist_runtime_config_to_workspace_db(workspace_id=actor.workspace_id,target_home=ctx.hermes_home)
                return result
            except Exception:
                save_config(before)
                if issued:remove_provider_env_credential(key_env)
                raise
    try:
        # Check the exact selection again after credential preparation, before configuration writes.
        current=await service.catalog(actor,body['model'],account_id=profile['account_id'])
        validate_assignment(body,current,supported_tasks())
        return await asyncio.to_thread(persist)
    except Exception:
        if issued:await service.store.revoke(actor.organization_id,actor.user_id,issued['token_id'])
        raise


async def patch_configuration(ctx, body):
    """Serialize hosted non-Cloud configuration with the existing assignment owner."""
    import asyncio
    import copy
    from agents.launcher import _workspace_bootstrap_lock
    from runtime_paths import bind_workspace_env

    if not isinstance(body,dict) or set(body)-{'config','env'} or not all(isinstance(body.get(k,{}),dict) for k in ('config','env')):
        raise HTTPException(422,'configuration and environment objects required')
    patch=copy.deepcopy(body.get('config') or {})
    environment=body.get('env') or {}
    def forbidden_reference(value):
        if isinstance(value,str):return '${' in value and '}' in value
        if isinstance(value,list):return any(forbidden_reference(v) for v in value)
        if not isinstance(value,dict):return False
        for key,item in value.items():
            if key in {'__proto__','constructor','prototype'}:return True
            if key in {'account_id','organization_id','capability_revision','credential_user_id'}:return True
            if key in {'key_env','api_key_env','credential_ref'} and str(item or '').startswith(('SEMANTIER_','LITELLM_','HERMES_CUSTOM_SEMANTIER_CLOUD_')):return True
            if key in {'api_key','token'} and str(item or '').startswith('smg_'):return True
            if forbidden_reference(item):return True
        return False
    if forbidden_reference(patch) or 'semantier_gateway' in patch or (isinstance(patch.get('providers'),dict) and 'semantier-cloud' in patch['providers']):
        raise HTTPException(422,'use the validated Cloud model selector')
    import re
    if any(not isinstance(key,str) or not re.fullmatch('[A-Z][A-Z0-9_]*',key) or re.match(r'^(SEMANTIER_|LITELLM_|HERMES_CUSTOM_SEMANTIER_CLOUD_|HERMES_INFERENCE_|HERMES_DEFAULT_MODEL$|HERMES_MODEL$|API_AUTH_KEY$|DATABASE_URL$|SESSION_SECRET$)',key) for key in environment):
        raise HTTPException(422,'infrastructure and managed credentials are server-owned')
    if forbidden_reference(environment):
        raise HTTPException(422,'SecretRef templates must use the credential owner')
    if any(value is not None and not isinstance(value,str) for value in environment.values()):
        raise HTTPException(422,'environment values must be strings or null')

    def persist():
        from hermes_cli.config import (_CONFIG_LOCK,require_readable_config_before_write,save_config,
            save_env_value_secure,remove_env_value,load_env,_deep_merge)
        from hermes_cli.credential_lifecycle import remove_provider_env_credential
        from hermes_cli.web_server_config import _apply_model_assignment_sync,_redact_managed_config
        from agents.webapi.system_skill_service import persist_runtime_config_to_workspace_db
        from agent.secret_scope import set_secret_scope,reset_secret_scope,build_profile_secret_scope,set_multiplex_context,reset_multiplex_context
        with _workspace_bootstrap_lock(ctx.hermes_home),bind_workspace_env(ctx.hermes_home),_CONFIG_LOCK:
            before=require_readable_config_before_write()
            before_env=load_env()
            if 'model' in patch and not isinstance(patch['model'],dict):
                raise HTTPException(422,'use the validated model selector')
            if 'auxiliary' in patch and not isinstance(patch['auxiliary'],dict):
                raise HTTPException(422,'purpose configuration must be an object')
            model=patch.pop('model',None)
            if model is not None and (not isinstance(model,dict) or model.get('provider') in (None,'','auto','semantier-cloud','semantier_gateway') or not model.get('default')):
                raise HTTPException(422,'use the validated Cloud selector or declare an explicit custom provider/model')
            if 'providers' in patch and not isinstance(patch['providers'],dict):
                raise HTTPException(422,'provider connections must be changed individually')
            routing={'provider','model','base_url','api_key','key_env','api_key_env','execution_transport'}
            for task,slot in (patch.get('auxiliary') or {}).items():
                old=(before.get('auxiliary') or {}).get(task) or {}
                if (old.get('account_id') or str(old.get('key_env') or '').startswith('HERMES_CUSTOM_SEMANTIER_CLOUD_')) and (not isinstance(slot,dict) or routing & set(slot)):
                    raise HTTPException(422,'use the validated Cloud purpose selector')
                if isinstance(slot,dict) and slot.get('provider') in {'semantier-cloud','semantier_gateway'}:
                    raise HTTPException(422,'use the validated Cloud purpose selector')
            try:
                for key,value in environment.items():
                    if value in (None,''):remove_provider_env_credential(key)
                    else:save_env_value_secure(key,value)
                current=copy.deepcopy(require_readable_config_before_write())
                for key,value in list(patch.items()):
                    if value is None:
                        current.pop(key,None);patch.pop(key)
                save_config(_deep_merge(current,patch))
                if model is not None:
                    key=model.get('api_key') or ''
                    if model.get('key_env'):
                        key=load_env().get(model['key_env'],'')
                    if forbidden_reference(key) or str(key).startswith('smg_'):
                        raise HTTPException(422,'Cloud credentials require the validated selector')
                    from agent.semantier_gateway import is_gateway_route
                    if is_gateway_route(str(model.get('base_url') or '')):
                        raise HTTPException(422,'Cloud connections require the validated selector')
                    scope=set_secret_scope(build_profile_secret_scope(ctx.hermes_home),profile_home=str(ctx.hermes_home))
                    multiplex=set_multiplex_context(True)
                    try:
                        _apply_model_assignment_sync('main',model['provider'],model['default'],'',model.get('base_url') or '',key)
                        custom_config=require_readable_config_before_write()
                        for field in ('api_mode','context_length','transport'):
                            if field in model:custom_config['model'][field]=model[field]
                        save_config(custom_config)
                    finally:
                        reset_multiplex_context(multiplex);reset_secret_scope(scope)
                persist_runtime_config_to_workspace_db(workspace_id=ctx.workspace_id,target_home=ctx.hermes_home)
                return {'ok':True,'message':'Configuration saved','payload':_redact_managed_config(require_readable_config_before_write())}
            except Exception:
                save_config(before)
                for key in set(load_env())-set(before_env):remove_env_value(key)
                for key,value in before_env.items():save_env_value_secure(key,value)
                raise
    return await asyncio.to_thread(persist)
