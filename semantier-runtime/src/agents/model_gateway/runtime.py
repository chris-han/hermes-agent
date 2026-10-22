"""Explicit optional service startup; schema changes require the separate CLI.

Service credentials resolve through the existing secret policy from server environment
references. Authenticated clients never select secret names or database/Proxy URLs.
"""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import dataclass
import os
import re
from typing import Mapping
from agents.tenant_secret_policy import resolve_secret


@dataclass(frozen=True)
class GatewayConfiguration:
    dsn: str
    proxy_url: str
    proxy_key: str
    billing_schema: str = 'semantier_billing'
    native_target_instance_id: str = ''
    native_environment_namespace: str = ''
    credential_key_value: str = ''
    credential_key_ref: str = ''

    @classmethod
    def from_environment(cls, env: Mapping[str,str]):
        if env.get('SEMANTIER_MODEL_GATEWAY_ENABLED','').lower() not in {'1','true','yes','on'}:
            return None
        hosted=env.get('SEMANTIER_STORAGE_BACKEND')=='postgresql'
        target=env.get('SEMANTIER_NATIVE_TARGET_INSTANCE_ID','').strip()
        namespace=env.get('SEMANTIER_NATIVE_ENVIRONMENT_NAMESPACE','').strip()
        if hosted and not target:
            raise ValueError('explicit native target instance required')
        if hosted and not namespace:
            raise ValueError('explicit native environment namespace required')
        for value in (target,namespace):
            if value and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}',value):
                raise ValueError('invalid native target identity')
        def secret(config_key,default,*,required=True):
            reference=env.get(config_key,default)
            if not re.fullmatch('[A-Z][A-Z0-9_]*',reference):
                raise ValueError('invalid model gateway server secret reference')
            resolved=resolve_secret(reference,workspace_env=None,platform_env=env,auth_cache=None)
            if not resolved and required:
                raise ValueError('required model gateway server secret unavailable: '+reference)
            return resolved.value if resolved else ''
        url=env.get('SEMANTIER_MODEL_GATEWAY_PROXY_URL','').strip()
        if not url and not hosted:
            raise ValueError('explicit private Proxy URL required')
        proxy_key=secret('SEMANTIER_MODEL_GATEWAY_PROXY_SECRET_REF','SEMANTIER_MODEL_GATEWAY_PROXY_KEY',required=not hosted)
        credential_ref=env.get('SEMANTIER_NATIVE_CREDENTIAL_KEY_REF','SEMANTIER_NATIVE_CREDENTIAL_KEY')
        if 'SESSION' in credential_ref.upper() or credential_ref in {
            env.get('SEMANTIER_MODEL_GATEWAY_PROXY_SECRET_REF','SEMANTIER_MODEL_GATEWAY_PROXY_KEY'),
            env.get('SEMANTIER_MODEL_GATEWAY_DATABASE_SECRET_REF','SEMANTIER_MODEL_GATEWAY_DATABASE_URL'),
        }:
            raise ValueError('credential encryption requires a separate server secret')
        credential_key=secret('SEMANTIER_NATIVE_CREDENTIAL_KEY_REF','SEMANTIER_NATIVE_CREDENTIAL_KEY',required=False)
        if credential_key and credential_key in {(env.get('SEMANTIER_SESSION_SECRET') or '').strip(),proxy_key}:
            raise ValueError('credential encryption requires a separate server secret')
        return cls(secret('SEMANTIER_MODEL_GATEWAY_DATABASE_SECRET_REF','SEMANTIER_MODEL_GATEWAY_DATABASE_URL'),
                   url,proxy_key,
                   env.get('SEMANTIER_MODEL_GATEWAY_BILLING_SCHEMA','semantier_billing'),
                   target,namespace,credential_key,credential_ref)

    def __repr__(self):
        return 'GatewayConfiguration(<server secrets redacted>)'


def stores(config):
    from agents.model_gateway.store import GatewayStore
    from billing.store import BillingStore
    return GatewayStore(config.dsn),BillingStore(config.dsn,schema=config.billing_schema)


def credential_keyring(value,*,key_id=None):
    """Resolve only the operator-held AEAD key ring; never a signing-key fallback."""
    if not value:
        return None
    from agents.model_gateway.credential_store import credential_keyring as parse,CredentialDependencyError
    try:
        return parse(value,key_id=key_id)
    except CredentialDependencyError:
        return None


async def start_gateway(app, *, env=None):
    platform_env=os.environ if env is None else env
    config=GatewayConfiguration.from_environment(platform_env)
    if config is None:
        app.state.model_gateway=None
        return None
    from tools.environments.local_env_policy import protect_platform_secret_names
    protect_platform_secret_names(platform_env)
    store,billing=stores(config)
    # Fail clearly when the operator has not applied the separate explicit migration.
    async with await store._connection() as connection:
        version=await (await connection.execute('SELECT max(version) AS version FROM semantier_gateway.schema_version')).fetchone()
        if version['version'] != 2:
            raise ValueError('unsupported model gateway schema version')
        await connection.execute('SELECT token_id FROM semantier_gateway.tokens LIMIT 0')
        await connection.execute('SELECT permissions FROM semantier_gateway.grants LIMIT 0')
        await connection.execute('SELECT body FROM semantier_gateway.deployments LIMIT 0')
        await connection.execute('SELECT revision FROM semantier_gateway.deployment_versions LIMIT 0')
    async with billing.connection() as connection:
        financial_version=await (await connection.execute('SELECT max(version) AS version FROM schema_version')).fetchone()
        from billing.store import BILLING_SCHEMA_VERSION
        if financial_version['version'] != BILLING_SCHEMA_VERSION:
            raise ValueError('unsupported financial schema version; explicit migration required')
        await connection.execute('SELECT invocation_id FROM invocations LIMIT 0')
        await connection.execute('SELECT attempt_id FROM raw_observations LIMIT 0')
    from agents.model_gateway.proxy import ProxyClient
    from agents.model_gateway.service import GatewayService
    sync_store=None
    if config.native_target_instance_id and config.native_environment_namespace:
        from agents.model_gateway.account_sync_store import AccountSyncStore
        sync_store=AccountSyncStore(config.dsn,billing_schema=config.billing_schema,
            target_instance_id=config.native_target_instance_id,
            environment_namespace=config.native_environment_namespace,
            keyring=credential_keyring(config.credential_key_value,key_id=config.credential_key_ref))
        await sync_store.require_ready()
    proxy=ProxyClient(config.proxy_url,config.proxy_key) if config.proxy_url and config.proxy_key else None
    service=GatewayService(store=store,billing=billing,proxy=proxy,enabled=True,account_sync_store=sync_store)
    app.state.model_gateway=service
    if sync_store is not None:
        from agents.model_gateway.account_sync import AccountSyncWorker
        service.account_sync_worker=AccountSyncWorker(sync_store,proxy=proxy,keyring=sync_store.keyring)
        service.account_sync_worker.start()
    return service


async def stop_gateway(app):
    service=getattr(app.state,'model_gateway',None)
    if service:
        service.enabled=False
        worker=getattr(service,'account_sync_worker',None)
        if worker:
            await worker.stop()
        tasks=list(getattr(service,"_identity_projection_tasks",()))
        for task in tasks: task.cancel()
        if tasks: await asyncio.gather(*tasks,return_exceptions=True)
        if service.proxy:
            await service.proxy.close()
        app.state.model_gateway=None


async def migrate(config):
    store,billing=stores(config)
    financial=await billing.migrate()
    await store.migrate()
    await store.migrate_account_sync(config.billing_schema)
    async with await store._connection() as connection:
        await connection.execute('SELECT token_id FROM semantier_gateway.tokens LIMIT 0')
    return {'gateway_schema':'semantier_gateway','gateway_schema_version':2,'billing':financial,'status':'ready'}


def main():
    parser=argparse.ArgumentParser(description='Explicit model-gateway schema administration')
    parser.add_argument('operation',choices=['migrate','account-create','account-access','sync-retry','grant','bind'])
    args=parser.parse_args()
    config=GatewayConfiguration.from_environment(os.environ)
    if config is None:
        raise SystemExit('set SEMANTIER_MODEL_GATEWAY_ENABLED=1 with explicit server configuration')
    if args.operation=='migrate':
        result=asyncio.run(migrate(config))
    else:
        import sys
        import json
        from agents.model_gateway.provision import provision
        wire=sys.stdin.buffer.read(65537)
        if len(wire)>65536:
            raise SystemExit('provisioning JSON exceeds 64 KiB')
        result=asyncio.run(provision(config,args.operation,json.loads(wire)))
    import json
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
