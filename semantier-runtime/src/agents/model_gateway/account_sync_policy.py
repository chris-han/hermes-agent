"""One account policy for genuine synchronous and asynchronous owner transactions.

The synchronous seam lets identity changes join their existing transaction. It
does not open another connection, dispatch HTTP, commit, or install any schema.
"""
from __future__ import annotations

from datetime import datetime,timezone
import json
import os
from uuid import uuid4

from psycopg import sql
from psycopg.types.json import Jsonb


def _queries(org,aid,currency,schema):
    from agents.model_gateway.account_sync_store import _schema,_account_query
    schema=_schema(schema)
    return [
        (_account_query(schema),(org,aid,currency)),
        ('''SELECT a.user_id,a.status,a.revision,u.payload_json,g.permissions
          FROM semantier_gateway.account_access a JOIN semantier_identity.users u USING(user_id)
          LEFT JOIN semantier_gateway.grants g ON g.organization_id=a.organization_id AND g.user_id=a.user_id
          WHERE a.organization_id=%s AND a.account_id=%s AND a.currency=%s ORDER BY a.user_id''',(org,aid,currency)),
        ('''SELECT body FROM semantier_gateway.deployments WHERE organization_id=%s
          AND body->>'account_id'=%s AND body->>'currency'=%s AND (body->>'enabled')::boolean=true ORDER BY deployment_id''',(org,aid,currency)),
        (sql.SQL('''SELECT COALESCE(SUM(remaining),0) AS amount FROM {}.credit_grants
          WHERE organization_id=%s AND account_id=%s AND currency=%s AND expires_at<=statement_timestamp()''').format(sql.Identifier(schema)),(org,aid,currency)),
        (sql.SQL('''SELECT grant_id,starts_at,expires_at FROM {}.credit_grants WHERE organization_id=%s AND account_id=%s AND currency=%s
          AND expired=false AND starts_at<=statement_timestamp() AND (expires_at IS NULL OR expires_at>statement_timestamp()) ORDER BY grant_id''').format(sql.Identifier(schema)),(org,aid,currency)),
    ]


def _assemble(account,access,routes,expired,grants):
    from agents.model_gateway.account_sync_store import _member
    if not account:raise ValueError('Canonical billing account required')
    routes=[r['body'] for r in routes if r['body'].get('funding')==account['funding']]
    models=sorted({r['proxy_model'] for r in routes if isinstance(r.get('proxy_model'),str) and r['proxy_model']})
    token_limits = [r.get('tokens_per_minute', 60000) for r in routes]
    if any(type(limit) is not int or not 1 <= limit <= 1000000 for limit in token_limits):
        raise ValueError('invalid governed tokens_per_minute')
    reason='SUSPENDED' if account['blocked'] else 'BUDGET_UNAVAILABLE' if account['budget_limit']<=account['budget_spent'] else None
    eligible=reason is None and bool(models)
    if account['funding']=='SEMANTIER_CREDITS':
        # Expired credit cannot fund a new dispatch, including held remnants
        # after the financial expiry debit. Valid fully held credit remains
        # eligible; calls already sent finish under their existing hold.
        funded=account['wallet']-expired['amount']>0
        if not funded and reason is None:reason='NO_FUNDING'
        eligible=eligible and funded
    if not models and reason is None:reason='ROUTE_UNAVAILABLE'
    members=[]
    for row in access:
        active=row['status']=='ACTIVE' and _member(json.loads(row['payload_json']),account['organization_id'])
        permitted='model:invoke' in (row['permissions'] or [])
        members.append({'user_id':row['user_id'],'active':active,'invoke_permitted':permitted,
            'eligible':bool(active and permitted and eligible),'access_revision':row['revision']})
    basis=[]
    if account['funding']=='SEMANTIER_CREDITS':
        basis=[{k:(v.astimezone(timezone.utc).isoformat() if isinstance(v,datetime) else v) for k,v in r.items()} for r in grants]
    return {'schema_version':1,'funding':account['funding'],'account_eligible':bool(eligible),'eligibility_reason':reason,
        'models':models,'members':members,'limits':{'rpm_limit':60,'tpm_limit':max(token_limits, default=60000),'max_parallel_requests':2},
        'routes':[{'deployment_id':r['deployment_id'],'revision':r['revision'],'config_fingerprint':r['config_fingerprint']} for r in routes],
        'grant_basis':basis}


def policy_sync(c,org,aid,currency,billing_schema):
    queries=_queries(org,aid,currency,billing_schema)
    account=c.execute(*queries[0]).fetchone()
    access=c.execute(*queries[1]).fetchall()
    routes=c.execute(*queries[2]).fetchall()
    expired=c.execute(*queries[3]).fetchone()
    grants=c.execute(*queries[4]).fetchall()
    return _assemble(account,access,routes,expired,grants)


async def policy_async(c,org,aid,currency,billing_schema):
    queries=_queries(org,aid,currency,billing_schema)
    account=await (await c.execute(*queries[0])).fetchone()
    access=await (await c.execute(*queries[1])).fetchall()
    routes=await (await c.execute(*queries[2])).fetchall()
    expired=await (await c.execute(*queries[3])).fetchone()
    grants=await (await c.execute(*queries[4])).fetchall()
    return _assemble(account,access,routes,expired,grants)


def enqueue_membership_change(c,user_id):
    """Join current per-account desired revisions to the identity mutation."""
    from agents.model_gateway.account_sync_store import _hash,target_configuration,team_id
    present=c.execute("SELECT to_regclass('semantier_gateway.account_scopes') AS owner").fetchone()
    if not present['owner']:
        if os.environ.get('SEMANTIER_STORAGE_BACKEND')=='postgresql' and os.environ.get('SEMANTIER_MODEL_GATEWAY_ENABLED','').lower() in {'1','true','yes','on'}:
            raise ValueError('Explicit account synchronization migration required')
        return 0
    scopes=c.execute('''SELECT DISTINCT s.* FROM semantier_gateway.account_scopes s
      JOIN semantier_gateway.account_access a USING(organization_id,account_id,currency)
      WHERE a.user_id=%s ORDER BY s.organization_id,s.account_id,s.currency''',(user_id,)).fetchall()
    if not scopes:return 0
    target,namespace=target_configuration()
    changed=0
    for scope in scopes:
        org,aid,currency=scope['organization_id'],scope['account_id'],scope['currency']
        policy=policy_sync(c,org,aid,currency,scope['billing_schema'])
        key=(target,namespace,org,aid,currency);digest=_hash(policy)
        current=c.execute('''SELECT * FROM semantier_gateway.account_projections WHERE target_instance_id=%s
          AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s FOR UPDATE''',key).fetchone()
        if current and current['desired_policy_hash']==digest:continue
        revision=current['desired_revision']+1 if current else 1
        c.execute('''INSERT INTO semantier_gateway.account_projections(target_instance_id,environment_namespace,organization_id,account_id,currency,native_team_id,desired_revision,desired_policy,desired_policy_hash,status)
          VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'PENDING') ON CONFLICT(target_instance_id,environment_namespace,organization_id,account_id,currency)
          DO UPDATE SET desired_revision=excluded.desired_revision,desired_policy=excluded.desired_policy,desired_policy_hash=excluded.desired_policy_hash,status='PENDING',last_error_code=NULL,updated_at=now()''',
          (*key,team_id(*key,scope['funding']),revision,Jsonb(policy),digest))
        c.execute('''UPDATE semantier_gateway.account_sync_intents SET status='SUPERSEDED',completed_at=now()
          WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s
          AND revision<%s AND status IN ('PENDING','RETRY_WAIT','NEEDS_ATTENTION','BLOCKED_DEPENDENCY','OWNERSHIP_CONFLICT')''',(*key,revision))
        c.execute('''INSERT INTO semantier_gateway.account_sync_intents(event_id,target_instance_id,environment_namespace,organization_id,account_id,currency,revision,cause)
          VALUES(%s,%s,%s,%s,%s,%s,%s,'membership_change') ON CONFLICT DO NOTHING''',(uuid4().hex,*key,revision))
        changed+=1
    return changed
