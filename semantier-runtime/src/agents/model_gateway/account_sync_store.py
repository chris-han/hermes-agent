"""Account-scoped native projection authority and fenced durable synchronization.

All joined commands use the caller's genuine PostgreSQL transaction. No HTTP,
implicit migration, commit, connection replacement or LiteLLM SQL belongs here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
from uuid import UUID, uuid4, uuid5
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

SCOPE_COLUMNS=('target_instance_id','environment_namespace','organization_id','account_id','currency')
ROLES={'owner','billing_admin','reader','member'}
STATUSES={'ACTIVE','REVOKED','SUSPENDED'}
NAMESPACE=UUID('50e1e57e-3486-4a08-9ea6-529ec05f3117')


def _schema(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,62}',value):
        raise ValueError('Explicit billing schema required')
    return value


def target_configuration():
    target=os.environ.get('SEMANTIER_NATIVE_TARGET_INSTANCE_ID','').strip()
    namespace=os.environ.get('SEMANTIER_NATIVE_ENVIRONMENT_NAMESPACE','').strip()
    if not target or not namespace or not target.isascii() or not namespace.isascii():
        raise ValueError('Explicit native target instance and environment namespace required')
    return target,namespace


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def team_id(target,namespace,org,account,currency,funding):
    return str(uuid5(NAMESPACE,json.dumps([target,namespace,org,account,currency,funding],separators=(',',':'))))


def _member(user,org):
    # Reuse the identity owner's pure normalization rules, never its DB-loading
    # membership presentation function: signup must see this transaction's rows.
    from agents.gateway_identity import _canonical_organization_id, _normalize_membership_status
    for member in user.get('organization_memberships',[]) if isinstance(user.get('organization_memberships'),list) else ():
        if isinstance(member,dict) and _canonical_organization_id(str(member.get('organization_id') or ''))==org:
            return _normalize_membership_status(str(member.get('membership_status') or ''))=='active'
    return (_canonical_organization_id(str(user.get('organization_id') or ''))==org
            and _normalize_membership_status(str(user.get('membership_status') or 'active'))=='active')


def _account_query(schema):
    return sql.SQL('SELECT * FROM {}.accounts WHERE organization_id=%s AND account_id=%s AND currency=%s FOR UPDATE').format(sql.Identifier(_schema(schema)))


def _seed_values(c,*,user_id,organization_id,account_id,currency,funding,billing_schema):
    account=c.execute(_account_query(billing_schema),(organization_id,account_id,currency)).fetchone()
    if not account or account['funding']!=funding:raise ValueError('Canonical billing account required')
    user=c.execute('SELECT payload_json FROM semantier_identity.users WHERE user_id=%s',(user_id,)).fetchone()
    if not user or not _member(json.loads(user['payload_json']),organization_id):raise ValueError('Current canonical membership required')
    c.execute('INSERT INTO semantier_gateway.account_scopes VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',(organization_id,account_id,currency,billing_schema,funding))
    retained=c.execute('SELECT billing_schema,funding FROM semantier_gateway.account_scopes WHERE organization_id=%s AND account_id=%s AND currency=%s',(organization_id,account_id,currency)).fetchone()
    if retained!=dict(billing_schema=billing_schema,funding=funding):raise ValueError('Account authority scope conflict')


def seed_personal_sync(c,*,user_id,organization_id,account_id,currency='CNY',funding='SEMANTIER_CREDITS',billing_schema='semantier_billing'):
    """Idempotent insertion only; authenticated repair never reactivates old access."""
    target,namespace=target_configuration()
    _seed_values(c,user_id=user_id,organization_id=organization_id,account_id=account_id,currency=currency,funding=funding,billing_schema=billing_schema)
    c.execute("INSERT INTO semantier_gateway.account_access(organization_id,account_id,currency,user_id,role,status) VALUES(%s,%s,%s,%s,'owner','ACTIVE') ON CONFLICT DO NOTHING",(organization_id,account_id,currency,user_id))
    from agents.model_gateway.account_sync_policy import policy_sync
    policy=policy_sync(c,organization_id,account_id,currency,billing_schema)
    scope=(target,namespace,organization_id,account_id,currency)
    c.execute('''INSERT INTO semantier_gateway.account_projections(target_instance_id,environment_namespace,organization_id,account_id,currency,native_team_id,desired_revision,desired_policy,desired_policy_hash,status)
      VALUES(%s,%s,%s,%s,%s,%s,1,%s,%s,'PENDING') ON CONFLICT DO NOTHING''',(*scope,team_id(*scope,funding),Jsonb(policy),_hash(policy)))
    c.execute('''INSERT INTO semantier_gateway.account_sync_intents(event_id,target_instance_id,environment_namespace,organization_id,account_id,currency,revision,cause)
      SELECT %s,target_instance_id,environment_namespace,organization_id,account_id,currency,desired_revision,'signup' FROM semantier_gateway.account_projections
      WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s AND desired_revision=1 ON CONFLICT DO NOTHING''',(uuid4().hex,*scope))
    retained=c.execute('SELECT desired_revision,status FROM semantier_gateway.account_projections WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s',scope).fetchone()
    return {'desired_revision':retained['desired_revision'],'gateway_sync_status':retained['status'],'inference_ready':False}


def account_provisioning(c,*,user_id,organization_id,workspace_id,billing_schema,account_id=None,currency=None):
    """Read-only customer projection; ambiguity never selects another account."""
    user=c.execute('SELECT payload_json FROM semantier_identity.users WHERE user_id=%s',(user_id,)).fetchone()
    if not user:return {'local_status':'REPAIR_REQUIRED','inference_ready':False,'reason':'NO_ACCESS'}
    payload=json.loads(user['payload_json'])
    if payload.get('workspace_id')!=workspace_id or not _member(payload,organization_id):
        return {'local_status':'REPAIR_REQUIRED','inference_ready':False,'reason':'NO_ACCESS'}
    target,namespace=target_configuration()
    rows=c.execute('''SELECT a.account_id,a.currency,p.status,p.desired_revision,p.applied_revision,p.desired_policy,p.last_verified_at,p.desired_policy_hash,p.observed_state
      FROM semantier_gateway.account_access a JOIN semantier_gateway.account_scopes s USING(organization_id,account_id,currency)
      LEFT JOIN semantier_gateway.account_projections p ON p.organization_id=a.organization_id AND p.account_id=a.account_id AND p.currency=a.currency AND p.target_instance_id=%s AND p.environment_namespace=%s
      WHERE a.user_id=%s AND a.organization_id=%s AND a.status='ACTIVE' AND s.billing_schema=%s AND (%s::text IS NULL OR a.account_id=%s) AND (%s::text IS NULL OR a.currency=%s) ORDER BY a.account_id,a.currency''',(target,namespace,user_id,organization_id,_schema(billing_schema),account_id,account_id,currency,currency)).fetchall()
    if not rows:return {'local_status':'REPAIR_REQUIRED','inference_ready':False,'reason':'NO_ACCOUNT_ACCESS'}
    if len(rows)!=1:return {'local_status':'READY','organization_id':organization_id,'inference_ready':False,'reason':'ACCOUNT_SELECTOR_REQUIRED','gateway_sync_status':'PENDING'}
    row=rows[0];ready=row['status']=='SYNCED' and row['desired_revision']==row['applied_revision'] and any(m['user_id']==user_id and m['eligible'] for m in (row['desired_policy'] or {}).get('members',[]))
    credential=None
    if ready:
        credential=c.execute('''SELECT * FROM semantier_gateway.native_credentials WHERE target_instance_id=%s AND environment_namespace=%s
          AND organization_id=%s AND account_id=%s AND currency=%s AND user_id=%s AND status='ACTIVE' AND verified_at IS NOT NULL AND policy_revision=%s AND policy_hash=%s''',
          (target,namespace,organization_id,row['account_id'],row['currency'],user_id,row['desired_revision'],row['desired_policy_hash'])).fetchone()
        observed=row['observed_state'] or {}
        ready=bool(credential and observed.get('team_blocked') is False and observed.get('keys',{}).get(user_id)==credential['native_key_hash'])
    from agents.model_gateway.credential_store import dependency_keyring,CredentialDependencyError
    keyring=dependency_keyring()
    dependency=None if keyring is not None else 'BLOCKED_DEPENDENCY'
    if ready and credential and keyring is not None:
        try:
            secret=keyring.decrypt(credential)
            if hashlib.sha256(secret.encode()).hexdigest()!=credential['native_key_hash']:dependency='BLOCKED_DEPENDENCY'
        except CredentialDependencyError:dependency='BLOCKED_DEPENDENCY'
    return {'local_status':'READY','organization_id':organization_id,'billing_account_id':row['account_id'],'currency':row['currency'],'gateway_sync_status':dependency or row['status'] or 'PENDING','inference_ready':ready and not dependency,'reason':'PREPARING' if dependency else None if ready else 'PREPARING' if (row['desired_policy'] or {}).get('account_eligible') else 'NO_ACCESS'}


async def enqueue_current(c,org,aid,currency,cause,*,billing_schema=None):
    """Joined current-state enqueue; implementation follows the frozen owner seam."""
    return await AccountSyncStore.enqueue_on_connection(c,org,aid,currency,cause,billing_schema=billing_schema)


async def grant_account_access(c,*,user_id,organization_id,account_id,currency,role='member',status='ACTIVE',billing_schema=None):
    return await AccountSyncStore.grant_on_connection(c,user_id=user_id,organization_id=organization_id,account_id=account_id,currency=currency,role=role,status=status,billing_schema=billing_schema)


class AccountSyncStore:
    def __init__(self,dsn,*,billing_schema='semantier_billing',target_instance_id=None,environment_namespace=None,keyring=None):
        self.dsn,self.billing_schema,self.keyring=dsn,_schema(billing_schema),keyring
        self.target_instance_id,self.environment_namespace=(target_instance_id,environment_namespace) if target_instance_id and environment_namespace else target_configuration()

    async def connection(self):
        import psycopg
        return await psycopg.AsyncConnection.connect(self.dsn,row_factory=dict_row)


    async def require_ready(self):
        async with await self.connection() as c:
            version=await (await c.execute('SELECT max(version) AS version FROM semantier_gateway.schema_version')).fetchone()
            if version['version']!=2:raise ValueError('Explicit account synchronization migration required')
            expected={'account_scope_organization':('account_scopes','semantier_identity.organizations',['organization_id']),
                      'account_scope_financial':('account_scopes',self.billing_schema+'.accounts',['organization_id','account_id','currency']),
                      'account_access_user':('account_access','semantier_identity.users',['user_id']),
                      'native_mapping_user':('native_user_mappings','semantier_identity.users',['user_id'])}
            for name,(table,parent,columns) in expected.items():
                relation=await (await c.execute('''SELECT c.contype='f' AND c.convalidated AND c.confrelid=to_regclass(%s)
                    AND ARRAY(SELECT a.attname::text FROM unnest(c.conkey) WITH ORDINALITY k(n,i) JOIN pg_catalog.pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=k.n ORDER BY k.i)=%s::text[]
                    AND ARRAY(SELECT a.attname::text FROM unnest(c.confkey) WITH ORDINALITY k(n,i) JOIN pg_catalog.pg_attribute a ON a.attrelid=c.confrelid AND a.attnum=k.n ORDER BY k.i)=%s::text[] AS valid
                    FROM pg_catalog.pg_constraint c WHERE c.conrelid=to_regclass(%s) AND c.conname=%s''',
                    (parent,columns,columns,'semantier_gateway.'+table,name))).fetchone()
                if not relation or not relation['valid']:raise ValueError('Explicit account authority relation migration required')
            await c.execute('SELECT ciphertext,nonce,encryption_key_id FROM semantier_gateway.native_credentials LIMIT 0')
        return {'schema_version':2,'status':'ready'}

    @staticmethod
    async def _policy(c,org,aid,currency,billing_schema):
        from agents.model_gateway.account_sync_policy import policy_async
        return await policy_async(c,org,aid,currency,billing_schema)

    @staticmethod
    async def _write_current(c,org,aid,currency,cause,*,billing_schema,target,namespace):
        registered=await (await c.execute('SELECT * FROM semantier_gateway.account_scopes WHERE organization_id=%s AND account_id=%s AND currency=%s',(org,aid,currency))).fetchone()
        if not registered:return {'registered':False,'desired_revision':None,'gateway_sync_status':'BLOCKED_DEPENDENCY','inference_eligible':False}
        schema=_schema(billing_schema or registered['billing_schema'])
        if schema!=registered['billing_schema']:raise ValueError('Account financial authority scope mismatch')
        policy=await AccountSyncStore._policy(c,org,aid,currency,schema)
        scope=(target,namespace,org,aid,currency);digest=_hash(policy)
        current=await (await c.execute('''SELECT * FROM semantier_gateway.account_projections WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s FOR UPDATE''',scope)).fetchone()
        if current and current['desired_policy_hash']==digest:
            return {'registered':True,'desired_revision':current['desired_revision'],'gateway_sync_status':current['status'],'inference_eligible':policy['account_eligible']}
        revision=current['desired_revision']+1 if current else 1
        await c.execute('''INSERT INTO semantier_gateway.account_projections(target_instance_id,environment_namespace,organization_id,account_id,currency,native_team_id,desired_revision,desired_policy,desired_policy_hash,status)
          VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'PENDING') ON CONFLICT(target_instance_id,environment_namespace,organization_id,account_id,currency)
          DO UPDATE SET desired_revision=excluded.desired_revision,desired_policy=excluded.desired_policy,desired_policy_hash=excluded.desired_policy_hash,status='PENDING',last_error_code=NULL,updated_at=now()''',
          (*scope,team_id(*scope,registered['funding']),revision,Jsonb(policy),digest))
        await c.execute('''UPDATE semantier_gateway.account_sync_intents SET status='SUPERSEDED',completed_at=now()
          WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s AND revision<%s AND status IN ('PENDING','RETRY_WAIT','NEEDS_ATTENTION','BLOCKED_DEPENDENCY','OWNERSHIP_CONFLICT')''',(*scope,revision))
        await c.execute('''INSERT INTO semantier_gateway.account_sync_intents(event_id,target_instance_id,environment_namespace,organization_id,account_id,currency,revision,cause)
          VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',(uuid4().hex,*scope,revision,str(cause)[:80]))
        return {'registered':True,'desired_revision':revision,'gateway_sync_status':'PENDING','inference_eligible':policy['account_eligible']}

    @staticmethod
    async def enqueue_on_connection(c,org,aid,currency,cause,*,billing_schema=None):
        present=await (await c.execute("SELECT to_regclass('semantier_gateway.account_projections') AS table_name")).fetchone()
        if not present['table_name']:
            if os.environ.get('SEMANTIER_STORAGE_BACKEND')=='postgresql' and os.environ.get('SEMANTIER_MODEL_GATEWAY_ENABLED','').lower() in {'1','true','yes','on'}:
                raise ValueError('Explicit account synchronization migration required')
            return {'registered':False,'desired_revision':None,'gateway_sync_status':'BLOCKED_DEPENDENCY','inference_eligible':False}
        registered=await (await c.execute('SELECT 1 FROM semantier_gateway.account_scopes WHERE organization_id=%s AND account_id=%s AND currency=%s',(org,aid,currency))).fetchone()
        if not registered:return {'registered':False,'desired_revision':None,'gateway_sync_status':'BLOCKED_DEPENDENCY','inference_eligible':False}
        identity=await (await c.execute("SELECT to_regclass('semantier_identity.users') AS table_name")).fetchone()
        if not identity['table_name'] and os.environ.get('SEMANTIER_STORAGE_BACKEND')!='postgresql':
            return {'registered':False,'desired_revision':None,'gateway_sync_status':'BLOCKED_DEPENDENCY','inference_eligible':False}
        target,namespace=target_configuration()
        return await AccountSyncStore._write_current(c,org,aid,currency,cause,billing_schema=billing_schema,target=target,namespace=namespace)

    @staticmethod
    async def grant_on_connection(c,*,user_id,organization_id,account_id,currency,role,status,billing_schema=None):
        if role not in ROLES or status not in STATUSES:raise ValueError('Unknown account access policy')
        schema=_schema(billing_schema or os.environ.get('SEMANTIER_MODEL_GATEWAY_BILLING_SCHEMA','semantier_billing'))
        account=await (await c.execute(_account_query(schema),(organization_id,account_id,currency))).fetchone()
        user=await (await c.execute('SELECT payload_json FROM semantier_identity.users WHERE user_id=%s',(user_id,))).fetchone()
        if not account or not user or not _member(json.loads(user['payload_json']),organization_id):raise PermissionError('Current canonical account and membership required')
        await c.execute('INSERT INTO semantier_gateway.account_scopes VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',(organization_id,account_id,currency,schema,account['funding']))
        retained=await (await c.execute('SELECT billing_schema,funding FROM semantier_gateway.account_scopes WHERE organization_id=%s AND account_id=%s AND currency=%s',(organization_id,account_id,currency))).fetchone()
        if retained!=dict(billing_schema=schema,funding=account['funding']):raise ValueError('Account authority scope conflict')
        await c.execute('''INSERT INTO semantier_gateway.account_access(organization_id,account_id,currency,user_id,role,status) VALUES(%s,%s,%s,%s,%s,%s)
          ON CONFLICT(organization_id,account_id,currency,user_id) DO UPDATE SET role=excluded.role,status=excluded.status,
          revision=account_access.revision+1,updated_at=now() WHERE (account_access.role,account_access.status) IS DISTINCT FROM (excluded.role,excluded.status)''',(organization_id,account_id,currency,user_id,role,status))
        return await enqueue_current(c,organization_id,account_id,currency,'account_access',billing_schema=schema)

    async def retry(self,org,aid,currency):
        scope=(self.target_instance_id,self.environment_namespace,org,aid,currency)
        async with await self.connection() as c:
            current=await (await c.execute('''SELECT desired_revision FROM semantier_gateway.account_projections WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s FOR UPDATE''',scope)).fetchone()
            if not current:raise ValueError('Registered current account projection required')
            cursor=await c.execute('''UPDATE semantier_gateway.account_sync_intents SET status='PENDING',available_at=now(),failures=0,lease_owner=NULL,lease_token=NULL,lease_until=NULL,last_error_code=NULL,completed_at=NULL
              WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s AND revision=%s AND status IN ('NEEDS_ATTENTION','BLOCKED_DEPENDENCY','OWNERSHIP_CONFLICT','RETRY_WAIT','DONE')''',(*scope,current['desired_revision']))
            if cursor.rowcount:await c.execute('''UPDATE semantier_gateway.account_projections SET status='PENDING',last_error_code=NULL WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s''',scope)
        return {'desired_revision':current['desired_revision'],'retry_requested':bool(cursor.rowcount)}


    def _scope(self,row):return tuple(row[k] for k in SCOPE_COLUMNS)

    async def claim(self,owner,*,limit=2,lease_seconds=60):
        if not 1<=limit<=2 or not 45<lease_seconds<=120:raise ValueError('Bounded aggregate lease required')
        claims=[]
        async with await self.connection() as c:
            await c.execute('''UPDATE semantier_gateway.account_sync_intents SET status='SUPERSEDED',completed_at=now() WHERE event_id IN (
              SELECT i.event_id FROM semantier_gateway.account_sync_intents i JOIN semantier_gateway.account_projections p USING(target_instance_id,environment_namespace,organization_id,account_id,currency)
              WHERE i.target_instance_id=%s AND i.environment_namespace=%s AND i.revision<>p.desired_revision AND i.status='APPLYING' AND i.lease_until<=now() LIMIT 20)''',(self.target_instance_id,self.environment_namespace))
            rows=await (await c.execute('''SELECT i.event_id FROM semantier_gateway.account_sync_intents i
              JOIN semantier_gateway.account_projections p USING(target_instance_id,environment_namespace,organization_id,account_id,currency)
              WHERE i.target_instance_id=%s AND i.environment_namespace=%s AND i.revision=p.desired_revision
              AND i.status IN ('PENDING','RETRY_WAIT','APPLYING') AND i.available_at<=now()
              AND (i.lease_until IS NULL OR i.lease_until<=now()) AND (p.lease_until IS NULL OR p.lease_until<=now())
              ORDER BY i.available_at,i.event_id LIMIT %s FOR UPDATE OF p,i SKIP LOCKED''',(self.target_instance_id,self.environment_namespace,limit))).fetchall()
            for row in rows:
                token=uuid4().hex
                claimed=await (await c.execute('''UPDATE semantier_gateway.account_sync_intents SET status='APPLYING',attempts=attempts+1,lease_owner=%s,lease_token=%s,lease_until=now()+(%s*interval '1 second')
                  WHERE event_id=%s RETURNING *,now() AS started_at''',(owner,token,lease_seconds,row['event_id']))).fetchone()
                projection=await (await c.execute('''UPDATE semantier_gateway.account_projections SET status='APPLYING',lease_owner=%s,lease_token=%s,lease_until=%s
                  WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s RETURNING applied_revision''',(owner,token,claimed['lease_until'],*self._scope(claimed)))).fetchone()
                await self._audit(c,claimed,{**claimed,**projection},'CLAIMED')
                claims.append(dict(claimed))
        return claims

    async def _fenced(self,c,claim):
        row=await (await c.execute('''SELECT p.*,i.event_id,i.revision AS claimed_revision,i.lease_token AS intent_token,i.lease_until AS intent_lease_until,i.attempts,i.failures,i.cause
          FROM semantier_gateway.account_sync_intents i JOIN semantier_gateway.account_projections p USING(target_instance_id,environment_namespace,organization_id,account_id,currency)
          WHERE i.event_id=%s AND p.target_instance_id=%s AND p.environment_namespace=%s FOR UPDATE OF p,i''',(claim['event_id'],self.target_instance_id,self.environment_namespace))).fetchone()
        now=datetime.now(timezone.utc)
        if not row or row['lease_token']!=claim['lease_token'] or row['intent_token']!=claim['lease_token'] or not row['lease_until'] or row['lease_until']<=now or row['intent_lease_until']<=now:return None
        return row

    async def _supersede(self,c,claim,row):
        await c.execute("UPDATE semantier_gateway.account_sync_intents SET status='SUPERSEDED',completed_at=now(),lease_owner=NULL,lease_token=NULL,lease_until=NULL WHERE event_id=%s",(claim['event_id'],))
        await c.execute('''UPDATE semantier_gateway.account_projections SET lease_owner=NULL,lease_token=NULL,lease_until=NULL
          WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s AND lease_token=%s''',(*self._scope(row),claim['lease_token']))
        await c.execute('''UPDATE semantier_gateway.native_credentials SET status='BLOCK_PENDING' WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s AND policy_revision=%s AND status='PREPARED' ''',(*self._scope(row),claim['revision']))

    async def refresh(self,claim):
        async with await self.connection() as c:
            # Always acquire the financial row before the projection row, matching
            # entitlement mutations and preserving one account lock order.
            await c.execute(_account_query(self.billing_schema),(claim['organization_id'],claim['account_id'],claim['currency']))
            row=await self._fenced(c,claim)
            if row is None:return None
            if row['desired_revision']==claim['revision']:
                await self._write_current(c,row['organization_id'],row['account_id'],row['currency'],'current_authority',billing_schema=self.billing_schema,target=self.target_instance_id,namespace=self.environment_namespace)
                row=await self._fenced(c,claim)
                if row is None:return None
            if row['desired_revision']!=claim['revision']:
                await self._supersede(c,claim,row);return None
            return dict(row)

    async def credential_rows(self,claim):
        async with await self.connection() as c:
            row=await self._fenced(c,claim)
            if row is None:return []
            rows=await (await c.execute('''SELECT user_id,generation,policy_revision,policy_hash,native_key_hash,key_alias,status
              FROM semantier_gateway.native_credentials WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s ORDER BY user_id,generation''',self._scope(row))).fetchall()
            return [dict(r) for r in rows]

    async def prepare_credential(self,claim,user_id):
        from fastapi import HTTPException
        from agents.model_gateway.credential_store import CredentialDependencyError
        import secrets
        if self.keyring is None:raise CredentialDependencyError('CREDENTIAL_KEY_UNAVAILABLE')
        async with await self.connection() as c:
            row=await self._fenced(c,claim)
            if row is None or row['desired_revision']!=claim['revision']:raise HTTPException(503,'PREPARING')
            if not any(m['user_id']==user_id and m['eligible'] for m in row['desired_policy']['members']):raise HTTPException(403,'NO_ACCESS')
            scope=self._scope(row)
            old=await (await c.execute('''SELECT * FROM semantier_gateway.native_credentials WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s AND user_id=%s AND policy_revision=%s AND policy_hash=%s AND status IN ('PREPARED','ACTIVE') ORDER BY generation DESC LIMIT 1''',(*scope,user_id,row['desired_revision'],row['desired_policy_hash']))).fetchone()
            if old:return self._credential(old,row)
            previous=await (await c.execute('''SELECT COALESCE(MAX(generation),0) AS generation FROM semantier_gateway.native_credentials WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s AND user_id=%s''',(*scope,user_id))).fetchone()
            generation=previous['generation']+1;secret='sk-'+secrets.token_urlsafe(32)
            encrypted=self.keyring.encrypt(secret,scope=scope,user_id=user_id,generation=generation,policy_hash=row['desired_policy_hash'])
            native_hash=hashlib.sha256(secret.encode()).hexdigest()
            alias='semantier-'+str(uuid5(NAMESPACE,json.dumps([*scope,user_id,generation],separators=(',',':'))))
            prepared=await (await c.execute('''INSERT INTO semantier_gateway.native_credentials(target_instance_id,environment_namespace,organization_id,account_id,currency,user_id,generation,policy_revision,policy_hash,native_key_hash,key_alias,encryption_key_id,nonce,ciphertext,status)
              VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'PREPARED') RETURNING *''',(*scope,user_id,generation,row['desired_revision'],row['desired_policy_hash'],native_hash,alias,encrypted['encryption_key_id'],encrypted['nonce'],encrypted['ciphertext']))).fetchone()
            result=self._credential(prepared,row)
        # The transaction commits before the caller can submit native create.
        return result

    def _credential(self,credential,projection):
        from agents.model_gateway.credential_store import ScopedCredential
        secret=self.keyring.decrypt(credential)
        return ScopedCredential(secret,*self._scope(projection),credential['user_id'],credential['generation'],projection['desired_revision'],credential['native_key_hash'],projection['native_team_id'],credential['key_alias'],credential['policy_hash'],tuple(projection['desired_policy']['models']))

    async def _audit(self,c,claim,row,outcome,error_code=None,after=None):
        await c.execute('''INSERT INTO semantier_gateway.account_sync_audit(audit_id,event_id,target_instance_id,environment_namespace,organization_id,account_id,currency,revision,attempt,reason,outcome,error_code,before_revision,after_revision,started_at)
          VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',(uuid4().hex,claim['event_id'],*self._scope(row),claim['revision'],row['attempts'],row['cause'],outcome,error_code,row['applied_revision'],after,claim['started_at']))

    async def acknowledge(self,claim,observed):
        async with await self.connection() as c:
            await c.execute(_account_query(self.billing_schema),(claim['organization_id'],claim['account_id'],claim['currency']))
            row=await self._fenced(c,claim)
            if row is None:return False
            policy=await self._policy(c,row['organization_id'],row['account_id'],row['currency'],self.billing_schema)
            if row['desired_revision']!=claim['revision'] or _hash(policy)!=row['desired_policy_hash']:
                await self._write_current(c,row['organization_id'],row['account_id'],row['currency'],'superseded_ack',billing_schema=self.billing_schema,target=self.target_instance_id,namespace=self.environment_namespace)
                await self._supersede(c,claim,row)
                await self._audit(c,claim,row,'SUPERSEDED');return False
            eligible={m['user_id'] for m in policy['members'] if m['eligible']}
            keys=observed.get('keys',{})
            if set(keys)!=eligible or (eligible and observed.get('team_blocked') is not False) or (not policy['account_eligible'] and observed.get('team_blocked') is not True):raise ValueError('Verified native account state required')
            scope=self._scope(row)
            for user_id,native_hash in keys.items():
                credential=await (await c.execute('''SELECT generation FROM semantier_gateway.native_credentials WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s AND user_id=%s AND native_key_hash=%s AND policy_revision=%s AND policy_hash=%s AND status IN ('PREPARED','ACTIVE')''',(*scope,user_id,native_hash,row['desired_revision'],row['desired_policy_hash']))).fetchone()
                if not credential:raise ValueError('Prepared current credential required')
                await c.execute('''UPDATE semantier_gateway.native_credentials SET status='BLOCKED' WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s AND user_id=%s AND generation<>%s AND status IN ('ACTIVE','PREPARED','BLOCK_PENDING')''',(*scope,user_id,credential['generation']))
                await c.execute('''UPDATE semantier_gateway.native_credentials SET status='ACTIVE',verified_at=now() WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s AND user_id=%s AND generation=%s''',(*scope,user_id,credential['generation']))
            await c.execute('''UPDATE semantier_gateway.native_credentials SET status='BLOCKED' WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s AND NOT(user_id=ANY(%s)) AND status IN ('ACTIVE','PREPARED','BLOCK_PENDING')''',(*scope,list(eligible)))
            for member in policy['members']:
                if member.get('active'):
                    await c.execute('''INSERT INTO semantier_gateway.native_user_mappings(target_instance_id,environment_namespace,user_id,native_user_id,verified_at) VALUES(%s,%s,%s,%s,now()) ON CONFLICT(target_instance_id,environment_namespace,user_id) DO UPDATE SET verified_at=now()''',(self.target_instance_id,self.environment_namespace,member['user_id'],member['user_id']))
            await c.execute('''UPDATE semantier_gateway.account_projections SET applied_revision=desired_revision,status='SYNCED',observed_hash=%s,observed_state=%s,last_verified_at=now(),last_error_code=NULL,lease_owner=NULL,lease_token=NULL,lease_until=NULL
              WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s''',(_hash(observed),Jsonb(observed),*scope))
            await c.execute("UPDATE semantier_gateway.account_sync_intents SET status='DONE',failures=0,completed_at=now(),lease_owner=NULL,lease_token=NULL,lease_until=NULL,last_error_code=NULL WHERE event_id=%s",(claim['event_id'],))
            await self._audit(c,claim,row,'SYNCED',after=row['desired_revision'])
        return True

    async def fail(self,claim,code,*,retryable=True,retry_after=None):
        import secrets
        code=code if isinstance(code,str) and re.fullmatch(r'[A-Z][A-Z0-9_]{0,63}',code) else 'SYNC_FAILED'
        async with await self.connection() as c:
            row=await self._fenced(c,claim)
            if row is None:return False
            if row['desired_revision']!=claim['revision']:
                await self._supersede(c,claim,row);await self._audit(c,claim,row,'SUPERSEDED');return False
            failures=row['failures']+1;delays=(5,15,30,60,120,300,600,900)
            if code=='OWNERSHIP_CONFLICT':status='OWNERSHIP_CONFLICT'
            elif code=='BLOCKED_DEPENDENCY':status='BLOCKED_DEPENDENCY'
            elif not retryable or failures>len(delays):status='NEEDS_ATTENTION'
            else:status='RETRY_WAIT'
            delay=delays[min(failures-1,len(delays)-1)]
            if retry_after is not None:
                try:delay=max(delay,min(900,max(0,float(retry_after))))
                except (TypeError,ValueError):pass
            delay=min(900,delay+secrets.randbelow(max(1,int(delay/5)+1)))
            await c.execute('''UPDATE semantier_gateway.account_sync_intents SET status=%s,failures=%s,available_at=now()+(%s*interval '1 second'),last_error_code=%s,lease_owner=NULL,lease_token=NULL,lease_until=NULL WHERE event_id=%s''',(status,failures,delay,code,claim['event_id']))
            await c.execute('''UPDATE semantier_gateway.account_projections SET status=%s,last_error_code=%s,lease_owner=NULL,lease_token=NULL,lease_until=NULL
              WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s''',(status,code,*self._scope(row)))
            await self._audit(c,claim,row,status,code)
        return True

    async def reconcile_due(self,limit=20):
        if not 1<=limit<=20:raise ValueError('Bounded current account reconciliation required')
        async with await self.connection() as c:
            rows=await (await c.execute('''SELECT organization_id,account_id,currency FROM semantier_gateway.account_projections
              WHERE target_instance_id=%s AND environment_namespace=%s AND status NOT IN ('NEEDS_ATTENTION','OWNERSHIP_CONFLICT','BLOCKED_DEPENDENCY')
              AND (last_verified_at IS NULL OR last_verified_at<now()-interval '5 minutes') AND (lease_until IS NULL OR lease_until<=now())
              ORDER BY last_verified_at NULLS FIRST,organization_id,account_id,currency LIMIT %s''',(self.target_instance_id,self.environment_namespace,limit))).fetchall()
            for row in rows:
                await self._write_current(c,row['organization_id'],row['account_id'],row['currency'],'periodic_reconciliation',billing_schema=self.billing_schema,target=self.target_instance_id,namespace=self.environment_namespace)
                await c.execute('''UPDATE semantier_gateway.account_sync_intents i SET status='PENDING',available_at=now(),completed_at=NULL FROM semantier_gateway.account_projections p
                  WHERE i.target_instance_id=p.target_instance_id AND i.environment_namespace=p.environment_namespace AND i.organization_id=p.organization_id AND i.account_id=p.account_id AND i.currency=p.currency AND i.revision=p.desired_revision
                  AND p.target_instance_id=%s AND p.environment_namespace=%s AND p.organization_id=%s AND p.account_id=%s AND p.currency=%s AND i.status='DONE' ''',(self.target_instance_id,self.environment_namespace,row['organization_id'],row['account_id'],row['currency']))
            return len(rows)

    async def selected_credential(self,actor,deployment,current_revision=None):
        from fastapi import HTTPException
        from agents.model_gateway.credential_store import CredentialDependencyError
        if self.keyring is None:raise HTTPException(503,'BLOCKED_DEPENDENCY')
        scope=(self.target_instance_id,self.environment_namespace,actor.organization_id,deployment.account_id,deployment.currency)
        async with await self.connection() as c:
            policy=await self._policy(c,actor.organization_id,deployment.account_id,deployment.currency,self.billing_schema)
            route={'deployment_id':deployment.deployment_id,'revision':deployment.revision,'config_fingerprint':deployment.config_fingerprint}
            if route not in policy['routes']:
                raise HTTPException(409,'deployment revision changed; resolve the current route')
            member=next((m for m in policy['members'] if m['user_id']==actor.user_id),None)
            if not member or not member['active'] or not member['invoke_permitted'] or 'model:invoke' not in actor.permissions or actor.organization_id!=deployment.organization_id:raise HTTPException(403,'NO_ACCESS')
            reason=policy.get('eligibility_reason')
            if reason in {'NO_FUNDING','BUDGET_UNAVAILABLE'}:raise HTTPException(402,reason)
            if reason=='SUSPENDED':raise HTTPException(403,'SUSPENDED')
            if deployment.proxy_model not in policy['models']:raise HTTPException(503,'ROUTE_UNAVAILABLE')
            row=await (await c.execute('''SELECT * FROM semantier_gateway.account_projections WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s''',scope)).fetchone()
            if not row or row['desired_policy_hash']!=_hash(policy) or row['desired_revision']!=row['applied_revision'] or row['status'] not in ('SYNCED','APPLYING') or (current_revision is not None and current_revision!=row['desired_revision']):raise HTTPException(503,'PREPARING')
            if not row.get('observed_state') or row['observed_state'].get('team_blocked') is not False:raise HTTPException(503,'PREPARING')
            credential=await (await c.execute('''SELECT * FROM semantier_gateway.native_credentials WHERE target_instance_id=%s AND environment_namespace=%s AND organization_id=%s AND account_id=%s AND currency=%s AND user_id=%s AND status='ACTIVE' AND policy_revision=%s AND policy_hash=%s''',(*scope,actor.user_id,row['desired_revision'],row['desired_policy_hash']))).fetchone()
            if not credential or row['observed_state'].get('keys',{}).get(actor.user_id)!=credential['native_key_hash']:raise HTTPException(503,'PREPARING')
            user=await (await c.execute('SELECT payload_json FROM semantier_identity.users WHERE user_id=%s',(actor.user_id,))).fetchone()
            if not user or json.loads(user['payload_json']).get('workspace_id')!=actor.workspace_id:raise HTTPException(403,'NO_ACCESS')
            try:return self._credential(credential,row)
            except CredentialDependencyError:raise HTTPException(503,'BLOCKED_DEPENDENCY') from None


async def install_account_relations(c,billing_schema):
    """Explicit migration stage after canonical identity and financial owners."""
    billing_schema=_schema(billing_schema)
    relationships=[
        ('account_scopes','account_scope_organization','(organization_id)','semantier_identity','organizations','(organization_id)'),
        ('account_scopes','account_scope_financial','(organization_id,account_id,currency)',billing_schema,'accounts','(organization_id,account_id,currency)'),
        ('account_access','account_access_user','(user_id)','semantier_identity','users','(user_id)'),
        ('native_user_mappings','native_mapping_user','(user_id)','semantier_identity','users','(user_id)')]
    for table,name,columns,parent_schema,parent,parent_columns in relationships:
        found=await (await c.execute('''SELECT c.confrelid=to_regclass(%s) AS correct FROM pg_catalog.pg_constraint c
          WHERE c.conrelid=to_regclass(%s) AND c.conname=%s''',(parent_schema+'.'+parent,'semantier_gateway.'+table,name))).fetchone()
        if found:
            correct=found['correct'] if hasattr(found,'keys') else found[0]
            if not correct:raise ValueError('Account authority foreign key scope conflict')
            continue
        await c.execute(sql.SQL('ALTER TABLE semantier_gateway.{} ADD CONSTRAINT {} FOREIGN KEY {} REFERENCES {}.{} {}').format(sql.Identifier(table),sql.Identifier(name),sql.SQL(columns),sql.Identifier(parent_schema),sql.Identifier(parent),sql.SQL(parent_columns)))
