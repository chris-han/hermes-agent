"""Durable gateway credentials, explicit grants, route references and actor audit.

Only trusted server provisioning calls grant/bind; HTTP clients cannot create authority.
The existing identity owner remains authoritative for current membership.
"""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import secrets
import uuid

from contracts.model_gateway import Deployment, PERMISSIONS


GATEWAY_SCHEMA_VERSION = 3


class GatewayStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    async def _connection(self):
        import psycopg
        from psycopg.rows import dict_row
        connection=await psycopg.AsyncConnection.connect(self.dsn, row_factory=dict_row)
        await connection.execute("SET TIME ZONE 'UTC'")
        return connection

    async def migrate(self):
        async with await self._connection() as c:
            await c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('semantier-gateway-migrate',))
            await c.execute('CREATE SCHEMA IF NOT EXISTS semantier_gateway')
            await c.execute('CREATE TABLE IF NOT EXISTS semantier_gateway.schema_version (version integer PRIMARY KEY,applied_at timestamptz NOT NULL DEFAULT now())')
            await c.execute('INSERT INTO semantier_gateway.schema_version(version) VALUES(1) ON CONFLICT DO NOTHING')
            await c.execute('''CREATE TABLE IF NOT EXISTS semantier_gateway.grants (
              organization_id text NOT NULL,user_id text NOT NULL,permissions jsonb NOT NULL,
              PRIMARY KEY(organization_id,user_id))''')
            await c.execute('''CREATE TABLE IF NOT EXISTS semantier_gateway.tokens (
              token_id text PRIMARY KEY,token_hash text UNIQUE NOT NULL,user_id text NOT NULL,
              organization_id text NOT NULL,workspace_id text NOT NULL,permissions jsonb NOT NULL,
              expires_at timestamptz NOT NULL,revoked_at timestamptz)''')
            await c.execute('''CREATE TABLE IF NOT EXISTS semantier_gateway.deployments (
              organization_id text NOT NULL,deployment_id text NOT NULL,body jsonb NOT NULL,
              PRIMARY KEY(organization_id,deployment_id))''')
            await c.execute('''CREATE TABLE IF NOT EXISTS semantier_gateway.deployment_versions (
              organization_id text NOT NULL,deployment_id text NOT NULL,revision text NOT NULL,body jsonb NOT NULL,
              PRIMARY KEY(organization_id,deployment_id,revision))''')
            await c.execute('''CREATE TABLE IF NOT EXISTS semantier_gateway.audit (
              event_id text PRIMARY KEY,produced_at timestamptz NOT NULL,organization_id text NOT NULL,
              actor_user_id text NOT NULL,operation text NOT NULL,target text NOT NULL,revision text)''')
            await c.execute('''CREATE TABLE IF NOT EXISTS semantier_gateway.objects (
              organization_id text NOT NULL,kind text NOT NULL,object_id text NOT NULL,
              PRIMARY KEY(kind,object_id))''')
            from pathlib import Path
            await c.execute(Path(__file__).with_name('account_sync.sql').read_text())
            await c.execute(Path(__file__).with_name('device_connections.sql').read_text())

    async def migrate_account_sync(self, billing_schema):
        """Explicit cross-owner links, after identity and billing migrations."""
        from agents.model_gateway.account_sync_store import install_account_relations
        async with await self._connection() as c:
            await c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('semantier-gateway-migrate',))
            await install_account_relations(c,billing_schema)
        return {'gateway_schema_version':GATEWAY_SCHEMA_VERSION,'account_sync_links':'ready'}

    async def grant(self, organization_id, user_id, permissions):
        if not set(permissions) <= PERMISSIONS:
            raise ValueError('unknown permission')
        async with await self._connection() as c:
            await c.execute('''INSERT INTO semantier_gateway.grants VALUES(%s,%s,%s::jsonb)
              ON CONFLICT(organization_id,user_id) DO UPDATE SET permissions=excluded.permissions''',
              (organization_id,user_id,json.dumps(sorted(permissions))))
            from agents.model_gateway.account_sync_store import enqueue_current
            scopes=await (await c.execute('SELECT account_id,currency FROM semantier_gateway.account_access WHERE organization_id=%s AND user_id=%s ORDER BY account_id,currency',(organization_id,user_id))).fetchall()
            for scope in scopes:
                await enqueue_current(c,organization_id,scope['account_id'],scope['currency'],'coarse_grant')

    async def permissions(self, organization_id, user_id):
        async with await self._connection() as c:
            r = await (await c.execute('SELECT permissions FROM semantier_gateway.grants WHERE organization_id=%s AND user_id=%s', (organization_id,user_id))).fetchone()
            return set(r['permissions']) if r else set()

    async def issue(self, *, organization_id, user_id, workspace_id, permissions, expires_at, account_id=None, currency=None):
        if expires_at.tzinfo is None or expires_at <= datetime.now(timezone.utc):
            raise ValueError('future aware expiry required')
        expires_at=expires_at.astimezone(timezone.utc)
        if not set(permissions) <= await self.permissions(organization_id,user_id):
            raise PermissionError('credential cannot expand current grant')
        secret = 'smg_' + secrets.token_urlsafe(32)
        token_id = str(uuid.uuid4())
        async with await self._connection() as c:
            await c.execute('INSERT INTO semantier_gateway.tokens(token_id,token_hash,user_id,organization_id,workspace_id,permissions,expires_at,account_id,currency) VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s)',
              (token_id,hashlib.sha256(secret.encode()).hexdigest(),user_id,organization_id,workspace_id,json.dumps(sorted(permissions)),expires_at,account_id,currency))
        return {'token_id': token_id, 'secret': secret, 'expires_at': expires_at.isoformat()}

    async def token(self, secret):
        async with await self._connection() as c:
            return await (await c.execute('''SELECT token_id,user_id,organization_id,workspace_id,permissions,expires_at,account_id,currency
              FROM semantier_gateway.tokens WHERE token_hash=%s AND revoked_at IS NULL AND expires_at>now()''',
              (hashlib.sha256(secret.encode()).hexdigest(),))).fetchone()

    async def revoke(self, organization_id, user_id, token_id):
        async with await self._connection() as c:
            cursor = await c.execute('UPDATE semantier_gateway.tokens SET revoked_at=now() WHERE token_id=%s AND organization_id=%s AND user_id=%s', (token_id,organization_id,user_id))
            revoked = cursor.rowcount == 1
            if revoked:
                await c.execute("UPDATE semantier_gateway.device_connections SET status='revoked',ciphertext=NULL,nonce=NULL,encryption_key_id=NULL WHERE token_id=%s", (token_id,))
            return revoked

    async def bind(self, deployment: Deployment):
        body = asdict(deployment)
        if body['tokens_per_minute'] is None:
            del body['tokens_per_minute']
        body['max_cost'] = str(deployment.max_cost)
        async with await self._connection() as c:
            previous=await (await c.execute('SELECT body FROM semantier_gateway.deployments WHERE organization_id=%s AND deployment_id=%s FOR UPDATE',(deployment.organization_id,deployment.deployment_id))).fetchone()
            snapshot={key:value for key,value in body.items() if key!='enabled'}
            await c.execute('''INSERT INTO semantier_gateway.deployment_versions VALUES(%s,%s,%s,%s::jsonb)
              ON CONFLICT DO NOTHING''',(deployment.organization_id,deployment.deployment_id,deployment.revision,json.dumps(snapshot)))
            retained=await (await c.execute('''SELECT body FROM semantier_gateway.deployment_versions
              WHERE organization_id=%s AND deployment_id=%s AND revision=%s''',(deployment.organization_id,deployment.deployment_id,deployment.revision))).fetchone()
            if retained['body'] != json.loads(json.dumps(snapshot)):
                raise ValueError('deployment revision is immutable; bind a new revision')
            await c.execute('''INSERT INTO semantier_gateway.deployments VALUES(%s,%s,%s::jsonb)
              ON CONFLICT(organization_id,deployment_id) DO UPDATE SET body=excluded.body''',
              (deployment.organization_id,deployment.deployment_id,json.dumps(body)))
            from agents.model_gateway.account_sync_store import enqueue_current
            scopes={(deployment.account_id,deployment.currency)}
            if previous:scopes.add((previous['body']['account_id'],previous['body']['currency']))
            for account_id,currency in sorted(scopes):
                await enqueue_current(c,deployment.organization_id,account_id,currency,'route_policy')

    async def deployments(self, organization_id):
        async with await self._connection() as c:
            rows = await (await c.execute('SELECT body FROM semantier_gateway.deployments WHERE organization_id=%s ORDER BY deployment_id', (organization_id,))).fetchall()
        out = []
        for row in rows:
            body = row['body']; body['max_cost'] = Decimal(body['max_cost']); body['capabilities'] = tuple(body['capabilities'])
            out.append(Deployment(**body))
        return out

    async def invalidate(self, organization_id, native_model):
        for d in await self.deployments(organization_id):
            if d.native_model_id == native_model or d.proxy_model == native_model or d.deployment_id == native_model:
                body=asdict(d); body['enabled']=False
                await self.bind(Deployment(**body))

    async def register_object(self, organization_id, kind, object_id):
        async with await self._connection() as c:
            await c.execute('INSERT INTO semantier_gateway.objects VALUES(%s,%s,%s) ON CONFLICT DO NOTHING', (organization_id,kind,object_id))

    async def owns(self, organization_id, kind, object_id):
        async with await self._connection() as c:
            return bool(await (await c.execute('SELECT 1 FROM semantier_gateway.objects WHERE organization_id=%s AND kind=%s AND object_id=%s', (organization_id,kind,object_id))).fetchone())

    async def audit(self, organization_id, user_id, operation, target, revision=None):
        async with await self._connection() as c:
            await c.execute('INSERT INTO semantier_gateway.audit VALUES(%s,now(),%s,%s,%s,%s,%s)', (str(uuid.uuid4()),organization_id,user_id,operation,target,revision))

    async def account_rows(self,actor,billing_schema,*,manage=False):
        """Current gateway account access, joined only through its explicit scope owner."""
        from psycopg import sql
        async with await self._connection() as c:
            query=sql.SQL("""SELECT s.organization_id,s.account_id,s.currency,a.funding,x.role FROM semantier_gateway.account_scopes s
                JOIN semantier_gateway.account_access x USING(organization_id,account_id,currency)
                JOIN {}.accounts a USING(organization_id,account_id,currency)
                WHERE s.organization_id=%s AND x.user_id=%s AND x.status='ACTIVE' AND s.billing_schema=%s
                  AND s.funding=a.funding ORDER BY s.account_id,s.currency""").format(sql.Identifier(billing_schema))
            rows=await (await c.execute(query,(actor.organization_id,actor.user_id,billing_schema))).fetchall()
        out=[]
        for r in rows:
            if getattr(actor,'account_id',None) is not None and (r['account_id'],r['currency'])!=(actor.account_id,actor.currency):continue
            owner=r.pop('role') in {'owner','billing_admin'}
            if manage and not owner:continue
            out.append({**r,'can_manage':owner and getattr(actor,'token_id',None) is None and 'billing:manage' in actor.permissions})
        return out
