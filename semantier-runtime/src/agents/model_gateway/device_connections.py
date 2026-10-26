"""Persisted possession-bound device intents, using the existing gateway token owner.

No payment, entitlement or native credential is created by connection approval.
Only scoped Semantier client tokens cross the native channel.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import secrets
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import HTTPException
from contracts.cloud_device import CONTRACT
from contracts.model_gateway import canonical_hash
from agents.model_gateway.auth import current_membership

TTL = 600
INTERVAL = 5
TOKEN_TTL = 2592000


def origin(value):
    p = urlsplit(value)
    if (not p.netloc or p.username or p.password or p.query or p.fragment or p.path not in {'', '/'}
            or not (p.scheme == 'https' or (p.scheme == 'http' and p.hostname in {'localhost', '127.0.0.1'}))):
        raise HTTPException(503, 'approved_origin_unavailable')
    return value.rstrip('/')


class DeviceConnections:
    def __init__(self, store, keyring):
        self.store, self.keyring = store, keyring

    @staticmethod
    def envelope(row, status=None, data=None):
        return dict(contract_version=CONTRACT, intent_ref=row['intent_ref'], status=status or row['status'],
                    expires_at=row['expires_at'].isoformat(), retry_after=INTERVAL, data=data or {})

    async def start(self, command, portal):
        context = command.model_dump()
        digest = canonical_hash(context)
        async with await self.store._connection() as c:
            await c.execute('''INSERT INTO semantier_gateway.device_connections
                (intent_ref,challenge,idempotency_key,request_hash,context,expires_at)
                VALUES(%s,%s,%s,%s,%s::jsonb,now()+interval '600 seconds') ON CONFLICT(challenge,idempotency_key) DO NOTHING''',
                ('dci_'+secrets.token_urlsafe(24), command.possession_challenge, command.idempotency_key, digest, json.dumps(context)))
            row = await (await c.execute('SELECT * FROM semantier_gateway.device_connections WHERE challenge=%s AND idempotency_key=%s FOR UPDATE',
                (command.possession_challenge, command.idempotency_key))).fetchone()
            if row['request_hash'] != digest:
                raise HTTPException(409, 'start_conflict')
            row = await self.expire(c, row)
            return self.envelope(row, data={'verification_uri': portal+'/billing/connect?intent_ref='+row['intent_ref'],
                                           'renewal': 'reauthorization_required'})

    async def locked(self, c, ref, proof=None):
        row = await (await c.execute('SELECT * FROM semantier_gateway.device_connections WHERE intent_ref=%s FOR UPDATE', (ref,))).fetchone()
        if row is None: raise HTTPException(404, 'intent_unavailable')
        if proof is not None and not hmac.compare_digest(hashlib.sha256(proof.encode()).hexdigest(), row['challenge']):
            raise HTTPException(401, 'invalid_device_proof')
        row = await self.expire(c, row)
        if row['token_id'] and row['status'] in {'claimed','installed'}:
            token = await (await c.execute('SELECT revoked_at,expires_at FROM semantier_gateway.tokens WHERE token_id=%s', (row['token_id'],))).fetchone()
            if token['revoked_at'] or token['expires_at'] <= datetime.now(timezone.utc):
                await c.execute("UPDATE semantier_gateway.device_connections SET status='revoked',ciphertext=NULL,nonce=NULL,encryption_key_id=NULL WHERE intent_ref=%s", (ref,))
                row = {**row, 'status':'revoked', 'ciphertext':None}
        return row

    async def expire(self, c, row):
        if row['expires_at'] <= datetime.now(timezone.utc) and row['status'] in {'pending', 'approved', 'claimed'}:
            # Unacknowledged credentials cannot outlive the install/recovery deadline.
            await c.execute('UPDATE semantier_gateway.tokens SET revoked_at=coalesce(revoked_at,now()) WHERE token_id=%s', (row['token_id'],))
            await c.execute("UPDATE semantier_gateway.device_connections SET status='expired',ciphertext=NULL,nonce=NULL,encryption_key_id=NULL WHERE intent_ref=%s", (row['intent_ref'],))
            row = {**row, 'status': 'expired', 'ciphertext': None}
        return row

    @staticmethod
    def bind_actor(row, actor):
        if row['user_id'] is not None and (row['user_id'], row['organization_id'], row['workspace_id']) != (actor.user_id, actor.organization_id, actor.workspace_id):
            raise HTTPException(403, 'intent_actor_mismatch')

    async def browser(self, ref, actor, *, continuation=None, account=None, decision=None):
        async with await self.store._connection() as c:
            row = await self.locked(c, ref)
            self.bind_actor(row, actor)
            if continuation is None and account is None and decision is None:
                return self.envelope(row, data=self.browser_data(row))
            if row['status'] != 'pending':
                if decision and (row['status'] not in ({'approved','claimed','installed'} if decision == 'approve' else {'denied'}) or row['account_id'] != account['account_id']):
                    raise HTTPException(409, 'decision_conflict')
                return self.envelope(row, data=self.browser_data(row))
            fields = dict(user_id=actor.user_id, organization_id=actor.organization_id, workspace_id=actor.workspace_id)
            if continuation is not None:
                if (row['account_id'] == (account or {}).get('account_id')
                        and row['continuation'].get('order_id') and not continuation.get('order_id')):
                    continuation = {**continuation, **{k:row['continuation'][k] for k in ('order_id','product_id','term') if row['continuation'].get(k)}}
                fields['continuation'] = continuation
            if account is not None:
                await self.require_access(c, actor.user_id, actor.organization_id, actor.workspace_id, account['account_id'], account['currency'])
                fields.update(account_id=account['account_id'], currency=account['currency'])
            if decision:
                fields['status'] = 'approved' if decision == 'approve' else 'denied'
            row.update(fields)
            if decision:
                await c.execute("INSERT INTO semantier_gateway.audit VALUES(%s,now(),%s,%s,%s,%s,%s)", (str(uuid4()), actor.organization_id, actor.user_id, 'device.'+decision, ref, CONTRACT))
            await c.execute('''UPDATE semantier_gateway.device_connections SET user_id=%s,organization_id=%s,workspace_id=%s,
                account_id=%s,currency=%s,continuation=%s::jsonb,status=%s WHERE intent_ref=%s''',
                (row['user_id'], row['organization_id'], row['workspace_id'], row['account_id'], row['currency'], json.dumps(row['continuation']), row['status'], ref))
            return self.envelope(row, data=self.browser_data(row))

    @staticmethod
    def browser_data(row):
        return {'client_label': row['context']['client_label'], 'profile_ref': row['context']['profile_ref'],
                'generation': row['context']['generation'], 'continuation': row['continuation'],
                'account_id': row['account_id'], 'currency': row['currency']}

    async def require_access(self, c, uid, org, ws, aid, currency):
        if not current_membership(uid, org, ws): raise HTTPException(403, 'current_membership_required')
        # Locks serialize grant/access changes against issuance. Dispatch/read owners
        # check authority again, so later revocation never relies on token lifetime.
        access = await (await c.execute("SELECT status FROM semantier_gateway.account_access WHERE organization_id=%s AND account_id=%s AND currency=%s AND user_id=%s FOR SHARE", (org, aid, currency, uid))).fetchone()
        grant = await (await c.execute('SELECT permissions FROM semantier_gateway.grants WHERE organization_id=%s AND user_id=%s FOR SHARE', (org, uid))).fetchone()
        if not access or access['status'] != 'ACTIVE' or not grant or 'model:invoke' not in grant['permissions']:
            raise HTTPException(403, 'current_account_access_required')
        return sorted(set(grant['permissions']) & {'model:invoke', 'billing:read', 'usage:read_self'})

    async def poll(self, ref, proof):
        async with await self.store._connection() as c:
            row = await self.locked(c, ref, proof)
            if row['status'] not in {'pending', 'approved', 'claimed', 'installed'}:
                return self.envelope(row)
            if row['token_id']:
                token = await (await c.execute('SELECT revoked_at,expires_at FROM semantier_gateway.tokens WHERE token_id=%s', (row['token_id'],))).fetchone()
                if token['revoked_at'] or token['expires_at'] <= datetime.now(timezone.utc):
                    return self.envelope(row, 'revoked')
            if row['user_id']:
                if row['account_id']:
                    await self.require_access(c, row['user_id'], row['organization_id'], row['workspace_id'], row['account_id'], row['currency'])
            if row['next_poll_at'] and row['next_poll_at'] > datetime.now(timezone.utc):
                return self.envelope(row, 'slow_down')
            await c.execute("UPDATE semantier_gateway.device_connections SET next_poll_at=now()+interval '5 seconds' WHERE intent_ref=%s", (ref,))
            data = {}
            if row['status'] in {'approved','claimed','installed'}:
                data = {key:row['context'][key] for key in ('runtime_ref','profile_ref','generation')}
                data.update(account_id=row['account_id'], currency=row['currency'])
            return self.envelope(row, data=data)

    def encryption_scope(self, row):
        return ('cloud-device', 'v1', row['organization_id'], row['account_id'], row['currency'])

    def recover(self, row):
        return json.loads(self.keyring.decrypt({**row, 'target_instance_id':'cloud-device', 'environment_namespace':'v1',
            'generation':1, 'policy_hash': canonical_hash([row['intent_ref'], row['claim_id']])}))

    async def claim(self, ref, command, inference_origin, billing_origin, *, ack=False):
        async with await self.store._connection() as c:
            row = await self.locked(c, ref, command.proof)
            context = row['context']
            if any(getattr(command, k) != context[k] for k in ('runtime_ref','profile_ref','generation')):
                raise HTTPException(409, 'profile_generation_mismatch')
            if row['status'] in {'pending', 'denied', 'expired', 'cancelled', 'revoked'}:
                return self.envelope(row)
            if command.account_id != row['account_id']: raise HTTPException(409, 'account_mismatch')
            if row['claim_id'] and row['claim_id'] != command.claim_id: raise HTTPException(409, 'claim_conflict')
            permissions = await self.require_access(c, row['user_id'], row['organization_id'], row['workspace_id'], row['account_id'], row['currency'])
            if row['token_id']:
                token = await (await c.execute('SELECT * FROM semantier_gateway.tokens WHERE token_id=%s FOR UPDATE', (row['token_id'],))).fetchone()
                if token['revoked_at'] or token['expires_at'] <= datetime.now(timezone.utc):
                    return self.envelope(row, 'revoked')
            if ack:
                if row['status'] not in {'claimed', 'installed'}: raise HTTPException(409, 'claim_required')
                if command.installed_generation != context['generation']: raise HTTPException(409, 'installed_generation_mismatch')
                await c.execute('UPDATE semantier_gateway.tokens SET expires_at=%s WHERE token_id=%s', (row['credential_expires_at'], row['token_id']))
                await c.execute("UPDATE semantier_gateway.device_connections SET status='installed',installed_generation=%s,ciphertext=NULL,nonce=NULL,encryption_key_id=NULL WHERE intent_ref=%s", (command.installed_generation, ref))
                return self.envelope(row, 'installed', {'claim_id':row['claim_id'], 'installed_generation':command.installed_generation})
            if row['status'] == 'installed': return self.envelope(row, data={'claim_id':row['claim_id'], 'installed_generation':row['installed_generation']})
            if self.keyring is None: raise HTTPException(503, 'credential_custody_unavailable')
            if row['status'] == 'claimed':
                from agents.model_gateway.credential_store import CredentialDependencyError
                try: return self.envelope(row, data=self.recover(row))
                except CredentialDependencyError: raise HTTPException(503, 'credential_custody_unavailable') from None
            secret = 'smg_'+secrets.token_urlsafe(32)
            token_id = str(uuid4())
            expiry = datetime.now(timezone.utc)+timedelta(seconds=TOKEN_TTL)
            payload = dict(credential={'token_id':token_id, 'secret':secret, 'expires_at':expiry.isoformat()},
                account_id=row['account_id'], currency=row['currency'], organization_id=row['organization_id'],
                inference_origin=inference_origin, billing_origin=billing_origin,
                catalog_path='/api/cloud-provider/catalog', readiness_path='/api/cloud-provider/device/readiness',
                runtime_ref=context['runtime_ref'], profile_ref=context['profile_ref'], generation=context['generation'],
                claim_id=command.claim_id, installation_expires_at=row['expires_at'].isoformat(), renewal='reauthorization_required')
            encrypted = self.keyring.encrypt(json.dumps(payload), scope=self.encryption_scope(row), user_id=row['user_id'],
                generation=1, policy_hash=canonical_hash([ref, command.claim_id]))
            await c.execute('''INSERT INTO semantier_gateway.tokens(token_id,token_hash,user_id,organization_id,workspace_id,permissions,expires_at,account_id,currency)
                VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s)''', (token_id, hashlib.sha256(secret.encode()).hexdigest(), row['user_id'], row['organization_id'], row['workspace_id'], json.dumps(permissions), row['expires_at'], row['account_id'], row['currency']))
            await c.execute("UPDATE semantier_gateway.device_connections SET status='claimed',claim_id=%s,token_id=%s,credential_expires_at=%s,encryption_key_id=%s,nonce=%s,ciphertext=%s WHERE intent_ref=%s",
                (command.claim_id, token_id, expiry, encrypted['encryption_key_id'], encrypted['nonce'], encrypted['ciphertext'], ref))
            await c.execute("INSERT INTO semantier_gateway.audit VALUES(%s,now(),%s,%s,'device.claim',%s,%s)", (str(uuid4()), row['organization_id'], row['user_id'], ref, CONTRACT))
            return self.envelope(row, 'claimed', payload)

    async def cancel(self, ref, proof):
        async with await self.store._connection() as c:
            row = await self.locked(c, ref, proof)
            if row['status'] in {'denied','expired','revoked'}: return self.envelope(row)
            await c.execute('UPDATE semantier_gateway.tokens SET revoked_at=coalesce(revoked_at,now()) WHERE token_id=%s', (row['token_id'],))
            await c.execute("UPDATE semantier_gateway.device_connections SET status='cancelled',ciphertext=NULL,nonce=NULL,encryption_key_id=NULL WHERE intent_ref=%s", (ref,))
            return self.envelope(row, 'cancelled')
