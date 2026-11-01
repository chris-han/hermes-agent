"""Scoped price-source custody using the existing Billing PostgreSQL connection owner."""
from uuid import uuid4
from psycopg.types.json import Jsonb
from billing.store import _wire
from billing.supplier_pricing_projection import digest
from contracts.billing import BillingError


class SupplierPricingStore:
    def __init__(self, billing):
        self.billing = billing

    async def operator(self, user_id):
        async with self.billing.connection() as c:
            row=await (await c.execute('SELECT enabled FROM supplier_pricing_operators WHERE user_id=%s',(user_id,))).fetchone()
            return bool(row and row['enabled'])

    async def begin(self, org, key, models, *, provider_id='alibaba', source_ref='alibaba-model-studio-cn-beijing', partition='cn-beijing'):
        request={'models':sorted(models),'partition':partition}
        if (provider_id, source_ref, partition) != ('alibaba','alibaba-model-studio-cn-beijing','cn-beijing'):
            request.update(provider_id=provider_id,source_ref=source_ref)
        request_hash=digest(request)
        async with self.billing.connection() as c:
            await c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('supplier-pricing:'+org,))
            await c.execute("UPDATE supplier_pricing_runs SET state='INTERRUPTED',reason='CAPTURE_LEASE_EXPIRED',updated_at=now() WHERE organization_id=%s AND state='COLLECTING' AND lease_until<now()",(org,))
            old=await (await c.execute('SELECT * FROM supplier_pricing_runs WHERE organization_id=%s AND idempotency_key=%s',(org,key))).fetchone()
            if old:
                if old['request_hash']!=request_hash:raise BillingError('idempotency_conflict')
                return _wire(old),False
            running=await (await c.execute("SELECT run_id FROM supplier_pricing_runs WHERE organization_id=%s AND provider_id=%s AND source_ref=%s AND partition=%s AND state='COLLECTING'",(org,provider_id,source_ref,partition))).fetchone()
            if running:raise BillingError('collection_in_progress')
            row=await (await c.execute("INSERT INTO supplier_pricing_runs(organization_id,run_id,idempotency_key,request_hash,partition,provider_id,source_ref,state,lease_until) VALUES(%s,%s,%s,%s,%s,%s,%s,'COLLECTING',now()+interval '150 seconds') RETURNING *",(org,uuid4().hex,key,request_hash,partition,provider_id,source_ref))).fetchone()
            return _wire(row),True

    async def finish(self, org, run_id, snapshot=None, *, failure=None):
        async with self.billing.connection() as c:
            run=await self.billing._one(c,'SELECT * FROM supplier_pricing_runs WHERE organization_id=%s AND run_id=%s FOR UPDATE',(org,run_id))
            if run['state']!='COLLECTING':return _wire(run)
            version=snapshot['version_id'] if snapshot else None
            state='FAILED'
            if snapshot:
                from billing.supplier_catalog import _safe
                _safe(snapshot)
                pin=dict(snapshot);pin.pop('version_id')
                if version!='sha256:'+digest(pin):raise ValueError('SNAPSHOT_HASH_MISMATCH')
                payload=Jsonb(snapshot)
                await c.execute("INSERT INTO prices(version_id,kind,payload) VALUES(%s,'supplier',%s) ON CONFLICT DO NOTHING",('supplier-pricing-source:'+version,payload))
                prior=await self.billing._one(c,"SELECT payload FROM prices WHERE version_id=%s AND kind='supplier'",('supplier-pricing-source:'+version,))
                if prior['payload']!=snapshot:raise BillingError('immutable_price')
                state='COLLECTED' if all(snapshot[k]['status']=='COMPLETE' for k in ('api','document')) else 'PARTIAL'
            row=await (await c.execute('UPDATE supplier_pricing_runs SET state=%s,snapshot_version=%s,reason=%s,updated_at=now() WHERE organization_id=%s AND run_id=%s RETURNING *',(state,version,failure,org,run_id))).fetchone()
            return _wire(row)

    async def runs(self, org, *, limit=50, offset=0, provider_id=None, source_ref=None):
        if type(limit) is not int or type(offset) is not int or not 1<=limit<=50 or not 0<=offset<=100000:
            raise ValueError('INVALID_PAGE')
        if (provider_id is None) != (source_ref is None):
            raise ValueError('INVALID_SOURCE_SCOPE')
        async with self.billing.connection() as c:
            await c.execute("UPDATE supplier_pricing_runs SET state='INTERRUPTED',reason='CAPTURE_LEASE_EXPIRED',updated_at=now() WHERE organization_id=%s AND state='COLLECTING' AND lease_until<now()",(org,))
            if provider_id is None:
                query='SELECT * FROM supplier_pricing_runs WHERE organization_id=%s ORDER BY created_at DESC,run_id LIMIT %s OFFSET %s'
                params=(org,limit,offset)
            else:
                query='SELECT * FROM supplier_pricing_runs WHERE organization_id=%s AND provider_id=%s AND source_ref=%s ORDER BY created_at DESC,run_id LIMIT %s OFFSET %s'
                params=(org,provider_id,source_ref,limit,offset)
            rows=await (await c.execute(query,params)).fetchall()
            return _wire(rows)

    async def latest_snapshot(self, org, provider_id, source_ref):
        async with self.billing.connection() as c:
            row=await (await c.execute('SELECT snapshot_version FROM supplier_pricing_runs WHERE organization_id=%s AND provider_id=%s AND source_ref=%s AND snapshot_version IS NOT NULL ORDER BY created_at DESC,run_id LIMIT 1',(org,provider_id,source_ref))).fetchone()
            return row['snapshot_version'] if row else None

    async def run(self, org, run_id):
        await self.runs(org)
        async with self.billing.connection() as c:
            return _wire(await self.billing._one(c,'SELECT * FROM supplier_pricing_runs WHERE organization_id=%s AND run_id=%s',(org,run_id)))

    async def snapshot(self, org, version):
        async with self.billing.connection() as c:
            row=await self.billing._one(c,"SELECT p.payload FROM prices p JOIN supplier_pricing_runs r ON p.version_id='supplier-pricing-source:'||r.snapshot_version WHERE r.organization_id=%s AND r.snapshot_version=%s AND p.kind='supplier' LIMIT 1",(org,version))
            snapshot=row['payload']
            pin=dict(snapshot);pin.pop('version_id',None)
            if snapshot.get('version_id')!='sha256:'+digest(pin):raise ValueError('SNAPSHOT_HASH_MISMATCH')
            return snapshot

    async def base_map(self, base_hash):
        async with self.billing.connection() as c:
            row=await self.billing._one(c,"SELECT payload FROM prices WHERE version_id=%s AND kind='supplier'",('supplier-pricing-base:'+base_hash,))
            if digest(row['payload']) != base_hash:
                raise ValueError('BASE_HASH_MISMATCH')
            return row['payload']

    async def put_projection(self, org, run_id, projection, *, base_map=None):
        version='sha256:'+digest(projection)
        async with self.billing.connection() as c:
            await self.billing._one(c,'SELECT run_id FROM supplier_pricing_runs WHERE organization_id=%s AND run_id=%s',(org,run_id))
            if base_map is not None:
                if digest(base_map)!=projection['base_sha256']:raise ValueError('BASE_HASH_MISMATCH')
                key='supplier-pricing-base:'+projection['base_sha256']
                await c.execute("INSERT INTO prices(version_id,kind,payload) VALUES(%s,'supplier',%s) ON CONFLICT DO NOTHING",(key,Jsonb(base_map)))
                old=await self.billing._one(c,"SELECT payload FROM prices WHERE version_id=%s AND kind='supplier'",(key,))
                if old['payload']!=base_map:raise BillingError('immutable_price')
            await c.execute('INSERT INTO supplier_pricing_projections(organization_id,version_id,run_id,payload) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING',(org,version,run_id,Jsonb(projection)))
        return version

    async def select(self, org, version):
        async with self.billing.connection() as c:
            await c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('supplier-pricing-global-pointer',))
            await self.billing._one(c,'SELECT version_id FROM supplier_pricing_projections WHERE organization_id=%s AND version_id=%s',(org,version))
            old=await (await c.execute('SELECT * FROM supplier_pricing_active WHERE singleton FOR UPDATE')).fetchone()
            await c.execute("INSERT INTO supplier_pricing_active(singleton,organization_id,projection_version,state) VALUES(true,%s,%s,'SELECTED') ON CONFLICT(singleton) DO UPDATE SET previous_version=supplier_pricing_active.projection_version,organization_id=excluded.organization_id,projection_version=excluded.projection_version,state='SELECTED',updated_at=now()",(org,version))
            return _wire(old) if old else None

    async def active(self):
        async with self.billing.connection() as c:
            row=await (await c.execute('SELECT a.*,p.payload FROM supplier_pricing_active a JOIN supplier_pricing_projections p ON p.organization_id=a.organization_id AND p.version_id=a.projection_version WHERE a.singleton')).fetchone()
            return _wire(row) if row else None

    async def effective(self, version, revision, success, *, unknown=False):
        async with self.billing.connection() as c:
            state = 'EFFECTIVE_ONE_WORKER' if success else 'OBSERVATION_UNKNOWN' if unknown else 'SYNC_FAILED'
            await c.execute("UPDATE supplier_pricing_active SET state=%s,effective_revision=CASE WHEN %s THEN %s ELSE effective_revision END,updated_at=now() WHERE singleton AND projection_version=%s",(state,success,revision,version))
