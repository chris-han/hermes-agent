"""PostgreSQL owner for admission, retained metering and service-credit settlement.

Callers must supply identity resolved by the existing authority owner. Raw Proxy
capture can append evidence only to an existing attempt; it cannot select a payer
or post money. Recovery deliberately has no model transport dependency.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from functools import wraps
from pathlib import Path
import re
import json
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from billing.native_evidence import final_native_observation
from billing.rating import FxQuote, PriceVersion, normalize_usage, rate_usage, standard_usd_rates, effective_usd_per_million, resolve_price
from contracts.billing import BillingError, currency_code, decimal_amount


def _wire(value):
    if isinstance(value, Decimal): return format(value,'f')
    if isinstance(value, datetime): return value.astimezone(timezone.utc).isoformat().replace('+00:00','Z')
    if isinstance(value, dict): return {k:_wire(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_wire(v) for v in value]
    return value


def _exact_money(method):
    @wraps(method)
    async def invoke(*args, **kwargs):
        # NUMERIC(38,12) arithmetic must not inherit Decimal's 28-digit default.
        with localcontext() as context:
            context.prec = 50
            return await method(*args, **kwargs)
    return invoke


BILLING_SCHEMA_VERSION = 7


class BillingStore:
    def __init__(self, dsn: str, schema: str = 'semantier_billing'):
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,62}',schema): raise ValueError('invalid billing schema')
        self.dsn,self.schema=dsn,schema

    @asynccontextmanager
    async def connection(self):
        async with await psycopg.AsyncConnection.connect(self.dsn,row_factory=dict_row) as conn:
            await conn.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(sql.Identifier(self.schema)))
            await conn.execute("SET LOCAL TIME ZONE 'UTC'")
            yield conn

    async def migrate(self):
        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            await conn.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('billing-migrate:'+self.schema,))
            await conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(self.schema)))
            await conn.execute(sql.SQL('SET LOCAL search_path TO {}, pg_catalog').format(sql.Identifier(self.schema)))
            await conn.execute(Path(__file__).with_name('schema.sql').read_text())
            for table in ('raw_observations','usage_observations','prices','fx_quotes','ledger_transactions','ledger_entries','reconciliation_items','organization_customer_prices','financial_policy_events','commerce_quotes','commerce_payment_events','commerce_subscriptions','commerce_enterprise_offers'):
                await conn.execute(sql.SQL('DROP TRIGGER IF EXISTS immutable ON {}').format(sql.Identifier(table)))
                await conn.execute(sql.SQL('CREATE TRIGGER immutable BEFORE UPDATE OR DELETE ON {} FOR EACH ROW EXECUTE FUNCTION reject_financial_mutation()').format(sql.Identifier(table)))
            await conn.execute('DROP TRIGGER IF EXISTS balanced ON ledger_entries')
            await conn.execute('CREATE CONSTRAINT TRIGGER balanced AFTER INSERT ON ledger_entries DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION check_transaction_balance()')
        return {'schema':self.schema,'version':BILLING_SCHEMA_VERSION}

    async def _one(self,conn,query,params=(),*,required=True):
        row=await (await conn.execute(query,params)).fetchone()
        if row is None and required: raise BillingError('not_found')
        return row

    async def _invocation(self,conn,organization_id,invocation_id,*,lock=False):
        return await self._one(conn,'SELECT i.*,a.attempt_id FROM invocations i JOIN attempts a USING(invocation_id) WHERE i.organization_id=%s AND i.invocation_id=%s'+(' FOR UPDATE OF i' if lock else ''),(organization_id,invocation_id))

    async def _account(self,conn,org,account,currency):
        return await self._one(conn,'SELECT * FROM accounts WHERE organization_id=%s AND account_id=%s AND currency=%s FOR UPDATE',(org,account,currency))

    @_exact_money
    async def admit(self,organization_id,account_id,operation,idempotency_key,request_hash,deployment_id,funding,currency,max_cost,attribution=None):
        currency_code(currency)
        cap=decimal_amount(max_cost)
        if funding not in ('CUSTOMER_CLOUD','SEMANTIER_CREDITS'): raise BillingError('funding_unavailable')
        if not all(isinstance(v,str) and 0<len(v)<=512 for v in (organization_id,account_id,operation,idempotency_key,request_hash,deployment_id)):
            raise ValueError('nonempty bounded invocation references required')
        attribution=attribution or {}
        if currency=='CNY':
            from billing.commerce import configured
            commerce=configured(self)
            if commerce.development: await commerce.cycles(organization_id,account_id)
        async with self.connection() as conn:
            await conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(json.dumps([organization_id,account_id,operation,idempotency_key]),))
            old=await self._one(conn,'SELECT i.*,a.attempt_id FROM invocations i JOIN attempts a USING(invocation_id) WHERE organization_id=%s AND account_id=%s AND operation=%s AND idempotency_key=%s',(organization_id,account_id,operation,idempotency_key),required=False)
            if old:
                if old['request_hash']!=request_hash: raise BillingError('idempotency_conflict')
                return _wire(old)|{'created':False}
            a=await self._account(conn,organization_id,account_id,currency)
            await self._expire_grants(conn,organization_id,account_id,currency)
            a=await self._account(conn,organization_id,account_id,currency)
            if a['funding']!=funding: raise BillingError('funding_mismatch')
            if a['blocked']: raise BillingError('reconciliation_pending')
            if a['budget_spent']+a['held']+cap>a['budget_limit']: raise BillingError('budget_exceeded')
            if funding=='SEMANTIER_CREDITS' and a['wallet']-a['held']<cap: raise BillingError('credits_exhausted')
            task_id=attribution.get('parent_run_id') or attribution.get('run_id')
            task=None
            if task_id:
                task=await self._one(conn,'SELECT * FROM task_budgets WHERE organization_id=%s AND account_id=%s AND currency=%s AND task_id=%s FOR UPDATE',(organization_id,account_id,currency,task_id),required=False)
                if task and task['spent']+task['held']+cap>task['budget_limit']: raise BillingError('budget_exceeded','task budget_exceeded')
            invocation_id,attempt_id=uuid4().hex,uuid4().hex
            await conn.execute('UPDATE accounts SET held=held+%s WHERE organization_id=%s AND account_id=%s AND currency=%s',(cap,organization_id,account_id,currency))
            if task:
                await conn.execute('UPDATE task_budgets SET held=held+%s WHERE organization_id=%s AND account_id=%s AND currency=%s AND task_id=%s',(cap,organization_id,account_id,currency,task_id))
            await conn.execute("INSERT INTO invocations(invocation_id,organization_id,account_id,currency,operation,idempotency_key,request_hash,deployment_id,funding,max_cost,attribution,task_id,execution_state,settlement_state) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'ADMITTED',%s)",(invocation_id,organization_id,account_id,currency,operation,idempotency_key,request_hash,deployment_id,funding,cap,Jsonb(attribution),task_id if task else None,'HELD' if funding=='SEMANTIER_CREDITS' else 'NOT_APPLICABLE'))
            await self._allocate_grants(conn,organization_id,account_id,currency,invocation_id,cap)
            await conn.execute('INSERT INTO attempts(attempt_id,invocation_id) VALUES(%s,%s)',(attempt_id,invocation_id))
            return _wire(await self._invocation(conn,organization_id,invocation_id))|{'created':True}

    async def mark_dispatching(self,organization_id,invocation_id):
        async with self.connection() as conn:
            i=await self._invocation(conn,organization_id,invocation_id,lock=True)
            if i['execution_state']!='ADMITTED': raise BillingError('invalid_dispatch_state')
            await conn.execute("UPDATE invocations SET execution_state='DISPATCHING',updated_at=now() WHERE invocation_id=%s",(invocation_id,))
            await conn.execute("UPDATE attempts SET outbound_intent='POSSIBLY_SENT',dispatched_at=now() WHERE invocation_id=%s",(invocation_id,))

    async def mark_uncertain(self,organization_id,invocation_id):
        async with self.connection() as conn:
            i=await self._invocation(conn,organization_id,invocation_id,lock=True)
            if i['hold_active']:
                await conn.execute("UPDATE invocations SET execution_state='POSSIBLY_EXECUTED',settlement_state='PENDING_RECONCILIATION',updated_at=now() WHERE invocation_id=%s",(invocation_id,))

    async def fail_before_send(self,organization_id,invocation_id):
        # Lock account first, same order as admission and settlement.
        async with self.connection() as conn:
            initial=await self._invocation(conn,organization_id,invocation_id)
            await self._account(conn,organization_id,initial['account_id'],initial['currency'])
            i=await self._invocation(conn,organization_id,invocation_id,lock=True)
            if i['execution_state']=='FAILED_BEFORE_SEND': return
            attempts=await (await conn.execute('SELECT attempt_id,native_send_claimed_at FROM attempts WHERE invocation_id=%s FOR UPDATE',(invocation_id,))).fetchall()
            observed=await self._one(conn,'SELECT EXISTS(SELECT 1 FROM raw_observations r JOIN attempts a USING(attempt_id) WHERE a.invocation_id=%s) AS observed',(invocation_id,))
            if i['execution_state'] not in {'ADMITTED','DISPATCHING'} or not attempts or any(a['native_send_claimed_at'] is not None for a in attempts) or observed['observed']:
                raise BillingError('possibly_executed','cannot release possibly-executed hold')
            await self._close_hold(conn,i,Decimal(0))
            await conn.execute("UPDATE invocations SET execution_state='FAILED_BEFORE_SEND',settlement_state='NOT_APPLICABLE',updated_at=now() WHERE invocation_id=%s",(invocation_id,))
            await conn.execute("UPDATE attempts SET outbound_intent='DEFINITELY_UNSENT' WHERE invocation_id=%s",(invocation_id,))

    async def _close_hold(self,conn,i,budget_cost):
        if not i['hold_active']: return
        await self._release_grants(conn,i,budget_cost)
        await conn.execute('UPDATE accounts SET held=held-%s,budget_spent=budget_spent+%s WHERE organization_id=%s AND account_id=%s AND currency=%s',(i['max_cost'],budget_cost,i['organization_id'],i['account_id'],i['currency']))
        if i['task_id']:
            await conn.execute('UPDATE task_budgets SET held=held-%s,spent=spent+%s WHERE organization_id=%s AND account_id=%s AND currency=%s AND task_id=%s',(i['max_cost'],budget_cost,i['organization_id'],i['account_id'],i['currency'],i['task_id']))
        await conn.execute('UPDATE invocations SET hold_active=false WHERE invocation_id=%s',(i['invocation_id'],))

    async def _enqueue_account_sync(self,conn,org,aid,currency,cause):
        from agents.model_gateway.account_sync_store import enqueue_current
        return await enqueue_current(conn,org,aid,currency,cause,billing_schema=self.schema)

    async def _expire_grants(self,conn,org,aid,currency,at=None,*,notify=True):
        at=at or datetime.now(timezone.utc)
        rows=await (await conn.execute('SELECT * FROM credit_grants WHERE organization_id=%s AND account_id=%s AND currency=%s AND NOT expired AND expires_at<=%s FOR UPDATE',(org,aid,currency,at))).fetchall()
        for g in rows:
            amount=g['remaining']-g['held']
            await self._expire_posting(conn,g,amount,':expiry')
            await conn.execute('UPDATE credit_grants SET remaining=held,expired=true WHERE grant_id=%s',(g['grant_id'],))
        if rows and notify:
            await self._enqueue_account_sync(conn,org,aid,currency,'allowance_expiry')
        return bool(rows)

    async def _expire_posting(self,conn,g,amount,suffix):
        if not amount:return
        tx=g['grant_id']+suffix
        await conn.execute("INSERT INTO ledger_transactions(id,organization_id,account_id,currency,kind) VALUES(%s,%s,%s,%s,'allowance_expiry')",(tx,g['organization_id'],g['account_id'],g['currency']))
        await conn.execute("INSERT INTO ledger_entries(transaction_id,book_account,amount) VALUES(%s,'wallet',%s),(%s,'issuance',%s)",(tx,-amount,tx,amount))
        await conn.execute('UPDATE accounts SET wallet=wallet-%s WHERE organization_id=%s AND account_id=%s AND currency=%s',(amount,g['organization_id'],g['account_id'],g['currency']))

    async def _allocate_grants(self,conn,org,aid,currency,iid,cap):
        rows=await (await conn.execute('SELECT * FROM credit_grants WHERE organization_id=%s AND account_id=%s AND currency=%s AND NOT expired ORDER BY expires_at NULLS LAST,grant_id FOR UPDATE',(org,aid,currency))).fetchall()
        left=cap
        for g in rows:
            amount=min(left,g['remaining']-g['held'])
            if amount:
                await conn.execute('INSERT INTO credit_hold_grants(invocation_id,grant_id,amount) VALUES(%s,%s,%s)',(iid,g['grant_id'],amount))
                await conn.execute('UPDATE credit_grants SET held=held+%s WHERE grant_id=%s',(amount,g['grant_id']))
                left-=amount

    async def _release_grants(self,conn,i,cost):
        rows=await (await conn.execute('SELECT g.*,h.amount AS reserved FROM credit_hold_grants h JOIN credit_grants g USING(grant_id) WHERE invocation_id=%s ORDER BY expires_at NULLS LAST,grant_id FOR UPDATE OF g',(i['invocation_id'],))).fetchall()
        left=cost
        for g in rows:
            debit=min(left,g['reserved']);left-=debit
            release=g['reserved']-debit
            await conn.execute('UPDATE credit_grants SET held=held-%s,remaining=remaining-%s WHERE grant_id=%s',(g['reserved'],g['reserved'] if g['expired'] else debit,g['grant_id']))
            if g['expired']: await self._expire_posting(conn,g,release,':release:'+i['invocation_id'])

    async def append_raw_observation(self,attempt_id,event_id,provider,raw_usage,provenance,physical_send_count,completion_state):
        if isinstance(physical_send_count,bool) or not isinstance(physical_send_count,int) or physical_send_count<0:
            raise ValueError('invalid physical send count')
        if completion_state not in ('FINAL','PARTIAL','MISSING') or not event_id or not provider or not provenance:
            raise ValueError('native provenance and completion state required')
        async with self.connection() as conn:
            await self._one(conn,'SELECT attempt_id FROM attempts WHERE attempt_id=%s',(attempt_id,))
            values=(attempt_id,event_id,provider,Jsonb(raw_usage),Jsonb(provenance),physical_send_count,completion_state)
            row=await self._one(conn,'INSERT INTO raw_observations(attempt_id,event_id,provider,raw_usage,provenance,physical_send_count,completion_state) VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING *',values,required=False)
            if row is None:
                row=await self._one(conn,'SELECT * FROM raw_observations WHERE attempt_id=%s AND event_id=%s',(attempt_id,event_id))
                if (row['provider'],row['raw_usage'],row['provenance'],row['physical_send_count'],row['completion_state'])!=(provider,raw_usage,provenance,physical_send_count,completion_state):
                    raise BillingError('observation_conflict')
            return _wire(row)

    async def read_raw_observations(self,organization_id,attempt_id):
        async with self.connection() as conn:
            await self._one(conn,'SELECT a.attempt_id FROM attempts a JOIN invocations i USING(invocation_id) WHERE i.organization_id=%s AND a.attempt_id=%s',(organization_id,attempt_id))
            return _wire(await (await conn.execute('SELECT * FROM raw_observations WHERE attempt_id=%s ORDER BY id',(attempt_id,))).fetchall())

    async def retain_usage(self,organization_id,attempt_id,event_id,raw_usage,normalized,rated_customer_cost,supplier_cost,price_version,fx_version=None,*,supplier_price_version=None,supplier_native_cost=None,supplier_currency=None):
        customer=decimal_amount(rated_customer_cost) if rated_customer_cost is not None else None
        supplier=decimal_amount(supplier_cost) if supplier_cost is not None else None
        native=decimal_amount(supplier_native_cost) if supplier_native_cost is not None else None
        if supplier_currency is not None: currency_code(supplier_currency)
        if customer is not None and (not price_version or normalized.get('completeness') not in ('PROVIDER_REPORTED','RECONCILED')):
            raise BillingError('price_unavailable')
        async with self.connection() as conn:
            initial=await self._one(conn,'SELECT i.* FROM invocations i JOIN attempts a USING(invocation_id) WHERE i.organization_id=%s AND a.attempt_id=%s',(organization_id,attempt_id))
            await self._account(conn,organization_id,initial['account_id'],initial['currency'])
            i=await self._invocation(conn,organization_id,initial['invocation_id'],lock=True)
            old=await self._one(conn,'INSERT INTO usage_observations(attempt_id,event_id,raw_usage,normalized,customer_cost,supplier_cost,price_version,fx_version,supplier_price_version,supplier_native_cost,supplier_currency) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING *',(attempt_id,event_id,Jsonb(raw_usage),Jsonb(normalized),customer,supplier,price_version,fx_version,supplier_price_version,native,supplier_currency),required=False)
            if old is None:
                old=await self._one(conn,'SELECT * FROM usage_observations WHERE attempt_id=%s AND event_id=%s',(attempt_id,event_id))
                if (old['raw_usage'],old['normalized'],old['customer_cost'],old['supplier_cost'],old['price_version'],old['fx_version'],old['supplier_price_version'],old['supplier_native_cost'],old['supplier_currency'])!=(raw_usage,normalized,customer,supplier,price_version,fx_version,supplier_price_version,native,supplier_currency):
                    raise BillingError('observation_conflict')
                return _wire(old)
            if i['hold_active']:
                # An out-of-order partial callback cannot erase a final observation.
                state=normalized.get('completeness','PARTIAL')
                if i['usage_state'] in ('PROVIDER_REPORTED','RECONCILED') and state not in ('PROVIDER_REPORTED','RECONCILED'):
                    state=i['usage_state']
                await conn.execute('UPDATE invocations SET usage_state=%s,pricing_state=%s,updated_at=now() WHERE invocation_id=%s',(state,'RATED' if customer is not None or i['pricing_state']=='RATED' else 'UNPRICED',i['invocation_id']))
            elif customer is not None and i['settlement_id']:
                previous=await self._one(conn,"SELECT * FROM usage_observations WHERE attempt_id=%s AND id<%s AND customer_cost IS NOT NULL ORDER BY (coalesce((normalized->>'input_total')::bigint,0)+coalesce((normalized->>'output_total')::bigint,0)) DESC,id DESC LIMIT 1",(attempt_id,old['id']),required=False)
                if previous and (previous['customer_cost']!=customer or previous['supplier_cost']!=supplier):
                    await conn.execute('INSERT INTO reconciliation_items(invocation_id,usage_observation_id,reason) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING',(i['invocation_id'],old['id'],'late_cost_conflict'))
                    await conn.execute("UPDATE invocations SET settlement_state='PENDING_RECONCILIATION',updated_at=now() WHERE invocation_id=%s",(i['invocation_id'],))
                    await conn.execute('UPDATE accounts SET blocked=true WHERE organization_id=%s AND account_id=%s AND currency=%s',(organization_id,i['account_id'],i['currency']))
                    await self._enqueue_account_sync(conn,organization_id,i['account_id'],i['currency'],'late_usage_reconciliation')
            return _wire(old)

    async def put_price(self,price:PriceVersion,kind='customer'):
        async with self.connection() as conn:
            old=await self._one(conn,'INSERT INTO prices(version_id,kind,payload) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING RETURNING payload',(price.version_id,kind,Jsonb(price.to_dict())),required=False)
            if old is None:
                old=await self._one(conn,'SELECT payload FROM prices WHERE version_id=%s AND kind=%s',(price.version_id,kind))
                if old['payload']!=price.to_dict(): raise BillingError('immutable_price')

    async def get_price(self,version_id,kind='customer'):
        async with self.connection() as conn:
            row=await self._one(conn,'SELECT payload FROM prices WHERE version_id=%s AND kind=%s',(version_id,kind),required=False)
            if row is None: raise BillingError('price_unavailable')
            return PriceVersion(**row['payload'])

    async def put_fx(self,quote:FxQuote):
        payload=quote.__dict__
        async with self.connection() as conn:
            old=await self._one(conn,'INSERT INTO fx_quotes(quote_id,payload) VALUES(%s,%s) ON CONFLICT DO NOTHING RETURNING payload',(quote.quote_id,Jsonb(payload)),required=False)
            if old is None:
                old=await self._one(conn,'SELECT payload FROM fx_quotes WHERE quote_id=%s',(quote.quote_id,))
                if old['payload']!=payload: raise BillingError('immutable_fx')

    @_exact_money
    async def meter(self,organization_id,attempt_id,event_id,provider,raw_usage,customer_price_version,supplier_price_version=None,fx_version=None):
        normalized=normalize_usage(provider,raw_usage)
        price=await self.get_price(customer_price_version) if customer_price_version else None
        async with self.connection() as conn:
            i=await self._one(conn,'SELECT i.*,a.dispatched_at FROM invocations i JOIN attempts a USING(invocation_id) WHERE i.organization_id=%s AND a.attempt_id=%s',(organization_id,attempt_id))
        at=_wire(i['dispatched_at'] or i['created_at'])
        if price is not None and normalized.input_total is not None: resolve_price([price],normalized.input_total,at=at)
        if price is None and i['funding']=='SEMANTIER_CREDITS': raise BillingError('price_unavailable')
        if price is not None and price.currency!=i['currency']: raise BillingError('currency_mismatch')
        customer=rate_usage(normalized,price) if price is not None and normalized.completeness=='PROVIDER_REPORTED' else None
        supplier=None
        supplier_native=None
        supplier_currency=None
        if supplier_price_version and normalized.completeness=='PROVIDER_REPORTED':
            supplier_price=await self.get_price(supplier_price_version,'supplier')
            resolve_price([supplier_price],normalized.input_total,at=at)
            supplier_native=rate_usage(normalized,supplier_price)
            supplier_currency=supplier_price.currency
            if supplier_price.currency!=i['currency']:
                if not fx_version: raise BillingError('fx_unavailable')
                async with self.connection() as conn:
                    quote=FxQuote(**(await self._one(conn,'SELECT payload FROM fx_quotes WHERE quote_id=%s',(fx_version,)))['payload'])
                supplier=quote.convert(supplier_native,supplier_price.currency,i['currency']).quantize(Decimal('.000000000001'))
            elif normalized.completeness=='PROVIDER_REPORTED': supplier=supplier_native
        return await self.retain_usage(organization_id,attempt_id,event_id,raw_usage,normalized.to_dict(),customer,supplier,customer_price_version,fx_version,supplier_price_version=supplier_price_version,supplier_native_cost=supplier_native,supplier_currency=supplier_currency)

    @_exact_money
    async def settle(self,organization_id,invocation_id):
        async with self.connection() as conn:
            initial=await self._invocation(conn,organization_id,invocation_id)
            await self._account(conn,organization_id,initial['account_id'],initial['currency'])
            i=await self._invocation(conn,organization_id,invocation_id,lock=True)
            if not i['hold_active']: return _wire(i)
            raw=await (await conn.execute('SELECT * FROM raw_observations WHERE attempt_id=%s ORDER BY id',(i['attempt_id'],))).fetchall()
            if raw and final_native_observation(raw,i['attribution'].get('provider')) is None:
                await conn.execute("UPDATE invocations SET execution_state='POSSIBLY_EXECUTED',settlement_state='PENDING_RECONCILIATION' WHERE invocation_id=%s",(invocation_id,))
                return _wire(await self._invocation(conn,organization_id,invocation_id))
            # Cumulative snapshots replace, never sum. The greatest supported
            # cumulative volume wins; late smaller snapshots cannot regress it.
            observations=await (await conn.execute('SELECT * FROM usage_observations WHERE attempt_id=%s ORDER BY id',(i['attempt_id'],))).fetchall()
            priced=[u for u in observations if u['customer_cost'] is not None and u['normalized'].get('completeness') in ('PROVIDER_REPORTED','RECONCILED')]
            if not priced:
                await conn.execute("UPDATE invocations SET settlement_state='PENDING_RECONCILIATION' WHERE invocation_id=%s",(invocation_id,))
                return _wire(await self._invocation(conn,organization_id,invocation_id))
            u=max(priced,key=lambda x:((x['normalized'].get('input_total') or 0)+(x['normalized'].get('output_total') or 0),x['id']))
            debit=min(u['customer_cost'],i['max_cost'])
            supplier=u['supplier_cost'] if u['supplier_cost'] is not None else u['customer_cost']
            variance=max(supplier-i['max_cost'],Decimal(0))
            customer_overrun=u['customer_cost']>i['max_cost']
            if variance or customer_overrun:
                await conn.execute('UPDATE accounts SET blocked=true WHERE organization_id=%s AND account_id=%s AND currency=%s',(organization_id,i['account_id'],i['currency']))
            settlement_id=uuid4().hex
            if i['funding']=='SEMANTIER_CREDITS':
                await conn.execute("INSERT INTO ledger_transactions(id,organization_id,account_id,currency,kind,invocation_id,price_version) VALUES(%s,%s,%s,%s,'consumption',%s,%s)",(settlement_id,organization_id,i['account_id'],i['currency'],invocation_id,u['price_version']))
                await conn.execute("INSERT INTO ledger_entries(transaction_id,book_account,amount) VALUES(%s,'wallet',%s),(%s,'consumption',%s)",(settlement_id,-debit,settlement_id,debit))
                await conn.execute('UPDATE accounts SET wallet=wallet-%s WHERE organization_id=%s AND account_id=%s AND currency=%s',(debit,organization_id,i['account_id'],i['currency']))
            # BYOK budget exposure still counts, while its credit debit is absent.
            await self._close_hold(conn,i,debit)
            await conn.execute("UPDATE invocations SET execution_state='FINISHED',settlement_state=%s,customer_debit=%s,supplier_variance=%s,settlement_id=%s,updated_at=now() WHERE invocation_id=%s",('SETTLED' if i['funding']=='SEMANTIER_CREDITS' else 'NOT_APPLICABLE',debit if i['funding']=='SEMANTIER_CREDITS' else Decimal(0),variance,settlement_id,invocation_id))
            await conn.execute('INSERT INTO outbox(id,invocation_id,payload) VALUES(%s,%s,%s)',(uuid4().hex,invocation_id,Jsonb({'kind':'usage_settled','invocation_id':invocation_id,'settlement_id':settlement_id,'organization_id':organization_id})))
            await self._enqueue_account_sync(conn,organization_id,i['account_id'],i['currency'],'settlement_eligibility')
            return _wire(await self._invocation(conn,organization_id,invocation_id))

    async def reconcile(self,limit=100):
        async with self.connection() as conn:
            rows=await (await conn.execute('SELECT organization_id,invocation_id,execution_state FROM invocations WHERE hold_active ORDER BY created_at LIMIT %s',(min(limit,1000),))).fetchall()
        for row in rows:
            invocation=await self.get_invocation(row['organization_id'],row['invocation_id'])
            pinned=invocation['attribution']
            receipts=await self.read_raw_observations(row['organization_id'],invocation['attempt_id'])
            final=final_native_observation(receipts,pinned.get('provider'))
            if receipts and (final is None or not pinned.get('provider')):
                await self.mark_uncertain(row['organization_id'],row['invocation_id'])
                continue
            if final:
                try:
                    await self.meter(row['organization_id'],invocation['attempt_id'],'native:'+final['event_id'],final['provider'],final['raw_usage'],pinned.get('customer_price_version'),pinned.get('supplier_price_version'),pinned.get('fx_version'))
                except (BillingError,ValueError):
                    await self.mark_uncertain(row['organization_id'],row['invocation_id'])
                    continue
            result=await self.settle(row['organization_id'],row['invocation_id'])
            if result['hold_active'] and row['execution_state'] in ('DISPATCHING','STREAMING','POSSIBLY_EXECUTED'):
                await self.mark_uncertain(row['organization_id'],row['invocation_id'])
        return len(rows)

    async def claim_outbox(self,worker_id,limit=100,lease_seconds=30):
        if not worker_id or lease_seconds<1 or lease_seconds>3600: raise ValueError('invalid outbox lease')
        async with self.connection() as conn:
            rows=await (await conn.execute("WITH selected AS (SELECT id FROM outbox WHERE delivered_at IS NULL AND (lease_until IS NULL OR lease_until<now()) ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT %s) UPDATE outbox o SET lease_owner=%s,lease_until=now()+%s*interval '1 second',delivery_attempts=delivery_attempts+1 FROM selected s WHERE o.id=s.id RETURNING o.*",(min(limit,1000),worker_id,lease_seconds))).fetchall()
            return _wire(rows)

    async def ack_outbox(self,event_id,worker_id):
        async with self.connection() as conn:
            row=await self._one(conn,'UPDATE outbox SET delivered_at=now() WHERE id=%s AND lease_owner=%s AND lease_until>=now() AND delivered_at IS NULL RETURNING id',(event_id,worker_id),required=False)
            if row is None: raise BillingError('lease_lost')

    async def get_invocation(self,organization_id,invocation_id):
        async with self.connection() as conn: return _wire(await self._invocation(conn,organization_id,invocation_id))

    async def list_usage(self,organization_id,limit=100,offset=0):
        async with self.connection() as conn:
            return _wire(await (await conn.execute('SELECT i.*,a.attempt_id FROM invocations i JOIN attempts a USING(invocation_id) WHERE organization_id=%s ORDER BY created_at DESC LIMIT %s OFFSET %s',(organization_id,min(max(limit,1),500),max(offset,0)))).fetchall())

    @_exact_money
    async def account_summary(self,organization_id,account_id,currency=None):
        async with self.connection() as conn:
            rows=await (await conn.execute('SELECT * FROM accounts WHERE organization_id=%s AND account_id=%s'+(' AND currency=%s' if currency else '')+' ORDER BY currency',(organization_id,account_id,currency) if currency else (organization_id,account_id))).fetchall()
            if not rows: raise BillingError('not_found')
            if len(rows)>1 and currency is None: raise BillingError('currency_required')
            a=rows[0]
            unresolved=await self._one(conn,"SELECT count(*) AS count FROM invocations WHERE organization_id=%s AND account_id=%s AND currency=%s AND hold_active",(organization_id,account_id,a['currency']))
            return _wire({'billing_account_id':account_id,'organization_id':organization_id,'funding_source':a['funding'],'balances':[{'amount':a['wallet'],'currency':a['currency']}] if a['funding']=='SEMANTIER_CREDITS' else [],'held':{'amount':a['held'],'currency':a['currency']},'available':{'amount':a['wallet']-a['held'],'currency':a['currency']} if a['funding']=='SEMANTIER_CREDITS' else None,'budget':{'limit':a['budget_limit'],'spent':a['budget_spent'],'currency':a['currency']},'unresolved_attempts':unresolved['count'],'blocked':a['blocked']})

    async def list_postings(self,organization_id,account_id,currency=None):
        async with self.connection() as conn:
            return _wire(await (await conn.execute('SELECT t.*,e.book_account,e.amount FROM ledger_transactions t JOIN ledger_entries e ON e.transaction_id=t.id WHERE organization_id=%s AND account_id=%s'+(' AND currency=%s' if currency else '')+' ORDER BY t.created_at,e.id',(organization_id,account_id,currency) if currency else (organization_id,account_id))).fetchall())

    async def verify_ledger(self,organization_id,account_id):
        async with self.connection() as conn:
            bad=await self._one(conn,'SELECT count(*) AS count FROM (SELECT t.id FROM ledger_transactions t LEFT JOIN ledger_entries e ON e.transaction_id=t.id WHERE organization_id=%s AND account_id=%s GROUP BY t.id HAVING coalesce(sum(e.amount),0)<>0) unbalanced',(organization_id,account_id))
            mismatch=await self._one(conn,"SELECT count(*) AS count FROM accounts a WHERE organization_id=%s AND account_id=%s AND wallet<>(SELECT coalesce(sum(e.amount),0) FROM ledger_transactions t JOIN ledger_entries e ON e.transaction_id=t.id WHERE t.organization_id=a.organization_id AND t.account_id=a.account_id AND t.currency=a.currency AND e.book_account='wallet')",(organization_id,account_id))
            return {'balanced':bad['count']==0,'wallet_matches':mismatch['count']==0}

    async def compare(self,price_versions,provider,raw_usage,fx_version=None):
        """Pure scenario over server-authorized catalog IDs; no provider execution."""
        usage=normalize_usage(provider,raw_usage)
        quote=None
        if fx_version:
            async with self.connection() as conn:
                quote=FxQuote(**(await self._one(conn,'SELECT payload FROM fx_quotes WHERE quote_id=%s',(fx_version,)))['payload'])
        reports=[]
        for version in price_versions:
            price=await self.get_price(version)
            reason=None
            try:
                cost=rate_usage(usage,price)
            except ValueError as exc:
                cost=None
                reason=str(exc)
            standard=None
            try:
                standard=standard_usd_rates(price,quote)
                scenario=effective_usd_per_million(cost,price.currency,usage.total_tokens,quote,complete=cost is not None)
            except ValueError:
                scenario={'known_cost':{'amount':format(cost,'f') if cost is not None else None,'currency':price.currency},'usd_per_million':None,'reason':'fx_unavailable','complete':False}
            reports.append({'price_version':version,'native_currency':price.currency,'normalized':usage.to_dict(),'standard_usd_per_million':standard,'scenario':scenario,'pricing_reason':reason,'realized':{'usd_per_million':None,'reason':'period_cost_basis_unavailable'}})
        return reports

    async def export_invocation(self,organization_id,invocation_id):
        """Return a self-contained hashed metering bundle; never prompt content."""
        from billing.replay import bundle_hash
        async with self.connection() as conn:
            # A repeatable snapshot binds all exported rows to one DB view.
            await conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            invocation=await self._invocation(conn,organization_id,invocation_id)
            usage=await (await conn.execute('SELECT * FROM usage_observations WHERE attempt_id=%s ORDER BY id',(invocation['attempt_id'],))).fetchall()
            raw=await (await conn.execute('SELECT * FROM raw_observations WHERE attempt_id=%s ORDER BY id',(invocation['attempt_id'],))).fetchall()
            ids=list({u[key] for u in usage for key in ('price_version','supplier_price_version') if u[key]})
            prices=await (await conn.execute('SELECT * FROM prices WHERE version_id=ANY(%s) ORDER BY version_id,kind',(ids,))).fetchall() if ids else []
            fxids=list({u['fx_version'] for u in usage if u['fx_version']})
            quotes=await (await conn.execute('SELECT * FROM fx_quotes WHERE quote_id=ANY(%s) ORDER BY quote_id',(fxids,))).fetchall() if fxids else []
            postings=await (await conn.execute('SELECT t.*,e.book_account,e.amount FROM ledger_transactions t JOIN ledger_entries e ON t.id=e.transaction_id WHERE t.invocation_id=%s ORDER BY e.id',(invocation_id,))).fetchall()
            bundle=_wire({'schema_version':'semantier.metering-bundle.v1','invocation':invocation,'usage_observations':usage,'raw_observations':raw,'prices':prices,'fx_quotes':quotes,'postings':postings})
            bundle['sha256']=bundle_hash(bundle)
            return bundle

    async def capture_admission(self,attempt_id,invocation_id):
        """Private capture validates an already-admitted send, never a client payer."""
        async with self.connection() as conn:
            row=await self._one(conn,"SELECT i.organization_id,i.account_id,i.deployment_id,i.invocation_id,a.attempt_id,i.attribution,i.execution_state,a.outbound_intent,i.hold_active FROM invocations i JOIN attempts a USING(invocation_id) WHERE a.attempt_id=%s AND i.invocation_id=%s",(attempt_id,invocation_id),required=False)
            if row is None or row['execution_state']!='DISPATCHING' or row['outbound_intent']!='POSSIBLY_SENT' or not row['hold_active'] or not row['attribution'].get('provider'):
                raise BillingError('invalid_capture_admission')
            return _wire(row)|{'provider':row['attribution']['provider']}

    async def claim_native_send(self,attempt_id,invocation_id,provider):
        """Commit one durable physical-send claim before private network dispatch."""
        async with self.connection() as conn:
            row=await self._one(conn,'SELECT i.* FROM invocations i JOIN attempts a USING(invocation_id) WHERE a.attempt_id=%s AND i.invocation_id=%s FOR UPDATE OF i',(attempt_id,invocation_id),required=False)
            if row is None or row['execution_state']!='DISPATCHING' or not row['hold_active'] or not provider or row['attribution'].get('provider')!=provider:
                raise BillingError('invalid_capture_admission')
            claim=await self._one(conn,"UPDATE attempts SET native_send_claimed_at=now() WHERE attempt_id=%s AND invocation_id=%s AND outbound_intent='POSSIBLY_SENT' AND native_send_claimed_at IS NULL RETURNING attempt_id,invocation_id,native_send_claimed_at",(attempt_id,invocation_id),required=False)
            if claim is None: raise BillingError('native_send_already_claimed')
            return _wire(claim)|{'provider':provider,'organization_id':row['organization_id']}

    async def set_account_budget(self,organization_id,account_id,currency,limit,actor_user_id):
        """Change an explicitly authorized account ceiling without changing credit."""
        amount=decimal_amount(limit)
        currency_code(currency)
        if amount>=Decimal('1e26') or not actor_user_id:
            raise BillingError('invalid_budget')
        async with self.connection() as conn:
            account=await self._account(conn,organization_id,account_id,currency)
            if amount<account['budget_spent']+account['held']:
                raise BillingError('budget_below_committed_exposure')
            await conn.execute('UPDATE accounts SET budget_limit=%s WHERE organization_id=%s AND account_id=%s AND currency=%s',(amount,organization_id,account_id,currency))
            payload={'prior_limit':format(account['budget_limit'],'f'),'limit':format(amount,'f'),'currency':currency}
            await conn.execute('INSERT INTO financial_policy_events(event_id,organization_id,actor_user_id,operation,target,payload) VALUES(%s,%s,%s,%s,%s,%s)',(uuid4().hex,organization_id,actor_user_id,'budget.update',account_id,Jsonb(payload)))
            await self._enqueue_account_sync(conn,organization_id,account_id,currency,'budget_policy')
        return {'billing_account_id':account_id,'limit':{'amount':format(amount,'f'),'currency':currency}}

    async def create_customer_price(self,organization_id,price:PriceVersion,actor_user_id):
        """Append a tenant-owned price candidate. This does not activate any tariff."""
        if not organization_id or not actor_user_id:
            raise BillingError('price_owner_required')
        async with self.connection() as conn:
            inserted=await self._one(conn,"INSERT INTO prices(version_id,kind,payload) VALUES(%s,'customer',%s) ON CONFLICT DO NOTHING RETURNING version_id",(price.version_id,Jsonb(price.to_dict())),required=False)
            if inserted is None:
                raise BillingError('immutable_price')
            await conn.execute("INSERT INTO organization_customer_prices(organization_id,version_id) VALUES(%s,%s)",(organization_id,price.version_id))
            await conn.execute('INSERT INTO financial_policy_events(event_id,organization_id,actor_user_id,operation,target,payload) VALUES(%s,%s,%s,%s,%s,%s)',(uuid4().hex,organization_id,actor_user_id,'price.create',price.version_id,Jsonb(price.to_dict())))
        return {'version_id':price.version_id,'payload':price.to_dict(),'activation':'NOT_ACTIVATED'}

    async def list_customer_prices(self,organization_id):
        async with self.connection() as conn:
            return _wire(await (await conn.execute("SELECT p.* FROM prices p JOIN organization_customer_prices o USING(version_id,kind) WHERE o.organization_id=%s ORDER BY p.created_at DESC",(organization_id,))).fetchall())

    async def create_zero_credit_account(self,organization_id,account_id,currency,funding,budget_limit,*,connection=None):
        """Server provisioning only. Never mint a wallet balance or ledger issuance."""
        amount=decimal_amount(budget_limit)
        currency_code(currency)
        if funding not in {'CUSTOMER_CLOUD','SEMANTIER_CREDITS'} or not account_id or not account_id.isascii():
            raise BillingError('invalid_account')
        async def create(conn):
            inserted=await self._one(conn,'INSERT INTO accounts(organization_id,account_id,currency,funding,budget_limit) VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING account_id',(organization_id,account_id,currency,funding,amount),required=False)
            if inserted is None:
                raise BillingError('account_already_exists')
            await conn.execute('INSERT INTO financial_policy_events(event_id,organization_id,actor_user_id,operation,target,payload) VALUES(%s,%s,%s,%s,%s,%s)',(uuid4().hex,organization_id,'server-provisioning','account.create',account_id,Jsonb({'currency':currency,'funding':funding,'budget_limit':format(amount,'f'),'wallet':'0'})))
            return _wire(await self._account(conn,organization_id,account_id,currency))
        if connection is not None:
            if (not isinstance(connection,psycopg.AsyncConnection) or connection.autocommit
                    or connection.info.transaction_status!=psycopg.pq.TransactionStatus.INTRANS):
                raise ValueError('An active caller-owned PostgreSQL transaction is required')
            return await create(connection)
        async with self.connection() as conn:
            await create(conn)
        return await self.account_summary(organization_id,account_id,currency)
