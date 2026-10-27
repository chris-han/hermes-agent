"""Development commerce over the core financial owner. No live merchant connector."""
import asyncio
import calendar
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import urlsplit
from uuid import uuid4
from psycopg.types.json import Jsonb
from billing.store import _wire

VERSION='development-v2'
FIXTURE_TERMS={'terms_version':'local-test-terms-v1','manual_renewal':True,
               'allowance_expiry':'cycle_end','refund_policy':'local-test-no-real-refund-v1',
               'tax_basis':'local-test-no-tax-invoice'}
PRODUCTS=[
    dict(product_id='professional_monthly',plan='Professional',term='monthly',amount='199.00',currency='CNY',monthly_allowance='100.00',months=1,purchasable=True,platform_fee='99.00',allowance_total='100.00')|FIXTURE_TERMS,
    dict(product_id='professional_annual',plan='Professional',term='annual',amount='2190.00',currency='CNY',monthly_allowance='100.00',months=12,purchasable=True,platform_fee='990.00',allowance_total='1200.00')|FIXTURE_TERMS,
    dict(product_id='credits_100',plan='Credits',term='once',amount='100.00',currency='CNY',monthly_allowance='0',months=0,purchasable=True,platform_fee='0.00',allowance_total='0.00')|FIXTURE_TERMS|{'manual_renewal':False},
    dict(product_id='team_monthly',plan='Team',term='monthly',amount='599.00',currency='CNY',monthly_allowance='300.00',months=1,purchasable=True,platform_fee='299.00',allowance_total='300.00')|FIXTURE_TERMS,
    dict(product_id='team_annual',plan='Team',term='annual',amount='6590.00',currency='CNY',monthly_allowance='300.00',months=12,purchasable=True,platform_fee='2990.00',allowance_total='3600.00')|FIXTURE_TERMS,
    dict(product_id='enterprise',plan='Enterprise',term='custom',amount=None,currency='CNY',monthly_allowance='0',months=0,purchasable=False,quote_required=True),
]

def month(at, n):
    y,m=divmod(at.year*12+at.month-1+n,12)
    return at.replace(year=y,month=m+1,day=min(at.day,calendar.monthrange(y,m+1)[1]))

class CommerceService:
    def __init__(self,billing,*,development=False,channels=None):
        self.billing=billing
        self.development=development
        self.channels=dict(channels or {}) if development else {}

    @property
    def payment_environment(self):
        return 'ALIPAY_SANDBOX' if any(a.environment=='ALIPAY_SANDBOX' for a in self.channels.values()) else 'DEVELOPMENT'

    @property
    def merchant_id(self):
        return self.channels['alipay'].merchant_id if self.payment_environment=='ALIPAY_SANDBOX' else 'local-development'

    def catalog(self):
        return {'version':VERSION,'environment':self.payment_environment if self.development else 'UNAVAILABLE',
                'available':self.development,'merchant':{'id':self.merchant_id,'display_name':'ALIPAY SANDBOX — virtual funds only' if self.payment_environment=='ALIPAY_SANDBOX' else 'LOCAL DEVELOPMENT — no real merchant'} if self.development else None,
                'products':[(p|{'purchasable':p['purchasable'] and (self.payment_environment!='ALIPAY_SANDBOX' or p['product_id'] in {'professional_monthly','credits_100'})}) for p in PRODUCTS] if self.development else [],
                'payment_methods':list(self.channels) or ['wechat_test','alipay_test'] if self.development else [],'overseas_available':False}

    def enabled(self):
        if not self.development: raise ValueError('commercial service unavailable')

    async def bind_account(self,org,aid):
        """Deployment-owned DEVELOPMENT fixture provisioning, never an HTTP command."""
        self.enabled()
        async with self.billing.connection() as c:
            await self.billing._account(c,org,aid,'CNY')
            await c.execute("INSERT INTO commerce_account_bindings(organization_id,account_id,currency,merchant_id,environment) VALUES(%s,%s,'CNY','local-development','DEVELOPMENT') ON CONFLICT DO NOTHING",(org,aid))

    async def binding(self,c,org,aid):
        return await self.billing._one(c,"SELECT * FROM commerce_account_bindings WHERE organization_id=%s AND account_id=%s AND merchant_id='local-development' AND environment='DEVELOPMENT'",(org,aid))

    async def provision_enterprise_offer(self,org,aid,*,amount,monthly_allowance,months,reference):
        """Operator-only DEVELOPMENT offer; no HTTP route can set commercial terms."""
        self.enabled()
        total,allowance=Decimal(amount),Decimal(monthly_allowance)
        if (not total.is_finite() or not allowance.is_finite() or total<=0 or allowance<=0
                or type(months) is not int or months not in {1,12} or allowance*months>total
                or total!=total.quantize(Decimal('0.01')) or allowance!=allowance.quantize(Decimal('0.01'))
                or not isinstance(reference,str) or not 1<=len(reference)<=128):
            raise ValueError('invalid enterprise offer')
        offer_id=uuid4().hex
        product=dict(product_id='enterprise',plan='Enterprise',term='annual' if months==12 else 'monthly',
            amount=str(total),currency='CNY',monthly_allowance=str(allowance),months=months,purchasable=True,
            platform_fee=str(total-allowance*months),allowance_total=str(allowance*months),
            enterprise_offer_id=offer_id,contract_reference=reference)|FIXTURE_TERMS
        async with self.billing.connection() as c:
            await self.binding(c,org,aid)
            row=await self.billing._one(c,'INSERT INTO commerce_enterprise_offers(offer_id,organization_id,account_id,product,expires_at) VALUES(%s,%s,%s,%s,%s) RETURNING *',
                (offer_id,org,aid,Jsonb(product),datetime.now(timezone.utc)+timedelta(days=7)))
            return _wire(row)

    async def quote(self,org,aid,product_id):
        self.enabled()
        if self.payment_environment=='ALIPAY_SANDBOX' and product_id not in {'professional_monthly','credits_100'}:raise ValueError('sandbox product unavailable')
        product=next((p for p in PRODUCTS if p['product_id']==product_id and p['purchasable']),None)
        if product is None and product_id!='enterprise': raise ValueError('product unavailable')
        async with self.billing.connection() as c:
            await self.binding(c,org,aid)
            expiry=datetime.now(timezone.utc)+timedelta(minutes=15)
            if product_id=='enterprise':
                offer=await self.billing._one(c,'SELECT * FROM commerce_enterprise_offers WHERE organization_id=%s AND account_id=%s AND expires_at>now() ORDER BY created_at DESC LIMIT 1',(org,aid),required=False)
                if not offer: raise ValueError('enterprise offer required; contact your account administrator')
                product=offer['product']
                expiry=min(expiry,offer['expires_at'])
            a=await self.billing._account(c,org,aid,'CNY')
            if a['funding']!='SEMANTIER_CREDITS': raise ValueError('account ineligible')
            snapshot=product|{'version':VERSION,'environment':self.payment_environment,'merchant_id':self.merchant_id}
            if self.channels:
                snapshot['channel_bindings']={name:{'merchant_id':a.merchant_id,'app_id':a.app_id,'environment':a.environment} for name,a in self.channels.items()}
            q=await self.billing._one(c,'INSERT INTO commerce_quotes(quote_id,organization_id,account_id,product,expires_at) VALUES(%s,%s,%s,%s,%s) RETURNING *',(uuid4().hex,org,aid,Jsonb(snapshot),expiry))
            return _wire(q)|snapshot

    def wire_order(self,row):
        return _wire(row)|{'amount':row['product']['amount'],'currency':row['product']['currency'],'environment':row['product'].get('environment','DEVELOPMENT')}

    async def order(self,org,aid,quote_id,key,confirmed):
        self.enabled()
        if confirmed is not True or not isinstance(key,str) or not 1<=len(key)<=128: raise ValueError('explicit confirmation and bounded idempotency key required')
        async with self.billing.connection() as c:
            await c.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(str([org,aid,key]),))
            old=await self.billing._one(c,'SELECT * FROM commerce_orders WHERE organization_id=%s AND account_id=%s AND idempotency_key=%s',(org,aid,key),required=False)
            if old:
                if old['quote_id']!=quote_id: raise ValueError('idempotency conflict')
                return self.wire_order(old)
            q=await self.billing._one(c,'SELECT * FROM commerce_quotes WHERE organization_id=%s AND account_id=%s AND quote_id=%s',(org,aid,quote_id))
            if q['expires_at']<=datetime.now(timezone.utc): raise ValueError('quote expired; reconfirm required')
            if q['product']['version']!=VERSION: raise ValueError('quote version changed; reconfirm required')
            overlap=await self.billing._one(c,'SELECT order_id FROM commerce_subscriptions WHERE organization_id=%s AND account_id=%s AND ends_at>now()',(org,aid),required=False)
            if overlap and q['product']['months']: raise ValueError('active subscription; renewal unavailable until expiry')
            row=await self.billing._one(c,'INSERT INTO commerce_orders(order_id,organization_id,account_id,quote_id,idempotency_key,product) VALUES(%s,%s,%s,%s,%s,%s) RETURNING *',(uuid4().hex,org,aid,quote_id,key,Jsonb(q['product'])))
            return self.wire_order(row)

    async def get_order(self,org,aid,oid):
        async with self.billing.connection() as c:
            order=self.wire_order(await self.billing._one(c,'SELECT * FROM commerce_orders WHERE organization_id=%s AND account_id=%s AND order_id=%s FOR UPDATE',(org,aid,oid)))
            order['payment_attempt']=_wire(await self.payment_attempt(c,oid))
        if self.development and order['status']=='PAID_PENDING_FULFILLMENT':
            await self.fulfill(oid)
            return await self.get_order(org,aid,oid)
        if (self.development and order['status']=='PENDING_PAYMENT' and order['payment_attempt']
                and order['payment_attempt']['status']=='SUCCEEDED' and order['payment_attempt']['method'] in {'wechat_test','alipay_test'}):
            await self.ingest(self.fixture_event(order),verified=True)
            await self.fulfill(oid)
            return await self.get_order(org,aid,oid)
        return order

    async def orders(self,org,aid):
        async with self.billing.connection() as c:
            return [self.wire_order(r) for r in await (await c.execute('SELECT * FROM commerce_orders WHERE organization_id=%s AND account_id=%s ORDER BY created_at DESC LIMIT 100',(org,aid))).fetchall()]

    def fixture_event(self,o):
        return {'event_id':'local-test:'+o['order_id'],'order_id':o['order_id'],'merchant_id':'local-development','currency':o['currency'],'amount':o['amount'],'reference':'fixture:'+o['order_id'],'status':'SUCCESS','source':'LOCAL_TEST'}

    async def payment_attempt(self,c,oid):
        await c.execute("UPDATE commerce_payment_attempts SET status='EXPIRED' WHERE order_id=%s AND status='AWAITING_PAYMENT' AND expires_at<=now()",(oid,))
        return await self.billing._one(c,'SELECT * FROM commerce_payment_attempts WHERE order_id=%s ORDER BY created_at DESC,attempt_id DESC LIMIT 1',(oid,),required=False)

    async def start_payment(self,org,aid,oid,method,*,language='en'):
        self.enabled()
        if method in self.channels:
            return await self.start_channel_payment(org,aid,oid,method,language=language)
        if method not in {'wechat_test','alipay_test'} or self.channels: raise ValueError('payment method unavailable')
        async with self.billing.connection() as c:
            o=await self.billing._one(c,'SELECT * FROM commerce_orders WHERE organization_id=%s AND account_id=%s AND order_id=%s FOR UPDATE',(org,aid,oid))
            await self.binding(c,org,aid)
            if o['product'].get('channel_bindings'):raise ValueError('quoted SDK channel cannot use simulator')
            if o['status']!='PENDING_PAYMENT': raise ValueError('order already paid; refresh status')
            old=await self.payment_attempt(c,oid)
            if old and old['status']=='AWAITING_PAYMENT':
                if old['method']!=method: raise ValueError('cancel the pending attempt before changing payment method')
                return _wire(old)
            if old and old['status']=='SUCCEEDED': raise ValueError('payment confirming; refresh status')
            row=await self.billing._one(c,"INSERT INTO commerce_payment_attempts(attempt_id,order_id,method,status,expires_at) VALUES(%s,%s,%s,'AWAITING_PAYMENT',%s) RETURNING *",
                (uuid4().hex,oid,method,datetime.now(timezone.utc)+timedelta(minutes=10)))
            return _wire(row)

    async def resolve_payment(self,org,aid,oid,attempt_id,outcome):
        self.enabled()
        statuses={'success':'SUCCEEDED','cancel':'CANCELED','decline':'DECLINED'}
        if outcome not in statuses: raise ValueError('unsupported mock outcome')
        expired=False
        async with self.billing.connection() as c:
            order=await self.billing._one(c,'SELECT * FROM commerce_orders WHERE organization_id=%s AND account_id=%s AND order_id=%s FOR UPDATE',(org,aid,oid))
            if order['product'].get('channel_bindings'):raise ValueError('quoted SDK channel cannot use simulator')
            await self.binding(c,org,aid)
            await self.payment_attempt(c,oid)
            a=await self.billing._one(c,'SELECT * FROM commerce_payment_attempts WHERE order_id=%s AND attempt_id=%s',(oid,attempt_id))
            if a['method'] not in {'wechat_test','alipay_test'}:raise ValueError('channel payment cannot be simulated')
            if a['status']=='EXPIRED':
                expired=True
            elif a['status'] not in {'AWAITING_PAYMENT',statuses[outcome]}:
                raise ValueError('payment attempt already resolved')
            else:
                await c.execute('UPDATE commerce_payment_attempts SET status=%s WHERE attempt_id=%s',(statuses[outcome],attempt_id))
        if expired: raise ValueError('payment attempt expired; start a new attempt')
        if outcome=='success':
            # A crash here is recoverable by retrying this exact succeeded attempt.
            order=await self.get_order(org,aid,oid)
            await self.ingest(self.fixture_event(order),verified=True)
            await self.fulfill(oid)
        return await self.get_order(org,aid,oid)

    async def test_payment(self,org,aid,oid,method='wechat_test'):
        """Compatibility helper for controlled fixtures; UI uses explicit attempts."""
        self.enabled()
        if method not in {'wechat_test','alipay_test'}: raise ValueError('payment method unavailable')
        order=await self.get_order(org,aid,oid)
        if order['product'].get('channel_bindings'):raise ValueError('quoted SDK channel cannot use simulator')
        if order['status']!='PENDING_PAYMENT':
            await self.ingest(self.fixture_event(order),verified=True)
            await self.fulfill(oid)
            return await self.get_order(org,aid,oid)
        attempt=order.get('payment_attempt')
        if not attempt or attempt['status']!='SUCCEEDED':
            attempt=await self.start_payment(org,aid,oid,method)
        return await self.resolve_payment(org,aid,oid,attempt['attempt_id'],'success')

    async def ingest(self,event,*,verified=False):
        self.enabled()
        if not verified or event.get('source')!='LOCAL_TEST': raise ValueError('unverified payment')
        async with self.billing.connection() as c:
            o=await self.billing._one(c,'SELECT * FROM commerce_orders WHERE order_id=%s FOR UPDATE',(event.get('order_id'),))
            await self.binding(c,o['organization_id'],o['account_id'])
            if o['product'].get('channel_bindings'):raise ValueError('quoted SDK channel cannot use simulator')
            expected=self.fixture_event(self.wire_order(o))
            if event.get('status') in {'REFUND_REQUESTED','ADJUSTMENT_REQUIRED'}:
                expected.update(status=event['status'],event_id=expected['event_id']+':'+event['status'])
            if event!=expected: raise ValueError('payment binding mismatch')
            old=await self.billing._one(c,'SELECT payload FROM commerce_payment_events WHERE event_id=%s',(event['event_id'],),required=False)
            if old:
                if old['payload']!=event: raise ValueError('payment event conflict')
                return
            await c.execute('INSERT INTO commerce_payment_events(event_id,order_id,payload) VALUES(%s,%s,%s)',(event['event_id'],o['order_id'],Jsonb(event)))
            if event['status'] in {'REFUND_REQUESTED','ADJUSTMENT_REQUIRED'}:
                if o['status'] not in {'ACTIVE','REFUND_PENDING','ADJUSTMENT_PENDING'}: raise ValueError('invalid payment transition')
                await c.execute('UPDATE commerce_orders SET status=%s WHERE order_id=%s',('REFUND_PENDING' if event['status']=='REFUND_REQUESTED' else 'ADJUSTMENT_PENDING',o['order_id']))
                return
            if o['status']=='PENDING_PAYMENT':
                await c.execute("UPDATE commerce_orders SET status='PAID_PENDING_FULFILLMENT' WHERE order_id=%s",(o['order_id'],))
                await c.execute('INSERT INTO commerce_fulfillment(order_id) VALUES(%s) ON CONFLICT DO NOTHING',(o['order_id'],))

    def channel(self,method):
        self.enabled()
        if method not in self.channels:raise ValueError('payment channel unavailable')
        return self.channels[method]

    def check_channel_binding(self,order,method,adapter):
        binding=order['product'].get('channel_bindings',{}).get(method)
        expected={'merchant_id':adapter.merchant_id,'app_id':adapter.app_id,'environment':adapter.environment}
        if binding!=expected or adapter.environment not in {'DEVELOPMENT','ALIPAY_SANDBOX'} or (adapter.environment=='ALIPAY_SANDBOX' and method!='alipay'):raise ValueError('immutable channel binding mismatch')

    async def start_channel_payment(self,org,aid,oid,method,*,language='en'):
        adapter=self.channel(method)
        async with self.billing.connection() as c:
            o=await self.billing._one(c,'SELECT * FROM commerce_orders WHERE organization_id=%s AND account_id=%s AND order_id=%s FOR UPDATE',(org,aid,oid))
            await self.binding(c,org,aid)
            self.check_channel_binding(o,method,adapter)
            if o['status']!='PENDING_PAYMENT':raise ValueError('order already paid; refresh status')
            old=await self.payment_attempt(c,oid)
            if old and old['status']!='CANCELED':
                if old['method']!=method:raise ValueError('resolve existing payment attempt before changing method')
                # Even locally expired attempts may have succeeded externally.
                attempt=old
            else:
                attempt=await self.billing._one(c,"INSERT INTO commerce_payment_attempts(attempt_id,order_id,method,status,expires_at) VALUES(%s,%s,%s,'AWAITING_PAYMENT',%s) RETURNING *",(uuid4().hex,oid,method,datetime.now(timezone.utc)+timedelta(minutes=10)))
        local_page_recovery=(old and old['status'] in {'AWAITING_PAYMENT','EXPIRED'} and not old.get('checkout') and adapter.environment=='ALIPAY_SANDBOX')
        if old and old['status']!='CANCELED' and not local_page_recovery:
            if old.get('checkout') and old['status']=='AWAITING_PAYMENT':return _wire(old)
            event=await asyncio.to_thread(adapter.query,attempt['attempt_id'])
            if event:
                if event['attempt_id']!=attempt['attempt_id']:raise ValueError('query attempt mismatch')
                await self.ingest_channel(event)
                return _wire(attempt)|{'status':'SUCCEEDED','checkout':None}
            raise ValueError('existing channel attempt pending; query or close before retry')
        checkout=await asyncio.to_thread(adapter.create,attempt['attempt_id'],o['product']|{'order_id':oid,'account_id':aid,'language':language})
        async with self.billing.connection() as c:
            row=await self.billing._one(c,'UPDATE commerce_payment_attempts SET checkout=%s WHERE attempt_id=%s AND checkout IS NULL RETURNING *',(Jsonb(checkout),attempt['attempt_id']),required=False)
            if row is None:
                row=await self.billing._one(c,'SELECT * FROM commerce_payment_attempts WHERE attempt_id=%s',(attempt['attempt_id'],))
        return _wire(row)

    async def notify(self,method,headers,body):
        if len(body)>65536:raise ValueError('notification too large')
        event=await asyncio.to_thread(self.channel(method).notification,headers,body)
        if event is None:raise ValueError('payment not confirmed')
        await self.ingest_channel(event)
        return {'accepted':True}

    async def ingest_channel(self,event):
        adapter=self.channel(event.get('method'))
        async with self.billing.connection() as c:
            attempt=await self.billing._one(c,'SELECT * FROM commerce_payment_attempts WHERE attempt_id=%s',(event.get('attempt_id'),))
            o=await self.billing._one(c,'SELECT * FROM commerce_orders WHERE order_id=%s FOR UPDATE',(attempt['order_id'],))
            self.check_channel_binding(o,event['method'],adapter)
            await self.binding(c,o['organization_id'],o['account_id'])
            from billing.payment_channels import amount
            expected={'attempt_id':attempt['attempt_id'],'reference':event.get('reference'),'amount':amount(o['product']['amount']),'currency':o['product']['currency'],'merchant_id':adapter.merchant_id,'app_id':adapter.app_id,'environment':adapter.environment,'method':attempt['method'],'status':'SUCCESS'}
            if event!=expected or not isinstance(event.get('reference'),str) or not 0<len(event['reference'])<=128:raise ValueError('payment binding mismatch')
            # Channel transaction is canonical across callback/query notification IDs.
            eid=event['method']+':'+adapter.merchant_id+':'+event['reference']
            if adapter.environment=='ALIPAY_SANDBOX':
                import hashlib,json
                eid='alipay:ALIPAY_SANDBOX:'+hashlib.sha256(json.dumps([adapter.environment,adapter.app_id,adapter.merchant_id,event['reference']],separators=(',',':')).encode()).hexdigest()
            await c.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(eid,))
            old=await self.billing._one(c,'SELECT order_id,payload FROM commerce_payment_events WHERE event_id=%s',(eid,),required=False)
            if old:
                if old['order_id']!=o['order_id'] or old['payload']!=event:raise ValueError('payment event conflict')
                return
            # A second successful channel transaction is retained as an exception,
            # never another grant or silently ignored cash.
            paid=await self.billing._one(c,'SELECT event_id FROM commerce_payment_events WHERE order_id=%s',(o['order_id'],),required=False)
            await c.execute('INSERT INTO commerce_payment_events(event_id,order_id,payload) VALUES(%s,%s,%s)',(eid,o['order_id'],Jsonb(event)))
            await c.execute("UPDATE commerce_payment_attempts SET status='SUCCEEDED' WHERE attempt_id=%s",(attempt['attempt_id'],))
            if paid:
                await c.execute("UPDATE commerce_orders SET status='PAID_REQUIRES_RECONCILIATION' WHERE order_id=%s",(o['order_id'],))
                await c.execute("UPDATE commerce_fulfillment SET reconciliation_reason='multiple_payment_transactions' WHERE order_id=%s",(o['order_id'],))
            elif o['status']=='PENDING_PAYMENT':
                await c.execute("UPDATE commerce_orders SET status='PAID_PENDING_FULFILLMENT' WHERE order_id=%s",(o['order_id'],))
                await c.execute('INSERT INTO commerce_fulfillment(order_id) VALUES(%s) ON CONFLICT DO NOTHING',(o['order_id'],))

    async def query_channel_payment(self,org,aid,oid,attempt_id,*,close=False):
        self.enabled()
        async with self.billing.connection() as c:
            o=await self.billing._one(c,'SELECT * FROM commerce_orders WHERE organization_id=%s AND account_id=%s AND order_id=%s',(org,aid,oid))
            a=await self.billing._one(c,'SELECT * FROM commerce_payment_attempts WHERE order_id=%s AND attempt_id=%s',(oid,attempt_id))
            adapter=self.channel(a['method']);self.check_channel_binding(o,a['method'],adapter)
        event=await asyncio.to_thread(adapter.query,attempt_id)
        if event:
            if event['attempt_id']!=attempt_id:raise ValueError('query attempt mismatch')
            await self.ingest_channel(event)
        elif close:
            await asyncio.to_thread(adapter.close,attempt_id)
            async with self.billing.connection() as c:
                await c.execute("UPDATE commerce_payment_attempts SET status='CANCELED' WHERE attempt_id=%s AND status<>'SUCCEEDED'",(attempt_id,))
        return await self.get_order(org,aid,oid)

    async def grant(self,c,org,aid,gid,kind,amount,start,end):
        inserted=await self.billing._one(c,'INSERT INTO credit_grants(grant_id,organization_id,account_id,currency,kind,amount,remaining,starts_at,expires_at) VALUES(%s,%s,%s,\'CNY\',%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING grant_id',(gid,org,aid,kind,amount,amount,start,end),required=False)
        if not inserted:return False
        await c.execute("INSERT INTO ledger_transactions(id,organization_id,account_id,currency,kind) VALUES(%s,%s,%s,'CNY',%s)",(gid,org,aid,'commerce_'+kind))
        await c.execute("INSERT INTO ledger_entries(transaction_id,book_account,amount) VALUES(%s,'wallet',%s),(%s,'issuance',%s)",(gid,amount,gid,-amount))
        await c.execute("UPDATE accounts SET wallet=wallet+%s WHERE organization_id=%s AND account_id=%s AND currency='CNY'",(amount,org,aid))
        return True

    async def fulfill(self,oid):
        self.enabled()
        async with self.billing.connection() as c:
            initial=await self.billing._one(c,'SELECT * FROM commerce_orders WHERE order_id=%s',(oid,))
            await self.billing._account(c,initial['organization_id'],initial['account_id'],'CNY')
            o=await self.billing._one(c,'SELECT * FROM commerce_orders WHERE order_id=%s FOR UPDATE',(oid,))
            job=await self.billing._one(c,'SELECT * FROM commerce_fulfillment WHERE order_id=%s FOR UPDATE',(oid,))
            if job['fulfilled_at'] or job['reconciliation_reason']:return
            start=datetime.now(timezone.utc); p=o['product'];org=o['organization_id'];aid=o['account_id']
            if p['months']:
                overlap=await self.billing._one(c,'SELECT order_id FROM commerce_subscriptions WHERE organization_id=%s AND account_id=%s AND ends_at>%s',(org,aid,start),required=False)
                if overlap:
                    await c.execute("UPDATE commerce_fulfillment SET reconciliation_reason='subscription_overlap' WHERE order_id=%s",(oid,))
                    await c.execute("UPDATE commerce_orders SET status='PAID_REQUIRES_RECONCILIATION' WHERE order_id=%s",(oid,))
                    return
                await c.execute('INSERT INTO commerce_subscriptions(order_id,organization_id,account_id,product,starts_at,ends_at) VALUES(%s,%s,%s,%s,%s,%s)',(oid,org,aid,Jsonb(p),start,month(start,p['months'])))
                await self.grant(c,org,aid,oid+':0','allowance',Decimal(p['monthly_allowance']),start,month(start,1))
            else:
                await self.grant(c,org,aid,oid+':credit','purchased',Decimal(p['amount']),start,None)
            await c.execute("UPDATE commerce_orders SET status='ACTIVE' WHERE order_id=%s",(oid,))
            await c.execute('UPDATE commerce_fulfillment SET fulfilled_at=now() WHERE order_id=%s',(oid,))
            await self.billing._enqueue_account_sync(c,org,aid,'CNY','payment_fulfillment')

    async def recover(self):
        self.enabled()
        async with self.billing.connection() as c:
            rows=await (await c.execute('SELECT order_id FROM commerce_fulfillment WHERE fulfilled_at IS NULL AND reconciliation_reason IS NULL ORDER BY order_id LIMIT 100')).fetchall()
        for r in rows: await self.fulfill(r['order_id'])
        return len(rows)

    async def cycles(self,org,aid,at=None):
        self.enabled()
        at=at or datetime.now(timezone.utc)
        if at.tzinfo is None: raise ValueError('UTC-aware time required')
        async with self.billing.connection() as c:
            await self.billing._account(c,org,aid,'CNY')
            changed=await self.billing._expire_grants(c,org,aid,'CNY',at,notify=False)
            rows=await (await c.execute('SELECT * FROM commerce_subscriptions WHERE organization_id=%s AND account_id=%s',(org,aid))).fetchall()
            for s in rows:
                for n in range(s['product']['months']):
                    start,end=month(s['starts_at'],n),month(s['starts_at'],n+1)
                    if start<=at<end:
                        changed=await self.grant(c,org,aid,s['order_id']+':'+str(n),'allowance',Decimal(s['product']['monthly_allowance']),start,end) or changed
            if changed:
                await self.billing._enqueue_account_sync(c,org,aid,'CNY','allowance_cycle')

    async def state(self,org,aid):
        async with self.billing.connection() as c:
            a=await self.billing._one(c,'SELECT * FROM accounts WHERE organization_id=%s AND account_id=%s',(org,aid))
        if a['currency']=='CNY' and self.development: await self.cycles(org,aid)
        summary=await self.billing.account_summary(org,aid,a['currency'])
        async with self.billing.connection() as c:
            binding=await self.billing._one(c,'SELECT merchant_id FROM commerce_account_bindings WHERE organization_id=%s AND account_id=%s',(org,aid),required=False)
            s=await self.billing._one(c,'SELECT * FROM commerce_subscriptions WHERE organization_id=%s AND account_id=%s ORDER BY starts_at DESC LIMIT 1',(org,aid),required=False)
            grants=await (await c.execute('SELECT kind,sum(remaining-held) AS amount FROM credit_grants WHERE organization_id=%s AND account_id=%s AND NOT expired GROUP BY kind',(org,aid))).fetchall()
        amounts={g['kind']:format(g['amount'],'f') for g in grants}
        now=datetime.now(timezone.utc)
        subscription=None if not s else {'plan':s['product']['plan'],'term':s['product']['term'],'status':'active' if s['ends_at']>now else 'expired','starts_at':_wire(s['starts_at']),'ends_at':_wire(s['ends_at']),'manual_renewal':True}
        return {'schema_version':'semantier.billing.v2','backend':'semantier','organization':{'id':org},'billing_account_id':aid,'currency':a['currency'],'environment':'DEVELOPMENT' if self.development else 'UNAVAILABLE','subscription':subscription,'merchant':{'id':binding['merchant_id'],'display_name':'LOCAL DEVELOPMENT — no real merchant'} if binding and self.development else None,'credits':{'allowance':{'amount':amounts.get('allowance','0'),'currency':a['currency']},'purchased':{'amount':amounts.get('purchased','0'),'currency':a['currency']},'held':summary['held'],'available':summary['available']},'usage':{'known_cost':{'amount':summary['budget']['spent'],'currency':a['currency']},'period':'lifetime'},'as_of':now.isoformat()}


def configured(billing,*,channels=None):
    development=(os.getenv('SEMANTIER_COMMERCE_ENVIRONMENT')=='DEVELOPMENT'
                 and os.getenv('SEMANTIER_ENVIRONMENT','').lower()=='development'
                 and all(os.getenv(name,'').lower() not in {'production','prod'}
                         for name in ('NODE_ENV','ENVIRONMENT')))
    if development and billing is not None and os.getenv('SEMANTIER_ALIPAY_SANDBOX_CONFIG'):
        from psycopg.conninfo import conninfo_to_dict
        if not conninfo_to_dict(billing.dsn).get('dbname','').startswith('alipay_sandbox_'):
            raise ValueError('isolated Alipay sandbox Billing store required')
    if development and channels is None:
        from billing.payment_channels import configured_channels
        channels=configured_channels()
    return CommerceService(billing,development=development,channels=channels)
