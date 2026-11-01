"""Commerce-owned event adapter, sharing the business PostgreSQL transaction."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from psycopg.types.json import Jsonb
from agents.operations_journey import fingerprint,utc

_attribution=ContextVar('operations_commerce_actor',default=None)

@contextmanager
def commerce_actor(user_id, surface):
    token=_attribution.set({'user_id':user_id,'surface':surface})
    try:yield
    finally:_attribution.reset(token)


def attribution():return _attribution.get()


async def capture(c,record,name,*,status=None,receipt_ref=None,subject_user_id=None,properties=None):
    binding=record.get('product',{}).get('operations_attribution')
    if not binding:return False
    now=datetime.now(timezone.utc)
    receipt=receipt_ref or record.get('order_id') or record.get('quote_id')
    eid='commerce:'+fingerprint([name,receipt,status])
    event=dict(event_id=eid,schema_version='journey-event-v1',event_name=name,
               producer_id='entitlement' if name in {'entitlement_status_changed','paid_entitlement_consumed'} else 'commerce',
               producer_version='v1',surface='server' if subject_user_id else binding['surface'],subject_ref='user:'+(subject_user_id or binding['user_id']),
               occurred_at=utc(now),received_at=utc(now),evidence_type='server_fact',
               receipt_ref=receipt,status=status,clock_status='accepted',
               properties={'purchase_origin':binding['surface'],'payment_environment':record['product'].get('environment','unknown'),**(properties or {})})
    try:
        async with c.transaction():
            old=await (await c.execute('SELECT payload FROM semantier_operations.events WHERE event_id=%s',(eid,))).fetchone()
            if old:return True
            await c.execute('INSERT INTO semantier_operations.events VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                            (eid,fingerprint({k:v for k,v in event.items() if k!='received_at'}),event['subject_ref'],name,now,now,Jsonb(event)))
            await c.execute("INSERT INTO semantier_operations.receipt_counts(producer_id,kind,count) VALUES(%s,'accepted',1) ON CONFLICT(producer_id,kind) DO UPDATE SET count=semantier_operations.receipt_counts.count+1,last_at=now()",(event['producer_id'],))
        return True
    except Exception:
        import logging
        logging.getLogger(__name__).warning('Commerce observation unavailable')
        return False


async def capture_consumption(c, invocation, grant, debit):
    """Actual paid grant allocation debit, never inferred from task completion."""
    user_id=invocation.get('attribution',{}).get('actor_user_id')
    if invocation['funding']!='SEMANTIER_CREDITS' or debit<=0 or not user_id:
        return False
    try:
        async with c.transaction():
            order=await (await c.execute("SELECT o.* FROM commerce_orders o JOIN commerce_fulfillment f USING(order_id) WHERE o.order_id=%s AND o.organization_id=%s AND o.account_id=%s AND o.status='ACTIVE' AND f.fulfilled_at IS NOT NULL",(grant['grant_id'].split(':',1)[0],invocation['organization_id'],invocation['account_id']))).fetchone()
            if not order:return False
            return await capture(c,order,'paid_entitlement_consumed',status='consumed',subject_user_id=user_id,
                receipt_ref=invocation['invocation_id']+':'+grant['grant_id'],
                properties={'invocation_id':invocation['invocation_id'],'grant_id':grant['grant_id'],'order_id':order['order_id'],'amount':str(debit),'currency':invocation['currency'],'consumption_kind':'paid_credit_allocation'})
    except Exception:
        import logging
        logging.getLogger(__name__).warning('Paid credit observation unavailable')
        return False
