"""Restartable lease-based delivery; recovery never reissues model calls."""


async def run_once(store, deliver, *, worker_id='billing-worker', limit=100):
    from billing.commerce import configured
    commerce=configured(store)
    if commerce.development:
        await commerce.recover()
        async with store.connection() as conn:
            accounts=await (await conn.execute('SELECT organization_id,account_id FROM commerce_account_bindings ORDER BY organization_id,account_id')).fetchall()
        for account in accounts:
            await commerce.cycles(account['organization_id'],account['account_id'])
    reconciled=await store.reconcile(limit=limit)
    events=await store.claim_outbox(worker_id,limit=limit)
    delivered=0
    for event in events:
        # Consumer must deduplicate event['id']; crash after delivery/before ack
        # intentionally permits repeat delivery, not repeated financial effects.
        await deliver(event)
        await store.ack_outbox(event['id'],worker_id)
        delivered+=1
    return {'reconciled':reconciled,'delivered':delivered}
