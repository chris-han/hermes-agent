"""Superseded v1 entry points; retained legacy intent rows are historical only.

Current account synchronization owns desired state and scoped credentials. Legacy
user/team/key aliases are never adopted or rewritten by a login or repair.
"""
from __future__ import annotations

import asyncio


async def reconcile(proxy, *, user_id=None, limit=100):
    """Explicit compatibility result, without legacy HTTP or history mutation."""
    if not 1<=limit<=100:raise ValueError('bounded identity projection limit required')
    return {'complete':0,'pending':0,'status':'SUPERSEDED','replacement':'account-sync'}


def schedule(service, *, user_id=None):
    """Wake the current durable worker, whose claim limit and lease remain owner controlled."""
    worker=getattr(service,'account_sync_worker',None) if service is not None else None
    if worker is None:return
    task=asyncio.create_task(worker.run_once())
    tasks=getattr(service,'_identity_projection_tasks',None)
    if tasks is None:
        tasks=set();service._identity_projection_tasks=tasks
    tasks.add(task)
    def done(finished):
        tasks.discard(finished)
        if not finished.cancelled():finished.exception()
    task.add_done_callback(done)


if __name__=='__main__':
    raise SystemExit('Legacy identity projection is superseded; use explicit account-sync operator commands')
