from __future__ import annotations

import asyncio
from pathlib import Path

from agents.hermes_embedded_gateway import get_embedded_hermes_gateway_service
from agents.internal_continuation import (
    ContinuationAdmissionStore,
    ContinuationDispatchService,
    HermesGatewayInternalContinuationProvider,
)


def schedule_admitted_internal_continuation(
    *,
    state_db_path: str | Path,
    continuation_id: str,
) -> bool:
    """Schedule a persisted continuation on the live embedded Hermes runner.

    False means no live runner accepted scheduling; the durable row remains
    PENDING and can be retried later.
    """
    store = ContinuationAdmissionStore(state_db_path)
    record = store.get(continuation_id)
    if record is None: raise KeyError(continuation_id)
    import json
    if 'api_actor_origin_hash' in json.loads(record.payload_json):
        from agents.hermes_embedded_api import get_embedded_hermes_api_service, APIInternalContinuationProvider
        from feishu_meeting_coordinator.store import MeetingCoordinatorStore
        api = get_embedded_hermes_api_service()
        loop = api._loop
        if api._adapter is None or loop is None or loop.is_closed():return False
        provider=APIInternalContinuationProvider(api,MeetingCoordinatorStore(state_db_path))
        coro=ContinuationDispatchService(store).dispatch(continuation_id=continuation_id,provider=provider)
        try: current=asyncio.get_running_loop()
        except RuntimeError:current=None
        if current is loop:
            task=loop.create_task(coro);api._continuation_tasks.add(task);task.add_done_callback(api._continuation_tasks.discard)
        else:asyncio.run_coroutine_threadsafe(coro,loop)
        return True

    service = get_embedded_hermes_gateway_service()
    runtime = getattr(service, "_shared_runtime", None)
    if runtime is None:
        return False
    runner = runtime.runner
    loop = getattr(runner, "_gateway_loop", None)
    if loop is None or loop.is_closed():
        return False

    store = ContinuationAdmissionStore(state_db_path)
    provider = HermesGatewayInternalContinuationProvider(runner)
    coro = ContinuationDispatchService(store).dispatch(
        continuation_id=continuation_id,
        provider=provider,
    )
    try:
        current = asyncio.get_running_loop()
    except RuntimeError:
        current = None
    if current is loop:
        task = loop.create_task(coro)
        background = getattr(runner, "_background_tasks", None)
        if isinstance(background, set):
            background.add(task)
            task.add_done_callback(background.discard)
        return True
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception:
        coro.close()
        return False
    return True
