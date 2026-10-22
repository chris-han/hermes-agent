"""Own a worker lease through response completion, disconnect or deadline."""
import asyncio
from functools import wraps
import json
import os
from fastapi import HTTPException
from starlette.background import BackgroundTasks
from agents.chat_execution_pool import ChatQueueFull
from starlette.responses import StreamingResponse


def isolated_chat_turn(function):
    @wraps(function)
    async def execute(**kwargs):
        # Explicit rollout switch; protocol-only tests can exercise the old transport.
        if os.environ.get('SEMANTIER_CHAT_PROCESS_ISOLATION','false').lower() not in {'1','true','yes'}:
            return await function(**kwargs)
        from agents.hermes_embedded_api import get_embedded_hermes_api_service
        owner = get_embedded_hermes_api_service()
        await owner.ensure_started()
        pool = owner.chat_execution_pool()
        try:
            lease = await pool.acquire(kwargs.get('workspace_id'),kwargs['session_id'])
        except (TimeoutError, ChatQueueFull) as exc:
            raise HTTPException(503,str(exc) if isinstance(exc,ChatQueueFull) else 'CHAT_EXECUTION_QUEUE_TIMEOUT') from exc
        try:
            async with asyncio.timeout(lease.remaining):
                response = await function(**kwargs,_execution_service=lease)
        except BaseException as exc:
            await asyncio.shield(lease.close())
            if isinstance(exc,TimeoutError):
                raise HTTPException(504,'CHAT_TURN_TIMEOUT') from exc
            raise
        if not isinstance(response,StreamingResponse):
            await lease.close()
            return response
        original = response.body_iterator
        async def stream():
            try:
                async with asyncio.timeout(lease.remaining):
                    async for chunk in original:
                        yield chunk
            except TimeoutError:
                payload=json.dumps({'error':{'code':'CHAT_TURN_TIMEOUT','message':'Chat execution exceeded its deadline.'},'choices':[{'index':0,'delta':{},'finish_reason':'error'}]})
                yield f'event: error\ndata: {payload}\n\n'.encode()
            finally:
                try:
                    if hasattr(original,'aclose'):
                        await original.aclose()
                finally:
                    await asyncio.shield(lease.close())
        response.body_iterator = stream()
        background = BackgroundTasks()
        background.add_task(lease.close)
        if response.background is not None:
            background.add_task(response.background)
        response.background = background
        return response
    return execute
