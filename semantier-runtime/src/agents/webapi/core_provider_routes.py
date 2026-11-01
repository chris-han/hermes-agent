"""Scoped configured-state readers; discovery and execution remain separate facts."""
import asyncio
from fastapi import APIRouter,HTTPException,Request
from agents.model_gateway.selection import task_provider_configuration


def router(*,context):
    routes=APIRouter()
    async def read(request):
        ctx=await asyncio.to_thread(context,request)
        try:return await asyncio.to_thread(task_provider_configuration,ctx)
        except HTTPException:raise
        except Exception as exc:raise HTTPException(503,'Provider configuration unavailable') from exc

    @routes.get('/api/task-provider')
    async def task_provider_read(request:Request):return await read(request)

    @routes.patch('/api/task-provider')
    async def task_provider_update(request:Request):
        ctx=await asyncio.to_thread(context,request)
        from agents.model_gateway.auth import csrf
        csrf(request)
        body=await request.json()
        try:return await asyncio.to_thread(task_provider_configuration,ctx,body)
        except HTTPException:raise
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc
        except Exception as exc:raise HTTPException(503,'Provider configuration unavailable; read before retrying') from exc

    @routes.get('/api/model/info')
    async def core_model_info(request:Request):
        import os
        if os.environ.get('SEMANTIER_RUNTIME_MODE')=='tasks':
            value=await read(request)
            route=value['route'] or {}
            return {**value,'provider':route.get('provider'),'model':route.get('model'),
                'supportsRuntimeSwitching':False,'vanillaAgent':False}
        from agents.auth_db import get_workspace_config
        ctx=await asyncio.to_thread(context,request)
        try:config=await asyncio.to_thread(get_workspace_config,ctx.workspace_id)
        except Exception as exc:raise HTTPException(503,'Provider configuration unavailable') from exc
        configured=(config or {}).get('model')
        model=configured.get('default') if isinstance(configured,dict) else configured if isinstance(configured,str) else None
        provider=configured.get('provider') if isinstance(configured,dict) else (config or {}).get('provider')
        return {'source':'core_workspace_configuration','provider':provider,'model':model,
            'configured':bool(model and provider),'availability':'UNVERIFIED','verified_execution':False,
            'supportsRuntimeSwitching':None,'vanillaAgent':None}
    return routes
