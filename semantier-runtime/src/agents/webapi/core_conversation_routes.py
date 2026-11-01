"""Authenticated core Chat routes; no embedded/native session adapter imports."""
import asyncio
import json
from fastapi import APIRouter,Request,HTTPException
from fastapi.responses import StreamingResponse,JSONResponse
from agents import job_store
from services.core_conversation_service import CoreConversationService
from services.core_scheduled_work_service import live_context
from agents.durable_responsibility import DurableResponsibilityError


def router(*, context, execution):
    routes=APIRouter(prefix='/api/conversations')
    owner=CoreConversationService()
    def invoke(fn,*args):
        try:return fn(*args)
        except (KeyError,StopIteration) as exc:raise HTTPException(404,'Conversation or turn not found') from exc
        except (ValueError,DurableResponsibilityError) as exc:raise HTTPException(409,str(exc)) from exc
        except PermissionError as exc:raise HTTPException(403,'Current workspace authority required') from exc
    async def body(request):
        if request.headers.get('content-type','').split(';')[0].strip()!='application/json':
            raise HTTPException(415,'JSON content type required')
        return await request.json()
    @routes.get('')
    async def listing(request:Request):
        ctx=await asyncio.to_thread(context,request)
        return await asyncio.to_thread(invoke,owner.list,ctx)
    @routes.post('')
    async def create(request:Request):
        ctx=await asyncio.to_thread(context,request);data=await body(request)
        return await asyncio.to_thread(invoke,owner.create,ctx,data)
    @routes.get('/{ident}')
    async def detail(ident:str,request:Request):
        ctx=await asyncio.to_thread(context,request)
        return await asyncio.to_thread(invoke,owner.get,ctx,ident)
    @routes.post('/{ident}/turns')
    async def send(ident:str,request:Request):
        ctx=await asyncio.to_thread(context,request);data=await body(request)
        current=await asyncio.to_thread(invoke,owner.accept,ctx,ident,data)
        turn=next(t for t in current['turns'] if t['id']==data['request_id'])
        if turn['state']=='accepted':
            work_id=await asyncio.to_thread(invoke,owner.prepare,ctx,ident,turn['id'])
            scope={k:getattr(ctx,k) for k in ('user_id','workspace_id','workspace_slug','organization_id')}
            try:
                await asyncio.to_thread(execution.submit,ctx,work_id,resolve_context=lambda:live_context(scope))
            except (ValueError,DurableResponsibilityError) as exc:raise HTTPException(409,str(exc)) from exc
        return JSONResponse(status_code=202,content=await asyncio.to_thread(invoke,owner.get,ctx,ident))
    @routes.post('/{ident}/turns/{turn_id}/cancel')
    async def cancel(ident:str,turn_id:str,request:Request):
        await body(request)
        ctx=await asyncio.to_thread(context,request)
        return await asyncio.to_thread(invoke,owner.cancel,ctx,ident,turn_id,execution)
    @routes.get('/{ident}/events')
    async def events(ident:str,request:Request):
        ctx=await asyncio.to_thread(context,request)
        await asyncio.to_thread(invoke,owner.get,ctx,ident)
        async def stream():
            cursor=0;work_id=None
            while not await request.is_disconnected():
                current_ctx=await asyncio.to_thread(context,request)
                item=await asyncio.to_thread(invoke,owner.get,current_ctx,ident)
                if not item['turns']:
                    yield 'event: result\ndata: '+json.dumps(item)+'\n\n';return
                turn=item['turns'][-1]
                if work_id!=turn['work_id']:work_id=turn['work_id'];cursor=0
                row=await asyncio.to_thread(job_store.get_job,user_id=current_ctx.user_id,workspace_id=current_ctx.workspace_id,job_id=work_id)
                pending=(row or {}).get('progress_events',[])
                for event in pending[cursor:]:
                    yield 'event: progress\ndata: '+json.dumps(event)+'\n\n';cursor+=1
                if turn['state'] in ('completed','cancelled','uncertain','failed'):
                    yield 'event: result\ndata: '+json.dumps(item)+'\n\n';return
                await asyncio.sleep(.1)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-store'})
    return routes
