"""Signed platform ingress is public transport; administration and reads require auth."""
import asyncio
from fastapi import APIRouter,Request,HTTPException


def router(*,context,service):
    routes=APIRouter(prefix='/api/channels/feishu')
    async def invoke(fn,*args):
        try:return await asyncio.to_thread(fn,*args)
        except PermissionError as exc:raise HTTPException(403,'Current channel authority or signature required') from exc
        except (KeyError,StopIteration) as exc:raise HTTPException(404,'Channel unavailable') from exc
        except ValueError as exc:raise HTTPException(409,str(exc)) from exc
    @routes.get('/configuration')
    async def configuration(request:Request):
        return await invoke(service.configuration,await asyncio.to_thread(context,request))
    @routes.put('/configuration')
    async def configure(request:Request):
        ctx=await asyncio.to_thread(context,request)
        if request.headers.get('content-type','').split(';')[0]!='application/json':raise HTTPException(415,'JSON required')
        return await invoke(service.configure,ctx,await request.json())
    @routes.post('/enable')
    async def enable(request:Request):
        ctx=await asyncio.to_thread(context,request)
        if request.headers.get('content-type','').split(';')[0]!='application/json':raise HTTPException(415,'JSON required')
        if await request.json()!={}:raise HTTPException(400,'Empty enable body required')
        return await invoke(service.enable,ctx)
    @routes.post('/{account}/events')
    async def receive(account:str,request:Request):
        if request.headers.get('content-type','').split(';')[0]!='application/json':raise HTTPException(415,'JSON required')
        body=await request.body()
        headers={key:request.headers.get(key,'') for key in ('X-Lark-Request-Timestamp','X-Lark-Request-Nonce','X-Lark-Signature')}
        return await invoke(service.receive,account,body,headers)
    @routes.get('/conversations/{ident}')
    async def detail(ident:str,request:Request):
        ctx=await asyncio.to_thread(context,request)
        return await invoke(service.get,ctx,ident)
    @routes.post('/conversations/{ident}/messages/{message_id}/cancel')
    async def cancel(ident:str,message_id:str,request:Request):
        ctx=await asyncio.to_thread(context,request)
        if await request.json()!={}:raise HTTPException(400,'Empty cancellation body required')
        return await invoke(service.cancel,ctx,ident,message_id)
    return routes
