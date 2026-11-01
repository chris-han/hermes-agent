"""Scoped history reads from conversation, managed-result and effect owners."""
import asyncio
from fastapi import APIRouter, HTTPException, Request
from services.core_conversation_service import CoreConversationService, ArtifactReadLimitExceeded


def router(*, context):
    routes=APIRouter(prefix='/api/conversation-history')
    owner=CoreConversationService()

    async def read(request, ident=None, turn_id=None):
        ctx=await asyncio.to_thread(context,request)
        try:
            return await asyncio.to_thread(owner.history,ctx,ident,turn_id)
        except KeyError as exc:
            raise HTTPException(404,'Conversation or turn not found') from exc
        except Exception as exc:
            # Never substitute filesystem history or expose persistence details.
            raise HTTPException(503,'Conversation history is unavailable; refresh to retry the read') from exc

    @routes.get('')
    async def conversation_history_list(request:Request):
        return await read(request)

    @routes.get('/{ident}')
    async def conversation_history_detail(ident:str,request:Request):
        return await read(request,ident)

    @routes.get('/{ident}/turns/{turn_id}')
    async def conversation_history_result(ident:str,turn_id:str,request:Request):
        return await read(request,ident,turn_id)
    @routes.get('/{ident}/turns/{turn_id}/inspection')
    async def conversation_turn_inspection(ident:str,turn_id:str,request:Request):
        ctx=await asyncio.to_thread(context,request)
        try:return await asyncio.to_thread(owner.inspection,ctx,ident,turn_id)
        except KeyError as exc:raise HTTPException(404,'Conversation or turn not found') from exc
        except Exception as exc:raise HTTPException(503,'Execution evidence unavailable; refresh to retry') from exc

    @routes.get('/{ident}/turns/{turn_id}/artifacts/{artifact_id}')
    async def conversation_result_artifact(ident:str,turn_id:str,artifact_id:str,request:Request):
        from fastapi.responses import Response
        ctx=await asyncio.to_thread(context,request)
        if artifact_id not in ('result','stdout','stderr'):raise HTTPException(404,'Artifact not found')
        try:
            content,media_type=await asyncio.to_thread(owner.artifact_bytes,ctx,ident,turn_id,artifact_id)
            return Response(content,media_type=media_type,headers={'Cache-Control':'no-store','Content-Disposition':f'attachment; filename="retained-{artifact_id}.bin"','X-Content-Type-Options':'nosniff'})
        except KeyError as exc:raise HTTPException(404,'Artifact not found') from exc
        except ArtifactReadLimitExceeded as exc:raise HTTPException(413,'Retained artifact exceeds bounded read limit') from exc
        except Exception as exc:raise HTTPException(503,'Execution artifact unavailable') from exc
    return routes
