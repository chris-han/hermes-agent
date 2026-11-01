"""Serve only the packaged management dashboard on the authenticated backend origin."""
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from agents.model_gateway.auth import actor_for
from agents.webapi.model_gateway_routes import gateway
router=APIRouter()

def packaged_file(asset):
    root=Path('/app/model-gateway-ui')
    path=(root/asset).resolve()
    if not path.is_relative_to(root) or any(p.startswith('.') for p in Path(asset).parts):
        raise HTTPException(404,'dashboard asset not found')
    if path.is_dir():path=path/'index.html'
    if not path.is_file():raise HTTPException(404,'dashboard asset not found')
    return path

@router.get('/ui/{asset:path}',include_in_schema=False)
async def dashboard(asset:str,request:Request):
    if os.environ.get('SEMANTIER_MODEL_GATEWAY_UI_ENABLED')!='1':raise HTTPException(404)
    await actor_for(request,gateway(request),'billing:manage',browser_only=True)
    return FileResponse(packaged_file(asset))

@router.get('/litellm-asset-prefix/{asset:path}',include_in_schema=False)
async def dashboard_asset(asset:str):
    if os.environ.get('SEMANTIER_MODEL_GATEWAY_UI_ENABLED')!='1':raise HTTPException(404)
    # Public immutable build assets contain no user data or server credentials.
    if not asset.startswith('_next/'):raise HTTPException(404)
    return FileResponse(packaged_file(asset))
