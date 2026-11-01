"""Unrelated embedded model surface; native cron HTTP access is retired.

Managed scheduling is owned by TaskAPI /api/scheduled-work. Native web cron
creation, actions, scripts and detail/output endpoints have been retired.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
router=APIRouter(tags=['webapi-gateway'])

@router.post('/api/start-hermes')
async def start_hermes():
    return JSONResponse({'ok':True,'status':'already-running',
        'message':'semantier webapi is the required backend surface on :8899.'})
