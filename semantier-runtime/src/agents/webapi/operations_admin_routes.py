"""Read-only platform operations. Every request rechecks DB-backed authority."""
from __future__ import annotations
import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from agents.webapi.deps import require_authenticated_context
from agents.operations_admin import is_operations_admin, operations_snapshot
from agents.gateway_identity import get_user_record

router = APIRouter(tags=["platform-operations"])

async def reserve_operations_login(request: Request):
    """Public signup cannot claim the bootstrap name before provisioning."""
    if request.method != "POST" or request.url.path.rstrip("/") != "/auth/password/register":
        return
    try:
        body = await request.json()
    except ValueError:
        return
    from agents.operations_admin import OPS_LOGIN
    if isinstance(body, dict) and str(body.get("login") or "").strip().lower() == OPS_LOGIN:
        raise HTTPException(400, "Reserved operations login")


@router.get("/api/admin/operations")
async def get_operations(request: Request, snapshot: str | None = None, definition_version: str | None = None, traffic: str = "REAL", branch: str = "all", source: str = "all", role: str = "all", scenario: str = "all", page_version: str = "all", has_current_task: str = "all", cohort_from: str | None = None, cohort_to: str | None = None, check_access: bool = False, view: str = "funnels", analysis_ref: str = "acquisition", edge_ref: str | None = None, comparison_dimension: str = "branch", group: str | None = None, journey_ref: str | None = None, cursor: int = 0, sort: str = "newest", limit: int = 25):
    if traffic not in {"REAL", "TEST"}:
        raise HTTPException(422, "Unsupported traffic scope")
    ctx = await asyncio.to_thread(require_authenticated_context, request)
    user = await asyncio.to_thread(get_user_record, ctx.user_id)
    if not is_operations_admin(user):
        raise HTTPException(403, "Platform operations administrator required")
    if check_access:
        return JSONResponse({"allowed": True}, headers={"Cache-Control": "no-store", "Vary": "Cookie"})
    from agents.operations_journey import SAFE_ID, stamp, utc
    from datetime import datetime, timedelta, timezone
    if definition_version is not None and definition_version not in {'metric-definition-v1','metric-definition-v2'}:raise HTTPException(422,'Unsupported definition')
    scope={"traffic":traffic,"branch":branch,"source":source,"role":role,"scenario":scenario,"page_version":page_version,"has_current_task":has_current_task}
    if any(not SAFE_ID.fullmatch(v) for v in scope.values()) or branch not in {"all","workspace","windows","both","unknown"}:
        raise HTTPException(422,"Unsupported scope")
    if bool(cohort_from)!=bool(cohort_to):raise HTTPException(422,"Both cohort boundaries required")
    if not cohort_from:
        from agents.operations_journey import CATALOG
        end=datetime.now(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=1)
        cohort_from=utc(end-timedelta(days=CATALOG['cohort_days']));cohort_to=utc(end)
    if cohort_from:
        try:
            start,end=stamp(cohort_from),stamp(cohort_to)
            if start>=end or end-start>timedelta(days=366):raise ValueError()
            scope.update(cohort_from=utc(start),cohort_to=utc(end))
        except ValueError:raise HTTPException(422,"Invalid UTC cohort interval") from None
    from agents import auth_db
    if auth_db.is_postgresql():
        from agents.operations_store import snapshot as create_snapshot
        try:
            result = await asyncio.to_thread(create_snapshot, scope=scope, snapshot_id=snapshot, definition_version=definition_version)
            from agents.operations_journey import snapshot_page
            result=snapshot_page(result,view=view,analysis=analysis_ref,edge=edge_ref,dimension=comparison_dimension,group=group,journey=journey_ref,cursor=cursor,sort=sort,limit=limit)
        except LookupError:
            raise HTTPException(410, "Snapshot unavailable; refresh explicitly") from None
        except ValueError:
            raise HTTPException(409, "Snapshot scope mismatch") from None
    else:
        # Explicit local unit-fixture path; the bootstrap CLI disallows SQLite.
        result = await asyncio.to_thread(operations_snapshot)
    # A long snapshot query must not publish after a role was revoked in flight.
    await asyncio.to_thread(require_authenticated_context, request)
    current = await asyncio.to_thread(get_user_record, ctx.user_id)
    if not is_operations_admin(current):
        raise HTTPException(403, "Platform operations administrator required")
    return JSONResponse(result, headers={"Cache-Control": "no-store", "Vary": "Cookie"})
