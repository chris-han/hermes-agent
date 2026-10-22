"""Authenticated ETL commands over the incumbent workbench owner composition."""

from contextlib import contextmanager
from threading import Event, Lock

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from pydantic import ValidationError

from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import require_route_authorization
from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
from contracts.etl_pipeline import (
    EtlCommand,
    EtlSourceRowCommand,
    EtlSourceProfileCommand,
)
from services.etl_pipeline_service import EtlPipelineService
from services.runtime_errors import RuntimeContextError

router = APIRouter()
# Ephemeral cancellation signals only; all result truth lives in retained receipts.
_running = {}
_lock = Lock()
_HEADERS = {"Cache-Control": "no-store"}


@contextmanager
def owners_for(request):
    ctx = require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
        route_hint=request.url.path,
    )
    sessions = request.headers.getlist("X-Workbench-Session-Ref")
    if len(sessions) != 1 or not sessions[0].strip():
        raise HTTPException(401, detail="AUTHORITY_CONTEXT_REQUIRED")
    owners = _workbench_owner_context(
        ctx=ctx, session_locator=sessions[0], channel="web"
    )
    try:
        yield (
            ctx,
            owners,
            EtlPipelineService(
                ctx=ctx,
                build_store=owners.build_store,
                authority_resolver=owners.authority_resolver,
            ),
        )
    finally:
        try:
            owners.build_store.close()
        finally:
            owners.source_registry.close()


def failure(exc):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            {"code": str(exc.detail)}, status_code=exc.status_code, headers=_HEADERS
        )
    if isinstance(exc, RuntimeContextError):
        return JSONResponse(
            {"code": "ACCESS_UNAVAILABLE"}, status_code=403, headers=_HEADERS
        )
    if isinstance(exc, (ValidationError, ValueError)):
        code = (
            str(exc)
            if isinstance(exc, ValueError) and not isinstance(exc, ValidationError)
            else "INVALID_ETL_COMMAND"
        )
        # Never reflect arbitrary parser values, file paths or validation input.
        if not code.replace("_", "").isalnum() or code.upper() != code:
            code = "INVALID_ETL_COMMAND"
        status = (
            409
            if code.startswith("STALE_")
            else 404
            if code
            in {
                "ETL_RECEIPT_UNAVAILABLE",
                "SOURCE_UNAVAILABLE",
                "ARTIFACT_NOT_FOUND",
                "SAVED_RECIPE_REQUIRED",
            }
            else 400
        )
        return JSONResponse({"code": code}, status_code=status, headers=_HEADERS)
    if isinstance(exc, (OSError, PermissionError)):
        return JSONResponse(
            {"code": "SOURCE_UNAVAILABLE"}, status_code=404, headers=_HEADERS
        )
    raise exc


@router.post("/api/etl/evaluate")
async def evaluate_recipe(request: Request):
    try:
        command = EtlCommand.model_validate(await request.json())

        def execute():
            # Compose, use and close SQLite owners in the same worker thread.
            with owners_for(request) as (ctx, owners, service):
                key = (
                    ctx.organization_id,
                    ctx.workspace_id,
                    ctx.user_id,
                    owners.session_ref,
                    command.request_id,
                )
                event = Event()
                with _lock:
                    if key in _running:
                        raise ValueError("ETL_REQUEST_RUNNING")
                    _running[key] = event
                try:
                    return service.evaluate(command, cancelled=event.is_set)
                finally:
                    with _lock:
                        _running.pop(key, None)

        return JSONResponse(await run_in_threadpool(execute), headers=_HEADERS)
    except (HTTPException, RuntimeContextError, ValueError, OSError) as exc:
        return failure(exc)


@router.post("/api/etl/cancel")
async def cancel_recipe(request: Request):
    try:
        command = EtlCommand.model_validate(await request.json())
        with owners_for(request) as (ctx, owners, service):
            service.access()
            key = (
                ctx.organization_id,
                ctx.workspace_id,
                ctx.user_id,
                owners.session_ref,
                command.request_id,
            )
            with _lock:
                event = _running.get(key)
                if event is not None:
                    event.set()
            return JSONResponse(
                {"cancel_requested": event is not None}, headers=_HEADERS
            )
    except (HTTPException, RuntimeContextError, ValueError, OSError) as exc:
        return failure(exc)


@router.get("/api/etl/runs")
def list_runs(request: Request):
    try:
        with owners_for(request) as (_, __, service):
            return JSONResponse({"runs": service.list_runs()}, headers=_HEADERS)
    except (HTTPException, RuntimeContextError, ValueError, OSError) as exc:
        return failure(exc)


@router.get("/api/etl/runs/{receipt_ref}")
def read_run(request: Request, receipt_ref: str):
    try:
        with owners_for(request) as (_, __, service):
            receipt, current = service.read(receipt_ref)
            from services.etl_pipeline_service import read_saved_recipe

            definition = read_saved_recipe(
                service.ctx, receipt.workflow_hash, current=False
            )["definition"]
            return JSONResponse(
                {
                    **receipt.model_dump(mode="json"),
                    "is_current": current,
                    "definition": definition,
                },
                headers=_HEADERS,
            )
    except (HTTPException, RuntimeContextError, ValueError, OSError) as exc:
        return failure(exc)


@router.get("/api/etl/runs/{receipt_ref}/exports/{filename}")
def export_run(request: Request, receipt_ref: str, filename: str):
    try:
        with owners_for(request) as (_, __, service):
            receipt, current = service.read(receipt_ref)
            if (
                filename
                not in {"result.csv", "exceptions.csv", "manifest.json", "pipeline.py", "pipeline_polars.py", "requirements.txt", "code-map.json"}
                or filename not in receipt.exports
            ):
                raise ValueError("ETL_EXPORT_UNAVAILABLE")
            return Response(
                receipt.exports[filename],
                media_type="text/plain; charset=utf-8",
                headers={
                    **_HEADERS,
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "X-ETL-Result-Basis": "current" if current else "historical",
                },
            )
    except (HTTPException, RuntimeContextError, ValueError, OSError) as exc:
        return failure(exc)


@router.post("/api/etl/source-row")
async def source_row(request: Request):
    try:
        command = EtlSourceRowCommand.model_validate(await request.json())

        def inspect():
            with owners_for(request) as (_, __, service):
                return service.source_row(command)

        return JSONResponse(await run_in_threadpool(inspect), headers=_HEADERS)
    except (HTTPException, RuntimeContextError, ValueError, OSError) as exc:
        return failure(exc)


@router.post("/api/etl/source-profile")
async def source_profile(request: Request):
    try:
        command = EtlSourceProfileCommand.model_validate(await request.json())

        def inspect():
            with owners_for(request) as (_, __, service):
                return service.source_profile(command)

        return JSONResponse(await run_in_threadpool(inspect), headers=_HEADERS)
    except (HTTPException, RuntimeContextError, ValueError, OSError) as exc:
        return failure(exc)
