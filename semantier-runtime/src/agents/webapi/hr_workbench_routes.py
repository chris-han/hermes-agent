"""Scoped auxiliary HR task operations over existing upload/review/store owners."""

from contextlib import contextmanager
from starlette.concurrency import run_in_threadpool
from fastapi import APIRouter, Request, Response, HTTPException
from agents.webapi.jd_review_routes import review_service, _ctx, _session
from agents.webapi.tender_review_routes import _command, _query
from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context
from services.workspace_artifact_service import WorkspaceArtifactService
from services.hr_workbench import HrWorkbenchService
from storage.workbench_projection_cursor_store import WorkbenchProjectionCursorStore
from contracts.hr_workbench import (
    HrImportCommand,
    HrModelCommand,
    HrRunCommand,
    HrRemediateCommand,
    HrInterpretationCommand,
    HrInterpretationEvidenceCommand,
)
from contracts.artifact_pin import ArtifactPinV1
from eos.artifact_hashing import canonical_json

router = APIRouter(tags=["hr-workbench"])


@contextmanager
def owner(request):
    ctx = _ctx(request)
    session = _session(request)
    owners = None
    try:
        owners = _workbench_owner_context(
            ctx=ctx, session_locator=session, channel="web", authorization_class="WORKSPACE_LOCAL"
        )
        cursors = WorkbenchProjectionCursorStore(
            build_store=owners.build_store,
            authority_resolver=owners.authority_resolver,
            session_resolver=owners.session_resolver,
            session_ref=owners.session_ref,
        )
        artifacts = WorkspaceArtifactService(
            build_store=owners.build_store,
            authority_resolver=owners.authority_resolver,
            max_bytes=20 * 1024 * 1024,
            cursor_store=cursors,
        )
        artifacts._access(ctx)
        with review_service(ctx, session) as (review, _, __):
            yield HrWorkbenchService(
                review_service=review, artifact_service=artifacts, request_context=ctx
            )
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(403, "FORBIDDEN_SCOPE") from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc
    finally:
        if owners:
            owners.build_store.close()
            owners.source_registry.close()


@router.get("/api/hr-workbench/context")
def context(request: Request):
    _query(request, ())
    with owner(request) as service:
        return service.context()


@router.post("/api/hr-workbench/sources")
async def import_source(request: Request):
    # Authenticate before body consumption or source resolution.
    with owner(request) as service:
        command = await _command(request, HrImportCommand)
        return service.import_artifact(
            **command.model_dump(exclude={"artifact_pin"}),
            artifact_pin=command.artifact_pin,
        ).model_dump(mode="json")


@router.post("/api/hr-workbench/models")
async def model(request: Request):
    with owner(request) as service:
        command = await _command(request, HrModelCommand)
        return service.model(
            command.review_id,
            job_ref=command.job_ref,
            predecessor_pin=command.predecessor_pin,
        ).model_dump(mode="json")


@router.post("/api/hr-workbench/runs")
async def start(request: Request):
    # Authenticate/body-validate first, then create producer-owned connections
    # inside the worker. Stop/status requests remain serviceable during compute.
    with owner(request):
        command = await _command(request, HrRunCommand)
    def execute():
        with owner(request) as service:
            return service.start(**dict(command)).model_dump(mode="json")
    return await run_in_threadpool(execute)


@router.get("/api/hr-workbench/runs/{run_ref}")
def read_run(request: Request, run_ref: str):
    query = _query(request, ("hash",), ("hash",))
    with owner(request) as service:
        return service.get_run(
            ArtifactPinV1(
                artifact_kind="hr_screening_run",
                artifact_ref=run_ref,
                artifact_hash=query["hash"],
            )
        ).model_dump(mode="json")


@router.get("/api/hr-workbench/runs/{run_ref}/export")
def export_run(request: Request, run_ref: str):
    query = _query(request, ("hash",), ("hash",))
    with owner(request) as service:
        run = service.get_run(
            ArtifactPinV1(
                artifact_kind="hr_screening_run",
                artifact_ref=run_ref,
                artifact_hash=query["hash"],
            )
        )
        return Response(
            canonical_json(run.model_dump(mode="json")),
            media_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="screening-evidence.json"',
                "Cache-Control": "no-store",
            },
        )


@router.post("/api/hr-workbench/remediation")
async def remediate(request: Request):
    with owner(request) as service:
        command = await _command(request, HrRemediateCommand)
        return service.remediate(
            model_pin=command.model_pin, request_id=command.request_id
        ).model_dump(mode="json")


@router.post("/api/hr-workbench/interpretations")
async def interpretation(request: Request):
    with owner(request) as service:
        command = await _command(request, HrInterpretationCommand)
        return service.interpretation(**dict(command)).model_dump(mode="json")


@router.post("/api/hr-workbench/interpretation-evidence")
async def interpretation_evidence(request: Request):
    with owner(request) as service:
        command = await _command(request, HrInterpretationEvidenceCommand)
        return service.interpretation_evidence(**dict(command))


@router.get("/api/hr-workbench/operations/{request_id}")
def operation_status(request: Request, request_id: str):
    _query(request, ())
    with owner(request) as service:
        return service.operation_status(request_id).model_dump(mode="json")


@router.post("/api/hr-workbench/operations/{request_id}/stop")
async def stop(request: Request, request_id: str):
    with owner(request):
        command = await _command(request, HrRunCommand)
        if command.request_id != request_id:
            raise HTTPException(409, "REQUEST_CONFLICT")
    def execute():
        with owner(request) as service:
            return service.stop(**dict(command)).model_dump(mode="json")
    return await run_in_threadpool(execute)
