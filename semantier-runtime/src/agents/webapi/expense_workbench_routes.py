"""Authenticated expense teaching task; no reimbursement submission operation."""
from contextlib import contextmanager
from starlette.concurrency import run_in_threadpool
from pathlib import Path
from datetime import date
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, ConfigDict
from agents.webapi.jd_review_routes import _ctx, _session
from agents.webapi.tender_review_routes import _command, _query
from agents.webapi.ontology_workbench_context_routes import _workbench_owner_context, _temporal_upload_boundary
from contracts.artifact_pin import ArtifactPinV1
from contracts.expense_reimbursement import ApplicantFields
from plugins.document_extraction.storage import BoundSession
from services.workspace_artifact_service import WorkspaceArtifactService
from services.expense_teaching_policy_service import ExpenseTeachingPolicyService
from services.expense_workbench_service import ExpenseWorkbenchService
from storage.workbench_projection_cursor_store import WorkbenchProjectionCursorStore

router = APIRouter(tags=['expense-workbench'])


class ResultCommand(BaseModel):
    model_config = ConfigDict(extra='forbid')
    result_ref: str
    result_hash: str


class EvaluateCommand(BaseModel):
    model_config = ConfigDict(extra='forbid')
    artifact_pin: ArtifactPinV1
    submitted_on: date
    applicant_fields: ApplicantFields
    predecessor_result: ResultCommand | None = None


@contextmanager
def owner(request):
    ctx, session = _ctx(request), _session(request)
    owners = None
    try:
        owners = _workbench_owner_context(ctx=ctx, session_locator=session, channel='web', authorization_class='WORKSPACE_LOCAL')
        boundary, _ = _temporal_upload_boundary(ctx=ctx, session_ref=owners.session_ref)
        bound = BoundSession(Path(boundary.paths.hermes_home), owners.session_ref, boundary.paths.uploads_root.parent)
        cursors = WorkbenchProjectionCursorStore(build_store=owners.build_store,
            authority_resolver=owners.authority_resolver, session_resolver=owners.session_resolver,
            session_ref=owners.session_ref)
        artifacts = WorkspaceArtifactService(build_store=owners.build_store,
            authority_resolver=owners.authority_resolver, max_bytes=20 * 1024 * 1024, cursor_store=cursors)
        artifacts._access(ctx)
        yield ExpenseWorkbenchService(ctx=ctx, bound=bound, artifacts=artifacts,
            policy_owner=ExpenseTeachingPolicyService(ctx=ctx, store=owners.build_store, source_registry=owners.source_registry))
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(403, 'FORBIDDEN_SCOPE') from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc
    finally:
        if owners:
            owners.build_store.close()
            owners.source_registry.close()


@router.get('/api/expense-workbench/context')
def context(request: Request):
    _query(request, ())
    with owner(request) as service:
        return service.context()


@router.post('/api/expense-workbench/evaluations')
async def evaluate(request: Request):
    command = await _command(request, EvaluateCommand)
    # Create/use/close SQLite owners in one worker; PDF extraction must not block
    # the ASGI event loop while the other classroom accounts log in or read files.
    def evaluate_in_worker():
        with owner(request) as service:
            return service.evaluate(artifact_pin=command.artifact_pin, submitted_on=command.submitted_on.isoformat(),
                                    applicant_fields=command.applicant_fields.model_dump(),
                                    predecessor_result=command.predecessor_result.model_dump() if command.predecessor_result else None)
    return await run_in_threadpool(evaluate_in_worker)


@router.post('/api/expense-workbench/drafts')
async def save_draft(request: Request):
    command = await _command(request, ResultCommand)
    def execute_in_worker():
        with owner(request) as service:
            return service.business.save_draft(command.result_ref, command.result_hash)
    return await run_in_threadpool(execute_in_worker)


@router.post('/api/expense-workbench/replay')
async def replay(request: Request):
    command = await _command(request, ResultCommand)
    def execute_in_worker():
        with owner(request) as service:
            return service.business.replay(command.result_ref, command.result_hash)
    return await run_in_threadpool(execute_in_worker)


@router.post('/api/expense-workbench/inferences')
async def retain_inference(request: Request):
    command = await _command(request, ResultCommand)
    def execute_in_worker():
        with owner(request) as service:
            return service.retain_inference(**command.model_dump())
    return await run_in_threadpool(execute_in_worker)


class PolicySourceCommand(BaseModel):
    model_config = ConfigDict(extra='forbid')
    parent_ref: str
    parent_hash: str
    markdown: str
    source_hash: str


class PolicyPinCommand(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_ref: str
    source_hash: str


class RebuildCommand(BaseModel):
    model_config = ConfigDict(extra='forbid')
    event_ref: str
    event_hash: str


def policy_revision_owner(service):
    from services.policy_revision_service import PolicyRevisionService
    policy = service.business.policy_owner
    return PolicyRevisionService(ctx=service.ctx, store=policy.store, policy_owner=policy)


@router.get('/api/expense-workbench/policy')
def policy_context(request: Request):
    _query(request, ())
    with owner(request) as service:
        return policy_revision_owner(service).context()


@router.post('/api/expense-workbench/policy/drafts')
async def policy_draft(request: Request):
    command = await _command(request, PolicySourceCommand)
    def execute():
        with owner(request) as service:
            return policy_revision_owner(service).save_draft(**command.model_dump())
    return await run_in_threadpool(execute)


@router.post('/api/expense-workbench/policy/publications')
async def policy_publication(request: Request):
    command = await _command(request, PolicySourceCommand)
    def execute():
        with owner(request) as service:
            return policy_revision_owner(service).publish(**command.model_dump())
    return await run_in_threadpool(execute)


@router.post('/api/expense-workbench/policy/rebuilds')
async def policy_rebuild(request: Request):
    command = await _command(request, RebuildCommand)
    def execute():
        with owner(request) as service:
            return policy_revision_owner(service).rebuild(**command.model_dump())
    return await run_in_threadpool(execute)


@router.post('/api/expense-workbench/policy/activate')
async def policy_activate(request: Request):
    await _command(request, PolicyPinCommand)
    with owner(request) as service:
        policy_revision_owner(service)._access()
        raise HTTPException(403, 'DEVELOPMENT_EVALUATION_CANNOT_GRANT_ACTIVATION')


class PolicyGraphEvalCommand(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_ref: str
    source_hash: str
    build_ref: str
    build_hash: str
    reference_complete: bool = True


class ConsumerCandidateCommand(BaseModel):
    model_config = ConfigDict(extra='forbid')
    parent_ref: str
    parent_hash: str
    binding_ids: list[str]


class ConsumerRunCommand(BaseModel):
    model_config = ConfigDict(extra='forbid')
    candidate_ref: str
    candidate_hash: str


def consumer_owner(service):
    from services.source_consumer_product_service import SourceConsumerProductService
    return SourceConsumerProductService(service)


@router.get('/api/expense-workbench/policy/consumer-context')
def consumer_context(request: Request):
    _query(request, ())
    with owner(request) as service:
        return consumer_owner(service).context()


@router.post('/api/expense-workbench/policy/consumer-candidates')
async def consumer_candidate(request: Request):
    command = await _command(request, ConsumerCandidateCommand)
    with owner(request) as service:
        return consumer_owner(service).successor(**command.model_dump())


@router.post('/api/expense-workbench/policy/consumer-runs')
async def consumer_execute(request: Request):
    command = await _command(request, ConsumerRunCommand)
    async def invoke(body):
        import json
        scope = {**request.scope, 'path':'/api/expense-workbench/evaluations', 'query_string':b''}
        async def receive():
            return {'type':'http.request','body':json.dumps(body).encode(),'more_body':False}
        child = Request(scope, receive)
        try:
            return 200, await evaluate(child)
        except HTTPException as exc:
            return exc.status_code, {'detail':exc.detail}
    with owner(request) as service:
        return await consumer_owner(service).execute(**command.model_dump(), invoke=invoke)


@router.get('/api/expense-workbench/policy/consumer-run')
def consumer_read(request: Request):
    _query(request, ('ref','hash'))
    with owner(request) as service:
        return consumer_owner(service).read_run(request.query_params['ref'],request.query_params['hash'])
