"""Authenticated personal learning reads and typed owner-native actions."""
from contextlib import contextmanager
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import require_route_authorization
from eos.db import init_db
from runtime_paths import workspace_hermes_home_path, workspace_root_path
from services.learning_workbench_service import LearningWorkbenchService
from storage.governance_store import GovernanceStore, _db_path
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path

router = APIRouter(tags=["learning-workbench"])


class LearningOrigin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_key: str = Field(min_length=1)
    message_id: str = Field(min_length=1)


class LearningAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["correct_context", "suppress_context", "propose_procedure", "assess_candidate",
                       "approve_candidate", "activate_candidate", "publish_procedure", "recover_procedure",
                       "import_pending", "reject_candidate", "mark_notice_read", "defer_candidate",
                       "preview_organization_contribution", "submit_organization_contribution",
                       "approve_organization_contribution", "reject_organization_contribution"]
    request_id: str = Field(min_length=1, max_length=200)
    origin: LearningOrigin | None = None
    item_ref: str | None = None
    expected_revision: str | None = None
    new_content: str | None = None
    baseline_ref: str | None = None
    baseline_hash: str | None = None
    candidate_ref: str | None = None
    candidate_hash: str | None = None
    assessment_ref: str | None = None
    assessment_hash: str | None = None
    version_ref: str | None = None
    version_hash: str | None = None
    native_id: str | None = None
    payload_hash: str | None = None
    notice_id: str | None = None
    destination_organization_id: str | None = None
    disclosure_hash: str | None = None
    consent: bool | None = None

    @model_validator(mode="after")
    def exact_action_inputs(self):
        required = {
            "correct_context": ("item_ref", "expected_revision", "new_content"),
            "suppress_context": ("item_ref", "expected_revision"),
            "propose_procedure": ("baseline_ref", "baseline_hash", "new_content"),
            "assess_candidate": ("candidate_ref", "candidate_hash"),
            "approve_candidate": ("candidate_ref", "candidate_hash", "assessment_ref", "assessment_hash"),
            "activate_candidate": ("candidate_ref", "candidate_hash"),
            "publish_procedure": ("candidate_ref", "candidate_hash"),
            "recover_procedure": ("version_ref", "version_hash"),
            "import_pending": ("native_id", "payload_hash"),
            "reject_candidate": ("candidate_ref", "candidate_hash"),
            "mark_notice_read": ("candidate_ref", "candidate_hash"),
            "defer_candidate": ("candidate_ref", "candidate_hash"),
            "preview_organization_contribution": ("candidate_ref", "candidate_hash", "destination_organization_id"),
            "submit_organization_contribution": ("candidate_ref", "candidate_hash", "destination_organization_id", "disclosure_hash", "consent"),
            "approve_organization_contribution": ("candidate_ref", "candidate_hash"),
            "reject_organization_contribution": ("candidate_ref", "candidate_hash"),
        }[self.operation]
        if any(not getattr(self, name) for name in required):
            raise ValueError("Action requires its exact scoped object and revision pins")
        return self


@contextmanager
def _owners(request: Request, route: str, *, personal_read: bool = False):
    ctx = require_route_authorization(request, authorization_class=(RouteAuthorizationClass.SELF_ONLY
                                      if personal_read else RouteAuthorizationClass.TENANT_MEMBER),
                                      route_hint=route)
    if (ctx.workspace_root.resolve() != workspace_root_path(ctx.workspace_id).resolve()
            or ctx.hermes_home.resolve() != workspace_hermes_home_path(ctx.workspace_id).resolve()):
        raise HTTPException(403, "Outside current learning scope")
    init_db(_db_path())
    store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
    governance = GovernanceStore()
    try:
        yield ctx, LearningWorkbenchService(store=store, governance=governance)
    except PermissionError as exc:
        raise HTTPException(403, "Outside current learning scope") from exc
    except LookupError as exc:
        raise HTTPException(404, "Learning object not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, "Runtime materialization unavailable; retained state is preserved") from exc
    finally:
        governance.close()
        store.close()


@router.get("/api/learning")
def learning_inventory(request: Request, limit: int = Query(default=100, ge=1, le=200),
                       cursor: str | None = Query(default=None, max_length=4096)):
    with _owners(request, "/api/learning", personal_read=True) as (context, service):
        return service.inventory(context=context, limit=limit, cursor=cursor)


@router.get("/api/learning/inspect")
def inspect_learning(request: Request, kind: Literal["result", "context", "procedure", "candidate", "pending", "contribution"] = Query(),
                     ref: str = "", session_key: str | None = None, message_id: str | None = None,
                     payload_hash: str | None = None):
    with _owners(request, "/api/learning/inspect", personal_read=kind in ("result", "pending")) as (context, service):
        return service.inspect(context=context, kind=kind, ref=ref, session_key=session_key,
                               message_id=message_id, payload_hash=payload_hash)


@router.get("/api/learning/context-snapshot")
def learning_context_snapshot(request: Request, session_key: str = Query(min_length=1)):
    with _owners(request, "/api/learning/context-snapshot") as (context, service):
        return service.context_snapshot(context=context, session_key=session_key)


@router.post("/api/learning/action")
def learning_action(body: LearningAction, request: Request):
    with _owners(request, "/api/learning/action", personal_read=body.operation in ('mark_notice_read', 'defer_candidate')) as (context, service):
        return service.act(context=context, operation=body.operation,
                           values=body.model_dump(exclude={"operation"}, exclude_none=True))


class LearningSlashCommand(BaseModel):
    model_config = ConfigDict(extra='forbid')
    message: str = Field(min_length=1, max_length=4096)
    request_id: str | None = Field(default=None, max_length=200)


@router.post('/api/learning/slash')
def learning_slash(body: LearningSlashCommand, request: Request):
    from services.learning_review_commands import run_review_command
    with _owners(request, '/api/learning/slash', personal_read=True) as (context, service):
        return run_review_command(context, service, body.message, body.request_id)
