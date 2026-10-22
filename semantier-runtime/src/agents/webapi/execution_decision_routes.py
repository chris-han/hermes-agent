"""Authenticated decision projections over explicit Core execution associations."""
from contextlib import contextmanager
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from agents.coordination_association import CoordinationAssociationConflict, CoordinationAssociationStore
from agents.durable_responsibility import DurableResponsibilityError, DurableResponsibilityStore
from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import require_route_authorization
from runtime_paths import workspace_hermes_home_path, workspace_root_path

router = APIRouter(tags=["execution-decisions"])


class UserWaitSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    coordination_id: str = Field(min_length=1, max_length=200)
    responsibility_id: str = Field(min_length=1, max_length=200)
    request_id: str = Field(min_length=1, max_length=200)
    wait_id: str = Field(min_length=1, max_length=200)
    expected_checkpoint_version: int = Field(ge=0, strict=True)
    expected_checkpoint_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


@contextmanager
def _owners(request: Request, route: str):
    ctx = require_route_authorization(
        request, authorization_class=RouteAuthorizationClass.TENANT_MEMBER, route_hint=route,
    )
    if (ctx.workspace_root.resolve() != workspace_root_path(ctx.workspace_id).resolve()
            or ctx.hermes_home.resolve() != workspace_hermes_home_path(ctx.workspace_id).resolve()):
        raise HTTPException(403, "Outside execution scope")
    path = ctx.hermes_home / "state.db"
    try:
        yield ctx, CoordinationAssociationStore(path), DurableResponsibilityStore(path)
    except PermissionError as exc:
        raise HTTPException(403, "Outside execution scope") from exc
    except KeyError as exc:
        raise HTTPException(404, "Execution owner not found") from exc
    except (CoordinationAssociationConflict, DurableResponsibilityError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


def _context(associations, coordination_id, workspace_id):
    context = associations.get_context(coordination_id)
    if context is None or context.workspace_id != workspace_id:
        raise KeyError(coordination_id)
    return context, associations.project(coordination_id)


@router.post("/api/presentation-context")
async def presentation_context(request: Request):
    from agents.webapi.presentation_context import resolve_presentation_context
    ctx = require_route_authorization(
        request, authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
        route_hint="POST /api/presentation-context",
    )
    body = await request.json()
    if not isinstance(body, dict) or not isinstance(body.get("message"), str) or len(body["message"]) > 10000:
        raise HTTPException(400, "bounded message required")
    return resolve_presentation_context(ctx, body["message"], body.get("page"))


@router.get("/api/executions")
def execution_inventory(request: Request):
    with _owners(request, "GET /api/executions") as (ctx, associations, responsibilities):
        executions = []
        for context in associations.list_contexts(ctx.workspace_id):
            projection = associations.project(context.coordination_id)
            decisions = [responsibilities.decision_snapshot(
                rid, workspace_id=ctx.workspace_id, actor_ref=f"user:{ctx.user_id}",
            ) for rid in projection.responsibility_refs]
            executions.append({
                **asdict(context),
                "waiting_count": sum(len(d["waiting_basis"]) for d in decisions),
                "needs_current_user": any(d["eligible_interventions"] for d in decisions),
                "decision_capability": "AVAILABLE" if decisions else "UNAVAILABLE",
            })
        return {"schema_version": "execution_inventory.v1", "executions": executions}


@router.get("/api/executions/decision")
def execution_decision(request: Request, coordination_id: str = Query(min_length=1, max_length=200)):
    with _owners(request, "GET /api/executions/decision") as (ctx, associations, responsibilities):
        context, projection = _context(associations, coordination_id, ctx.workspace_id)
        decisions = [responsibilities.decision_snapshot(
            rid, workspace_id=ctx.workspace_id, actor_ref=f"user:{ctx.user_id}",
        ) for rid in projection.responsibility_refs]
        return {
            "schema_version": "execution_decision.v1", "context": asdict(context),
            "association": asdict(projection),
            "capability": "AVAILABLE" if decisions else "UNAVAILABLE",
            "unavailable_reason": None if decisions else "no_bound_responsibility",
            "decisions": decisions,
            "coordination_receipts": [asdict(r) for r in associations.list_receipts(coordination_id)],
        }


@router.post("/api/executions/signal-user-wait")
def signal_user_wait(body: UserWaitSignal, request: Request):
    with _owners(request, "POST /api/executions/signal-user-wait") as (ctx, associations, responsibilities):
        _, projection = _context(associations, body.coordination_id, ctx.workspace_id)
        if body.responsibility_id not in projection.responsibility_refs:
            raise KeyError(body.responsibility_id)
        receipt = responsibilities.signal_user_wait(
            responsibility_id=body.responsibility_id, workspace_id=ctx.workspace_id,
            actor_ref=f"user:{ctx.user_id}", request_id=body.request_id, wait_id=body.wait_id,
            expected_checkpoint_version=body.expected_checkpoint_version,
            expected_checkpoint_hash=body.expected_checkpoint_hash,
        )
        return {"receipt": receipt}
