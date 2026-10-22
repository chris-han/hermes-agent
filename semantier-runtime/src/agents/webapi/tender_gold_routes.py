"""Authenticated Tender task transport; actor/scope always come from the server."""

from typing import Literal
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import require_route_authorization
from contracts.tender_gold import Annotation, DraftAnnotation, Role
from services.tender_gold_service import TenderGoldService
from storage.hosted_postgres import sql_query

router = APIRouter(tags=["tender-gold"])
BASE = "/api/tender-gold/packages"


class DraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Role
    case_id: str
    annotation: DraftAnnotation
    expected_revision: int = Field(ge=0)


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["R1", "R2"]
    annotation: Annotation
    expected_revision: int = Field(ge=0)


class SealRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["R1", "R2"]
    attestation: str = Field(min_length=1)


class AdjudicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    annotation: Annotation
    primary_hashes: list[str] = Field(min_length=2, max_length=2)
    expected_revision: int = Field(ge=0)
    accept: bool = False


class AcceptanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    reference_hash: str
    disputed: bool = False


def _call(request, operation):
    ctx = require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
        route_hint=f"{request.method} {request.url.path}",
    )
    if not ctx.workspace_id or not ctx.organization_id:
        raise HTTPException(409, "GOVERNED_WORKSPACE_REQUIRED")
    service = TenderGoldService(
        ctx.workspace_root,
        organization_id=str(ctx.organization_id),
        workspace_id=ctx.workspace_id,
    )
    try:
        return operation(service, ctx.user_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, "TENDER_PACKAGE_OR_CASE_NOT_FOUND") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    finally:
        service.close()


@router.get(BASE)
def packages(request: Request):
    def project(s, actor):
        rows = s.db.execute(
            sql_query("SELECT id,payload FROM tender_gold_packages WHERE org=? AND workspace=? ORDER BY id",'SELECT id,payload FROM tender_gold_packages WHERE org=%s AND workspace=%s ORDER BY id'),
            (s.org, s.workspace),
        )
        result = []
        for row in rows:
            roles = [
                r["role"]
                for r in s.db.execute(
                    sql_query("SELECT role FROM tender_gold_assignments WHERE package=? AND actor=? ORDER BY role",'SELECT role FROM tender_gold_assignments WHERE package=%s AND actor=%s ORDER BY role'),
                    (row["id"], actor),
                )
            ]
            if roles:
                result.append(dict(package_id=row["id"], roles=roles))
        return dict(
            actor_id=actor,
            packages=result,
            genuine_human_reviews=0,
            qualified_references=0,
            unavailable_reason=None
            if result
            else "APPROVED_ASSIGNMENT_AND_SOURCE_ACCESS_REQUIRED",
        )

    return _call(request, project)


@router.get(BASE + "/{package_id}")
def package(package_id: str, request: Request, role: Role = "R1"):
    return _call(
        request, lambda s, actor: s.view(package_id, actor_id=actor, role=role)
    )


@router.put(BASE + "/{package_id}/draft")
def draft(package_id: str, request: Request, body: DraftRequest):
    return _call(
        request,
        lambda s, actor: s.save_draft(
            package_id,
            actor_id=actor,
            **body.model_dump(exclude={"annotation"}),
            annotation=body.annotation,
        ),
    )


@router.post(BASE + "/{package_id}/cases/{case_id}/annotations")
def submit(package_id: str, case_id: str, request: Request, body: SubmitRequest):
    return _call(
        request,
        lambda s, actor: s.submit(
            package_id,
            case_id=case_id,
            actor_id=actor,
            role=body.role,
            annotation=body.annotation,
            expected_revision=body.expected_revision,
        ),
    )


@router.post(BASE + "/{package_id}/submissions")
def seal(package_id: str, request: Request, body: SealRequest):
    return _call(
        request,
        lambda s, actor: s.seal(package_id, actor_id=actor, **body.model_dump()),
    )


@router.get(BASE + "/{package_id}/comparison")
def comparison(package_id: str, request: Request):
    return _call(request, lambda s, actor: s.comparison(package_id, actor_id=actor))


@router.post(BASE + "/{package_id}/adjudications")
def adjudicate(package_id: str, request: Request, body: AdjudicationRequest):
    return _call(
        request,
        lambda s, actor: s.adjudicate(
            package_id,
            actor_id=actor,
            **body.model_dump(exclude={"annotation"}),
            annotation=body.annotation,
        ),
    )


@router.post(BASE + "/{package_id}/qualifications")
def qualify(package_id: str, request: Request, body: AcceptanceRequest):
    return _call(
        request,
        lambda s, actor: s.accept(package_id, actor_id=actor, **body.model_dump()),
    )


@router.get(BASE + "/{package_id}/dataset")
def dataset(
    package_id: str,
    request: Request,
    role: Role = "R1",
    kind: Literal["candidate", "qualified"] = "candidate",
    format: Literal["json", "zip"] = "zip",
    version: str | None = None,
):
    content = _call(
        request,
        lambda s, actor: (s.bundle if format == "zip" else s.export)(
            package_id, actor_id=actor, role=role, kind=kind, version=version
        ),
    )
    return Response(
        content,
        media_type="application/zip" if format == "zip" else "application/json",
        headers={
            "Content-Disposition": f'attachment; filename="tender-{kind}.{format}"',
            "Cache-Control": "no-store",
        },
    )
