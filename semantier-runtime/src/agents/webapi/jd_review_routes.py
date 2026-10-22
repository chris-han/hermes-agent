"""Authenticated JD evidence operations. HTTP never supplies policy authority or edits."""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import re
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from starlette.datastructures import UploadFile
from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import require_route_authorization
from agents.webapi.ontology_workbench_context_routes import (
    _workbench_owner_context,
    _temporal_upload_boundary,
)
from agents.workspace_uploads import (
    save_workspace_upload,
    read_document_workspace_upload,
)
from contracts.jd_compliance_review import (
    JdReviewV1,
    JdBasisEvidenceV1,
    JdSuccessorCommandV1,
)
from plugins.document_extraction.storage import BoundSession, sha256_bytes
from services.jd_compliance_review import JdComplianceReviewService
from services.activation_snapshot_resolver_service import (
    StoreBackedActivationSnapshotResolverService,
)
from storage.governed_rule_store import GovernedRuleStore
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path

router = APIRouter()
MAX_BYTES = 20 * 1024 * 1024


@contextmanager
def review_service(ctx, session_locator):
    rule_store = None
    owners = _workbench_owner_context(
        ctx=ctx, session_locator=session_locator, channel="web", authorization_class="WORKSPACE_LOCAL"
    )
    try:
        boundary, segment = _temporal_upload_boundary(
            ctx=ctx, session_ref=owners.session_ref
        )
        bound = BoundSession(
            Path(boundary.paths.hermes_home),
            owners.session_ref,
            boundary.paths.uploads_root.parent,
        )
        rule_store = GovernedRuleStore()
        service = JdComplianceReviewService(
            store=owners.build_store,
            workspace_root=ctx.workspace_root,
            actor_context=owners.actor_context,
            trusted_session=bound,
            snapshot_resolver=StoreBackedActivationSnapshotResolverService(rule_store),
        )
        yield service, boundary, segment
    finally:
        if rule_store is not None:
            rule_store._conn.close()
        owners.build_store.close()
        owners.source_registry.close()


def _session(request):
    values = request.headers.getlist("X-Workbench-Session-Ref")
    if len(values) != 1 or not values[0].strip():
        raise HTTPException(401, "AUTHORITY_CONTEXT_REQUIRED")
    return values[0]


def _ctx(request):
    return require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.SELF_ONLY,
        route_hint=request.url.path,
    )


def _version(service, review_id, version_id):
    review = service.get(review_id)
    if not version_id or version_id == review.version_id:
        return review
    for pin in service.store.list_jd_review_lineage_pins(
        organization_id=service.context.organization_id,
        workspace_id=service.context.workspace_id,
        actor_id=service.context.actor_id,
        session_id=service.context.session_id,
        lineage_review_id=(review.successor or {}).get(
            "lineage_review_id", review.review_id
        ),
    ):
        candidate = service.store.get(
            organization_id=service.context.organization_id,
            workspace_id=service.context.workspace_id,
            artifact_ref=pin.artifact_ref,
            contract_type=JdReviewV1,
        )
        if (
            candidate.version_id == version_id
            and candidate.successor
            and candidate.successor.get(
                "lineage_review_id", candidate.successor.get("supersedes_review_id")
            )
            == (review.successor or {}).get("lineage_review_id", review.review_id)
        ):
            service._access(candidate)
            return candidate
    raise ValueError("VERSION_NOT_FOUND")


def _payload(service, review):
    if review.status in {"CHECKED", "PARTIAL"}:
        service.get_graph(review.review_id)
    result = review.model_dump(mode="json")
    root = (
        review.successor.get(
            "lineage_review_id", review.successor.get("supersedes_review_id")
        )
        if review.successor
        else review.review_id
    )
    versions = []
    for pin in service.store.list_jd_review_lineage_pins(
        organization_id=service.context.organization_id,
        workspace_id=service.context.workspace_id,
        actor_id=service.context.actor_id,
        session_id=service.context.session_id,
        lineage_review_id=root,
    ):
        candidate = service.store.get(
            organization_id=service.context.organization_id,
            workspace_id=service.context.workspace_id,
            artifact_ref=pin.artifact_ref,
            contract_type=JdReviewV1,
        )
        if (
            candidate.actor_id != service.context.actor_id
            or candidate.session_id != service.context.session_id
        ):
            continue
        if candidate.status not in {"CHECKED", "PARTIAL"}:
            continue
        if candidate.review_id == root or (
            candidate.successor
            and candidate.successor.get(
                "lineage_review_id", candidate.successor.get("supersedes_review_id")
            )
            == root
        ):
            versions.append(
                dict(
                    review_id=candidate.review_id,
                    version_id=candidate.version_id,
                    result_hash=candidate.result_hash,
                    status=candidate.status,
                    filename=candidate.source["filename"],
                    source_identity_ref=candidate.source["source_identity_ref"],
                    source_hash=candidate.source["source_hash"],
                )
            )
    result["result_artifact_ref"] = review.review_id
    result["result_hash_scope"] = (
        "saved jd_review.v1 artifact; versions and lineage API projection excluded"
    )
    result["review_id"] = root
    result["versions"] = versions
    return result


def _error(exc):
    from services.runtime_errors import RuntimeContextError

    if isinstance(exc, (PermissionError, RuntimeContextError)):
        raise HTTPException(403, "FORBIDDEN_SCOPE") from None
    code = str(exc)
    raise HTTPException(
        404
        if code in {"REVIEW_NOT_FOUND", "VERSION_NOT_FOUND"}
        else 409
        if code.startswith("STALE_")
        else 422,
        code if re.fullmatch("[A-Z_]+", code) else "REVIEW_OPERATION_FAILED",
    ) from None


@router.post("/api/jd-reviews")
async def create_jd_review(request: Request):
    ctx = _ctx(request)
    session = _session(request)
    try:
        with review_service(ctx, session) as (service, boundary, segment):
            # Scope is resolved before body/file access. Limit the entire multipart envelope.
            data = bytearray()
            async for chunk in request.stream():
                if len(data) + len(chunk) > MAX_BYTES + 65536:
                    raise HTTPException(413, "INVALID_BYTE_LENGTH")
                data.extend(chunk)
            request._body = bytes(data)
            async with request.form(
                max_files=1, max_fields=0, max_part_size=MAX_BYTES + 65536
            ) as form:
                if (
                    len(form.multi_items()) != 1
                    or set(form) != {"file"}
                    or not isinstance(form["file"], UploadFile)
                ):
                    raise HTTPException(422, "INVALID_UPLOAD")
                file = form["file"]
                content = await file.read(MAX_BYTES + 1)
                if len(content) > MAX_BYTES:
                    raise HTTPException(413, "INVALID_BYTE_LENGTH")
                uploaded = save_workspace_upload(
                    workspace_root=ctx.workspace_root,
                    session_id=segment,
                    filename=file.filename or "",
                    content=content,
                    content_type=file.content_type or "",
                    allowed_extensions=frozenset({".docx", ".pdf", ".md", ".txt"}),
                    boundary=boundary,
                )
                read_document_workspace_upload(
                    boundary=boundary,
                    upload_ref=uploaded.relative_path,
                    upload_hash=sha256_bytes(content),
                    max_bytes=MAX_BYTES,
                )
                review = service.create(
                    document_ref="uploads/" + uploaded.filename,
                    filename=uploaded.filename,
                )
                return JSONResponse(
                    _payload(service, review), headers={"Cache-Control": "no-store"}
                )
    except HTTPException:
        raise
    except Exception as exc:
        _error(exc)


@router.get("/api/jd-reviews/{review_id}")
async def get_jd_review(request: Request, review_id: str):
    ctx = _ctx(request)
    session = _session(request)
    try:
        if (
            set(request.query_params) - {"version_id"}
            or len(request.query_params.getlist("version_id")) > 1
        ):
            raise ValueError("INVALID_VERSION_QUERY")
        with review_service(ctx, session) as (service, _, __):
            return JSONResponse(
                _payload(
                    service,
                    _version(
                        service, review_id, request.query_params.get("version_id")
                    ),
                ),
                headers={"Cache-Control": "no-store"},
            )
    except Exception as exc:
        _error(exc)


@router.post("/api/jd-reviews/{review_id}/successors")
async def create_jd_successor(request: Request, review_id: str):
    ctx = _ctx(request)
    session = _session(request)
    try:
        with review_service(ctx, session) as (service, _, __):
            command = JdSuccessorCommandV1.model_validate(await request.json())
            parent = _version(service, review_id, command.version_id)
            return JSONResponse(
                _payload(service, service.successor(parent.review_id, command)),
                headers={"Cache-Control": "no-store"},
            )
    except Exception as exc:
        _error(exc)


@router.get("/api/jd-reviews/{review_id}/versions/{version_id}/download")
async def download_jd_version(request: Request, review_id: str, version_id: str):
    ctx = _ctx(request)
    session = _session(request)
    try:
        with review_service(ctx, session) as (service, _, __):
            review = _version(service, review_id, version_id)
            if review.status not in {"CHECKED", "PARTIAL"} and not (
                review.status == "RECHECK_FAILED"
                and review.successor.get("generated_artifact_ref")
                and review.successor.get("generation_status") == "COMPLETE"
            ):
                raise ValueError("INCOMPLETE_SUCCESSOR")
            path = service.source_path(review.review_id)
            return FileResponse(
                path,
                media_type=review.source["identity"]["mediaType"],
                filename=review.source["identity"]["originalName"],
                headers={
                    "Cache-Control": "no-store",
                    "X-Source-Hash": review.source["source_hash"],
                },
            )
    except Exception as exc:
        _error(exc)


def serve_jd_source(request, ctx, identity):
    """Only registered owner/session evidence identities reach a byte read."""
    store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
    try:
        if identity.startswith("jd-source:"):
            ref = identity.removeprefix("jd-source:")
            model = JdReviewV1
        elif identity.startswith("jd-basis-source:"):
            artifact = None
            for pin in store.list_artifact_pins(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                schema_version="jd_basis_evidence.v1",
            ):
                candidate = store.get(
                    organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id,
                    artifact_ref=pin.artifact_ref,
                    contract_type=JdBasisEvidenceV1,
                )
                if (
                    candidate.actor_id == ctx.user_id
                    and candidate.document["source"]["source_identity_ref"] == identity
                ):
                    artifact = candidate
                    break
            model = JdBasisEvidenceV1
            if artifact is None:
                raise ValueError("SOURCE_NOT_FOUND")
        else:
            raise ValueError("SOURCE_NOT_FOUND")
        if model is JdReviewV1:
            artifact = store.get(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_ref=ref,
                contract_type=model,
            )
        if artifact is None or artifact.actor_id != ctx.user_id:
            raise PermissionError("FORBIDDEN_SCOPE")
        session = artifact.session_id
        with review_service(ctx, session) as (service, _, __):
            if model is JdReviewV1:
                if artifact.status not in {"CHECKED", "PARTIAL"} and not (
                    artifact.status == "RECHECK_FAILED"
                    and (artifact.successor or {}).get("generation_status")
                    == "COMPLETE"
                ):
                    raise ValueError("INCOMPLETE_SUCCESSOR")
                source = artifact.source
                path = service.source_path(artifact.review_id)
            else:
                source = artifact.document["source"]
                path = (service.bound.session_root / source["artifact_ref"]).resolve()
                path.relative_to(service.bound.session_root.resolve())
                if sha256_bytes(path.read_bytes()) != source["source_hash"]:
                    raise ValueError("STALE_SOURCE_HASH")
            if source["source_identity_ref"] != identity:
                raise ValueError("SOURCE_IDENTITY_MISMATCH")
            return FileResponse(
                path,
                media_type=source["identity"]["mediaType"],
                filename=source["identity"]["originalName"],
                content_disposition_type="inline",
                headers={
                    "Cache-Control": "no-store",
                    "ETag": '"' + source["source_hash"] + '"',
                    "X-Source-Hash": source["source_hash"],
                },
            )
    except Exception as exc:
        _error(exc)
    finally:
        store.close()


@router.post("/api/jd-reviews/{review_id}/feedback")
async def submit_jd_feedback(request: Request, review_id: str):
    from contracts.jd_compliance_review import JdFeedbackCommandV1

    ctx = _ctx(request)
    session = _session(request)
    try:
        with review_service(ctx, session) as (service, _, __):
            command = JdFeedbackCommandV1.model_validate(await request.json())
            review = _version(service, review_id, command.version_id)
            return JSONResponse(
                service.feedback(review.review_id, command).model_dump(mode="json"),
                headers={"Cache-Control": "no-store"},
            )
    except Exception as exc:
        _error(exc)
