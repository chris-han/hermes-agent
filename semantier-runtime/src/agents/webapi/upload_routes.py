from __future__ import annotations

from fastapi import APIRouter

from pathlib import Path
from hashlib import sha256
from typing import Any

from fastapi import HTTPException, Request
from starlette.exceptions import HTTPException as StarletteHTTPException

from agents.route_policy import RouteAuthorizationClass
from agents.workspace_uploads import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    WorkspaceUploadError,
    save_workspace_upload,
)
from agents.webapi.deps import (
    require_route_authorization as _require_route_authorization,
)
from agents.webapi.organization_service import __getattr_from_gateway_identity

router = APIRouter(tags=["webapi-gateway"])


async def _upload_temporal_notebook(request: Request, ctx):
    from agents.webapi.ontology_workbench_context_routes import (
        _workbench_owner_context,
        _temporal_upload_boundary,
    )
    from storage.runtime_working_context_store import RuntimeContextError

    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > 1_500_000:
            raise HTTPException(413, "RESOURCE_LIMIT_EXCEEDED")
        content.extend(chunk)

    async def receive():
        return {"type": "http.request", "body": bytes(content), "more_body": False}

    bounded = Request(request.scope, receive=receive)
    async with bounded.form(max_files=1, max_fields=1) as form:
        items = form.multi_items()
        if len(items) != 2 or {key for key, _ in items} != {"session_id", "file"}:
            raise HTTPException(422, "MALFORMED_CONTRACT")
        session = form["session_id"]
        file = form["file"]
        if not isinstance(session, str) or not session or not hasattr(file, "read"):
            raise HTTPException(422, "MALFORMED_CONTRACT")
        data = await file.read(1_048_577)
        if len(data) > 1_048_576:
            raise HTTPException(413, "RESOURCE_LIMIT_EXCEEDED")
        context = None
        try:
            context = _workbench_owner_context(
                ctx=ctx, session_locator=session, channel="web"
            )
            boundary, segment = _temporal_upload_boundary(
                ctx=ctx, session_ref=context.session_ref
            )
            saved = save_workspace_upload(
                workspace_root=ctx.workspace_root.resolve(),
                session_id=segment,
                filename=file.filename or "",
                content=data,
                content_type=file.content_type or "",
                allowed_extensions=frozenset({".ipynb"}),
                boundary=boundary,
            )
            return {
                "schema_version": "runtime_temporal_upload.v1",
                "session_ref": context.session_ref,
                "upload_ref": saved.relative_path,
                "upload_hash": "sha256:" + sha256(data).hexdigest(),
                "size": saved.size,
            }
        except RuntimeContextError:
            raise HTTPException(403, "FORBIDDEN_SCOPE") from None
        except WorkspaceUploadError as exc:
            raise HTTPException(
                403 if exc.status_code == 403 else 422, "INVALID_SOURCE"
            ) from None
        finally:
            if context is not None:
                try:
                    context.build_store.close()
                finally:
                    context.source_registry.close()


@router.post("/upload")
async def upload_workspace_document(request: Request):
    ctx = _require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.SELF_ONLY,
        route_hint="POST /upload",
    )
    versions = request.headers.getlist("Accept-Workspace-Upload-Version")
    if versions:
        if versions != ["runtime-temporal.v1"]:
            raise HTTPException(422, "MALFORMED_CONTRACT")
        try:
            return await _upload_temporal_notebook(request, ctx)
        except StarletteHTTPException as exc:
            if exc.status_code == 400:
                raise HTTPException(422, "MALFORMED_CONTRACT") from None
            raise
    form = await request.form()
    file_field = form.get("file")
    if (
        file_field is None
        or not hasattr(file_field, "filename")
        or not hasattr(file_field, "read")
    ):
        raise HTTPException(status_code=400, detail="file required")

    ensure_workspace_paths = __getattr_from_gateway_identity("ensure_workspace_paths")
    workspace_root, _ = ensure_workspace_paths(ctx.workspace_id)

    session_id = str(form.get("session_id") or "").strip()
    content = await file_field.read()
    try:
        saved = save_workspace_upload(
            workspace_root=Path(workspace_root),
            session_id=session_id,
            filename=file_field.filename or "",
            content=content,
            content_type=file_field.content_type or "",
        )
    except WorkspaceUploadError as exc:
        if exc.code == "unsupported_extension":
            detail: str | dict[str, Any] = {
                "error": str(exc),
                "allowed_extensions": sorted(ALLOWED_DOCUMENT_EXTENSIONS),
            }
        else:
            detail = str(exc)
        raise HTTPException(status_code=exc.status_code, detail=detail) from exc

    return {
        "status": "ok",
        "filename": saved.filename,
        "file_path": str(saved.file_path),
        "relative_path": saved.relative_path,
        "content_type": saved.content_type,
        "size": saved.size,
        "workspace_id": ctx.workspace_id,
    }
