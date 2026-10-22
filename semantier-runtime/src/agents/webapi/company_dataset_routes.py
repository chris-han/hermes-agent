from __future__ import annotations

from fastapi import APIRouter

import json
from typing import Any, Literal

from fastapi import File, Form, HTTPException, Request, UploadFile

from agents import company_dataset_imports
from agents.route_policy import classify_authorization_class
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from agents.webapi.organization_service import get_user_organization_context

router = APIRouter(tags=["webapi-gateway"])

@router.get("/company-data-context/setup-status")
async def company_data_context_setup_status(request: Request):
    authz_class = classify_authorization_class("GET", "/company-data-context/setup-status")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="GET /company-data-context/setup-status",
    )
    org_context = get_user_organization_context(ctx.user_id) or {}
    return {
        "ok": True,
        **company_dataset_imports.setup_status_for_context(ctx, org_context),
    }



def _raise_company_dataset_error(exc: Exception) -> None:
    if isinstance(exc, company_dataset_imports.CompanyDatasetAuthorizationError):
        detail = str(exc)
        status_code = 403 if "admin" in detail.lower() else 409
        raise HTTPException(status_code=status_code, detail=detail) from exc
    if isinstance(exc, company_dataset_imports.CompanyDatasetValidationError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail="import not found") from exc
    raise HTTPException(status_code=400, detail=str(exc)) from exc


def _json_object_from_form(value: str | None, *, default: dict[str, Any]) -> dict[str, Any]:
    if value is None or not value.strip():
        return dict(default)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON form field") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="JSON form field must be an object")
    return parsed


@router.post("/company-dataset-imports/upload")
async def company_dataset_imports_upload(
    request: Request,
    idempotency_key: str = Form(...),
    purpose: Literal["company_dataset", "receivables_candidate"] = Form("company_dataset"),
    parser_profile: str | None = Form(None),
    source_path: str | None = Form(None),
    files: list[UploadFile] | None = File(None),
):
    authz_class = classify_authorization_class("POST", "/company-dataset-imports/upload")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /company-dataset-imports/upload",
    )
    try:
        if purpose == "receivables_candidate" and source_path:
            raise company_dataset_imports.CompanyDatasetValidationError("Candidate imports require uploaded files")
        profile_payload = _json_object_from_form(parser_profile, default={})
        profile = company_dataset_imports.build_parser_profile(
            selected_sheets=profile_payload.get("selected_sheets")
            if isinstance(profile_payload.get("selected_sheets"), list)
            else None,
            header_rows=profile_payload.get("header_rows")
            if isinstance(profile_payload.get("header_rows"), dict)
            else None,
            all_sheets=profile_payload.get("all_sheets") is True,
            auto_header=profile_payload.get("auto_header") is True,
            locale=str(profile_payload.get("locale") or "zh-CN"),
            timezone=str(profile_payload.get("timezone") or "UTC"),
            encoding=str(profile_payload.get("encoding") or "utf-8"),
            required_column_profile=str(
                profile_payload.get("required_column_profile") or "company_dataset_v1"
            ),
        )
        uploaded = [
            company_dataset_imports.UploadedDatasetFile(
                filename=str(file.filename or ""),
                content_type=str(file.content_type or ""),
                content=await file.read(),
            )
            for file in files or []
        ]
        if source_path and source_path.strip():
            uploaded.extend(
                company_dataset_imports.uploaded_dataset_files_from_path(source_path)
            )
        record = company_dataset_imports.create_dataset_import(
            ctx=ctx,
            idempotency_key=idempotency_key,
            files=uploaded,
            parser_profile=profile,
            purpose=purpose,
        )
    except Exception as exc:
        _raise_company_dataset_error(exc)
    return {"ok": True, "import": record}


@router.post("/company-dataset-imports/preview")
async def company_dataset_imports_preview(
    request: Request,
    purpose: Literal["company_dataset", "receivables_candidate"] = Form("company_dataset"),
    source_path: str | None = Form(None),
    files: list[UploadFile] | None = File(None),
):
    authz_class = classify_authorization_class("POST", "/company-dataset-imports/preview")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /company-dataset-imports/preview",
    )
    try:
        if purpose == "receivables_candidate":
            company_dataset_imports._assert_candidate_context(ctx)
            if source_path:
                raise company_dataset_imports.CompanyDatasetValidationError("Candidate previews require uploaded files")
        else:
            company_dataset_imports._assert_active_real_context(ctx)
        uploaded = [
            company_dataset_imports.UploadedDatasetFile(
                filename=str(file.filename or ""),
                content_type=str(file.content_type or ""),
                content=await file.read(),
            )
            for file in files or []
        ]
        if source_path and source_path.strip():
            uploaded.extend(
                company_dataset_imports.uploaded_dataset_files_from_path(source_path)
            )
        preview = company_dataset_imports.preview_dataset_files(uploaded)
    except Exception as exc:
        _raise_company_dataset_error(exc)
    return {"ok": True, "preview": preview}


@router.get("/company-dataset-imports")
async def company_dataset_imports_list(request: Request):
    authz_class = classify_authorization_class("GET", "/company-dataset-imports")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="GET /company-dataset-imports",
    )
    try:
        imports = company_dataset_imports.list_dataset_imports(ctx=ctx)
    except Exception as exc:
        _raise_company_dataset_error(exc)
    return {"ok": True, "imports": imports}


@router.get("/company-dataset-imports/active-dataset")
async def company_dataset_imports_active_dataset(request: Request):
    authz_class = classify_authorization_class("GET", "/company-dataset-imports/active-dataset")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="GET /company-dataset-imports/active-dataset",
    )
    try:
        dataset = company_dataset_imports.get_active_dataset_explorer(
            ctx=ctx,
            file_id=request.query_params.get("file_id"),
            page=int(request.query_params.get("page") or "1"),
            page_size=int(request.query_params.get("page_size") or "25"),
            filter_text=str(request.query_params.get("filter") or ""),
        )
    except Exception as exc:
        _raise_company_dataset_error(exc)
    return {"ok": True, "dataset": dataset}


@router.get("/company-dataset-imports/{import_id}")
async def company_dataset_imports_get(import_id: str, request: Request):
    authz_class = classify_authorization_class("GET", "/company-dataset-imports/{import_id}")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="GET /company-dataset-imports/{import_id}",
    )
    try:
        record = company_dataset_imports.get_dataset_import(ctx=ctx, import_id=import_id)
    except Exception as exc:
        _raise_company_dataset_error(exc)
    if record is None:
        raise HTTPException(status_code=404, detail="import not found")
    return {"ok": True, "import": record}


@router.post("/company-dataset-imports/{import_id}/validate")
async def company_dataset_imports_validate(import_id: str, request: Request):
    authz_class = classify_authorization_class("POST", "/company-dataset-imports/{import_id}/validate")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /company-dataset-imports/{import_id}/validate",
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    idempotency_key = body.get("idempotency_key")
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        raise HTTPException(status_code=400, detail="idempotency_key required")
    try:
        record = company_dataset_imports.validate_dataset_import(
            ctx=ctx,
            import_id=import_id,
            idempotency_key=idempotency_key,
        )
    except Exception as exc:
        _raise_company_dataset_error(exc)
    return {"ok": True, "import": record}


@router.post("/company-dataset-imports/{import_id}/promote")
async def company_dataset_imports_promote(import_id: str, request: Request):
    authz_class = classify_authorization_class("POST", "/company-dataset-imports/{import_id}/promote")
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    ctx = _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint="POST /company-dataset-imports/{import_id}/promote",
    )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    idempotency_key = body.get("idempotency_key")
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        raise HTTPException(status_code=400, detail="idempotency_key required")
    try:
        record = company_dataset_imports.promote_dataset_import(
            ctx=ctx,
            import_id=import_id,
            idempotency_key=idempotency_key,
        )
    except Exception as exc:
        _raise_company_dataset_error(exc)
    return {"ok": True, "import": record}



