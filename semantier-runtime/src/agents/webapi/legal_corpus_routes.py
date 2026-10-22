from __future__ import annotations

import re
from pathlib import PurePosixPath

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from agents.route_policy import classify_authorization_class
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from eos.db import init_db
from eos.legal_authority_runtime import (
    LEGAL_RUNTIME_REPLAY_PIN_MISSING,
    LegalAuthorityRuntime,
)
from eos.legal_corpus_store import (
    LEGAL_ACTIVATION_EVIDENCE_REQUIRED,
    LEGAL_DEMO_APPROVAL_NOT_ACTIVATABLE,
    LEGAL_DEMO_ARTIFACT_NOT_CERTIFIABLE,
    LegalCorpusStore,
)
from eos.legal_regulatory_monitor import LegalRegulatoryMonitor
from eos.legal_source_acquisition import GovernedManualFallbackAdapter


router = APIRouter(tags=["legal-corpus"])

ROUTE_INVENTORY = "/api/legal-corpus/inventory"
ROUTE_DASHBOARD = "/api/legal-corpus/dashboard"
ROUTE_PIPELINE2_EVIDENCE_CONTRACT = "/api/legal-corpus/pipeline2-evidence-contract"
ROUTE_ACCEPTANCE_EVIDENCE_EXPORT = "/api/legal-corpus/acceptance-evidence-export"
ROUTE_ACCEPTANCE_EVIDENCE_EXPORTS = "/api/legal-corpus/acceptance-evidence-exports"
ROUTE_REGISTER_SOURCE = "/api/legal-corpus/sources"
ROUTE_CREATE_VERSION = "/api/legal-corpus/sources/{source_id}/versions"
ROUTE_PRESERVE_ARTIFACT = "/api/legal-corpus/versions/{version_id}/artifacts"
ROUTE_UPLOAD_SOURCE_ARTIFACT = "/api/legal-corpus/versions/{version_id}/source-upload"
ROUTE_MANUAL_FALLBACK = "/api/legal-corpus/sources/{source_id}/manual-fallback"
ROUTE_REFRESH_SCHEDULE = "/api/legal-corpus/sources/{source_id}/refresh-schedule"
ROUTE_REFRESH_CHECK = "/api/legal-corpus/sources/{source_id}/refresh-check"
ROUTE_SCAN_RUNS = "/api/legal-corpus/scan-runs"
ROUTE_SCAN_RUN = "/api/legal-corpus/scan-runs/{scan_run_id}"
ROUTE_DUE_SOURCES = "/api/legal-corpus/sources/due"
ROUTE_SOURCE_STATUS = "/api/legal-corpus/sources/{source_id}/status"
ROUTE_SCAN_ITEM_RETRY = "/api/legal-corpus/scan-items/{scan_item_id}/retry"
ROUTE_CHANGE_CANDIDATES = "/api/legal-corpus/change-candidates"
ROUTE_CHANGE_CANDIDATE_IMPACT = "/api/legal-corpus/change-candidates/{candidate_id}/impact"
ROUTE_ALERT_ACKNOWLEDGE = "/api/legal-corpus/alerts/{alert_id}/acknowledge"
ROUTE_CERTIFY = "/api/legal-corpus/certifications"
ROUTE_ACTIVATE = "/api/legal-corpus/activations"
ROUTE_RUNTIME_EXECUTE = "/api/legal-runtime/execute"
ROUTE_RUNTIME_REPLAY = "/api/legal-runtime/replay"


def _store() -> LegalCorpusStore:
    init_db()
    return LegalCorpusStore()


def _runtime() -> LegalAuthorityRuntime:
    init_db()
    return LegalAuthorityRuntime()


def _monitor() -> LegalRegulatoryMonitor:
    return LegalRegulatoryMonitor(_store())


def _authz(request: Request, method: str, path: str):
    authz_class = classify_authorization_class(method, path)
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    return _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint=f"{method} {path}",
    )


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return body


def _organization_id(ctx) -> str:
    organization_id = str(ctx.organization_id or "").strip()
    if not organization_id:
        raise HTTPException(status_code=409, detail="No organization selected")
    return organization_id


def _correlation_id(request: Request, body: dict) -> str:
    return str(
        body.get("correlation_id")
        or request.headers.get("x-correlation-id")
        or request.headers.get("x-request-id")
        or ""
    ).strip()


def _safe_upload_filename(filename: str) -> str:
    name = PurePosixPath(str(filename or "source.pdf").replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "-", name).strip(".-")
    return name or "source.pdf"


def _extract_pdf_text(content: bytes) -> str:
    """Metadata preview through the qualified born-digital PyMuPDF observation provider."""
    import hashlib
    from plugins.document_extraction.parsers import pymupdf_native

    try:
        parsed = pymupdf_native.parse_bytes_bound(
            content,
            artifact_ref="upload:legal-source-preview",
            source_sha256="sha256:" + hashlib.sha256(content).hexdigest(),
            media_type="application/pdf",
            byte_length=len(content),
            execution_scope_ref="legal-source-preview",
        )
        return "\n".join(
            str(block.get("text") or "")
            for block in parsed.get("blocks", [])
            if int(block.get("page") or 1) <= 2
        )
    except Exception:
        return ""


def _metadata_suggestions_from_upload(filename: str, content: bytes, mime_type: str) -> dict:
    text = _extract_pdf_text(content) if mime_type == "application/pdf" else content[:12000].decode("utf-8", "ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename).replace("_", " ").replace("-", " ").strip()
    title = next((line for line in lines if 4 <= len(line) <= 120), stem)
    issuer = next(
        (
            line
            for line in lines[:40]
            if re.search(r"(人民代表大会|委员会|国务院|政府|法院|检察院|部|局|厅|Ministry|Commission|Council)", line)
        ),
        "",
    )
    document_number_match = re.search(r"[\w\u4e00-\u9fff]*〔?\d{4}[〕\]][\w\u4e00-\u9fff第号\-]+", text)
    date_match = re.search(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})日?", text)
    effective_from = ""
    if date_match:
        year, month, day = date_match.groups()
        effective_from = f"{int(year):04d}-{int(month):02d}-{int(day):02d}T00:00:00+00:00"
    return {
        "canonical_title": title,
        "issuer": issuer,
        "document_number": document_number_match.group(0) if document_number_match else "",
        "effective_from": effective_from,
        "extraction_method": "pdf_text_layer_metadata" if text and mime_type == "application/pdf" else "filename_metadata",
    }


def _record_failure(
    *,
    request: Request,
    body: dict,
    ctx,
    attempted_endpoint: str,
    reason_code: str,
) -> dict:
    correlation_id = _correlation_id(request, body)
    if not correlation_id:
        correlation_id = f"legal-action-{str(body.get('idempotency_key') or 'no-idempotency')}"
    object_ref = str(body.get("object_ref") or body.get("authority_bundle_version_id") or "").strip()
    if not object_ref:
        object_ref = "unknown"
    object_type = str(body.get("object_type") or body.get("activation_source_type") or "").strip()
    if not object_type:
        object_type = "unknown"
    event = _store().record_denial_event(
        object_ref=object_ref,
        object_type=object_type,
        lifecycle_state=str(body.get("lifecycle_state") or "").strip() or None,
        attempted_endpoint=attempted_endpoint,
        actor_ref=str(ctx.user_id),
        organization_id=_organization_id(ctx),
        reason_code=reason_code,
        route_policy_id=f"POST {attempted_endpoint}",
        correlation_id=correlation_id,
        event_payload={
            "schema": "legal_action_failure.v1",
            "attempted_endpoint": attempted_endpoint,
            "safe_object_ref": object_ref,
            "object_type": object_type,
            "reason_code": reason_code,
            "correlation_id": correlation_id,
        },
    )
    return {
        "schema": "legal_action_failure.v1",
        "route_policy_id": f"POST {attempted_endpoint}",
        "reason_code": reason_code,
        "correlation_id": correlation_id,
        "actor_ref": str(ctx.user_id),
        "organization_id": _organization_id(ctx),
        "object_ref": object_ref,
        "object_type": object_type,
        "audit_event": event,
    }


@router.get(ROUTE_INVENTORY)
async def get_legal_corpus_inventory(request: Request):
    ctx = _authz(request, "GET", ROUTE_INVENTORY)
    inventory = _store().list_inventory(organization_id=_organization_id(ctx))
    return {"ok": True, "inventory": inventory}


@router.get(ROUTE_DASHBOARD)
async def get_legal_corpus_dashboard(request: Request, expected_source_count: int | None = None):
    ctx = _authz(request, "GET", ROUTE_DASHBOARD)
    dashboard = _store().dashboard_metrics(
        organization_id=_organization_id(ctx),
        expected_source_count=expected_source_count,
    )
    return {"ok": True, "dashboard": dashboard}


@router.get(ROUTE_PIPELINE2_EVIDENCE_CONTRACT)
async def get_legal_pipeline2_evidence_contract(request: Request):
    ctx = _authz(request, "GET", ROUTE_PIPELINE2_EVIDENCE_CONTRACT)
    contract = _store().build_pipeline2_evidence_contract(
        organization_id=_organization_id(ctx),
    )
    return {"ok": True, "evidence_contract": contract}


@router.get(ROUTE_ACCEPTANCE_EVIDENCE_EXPORT)
async def get_legal_acceptance_evidence_export(
    request: Request,
    test_run_id: str,
):
    ctx = _authz(request, "GET", ROUTE_ACCEPTANCE_EVIDENCE_EXPORT)
    test_run_id = str(test_run_id or "").strip()
    if not test_run_id:
        raise HTTPException(status_code=400, detail="test_run_id is required")
    export = _store().build_acceptance_evidence_export(
        organization_id=_organization_id(ctx),
        test_run_id=test_run_id,
    )
    return {"ok": True, "acceptance_evidence": export}


@router.post(ROUTE_ACCEPTANCE_EVIDENCE_EXPORT)
async def build_legal_acceptance_evidence_export(request: Request):
    ctx = _authz(request, "POST", ROUTE_ACCEPTANCE_EVIDENCE_EXPORT)
    body = await _json_body(request)
    test_run_id = str(body.get("test_run_id") or "").strip()
    if not test_run_id:
        raise HTTPException(status_code=400, detail="test_run_id is required")
    export = _store().build_acceptance_evidence_export(
        organization_id=_organization_id(ctx),
        workspace_id=str(ctx.workspace_id or "").strip() or None,
        test_run_id=test_run_id,
        code_revision=str(body.get("code_revision") or "").strip() or None,
        evidence_bundle_ref=str(body.get("evidence_bundle_ref") or "").strip()
        or None,
        automated_verification_summary=list(
            body.get("automated_verification_summary") or []
        ),
        screenshots=list(body.get("screenshots") or []),
        uat_cases=list(body.get("uat_cases") or []),
        user_signoff=dict(body.get("user_signoff") or {}),
        demo_operator=str(body.get("demo_operator") or "").strip() or None,
        domain_reviewer=str(body.get("domain_reviewer") or "").strip() or None,
        product_acceptor=str(body.get("product_acceptor") or "").strip() or None,
    )
    persisted = _store().persist_acceptance_evidence_export(export)
    return {"ok": True, "acceptance_evidence": export, "persisted_export": persisted}


@router.get(ROUTE_ACCEPTANCE_EVIDENCE_EXPORTS)
async def list_legal_acceptance_evidence_exports(request: Request):
    ctx = _authz(request, "GET", ROUTE_ACCEPTANCE_EVIDENCE_EXPORTS)
    exports = _store().list_acceptance_evidence_exports(
        organization_id=_organization_id(ctx),
    )
    return {"ok": True, "acceptance_evidence_exports": exports}


@router.post(ROUTE_REGISTER_SOURCE)
async def register_legal_source(request: Request):
    ctx = _authz(request, "POST", ROUTE_REGISTER_SOURCE)
    body = await _json_body(request)
    try:
        source = _store().register_source(
            organization_id=_organization_id(ctx),
            workspace_id=str(ctx.workspace_id or "").strip() or None,
            canonical_title=str(body.get("canonical_title") or "").strip(),
            issuer=str(body.get("issuer") or "").strip(),
            authority_tier=str(body.get("authority_tier") or "").strip(),
            jurisdiction=str(body.get("jurisdiction") or "").strip(),
            document_number=str(body.get("document_number") or "").strip() or None,
            expected_status=str(body.get("expected_status") or "").strip(),
            governance_state=str(body.get("governance_state") or "LISTED").strip(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "source": source}


@router.post(ROUTE_CREATE_VERSION)
async def create_legal_source_version(source_id: str, request: Request):
    _authz(request, "POST", ROUTE_CREATE_VERSION)
    body = await _json_body(request)
    try:
        version = _store().create_source_version(
            source_id=source_id,
            version_identity=str(body.get("version_identity") or "").strip(),
            source_hash=str(body.get("source_hash") or body.get("content_hash") or "").strip(),
            normalized_hash=str(body.get("normalized_hash") or "").strip() or None,
            effective_from=str(body.get("effective_from") or "").strip(),
            effective_to=str(body.get("effective_to") or "").strip() or None,
            promulgated_at=str(body.get("promulgated_at") or "").strip() or None,
            source_provenance=body.get("source_provenance") if isinstance(body.get("source_provenance"), dict) else {},
            lifecycle_state=str(body.get("lifecycle_state") or "REGISTERED").strip(),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "version": version.__dict__}


@router.post(ROUTE_PRESERVE_ARTIFACT)
async def preserve_legal_source_artifact(version_id: str, request: Request):
    _authz(request, "POST", ROUTE_PRESERVE_ARTIFACT)
    body = await _json_body(request)
    try:
        artifact = _store().preserve_artifact(
            version_id=version_id,
            artifact_kind=str(body.get("artifact_kind") or "official_artifact").strip(),
            uri=str(body.get("uri") or "").strip(),
            content=str(body.get("content_text") or "").encode("utf-8"),
            mime_type=str(body.get("mime_type") or "text/plain").strip(),
            storage_ref=str(body.get("storage_ref") or "").strip(),
            http_metadata=dict(body.get("http_metadata") or {}),
            redirect_metadata=list(body.get("redirect_metadata") or []),
            retrieved_at=str(body.get("retrieved_at") or "").strip() or None,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "artifact": artifact}


@router.post(ROUTE_UPLOAD_SOURCE_ARTIFACT)
async def upload_legal_source_artifact(
    version_id: str,
    request: Request,
    file: UploadFile = File(...),
    artifact_kind: str = Form("official_artifact_upload"),
    source_uri: str = Form(""),
):
    _authz(request, "POST", ROUTE_UPLOAD_SOURCE_ARTIFACT)
    filename = _safe_upload_filename(file.filename or "")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="file content is required")
    mime_type = str(file.content_type or "application/octet-stream")
    try:
        artifact = _store().preserve_artifact(
            version_id=version_id,
            artifact_kind=str(artifact_kind or "official_artifact_upload").strip(),
            uri=str(source_uri or f"uploads/{filename}").strip(),
            content=content,
            mime_type=mime_type,
            storage_ref=f"artifacts/legal/{version_id}/{filename}",
            http_metadata={
                "uploaded_filename": filename,
                "upload_size": len(content),
                "metadata_extraction": True,
            },
            redirect_metadata=[],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "artifact": artifact,
        "metadata_suggestions": _metadata_suggestions_from_upload(
            filename,
            content,
            mime_type,
        ),
    }


@router.post(ROUTE_MANUAL_FALLBACK)
async def ingest_legal_manual_fallback(source_id: str, request: Request):
    ctx = _authz(request, "POST", ROUTE_MANUAL_FALLBACK)
    body = await _json_body(request)
    try:
        result = GovernedManualFallbackAdapter().ingest(
            store=_store(),
            source_id=source_id,
            version_id=str(body.get("version_id") or "").strip(),
            governed_upload_ref=str(body.get("governed_upload_ref") or "").strip(),
            content=str(body.get("content_text") or "").encode("utf-8"),
            mime_type=str(body.get("mime_type") or "text/plain").strip(),
            storage_ref=str(body.get("storage_ref") or "").strip(),
            actor_ref=str(ctx.user_id),
            organization_id=_organization_id(ctx),
            correlation_id=_correlation_id(request, body) or "manual-fallback",
            provenance_reviewed=bool(body.get("provenance_reviewed")),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "result": result}


@router.post(ROUTE_REFRESH_SCHEDULE)
async def schedule_legal_source_refresh(source_id: str, request: Request):
    ctx = _authz(request, "POST", ROUTE_REFRESH_SCHEDULE)
    body = await _json_body(request)
    try:
        schedule = _store().schedule_refresh(
            source_id=source_id,
            interval_seconds=int(body.get("interval_seconds") or 0),
            next_check_at=str(body.get("next_check_at") or "").strip(),
            operator_ref=str(ctx.user_id),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "schedule": schedule}


@router.post(ROUTE_REFRESH_CHECK)
async def run_legal_source_refresh_check(source_id: str, request: Request):
    ctx = _authz(request, "POST", ROUTE_REFRESH_CHECK)
    body = await _json_body(request)
    try:
        check = _store().run_operator_refresh_check(
            source_id=source_id,
            checked_uri=str(body.get("checked_uri") or "").strip(),
            observed_content=str(body.get("observed_content_text") or "").encode("utf-8"),
            active_content_hash=str(body.get("active_content_hash") or "").strip() or None,
            operator_ref=str(ctx.user_id),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "refresh_check": check}


@router.post(ROUTE_SCAN_RUNS)
async def create_legal_scan_run(request: Request):
    ctx = _authz(request, "POST", ROUTE_SCAN_RUNS)
    body = await _json_body(request)
    scheduled_at = str(body.get("scheduled_at") or "").strip()
    if not scheduled_at:
        raise HTTPException(status_code=400, detail="scheduled_at is required")
    try:
        result = _monitor().execute_tick(
            organization_id=_organization_id(ctx),
            workspace_id=str(ctx.workspace_id or "").strip() or None,
            scheduled_at=scheduled_at,
            trigger="MANUAL",
            limit=int(body["limit"]) if body.get("limit") is not None else None,
            offset=int(body.get("offset") or 0),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, **result}


@router.get(ROUTE_SCAN_RUNS)
async def list_legal_scan_runs(request: Request, limit: int = 50, offset: int = 0):
    ctx = _authz(request, "GET", ROUTE_SCAN_RUNS)
    try:
        runs = _store().list_scan_runs(
            organization_id=_organization_id(ctx),
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "scan_runs": runs}


@router.get(ROUTE_SCAN_RUN)
async def get_legal_scan_run(scan_run_id: str, request: Request):
    ctx = _authz(request, "GET", ROUTE_SCAN_RUN)
    try:
        result = _store().get_scan_run(
            organization_id=_organization_id(ctx),
            scan_run_id=scan_run_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, **result}


@router.get(ROUTE_DUE_SOURCES)
async def list_legal_due_sources(
    request: Request,
    as_of: str,
    limit: int | None = None,
    offset: int = 0,
):
    ctx = _authz(request, "GET", ROUTE_DUE_SOURCES)
    try:
        due_sources = _monitor().list_due_sources(
            organization_id=_organization_id(ctx),
            as_of=as_of,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "due_sources": due_sources}


@router.get(ROUTE_SOURCE_STATUS)
async def get_legal_source_status(source_id: str, request: Request, as_of: str | None = None):
    ctx = _authz(request, "GET", ROUTE_SOURCE_STATUS)
    try:
        status = _monitor().get_status(
            organization_id=_organization_id(ctx),
            source_id=source_id,
            as_of=as_of,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "status": status}


@router.post(ROUTE_SCAN_ITEM_RETRY)
async def retry_legal_scan_item(scan_item_id: str, request: Request):
    ctx = _authz(request, "POST", ROUTE_SCAN_ITEM_RETRY)
    try:
        item = _store().retry_scan_item(
            organization_id=_organization_id(ctx),
            scan_item_id=scan_item_id,
            actor_ref=str(ctx.user_id),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "scan_item": item}


@router.get(ROUTE_CHANGE_CANDIDATES)
async def list_legal_change_candidates(
    request: Request,
    review_state: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    ctx = _authz(request, "GET", ROUTE_CHANGE_CANDIDATES)
    try:
        candidates = _store().list_change_candidates(
            organization_id=_organization_id(ctx),
            review_state=review_state,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "change_candidates": candidates}


@router.get(ROUTE_CHANGE_CANDIDATE_IMPACT)
async def get_legal_change_candidate_impact(candidate_id: str, request: Request):
    ctx = _authz(request, "GET", ROUTE_CHANGE_CANDIDATE_IMPACT)
    try:
        impact = _monitor().compute_impact(
            organization_id=_organization_id(ctx),
            candidate_id=candidate_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "impact": impact}


@router.post(ROUTE_ALERT_ACKNOWLEDGE)
async def acknowledge_legal_regulatory_alert(alert_id: str, request: Request):
    ctx = _authz(request, "POST", ROUTE_ALERT_ACKNOWLEDGE)
    body = await _json_body(request)
    try:
        acknowledgement = _store().acknowledge_regulatory_alert(
            organization_id=_organization_id(ctx),
            alert_id=alert_id,
            acknowledged_by=str(ctx.user_id),
            acknowledgement=dict(body.get("acknowledgement") or {}),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "acknowledgement": acknowledgement}


@router.post(ROUTE_CERTIFY)
async def certify_legal_object(request: Request):
    ctx = _authz(request, "POST", ROUTE_CERTIFY)
    body = await _json_body(request)
    try:
        event = _store().record_certification_event(
            object_ref=str(body.get("object_ref") or "").strip(),
            object_type=str(body.get("object_type") or "").strip(),
            decision=str(body.get("decision") or "").strip(),
            reviewer_ref=str(body.get("reviewer_ref") or ctx.user_id).strip(),
            reviewer_role=str(body.get("reviewer_role") or "").strip(),
            evidence_hash=str(body.get("evidence_hash") or "").strip(),
            event_payload=body,
        )
    except ValueError as exc:
        reason = str(exc)
        if reason == LEGAL_DEMO_ARTIFACT_NOT_CERTIFIABLE:
            failure = _record_failure(
                request=request,
                body=body,
                ctx=ctx,
                attempted_endpoint=ROUTE_CERTIFY,
                reason_code=reason,
            )
            raise HTTPException(status_code=409, detail=failure) from exc
        raise HTTPException(status_code=409, detail=reason) from exc
    return {"ok": True, "certification": event}


@router.post(ROUTE_ACTIVATE)
async def activate_legal_bundle(request: Request):
    ctx = _authz(request, "POST", ROUTE_ACTIVATE)
    body = await _json_body(request)
    activation_source_type = str(body.get("activation_source_type") or body.get("object_type") or "").strip()
    if activation_source_type in {"DemoPromotionRecord", "CuratedKnowledgeArtifact"}:
        failure = _record_failure(
            request=request,
            body=body,
            ctx=ctx,
            attempted_endpoint=ROUTE_ACTIVATE,
            reason_code=LEGAL_DEMO_APPROVAL_NOT_ACTIVATABLE,
        )
        raise HTTPException(status_code=409, detail=failure)
    try:
        event = _store().record_activation_event(
            source_registry_snapshot_id=str(body.get("source_registry_snapshot_id") or "").strip(),
            knowledge_context_version_id=str(body.get("knowledge_context_version_id") or "").strip(),
            authority_bundle_version_id=str(body.get("authority_bundle_version_id") or "").strip(),
            activation_performed=bool(body.get("activation_performed")),
            approver_refs=list(body.get("approver_refs") or []),
            content_hashes=dict(body.get("content_hashes") or {}),
            idempotency_key=str(body.get("idempotency_key") or "").strip(),
        )
    except ValueError as exc:
        reason = str(exc)
        if reason == LEGAL_ACTIVATION_EVIDENCE_REQUIRED:
            failure = _record_failure(
                request=request,
                body=body,
                ctx=ctx,
                attempted_endpoint=ROUTE_ACTIVATE,
                reason_code=reason,
            )
            raise HTTPException(status_code=409, detail=failure) from exc
        raise HTTPException(status_code=409, detail=reason) from exc
    return {"ok": True, "activation": event}


@router.post(ROUTE_RUNTIME_EXECUTE)
async def execute_legal_runtime(request: Request):
    ctx = _authz(request, "POST", ROUTE_RUNTIME_EXECUTE)
    body = await _json_body(request)
    try:
        result = _runtime().execute(
            organization_id=_organization_id(ctx),
            operation=str(body.get("operation") or "").strip(),
            issue_model_id=str(body.get("issue_model_id") or "").strip(),
            jurisdiction=str(body.get("jurisdiction") or "").strip(),
            as_of_date=str(body.get("as_of_date") or "").strip(),
            facts=dict(body.get("facts") or {}),
            requested_remedy=str(body.get("requested_remedy") or "").strip() or None,
            governed_defaults=dict(body.get("governed_defaults") or {}),
            renderer_contract_version=str(
                body.get("renderer_contract_version")
                or "legal_renderer_contract.v1"
            ).strip(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "legal_runtime": result}


@router.post(ROUTE_RUNTIME_REPLAY)
async def replay_legal_runtime(request: Request):
    _authz(request, "POST", ROUTE_RUNTIME_REPLAY)
    body = await _json_body(request)
    try:
        result = _runtime().replay(
            execution_manifest=dict(body.get("execution_manifest") or {}),
            facts=dict(body.get("facts") or {}),
            renderer_contract_version=str(
                body.get("renderer_contract_version")
                or "legal_renderer_contract.v1"
            ).strip(),
        )
    except ValueError as exc:
        detail = str(exc)
        status_code = 409 if LEGAL_RUNTIME_REPLAY_PIN_MISSING in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return {"ok": True, "legal_runtime": result}
