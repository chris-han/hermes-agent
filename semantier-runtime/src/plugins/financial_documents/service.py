from __future__ import annotations

import json
from typing import Any

from plugins.document_extraction.storage import BoundSession, bound_session, resolve_session_ref, utc_now_iso, write_json_artifact, sha256_payload

from . import schemas
from .adapters import (
    china_digital_invoice_v1,
    china_vat_ordinary_v1,
    china_vat_special_v1,
    generic_cloud_usage_v1,
    generic_subscription_v1,
    google_workspace_v1,
    microsoft_azure_v1,
)
from .adapters.base import canonical_text


def _select_adapter(canonical: dict[str, Any], hint: str | None, jurisdiction: str | None):
    text = canonical_text(canonical).lower()
    if jurisdiction and jurisdiction.upper() in {"CN", "CN-VAT", "CHINA"}:
        if "专用发票" in text:
            return china_vat_special_v1
        if "电子发票" in text or "全电" in text:
            return china_digital_invoice_v1
        return china_vat_ordinary_v1
    if "google workspace" in text or "google cloud" in text:
        return google_workspace_v1
    if "microsoft azure" in text or "azure" in text:
        return microsoft_azure_v1
    if "usage" in text or "meter" in text:
        return generic_cloud_usage_v1
    return generic_subscription_v1


def extract_live(args: dict[str, Any], *, trusted_session: BoundSession | None = None) -> dict[str, Any]:
    if trusted_session is not None:
        if not isinstance(trusted_session, BoundSession):
            raise TypeError('TRUSTED_BOUND_SESSION_REQUIRED')
        if args.get('session_id') and args['session_id'] != trusted_session.session_id:
            raise RuntimeError('SESSION_CLAIM_MISMATCH')
    bound = trusted_session if trusted_session is not None else bound_session(args)
    canonical_ref = str(args.get("canonical_document_artifact_ref") or "")
    if not canonical_ref:
        raise RuntimeError("CANONICAL_DOCUMENT_ARTIFACT_REF_REQUIRED")
    canonical_path = resolve_session_ref(bound, canonical_ref)
    artifact = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical = artifact.get("document")
    if not isinstance(canonical, dict) or canonical.get("schema_version") not in {
        "canonical_document.v1",
        "canonical_document.v2",
    }:
        raise RuntimeError("INVALID_CANONICAL_DOCUMENT")
    adapter = _select_adapter(canonical, args.get("document_type_hint"), args.get("jurisdiction_hint"))
    result = adapter.extract(canonical, preferred_currency=args.get("preferred_currency"))
    review_tasks = [
        {"field": field, "reason": "missing_or_conflicting_extraction"}
        for field in result.get("unresolved_fields", [])
    ]
    payload = {
        "document": result["document"],
        "findings": result.get("findings", []),
        "unresolved_fields": result.get("unresolved_fields", []),
        "canonical_document_artifact_ref": canonical_ref,
    }
    financial_ref, _ = write_json_artifact(
        bound,
        area="financial_documents",
        name=f"invoice-{sha256_payload(payload).removeprefix('sha256:')}",
        payload=payload,
    )
    return {
        "status": result["status"],
        "document": result["document"],
        "findings": result.get("findings", []),
        "unresolved_fields": result.get("unresolved_fields", []),
        "review_tasks": review_tasks,
        "artifacts": {
            "canonical_document_artifact_ref": canonical_ref,
            "financial_extraction_artifact_ref": financial_ref,
            "model_artifact_refs": [],
        },
        "completed_at": utc_now_iso(),
    }


def extract_replay(args: dict[str, Any]) -> dict[str, Any]:
    bound = bound_session(args)
    financial_ref = str(args.get("financial_extraction_artifact_ref") or "")
    if not financial_ref:
        raise RuntimeError("FINANCIAL_EXTRACTION_ARTIFACT_REF_REQUIRED")
    artifact = json.loads(resolve_session_ref(bound, financial_ref).read_text(encoding="utf-8"))
    document = artifact.get("document")
    if not isinstance(document, dict):
        raise RuntimeError("INVALID_FINANCIAL_EXTRACTION_ARTIFACT")
    pins = args.get("pins") or {}
    expected_schema = pins.get("invoice_schema_version")
    if expected_schema and expected_schema != document.get("schema_version"):
        raise RuntimeError("INVOICE_SCHEMA_PIN_MISMATCH")
    return {
        "status": "review_required" if artifact.get("unresolved_fields") else "admissible",
        "document": document,
        "findings": artifact.get("findings", []),
        "unresolved_fields": artifact.get("unresolved_fields", []),
        "review_tasks": [
            {"field": field, "reason": "missing_or_conflicting_extraction"}
            for field in artifact.get("unresolved_fields", [])
        ],
        "artifacts": {
            "canonical_document_artifact_ref": artifact.get("canonical_document_artifact_ref") or args.get("canonical_document_artifact_ref"),
            "financial_extraction_artifact_ref": financial_ref,
            "model_artifact_refs": list(args.get("model_artifact_refs") or []),
        },
        "completed_at": utc_now_iso(),
    }
