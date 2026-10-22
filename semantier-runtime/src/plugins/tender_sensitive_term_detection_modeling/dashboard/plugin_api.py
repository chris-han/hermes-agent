from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from contracts.tender_sensitive_term_detection_modeling_core import RuntimeContextDTO
from services.detection_evidence_service import StoreBackedDetectionEvidenceService
from services.governed_artifact_service import StoreBackedGovernedArtifactService
from services.knowledge_source_compilation_service import (
    StoreBackedKnowledgeSourceCompilationService,
)
from services.policy_rule_candidate_service import StoreBackedPolicyRuleCandidateService
from services.sensitive_labeling_report_service import StoreBackedSensitiveLabelingReportService
from .. import schemas, service

CONTRACT_VERSION = schemas.DASHBOARD_CONTRACT_VERSION
PLUGIN_VERSION = "0.1.0"

ROUTES = (
    {
        "method": "GET",
        "path": "/api/plugins/tender-sensitive-term-detection-modeling/v1/capabilities",
        "handler": "capabilities",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "GET",
        "path": "/api/plugins/tender-sensitive-term-detection-modeling/v1/compilation-runs/{run_id}",
        "handler": "get_compilation_run",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "GET",
        "path": "/api/plugins/tender-sensitive-term-detection-modeling/v1/candidates",
        "handler": "list_candidates",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "POST",
        "path": "/api/plugins/tender-sensitive-term-detection-modeling/v1/candidates/{candidate_id}/review",
        "handler": "review_candidate",
        "authorization": "governed_reviewer_capability",
    },
    {
        "method": "POST",
        "path": "/api/plugins/tender-sensitive-term-detection-modeling/v1/detections",
        "handler": "create_detection",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "GET",
        "path": "/api/plugins/tender-sensitive-term-detection-modeling/v1/detections/{run_id}",
        "handler": "get_detection",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "POST",
        "path": "/api/plugins/tender-sensitive-term-detection-modeling/v1/reports",
        "handler": "create_report",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "GET",
        "path": "/api/plugins/tender-sensitive-term-detection-modeling/v1/reports/{report_id}",
        "handler": "get_report",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "POST",
        "path": "/api/plugins/tender-sensitive-term-detection-modeling/v1/labeled-docx",
        "handler": "create_labeled_docx",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "GET",
        "path": "/api/plugins/tender-sensitive-term-detection-modeling/v1/labeled-docx/{artifact_id}",
        "handler": "get_labeled_docx",
        "authorization": "authenticated_tenant_member",
    },
    {
        "method": "POST",
        "path": "/api/plugins/tender-sensitive-term-detection-modeling/v1/replay/{replay_binding_id}/verify",
        "handler": "verify_replay",
        "authorization": "authenticated_tenant_member",
    },
)


def _envelope(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "plugin_id": schemas.TOOLSET_NAME,
        "plugin_version": PLUGIN_VERSION,
        "contract_version": CONTRACT_VERSION,
        **result,
    }


def _fail_closed(error_code: str, **details: Any) -> dict[str, Any]:
    return _envelope(
        {
            "status": "error",
            "error_code": error_code,
            "redaction_policy": {
                "raw_authority_rows": False,
                "approval_credentials": False,
                "source_excerpts": "authorized_anchored_spans_only",
                "sensitive_target_text": "minimized_in_list_payloads",
                "replay_manifests": "refs_and_status_by_default",
            },
            "details": details,
        }
    )


def capabilities() -> dict[str, Any]:
    return _envelope(
        {
            "status": "ok",
            "capabilities": [tool["name"] for tool in schemas.TOOLS],
            "routes": list(ROUTES),
            "redaction_policy": {
                "raw_authority_rows": False,
                "approval_credentials": False,
                "source_excerpts": "authorized_anchored_spans_only",
                "sensitive_target_text": "minimized_in_list_payloads",
                "replay_manifests": "refs_and_status_by_default",
            },
        }
    )


def _runtime_context(ctx: Any) -> RuntimeContextDTO:
    return RuntimeContextDTO(
        organization_id=str(getattr(ctx, "organization_id", "") or ""),
        workspace_id=getattr(ctx, "workspace_id", None),
        actor_id=str(getattr(ctx, "user_id", "") or ""),
        actor_role=str(getattr(ctx, "member_role", "") or ""),
    )


def get_compilation_run(*, ctx: Any, run_id: str) -> dict[str, Any]:
    if ctx is None:
        return _fail_closed("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED", run_id=run_id)
    run = StoreBackedKnowledgeSourceCompilationService().get_run(run_id)
    if run is None or run.get("organization_id") != str(getattr(ctx, "organization_id", "") or ""):
        return _fail_closed("COMPILATION_RUN_NOT_FOUND", run_id=run_id)
    if getattr(ctx, "workspace_id", None) and run.get("workspace_id") not in {None, getattr(ctx, "workspace_id", None)}:
        return _fail_closed("COMPILATION_RUN_NOT_FOUND", run_id=run_id)
    anchors = StoreBackedKnowledgeSourceCompilationService().list_anchors(run_id)
    return _envelope({"status": "ok", "compilation_run": run, "anchors": anchors})


def list_candidates(*, ctx: Any, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    if ctx is None:
        return _fail_closed("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED", filters=filters or {})
    filters = filters or {}
    candidates = StoreBackedPolicyRuleCandidateService().list_candidates(
        rule_family=filters.get("rule_family"),
        candidate_state=filters.get("candidate_state"),
    )
    org_id = str(getattr(ctx, "organization_id", "") or "")
    workspace_id = getattr(ctx, "workspace_id", None)
    scoped = [
        candidate
        for candidate in candidates
        if candidate.get("organization_id") == org_id
        and (
            workspace_id is None
            or candidate.get("proposed_workspace_id") in {None, workspace_id}
        )
    ]
    return _envelope({"status": "ok", "candidates": scoped})


def review_candidate(*, ctx: Any, candidate_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = service.review_sensitive_lexicon_candidates(
        args={**payload, "candidate_id": candidate_id},
        ctx=ctx,
    )
    return _envelope({"status": "ok", "candidate": result})


def create_detection(*, ctx: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return _envelope({"status": "ok", "detection": service.label_tender_document(args=payload, ctx=ctx)})


def get_detection(*, ctx: Any, run_id: str) -> dict[str, Any]:
    if ctx is None:
        return _fail_closed("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED", run_id=run_id)
    runtime_context = _runtime_context(ctx)
    try:
        evidence = StoreBackedDetectionEvidenceService().get_detection_evidence(
            run_id,
            runtime_context,
        )
    except Exception:
        return _fail_closed("DETECTION_RUN_NOT_FOUND", run_id=run_id)
    return _envelope({"status": "ok", "detection": asdict(evidence)})


def create_report(payload: dict[str, Any]) -> dict[str, Any]:
    return _envelope({"status": "ok", "report": service.generate_sensitive_labeling_report(args=payload)})


def get_report(*, ctx: Any, report_id: str) -> dict[str, Any]:
    if ctx is None:
        return _fail_closed("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED", report_id=report_id)
    report = StoreBackedSensitiveLabelingReportService().get_report(report_id)
    if report is None or report.get("organization_id") != str(getattr(ctx, "organization_id", "") or ""):
        return _fail_closed("REPORT_NOT_FOUND", report_id=report_id)
    return _envelope({"status": "ok", "report": report})


def create_labeled_docx(*, ctx: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return _envelope({"status": "ok", "artifact": service.generate_labeled_docx(args=payload, ctx=ctx)})


def get_labeled_docx(*, ctx: Any, artifact_id: str) -> dict[str, Any]:
    if ctx is None:
        return _fail_closed("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED", artifact_id=artifact_id)
    artifact = StoreBackedGovernedArtifactService().get_labeled_docx_derivative(
        artifact_id,
        _runtime_context(ctx),
    )
    if artifact is None:
        return _fail_closed("LABELED_DOCX_NOT_FOUND", artifact_id=artifact_id)
    return _envelope({"status": "ok", "artifact": artifact})


def verify_replay(*, ctx: Any, replay_binding_id: str) -> dict[str, Any]:
    result = service.verify_sensitive_labeling_replay(
        args={"replay_binding_id": replay_binding_id},
        ctx=ctx,
    )
    return _envelope({"status": "ok", "replay": result})


