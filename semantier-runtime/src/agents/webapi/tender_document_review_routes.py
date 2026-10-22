from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from agents.replay import build_tender_detection_replay_bundle
from agents.hermes_current_agent_llm import HermesCurrentAgentStructuredLlm
from agents.route_policy import classify_authorization_class
from agents.runtime_feedback_to_candidate import convert_tender_feedback_to_candidate_delta
from agents.plugins.tender_document_review import validate_graph_release_focus
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from plugins.tender_sensitive_term_detection_modeling.dashboard import plugin_api as tender_sensitive_plugin_api
from storage.semantic_feedback_event_store import SemanticFeedbackEventStore
from storage.tender_detection_report_store import TenderDetectionReportStore
from storage.tender_detection_run_store import TenderDetectionRunStore
from runtime_paths import sqlite_db_path
from services.runtime_semantic_inspection_service import RuntimeSemanticInspectionService

router = APIRouter(tags=["webapi-gateway"])

@router.post("/api/tender-document-review/graph-focus")
async def validate_tender_graph_focus(request: Request):
    route_path = "/api/tender-document-review/graph-focus"
    _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        return {"ok": True, "focus": validate_graph_release_focus(
            finding=dict(body.get("finding") or {}),
            accepted_release_hash=str(body.get("accepted_release_hash") or ""),
        )}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _authz(request: Request, method: str, path: str):
    authz_class = classify_authorization_class(method, path)
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    return _require_route_authorization(
        request,
        authorization_class=authz_class,
        route_hint=f"{method} {path}",
    )


def _can_view_audit_details(ctx) -> bool:
    return bool(ctx.authenticated)


def _first_text(*values) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _semantic_action_for_disposition(disposition: str) -> str | None:
    return {
        "accepted": "confirm",
        "edited": "change",
        "rejected": "dismiss",
    }.get(disposition)


def _semantic_feedback_event_for_disposition(
    *,
    ctx,
    run: dict,
    finding: dict,
    disposition: dict,
    body: dict,
) -> dict | None:
    action = _semantic_action_for_disposition(str(disposition["disposition"]))
    if action is None:
        return None
    workspace_id = _first_text(ctx.workspace_id, run.get("workspace_id"))
    workspace_root = getattr(ctx, "workspace_root", None)
    if not workspace_id or workspace_root is None:
        raise ValueError("semantic feedback requires workspace context")
    project_metadata = dict(run.get("project_metadata") or {})
    graph_version_ref = _first_text(
        finding.get("source_graph_version"),
        project_metadata.get("selected_graph_version_ref"),
        project_metadata.get("graph_version_ref"),
        project_metadata.get("expected_graph_version"),
        finding.get("source_graph_release_hash"),
        run.get("activation_set_snapshot_id"),
    )
    evidence_ref = _first_text(
        finding.get("target_evidence_ref"),
        finding.get("target_anchor_ref"),
        finding.get("resolver_evidence_ref"),
        finding.get("content_hash"),
    )
    original_result_ref = _first_text(
        finding.get("match_id"),
        finding.get("content_hash"),
        finding.get("finding_id"),
    )
    profile_ref = _first_text(
        project_metadata.get("profile_ref"),
        project_metadata.get("profileRef"),
        "profile://legacy/tender-document-review",
    )
    profile_version = _first_text(
        project_metadata.get("profile_version"),
        project_metadata.get("profileVersion"),
        "1",
    )
    skill_ref = _first_text(
        project_metadata.get("skill_ref"),
        project_metadata.get("skillRef"),
        "tender-sensitive-term-detection-modeling@compat",
    )
    if not graph_version_ref or not evidence_ref or not original_result_ref:
        raise ValueError("semantic feedback event missing finding provenance")
    justification = _first_text(
        body.get("justification"),
        body.get("rejectionRationale"),
        body.get("rejection_rationale"),
        body.get("editedReplacement"),
        body.get("edited_replacement"),
    )
    corrected_result = None
    if action == "change":
        edited_replacement = _first_text(
            body.get("editedReplacement"),
            body.get("edited_replacement"),
        )
        corrected_result = {
            "suggested_replacement": edited_replacement,
            "review_disposition": "edited",
        }
    reason_code = _first_text(body.get("reasonCode"), body.get("reason_code"))
    if action == "dismiss" and reason_code is None:
        reason_code = "human_dismissed"
    if action == "change" and reason_code is None:
        reason_code = "human_changed"
    event = SemanticFeedbackEventStore(
        workspace_root,
        workspace_id=workspace_id,
    ).create_event(
        run_id=str(run["run_id"]),
        finding_id=str(finding["finding_id"]),
        action=action,
        original_result_ref=original_result_ref,
        corrected_result=corrected_result,
        reason_code=reason_code,
        justification=justification,
        evidence_ref=evidence_ref,
        graph_version_ref=graph_version_ref,
        ontology_version_ref=_first_text(
            project_metadata.get("ontology_version_ref"),
            project_metadata.get("ontologyVersionRef"),
        ),
        profile_ref=profile_ref,
        profile_version=profile_version,
        skill_ref=skill_ref,
        derivation_refs=[
            ref
            for ref in (
                _first_text(finding.get("source_graph_rule_id")),
                _first_text(finding.get("triggered_rule_version_id")),
                _first_text(finding.get("resolver_evidence_ref")),
            )
            if ref
        ],
        reviewer=str(ctx.user_id),
    )
    return event.as_dict()


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="JSON body required") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object body required")
    return body


def _eos_path() -> Path:
    return Path(
        os.environ.get(
            "SEMANTIER_EOS_DB_PATH",
            str(sqlite_db_path("eos.db", ".semantier/eos.db")),
        )
    )


def _runtime_semantic_inspection_service(ctx, *, read_only: bool = False) -> RuntimeSemanticInspectionService:
    requested = Path(ctx.workspace_root).resolve()
    workspace_id = str(ctx.workspace_id)
    runtime_root = (
        requested.parent.parent
        if requested.name == workspace_id and requested.parent.name == "workspaces"
        else requested
    )
    return RuntimeSemanticInspectionService(
        runtime_root=runtime_root,
        workspace_root=requested,
        db_path=_eos_path(),
        actor_id=str(ctx.user_id),
        generator=None if read_only else HermesCurrentAgentStructuredLlm(
            hermes_home=Path(ctx.hermes_home), require_primary_binding=False),
    )


@router.post("/api/tender-document-review/detections")
async def create_tender_document_detection(request: Request):
    route_path = "/api/tender-document-review/detections"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        metadata = dict(body.get("projectMetadata") or body.get("project_metadata") or {})
        metadata.pop("semantic_runtime", None)
        semantic_runtime = None
        if body.get("semanticRuntime") is True:
            file_ref = str(body.get("fileRef") or body.get("file_ref") or "").strip()
            if not file_ref:
                raise ValueError("semantic runtime inspection requires fileRef")
            semantic_runtime = await _runtime_semantic_inspection_service(ctx).inspect(
                organization_id=str(ctx.organization_id),
                workspace_id=str(ctx.workspace_id),
                normalized_document_artifact_ref=file_ref,
            )
            metadata["semantic_runtime"] = semantic_runtime
        body = {**body, "projectMetadata": metadata}
        result = tender_sensitive_plugin_api.create_detection(ctx=ctx, payload=body)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "ok": True,
        "run": result["detection"],
        "semantic_runtime": semantic_runtime,
    }


@router.post("/api/tender-document-review/runs/{run_id}/rerun")
async def rerun_tender_document_detection(run_id: str, request: Request):
    route_path = "/api/tender-document-review/runs/{run_id}/rerun"
    ctx = _authz(request, "POST", route_path)
    prior = TenderDetectionRunStore().get_run(run_id)
    if prior is None or prior["organization_id"] != ctx.organization_id:
        raise HTTPException(status_code=404, detail="run not found")
    body = await _json_body(request)
    corrected_text = str(body.get("documentText") or body.get("document_text") or "").strip()
    if not corrected_text:
        raise HTTPException(status_code=400, detail="documentText is required for rerun")
    project_metadata = dict(prior["project_metadata"] or {})
    project_metadata.update(body.get("projectMetadata") or body.get("project_metadata") or {})
    project_metadata["parent_run_id"] = run_id
    payload = {
        "tenderDocumentId": prior["tender_document_id"],
        "documentText": corrected_text,
        "organizationId": prior["organization_id"],
        "workspaceId": prior["workspace_id"],
        "projectMetadata": project_metadata,
        "reviewPurpose": prior["review_purpose"],
        "callerRole": prior["caller_role"],
        "requestedRuleFamilies": prior["requested_rule_families"],
    }
    try:
        result = tender_sensitive_plugin_api.create_detection(ctx=ctx, payload=payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "run": result["detection"]}


@router.get("/api/tender-document-review/runs/{run_id}")
async def get_tender_document_detection_run(run_id: str, request: Request):
    route_path = "/api/tender-document-review/runs/{run_id}"
    ctx = _authz(request, "GET", route_path)
    run = TenderDetectionRunStore().get_run(run_id)
    if run is None or run["organization_id"] != ctx.organization_id or run.get("workspace_id") != ctx.workspace_id:
        raise HTTPException(status_code=404, detail="run not found")
    return {"ok": True, "run": run}


@router.post("/api/tender-document-review/runs/{run_id}/findings/{finding_id}/disposition")
async def record_tender_document_finding_disposition(
    run_id: str,
    finding_id: str,
    request: Request,
):
    route_path = "/api/tender-document-review/runs/{run_id}/findings/{finding_id}/disposition"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    store = TenderDetectionRunStore()
    run = store.get_run(run_id)
    if run is None or run["organization_id"] != ctx.organization_id or run.get("workspace_id") != ctx.workspace_id:
        raise HTTPException(status_code=404, detail="run not found")
    finding = next(
        (item for item in run["findings"] if item["finding_id"] == finding_id),
        None,
    )
    if finding is None:
        raise HTTPException(status_code=404, detail="finding not found")
    try:
        disposition = store.record_disposition(
            run_id=run_id,
            finding_id=finding_id,
            disposition=str(body.get("disposition") or ""),
            edited_replacement=body.get("editedReplacement") or body.get("edited_replacement"),
            rejection_rationale=body.get("rejectionRationale") or body.get("rejection_rationale"),
            decided_by=str(ctx.user_id),
        )
        semantic_feedback_event = _semantic_feedback_event_for_disposition(
            ctx=ctx,
            run=run,
            finding=finding,
            disposition=disposition,
            body=body,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "disposition": disposition,
        "semantic_feedback_event": semantic_feedback_event,
    }


@router.post("/api/tender-document-review/runs/{run_id}/findings/{finding_id}/feedback")
async def record_tender_document_finding_feedback(
    run_id: str,
    finding_id: str,
    request: Request,
):
    route_path = "/api/tender-document-review/runs/{run_id}/findings/{finding_id}/feedback"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    try:
        feedback = convert_tender_feedback_to_candidate_delta(
            organization_id=str(ctx.organization_id),
            workspace_id=ctx.workspace_id,
            run_id=run_id,
            finding_id=finding_id,
            feedback_type=str(body.get("feedbackType") or body.get("feedback_type") or ""),
            user_disposition=dict(body.get("userDisposition") or body.get("user_disposition") or {}),
            escalation_outcome=body.get("escalationOutcome") or body.get("escalation_outcome"),
            reviewer_notes=body.get("reviewerNotes") or body.get("reviewer_notes"),
            edited_remediation=body.get("editedRemediation") or body.get("edited_remediation"),
            created_by=str(ctx.user_id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "feedback": feedback}


@router.post("/api/tender-document-review/runs/{run_id}/feedback")
async def record_tender_document_run_feedback(run_id: str, request: Request):
    route_path = "/api/tender-document-review/runs/{run_id}/feedback"
    ctx = _authz(request, "POST", route_path)
    body = await _json_body(request)
    if str(body.get("feedbackType") or body.get("feedback_type") or "") != "false_negative":
        raise HTTPException(status_code=400, detail="run-level feedbackType must be false_negative")
    source_span = body.get("sourceSpan") or body.get("source_span")
    if not isinstance(source_span, dict):
        raise HTTPException(status_code=400, detail="sourceSpan is required")
    try:
        feedback = convert_tender_feedback_to_candidate_delta(
            organization_id=str(ctx.organization_id),
            workspace_id=ctx.workspace_id,
            run_id=run_id,
            finding_id=None,
            feedback_type="false_negative",
            user_disposition={},
            escalation_outcome=None,
            reviewer_notes=body.get("reviewerNotes") or body.get("reviewer_notes"),
            edited_remediation=None,
            created_by=str(ctx.user_id),
            source_span_override=source_span,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "feedback": feedback}


@router.post("/api/tender-document-review/runs/{run_id}/report")
async def create_tender_document_report(run_id: str, request: Request):
    route_path = "/api/tender-document-review/runs/{run_id}/report"
    ctx = _authz(request, "POST", route_path)
    run = TenderDetectionRunStore().get_run(run_id)
    if run is None or run["organization_id"] != ctx.organization_id or run.get("workspace_id") != ctx.workspace_id:
        raise HTTPException(status_code=404, detail="run not found")
    result = tender_sensitive_plugin_api.create_report(
        {"run_id": run_id, "created_by": str(ctx.user_id)}
    )
    report = result["report"]
    return {"ok": True, "report": report}


@router.post("/api/tender-document-review/runs/{run_id}/labeled-docx")
async def create_tender_document_labeled_docx(run_id: str, request: Request):
    route_path = "/api/tender-document-review/runs/{run_id}/labeled-docx"
    ctx = _authz(request, "POST", route_path)
    run = TenderDetectionRunStore().get_run(run_id)
    if run is None or run["organization_id"] != ctx.organization_id or run.get("workspace_id") != ctx.workspace_id:
        raise HTTPException(status_code=404, detail="run not found")
    try:
        result = tender_sensitive_plugin_api.create_labeled_docx(
            ctx=ctx,
            payload={"run": run, "created_by": str(ctx.user_id)},
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    derivative = result["artifact"]
    return {"ok": True, "derivative": derivative}


@router.get("/api/tender-document-review/reports/{report_id}")
async def get_tender_document_report(report_id: str, request: Request):
    route_path = "/api/tender-document-review/reports/{report_id}"
    ctx = _authz(request, "GET", route_path)
    report = TenderDetectionReportStore().get_report(report_id)
    if report is None or report["organization_id"] != ctx.organization_id:
        raise HTTPException(status_code=404, detail="report not found")
    run = TenderDetectionRunStore().get_run(report['run_id'])
    if run is None or run.get('workspace_id') != ctx.workspace_id:
        raise HTTPException(status_code=404, detail='report not found')
    if not _can_view_audit_details(ctx):
        report = dict(report)
        report.pop("replay_binding", None)
    return {"ok": True, "report": report}


@router.get("/api/tender-document-review/runs/{run_id}/replay")
async def get_tender_document_replay_bundle(run_id: str, request: Request):
    route_path = "/api/tender-document-review/runs/{run_id}/replay"
    ctx = _authz(request, "GET", route_path)
    run = TenderDetectionRunStore().get_run(run_id)
    if run is None or run['organization_id'] != ctx.organization_id or run.get('workspace_id') != ctx.workspace_id:
        raise HTTPException(status_code=404, detail='run not found')
    try:
        bundle = build_tender_detection_replay_bundle(
            run_id=run_id,
            organization_id=str(ctx.organization_id),
            include_audit_details=_can_view_audit_details(ctx),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "bundle": bundle}
