from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from pathlib import Path
from uuid import uuid4

from contracts.tender_sensitive_term_detection_modeling_core import (
    DetectionRunCreateDTO,
    OccurrenceFindingDTO,
    RuntimeConflictDTO,
    RuntimeContextDTO,
    validate_ascii_identifier,
)
from eos.artifact_hashing import canonical_json, content_hash
from services.activation_snapshot_resolver_service import (
    StoreBackedActivationSnapshotResolverService,
)
from services.detection_evidence_service import StoreBackedDetectionEvidenceService
from services.generic_replay_envelope_service import StoreBackedGenericReplayEnvelopeService
from services.governed_artifact_service import StoreBackedGovernedArtifactService
from services.governed_document_service import load_verified_normalized_document_artifact
from services.tender_precedent_observed_expression_lookup_service import (
    StoreBackedTenderPrecedentObservedExpressionLookupService,
)
from agents.governed_tool_context import resolve_authenticated_tool_context
from . import compiler, detector, docx_output, report


@dataclass(frozen=True)
class TenderDetectionRequestDTO:
    tender_document_id: str
    document_text: str
    organization_id: str
    workspace_id: str | None
    project_metadata: dict[str, Any]
    review_purpose: str
    caller_role: str
    requested_rule_families: list[str]
    local_rules: list[dict[str, Any]]
    expected_graph_version: str | None = None
    expected_accepted_release_id: str | None = None
    canonical_document: dict[str, Any] | None = None
    target_session_id: str | None = None
    target_source_hash: str | None = None
    target_normalized_document_artifact_ref: str | None = None
    target_normalized_document_artifact_hash: str | None = None


def _canonical_document_text(document: dict[str, Any]) -> str:
    return "\n".join(str(element.get("text") or "") for element in document.get("elements") or [])


def _canonical_document_kind(document: dict[str, Any]) -> str:
    elements = document.get("elements") or []
    if any(str(element.get("kind") or "").startswith("pdf_") for element in elements):
        return "pdf"
    if elements:
        return "docx"
    return "canonical_source_ir"


def _request_from_payload(payload: dict[str, Any], *, ctx: Any) -> TenderDetectionRequestDTO:
    document_text = str(
        payload.get("documentText") or payload.get("document_text") or ""
    ).strip()
    file_ref = str(payload.get("fileRef") or payload.get("file_ref") or "").strip()
    canonical_document: dict[str, Any] | None = None
    target_session_id = None
    target_source_hash = None
    target_normalized_document_artifact_hash = None
    target_normalized_document_artifact_ref = None
    if not document_text and file_ref:
        target_session_id = str(payload.get('sessionId') or payload.get('session_id') or os.environ.get('HERMES_SESSION_ID') or os.environ.get('SESSION_ID') or '').strip()
        home = getattr(ctx, 'hermes_home', None)
        if not home or not target_session_id:
            raise ValueError('governed workspace and session binding required for tender artifact')
        from runtime_paths import workspace_session_runs_root_from_hermes_home
        session_root = workspace_session_runs_root_from_hermes_home(Path(home), target_session_id).parent.resolve()
        if Path(file_ref).is_absolute():
            raise ValueError('fileRef must be governed session-relative')
        try:
            (session_root / file_ref).resolve().relative_to(session_root)
        except ValueError as exc:
            raise ValueError('fileRef must remain inside the governed session') from exc
        loaded = load_verified_normalized_document_artifact(
            normalized_document_artifact_ref=file_ref,
            hermes_home=getattr(ctx, "hermes_home", None),
            session_id=target_session_id,
        )
        canonical_document = loaded["document"]
        if canonical_document.get("schema_version") != "canonical_document.v2":
            raise ValueError("canonical_document.v2 is required for fileRef occurrence labeling")
        target_source_hash = loaded["source_hash"]
        target_normalized_document_artifact_hash = loaded["artifact_hash"]
        target_normalized_document_artifact_ref = file_ref
        document_text = _canonical_document_text(canonical_document)
    if not document_text:
        raise ValueError("documentText is required")
    org_id = str(
        getattr(ctx, "organization_id", None)
        or payload.get("organizationId")
        or payload.get("organization_id")
        or ""
    ).strip()
    if not org_id:
        raise ValueError("organization context required")
    requested = (
        payload.get("requestedRuleFamilies")
        or payload.get("requested_rule_families")
        or ["tender_compliance"]
    )
    if not isinstance(requested, list) or not requested:
        raise ValueError("requestedRuleFamilies must be a non-empty list")
    local_rules = payload.get("localRules") or payload.get("local_rules") or []
    if not isinstance(local_rules, list):
        raise ValueError("localRules must be a list when provided")
    project_metadata = payload.get("projectMetadata") or payload.get("project_metadata") or {}
    if not isinstance(project_metadata, dict):
        raise ValueError("projectMetadata must be an object")
    expected_graph_version = str(
        payload.get("expectedGraphVersion")
        or payload.get("expected_graph_version")
        or project_metadata.get("expectedGraphVersion")
        or project_metadata.get("expected_graph_version")
        or ""
    ).strip() or None
    expected_accepted_release_id = str(
        payload.get("expectedAcceptedReleaseId")
        or payload.get("expected_accepted_release_id")
        or project_metadata.get("expectedAcceptedReleaseId")
        or project_metadata.get("expected_accepted_release_id")
        or ""
    ).strip() or None
    if expected_graph_version:
        project_metadata = {**project_metadata, "expected_graph_version": expected_graph_version}
    if expected_accepted_release_id:
        project_metadata = {
            **project_metadata,
            "expected_accepted_release_id": expected_accepted_release_id,
        }
    workspace_id = str(
        getattr(ctx, "workspace_id", None)
        or payload.get("workspaceId")
        or payload.get("workspace_id")
        or ""
    ).strip() or None
    return TenderDetectionRequestDTO(
        tender_document_id=str(
            payload.get("tenderDocumentId")
            or payload.get("tender_document_id")
            or f"tdoc_{uuid4().hex}"
        ).strip(),
        document_text=document_text,
        organization_id=org_id,
        workspace_id=workspace_id,
        project_metadata=project_metadata,
        review_purpose=str(
            payload.get("reviewPurpose")
            or payload.get("review_purpose")
            or "pre_release_check"
        ).strip(),
        caller_role=str(
            payload.get("callerRole")
            or payload.get("caller_role")
            or getattr(ctx, "member_role", None)
            or "member"
        ).strip(),
        requested_rule_families=[str(item).strip() for item in requested if str(item).strip()],
        local_rules=[item for item in local_rules if isinstance(item, dict)],
        expected_graph_version=expected_graph_version,
        expected_accepted_release_id=expected_accepted_release_id,
        canonical_document=canonical_document,
        target_session_id=target_session_id,
        target_source_hash=target_source_hash,
        target_normalized_document_artifact_ref=target_normalized_document_artifact_ref,
        target_normalized_document_artifact_hash=target_normalized_document_artifact_hash,
    )


def compile_sensitive_lexicon(*, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    ctx = kwargs.get("ctx") or kwargs.get("request_context") or resolve_authenticated_tool_context()
    if ctx is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")

    organization_id = validate_ascii_identifier(
        str(getattr(ctx, "organization_id", "") or "").strip(),
        "organization_id",
    )
    actor_id = validate_ascii_identifier(
        str(getattr(ctx, "user_id", "") or "").strip(),
        "created_by",
    )
    workspace_id = str(getattr(ctx, "workspace_id", "") or "").strip() or None
    if workspace_id is not None:
        workspace_id = validate_ascii_identifier(workspace_id, "workspace_id")

    normalized_ref = str(
        args.get("normalized_document_artifact_ref")
        or args.get("normalizedDocumentArtifactRef")
        or ""
    ).strip()
    if not normalized_ref:
        raise ValueError("normalized_document_artifact_ref is required")
    knowledge_source_id = validate_ascii_identifier(
        str(args.get("knowledge_source_id") or args.get("knowledgeSourceId") or "").strip(),
        "knowledge_source_id",
    )
    source_ref = str(args.get("source_ref") or args.get("sourceRef") or "").strip()
    if not source_ref:
        raise ValueError("source_ref is required")

    compiler_args = {
        key: args[key]
        for key in (
            "proposed_workspace_scope",
            "proposed_workspace_id",
            "source_version",
            "db_path",
            "compiler_profile_version",
        )
        if key in args
    }
    return compiler.compile_sensitive_lexicon(
        normalized_document_artifact_ref=normalized_ref,
        organization_id=organization_id,
        workspace_id=workspace_id,
        knowledge_source_id=knowledge_source_id,
        source_ref=source_ref,
        created_by=actor_id,
        **compiler_args,
    )


def review_sensitive_lexicon_candidates(*, args: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    candidate_id = str(args.get("candidate_id") or "").strip()
    action = str(args.get("action") or "request_review").strip().lower()
    if not candidate_id:
        raise ValueError("candidate_id is required")
    policy = _candidate_review_governance_policy()
    ctx = (
        _kwargs.get("ctx")
        or _kwargs.get("request_context")
        or resolve_authenticated_tool_context()
    )
    runtime_context = _resolve_candidate_review_runtime_context(
        ctx=ctx,
        args=args,
        action=action,
        policy=policy,
    )
    candidate_service = _kwargs.get("candidate_service") or StoreBackedPolicyRuleCandidateService()
    if action not in {"edit", "approve", "get", "request_review"}:
        raise ValueError("unsupported candidate review action")
    if action == "edit":
        edit_args = dict(args)
        if not edit_args.get("actor_authority") and policy.default_actor_authority:
            edit_args["actor_authority"] = policy.default_actor_authority
        if not edit_args.get("review_comments") and policy.default_review_comments:
            edit_args["review_comments"] = policy.default_review_comments
        return candidate_service.record_reviewer_edit(candidate_id, edit_args, runtime_context)
    if action == "approve":
        approval_args = _approval_payload_with_policy_defaults(args, policy=policy)
        return candidate_service.approve(candidate_id, approval_args, runtime_context)
    if action == "get":
        return candidate_service.get_candidate_state(candidate_id, runtime_context)
    if action == "request_review":
        review_result = candidate_service.request_review(candidate_id, runtime_context)
        if not policy.auto_approve_after_request_review:
            return review_result
        approval_args = _approval_payload_with_policy_defaults(args, policy=policy)
        approved = candidate_service.approve(candidate_id, approval_args, runtime_context)
        return {
            **approved,
            "lifecycle_shortcut": {
                "auto_approve_after_request_review": True,
                "review_state": review_result.get("candidate_state"),
            },
        }
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class _CandidateReviewGovernancePolicy:
    allow_contextless_review: bool
    allowed_contextless_actions: tuple[str, ...]
    auto_approve_after_request_review: bool
    fallback_organization_id: str | None
    fallback_workspace_id: str | None
    fallback_actor_id: str | None
    fallback_actor_role: str
    default_actor_authority: str | None
    default_approval_evidence: dict[str, Any]
    default_test_evidence: dict[str, Any]
    default_review_comments: str | None


def _candidate_review_governance_policy() -> _CandidateReviewGovernancePolicy:
    raw = os.environ.get("SEMANTIER_TENDER_SENSITIVE_REVIEW_POLICY_JSON", "").strip()
    payload: dict[str, Any] = {}
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("SEMANTIER_TENDER_SENSITIVE_REVIEW_POLICY_JSON must be an object")
        payload = parsed
    allowed = payload.get("allowed_contextless_actions") or []
    if isinstance(allowed, str):
        allowed_actions = tuple(item.strip() for item in allowed.split(",") if item.strip())
    elif isinstance(allowed, list):
        allowed_actions = tuple(str(item).strip() for item in allowed if str(item).strip())
    else:
        allowed_actions = ()
    return _CandidateReviewGovernancePolicy(
        allow_contextless_review=bool(payload.get("allow_contextless_review", False)),
        allowed_contextless_actions=allowed_actions,
        auto_approve_after_request_review=bool(
            payload.get("auto_approve_after_request_review", False)
        ),
        fallback_organization_id=_optional_nonempty(payload.get("fallback_organization_id")),
        fallback_workspace_id=_optional_nonempty(payload.get("fallback_workspace_id")),
        fallback_actor_id=_optional_nonempty(payload.get("fallback_actor_id")),
        fallback_actor_role=str(payload.get("fallback_actor_role") or "system").strip() or "system",
        default_actor_authority=_optional_nonempty(payload.get("default_actor_authority")),
        default_approval_evidence=_as_object(payload.get("default_approval_evidence")),
        default_test_evidence=_as_object(payload.get("default_test_evidence")),
        default_review_comments=_optional_nonempty(payload.get("default_review_comments")),
    )


def _resolve_candidate_review_runtime_context(
    *,
    ctx: Any,
    args: dict[str, Any],
    action: str,
    policy: _CandidateReviewGovernancePolicy,
) -> RuntimeContextDTO:
    if ctx is not None:
        return RuntimeContextDTO(
            organization_id=str(
                getattr(ctx, "organization_id", "") or args.get("organization_id") or ""
            ),
            workspace_id=getattr(ctx, "workspace_id", None) or args.get("workspace_id"),
            actor_id=str(getattr(ctx, "user_id", "") or args.get("actor_id") or ""),
            actor_role=str(getattr(ctx, "member_role", "") or args.get("actor_role") or ""),
        )
    if not policy.allow_contextless_review:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    if action not in set(policy.allowed_contextless_actions):
        raise ValueError("CANDIDATE_REVIEW_ACTION_NOT_ALLOWED_WITHOUT_CONTEXT")
    organization_id = str(args.get("organization_id") or policy.fallback_organization_id or "").strip()
    actor_id = str(args.get("actor_id") or policy.fallback_actor_id or "").strip()
    if not organization_id:
        raise ValueError("CANDIDATE_REVIEW_POLICY_MISSING_ORGANIZATION_ID")
    if not actor_id:
        raise ValueError("CANDIDATE_REVIEW_POLICY_MISSING_ACTOR_ID")
    workspace_id = args.get("workspace_id") or policy.fallback_workspace_id
    actor_role = str(args.get("actor_role") or policy.fallback_actor_role or "system").strip() or "system"
    return RuntimeContextDTO(
        organization_id=organization_id,
        workspace_id=workspace_id,
        actor_id=actor_id,
        actor_role=actor_role,
    )


def _approval_payload_with_policy_defaults(
    args: dict[str, Any],
    *,
    policy: _CandidateReviewGovernancePolicy,
) -> dict[str, Any]:
    payload = dict(args)
    if not payload.get("actor_authority") and policy.default_actor_authority:
        payload["actor_authority"] = policy.default_actor_authority
    if not payload.get("approval_evidence") and policy.default_approval_evidence:
        payload["approval_evidence"] = policy.default_approval_evidence
    if not payload.get("test_evidence") and policy.default_test_evidence:
        payload["test_evidence"] = policy.default_test_evidence
    return payload


def _optional_nonempty(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _as_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _resolve_precedent_lookup_service(
    *,
    kwargs: dict[str, Any],
    detection_service: StoreBackedDetectionEvidenceService,
):
    injected = kwargs.get("precedent_observed_expression_lookup_service")
    if injected is not None:
        return injected
    db_path = kwargs.get("db_path") or detection_service.authoritative_db_path
    return StoreBackedTenderPrecedentObservedExpressionLookupService(db_path=db_path)


def label_tender_document(*, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    ctx = kwargs.get("ctx") or kwargs.get("request_context")
    if ctx is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    request = _request_from_payload(args, ctx=ctx)
    resolver = kwargs.get("activation_snapshot_resolver") or StoreBackedActivationSnapshotResolverService(
        rule_store=kwargs.get("rule_store")
    )
    detection_service = kwargs.get("detection_evidence_service") or StoreBackedDetectionEvidenceService(
        run_store=kwargs.get("run_store")
    )
    runtime_context = RuntimeContextDTO(
        organization_id=request.organization_id,
        workspace_id=request.workspace_id,
        actor_id=str(getattr(ctx, "user_id")),
        actor_role=str(getattr(ctx, "member_role", "") or ""),
    )
    governed_context = resolver.materialize_snapshot(
        requested_rule_families=request.requested_rule_families,
        context=runtime_context,
    )
    if not governed_context:
        raise ValueError("governed activation context required for tender detection")
    snapshot_id = str(governed_context.get("activation_set_snapshot_id") or "")
    snapshot_hash = str(governed_context.get("resolved_activation_set_hash") or "")
    resolver_policy = str(governed_context.get("activation_resolver_policy_version") or "")
    if not snapshot_id or not snapshot_hash or not resolver_policy:
        raise ValueError("resolver snapshot evidence required for tender detection")
    active_rules, rule_version_set_hash = resolver.list_rules_for_activation_snapshot(
        activation_set_snapshot_id=snapshot_id,
        resolved_activation_set_hash=snapshot_hash,
        context=runtime_context,
        requested_rule_families=request.requested_rule_families,
    )
    ignored_local_rules = [
        {
            "rule_ref": str(item.get("rule_ref") or item.get("name") or "local_rule"),
            "reason_code": "LOCAL_RULE_NOT_AUTHORITY",
        }
        for item in request.local_rules
    ]
    graph_lineage = []
    for rule in active_rules:
        lineage = tuple(rule.get(field) for field in (
            "source_graph_version", "source_graph_release_hash", "source_graph_rule_id",
        ))
        if any(value is not None for value in lineage) and not all(lineage):
            raise ValueError("activated governed rule has partial graph lineage")
        if all(lineage):
            graph_lineage.append({
                "source_graph_version": lineage[0],
                "source_graph_release_hash": lineage[1],
                "source_graph_rule_id": lineage[2],
            })
    expected_release_hash = str(
        request.project_metadata.get("accepted_graph_release_hash")
        or request.project_metadata.get("acceptedGraphReleaseHash")
        or ""
    )
    if expected_release_hash and any(
        item["source_graph_release_hash"] != expected_release_hash for item in graph_lineage
    ):
        raise ValueError("activated rule lineage does not match accepted graph release")
    expected_versions = {
        value
        for value in (request.expected_graph_version, request.expected_accepted_release_id)
        if value
    }
    if expected_versions:
        active_versions = {str(item["source_graph_version"]) for item in graph_lineage}
        if not active_versions or not expected_versions.issubset(active_versions):
            raise ValueError("graph_not_active_in_governed_context")
    active_concept_ids = sorted(
        {
            str(rule.get("source_graph_rule_id"))
            for rule in active_rules
            if rule.get("source_graph_rule_id")
        }
    )
    active_release_ids = sorted(
        {
            str(item["source_graph_version"])
            for item in graph_lineage
            if item.get("source_graph_version")
        }
    )
    precedent_lookup_service = _resolve_precedent_lookup_service(
        kwargs=kwargs,
        detection_service=detection_service,
    )
    precedent_observed_expressions = (
        precedent_lookup_service.list_human_accepted_observed_expressions(
            concept_ids=active_concept_ids,
            accepted_graph_release_ids=active_release_ids,
        )
    )
    precedent_model_hash = content_hash(precedent_observed_expressions)
    precedent_model_ref = f"contextgraph_observed_expression_precedent:{precedent_model_hash}"
    domain_result = detector.detect_tender_occurrences(
        document_text=request.document_text,
        canonical_document=request.canonical_document,
        active_rules=active_rules,
        resolver_evidence_ref=snapshot_id,
        precedent_observed_expressions=precedent_observed_expressions,
        precedent_model_ref=precedent_model_ref,
    )
    run_request = DetectionRunCreateDTO(
        tender_document_id=request.tender_document_id,
        activation_set_snapshot_id=snapshot_id,
        resolved_activation_set_hash=snapshot_hash,
        rule_version_set_hash=rule_version_set_hash,
        target_artifact_ref=request.target_normalized_document_artifact_ref,
        target_artifact_hash=request.target_normalized_document_artifact_hash,
        idempotency_key=str(args.get("idempotencyKey") or args.get("idempotency_key") or "") or None,
        metadata={
            **request.project_metadata,
            "document_text": request.document_text,
            "parent_run_id": str(request.project_metadata.get("parent_run_id") or "") or None,
            "review_purpose": request.review_purpose,
            "caller_role": request.caller_role,
            "requested_rule_families": request.requested_rule_families,
            "activation_resolver_policy_version": resolver_policy,
            "ignored_local_rules": ignored_local_rules,
            "rule_version_set_hash": rule_version_set_hash,
            "rule_version_refs": [
                {
                    "rule_version_id": rule["rule_version_id"],
                    "content_hash": rule["content_hash"],
                }
                for rule in active_rules
            ],
            "governed_rule_activation_ids": sorted(
                {
                    str(activation_id)
                    for rule in active_rules
                    for activation_id in (rule.get("governed_rule_activation_ids") or [])
                }
            ),
            "source_graph_lineage": graph_lineage,
            "precedent_lookup_model_ref": precedent_model_ref,
            "precedent_lookup_model_hash": precedent_model_hash,
            "precedent_lookup_observed_expression_count": len(precedent_observed_expressions),
            "semantic_profile_ids": sorted(
                {
                    str(finding["semantic_profile_id"])
                    for finding in domain_result.findings
                    if finding.get("semantic_profile_id")
                }
            ),
            "target_session_id": request.target_session_id,
            "target_source_hash": request.target_source_hash,
            "target_normalized_document_artifact_ref": request.target_normalized_document_artifact_ref,
            "target_normalized_document_artifact_hash": request.target_normalized_document_artifact_hash,
            "document_kind": _canonical_document_kind(request.canonical_document)
            if request.canonical_document is not None
            else "canonical_source_ir",
            "document_schema_version": (
                (request.canonical_document or {}).get("schema_version")
                if request.canonical_document is not None
                else "plain_text.v1"
            ),
        },
    )
    detection_run = detection_service.create_detection_run_idempotent(
        run_request,
        runtime_context,
    )
    findings = [
        _finding_dto(finding)
        for finding in domain_result.findings
    ]
    conflicts = [
        _runtime_conflict_dto(conflict)
        for conflict in domain_result.runtime_conflicts
    ]
    finding_set = detection_service.persist_occurrence_findings(
        detection_run.run_id,
        findings,
        runtime_context,
    )
    conflict_set = detection_service.persist_runtime_conflicts(
        detection_run.run_id,
        conflicts,
        runtime_context,
    )
    finalized = detection_service.finalize_detection_run(
        detection_run.run_id,
        finding_set.finding_set_hash,
        conflict_set.conflict_set_hash,
        "finalized",
        runtime_context,
    )
    return dict(finalized.metadata["run"])


def _finding_dto(finding: dict[str, Any]) -> OccurrenceFindingDTO:
    triggered_refs = finding.get("triggered_rule_version_refs") or []
    rule_version_id = str(
        finding.get("winning_rule_version_ref")
        or finding.get("triggered_rule_version_id")
        or (
            triggered_refs[0].get("rule_version_id")
            if triggered_refs and isinstance(triggered_refs[0], dict)
            else ""
        )
    )
    return OccurrenceFindingDTO(
        finding_id=str(finding.get("finding_id") or f"tdf_{uuid4().hex}"),
        rule_version_id=rule_version_id,
        target_anchor_ref=str(finding.get("target_anchor_ref") or ""),
        start_offset=int(finding.get("start_offset") or 0),
        end_offset=int(finding.get("end_offset") or 0),
        occurrence_index=int(finding.get("occurrence_index") or 0),
        element_hash=str(finding.get("target_element_hash") or ""),
        content_hash=str(finding.get("content_hash") or content_hash(canonical_json(finding))),
        metadata={"store_payload": finding},
    )


def _runtime_conflict_dto(conflict: dict[str, Any]) -> RuntimeConflictDTO:
    return RuntimeConflictDTO(
        conflict_id=str(conflict.get("conflict_id") or f"tdrc_{uuid4().hex}"),
        conflict_type=str(conflict.get("conflict_type") or ""),
        target_anchor_ref=str(conflict.get("target_anchor_ref") or ""),
        rule_version_ids=tuple(str(item) for item in conflict.get("rule_version_ids") or ()),
        start_offset=int(conflict.get("start_offset") or 0),
        end_offset=int(conflict.get("end_offset") or 0),
        content_hash=str(conflict.get("content_hash") or content_hash(canonical_json(conflict))),
        metadata={"store_payload": conflict},
    )


def generate_sensitive_labeling_report(*, args: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    return report.create_sensitive_labeling_report(**args)


def generate_labeled_docx(*, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    ctx = kwargs.get("ctx") or kwargs.get("request_context")
    if ctx is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    return docx_output.generate_labeled_docx(ctx=ctx, **args)


def verify_sensitive_labeling_replay(*, args: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    ctx = _kwargs.get("ctx") or _kwargs.get("request_context")
    if ctx is None:
        raise ValueError("AUTHENTICATED_REQUEST_CONTEXT_REQUIRED")
    replay_binding_id = str(args.get("replay_binding_id") or "").strip()
    if not replay_binding_id:
        raise ValueError("replay_binding_id is required")
    runtime_context = RuntimeContextDTO(
        organization_id=str(getattr(ctx, "organization_id", "") or args.get("organization_id") or ""),
        workspace_id=getattr(ctx, "workspace_id", None) or args.get("workspace_id"),
        actor_id=str(getattr(ctx, "user_id", "") or args.get("actor_id") or ""),
        actor_role=str(getattr(ctx, "member_role", "") or args.get("actor_role") or ""),
    )
    replay_service = _kwargs.get("replay_service") or StoreBackedGenericReplayEnvelopeService()
    return replay_service.verify_typed_replay_binding(replay_binding_id, runtime_context)
