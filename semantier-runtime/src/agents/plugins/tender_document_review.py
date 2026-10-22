from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents.auth_session import RequestContext
from agents.governed_context import resolve_governed_activation_context
from eos.artifact_hashing import canonical_json, content_hash
from runtime_paths import workspace_session_runs_root_from_hermes_home
from storage.governed_rule_store import GovernedRuleStore
from storage.knowledge_extraction_store import KnowledgeExtractionStore
from storage.tender_detection_run_store import TenderDetectionRunStore
from plugins.tender_sensitive_term_detection_modeling.detector import detect_tender_occurrences
from contracts.evidence_location import SourceIdentity
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import adapt_docx_bytes
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_for_block


@dataclass(frozen=True)
class TenderDetectionRequest:
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
    target_source_hash: str | None = None
    target_normalized_document_artifact_ref: str | None = None
    target_normalized_document_artifact_hash: str | None = None
    target_source_identity: dict[str, Any] | None = None
    target_canonical_source_document: dict[str, Any] | None = None

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        ctx: RequestContext,
    ) -> "TenderDetectionRequest":
        document_text = str(
            payload.get("documentText") or payload.get("document_text") or ""
        ).strip()
        file_ref = str(payload.get("fileRef") or payload.get("file_ref") or "").strip()
        canonical_document: dict[str, Any] | None = None
        target_source_hash = None
        target_normalized_document_artifact_hash = None
        target_normalized_document_artifact_ref = None
        target_source_identity = payload.get("sourceIdentity") or payload.get("source_identity")
        if target_source_identity is not None and not isinstance(target_source_identity, dict):
            raise ValueError("sourceIdentity must be an object when provided")
        target_canonical_source_document = payload.get("canonicalSourceDocument") or payload.get("canonical_source_document")
        if not document_text and file_ref:
            loaded = _load_canonical_document_artifact(ctx=ctx, file_ref=file_ref)
            canonical_document = loaded["document"]
            target_source_hash = loaded["source_hash"]
            target_normalized_document_artifact_hash = loaded["artifact_hash"]
            target_normalized_document_artifact_ref = file_ref
            document_text = _canonical_document_text(canonical_document)
            target_source_identity = target_source_identity or loaded.get("source_identity")
            target_canonical_source_document = loaded.get("canonical_source_document")
        if not document_text:
            raise ValueError("documentText is required")
        org_id = str(
            ctx.organization_id
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
            or project_metadata.get("expected_graph_version")
            or project_metadata.get("expectedGraphVersion")
            or ""
        ).strip() or None
        expected_accepted_release_id = str(
            payload.get("expectedAcceptedReleaseId")
            or payload.get("expected_accepted_release_id")
            or project_metadata.get("expected_accepted_release_id")
            or project_metadata.get("expectedAcceptedReleaseId")
            or ""
        ).strip() or None
        if expected_graph_version:
            project_metadata = {**project_metadata, "expected_graph_version": expected_graph_version}
        if expected_accepted_release_id:
            project_metadata = {
                **project_metadata,
                "expected_accepted_release_id": expected_accepted_release_id,
            }
        return cls(
            tender_document_id=str(
                payload.get("tenderDocumentId")
                or payload.get("tender_document_id")
                or f"tdoc_{uuid4().hex}"
            ).strip(),
            document_text=document_text,
            organization_id=org_id,
            workspace_id=str(
                ctx.workspace_id
                or payload.get("workspaceId")
                or payload.get("workspace_id")
                or ""
            ).strip()
            or None,
            project_metadata=project_metadata,
            review_purpose=str(
                payload.get("reviewPurpose")
                or payload.get("review_purpose")
                or "pre_release_check"
            ).strip(),
            caller_role=str(
                payload.get("callerRole") or payload.get("caller_role") or ctx.member_role or "member"
            ).strip(),
            requested_rule_families=[str(item).strip() for item in requested if str(item).strip()],
            local_rules=[item for item in local_rules if isinstance(item, dict)],
            expected_graph_version=expected_graph_version,
            expected_accepted_release_id=expected_accepted_release_id,
            canonical_document=canonical_document,
            target_source_hash=target_source_hash,
            target_normalized_document_artifact_ref=target_normalized_document_artifact_ref,
            target_normalized_document_artifact_hash=target_normalized_document_artifact_hash,
            target_source_identity=target_source_identity,
            target_canonical_source_document=target_canonical_source_document,
        )


class TenderDocumentReviewRuntime:
    def __init__(
        self,
        *,
        rule_store: GovernedRuleStore | None = None,
        run_store: TenderDetectionRunStore | None = None,
    ) -> None:
        self._rule_store = rule_store or GovernedRuleStore()
        self._run_store = run_store or TenderDetectionRunStore()

    def detect(self, *, ctx: RequestContext, request: TenderDetectionRequest) -> dict[str, Any]:
        governed_context = resolve_governed_activation_context(
            user_id=str(ctx.user_id),
            workspace_id=request.workspace_id,
            persist_snapshot=True,
        )
        if not governed_context:
            raise ValueError("governed activation context required for tender detection")
        snapshot_id = str(governed_context.get("activation_set_snapshot_id") or "")
        snapshot_hash = str(governed_context.get("resolved_activation_set_hash") or "")
        resolver_policy = str(governed_context.get("activation_resolver_policy_version") or "")
        if not snapshot_id or not snapshot_hash or not resolver_policy:
            raise ValueError("resolver snapshot evidence required for tender detection")

        active_rules, rule_version_set_hash = self._rule_store.list_rules_for_activation_snapshot(
            activation_set_snapshot_id=snapshot_id,
            resolved_activation_set_hash=snapshot_hash,
            organization_id=request.organization_id,
            workspace_id=request.workspace_id,
            requested_rule_families=request.requested_rule_families,
        )
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
        precedent_observed_expressions = KnowledgeExtractionStore(
            self._run_store._db_path
        ).list_human_accepted_observed_expressions(
            concept_ids=active_concept_ids,
            accepted_graph_release_ids=active_release_ids,
        )
        precedent_model_hash = content_hash(precedent_observed_expressions)
        precedent_model_ref = f"contextgraph_observed_expression_precedent:{precedent_model_hash}"

        ignored_local_rules = [
            {
                "rule_ref": str(item.get("rule_ref") or item.get("name") or "local_rule"),
                "reason_code": "LOCAL_RULE_NOT_AUTHORITY",
            }
            for item in request.local_rules
        ]
        domain_result = detect_tender_occurrences(
            document_text=request.document_text,
            canonical_document=request.canonical_document,
            active_rules=active_rules,
            resolver_evidence_ref=snapshot_id,
            precedent_observed_expressions=precedent_observed_expressions,
            precedent_model_ref=precedent_model_ref,
        )
        findings = domain_result.findings
        findings = _attach_target_evidence(
            findings,
            canonical_source_document=request.target_canonical_source_document,
            source_identity=request.target_source_identity,
            artifact_ref=request.target_normalized_document_artifact_ref,
        )
        runtime_conflicts = domain_result.runtime_conflicts
        return self._run_store.create_run(
            run_id=None,
            tender_document_id=request.tender_document_id,
            document_text=request.document_text,
            organization_id=request.organization_id,
            workspace_id=request.workspace_id,
            parent_run_id=str(request.project_metadata.get("parent_run_id") or "") or None,
            project_metadata={
                **request.project_metadata,
                "rule_version_set_hash": rule_version_set_hash,
                "rule_version_refs": [
                    {
                        "rule_version_id": rule["rule_version_id"],
                        "content_hash": rule["content_hash"],
                    }
                    for rule in active_rules
                ],
                "source_graph_lineage": graph_lineage,
                "precedent_lookup_model_ref": precedent_model_ref,
                "precedent_lookup_model_hash": precedent_model_hash,
                "precedent_lookup_observed_expression_count": len(precedent_observed_expressions),
                "governed_rule_activation_ids": sorted(
                    {
                        str(activation_id)
                        for rule in active_rules
                        for activation_id in (
                            rule.get("governed_rule_activation_ids") or []
                        )
                    }
                ),
                "target_source_hash": request.target_source_hash,
                "target_normalized_document_artifact_ref": request.target_normalized_document_artifact_ref,
                "target_normalized_document_artifact_hash": request.target_normalized_document_artifact_hash,
                "document_schema_version": (
                    (request.canonical_document or {}).get("schema_version")
                    if request.canonical_document is not None
                    else "plain_text.v1"
                ),
                "runtime_conflicts": runtime_conflicts,
                "target_source_identity": request.target_source_identity,
                "target_canonical_source_document": request.target_canonical_source_document,
            },
            review_purpose=request.review_purpose,
            caller_role=request.caller_role,
            requested_rule_families=request.requested_rule_families,
            activation_set_snapshot_id=snapshot_id,
            activation_resolver_policy_version=resolver_policy,
            resolved_activation_set_hash=snapshot_hash,
            ignored_local_rules=ignored_local_rules,
            findings=findings,
        )

    def _detect_findings(
        self,
        *,
        request: TenderDetectionRequest,
        active_rules: list[dict[str, Any]],
        resolver_evidence_ref: str,
    ) -> list[dict[str, Any]]:
        if request.canonical_document is not None:
            return self._detect_canonical_occurrence_findings(
                request=request,
                active_rules=active_rules,
                resolver_evidence_ref=resolver_evidence_ref,
            )
        matched: dict[str, list[dict[str, Any]]] = {}
        for rule in active_rules:
            scope = rule.get("applicability_scope") or {}
            terms = scope.get("match_terms") if isinstance(scope, dict) else None
            normalized_terms = [str(term).strip() for term in (terms or []) if str(term).strip()]
            for term in normalized_terms:
                if term in request.document_text:
                    matched.setdefault(term, []).append(rule)

        findings: list[dict[str, Any]] = []
        for term, rules in matched.items():
            if len(rules) > 1:
                rule_ids = ",".join(sorted(str(rule["rule_version_id"]) for rule in rules))
                findings.append(
                    self._finding(
                        term=term,
                        rule=rules[0],
                        issue_type="conflict_escalation",
                        triggered_rule_version_id=rule_ids,
                        resolver_evidence_ref=resolver_evidence_ref,
                        escalation_flag=True,
                    )
                )
                continue
            rule = rules[0]
            scope = rule.get("applicability_scope") or {}
            issue_type = (
                str(scope.get("issue_type") or "compliance_violation")
                if isinstance(scope, dict)
                else "compliance_violation"
            )
            findings.append(
                self._finding(
                    term=term,
                    rule=rule,
                    issue_type=issue_type,
                    triggered_rule_version_id=str(rule["rule_version_id"]),
                    resolver_evidence_ref=resolver_evidence_ref,
                    escalation_flag=False,
                )
            )
        return findings

    def _detect_canonical_occurrence_findings(
        self,
        *,
        request: TenderDetectionRequest,
        active_rules: list[dict[str, Any]],
        resolver_evidence_ref: str,
    ) -> list[dict[str, Any]]:
        document = request.canonical_document or {}
        if document.get("schema_version") != "canonical_document.v2":
            raise ValueError("canonical_document.v2 is required for fileRef occurrence labeling")
        term_rules: dict[str, list[dict[str, Any]]] = {}
        for rule in active_rules:
            scope = rule.get("applicability_scope") or {}
            terms = scope.get("match_terms") if isinstance(scope, dict) else None
            for term in [str(item).strip() for item in (terms or []) if str(item).strip()]:
                term_rules.setdefault(term, []).append(rule)

        findings: list[dict[str, Any]] = []
        occurrence_counts: dict[str, int] = {}
        for element in document.get("elements") or []:
            text = str(element.get("text") or "")
            if not text:
                continue
            for term, rules in term_rules.items():
                start = text.find(term)
                while start >= 0:
                    end = start + len(term)
                    occurrence_counts[term] = occurrence_counts.get(term, 0) + 1
                    occurrence_index = occurrence_counts[term]
                    if len(rules) > 1:
                        rule_ids = sorted(str(rule["rule_version_id"]) for rule in rules)
                        findings.append(
                            self._finding(
                                term=term,
                                rule=rules[0],
                                issue_type="conflict_escalation",
                                triggered_rule_version_id=",".join(rule_ids),
                                resolver_evidence_ref=resolver_evidence_ref,
                                escalation_flag=True,
                                occurrence={
                                    "element": element,
                                    "start_offset": start,
                                    "end_offset": end,
                                    "occurrence_index": occurrence_index,
                                },
                                triggered_rule_version_refs=[
                                    {
                                        "rule_version_id": rule["rule_version_id"],
                                        "content_hash": rule["content_hash"],
                                    }
                                    for rule in rules
                                ],
                                winning_rule_version_ref=None,
                                conflicting_rule_version_refs=[
                                    {
                                        "rule_version_id": rule["rule_version_id"],
                                        "content_hash": rule["content_hash"],
                                    }
                                    for rule in rules
                                ],
                                conflict_resolution_ref="conflict_escalation",
                            )
                        )
                    else:
                        rule = rules[0]
                        scope = rule.get("applicability_scope") or {}
                        issue_type = (
                            str(scope.get("issue_type") or "compliance_violation")
                            if isinstance(scope, dict)
                            else "compliance_violation"
                        )
                        findings.append(
                            self._finding(
                                term=term,
                                rule=rule,
                                issue_type=issue_type,
                                triggered_rule_version_id=str(rule["rule_version_id"]),
                                resolver_evidence_ref=resolver_evidence_ref,
                                escalation_flag=False,
                                occurrence={
                                    "element": element,
                                    "start_offset": start,
                                    "end_offset": end,
                                    "occurrence_index": occurrence_index,
                                },
                                triggered_rule_version_refs=[
                                    {
                                        "rule_version_id": rule["rule_version_id"],
                                        "content_hash": rule["content_hash"],
                                    }
                                ],
                                winning_rule_version_ref=str(rule["rule_version_id"]),
                                conflicting_rule_version_refs=[],
                                conflict_resolution_ref=None,
                            )
                        )
                    start = text.find(term, end)
        return findings

    def _finding(
        self,
        *,
        term: str,
        rule: dict[str, Any],
        issue_type: str,
        triggered_rule_version_id: str,
        resolver_evidence_ref: str,
        escalation_flag: bool,
        occurrence: dict[str, Any] | None = None,
        triggered_rule_version_refs: list[dict[str, Any]] | None = None,
        winning_rule_version_ref: str | None = None,
        conflicting_rule_version_refs: list[dict[str, Any]] | None = None,
        conflict_resolution_ref: str | None = None,
    ) -> dict[str, Any]:
        element = (occurrence or {}).get("element") or {}
        target_anchor_ref = element.get("anchor_ref") if occurrence else None
        target_element_kind = element.get("kind") if occurrence else None
        target_element_id = element.get("id") if occurrence else None
        target_element_hash = element.get("local_text_hash") if occurrence else None
        start_offset = (occurrence or {}).get("start_offset")
        end_offset = (occurrence or {}).get("end_offset")
        occurrence_index = (occurrence or {}).get("occurrence_index")
        return {
            "finding_id": f"tdf_{uuid4().hex}",
            "clause_id": "document",
            "clause_anchor": target_anchor_ref or f"text:contains:{term}",
            "issue_type": issue_type,
            "matched_text": term,
            "normalized_match": term.strip().lower(),
            "triggered_rule_version_id": triggered_rule_version_id,
            "triggered_rule_version_refs": triggered_rule_version_refs or [],
            "winning_rule_version_ref": winning_rule_version_ref,
            "conflicting_rule_version_refs": conflicting_rule_version_refs or [],
            "conflict_resolution_ref": conflict_resolution_ref,
            "source_anchor_refs": rule.get("source_anchor_refs") or [],
            "judgment_basis": str(rule.get("extracted_rationale") or "governed rule matched tender text"),
            "severity": str(rule.get("severity") or "medium"),
            "confidence": float(rule.get("confidence") or 1.0),
            "suggested_replacement": self._suggested_replacement(term=term, rule=rule),
            "escalation_flag": escalation_flag,
            "resolver_evidence_ref": resolver_evidence_ref,
            "target_anchor_ref": target_anchor_ref,
            "target_element_kind": target_element_kind,
            "target_element_id": target_element_id,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "occurrence_index": occurrence_index,
            "target_element_hash": target_element_hash,
            "source_graph_version": rule.get("source_graph_version"),
            "source_graph_release_hash": rule.get("source_graph_release_hash"),
            "source_graph_rule_id": rule.get("source_graph_rule_id"),
        }

    def _suggested_replacement(self, *, term: str, rule: dict[str, Any]) -> str | None:
        scope = rule.get("applicability_scope") or {}
        if isinstance(scope, dict) and scope.get("suggested_replacement"):
            return str(scope["suggested_replacement"])
        if term:
            return f"AI-assisted recommendation: remove or revise '{term}' to satisfy the governed rule."
        return None


def detect_tender_document(*, ctx: RequestContext, payload: dict[str, Any]) -> dict[str, Any]:
    request = TenderDetectionRequest.from_payload(payload, ctx=ctx)
    return TenderDocumentReviewRuntime().detect(ctx=ctx, request=request)


def _canonical_document_text(document: dict[str, Any]) -> str:
    return "\n".join(str(element.get("text") or "") for element in document.get("elements") or [])


def _load_canonical_document_artifact(*, ctx: RequestContext, file_ref: str) -> dict[str, Any]:
    raw = str(file_ref or "").strip()
    if not raw:
        raise ValueError("fileRef is required")
    if Path(raw).is_absolute() or raw.startswith(("http://", "https://", "file://")):
        raise ValueError("fileRef must be a governed session artifact reference")
    session_id = os.environ.get("HERMES_SESSION_ID") or os.environ.get("SESSION_ID") or ""
    workspace_home = os.environ.get("HERMES_HOME") or os.environ.get("SESSION_HERMES_HOME") or ""
    if not workspace_home:
        workspace_home = str(ctx.hermes_home)
    if not session_id:
        raise ValueError("session binding required for fileRef")
    session_root = workspace_session_runs_root_from_hermes_home(Path(workspace_home), session_id).parent
    path = (session_root / raw).resolve()
    try:
        path.relative_to(session_root.resolve())
    except ValueError as exc:
        raise ValueError("fileRef must remain inside the governed session") from exc
    if not path.is_file():
        raise ValueError("fileRef artifact not found")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in artifact.items() if key != "artifact_hash"}
    actual_hash = "sha256:" + sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    if artifact.get("artifact_hash") != actual_hash:
        raise ValueError("normalized document artifact hash mismatch")
    document = artifact.get("document")
    if not isinstance(document, dict):
        raise ValueError("invalid normalized document artifact")
    if document.get("schema_version") != "canonical_document.v2":
        raise ValueError("canonical_document.v2 is required for fileRef occurrence labeling")
    source_hash = str(artifact.get("source_hash") or "")
    source_artifact_ref = str(artifact.get("source_artifact_ref") or "")
    source_identity_ref = f"source:tender:{source_hash.removeprefix('sha256:')}" if source_hash else f"source:tender:{sha256(raw.encode()).hexdigest()}"
    source_identity = SourceIdentity(
        schemaVersion="semantier.source_identity.v1",
        sourceIdentityRef=source_identity_ref,
        organizationId=str(ctx.organization_id),
        workspaceId=str(ctx.workspace_id or ""),
        sourceHash=source_hash or "sha256:" + sha256(canonical_json(document).encode()).hexdigest(),
        mediaType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        originalName=Path(source_artifact_ref or raw).name,
        artifactRef=source_artifact_ref or raw,
        sourceVersion=None,
        supersedesSourceIdentityRef=None,
        createdAt=datetime.now(timezone.utc),
    ).model_dump(mode="json", by_alias=True)
    canonical_source_document = None
    if source_artifact_ref:
        source_path = (session_root / source_artifact_ref).resolve()
        try:
            source_path.relative_to(session_root.resolve())
            if source_path.is_file():
                canonical_source_document = adapt_docx_bytes(
                    source_path.read_bytes(), source_identity_ref=source_identity_ref
                ).model_dump(mode="json", by_alias=True)
        except ValueError:
            raise ValueError("source artifact escaped governed session")
    return {
        "document": document,
        "source_hash": source_hash,
        "artifact_hash": artifact.get("artifact_hash"),
        "source_identity": source_identity,
        "canonical_source_document": canonical_source_document,
    }


def _attach_target_evidence(
    findings: list[dict[str, Any]],
    *,
    canonical_source_document: dict[str, Any] | None,
    source_identity: dict[str, Any] | None,
    artifact_ref: str | None,
) -> list[dict[str, Any]]:
    """Attach stable target EvidenceRefs without making offsets durable identity."""
    if not canonical_source_document or not source_identity:
        return findings
    try:
        document = __import__("contracts.canonical_source_ir", fromlist=["CanonicalSourceDocument"]).CanonicalSourceDocument.model_validate(canonical_source_document)
        identity = SourceIdentity.model_validate(source_identity)
    except Exception:
        return findings
    result: list[dict[str, Any]] = []
    used: set[str] = set()
    for finding in findings:
        match = str(finding.get("matched_text") or "")
        candidates = [block for block in document.blocks if isinstance(block.content, str) and match in block.content]
        block = next((candidate for candidate in candidates if candidate.block_id not in used), candidates[0] if candidates else None)
        enriched = dict(finding)
        if block is not None:
            ref = evidence_ref_for_block(document, block, source_identity=identity, evidence_role="tender_target")
            enriched["target_evidence_ref"] = ref.evidence_ref
            enriched["target_evidence"] = ref.model_dump(mode="json", by_alias=True)
            used.add(block.block_id)
        else:
            enriched["target_evidence_ref"] = None
        result.append(enriched)
    return result


def validate_graph_release_focus(*, finding: dict[str, Any], accepted_release_hash: str) -> dict[str, str]:
    """Validate the immutable release pin before a Studio graph focus is applied."""
    expected = str(finding.get("source_graph_release_hash") or "")
    if not expected or expected != str(accepted_release_hash):
        raise ValueError("GRAPH_RELEASE_HASH_MISMATCH")
    return {
        "source_graph_release_hash": expected,
        "source_graph_version": str(finding.get("source_graph_version") or ""),
        "source_graph_rule_id": str(finding.get("source_graph_rule_id") or ""),
    }
