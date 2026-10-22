from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from agents.auth_session import RequestContext
from security.sensitive_data.authority.bundle_loader import load_demo_bundle
from security.sensitive_data.authority.capabilities import (
    SensitiveGovernanceCapabilityResolver,
)
from security.sensitive_data.authority.escalation import (
    DeterministicEscalationReviewerResolver,
)
from security.sensitive_data.classification.ontology import bind_findings_to_t1
from security.sensitive_data.detection.deterministic import SensitiveDataDetector
from security.sensitive_data.domain.agent_adapter import LocalSimulatedGovernedAgentAdapter
from security.sensitive_data.domain.execution_boundary import bind_governed_execution_boundary
from security.sensitive_data.ingestion.source import resolve_governed_source_content
from security.sensitive_data.persistence.store import SensitiveGovernanceStore
from security.sensitive_data.policy.evaluator import SensitivePolicyEvaluator
from security.sensitive_data.policy.validator import validate_input_manifest
from security.sensitive_data.transformation.transformer import transform_input
from semantier.sensitive_governance_contracts.hashing import canonical_hash
from semantier.sensitive_governance_contracts.registry import CURRENT_CONTRACT_VERSION
from semantier.sensitive_governance_contracts.requests import (
    CopySegmentRequest,
    ExecuteAssessmentRequest,
    ReplayRunRequest,
    RevealSegmentRequest,
    ReviewEscalationRequest,
    SensitiveGovernanceRequest,
)


def _organization_id(ctx: RequestContext) -> str:
    org_id = str(ctx.organization_id or "").strip()
    if not org_id:
        raise ValueError("active organization context required")
    return org_id


def _authority_object(object_type: str, object_ref: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"object_type": object_type, "object_ref": object_ref, "payload": payload}


def _execution_expires_at() -> str:
    return (datetime.now(UTC) + timedelta(minutes=15)).isoformat()


class SensitiveGovernanceService:
    def __init__(
        self,
        store: SensitiveGovernanceStore | None = None,
        *,
        detector: SensitiveDataDetector | None = None,
        residual_detector: SensitiveDataDetector | None = None,
        bundle_loader: Callable[[], Any] = load_demo_bundle,
        evaluator_factory: Callable[[Any], SensitivePolicyEvaluator] = SensitivePolicyEvaluator,
        agent_adapter_factory: Callable[[], LocalSimulatedGovernedAgentAdapter]
        | None = None,
        reviewer_resolver: DeterministicEscalationReviewerResolver | None = None,
    ) -> None:
        self._store = store or SensitiveGovernanceStore()
        self._detector = detector or SensitiveDataDetector()
        self._residual_detector = residual_detector
        self._bundle_loader = bundle_loader
        self._evaluator_factory = evaluator_factory
        self._agent_adapter_factory = (
            agent_adapter_factory or LocalSimulatedGovernedAgentAdapter
        )
        self._reviewer_resolver = reviewer_resolver or DeterministicEscalationReviewerResolver()

    def create_assessment(
        self,
        *,
        ctx: RequestContext,
        request: SensitiveGovernanceRequest,
        session_id: str | None,
    ) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        if not ctx.workspace_id:
            raise ValueError("workspace context required")
        if not session_id:
            session_id = "default"
        governed_source = resolve_governed_source_content(
            ctx=ctx,
            session_id=session_id,
            request=request,
        )
        if request.purpose == "workshop_finance_annotations":
            from agents.webapi.finance_sensitive_demo import bind_finance_demo
            from pathlib import Path
            result = bind_finance_demo(
                source_hash=governed_source.source_evidence.get("source_hash", ""),
                session_id=session_id,
                filename=Path(governed_source.source_evidence["governed_upload_ref"]).name,
            )
            return self._store.create_assessment(
                organization_id=organization_id, user_id=ctx.user_id,
                workspace_id=ctx.workspace_id, session_id=session_id,
                contract_version=request.contract_version, source_kind=request.source_kind,
                purpose=request.purpose, destination=request.destination,
                request_hash=canonical_hash(result), idempotency_key=request.idempotency_key,
                prepared_payload={"finance_teaching_result": result},
            )
        content = governed_source.content
        bundle = self._bundle_loader()
        findings = self._detector.detect(content)
        bound_findings = bind_findings_to_t1(findings, bundle)
        # Restricted values belong to one scoped assessment request. Detector
        # finding IDs describe content and may repeat across files or groups.
        payload_scope = canonical_hash({'organization_id': organization_id,
            'user_id': ctx.user_id, 'workspace_id': ctx.workspace_id,
            'session_id': session_id, 'idempotency_key': request.idempotency_key})
        bound_findings = [replace(finding, original_value_ref=
            f'restricted://sensitive-governance/input/{payload_scope}/{finding.finding_id}')
            for finding in bound_findings]
        evaluator = self._evaluator_factory(bundle)
        policy_decisions = evaluator.decide(bound_findings, destination=request.destination,
            **({'display_preferences':request.display_preferences} if request.display_preferences else {}))
        transformed = transform_input(content, bound_findings, policy_decisions)

        request_hash_payload = {
            "contract_version": request.contract_version,
            "source_kind": request.source_kind,
            "purpose": request.purpose,
            "destination": request.destination,
            "content": content,
        }
        if request.source_ref:
            request_hash_payload["source_ref"] = request.source_ref
        if request.display_preferences:
            request_hash_payload['display_preferences'] = request.display_preferences
        request_hash = canonical_hash(request_hash_payload)
        fact_context = {
            "fact_context_ref": "sg_fact_context_" + canonical_hash({"request_hash": request_hash})[:24],
            "source_artifact_refs": (
                [governed_source.source_evidence]
                if governed_source.source_evidence.get("governed_upload_ref")
                else []
            ),
            "admitted_fact_refs": [finding.evidence_ref for finding in bound_findings],
            "unresolved_fact_refs": [],
            "qualification_status": "qualified" if bound_findings else "partial",
        }
        semantic_binding = {
            "semantic_binding_ref": "sg_semantic_binding_" + canonical_hash(
                [finding.to_public_dict() for finding in bound_findings]
            )[:24],
            "ontology_bundle_hash": bundle.artifact_hashes["soe_sensitive_governance_t1_v1"],
            "recognizer_bundle_hash": canonical_hash({"detector_version": SensitiveDataDetector.detector_version}),
            "label_bindings": [finding.to_public_dict() for finding in bound_findings],
            "ambiguity_register": [],
        }
        authority_resolution = {
            "authority_resolution_ref": "sg_authority_resolution_" + bundle.manifest_hash[:24],
            "authority_source_manifest_hash": bundle.manifest_hash,
            "t2_legal_authority_registry_hash": bundle.t2_authority_registry.registry_hash,
            "t2_legal_authority_provenance_hash": bundle.t2_authority_registry.provenance_hash,
            "authority_effectivity_snapshot_hash": canonical_hash(
                {
                    "effective_from": bundle.manifest.get("effective_from"),
                    "effective_to": bundle.manifest.get("effective_to"),
                    "t2_effective_from": bundle.t2_authority_registry.effective_from,
                    "t2_effective_to": bundle.t2_authority_registry.effective_to,
                }
            ),
            "precedence_graph_hash": canonical_hash({"precedence": ["T2", "T3", "T4"]}),
            "resolution_rules_hash": canonical_hash({"rule": "T4_cannot_weaken_T2_T3"}),
            "resolution_status": "resolved",
        }
        consent_snapshot = {
            "consent_snapshot_ref": "sg_consent_" + canonical_hash(
                {"organization_id": organization_id, "purpose": request.purpose}
            )[:24],
            "institutional_intent": request.purpose,
            "policy_scope": ["soe_tender_sensitive_governance_demo"],
            "risk_boundaries": {"destination": request.destination},
            "prohibited_actions": ["send_existing_state_secret_marking"],
            "accountable_owner": organization_id,
        }
        authorization = SensitiveGovernanceCapabilityResolver(self._store).resolve(
            ctx=ctx,
            organization_id=organization_id,
            purpose=request.purpose,
            destination=request.destination,
        )
        authorization_snapshot = authorization.snapshot
        policy_decision_payload = {
            "policy_decision_ref": "sg_policy_decision_set_" + canonical_hash(
                [decision.to_dict() for decision in policy_decisions]
            )[:24],
            "institutional_route": evaluator.aggregate_route(policy_decisions),
            "enforcement_action": evaluator.aggregate_action(policy_decisions).value,
            "decisions": [decision.to_dict() for decision in policy_decisions],
            "policy_bundle_hash": bundle.artifact_hashes["soe_sensitive_governance_t4_v1"],
            "resolution_trace_hash": canonical_hash([decision.resolution_trace_hash for decision in policy_decisions]),
        }
        disclosure_justification = {
            "disclosure_justification_ref": "sg_disclosure_justification_" + canonical_hash(
                {"policy_decision_ref": policy_decision_payload["policy_decision_ref"]}
            )[:24],
            "fact_context_ref": fact_context["fact_context_ref"],
            "semantic_binding_ref": semantic_binding["semantic_binding_ref"],
            "authority_resolution_ref": authority_resolution["authority_resolution_ref"],
            "enterprise_consent_snapshot_ref": consent_snapshot["consent_snapshot_ref"],
            "authorization_snapshot_ref": authorization_snapshot["authorization_snapshot_ref"],
            "policy_decision_ref": policy_decision_payload["policy_decision_ref"],
            "route_proposal": policy_decision_payload["institutional_route"],
            "enforcement_action": policy_decision_payload["enforcement_action"],
            "reason_codes": sorted(
                {
                    reason
                    for decision in policy_decisions
                    for reason in decision.reason_codes
                }
            ),
            "expected_effects": ["transformed_input_only_crosses_agent_boundary"],
            "reversibility": "reversible",
            "responsible_principal": ctx.user_id,
            "generated_by": "repo_sensitive_policy_evaluator_v1",
            "evidence_refs": [finding.evidence_ref for finding in bound_findings],
        }
        residual_evidence = {
            "residual_scan_evidence_ref": "sg_residual_scan_" + canonical_hash(
                {"transformed_input_hash": transformed.transformed_input_hash}
            )[:24],
            "transformed_input_hash": transformed.transformed_input_hash,
        }
        input_manifest = {
            "input_manifest_ref": "sg_input_manifest_" + canonical_hash(
                {"transformed_input_hash": transformed.transformed_input_hash}
            )[:24],
            "fact_context_hash": canonical_hash(fact_context),
            "semantic_binding_hash": canonical_hash(semantic_binding),
            "authority_resolution_hash": canonical_hash(authority_resolution),
            "enterprise_consent_hash": canonical_hash(consent_snapshot),
            "authorization_snapshot_hash": canonical_hash(authorization_snapshot),
            "authorization_valid": authorization.valid,
            "authorization_reason_codes": authorization.reason_codes,
            "policy_decision_hash": canonical_hash(policy_decision_payload),
            "disclosure_justification_hash": canonical_hash(disclosure_justification),
            "transformed_input_hash": transformed.transformed_input_hash,
            "residual_scan_evidence_ref": residual_evidence["residual_scan_evidence_ref"],
            "residual_scan_evidence_hash": canonical_hash(residual_evidence),
        }
        validation = validate_input_manifest(
            manifest=input_manifest,
            policy_decisions=policy_decisions,
            transformed_text=transformed.transformed_text,
            residual_detector=self._residual_detector,
        )
        final_route = validation.route
        final_reason_codes = list(validation.reason_codes)
        escalation_case = None
        if validation.route == "ESCALATE":
            assignment = self._reviewer_resolver.resolve(organization_id=organization_id)
            if assignment is None:
                final_route = "FAIL_HARD"
                final_reason_codes.append("ESCALATION_REVIEWER_UNAVAILABLE")
                escalation_case = {
                    "escalation_id": "sg_escalation_" + canonical_hash(
                        {
                            "assessment_request_hash": request_hash,
                            "assessment_scope_hash": payload_scope,
                            "organization_id": organization_id,
                            "reviewer_assignment": None,
                        }
                    )[:24],
                    "admissibility_ref": "pending",
                    "reason_codes": final_reason_codes,
                    "assigned_role": "governance_security_reviewer",
                    "assigned_principal": None,
                    "initial_status": "blocked_no_reviewer",
                    "execution_blocked": True,
                }
            else:
                escalation_case = {
                    "escalation_id": "sg_escalation_" + canonical_hash(
                        {
                            "assessment_request_hash": request_hash,
                            "assessment_scope_hash": payload_scope,
                            "organization_id": organization_id,
                            "reviewer_assignment": assignment.assigned_principal,
                        }
                    )[:24],
                    "admissibility_ref": "pending",
                    "reason_codes": final_reason_codes,
                    "assigned_role": assignment.assigned_role,
                    "assigned_principal": assignment.assigned_principal,
                    "initial_status": "open",
                    "execution_blocked": True,
                }
        admissibility_decision = {
            "disclosure_admissibility_ref": "sg_disclosure_admissibility_" + canonical_hash(
                {
                    "validation_result_ref": validation.validation_result_ref,
                    "final_route": final_route,
                }
            )[:24],
            "justification_ref": disclosure_justification["disclosure_justification_ref"],
            "validation_result_ref": validation.validation_result_ref,
            "route": final_route,
            "autonomous_execution_allowed": final_route == "PASS_AUTO",
            "reason_codes": final_reason_codes,
            "validator_ref": "repo_sensitive_admissibility_validator_v1",
        }
        if escalation_case:
            escalation_case["admissibility_ref"] = admissibility_decision[
                "disclosure_admissibility_ref"
            ]
        state = {
            "PASS_AUTO": "PREPARED",
            "ESCALATE": "ESCALATED",
            "FAIL_HARD": "BLOCKED",
        }[final_route]
        authority_objects = [
            _authority_object("FactContextEnvelope", fact_context["fact_context_ref"], fact_context),
            _authority_object("SensitiveSemanticBinding", semantic_binding["semantic_binding_ref"], semantic_binding),
            _authority_object("AuthorityResolution", authority_resolution["authority_resolution_ref"], authority_resolution),
            _authority_object("EnterpriseConsentSnapshot", consent_snapshot["consent_snapshot_ref"], consent_snapshot),
            _authority_object("ActionAuthorizationSnapshot", authorization_snapshot["authorization_snapshot_ref"], authorization_snapshot),
            _authority_object("PolicyDecision", policy_decision_payload["policy_decision_ref"], policy_decision_payload),
            _authority_object("DisclosureJustification", disclosure_justification["disclosure_justification_ref"], disclosure_justification),
            _authority_object("SensitiveValidationInputManifest", input_manifest["input_manifest_ref"], input_manifest),
            _authority_object("SensitiveValidationResult", validation.validation_result_ref, validation.to_dict()),
            _authority_object("DisclosureAdmissibilityDecision", admissibility_decision["disclosure_admissibility_ref"], admissibility_decision),
        ]
        if escalation_case:
            authority_objects.append(
                _authority_object(
                    "SensitiveGovernanceEscalationCaseCreated",
                    escalation_case["escalation_id"],
                    escalation_case,
                )
            )
        workshop_fixture = None
        restricted_payloads: list[dict[str, Any]] = []
        if (
            request.purpose.startswith("workshop_")
            and governed_source.source_evidence.get("governed_upload_ref")
            and "模拟培训材料" in content
        ):
            workshop_fixture = {
                "fictional_training_only": True,
                "purpose": request.purpose,
                "source_evidence": governed_source.source_evidence,
            }
            restricted_payloads = [
                {
                    "restricted_payload_ref": finding.original_value_ref,
                    "classification": finding.finding_class.value,
                    "value": content[finding.start:finding.end],
                }
                for finding in bound_findings
            ]

        prepared_payload = {
            "contract_version": request.contract_version,
            "input_governance": {
                "status": state,
                "route": final_route,
                "finding_count": len(bound_findings),
                "findings": [finding.to_public_dict() for finding in bound_findings],
                "decisions": [decision.to_dict() for decision in policy_decisions],
                "transformations": transformed.transformation_refs,
                "policy_bundle_version": "soe_sensitive_governance_t4_v1",
                "policy_decision_ref": policy_decision_payload["policy_decision_ref"],
                "validation_result_ref": validation.validation_result_ref,
                "admissibility_ref": admissibility_decision["disclosure_admissibility_ref"],
                "destination": request.destination,
                "reason_codes": final_reason_codes,
                "escalation": escalation_case,
            },
            "execution_intent": {
                "requires_confirmation": True,
                "single_use": validation.route == "PASS_AUTO",
                "agent_boundary": "local_simulated_agent",
                "transformed_input_hash": transformed.transformed_input_hash,
                "expires_at": _execution_expires_at(),
            },
            "agent_boundary_comparison": {
                "raw_input_label": "synthetic_tender_source",
                "raw_request_hash": request_hash,
                "raw_detected_classes": sorted(
                    {finding.finding_class.value for finding in bound_findings}
                ),
                "governed_payload_label": "server_owned_transformed_input",
                "governed_payload_hash": transformed.transformed_input_hash,
                "restricted_request_ref": (
                    "restricted://sensitive-governance/transformed-input/"
                    + transformed.transformed_input_hash[:24]
                ),
                "destination": request.destination,
                "agent_boundary": "local_simulated_agent",
                "transformation_actions": sorted(
                    {decision.enforcement_action.value for decision in policy_decisions}
                ),
                "transformation_count": len(transformed.transformation_refs),
                "residual_scan_evidence_ref": residual_evidence[
                    "residual_scan_evidence_ref"
                ],
                "residual_scan_evidence_hash": canonical_hash(residual_evidence),
                "critical_residual_sensitive_data": False,
                "browser_payload_custody": "transformed_text_not_serialized",
            },
            "server_owned_transformed_input": transformed.transformed_text,
            "workshop_fixture": workshop_fixture,
            "bundle": {
                "bundle_id": bundle.bundle_id,
                "manifest_hash": bundle.manifest_hash,
                "artifact_hashes": bundle.artifact_hashes,
            },
        }
        if request.display_preferences:
            prepared_payload['display_preferences'] = dict(request.display_preferences)
        return self._store.create_assessment(
            organization_id=organization_id,
            user_id=ctx.user_id,
            workspace_id=ctx.workspace_id,
            session_id=session_id,
            contract_version=request.contract_version,
            source_kind=request.source_kind,
            purpose=request.purpose,
            destination=request.destination,
            request_hash=request_hash,
            idempotency_key=request.idempotency_key,
            prepared_payload=prepared_payload,
            state=state,
            authority_objects=authority_objects,
            bundle_activation={
                "bundle_id": bundle.bundle_id,
                "manifest_hash": bundle.manifest_hash,
                "artifact_hashes": bundle.artifact_hashes,
            },
            escalation_case=escalation_case,
            restricted_payloads=restricted_payloads,
        )

    def review_escalation(
        self,
        *,
        ctx: RequestContext,
        escalation_id: str,
        request: ReviewEscalationRequest,
    ) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        if (
            not ctx.authenticated
            or not ctx.user_id
        ):
            raise PermissionError("authenticated identity required")
        return self._store.review_escalation(
            organization_id=organization_id,
            escalation_id=escalation_id,
            reviewer_principal=ctx.user_id,
            reviewer_role=ctx.member_role,
            decision=request.decision,
            reason=request.reason,
            idempotency_key=request.idempotency_key,
        )

    def execute_assessment(
        self,
        *,
        ctx: RequestContext,
        assessment_id: str,
        request: ExecuteAssessmentRequest,
    ) -> dict[str, Any]:
        organization_id = _organization_id(ctx)
        run_id = f"sg_run_{uuid4().hex}"
        assessment = self._store.get_assessment(
            assessment_id,
            organization_id=organization_id,
        )
        if (
            assessment is None
            or not ctx.workspace_id
            or assessment["workspace_id"] != ctx.workspace_id
        ):
            raise KeyError("assessment not found")
        if assessment['purpose'] == 'workshop_finance_annotations':
            raise ValueError('Teaching annotations are read-only; no governed agent execution')
        existing_run = self._store.get_run_by_execution_idempotency_key(
            organization_id=organization_id,
            execution_idempotency_key=request.idempotency_key,
        )
        if existing_run is not None:
            if (
                existing_run["assessment_id"] != assessment_id
                or existing_run["workspace_id"] != ctx.workspace_id
            ):
                raise ValueError("execution idempotency conflict")
            return existing_run
        if assessment["state"] != "PREPARED":
            raise ValueError("assessment is not executable")
        prepared_payload = assessment["prepared_payload"]
        execution_boundary = bind_governed_execution_boundary(
            prepared_payload=prepared_payload,
        )
        execution = self._agent_adapter_factory().execute(
            run_id=run_id,
            assessment_id=assessment_id,
            assessment_hash=request.assessment_hash,
            transformed_input=prepared_payload["server_owned_transformed_input"],
            transformed_input_hash=prepared_payload["execution_intent"]["transformed_input_hash"],
            execution_boundary=execution_boundary,
            purpose=assessment["purpose"],
            source_evidence=(prepared_payload.get("workshop_fixture") or {}).get("source_evidence"),
            findings=prepared_payload["input_governance"].get("findings") or [],
            decisions=prepared_payload["input_governance"].get("decisions") or [],
        )
        return self._store.consume_assessment_and_create_run(
            assessment_id=assessment_id,
            organization_id=organization_id,
            execution_idempotency_key=request.idempotency_key,
            assessment_hash=request.assessment_hash,
            governed_response=execution.governed_response,
            response_authority_objects=execution.authority_objects,
        )

    def get_run(self, *, ctx: RequestContext, run_id: str) -> dict[str, Any] | None:
        run = self._store.get_run(run_id, organization_id=_organization_id(ctx))
        if run is None or not ctx.workspace_id or run["workspace_id"] != ctx.workspace_id:
            return None
        return run

    def reveal_segment(
        self,
        *,
        ctx: RequestContext,
        response_id: str,
        segment_id: str,
        request: RevealSegmentRequest,
    ) -> dict[str, Any]:
        authorization = SensitiveGovernanceCapabilityResolver(self._store).resolve_operation(
            ctx=ctx,
            organization_id=_organization_id(ctx),
            purpose=request.purpose,
            operation="reveal",
        )
        return self._store.reveal_segment(
            organization_id=_organization_id(ctx),
            principal_id=ctx.user_id,
            workspace_id=str(ctx.workspace_id or ""),
            response_id=response_id,
            segment_id=segment_id,
            purpose=request.purpose,
            operation="reveal",
            authorization_valid=authorization.valid,
            authorization_reason_codes=authorization.reason_codes,
            authorization_snapshot=authorization.snapshot,
        )

    def copy_segment(
        self,
        *,
        ctx: RequestContext,
        response_id: str,
        segment_id: str,
        request: CopySegmentRequest,
    ) -> dict[str, Any]:
        authorization = SensitiveGovernanceCapabilityResolver(self._store).resolve_operation(
            ctx=ctx,
            organization_id=_organization_id(ctx),
            purpose=request.purpose,
            operation="copy",
        )
        return self._store.reveal_segment(
            organization_id=_organization_id(ctx),
            principal_id=ctx.user_id,
            workspace_id=str(ctx.workspace_id or ""),
            response_id=response_id,
            segment_id=segment_id,
            purpose=request.purpose,
            operation="copy",
            authorization_valid=authorization.valid,
            authorization_reason_codes=authorization.reason_codes,
            authorization_snapshot=authorization.snapshot,
        )

    def replay_run(
        self,
        *,
        ctx: RequestContext,
        run_id: str,
        request: ReplayRunRequest,
    ) -> dict[str, Any]:
        del request
        if self.get_run(ctx=ctx, run_id=run_id) is None:
            raise KeyError("run not found")
        return self._store.replay_run(
            organization_id=_organization_id(ctx),
            run_id=run_id,
        )


def contract_metadata() -> dict[str, Any]:
    return {
        "contract_version": CURRENT_CONTRACT_VERSION,
        "supported_source_kinds": ["pasted_text", "txt", "md", "docx", "pdf", "xlsx"],
    }
