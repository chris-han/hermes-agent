"""Consume previously evaluated A1 quality evidence; never produce quality during certification."""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path

from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.ontology_admission import CertifiedSemanticExecutionPlanV1
from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_paired_evaluation import ArtifactPayloadPinV1
from contracts.ontology_quality import QualityEvaluationRunV1
from contracts.ontology_validation import CompiledRulePlanV1
from contracts.validation_contract import ValidationContractV1
from eos.artifact_hashing import content_hash


class OntologyA1CertificationEvaluator:
    """Check the exact retained qualification basis before issuing certification."""

    def verify_execution_inputs(
        self, *, quality_evaluation: QualityEvaluationRunV1, validation: ValidationContractV1,
        candidate: NormativeSemanticCandidateSetV1, plan: CompiledRulePlanV1,
        store=None,
    ) -> None:
        quality_evaluation = QualityEvaluationRunV1.model_validate(quality_evaluation)
        if len({(item.organization_id, item.workspace_id)
            for item in (quality_evaluation, validation, candidate, plan)}) != 1:
            raise PermissionError("certification inputs cross tenant scope")
        from services.ontology_quality_profile_service import QualityProfileService
        profile = quality_evaluation.profile_definition
        QualityProfileService.verify_consumer_basis(store=store, profile=profile, run=quality_evaluation)
        quality_evaluation.require_basis(quality_profile=profile.pin(),
            subject=ArtifactPinV1(artifact_kind="normative_semantic_candidate_set",
                artifact_ref=candidate.candidate_set_ref, artifact_hash=candidate.candidate_set_hash),
            competency=ArtifactPinV1(artifact_kind="competency_contract",
                artifact_ref=candidate.competency_contract_ref, artifact_hash=candidate.competency_contract_hash),
            validation=ArtifactPinV1(artifact_kind="validation_contract",
                artifact_ref=validation.validation_contract_ref, artifact_hash=validation.validation_contract_hash))
        if quality_evaluation.disposition not in {"QUALIFIED", "QUALIFIED_WITH_WARNINGS"}:
            raise ValueError("certification requires a qualified quality evaluation")
        if quality_evaluation.disposition == "QUALIFIED_WITH_WARNINGS" and not profile.allow_warnings:
            raise ValueError("assurance policy does not permit quality warnings")
        if validation.review_state != "accepted":
            raise ValueError("certification requires an accepted independent validation oracle")
        if ((plan.candidate_set_ref, plan.candidate_set_hash) != (candidate.candidate_set_ref, candidate.candidate_set_hash)
            or (plan.validation_contract_ref, plan.validation_contract_hash) != (
                validation.validation_contract_ref, validation.validation_contract_hash)):
            raise ValueError("certification compiled plan basis pin mismatch")
        results = {item.requirement_id: item for item in quality_evaluation.results}
        retained = results['historical_immutability'].evidence
        for artifact, ref, hash_ in ((candidate, 'candidate_set_ref', 'candidate_set_hash'),
            (validation, 'validation_contract_ref', 'validation_contract_hash'), (plan, 'plan_ref', 'plan_hash')):
            payload = ArtifactPayloadPinV1.of(artifact, ref, hash_)
            expected = ArtifactPinV1(artifact_kind='artifact_payload', artifact_ref=payload.artifact_ref,
                                     artifact_hash=payload.payload_hash)
            if expected not in retained:
                raise ValueError('certification retained input payload pin mismatch')
        plan_pin = ArtifactPinV1(artifact_kind='compiled_rule_plan', artifact_ref=plan.plan_ref, artifact_hash=plan.plan_hash)
        if plan_pin not in results['deterministic_reconstruction'].evidence:
            raise ValueError('certification reconstruction plan pin mismatch')

    def execution_plan(
        self,
        *,
        quality_evaluation: QualityEvaluationRunV1,
        validation: ValidationContractV1,
        candidate: NormativeSemanticCandidateSetV1,
        plan: CompiledRulePlanV1,
        certified_at: datetime,
        runtime_program_hash: str | None = None,
        environment_fingerprint: str | None = None,
        store=None,
    ) -> CertifiedSemanticExecutionPlanV1:
        self.verify_execution_inputs(quality_evaluation=quality_evaluation, validation=validation,
            candidate=candidate, plan=plan, store=store)
        assurance_policy_ref = "assurance-policy:A1_INTERNAL_POLICY_MVL:tender-screening-v1"
        assurance_policy_hash = "sha256:" + content_hash(
            {
                "profile": "A1_INTERNAL_POLICY_MVL",
                "scope": "bounded_internal_tender_restriction_screening",
                "binding_legal_conclusion": False,
            }
        )
        runtime_program_hash = runtime_program_hash or "sha256:" + sha256(Path(__file__).read_bytes()).hexdigest()
        projection_compiler_ref = "compiler:ontology-rule-plan:v1"
        projection_compiler_hash = plan.compiler_hash
        environment_fingerprint = environment_fingerprint or "sha256:" + content_hash(
            {
                "semantic_profile": plan.semantic_profile,
                "runtime": "python-in-process",
                "evaluator": "semantier_deterministic_v1",
            }
        )
        basis = {
            "organization_id": candidate.organization_id,
            "workspace_id": candidate.workspace_id,
            "assurance_profile": "A1_INTERNAL_POLICY_MVL",
            "assurance_policy_ref": assurance_policy_ref,
            "assurance_policy_hash": assurance_policy_hash,
            "canonical_normative_ir_ref": candidate.candidate_set_ref,
            "canonical_normative_ir_hash": candidate.candidate_set_hash,
            "validation_contract_ref": validation.validation_contract_ref,
            "validation_contract_hash": validation.validation_contract_hash,
            "quality_profile": quality_evaluation.quality_profile,
            "quality_evaluation": quality_evaluation.pin(),
            "compiled_rule_plan_ref": plan.plan_ref,
            "compiled_rule_plan_hash": plan.plan_hash,
            "runtime_program_ref": "runtime:semantier_deterministic_v1",
            "runtime_program_hash": runtime_program_hash,
            "projection_compiler_ref": projection_compiler_ref,
            "projection_compiler_hash": projection_compiler_hash,
            "environment_fingerprint": environment_fingerprint,
            "justification_projection": "runtime_justification.v1",
        }
        digest = content_hash({key: value.model_dump(mode="json") if isinstance(value, ArtifactPinV1) else value
                               for key, value in basis.items()})
        return CertifiedSemanticExecutionPlanV1(
            execution_plan_ref=f"certified-execution-plan:{digest}",
            execution_plan_hash=f"sha256:{digest}",
            certified_at=certified_at,
            **basis,
        )


__all__ = ["OntologyA1CertificationEvaluator"]
