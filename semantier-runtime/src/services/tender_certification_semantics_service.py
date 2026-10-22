"""Deterministic Tender A1 semantics and reviewed-oracle inputs for certification."""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from contracts.normative_semantics import (
    EntityDistinctionFrameV1,
    ExceptionFrameV1,
    NormativeSemanticCandidateSetV1,
    PrecedenceFrameV1,
    QualificationFrameV1,
    RuleFrameV1,
    ScopeFrameV1,
    SemanticPredicateV1,
    TemporalConditionFrameV1,
)
from contracts.normative_schema import NormativeSchemaContractV1
from contracts.ontology_building import CompetencyContractV1
from contracts.ontology_certification import NormativeAuthorityDependencySetV1
from eos.artifact_hashing import content_hash


class TenderCertificationSemanticsService:
    """Compile the bounded Tender screening semantics used by the first A1 proving loop."""

    COMPILER_REF = "compiler:tender-certification-semantics:v1"
    PROFILE_REF = "tender_corpus_schema_induction.v0_1"
    EFFECTIVE_FROM = "2024-05-01"
    EFFECTIVE_UNTIL = "9999-12-31"

    @staticmethod
    def _frame(contract, **payload):
        def jsonable(value):
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
            if isinstance(value, tuple):
                return [jsonable(item) for item in value]
            if isinstance(value, dict):
                return {key: jsonable(item) for key, item in value.items()}
            return value

        digest = content_hash(jsonable(payload))
        return contract(frame_hash=f"sha256:{digest}", **payload)

    @staticmethod
    def _evidence_by_role(
        authority: NormativeAuthorityDependencySetV1,
    ) -> dict[str, tuple[str, ...]]:
        values: dict[str, list[str]] = {}
        for dependency in authority.dependencies:
            for section in dependency.sections:
                evidence_ref = f"legal-section:{section.section_id}"
                for role in section.dependency_roles:
                    values.setdefault(role, []).append(evidence_ref)
        return {
            role: tuple(dict.fromkeys(refs))
            for role, refs in values.items()
        }

    def compile(
        self,
        *,
        authority: NormativeAuthorityDependencySetV1,
        competency: CompetencyContractV1,
        schema: NormativeSchemaContractV1,
        source_identity_ref: str,
        source_hash: str,
        canonical_source_ir_ref: str,
        canonical_source_ir_hash: str,
        compiled_at: datetime,
    ) -> NormativeSemanticCandidateSetV1:
        question = next(
            (item for item in competency.questions if item.competency_id == "cq.constraint"),
            None,
        )
        if question is None or question.review_state != "confirmed" or question.criticality != "critical":
            raise ValueError("Tender certification requires confirmed critical cq.constraint")
        if schema.competency_contract_ref != competency.competency_contract_ref:
            raise ValueError("normative schema does not bind the reviewed competency")
        roles = self._evidence_by_role(authority)
        required_roles = {
            "rule_authority",
            "scope_authority",
            "exception_authority",
            "qualification_authority",
            "effectivity_authority",
            "precedence_authority",
        }
        if not required_roles.issubset(roles):
            raise ValueError("authority dependency set lacks required Tender semantic roles")
        source_evidence = tuple(question.evidence_refs)
        rule_evidence = tuple(dict.fromkeys((*roles["rule_authority"], *source_evidence)))
        scope_evidence = tuple(dict.fromkeys((*roles["scope_authority"], *source_evidence)))
        qualification_evidence = tuple(
            dict.fromkeys((*roles["qualification_authority"], *source_evidence))
        )
        exception_evidence = roles["exception_authority"]
        precedence_evidence = roles["precedence_authority"]
        temporal_evidence = roles["effectivity_authority"]
        all_authority = tuple(
            dict.fromkeys(ref for refs in roles.values() for ref in refs)
        )

        rule_id = "rule.unreasonable_eligibility_restriction"
        exception_id = "exception.authorized_legal_exception"
        scope_id = "scope.tender_eligibility_condition"
        qualification_id = "qualification.unreasonable_restriction"
        precedence_id = "precedence.authorized_exception_over_screening_rule"
        temporal_id = "temporal.post_2024_fair_competition_rule"
        frames = [
            self._frame(
                EntityDistinctionFrameV1,
                frame_id="entity_distinction.screening_reference_vs_legal_authority",
                display_name="Screening reference versus binding legal authority",
                evidence_refs=tuple(dict.fromkeys((*source_evidence, *all_authority))),
                source_region_refs=("poc.screening_reference", "authority.official_sources"),
                left_entity_ref="entity.screening_reference",
                right_entity_ref="entity.binding_legal_authority",
                outcome_effect=(
                    "A sensitive-term or screening match is evidence for review; it is not by itself a binding legal conclusion."
                ),
            ),
            self._frame(
                ExceptionFrameV1,
                frame_id=exception_id,
                display_name="Authorized statutory or regulatory exception",
                evidence_refs=exception_evidence,
                source_region_refs=("authority.fair_competition.exception",),
                applies_to_rule_ref=rule_id,
                conditions=(
                    SemanticPredicateV1(
                        predicate_id="predicate.authorized_legal_exception",
                        operator="EQ",
                        fact_ref="fact.authorized_legal_exception",
                        expected_value=True,
                        evidence_refs=exception_evidence,
                    ),
                ),
            ),
            self._frame(
                PrecedenceFrameV1,
                frame_id=precedence_id,
                display_name="Authorized exception precedence over screening rule",
                evidence_refs=precedence_evidence,
                source_region_refs=("authority.fair_competition.precedence",),
                higher_priority_ref=exception_id,
                lower_priority_ref=rule_id,
                condition_refs=("predicate.authorized_legal_exception",),
            ),
            self._frame(
                QualificationFrameV1,
                frame_id=qualification_id,
                display_name="Unreasonable restriction qualification",
                evidence_refs=qualification_evidence,
                source_region_refs=("authority.tender.qualification", "poc.reference_examples"),
                qualified_term="unreasonable_eligibility_restriction",
                definition=(
                    "A condition is a screening match only when normalized evidence establishes that it constrains tender eligibility "
                    "and that the restriction is unreasonable under the pinned authority. A keyword or example alone is insufficient."
                ),
            ),
            self._frame(
                RuleFrameV1,
                frame_id=rule_id,
                display_name="Screen unreasonable tender eligibility restrictions",
                evidence_refs=rule_evidence,
                source_region_refs=("authority.tender.rule", "poc.reference_examples"),
                modality="prohibition",
                actor_ref="actor.tenderer_or_policy_maker",
                object_ref="object.tender_eligibility_condition",
                conditions=(
                    SemanticPredicateV1(
                        predicate_id="predicate.condition_constrains_eligibility",
                        operator="EQ",
                        fact_ref="fact.condition_constrains_eligibility",
                        expected_value=True,
                        evidence_refs=rule_evidence,
                    ),
                    SemanticPredicateV1(
                        predicate_id="predicate.condition_is_unreasonable_restriction",
                        operator="EQ",
                        fact_ref="fact.condition_is_unreasonable_restriction",
                        expected_value=True,
                        evidence_refs=qualification_evidence,
                    ),
                    SemanticPredicateV1(
                        predicate_id="predicate.document_date_on_or_after_2024_05_01",
                        operator="DATE_ON_OR_AFTER",
                        fact_ref="fact.document_date",
                        expected_value=self.EFFECTIVE_FROM,
                        evidence_refs=temporal_evidence,
                    ),
                ),
                consequence="screening_match_requires_evidence_review",
                exception_refs=(exception_id,),
                scope_refs=(scope_id,),
                qualification_refs=(qualification_id,),
                temporal_condition_refs=(temporal_id,),
                precedence_refs=(precedence_id,),
            ),
            self._frame(
                ScopeFrameV1,
                frame_id=scope_id,
                display_name="Tender eligibility screening scope",
                evidence_refs=scope_evidence,
                source_region_refs=("authority.tender.scope", "poc.screening_scope"),
                applies_to_rule_ref=rule_id,
                actor_scope=("actor.tenderer", "actor.policy_maker"),
                object_scope=("object.qualification_condition", "object.evaluation_condition"),
            ),
            self._frame(
                TemporalConditionFrameV1,
                frame_id=temporal_id,
                display_name="Post-2024 fair-competition screening effectivity",
                evidence_refs=temporal_evidence,
                source_region_refs=("authority.fair_competition.effectivity",),
                effective_from=self.EFFECTIVE_FROM,
                effective_until=self.EFFECTIVE_UNTIL,
                event_time_ref="fact.document_date",
            ),
        ]
        frames.sort(key=lambda frame: frame.frame_id.encode("utf-8"))
        compiler_hash = "sha256:" + sha256(Path(__file__).read_bytes()).hexdigest()
        config_basis = {
            "effective_from": self.EFFECTIVE_FROM,
            "effective_until": self.EFFECTIVE_UNTIL,
            "authority_set": authority.set_hash,
            "schema": schema.schema_hash,
        }
        values = {
            "organization_id": competency.organization_id,
            "workspace_id": competency.workspace_id,
            "source_identity_ref": source_identity_ref,
            "source_hash": source_hash,
            "canonical_source_ir_ref": canonical_source_ir_ref,
            "canonical_source_ir_hash": canonical_source_ir_hash,
            "competency_contract_ref": competency.competency_contract_ref,
            "competency_contract_hash": competency.competency_contract_hash,
            "normative_schema_ref": schema.schema_ref,
            "normative_schema_hash": schema.schema_hash,
            "profile_ref": schema.profile_ref,
            "profile_hash": schema.profile_hash,
            "compiler_ref": self.COMPILER_REF,
            "compiler_hash": compiler_hash,
            "prompt_hash": "sha256:" + content_hash({"prompt": "none:deterministic"}),
            "provider_ref": "provider:deterministic-authority-pinned",
            "model_ref": "none",
            "config_hash": "sha256:" + content_hash(config_basis),
            "frames": [frame.model_dump(mode="json") for frame in frames],
        }
        digest = content_hash(values)
        return NormativeSemanticCandidateSetV1(
            candidate_set_ref=f"normative-candidate-set:{digest}",
            candidate_set_hash=f"sha256:{digest}",
            frames=tuple(frames),
            completeness_state="COMPLETE",
            unresolved_slot_refs=(),
            compiled_at=compiled_at,
            **{key: value for key, value in values.items() if key != "frames"},
        )

    def validation_source(
        self,
        *,
        authority: NormativeAuthorityDependencySetV1,
        competency: CompetencyContractV1,
    ) -> dict[str, Any]:
        roles = self._evidence_by_role(authority)
        question = next(item for item in competency.questions if item.competency_id == "cq.constraint")
        source_evidence = tuple(question.evidence_refs)
        dimension_evidence = {
            "rule": tuple(dict.fromkeys((*roles["rule_authority"], *source_evidence))),
            "modality": tuple(dict.fromkeys((*roles["rule_authority"], *source_evidence))),
            "exception": roles["exception_authority"],
            "scope": tuple(dict.fromkeys((*roles["scope_authority"], *source_evidence))),
            "qualification": tuple(dict.fromkeys((*roles["qualification_authority"], *source_evidence))),
            "precedence": roles["precedence_authority"],
            "temporal_condition": roles["effectivity_authority"],
            "entity_distinction": tuple(dict.fromkeys((*source_evidence, *roles["rule_authority"]))),
            "evidence_binding": tuple(dict.fromkeys((*source_evidence, *(ref for refs in roles.values() for ref in refs)))),
        }
        cases = self.behavioral_cases(authority=authority, competency=competency)
        return {
            "semantic_dimensions": [
                {"dimension": dimension, "critical": True, "evidence_refs": list(refs)}
                for dimension, refs in dimension_evidence.items()
            ],
            "competency_oracle": [
                {
                    "competency_id": case["competency_id"],
                    "case_kind": case["case_kind"],
                    "criticality": "critical",
                    "question": case["question"],
                    "expected_result": case["expected_result"],
                    "required_semantic_dimensions": list(question.required_semantic_dimensions),
                    "evidence_refs": list(case["evidence_refs"]),
                }
                for case in cases
            ],
        }

    def behavioral_cases(
        self,
        *,
        authority: NormativeAuthorityDependencySetV1,
        competency: CompetencyContractV1,
    ) -> tuple[dict[str, Any], ...]:
        question = next(item for item in competency.questions if item.competency_id == "cq.constraint")
        roles = self._evidence_by_role(authority)
        evidence = tuple(dict.fromkeys((*question.evidence_refs, *roles["rule_authority"])))
        common = {"fact.condition_constrains_eligibility": True, "fact.document_date": "2026-09-17"}
        return (
            {
                "competency_id": "cq.constraint.positive",
                "case_kind": "positive",
                "question": "Does an evidenced unreasonable eligibility restriction require screening review?",
                "expected_result": "MATCH",
                "observed_facts": {**common, "fact.condition_is_unreasonable_restriction": True},
                "evidence_refs": evidence,
            },
            {
                "competency_id": "cq.constraint.hard_negative",
                "case_kind": "hard_negative",
                "question": "Does a condition that constrains eligibility but is evidenced as reasonable match the restriction rule?",
                "expected_result": "NO_MATCH",
                "observed_facts": {**common, "fact.condition_is_unreasonable_restriction": False},
                "evidence_refs": evidence,
            },
            {
                "competency_id": "cq.constraint.authorized_exception",
                "case_kind": "boundary",
                "question": "Does an otherwise matching restriction remain a match when an evidenced authorized legal exception applies?",
                "expected_result": "NO_MATCH",
                "observed_facts": {
                    **common,
                    "fact.condition_is_unreasonable_restriction": True,
                    "fact.authorized_legal_exception": True,
                },
                "evidence_refs": tuple(dict.fromkeys((*evidence, *roles["exception_authority"]))),
            },
            {
                "competency_id": "cq.constraint.missing_material_fact",
                "case_kind": "ambiguity",
                "question": "What happens when the source constrains eligibility but there is no evidence establishing reasonableness?",
                "expected_result": "UNKNOWN",
                "observed_facts": common,
                "evidence_refs": evidence,
            },
        )


__all__ = ["TenderCertificationSemanticsService"]
