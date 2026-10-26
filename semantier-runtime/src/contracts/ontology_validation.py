"""Deterministic A1 runtime-plan and evaluation evidence contracts."""

from __future__ import annotations

from datetime import datetime, timedelta
from graphlib import CycleError, TopologicalSorter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


from eos.artifact_hashing import content_hash
from contracts.normative_conflict import NormativeDomainBasisV2


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def timestamps_are_utc(self) -> "_Contract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class CompiledPredicateV1(_Contract):
    predicate_id: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    fact_ref: str = Field(min_length=1)
    expected_value: Any = None
    ir_node_ref: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    strict_operands: bool | None = Field(default=None, exclude_if=lambda value: value is None)
    comparison_fact_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)


class CompiledExceptionGroupV1(_Contract):
    ir_node_ref: str = Field(min_length=1)
    predicate_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class CompiledRuleV1(_Contract):
    rule_id: str = Field(min_length=1)
    ir_node_ref: str = Field(min_length=1)
    competency_refs: tuple[str, ...] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    predicates: tuple[CompiledPredicateV1, ...] = Field(min_length=1)
    exception_predicates: tuple[CompiledPredicateV1, ...]
    input_requirements: tuple[CompiledPredicateV1, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None)
    exception_groups: tuple[CompiledExceptionGroupV1, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None)
    source_modality: Literal["obligation", "prohibition", "permission"] | None = Field(
        default=None, exclude_if=lambda value: value is None)
    disposition_effects: tuple[CompiledPredicateV1, ...] | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None)
    result_when_matched: Literal["MATCH"] = "MATCH"
    interpretation_required: str | None = Field(default=None, min_length=1,
        exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def exact_exception_groups(self):
        if (self.source_modality is None) != (self.disposition_effects is None):
            raise ValueError("source modality and disposition effects must be paired")
        if self.exception_groups is None:
            return self  # Historical flat conjunction and serialized bytes remain unchanged.
        groups = self.exception_groups
        refs = [group.ir_node_ref for group in groups]
        predicates = {item.predicate_id: item for item in self.exception_predicates}
        flattened = tuple(ref for group in groups for ref in group.predicate_ids)
        if (len(refs) != len(set(refs)) or len(predicates) != len(self.exception_predicates)
            or flattened != tuple(item.predicate_id for item in self.exception_predicates)):
            raise ValueError("exception groups must exactly partition exception predicates")
        for group in groups:
            if any(predicates[ref].ir_node_ref != group.ir_node_ref
                or not set(predicates[ref].evidence_refs).issubset(group.evidence_refs)
                for ref in group.predicate_ids):
                raise ValueError("exception group source binding mismatch")
        return self


class CompiledPrecedenceV1(_Contract):
    ir_node_ref: str = Field(min_length=1)
    higher_priority_ref: str = Field(min_length=1)
    lower_priority_ref: str = Field(min_length=1)
    higher_priority_kind: Literal["rule", "exception"]
    conditions: tuple[CompiledPredicateV1, ...]
    evidence_refs: tuple[str, ...] = Field(min_length=1)



class CompiledRulePlanV1(_Contract):
    domain_basis: NormativeDomainBasisV2 | None = Field(default=None, exclude_if=lambda v: v is None)
    teaching_candidate: Literal[True] | None = Field(default=None, exclude_if=lambda value: value is None)
    schema_version: Literal["compiled_rule_plan.v1"] = "compiled_rule_plan.v1"
    plan_ref: str = Field(min_length=1)
    plan_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    candidate_set_ref: str = Field(min_length=1)
    candidate_set_hash: str = Field(pattern=SHA256_PATTERN)
    validation_contract_ref: str = Field(min_length=1)
    validation_contract_hash: str = Field(pattern=SHA256_PATTERN)
    compiler_ref: str = Field(min_length=1)
    compiler_hash: str = Field(pattern=SHA256_PATTERN)
    semantic_profile: Literal["semantier_deterministic_v1"]
    rules: tuple[CompiledRuleV1, ...] = Field(min_length=1)
    precedences: tuple[CompiledPrecedenceV1, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None)
    compiled_at: datetime

    @model_validator(mode="after")
    def executable_precedence(self):
        if self.precedences is None:
            return self
        rules = {rule.rule_id: rule for rule in self.rules}
        refs = [item.ir_node_ref for item in self.precedences]
        if len(rules) != len(self.rules) or len(refs) != len(set(refs)):
            raise ValueError("precedence requires unique rule and relation identities")
        dependencies = {ref: set() for ref in rules}
        for item in self.precedences:
            lower = rules.get(item.lower_priority_ref)
            if lower is None or item.higher_priority_ref == item.lower_priority_ref:
                raise ValueError("precedence lower endpoint must be a distinct rule")
            if item.higher_priority_kind == "rule":
                if item.higher_priority_ref not in rules:
                    raise ValueError("precedence higher rule is missing")
                dependencies[lower.rule_id].add(item.higher_priority_ref)
            elif item.higher_priority_ref not in {
                group.ir_node_ref for group in (lower.exception_groups or ())
            }:
                raise ValueError("precedence exception must belong to its lower rule")
        try:
            tuple(TopologicalSorter(dependencies).static_order())
        except CycleError as exc:
            raise ValueError("precedence cycle is not executable") from exc
        return self


class LegalClaimRuleBindingV2(_Contract):
    claim_ref: str
    claim_hash: str
    source_span_hash: str
    rule_ref: str


class LegalPrecedenceBasisV2(_Contract):
    """Explicit mapping certified with the existing legal issue-model owner."""
    schema_version: Literal["legal_precedence_basis.v2"] = "legal_precedence_basis.v2"
    organization_id: str
    issue_model_id: str
    ontology_version_ref: str
    plan: CompiledRulePlanV1
    bindings: tuple[LegalClaimRuleBindingV2, ...]

    @model_validator(mode="after")
    def exact_plan_and_mapping(self):
        if self.organization_id != self.plan.organization_id:
            raise ValueError("legal precedence organization mismatch")
        if len({b.claim_ref for b in self.bindings}) != len(self.bindings) or len({b.rule_ref for b in self.bindings}) != len(self.bindings):
            raise ValueError("legal precedence requires unique exact endpoints")
        if {b.rule_ref for b in self.bindings} != {r.rule_id for r in self.plan.rules}:
            raise ValueError("legal precedence rule mapping is incomplete")
        payload = self.plan.model_dump(mode="json", exclude={"schema_version", "plan_ref", "plan_hash", "organization_id", "workspace_id", "compiled_at"})
        digest = content_hash(dict(payload, compiled_at=self.plan.compiled_at.isoformat()))
        if (self.plan.plan_ref, self.plan.plan_hash) != ("compiled-rule-plan:"+digest, "sha256:"+digest):
            raise ValueError("legal precedence compiled plan identity mismatch")
        return self


class PredicateEvaluationV1(_Contract):
    predicate_id: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    fact_ref: str = Field(min_length=1)
    result: Literal["SATISFIED", "FAILED", "UNKNOWN", "UNSUPPORTED"]
    observed_value: Any = None
    expected_value: Any = None
    ir_node_ref: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    comparison_fact_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)
    comparison_observed_value: Any = Field(default=None, exclude_if=lambda value: value is None)


class PrecedenceEvaluationV1(_Contract):
    ir_node_ref: str
    higher_priority_ref: str
    lower_priority_ref: str
    higher_outcome: Literal["MATCH", "NO_MATCH", "UNKNOWN"]
    condition_results: tuple[PredicateEvaluationV1, ...]
    outcome: Literal["MATCH", "NO_MATCH", "UNKNOWN"]
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class CandidateRuleEvaluationV1(_Contract):
    rule_id: str = Field(min_length=1)
    outcome: Literal["MATCH", "NO_MATCH", "UNKNOWN"]
    predicate_results: tuple[PredicateEvaluationV1, ...]
    exception_results: tuple[PredicateEvaluationV1, ...]
    rejection_reasons: tuple[str, ...]
    input_requirement_results: tuple[PredicateEvaluationV1, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None)
    precedence_results: tuple[PrecedenceEvaluationV1, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None)


class RuntimeJustificationV1(_Contract):
    schema_version: Literal["runtime_justification.v1"] = "runtime_justification.v1"
    result: Literal["MATCH", "NO_MATCH", "UNKNOWN"]
    reason_code: str = Field(min_length=1)
    rule_refs: tuple[str, ...]
    ir_node_refs: tuple[str, ...]
    competency_refs: tuple[str, ...]
    source_evidence_refs: tuple[str, ...]


class RuntimeSemanticEvaluationBundleV1(_Contract):
    schema_version: Literal["runtime_semantic_evaluation_bundle.v1"] = (
        "runtime_semantic_evaluation_bundle.v1"
    )
    bundle_ref: str = Field(min_length=1)
    bundle_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    execution_epoch_ref: str = Field(min_length=1)
    plan_ref: str = Field(min_length=1)
    plan_hash: str = Field(pattern=SHA256_PATTERN)
    result: Literal["MATCH", "NO_MATCH", "UNKNOWN"]
    observed_facts: dict[str, Any]
    target_evidence_refs: tuple[str, ...] = Field(min_length=1)
    candidate_rule_evaluations: tuple[CandidateRuleEvaluationV1, ...] = Field(
        min_length=1
    )
    satisfied_predicates: tuple[str, ...]
    failed_predicates: tuple[str, ...]
    exception_checks: tuple[PredicateEvaluationV1, ...]
    unknown_inputs: tuple[str, ...]
    unsupported_semantics: tuple[str, ...]
    competency_refs: tuple[str, ...] = Field(min_length=1)
    ir_node_refs: tuple[str, ...] = Field(min_length=1)
    source_evidence_refs: tuple[str, ...] = Field(min_length=1)
    justification: RuntimeJustificationV1
    all_rule_justifications: tuple[RuntimeJustificationV1, ...] | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None)
    evaluated_at: datetime

    @model_validator(mode="after")
    def result_has_learning_grade_trace(self) -> "RuntimeSemanticEvaluationBundleV1":
        if self.all_rule_justifications is not None:
            evaluations = self.candidate_rule_evaluations
            if (len(evaluations) != len(self.all_rule_justifications)
                or len({item.rule_id for item in evaluations}) != len(evaluations)
                or any(proof.rule_refs != (item.rule_id,) or proof.result != item.outcome
                    for item, proof in zip(evaluations, self.all_rule_justifications, strict=True))):
                raise ValueError("complete rule evaluations require exact per-rule justifications")
        if self.result == "MATCH" and not self.satisfied_predicates:
            raise ValueError("MATCH requires supporting predicates")
        if self.result == "NO_MATCH" and not any(
            item.rejection_reasons or item.exception_results
            for item in self.candidate_rule_evaluations
        ):
            raise ValueError("NO_MATCH requires bounded rejection or exception trace")
        if self.result == "UNKNOWN" and not (
            self.unknown_inputs or self.unsupported_semantics
        ):
            raise ValueError("UNKNOWN requires exact unknown inputs or unsupported semantics")
        return self


__all__ = [
    "CandidateRuleEvaluationV1",
    "CompiledPredicateV1",
    "CompiledPrecedenceV1",
    "CompiledExceptionGroupV1",
    "CompiledRulePlanV1",
    "CompiledRuleV1",
    "PredicateEvaluationV1",
    "PrecedenceEvaluationV1",
    "RuntimeJustificationV1",
    "RuntimeSemanticEvaluationBundleV1",
]
