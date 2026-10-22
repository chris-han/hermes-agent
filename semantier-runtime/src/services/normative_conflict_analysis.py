"""Deterministic bounded normative-set compatibility analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import TypeAdapter

from contracts.normative_conflict import (
    ConflictCoreV1,
    ConflictResolutionV1,
    NormativeCompatibilityResultV1,
    NormativeEffectV1,
    NormativeSetV1,
)
from contracts.knowledge_conflict_review import (
    ConflictType,
    KnowledgeConflictV1,
    create_knowledge_conflict,
)
from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.ontology_validation import CompiledRulePlanV1, RuntimeSemanticEvaluationBundleV1
from eos.artifact_hashing import content_hash
from services.deterministic_rule_evaluator import DeterministicRuleEvaluator


def _digest(payload: dict) -> str:
    return content_hash(TypeAdapter(dict[str, Any]).dump_python(payload, mode="json"))


class NormativeSetCompiler:
    """Forms the applicable set without using precedence as a conflict suppressor."""

    @staticmethod
    def _effect(rule, effect) -> NormativeEffectV1:
        payload = dict(
            schema_version="normative_effect.v1",
            rule_ref=rule.frame_id,
            modality=rule.modality,
            fact_ref=effect.fact_ref,
            operator=effect.operator,
            expected_value=effect.expected_value,
            evidence_refs=tuple(effect.evidence_refs),
            ir_node_refs=(rule.frame_id, effect.effect_id),
        )
        digest = _digest(payload)
        return NormativeEffectV1(
            effect_ref="normative-effect:" + digest,
            effect_hash="sha256:" + digest,
            **payload,
        )

    def compile(
        self,
        *,
        candidate: NormativeSemanticCandidateSetV1,
        plan: CompiledRulePlanV1,
        evaluation: RuntimeSemanticEvaluationBundleV1,
        context_ref: str,
        context_hash: str,
        formed_at: datetime,
    ) -> NormativeSetV1:
        if (plan.candidate_set_ref, plan.candidate_set_hash) != (
            candidate.candidate_set_ref, candidate.candidate_set_hash
        ):
            raise ValueError("normative set candidate/plan mismatch")
        if (evaluation.plan_ref, evaluation.plan_hash) != (plan.plan_ref, plan.plan_hash):
            raise ValueError("normative set evaluation/plan mismatch")

        rules = {frame.frame_id: frame for frame in candidate.frames if frame.frame_kind == "rule"}
        intrinsic = DeterministicRuleEvaluator().evaluate_rules_without_precedence(
            plan, evaluation.observed_facts
        )
        outcomes = {item.rule_id: item.outcome for item in intrinsic}
        member, inactive, unknown, unresolved, effects = [], [], [], [], [],

        for rule_id in sorted(rules, key=lambda x: x.encode("utf-8")):
            rule = rules[rule_id]
            outcome = outcomes.get(rule_id, "UNKNOWN")
            if outcome == "NO_MATCH":
                inactive.append(rule_id)
                continue
            if outcome == "UNKNOWN":
                unknown.append(rule_id)
                unresolved.append(rule_id)
                continue
            member.append(rule_id)
            if rule.modality not in {"obligation", "prohibition"} or not rule.rule_effects:
                unresolved.append(rule_id)
                continue
            effects.extend(self._effect(rule, effect) for effect in rule.rule_effects)

        if candidate.completeness_state != "COMPLETE":
            unresolved.extend(candidate.unresolved_slot_refs)
        unresolved = sorted(set(unresolved), key=lambda x: x.encode("utf-8"))
        payload = dict(
            schema_version="normative_set.v1",
            organization_id=plan.organization_id,
            workspace_id=plan.workspace_id,
            context_ref=context_ref,
            context_hash=context_hash,
            compiled_rule_plan_ref=plan.plan_ref,
            compiled_rule_plan_hash=plan.plan_hash,
            member_rule_refs=tuple(member),
            effects=tuple(sorted(effects, key=lambda x: x.effect_ref.encode("utf-8"))),
            inactive_rule_refs=tuple(inactive),
            unknown_rule_refs=tuple(unknown),
            closure_state="INCOMPLETE" if unresolved else "COMPLETE",
            unresolved_refs=tuple(unresolved),
            formed_at=formed_at,
        )
        digest = _digest({**payload, "effects": [x.model_dump(mode="json") for x in payload["effects"]]})
        return NormativeSetV1(
            normative_set_ref="normative-set:" + digest,
            normative_set_hash="sha256:" + digest,
            **payload,
        )


class NormativeCompatibilityEvaluator:
    def __init__(self, *, evaluator_ref: str, evaluator_hash: str) -> None:
        self.evaluator_ref = evaluator_ref
        self.evaluator_hash = evaluator_hash

    @staticmethod
    def _numeric_conflict(effects):
        lower = None
        upper = None
        lower_strict = upper_strict = False
        lower_refs, upper_refs = [], []
        for effect in effects:
            if effect.modality != "obligation" or effect.operator not in {"GE", "GT", "LE", "LT"}:
                continue
            value = effect.expected_value
            if type(value) not in {int, float}:
                continue
            if effect.operator in {"GE", "GT"} and (
                lower is None or value > lower or (value == lower and effect.operator == "GT")
            ):
                lower, lower_strict, lower_refs = value, effect.operator == "GT", [effect.effect_ref]
            elif effect.operator in {"LE", "LT"} and (
                upper is None or value < upper or (value == upper and effect.operator == "LT")
            ):
                upper, upper_strict, upper_refs = value, effect.operator == "LT", [effect.effect_ref]
        if (
            lower is not None
            and upper is not None
            and (lower > upper or (lower == upper and (lower_strict or upper_strict)))
        ):
            return tuple(lower_refs + upper_refs)
        return ()

    @staticmethod
    def _categorical_conflict(effects):
        obligations = [e for e in effects if e.modality == "obligation" and e.operator in {"EQ", "IN"}]
        prohibited = [e for e in effects if e.modality == "prohibition" and e.operator in {"EQ", "IN"}]
        allowed = None
        support = []
        for effect in obligations:
            values = {effect.expected_value} if effect.operator == "EQ" else set(effect.expected_value)
            allowed = values if allowed is None else allowed & values
            support.append(effect.effect_ref)
        if obligations and not allowed:
            return tuple(support)
        if allowed is not None:
            for effect in prohibited:
                denied = {effect.expected_value} if effect.operator == "EQ" else set(effect.expected_value)
                if allowed and allowed.issubset(denied):
                    return tuple(support + [effect.effect_ref])
        return ()

    def _conflicting_refs(self, normative_set: NormativeSetV1) -> tuple[str, ...]:
        by_fact = {}
        for effect in normative_set.effects:
            by_fact.setdefault(effect.fact_ref, []).append(effect)
        for effects in by_fact.values():
            refs = self._categorical_conflict(effects) or self._numeric_conflict(effects)
            if refs:
                return tuple(sorted(set(refs), key=lambda x: x.encode("utf-8")))
        return ()

    def evaluate(self, *, normative_set: NormativeSetV1, evaluated_at: datetime) -> NormativeCompatibilityResultV1:
        if normative_set.closure_state != "COMPLETE":
            outcome, proof, refs = "UNKNOWN", "incomplete_semantics", ()
        else:
            refs = self._conflicting_refs(normative_set)
            outcome, proof = ("CONFLICT", "bounded_effect_algebra") if refs else ("COMPATIBLE", "bounded_effect_algebra")
        payload = dict(
            schema_version="normative_compatibility_result.v1",
            normative_set_ref=normative_set.normative_set_ref,
            normative_set_hash=normative_set.normative_set_hash,
            outcome=outcome,
            proof_kind=proof,
            conflicting_effect_refs=refs,
            unresolved_refs=normative_set.unresolved_refs,
            evaluator_ref=self.evaluator_ref,
            evaluator_hash=self.evaluator_hash,
            evaluated_at=evaluated_at,
        )
        digest = _digest(payload)
        return NormativeCompatibilityResultV1(
            result_ref="normative-compatibility:" + digest,
            result_hash="sha256:" + digest,
            **payload,
        )

    def extract_core(
        self,
        *,
        normative_set: NormativeSetV1,
        result: NormativeCompatibilityResultV1,
    ) -> ConflictCoreV1:
        if result.outcome != "CONFLICT":
            raise ValueError("conflict core requires a verified conflict")
        effects = {e.effect_ref: e for e in normative_set.effects}
        selected = [effects[ref] for ref in result.conflicting_effect_refs]
        rule_refs = tuple(sorted({e.rule_ref for e in selected}, key=lambda x: x.encode("utf-8")))
        evidence_refs = tuple(sorted({ref for e in selected for ref in e.evidence_refs}, key=lambda x: x.encode("utf-8")))
        payload = dict(
            schema_version="conflict_core.v1",
            compatibility_result_ref=result.result_ref,
            normative_set_ref=normative_set.normative_set_ref,
            context_ref=normative_set.context_ref,
            member_rule_refs=rule_refs,
            member_effect_refs=tuple(sorted(result.conflicting_effect_refs, key=lambda x: x.encode("utf-8"))),
            minimality="VERIFIED_MINIMAL" if len(selected) == 2 else "NOT_CHECKED",
            evidence_refs=evidence_refs,
        )
        digest = _digest(payload)
        return ConflictCoreV1(
            conflict_core_ref="conflict-core:" + digest,
            conflict_core_hash="sha256:" + digest,
            **payload,
        )


class ConflictResolver:
    """Apply governed precedence only after a verified conflict core exists."""

    def resolve(
        self,
        *,
        core: ConflictCoreV1,
        plan: CompiledRulePlanV1,
        observed_facts: dict[str, Any],
        resolved_at: datetime,
    ) -> ConflictResolutionV1:
        members = set(core.member_rule_refs)
        checks = tuple(
            item
            for item in DeterministicRuleEvaluator().evaluate_rule_precedences(
                plan, observed_facts
            )
            if item.outcome == "MATCH"
            and item.higher_priority_ref in members
            and item.lower_priority_ref in members
        )
        candidates = []
        for higher in sorted(members, key=lambda x: x.encode("utf-8")):
            covered = {
                item.lower_priority_ref
                for item in checks
                if item.higher_priority_ref == higher
            }
            if members - {higher} <= covered:
                candidates.append(higher)
        if len(candidates) == 1:
            outcome = "RESOLVED_BY_PRECEDENCE"
            controlling = (candidates[0],)
            evidence = tuple(sorted({
                ref for item in checks
                if item.higher_priority_ref == candidates[0]
                for ref in item.evidence_refs
            }, key=lambda x: x.encode("utf-8")))
        else:
            outcome = "UNRESOLVED"
            controlling = ()
            evidence = tuple(sorted({
                ref for item in checks for ref in item.evidence_refs
            }, key=lambda x: x.encode("utf-8")))
        payload = dict(
            schema_version="conflict_resolution.v1",
            conflict_core_ref=core.conflict_core_ref,
            outcome=outcome,
            controlling_rule_refs=controlling,
            resolution_evidence_refs=evidence,
            resolved_at=resolved_at,
        )
        digest = _digest(payload)
        return ConflictResolutionV1(
            resolution_ref="conflict-resolution:" + digest,
            resolution_hash="sha256:" + digest,
            **payload,
        )


def project_conflict_core(
    *,
    core: ConflictCoreV1,
    detected_at: str,
    detector_ref: str,
    required_authority: str | None = None,
) -> KnowledgeConflictV1:
    """Project a verified N-ary core without fabricating pairwise graph conflicts."""
    return create_knowledge_conflict(
        conflict_type=ConflictType.ONTOLOGY_CONFLICT,
        object_refs=core.member_rule_refs,
        evidence_refs=core.evidence_refs,
        detected_at=detected_at,
        detected_by="system",
        detector_ref=detector_ref,
        required_authority=required_authority,
        proof_ref=core.conflict_core_ref,
        normative_set_ref=core.normative_set_ref,
        context_ref=core.context_ref,
    )
