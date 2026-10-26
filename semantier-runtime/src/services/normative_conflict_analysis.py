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
    NormativeAnalysisScopeV2,
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
            unresolved.extend(candidate.unresolved_slot_refs or (candidate.candidate_set_ref,))
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
        if plan.domain_basis is not None:
            plan.domain_basis.verify_candidate(candidate)
            payload["domain_basis"] = plan.domain_basis.model_dump(mode="json")
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
    def _group(effects, definition):
        """Intersect supported restrictions using the declared source domain."""
        import math
        if definition is None or definition.value_type not in {"number", "integer", "string", "boolean"}:
            return (), tuple(e.effect_ref for e in effects)
        supported, unsupported = [], []
        for e in effects:
            if definition.accepts_predicate(e.operator, e.expected_value):
                supported.append(e)
            else:
                unsupported.append(e.effect_ref)
        if not supported:
            return (), tuple(unsupported)
        if definition.value_type in {"number", "integer"}:
            lower, upper = float("-inf"), float("inf")
            ls = us = False
            excluded = set()
            for e in supported:
                op, v = e.operator, e.expected_value
                if op == "IN":
                    unsupported.append(e.effect_ref)
                    continue  # Numeric disjunction is outside this algebra.
                if e.modality == "prohibition":
                    if op == "EQ":
                        excluded.add(v)
                        continue
                    op = {"GE": "LT", "GT": "LE", "LE": "GT", "LT": "GE"}[op]
                if op == "EQ":
                    if v > lower:
                        lower, ls = v, False
                    if v < upper:
                        upper, us = v, False
                elif op in {"GE", "GT"}:
                    if v > lower or (v == lower and op == "GT"):
                        lower, ls = v, op == "GT"
                elif v < upper or (v == upper and op == "LT"):
                    upper, us = v, op == "LT"
            if definition.value_type == "integer":
                if lower != float("-inf"):
                    lower = math.floor(lower) + 1 if ls else math.ceil(lower)
                if upper != float("inf"):
                    upper = math.ceil(upper) - 1 if us else math.floor(upper)
                conflict = lower > upper or (
                    lower != float("-inf") and upper != float("inf")
                    and upper - lower + 1 <= len(excluded)
                    and sum(lower <= v <= upper for v in excluded) == upper - lower + 1)
            else:
                conflict = lower > upper or (lower == upper and (ls or us or lower in excluded))
        else:
            allowed = set(definition.allowed_values) if definition.allowed_values is not None else (
                {True, False} if definition.value_type == "boolean" else None)
            denied = set()
            for e in supported:
                values = {e.expected_value} if e.operator == "EQ" else set(e.expected_value)
                if e.modality == "obligation":
                    allowed = values if allowed is None else allowed & values
                else:
                    denied |= values
            conflict = allowed is not None and not (allowed - denied)
        checked = tuple(sorted({e.effect_ref for e in supported} - set(unsupported)))
        return (checked if conflict else ()), tuple(unsupported)

    def _analysis(self, normative_set):
        by_fact = {}
        for effect in normative_set.effects:
            by_fact.setdefault(effect.fact_ref, []).append(effect)
        definitions = {f.fact_ref: f for f in normative_set.domain_basis.facts} if normative_set.domain_basis else {}
        groups, unsupported = [], []
        for fact, effects in sorted(by_fact.items()):
            refs, unknown = self._group(effects, definitions.get(fact))
            if refs:
                groups.append(refs)
            unsupported.extend(unknown)
        return tuple(groups), tuple(sorted(set(unsupported)))

    def _conflicting_refs(self, normative_set):
        return tuple(sorted({ref for group in self._analysis(normative_set)[0] for ref in group}))

    def evaluate(self, *, normative_set: NormativeSetV1, evaluated_at: datetime) -> NormativeCompatibilityResultV1:
        groups, unsupported = self._analysis(normative_set)
        refs = tuple(sorted({ref for group in groups for ref in group}))
        unresolved = tuple(sorted(set(normative_set.unresolved_refs) | set(unsupported)))
        if refs:
            outcome, proof = "CONFLICT", "bounded_effect_algebra"
        elif normative_set.closure_state != "COMPLETE" or unresolved:
            outcome, proof = "UNKNOWN", "incomplete_semantics"
        else:
            outcome, proof = "COMPATIBLE", "bounded_effect_algebra"
        payload = dict(
            schema_version="normative_compatibility_result.v1",
            normative_set_ref=normative_set.normative_set_ref,
            normative_set_hash=normative_set.normative_set_hash,
            outcome=outcome,
            proof_kind=proof,
            conflicting_effect_refs=refs,
            unresolved_refs=unresolved,
            evaluator_ref=self.evaluator_ref,
            evaluator_hash=self.evaluator_hash,
            analysis_scope=NormativeAnalysisScopeV2(
                coverage="INCOMPLETE" if unresolved or normative_set.closure_state != "COMPLETE" else "COMPLETE",
                checked_effect_refs=tuple(sorted({e.effect_ref for e in normative_set.effects} - set(unsupported))),
                conflict_groups=groups,
            ).model_dump(mode="json"),
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
        NormativeSetV1.model_validate(normative_set.model_dump(mode="json"))
        NormativeCompatibilityResultV1.model_validate(result.model_dump(mode="json"))
        if (result.normative_set_ref, result.normative_set_hash) != (
            normative_set.normative_set_ref, normative_set.normative_set_hash
        ):
            raise ValueError("conflict result/set binding mismatch")
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
            minimality="NOT_CHECKED",
            analysis_policy="typed_bounded_algebra.v2" if result.analysis_scope is not None else None,
            evidence_refs=evidence_refs,
        )
        if payload["analysis_policy"] is None:
            payload.pop("analysis_policy")
        digest = _digest(payload)
        return ConflictCoreV1(
            conflict_core_ref="conflict-core:" + digest,
            conflict_core_hash="sha256:" + digest,
            **payload,
        )


class ConflictResolver:
    """Apply governed precedence only after a verified conflict core exists."""

    @staticmethod
    def controllers(*, plan, members, observed_facts):
        """Shared explicit-controller seam for normative and legal consumers."""
        members = set(members)
        if len(members) < 2:
            return (), ()
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
        return tuple(candidates), checks

    def resolve(
        self,
        *,
        core: ConflictCoreV1,
        plan: CompiledRulePlanV1,
        observed_facts: dict[str, Any],
        resolved_at: datetime,
    ) -> ConflictResolutionV1:
        candidates, checks = self.controllers(plan=plan, members=core.member_rule_refs, observed_facts=observed_facts)
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
