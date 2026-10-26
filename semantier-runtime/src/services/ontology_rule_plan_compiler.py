"""Deterministic projection of normative IR into the bounded runtime plan."""

from __future__ import annotations

from datetime import date, datetime

from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.ontology_validation import (
    CompiledExceptionGroupV1,
    CompiledPredicateV1,
    CompiledPrecedenceV1,
    CompiledRulePlanV1,
    CompiledRuleV1,
)
from eos.artifact_hashing import content_hash


class OntologyRulePlanCompiler:
    def __init__(self, *, compiler_ref: str, compiler_hash: str) -> None:
        self._compiler_ref = compiler_ref
        self._compiler_hash = compiler_hash

    @staticmethod
    def _predicate(frame_id: str, predicate, *, strict_operands=None) -> CompiledPredicateV1:
        return CompiledPredicateV1(
            predicate_id=predicate.predicate_id,
            operator=predicate.operator,
            fact_ref=predicate.fact_ref,
            expected_value=predicate.expected_value,
            ir_node_ref=frame_id,
            evidence_refs=predicate.evidence_refs,
            strict_operands=strict_operands,
            comparison_fact_ref=predicate.comparison_fact_ref,
        )

    def compile(
        self,
        *,
        candidate: NormativeSemanticCandidateSetV1,
        validation_contract_ref: str,
        validation_contract_hash: str,
        competency_refs: tuple[str, ...],
        compiled_at: datetime,
        teaching_context=None,
        domain_basis=None,
    ) -> CompiledRulePlanV1:
        if domain_basis is not None:
            domain_basis.verify_candidate(candidate)
        if candidate.teaching_candidate:
            from services.teaching_candidate_scope import require_teaching_scope
            require_teaching_scope(teaching_context, organization_id=candidate.organization_id, workspace_id=candidate.workspace_id)
        if candidate.completeness_state != "COMPLETE":
            raise ValueError("runtime plans require complete normative semantics")
        exceptions = {
            frame.frame_id: frame
            for frame in candidate.frames
            if frame.frame_kind == "exception"
        }
        temporal_frames = {frame.frame_id: frame for frame in candidate.frames
            if frame.frame_kind in {"temporal_condition", "relative_temporal_condition"}}
        scopes = {frame.frame_id: frame for frame in candidate.frames if frame.frame_kind == "scope"}
        qualifications = {frame.frame_id: frame for frame in candidate.frames if frame.frame_kind == "qualification"}
        rule_ids = {frame.frame_id for frame in candidate.frames if frame.frame_kind == "rule"}
        entity_effects = {ref: [] for ref in rule_ids}
        for distinction in candidate.frames:
            if distinction.frame_kind != "entity_distinction":
                continue
            if distinction.epistemic_status != "FOUND" or not distinction.rule_effects:
                raise ValueError(f"runtime plan entity distinction has no executable effects: {distinction.frame_id}")
            for effect in distinction.rule_effects:
                if effect.rule_ref not in rule_ids:
                    raise ValueError(f"runtime plan entity distinction rule binding mismatch: {effect.rule_ref}")
                entity_effects[effect.rule_ref].append((distinction, effect))
        rules = []
        for frame in candidate.frames:
            if frame.frame_kind != "rule":
                continue
            input_requirements = []
            for ref in frame.qualification_refs:
                qualification = qualifications.get(ref)
                if qualification is None or qualification.epistemic_status != "FOUND" or not qualification.conditions:
                    raise ValueError(f"runtime plan qualification binding is not executable: {frame.frame_id} -> {ref}")
                input_requirements.extend(self._predicate(ref, predicate) for predicate in qualification.conditions)
            entity_predicates = []
            for distinction, effect in entity_effects[frame.frame_id]:
                target = input_requirements if effect.effect == "input_requirement" else entity_predicates
                target.extend(self._predicate(distinction.frame_id, predicate, strict_operands=True)
                    for predicate in effect.conditions)
            scope_predicates = []
            for ref in frame.scope_refs:
                scope = scopes.get(ref)
                if (scope is None or scope.applies_to_rule_ref != frame.frame_id
                    or scope.epistemic_status != "FOUND" or not scope.conditions):
                    raise ValueError(f"runtime plan scope binding is not executable: {frame.frame_id} -> {ref}")
                scope_predicates.extend(self._predicate(ref, predicate) for predicate in scope.conditions)
            temporal_predicates = []
            for ref in frame.temporal_condition_refs:
                temporal = temporal_frames.get(ref)
                if temporal is None or temporal.epistemic_status != "FOUND":
                    raise ValueError(f"runtime plan temporal binding mismatch: {frame.frame_id} -> {ref}")
                if temporal.frame_kind == "relative_temporal_condition":
                    temporal_predicates.extend(self._predicate(ref, predicate, strict_operands=True)
                        for predicate in temporal.conditions)
                    continue
                try:
                    start = date.fromisoformat(temporal.effective_from)
                    end = date.fromisoformat(temporal.effective_until)
                    if start.isoformat() != temporal.effective_from or end.isoformat() != temporal.effective_until or start > end:
                        raise ValueError("invalid date interval")
                except ValueError as exc:
                    raise ValueError(f"runtime plan temporal interval invalid: {ref}") from exc
                for operator, value in (("DATE_ON_OR_AFTER", temporal.effective_from), ("DATE_ON_OR_BEFORE", temporal.effective_until)):
                    temporal_predicates.append(CompiledPredicateV1(
                        predicate_id="predicate:temporal:" + content_hash(dict(rule=frame.frame_id, frame=ref, operator=operator)),
                        operator=operator, fact_ref=temporal.event_time_ref, expected_value=value,
                        ir_node_ref=ref, evidence_refs=temporal.evidence_refs))
            for ref in frame.exception_refs:
                exception = exceptions.get(ref)
                if exception is None or exception.applies_to_rule_ref != frame.frame_id:
                    raise ValueError(
                        f"runtime plan exception binding mismatch: {frame.frame_id} -> {ref}"
                    )
            exception_predicates = tuple(
                self._predicate(exception.frame_id, predicate)
                for ref in frame.exception_refs
                for exception in (exceptions[ref],)
                for predicate in exception.conditions
            )
            rules.append(
                CompiledRuleV1(
                    rule_id=frame.frame_id,
                    source_modality=frame.modality if frame.rule_effects else None,
                    disposition_effects=tuple(CompiledPredicateV1(
                        predicate_id=effect.effect_id, operator=effect.operator,
                        fact_ref=effect.fact_ref, expected_value=effect.expected_value,
                        ir_node_ref=frame.frame_id, evidence_refs=effect.evidence_refs,
                        strict_operands=True,
                    ) for effect in frame.rule_effects) if frame.rule_effects else None,
                    interpretation_required=frame.interpretation_required,
                    ir_node_ref=frame.frame_id,
                    competency_refs=competency_refs,
                    evidence_refs=frame.evidence_refs,
                    predicates=tuple(
                        self._predicate(frame.frame_id, predicate)
                        for predicate in frame.conditions
                    ) + tuple(scope_predicates) + tuple(temporal_predicates) + tuple(entity_predicates),
                    exception_predicates=exception_predicates,
                    input_requirements=tuple(input_requirements) or None,
                    exception_groups=tuple(CompiledExceptionGroupV1(
                        ir_node_ref=exceptions[ref].frame_id,
                        predicate_ids=tuple(item.predicate_id for item in exceptions[ref].conditions),
                        evidence_refs=exceptions[ref].evidence_refs,
                    ) for ref in frame.exception_refs) if frame.exception_refs else None,
                )
            )
        if not rules:
            raise ValueError("runtime plan requires at least one normative rule")
        frames = {frame.frame_id: frame for frame in candidate.frames}
        precedence_frames = {ref: frame for ref, frame in frames.items() if frame.frame_kind == "precedence"}
        predicates = {}
        for frame in candidate.frames:
            for predicate in getattr(frame, "conditions", ()) or ():
                if predicate.predicate_id in predicates:
                    raise ValueError("precedence predicate identity must be unique")
                predicates[predicate.predicate_id] = (frame, predicate)
        for frame in candidate.frames:
            if frame.frame_kind != "rule":
                continue
            for ref in frame.precedence_refs:
                relation = precedence_frames.get(ref)
                if relation is None or frame.frame_id not in (relation.higher_priority_ref, relation.lower_priority_ref):
                    raise ValueError("runtime plan precedence reference is not bound to this rule")
        precedences = []
        for ref, frame in precedence_frames.items():
            higher = frames.get(frame.higher_priority_ref)
            lower = frames.get(frame.lower_priority_ref)
            if (frame.epistemic_status != "FOUND" or higher is None or lower is None
                or higher.frame_kind not in {"rule", "exception"} or lower.frame_kind != "rule"
                or higher.epistemic_status != "FOUND" or lower.epistemic_status != "FOUND"):
                raise ValueError(f"runtime plan precedence endpoints are not executable: {ref}")
            conditions = []
            for predicate_ref in frame.condition_refs:
                owner_predicate = predicates.get(predicate_ref)
                if owner_predicate is None or owner_predicate[0].epistemic_status != "FOUND":
                    raise ValueError(f"runtime plan precedence condition is not executable: {predicate_ref}")
                owner, predicate = owner_predicate
                conditions.append(self._predicate(owner.frame_id, predicate,
                    strict_operands=True if owner.frame_kind == "entity_distinction" else None))
            precedences.append(CompiledPrecedenceV1(
                ir_node_ref=ref, higher_priority_ref=frame.higher_priority_ref,
                lower_priority_ref=frame.lower_priority_ref, higher_priority_kind=higher.frame_kind,
                conditions=tuple(conditions), evidence_refs=frame.evidence_refs))
        basis = {
            "candidate_set_ref": candidate.candidate_set_ref,
            "candidate_set_hash": candidate.candidate_set_hash,
            "validation_contract_ref": validation_contract_ref,
            "validation_contract_hash": validation_contract_hash,
            "compiler_ref": self._compiler_ref,
            "compiler_hash": self._compiler_hash,
            "semantic_profile": "semantier_deterministic_v1",
            "rules": [rule.model_dump(mode="json") for rule in rules],
        }
        if domain_basis is not None:
            basis["domain_basis"] = domain_basis.model_dump(mode="json")
        if candidate.teaching_candidate:
            basis["teaching_candidate"] = True
        if precedences:
            basis["precedences"] = [item.model_dump(mode="json") for item in precedences]
        # Production time is persisted payload and must distinguish fresh generations.
        digest = content_hash(dict(basis, compiled_at=compiled_at.isoformat()))
        return CompiledRulePlanV1(
            plan_ref=f"compiled-rule-plan:{digest}",
            plan_hash=f"sha256:{digest}",
            organization_id=candidate.organization_id,
            workspace_id=candidate.workspace_id,
            compiled_at=compiled_at,
            **basis,
        )


__all__ = ["OntologyRulePlanCompiler"]
