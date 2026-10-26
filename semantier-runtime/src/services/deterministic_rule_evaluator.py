"""Backend-neutral bounded evaluator for ``semantier_deterministic_v1``."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from calendar import monthrange
from fractions import Fraction
from graphlib import TopologicalSorter
import math
from typing import TYPE_CHECKING, Any, NamedTuple

from contracts.ontology_validation import (
    CandidateRuleEvaluationV1,
    CompiledPredicateV1,
    CompiledPrecedenceV1,
    CompiledRulePlanV1,
    CompiledRuleV1,
    PredicateEvaluationV1,
    PrecedenceEvaluationV1,
    RuntimeJustificationV1,
    RuntimeSemanticEvaluationBundleV1,
)
from eos.artifact_hashing import content_hash

if TYPE_CHECKING:
    from contracts.ontology_building import ArtifactPinV1
    from contracts.runtime_semantic_evaluation import RuntimeRuleEvaluationSetV1


SUPPORTED_OPERATORS = frozenset(
    {
        "EQ",
        "IN",
        "EXISTS",
        "NOT",
        "AND",
        "OR",
        "GE",
        "GT",
        "LE",
        "LT",
        "DATE_ON_OR_AFTER",
        "DATE_ON_OR_BEFORE",
        "RATIO_LE", "RATIO_GT", "TIMESTAMP_BEFORE", "TIMESTAMP_AFTER", "CALENDAR_YEARS_AFTER",
    }
)


class _PerRuleEvaluation(NamedTuple):
    candidate_evaluation: CandidateRuleEvaluationV1
    satisfied_predicates: tuple[str, ...]
    failed_predicates: tuple[str, ...]
    exception_checks: tuple[PredicateEvaluationV1, ...]
    unknown_inputs: tuple[str, ...]
    unsupported_semantics: tuple[str, ...]
    competency_refs: tuple[str, ...]
    ir_node_refs: tuple[str, ...]
    source_evidence_refs: tuple[str, ...]
    justification: RuntimeJustificationV1


class DeterministicRuleEvaluator:
    def _evaluate_input_requirement(self, predicate, facts):
        return self._evaluate_predicate(predicate, facts, optional_absence_is_failure=False, strict_operands=True)

    @staticmethod
    def _operand_kinds_valid(predicate, observed):
        expected = predicate.expected_value

        def same_scalar_kind(value, exemplar):
            if type(exemplar) is bool:
                return type(value) is bool
            if type(exemplar) in {int, float}:
                return type(value) is int or (type(value) is float and math.isfinite(value))
            return type(value) is type(exemplar)

        operator = predicate.operator
        if operator == "NOT":
            valid = type(observed) is bool
        elif operator in {"AND", "OR"}:
            valid = isinstance(observed, (list, tuple)) and all(type(item) is bool for item in observed)
        elif operator in {"EQ", "GE", "GT", "LE", "LT"}:
            valid = same_scalar_kind(observed, expected)
        elif operator == "IN":
            valid = isinstance(expected, (list, tuple)) and any(same_scalar_kind(observed, item) for item in expected)
        elif operator in {"DATE_ON_OR_AFTER", "DATE_ON_OR_BEFORE"}:
            try:
                valid = isinstance(observed, str) and date.fromisoformat(observed).isoformat() == observed
            except ValueError:
                valid = False
        else:
            valid = True  # EXISTS establishes presence, not an inferred value type.
        return valid

    @staticmethod
    def _conjunction(results):
        if any(item.result == "FAILED" for item in results):
            return "NO_MATCH"
        if any(item.result in {"UNKNOWN", "UNSUPPORTED"} for item in results):
            return "UNKNOWN"
        return "MATCH"

    def _precedence_check(self, relation, higher_outcome, observed_facts):
        conditions = tuple(self._evaluate_predicate(
            predicate, observed_facts, optional_absence_is_failure=False)
            for predicate in relation.conditions)
        guard = self._conjunction(conditions)
        outcome = ("NO_MATCH" if "NO_MATCH" in (higher_outcome, guard) else
                   "UNKNOWN" if "UNKNOWN" in (higher_outcome, guard) else "MATCH")
        return PrecedenceEvaluationV1(
            ir_node_ref=relation.ir_node_ref, higher_priority_ref=relation.higher_priority_ref,
            lower_priority_ref=relation.lower_priority_ref, higher_outcome=higher_outcome,
            condition_results=conditions, outcome=outcome, evidence_refs=relation.evidence_refs)

    @staticmethod
    def _evaluate_predicate(
        predicate: CompiledPredicateV1,
        facts: dict[str, Any],
        *,
        optional_absence_is_failure: bool,
        strict_operands: bool = False,
    ) -> PredicateEvaluationV1:
        operator = predicate.operator
        present = predicate.fact_ref in facts
        observed = facts.get(predicate.fact_ref)
        comparison = facts.get(predicate.comparison_fact_ref)
        if operator not in SUPPORTED_OPERATORS:
            result = "UNSUPPORTED"
        elif operator == "EXISTS":
            result = "SATISFIED" if present and observed not in (None, "") else "FAILED"
        elif not present:
            result = "FAILED" if optional_absence_is_failure else "UNKNOWN"
        else:
            expected = predicate.expected_value
            try:
                if (strict_operands or predicate.strict_operands) and not DeterministicRuleEvaluator._operand_kinds_valid(predicate, observed):
                    raise ValueError("predicate operand kind mismatch")
                if operator in {"RATIO_LE", "RATIO_GT", "TIMESTAMP_BEFORE", "TIMESTAMP_AFTER"}:
                    if predicate.comparison_fact_ref is None or comparison is None:
                        raise ValueError("comparison observation missing")
                    if operator in {"RATIO_LE", "RATIO_GT"}:
                        if any(type(value) not in {int, float} or not math.isfinite(value)
                            for value in (observed, comparison, expected)) or observed < 0 or comparison <= 0 or expected < 0:
                            raise ValueError("invalid ratio operands")
                        matched = Fraction(str(observed)) <= Fraction(str(comparison)) * Fraction(str(expected))
                        if operator == "RATIO_GT":
                            matched = not matched
                    else:
                        if expected is not None or any(not isinstance(value, str) or 'T' not in value for value in (observed, comparison)):
                            raise ValueError("invalid timestamp operands")
                        left, right = (datetime.fromisoformat(value) for value in (observed, comparison))
                        if any(value.utcoffset() != timedelta(0) for value in (left, right)):
                            raise ValueError("timestamps must already be normalized UTC")
                        matched = left < right if operator == "TIMESTAMP_BEFORE" else left > right
                elif operator == "CALENDAR_YEARS_AFTER":
                    if type(expected) is not int or expected <= 0:
                        raise ValueError("invalid calendar duration")
                    if any(not isinstance(value, str) for value in (observed, comparison)):
                        raise ValueError("calendar dates required")
                    submitted, issued = date.fromisoformat(observed), date.fromisoformat(comparison)
                    if submitted.isoformat() != observed or issued.isoformat() != comparison:
                        raise ValueError("canonical calendar dates required")
                    year = issued.year + expected
                    anniversary = date(year, issued.month, min(issued.day, monthrange(year, issued.month)[1]))
                    matched = submitted > anniversary
                elif operator == "EQ":
                    matched = observed == expected
                elif operator == "IN":
                    matched = observed in expected
                elif operator == "NOT":
                    matched = not bool(observed)
                elif operator == "AND":
                    matched = all(bool(item) for item in observed)
                elif operator == "OR":
                    matched = any(bool(item) for item in observed)
                elif operator == "GE":
                    matched = observed >= expected
                elif operator == "GT":
                    matched = observed > expected
                elif operator == "LE":
                    matched = observed <= expected
                elif operator == "LT":
                    matched = observed < expected
                elif operator == "DATE_ON_OR_AFTER":
                    matched = date.fromisoformat(str(observed)) >= date.fromisoformat(
                        str(expected)
                    )
                else:
                    matched = date.fromisoformat(str(observed)) <= date.fromisoformat(
                        str(expected)
                    )
                result = "SATISFIED" if matched else "FAILED"
            except (TypeError, ValueError, OverflowError):
                result = "UNKNOWN"
        return PredicateEvaluationV1(
            predicate_id=predicate.predicate_id,
            operator=operator,
            fact_ref=predicate.fact_ref,
            result=result,
            observed_value=observed,
            expected_value=predicate.expected_value,
            ir_node_ref=predicate.ir_node_ref,
            evidence_refs=predicate.evidence_refs,
            comparison_fact_ref=predicate.comparison_fact_ref,
            comparison_observed_value=comparison,
        )

    def _evaluate_rule(
        self,
        rule: CompiledRuleV1,
        observed_facts: dict[str, Any],
        precedences: tuple[CompiledPrecedenceV1, ...] = (),
    ) -> _PerRuleEvaluation:
        input_results = tuple(self._evaluate_input_requirement(predicate, observed_facts)
            for predicate in (rule.input_requirements or ()))
        predicate_results = tuple(
            self._evaluate_predicate(
                predicate, observed_facts, optional_absence_is_failure=False
            )
            for predicate in rule.predicates
        )
        exception_results = tuple(
            self._evaluate_predicate(
                predicate, observed_facts, optional_absence_is_failure=rule.exception_groups is None
            )
            for predicate in rule.exception_predicates
        )
        precedence_checks = []
        if rule.exception_groups is None:
            exception_applies = bool(exception_results) and all(
                item.result == "SATISFIED" for item in exception_results)
            material_exception_results = ()
            unsupported_exception_results = exception_results
        else:
            by_id = {item.predicate_id: item for item in exception_results}
            groups = []
            for group in rule.exception_groups:
                body = tuple(by_id[ref] for ref in group.predicate_ids)
                relations = tuple(item for item in precedences if item.higher_priority_ref == group.ir_node_ref)
                if not relations:
                    groups.append(body)
                for relation in relations:
                    check = self._precedence_check(relation, self._conjunction(body), observed_facts)
                    precedence_checks.append(check)
                    groups.append(body + check.condition_results)
            exception_applies = any(all(item.result == "SATISFIED" for item in group) for group in groups)
            # A failed conjunct defeats its exception; a satisfied alternative
            # settles exemption even if another alternative has unknown facts.
            material_exception_results = () if exception_applies else tuple(
                item for group in groups if not any(item.result == "FAILED" for item in group)
                for item in group if item.result in {"UNKNOWN", "UNSUPPORTED"})
            unsupported_exception_results = material_exception_results
        unsupported = tuple(
            sorted(
                {
                    item.operator
                    for item in (*predicate_results, *unsupported_exception_results)
                    if item.result == "UNSUPPORTED"
                }
            )
        )
        unknown_inputs = tuple(
            sorted(
                {
                    ref
                    for item in (*predicate_results, *material_exception_results)
                    if item.result == "UNKNOWN"
                    for ref in (item.fact_ref, item.comparison_fact_ref) if ref is not None
                }
            )
        )
        satisfied = tuple(
            item.predicate_id
            for item in predicate_results
            if item.result == "SATISFIED"
        )
        failed = tuple(
            item.predicate_id for item in predicate_results if item.result == "FAILED"
        )
        if any(item.result != "SATISFIED" for item in input_results):
            result, reason = "UNKNOWN", "QUALIFIED_INPUT_NOT_ESTABLISHED"
            unknown_inputs = tuple(sorted(set((*unknown_inputs,
                *(ref for item in input_results if item.result in {"FAILED", "UNKNOWN"}
                    for ref in (item.fact_ref,item.comparison_fact_ref) if ref is not None)))))
            unsupported = tuple(sorted(set((*unsupported,
                *(item.operator for item in input_results if item.result == "UNSUPPORTED")))))
        elif rule.exception_groups is not None and exception_applies:
            result, reason = "NO_MATCH", "EXPLICIT_EXCEPTION_APPLIES"
            unknown_inputs, unsupported = (), ()
        elif unsupported:
            result, reason = "UNKNOWN", "UNSUPPORTED_SEMANTICS"
        elif unknown_inputs:
            result, reason = "UNKNOWN", "MISSING_MATERIAL_INPUT"
        elif failed:
            result, reason = "NO_MATCH", "RULE_CONDITIONS_NOT_SATISFIED"
        elif exception_applies:
            result, reason = "NO_MATCH", "EXPLICIT_EXCEPTION_APPLIES"
        else:
            result, reason = "MATCH", "RULE_CONDITIONS_SATISFIED"

        rejection_reasons = (
            (reason,)
            if result == "NO_MATCH" and (reason != "EXPLICIT_EXCEPTION_APPLIES" or not exception_results)
            else ()
        )
        candidate_evaluation = CandidateRuleEvaluationV1(
            rule_id=rule.rule_id,
            outcome=result,
            predicate_results=predicate_results,
            exception_results=exception_results,
            rejection_reasons=rejection_reasons,
            input_requirement_results=input_results if rule.input_requirements is not None else None,
            precedence_results=tuple(precedence_checks) or None,
        )
        competency_refs = rule.competency_refs
        ir_node_refs = tuple(
            dict.fromkeys(
                [
                    rule.ir_node_ref,
                    *(item.ir_node_ref for item in input_results),
                    *(item.ir_node_ref for item in predicate_results),
                    *(item.ir_node_ref for item in exception_results),
                    *(group.ir_node_ref for group in (rule.exception_groups or ())),
                    *(check.ir_node_ref for check in precedence_checks),
                    *(item.ir_node_ref for check in precedence_checks for item in check.condition_results),
                ]
            )
        )
        source_evidence_refs = tuple(
            dict.fromkeys(
                [
                    *rule.evidence_refs,
                    *(ref for item in input_results for ref in item.evidence_refs),
                    *(ref for item in predicate_results for ref in item.evidence_refs),
                    *(ref for item in exception_results for ref in item.evidence_refs),
                    *(ref for group in (rule.exception_groups or ()) for ref in group.evidence_refs),
                    *(ref for check in precedence_checks for ref in check.evidence_refs),
                    *(ref for check in precedence_checks for item in check.condition_results for ref in item.evidence_refs),
                ]
            )
        )
        justification = RuntimeJustificationV1(
            result=result,
            reason_code=reason,
            rule_refs=(rule.rule_id,),
            ir_node_refs=ir_node_refs,
            competency_refs=competency_refs,
            source_evidence_refs=source_evidence_refs,
        )
        return _PerRuleEvaluation(
            candidate_evaluation=candidate_evaluation,
            satisfied_predicates=satisfied,
            failed_predicates=failed,
            exception_checks=exception_results,
            unknown_inputs=unknown_inputs,
            unsupported_semantics=unsupported,
            competency_refs=competency_refs,
            ir_node_refs=ir_node_refs,
            source_evidence_refs=source_evidence_refs,
            justification=justification,
        )

    def evaluate_rules_without_precedence(self, plan, observed_facts):
        """Evaluate intrinsic rule applicability without cross-rule priority suppression."""
        return tuple(
            self._evaluate_rule(rule, observed_facts).candidate_evaluation
            for rule in plan.rules
        )

    def evaluate_rule_precedences(self, plan, observed_facts):
        """Evaluate rule-to-rule precedence evidence without suppressing either rule."""
        intrinsic = {
            item.rule_id: item.outcome
            for item in self.evaluate_rules_without_precedence(plan, observed_facts)
        }
        return tuple(
            self._precedence_check(
                relation, intrinsic.get(relation.higher_priority_ref, "UNKNOWN"), observed_facts
            )
            for relation in (plan.precedences or ())
            if relation.higher_priority_kind == "rule"
        )

    def _evaluate_plan(self, plan, observed_facts):
        if not plan.precedences:
            return tuple(self._evaluate_rule(rule, observed_facts) for rule in plan.rules)
        rules = {rule.rule_id: rule for rule in plan.rules}
        dependencies = {ref: set() for ref in rules}
        for relation in plan.precedences:
            if relation.higher_priority_kind == "rule":
                dependencies[relation.lower_priority_ref].add(relation.higher_priority_ref)
        evaluated = {}
        for ref in TopologicalSorter(dependencies).static_order():
            relations = tuple(item for item in plan.precedences if item.lower_priority_ref == ref)
            value = self._evaluate_rule(rules[ref], observed_facts, tuple(
                item for item in relations if item.higher_priority_kind == "exception"))
            checks = tuple(self._precedence_check(
                item, evaluated[item.higher_priority_ref].candidate_evaluation.outcome, observed_facts)
                for item in relations if item.higher_priority_kind == "rule")
            if not checks:
                evaluated[ref] = value
                continue
            result = value.candidate_evaluation.outcome
            reason = value.justification.reason_code
            material = ()
            decisive_override = False
            if result != "NO_MATCH":
                if any(check.outcome == "MATCH" for check in checks):
                    result, reason = "NO_MATCH", "HIGHER_PRIORITY_RULE_APPLIES"
                    decisive_override = True
                elif any(check.outcome == "UNKNOWN" for check in checks):
                    result, reason = "UNKNOWN", "PRECEDENCE_INPUT_UNRESOLVED"
                    material = tuple(check for check in checks if check.outcome == "UNKNOWN")
            higher_values = tuple(evaluated[check.higher_priority_ref] for check in checks)
            ir_refs = tuple(dict.fromkeys((*value.ir_node_refs,
                *(check.ir_node_ref for check in checks),
                *(item.ir_node_ref for check in checks for item in check.condition_results),
                *(node for higher in higher_values for node in higher.ir_node_refs))))
            evidence = tuple(dict.fromkeys((*value.source_evidence_refs,
                *(item for check in checks for item in check.evidence_refs),
                *(item for check in checks for predicate in check.condition_results for item in predicate.evidence_refs),
                *(item for higher in higher_values for item in higher.source_evidence_refs))))
            unknown = tuple(sorted(set((*(() if decisive_override else value.unknown_inputs),
                *(ref for check in material for item in check.condition_results if item.result == "UNKNOWN"
                    for ref in (item.fact_ref,item.comparison_fact_ref) if ref is not None),
                *(item for check in material for item in evaluated[check.higher_priority_ref].unknown_inputs)))))
            unsupported = tuple(sorted(set((*(() if decisive_override else value.unsupported_semantics),
                *(item.operator for check in material for item in check.condition_results if item.result == "UNSUPPORTED"),
                *(item for check in material for item in evaluated[check.higher_priority_ref].unsupported_semantics)))))
            competency_refs = tuple(dict.fromkeys((*value.competency_refs,
                *(item for higher in higher_values for item in higher.competency_refs))))
            candidate = value.candidate_evaluation.model_copy(update=dict(
                outcome=result, precedence_results=(*(value.candidate_evaluation.precedence_results or ()), *checks),
                rejection_reasons=(reason,) if result == "NO_MATCH" else ()))
            justification = RuntimeJustificationV1(result=result, reason_code=reason,
                rule_refs=(ref,),
                ir_node_refs=ir_refs, competency_refs=competency_refs, source_evidence_refs=evidence)
            evaluated[ref] = value._replace(candidate_evaluation=candidate, justification=justification,
                ir_node_refs=ir_refs, source_evidence_refs=evidence, competency_refs=competency_refs,
                unknown_inputs=unknown, unsupported_semantics=unsupported)
        return tuple(evaluated[rule.rule_id] for rule in plan.rules)

    def evaluate(
        self,
        *,
        plan: CompiledRulePlanV1,
        execution_epoch_ref: str,
        observed_facts: dict[str, Any],
        target_evidence_refs: tuple[str, ...],
        evaluated_at: datetime,
        include_all_rules: bool = False,
    ) -> RuntimeSemanticEvaluationBundleV1:
        rule_results = self._evaluate_plan(plan, observed_facts)
        rule_evaluation = rule_results[0]
        # The legacy scalar result still concerns the first rule. For precedence
        # plans retain all per-rule decisions that can explain its dependencies.
        candidate_evaluations = tuple(item.candidate_evaluation for item in rule_results) if include_all_rules or plan.precedences else (rule_evaluation.candidate_evaluation,)
        candidate_evaluation = rule_evaluation.candidate_evaluation
        satisfied = rule_evaluation.satisfied_predicates
        failed = rule_evaluation.failed_predicates
        exception_results = rule_evaluation.exception_checks
        unknown_inputs = rule_evaluation.unknown_inputs
        unsupported = rule_evaluation.unsupported_semantics
        competency_refs = rule_evaluation.competency_refs
        ir_node_refs = rule_evaluation.ir_node_refs
        source_evidence_refs = rule_evaluation.source_evidence_refs
        justification = rule_evaluation.justification
        result = candidate_evaluation.outcome
        basis = {
            "execution_epoch_ref": execution_epoch_ref,
            "plan_ref": plan.plan_ref,
            "plan_hash": plan.plan_hash,
            "result": result,
            "observed_facts": observed_facts,
            "target_evidence_refs": target_evidence_refs,
            "candidate_rule_evaluations": [
                item.model_dump(mode="json") for item in candidate_evaluations
            ],
            "justification": justification.model_dump(mode="json"),
        }
        complete = {"all_rule_justifications": tuple(item.justification for item in rule_results)} if include_all_rules else {}
        if complete:
            basis["all_rule_justifications"] = [item.model_dump(mode="json") for item in complete["all_rule_justifications"]]
        # Fresh evaluations must not collide with an earlier timestamped receipt.
        digest = content_hash(dict(basis, evaluated_at=evaluated_at.isoformat()))
        return RuntimeSemanticEvaluationBundleV1(
            bundle_ref=f"runtime-semantic-evaluation:{digest}",
            bundle_hash=f"sha256:{digest}",
            organization_id=plan.organization_id,
            workspace_id=plan.workspace_id,
            execution_epoch_ref=execution_epoch_ref,
            plan_ref=plan.plan_ref,
            plan_hash=plan.plan_hash,
            result=result,
            observed_facts=observed_facts,
            target_evidence_refs=target_evidence_refs,
            candidate_rule_evaluations=candidate_evaluations,
            satisfied_predicates=satisfied,
            failed_predicates=failed,
            exception_checks=exception_results,
            unknown_inputs=unknown_inputs,
            unsupported_semantics=unsupported,
            competency_refs=competency_refs,
            ir_node_refs=ir_node_refs,
            source_evidence_refs=source_evidence_refs,
            justification=justification,
            evaluated_at=evaluated_at,
            **complete,
        )

    def evaluate_scoped_disposition(
        self, *, candidate, plan, rule_id, observed_facts, context_pin,
        execution_epoch_ref, target_evidence_refs, evaluated_at,
    ):
        """Compute a proposal, never authority, from one exact source rule.

        Publication additionally requires the Tender verified binding/record owner.
        Legacy MATCH semantics and the evaluator's first-rule scalar stay unchanged.
        """
        from services.ontology_rule_plan_compiler import OntologyRulePlanCompiler

        rules = [rule for rule in plan.rules if rule.rule_id == rule_id]
        frames = [frame for frame in candidate.frames
                  if frame.frame_kind == "rule" and frame.frame_id == rule_id]
        if len(rules) != 1 or len(frames) != 1:
            raise ValueError("EXACT_NORMATIVE_RULE_ID_REQUIRED")
        if (plan.organization_id, plan.workspace_id, plan.candidate_set_ref, plan.candidate_set_hash) != (
            candidate.organization_id, candidate.workspace_id,
            candidate.candidate_set_ref, candidate.candidate_set_hash,
        ):
            raise ValueError("NORMATIVE_PLAN_BINDING_MISMATCH")
        recompiled = OntologyRulePlanCompiler(
            compiler_ref=plan.compiler_ref, compiler_hash=plan.compiler_hash,
        ).compile(candidate=candidate, validation_contract_ref=plan.validation_contract_ref,
                  validation_contract_hash=plan.validation_contract_hash,
                  competency_refs=rules[0].competency_refs, compiled_at=plan.compiled_at, domain_basis=plan.domain_basis)
        if recompiled != plan:
            raise ValueError("NORMATIVE_PLAN_CONTENT_MISMATCH")
        bundle = self.evaluate(
            plan=plan, observed_facts=observed_facts, execution_epoch_ref=execution_epoch_ref,
            target_evidence_refs=target_evidence_refs, evaluated_at=evaluated_at,
            include_all_rules=True,
        )
        selected = next(item for item in bundle.candidate_rule_evaluations if item.rule_id == rule_id)
        justification = next(item for item in bundle.all_rule_justifications if rule_id in item.rule_refs)
        rule, frame = rules[0], frames[0]
        applicability_refs = set((*frame.scope_refs, *frame.temporal_condition_refs))
        for distinction in candidate.frames:
            if distinction.frame_kind == "entity_distinction" and any(
                effect.rule_ref == rule_id and effect.effect == "applicability"
                for effect in (distinction.rule_effects or ())
            ):
                applicability_refs.add(distinction.frame_id)
        applicability = tuple(item for item in selected.predicate_results
                              if item.ir_node_ref in applicability_refs)
        effects = tuple(self._evaluate_predicate(
            effect, observed_facts, optional_absence_is_failure=False,
        ) for effect in (rule.disposition_effects or ()))
        outcome, reason = "UNKNOWN", "SOURCE_DISPOSITION_POLICY_UNRESOLVED"
        if frame.interpretation_required:
            reason = "QUALIFIED_INTERPRETATION_REQUIRED"
        elif any(item.result != "SATISFIED" for item in (selected.input_requirement_results or ())):
            reason = "QUALIFIED_INPUT_NOT_ESTABLISHED"
        elif any(item.result == "FAILED" for item in applicability):
            outcome, reason = "NOT_APPLICABLE", "RULE_SCOPE_NOT_SATISFIED"
        elif any(item.result in {"UNKNOWN", "UNSUPPORTED"} for item in applicability):
            reason = "APPLICABILITY_UNRESOLVED"
        elif selected.outcome == "UNKNOWN":
            reason = justification.reason_code
        elif selected.outcome == "NO_MATCH":
            outcome, reason = "NO_VIOLATION_IN_SCOPE", justification.reason_code
        elif effects and not any(item.result in {"UNKNOWN", "UNSUPPORTED"} for item in effects):
            satisfied = all(item.result == "SATISFIED" for item in effects)
            breached = ((frame.modality == "prohibition" and satisfied)
                        or (frame.modality == "obligation" and not satisfied))
            outcome = "VIOLATION" if breached else "NO_VIOLATION_IN_SCOPE"
            reason = "SOURCE_DEFINED_EFFECT_BREACHED" if breached else "NO_SOURCE_DEFINED_BREACH"
        elif effects:
            reason = "MISSING_EFFECT_FACT"
        # Conflicts are formed intrinsically before precedence can suppress a rule.
        from services.normative_conflict_analysis import (
            NormativeSetCompiler, NormativeCompatibilityEvaluator, ConflictResolver,
        )
        normative_set = NormativeSetCompiler().compile(
            candidate=candidate, plan=plan, evaluation=bundle,
            context_ref=context_pin.artifact_ref, context_hash=context_pin.artifact_hash, formed_at=evaluated_at,
        )
        compatibility = None
        resolution = None
        if len(normative_set.member_rule_refs) > 1 or normative_set.unknown_rule_refs:
            evaluator = NormativeCompatibilityEvaluator(
                evaluator_ref="source-rule-effect-compatibility:v2",
                evaluator_hash="sha256:" + content_hash({"policy": "source_rule_effect_disposition.v2"}),
            )
            compatibility = evaluator.evaluate(normative_set=normative_set, evaluated_at=evaluated_at)
            if compatibility.outcome == "CONFLICT":
                core = evaluator.extract_core(normative_set=normative_set, result=compatibility)
                resolution = ConflictResolver().resolve(
                    core=core, plan=plan, observed_facts=observed_facts, resolved_at=evaluated_at,
                )
                if outcome != "NOT_APPLICABLE":
                    if resolution.outcome == "UNRESOLVED" and rule_id in core.member_rule_refs:
                        outcome, reason = "CONFLICT", "GOVERNED_PRECEDENCE_UNRESOLVED"
                    else:
                        # One extracted core does not prove that all conflicts
                        # affecting this selected rule have been resolved.
                        outcome, reason = "UNKNOWN", "SCOPED_CONFLICT_CLOSURE_UNRESOLVED"
            elif compatibility.outcome != "COMPATIBLE" and outcome != "NOT_APPLICABLE":
                outcome, reason = "UNKNOWN", "NORMATIVE_DEPENDENCY_CLOSURE_UNRESOLVED"
        return dict(
            policy_version="source_rule_effect_disposition.v2", rule_id=rule_id,
            qualification="REQUIRES_VERIFIED_BINDING", proposed_outcome=outcome,
            reason_code=reason, evaluation=bundle, effect_results=effects,
            applicability_results=applicability, normative_set=normative_set,
            compatibility=compatibility, conflict_resolution=resolution,
        )

    def evaluate_rules(
        self,
        *,
        plan: CompiledRulePlanV1,
        execution_epoch: ArtifactPinV1,
        observed_facts: dict[str, Any],
        observed_facts_pin: ArtifactPinV1,
        target_evidence_refs: tuple[str, ...],
        evaluated_at: datetime,
    ) -> RuntimeRuleEvaluationSetV1:
        from contracts.ontology_building import ArtifactPinV1
        from contracts.runtime_semantic_evaluation import RuntimeRuleEvaluationSetV1

        rule_results = self._evaluate_plan(plan, observed_facts)
        evaluations = tuple(item.candidate_evaluation for item in rule_results)
        justifications = tuple(item.justification for item in rule_results)
        counts = {
            outcome: sum(evaluation.outcome == outcome for evaluation in evaluations)
            for outcome in ("MATCH", "NO_MATCH", "UNKNOWN")
        }
        return RuntimeRuleEvaluationSetV1.create(
            schema_version="runtime_rule_evaluation_set.v1",
            organization_id=plan.organization_id,
            workspace_id=plan.workspace_id,
            plan=ArtifactPinV1(
                artifact_kind="compiled_rule_plan",
                artifact_ref=plan.plan_ref,
                artifact_hash=plan.plan_hash,
            ),
            execution_epoch=execution_epoch,
            observed_facts=observed_facts_pin,
            target_evidence_refs=target_evidence_refs,
            rule_evaluations=evaluations,
            justifications=justifications,
            counts=counts,
            evaluated_at=evaluated_at,
        )


__all__ = ["DeterministicRuleEvaluator", "SUPPORTED_OPERATORS"]
