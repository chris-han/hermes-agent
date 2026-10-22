"""Server-owned bounded comparison for ChangeCandidate assessment.

This service compares exact semantic DecisionRecords under the frozen evaluation
basis and persists one comparison DecisionRecord. It creates no learning store,
episode object, or paired-comparison semantic authority.
"""

from __future__ import annotations

from storage.hosted_postgres import sql_query,begin_write,in_transaction

from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import (
    ChangeCandidate,
    ContextGraph,
    ContextGraphOrigin,
    DecisionRecord,
    ReasonerInvocation,
    ReasoningTrace,
)
from storage.ontology_build_store import OntologyBuildStore


class ChangeCandidateComparisonError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _key(pin: ArtifactPinV1) -> tuple[str, str, str]:
    return pin.artifact_kind, pin.artifact_ref, pin.artifact_hash


class ChangeCandidateComparisonService:
    """Evaluate one matched case plus one independent holdout from frozen truth."""

    def __init__(self, *, store: OntologyBuildStore, context_evaluator=None, actor_context=None) -> None:
        if not isinstance(store, OntologyBuildStore):
            raise TypeError("ChangeCandidate comparison requires the shared kernel store")
        self._store = store
        self._context_evaluator = context_evaluator
        self._actor_context = actor_context

    def _load(self, *, scope: tuple[str, str], pin: ArtifactPinV1, model):
        if pin.artifact_kind != model.ARTIFACT_KIND:
            raise ChangeCandidateComparisonError("INVALID_PIN_KIND")
        value = self._store.get(
            organization_id=scope[0],
            workspace_id=scope[1],
            artifact_ref=pin.artifact_ref,
            contract_type=model,
        )
        if value is None:
            raise ChangeCandidateComparisonError("OBJECT_NOT_FOUND")
        if value.pin() != pin:
            raise ChangeCandidateComparisonError("HASH_MISMATCH")
        return value

    def compare(
        self,
        *,
        proposal_pin: ArtifactPinV1,
        matched_baseline_decision_pin: ArtifactPinV1,
        matched_successor_decision_pin: ArtifactPinV1,
        holdout_baseline_decision_pin: ArtifactPinV1,
        holdout_successor_decision_pin: ArtifactPinV1,
        successor_input_pin: ArtifactPinV1,
        actor_id: str,
        persist: bool = True,
        retained_evaluations: tuple[DecisionRecord, ...] | None = None,
    ) -> DecisionRecord:
        # Recover scope from persisted row metadata, never from caller-provided
        # semantic fields.
        row = self._store._connection.execute(
            sql_query("SELECT organization_id,workspace_id,artifact_hash FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT organization_id,workspace_id,artifact_hash FROM ontology_build_artifacts WHERE artifact_ref=%s'),
            (proposal_pin.artifact_ref,),
        ).fetchone()
        if row is None:
            raise ChangeCandidateComparisonError("OBJECT_NOT_FOUND")
        scope = (str(row[0]), str(row[1]))
        if row[2] != proposal_pin.artifact_hash:
            raise ChangeCandidateComparisonError("HASH_MISMATCH")
        proposal = self._load(scope=scope, pin=proposal_pin, model=ChangeCandidate)
        if proposal.stage != "PROPOSAL":
            raise ChangeCandidateComparisonError("PROPOSAL_REQUIRED")
        expectation = proposal.evaluation_basis.comparison_expectation
        if expectation is None:
            raise ChangeCandidateComparisonError("COMPARISON_EXPECTATION_REQUIRED")
        if not isinstance(proposal.origin, ContextGraphOrigin):
            raise ChangeCandidateComparisonError("CONTEXT_ORIGIN_REQUIRED")
        context = self._load(scope=scope, pin=proposal.origin.context, model=ContextGraph)
        decisions = retained_evaluations or tuple(
            self._load(scope=scope, pin=pin, model=DecisionRecord)
            for pin in (
                matched_baseline_decision_pin,
                matched_successor_decision_pin,
                holdout_baseline_decision_pin,
                holdout_successor_decision_pin,
            )
        )
        if retained_evaluations is not None and (persist or len(decisions) != 4 or tuple(item.pin() for item in decisions) != (
                matched_baseline_decision_pin, matched_successor_decision_pin,
                holdout_baseline_decision_pin, holdout_successor_decision_pin)
                or proposal.evaluation_basis.obligation_suite.artifact_kind != "runtime_context_reference_panel"):
            raise ChangeCandidateComparisonError("INVALID_RETAINED_EVALUATIONS")
        if any((item.organization_id, item.workspace_id) != scope for item in decisions):
            raise ChangeCandidateComparisonError("FORBIDDEN_SCOPE")
        panel = None
        if proposal.evaluation_basis.obligation_suite.artifact_kind == "runtime_context_reference_panel":
            if (expectation.matched_baseline_result, expectation.matched_successor_result) != ("NO_MATCH", "MATCH"):
                raise ChangeCandidateComparisonError("REFERENCE_CORRECTNESS_EXPECTATION_REQUIRED")
            if self._context_evaluator is None or self._actor_context is None:
                raise ChangeCandidateComparisonError("CONTEXT_EVALUATOR_REQUIRED")
            if self._context_evaluator._store._build_store is not self._store:
                raise ChangeCandidateComparisonError("INVALID_CONSUMER_STORE")
            panel = self._context_evaluator.reference_panel(
                actor_context=self._actor_context, pin=proposal.evaluation_basis.obligation_suite)
            if (panel.scorer != proposal.evaluation_basis.evaluator
                    or panel.comparison_policy != proposal.evaluation_basis.comparison_policy
                    or panel.matched.case != proposal.matched_cases[0]
                    or panel.holdout.case != proposal.independent_holdout_cases[0]):
                raise ChangeCandidateComparisonError("FROZEN_PANEL_BASIS_MISMATCH")
            for decision, expected_request in zip(decisions, (
                    panel.matched.baseline_request, panel.matched.successor_request,
                    panel.holdout.baseline_request, panel.holdout.successor_request)):
                if decision.outcome_observation is None:
                    raise ChangeCandidateComparisonError("PRODUCER_RUN_REQUIRED")
                expected = self._context_evaluator.grade(
                    actor_context=self._actor_context, panel_pin=panel.artifact_pin(),
                    run=decision.outcome_observation, persist=False)
                if (decision != expected
                        or expected_request not in decision.reasoning.invocations[0].inputs):
                    raise ChangeCandidateComparisonError("REFERENCE_EVALUATION_MISMATCH")
        else:
            # Legacy comparison remains readable, but changed semantic scorer is
            # not made comparable merely by unchanged outputs or case labels.
            identities = tuple(tuple((item.executable_engine, item.rules, item.strategy)
                                     for item in decision.reasoning.invocations)
                               for decision in decisions)
            if not identities[0] or any(value != identities[0] for value in identities[1:]):
                raise ChangeCandidateComparisonError("SEMANTIC_SCORER_BASIS_MISMATCH")
        if any(item.purpose != "EVALUATION" for item in decisions):
            raise ChangeCandidateComparisonError("EVALUATION_DECISION_REQUIRED")
        if len(proposal.matched_cases) != 1 or len(proposal.independent_holdout_cases) != 1:
            raise ChangeCandidateComparisonError("EXACT_MATCHED_AND_HOLDOUT_CASES_REQUIRED")
        matched_case = proposal.matched_cases[0]
        holdout_case = proposal.independent_holdout_cases[0]
        if matched_case != proposal.origin.context:
            raise ChangeCandidateComparisonError("MATCHED_CASE_MUST_EQUAL_ORIGIN_CONTEXT")
        if matched_case.artifact_kind != "context_graph" or holdout_case.artifact_kind != "context_graph":
            raise ChangeCandidateComparisonError("CASE_CONTEXT_REQUIRED")
        matched_baseline, matched_successor, holdout_baseline, holdout_successor = decisions
        if matched_baseline.context != matched_case or holdout_baseline.context != holdout_case:
            raise ChangeCandidateComparisonError("BASELINE_CASE_BINDING_MISMATCH")
        if matched_baseline.subject != matched_successor.subject or holdout_baseline.subject != holdout_successor.subject:
            raise ChangeCandidateComparisonError("CASE_TARGET_CHANGED")
        holdout_context = self._load(scope=scope, pin=holdout_case, model=ContextGraph)
        matched_successor_context = self._load(
            scope=scope, pin=matched_successor.context, model=ContextGraph
        )
        holdout_successor_context = self._load(
            scope=scope, pin=holdout_successor.context, model=ContextGraph
        )
        if any(
            item.state != "ADMITTED"
            for item in (context, holdout_context, matched_successor_context, holdout_successor_context)
        ):
            raise ChangeCandidateComparisonError("ADMITTED_CONTEXT_REQUIRED")
        if (
            holdout_context.ontology != context.ontology
            or holdout_context.intent != context.intent
            or holdout_context.formation.source_inputs != context.formation.source_inputs
            or holdout_context.formation.formation_policy != context.formation.formation_policy
        ):
            raise ChangeCandidateComparisonError("HOLDOUT_BASIS_MISMATCH")
        for decision, decision_context in (
            (matched_baseline, context),
            (matched_successor, matched_successor_context),
            (holdout_baseline, holdout_context),
            (holdout_successor, holdout_successor_context),
        ):
            if decision.ontology != decision_context.ontology:
                raise ChangeCandidateComparisonError("DECISION_ONTOLOGY_BINDING_MISMATCH")
        if panel is None and (matched_baseline.evidence != context.evidence or holdout_baseline.evidence != holdout_context.evidence):
            raise ChangeCandidateComparisonError("BASELINE_EVIDENCE_BINDING_MISMATCH")
        if panel is None and (matched_successor.evidence != matched_successor_context.evidence or holdout_successor.evidence != holdout_successor_context.evidence):
            raise ChangeCandidateComparisonError("SUCCESSOR_EVIDENCE_BINDING_MISMATCH")
        if panel is not None:
            if (proposal.target != "CONTEXT_FORMATION"
                    or successor_input_pin != matched_successor_context.formation.formation_policy
                    or successor_input_pin != holdout_successor_context.formation.formation_policy
                    or successor_input_pin == context.formation.formation_policy):
                raise ChangeCandidateComparisonError("DECLARED_CONTEXT_SUCCESSOR_INPUT_REQUIRED")
        elif proposal.target == "CONTEXT_FORMATION":
            if successor_input_pin == context.formation.formation_policy:
                raise ChangeCandidateComparisonError("FORMATION_POLICY_UNCHANGED")
            for baseline_context, successor_context in (
                (context, matched_successor_context),
                (holdout_context, holdout_successor_context),
            ):
                if successor_context.formation.formation_policy != successor_input_pin:
                    raise ChangeCandidateComparisonError("SUCCESSOR_FORMATION_POLICY_MISMATCH")
                if (
                    successor_context.intent != baseline_context.intent
                    or successor_context.ontology != baseline_context.ontology
                    or successor_context.world != baseline_context.world
                    or successor_context.formation.source_inputs
                    != baseline_context.formation.source_inputs
                ):
                    raise ChangeCandidateComparisonError("CONTEXT_FORMATION_DELTA_ESCAPED_BOUND")
        elif proposal.target == "ONTOLOGY":
            if successor_input_pin.artifact_kind != "ontology":
                raise ChangeCandidateComparisonError("ONTOLOGY_SUCCESSOR_REQUIRED")
            for baseline_context, successor_context in (
                (context, matched_successor_context),
                (holdout_context, holdout_successor_context),
            ):
                if successor_context.ontology != successor_input_pin:
                    raise ChangeCandidateComparisonError("SUCCESSOR_ONTOLOGY_MISMATCH")
                if (
                    successor_context.intent != baseline_context.intent
                    or successor_context.world != baseline_context.world
                    or successor_context.evidence != baseline_context.evidence
                    or successor_context.formation.formation_policy
                    != baseline_context.formation.formation_policy
                ):
                    raise ChangeCandidateComparisonError("ONTOLOGY_DELTA_ESCAPED_BOUND")
        else:
            raise ChangeCandidateComparisonError("REASONING_COMPARISON_REQUIRES_REASONING_EVALUATOR")
        if any(item.epistemic_status != "KNOWN" for item in decisions):
            verdict = "INDETERMINATE"
            reason = "UNKNOWN_SEMANTIC_RESULT"
        elif panel is not None:
            # MATCH means oracle correctness here, unlike legacy predicate truth.
            if any(before.result_code == "MATCH" and after.result_code == "NO_MATCH"
                   for before, after in ((matched_baseline, matched_successor),
                                         (holdout_baseline, holdout_successor))):
                verdict, reason = "WORSE", "CRITICAL_REFERENCE_REGRESSION"
            elif holdout_baseline.result_code != "MATCH" or holdout_successor.result_code != "MATCH":
                verdict, reason = "INDETERMINATE", "HOLDOUT_REFERENCE_CORRECTNESS_NOT_ESTABLISHED"
            elif matched_baseline.result_code == "NO_MATCH" and matched_successor.result_code == "MATCH":
                verdict, reason = "BETTER", "MATCHED_CORRECTNESS_IMPROVED_HOLDOUT_CORRECT"
            else:
                verdict, reason = "INDETERMINATE", "MATCHED_EXPECTATION_NOT_PROVEN"
        else:
            matched_baseline, matched_successor, holdout_baseline, holdout_successor = decisions
            holdout_stable = holdout_successor.result_code == holdout_baseline.result_code
            matched_expected = (
                matched_baseline.result_code == expectation.matched_baseline_result
                and matched_successor.result_code == expectation.matched_successor_result
            )
            if not holdout_stable:
                verdict, reason = "WORSE", "INDEPENDENT_HOLDOUT_REGRESSED"
            elif matched_expected:
                verdict, reason = "BETTER", "MATCHED_IMPROVED_HOLDOUT_STABLE"
            else:
                verdict, reason = "INDETERMINATE", "MATCHED_EXPECTATION_NOT_PROVEN"
        evidence = tuple(
            sorted(
                {pin for item in decisions for pin in item.evidence},
                key=_key,
            )
        )
        if not evidence:
            raise ChangeCandidateComparisonError("EVIDENCE_REQUIRED")
        inputs = (*tuple(item.pin() for item in decisions), successor_input_pin)
        invocation = ReasonerInvocation(
            invocation_id="change-candidate-comparison",
            strategy="bounded-matched-holdout-comparison",
            executable_engine=proposal.evaluation_basis.evaluator,
            rules=(
                proposal.evaluation_basis.obligation_suite,
                proposal.evaluation_basis.comparison_policy,
            ),
            inputs=inputs,
            outputs=(verdict,),
            proof_steps=(
                f"matched-baseline={decisions[0].result_code}",
                f"matched-successor={decisions[1].result_code}",
                f"holdout-baseline={decisions[2].result_code}",
                f"holdout-successor={decisions[3].result_code}",
                reason,
            ),
            evidence=evidence,
        )
        comparison = DecisionRecord.create(
            schema_version="semantier.decision_record.v1",
            organization_id=scope[0],
            workspace_id=scope[1],
            purpose="EVALUATION",
            subject=proposal.pin(),
            context=proposal.origin.context,
            ontology=context.ontology,
            result_code=verdict,
            epistemic_status="KNOWN",
            evidence=evidence,
            admissibility=None,
            reasoning=ReasoningTrace(
                strategy="change-candidate-comparison",
                reasoning_version="context-reference-panel-comparison.v1" if panel is not None else "change-candidate-comparison.v1",
                invocations=(invocation,),
                derived_assertion_ids=(),
                diagnostics=(),
            ),
            actor_id=actor_id,
            reasoner_identity=proposal.evaluation_basis.evaluator.artifact_ref,
            authorization=None,
            outcome_observation=None,
            recorded_at=max(item.recorded_at for item in decisions),
        )
        if persist:
            self._store.put(comparison)
        return comparison


__all__ = ["ChangeCandidateComparisonError", "ChangeCandidateComparisonService"]
