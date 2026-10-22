"""Immutable ChangeCandidate lifecycle over the shared five-object kernel store."""

from __future__ import annotations

from storage.hosted_postgres import sql_query,begin_write,in_transaction

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from pydantic import ValidationError

from contracts.change_candidate_v2 import (
    ChangeCandidateV2,
    validate_change_candidate_v2_revision,
)
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import (
    ChangeAssessment,
    ChangeCandidate,
    ChangeDelta,
    ChangeObservation,
    ChangeOrigin,
    ContextGraph,
    ContextGraphOrigin,
    DecisionRecord,
    EvaluationBasis,
    Evidence,
    PreContextOrigin,
    WorkflowEditOrigin,
    validate_change_candidate_revision,
)
from storage.ontology_build_store import OntologyBuildStore


class ChangeCandidateError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _key(pin: ArtifactPinV1) -> tuple[str, str, str]:
    return pin.artifact_kind, pin.artifact_ref, pin.artifact_hash


@dataclass(frozen=True)
class EpistemicLearningProjectionV1:
    """Pure, non-persisted learning context reconstructed from kernel records."""

    current_context: ArtifactPinV1 | None
    accepted_candidates: tuple[ChangeCandidate, ...]
    rejected_candidates: tuple[ChangeCandidate, ...]
    pending_candidates: tuple[ChangeCandidate, ...]
    candidate_lineage: tuple[tuple[ChangeCandidate, ...], ...]
    evidence_pins: tuple[ArtifactPinV1, ...]
    decision_pins: tuple[ArtifactPinV1, ...]
    unresolved_questions: tuple[str, ...]
    qualified_successor_inputs: tuple[ArtifactPinV1, ...] = ()

    @property
    def accepted_successor_inputs(self) -> tuple[ArtifactPinV1, ...]:
        """Explicit attributed inputs available to the ordinary formation path."""
        return self.qualified_successor_inputs

    def canonical(self) -> dict[str, object]:
        return {
            "current_context": self.current_context.model_dump(mode="json") if self.current_context else None,
            "accepted_candidates": [item.model_dump(mode="json") for item in self.accepted_candidates],
            "rejected_candidates": [item.model_dump(mode="json") for item in self.rejected_candidates],
            "pending_candidates": [item.model_dump(mode="json") for item in self.pending_candidates],
            "candidate_lineage": [[item.model_dump(mode="json") for item in chain] for chain in self.candidate_lineage],
            "evidence_pins": [item.model_dump(mode="json") for item in self.evidence_pins],
            "decision_pins": [item.model_dump(mode="json") for item in self.decision_pins],
            "unresolved_questions": list(self.unresolved_questions),
        }


class ChangeCandidateService:
    """Persist A/B/C candidate revisions without acquiring admission authority."""

    def __init__(self, *, store: OntologyBuildStore) -> None:
        if not isinstance(store, OntologyBuildStore):
            raise TypeError("ChangeCandidate requires the shared kernel store")
        self._store = store

    def _load(self, *, scope: tuple[str, str], pin: ArtifactPinV1, model):
        if pin.artifact_kind != model.ARTIFACT_KIND:
            raise ChangeCandidateError("INVALID_PIN_KIND")
        try:
            value = self._store.get(
                organization_id=scope[0],
                workspace_id=scope[1],
                artifact_ref=pin.artifact_ref,
                contract_type=model,
            )
        except ValidationError as exc:
            raise ChangeCandidateError("HASH_MISMATCH") from exc
        if value is None:
            raise ChangeCandidateError("OBJECT_NOT_FOUND")
        if value.pin() != pin:
            raise ChangeCandidateError("HASH_MISMATCH")
        return value

    def _verify_origin(self, *, scope: tuple[str, str], origin: ChangeOrigin) -> None:
        if isinstance(origin, ContextGraphOrigin):
            context = self._load(scope=scope, pin=origin.context, model=ContextGraph)
            if context.state not in {"PROPOSED", "ADMITTED"}:
                raise ChangeCandidateError("INVALID_ORIGIN_CONTEXT")
            allowed = {_key(pin) for pin in context.formation.source_inputs}
            if any(_key(pin) not in allowed for pin in origin.evidence):
                raise ChangeCandidateError("ORIGIN_EVIDENCE_NOT_IN_FORMATION_CLOSURE")
            for pin in origin.evidence:
                self._load(scope=scope, pin=pin, model=Evidence)
        elif isinstance(origin, PreContextOrigin):
            formation = origin.formation_input
            if (formation.organization_id, formation.workspace_id) != scope:
                raise ChangeCandidateError("FORBIDDEN_SCOPE")
            allowed = {_key(pin) for pin in formation.evidence}
            if any(_key(pin) not in allowed for pin in origin.evidence):
                raise ChangeCandidateError("ORIGIN_EVIDENCE_NOT_IN_FORMATION")
            for pin in origin.evidence:
                self._load(scope=scope, pin=pin, model=Evidence)
        elif isinstance(origin, WorkflowEditOrigin):
            for pin in origin.evidence:
                self._load(scope=scope, pin=pin, model=Evidence)
        else:
            raise ChangeCandidateError("INVALID_ORIGIN")

    def reconcile_semantic_graph_edit(self, *, organization_id: str, workspace_id: str, command):
        """Replay the semantic delta using only exact original kernel pins."""
        from contracts.ontology_workbench_context import SemanticGraphEditCommandV1
        from contracts.semantic_kernel import Ontology, ProtectedInput
        from eos.artifact_hashing import content_hash

        command = SemanticGraphEditCommandV1.model_validate(command.model_dump(mode="json"))
        if command.graph_pin.artifact_kind != "context_graph":
            raise ChangeCandidateError("NOT_RUN_CONTRACT_MISMATCH")
        scope = (organization_id, workspace_id)
        context = self._load(scope=scope, pin=command.graph_pin, model=ContextGraph)
        self._load(scope=scope, pin=context.ontology, model=Ontology)
        op = command.edit.operation
        nodes = {node.node_id: node for node in context.nodes}
        edges = {edge.edge_id: edge for edge in context.edges}
        if op.kind == "RENAME_NODE":
            node = nodes.get(op.node_ref)
            if node is None:
                raise ChangeCandidateError("UNKNOWN_OBJECT")
            if node.label != op.previous_label:
                raise ChangeCandidateError("STALE_SEMANTIC_BASIS")
        else:
            if op.source_node_ref not in nodes or op.target_node_ref not in nodes:
                raise ChangeCandidateError("UNKNOWN_OBJECT")
            if op.kind != "ADD_RELATION":
                edge = edges.get(op.edge_ref)
                if edge is None or (edge.source_id, edge.target_id) != (op.source_node_ref, op.target_node_ref):
                    raise ChangeCandidateError("UNKNOWN_OBJECT")
                if edge.relation_type != op.previous_relation_type:
                    raise ChangeCandidateError("STALE_SEMANTIC_BASIS")
            if op.kind == "ADD_RELATION" and any(
                (edge.source_id, edge.target_id, edge.relation_type) == (op.source_node_ref, op.target_node_ref, op.proposed_relation_type)
                for edge in context.edges
            ):
                raise ChangeCandidateError("NO_CHANGE_REQUIRED")
        # The kernel declares evidence closure on the formed context; assertions
        # narrow it when present, never browser labels or renderer adjacency.
        attached = {
            _key(pin) for assertion in context.assertions
            if assertion.subject_ref in command.edit.affected_object_refs or assertion.object_ref in command.edit.affected_object_refs
            for pin in assertion.evidence
        }
        allowed = attached if context.assertions else {_key(pin) for pin in context.evidence}
        if any(_key(pin) not in allowed for pin in command.evidence_pins):
            raise ChangeCandidateError("EVIDENCE_OUT_OF_CLOSURE")
        for pin in command.evidence_pins:
            self._load(scope=scope, pin=pin, model=Evidence)
        semantic = {"graph": command.graph_pin.model_dump(mode="json"), "operation": op.model_dump(mode="json"), "evidence": [pin.model_dump(mode="json") for pin in command.evidence_pins], "rationale": command.rationale}
        def protocol_pin(kind: str, protocol: str) -> ArtifactPinV1:
            digest = content_hash({"protocol": protocol})
            return ArtifactPinV1(artifact_kind=kind, artifact_ref=f"{kind}:{digest}", artifact_hash=f"sha256:{digest}")
        basis = EvaluationBasis(
            evaluator=protocol_pin("evaluator", "semantic-graph-edit.v1"),
            obligation_suite=protocol_pin("obligation_suite", "semantic-graph-edit-evidence.v1"),
            comparison_policy=protocol_pin("comparison_policy", "semantic-graph-edit-bounded.v1"),
            protected_inputs=(ProtectedInput(slot_id="context", artifact=context.pin()), ProtectedInput(slot_id="ontology", artifact=context.ontology),
                *tuple(ProtectedInput(slot_id=f"evidence:{index}", artifact=pin) for index, pin in enumerate(command.evidence_pins))),
            candidate_delta_bounds="Only the pinned semantic operation may change; graph and evidence basis remain frozen",
        )
        import json
        delta = ChangeDelta(description=json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")), claims_semantic_graph_change=True, basis_refs=(context.pin(), *command.evidence_pins))
        return context, basis, delta

    def propose_semantic_graph_edit(
        self, *, organization_id: str, workspace_id: str, command,
        current_graph_pin: ArtifactPinV1, actor_ref: str, request_id: str,
        request_hash: str, recorded_at: datetime, verify_material_basis: Callable[[], None] | None = None,
    ) -> ChangeCandidate:
        """Current-basis guarded ingress into the existing proposal writer."""
        if command.graph_pin != current_graph_pin:
            raise ChangeCandidateError("STALE_SEMANTIC_BASIS")
        context, basis, delta = self.reconcile_semantic_graph_edit(
            organization_id=organization_id, workspace_id=workspace_id, command=command,
        )
        def verify_basis():
            if verify_material_basis is not None:
                verify_material_basis()
            successor = self._store._connection.execute(
                sql_query("SELECT 1 FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? AND schema_version='semantier.context_graph.v1' AND json_extract(payload_json, '$.predecessor.artifact_ref')=? LIMIT 1","SELECT 1 FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND schema_version='semantier.context_graph.v1' AND (payload_json::jsonb #>> '{predecessor,artifact_ref}')=%s LIMIT 1"),
                (organization_id, workspace_id, context.artifact_ref),
            ).fetchone()
            if successor is not None:
                raise ChangeCandidateError("STALE_SEMANTIC_BASIS")
        return self.propose_with_request(
            organization_id=organization_id, workspace_id=workspace_id,
            origin=ContextGraphOrigin(context=context.pin(), evidence=command.evidence_pins),
            failure_class="STRUCTURE", target="ONTOLOGY", hypothesis=command.rationale,
            evaluation_basis=basis,
            delta=delta,
            matched_cases=(context.pin(),), independent_holdout_cases=(),
            recorded_at=recorded_at, actor_ref=actor_ref, request_id=request_id, request_hash=request_hash,
            verify_proposal_basis=verify_basis,
        )

    def propose(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        origin: ChangeOrigin,
        failure_class: str,
        target: str,
        hypothesis: str,
        evaluation_basis: EvaluationBasis,
        delta: ChangeDelta,
        matched_cases: tuple[ArtifactPinV1, ...],
        independent_holdout_cases: tuple[ArtifactPinV1, ...],
        recorded_at: datetime,
    ) -> ChangeCandidate:
        scope = (organization_id, workspace_id)
        self._verify_origin(scope=scope, origin=origin)
        candidate = ChangeCandidate.create(
            schema_version="semantier.change_candidate.v1",
            organization_id=organization_id,
            workspace_id=workspace_id,
            revision=0,
            stage="PROPOSAL",
            predecessor=None,
            origin=origin,
            failure_class=failure_class,
            target=target,
            hypothesis=hypothesis,
            evaluation_basis=evaluation_basis,
            delta=delta,
            matched_cases=matched_cases,
            independent_holdout_cases=independent_holdout_cases,
            observations=(),
            assessment=None,
            successor_inputs=(),
            recorded_at=recorded_at,
        )
        self._store.put(candidate)
        return candidate

    def _request_candidate(
        self,
        *,
        scope: tuple[str, str],
        actor_ref: str,
        operation: str,
        request_id: str,
        request_hash: str,
    ) -> ChangeCandidate | None:
        row = self._store._connection.execute(
            sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
            "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
            (scope[0], scope[1], actor_ref, operation, request_id),
        ).fetchone()
        if row is None:
            return None
        if row[0] != request_hash:
            raise ChangeCandidateError("REQUEST_CONFLICT")
        return self._load(
            scope=scope,
            pin=ArtifactPinV1(
                artifact_kind="change_candidate",
                artifact_ref=row[1],
                artifact_hash=row[2],
            ),
            model=ChangeCandidate,
        )

    def propose_with_request(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        origin: ChangeOrigin,
        failure_class: str,
        target: str,
        hypothesis: str,
        evaluation_basis: EvaluationBasis,
        delta: ChangeDelta,
        matched_cases: tuple[ArtifactPinV1, ...],
        independent_holdout_cases: tuple[ArtifactPinV1, ...],
        recorded_at: datetime,
        actor_ref: str,
        request_id: str,
        request_hash: str,
        verify_proposal_basis: Callable[[], None] | None = None,
    ) -> ChangeCandidate:
        scope = (organization_id, workspace_id)
        existing = self._request_candidate(
            scope=scope,
            actor_ref=actor_ref,
            operation="SUBMIT_CHANGE_CANDIDATE",
            request_id=request_id,
            request_hash=request_hash,
        )
        if existing is not None:
            return existing
        self._verify_origin(scope=scope, origin=origin)
        candidate = ChangeCandidate.create(
            schema_version="semantier.change_candidate.v1",
            organization_id=organization_id,
            workspace_id=workspace_id,
            revision=0,
            stage="PROPOSAL",
            predecessor=None,
            origin=origin,
            failure_class=failure_class,
            target=target,
            hypothesis=hypothesis,
            evaluation_basis=evaluation_basis,
            delta=delta,
            matched_cases=matched_cases,
            independent_holdout_cases=independent_holdout_cases,
            observations=(),
            assessment=None,
            successor_inputs=(),
            recorded_at=recorded_at,
        )

        def validate_before_insert(connection):
            if connection is not self._store._connection:
                raise ChangeCandidateError("INVALID_STATE")
            if verify_proposal_basis is not None:
                verify_proposal_basis()
            return self._request_candidate(
                scope=scope,
                actor_ref=actor_ref,
                operation="SUBMIT_CHANGE_CANDIDATE",
                request_id=request_id,
                request_hash=request_hash,
            ) is None

        def mutate_indexes(connection):
            connection.execute(
                sql_query("INSERT INTO workspace_operation_request_index "
                "(organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) "
                "VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index (organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                (
                    organization_id,
                    workspace_id,
                    actor_ref,
                    "SUBMIT_CHANGE_CANDIDATE",
                    request_id,
                    request_hash,
                    candidate.artifact_ref,
                    candidate.artifact_hash,
                ),
            )

        inserted = self._store.atomic_write(
            artifacts=(candidate,),
            validate_before_insert=validate_before_insert,
            mutate_indexes=mutate_indexes,
        )
        if inserted is None:
            retry = self._request_candidate(
                scope=scope,
                actor_ref=actor_ref,
                operation="SUBMIT_CHANGE_CANDIDATE",
                request_id=request_id,
                request_hash=request_hash,
            )
            if retry is None:
                raise ChangeCandidateError("INCOMPLETE_CLOSURE")
            return retry
        return candidate

    def evaluate_pair_with_request(
        self,
        *,
        proposal: ArtifactPinV1,
        matched_execution_attempt: ArtifactPinV1,
        matched_metric_refs: tuple[ArtifactPinV1, ...],
        matched_status: str,
        holdout_execution_attempt: ArtifactPinV1,
        holdout_metric_refs: tuple[ArtifactPinV1, ...],
        holdout_status: str,
        comparison_decision: ArtifactPinV1 | None,
        successor_input: ArtifactPinV1,
        recorded_at: datetime,
        actor_ref: str,
        request_id: str,
        request_hash: str,
        comparison_record: DecisionRecord | None = None,
        context_evaluator=None,
        actor_context=None,
    ) -> ChangeCandidate:
        scope = self._scope(proposal)
        existing = self._request_candidate(
            scope=scope,
            actor_ref=actor_ref,
            operation="ASSESS_CHANGE_CANDIDATE",
            request_id=request_id,
            request_hash=request_hash,
        )
        previous = self._load(scope=scope, pin=proposal, model=ChangeCandidate)
        if previous.stage != "PROPOSAL":
            raise ChangeCandidateError("PROPOSAL_REQUIRED")
        if len(previous.matched_cases) != 1 or len(previous.independent_holdout_cases) != 1:
            raise ChangeCandidateError("BOUNDED_PAIR_REQUIRED")
        if previous.evaluation_basis.obligation_suite.artifact_kind != "runtime_context_reference_panel":
            raise ChangeCandidateError("REFERENCE_BOUND_PAIR_REQUIRED")
        if context_evaluator is None or actor_context is None:
            raise ChangeCandidateError("CONTEXT_EVALUATOR_REQUIRED")
        from services.change_candidate_comparison_service import ChangeCandidateComparisonService

        if comparison_decision is None:
            raise ChangeCandidateError("COMPARISON_DECISION_REQUIRED")
        supplied = comparison_record or self._load(scope=scope, pin=comparison_decision, model=DecisionRecord)
        if supplied.pin() != comparison_decision:
            raise ChangeCandidateError("INVALID_COMPARISON_DECISION")
        inputs = supplied.reasoning.invocations[0].inputs
        if len(inputs) != 5:
            raise ChangeCandidateError("INVALID_COMPARISON_DECISION")
        panel = context_evaluator.reference_panel(
            actor_context=actor_context, pin=previous.evaluation_basis.obligation_suite)
        decision_inputs = tuple(self._load(scope=scope, pin=pin, model=DecisionRecord)
                                for pin in inputs if pin.artifact_kind == "decision_record")
        decisions = []
        for request in (panel.matched.baseline_request, panel.matched.successor_request,
                        panel.holdout.baseline_request, panel.holdout.successor_request):
            matches = tuple(decision for decision in decision_inputs
                            if request in decision.reasoning.invocations[0].inputs)
            if len(matches) != 1:
                raise ChangeCandidateError("INVALID_COMPARISON_DECISION")
            decisions.append(matches[0])
        if successor_input not in inputs:
            raise ChangeCandidateError("INVALID_COMPARISON_DECISION")
        validated = ChangeCandidateComparisonService(
            store=self._store, context_evaluator=context_evaluator, actor_context=actor_context).compare(
                proposal_pin=proposal, matched_baseline_decision_pin=decisions[0].pin(),
                matched_successor_decision_pin=decisions[1].pin(), holdout_baseline_decision_pin=decisions[2].pin(),
                holdout_successor_decision_pin=decisions[3].pin(), successor_input_pin=successor_input,
                actor_id=actor_ref, persist=False)
        if validated != supplied:
            raise ChangeCandidateError("INVALID_COMPARISON_DECISION")
        if (matched_execution_attempt != decisions[1].outcome_observation
                or holdout_execution_attempt != decisions[3].outcome_observation):
            raise ChangeCandidateError("PRODUCER_ATTEMPT_BINDING_MISMATCH")
        # Status and metric evidence come from validated producer/reference joins.
        matched_status = holdout_status = "COMPLETE"
        matched_metric_refs = (decisions[0].pin(), decisions[1].pin())
        holdout_metric_refs = (decisions[2].pin(), decisions[3].pin())
        observations = (
            ChangeObservation(
                case=previous.matched_cases[0],
                execution_attempt=matched_execution_attempt,
                status=matched_status,
                metric_refs=matched_metric_refs,
                diagnostics=(),
            ),
            ChangeObservation(
                case=previous.independent_holdout_cases[0],
                execution_attempt=holdout_execution_attempt,
                status=holdout_status,
                metric_refs=holdout_metric_refs,
                diagnostics=(),
            ),
        )
        if existing is not None:
            if existing.observations != observations:
                raise ChangeCandidateError("RETAINED_ASSESSMENT_EVIDENCE_CHANGED")
            return existing
        observation_revision = ChangeCandidate.create(
            **self._revision_values(previous),
            revision=1,
            stage="OBSERVATION",
            predecessor=previous.pin(),
            observations=observations,
            assessment=None,
            successor_inputs=(),
            recorded_at=recorded_at,
        )
        validate_change_candidate_revision(previous, observation_revision)
        both_complete = matched_status == "COMPLETE" and holdout_status == "COMPLETE"
        rejected = matched_status in {"FAILED", "REJECTED"} or holdout_status in {"FAILED", "REJECTED"}
        comparison = None
        comparison_verdict = "INDETERMINATE"
        if both_complete:
            if comparison_decision is None:
                raise ChangeCandidateError("COMPARISON_DECISION_REQUIRED")
            comparison = supplied
            if (
                comparison.purpose != "EVALUATION"
                or comparison.subject != previous.pin()
                or not isinstance(previous.origin, ContextGraphOrigin)
                or comparison.context != previous.origin.context
                or comparison.authorization is not None
            ):
                raise ChangeCandidateError("INVALID_COMPARISON_DECISION")
            basis = previous.evaluation_basis
            invocation = next(
                (
                    item
                    for item in comparison.reasoning.invocations
                    if item.executable_engine == basis.evaluator
                    and basis.obligation_suite in item.rules
                    and basis.comparison_policy in item.rules
                ),
                None,
            )
            if invocation is None:
                raise ChangeCandidateError("COMPARISON_BASIS_MISMATCH")
            if comparison.result_code not in {"BETTER", "WORSE", "INDETERMINATE", "LOCAL_FIX_VERIFIED"}:
                raise ChangeCandidateError("INVALID_COMPARISON_VERDICT")
            comparison_verdict = comparison.result_code
        elif comparison_decision is not None:
            raise ChangeCandidateError("COMPARISON_REQUIRES_COMPLETE_PAIR")

        if comparison_verdict == "BETTER":
            disposition = "ACCEPTED_FOR_SUCCESSOR_INPUT"
            reason_code = "MATCHED_IMPROVED_INDEPENDENT_HOLDOUT_STABLE"
        elif comparison_verdict == "WORSE" or rejected:
            disposition = "REJECTED_FOR_USE"
            reason_code = "COMPARISON_WORSE" if comparison_verdict == "WORSE" else "TRIAL_REJECTED"
        else:
            disposition = "NOT_YET_ADMISSIBLE"
            reason_code = "COMPARISON_INDETERMINATE" if both_complete else "TRIAL_INCOMPLETE"
        metric_refs = tuple(
            sorted(
                (*matched_metric_refs, *holdout_metric_refs, *((comparison.pin(),) if comparison is not None else ())),
                key=_key,
            )
        )
        assessment = ChangeAssessment(
            trial_validity="VALID" if both_complete or rejected else "INCOMPLETE_EVIDENCE",
            verdict=comparison_verdict if both_complete else "INDETERMINATE",
            metric_refs=metric_refs,
            disposition=disposition,
            reason_code=reason_code,
        )
        assessment_revision = ChangeCandidate.create(
            **self._revision_values(observation_revision),
            revision=2,
            stage="ASSESSMENT",
            predecessor=observation_revision.pin(),
            observations=observations,
            assessment=assessment,
            successor_inputs=((successor_input,) if comparison_verdict == "BETTER" else ()),
            recorded_at=recorded_at,
        )
        validate_change_candidate_revision(observation_revision, assessment_revision)

        def validate_before_insert(connection):
            if connection is not self._store._connection:
                raise ChangeCandidateError("INVALID_STATE")
            return self._request_candidate(
                scope=scope,
                actor_ref=actor_ref,
                operation="ASSESS_CHANGE_CANDIDATE",
                request_id=request_id,
                request_hash=request_hash,
            ) is None

        def mutate_indexes(connection):
            connection.execute(
                sql_query("INSERT INTO workspace_operation_request_index "
                "(organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) "
                "VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index (organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                (
                    scope[0],
                    scope[1],
                    actor_ref,
                    "ASSESS_CHANGE_CANDIDATE",
                    request_id,
                    request_hash,
                    assessment_revision.artifact_ref,
                    assessment_revision.artifact_hash,
                ),
            )

        inserted = self._store.atomic_write(
            artifacts=(supplied, observation_revision, assessment_revision),
            validate_before_insert=validate_before_insert,
            mutate_indexes=mutate_indexes,
        )
        if inserted is None:
            retry = self._request_candidate(
                scope=scope,
                actor_ref=actor_ref,
                operation="ASSESS_CHANGE_CANDIDATE",
                request_id=request_id,
                request_hash=request_hash,
            )
            if retry is None:
                raise ChangeCandidateError("INCOMPLETE_CLOSURE")
            return retry
        return assessment_revision

    def record_observations(
        self,
        *,
        proposal: ArtifactPinV1,
        observations: tuple[ChangeObservation, ...],
        recorded_at: datetime,
    ) -> ChangeCandidate:
        scope = self._scope(proposal)
        previous = self._load(scope=scope, pin=proposal, model=ChangeCandidate)
        if previous.stage != "PROPOSAL":
            raise ChangeCandidateError("PROPOSAL_REQUIRED")
        if not observations:
            raise ChangeCandidateError("OBSERVATIONS_REQUIRED")
        allowed_cases = {_key(pin) for pin in (*previous.matched_cases, *previous.independent_holdout_cases)}
        seen_cases = [_key(item.case) for item in observations]
        if len(seen_cases) != len(set(seen_cases)):
            raise ChangeCandidateError("DUPLICATE_CASE_OBSERVATION")
        if any(item not in allowed_cases for item in seen_cases):
            raise ChangeCandidateError("UNBOUND_CASE_OBSERVATION")
        current = ChangeCandidate.create(
            **self._revision_values(previous),
            revision=previous.revision + 1,
            stage="OBSERVATION",
            predecessor=previous.pin(),
            observations=observations,
            assessment=None,
            successor_inputs=(),
            recorded_at=recorded_at,
        )
        validate_change_candidate_revision(previous, current)
        self._store.put(current)
        return current

    def assess(
        self,
        *,
        observation_revision: ArtifactPinV1,
        assessment: ChangeAssessment,
        successor_inputs: tuple[ArtifactPinV1, ...],
        recorded_at: datetime,
    ) -> ChangeCandidate:
        scope = self._scope(observation_revision)
        previous = self._load(scope=scope, pin=observation_revision, model=ChangeCandidate)
        if previous.stage != "OBSERVATION":
            raise ChangeCandidateError("OBSERVATION_REVISION_REQUIRED")
        if previous.evaluation_basis.obligation_suite.artifact_kind == "runtime_context_reference_panel":
            raise ChangeCandidateError("REFERENCE_BOUND_PAIR_INGRESS_REQUIRED")
        observed = {_key(item.case): item for item in previous.observations}
        required_matched = {_key(pin) for pin in previous.matched_cases}
        required_holdout = {_key(pin) for pin in previous.independent_holdout_cases}
        if assessment.trial_validity == "VALID":
            if not required_matched <= observed.keys():
                raise ChangeCandidateError("MATCHED_CASES_NOT_OBSERVED")
            if assessment.verdict == "BETTER":
                if not required_holdout <= observed.keys():
                    raise ChangeCandidateError("HOLDOUT_CASES_NOT_OBSERVED")
                if any(
                    observed[key].status != "COMPLETE"
                    for key in required_matched | required_holdout
                ):
                    raise ChangeCandidateError("GENERALIZATION_REQUIRES_COMPLETE_CASES")
        if assessment.disposition == "ACCEPTED_FOR_SUCCESSOR_INPUT" and not successor_inputs:
            raise ChangeCandidateError("ACCEPTED_CANDIDATE_REQUIRES_SUCCESSOR_INPUT")
        if assessment.disposition != "ACCEPTED_FOR_SUCCESSOR_INPUT" and successor_inputs:
            raise ChangeCandidateError("NON_ACCEPTED_CANDIDATE_CANNOT_BIND_SUCCESSOR_INPUT")
        current = ChangeCandidate.create(
            **self._revision_values(previous),
            revision=previous.revision + 1,
            stage="ASSESSMENT",
            predecessor=previous.pin(),
            observations=previous.observations,
            assessment=assessment,
            successor_inputs=successor_inputs,
            recorded_at=recorded_at,
        )
        validate_change_candidate_revision(previous, current)
        self._store.put(current)
        return current

    def get(self, *, pin: ArtifactPinV1) -> ChangeCandidate:
        """Legacy v1-only reader retained for existing empirical-learning consumers."""

        scope = self._scope(pin)
        raw = self._store.get_raw(
            organization_id=scope[0],
            workspace_id=scope[1],
            artifact_ref=pin.artifact_ref,
        )
        if raw is None:
            raise ChangeCandidateError("OBJECT_NOT_FOUND")
        if raw["schema_version"] != "semantier.change_candidate.v1":
            raise ChangeCandidateError("UNSUPPORTED_SCHEMA_VERSION")
        return self._load(scope=scope, pin=pin, model=ChangeCandidate)

    def _load_versioned_candidate(
        self, *, scope: tuple[str, str], pin: ArtifactPinV1
    ) -> ChangeCandidate | ChangeCandidateV2:
        if pin.artifact_kind != "change_candidate":
            raise ChangeCandidateError("INVALID_PIN_KIND")
        raw = self._store.get_raw(
            organization_id=scope[0],
            workspace_id=scope[1],
            artifact_ref=pin.artifact_ref,
        )
        if raw is None:
            raise ChangeCandidateError("OBJECT_NOT_FOUND")
        if raw["artifact_hash"] != pin.artifact_hash:
            raise ChangeCandidateError("HASH_MISMATCH")
        payload = raw["payload"]
        if not isinstance(payload, dict):
            raise ChangeCandidateError("HASH_MISMATCH")
        schema_version = str(raw["schema_version"])
        if payload.get("schema_version") != schema_version:
            raise ChangeCandidateError("HASH_MISMATCH")
        model = {
            "semantier.change_candidate.v1": ChangeCandidate,
            "semantier.change_candidate.v2": ChangeCandidateV2,
        }.get(schema_version)
        if model is None:
            raise ChangeCandidateError("UNSUPPORTED_SCHEMA_VERSION")
        try:
            value = model.model_validate(payload)
        except ValidationError as exc:
            raise ChangeCandidateError("HASH_MISMATCH") from exc
        if value.pin() != pin:
            raise ChangeCandidateError("HASH_MISMATCH")
        return value

    def get_versioned(
        self, *, pin: ArtifactPinV1
    ) -> ChangeCandidate | ChangeCandidateV2:
        """Read one exact ChangeCandidate by stored schema version without migration."""

        scope = self._scope(pin)
        return self._load_versioned_candidate(scope=scope, pin=pin)

    def get_learning_candidate(
        self, *, pin: ArtifactPinV1
    ) -> ChangeCandidate | ChangeCandidateV2:
        """Read a candidate only when it has empirical-learning semantics."""

        candidate = self.get_versioned(pin=pin)
        if isinstance(candidate, ChangeCandidateV2) and (
            candidate.candidate_kind != "LEARNING_HYPOTHESIS"
        ):
            raise ChangeCandidateError("UNSUPPORTED_CANDIDATE_PROFILE")
        return candidate

    def list_versioned(
        self, *, organization_id: str, workspace_id: str
    ) -> tuple[ChangeCandidate | ChangeCandidateV2, ...]:
        """List all supported persisted ChangeCandidate versions for one scope."""

        result: list[ChangeCandidate | ChangeCandidateV2] = []
        for schema_version in (
            "semantier.change_candidate.v1",
            "semantier.change_candidate.v2",
        ):
            for pin in self._store.list_artifact_pins(
                organization_id=organization_id,
                workspace_id=workspace_id,
                schema_version=schema_version,
            ):
                result.append(
                    self._load_versioned_candidate(
                        scope=(organization_id, workspace_id),
                        pin=pin,
                    )
                )
        return tuple(
            sorted(
                result,
                key=lambda item: (
                    item.schema_version,
                    item.revision,
                    item.artifact_ref,
                ),
            )
        )

    def reconstruct_versioned_revision_chain(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        candidate: ArtifactPinV1,
    ) -> tuple[ChangeCandidate | ChangeCandidateV2, ...]:
        """Reconstruct one exact homogeneous v1 or v2 predecessor chain."""

        chain: list[ChangeCandidate | ChangeCandidateV2] = []
        current = self._load_versioned_candidate(
            scope=(organization_id, workspace_id),
            pin=candidate,
        )
        expected_schema = current.schema_version
        seen: set[tuple[str, str, str]] = set()
        while True:
            key = _key(current.pin())
            if key in seen:
                raise ChangeCandidateError("REVISION_CYCLE")
            seen.add(key)
            chain.append(current)
            if current.predecessor is None:
                break
            previous = self._load_versioned_candidate(
                scope=(organization_id, workspace_id),
                pin=current.predecessor,
            )
            if previous.schema_version != expected_schema:
                raise ChangeCandidateError("MIXED_SCHEMA_REVISION_CHAIN")
            try:
                if isinstance(current, ChangeCandidate):
                    if not isinstance(previous, ChangeCandidate):
                        raise ChangeCandidateError("MIXED_SCHEMA_REVISION_CHAIN")
                    validate_change_candidate_revision(previous, current)
                else:
                    if not isinstance(previous, ChangeCandidateV2):
                        raise ChangeCandidateError("MIXED_SCHEMA_REVISION_CHAIN")
                    validate_change_candidate_v2_revision(previous, current)
            except ChangeCandidateError:
                raise
            except ValueError as exc:
                raise ChangeCandidateError("INVALID_REVISION_CHAIN") from exc
            current = previous
        chain.reverse()
        return tuple(chain)

    def list_for_origin(
        self, *, organization_id: str, workspace_id: str, origin_ref: str,
        origin_hash: str | None = None,
    ) -> tuple[ChangeCandidate, ...]:
        """List all revisions whose origin is the exact bounded context."""
        result = []
        for pin in self._store.list_artifact_pins(
            organization_id=organization_id,
            workspace_id=workspace_id,
            schema_version="semantier.change_candidate.v1",
        ):
            candidate = self._load(scope=(organization_id, workspace_id), pin=pin, model=ChangeCandidate)
            origin_pin = (
                candidate.origin.context
                if isinstance(candidate.origin, ContextGraphOrigin)
                else candidate.origin.formation_input.pin()
                if isinstance(candidate.origin, PreContextOrigin)
                else candidate.origin.workflow
            )
            if origin_pin.artifact_ref == origin_ref:
                if origin_hash is not None and origin_pin.artifact_hash != origin_hash:
                    raise ChangeCandidateError("HASH_MISMATCH")
                result.append(candidate)

        def origin_ref(item: ChangeCandidate) -> str:
            if isinstance(item.origin, ContextGraphOrigin):
                return item.origin.context.artifact_ref
            if isinstance(item.origin, PreContextOrigin):
                return item.origin.formation_input.pin().artifact_ref
            return item.origin.workflow.artifact_ref

        return tuple(
            sorted(
                result,
                key=lambda item: (origin_ref(item), item.revision, item.artifact_ref),
            )
        )

    def reconstruct_revision_chain(
        self, *, organization_id: str, workspace_id: str, candidate: ArtifactPinV1,
    ) -> tuple[ChangeCandidate, ...]:
        """Reconstruct a candidate's immutable predecessor chain oldest-first."""
        chain = []
        current = self._load(scope=(organization_id, workspace_id), pin=candidate, model=ChangeCandidate)
        while current is not None:
            chain.append(current)
            if current.predecessor is None:
                break
            current = self._load(scope=(organization_id, workspace_id), pin=current.predecessor, model=ChangeCandidate)
        chain.reverse()
        return tuple(chain)

    def verify_reference_assessment(self, *, candidate, context_evaluator, actor_context):
        """Revalidate retained assessment joins for current read qualification."""
        if candidate.evaluation_basis.obligation_suite.artifact_kind != "runtime_context_reference_panel":
            raise ChangeCandidateError("REFERENCE_BOUND_PAIR_REQUIRED")
        scope = (candidate.organization_id, candidate.workspace_id)
        chain = self.reconstruct_revision_chain(
            organization_id=scope[0], workspace_id=scope[1], candidate=candidate.pin())
        proposal = chain[0]
        if (len(chain) != 3 or candidate.assessment is None
                or candidate.assessment.trial_validity != "VALID"):
            raise ChangeCandidateError("INVALID_REFERENCE_ASSESSMENT")
        records = tuple(self._load(scope=scope, pin=pin, model=DecisionRecord)
                        for pin in candidate.assessment.metric_refs)
        comparisons = tuple(record for record in records if record.subject == proposal.pin())
        if len(comparisons) != 1:
            raise ChangeCandidateError("INVALID_REFERENCE_ASSESSMENT")
        comparison = comparisons[0]
        panel = context_evaluator.reference_panel(
            actor_context=actor_context, pin=proposal.evaluation_basis.obligation_suite)
        ordered = []
        for request in (panel.matched.baseline_request, panel.matched.successor_request,
                        panel.holdout.baseline_request, panel.holdout.successor_request):
            matches = tuple(record for record in records
                            if request in record.reasoning.invocations[0].inputs)
            if len(matches) != 1:
                raise ChangeCandidateError("INVALID_REFERENCE_ASSESSMENT")
            ordered.append(matches[0])
        successor_context = self._load(scope=scope, pin=ordered[1].context, model=ContextGraph)
        from services.change_candidate_comparison_service import ChangeCandidateComparisonService

        expected = ChangeCandidateComparisonService(store=self._store,
            context_evaluator=context_evaluator, actor_context=actor_context).compare(
                proposal_pin=proposal.pin(), matched_baseline_decision_pin=ordered[0].pin(),
                matched_successor_decision_pin=ordered[1].pin(),holdout_baseline_decision_pin=ordered[2].pin(),
                holdout_successor_decision_pin=ordered[3].pin(),
                successor_input_pin=successor_context.formation.formation_policy,
                actor_id=comparison.actor_id, persist=False)
        if (expected != comparison or candidate.assessment.verdict != expected.result_code
                or tuple(o.execution_attempt for o in candidate.observations) != tuple(
                    decision.outcome_observation for decision in (ordered[1],ordered[3]))):
            raise ChangeCandidateError("INVALID_REFERENCE_ASSESSMENT")
        expected_disposition = ("ACCEPTED_FOR_SUCCESSOR_INPUT" if expected.result_code == "BETTER"
                                else "REJECTED_FOR_USE" if expected.result_code == "WORSE"
                                else "NOT_YET_ADMISSIBLE")
        expected_successors = ((successor_context.formation.formation_policy,)
                               if expected.result_code == "BETTER" else ())
        if (candidate.assessment.disposition != expected_disposition
                or candidate.successor_inputs != expected_successors
                or set(candidate.assessment.metric_refs) != {comparison.pin(), *(record.pin() for record in ordered)}
                or tuple(o.case for o in candidate.observations) != (panel.matched.case, panel.holdout.case)
                or any(o.status != "COMPLETE" or o.diagnostics for o in candidate.observations)
                or tuple(frozenset(o.metric_refs) for o in candidate.observations) != (
                    frozenset((ordered[0].pin(),ordered[1].pin())), frozenset((ordered[2].pin(),ordered[3].pin())))):
            raise ChangeCandidateError("INVALID_REFERENCE_ASSESSMENT")
        return expected

    def project_learning_context(
        self, *, organization_id: str, workspace_id: str, origin_ref: str,
        origin_hash: str | None = None, current_context: ArtifactPinV1 | None = None,
        context_evaluator=None, actor_context=None,
    ) -> EpistemicLearningProjectionV1:
        candidates = self.list_for_origin(
            organization_id=organization_id, workspace_id=workspace_id,
            origin_ref=origin_ref, origin_hash=origin_hash,
        )
        unresolved_questions: tuple[str, ...] = ()
        if current_context is not None:
            context = self._load(
                scope=(organization_id, workspace_id),
                pin=current_context,
                model=ContextGraph,
            )
            unresolved_questions = tuple(context.intent.unresolved_questions)
        elif candidates:
            origin = candidates[0].origin
            if isinstance(origin, ContextGraphOrigin):
                context = self._load(
                    scope=(organization_id, workspace_id),
                    pin=origin.context,
                    model=ContextGraph,
                )
                unresolved_questions = tuple(context.intent.unresolved_questions)
            elif isinstance(origin, PreContextOrigin):
                unresolved_questions = tuple(origin.formation_input.intent.unresolved_questions)
        latest = {}
        for item in candidates:
            chain = self.reconstruct_revision_chain(
                organization_id=organization_id, workspace_id=workspace_id,
                candidate=item.pin(),
            )
            latest[chain[0].artifact_ref] = max(
                latest.get(chain[0].artifact_ref, item), item,
                key=lambda value: (value.revision, value.artifact_ref),
            )
        terminal = tuple(sorted(latest.values(), key=lambda item: item.artifact_ref))
        accepted_records = tuple(item for item in terminal if item.assessment and item.assessment.disposition == "ACCEPTED_FOR_SUCCESSOR_INPUT")
        accepted = []
        unqualified = []
        for item in accepted_records:
            if context_evaluator is None or actor_context is None:
                unqualified.append(item)
                continue
            try:
                self.verify_reference_assessment(candidate=item,
                    context_evaluator=context_evaluator, actor_context=actor_context)
            except ValueError:
                unqualified.append(item)
            else:
                accepted.append(item)
        accepted = tuple(accepted)
        unresolved_questions += tuple("EVALUATION_BASIS_UNQUALIFIED:" + item.artifact_ref for item in unqualified)
        rejected = tuple(item for item in terminal if item.assessment and item.assessment.disposition == "REJECTED_FOR_USE")
        pending = tuple(item for item in terminal if item.assessment is None or item.assessment.disposition in {"REVIEW_CANDIDATE", "NOT_YET_ADMISSIBLE"})
        chains = tuple(self.reconstruct_revision_chain(organization_id=organization_id, workspace_id=workspace_id, candidate=item.pin()) for item in terminal)
        evidence, decisions = set(), set()
        for item in terminal:
            evidence.update(_key(pin) for pin in item.origin.evidence)
            if isinstance(item.origin, ContextGraphOrigin) and item.origin.decision is not None:
                decisions.add(_key(item.origin.decision))
            if item.assessment:
                for pin in item.assessment.metric_refs:
                    if pin.artifact_kind == "decision_record": decisions.add(_key(pin))
        return EpistemicLearningProjectionV1(
            current_context=current_context, accepted_candidates=accepted_records,
            qualified_successor_inputs=tuple(pin for item in accepted for pin in item.successor_inputs),
            rejected_candidates=rejected, pending_candidates=pending,
            candidate_lineage=chains,
            evidence_pins=tuple(ArtifactPinV1(artifact_kind=k, artifact_ref=r, artifact_hash=h) for k,r,h in sorted(evidence)),
            decision_pins=tuple(ArtifactPinV1(artifact_kind=k, artifact_ref=r, artifact_hash=h) for k,r,h in sorted(decisions)),
            unresolved_questions=unresolved_questions,
        )

    def _scope(self, pin: ArtifactPinV1) -> tuple[str, str]:
        row = self._store._connection.execute(
            sql_query("SELECT organization_id,workspace_id,artifact_hash FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT organization_id,workspace_id,artifact_hash FROM ontology_build_artifacts WHERE artifact_ref=%s'),
            (pin.artifact_ref,),
        ).fetchone()
        if row is None:
            raise ChangeCandidateError("OBJECT_NOT_FOUND")
        if row[2] != pin.artifact_hash:
            raise ChangeCandidateError("HASH_MISMATCH")
        return str(row[0]), str(row[1])

    @staticmethod
    def _revision_values(previous: ChangeCandidate) -> dict[str, object]:
        return {
            "schema_version": "semantier.change_candidate.v1",
            "organization_id": previous.organization_id,
            "workspace_id": previous.workspace_id,
            "origin": previous.origin,
            "failure_class": previous.failure_class,
            "target": previous.target,
            "hypothesis": previous.hypothesis,
            "evaluation_basis": previous.evaluation_basis,
            "delta": previous.delta,
            "matched_cases": previous.matched_cases,
            "independent_holdout_cases": previous.independent_holdout_cases,
        }


__all__ = ["ChangeCandidateError", "ChangeCandidateService", "EpistemicLearningProjectionV1"]
