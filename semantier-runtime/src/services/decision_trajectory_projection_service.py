"""Deterministic builders for decision transition/trajectory projections.

The service consumes existing durable owners and emits content-addressed derived views plus the
machine-verifiable experiment-readiness receipt. It does not schedule learning or admit semantic
changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from contracts.decision_local_projection import DecisionLocalProjectionV1
from contracts.semantic_kernel import DecisionRecord, Evidence, WorkingContext
from contracts.experiment_integrity_snapshot import (
    ExperimentIntegritySnapshotV1,
    PrePilotAuthorityArtifactV1,
)
from contracts.decision_trajectory_projection import (
    DecisionStateEffectV1,
    DecisionTrajectoryProjectionV1,
    DecisionTransitionBasisV1,
    DecisionTransitionObservedV1,
    DecisionTransitionProjectionV1,
    DecisionTransitionSourceV1,
    InvocationUsageV1,
    TRAJECTORY_INSTRUMENTATION_CHECK_IDS,
    TrajectoryInstrumentationGateReceiptV1,
    TrajectoryInstrumentationCheckResultV1,
)
from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_quality import QualityEvaluationRunV1, QualityProfileV1
from eos.artifact_hashing import content_hash
from semantier.cognitive_context.models import DecisionStep
from semantier.cognitive_context.store import CognitiveContextStore
from storage.ontology_build_store import OntologyBuildStore
from services.tender_decision_projection_replay_gate_service import TenderDecisionProjectionReplayGateReceiptV1


@dataclass(frozen=True)
class TrajectoryInstrumentationVerification:
    """Verifier output; a receipt is available only when persisted checks all pass."""

    check_results: tuple[TrajectoryInstrumentationCheckResultV1, ...]
    _green_receipt: TrajectoryInstrumentationGateReceiptV1 | None

    @property
    def all_checks_pass(self) -> bool:
        return bool(self.check_results) and all(item.status == "PASS" for item in self.check_results)

    def to_green_receipt(self) -> TrajectoryInstrumentationGateReceiptV1 | None:
        return self._green_receipt if self.all_checks_pass else None

    def __iter__(self):
        return iter(self.check_results)

    def __len__(self) -> int:
        return len(self.check_results)


class DecisionTrajectoryProjectionService:
    @staticmethod
    def _decision_local_pin(projection: DecisionLocalProjectionV1) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="decision_local_projection",
            artifact_ref=projection.projection_ref,
            artifact_hash=projection.projection_hash,
        )

    @staticmethod
    def _decision_contract_pin(projection: DecisionLocalProjectionV1) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="decision_contract",
            artifact_ref=projection.decision_contract_ref,
            artifact_hash=projection.decision_contract_hash,
        )

    @staticmethod
    def _world_time(projection: DecisionLocalProjectionV1) -> str | None:
        for key in ("at_time", "start_time", "end_time", "governance_time"):
            value = projection.world_time_basis.get(key)
            if value:
                return value
        return None

    def create_transition(
        self,
        *,
        step: DecisionStep,
        decision_local_projection: DecisionLocalProjectionV1,
        logical_step_id: str,
        execution_attempt: ArtifactPinV1,
        attempt_index: int,
        lane: Literal["TASK", "RECOVERY", "DIAGNOSTIC"],
        trajectory_snapshot: ArtifactPinV1,
        projection_policy: ArtifactPinV1,
        producer_config: ArtifactPinV1,
        rendered_request: ArtifactPinV1,
        semantic_proposal: ArtifactPinV1 | None,
        deterministic_evaluation: ArtifactPinV1 | None,
        decision_record: ArtifactPinV1 | None,
        verification: ArtifactPinV1 | None,
        outcome: ArtifactPinV1 | None,
        execution_status: Literal["COMPLETE", "FAILED", "REJECTED", "PARTIAL", "INCONCLUSIVE"],
        outcome_provenance: Literal["OBSERVED", "REPLAYED", "SIMULATED", "INFERRED", "ASSUMED"] | None,
        state_effect_status: Literal["UNCHANGED", "CHANGED", "NOT_EXECUTED", "UNKNOWN"],
        invocation_usage: tuple[InvocationUsageV1, ...],
        started_at: datetime,
        observed_through: datetime,
        quality_evaluation: QualityEvaluationRunV1 | None = None,
        counterfactual_of: ArtifactPinV1 | None = None,
        counterfactual_kind: str | None = None,
        context_graph_after: ArtifactPinV1 | None = None,
        committed_delta: ArtifactPinV1 | None = None,
        commit_receipt: ArtifactPinV1 | None = None,
        action_receipts: tuple[ArtifactPinV1, ...] = (),
        tool_receipts: tuple[ArtifactPinV1, ...] = (),
        information_cutoff: str | None = None,
        closure_status: Literal["COMPLETE", "PARTIAL", "INVALID"] = "COMPLETE",
        missing_dependencies: tuple[str, ...] = (),
        resolution_of_transition: ArtifactPinV1 | None = None,
        successor_working_context: ArtifactPinV1 | None = None,
        successor_decision_local_projection: ArtifactPinV1 | None = None,
        language_evidence: ArtifactPinV1 | None = None,
        test_control_evidence: ArtifactPinV1 | None = None,
        semantic_result: Literal["SEMANTIC_TRUE", "SEMANTIC_FALSE", "SEMANTIC_UNKNOWN"] | None = None,
        producer_state: Literal["PRODUCER_OBSERVATION", "PRODUCER_ABSTENTION", "INVALID_OR_CENSORED_SCORE", "PROVIDER_FAILURE"] | None = None,
        semantic_error: bool = False,
    ) -> DecisionTransitionProjectionV1:
        if step.organization_id != decision_local_projection.organization_id:
            raise ValueError("DECISION_STEP_ORGANIZATION_MISMATCH")
        if step.workspace_id != decision_local_projection.workspace_id:
            raise ValueError("DECISION_STEP_WORKSPACE_MISMATCH")
        if outcome is not None and outcome.artifact_ref != step.outcome_ref:
            raise ValueError("DECISION_STEP_OUTCOME_MISMATCH")
        if not logical_step_id:
            raise ValueError("logical_step_id is required")
        if step.logical_step_id is None or step.execution_attempt_ref is None or step.execution_attempt_hash is None:
            raise ValueError("DECISION_STEP_TRAJECTORY_INSTRUMENTATION_REQUIRED")
        if step.logical_step_id != logical_step_id:
            raise ValueError("DECISION_STEP_LOGICAL_ID_MISMATCH")
        if (
            step.execution_attempt_ref != execution_attempt.artifact_ref
            or step.execution_attempt_hash != execution_attempt.artifact_hash
        ):
            raise ValueError("DECISION_STEP_EXECUTION_ATTEMPT_MISMATCH")
        if step.attempt_index != attempt_index or step.lane != lane:
            raise ValueError("DECISION_STEP_ATTEMPT_METADATA_MISMATCH")
        if any(logical_step_id not in usage.beneficiary_logical_step_ids for usage in invocation_usage):
            raise ValueError("every invocation usage must bind the transition logical_step_id")

        source = DecisionTransitionSourceV1(
            trajectory_ref=step.trajectory_ref,
            trajectory_snapshot=trajectory_snapshot,
            session_id=step.session_id,
            decision_step_ref=f"decision-step:{step.step_key}",
            logical_step_id=logical_step_id,
            execution_attempt=execution_attempt,
            attempt_index=attempt_index,
            lane=lane,
        )
        basis = DecisionTransitionBasisV1(
            quality_evaluation=quality_evaluation.pin() if quality_evaluation else None,
            context_graph_before=decision_local_projection.context_graph,
            working_context=decision_local_projection.working_context,
            decision_local_projection=self._decision_local_pin(decision_local_projection),
            decision_contract=self._decision_contract_pin(decision_local_projection),
            input_evidence=decision_local_projection.evidence,
            world_time=self._world_time(decision_local_projection),
            information_cutoff=information_cutoff,
            producer_config=producer_config,
            rendered_request=rendered_request,
        )
        observed = DecisionTransitionObservedV1(
            semantic_proposal=semantic_proposal,
            deterministic_evaluation=deterministic_evaluation,
            decision_record=decision_record,
            verification=verification,
            outcome=outcome,
            execution_status=execution_status,
            outcome_provenance=outcome_provenance,
            semantic_result=semantic_result,
            producer_state=producer_state,
            semantic_error=semantic_error,
            action_receipts=action_receipts,
            tool_receipts=tool_receipts,
        )
        state_effect = DecisionStateEffectV1(
            status=state_effect_status,
            context_graph_after=context_graph_after,
            committed_delta=committed_delta,
            commit_receipt=commit_receipt,
        )
        core = {
            "schema_version": "decision_transition_projection.v1",
            "organization_id": step.organization_id,
            "workspace_id": step.workspace_id,
            "projection_policy": projection_policy,
            "source": source,
            "basis": basis,
            "observed": observed,
            "state_effect": state_effect,
            "invocation_usage": invocation_usage,
            "started_at": started_at,
            "observed_through": observed_through,
            "closure_status": closure_status,
            "missing_dependencies": missing_dependencies,
            "resolution_of_transition": resolution_of_transition,
            "successor_working_context": successor_working_context,
            "successor_decision_local_projection": successor_decision_local_projection,
            "language_evidence": language_evidence,
            "test_control_evidence": test_control_evidence,
            "counterfactual_of": counterfactual_of,
            "counterfactual_kind": counterfactual_kind,
        }
        digest = content_hash(
            DecisionTransitionProjectionV1.model_construct(
                projection_ref="decision-transition-projection:pending",
                projection_hash="sha256:" + "0" * 64,
                **core,
            ).hash_payload()
        )
        return DecisionTransitionProjectionV1.model_validate(
            {
                **core,
                "projection_ref": f"decision-transition-projection:{digest}",
                "projection_hash": f"sha256:{digest}",
            }
        )

    def create_trajectory(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        trajectory_ref: str,
        branch_id: str,
        source_snapshot: ArtifactPinV1,
        projection_policy: ArtifactPinV1,
        transitions: tuple[DecisionTransitionProjectionV1, ...],
        coverage_scope: Literal["WHOLE_RUN", "DEPENDENCY_SLICE"] = "WHOLE_RUN",
        boundary_dependencies: tuple[ArtifactPinV1, ...] = (),
        task_definition: ArtifactPinV1 | None = None,
        experiment_integrity_snapshot: ExperimentIntegritySnapshotV1 | None = None,
        continuation_basis: ArtifactPinV1 | None = None,
        dependencies: tuple = (),
    ) -> DecisionTrajectoryProjectionV1:
        if not transitions:
            raise ValueError("trajectory requires at least one transition")
        for transition in transitions:
            if transition.organization_id != organization_id or transition.workspace_id != workspace_id:
                raise ValueError("TRAJECTORY_SCOPE_MISMATCH")
            if transition.source.trajectory_ref != trajectory_ref:
                raise ValueError("TRAJECTORY_REF_MISMATCH")

        seen_attempts: set[tuple[str, int]] = set()
        last_attempt: dict[str, int] = {}
        for transition in transitions:
            key = (transition.source.logical_step_id, transition.source.attempt_index)
            if key in seen_attempts:
                raise ValueError("duplicate logical_step_id/attempt_index")
            seen_attempts.add(key)
            previous = last_attempt.get(transition.source.logical_step_id)
            if previous is not None and transition.source.attempt_index <= previous:
                raise ValueError("trajectory attempts must be ordered monotonically per logical step")
            last_attempt[transition.source.logical_step_id] = transition.source.attempt_index

        core = {
            "schema_version": "decision_trajectory_projection.v1",
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "trajectory_ref": trajectory_ref,
            "branch_id": branch_id,
            "source_snapshot": source_snapshot,
            "projection_policy": projection_policy,
            "coverage_scope": coverage_scope,
            "boundary_dependencies": boundary_dependencies,
            "task_definition": task_definition,
            "experiment_integrity_snapshot": experiment_integrity_snapshot.pin() if experiment_integrity_snapshot else None,
            "transition_sequence": tuple(item.pin() for item in transitions),
            "continuation_basis": continuation_basis,
            "dependencies": dependencies,
        }
        draft = DecisionTrajectoryProjectionV1.model_construct(
            projection_ref="decision-trajectory-projection:pending",
            projection_hash="sha256:" + "0" * 64,
            **core,
        )
        digest = content_hash(draft.hash_payload())
        return DecisionTrajectoryProjectionV1.model_validate(
            {
                **core,
                "projection_ref": f"decision-trajectory-projection:{digest}",
                "projection_hash": f"sha256:{digest}",
            }
        )

    @staticmethod
    def create_continuation_basis(
        *, checkpoint, working_context: WorkingContext, graph,
        trajectory_ref: str, branch_id: str, pending_owner_refs: tuple[str, ...] = (),
    ):
        """Project existing pinned values; reconstruction separately verifies producers."""
        from contracts.decision_trajectory_projection import DependencyEdgeV1, SemanticContinuationBasisV1
        scope = dict(organization_id=working_context.organization_id,
                     workspace_id=working_context.workspace_id)
        if (checkpoint.organization_id, checkpoint.workspace_id) != tuple(scope.values()) or (graph.organization_id, graph.workspace_id) != tuple(scope.values()):
            raise ValueError("CONTINUATION_SCOPE_MISMATCH")
        if working_context.context_graph != graph.pin() or graph.admission_decision is None:
            raise ValueError("CONTINUATION_CONTEXT_BINDING_MISMATCH")
        cp_pin = ArtifactPinV1(artifact_kind="effective_context_checkpoint",
            artifact_ref=checkpoint.checkpoint_id, artifact_hash="sha256:"+str(checkpoint.checkpoint_hash))
        wc_pin = working_context.pin()
        edges = [DependencyEdgeV1(**scope,relation_class=kind,relation_type=relation,
                                 source=source,target=target) for kind,relation,source,target in (
            ("EXECUTION","CONTINUES",cp_pin,wc_pin),
            ("SEMANTIC","WORKING_CONTEXT_OF",wc_pin,graph.pin()),
            ("AUTHORITY","AUTHORIZED_BY",wc_pin,graph.admission_decision),
            *(("EPISTEMIC","SUPPORTED_BY",wc_pin,pin) for pin in working_context.evidence),
        )]
        return SemanticContinuationBasisV1.create(**scope, trajectory_ref=trajectory_ref,
            branch_id=branch_id,source_session_id=checkpoint.session_id,checkpoint=cp_pin,
            working_context=wc_pin,context_graph=graph.pin(),evidence=working_context.evidence,
            admission_decision=graph.admission_decision,dependencies=tuple(edges),
            unresolved_questions=working_context.unresolved_questions,
            pending_owner_refs=pending_owner_refs)

    @staticmethod
    def reconstruct_continuation(
        *, basis_pin: ArtifactPinV1, projection_store: CognitiveContextStore,
        checkpoint_service, build_store: OntologyBuildStore, authority_resolver,
        actor_context, trajectory_ref: str, branch_id: str,
    ) -> WorkingContext:
        """Resolve frozen semantic state; native owners still authorize each action."""
        from contracts.decision_trajectory_projection import SemanticContinuationBasisV1
        from contracts.semantic_kernel import ContextGraph, Ontology, validate_context_admission_transition
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from semantier.effective_context.service import EffectiveContextCheckpointService
        if not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("CONTINUATION_GOVERNED_AUTHORITY_REQUIRED")
        if not isinstance(checkpoint_service, EffectiveContextCheckpointService):
            raise TypeError("CONTINUATION_CHECKPOINT_OWNER_REQUIRED")
        authority_resolver.require_current_actor(actor_context=actor_context)
        org, ws = actor_context.organization_id, str(actor_context.workspace_id or "")
        if basis_pin.artifact_kind != "semantic_continuation_basis":
            raise ValueError("CONTINUATION_BASIS_KIND_MISMATCH")
        basis = projection_store.get_derived_projection(basis_pin.artifact_ref,
            organization_id=org, workspace_id=ws, contract_type=SemanticContinuationBasisV1)
        if basis.pin() != basis_pin:
            raise ValueError("CONTINUATION_BASIS_HASH_MISMATCH")
        if (basis.trajectory_ref, basis.branch_id) != (trajectory_ref, branch_id):
            raise ValueError("CONTINUATION_BRANCH_MISMATCH")
        cp = checkpoint_service.store.get_checkpoint(org, ws, basis.checkpoint.artifact_ref)
        if ("sha256:" + str(cp.checkpoint_hash)) != basis.checkpoint.artifact_hash or cp.session_id != basis.source_session_id:
            raise ValueError("CONTINUATION_CHECKPOINT_MISMATCH")
        if cp.verification_status != "verified" or not checkpoint_service.verify_checkpoint(org, ws, cp.checkpoint_id):
            raise ValueError("CONTINUATION_CHECKPOINT_UNVERIFIED")
        historical_authority = checkpoint_service.store.get_payload(org, ws, cp.authority_pins_ref)
        if historical_authority.get("historical_admission_proof_ref") != basis.admission_decision.artifact_ref:
            raise ValueError("CONTINUATION_CHECKPOINT_AUTHORITY_MISMATCH")
        knowledge = checkpoint_service.store.get_payload(org, ws, cp.knowledge_state_ref)
        if (knowledge.get("trajectory_ref"), knowledge.get("branch_id")) != (basis.trajectory_ref, basis.branch_id):
            raise ValueError("CONTINUATION_CHECKPOINT_BRANCH_MISMATCH")
        if knowledge.get("working_context") != basis.working_context.model_dump(mode="json"):
            raise ValueError("CONTINUATION_CHECKPOINT_SEMANTIC_BINDING_MISMATCH")
        # No transcript or live head/memory lookup participates in reconstruction.
        def exact(pin, model):
            value = build_store.get(organization_id=org, workspace_id=ws,
                                    artifact_ref=pin.artifact_ref, contract_type=model)
            if value is None or value.pin() != pin:
                raise ValueError("CONTINUATION_OWNER_PIN_MISMATCH")
            return value
        wc = projection_store.get_derived_projection(basis.working_context.artifact_ref,
            organization_id=org, workspace_id=ws, contract_type=WorkingContext)
        if wc.pin() != basis.working_context:
            raise ValueError("CONTINUATION_OWNER_PIN_MISMATCH")
        graph = exact(basis.context_graph, ContextGraph)
        decision = exact(basis.admission_decision, DecisionRecord)
        if graph.predecessor is None:
            raise ValueError("CONTINUATION_ADMISSION_PREDECESSOR_REQUIRED")
        proposed = exact(graph.predecessor, ContextGraph)
        validate_context_admission_transition(proposed, decision, graph)
        exact(wc.ontology, Ontology)
        for pin in wc.evidence:
            exact(pin, Evidence)
        if (wc.context_graph != graph.pin() or wc.ontology != graph.ontology
                or wc.evidence != basis.evidence or graph.admission_decision != basis.admission_decision
                or wc.unresolved_questions != basis.unresolved_questions):
            raise ValueError("CONTINUATION_SEMANTIC_CLOSURE_MISMATCH")
        if wc.responsibility.responsibility_bearer_id != actor_context.actor_id:
            raise ValueError("CONTINUATION_RESPONSIBILITY_MISMATCH")
        from services.working_context_projection_service import WorkingContextProjectionService
        checked = WorkingContextProjectionService(build_store=build_store,
            authority_resolver=authority_resolver,
            runtime_actor_id=wc.responsibility.delegated_actor_id).project(
                actor_context=actor_context,context=graph.pin(),task_ref=wc.task_ref,task_goal=wc.task_goal,
                selected_node_ids=wc.selected_node_ids,selected_edge_ids=wc.selected_edge_ids,
                selected_assertion_ids=wc.selected_assertion_ids,
                delegated_operation=wc.responsibility.delegated_operation,
                commitment_mode=wc.responsibility.commitment_mode,formed_at=wc.formed_at)
        if checked.pin() != wc.pin():
            raise ValueError("CONTINUATION_WORKING_CONTEXT_REFORMATION_MISMATCH")
        return wc

    @staticmethod
    def resolve_meeting_continuation(*, ctx, projection_store, checkpoint_service,
                                    build_store, authority_resolver, basis_pin,
                                    comparison, read_mode, native_state):
        """Exact native-consumer input resolution; task goal remains a request."""
        import json
        from pathlib import Path
        from contracts.decision_trajectory_projection import SemanticContinuationBasisV1
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.workflow_execution import hash_value, verify_meeting_binding
        from semantier.effective_context.service import EffectiveContextCheckpointService
        if not ctx or not ctx.authenticated or not ctx.user_id:
            raise ValueError("CONTINUATION_AUTHENTICATED_CONTEXT_REQUIRED")
        if (not isinstance(projection_store, CognitiveContextStore)
                or not isinstance(build_store, OntologyBuildStore)
                or not isinstance(checkpoint_service, EffectiveContextCheckpointService)):
            raise TypeError("CONTINUATION_OWNER_STORES_REQUIRED")
        root = Path(ctx.workspace_root).resolve()
        for path in (projection_store.db_path, build_store._path, checkpoint_service.store._db_path):
            if not path or not Path(path).resolve().is_relative_to(root):
                raise ValueError("CONTINUATION_PATH_OUTSIDE_WORKSPACE")
        if read_mode not in ("original", "reconstructed"):
            raise ValueError("CONTINUATION_READ_MODE_INVALID")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_id=ctx.user_id, session_id="native:continuation")
        authority_resolver.require_current_actor(actor_context=actor)
        basis = projection_store.get_derived_projection(basis_pin.artifact_ref,
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            contract_type=SemanticContinuationBasisV1)
        if basis.pin() != basis_pin:
            raise ValueError("CONTINUATION_BASIS_HASH_MISMATCH")
        checked = DecisionTrajectoryProjectionService.reconstruct_continuation(
            basis_pin=basis_pin, projection_store=projection_store,
            checkpoint_service=checkpoint_service, build_store=build_store,
            authority_resolver=authority_resolver, actor_context=actor,
            trajectory_ref=basis.trajectory_ref, branch_id=basis.branch_id)
        # A direct read remains distinct from reconstruction. Common exact
        # verification does not substitute for the actual selected consumer read.
        if read_mode == "original":
            wc = projection_store.get_derived_projection(basis.working_context.artifact_ref,
                organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                contract_type=WorkingContext)
        else:
            wc = checked
        if wc.pin() != checked.pin():
            raise ValueError("CONTINUATION_CONSUMED_BASIS_MISMATCH")
        case = native_state["case"]
        if (case["organization_id"], case["workspace_id"], case["creator_user_id"]) != (
                ctx.organization_id, ctx.workspace_id, ctx.user_id):
            raise ValueError("CONTINUATION_NATIVE_AUTHORITY_MISMATCH")
        binding = json.loads(case["workflow_binding_json"])
        verify_meeting_binding(binding, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        if basis.pending_owner_refs != (case["negotiation_id"],):
            raise ValueError("CONTINUATION_CASE_MISMATCH")
        if wc.responsibility.delegated_operation != "CONTINUE_EXISTING_MEETING":
            raise ValueError("CONTINUATION_OPERATION_MISMATCH")
        command = json.loads(wc.task_goal)
        if set(command) != {"case_id", "boundary", "proposal", "event", "expected_slot"}:
            raise ValueError("CONTINUATION_REQUEST_SCHEMA_MISMATCH")
        if command["case_id"] != case["negotiation_id"] or command["event"] != case["event_id"]:
            raise ValueError("CONTINUATION_CASE_MISMATCH")
        required = {"before_proposal":"pending_decliner_input", "before_vote":"collecting_votes", "before_finalize":"consented"}
        if required.get(command["boundary"]) != case["status"]:
            raise ValueError("CONTINUATION_BOUNDARY_MISMATCH")
        owner_evidence = [build_store.get(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, artifact_ref=pin.artifact_ref,
            contract_type=Evidence) for pin in wc.evidence]
        if not any(e.source_ref == "meeting-native:" + case["negotiation_id"]
                   and e.source_hash == hash_value(native_state) for e in owner_evidence):
            raise ValueError("CONTINUATION_NATIVE_PREDECESSOR_MISMATCH")
        if set(comparison) != {"contract", "contract_hash", "pair_id", "seed", "expected_basis"}:
            raise ValueError("CONTINUATION_COMPARISON_SCHEMA_MISMATCH")
        contract = comparison["contract"]
        if (contract.get("schema_version") != "producer-bound-continuation-comparison.v1"
                or binding["definition"]["workflowVersion"] != contract.get("workflow_version")):
            raise ValueError("CONTINUATION_COMPARISON_DEFINITION_MISMATCH")
        if comparison["contract_hash"] != hash_value(contract):
            raise ValueError("CONTINUATION_COMPARISON_CONTRACT_MISMATCH")
        if comparison["expected_basis"] != basis.pin().model_dump(mode="json"):
            raise ValueError("CONTINUATION_DECLARED_BASIS_MISMATCH")
        if (comparison["seed"] not in contract["seeds"]
                or command["boundary"] not in contract["boundaries"]
                or comparison["pair_id"] != case["event_id"]
                or {"pair_id":case["event_id"], "boundary":command["boundary"], "seed":comparison["seed"]} not in contract["runs"]):
            raise ValueError("CONTINUATION_COMPARISON_RUN_MISMATCH")
        consumed = {"schema_version":"meeting_semantic_consumption.v1",
            "reader_version":"meeting-semantic-consumer/v1", "read_mode":read_mode,
            "pair_id":comparison["pair_id"], "run_id":comparison["pair_id"] + ":" + read_mode,
            "seed":comparison["seed"], "boundary":command["boundary"],
            "comparison_contract_hash":comparison["contract_hash"],
            "basis":basis.pin().model_dump(mode="json"),
            "checkpoint":basis.checkpoint.model_dump(mode="json"),
            "working_context":wc.pin().model_dump(mode="json"),
            "consumed_request":command, "consumed_request_hash":hash_value(command),
            "native_predecessor_hash":hash_value(native_state),
            "workflow_binding_hash":hash_value(binding), "instance_id":case["negotiation_id"],
            "organization_id":ctx.organization_id, "workspace_id":ctx.workspace_id,
            "actor_id":ctx.user_id}
        return command, consumed

    def create_gate_receipt(
        self,
        **_caller_assertions,
    ) -> TrajectoryInstrumentationGateReceiptV1:
        raise RuntimeError("ASSERTION_BASED_TRAJECTORY_RECEIPTS_DISABLED_USE_VERIFIER")

    @staticmethod
    def assert_trajectory_instrumentation_green(
        receipt: TrajectoryInstrumentationGateReceiptV1 | None,
        *,
        runtime_contract: ArtifactPinV1,
        scorer_contract: ArtifactPinV1,
        collector_contract: ArtifactPinV1,
    ) -> None:
        if receipt is None or receipt.status != "GREEN":
            raise RuntimeError("TRAJECTORY_INSTRUMENTATION_GREEN_REQUIRED")
        if (receipt.runtime_contract, receipt.scorer_contract, receipt.collector_contract) != (
            runtime_contract, scorer_contract, collector_contract
        ):
            raise RuntimeError("TRAJECTORY_INSTRUMENTATION_GREEN_STALE")
        if any(result.status != "PASS" for result in receipt.check_results):
            raise RuntimeError("TRAJECTORY_INSTRUMENTATION_GREEN_INCOMPLETE")


class TrajectoryInstrumentationVerifier:
    """Derive trajectory admissibility checks from scope-bound persisted projections."""

    def __init__(self, store: CognitiveContextStore, ontology_store: OntologyBuildStore | None = None) -> None:
        self._store = store
        self._ontology_store = ontology_store

    def _verify_experiment_snapshot(
        self,
        trajectory: DecisionTrajectoryProjectionV1,
        *,
        required_stage: str = "PRE_PILOT",
        runtime_contract: ArtifactPinV1 | None = None,
        scorer_contract: ArtifactPinV1 | None = None,
        collector_contract: ArtifactPinV1 | None = None,
    ) -> tuple[bool, tuple[ArtifactPinV1, ...], str]:
        pin = trajectory.experiment_integrity_snapshot
        if pin is None:
            return False, (), "experiment integrity snapshot pin missing"
        try:
            snapshot = self._store.get_derived_projection(
                pin.artifact_ref,
                organization_id=trajectory.organization_id,
                workspace_id=trajectory.workspace_id,
                contract_type=ExperimentIntegritySnapshotV1,
            )
        except (KeyError, ValueError) as exc:
            return False, (pin,), f"experiment integrity snapshot unresolved: {type(exc).__name__}"
        if (snapshot.snapshot_hash, snapshot.pin()) != (pin.artifact_hash, pin):
            return False, (pin,), "trajectory experiment integrity pin differs from persisted snapshot"
        child_pins = snapshot.bound_pins
        expected_contracts = (runtime_contract, scorer_contract, collector_contract)
        snapshot_contracts = (snapshot.runtime_contract, snapshot.scorer_contract, snapshot.collector_contract)
        contracts_match = all(expected is None or expected == actual
                              for expected, actual in zip(expected_contracts, snapshot_contracts))
        valid = (
            snapshot.integrity_stage == required_stage
            and contracts_match
            and all(p.artifact_ref and p.artifact_hash.startswith("sha256:") for p in child_pins)
            and snapshot.verify_resolved_authorities()
        )
        if valid:
            try:
                from services.pre_pilot_authorities import validate_pre_pilot_authority_sources

                for child_pin in child_pins:
                    if child_pin.artifact_kind == "quality_profile":
                        row = next(row for row in snapshot.authority_payloads
                                   if row.get("artifact_ref") == child_pin.artifact_ref)
                        profile = QualityProfileV1.model_validate(row["payload"])
                        if profile.pin() != child_pin:
                            raise ValueError("quality profile mismatch")
                        continue
                    child = self._store.get_derived_projection(
                        child_pin.artifact_ref,
                        organization_id=trajectory.organization_id,
                        workspace_id=trajectory.workspace_id,
                        contract_type=PrePilotAuthorityArtifactV1,
                    )
                    if child.pin() != child_pin:
                        valid = False
                        break
                    frozen_row = next(
                        row for row in snapshot.authority_payloads
                        if (row.get("artifact_kind"), row.get("artifact_ref"), row.get("artifact_hash"))
                        == (child_pin.artifact_kind, child_pin.artifact_ref, child_pin.artifact_hash)
                    )
                    if frozen_row.get("payload") != child.payload:
                        valid = False
                        break
                    if not validate_pre_pilot_authority_sources((child,)):
                        valid = False
                        break
            except (KeyError, StopIteration, ValueError, TypeError):
                valid = False
        if valid:
            try:
                if self._ontology_store is None:
                    raise ValueError("quality store unavailable")
                for transition_pin in trajectory.transition_sequence:
                    transition = self._store.get_derived_projection(transition_pin.artifact_ref,
                        organization_id=trajectory.organization_id, workspace_id=trajectory.workspace_id,
                        contract_type=DecisionTransitionProjectionV1)
                    if transition.pin() != transition_pin or transition.basis.quality_evaluation is None:
                        raise ValueError("transition quality evaluation missing")
                    if transition.counterfactual_of is not None:
                        baseline = self._store.get_derived_projection(transition.counterfactual_of.artifact_ref,
                            organization_id=trajectory.organization_id, workspace_id=trajectory.workspace_id,
                            contract_type=DecisionTransitionProjectionV1)
                        if baseline.pin() != transition.counterfactual_of or baseline.counterfactual_of is not None:
                            raise ValueError("counterfactual baseline must be an exact unperturbed transition")
                        if baseline.basis.quality_evaluation != transition.basis.quality_evaluation:
                            raise ValueError("counterfactual cannot substitute baseline quality evidence")
                        transition = baseline
                    quality_pin = transition.basis.quality_evaluation
                    run = self._ontology_store.get(organization_id=trajectory.organization_id,
                        workspace_id=trajectory.workspace_id, artifact_ref=quality_pin.artifact_ref,
                        contract_type=QualityEvaluationRunV1)
                    dlp_pin = transition.basis.decision_local_projection
                    dlp = self._store.get_derived_projection(dlp_pin.artifact_ref,
                        organization_id=trajectory.organization_id, workspace_id=trajectory.workspace_id,
                        contract_type=DecisionLocalProjectionV1)
                    if run.pin() != quality_pin or self._store_projection_pin(dlp) != dlp_pin:
                        raise ValueError("quality evaluation or DLP identity mismatch")
                    if transition.basis.context_graph_before != dlp.context_graph:
                        raise ValueError("transition graph differs from DLP")
                    snapshot.require_quality_basis(evaluation=run, decision_local_projection=dlp)
            except (KeyError, ValueError, TypeError, AttributeError):
                valid = False
        reason = "" if valid else f"snapshot is stale for {required_stage} or child authority payload is absent/unresolved/mismatched"
        return valid, (pin, *child_pins), reason

    def verify(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        trajectory_ref: str,
        trajectory_hash: str,
        related_trajectories: tuple[ArtifactPinV1, ...] = (),
        runtime_contract: ArtifactPinV1,
        scorer_contract: ArtifactPinV1,
        collector_contract: ArtifactPinV1,
        validated_at: datetime,
        required_stage: str = "PRE_PILOT",
    ) -> TrajectoryInstrumentationVerification:
        """No pass/failure/language/restart booleans are accepted; facts are reconstructed here."""
        trajectory = self._store.get_derived_projection(
            trajectory_ref,
            organization_id=organization_id,
            workspace_id=workspace_id,
            contract_type=DecisionTrajectoryProjectionV1,
        )
        if trajectory.projection_hash != trajectory_hash:
            raise ValueError("TRAJECTORY_HASH_MISMATCH")
        trajectories = [trajectory]
        for trajectory_pin in related_trajectories:
            related = self._store.get_derived_projection(
                trajectory_pin.artifact_ref, organization_id=organization_id,
                workspace_id=workspace_id, contract_type=DecisionTrajectoryProjectionV1,
            )
            if related.pin() != trajectory_pin:
                raise ValueError("RELATED_TRAJECTORY_HASH_MISMATCH")
            trajectories.append(related)
        if len({item.projection_ref for item in trajectories}) != len(trajectories):
            raise ValueError("DUPLICATE_TRAJECTORY_REF")
        parent_by_transition: dict[str, DecisionTrajectoryProjectionV1] = {}
        transitions_list = []
        for parent in trajectories:
            for transition_pin in parent.transition_sequence:
                transition = self._store.get_derived_projection(
                    transition_pin.artifact_ref, organization_id=organization_id,
                    workspace_id=workspace_id, contract_type=DecisionTransitionProjectionV1,
                )
                parent_by_transition[transition.projection_ref] = parent
                transitions_list.append(transition)
        transitions = tuple(transitions_list)
        transition_pins_resolve_exactly = all(
            transition.pin() == pin
            and transition.source.trajectory_ref == parent_by_transition[transition.projection_ref].trajectory_ref
            for parent in trajectories for pin in parent.transition_sequence
            for transition in (next(item for item in transitions if item.projection_ref == pin.artifact_ref),)
        )
        results: list[TrajectoryInstrumentationCheckResultV1] = []

        def add(check_id: str, ok: bool, refs: tuple[ArtifactPinV1, ...], reason: str) -> None:
            results.append(TrajectoryInstrumentationCheckResultV1(
                check_id=check_id,
                status="PASS" if ok else "FAIL",
                evidence_refs=refs,
                failure_reasons=() if ok else (reason,),
            ))

        refs = tuple(item.pin() for item in transitions)
        exact = all(item.organization_id == organization_id and item.workspace_id == workspace_id
                    and item.source.trajectory_ref == parent_by_transition[item.projection_ref].trajectory_ref for item in transitions)
        dlps: dict[str, DecisionLocalProjectionV1] = {}
        dlp_support_refs: list[ArtifactPinV1] = []
        exact_dlp = exact and self._ontology_store is not None
        for transition in transitions:
            try:
                pin = transition.basis.decision_local_projection
                dlp = self._store.get_derived_projection(
                    pin.artifact_ref, organization_id=organization_id,
                    workspace_id=workspace_id, contract_type=DecisionLocalProjectionV1,
                )
                wc = self._store.get_derived_projection(
                    dlp.working_context.artifact_ref, organization_id=organization_id,
                    workspace_id=workspace_id, contract_type=WorkingContext,
                )
                graph = self._ontology_store.get(
                    organization_id=organization_id, workspace_id=workspace_id,
                    artifact_ref=dlp.context_graph.artifact_ref,
                    contract_type=__import__("contracts.semantic_kernel", fromlist=["ContextGraph"]).ContextGraph,
                ) if self._ontology_store is not None else None
                valid = (
                    self._store_projection_pin(dlp) == pin
                    and (dlp.organization_id, dlp.workspace_id) == (organization_id, workspace_id)
                    and wc.pin() == dlp.working_context == transition.basis.working_context
                    and wc.context_graph == dlp.context_graph == transition.basis.context_graph_before
                    and transition.basis.decision_contract is not None
                    and (transition.basis.decision_contract.artifact_ref,
                         transition.basis.decision_contract.artifact_hash)
                    == (dlp.decision_contract_ref, dlp.decision_contract_hash)
                    and transition.basis.input_evidence == dlp.evidence
                    and graph is not None and graph.pin() == dlp.context_graph
                )
                exact_dlp = exact_dlp and valid
                dlps[transition.projection_ref] = dlp
                dlp_support_refs.extend((pin, dlp.working_context, dlp.context_graph,
                                         transition.basis.decision_contract, *dlp.evidence))
            except (KeyError, ValueError, PermissionError, TypeError):
                exact_dlp = False
        add("TIG-01", exact_dlp, tuple(dict.fromkeys((*refs, *(pin for pin in dlp_support_refs if pin is not None)))), "persisted exact DecisionLocalProjection/WorkingContext/contract/context binding absent or mismatched")
        joins = bool(self._ontology_store is not None)
        lifecycle_refs: list[ArtifactPinV1] = []
        for item in transitions:
            if item.observed.execution_status != "COMPLETE":
                continue
            try:
                dlp = dlps[item.projection_ref]
                proposal = self._resolve_evidence(item.observed.semantic_proposal, organization_id, workspace_id)
                evaluation = self._resolve_evidence(item.observed.deterministic_evaluation, organization_id, workspace_id)
                verification = self._resolve_verification(item.observed.verification, organization_id, workspace_id)
                outcome = self._resolve_evidence(item.observed.outcome, organization_id, workspace_id)
                decision = self._ontology_store.get(
                    organization_id=organization_id, workspace_id=workspace_id,
                    artifact_ref=item.observed.decision_record.artifact_ref,
                    contract_type=DecisionRecord,
                )
                valid = all(x is not None for x in (proposal, evaluation, verification, outcome, decision))
                valid = valid and (proposal.evidence_role == "test_control_producer_observation")
                valid = valid and (evaluation.evidence_role == "deterministic_evaluation")
                valid = valid and verification.status == "GREEN"
                valid = valid and verification.persisted_projection == self._store_projection_pin(dlp)
                valid = valid and verification.replayed_projection == self._store_projection_pin(dlp)
                valid = valid and verification.working_context == dlp.working_context
                valid = valid and set(verification.checks) == {
                    "PERSISTED_ARTIFACTS_VALID", "WORKING_CONTEXT_RECONSTRUCTED",
                    "HISTORICAL_AUTHORITY_REPLAYED", "PROJECTION_IDENTITY_EQUAL",
                }
                valid = valid and (outcome.evidence_role == "outcome")
                valid = valid and decision.pin() == item.observed.decision_record
                valid = valid and decision.subject.artifact_ref == dlp.projection_ref
                valid = valid and decision.subject.artifact_hash == dlp.projection_hash
                valid = valid and decision.context == dlp.context_graph
                valid = valid and decision.outcome_observation == item.observed.outcome
                valid = valid and decision.result_code == item.observed.semantic_result
                valid = valid and any(
                    item.observed.semantic_proposal in invocation.inputs
                    and item.observed.deterministic_evaluation in invocation.inputs
                    and item.observed.verification in invocation.inputs
                    for invocation in decision.reasoning.invocations
                )
                joins = joins and valid
                lifecycle_refs.extend(pin for pin in (item.observed.semantic_proposal,
                    item.observed.deterministic_evaluation, item.observed.verification,
                    item.observed.decision_record, item.observed.outcome) if pin is not None)
            except (KeyError, ValueError, PermissionError, TypeError, AttributeError):
                joins = False
        add("TIG-02", joins, tuple(dict.fromkeys((*refs, *lifecycle_refs))), "persisted typed producer/evaluation/verification/DecisionRecord/outcome join absent or mismatched")
        coherent = True
        attempt_refs: list[ArtifactPinV1] = []
        for item in transitions:
            try:
                step = self._store.get_decision_step_by_key(
                    item.source.decision_step_ref.removeprefix("decision-step:"),
                    organization_id=organization_id, workspace_id=workspace_id,
                )
                coherent = coherent and step.trajectory_ref == parent_by_transition[item.projection_ref].trajectory_ref
                coherent = coherent and step.logical_step_id == item.source.logical_step_id
                coherent = coherent and step.execution_attempt_ref == item.source.execution_attempt.artifact_ref
                coherent = coherent and step.execution_attempt_hash == item.source.execution_attempt.artifact_hash
                coherent = coherent and step.attempt_index == item.source.attempt_index and step.lane == item.source.lane
                coherent = coherent and self._resolve_evidence(
                    item.source.execution_attempt, organization_id, workspace_id
                ) is not None
                attempt_refs.append(item.source.execution_attempt)
            except (KeyError, ValueError, PermissionError, TypeError):
                coherent = False
        add("TIG-03", coherent, tuple(dict.fromkeys((*refs, *attempt_refs))), "attempt identity or lane is missing/incoherent")
        state_kinds = {item.observed.semantic_result for item in transitions}
        producer_kinds = {item.observed.producer_state for item in transitions}
        control_evidence_ok = all(
            item.test_control_evidence is not None
            and (control := self._resolve_evidence(item.test_control_evidence, organization_id, workspace_id)) is not None
            and control.evidence_role == "test_control_producer_observation"
            for item in transitions if item.observed.producer_state is not None
        )
        add("TIG-04", {"SEMANTIC_TRUE", "SEMANTIC_FALSE", "SEMANTIC_UNKNOWN"}.issubset(state_kinds) and {"PRODUCER_OBSERVATION", "PRODUCER_ABSTENTION", "INVALID_OR_CENSORED_SCORE", "PROVIDER_FAILURE"}.issubset(producer_kinds) and any(item.observed.execution_status == "REJECTED" for item in transitions) and any(item.observed.semantic_error for item in transitions) and control_evidence_ok, tuple(dict.fromkeys((*refs, *(t.test_control_evidence for t in transitions if t.test_control_evidence is not None)))), "semantic/provider/execution/error states are not separately represented or controlled observations lack typed evidence")
        by_ref = {item.projection_ref: item for item in transitions}
        resolution_languages: set[str] = set()
        resolution_ok = False
        for item in transitions:
            if item.resolution_of_transition is None or item.resolution_of_transition.artifact_ref not in by_ref:
                continue
            prior = by_ref[item.resolution_of_transition.artifact_ref]
            successor_result = item.observed.semantic_result
            prior_language = self._language_for_transition(prior, organization_id, workspace_id)
            if prior_language is not None and successor_result in {"SEMANTIC_TRUE", "SEMANTIC_FALSE"}:
                resolution_languages.add(prior_language)
            resolution_ok = resolution_ok or (
                prior.pin() == item.resolution_of_transition
                and prior.observed.semantic_result == "SEMANTIC_UNKNOWN"
                and successor_result in {"SEMANTIC_TRUE", "SEMANTIC_FALSE"}
                and item.successor_working_context is not None
                and item.successor_decision_local_projection is not None
                and item.successor_working_context == item.basis.working_context
                and item.successor_decision_local_projection == item.basis.decision_local_projection
                and prior.basis.working_context != item.successor_working_context
                and prior.basis.decision_local_projection != item.successor_decision_local_projection
            )
        resolution_ok = resolution_ok and resolution_languages == {"en", "zh"}
        successor_refs = tuple(pin for t in transitions for pin in (t.resolution_of_transition,
            t.successor_working_context, t.successor_decision_local_projection) if pin is not None)
        add("TIG-05", resolution_ok, tuple(dict.fromkeys((*refs, *successor_refs))), "explicit retained C-to-A/B predecessor and successor bindings absent")
        failed = [t for t in transitions if t.observed.execution_status in {"FAILED", "REJECTED", "PARTIAL", "INCONCLUSIVE"}]
        recovered = any(any(ok.source.logical_step_id == bad.source.logical_step_id and ok.source.attempt_index > bad.source.attempt_index and ok.observed.execution_status == "COMPLETE" for ok in transitions) for bad in failed)
        attempt_pair_refs = tuple(pin for t in failed + [item for item in transitions
            if item.source.logical_step_id in {bad.source.logical_step_id for bad in failed}
            and item.observed.execution_status == "COMPLETE"] for pin in (t.pin(), t.source.execution_attempt))
        add("TIG-06", bool(failed) and recovered, tuple(dict.fromkeys((*refs, *attempt_pair_refs))), "durable failed/recovery attempt pair absent")
        usage_rows = [u for t in transitions for u in t.invocation_usage]
        usage_ok = all(u.measurement_evidence is not None for u in usage_rows if any((u.model_visible_input_tokens, u.generated_output_tokens, u.reported_reasoning_tokens, u.cached_input_tokens, u.physical_prefill_tokens, u.wall_latency_ms)))
        measurement_refs = tuple(u.measurement_evidence for u in usage_rows if u.measurement_evidence is not None)
        add("TIG-07", usage_ok, tuple(dict.fromkeys((*refs, *measurement_refs))), "usage metrics lack observed measurement evidence")
        add(
            "TIG-08",
            all(item.pin().artifact_ref and item.organization_id == organization_id
                and item.workspace_id == workspace_id for item in trajectories)
            and trajectory.pin().artifact_ref == trajectory_ref
            and trajectory.pin().artifact_hash == trajectory_hash
            and transition_pins_resolve_exactly,
            tuple(item.pin() for item in trajectories) + refs,
            "persisted trajectory or one of its exact transition pins could not be reconstructed",
        )
        add("TIG-09", all("transcript" not in item.model_dump(mode="json") for item in transitions), refs, "canonical trajectory contains transcript state")
        unknown = any(t.observed.semantic_result == "SEMANTIC_UNKNOWN" and t.closure_status == "PARTIAL" and t.missing_dependencies for t in transitions)
        no_structure_attribution = all(not t.observed.semantic_error or t.test_control_evidence is not None for t in transitions if t.observed.semantic_result == "SEMANTIC_UNKNOWN")
        unknown_refs = tuple(pin for t in transitions if t.observed.semantic_result == "SEMANTIC_UNKNOWN"
            for pin in (t.pin(), t.basis.decision_local_projection, *t.basis.input_evidence))
        add("TIG-10", unknown and no_structure_attribution, tuple(dict.fromkeys((*refs, *unknown_refs))), "UNKNOWN lacks material-dependency evidence or semantic-error attribution boundary is ambiguous")
        integrity_checks = tuple(self._verify_experiment_snapshot(
            item, required_stage=required_stage, runtime_contract=runtime_contract,
            scorer_contract=scorer_contract, collector_contract=collector_contract,
        ) for item in trajectories)
        integrity_ok = all(item[0] for item in integrity_checks)
        integrity_refs = tuple(pin for item in integrity_checks for pin in item[1])
        integrity_reason = "; ".join(item[2] for item in integrity_checks if item[2])
        add("TIG-11", integrity_ok, integrity_refs, integrity_reason)
        add("TIG-12", True, (trajectory.pin(),), "receipt records admissibility/binding/replay only; it asserts no correctness")
        check_results = tuple(results)
        verification_inputs = {
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "trajectory_ref": trajectory_ref,
            "trajectory_hash": trajectory_hash,
            "related_trajectories": related_trajectories,
            "runtime_contract": runtime_contract,
            "scorer_contract": scorer_contract,
            "collector_contract": collector_contract,
            "validated_at": validated_at,
        }
        receipt = self._build_green_receipt(verification_inputs, check_results) if all(
            item.status == "PASS" for item in check_results
        ) else None
        return TrajectoryInstrumentationVerification(check_results, receipt)

    def _build_green_receipt(
        self,
        kwargs: dict,
        results: tuple[TrajectoryInstrumentationCheckResultV1, ...],
    ) -> TrajectoryInstrumentationGateReceiptV1:
        trajectory = self._store.get_derived_projection(kwargs["trajectory_ref"], organization_id=kwargs["organization_id"], workspace_id=kwargs["workspace_id"], contract_type=DecisionTrajectoryProjectionV1)
        related_trajectories = tuple(self._store.get_derived_projection(
            pin.artifact_ref, organization_id=kwargs["organization_id"], workspace_id=kwargs["workspace_id"],
            contract_type=DecisionTrajectoryProjectionV1,
        ) for pin in kwargs.get("related_trajectories", ()))
        trajectories = (trajectory, *related_trajectories)
        transitions = tuple(self._store.get_derived_projection(pin.artifact_ref,
            organization_id=kwargs["organization_id"], workspace_id=kwargs["workspace_id"],
            contract_type=DecisionTransitionProjectionV1) for current in trajectories for pin in current.transition_sequence)
        languages_seen: set[str] = set()
        for transition in transitions:
            language_source = self._resolve_evidence(
                transition.language_evidence, kwargs["organization_id"], kwargs["workspace_id"]
            )
            if language_source is None:
                continue
            text = language_source.selector.exact_quote
            has_cjk = any("\u3400" <= character <= "\u9fff" for character in text)
            languages_seen.add("zh" if has_cjk else "en")
        languages = tuple(sorted(languages_seen))
        failed_recovery = any(t.observed.execution_status in {"FAILED", "REJECTED", "PARTIAL", "INCONCLUSIVE"} for t in transitions)
        core = dict(schema_version="trajectory_instrumentation_gate_receipt.v1", organization_id=kwargs["organization_id"], workspace_id=kwargs["workspace_id"], projection_policy=trajectory.projection_policy, runtime_contract=kwargs["runtime_contract"], scorer_contract=kwargs["scorer_contract"], collector_contract=kwargs["collector_contract"], trajectory=trajectory.pin(), related_trajectories=tuple(item.pin() for item in related_trajectories), verification_evidence=tuple(dict.fromkeys(pin for result in results for pin in result.evidence_refs)), languages=languages, has_failed_recovery_attempt=failed_recovery, fresh_session_reconstruction=next(item for item in results if item.check_id == "TIG-08").status == "PASS", passed_check_ids=tuple(item.check_id for item in results), check_results=results, attested_properties=("admissibility", "exact_binding", "replayability"), status="GREEN", validated_at=kwargs["validated_at"])
        draft = TrajectoryInstrumentationGateReceiptV1.model_construct(gate_ref="trajectory-instrumentation-gate:pending", gate_hash="sha256:" + "0" * 64, **core)
        digest = content_hash(draft.hash_payload())
        return TrajectoryInstrumentationGateReceiptV1.model_validate({**core, "gate_ref": f"trajectory-instrumentation-gate:{digest}", "gate_hash": f"sha256:{digest}"})

    def _resolve_evidence(
        self, pin: ArtifactPinV1 | None, organization_id: str, workspace_id: str,
    ) -> Evidence | None:
        if pin is None or pin.artifact_kind != "evidence" or self._ontology_store is None:
            return None
        evidence = self._ontology_store.get(
            organization_id=organization_id, workspace_id=workspace_id,
            artifact_ref=pin.artifact_ref, contract_type=Evidence,
        )
        return evidence if evidence is not None and evidence.pin() == pin else None

    def _resolve_verification(
        self, pin: ArtifactPinV1 | None, organization_id: str, workspace_id: str,
    ) -> TenderDecisionProjectionReplayGateReceiptV1 | None:
        if pin is None or pin.artifact_kind != "tender_decision_projection_replay_gate_receipt":
            return None
        receipt = self._store.get_derived_projection(
            pin.artifact_ref, organization_id=organization_id, workspace_id=workspace_id,
            contract_type=TenderDecisionProjectionReplayGateReceiptV1,
        )
        if (receipt.receipt_ref, receipt.receipt_hash) != (pin.artifact_ref, pin.artifact_hash):
            return None
        return receipt

    def _language_for_transition(
        self, transition: DecisionTransitionProjectionV1, organization_id: str, workspace_id: str,
    ) -> str | None:
        evidence = self._resolve_evidence(transition.language_evidence, organization_id, workspace_id)
        if evidence is None:
            return None
        return "zh" if any("\u3400" <= character <= "\u9fff" for character in evidence.selector.exact_quote) else "en"

    @staticmethod
    def _store_projection_pin(projection: DecisionLocalProjectionV1) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="decision_local_projection",
            artifact_ref=projection.projection_ref,
            artifact_hash=projection.projection_hash,
        )

    @staticmethod
    def assert_trajectory_instrumentation_green(
        receipt: TrajectoryInstrumentationGateReceiptV1 | None,
        *,
        runtime_contract: ArtifactPinV1,
        scorer_contract: ArtifactPinV1,
        collector_contract: ArtifactPinV1,
    ) -> None:
        if receipt is None or receipt.status != "GREEN":
            raise RuntimeError("TRAJECTORY_INSTRUMENTATION_GREEN_REQUIRED")
        if (
            receipt.runtime_contract != runtime_contract
            or receipt.scorer_contract != scorer_contract
            or receipt.collector_contract != collector_contract
        ):
            raise RuntimeError("TRAJECTORY_INSTRUMENTATION_GREEN_STALE")
        if set(receipt.passed_check_ids) != set(TRAJECTORY_INSTRUMENTATION_CHECK_IDS):
            raise RuntimeError("TRAJECTORY_INSTRUMENTATION_GREEN_INCOMPLETE")


__all__ = ["DecisionTrajectoryProjectionService", "TrajectoryInstrumentationVerifier"]
