"""Compile one exact WorkingContext into a deterministic DecisionLocalProjectionV1."""

from __future__ import annotations

from datetime import datetime

from contracts.decision_local_projection import (
    DecisionContractV1,
    DecisionEvidenceProjectionV1,
    DecisionLocalProjectionV1,
    MaterialDependencyResolutionV1,
)
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import ContextAssertion, ContextGraph, Evidence, WorkingContext
from eos.artifact_hashing import content_hash
from services.working_context_selection_policy import working_context_selection_policy
from storage.ontology_build_store import OntologyBuildStore

_COMPILER_PAYLOAD = {
    "compiler": "decision_local_projection",
    "version": "v1",
    "behavior": "exact WorkingContext selected assertions and evidence closure to model-visible projection",
}
_COMPILER_HASH = "sha256:" + content_hash(_COMPILER_PAYLOAD)
_COMPILER_REF = "decision-local-projection-compiler:v1:" + _COMPILER_HASH[7:]


class DecisionLocalProjectionService:
    """Derive decision-local model input without creating a new semantic authority."""

    def __init__(self, *, build_store: OntologyBuildStore) -> None:
        if not isinstance(build_store, OntologyBuildStore):
            raise TypeError("decision-local projection requires the shared kernel store")
        self._store = build_store

    @staticmethod
    def compiler_pin() -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="decision_projection_compiler",
            artifact_ref=_COMPILER_REF,
            artifact_hash=_COMPILER_HASH,
        )

    def _context(self, working_context: WorkingContext) -> ContextGraph:
        if working_context.context_graph.artifact_kind != "context_graph":
            raise ValueError("CONTEXT_GRAPH_REQUIRED")
        context = self._store.get(
            organization_id=working_context.organization_id,
            workspace_id=working_context.workspace_id,
            artifact_ref=working_context.context_graph.artifact_ref,
            contract_type=ContextGraph,
        )
        if context is None:
            raise ValueError("CONTEXT_NOT_FOUND")
        if context.pin() != working_context.context_graph:
            raise ValueError("WORKING_CONTEXT_SOURCE_MISMATCH")
        if context.state != "ADMITTED":
            raise ValueError("ADMITTED_CONTEXT_REQUIRED")
        if working_context.ontology != context.ontology:
            raise ValueError("WORKING_CONTEXT_ONTOLOGY_MISMATCH")
        if working_context.selection_policy != working_context_selection_policy():
            raise ValueError("UNTRUSTED_WORKING_CONTEXT_SELECTOR")
        return context

    def _selected_assertions(
        self,
        *,
        working_context: WorkingContext,
        context: ContextGraph,
    ) -> tuple[ContextAssertion, ...]:
        by_id = {item.assertion_id: item for item in context.assertions}
        if not working_context.selected_assertion_ids:
            raise ValueError("DECISION_PROJECTION_REQUIRES_CONTEXT_FACTS")
        if any(item not in by_id for item in working_context.selected_assertion_ids):
            raise ValueError("UNKNOWN_ASSERTION_SELECTION")
        assertions = tuple(by_id[item] for item in working_context.selected_assertion_ids)

        expected_evidence = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash): pin
            for assertion in assertions
            for pin in assertion.evidence
        }
        actual_evidence = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash): pin
            for pin in working_context.evidence
        }
        if expected_evidence != actual_evidence:
            raise ValueError("WORKING_CONTEXT_EVIDENCE_MISMATCH")

        context_evidence = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
            for pin in context.evidence
        }
        if not set(expected_evidence) <= context_evidence:
            raise ValueError("OUT_OF_GRAPH_EVIDENCE")
        for pin in expected_evidence.values():
            evidence = self._store.get(
                organization_id=context.organization_id,
                workspace_id=context.workspace_id,
                artifact_ref=pin.artifact_ref,
                contract_type=Evidence,
            )
            if evidence is None or evidence.pin() != pin:
                raise ValueError("UNRESOLVABLE_DECISION_EVIDENCE")
        return assertions

    def evidence_closure(self, working_context: WorkingContext) -> tuple[Evidence, ...]:
        """Resolve the exact selected WorkingContext evidence closure through its persisted owner."""
        context = self._context(working_context)
        self._selected_assertions(working_context=working_context, context=context)
        values: list[Evidence] = []
        for pin in working_context.evidence:
            evidence = self._store.get(
                organization_id=context.organization_id,
                workspace_id=context.workspace_id,
                artifact_ref=pin.artifact_ref,
                contract_type=Evidence,
            )
            if evidence is None or evidence.pin() != pin:
                raise ValueError("UNRESOLVABLE_DECISION_EVIDENCE")
            values.append(evidence)
        return tuple(values)

    @staticmethod
    def evaluate_semantic_proposal(
        projection: DecisionLocalProjectionV1,
        *,
        producer_state: str,
        proposed_semantic_result: str | None,
    ) -> tuple[str, str | None]:
        """Apply epistemic completeness without collapsing producer transport states into UNKNOWN."""
        producer_states = {
            "PRODUCER_OBSERVATION",
            "PRODUCER_ABSTENTION",
            "INVALID_OR_CENSORED_SCORE",
            "PROVIDER_FAILURE",
        }
        if producer_state not in producer_states:
            raise ValueError("UNKNOWN_PRODUCER_STATE")
        if proposed_semantic_result not in {None, "SEMANTIC_TRUE", "SEMANTIC_FALSE", "SEMANTIC_UNKNOWN"}:
            raise ValueError("UNKNOWN_SEMANTIC_RESULT")
        if producer_state != "PRODUCER_OBSERVATION":
            if proposed_semantic_result is not None:
                raise ValueError("NON_OBSERVATION_CANNOT_PROPOSE_SEMANTIC_RESULT")
            return producer_state, None
        if proposed_semantic_result is None:
            raise ValueError("PRODUCER_OBSERVATION_REQUIRES_SEMANTIC_RESULT")
        if any(item.status == "MISSING" for item in projection.material_dependency_resolutions):
            return producer_state, "SEMANTIC_UNKNOWN"
        return producer_state, proposed_semantic_result

    @staticmethod
    def _world_time_basis(context: ContextGraph) -> dict[str, str | None]:
        def iso(value: datetime | None) -> str | None:
            return None if value is None else value.isoformat().replace("+00:00", "Z")

        return {
            "at_time": iso(context.world.at_time),
            "start_time": iso(context.world.start_time),
            "end_time": iso(context.world.end_time),
            "governance_time": iso(context.world.governance_time),
            "reasoning_version": context.world.reasoning_version,
        }

    @staticmethod
    def _projection_content(
        *,
        working_context: WorkingContext,
        context: ContextGraph,
        decision_subject_ref: str,
        decision_contract: DecisionContractV1,
        assertions: tuple[ContextAssertion, ...],
        evidence_closure: tuple[Evidence, ...],
    ) -> dict:
        nodes = {item.node_id: item for item in context.nodes}
        if decision_subject_ref not in working_context.selected_node_ids:
            raise ValueError("DECISION_SUBJECT_OUTSIDE_WORKING_CONTEXT")
        if decision_subject_ref not in nodes:
            raise ValueError("DECISION_SUBJECT_NOT_FOUND")

        evidence_projection: list[DecisionEvidenceProjectionV1] = []
        for assertion in assertions:
            if assertion.subject_ref not in nodes or assertion.object_ref not in nodes:
                raise ValueError("DECISION_ASSERTION_NODE_BINDING_REQUIRED")
            evidence_projection.append(
                DecisionEvidenceProjectionV1(
                    assertion_id=assertion.assertion_id,
                    subject_ref=assertion.subject_ref,
                    subject_label=nodes[assertion.subject_ref].label,
                    predicate=assertion.predicate,
                    object_ref=assertion.object_ref,
                    object_label=nodes[assertion.object_ref].label,
                    evidence=assertion.evidence,
                )
            )

        evidence = tuple(
            sorted(
                working_context.evidence,
                key=lambda pin: (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash),
            )
        )
        dependency_resolutions: list[MaterialDependencyResolutionV1] = []
        for predicate in decision_contract.semantic_predicates:
            for requirement in predicate.material_requirements:
                matches = tuple(
                    item.pin()
                    for item in evidence_closure
                    if item.evidence_role == requirement.required_evidence_role
                )
                satisfied = len(matches) >= requirement.min_count
                dependency_resolutions.append(MaterialDependencyResolutionV1(
                    predicate_id=predicate.predicate_id,
                    dependency_id=requirement.dependency_id,
                    required_evidence_role=requirement.required_evidence_role,
                    min_count=requirement.min_count,
                    status="SATISFIED" if satisfied else "MISSING",
                    matched_evidence=matches if satisfied else (),
                ))
        return {
            "schema_version": "decision_local_projection.v1",
            "organization_id": working_context.organization_id,
            "workspace_id": working_context.workspace_id,
            "context_graph": context.pin(),
            "working_context": working_context.pin(),
            "ontology": context.ontology,
            "decision_subject_ref": decision_subject_ref,
            "decision_contract_ref": decision_contract.contract_ref,
            "decision_contract_hash": decision_contract.contract_hash,
            "selected_assertion_ids": working_context.selected_assertion_ids,
            "evidence": evidence,
            "evidence_projection": tuple(evidence_projection),
            "semantic_predicates": decision_contract.semantic_predicates,
            "deterministic_constraints": decision_contract.deterministic_constraints,
            "material_dependency_resolutions": tuple(dependency_resolutions),
            "world_mode": context.world.mode,
            "world_time_basis": DecisionLocalProjectionService._world_time_basis(context),
            "compiler_ref": _COMPILER_REF,
            "compiler_hash": _COMPILER_HASH,
        }

    def compile(
        self,
        *,
        working_context: WorkingContext,
        decision_subject_ref: str,
        decision_contract: DecisionContractV1,
        required_source_graph: ArtifactPinV1 | None = None,
    ) -> DecisionLocalProjectionV1:
        context = self._context(working_context)
        if required_source_graph is not None and context.world.source_graph != required_source_graph:
            raise ValueError("DECISION_SOURCE_GRAPH_MISMATCH")
        if (working_context.organization_id, working_context.workspace_id) != (
            context.organization_id,
            context.workspace_id,
        ):
            raise ValueError("WORKING_CONTEXT_SCOPE_MISMATCH")
        assertions = self._selected_assertions(working_context=working_context, context=context)
        evidence_closure = self.evidence_closure(working_context)
        content = self._projection_content(
            working_context=working_context,
            context=context,
            decision_subject_ref=decision_subject_ref,
            decision_contract=decision_contract,
            assertions=assertions,
            evidence_closure=evidence_closure,
        )

        provisional = DecisionLocalProjectionV1.model_construct(
            **{
                **content,
                "projection_ref": "decision-local-projection:pending",
                "projection_hash": "sha256:" + "0" * 64,
                "model_input_hash": "sha256:" + "0" * 64,
            }
        )
        model_input_hash = "sha256:" + content_hash(provisional.model_input_payload())
        with_input = DecisionLocalProjectionV1.model_construct(
            **{
                **content,
                "projection_ref": "decision-local-projection:pending",
                "projection_hash": "sha256:" + "0" * 64,
                "model_input_hash": model_input_hash,
            }
        )
        projection_hash = "sha256:" + content_hash(with_input.hash_payload())
        return DecisionLocalProjectionV1.model_validate(
            {
                **content,
                "projection_ref": "decision-local-projection:" + projection_hash[7:],
                "projection_hash": projection_hash,
                "model_input_hash": model_input_hash,
            }
        )


__all__ = ["DecisionLocalProjectionService"]
