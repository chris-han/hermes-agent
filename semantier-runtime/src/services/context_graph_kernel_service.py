"""Clean-break ContextGraph formation and admission over the five-object kernel.

This service deliberately reuses OntologyBuildStore as the scoped content-addressed
UoW. It owns no new database and no authority source. Admission requires an injected
server-side verifier so a DecisionRecord cannot manufacture authorization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash
from contracts.semantic_kernel import (
    AuthorizationBasis,
    ChangeCandidate,
    ContextAssertion,
    ContextEdge,
    ContextFormation,
    ContextGraph,
    ContextNode,
    ContextWorld,
    DecisionRecord,
    Evidence,
    ExcludedFormationInput,
    FormationInput,
    Ontology,
    ReasoningTrace,
    validate_context_admission_transition,
)
from storage.ontology_build_store import OntologyBuildStore


class KernelContextError(ValueError):
    """Bounded error code for the new kernel seam."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ContextGraphKernelService:
    def __init__(
        self,
        *,
        store: OntologyBuildStore,
        verify_external_pin: Callable[[ArtifactPinV1], None] | None = None,
        verify_authorization: Callable[[AuthorizationBasis], None] | None = None,
    ) -> None:
        if not isinstance(store, OntologyBuildStore):
            raise TypeError("store must be OntologyBuildStore")
        if verify_external_pin is not None and not callable(verify_external_pin):
            raise TypeError("verify_external_pin must be callable")
        if verify_authorization is not None and not callable(verify_authorization):
            raise TypeError("verify_authorization must be callable")
        self._store = store
        self._verify_external_pin = verify_external_pin
        self._verify_authorization = verify_authorization

    def _load_exact(self, *, scope: tuple[str, str], pin: ArtifactPinV1, model):
        organization_id, workspace_id = scope
        value = self._store.get(
            organization_id=organization_id,
            workspace_id=workspace_id,
            artifact_ref=pin.artifact_ref,
            contract_type=model,
        )
        if value is None:
            raise KernelContextError("OBJECT_NOT_FOUND")
        if value.pin() != pin:
            raise KernelContextError("HASH_MISMATCH")
        return value

    def form_context(
        self,
        *,
        formation_input: FormationInput,
        world: ContextWorld,
        nodes: tuple[ContextNode, ...],
        edges: tuple[ContextEdge, ...],
        assertions: tuple[ContextAssertion, ...],
        selected_evidence: tuple[ArtifactPinV1, ...],
        excluded_evidence: tuple[ExcludedFormationInput, ...],
        conflicts: tuple[str, ...],
        formation_policy: ArtifactPinV1,
        formation_policy_version: str,
        applied_change_candidate: ArtifactPinV1 | None,
        recorded_at: datetime,
        persist: bool = True,
    ) -> ContextGraph:
        """Form and persist one immutable PROPOSED ContextGraph revision."""

        formation_input = FormationInput.model_validate(
            formation_input.model_dump(mode="json")
        )
        world = ContextWorld.model_validate(world.model_dump(mode="json"))
        nodes = tuple(ContextNode.model_validate(item.model_dump(mode="json")) for item in nodes)
        edges = tuple(ContextEdge.model_validate(item.model_dump(mode="json")) for item in edges)
        assertions = tuple(
            ContextAssertion.model_validate(item.model_dump(mode="json"))
            for item in assertions
        )
        excluded_evidence = tuple(
            ExcludedFormationInput.model_validate(item.model_dump(mode="json"))
            for item in excluded_evidence
        )
        scope = (formation_input.organization_id, formation_input.workspace_id)
        ontology = self._load_exact(
            scope=scope, pin=formation_input.ontology, model=Ontology
        )
        known_node_types = {item.type_id for item in ontology.entity_types}
        known_relation_types = {item.type_id for item in ontology.relation_types}
        if any(node.node_type not in known_node_types for node in nodes):
            raise KernelContextError("UNKNOWN_ONTOLOGY_ENTITY_TYPE")
        if any(edge.relation_type not in known_relation_types for edge in edges):
            raise KernelContextError("UNKNOWN_ONTOLOGY_RELATION_TYPE")
        evidence_by_key: dict[tuple[str, str, str], Evidence] = {}
        for pin in formation_input.evidence:
            value = self._load_exact(scope=scope, pin=pin, model=Evidence)
            evidence_by_key[(pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)] = value
        if formation_policy not in formation_input.policy_pins:
            raise KernelContextError("FORMATION_POLICY_NOT_PINNED")
        if self._verify_external_pin is None:
            raise KernelContextError("EXTERNAL_PIN_VERIFIER_REQUIRED")
        for policy_pin in formation_input.policy_pins:
            self._verify_external_pin(policy_pin)
        self._verify_external_pin(world.source_graph)

        selected_keys = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
            for pin in selected_evidence
        }
        available_keys = set(evidence_by_key)
        if not selected_keys <= available_keys:
            raise KernelContextError("UNBOUND_EVIDENCE")
        excluded_keys = {
            (item.input.artifact_kind, item.input.artifact_ref, item.input.artifact_hash)
            for item in excluded_evidence
        }
        if not excluded_keys <= available_keys:
            raise KernelContextError("UNBOUND_EVIDENCE")

        predecessor: ContextGraph | None = None
        predecessor_pin = formation_input.predecessor_context
        revision = 0
        if predecessor_pin is not None:
            predecessor = self._load_exact(
                scope=scope, pin=predecessor_pin, model=ContextGraph
            )
            revision = predecessor.revision + 1

        if applied_change_candidate is not None:
            candidate = self._load_exact(
                scope=scope,
                pin=applied_change_candidate,
                model=ChangeCandidate,
            )
            if (
                candidate.stage != "ASSESSMENT"
                or candidate.assessment is None
                or candidate.assessment.disposition
                != "ACCEPTED_FOR_SUCCESSOR_INPUT"
            ):
                raise KernelContextError("CHANGE_CANDIDATE_NOT_ACCEPTED_FOR_SUCCESSOR")
            if candidate.target != "CONTEXT_FORMATION":
                raise KernelContextError("CHANGE_CANDIDATE_TARGET_MISMATCH")
            if formation_policy not in candidate.successor_inputs:
                raise KernelContextError("CHANGE_CANDIDATE_SUCCESSOR_INPUT_MISMATCH")

        formation = ContextFormation(
            source_inputs=formation_input.evidence,
            selected_inputs=selected_evidence,
            excluded_inputs=excluded_evidence,
            conflicts=conflicts,
            formation_policy=formation_policy,
            formation_policy_version=formation_policy_version,
            predecessor_context=predecessor_pin,
            applied_change_candidate=applied_change_candidate,
        )
        context = ContextGraph.create(
            schema_version="semantier.context_graph.v1",
            organization_id=formation_input.organization_id,
            workspace_id=formation_input.workspace_id,
            revision=revision,
            state="PROPOSED",
            predecessor=predecessor_pin,
            intent=formation_input.intent,
            ontology=formation_input.ontology,
            world=world,
            nodes=nodes,
            edges=edges,
            assertions=assertions,
            evidence=selected_evidence,
            formation=formation,
            recorded_at=recorded_at,
        )
        if persist:
            self._store.put(context)
        return context

    def admit_new_context_with_request(
        self,
        *,
        proposed: ContextGraph,
        authorization: AuthorizationBasis,
        reasoning: ReasoningTrace,
        evidence: tuple[ArtifactPinV1, ...],
        actor_id: str,
        reasoner_identity: str,
        recorded_at: datetime,
        request_id: str,
        request_hash: str,
        operation: str = "START_RUNTIME",
    ) -> tuple[DecisionRecord, ContextGraph]:
        """Atomically publish proposal, admission decision, successor and retry binding."""

        if self._verify_authorization is None:
            raise KernelContextError("AUTHORITY_VERIFIER_REQUIRED")
        proposed = ContextGraph.model_validate(proposed.model_dump(mode="json"))
        authorization = AuthorizationBasis.model_validate(
            authorization.model_dump(mode="json")
        )
        reasoning = ReasoningTrace.model_validate(reasoning.model_dump(mode="json"))
        if proposed.state != "PROPOSED":
            raise KernelContextError("CONTEXT_NOT_PROPOSED")
        if authorization.subject != proposed.pin():
            raise KernelContextError("AUTHORITY_SUBJECT_MISMATCH")
        self._verify_authorization(authorization)
        selected_keys = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
            for pin in proposed.evidence
        }
        decision_evidence_keys = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) for pin in evidence
        }
        if not decision_evidence_keys <= selected_keys:
            raise KernelContextError("UNBOUND_EVIDENCE")

        decision = DecisionRecord.create(
            schema_version="semantier.decision_record.v1",
            organization_id=proposed.organization_id,
            workspace_id=proposed.workspace_id,
            purpose="CONTEXT_ADMISSION",
            subject=proposed.pin(),
            context=proposed.pin(),
            ontology=proposed.ontology,
            result_code="ADMITTED",
            epistemic_status="KNOWN",
            evidence=evidence,
            admissibility=None,
            reasoning=reasoning,
            actor_id=actor_id,
            reasoner_identity=reasoner_identity,
            authorization=authorization,
            outcome_observation=None,
            recorded_at=recorded_at,
        )
        admitted_formation = ContextFormation(
            source_inputs=proposed.formation.source_inputs,
            selected_inputs=proposed.formation.selected_inputs,
            excluded_inputs=proposed.formation.excluded_inputs,
            conflicts=proposed.formation.conflicts,
            formation_policy=proposed.formation.formation_policy,
            formation_policy_version=proposed.formation.formation_policy_version,
            predecessor_context=proposed.pin(),
            applied_change_candidate=proposed.formation.applied_change_candidate,
        )
        admitted = ContextGraph.create(
            schema_version="semantier.context_graph.v1",
            organization_id=proposed.organization_id,
            workspace_id=proposed.workspace_id,
            revision=proposed.revision + 1,
            state="ADMITTED",
            predecessor=proposed.pin(),
            admission_decision=decision.pin(),
            intent=proposed.intent,
            ontology=proposed.ontology,
            world=proposed.world,
            nodes=proposed.nodes,
            edges=proposed.edges,
            assertions=proposed.assertions,
            evidence=proposed.evidence,
            formation=admitted_formation,
            recorded_at=recorded_at,
        )
        validate_context_admission_transition(proposed, decision, admitted)
        winner = self._store.put_many_with_operation_result(
            artifacts=(proposed, decision, admitted),
            organization_id=proposed.organization_id,
            workspace_id=proposed.workspace_id,
            actor_ref=actor_id,
            operation=operation,
            request_id=request_id,
            request_hash=request_hash,
            result_pin=admitted.pin(),
            verify_before_commit=lambda: self._verify_authorization(authorization),
        )
        if winner != admitted.pin():
            existing = self._load_exact(
                scope=(proposed.organization_id, proposed.workspace_id),
                pin=winner,
                model=ContextGraph,
            )
            if existing.admission_decision is None:
                raise KernelContextError("INTEGRITY_UNPROVABLE")
            existing_decision = self._load_exact(
                scope=(proposed.organization_id, proposed.workspace_id),
                pin=existing.admission_decision,
                model=DecisionRecord,
            )
            return existing_decision, existing
        return decision, admitted

    def decide_context_admission(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        proposed: ArtifactPinV1,
        accepted: bool,
        authorization: AuthorizationBasis,
        reasoning: ReasoningTrace,
        evidence: tuple[ArtifactPinV1, ...],
        actor_id: str,
        reasoner_identity: str,
        recorded_at: datetime,
    ) -> tuple[DecisionRecord, ContextGraph | None]:
        """Persist admission decision and, on acceptance, one ADMITTED successor."""

        if self._verify_authorization is None:
            raise KernelContextError("AUTHORITY_VERIFIER_REQUIRED")
        authorization = AuthorizationBasis.model_validate(
            authorization.model_dump(mode="json")
        )
        reasoning = ReasoningTrace.model_validate(reasoning.model_dump(mode="json"))
        if proposed.artifact_kind != "context_graph":
            raise KernelContextError("INVALID_SUBJECT")

        scope = (organization_id, workspace_id)
        try:
            context = self._load_exact(scope=scope, pin=proposed, model=ContextGraph)
        except PermissionError as exc:
            raise KernelContextError("FORBIDDEN_SCOPE") from exc
        if context.state != "PROPOSED":
            raise KernelContextError("CONTEXT_NOT_PROPOSED")
        if authorization.subject != proposed:
            raise KernelContextError("AUTHORITY_SUBJECT_MISMATCH")
        self._verify_authorization(authorization)

        selected_keys = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
            for pin in context.evidence
        }
        decision_evidence_keys = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) for pin in evidence
        }
        if not decision_evidence_keys <= selected_keys:
            raise KernelContextError("UNBOUND_EVIDENCE")

        decision = DecisionRecord.create(
            schema_version="semantier.decision_record.v1",
            organization_id=context.organization_id,
            workspace_id=context.workspace_id,
            purpose="CONTEXT_ADMISSION",
            subject=context.pin(),
            context=context.pin(),
            ontology=context.ontology,
            result_code="ADMITTED" if accepted else "REJECTED",
            epistemic_status="KNOWN",
            evidence=evidence,
            admissibility=None,
            reasoning=reasoning,
            actor_id=actor_id,
            reasoner_identity=reasoner_identity,
            authorization=authorization,
            outcome_observation=None,
            recorded_at=recorded_at,
        )
        if not accepted:
            self._store.put(decision)
            return decision, None

        admitted_formation = ContextFormation(
            source_inputs=context.formation.source_inputs,
            selected_inputs=context.formation.selected_inputs,
            excluded_inputs=context.formation.excluded_inputs,
            conflicts=context.formation.conflicts,
            formation_policy=context.formation.formation_policy,
            formation_policy_version=context.formation.formation_policy_version,
            predecessor_context=context.pin(),
            applied_change_candidate=context.formation.applied_change_candidate,
        )
        admitted = ContextGraph.create(
            schema_version="semantier.context_graph.v1",
            organization_id=context.organization_id,
            workspace_id=context.workspace_id,
            revision=context.revision + 1,
            state="ADMITTED",
            predecessor=context.pin(),
            admission_decision=decision.pin(),
            intent=context.intent,
            ontology=context.ontology,
            world=context.world,
            nodes=context.nodes,
            edges=context.edges,
            assertions=context.assertions,
            evidence=context.evidence,
            formation=admitted_formation,
            recorded_at=recorded_at,
        )
        validate_context_admission_transition(context, decision, admitted)
        # One existing-store UoW: either both the decision and successor are durable,
        # or neither is.
        self._store.put_many((decision, admitted))
        return decision, admitted


__all__ = ["ContextGraphKernelService", "KernelContextError"]
