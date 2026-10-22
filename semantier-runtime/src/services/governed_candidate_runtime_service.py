"""W5 governed candidate admission binding and accepted-only runtime resolution."""

from __future__ import annotations

from storage.hosted_postgres import sql_query,begin_write,in_transaction

from contracts.artifact_pin import ArtifactPinV1
from contracts.candidate_stage_binding import OwningStage
from contracts.governed_candidate_runtime import (
    CandidateAdmissionBindingV1,
    RuntimeCandidateResolutionV1,
)
from contracts.semantic_kernel import ContextGraph, ContextGraphOrigin, DecisionRecord, PreContextOrigin
from semantier.cognitive_context.store import CognitiveContextStore
from services.candidate_admission_routing_service import (
    CandidateAdmissionRouteV1,
    CandidateAdmissionRoutingService,
)
from services.candidate_stage_routing_service import CandidateStageRoutingService
from services.change_candidate_service import ChangeCandidateService
from storage.ontology_build_store import OntologyBuildStore


class GovernedCandidateRuntimeService:
    """Bind existing owner receipts; does not create semantic admission or activation authority."""

    def __init__(
        self,
        *,
        store: CognitiveContextStore,
        candidate_service: ChangeCandidateService,
        kernel_store: OntologyBuildStore | None = None,
    ) -> None:
        if not isinstance(store, CognitiveContextStore):
            raise TypeError("candidate runtime requires CognitiveContextStore")
        if not isinstance(candidate_service, ChangeCandidateService):
            raise TypeError("candidate runtime requires ChangeCandidateService")
        resolved_kernel_store = kernel_store if kernel_store is not None else candidate_service._store
        if not isinstance(resolved_kernel_store, OntologyBuildStore):
            raise TypeError("candidate runtime requires OntologyBuildStore")
        self._store = store
        self._kernel_store = resolved_kernel_store
        self._candidates = candidate_service
        self._routing = CandidateStageRoutingService()

    @staticmethod
    def _origin_pin(candidate) -> ArtifactPinV1:
        if isinstance(candidate.origin, ContextGraphOrigin):
            return candidate.origin.context
        if isinstance(candidate.origin, PreContextOrigin):
            return candidate.origin.formation_input.pin()
        raise ValueError("CANDIDATE_ORIGIN_UNSUPPORTED")

    def _accepted_candidate(self, candidate_pin: ArtifactPinV1):
        candidate = self._candidates.get(pin=candidate_pin)
        if (
            candidate.stage != "ASSESSMENT"
            or candidate.assessment is None
            or candidate.assessment.disposition != "ACCEPTED_FOR_SUCCESSOR_INPUT"
        ):
            raise ValueError("CANDIDATE_NOT_ACCEPTED_FOR_SUCCESSOR")
        if len(candidate.successor_inputs) != 1:
            raise ValueError("ACCEPTED_CANDIDATE_REQUIRES_EXACT_SUCCESSOR_INPUT")
        return candidate

    def _kernel_get(self, *, pin: ArtifactPinV1, model):
        value = self._kernel_store.get(
            organization_id=self._scope_for_pin(pin)[0],
            workspace_id=self._scope_for_pin(pin)[1],
            artifact_ref=pin.artifact_ref,
            contract_type=model,
        )
        if value is None or value.pin() != pin:
            raise ValueError("OWNER_RECEIPT_NOT_FOUND")
        return value

    def _scope_for_pin(self, pin: ArtifactPinV1) -> tuple[str, str]:
        row = self._kernel_store._connection.execute(
            sql_query("SELECT organization_id,workspace_id,artifact_hash FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT organization_id,workspace_id,artifact_hash FROM ontology_build_artifacts WHERE artifact_ref=%s'),
            (pin.artifact_ref,),
        ).fetchone()
        if row is None or row[2] != pin.artifact_hash:
            raise ValueError("OWNER_RECEIPT_NOT_FOUND")
        return str(row[0]), str(row[1])

    def _verify_context_owner_admission(
        self,
        *,
        candidate,
        owner_receipt: ArtifactPinV1,
    ) -> None:
        if owner_receipt.artifact_kind != "decision_record":
            raise ValueError("CONTEXT_OWNER_ADMISSION_DECISION_REQUIRED")
        decision = self._kernel_get(pin=owner_receipt, model=DecisionRecord)
        if decision.purpose != "CONTEXT_ADMISSION" or decision.result_code != "ADMITTED":
            raise ValueError("INVALID_CONTEXT_OWNER_ADMISSION_DECISION")
        proposed = self._kernel_get(pin=decision.context, model=ContextGraph)
        if (
            proposed.state != "PROPOSED"
            or proposed.formation.applied_change_candidate != candidate.pin()
            or proposed.formation.formation_policy != candidate.successor_inputs[0]
        ):
            raise ValueError("CONTEXT_OWNER_ADMISSION_BINDING_MISMATCH")

    def _verify_context_owner_activation(
        self,
        *,
        candidate,
        admitted_binding: CandidateAdmissionBindingV1,
        owner_activation_receipt: ArtifactPinV1,
    ) -> None:
        if owner_activation_receipt.artifact_kind != "context_graph":
            raise ValueError("CONTEXT_OWNER_ACTIVE_GRAPH_REQUIRED")
        context = self._kernel_get(pin=owner_activation_receipt, model=ContextGraph)
        if (
            context.state != "ADMITTED"
            or context.admission_decision != admitted_binding.owner_receipt
            or context.formation.applied_change_candidate != candidate.pin()
            or context.formation.formation_policy != candidate.successor_inputs[0]
        ):
            raise ValueError("CONTEXT_OWNER_ACTIVATION_BINDING_MISMATCH")

    def admit(
        self,
        *,
        candidate_pin: ArtifactPinV1,
        owning_stage: OwningStage,
        current_basis: ArtifactPinV1,
        owner_receipt: ArtifactPinV1,
    ) -> CandidateAdmissionBindingV1:
        candidate = self._accepted_candidate(candidate_pin)
        origin = self._origin_pin(candidate)
        if current_basis != origin:
            raise ValueError("STALE_ADMISSION_BASIS")
        stage_binding = self._routing.bind(
            target=candidate.target,
            failure_class=candidate.failure_class,
            owning_stage=owning_stage,
        )
        if candidate.target == "CONTEXT_FORMATION" and owning_stage == "DECISION":
            self._verify_context_owner_admission(
                candidate=candidate,
                owner_receipt=owner_receipt,
            )
        else:
            raise ValueError("OWNER_ADMISSION_NOT_IMPLEMENTED_FOR_STAGE_TARGET")
        binding = CandidateAdmissionBindingV1.create(
            organization_id=candidate.organization_id,
            workspace_id=candidate.workspace_id,
            candidate=candidate.pin(),
            owning_stage=owning_stage,
            owner_ref=stage_binding.owner_ref,
            predecessor_basis=current_basis,
            successor_input=candidate.successor_inputs[0],
            owner_receipt=owner_receipt,
            state="ADMITTED",
            predecessor_binding=None,
            successor_binding=None,
            reason=None,
        )
        return self._store.save_or_get_derived_projection(binding)

    def activate(
        self,
        *,
        admitted: CandidateAdmissionBindingV1,
        owner_activation_receipt: ArtifactPinV1,
    ) -> CandidateAdmissionBindingV1:
        if admitted.state != "ADMITTED":
            raise ValueError("ADMITTED_BINDING_REQUIRED")
        candidate = self._accepted_candidate(admitted.candidate)
        stage_binding = self._routing.bind(
            target=candidate.target,
            failure_class=candidate.failure_class,
            owning_stage=admitted.owning_stage,
        )
        if admitted.owner_ref != stage_binding.owner_ref:
            raise ValueError("OWNER_BINDING_MISMATCH")
        if candidate.target == "CONTEXT_FORMATION" and admitted.owning_stage == "DECISION":
            self._verify_context_owner_activation(
                candidate=candidate,
                admitted_binding=admitted,
                owner_activation_receipt=owner_activation_receipt,
            )
        else:
            raise ValueError("OWNER_ACTIVATION_NOT_IMPLEMENTED_FOR_STAGE_TARGET")
        active = CandidateAdmissionBindingV1.create(
            organization_id=admitted.organization_id,
            workspace_id=admitted.workspace_id,
            candidate=admitted.candidate,
            owning_stage=admitted.owning_stage,
            owner_ref=admitted.owner_ref,
            predecessor_basis=admitted.predecessor_basis,
            successor_input=admitted.successor_input,
            owner_receipt=owner_activation_receipt,
            state="ACTIVE",
            predecessor_binding=admitted.pin(),
            successor_binding=None,
            reason=None,
        )
        return self._store.save_or_get_derived_projection(active)

    def supersede(
        self,
        *,
        active: CandidateAdmissionBindingV1,
        successor_active: CandidateAdmissionBindingV1,
        owner_receipt: ArtifactPinV1,
        reason: str,
    ) -> CandidateAdmissionBindingV1:
        if active.state != "ACTIVE" or successor_active.state != "ACTIVE":
            raise ValueError("ACTIVE_BINDINGS_REQUIRED")
        if (
            active.organization_id,
            active.workspace_id,
            active.owning_stage,
            active.owner_ref,
        ) != (
            successor_active.organization_id,
            successor_active.workspace_id,
            successor_active.owning_stage,
            successor_active.owner_ref,
        ):
            raise ValueError("CROSS_OWNER_SUPERSESSION_FORBIDDEN")

        successor_candidate = self._accepted_candidate(successor_active.candidate)
        if (
            successor_candidate.target == "CONTEXT_FORMATION"
            and successor_active.owning_stage == "DECISION"
        ):
            if owner_receipt.artifact_kind != "decision_record":
                raise ValueError("CONTEXT_OWNER_SUPERSESSION_DECISION_REQUIRED")
            decision = self._kernel_get(pin=owner_receipt, model=DecisionRecord)
            successor_context = self._kernel_get(
                pin=successor_active.owner_receipt,
                model=ContextGraph,
            )
            if (
                decision.purpose != "CONTEXT_ADMISSION"
                or decision.result_code != "ADMITTED"
                or successor_context.admission_decision != decision.pin()
                or successor_context.formation.applied_change_candidate
                != successor_candidate.pin()
                or successor_context.formation.formation_policy
                != successor_candidate.successor_inputs[0]
            ):
                raise ValueError("CONTEXT_OWNER_SUPERSESSION_BINDING_MISMATCH")
        else:
            raise ValueError("OWNER_SUPERSESSION_NOT_IMPLEMENTED_FOR_STAGE_TARGET")

        terminal = CandidateAdmissionBindingV1.create(
            organization_id=active.organization_id,
            workspace_id=active.workspace_id,
            candidate=active.candidate,
            owning_stage=active.owning_stage,
            owner_ref=active.owner_ref,
            predecessor_basis=active.predecessor_basis,
            successor_input=active.successor_input,
            owner_receipt=owner_receipt,
            state="SUPERSEDED",
            predecessor_binding=active.pin(),
            successor_binding=successor_active.pin(),
            reason=reason,
        )
        return self._store.save_or_get_derived_projection(terminal)

    def resolve_active(
        self,
        *,
        active_binding: ArtifactPinV1,
        organization_id: str,
        workspace_id: str,
        admission_route: CandidateAdmissionRouteV1 | None = None,
    ) -> RuntimeCandidateResolutionV1:
        active = self._store.get_derived_projection(
            active_binding.artifact_ref,
            organization_id=organization_id,
            workspace_id=workspace_id,
            contract_type=CandidateAdmissionBindingV1,
        )
        if active.pin() != active_binding or active.state != "ACTIVE":
            raise ValueError("ACTIVE_BINDING_REQUIRED")

        all_bindings = self._store.list_derived_projections(
            organization_id=organization_id,
            workspace_id=workspace_id,
            projection_kind="candidate_admission_binding.v1",
            contract_type=CandidateAdmissionBindingV1,
        )
        if any(
            item.state == "SUPERSEDED"
            and item.predecessor_binding == active.pin()
            for item in all_bindings
        ):
            raise ValueError("ACTIVE_BINDING_NO_LONGER_CURRENT")

        candidate = self._accepted_candidate(active.candidate)
        if admission_route is None:
            raise ValueError("RUNTIME_ADMISSION_ROUTE_REQUIRED")
        if admission_route.candidate != candidate.pin():
            raise ValueError("RUNTIME_ADMISSION_ROUTE_CANDIDATE_MISMATCH")
        expected_route = CandidateAdmissionRoutingService().route(
            candidate=candidate,
            classification=admission_route.classification,
        )
        if expected_route != admission_route:
            raise ValueError("RUNTIME_ADMISSION_ROUTE_INVALID")
        if admission_route.destination not in {
            "PROCEDURAL_PROJECTION",
            "GOVERNED_ADMISSION",
        }:
            raise ValueError("RUNTIME_ADMISSION_ROUTE_BLOCKED")
        if candidate.successor_inputs != (active.successor_input,):
            raise ValueError("ACTIVE_SUCCESSOR_INPUT_MISMATCH")

        resolution = RuntimeCandidateResolutionV1.create(
            organization_id=organization_id,
            workspace_id=workspace_id,
            owning_stage=active.owning_stage,
            owner_ref=active.owner_ref,
            active_binding=active.pin(),
            candidate=active.candidate,
            successor_input=active.successor_input,
        )
        return self._store.save_or_get_derived_projection(resolution)

    def assert_cached_resolution_current(
        self,
        cached: RuntimeCandidateResolutionV1,
        *,
        admission_route: CandidateAdmissionRouteV1 | None = None,
    ) -> RuntimeCandidateResolutionV1:
        current = self.resolve_active(
            active_binding=cached.active_binding,
            organization_id=cached.organization_id,
            workspace_id=cached.workspace_id,
            admission_route=admission_route,
        )
        if current != cached:
            raise ValueError("CACHED_RUNTIME_RESOLUTION_STALE")
        return current
