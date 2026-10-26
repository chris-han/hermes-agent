"""Receipt-driven transition engine for the Tender review journey."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agents.auth_session import RequestContext
from services.workspace_artifact_preparation_service import WorkspaceArtifactPreparationService

from contracts.tender_journey import (
    JourneyArtifactBindingV1,
    TenderJourneyHeadV1,
    TenderJourneyPhase,
    TenderJourneySurface,
    TenderJourneyTransitionV1,
    TenderJourneyProjectionV1,
)
from storage.tender_journey_store import TenderJourneyStore
from storage.ontology_build_store import ontology_build_store_path
from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash
from services.effective_restriction_runtime_context_service import (
    EffectiveRestrictionRuntimeContextService,
)


def _selection_payload(item):
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if is_dataclass(item):
        return asdict(item)
    raise TypeError("certification selection must be a contract model or dataclass")


_NEXT_ACTION: dict[TenderJourneyPhase, tuple[str, bool]] = {
    "NEEDS_POLICY": ("SELECT_OR_UPLOAD_POLICY", True),
    "POLICY_READY": ("BUILD_CONTEXT", False),
    "BUILDING_CONTEXT": ("CONTINUE_CONTEXT_BUILD", False),
    "REVIEW_INTENT": ("REVIEW_MATERIAL_GRAPH_INTENT", True),
    "CERTIFYING_CONTEXT": ("CONTINUE_CERTIFICATION", False),
    "RUNTIME_BINDING_REQUIRED": ("COMPLETE_GOVERNED_RUNTIME_BINDING", True),
    "CONTEXT_READY": ("SELECT_RUNTIME_DOCUMENT", True),
    "NEEDS_RUNTIME_DOCUMENT": ("SELECT_RUNTIME_DOCUMENT", True),
    "FORMING_RUNTIME_CONTEXT": ("FORM_RUNTIME_CONTEXT", False),
    "INSPECTING": ("CONTINUE_INSPECTION", False),
    "REVIEW_RESULTS": ("REVIEW_OR_INSPECT_NEXT_DOCUMENT", True),
}

_ALLOWED_NEXT: dict[TenderJourneyPhase, frozenset[TenderJourneyPhase]] = {
    "NEEDS_POLICY": frozenset({"POLICY_READY"}),
    "POLICY_READY": frozenset({"BUILDING_CONTEXT"}),
    "BUILDING_CONTEXT": frozenset({"BUILDING_CONTEXT", "REVIEW_INTENT", "CERTIFYING_CONTEXT"}),
    "REVIEW_INTENT": frozenset({"REVIEW_INTENT", "BUILDING_CONTEXT", "CERTIFYING_CONTEXT"}),
    "CERTIFYING_CONTEXT": frozenset({"CERTIFYING_CONTEXT", "RUNTIME_BINDING_REQUIRED"}),
    "RUNTIME_BINDING_REQUIRED": frozenset({"CONTEXT_READY"}),
    "CONTEXT_READY": frozenset({"NEEDS_RUNTIME_DOCUMENT", "FORMING_RUNTIME_CONTEXT"}),
    "NEEDS_RUNTIME_DOCUMENT": frozenset({"FORMING_RUNTIME_CONTEXT"}),
    "FORMING_RUNTIME_CONTEXT": frozenset({"INSPECTING"}),
    "INSPECTING": frozenset({"REVIEW_RESULTS"}),
    "REVIEW_RESULTS": frozenset({"CONTEXT_READY", "NEEDS_RUNTIME_DOCUMENT"}),
}


class TenderJourneyService:
    """Own workflow transitions; specialist services continue to own all artifacts."""

    def __init__(self, workspace_root: str | Path) -> None:
        self._store = TenderJourneyStore(workspace_root)

    def create_for_session(self, *, organization_id: str, workspace_id: str, actor_ref: str,
                           session_ref: str, command_id: str) -> TenderJourneyHeadV1:
        return self._store.create_for_session(organization_id=organization_id, workspace_id=workspace_id,
                                             actor_ref=actor_ref, session_ref=session_ref, command_id=command_id)

    def resolve_for_session(self, *, organization_id: str, workspace_id: str, actor_ref: str,
                            session_ref: str, journey_id: str | None = None
                            ) -> tuple[TenderJourneyHeadV1 | None, tuple[TenderJourneyHeadV1, ...]]:
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        binding = dict(**scope, actor_ref=actor_ref, session_ref=session_ref)
        if journey_id is not None:
            return self._store.bind_session(**binding, journey_id=journey_id), ()
        selected = self._store.get_session_journey(**binding)
        if selected is not None:
            return selected, ()
        choices = self._store.list_resumable_heads(**scope)
        if len(choices) == 1:
            return self._store.bind_session(**binding, journey_id=choices[0].journey_id), ()
        return None, choices

    def close(self) -> None:
        self._store.close()

    def create(
        self, *, organization_id: str, workspace_id: str, actor_ref: str
    ) -> TenderJourneyHeadV1:
        return self._store.create(
            organization_id=organization_id,
            workspace_id=workspace_id,
            created_by=actor_ref,
        )

    def get(
        self, *, journey_id: str, organization_id: str, workspace_id: str
    ) -> TenderJourneyHeadV1:
        return self._store.get_head(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )

    def history(
        self, *, journey_id: str, organization_id: str, workspace_id: str
    ) -> tuple[TenderJourneyTransitionV1, ...]:
        return self._store.list_transitions(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )

    def list_active(
        self, *, organization_id: str, workspace_id: str
    ) -> tuple[TenderJourneyHeadV1, ...]:
        return self._store.list_active_heads(
            organization_id=organization_id, workspace_id=workspace_id
        )

    def _requirements_access(
        self,
        *,
        ctx,
        journey_id,
        build_store,
        authority_resolver,
        authorization_class="TENANT_MEMBER",
    ):
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import OntologyBuildStore

        if not isinstance(
            authority_resolver, RuntimeAuthorityResolver
        ) or not isinstance(build_store, OntologyBuildStore):
            raise TypeError("requirements review requires production owners")
        if (
            not ctx.authenticated
            or build_store._path.resolve() != self._store.db_path.resolve()
            or self._store.db_path.resolve()
            != ontology_build_store_path(ctx.workspace_root).resolve()
        ):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        actor = RuntimeContextDTO(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            actor_id=ctx.user_id,
        )
        authority_resolver.require_current_actor(
            actor_context=actor, authorization_class=authorization_class
        )
        # Identity resolution and authorization are separate: the existing
        # resolver exposes the current governed membership, without inferring it
        # from a cached RequestContext or from a saved answer.
        _, membership = authority_resolver._current_identity_context(actor)
        if membership["membership_status"] != "active" or (
            authorization_class == "TENANT_ADMIN"
            and not membership["can_change_settings"]
        ):
            from services.runtime_errors import RuntimeContextError

            raise RuntimeContextError("FORBIDDEN_ACTOR")
        return self.get(
            journey_id=journey_id,
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
        )

    @staticmethod
    def _requirements_source_matches(
        *, head, candidate, source_role, build_store, scope
    ):
        pin = ArtifactPinV1(
            artifact_kind="normative_semantic_candidate_set",
            artifact_ref=candidate.candidate_set_ref,
            artifact_hash=candidate.candidate_set_hash,
        )
        if source_role == "ACTIVE_RULE_BASIS":
            return False  # No active rule authority may be inferred from a candidate.
        if source_role != "RUNTIME_TARGET":
            role = (
                "normative_semantics"
                if source_role == "DESIGN_SOURCE"
                else "sensitivity_normative_semantics"
            )
            return any(b.role == role and b.pin == pin for b in head.artifact_bindings)
        from contracts.ontology_workbench_context import (
            WorkspaceArtifactPreparationV1,
            WorkspaceArtifactV1,
        )

        preparations = [
            b.pin
            for b in head.artifact_bindings
            if b.role == "runtime_document_preparation"
        ]
        documents = [
            b.pin for b in head.artifact_bindings if b.role == "runtime_document"
        ]
        if len(preparations) != 1 or len(documents) != 1:
            return False
        prepared = build_store.get(
            **scope,
            artifact_ref=preparations[0].artifact_ref,
            contract_type=WorkspaceArtifactPreparationV1,
        )
        document = build_store.get(
            **scope,
            artifact_ref=documents[0].artifact_ref,
            contract_type=WorkspaceArtifactV1,
        )
        return (
            prepared is not None
            and document is not None
            and prepared.preparation_hash == preparations[0].artifact_hash
            and document.artifact_hash == documents[0].artifact_hash
            and prepared.artifact_pin == documents[0]
            and prepared.purpose == "RUNTIME_TARGET"
            and prepared.outcome == "READY"
            and prepared.source_identity_pin is not None
            and prepared.canonical_source_ir_pin is not None
            and candidate.source_identity_ref
            == prepared.source_identity_pin.artifact_ref
            and (candidate.canonical_source_ir_ref, candidate.canonical_source_ir_hash)
            == (
                prepared.canonical_source_ir_pin.artifact_ref,
                prepared.canonical_source_ir_pin.artifact_hash,
            )
            and candidate.source_hash == document.content_hash
        )

    def read_requirements(
        self,
        *,
        ctx,
        journey_id,
        build_store,
        authority_resolver,
        source_role=None,
    ):
        head = self._requirements_access(
            ctx=ctx,
            journey_id=journey_id,
            build_store=build_store,
            authority_resolver=authority_resolver,
        )
        if source_role is None:
            from contracts.requirements_reconciliation import RequirementsReviewV1

            pins = [
                b.pin for b in head.artifact_bindings if b.role == "requirements_review"
            ]
            source_role = "DESIGN_SOURCE"
            if pins:
                if len(pins) != 1:
                    raise ValueError("REQUIREMENTS_REVIEW_PIN_MISMATCH")
                selected = build_store.get(
                    organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id,
                    artifact_ref=pins[0].artifact_ref,
                    contract_type=RequirementsReviewV1,
                )
                if (
                    selected is None
                    or selected.pin() != pins[0]
                    or selected.journey_id != journey_id
                ):
                    raise ValueError("REQUIREMENTS_REVIEW_PIN_MISMATCH")
                source_role = selected.source_role
        return build_store.requirements_head(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            journey_id=journey_id,
            source_role=source_role,
        )

    def requirements_command(
        self,
        *,
        ctx,
        journey_id,
        command,
        build_store,
        authority_resolver,
        surface,
        source_registry=None,
    ):
        from contracts.requirements_reconciliation import (
            AssessRequirementsV1,
            ReviewRequirementV1,
            RequirementsReviewV1,
            RequirementDispositionV1,
        )
        from contracts.normative_semantics import NormativeSemanticCandidateSetV1
        from services.requirements_reconciliation_service import (
            RequirementsReconciliationService,
        )

        head = self._requirements_access(
            ctx=ctx,
            journey_id=journey_id,
            build_store=build_store,
            authority_resolver=authority_resolver,
        )
        scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        actor_ref = "user:" + ctx.user_id
        request_hash = "sha256:" + content_hash(
            dict(journey_id=journey_id, command=command.model_dump(mode="json"))
        )
        # Bind key to the whole semantic command, independent of transport.
        request_id = journey_id + ":" + command.idempotency_key
        saved = build_store.get_operation_result(
            **scope,
            actor_ref=actor_ref,
            operation="REQUIREMENTS_COMMAND",
            request_id=request_id,
            request_hash=request_hash,
            artifact_kind="requirements_review",
        )
        adoption_revision = command.expected_revision

        def is_adopted(pin):
            return any(b.pin == pin for b in head.artifact_bindings) or any(
                t.operation_receipt is not None and t.operation_receipt.pin == pin
                for t in self.history(**scope, journey_id=journey_id)
            )

        if saved is not None:
            review = build_store.get(
                **scope,
                artifact_ref=saved.artifact_ref,
                contract_type=RequirementsReviewV1,
            )
            if any(b.pin == saved for b in head.artifact_bindings):
                return head, review
            history = self.history(**scope, journey_id=journey_id)
            if any(
                t.operation_receipt is not None and t.operation_receipt.pin == saved
                for t in history
            ):
                return head, review
            current = build_store.requirements_head(
                **scope, journey_id=journey_id, source_role=review.source_role
            )
            if current is None or current.pin() != saved:
                raise ValueError("REQUIREMENTS_REVIEW_SUPERSEDED")
            if head.lifecycle_status != "ACTIVE":
                raise ValueError("JOURNEY_NOT_ACTIVE")
            if type(command) is AssessRequirementsV1:
                input_pin = command.candidate_pin
            else:
                original = build_store.get(
                    **scope,
                    artifact_ref=command.review_pin.artifact_ref,
                    contract_type=RequirementsReviewV1,
                )
                input_pin = ArtifactPinV1(
                    artifact_kind="normative_semantic_candidate_set",
                    artifact_ref=original.candidate.candidate_set_ref,
                    artifact_hash=original.candidate.candidate_set_hash,
                )
            input_candidate = build_store.get(
                **scope,
                artifact_ref=input_pin.artifact_ref,
                contract_type=NormativeSemanticCandidateSetV1,
            )
            if (
                input_candidate is None
                or input_candidate.candidate_set_hash != input_pin.artifact_hash
                or not self._requirements_source_matches(
                    head=head,
                    candidate=input_candidate,
                    source_role=review.source_role,
                    build_store=build_store,
                    scope=scope,
                )
            ):
                raise ValueError("REQUIREMENTS_INPUT_CHANGED")
            adoption_revision = head.revision
        else:
            if head.revision != command.expected_revision:
                raise ValueError("STALE_JOURNEY_REVISION")
            if head.lifecycle_status != "ACTIVE":
                raise ValueError("JOURNEY_NOT_ACTIVE")
            if type(command) is AssessRequirementsV1:
                if (
                    command.candidate_pin.artifact_kind
                    != "normative_semantic_candidate_set"
                ):
                    raise ValueError("REQUIREMENTS_CANDIDATE_PIN_MISMATCH")
                candidate = build_store.get(
                    **scope,
                    artifact_ref=command.candidate_pin.artifact_ref,
                    contract_type=NormativeSemanticCandidateSetV1,
                )
                if (
                    candidate is None
                    or candidate.candidate_set_hash
                    != command.candidate_pin.artifact_hash
                ):
                    raise ValueError("REQUIREMENTS_CANDIDATE_PIN_MISMATCH")
                if not self._requirements_source_matches(
                    head=head,
                    candidate=candidate,
                    source_role=command.source_role,
                    build_store=build_store,
                    scope=scope,
                ):
                    raise ValueError("REQUIREMENTS_TASK_SOURCE_MISMATCH")
                if command.source_role == "RUNTIME_TARGET":
                    from services.normative_semantic_production_service import (
                        NormativeSemanticProductionService,
                    )

                    candidate = NormativeSemanticProductionService.read_persisted(
                        store=build_store, **scope, candidate_pin=command.candidate_pin
                    )
                prior = build_store.requirements_head(
                    **scope, journey_id=journey_id, source_role=command.source_role
                )
                if (
                    prior is not None
                    and not is_adopted(prior.pin())
                    and prior.candidate.candidate_set_hash
                    == candidate.candidate_set_hash
                ):
                    raise ValueError("REQUIREMENTS_ADOPTION_REQUIRED")
                review = RequirementsReconciliationService.assess(
                    candidate=candidate,
                    journey_id=journey_id,
                    source_role=command.source_role,
                    actor_ref=actor_ref,
                    prior=prior,
                    source_evidence=RequirementsReconciliationService.persisted_source_evidence(
                        build_store, candidate, prior=prior
                    ),
                    journey_revision=command.expected_revision,
                    domain_basis=RequirementsReconciliationService.persisted_domain_basis(
                        build_store, candidate
                    ),
                    source_predecessor_ref=RequirementsReconciliationService.persisted_source_predecessor(
                        source_registry, candidate, prior
                    ),
                )
            elif type(command) is ReviewRequirementV1:
                prior = build_store.get(
                    **scope,
                    artifact_ref=command.review_pin.artifact_ref,
                    contract_type=RequirementsReviewV1,
                )
                if (
                    prior is None
                    or prior.pin() != command.review_pin
                    or prior.journey_id != journey_id
                ):
                    raise ValueError("REQUIREMENTS_REVIEW_PIN_MISMATCH")
                current = build_store.requirements_head(
                    **scope, journey_id=journey_id, source_role=prior.source_role
                )
                if current is None or current.pin() != prior.pin():
                    raise ValueError("STALE_REQUIREMENTS_REVIEW")
                if not is_adopted(prior.pin()):
                    raise ValueError("REQUIREMENTS_ADOPTION_REQUIRED")
                if not self._requirements_source_matches(
                    head=head,
                    candidate=prior.candidate,
                    source_role=prior.source_role,
                    build_store=build_store,
                    scope=scope,
                ):
                    raise ValueError("REQUIREMENTS_INPUT_CHANGED")
                q = next(
                    (
                        q
                        for q in prior.questions
                        if (q.question_id, q.dependency_hash)
                        == (command.question_id, command.dependency_hash)
                    ),
                    None,
                )
                if q is None:
                    raise ValueError("REQUIREMENTS_QUESTION_BASIS_MISMATCH")
                if any(
                    d.question_id == q.question_id and d.decision != "defer"
                    for d in prior.dispositions
                ):
                    raise ValueError("REQUIREMENTS_ANSWER_ALREADY_COMMITTED")
                candidate = prior.candidate
                if command.decision == "accept_repair":
                    if command.successor_candidate_pin is None:
                        raise ValueError("REQUIREMENTS_ACCEPTED_SUCCESSOR_REQUIRED")
                    from services.normative_semantic_production_service import (
                        NormativeSemanticProductionService,
                    )

                    successor = NormativeSemanticProductionService.read_persisted(
                        store=build_store,
                        **scope,
                        candidate_pin=command.successor_candidate_pin,
                    )
                    if (
                        successor is None
                        or successor.candidate_set_hash
                        != command.successor_candidate_pin.artifact_hash
                        or (
                            successor.predecessor_candidate_set_ref,
                            successor.predecessor_candidate_set_hash,
                        )
                        != (candidate.candidate_set_ref, candidate.candidate_set_hash)
                        or (successor.source_identity_ref, successor.source_hash)
                        != (candidate.source_identity_ref, candidate.source_hash)
                        or successor.source_review_ref is None
                    ):
                        raise ValueError("REQUIREMENTS_ACCEPTED_SUCCESSOR_REQUIRED")
                    from contracts.knowledge_classification import RuntimeContextDTO

                    authority_resolver.require_current_owner(
                        actor_context=RuntimeContextDTO(
                            organization_id=ctx.organization_id,
                            workspace_id=ctx.workspace_id,
                            actor_id=ctx.user_id,
                        )
                    )
                    candidate = successor
                elif command.successor_candidate_pin is not None:
                    raise ValueError("REQUIREMENTS_UNACCEPTED_SUCCESSOR")
                disposition = RequirementDispositionV1(
                    question_id=q.question_id,
                    dependency_hash=q.dependency_hash,
                    decision=command.decision,
                    rationale=command.rationale,
                    actor_ref=actor_ref,
                    successor_candidate_pin=command.successor_candidate_pin,
                )
                dispositions = tuple(
                    d for d in prior.dispositions if d.question_id != q.question_id
                ) + (disposition,)
                review = RequirementsReconciliationService.assess(
                    candidate=candidate,
                    journey_id=journey_id,
                    source_role=prior.source_role,
                    actor_ref=actor_ref,
                    prior=prior,
                    dispositions=dispositions,
                    source_evidence=RequirementsReconciliationService.persisted_source_evidence(
                        build_store, candidate, prior=prior
                    ),
                    journey_revision=command.expected_revision,
                    domain_basis=RequirementsReconciliationService.persisted_domain_basis(
                        build_store, candidate
                    ),
                    source_predecessor_ref=RequirementsReconciliationService.persisted_source_predecessor(
                        source_registry, candidate, prior
                    ),
                )
            else:
                raise TypeError("typed requirements command required")

            def verify(*, replay=False):
                current_head = self._requirements_access(
                    ctx=ctx,
                    journey_id=journey_id,
                    build_store=build_store,
                    authority_resolver=authority_resolver,
                    authorization_class="TENANT_ADMIN"
                    if type(command) is ReviewRequirementV1
                    and command.decision == "accept_repair"
                    else "TENANT_MEMBER",
                )
                if not replay and current_head.revision != command.expected_revision:
                    raise ValueError("STALE_JOURNEY_REVISION")

            saved = build_store.publish_requirements_review(
                review=review,
                request_id=request_id,
                request_hash=request_hash,
                verify_before_commit=verify,
            )
            review = build_store.get(
                **scope,
                artifact_ref=saved.artifact_ref,
                contract_type=RequirementsReviewV1,
            )
        # Result is committed before adoption. An exact retry never reassesses.
        self._requirements_access(
            ctx=ctx,
            journey_id=journey_id,
            build_store=build_store,
            authority_resolver=authority_resolver,
            authorization_class="TENANT_ADMIN"
            if type(command) is ReviewRequirementV1
            and command.decision == "accept_repair"
            else "TENANT_MEMBER",
        )
        binding = JourneyArtifactBindingV1(
            role="requirements_review",
            owner_kind="RequirementsReconciliationService",
            pin=saved,
        )
        bindings = (binding,)
        if (
            type(command) is AssessRequirementsV1
            and command.source_role == "RUNTIME_TARGET"
        ):
            bindings += (
                JourneyArtifactBindingV1(
                    role="runtime_normative_semantics",
                    owner_kind="NormativeSemanticProductionService",
                    pin=command.candidate_pin,
                ),
            )
        if type(command) is ReviewRequirementV1 and command.decision == "accept_repair":
            from contracts.knowledge_classification import RuntimeContextDTO

            authority_resolver.require_current_owner(
                actor_context=RuntimeContextDTO(
                    organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id,
                    actor_id=ctx.user_id,
                )
            )
            candidate_binding = JourneyArtifactBindingV1(
                role={
                    "DESIGN_SOURCE": "normative_semantics",
                    "ACTIVE_RULE_BASIS": "normative_semantics",
                    "RUNTIME_TARGET": "runtime_normative_semantics",
                    "PREPARED_SENSITIVITY": "sensitivity_normative_semantics",
                }[review.source_role],
                owner_kind="NormativeSemanticProductionService",
                pin=command.successor_candidate_pin,
            )
            bindings += (candidate_binding,)
        transition = TenderJourneyTransitionV1(
            transition_id="tender-journey-transition:" + uuid4().hex,
            journey_id=journey_id,
            predecessor_revision=adoption_revision,
            successor_revision=adoption_revision + 1,
            from_phase=head.current_phase,
            to_phase=head.current_phase,
            cause_code="REQUIREMENTS_REVIEW_COMMITTED",
            actor_ref=actor_ref,
            surface=surface,
            command_id=command.command_id,
            idempotency_key="requirements:" + command.idempotency_key,
            artifact_bindings=bindings,
            operation_receipt=binding,
            blocker_code=head.blocker_code,
            created_at=datetime.now(timezone.utc),
        )
        result = self._store.append_transition(
            **scope, expected_revision=adoption_revision, transition=transition
        )
        return result, review

    def project(self, head: TenderJourneyHeadV1) -> TenderJourneyProjectionV1:
        if head.lifecycle_status != "ACTIVE":
            return TenderJourneyProjectionV1(
                journey_id=head.journey_id,
                revision=head.revision,
                current_phase=head.current_phase,
                lifecycle_status=head.lifecycle_status,
                blocker_code=head.blocker_code,
                next_action_code="RESUME_JOURNEY"
                if head.lifecycle_status == "PAUSED"
                else "NONE",
                next_action_requires_user=head.lifecycle_status == "PAUSED",
                artifact_bindings=head.artifact_bindings,
            )
        action, requires_user = _NEXT_ACTION[head.current_phase]
        if head.blocker_code:
            action = f"RESOLVE_{head.blocker_code}"
            requires_user = True
            if head.blocker_code == "RUNTIME_INSPECTION_FAILED":
                action = "RETRY_INSPECTION"
            elif head.blocker_code == "GRAPH_INTENT_DISCOVERY_FAILED":
                action = "RETRY_GRAPH_INTENT_DISCOVERY"
        return TenderJourneyProjectionV1(
            journey_id=head.journey_id,
            revision=head.revision,
            current_phase=head.current_phase,
            lifecycle_status=head.lifecycle_status,
            blocker_code=head.blocker_code,
            next_action_code=action,
            next_action_requires_user=requires_user,
            artifact_bindings=head.artifact_bindings,
        )

    def pause(self, *, journey_id: str, organization_id: str, workspace_id: str, expected_revision: int, actor_ref: str = "system:journey", surface: TenderJourneySurface = "SYSTEM") -> TenderJourneyHeadV1:
        return self._store.set_lifecycle(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            expected_revision=expected_revision, actor_ref=actor_ref, surface=surface,
            lifecycle_status="PAUSED",
        )

    def resume(self, *, journey_id: str, organization_id: str, workspace_id: str, expected_revision: int, actor_ref: str = "system:journey", surface: TenderJourneySurface = "SYSTEM") -> TenderJourneyHeadV1:
        return self._store.set_lifecycle(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            expected_revision=expected_revision, actor_ref=actor_ref, surface=surface,
            lifecycle_status="ACTIVE",
        )

    def cancel(self, *, journey_id: str, organization_id: str, workspace_id: str, expected_revision: int, actor_ref: str = "system:journey", surface: TenderJourneySurface = "SYSTEM") -> TenderJourneyHeadV1:
        return self._store.set_lifecycle(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            expected_revision=expected_revision, actor_ref=actor_ref, surface=surface,
            lifecycle_status="CANCELLED",
        )

    def select_policy(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        artifact_pin: ArtifactPinV1, preparation_pin: ArtifactPinV1,
        preparation_service: WorkspaceArtifactPreparationService,
        command_id: str, idempotency_key: str, surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        """Bind verified source-owner output; callers only select governed artifacts."""
        if not isinstance(preparation_service, WorkspaceArtifactPreparationService):
            raise TypeError("policy selection requires the production preparation owner")
        self.get(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        if preparation_service._build._path.resolve() != self._store.db_path.resolve():
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        from services.tender_graph_preflight import require_tender_graph_preflight
        require_tender_graph_preflight(ctx=ctx, artifacts=preparation_service._artifacts)
        preparation = preparation_service.resolve(
            ctx=ctx, artifact_pin=artifact_pin, preparation_pin=preparation_pin,
            purpose="DESIGN_SOURCE",
        )
        return self.record_policy_preparation(
            journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            expected_revision=expected_revision,
            source_identity_pin=preparation.source_identity_pin,
            canonical_source_ir_pin=preparation.canonical_source_ir_pin,
            preparation_pin=preparation_pin, actor_ref=f"user:{ctx.user_id}", surface=surface,
            command_id=command_id, idempotency_key=idempotency_key,
        )

    def record_policy_preparation(
        self,
        *,
        journey_id: str,
        organization_id: str,
        workspace_id: str,
        expected_revision: int,
        source_identity_pin: ArtifactPinV1,
        canonical_source_ir_pin: ArtifactPinV1,
        preparation_pin: ArtifactPinV1,
        actor_ref: str,
        surface: TenderJourneySurface,
        command_id: str,
        idempotency_key: str,
    ) -> TenderJourneyHeadV1:
        if source_identity_pin.artifact_kind != "source_identity":
            raise ValueError("POLICY_SOURCE_IDENTITY_REQUIRED")
        if canonical_source_ir_pin.artifact_kind != "canonical_source_ir":
            raise ValueError("POLICY_CANONICAL_SOURCE_IR_REQUIRED")
        if preparation_pin.artifact_kind != "workspace_artifact_preparation":
            raise ValueError("POLICY_PREPARATION_RECEIPT_REQUIRED")
        bindings = (
            JourneyArtifactBindingV1(pin=source_identity_pin, owner_kind="WorkspaceArtifactPreparationService", role="policy_source"),
            JourneyArtifactBindingV1(pin=canonical_source_ir_pin, owner_kind="WorkspaceArtifactPreparationService", role="canonical_ir"),
        )
        receipt = JourneyArtifactBindingV1(
            pin=preparation_pin,
            owner_kind="WorkspaceArtifactPreparationService",
            role="operation_receipt",
        )
        return self.advance(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            expected_revision=expected_revision,
            to_phase="POLICY_READY",
            cause_code="GOVERNED_POLICY_SOURCE_PREPARED",
            actor_ref=actor_ref,
            surface=surface,
            command_id=command_id,
            idempotency_key=idempotency_key,
            operation_receipt=receipt,
            artifact_bindings=bindings,
        )

    def replay_intent_discovery(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        command_id: str, idempotency_key: str, surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1 | None:
        """Read a completed command without loading live inference configuration."""
        if not ctx.authenticated:
            raise ValueError("AUTHORITY_CONTEXT_REQUIRED")
        return self._store.replay_command(
            journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            idempotency_key=idempotency_key,
            command={"command_id": command_id, "actor_ref": f"user:{ctx.user_id}", "surface": surface,
                     "predecessor_revision": expected_revision + 1, "cause_code": "GRAPH_INTENT_DISCOVERY_COMPLETED"},
        )

    def discover_intent(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        capability, authority_resolver, command_id: str, idempotency_key: str,
        surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        """Propose/review through the source owner with a durable single-call claim."""
        from storage.ontology_build_store import ontology_build_store_path
        from contracts.graph_intent_brief import GraphIntentSourceBinding
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.corpus_graph_schema_capability_service import CorpusGraphSchemaCapabilityService
        from services.graph_intent_discovery_service import GraphIntentDiscoveryService
        from services.runtime_context_transition_service import RuntimeAuthorityResolver

        if not isinstance(capability, CorpusGraphSchemaCapabilityService) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("intent discovery requires production capability and authority owners")
        authority_resolver.require_current_actor(actor_context=RuntimeContextDTO(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id,
        ))
        if self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        if not ctx.authenticated or (
            capability.organization_id, capability.workspace_id, capability.user_id, capability.workspace_root
        ) != (ctx.organization_id, ctx.workspace_id, ctx.user_id, ctx.workspace_root.resolve()):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        actor_ref = f"user:{ctx.user_id}"
        identity = dict(command_id=command_id, actor_ref=actor_ref, surface=surface)
        completed = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision": expected_revision + 1,
                     "cause_code": "GRAPH_INTENT_DISCOVERY_COMPLETED"})
        if completed is not None:
            return completed
        pending = self._store.replay_command(**scope, idempotency_key=idempotency_key + ":start",
            command={**identity, "predecessor_revision": expected_revision,
                     "cause_code": "GRAPH_INTENT_DISCOVERY_STARTED"})
        head = self.get(**scope)
        if pending is not None and (head != pending or head.lifecycle_status != "ACTIVE"):
            raise ValueError("GRAPH_INTENT_DISCOVERY_PENDING_RECOVERY")
        if pending is None and head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        pins = {item.role: item.pin for item in head.artifact_bindings}
        intents = capability._graph_intent_store()
        discovery = GraphIntentDiscoveryService(ctx.workspace_root, intent_store=intents,
            profile_registry=capability.profile_registry)
        rejected_proposal_retry = False
        if head.current_phase == "REVIEW_INTENT" and "graph_intent" in pins:
            pin = pins["graph_intent"]
            prior_brief = intents.get(pin.artifact_ref, graph_intent_hash=pin.artifact_hash)
            rejected_proposal_retry = (
                prior_brief.review.state == "proposed"
                and discovery.proposal_validation_failure(prior_brief) is not None
            )
        if not rejected_proposal_retry and head.current_phase != "POLICY_READY" and not (
            head.current_phase == "BUILDING_CONTEXT" and (head.blocker_code == "GRAPH_INTENT_DISCOVERY_FAILED" or
                (pending is not None and head.blocker_code == "GRAPH_INTENT_DISCOVERY_IN_PROGRESS"))
        ):
            raise ValueError("GRAPH_INTENT_DISCOVERY_NOT_REQUIRED_IN_CURRENT_PHASE")
        if "policy_source" not in pins or "canonical_ir" not in pins:
            raise ValueError("POLICY_SOURCE_IDENTITY_REQUIRED")
        lineage, source_identity = intents.source_registry.resolve_source_identity_lineage(
            source_identity_ref=pins["policy_source"].artifact_ref,
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
        )
        if pins["policy_source"].artifact_hash != "sha256:" + content_hash(source_identity.model_dump(mode="json", by_alias=True, exclude_none=False)):
            raise ValueError("JOURNEY_POLICY_SOURCE_MISMATCH")
        intents.verify_source_binding(GraphIntentSourceBinding(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, lineage_id=lineage,
            source_identity_ref=source_identity.source_identity_ref, source_hash=source_identity.source_hash,
            source_version=source_identity.source_version, canonical_source_ir_ref=pins["canonical_ir"].artifact_ref,
            canonical_source_ir_hash=pins["canonical_ir"].artifact_hash,
        ), source_identity=source_identity)
        profile_id = "tender_corpus_schema_induction.v0_1"
        capability.profile_registry.resolve(profile_id)
        if pending is not None:
            started = pending
            claim_id = pending.latest_transition_id
        else:
            claim_id = f"tender-journey-transition:{uuid4().hex}"
            claim = TenderJourneyTransitionV1(
                transition_id=claim_id, journey_id=journey_id,
                predecessor_revision=expected_revision, successor_revision=expected_revision + 1,
                from_phase=head.current_phase, to_phase="BUILDING_CONTEXT",
                cause_code="GRAPH_INTENT_DISCOVERY_STARTED", **identity,
                idempotency_key=idempotency_key + ":start", operation_receipt=head.operation_receipt,
                blocker_code="GRAPH_INTENT_DISCOVERY_IN_PROGRESS", created_at=datetime.now(timezone.utc),
            )
            started = self._store.append_transition(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                expected_revision=expected_revision, transition=claim)
            if started.latest_transition_id != claim_id:
                raise ValueError("GRAPH_INTENT_DISCOVERY_PENDING_RECOVERY")
        try:
            if pending is not None:
                from services.graph_intent_discovery_service import GraphIntentDiscoveryService
                reader = GraphIntentDiscoveryService(ctx.workspace_root, intent_store=intents,
                    profile_registry=capability.profile_registry)
                matches = []
                for candidate in intents.list(lineage_id=lineage, source_identity_ref=source_identity.source_identity_ref):
                    source = candidate.source_binding
                    if (source.canonical_source_ir_ref, source.canonical_source_ir_hash) != (
                        pins["canonical_ir"].artifact_ref, pins["canonical_ir"].artifact_hash,
                    ):
                        continue
                    if not any(ref.startswith("evidence:graph-intent-proposal:") for ref in candidate.evidence_refs):
                        continue
                    evidence = reader.load_proposal_evidence(candidate)["llm_evidence"]
                    if evidence.get("attempt_ref") != claim_id:
                        continue
                    if (candidate.compiler_eligible or any(question.material for question in candidate.unresolved_questions)
                            or evidence["status"] != "passed"):
                        matches.append(candidate)
                if len(matches) != 1:
                    raise ValueError("GRAPH_INTENT_DISCOVERY_PENDING_RECOVERY")
                candidate = matches[0]
                envelope = capability.get_graph_intent(candidate.graph_intent_id,
                    graph_intent_hash=candidate.graph_intent_hash)
            else:
                envelope = capability.propose_graph_intent(
                    lineage_id=lineage, source_identity_ref=source_identity.source_identity_ref,
                    canonical_source_ir_ref=pins["canonical_ir"].artifact_ref,
                    profile_id=profile_id, operating_mode="auto_pilot", attempt_ref=claim_id,
                )
        except (ValueError, RuntimeError, OSError) as exc:
            if pending is not None:
                raise ValueError("GRAPH_INTENT_DISCOVERY_PENDING_RECOVERY") from exc
            return self.block(**scope, expected_revision=started.revision, blocker_code="GRAPH_INTENT_DISCOVERY_FAILED",
                cause_code="GRAPH_INTENT_DISCOVERY_COMPLETED", **identity, idempotency_key=idempotency_key)
        pin = ArtifactPinV1(artifact_kind="graph_intent_brief", artifact_ref=envelope["pins"]["graphIntentRef"],
            artifact_hash=envelope["pins"]["graphIntentHash"])
        brief = intents.get(pin.artifact_ref, graph_intent_hash=pin.artifact_hash)
        binding = JourneyArtifactBindingV1(pin=pin, owner_kind="GraphIntentBriefStore", role="graph_intent")
        if discovery.proposal_validation_failure(brief) is not None or not brief.evidence_refs:
            return self.block(**scope, expected_revision=started.revision, blocker_code="GRAPH_INTENT_DISCOVERY_FAILED",
                cause_code="GRAPH_INTENT_DISCOVERY_COMPLETED", **identity, idempotency_key=idempotency_key,
                artifact_bindings=(binding,))
        return self.advance(**scope, expected_revision=started.revision,
            to_phase="BUILDING_CONTEXT" if brief.compiler_eligible else "REVIEW_INTENT",
            cause_code="GRAPH_INTENT_DISCOVERY_COMPLETED", **identity, idempotency_key=idempotency_key,
            operation_receipt=binding, artifact_bindings=(binding,))

    def review_intent(self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
                      graph_intent_pin: ArtifactPinV1, state: str, relation_edit: dict | None,
                      selected_question_ids: list[str], capability, authority_resolver,
                      command_id: str, idempotency_key: str, surface: TenderJourneySurface):
        """Apply an exact user review through the existing typed GraphIntent owner."""
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.corpus_graph_schema_capability_service import CorpusGraphSchemaCapabilityService
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        if not isinstance(capability, CorpusGraphSchemaCapabilityService) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("intent review requires production capability and authority owners")
        authority_resolver.require_current_actor(actor_context=RuntimeContextDTO(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id))
        if (not ctx.authenticated or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve()
                or (capability.organization_id, capability.workspace_id, capability.user_id, capability.workspace_root)
                != (ctx.organization_id, ctx.workspace_id, ctx.user_id, ctx.workspace_root.resolve())):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        normalized = capability.normalize_graph_intent_review(channel="chat" if surface == "CHAT" else "direct",
            state=state, operating_mode="assisted", relation_edit=relation_edit, selected_question_ids=selected_question_ids)
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        digest = content_hash({"graph_intent_pin": graph_intent_pin.model_dump(mode="json"), "review": normalized})
        identity = dict(command_id=command_id + ":" + digest, actor_ref=f"user:{ctx.user_id}", surface=surface,
                        cause_code="GRAPH_INTENT_REVIEWED")
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision": expected_revision})
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if head.current_phase != "REVIEW_INTENT":
            raise ValueError("GRAPH_INTENT_REVIEW_NOT_REQUIRED")
        if [item.pin for item in head.artifact_bindings if item.role == "graph_intent"] != [graph_intent_pin]:
            raise ValueError("JOURNEY_GRAPH_INTENT_MISMATCH")
        envelope = capability.review_graph_intent(graph_intent_pin.artifact_ref,
            graph_intent_hash=graph_intent_pin.artifact_hash, channel="chat" if surface == "CHAT" else "direct",
            state=state, operating_mode="assisted", relation_edit=relation_edit,
            selected_question_ids=selected_question_ids)
        pin = ArtifactPinV1(artifact_kind="graph_intent_brief", artifact_ref=envelope["pins"]["graphIntentRef"],
                           artifact_hash=envelope["pins"]["graphIntentHash"])
        successor = capability._graph_intent_store().get(pin.artifact_ref, graph_intent_hash=pin.artifact_hash)
        if successor.predecessor_graph_intent_ref != graph_intent_pin.artifact_ref:
            raise ValueError("GRAPH_INTENT_REVIEW_LINEAGE_MISMATCH")
        binding = JourneyArtifactBindingV1(pin=pin, owner_kind="GraphIntentBriefStore", role="graph_intent")
        if state == "rejected":
            return self.block(**scope, expected_revision=expected_revision, **identity, idempotency_key=idempotency_key,
                blocker_code="GRAPH_INTENT_REJECTED", artifact_bindings=(binding,))
        return self.advance(**scope, expected_revision=expected_revision, **identity, idempotency_key=idempotency_key,
            to_phase="BUILDING_CONTEXT" if successor.compiler_eligible else "REVIEW_INTENT",
            operation_receipt=binding, artifact_bindings=(binding,))

    def build_context(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        graph_intent_pin: ArtifactPinV1, intent_store, build_store, authority_resolver,
        command_id: str, idempotency_key: str, surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        """Compile the journey's exact reviewed source through the existing A0 owner."""
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.ontology_build_start_service import OntologyBuildStartRequest, OntologyBuildStartService
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.graph_intent_brief_store import GraphIntentBriefStore
        from storage.ontology_build_store import OntologyBuildStore

        if not isinstance(intent_store, GraphIntentBriefStore) or not isinstance(build_store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("journey builds require production artifact and authority owners")
        authority_resolver.require_current_actor(actor_context=RuntimeContextDTO(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id,
        ))
        if (
            not ctx.authenticated
            or build_store._path.resolve() != self._store.db_path.resolve()
            or intent_store.workspace_path.resolve() != ctx.workspace_root.resolve()
            or (intent_store.organization_id, intent_store.workspace_id) != (ctx.organization_id, ctx.workspace_id)
        ):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        if graph_intent_pin.artifact_kind != "graph_intent_brief":
            raise ValueError("GRAPH_INTENT_PIN_REQUIRED")
        head = self.get(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        from services.corpus_graph_schema_profile_registry import CorpusGraphSchemaProfileRegistry
        from services.tender_graph_preflight import TENDER_PROFILE_ID
        CorpusGraphSchemaProfileRegistry().resolve(TENDER_PROFILE_ID)
        brief = intent_store.get(graph_intent_pin.artifact_ref, graph_intent_hash=graph_intent_pin.artifact_hash)
        if not brief.compiler_eligible:
            raise ValueError("GRAPH_INTENT_REVIEW_REQUIRED")
        source = brief.source_binding
        identity = intent_store.source_registry.get_source_identity(
            source_identity_ref=source.source_identity_ref, organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, lineage_id=source.lineage_id,
        )
        if identity is None:
            raise ValueError("POLICY_SOURCE_IDENTITY_REQUIRED")
        intent_store.verify_source_binding(source, source_identity=identity)
        pins = {binding.role: binding.pin for binding in head.artifact_bindings}
        if (
            pins.get("policy_source") != ArtifactPinV1(artifact_kind="source_identity",
                artifact_ref=identity.source_identity_ref,
                artifact_hash="sha256:" + content_hash(identity.model_dump(mode="json", by_alias=True, exclude_none=False)))
            or pins.get("canonical_ir") != ArtifactPinV1(artifact_kind="canonical_source_ir",
                artifact_ref=source.canonical_source_ir_ref, artifact_hash=source.canonical_source_ir_hash)
        ):
            raise ValueError("JOURNEY_POLICY_SOURCE_MISMATCH")
        intent_binding = JourneyArtifactBindingV1(pin=graph_intent_pin, owner_kind="GraphIntentBriefStore", role="graph_intent")
        common = dict(journey_id=journey_id, organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_ref=f"user:{ctx.user_id}", surface=surface)
        started = self._store.replay_command(
            journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            idempotency_key=idempotency_key + ":start",
            command={"predecessor_revision": expected_revision, "command_id": command_id + ":start",
                     "actor_ref": common["actor_ref"], "surface": surface,
                     "cause_code": "REVIEWED_GRAPH_INTENT_BUILD_STARTED",
                     "operation_receipt": intent_binding, "artifact_bindings": (intent_binding,)},
        )
        if started is None:
            started = self.advance(
                **common, expected_revision=expected_revision,
                to_phase="REVIEW_INTENT" if head.current_phase == "REVIEW_INTENT" else "BUILDING_CONTEXT",
                cause_code="REVIEWED_GRAPH_INTENT_BUILD_STARTED", command_id=command_id + ":start",
                idempotency_key=idempotency_key + ":start", operation_receipt=intent_binding,
                artifact_bindings=(intent_binding,),
            )
        # A completed command returns its original result before invoking the compiler.
        replay = self._store.replay_command(
            journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            idempotency_key=idempotency_key,
            command={"predecessor_revision": started.revision, "command_id": command_id,
                     "actor_ref": common["actor_ref"], "surface": surface,
                     "cause_code": "A0_CONTEXT_BUILD_READY"},
        )
        if replay is not None:
            return replay
        current = self.get(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        if current != started:
            raise ValueError("STALE_JOURNEY_REVISION")
        if current.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        manifest = OntologyBuildStartService().start(
            command=OntologyBuildStartRequest(graph_intent_ref=brief.graph_intent_id, graph_intent_hash=brief.graph_intent_hash),
            intent_store=intent_store, build_store=build_store,
        )
        build_binding = JourneyArtifactBindingV1(
            pin=ArtifactPinV1(artifact_kind="ontology_build_manifest", artifact_ref=manifest.build_ref, artifact_hash=manifest.build_hash),
            owner_kind="OntologyBuildStore", role="context_build",
        )
        return self.advance(
            **common, expected_revision=started.revision, to_phase="CERTIFYING_CONTEXT",
            cause_code="A0_CONTEXT_BUILD_READY", command_id=command_id, idempotency_key=idempotency_key,
            operation_receipt=build_binding, artifact_bindings=(intent_binding, build_binding),
        )

    def record_context_build(
        self,
        *,
        journey_id: str,
        organization_id: str,
        workspace_id: str,
        expected_revision: int,
        graph_intent_pin: ArtifactPinV1,
        build_pin: ArtifactPinV1,
        build_receipt_pin: ArtifactPinV1,
        actor_ref: str,
        surface: TenderJourneySurface,
        command_id: str,
        idempotency_key: str,
        material_review_required: bool = False,
    ) -> TenderJourneyHeadV1:
        bindings = (
            JourneyArtifactBindingV1(pin=graph_intent_pin, owner_kind="knowledge-graph-building", role="graph_intent"),
            JourneyArtifactBindingV1(pin=build_pin, owner_kind="OntologyBuildStore", role="context_build"),
        )
        receipt = JourneyArtifactBindingV1(pin=build_receipt_pin, owner_kind="knowledge-graph-building", role="operation_receipt")
        return self.advance(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            expected_revision=expected_revision,
            to_phase="REVIEW_INTENT" if material_review_required else "CERTIFYING_CONTEXT",
            cause_code="MATERIAL_GRAPH_INTENT_REVIEW_REQUIRED" if material_review_required else "A0_CONTEXT_BUILD_READY",
            actor_ref=actor_ref,
            surface=surface,
            command_id=command_id,
            idempotency_key=idempotency_key,
            operation_receipt=receipt,
            artifact_bindings=bindings,
        )

    def review_competency(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        competency_pin: ArtifactPinV1, question_id: str, decision_kind: str, rationale: str,
        build_store, authority_resolver, command_id: str, idempotency_key: str,
        surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from services.competency_contract_service import CompetencyContractService
        from storage.ontology_build_store import OntologyBuildStore
        if not isinstance(build_store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("competency review requires production owners")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor)
        if not ctx.authenticated or build_store._path.resolve() != self._store.db_path.resolve() or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        if surface not in {"CHAT", "WORKBENCH"}:
            raise ValueError("HUMAN_REVIEW_SURFACE_REQUIRED")
        digest = content_hash(dict(competency=competency_pin.model_dump(mode="json"),
            question_id=question_id, decision_kind=decision_kind, rationale=rationale))
        identity = dict(command_id=command_id + ":" + digest, actor_ref=f"user:{ctx.user_id}",
                        surface=surface, cause_code="COMPETENCY_REVIEWED")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision":expected_revision})
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if head.current_phase != "CERTIFYING_CONTEXT":
            raise ValueError("CERTIFICATION_NOT_REQUIRED_IN_CURRENT_PHASE")
        pins = [item.pin for item in head.artifact_bindings if item.role == "competency_contract"]
        if pins != [competency_pin]:
            raise ValueError("JOURNEY_COMPETENCY_PIN_MISMATCH")
        result = CompetencyContractService().review_persisted(store=build_store,
            authority_resolver=authority_resolver, actor_context=actor, competency_pin=competency_pin,
            question_id=question_id, decision_kind=decision_kind, rationale=rationale,
            channel="chat" if surface == "CHAT" else "direct_ui", decided_at=datetime.now(timezone.utc),
            request_id="journey-review-competency:" + content_hash(dict(journey_id=journey_id,
                idempotency_key=idempotency_key, command_id=identity["command_id"])))
        binding = JourneyArtifactBindingV1(role="competency_contract", owner_kind="CompetencyContractService",
            pin=ArtifactPinV1(artifact_kind="competency_contract", artifact_ref=result.competency_contract_ref,
                             artifact_hash=result.competency_contract_hash))
        return self.advance(**scope, expected_revision=expected_revision, to_phase="CERTIFYING_CONTEXT",
            **identity, idempotency_key=idempotency_key, operation_receipt=binding, artifact_bindings=(binding,))

    def propose_competency(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        intent_store, build_store, authority_resolver, command_id: str,
        idempotency_key: str, surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from services.competency_contract_service import CompetencyContractService
        from storage.ontology_build_store import OntologyBuildStore
        if not isinstance(build_store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("competency preparation requires production owners")
        authority_resolver.require_current_actor(actor_context=RuntimeContextDTO(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id))
        if not ctx.authenticated or build_store._path.resolve() != self._store.db_path.resolve() or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        identity = dict(command_id=command_id, actor_ref=f"user:{ctx.user_id}",
                        surface=surface, cause_code="COMPETENCY_PROPOSED")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision":expected_revision})
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if head.current_phase != "CERTIFYING_CONTEXT":
            raise ValueError("CERTIFICATION_NOT_REQUIRED_IN_CURRENT_PHASE")
        from contracts.ontology_building import CompetencyContractV1
        for existing in head.artifact_bindings:
            if existing.role != "competency_contract":
                continue
            contract = build_store.get(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                artifact_ref=existing.pin.artifact_ref, contract_type=CompetencyContractV1)
            if contract is None or contract.competency_contract_hash != existing.pin.artifact_hash:
                raise ValueError("COMPETENCY_PIN_MISMATCH")
            if contract.predecessor_ref is not None or any(item.review_state != "unresolved" for item in contract.questions):
                raise ValueError("COMPETENCY_REVIEW_ALREADY_RECORDED")
        builds = [item.pin for item in head.artifact_bindings if item.role == "context_build"]
        if len(builds) != 1:
            raise ValueError("CERTIFICATION_ORIGIN_REQUIRED")
        result = CompetencyContractService().propose_persisted(store=build_store, intent_store=intent_store,
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, origin_pin=builds[0],
            actor_ref=identity["actor_ref"], request_id="journey-competency:" + content_hash(dict(
                journey_id=journey_id, idempotency_key=idempotency_key, command_id=command_id)),
            created_at=datetime.now(timezone.utc))
        binding = JourneyArtifactBindingV1(role="competency_contract", owner_kind="CompetencyContractService",
            pin=ArtifactPinV1(artifact_kind="competency_contract", artifact_ref=result.competency_contract_ref,
                             artifact_hash=result.competency_contract_hash))
        return self.advance(**scope, expected_revision=expected_revision, to_phase="CERTIFYING_CONTEXT",
            **identity, idempotency_key=idempotency_key, operation_receipt=binding, artifact_bindings=(binding,))

    async def produce_normative_schema(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        schema_decision_pin: ArtifactPinV1, capability, producer,
        build_store, authority_resolver, command_id: str, idempotency_key: str,
        surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.ontology_building import CompetencyContractV1
        from services.corpus_graph_schema_capability_service import CorpusGraphSchemaCapabilityService
        from services.normative_schema_production_service import NormativeSchemaProductionService
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import OntologyBuildStore
        if (not isinstance(build_store, OntologyBuildStore)
            or not isinstance(authority_resolver, RuntimeAuthorityResolver)
            or not isinstance(capability, CorpusGraphSchemaCapabilityService)
            or not isinstance(producer, NormativeSchemaProductionService)):
            raise TypeError("normative schema preparation requires production owners")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor)
        if (not ctx.authenticated or build_store._path.resolve() != self._store.db_path.resolve()
            or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve()
            or (capability.organization_id, capability.workspace_id, capability.user_id)
                != (ctx.organization_id, ctx.workspace_id, ctx.user_id)):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        identity = dict(command_id=command_id + ":" + content_hash(schema_decision_pin.model_dump(mode="json")),
            actor_ref=f"user:{ctx.user_id}", surface=surface, cause_code="NORMATIVE_SCHEMA_PRODUCED")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision": expected_revision})
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if head.current_phase != "CERTIFYING_CONTEXT":
            raise ValueError("CERTIFICATION_NOT_REQUIRED_IN_CURRENT_PHASE")
        pins = [item.pin for item in head.artifact_bindings if item.role == "competency_contract"]
        if len(pins) != 1:
            raise ValueError("JOURNEY_COMPETENCY_PIN_MISMATCH")
        competency = build_store.get(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            artifact_ref=pins[0].artifact_ref, contract_type=CompetencyContractV1)
        intents = [item.pin for item in head.artifact_bindings if item.role == "graph_intent"]
        if (competency is None or competency.competency_contract_hash != pins[0].artifact_hash
            or len(intents) != 1 or (intents[0].artifact_ref, intents[0].artifact_hash)
                != (competency.graph_intent_brief_ref, competency.graph_intent_brief_hash)):
            raise ValueError("JOURNEY_COMPETENCY_INTENT_MISMATCH")
        authority_pins = [item.pin for item in head.artifact_bindings if item.role == "authority_dependencies"]
        coverage_pins = [item.pin for item in head.artifact_bindings if item.role == "source_search_coverage"]
        if len(authority_pins) > 1 or len(coverage_pins) > 1:
            raise ValueError("NORMATIVE_LEGAL_SOURCE_BINDING_AMBIGUOUS")
        parent_inputs = {}
        if coverage_pins and not authority_pins:
            raise ValueError("NORMATIVE_LEGAL_SOURCE_AUTHORITY_REQUIRED")
        if authority_pins:
            if not coverage_pins:
                raise ValueError("NORMATIVE_LEGAL_SOURCE_COVERAGE_REQUIRED")
            from eos.legal_corpus_store import LegalCorpusStore
            parent_inputs = dict(authority_pin=authority_pins[0], coverage_pin=coverage_pins[0],
                legal_store=LegalCorpusStore())
        schema, report = await producer.produce_persisted(store=build_store, capability=capability,
            competency_pin=pins[0], schema_decision_pin=schema_decision_pin, **parent_inputs,
            request_id="journey-normative-schema:" + content_hash(dict(journey_id=journey_id,
                idempotency_key=idempotency_key, command_id=identity["command_id"])),
            created_at=datetime.now(timezone.utc))
        bindings = (
            JourneyArtifactBindingV1(role="normative_schema", owner_kind="NormativeSchemaProductionService",
                pin=ArtifactPinV1(artifact_kind="normative_schema_contract", artifact_ref=schema.schema_ref,
                    artifact_hash=schema.schema_hash)),
            JourneyArtifactBindingV1(role="schema_adequacy", owner_kind="OntologySchemaAdequacyService",
                pin=ArtifactPinV1(artifact_kind="schema_adequacy_report", artifact_ref=report.report_ref,
                    artifact_hash=report.report_hash)),
        )
        authority_resolver.require_current_actor(actor_context=actor)
        return self.advance(**scope, expected_revision=expected_revision, to_phase="CERTIFYING_CONTEXT",
            **identity, idempotency_key=idempotency_key, operation_receipt=bindings[0], artifact_bindings=bindings)

    def review_normative_precedence_mapping(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        schema_pin: ArtifactPinV1, schema_object_ref: str, binding_record_hash: str,
        source_evidence_ref: str, package_hash: str, rationale: str, reviewed_at: datetime,
        build_store, authority_resolver, command_id: str, idempotency_key: str,
        surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.normative_schema_production_service import NormativeSchemaProductionService
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import OntologyBuildStore
        if not isinstance(build_store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("mapping review requires production owners")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor, authorization_class="TENANT_ADMIN")
        if (not ctx.authenticated or build_store._path.resolve() != self._store.db_path.resolve()
            or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve()):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        inputs = dict(schema_pin=schema_pin, schema_object_ref=schema_object_ref,
            binding_record_hash=binding_record_hash, source_evidence_ref=source_evidence_ref,
            package_hash=package_hash, rationale=rationale, reviewed_at=reviewed_at)
        digest = content_hash({**inputs, "schema_pin":schema_pin.model_dump(mode="json"),
            "reviewed_at":reviewed_at.isoformat()})
        identity = dict(actor_ref="user:"+ctx.user_id, surface=surface,
            command_id=command_id+":"+digest, cause_code="NORMATIVE_PRECEDENCE_MAPPING_REVIEWED")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision":expected_revision})
        if replay is not None:
            pin = next(b.pin for b in replay.artifact_bindings if b.role == "normative_schema")
            NormativeSchemaProductionService.read_persisted(store=build_store,
                organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, schema_pin=pin)
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE" or head.current_phase != "CERTIFYING_CONTEXT":
            raise ValueError("CERTIFICATION_NOT_REQUIRED_IN_CURRENT_PHASE")
        if [b.pin for b in head.artifact_bindings if b.role == "normative_schema"] != [schema_pin]:
            raise ValueError("JOURNEY_NORMATIVE_SCHEMA_PIN_MISMATCH")
        original, _ = NormativeSchemaProductionService.read_persisted(store=build_store,
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, schema_pin=schema_pin)
        competency = [b.pin for b in head.artifact_bindings if b.role == "competency_contract"]
        if len(competency) != 1 or (competency[0].artifact_ref, competency[0].artifact_hash) != (
            original.competency_contract_ref, original.competency_contract_hash):
            raise ValueError("JOURNEY_COMPETENCY_PIN_MISMATCH")
        schema, report = NormativeSchemaProductionService.review_precedence_mapping_persisted(
            store=build_store, authority_resolver=authority_resolver, ctx=ctx, **inputs,
            request_id="journey-precedence-review:"+content_hash(dict(journey_id=journey_id,
                idempotency_key=idempotency_key, command_id=identity["command_id"])))
        bindings = (
            JourneyArtifactBindingV1(role="normative_schema", owner_kind="NormativeSchemaProductionService",
                pin=ArtifactPinV1(artifact_kind="normative_schema_contract",
                    artifact_ref=schema.schema_ref, artifact_hash=schema.schema_hash)),
            JourneyArtifactBindingV1(role="schema_adequacy", owner_kind="OntologySchemaAdequacyService",
                pin=ArtifactPinV1(artifact_kind="schema_adequacy_report",
                    artifact_ref=report.report_ref, artifact_hash=report.report_hash)),
        )
        authority_resolver.require_current_actor(actor_context=actor, authorization_class="TENANT_ADMIN")
        return self.advance(**scope, expected_revision=expected_revision, to_phase="CERTIFYING_CONTEXT",
            **identity, idempotency_key=idempotency_key, operation_receipt=bindings[0], artifact_bindings=bindings)

    async def produce_normative_semantics(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        producer, build_store, authority_resolver, command_id: str,
        idempotency_key: str, surface: TenderJourneySurface, source_registry=None,
        source_role="DESIGN_SOURCE", preparation_service=None,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.ontology_building import CompetencyContractV1
        from services.normative_schema_production_service import NormativeSchemaProductionService
        from services.normative_semantic_production_service import NormativeSemanticProductionService
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import OntologyBuildStore
        if (not isinstance(build_store, OntologyBuildStore)
            or not isinstance(authority_resolver, RuntimeAuthorityResolver)
            or not isinstance(producer, NormativeSemanticProductionService)):
            raise TypeError("normative semantic preparation requires production owners")
        self._requirements_access(ctx=ctx,journey_id=journey_id,build_store=build_store,authority_resolver=authority_resolver)
        actor = RuntimeContextDTO(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor)
        if (not ctx.authenticated or build_store._path.resolve() != self._store.db_path.resolve()
            or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve()):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        if source_role not in {"DESIGN_SOURCE", "RUNTIME_TARGET"}:
            raise ValueError("NORMATIVE_SOURCE_ROLE_UNSUPPORTED")
        identity = dict(command_id=command_id if source_role == "DESIGN_SOURCE" else command_id + ":runtime-target", actor_ref=f"user:{ctx.user_id}",
            surface=surface, cause_code="NORMATIVE_SEMANTICS_PRODUCED")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision": expected_revision})
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if (
            source_role == "DESIGN_SOURCE"
            and head.current_phase != "CERTIFYING_CONTEXT"
        ):
            raise ValueError("CERTIFICATION_NOT_REQUIRED_IN_CURRENT_PHASE")
        selected = {}
        for role in ("normative_schema", "competency_contract", "graph_intent"):
            pins = [item.pin for item in head.artifact_bindings if item.role == role]
            if len(pins) != 1:
                raise ValueError("JOURNEY_SEMANTIC_INPUT_MISMATCH")
            selected[role] = pins[0]
        artifact_scope = dict(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id
        )
        schema, _ = NormativeSchemaProductionService.read_persisted(
            store=build_store, **artifact_scope, schema_pin=selected["normative_schema"]
        )
        competency_pin, intent_pin = (
            selected["competency_contract"],
            selected["graph_intent"],
        )
        competency = build_store.get(
            **artifact_scope,
            artifact_ref=competency_pin.artifact_ref,
            contract_type=CompetencyContractV1,
        )
        if (
            competency is None
            or competency.competency_contract_hash != competency_pin.artifact_hash
            or (schema.competency_contract_ref, schema.competency_contract_hash)
            != (competency_pin.artifact_ref, competency_pin.artifact_hash)
            or (competency.graph_intent_brief_ref, competency.graph_intent_brief_hash)
            != (intent_pin.artifact_ref, intent_pin.artifact_hash)
        ):
            raise ValueError("JOURNEY_SEMANTIC_INPUT_MISMATCH")
        runtime_inputs = {}
        if source_role == "RUNTIME_TARGET":
            if (
                not isinstance(preparation_service, WorkspaceArtifactPreparationService)
                or preparation_service._build._path.resolve()
                != build_store._path.resolve()
            ):
                raise TypeError(
                    "runtime requirements need the current preparation owner"
                )
            documents = [
                b.pin for b in head.artifact_bindings if b.role == "runtime_document"
            ]
            preparations = [
                b.pin
                for b in head.artifact_bindings
                if b.role == "runtime_document_preparation"
            ]
            if len(documents) != 1 or len(preparations) != 1:
                raise ValueError("RUNTIME_DOCUMENT_REQUIRED")
            prepared = preparation_service.resolve(
                ctx=ctx,
                artifact_pin=documents[0],
                preparation_pin=preparations[0],
                purpose="RUNTIME_TARGET",
            )
            source = preparation_service.resolve_runtime_target(
                ctx=ctx, artifact_pin=documents[0], preparation_pin=preparations[0]
            )
            runtime_inputs = dict(runtime_preparation=prepared, runtime_source=source)
        candidate = await producer.produce_persisted(
            store=build_store,
            **artifact_scope,
            actor_ref=identity["actor_ref"],
            schema_pin=selected["normative_schema"],
            request_id="journey-normative-semantics:"
            + content_hash(
                dict(
                    journey_id=journey_id,
                    idempotency_key=idempotency_key,
                    command_id=identity["command_id"],
                )
            ),
            created_at=datetime.now(timezone.utc),
            **runtime_inputs,
            verify_before_commit=lambda: self._requirements_access(
                ctx=ctx,
                journey_id=journey_id,
                build_store=build_store,
                authority_resolver=authority_resolver,
            ),
        )
        binding = JourneyArtifactBindingV1(
            role="normative_semantics"
            if source_role == "DESIGN_SOURCE"
            else "runtime_normative_semantics",
            owner_kind="NormativeSemanticProductionService",
            pin=ArtifactPinV1(
                artifact_kind="normative_semantic_candidate_set",
                artifact_ref=candidate.candidate_set_ref,
                artifact_hash=candidate.candidate_set_hash,
            ),
        )
        from services.requirements_reconciliation_service import (
            RequirementsReconciliationService,
        )

        prior = build_store.requirements_head(
            **artifact_scope, journey_id=journey_id, source_role=source_role
        )
        review_request = (
            "source-requirements:"
            if source_role == "DESIGN_SOURCE"
            else "runtime-source-requirements:"
        ) + content_hash(
            dict(journey_id=journey_id, candidate=candidate.candidate_set_hash)
        )
        review_hash = "sha256:" + content_hash(
            dict(candidate=candidate.candidate_set_hash, source_role=source_role)
        )
        review_pin = build_store.get_operation_result(
            **artifact_scope,
            actor_ref=identity["actor_ref"],
            operation="REQUIREMENTS_COMMAND",
            request_id=review_request,
            request_hash=review_hash,
            artifact_kind="requirements_review",
        )
        if review_pin is None:
            review = RequirementsReconciliationService.assess(
                candidate=candidate,
                journey_id=journey_id,
                source_role=source_role,
                actor_ref=identity["actor_ref"],
                prior=prior,
                source_evidence=RequirementsReconciliationService.persisted_source_evidence(
                    build_store, candidate, prior=prior
                ),
                journey_revision=expected_revision,
                domain_basis=RequirementsReconciliationService.persisted_domain_basis(
                    build_store, candidate
                ),
                source_predecessor_ref=RequirementsReconciliationService.persisted_source_predecessor(
                    source_registry, candidate, prior
                ),
            )
            review_pin = build_store.publish_requirements_review(
                review=review,
                request_id=review_request,
                request_hash=review_hash,
                verify_before_commit=lambda **kw: self._requirements_access(
                    ctx=ctx,
                    journey_id=journey_id,
                    build_store=build_store,
                    authority_resolver=authority_resolver,
                ),
            )
        review_binding = JourneyArtifactBindingV1(
            role="requirements_review",
            owner_kind="RequirementsReconciliationService",
            pin=review_pin,
        )
        authority_resolver.require_current_actor(actor_context=actor)
        if source_role == "RUNTIME_TARGET":
            self._requirements_access(
                ctx=ctx,
                journey_id=journey_id,
                build_store=build_store,
                authority_resolver=authority_resolver,
            )
            transition = TenderJourneyTransitionV1(
                transition_id="tender-journey-transition:" + uuid4().hex,
                journey_id=journey_id,
                predecessor_revision=expected_revision,
                successor_revision=expected_revision + 1,
                from_phase=head.current_phase,
                to_phase=head.current_phase,
                **identity,
                idempotency_key=idempotency_key,
                operation_receipt=binding,
                artifact_bindings=(binding, review_binding),
                created_at=datetime.now(timezone.utc),
            )
            return self._store.append_transition(
                **artifact_scope, expected_revision=expected_revision, transition=transition
            )
        return self.advance(
            **scope,
            expected_revision=expected_revision,
            to_phase="CERTIFYING_CONTEXT",
            **identity,
            idempotency_key=idempotency_key,
            operation_receipt=binding,
            artifact_bindings=(binding, review_binding),
        )

    async def propose_validation_cases(
        self,
        *,
        ctx: RequestContext,
        journey_id: str,
        expected_revision: int,
        producer,
        build_store,
        authority_resolver,
        command_id: str,
        idempotency_key: str,
        surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.ontology_building import CompetencyContractV1
        from services.normative_schema_production_service import (
            NormativeSchemaProductionService,
        )
        from services.source_validation_case_production_service import (
            SourceValidationCaseProductionService,
        )
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import OntologyBuildStore

        if (
            not isinstance(build_store, OntologyBuildStore)
            or not isinstance(authority_resolver, RuntimeAuthorityResolver)
            or not isinstance(producer, SourceValidationCaseProductionService)
        ):
            raise TypeError("validation case proposal requires production owners")
        actor = RuntimeContextDTO(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            actor_id=ctx.user_id,
        )
        authority_resolver.require_current_actor(actor_context=actor)
        if (
            not ctx.authenticated
            or build_store._path.resolve() != self._store.db_path.resolve()
            or self._store.db_path.resolve()
            != ontology_build_store_path(ctx.workspace_root).resolve()
        ):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        identity = dict(
            command_id=command_id,
            actor_ref=f"user:{ctx.user_id}",
            surface=surface,
            cause_code="VALIDATION_CASES_PROPOSED",
        )
        scope = dict(
            journey_id=journey_id,
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
        )
        replay = self._store.replay_command(
            **scope,
            idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision": expected_revision},
        )
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if head.current_phase != "CERTIFYING_CONTEXT":
            raise ValueError("CERTIFICATION_NOT_REQUIRED_IN_CURRENT_PHASE")
        selected = {}
        for role in ("normative_schema", "competency_contract", "graph_intent"):
            pins = [item.pin for item in head.artifact_bindings if item.role == role]
            if len(pins) != 1:
                raise ValueError("JOURNEY_VALIDATION_INPUT_MISMATCH")
            selected[role] = pins[0]
        artifact_scope = dict(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id
        )
        schema, _ = NormativeSchemaProductionService.read_persisted(
            store=build_store, **artifact_scope, schema_pin=selected["normative_schema"]
        )
        competency_pin, intent_pin = (
            selected["competency_contract"],
            selected["graph_intent"],
        )
        competency = build_store.get(
            **artifact_scope,
            artifact_ref=competency_pin.artifact_ref,
            contract_type=CompetencyContractV1,
        )
        if (
            competency is None
            or competency.competency_contract_hash != competency_pin.artifact_hash
            or (schema.competency_contract_ref, schema.competency_contract_hash)
            != (competency_pin.artifact_ref, competency_pin.artifact_hash)
            or (competency.graph_intent_brief_ref, competency.graph_intent_brief_hash)
            != (intent_pin.artifact_ref, intent_pin.artifact_hash)
        ):
            raise ValueError("JOURNEY_VALIDATION_INPUT_MISMATCH")
        receipt, _ = await producer.produce_persisted(
            store=build_store,
            **artifact_scope,
            actor_ref=identity["actor_ref"],
            schema_pin=selected["normative_schema"],
            request_id="journey-validation-cases:"
            + content_hash(
                dict(
                    journey_id=journey_id,
                    idempotency_key=idempotency_key,
                    command_id=command_id,
                )
            ),
            created_at=datetime.now(timezone.utc),
        )
        binding = JourneyArtifactBindingV1(
            role="validation_case_proposal",
            owner_kind="SourceValidationCaseProductionService",
            pin=ArtifactPinV1(
                artifact_kind="source_validation_proposal_evidence",
                artifact_ref=receipt.evidence_ref,
                artifact_hash=receipt.evidence_hash,
            ),
        )
        authority_resolver.require_current_actor(actor_context=actor)
        return self.advance(
            **scope,
            expected_revision=expected_revision,
            to_phase="CERTIFYING_CONTEXT",
            **identity,
            idempotency_key=idempotency_key,
            operation_receipt=binding,
            artifact_bindings=(binding,),
        )

    @staticmethod
    def source_validation_proposal_is_current(head, receipt) -> bool:
        """A retained review only applies to its exact current source selection."""
        import json

        schema = json.loads(receipt.input_text)["normative_schema"]
        expected = {
            "normative_schema": (
                "normative_schema_contract",
                schema["schema_ref"],
                schema["schema_hash"],
            ),
            "competency_contract": (
                "competency_contract",
                schema["competency_contract_ref"],
                schema["competency_contract_hash"],
            ),
            "graph_intent": (
                "graph_intent_brief",
                schema["graph_intent_brief_ref"],
                schema["graph_intent_brief_hash"],
            ),
            "validation_case_proposal": (
                "source_validation_proposal_evidence",
                receipt.evidence_ref,
                receipt.evidence_hash,
            ),
        }
        for role, values in expected.items():
            pins = [binding.pin for binding in head.artifact_bindings if binding.role == role]
            if len(pins) != 1 or (pins[0].artifact_kind, pins[0].artifact_ref, pins[0].artifact_hash) != values:
                return False
        return (head.organization_id, head.workspace_id) == (receipt.organization_id, receipt.workspace_id)

    def review_validation_cases(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        proposal_pin: ArtifactPinV1, decision: str, rationale: str,
        build_store, authority_resolver, command_id: str, idempotency_key: str,
        surface: TenderJourneySurface, case_review_pins: tuple[ArtifactPinV1, ...] = (), case_rule_expectations=None,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from services.source_validation_case_production_service import SourceValidationCaseProductionService
        from services.validation_contract_review_service import ValidationContractReviewService
        from storage.ontology_build_store import OntologyBuildStore
        if not isinstance(build_store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("validation case review requires production owners")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor)
        if (not ctx.authenticated or build_store._path.resolve() != self._store.db_path.resolve()
            or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve()):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        selected_reviews = tuple(ArtifactPinV1.model_validate(pin) for pin in case_review_pins)
        case_basis = {"case_review_pins": [pin.model_dump(mode="json") for pin in selected_reviews]} if selected_reviews else {}
        if case_rule_expectations is not None:
            from pydantic import TypeAdapter
            from contracts.validation_contract import CaseRuleExpectations
            case_basis["case_rule_expectations"] = TypeAdapter(CaseRuleExpectations).validate_python(case_rule_expectations)
        digest = content_hash(dict(proposal=proposal_pin.model_dump(mode="json"), decision=decision, rationale=rationale, **case_basis))
        identity = dict(command_id=command_id + ":" + digest, actor_ref="user:" + ctx.user_id,
            surface=surface, cause_code="VALIDATION_CASES_REVIEWED")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision": expected_revision})
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if head.current_phase != "CERTIFYING_CONTEXT":
            raise ValueError("CERTIFICATION_NOT_REQUIRED_IN_CURRENT_PHASE")
        selected = [binding.pin for binding in head.artifact_bindings if binding.role == "validation_case_proposal"]
        if selected != [proposal_pin]:
            raise ValueError("JOURNEY_VALIDATION_PROPOSAL_MISMATCH")
        receipt, _ = SourceValidationCaseProductionService.read_persisted(store=build_store,
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, proposal_pin=proposal_pin)
        if not self.source_validation_proposal_is_current(head, receipt):
            raise ValueError("JOURNEY_VALIDATION_SOURCE_MISMATCH")
        review, _ = ValidationContractReviewService().decide_source_cases_persisted(ctx=ctx,
            authority_resolver=authority_resolver, store=build_store, proposal_pin=proposal_pin,
            case_review_pins=selected_reviews, case_rule_expectations=case_basis.get("case_rule_expectations"),
            decision=decision, rationale=rationale, decided_at=datetime.now(timezone.utc),
            request_id="journey-validation-review:" + content_hash(dict(journey_id=journey_id,
                idempotency_key=idempotency_key, command_id=identity["command_id"])))
        binding = JourneyArtifactBindingV1(role="validation_case_review", owner_kind="ValidationContractReviewService",
            pin=ArtifactPinV1(artifact_kind="source_validation_review_decision",
                artifact_ref=review.decision_ref, artifact_hash=review.decision_hash))
        authority_resolver.require_current_actor(actor_context=actor)
        return self.advance(**scope, expected_revision=expected_revision, to_phase="CERTIFYING_CONTEXT",
            **identity, idempotency_key=idempotency_key, operation_receipt=binding, artifact_bindings=(binding,))

    @classmethod
    def resolve_validation_review(cls, *, head, build_store):
        from services.source_validation_case_production_service import SourceValidationCaseProductionService
        from services.validation_contract_review_service import ValidationContractReviewService
        review_pins = [binding.pin for binding in head.artifact_bindings if binding.role == "validation_case_review"]
        if not review_pins:
            return None, None, "not_reviewed"
        if len(review_pins) != 1:
            raise ValueError("VALIDATION_REVIEW_PIN_MISMATCH")
        scope = dict(organization_id=head.organization_id, workspace_id=head.workspace_id)
        review, contract = ValidationContractReviewService.read_source_decision_persisted(
            store=build_store, **scope, decision_pin=review_pins[0])
        case_pins = [binding.pin for binding in head.artifact_bindings if binding.role == "validation_case_proposal"]
        if len(case_pins) != 1 or (review.proposal_ref, review.proposal_hash) != (
            case_pins[0].artifact_ref, case_pins[0].artifact_hash):
            return review, None, "stale"
        receipt, _ = SourceValidationCaseProductionService.read_persisted(store=build_store, **scope,
            proposal_pin=case_pins[0])
        if not cls.source_validation_proposal_is_current(head, receipt):
            return review, None, "stale"
        return review, contract, review.decision

    def validate_source_semantics(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        build_store, authority_resolver, command_id: str, idempotency_key: str,
        surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from services.ontology_quality_evaluation_service import QualityEvaluationService
        from storage.ontology_build_store import OntologyBuildStore
        if not isinstance(build_store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("source validation requires production owners")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor)
        if (not ctx.authenticated or build_store._path.resolve() != self._store.db_path.resolve()
            or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve()):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        identity = dict(command_id=command_id, actor_ref="user:" + ctx.user_id,
            surface=surface, cause_code="SOURCE_SEMANTICS_VALIDATED")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision": expected_revision})
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if head.current_phase != "CERTIFYING_CONTEXT":
            raise ValueError("CERTIFICATION_NOT_REQUIRED_IN_CURRENT_PHASE")
        review, contract, status = self.resolve_validation_review(head=head, build_store=build_store)
        if status != "accepted" or contract is None:
            raise ValueError("JOURNEY_VALIDATION_REVIEW_REQUIRED")
        candidates = [binding.pin for binding in head.artifact_bindings if binding.role == "normative_semantics"]
        if len(candidates) != 1:
            raise ValueError("NORMATIVE_SEMANTIC_CANDIDATE_REQUIRED")
        authority_pins = [binding.pin for binding in head.artifact_bindings if binding.role == "authority_dependencies"]
        if len(authority_pins) > 1:
            raise ValueError("CERTIFICATION_AUTHORITY_AMBIGUOUS")
        receipt, _ = QualityEvaluationService().run_source_persisted(ctx=ctx,
            authority_pin=authority_pins[0] if authority_pins else None,
            authority_resolver=authority_resolver, store=build_store, candidate_pin=candidates[0],
            review_pin=ArtifactPinV1(artifact_kind="source_validation_review_decision",
                artifact_ref=review.decision_ref, artifact_hash=review.decision_hash),
            request_id="journey-source-validation:" + content_hash(dict(journey_id=journey_id,
                idempotency_key=idempotency_key, command_id=command_id)), executed_at=datetime.now(timezone.utc))
        binding = JourneyArtifactBindingV1(role="source_validation_execution", owner_kind="QualityEvaluationService",
            pin=ArtifactPinV1(artifact_kind="source_validation_execution",
                artifact_ref=receipt.execution_ref, artifact_hash=receipt.execution_hash))
        authority_resolver.require_current_actor(actor_context=actor)
        return self.advance(**scope, expected_revision=expected_revision, to_phase="CERTIFYING_CONTEXT",
            **identity, idempotency_key=idempotency_key, operation_receipt=binding, artifact_bindings=(binding,))

    @classmethod
    def resolve_source_validation(cls, *, head, build_store):
        from services.ontology_quality_evaluation_service import QualityEvaluationService
        pins = [binding.pin for binding in head.artifact_bindings if binding.role == "source_validation_execution"]
        if not pins:
            return None, None, "not_run"
        if len(pins) != 1:
            raise ValueError("SOURCE_VALIDATION_EXECUTION_PIN_MISMATCH")
        receipt, run = QualityEvaluationService.read_source_execution(store=build_store,
            organization_id=head.organization_id, workspace_id=head.workspace_id, execution_pin=pins[0])
        candidate_pins = [binding.pin for binding in head.artifact_bindings if binding.role == "normative_semantics"]
        expected = receipt.candidate_pin
        if len(candidate_pins) != 1 or (candidate_pins[0].artifact_kind,
            candidate_pins[0].artifact_ref, candidate_pins[0].artifact_hash) != (
                "normative_semantic_candidate_set", expected.artifact_ref, expected.artifact_hash):
            return receipt, run, "stale"
        review, contract, status = cls.resolve_validation_review(head=head, build_store=build_store)
        if (status != "accepted" or contract is None
            or (review.decision_ref, review.decision_hash) != (receipt.review_pin.artifact_ref, receipt.review_pin.artifact_hash)
            or (contract.validation_contract_ref, contract.validation_contract_hash) != (
                receipt.validation_pin.artifact_ref, receipt.validation_pin.artifact_hash)):
            return receipt, run, "stale"
        return receipt, run, run.disposition

    def review_source_coverage(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        selections,
        build_store, authority_resolver, command_id: str, idempotency_key: str,
        surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from services.ontology_certification_successor_service import (
            CertificationLegalSelection, OntologyCertificationSuccessorService,
        )
        from storage.ontology_build_store import OntologyBuildStore
        from eos.legal_corpus_store import LegalCorpusStore
        if not isinstance(build_store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("certification preparation requires production owners")
        authority_resolver.require_current_actor(actor_context=RuntimeContextDTO(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id))
        if not ctx.authenticated or build_store._path.resolve() != self._store.db_path.resolve() or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        if surface not in {"CHAT", "WORKBENCH"}:
            raise ValueError("HUMAN_REVIEW_SURFACE_REQUIRED")
        selection_payloads = [_selection_payload(item) for item in selections]
        digest = content_hash(selection_payloads)
        identity = dict(command_id=command_id + ":" + digest, actor_ref=f"user:{ctx.user_id}",
                        surface=surface, cause_code="SOURCE_COVERAGE_REVIEWED")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision":expected_revision})
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if head.current_phase != "CERTIFYING_CONTEXT":
            raise ValueError("CERTIFICATION_NOT_REQUIRED_IN_CURRENT_PHASE")
        result = OntologyCertificationSuccessorService().review_source_coverage(
            store=build_store, legal_store=LegalCorpusStore(), authority_resolver=authority_resolver,
            actor_context=RuntimeContextDTO(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id),
            selections=tuple(CertificationLegalSelection(**item) for item in selection_payloads),
            reviewed_at=datetime.now(timezone.utc), request_id="journey-coverage:" + content_hash(dict(
                journey_id=journey_id, idempotency_key=idempotency_key, command_id=identity["command_id"])))
        binding = JourneyArtifactBindingV1(role="source_search_coverage", owner_kind="OntologyCertificationSuccessorService",
            pin=ArtifactPinV1(artifact_kind="source_search_coverage_manifest", artifact_ref=result.coverage_ref, artifact_hash=result.coverage_hash))
        return self.advance(**scope, expected_revision=expected_revision, to_phase="CERTIFYING_CONTEXT",
            **identity, idempotency_key=idempotency_key, operation_receipt=binding, artifact_bindings=(binding,))

    def prepare_authority_dependencies(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        coverage_pin: ArtifactPinV1, selections, jurisdiction: str, as_of_time: datetime,
        build_store, authority_resolver, command_id: str, idempotency_key: str,
        surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from services.ontology_certification_successor_service import (
            CertificationLegalSelection, OntologyCertificationSuccessorService,
        )
        from storage.ontology_build_store import OntologyBuildStore
        from eos.legal_corpus_store import LegalCorpusStore
        if not isinstance(build_store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("certification preparation requires production owners")
        authority_resolver.require_current_actor(actor_context=RuntimeContextDTO(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id))
        if not ctx.authenticated or build_store._path.resolve() != self._store.db_path.resolve() or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        selection_payloads = [_selection_payload(item) for item in selections]
        digest = content_hash(dict(coverage=coverage_pin.model_dump(mode="json"),
            selections=selection_payloads,
            jurisdiction=jurisdiction, as_of_time=as_of_time.isoformat()))
        identity = dict(command_id=command_id + ":" + digest, actor_ref=f"user:{ctx.user_id}",
                        surface=surface, cause_code="AUTHORITY_DEPENDENCIES_PREPARED")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision":expected_revision})
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if head.current_phase != "CERTIFYING_CONTEXT":
            raise ValueError("CERTIFICATION_NOT_REQUIRED_IN_CURRENT_PHASE")
        builds = [item.pin for item in head.artifact_bindings if item.role == "context_build"]
        if len(builds) != 1:
            raise ValueError("CERTIFICATION_ORIGIN_REQUIRED")
        result = OntologyCertificationSuccessorService().publish_authority_dependency_set(
            store=build_store, legal_store=LegalCorpusStore(), organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, origin_pin=builds[0], coverage_pin=coverage_pin,
            selections=tuple(CertificationLegalSelection(**item) for item in selection_payloads),
            jurisdiction=jurisdiction, as_of_time=as_of_time, created_at=datetime.now(timezone.utc),
            actor_ref=identity["actor_ref"], request_id="journey-authority:" + content_hash(dict(
                journey_id=journey_id, idempotency_key=idempotency_key, command_id=identity["command_id"])))
        binding = JourneyArtifactBindingV1(role="authority_dependencies", owner_kind="OntologyCertificationSuccessorService",
            pin=ArtifactPinV1(artifact_kind="normative_authority_dependency_set", artifact_ref=result.set_ref, artifact_hash=result.set_hash))
        return self.advance(**scope, expected_revision=expected_revision, to_phase="CERTIFYING_CONTEXT",
            **identity, idempotency_key=idempotency_key, operation_receipt=binding, artifact_bindings=(binding,))

    def certify_context(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        competency_pin: ArtifactPinV1, authority_pin: ArtifactPinV1,
        execution_plan_pin: ArtifactPinV1, schema_adequacy_pin: ArtifactPinV1,
        representative_evaluation_pin: ArtifactPinV1, paired_suite_pin: ArtifactPinV1 | None,
        build_store, authority_resolver, command_id: str, idempotency_key: str,
        surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.ontology_building import OntologyBuildManifestV1
        from contracts.ontology_certification import OntologyCertificationCommandV1
        from services.ontology_certification_successor_service import OntologyCertificationSuccessorService
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import OntologyBuildStore

        if not isinstance(build_store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("certification requires production artifact and authority owners")
        authority_resolver.require_current_actor(actor_context=RuntimeContextDTO(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id,
        ))
        if not ctx.authenticated or build_store._path.resolve() != self._store.db_path.resolve() or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        if competency_pin.artifact_kind != "competency_contract" or authority_pin.artifact_kind != "normative_authority_dependency_set":
            raise ValueError("CERTIFICATION_INPUT_KIND_MISMATCH")
        inputs = (competency_pin, authority_pin, execution_plan_pin, schema_adequacy_pin,
                  representative_evaluation_pin, paired_suite_pin)
        owner_command_id = "journey-certify:" + content_hash({
            "command_id": command_id, "inputs": [pin.model_dump(mode="json") if pin else None for pin in inputs],
        })
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key, command={
            "predecessor_revision": expected_revision, "command_id": owner_command_id,
            "actor_ref": f"user:{ctx.user_id}", "surface": surface, "cause_code": "CERTIFICATION_OWNER_RESULT",
        })
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if head.current_phase != "CERTIFYING_CONTEXT":
            raise ValueError("CERTIFICATION_NOT_REQUIRED_IN_CURRENT_PHASE")
        if any(binding.role == "validation_case_proposal" for binding in head.artifact_bindings):
            _, reviewed_validation, status = self.resolve_validation_review(head=head, build_store=build_store)
            if status != "accepted" or reviewed_validation is None:
                raise ValueError("JOURNEY_VALIDATION_REVIEW_REQUIRED")
            receipt, quality, validation_status = self.resolve_source_validation(head=head, build_store=build_store)
            if validation_status not in {"QUALIFIED", "QUALIFIED_WITH_WARNINGS"} or receipt is None:
                raise ValueError("JOURNEY_VALIDATION_EXECUTION_MISMATCH")
            from contracts.ontology_admission import CertifiedSemanticExecutionPlanV1
            execution = build_store.get(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                artifact_ref=execution_plan_pin.artifact_ref, contract_type=CertifiedSemanticExecutionPlanV1)
            if (execution_plan_pin.artifact_kind != "certified_semantic_execution_plan" or execution is None
                or execution.execution_plan_hash != execution_plan_pin.artifact_hash
                or execution.quality_evaluation != quality.pin()
                or execution.quality_profile != quality.quality_profile):
                raise ValueError("JOURNEY_VALIDATION_EXECUTION_MISMATCH")
        builds = [item.pin for item in head.artifact_bindings if item.role == "context_build"]
        if len(builds) != 1:
            raise ValueError("CERTIFICATION_ORIGIN_REQUIRED")
        origin = build_store.get(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                                 artifact_ref=builds[0].artifact_ref, contract_type=OntologyBuildManifestV1)
        if origin is None or origin.build_hash != builds[0].artifact_hash:
            raise ValueError("CERTIFICATION_ORIGIN_NOT_FOUND")
        snapshots = [pin for pin in origin.child_artifacts if pin.artifact_kind == "exploratory_graph_snapshot"]
        if len(snapshots) != 1:
            raise ValueError("CERTIFICATION_REQUIRED_CHILD_MISSING:exploratory_graph_snapshot")
        command = OntologyCertificationCommandV1(
            request_id="journey-certify:" + content_hash({"journey_id": journey_id, "idempotency_key": idempotency_key,
                                                         "command_id": owner_command_id}),
            actor_ref=f"user:{ctx.user_id}", origin_build_ref=origin.build_ref, origin_build_hash=origin.build_hash,
            origin_snapshot_ref=snapshots[0].artifact_ref, origin_snapshot_hash=snapshots[0].artifact_hash,
            reviewed_competency_contract_ref=competency_pin.artifact_ref,
            reviewed_competency_contract_hash=competency_pin.artifact_hash,
            authority_dependency_set_ref=authority_pin.artifact_ref, authority_dependency_set_hash=authority_pin.artifact_hash,
        )
        _, receipt = OntologyCertificationSuccessorService().certify_persisted(
            store=build_store, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            command=command, execution_plan_pin=execution_plan_pin, schema_adequacy_pin=schema_adequacy_pin,
            representative_evaluation_pin=representative_evaluation_pin, paired_suite_pin=paired_suite_pin,
            created_at=datetime.now(timezone.utc),
        )
        return self.select_certification(
            ctx=ctx, journey_id=journey_id, expected_revision=expected_revision,
            receipt_pin=ArtifactPinV1(artifact_kind="ontology_certification_receipt.v1",
                                    artifact_ref=receipt.receipt_ref, artifact_hash=receipt.receipt_hash),
            build_store=build_store, authority_resolver=authority_resolver,
            command_id=owner_command_id, idempotency_key=idempotency_key, surface=surface,
        )

    def select_certification(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        receipt_pin: ArtifactPinV1, build_store, authority_resolver,
        command_id: str, idempotency_key: str, surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        """Adopt only the certification owner's verified successor of this build."""
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.ontology_certification_successor_service import OntologyCertificationSuccessorService
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
        if not isinstance(build_store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("certification selection requires production artifact and authority owners")
        authority_resolver.require_current_actor(actor_context=RuntimeContextDTO(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id,
        ))
        if not ctx.authenticated or build_store._path.resolve() != self._store.db_path.resolve() or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        if receipt_pin.artifact_kind != "ontology_certification_receipt.v1":
            raise ValueError("ONTOLOGY_CERTIFICATION_RECEIPT_REQUIRED")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        identity = dict(command_id=command_id, actor_ref=f"user:{ctx.user_id}", surface=surface,
                        cause_code="CERTIFICATION_OWNER_RESULT")
        binding = JourneyArtifactBindingV1(pin=receipt_pin, owner_kind="OntologyCertificationSuccessorService", role="certification")
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "predecessor_revision": expected_revision})
        if replay is not None:
            if binding not in replay.artifact_bindings:
                raise ValueError("IDEMPOTENCY_KEY_REUSED_WITH_DIFFERENT_TRANSITION")
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.current_phase != "CERTIFYING_CONTEXT":
            raise ValueError("CERTIFICATION_NOT_REQUIRED_IN_CURRENT_PHASE")
        builds = [item.pin for item in head.artifact_bindings if item.role == "context_build"]
        if len(builds) != 1:
            raise ValueError("CERTIFICATION_ORIGIN_REQUIRED")
        receipt, successor = OntologyCertificationSuccessorService().resolve_receipt(
            store=build_store, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            receipt_pin=receipt_pin, origin_pin=builds[0],
        )
        if successor is None:
            return self.block(**scope, expected_revision=expected_revision, **identity, idempotency_key=idempotency_key,
                blocker_code="CERTIFICATION_" + receipt.status, artifact_bindings=(binding,))
        build_binding = JourneyArtifactBindingV1(
            pin=ArtifactPinV1(artifact_kind="ontology_build_manifest", artifact_ref=successor.build_ref, artifact_hash=successor.build_hash),
            owner_kind="OntologyCertificationSuccessorService", role="context_build",
        )
        return self.advance(**scope, expected_revision=expected_revision, **identity, idempotency_key=idempotency_key,
            to_phase="RUNTIME_BINDING_REQUIRED", operation_receipt=binding, artifact_bindings=(build_binding, binding))

    def record_certification(
        self,
        *,
        journey_id: str,
        organization_id: str,
        workspace_id: str,
        expected_revision: int,
        successor_build_pin: ArtifactPinV1,
        certification_receipt_pin: ArtifactPinV1,
        actor_ref: str,
        surface: TenderJourneySurface,
        command_id: str,
        idempotency_key: str,
    ) -> TenderJourneyHeadV1:
        if successor_build_pin.artifact_kind != "ontology_build_manifest":
            raise ValueError("A1_SUCCESSOR_BUILD_REQUIRED")
        if certification_receipt_pin.artifact_kind != "ontology_certification_receipt.v1":
            raise ValueError("ONTOLOGY_CERTIFICATION_RECEIPT_REQUIRED")
        build_binding = JourneyArtifactBindingV1(
            pin=successor_build_pin,
            owner_kind="OntologyCertificationSuccessorService",
            role="context_build",
        )
        receipt = JourneyArtifactBindingV1(
            pin=certification_receipt_pin,
            owner_kind="OntologyCertificationSuccessorService",
            role="operation_receipt",
        )
        return self.advance(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            expected_revision=expected_revision,
            to_phase="RUNTIME_BINDING_REQUIRED",
            cause_code="A1_BOUNDED_CERTIFICATION_COMPLETED",
            actor_ref=actor_ref,
            surface=surface,
            command_id=command_id,
            idempotency_key=idempotency_key,
            operation_receipt=receipt,
            artifact_bindings=(
                build_binding,
                JourneyArtifactBindingV1(
                    pin=certification_receipt_pin,
                    owner_kind="OntologyCertificationSuccessorService",
                    role="certification",
                ),
            ),
        )

    def next_document(
        self, *, journey_id: str, organization_id: str, workspace_id: str,
        expected_revision: int, actor_ref: str, surface: TenderJourneySurface,
        command_id: str, idempotency_key: str,
    ) -> TenderJourneyHeadV1:
        scope = dict(journey_id=journey_id, organization_id=organization_id, workspace_id=workspace_id)
        identity = dict(actor_ref=actor_ref, surface=surface, command_id=command_id)
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key, command={
            **identity, "predecessor_revision": expected_revision,
            "cause_code": "NEXT_RUNTIME_DOCUMENT_REQUESTED", "to_phase": "NEEDS_RUNTIME_DOCUMENT",
        })
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.current_phase != "REVIEW_RESULTS":
            raise ValueError("RUNTIME_RESULTS_REVIEW_REQUIRED")
        results = [binding for binding in head.artifact_bindings if binding.role == "inspection_result"]
        if len(results) != 1:
            raise ValueError("INSPECTION_RESULT_REQUIRED")
        return self.advance(**scope, expected_revision=expected_revision,
            to_phase="NEEDS_RUNTIME_DOCUMENT", cause_code="NEXT_RUNTIME_DOCUMENT_REQUESTED",
            **identity, idempotency_key=idempotency_key, operation_receipt=results[0])

    def select_runtime_document(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        artifact_pin: ArtifactPinV1, preparation_pin: ArtifactPinV1,
        preparation_service: WorkspaceArtifactPreparationService,
        command_id: str, idempotency_key: str, surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        """Select a governed prepared document for the next inspection episode."""
        if not isinstance(preparation_service, WorkspaceArtifactPreparationService):
            raise TypeError("document selection requires the production preparation owner")
        self.get(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        if preparation_service._build._path.resolve() != self._store.db_path.resolve():
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        preparation = preparation_service.resolve(
            ctx=ctx, artifact_pin=artifact_pin, preparation_pin=preparation_pin,
            purpose="RUNTIME_TARGET",
        )
        if preparation.source_identity_pin is None or preparation.canonical_source_ir_pin is None:
            raise ValueError("RUNTIME_DOCUMENT_CANONICAL_SOURCE_REQUIRED")
        return self.record_runtime_target(
            journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            expected_revision=expected_revision, runtime_document_pin=artifact_pin,
            preparation_pin=preparation_pin, actor_ref=f"user:{ctx.user_id}", surface=surface,
            command_id=command_id, idempotency_key=idempotency_key,
        )

    def record_runtime_target(
        self,
        *,
        journey_id: str,
        organization_id: str,
        workspace_id: str,
        expected_revision: int,
        runtime_document_pin: ArtifactPinV1,
        preparation_pin: ArtifactPinV1,
        actor_ref: str,
        surface: TenderJourneySurface,
        command_id: str,
        idempotency_key: str,
    ) -> TenderJourneyHeadV1:
        binding = JourneyArtifactBindingV1(pin=runtime_document_pin, owner_kind="WorkspaceArtifactPreparationService", role="runtime_document")
        receipt = JourneyArtifactBindingV1(pin=preparation_pin, owner_kind="WorkspaceArtifactPreparationService", role="runtime_document_preparation")
        return self.advance(
            journey_id=journey_id, organization_id=organization_id, workspace_id=workspace_id,
            expected_revision=expected_revision, to_phase="FORMING_RUNTIME_CONTEXT",
            cause_code="GOVERNED_RUNTIME_TARGET_PREPARED", actor_ref=actor_ref, surface=surface,
            command_id=command_id, idempotency_key=idempotency_key, operation_receipt=receipt,
            artifact_bindings=(binding, receipt),
        )

    def confirm_runtime_context_command(
        self, *, ctx: RequestContext, journey_id: str, session_ref: str,
        command_id: str, idempotency_key: str, expected_revision: int,
    ):
        """Observe durable formation result only; never form or retry a command."""
        from contracts.tender_journey import TenderRuntimeCommandCompletionV1
        if (not ctx.authenticated or self._store.db_path.resolve()
                != ontology_build_store_path(ctx.workspace_root).resolve()):
            raise ValueError("JOURNEY_COMMAND_SCOPE_MISMATCH")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id,
                     workspace_id=ctx.workspace_id)
        current = self.get(**scope)  # Scope authorization precedes receipt lookup.
        selected = self._store.get_session_journey(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_ref=f"user:{ctx.user_id}", session_ref=session_ref)
        if not ctx.authenticated or selected is None or selected.journey_id != journey_id:
            raise ValueError("JOURNEY_COMMAND_SCOPE_MISMATCH")
        identity = dict(predecessor_revision=expected_revision,
            cause_code="GOVERNED_RUNTIME_CONTEXT_FORMED", actor_ref=f"user:{ctx.user_id}",
            surface="WORKBENCH", command_id=command_id, to_phase="INSPECTING", blocker_code=None)
        result = self._store.replay_command(**scope, idempotency_key=idempotency_key, command=identity)
        common = dict(journey_id=journey_id, command_id=command_id,
            idempotency_key=idempotency_key, expected_revision=expected_revision)
        if result is None:
            return TenderRuntimeCommandCompletionV1(status="NOT_CONFIRMED", **common)
        transition = next(item for item in self.history(**scope)
                          if item.transition_id == result.latest_transition_id)
        if (transition.operation_receipt is None or transition.operation_receipt.role != "runtime_context"
                or result.operation_receipt != transition.operation_receipt):
            raise ValueError("JOURNEY_COMMAND_RESULT_UNAVAILABLE")
        return TenderRuntimeCommandCompletionV1(status="COMPLETED", **common,
            compatible=current == result and current.lifecycle_status == "ACTIVE",
            transition=transition, result=self.project(result))

    def form_runtime_context(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        preparation_service: WorkspaceArtifactPreparationService, inspection_service,
        authority_resolver, command_id: str, idempotency_key: str, surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_semantic_inspection_service import RuntimeSemanticInspectionService
        from services.runtime_context_transition_service import RuntimeAuthorityResolver

        if not isinstance(inspection_service, RuntimeSemanticInspectionService) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("runtime formation requires production inspection and authority owners")
        authority_resolver.require_current_actor(actor_context=RuntimeContextDTO(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id,
        ))
        if (
            not ctx.authenticated or inspection_service._workspace_root.resolve() != ctx.workspace_root.resolve()
            or inspection_service._actor_id != ctx.user_id
            or not isinstance(preparation_service, WorkspaceArtifactPreparationService)
            or preparation_service._build._path.resolve() != self._store.db_path.resolve()
        ):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key, command={
            "predecessor_revision": expected_revision, "cause_code": "GOVERNED_RUNTIME_CONTEXT_FORMED",
            "actor_ref": f"user:{ctx.user_id}", "surface": surface, "command_id": command_id,
            "to_phase": "INSPECTING", "blocker_code": None,
        })
        if replay is not None:
            return replay
        head = self.get(**scope)
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if head.current_phase != "FORMING_RUNTIME_CONTEXT":
            raise ValueError("RUNTIME_CONTEXT_FORMATION_NOT_REQUIRED")
        pins = {}
        for role in ("runtime_document", "runtime_document_preparation", "runtime_binding"):
            matches = [binding.pin for binding in head.artifact_bindings if binding.role == role]
            if len(matches) != 1:
                raise ValueError("RUNTIME_CONTEXT_INPUT_REQUIRED")
            pins[role] = matches[0]
        prepared = preparation_service.resolve(
            ctx=ctx, artifact_pin=pins["runtime_document"],
            preparation_pin=pins["runtime_document_preparation"], purpose="RUNTIME_TARGET",
        )
        if prepared.canonical_source_ir_pin is None:
            raise ValueError("RUNTIME_DOCUMENT_CANONICAL_SOURCE_REQUIRED")
        import asyncio
        formed = asyncio.run(inspection_service.prepare_context(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            normalized_document_artifact_ref=prepared.canonical_source_ir_pin.artifact_ref,
            runtime_context_pin=pins["runtime_binding"],
            request_id="journey-inspection:" + content_hash({
                "journey_id": journey_id, "selection_transition": head.latest_transition_id,
                "command_id": command_id, "idempotency_key": idempotency_key,
            }),
        ))
        return self.record_runtime_context_formed(
            **scope, expected_revision=expected_revision,
            runtime_context_pin=ArtifactPinV1(artifact_kind=formed.schema_version,
                artifact_ref=formed.context_ref, artifact_hash=formed.context_hash),
            actor_ref=f"user:{ctx.user_id}", surface=surface,
            command_id=command_id, idempotency_key=idempotency_key,
        )

    def record_runtime_context_formed(
        self,
        *,
        journey_id: str,
        organization_id: str,
        workspace_id: str,
        expected_revision: int,
        runtime_context_pin: ArtifactPinV1,
        actor_ref: str,
        surface: TenderJourneySurface,
        command_id: str,
        idempotency_key: str,
    ) -> TenderJourneyHeadV1:
        binding = JourneyArtifactBindingV1(pin=runtime_context_pin, owner_kind="RuntimeSemanticInspectionService", role="runtime_context")
        return self.advance(
            journey_id=journey_id, organization_id=organization_id, workspace_id=workspace_id,
            expected_revision=expected_revision, to_phase="INSPECTING", cause_code="GOVERNED_RUNTIME_CONTEXT_FORMED",
            actor_ref=actor_ref, surface=surface, command_id=command_id, idempotency_key=idempotency_key,
            operation_receipt=binding, artifact_bindings=(binding,),
        )

    async def inspect_document(
        self, *, ctx: RequestContext, journey_id: str, expected_revision: int,
        inspection_service, authority_resolver,
        command_id: str, idempotency_key: str, surface: TenderJourneySurface,
    ) -> TenderJourneyHeadV1:
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.runtime_semantic_inspection_result import RuntimeSemanticInspectionResultV1
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from services.runtime_semantic_inspection_service import RuntimeSemanticInspectionService
        if not isinstance(inspection_service, RuntimeSemanticInspectionService) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("journey inspection requires production owners")
        authority_resolver.require_current_actor(actor_context=RuntimeContextDTO(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id,
        ))
        if (not ctx.authenticated or inspection_service._workspace_root.resolve() != ctx.workspace_root.resolve()
                or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve()
                or inspection_service._actor_id != ctx.user_id):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        identity = dict(actor_ref=f"user:{ctx.user_id}", surface=surface, command_id=command_id)
        started = self._store.replay_command(**scope, idempotency_key=idempotency_key + ":start",
            command={**identity, "predecessor_revision": expected_revision, "cause_code": "RUNTIME_INSPECTION_STARTED"})
        # The start receipt fixes the caller's original revision. Completion may
        # follow pause/resume revisions, so it need not immediately follow start.
        replay = self._store.replay_command(**scope, idempotency_key=idempotency_key,
            command={**identity, "cause_code": "RUNTIME_INSPECTION_COMPLETED"})
        if replay is not None:
            if started is None:
                raise ValueError("JOURNEY_INSPECTION_CLAIM_REQUIRED")
            self._publish_inspection_compliance(ctx=ctx, head=replay,
                inspection_service=inspection_service, authority_resolver=authority_resolver)
            return replay

        def current_attempt_head():
            current = self.get(**scope)
            if current.lifecycle_status != "ACTIVE":
                raise ValueError("JOURNEY_NOT_ACTIVE")
            revision = started.revision if started is not None else expected_revision
            if current.revision != revision:
                if started is None or current.revision < revision:
                    raise ValueError("STALE_JOURNEY_REVISION")
                intervening = [item for item in self.history(**scope)
                               if revision < item.successor_revision <= current.revision]
                if (len(intervening) != current.revision - revision
                        or any(item.to_lifecycle_status is None for item in intervening)
                        or current.artifact_bindings != started.artifact_bindings
                        or current.current_phase != started.current_phase):
                    raise ValueError("STALE_JOURNEY_REVISION")
            return current

        head = current_attempt_head()
        if head.current_phase != "INSPECTING":
            raise ValueError("RUNTIME_INSPECTION_NOT_REQUIRED")
        if started is None and head.blocker_code == "RUNTIME_INSPECTION_IN_PROGRESS":
            raise ValueError("RUNTIME_INSPECTION_PENDING_RECOVERY")
        pins = {}
        for role in ("runtime_context", "runtime_binding"):
            matches = [binding.pin for binding in head.artifact_bindings if binding.role == role]
            if len(matches) != 1:
                raise ValueError("RUNTIME_CONTEXT_INPUT_REQUIRED")
            pins[role] = matches[0]
        prepared = inspection_service.resolve_context(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, pin=pins["runtime_context"],
        )
        if prepared.runtime_context_pin != pins["runtime_binding"] or not prepared.request_id:
            raise ValueError("JOURNEY_RUNTIME_CONTEXT_MISMATCH")
        owner_args = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            normalized_document_artifact_ref=prepared.normalized_document_artifact_ref,
            runtime_context_pin=prepared.runtime_context_pin, request_id=prepared.request_id)
        completed = inspection_service.get_completed_inspection(**owner_args)
        if started is None:
            claim_id = f"tender-journey-transition:{uuid4().hex}"
            started = self._store.append_transition(
                organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, expected_revision=expected_revision,
                transition=TenderJourneyTransitionV1(
                    transition_id=claim_id, journey_id=journey_id, predecessor_revision=expected_revision,
                    successor_revision=expected_revision + 1, from_phase="INSPECTING", to_phase="INSPECTING",
                    cause_code="RUNTIME_INSPECTION_STARTED", **identity, idempotency_key=idempotency_key + ":start",
                    operation_receipt=head.operation_receipt, blocker_code="RUNTIME_INSPECTION_IN_PROGRESS",
                    created_at=datetime.now(timezone.utc),
                ),
            )
            if started.latest_transition_id == claim_id and completed is None:
                try:
                    completed = RuntimeSemanticInspectionResultV1.model_validate(await inspection_service.inspect(**owner_args))
                except (RuntimeError, OSError):
                    # A returned failure ends this attempt. Cancellation/crash is
                    # deliberately not caught: an uncertain attempt stays claimed.
                    # Publication may have succeeded before a transport failure.
                    completed = inspection_service.get_completed_inspection(**owner_args)
                    if completed is None:
                        return self.block(**scope, expected_revision=current_attempt_head().revision,
                            blocker_code="RUNTIME_INSPECTION_FAILED", cause_code="RUNTIME_INSPECTION_COMPLETED",
                            **identity, idempotency_key=idempotency_key)
        if completed is None:
            raise ValueError("RUNTIME_INSPECTION_PENDING_RECOVERY")
        self._verify_inspection_lineage(prepared, completed)
        recorded = self.record_inspection_result(
            **scope, expected_revision=current_attempt_head().revision, inspection_ref=completed.result_ref,
            inspection_hash=completed.result_hash, **identity, idempotency_key=idempotency_key,
        )
        self._publish_inspection_compliance(ctx=ctx, head=recorded,
            inspection_service=inspection_service, authority_resolver=authority_resolver)
        return recorded

    def _publish_inspection_compliance(self, *, ctx, head, inspection_service, authority_resolver):
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.tender_legal_compliance_service import TenderLegalComplianceService
        from storage.ontology_build_store import OntologyBuildStore
        if head.current_phase != "REVIEW_RESULTS":
            return  # Failed attempts have no completed episode or legal result.
        episode = self.resolve_inspection_episode(ctx=ctx, journey_id=head.journey_id,
            transition_id=head.latest_transition_id, inspection_service=inspection_service,
            authority_resolver=authority_resolver)
        store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
        try:
            return TenderLegalComplianceService(build_store=store, authority_resolver=authority_resolver,
                workspace_root=ctx.workspace_root, db_path=inspection_service._db_path).publish_episode(
                    actor_context=RuntimeContextDTO(organization_id=ctx.organization_id,
                        workspace_id=ctx.workspace_id, actor_id=ctx.user_id),
                    episode=episode, inspection_service=inspection_service)
        finally:
            store.close()

    @staticmethod
    def _verify_inspection_lineage(prepared, completed):
        if (completed.run_id != prepared.run_id
                or completed.retrieval_snapshot_refs != prepared.retrieval_snapshot_refs
                or completed.source_identity_ref != prepared.source_regions[0].source_binding.source_identity_ref
                or completed.source_hash != prepared.source_regions[0].source_binding.source_hash
                or completed.canonical_source_ir_ref != prepared.source_regions[0].source_binding.canonical_source_ir.artifact_ref
                or completed.canonical_source_ir_hash != prepared.source_regions[0].source_binding.canonical_source_ir.artifact_hash):
            raise ValueError("JOURNEY_INSPECTION_RESULT_MISMATCH")

    def resolve_inspection_episode(self, *, ctx: RequestContext, journey_id: str,
                                   transition_id: str, inspection_service, authority_resolver):
        """Resolve a completed historical episode using only its pinned artifacts."""
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.tender_journey import TenderJourneyInspectionEpisodeV1
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from services.runtime_semantic_inspection_service import RuntimeSemanticInspectionService
        if not isinstance(inspection_service, RuntimeSemanticInspectionService) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("episode reads require production artifact and authority owners")
        authority_resolver.require_current_actor(actor_context=RuntimeContextDTO(
            organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id,
        ))
        if (not ctx.authenticated or inspection_service._workspace_root.resolve() != ctx.workspace_root.resolve()
                or self._store.db_path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve()
                or inspection_service._actor_id != ctx.user_id):
            raise ValueError("JOURNEY_WORKSPACE_OWNER_MISMATCH")
        scope = dict(journey_id=journey_id, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        transition, head = self._store.get_transition_result(**scope, transition_id=transition_id)
        if (transition.cause_code != "RUNTIME_INSPECTION_COMPLETED" or transition.to_phase != "REVIEW_RESULTS"
                or transition.operation_receipt is None or transition.operation_receipt.role != "inspection_result"):
            raise ValueError("INSPECTION_EPISODE_REQUIRED")
        pins = {}
        for role in ("runtime_binding", "runtime_context", "inspection_result"):
            matches = [item.pin for item in head.artifact_bindings if item.role == role]
            if len(matches) != 1:
                raise ValueError("INSPECTION_EPISODE_INPUT_REQUIRED")
            pins[role] = matches[0]
        if pins["inspection_result"] != transition.operation_receipt.pin:
            raise ValueError("INSPECTION_EPISODE_RESULT_MISMATCH")
        artifact_scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        prepared = inspection_service.resolve_context(**artifact_scope, pin=pins["runtime_context"])
        completed = inspection_service.resolve_result(**artifact_scope, pin=pins["inspection_result"])
        if prepared.runtime_context_pin != pins["runtime_binding"]:
            raise ValueError("JOURNEY_RUNTIME_CONTEXT_MISMATCH")
        self._verify_inspection_lineage(prepared, completed)
        return TenderJourneyInspectionEpisodeV1(transition_id=transition_id, journey=self.project(head),
            inspection_context=prepared, inspection_result=completed)

    def record_inspection_result(
        self,
        *,
        journey_id: str,
        organization_id: str,
        workspace_id: str,
        expected_revision: int,
        inspection_ref: str,
        inspection_hash: str,
        actor_ref: str,
        surface: TenderJourneySurface,
        command_id: str,
        idempotency_key: str,
    ) -> TenderJourneyHeadV1:
        binding = JourneyArtifactBindingV1(
            pin=ArtifactPinV1(artifact_kind="runtime_semantic_inspection_result.v1", artifact_ref=inspection_ref, artifact_hash=inspection_hash),
            owner_kind="RuntimeSemanticInspectionService", role="inspection_result",
        )
        return self.advance(
            journey_id=journey_id, organization_id=organization_id, workspace_id=workspace_id,
            expected_revision=expected_revision, to_phase="REVIEW_RESULTS", cause_code="RUNTIME_INSPECTION_COMPLETED",
            actor_ref=actor_ref, surface=surface, command_id=command_id, idempotency_key=idempotency_key,
            operation_receipt=binding, artifact_bindings=(binding,),
        )

    def bind_current_runtime_context(
        self,
        *,
        journey_id: str,
        organization_id: str,
        workspace_id: str,
        expected_revision: int,
        actor_ref: str,
        surface: TenderJourneySurface,
        command_id: str,
        idempotency_key: str,
        runtime_db_path: str | Path,
    ) -> TenderJourneyHeadV1:
        replay = self._store.replay_command(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            idempotency_key=idempotency_key,
            command={
                "predecessor_revision": expected_revision,
                "cause_code": "GOVERNED_RUNTIME_CONTEXT_RESOLVED",
                "actor_ref": actor_ref,
                "surface": surface,
                "command_id": command_id,
                "to_phase": "CONTEXT_READY",
                "blocker_code": None,
            },
        )
        if replay is not None:
            return replay
        head = self.get(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if head.lifecycle_status != "ACTIVE":
            raise ValueError("JOURNEY_NOT_ACTIVE")
        if head.current_phase != "RUNTIME_BINDING_REQUIRED":
            raise ValueError("RUNTIME_BINDING_NOT_REQUIRED_IN_CURRENT_PHASE")
        policy = {}
        for role in ("policy_source", "canonical_ir", "graph_intent", "context_build"):
            matches = [binding.pin for binding in head.artifact_bindings if binding.role == role]
            if len(matches) != 1:
                raise ValueError("RUNTIME_BINDING_POLICY_LINEAGE_REQUIRED")
            policy[role] = matches[0]
        from storage.ontology_build_store import OntologyBuildStore
        artifact_store = OntologyBuildStore(self._store.db_path)
        try:
            context_pin = EffectiveRestrictionRuntimeContextService(runtime_db_path).publish_for_policy(
                store=artifact_store, organization_id=organization_id, workspace_id=workspace_id,
                actor_ref=actor_ref,
                request_id="tender-journey-binding:" + content_hash({
                    "journey_id": journey_id, "idempotency_key": idempotency_key,
                }),
                request_hash="sha256:" + content_hash({
                    "journey_id": journey_id, "expected_revision": expected_revision,
                    "command_id": command_id, "surface": surface,
                    "policy": {role: pin.model_dump(mode="json") for role, pin in policy.items()},
                }),
                source_identity_ref=policy["policy_source"].artifact_ref,
                canonical_source_ir_pin=policy["canonical_ir"], graph_intent_pin=policy["graph_intent"],
            )
            context = EffectiveRestrictionRuntimeContextService.resolve_persisted(
                store=artifact_store, pin=context_pin, organization_id=organization_id, workspace_id=workspace_id,
            )
            self._require_runtime_release_link(store=artifact_store, organization_id=organization_id,
                workspace_id=workspace_id, build_pin=policy["context_build"],
                accepted_graph_release_ref=context.accepted_graph_release_ref)
        finally:
            artifact_store.close()
        binding = JourneyArtifactBindingV1(
            pin=context_pin,
            owner_kind="EffectiveRestrictionRuntimeContextService",
            role="runtime_binding",
        )
        return self.advance(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            expected_revision=expected_revision,
            to_phase="CONTEXT_READY",
            cause_code="GOVERNED_RUNTIME_CONTEXT_RESOLVED",
            actor_ref=actor_ref,
            surface=surface,
            command_id=command_id,
            idempotency_key=idempotency_key,
            operation_receipt=binding,
            artifact_bindings=(binding,),
        )

    @staticmethod
    def _require_runtime_release_link(*, store, organization_id, workspace_id, build_pin, accepted_graph_release_ref):
        """Consume an existing admitted release link; never manufacture the bridge."""
        from contracts.ontology_admission import OntologyReleaseBundleV1, OntologyAdmissionDecisionV1, CertifiedSemanticExecutionPlanV1
        from contracts.ontology_building import OntologyBuildManifestV1
        from contracts.ontology_quality import QualityEvaluationRunV1
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        if build_pin.artifact_kind != "ontology_build_manifest":
            raise ValueError("RUNTIME_CERTIFIED_BUILD_REQUIRED")
        manifest = store.get(**scope, artifact_ref=build_pin.artifact_ref, contract_type=OntologyBuildManifestV1)
        if manifest is None or manifest.build_hash != build_pin.artifact_hash:
            raise ValueError("RUNTIME_CERTIFIED_BUILD_PIN_MISMATCH")
        execution = [pin for pin in manifest.child_artifacts if pin.artifact_kind == "certified_semantic_execution_plan"]
        if manifest.assurance_profile == "A0_EXPLORATORY" or len(execution) != 1:
            raise ValueError("RUNTIME_CERTIFIED_BUILD_REQUIRED")
        plan = store.get(**scope, artifact_ref=execution[0].artifact_ref, contract_type=CertifiedSemanticExecutionPlanV1)
        if plan is None or plan.execution_plan_hash != execution[0].artifact_hash or plan.assurance_profile != manifest.assurance_profile:
            raise ValueError("RUNTIME_RELEASE_EXECUTION_PIN_MISMATCH")
        for pin in store.list_artifact_pins(**scope, schema_version="ontology_release_bundle.v1"):
            bundle = store.get(**scope, artifact_ref=pin.artifact_ref, contract_type=OntologyReleaseBundleV1)
            if bundle is None or bundle.release_bundle_hash != pin.artifact_hash:
                raise ValueError("RUNTIME_RELEASE_BUNDLE_PIN_MISMATCH")
            if (bundle.build_ref, bundle.build_hash, bundle.accepted_graph_release_ref) != (
                build_pin.artifact_ref, build_pin.artifact_hash, accepted_graph_release_ref
            ):
                continue
            decision = store.get(**scope, artifact_ref=bundle.admission_decision_ref, contract_type=OntologyAdmissionDecisionV1)
            if decision is None or decision.decision_hash != bundle.admission_decision_hash:
                raise ValueError("RUNTIME_RELEASE_ADMISSION_PIN_MISMATCH")
            if (decision.build_ref, decision.build_hash) != (build_pin.artifact_ref, build_pin.artifact_hash):
                raise ValueError("RUNTIME_RELEASE_ADMISSION_BUILD_MISMATCH")
            expected = (execution[0].artifact_ref, execution[0].artifact_hash)
            if (bundle.execution_plan_ref, bundle.execution_plan_hash) != expected or (
                decision.execution_plan_ref, decision.execution_plan_hash
            ) != expected:
                raise ValueError("RUNTIME_RELEASE_EXECUTION_PIN_MISMATCH")
            if decision.decision != "ADMIT":
                raise ValueError("RUNTIME_RELEASE_UNRESTRICTED_ADMISSION_REQUIRED")
            if (decision.assurance_profile != manifest.assurance_profile
                or decision.quality_evaluation != plan.quality_evaluation
                or decision.quality_profile != plan.quality_profile):
                raise ValueError("RUNTIME_RELEASE_ADMISSION_PLAN_MISMATCH")
            quality = store.get(**scope, artifact_ref=plan.quality_evaluation.artifact_ref, contract_type=QualityEvaluationRunV1)
            if (quality is None or quality.pin() != plan.quality_evaluation
                or quality.quality_profile != plan.quality_profile
                or (quality.validation.artifact_ref, quality.validation.artifact_hash) != (
                    plan.validation_contract_ref, plan.validation_contract_hash)
                or quality.disposition not in {"QUALIFIED", "QUALIFIED_WITH_WARNINGS"}
                or (quality.disposition == "QUALIFIED_WITH_WARNINGS" and not quality.profile_definition.allow_warnings)):
                raise ValueError("RUNTIME_RELEASE_QUALITY_PIN_MISMATCH")
            if quality.subject != ArtifactPinV1(artifact_kind="normative_semantic_candidate_set",
                artifact_ref=plan.canonical_normative_ir_ref, artifact_hash=plan.canonical_normative_ir_hash):
                raise ValueError("RUNTIME_RELEASE_QUALITY_CANDIDATE_MISMATCH")
            return
        raise ValueError("CERTIFIED_RUNTIME_RELEASE_BRIDGE_UNAVAILABLE")

    def advance(
        self,
        *,
        journey_id: str,
        organization_id: str,
        workspace_id: str,
        expected_revision: int,
        to_phase: TenderJourneyPhase,
        cause_code: str,
        actor_ref: str,
        surface: TenderJourneySurface,
        command_id: str,
        idempotency_key: str,
        operation_receipt: JourneyArtifactBindingV1,
        artifact_bindings: tuple[JourneyArtifactBindingV1, ...] = (),
        now: datetime | None = None,
    ) -> TenderJourneyHeadV1:
        replay = self._store.replay_command(
            journey_id=journey_id, organization_id=organization_id, workspace_id=workspace_id,
            idempotency_key=idempotency_key,
            command={
                "predecessor_revision": expected_revision, "cause_code": cause_code,
                "actor_ref": actor_ref, "surface": surface, "command_id": command_id,
                "artifact_bindings": artifact_bindings,
                "to_phase": to_phase, "operation_receipt": operation_receipt, "blocker_code": None,
            },
        )
        if replay is not None:
            return replay
        head = self.get(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        if to_phase not in _ALLOWED_NEXT[head.current_phase]:
            raise ValueError(
                f"INVALID_JOURNEY_TRANSITION:{head.current_phase}->{to_phase}"
            )
        transition = TenderJourneyTransitionV1(
            transition_id=f"tender-journey-transition:{uuid4().hex}",
            journey_id=journey_id,
            predecessor_revision=head.revision,
            successor_revision=head.revision + 1,
            from_phase=head.current_phase,
            to_phase=to_phase,
            cause_code=cause_code,
            actor_ref=actor_ref,
            surface=surface,
            command_id=command_id,
            idempotency_key=idempotency_key,
            artifact_bindings=artifact_bindings,
            operation_receipt=operation_receipt,
            created_at=now or datetime.now(timezone.utc),
        )
        return self._store.append_transition(
            organization_id=organization_id,
            workspace_id=workspace_id,
            expected_revision=expected_revision,
            transition=transition,
        )

    def block(
        self,
        *,
        journey_id: str,
        organization_id: str,
        workspace_id: str,
        expected_revision: int,
        blocker_code: str,
        cause_code: str,
        actor_ref: str,
        surface: TenderJourneySurface,
        command_id: str,
        idempotency_key: str,
        artifact_bindings: tuple[JourneyArtifactBindingV1, ...] = (),
        now: datetime | None = None,
    ) -> TenderJourneyHeadV1:
        """Persist a blocker without narratively advancing the workflow phase."""
        replay = self._store.replay_command(
            journey_id=journey_id, organization_id=organization_id, workspace_id=workspace_id,
            idempotency_key=idempotency_key,
            command={
                "predecessor_revision": expected_revision, "cause_code": cause_code,
                "actor_ref": actor_ref, "surface": surface, "command_id": command_id,
                "artifact_bindings": artifact_bindings,
                "blocker_code": blocker_code, "operation_receipt": None,
            },
        )
        if replay is not None:
            return replay
        head = self.get(
            journey_id=journey_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        if head.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        transition = TenderJourneyTransitionV1(
            transition_id=f"tender-journey-transition:{uuid4().hex}",
            journey_id=journey_id,
            predecessor_revision=head.revision,
            successor_revision=head.revision + 1,
            from_phase=head.current_phase,
            to_phase=head.current_phase,
            cause_code=cause_code,
            actor_ref=actor_ref,
            surface=surface,
            command_id=command_id,
            idempotency_key=idempotency_key,
            artifact_bindings=artifact_bindings,
            blocker_code=blocker_code,
            created_at=now or datetime.now(timezone.utc),
        )
        return self._store.append_transition(
            organization_id=organization_id,
            workspace_id=workspace_id,
            expected_revision=expected_revision,
            transition=transition,
        )


__all__ = ["TenderJourneyService"]
