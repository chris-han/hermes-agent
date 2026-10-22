"""Purpose-specific Tender inference producer using the existing kernel store."""

from datetime import datetime

from contracts.knowledge_classification import RuntimeContextDTO
from contracts.semantic_kernel import (
    ContextGraph,
    DecisionRecord,
    Evidence,
    Ontology,
    ReasonerInvocation,
    ReasoningTrace,
)
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from services.tender_decision_verification_service import (
    TenderDecisionVerificationService,
)
from storage.ontology_build_store import OntologyBuildStore


class TenderDecisionRecordService:
    def __init__(
        self,
        *,
        build_store: OntologyBuildStore,
        authority_resolver: RuntimeAuthorityResolver,
    ):
        if not isinstance(build_store, OntologyBuildStore):
            raise TypeError("Tender inference requires the shared kernel store")
        if not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("Tender inference requires governed actor resolution")
        self._store = build_store
        self._authority = authority_resolver

    def record_inference(
        self,
        *,
        actor_context: RuntimeContextDTO,
        recorded_at: datetime,
        verification,
        projection,
        frontier,
        model_input,
        distribution=None,
    ) -> DecisionRecord:
        self._authority.require_current_actor(actor_context=actor_context)
        handoff = TenderDecisionVerificationService().handoff(
            verification=verification,
            projection=projection,
            frontier=frontier,
            model_input=model_input,
            distribution=distribution,
        )
        scope = (actor_context.organization_id, actor_context.workspace_id)
        if scope != (handoff.organization_id, handoff.workspace_id):
            raise ValueError("FORBIDDEN_SCOPE")

        def load(pin, model):
            value = self._store.get(
                organization_id=scope[0],
                workspace_id=scope[1],
                artifact_ref=pin.artifact_ref,
                contract_type=model,
            )
            if value is None or value.pin() != pin:
                raise ValueError("UNRESOLVABLE_DECISION_BASIS")
            return value

        context = load(handoff.context, ContextGraph)
        if context.state != "ADMITTED" or context.ontology != handoff.ontology:
            raise ValueError("ADMITTED_CONTEXT_REQUIRED")
        load(handoff.ontology, Ontology)
        for evidence in handoff.evidence:
            load(evidence, Evidence)
        if not handoff.evidence:
            raise ValueError("SEMANTIC_INFERENCE_REQUIRES_EVIDENCE")
        verified = handoff.verification
        invocation = ReasonerInvocation(
            invocation_id=verified.verification_ref,
            strategy="tender_interpreted_predicate_verification",
            executable_engine=verified.policy,
            rules=(verified.policy,),
            inputs=(
                handoff.context,
                handoff.working_context,
                verified.projection,
                verified.frontier,
                verified.model_input,
                verified.distribution,
            ),
            outputs=(handoff.semantic_result,),
            # Embed the receipt instead of introducing a verification store.
            proof_steps=(
                verified.model_dump_json(),
                "scope:INTERPRETED_PREDICATE",
                "predicate:" + handoff.predicate_id,
            ),
            evidence=handoff.evidence,
        )
        decision = DecisionRecord.create(
            organization_id=scope[0],
            workspace_id=scope[1],
            purpose="INFERENCE",
            subject=handoff.subject,
            context=handoff.context,
            ontology=handoff.ontology,
            result_code=handoff.semantic_result,
            epistemic_status="UNKNOWN"
            if handoff.semantic_result == "UNKNOWN"
            else "KNOWN",
            evidence=handoff.evidence,
            admissibility=None,
            reasoning=ReasoningTrace(
                strategy=invocation.strategy,
                reasoning_version=verified.policy.artifact_hash,
                invocations=(invocation,),
                diagnostics=handoff.diagnostics,
            ),
            actor_id=actor_context.actor_id,
            reasoner_identity=verified.policy.artifact_ref,
            authorization=None,
            outcome_observation=None,
            recorded_at=recorded_at,
        )
        self._authority.require_current_actor(actor_context=actor_context)
        self._store.put(decision)
        return decision

    def record_evaluation(self, *, actor_context, episode, basis, projection, hypothesis, compliance_owner):
        """Append a scoped EVALUATION; never mutate the screening Finding."""
        from contracts.ontology_building import ArtifactPinV1
        from contracts.tender_legal_compliance import TenderLegalComplianceBasisV1
        from services.tender_legal_compliance_service import TenderLegalComplianceService
        if (not isinstance(compliance_owner, TenderLegalComplianceService)
            or compliance_owner.store is not self._store or compliance_owner.authority is not self._authority):
            raise TypeError("evaluation requires the same governed compliance owners")
        self._authority.require_current_actor(actor_context=actor_context)
        # The retained REVIEW convention is not a qualified legal-binding
        # producer. Until that owner exists, no supplied normative payload may
        # enter live publication, even if it contains an accepted review pin.
        if basis.normative is not None:
            raise ValueError("QUALIFIED_NORMATIVE_BINDING_OWNER_UNAVAILABLE")
        basis = TenderLegalComplianceBasisV1.model_validate(basis.model_dump(mode="json"))
        scope = dict(organization_id=actor_context.organization_id, workspace_id=actor_context.workspace_id)
        if (basis.organization_id,basis.workspace_id) != (actor_context.organization_id,actor_context.workspace_id):
            raise ValueError("FORBIDDEN_SCOPE")
        for value in (projection.context, projection.ontology, *projection.evidence):
            stored = self._store.get(**scope, artifact_ref=value.artifact_ref, contract_type=type(value))
            if stored != value:
                raise ValueError("UNRESOLVABLE_DECISION_BASIS")
        if projection.context.state != "ADMITTED" or projection.context.ontology != projection.ontology.pin():
            raise ValueError("ADMITTED_CONTEXT_REQUIRED")
        previous = compliance_owner.read_episode(store=self._store,episode=episode)
        expected_assessments = {entry['decision'].pin() for entry in previous if entry['basis'].finding==basis.finding}
        if basis.supersedes_assessment is None and expected_assessments:
            exact = [entry for entry in previous if entry['basis'] == basis]
            if len(exact) == 1:
                compliance_owner.verify_advisory_owners(actor_context=actor_context, basis=basis, projection=projection)
                return exact[0]['decision']
            raise ValueError("ASSESSMENT_SUCCESSOR_SELECTION_CHANGED")
        if basis.supersedes_assessment is not None:
            from contracts.tender_legal_compliance import freeze_assessment_selection
            previous = compliance_owner.read_episode(store=self._store,episode=episode)
            selected = freeze_assessment_selection(inventory=(ArtifactPinV1(artifact_kind="governed_frame_finding",
                artifact_ref=basis.finding.finding_id,artifact_hash=basis.finding.finding_hash),),entries=previous)[0]
            if selected.selected != basis.supersedes_assessment:
                raise ValueError("ASSESSMENT_SUCCESSOR_SELECTION_CHANGED")
            prior = next(entry['basis'] for entry in previous if entry['decision'].pin()==selected.selected)
            if prior.frozen_at >= basis.frozen_at:
                raise ValueError("ASSESSMENT_SUCCESSOR_TIME_MISMATCH")
        compliance_owner.verify_advisory_owners(actor_context=actor_context, basis=basis, projection=projection)
        review = compliance_owner.verify_normative_owners(basis=basis, projection=projection)
        verification = TenderDecisionVerificationService().verify_legal_compliance(
            basis=basis, episode=episode, projection=projection, hypothesis=hypothesis, binding_review=review)
        if verification.status != "VERIFIED":
            raise ValueError("INVALID_BINDING:" + ",".join(verification.diagnostics))
        evidence = tuple(item.pin() for item in projection.evidence)
        if not evidence:
            raise ValueError("SCOPED_EVALUATION_REQUIRES_EVIDENCE")
        subject = ArtifactPinV1(artifact_kind="governed_frame_finding", artifact_ref=basis.finding.finding_id,
                                artifact_hash=basis.finding.finding_hash)
        invocation = ReasonerInvocation(invocation_id=verification.verification_ref,
            strategy="tender_scoped_compliance_evaluation", executable_engine=verification.policy,
            rules=(verification.policy,), inputs=(basis.pin(),basis.episode,basis.inspection_result),
            outputs=(verification.outcome,), proof_steps=(verification.model_dump_json(),), evidence=evidence)
        decision = DecisionRecord.create(**scope, purpose="EVALUATION", subject=subject,
            context=projection.context.pin(), ontology=projection.ontology.pin(), result_code=verification.outcome,
            epistemic_status="UNKNOWN" if verification.outcome in {"UNKNOWN","CONFLICT"} else "KNOWN",
            evidence=evidence, admissibility=None,
            reasoning=ReasoningTrace(strategy=invocation.strategy, reasoning_version=verification.policy.artifact_hash,
                invocations=(invocation,), diagnostics=verification.diagnostics), actor_id=actor_context.actor_id,
            reasoner_identity=verification.policy.artifact_ref, authorization=None, outcome_observation=None,
            recorded_at=basis.frozen_at)
        self._authority.require_current_actor(actor_context=actor_context)
        if basis.advisory is None and basis.supersedes_assessment is None:
            self._store.put_many((basis, decision))
        else:
            def validate(_connection):
                current = compliance_owner.read_episode(store=self._store,episode=episode)
                current_pins = {entry['decision'].pin() for entry in current if entry['basis'].finding==basis.finding}
                if current_pins != expected_assessments:
                    raise ValueError("ASSESSMENT_SUCCESSOR_SELECTION_CHANGED")
                self._authority.require_current_actor(actor_context=actor_context)
                compliance_owner.verify_advisory_owners(actor_context=actor_context, basis=basis,
                    projection=projection, verify_facts=False)
                return True
            # Serialize the live adoption check with revocation in the same
            # existing store, so revoked policy cannot race publication.
            self._store._atomic_write(artifacts=(basis,decision), validate_before_insert=validate,
                mutate_indexes=lambda _connection: None)
        return decision

    def record_conflict_resolution(
        self,
        *,
        actor_context: RuntimeContextDTO,
        recorded_at: datetime,
        projection,
        conflict_membership,
        resolution,
        evidence,
    ) -> DecisionRecord:
        from contracts.decision_graph_projection import DecisionGraphProjectionV1
        from contracts.decision_verification import (
            ConflictMembershipReceiptV1,
            ConflictResolutionReceiptV1,
        )

        self._authority.require_current_actor(actor_context=actor_context)
        projection = DecisionGraphProjectionV1.model_validate(
            projection.model_dump(mode="json") if hasattr(projection, "model_dump") else projection
        )
        membership = ConflictMembershipReceiptV1.model_validate(
            conflict_membership.model_dump(mode="json")
            if hasattr(conflict_membership, "model_dump")
            else conflict_membership
        )
        resolved = ConflictResolutionReceiptV1.model_validate(
            resolution.model_dump(mode="json")
            if hasattr(resolution, "model_dump")
            else resolution
        )
        projection_pin = projection.pin()
        if membership.projection != projection_pin or resolved.projection != projection_pin:
            raise ValueError("CONFLICT_RESOLUTION_PROJECTION_MISMATCH")
        if resolved.conflict_membership != membership.pin():
            raise ValueError("CONFLICT_RESOLUTION_MEMBERSHIP_MISMATCH")

        scope = (actor_context.organization_id, actor_context.workspace_id)
        if scope != (projection.organization_id, projection.workspace_id):
            raise ValueError("FORBIDDEN_SCOPE")

        context = self._store.get(
            organization_id=scope[0],
            workspace_id=scope[1],
            artifact_ref=projection.canonical_context.artifact_ref,
            contract_type=ContextGraph,
        )
        if (
            context is None
            or context.pin() != projection.canonical_context
            or context.state != "ADMITTED"
            or context.ontology != projection.ontology
        ):
            raise ValueError("ADMITTED_CONTEXT_REQUIRED")
        ontology = self._store.get(
            organization_id=scope[0],
            workspace_id=scope[1],
            artifact_ref=projection.ontology.artifact_ref,
            contract_type=Ontology,
        )
        if ontology is None or ontology.pin() != projection.ontology:
            raise ValueError("UNRESOLVABLE_DECISION_BASIS")

        evidence = tuple(evidence)
        if not evidence:
            raise ValueError("SEMANTIC_INFERENCE_REQUIRES_EVIDENCE")
        if tuple(projection.evidence) != tuple(sorted(
            evidence,
            key=lambda pin: (
                pin.artifact_kind.encode("utf-8"),
                pin.artifact_ref.encode("utf-8"),
                pin.artifact_hash.encode("utf-8"),
            ),
        )):
            raise ValueError("CONFLICT_RESOLUTION_EVIDENCE_MISMATCH")
        for pin in evidence:
            value = self._store.get(
                organization_id=scope[0],
                workspace_id=scope[1],
                artifact_ref=pin.artifact_ref,
                contract_type=Evidence,
            )
            if value is None or value.pin() != pin:
                raise ValueError("UNRESOLVABLE_DECISION_BASIS")

        invocation = ReasonerInvocation(
            invocation_id=resolved.receipt_ref,
            strategy="tender_conflict_resolution",
            executable_engine=resolved.pin(),
            rules=(membership.pin(),),
            inputs=(
                projection_pin,
                membership.left_factor,
                membership.right_factor,
                membership.pin(),
            ),
            outputs=(resolved.resolution,),
            proof_steps=(
                membership.model_dump_json(),
                resolved.model_dump_json(),
                "scope:CONFLICT_RESOLUTION",
            ),
            evidence=evidence,
        )
        decision = DecisionRecord.create(
            organization_id=scope[0],
            workspace_id=scope[1],
            purpose="INFERENCE",
            subject=resolved.pin(),
            context=projection.canonical_context,
            ontology=projection.ontology,
            result_code=resolved.resolution,
            epistemic_status="UNKNOWN" if resolved.resolution == "UNKNOWN" else "KNOWN",
            evidence=evidence,
            admissibility=None,
            reasoning=ReasoningTrace(
                strategy=invocation.strategy,
                reasoning_version=resolved.receipt_hash,
                invocations=(invocation,),
                diagnostics=(
                    "decision_graph_projection:" + projection.projection_ref,
                    "conflict_membership:" + membership.receipt_ref,
                    "conflict_resolution:" + resolved.receipt_ref,
                ),
            ),
            actor_id=actor_context.actor_id,
            reasoner_identity="deterministic-conflict-resolution:v1",
            authorization=None,
            outcome_observation=None,
            recorded_at=recorded_at,
        )
        self._authority.require_current_actor(actor_context=actor_context)
        self._store.put(decision)
        return decision
