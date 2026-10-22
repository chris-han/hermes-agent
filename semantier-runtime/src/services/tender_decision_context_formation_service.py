"""Form an authorized kernel WorkingContext for one Tender semantic decision."""

from __future__ import annotations

from datetime import datetime, timezone

from contracts.evidence_location import EvidenceRef
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.observed_constraint_hypothesis import RuntimeSemanticRetrievalSnapshotV1
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_semantic_source_region import RuntimeSemanticSourceRegionV1
from contracts.semantic_kernel import (
    ContextAssertion,
    ContextNode,
    ContextWorld,
    Evidence,
    FormationInput,
    IntentSnapshot,
    Ontology,
    OntologyEntityType,
    OntologyRelationType,
    ReasonerInvocation,
    ReasoningTrace,
    WorkingContext,
)
from eos.artifact_hashing import content_hash
from services.context_graph_kernel_service import ContextGraphKernelService
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from services.working_context_projection_service import WorkingContextProjectionService
from storage.ontology_build_store import OntologyBuildStore


class TenderDecisionContextFormationService:
    """Bridge governed source-region evidence into the existing five-object kernel."""

    def __init__(
        self,
        *,
        build_store: OntologyBuildStore,
        authority_resolver: RuntimeAuthorityResolver,
        runtime_actor_id: str = "agent:tender-semantic-runtime",
    ) -> None:
        if not isinstance(build_store, OntologyBuildStore):
            raise TypeError("Tender decision context formation requires OntologyBuildStore")
        if not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("Tender decision context formation requires RuntimeAuthorityResolver")
        if not runtime_actor_id.strip():
            raise ValueError("runtime_actor_id is required")
        self._store = build_store
        self._authority = authority_resolver
        self._runtime_actor_id = runtime_actor_id

    @staticmethod
    def _release_pin(snapshot: RuntimeSemanticRetrievalSnapshotV1) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="accepted_graph_release",
            artifact_ref=snapshot.accepted_graph_release_ref,
            artifact_hash=snapshot.accepted_graph_release_hash,
        )

    @staticmethod
    def _retrieval_pin(snapshot: RuntimeSemanticRetrievalSnapshotV1) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="runtime_semantic_retrieval_snapshot",
            artifact_ref=snapshot.snapshot_id,
            artifact_hash=snapshot.snapshot_hash,
        )

    @staticmethod
    def formation_policy() -> ArtifactPinV1:
        payload = {
            "policy": "tender_decision_context_formation",
            "version": "v1",
            "selection": "exact source-region evidence plus one pinned canonical pattern",
            "authority": "accepted graph release remains source authority",
        }
        digest = content_hash(payload)
        return ArtifactPinV1(
            artifact_kind="context_formation_policy",
            artifact_ref=f"tender-decision-context-formation:v1:{digest}",
            artifact_hash=f"sha256:{digest}",
        )

    def _ontology(self, *, organization_id: str, workspace_id: str, recorded_at: datetime) -> Ontology:
        ontology = Ontology.create(
            organization_id=organization_id,
            workspace_id=workspace_id,
            version="tender-decision-context-projection-v1",
            predecessor=None,
            entity_types=(
                OntologyEntityType(type_id="SourceRegion", label="Source Region"),
                OntologyEntityType(type_id="CanonicalPattern", label="Canonical Pattern"),
            ),
            relation_types=(
                OntologyRelationType(
                    type_id="candidate_for",
                    label="candidate for governed semantic evaluation",
                    directional=True,
                ),
            ),
            rules=(),
            semantic_frames=(),
            provenance=(),
            recorded_at=recorded_at,
        )
        self._store.put(ontology)
        return ontology

    @staticmethod
    def _kernel_evidence(ref: EvidenceRef) -> Evidence:
        provenance = ref.provenance
        canonical_ref = str(provenance.get("canonical_source_ir_ref") or "")
        canonical_hash = str(provenance.get("canonical_source_ir_hash") or "")
        pins = ()
        if canonical_ref and canonical_hash:
            pins = (
                ArtifactPinV1(
                    artifact_kind="canonical_source_ir",
                    artifact_ref=canonical_ref,
                    artifact_hash=canonical_hash,
                ),
            )
        return Evidence.create(
            organization_id=ref.organization_id,
            workspace_id=ref.workspace_id,
            source_ref=ref.source_identity_ref,
            source_hash=str(provenance.get("source_hash") or ""),
            selector=ref.selector,
            selector_hash=ref.selector_hash,
            evidence_role=ref.evidence_role,
            provenance=pins,
            observed_at=ref.created_at,
            source_time=None,
            recorded_at=ref.created_at,
        )

    def form(
        self,
        *,
        actor_context: RuntimeContextDTO,
        source_region: RuntimeSemanticSourceRegionV1,
        retrieval_snapshot: RuntimeSemanticRetrievalSnapshotV1,
        canonical_pattern_ref: str,
        formed_at: datetime | None = None,
    ) -> WorkingContext:
        formed_at = (formed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if not actor_context.workspace_id:
            raise ValueError("Tender decision context requires workspace scope")
        scope = (actor_context.organization_id, actor_context.workspace_id)
        first = source_region.evidence[0].evidence_ref
        if scope != (first.organization_id, first.workspace_id) or scope != (
            retrieval_snapshot.organization_id,
            retrieval_snapshot.workspace_id,
        ):
            raise ValueError("TENDER_DECISION_CONTEXT_SCOPE_MISMATCH")
        matches = tuple(
            item
            for item in retrieval_snapshot.ordered_material
            if item.kind == "canonical_pattern" and item.ref == canonical_pattern_ref
        )
        if len(matches) != 1:
            raise ValueError("TENDER_CANONICAL_PATTERN_NOT_PINNED")
        pattern = matches[0]

        evidence = tuple(self._kernel_evidence(item.evidence_ref) for item in source_region.evidence)
        self._store.put_many(evidence)
        ontology = self._ontology(
            organization_id=actor_context.organization_id,
            workspace_id=actor_context.workspace_id,
            recorded_at=formed_at,
        )
        release_pin = self._release_pin(retrieval_snapshot)
        retrieval_pin = self._retrieval_pin(retrieval_snapshot)
        policy = self.formation_policy()
        intent = IntentSnapshot.create(
            organization_id=actor_context.organization_id,
            workspace_id=actor_context.workspace_id,
            goal="Evaluate one bounded Tender restriction semantic predicate",
            competency_questions=(
                "Does the exact source region establish the pinned Tender restriction pattern?",
            ),
            hard_negatives=(
                "Retrieval similarity alone must not become a positive semantic finding.",
            ),
            material_dimensions=("restriction", "evidence_binding", "runtime_authority"),
            source_roles=tuple(sorted({item.evidence_role for item in evidence})),
            required_semantics=("semantic_predicate",),
            unresolved_questions=(),
            captured_at=formed_at,
        )
        formation_input = FormationInput.create(
            organization_id=actor_context.organization_id,
            workspace_id=actor_context.workspace_id,
            intent=intent,
            ontology=ontology.pin(),
            evidence=tuple(item.pin() for item in evidence),
            policy_pins=(policy,),
            predecessor_context=None,
            captured_at=formed_at,
        )
        target_ref = source_region.source_region_ref
        pattern_node_ref = f"canonical-pattern:{pattern.ref}"
        assertion_id = f"decision-candidate:{content_hash({'target': target_ref, 'pattern': pattern.ref, 'retrieval': snapshot_identity(retrieval_snapshot)})}"
        assertion = ContextAssertion(
            assertion_id=assertion_id,
            subject_ref=target_ref,
            predicate="candidate_for",
            object_ref=pattern_node_ref,
            origin="DERIVED",
            evidence=tuple(item.pin() for item in evidence),
            derivation=retrieval_pin,
        )
        world = ContextWorld(
            source_graph=release_pin,
            mode="current",
            at_time=None,
            start_time=None,
            end_time=None,
            reasoning_version="tender-decision-context-projection-v1",
            governance_time=formed_at,
        )

        def verify_external(pin: ArtifactPinV1) -> None:
            if pin == policy or pin == release_pin:
                return
            raise ValueError("UNTRUSTED_TENDER_CONTEXT_EXTERNAL_PIN")

        kernel = ContextGraphKernelService(
            store=self._store,
            verify_external_pin=verify_external,
            verify_authorization=lambda basis: self._authority.verify_authorization_basis(
                actor_context=actor_context,
                basis=basis,
            ),
        )
        proposed = kernel.form_context(
            formation_input=formation_input,
            world=world,
            nodes=(
                ContextNode(node_id=target_ref, node_type="SourceRegion", label=source_region.text),
                ContextNode(
                    node_id=pattern_node_ref,
                    node_type="CanonicalPattern",
                    label=canonical_pattern_ref,
                ),
            ),
            edges=(),
            assertions=(assertion,),
            selected_evidence=tuple(item.pin() for item in evidence),
            excluded_evidence=(),
            conflicts=(),
            formation_policy=policy,
            formation_policy_version="tender-decision-context-formation.v1",
            applied_change_candidate=None,
            recorded_at=formed_at,
            persist=False,
        )
        authorization = self._authority.current_actor_authorization_basis(
            actor_context=actor_context,
            operation="ADMIT_CONTEXT",
            subject=proposed.pin(),
            authorization_class="TENANT_MEMBER",
        )
        reasoning = ReasoningTrace(
            strategy="tender_decision_context_formation",
            reasoning_version=policy.artifact_hash,
            invocations=(
                ReasonerInvocation(
                    invocation_id=f"tender-context:{content_hash({'proposal': proposed.artifact_hash, 'retrieval': retrieval_snapshot.snapshot_hash})}",
                    strategy="exact_source_region_plus_retrieval_candidate",
                    executable_engine=policy,
                    rules=(retrieval_pin,),
                    inputs=(release_pin, retrieval_pin, *tuple(item.pin() for item in evidence)),
                    outputs=(assertion_id,),
                    proof_steps=(
                        "source-region EvidenceRefs converted to exact kernel Evidence",
                        "canonical pattern selected only from the pinned retrieval snapshot",
                        "accepted graph release retained as ContextWorld source authority",
                    ),
                    evidence=tuple(item.pin() for item in evidence),
                ),
            ),
            working_context=None,
            derived_assertion_ids=(assertion_id,),
            diagnostics=(),
        )
        request_hash = "sha256:" + content_hash(
            {
                "operation": "FORM_TENDER_DECISION_CONTEXT",
                "source_region_ref": source_region.source_region_ref,
                "retrieval_snapshot_ref": retrieval_snapshot.snapshot_id,
                "retrieval_snapshot_hash": retrieval_snapshot.snapshot_hash,
                "canonical_pattern_ref": canonical_pattern_ref,
                "actor_id": actor_context.actor_id,
            }
        )
        _decision, admitted = kernel.admit_new_context_with_request(
            proposed=proposed,
            authorization=authorization,
            reasoning=reasoning,
            evidence=tuple(item.pin() for item in evidence),
            actor_id=actor_context.actor_id,
            reasoner_identity=policy.artifact_ref,
            recorded_at=formed_at,
            request_id=f"tender-decision-context:{request_hash[7:31]}",
            request_hash=request_hash,
            operation="FORM_TENDER_DECISION_CONTEXT",
        )
        return WorkingContextProjectionService(
            build_store=self._store,
            authority_resolver=self._authority,
            runtime_actor_id=self._runtime_actor_id,
        ).project_for_reasoning(
            actor_context=actor_context,
            context=admitted.pin(),
            task_ref=f"tender-semantic:{source_region.source_region_ref}:{canonical_pattern_ref}",
            task_goal="evaluate bounded Tender restriction semantic predicate",
            selected_assertion_ids=(assertion_id,),
            delegated_operation="EVALUATE_TENDER_PREDICATE",
            commitment_mode="AUTO",
            formed_at=formed_at,
        )


def snapshot_identity(snapshot: RuntimeSemanticRetrievalSnapshotV1) -> str:
    return f"{snapshot.snapshot_id}@{snapshot.snapshot_hash}"


__all__ = ["TenderDecisionContextFormationService"]
