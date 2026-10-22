"""W3 stage attribution and bounded structural-hypothesis proposal service."""

from __future__ import annotations

from typing import Protocol

from contracts.artifact_pin import ArtifactPinV1
from contracts.candidate_stage_binding import OwningStage
from contracts.episode_acquisition import AcquisitionDecisionV1
from contracts.governed_decision_episode import GovernedDecisionEpisodeV1
from contracts.institutional_adaptation import (
    InstitutionalAdaptationIngressV1,
    InstitutionalDivergenceDetectionV1,
)
from contracts.semantic_kernel import ChangeDelta
from contracts.stage_attribution import (
    OptimizerProposalV1,
    StageAttributionV1,
    StructuralHypothesisV1,
)
from eos.artifact_hashing import content_hash
from semantier.cognitive_context.store import CognitiveContextStore
from services.bounded_candidate_generation_service import CandidateDraft
from services.candidate_stage_routing_service import CandidateStageRoutingService


class StructuralHypothesisOptimizer(Protocol):
    def propose(
        self,
        *,
        attribution: StageAttributionV1,
        selected_episodes: tuple[GovernedDecisionEpisodeV1, ...],
        optional_human_evidence: tuple[ArtifactPinV1, ...],
        predecessor_candidate: ArtifactPinV1 | None,
        allowed_candidate_delta_schema: ArtifactPinV1,
        optimizer_config: ArtifactPinV1,
        budget: ArtifactPinV1,
    ) -> OptimizerProposalV1:
        ...


class DeterministicStructuralHypothesisOptimizer:
    """Test/reference adapter. It cannot persist or mutate governed state."""

    def __init__(
        self,
        *,
        optimizer_ref: str,
        hypothesis: str,
        delta: ChangeDelta,
        search_lane: str = "REFINE",
    ) -> None:
        self._optimizer_ref = optimizer_ref
        self._hypothesis = hypothesis
        self._delta = delta
        self._search_lane = search_lane

    def propose(self, **kwargs) -> OptimizerProposalV1:
        attribution = kwargs["attribution"]
        if not attribution.hypothesis_eligible:
            raise ValueError("ATTRIBUTION_NOT_HYPOTHESIS_ELIGIBLE")
        payload = {
            "optimizer_ref": self._optimizer_ref,
            "hypothesis": self._hypothesis,
            "delta": self._delta.model_dump(mode="json"),
            "search_lane": self._search_lane,
            "attribution": attribution.attribution_hash,
            "episodes": [item.episode_hash for item in kwargs["selected_episodes"]],
            "optimizer_config": kwargs["optimizer_config"].model_dump(mode="json"),
            "budget": kwargs["budget"].model_dump(mode="json"),
        }
        return OptimizerProposalV1(
            hypothesis=self._hypothesis,
            delta=self._delta,
            search_lane=self._search_lane,
            optimizer_ref=self._optimizer_ref,
            optimizer_hash="sha256:" + content_hash(payload),
            optimizer_config=kwargs["optimizer_config"],
            budget=kwargs["budget"],
        )


class StageAttributionOptimizationService:
    def __init__(self, *, store: CognitiveContextStore) -> None:
        self._store = store
        self._routing = CandidateStageRoutingService()

    def attribute(
        self,
        *,
        episode: GovernedDecisionEpisodeV1,
        acquisition: AcquisitionDecisionV1,
        failure_class: str,
        owning_stage: OwningStage,
        target: str,
        evidence: tuple[ArtifactPinV1, ...],
        observable: bool,
        intervenable: bool,
        outcome_sensitive: bool,
        cross_episode_reusable: bool,
        diagnostics: tuple[str, ...] = (),
        divergence_detection: InstitutionalDivergenceDetectionV1 | None = None,
        adaptation_ingress: InstitutionalAdaptationIngressV1 | None = None,
    ) -> StageAttributionV1:
        if not acquisition.eligible:
            raise ValueError("ACQUISITION_NOT_ELIGIBLE")
        if (
            acquisition.episode_ref,
            acquisition.episode_hash,
            acquisition.organization_id,
            acquisition.workspace_id,
        ) != (
            episode.episode_ref,
            episode.episode_hash,
            episode.organization_id,
            episode.workspace_id,
        ):
            raise ValueError("ATTRIBUTION_ACQUISITION_EPISODE_MISMATCH")

        acquisition_pin = ArtifactPinV1(
            artifact_kind="acquisition_decision",
            artifact_ref=acquisition.receipt_ref,
            artifact_hash=acquisition.receipt_hash,
        )
        if divergence_detection is not None:
            if (
                divergence_detection.organization_id,
                divergence_detection.workspace_id,
            ) != (episode.organization_id, episode.workspace_id):
                raise ValueError("DIVERGENCE_DETECTION_SCOPE_MISMATCH")
            self._store.save_or_get_derived_projection(divergence_detection)
            if divergence_detection.classification == "UNRESOLVED_MATERIALITY":
                raise ValueError("INSTITUTIONAL_DIVERGENCE_MATERIALITY_UNRESOLVED")
            if divergence_detection.classification == "MATERIAL_INSTITUTIONAL_DIVERGENCE":
                if adaptation_ingress is None:
                    raise ValueError("PRE_LEARNING_TRIAGE_REQUIRED")
            elif adaptation_ingress is not None:
                raise ValueError("UNEXPECTED_ADAPTATION_INGRESS")
        elif adaptation_ingress is not None:
            raise ValueError("ADAPTATION_INGRESS_REQUIRES_DIVERGENCE_DETECTION")

        if adaptation_ingress is not None:
            if (
                adaptation_ingress.organization_id,
                adaptation_ingress.workspace_id,
            ) != (episode.organization_id, episode.workspace_id):
                raise ValueError("ADAPTATION_INGRESS_SCOPE_MISMATCH")
            if adaptation_ingress.acquisition != acquisition_pin:
                raise ValueError("ADAPTATION_INGRESS_ACQUISITION_MISMATCH")
            if adaptation_ingress.divergence_detection != divergence_detection.pin():
                raise ValueError("ADAPTATION_INGRESS_DETECTION_MISMATCH")
            self._store.save_or_get_derived_projection(adaptation_ingress)
            if adaptation_ingress.route != "ATTRIBUTE":
                raise ValueError(
                    f"ADAPTATION_INGRESS_ROUTE_BLOCKS_ATTRIBUTION:{adaptation_ingress.route}"
                )

        binding = self._routing.bind(
            target=target,
            failure_class=failure_class,
            owning_stage=owning_stage,
        )
        attribution = StageAttributionV1.create(
            organization_id=episode.organization_id,
            workspace_id=episode.workspace_id,
            episode=ArtifactPinV1(
                artifact_kind="governed_decision_episode",
                artifact_ref=episode.episode_ref,
                artifact_hash=episode.episode_hash,
            ),
            acquisition=acquisition_pin,
            institutional_adaptation_ingress=(
                adaptation_ingress.pin() if adaptation_ingress is not None else None
            ),
            failure_class=failure_class,
            stage_binding=binding,
            evidence=evidence,
            observable=observable,
            intervenable=intervenable,
            outcome_sensitive=outcome_sensitive,
            cross_episode_reusable=cross_episode_reusable,
            diagnostics=diagnostics,
        )
        return self._store.save_or_get_derived_projection(attribution)

    def propose_hypothesis(
        self,
        *,
        attribution: StageAttributionV1,
        selected_episodes: tuple[GovernedDecisionEpisodeV1, ...],
        optimizer: StructuralHypothesisOptimizer,
        optional_human_evidence: tuple[ArtifactPinV1, ...],
        predecessor_candidate: ArtifactPinV1 | None,
        allowed_candidate_delta_schema: ArtifactPinV1,
        optimizer_config: ArtifactPinV1,
        budget: ArtifactPinV1,
    ) -> StructuralHypothesisV1:
        if not attribution.hypothesis_eligible:
            raise ValueError("ATTRIBUTION_NOT_HYPOTHESIS_ELIGIBLE")
        if not selected_episodes:
            raise ValueError("STRUCTURAL_HYPOTHESIS_REQUIRES_EPISODES")
        if any(
            (item.organization_id, item.workspace_id)
            != (attribution.organization_id, attribution.workspace_id)
            for item in selected_episodes
        ):
            raise ValueError("STRUCTURAL_HYPOTHESIS_SCOPE_MISMATCH")
        proposal = optimizer.propose(
            attribution=attribution,
            selected_episodes=selected_episodes,
            optional_human_evidence=optional_human_evidence,
            predecessor_candidate=predecessor_candidate,
            allowed_candidate_delta_schema=allowed_candidate_delta_schema,
            optimizer_config=optimizer_config,
            budget=budget,
        )
        hypothesis = StructuralHypothesisV1.create(
            organization_id=attribution.organization_id,
            workspace_id=attribution.workspace_id,
            attribution=attribution.pin(),
            owning_stage=attribution.stage_binding.owning_stage,
            stage_binding=attribution.stage_binding,
            selected_episodes=tuple(
                ArtifactPinV1(
                    artifact_kind="governed_decision_episode",
                    artifact_ref=item.episode_ref,
                    artifact_hash=item.episode_hash,
                )
                for item in selected_episodes
            ),
            optional_human_evidence=optional_human_evidence,
            predecessor_candidate=predecessor_candidate,
            allowed_candidate_delta_schema=allowed_candidate_delta_schema,
            proposal=proposal,
        )
        return self._store.save_or_get_derived_projection(hypothesis)

    def replay_divergence_detection(
        self,
        *,
        detection_ref: str,
        organization_id: str,
        workspace_id: str,
    ) -> InstitutionalDivergenceDetectionV1:
        return self._store.get_derived_projection(
            detection_ref,
            organization_id=organization_id,
            workspace_id=workspace_id,
            contract_type=InstitutionalDivergenceDetectionV1,
        )

    def replay_adaptation_ingress(
        self,
        *,
        ingress_ref: str,
        organization_id: str,
        workspace_id: str,
    ) -> InstitutionalAdaptationIngressV1:
        return self._store.get_derived_projection(
            ingress_ref,
            organization_id=organization_id,
            workspace_id=workspace_id,
            contract_type=InstitutionalAdaptationIngressV1,
        )

    @staticmethod
    def to_candidate_draft(hypothesis: StructuralHypothesisV1) -> CandidateDraft:
        target = hypothesis.stage_binding.target
        surface = {
            "ONTOLOGY": "OBJECT_ONTOLOGY",
            "POLICY": "OBJECT_POLICY",
            "CONTEXT_FORMATION": "OBJECT_CONTEXT_FORMATION",
            "REASONING": "OBJECT_REASONING",
        }[target]
        delta = ChangeDelta(
            description=hypothesis.proposal.delta.description,
            claims_semantic_graph_change=hypothesis.proposal.delta.claims_semantic_graph_change,
            basis_refs=(
                *hypothesis.proposal.delta.basis_refs,
                hypothesis.pin(),
            ),
            structural_diff=hypothesis.proposal.delta.structural_diff,
        )
        return CandidateDraft(
            change_surface=surface,
            search_lane=hypothesis.proposal.search_lane,
            target=target,
            failure_class=hypothesis.stage_binding.failure_class,
            hypothesis=hypothesis.proposal.hypothesis,
            delta=delta,
            source_candidate_refs=(
                (hypothesis.predecessor_candidate,)
                if hypothesis.predecessor_candidate is not None
                else ()
            ),
        )
