"""W6 unified Semantier learning runtime across Realtime, Sleeptime, and Slow-clock eligibility."""

from __future__ import annotations

from contracts.artifact_pin import ArtifactPinV1
from contracts.episode_acquisition import AcquisitionDecisionV1
from contracts.governed_decision_episode import GovernedDecisionEpisodeV1
from contracts.intervention_comparison import ComparisonCompatibilityV1
from contracts.learning_work_window import LearningWorkWindowV1
from contracts.stage_attribution import StageAttributionV1
from semantier.cognitive_context.store import CognitiveContextStore


class UnifiedLearningRuntimeService:
    """Plan clock-eligible learning work without admission, activation, or commitment authority."""

    def __init__(self, *, store: CognitiveContextStore) -> None:
        if not isinstance(store, CognitiveContextStore):
            raise TypeError("learning runtime requires CognitiveContextStore")
        self._store = store

    def _load(
        self,
        *,
        pin: ArtifactPinV1,
        organization_id: str,
        workspace_id: str,
        contract_type,
    ):
        value = self._store.get_derived_projection(
            pin.artifact_ref,
            organization_id=organization_id,
            workspace_id=workspace_id,
            contract_type=contract_type,
        )
        actual_pin = ArtifactPinV1(
            artifact_kind=pin.artifact_kind,
            artifact_ref=pin.artifact_ref,
            artifact_hash=pin.artifact_hash,
        )
        if isinstance(value, GovernedDecisionEpisodeV1):
            resolved = ArtifactPinV1(
                artifact_kind="governed_decision_episode",
                artifact_ref=value.episode_ref,
                artifact_hash=value.episode_hash,
            )
        elif isinstance(value, AcquisitionDecisionV1):
            resolved = ArtifactPinV1(
                artifact_kind="acquisition_decision",
                artifact_ref=value.receipt_ref,
                artifact_hash=value.receipt_hash,
            )
        elif isinstance(value, StageAttributionV1):
            resolved = value.pin()
        elif isinstance(value, ComparisonCompatibilityV1):
            resolved = ArtifactPinV1(
                artifact_kind="comparison_compatibility",
                artifact_ref=value.receipt_ref,
                artifact_hash=value.receipt_hash,
            )
        else:
            raise TypeError("unsupported learning-runtime dependency")
        if resolved != actual_pin:
            raise ValueError("LEARNING_RUNTIME_DEPENDENCY_BINDING_MISMATCH")
        return value

    def open_window(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        execution_mode: str,
        clock: str,
        loop: str,
        eligibility_basis: str,
        episodes: tuple[ArtifactPinV1, ...],
        acquisitions: tuple[ArtifactPinV1, ...] = (),
        attributions: tuple[ArtifactPinV1, ...] = (),
        comparison_receipts: tuple[ArtifactPinV1, ...] = (),
        independent_evaluation_basis: ArtifactPinV1 | None = None,
        scheduler_wake: ArtifactPinV1 | None = None,
        checkpoint: ArtifactPinV1,
        resource_policy: ArtifactPinV1,
    ) -> LearningWorkWindowV1:
        for pin in episodes:
            self._load(
                pin=pin,
                organization_id=organization_id,
                workspace_id=workspace_id,
                contract_type=GovernedDecisionEpisodeV1,
            )
        acquisition_values = tuple(
            self._load(
                pin=pin,
                organization_id=organization_id,
                workspace_id=workspace_id,
                contract_type=AcquisitionDecisionV1,
            )
            for pin in acquisitions
        )
        attribution_values = tuple(
            self._load(
                pin=pin,
                organization_id=organization_id,
                workspace_id=workspace_id,
                contract_type=StageAttributionV1,
            )
            for pin in attributions
        )
        comparison_values = tuple(
            self._load(
                pin=pin,
                organization_id=organization_id,
                workspace_id=workspace_id,
                contract_type=ComparisonCompatibilityV1,
            )
            for pin in comparison_receipts
        )

        if any(not item.eligible for item in acquisition_values):
            raise ValueError("LEARNING_WINDOW_REQUIRES_ELIGIBLE_ACQUISITIONS")
        if clock == "SLOW":
            if any(not item.hypothesis_eligible for item in attribution_values):
                raise ValueError("SLOW_CLOCK_REQUIRES_ELIGIBLE_ATTRIBUTIONS")
            if any(item.status != "COMPARABLE" for item in comparison_values):
                raise ValueError("SLOW_CLOCK_REQUIRES_COMPARABLE_HISTORY")

        window = LearningWorkWindowV1.create(
            organization_id=organization_id,
            workspace_id=workspace_id,
            execution_mode=execution_mode,
            clock=clock,
            loop=loop,
            eligibility_basis=eligibility_basis,
            episodes=episodes,
            acquisitions=acquisitions,
            attributions=attributions,
            comparison_receipts=comparison_receipts,
            independent_evaluation_basis=independent_evaluation_basis,
            scheduler_wake=scheduler_wake,
            checkpoint=checkpoint,
            resource_policy=resource_policy,
            foreground_decoupled=True,
        )
        return self._store.save_or_get_derived_projection(window)

    def replay(
        self,
        *,
        window_ref: str,
        organization_id: str,
        workspace_id: str,
    ) -> LearningWorkWindowV1:
        return self._store.get_derived_projection(
            window_ref,
            organization_id=organization_id,
            workspace_id=workspace_id,
            contract_type=LearningWorkWindowV1,
        )
