"""Resolve governed evidence and publish an append-only finding correction."""

from __future__ import annotations

from pathlib import Path
from contracts.governed_frame_finding_correction_v1 import (
    create_governed_frame_finding_correction_v1,
)
from contracts.knowledge_feature_pack import LearningEventEnvelope
from storage.governed_frame_finding_correction_store import (
    GovernedFrameFindingCorrectionStore,
    _FINDING_CORRECTION_WRITE_CAPABILITY,
)
from storage.human_grounding_store import HumanGroundingStore
from storage.runtime_semantic_hypothesis_artifact_store import (
    RuntimeSemanticHypothesisArtifactStore,
)


class GovernedFrameFindingCorrectionService:
    def __init__(
        self,
        *,
        runtime_root: str | Path,
        grounding_store: HumanGroundingStore,
    ) -> None:
        self._runtime_root = Path(runtime_root)
        self._grounding_store = grounding_store

    def create_and_persist(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        finding_ref: str,
        finding_hash: str,
        grounding_event_ref: str,
        grounding_event_hash: str,
    ):
        finding = RuntimeSemanticHypothesisArtifactStore(
            self._runtime_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
        ).get_governed_frame_finding(
            finding_ref=finding_ref,
            finding_hash=finding_hash,
        )
        stored_event = self._grounding_store.get_learning_event(grounding_event_ref)
        if stored_event is None:
            raise ValueError("committed Human Grounding event not found")
        actual_hash = stored_event.pop("event_hash")
        if actual_hash != grounding_event_hash:
            raise ValueError("committed Human Grounding event hash mismatch")
        event = LearningEventEnvelope.model_validate(stored_event)
        correction = create_governed_frame_finding_correction_v1(
            finding=finding,
            grounding_event=event,
            grounding_event_hash=actual_hash,
        )
        return GovernedFrameFindingCorrectionStore(
            self._runtime_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )._put(correction, capability=_FINDING_CORRECTION_WRITE_CAPABILITY)

    def get_verified(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        correction_ref: str,
        correction_hash: str,
    ):
        correction = GovernedFrameFindingCorrectionStore(
            self._runtime_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )._get(correction_ref=correction_ref, correction_hash=correction_hash)
        finding = RuntimeSemanticHypothesisArtifactStore(
            self._runtime_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
        ).get_governed_frame_finding(
            finding_ref=correction.predecessor_finding_ref,
            finding_hash=correction.predecessor_finding_hash,
        )
        stored_event = self._grounding_store.get_learning_event(
            correction.grounding_event_ref
        )
        if stored_event is None:
            raise ValueError("committed Human Grounding event not found")
        actual_hash = stored_event.pop("event_hash")
        if actual_hash != correction.grounding_event_hash:
            raise ValueError("committed Human Grounding event hash mismatch")
        expected = create_governed_frame_finding_correction_v1(
            finding=finding,
            grounding_event=LearningEventEnvelope.model_validate(stored_event),
            grounding_event_hash=actual_hash,
        )
        if expected != correction:
            raise ValueError("finding correction differs from governed predecessors")
        return correction


__all__ = ["GovernedFrameFindingCorrectionService"]
