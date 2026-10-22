"""Sealed creation and publication of governed frame findings."""

from pathlib import Path

from contracts.governed_frame_finding_v1 import create_governed_frame_finding_v1
from services.tender_finding_decision_service import TenderFindingDecisionService
from storage.runtime_semantic_hypothesis_artifact_store import (
    RuntimeSemanticHypothesisArtifactStore,
    _FRAME_FINDING_WRITE_CAPABILITY,
)


class GovernedFrameFindingService:
    def __init__(self, *, runtime_root: str | Path) -> None:
        self._runtime_root = Path(runtime_root)

    def create_and_persist(
        self, *, organization_id: str, workspace_id: str,
        frame_match_result_ref: str, frame_match_result_hash: str,
    ):
        store = RuntimeSemanticHypothesisArtifactStore(
            self._runtime_root, organization_id=organization_id, workspace_id=workspace_id
        )
        result = store.get_frame_match_result(
            result_ref=frame_match_result_ref, result_hash=frame_match_result_hash
        )
        decision = TenderFindingDecisionService().decide_frame_match_v2(frame_match=result)
        finding = create_governed_frame_finding_v1(
            frame_match=result, decision=decision
        )
        return store._put_governed_frame_finding(
            finding, capability=_FRAME_FINDING_WRITE_CAPABILITY
        )


__all__ = ["GovernedFrameFindingService"]
