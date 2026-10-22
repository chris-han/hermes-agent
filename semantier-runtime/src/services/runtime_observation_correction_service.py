"""Human-Grounding-authorized runtime observation hypothesis successors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from contracts.knowledge_feature_pack import HumanDecision, LearningEventEnvelope, LearningEventType
from contracts.observed_constraint_hypothesis import create_observed_constraint_hypothesis
from contracts.runtime_observation_correction_lineage_v1 import (
    create_runtime_observation_correction_lineage_v1,
)
from services.runtime_semantic_hypothesis_generation_service import _runtime_context
from storage.human_grounding_store import HumanGroundingStore
from storage.runtime_semantic_hypothesis_artifact_store import (
    RuntimeSemanticHypothesisArtifactStore,
)
from storage.runtime_observation_correction_store import (
    RuntimeObservationCorrectionStore,
    _RUNTIME_OBSERVATION_CORRECTION_WRITE_CAPABILITY,
)


_OUTCOME_FIELDS = {
    "schema_version",
    "selected_candidate_frame_indexes",
    "supporting_evidence_refs",
    "counter_evidence_refs",
    "uncertainty",
}


class RuntimeObservationCorrectionService:
    def __init__(
        self,
        *,
        runtime_root: str | Path,
        grounding_store: HumanGroundingStore,
    ) -> None:
        self._runtime_root = Path(runtime_root)
        self._grounding_store = grounding_store

    def correct_and_persist(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        predecessor_hypothesis_ref: str,
        predecessor_hypothesis_hash: str,
        grounding_event_ref: str,
        grounding_event_hash: str,
    ):
        store, predecessor, event, successor = self._derive(
            organization_id=organization_id,
            workspace_id=workspace_id,
            predecessor_hypothesis_ref=predecessor_hypothesis_ref,
            predecessor_hypothesis_hash=predecessor_hypothesis_hash,
            grounding_event_ref=grounding_event_ref,
            grounding_event_hash=grounding_event_hash,
        )
        successor = store.put_hypothesis(successor, judgment_request=store.get_judgment_request(
            hypothesis_ref=predecessor.hypothesis_id, hypothesis_hash=predecessor.hypothesis_hash))
        lineage = create_runtime_observation_correction_lineage_v1(
            predecessor=predecessor,
            grounding_event=event,
            grounding_event_hash=grounding_event_hash,
            successor=successor,
        )
        RuntimeObservationCorrectionStore(
            self._runtime_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )._put(
            lineage,
            capability=_RUNTIME_OBSERVATION_CORRECTION_WRITE_CAPABILITY,
        )
        return successor

    def get_verified_successor(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        successor_hypothesis_ref: str,
        successor_hypothesis_hash: str,
    ):
        lineage = RuntimeObservationCorrectionStore(
            self._runtime_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )._find_by_successor(successor_hypothesis_ref, successor_hypothesis_hash)
        assert lineage is not None
        store, predecessor, event, expected = self._derive(
            organization_id=organization_id,
            workspace_id=workspace_id,
            predecessor_hypothesis_ref=lineage.predecessor_hypothesis_ref,
            predecessor_hypothesis_hash=lineage.predecessor_hypothesis_hash,
            grounding_event_ref=lineage.grounding_event_ref,
            grounding_event_hash=lineage.grounding_event_hash,
        )
        actual = store.get_hypothesis(
            hypothesis_ref=successor_hypothesis_ref,
            hypothesis_hash=successor_hypothesis_hash,
        )
        if actual != expected:
            raise ValueError("runtime observation successor differs from governed correction")
        expected_lineage = create_runtime_observation_correction_lineage_v1(
            predecessor=predecessor,
            grounding_event=event,
            grounding_event_hash=lineage.grounding_event_hash,
            successor=actual,
        )
        if expected_lineage != lineage:
            raise ValueError("runtime observation correction lineage replay mismatch")
        return actual

    def _derive(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        predecessor_hypothesis_ref: str,
        predecessor_hypothesis_hash: str,
        grounding_event_ref: str,
        grounding_event_hash: str,
    ):
        store = RuntimeSemanticHypothesisArtifactStore(
            self._runtime_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        predecessor = store.get_hypothesis(
            hypothesis_ref=predecessor_hypothesis_ref,
            hypothesis_hash=predecessor_hypothesis_hash,
        )
        snapshot = store.get_retrieval_snapshot(
            snapshot_ref=predecessor.retrieval_snapshot_ref,
            snapshot_hash=predecessor.retrieval_snapshot_hash,
        )
        stored_event = self._grounding_store.get_learning_event(grounding_event_ref)
        if stored_event is None:
            raise ValueError("committed Human Grounding event not found")
        actual_event_hash = stored_event.pop("event_hash")
        if actual_event_hash != grounding_event_hash:
            raise ValueError("committed Human Grounding event hash mismatch")
        event = LearningEventEnvelope.model_validate(stored_event)
        self._validate_event(event=event, predecessor=predecessor, snapshot=snapshot)
        outcome = dict(event.downstream_outcome or {})
        indexes = tuple(outcome["selected_candidate_frame_indexes"])
        candidate_frames = [predecessor.candidate_frames[index] for index in indexes]
        successor = create_observed_constraint_hypothesis(
            organization_id=predecessor.organization_id,
            workspace_id=predecessor.workspace_id,
            run_id=predecessor.run_id,
            source_binding=predecessor.source_binding,
            evidence_refs=predecessor.evidence_refs,
            source_anchor_refs=predecessor.source_anchor_refs,
            candidate_frames=candidate_frames,
            retrieved_pattern_refs=predecessor.retrieved_pattern_refs,
            retrieved_pattern_hashes=predecessor.retrieved_pattern_hashes,
            assumptions=predecessor.assumptions,
            supporting_evidence_refs=outcome["supporting_evidence_refs"],
            counter_evidence_refs=outcome["counter_evidence_refs"],
            uncertainty=outcome["uncertainty"],
            provider_ref=predecessor.provider_ref,
            model_ref=predecessor.model_ref,
            prompt_asset_ref=predecessor.prompt_asset_ref,
            prompt_asset_hash=predecessor.prompt_asset_hash,
            schema_ref=predecessor.schema_ref,
            retrieval_snapshot_ref=predecessor.retrieval_snapshot_ref,
            retrieval_snapshot_hash=predecessor.retrieval_snapshot_hash,
            runtime_context=_runtime_context(snapshot),
            lifecycle_state="proposed",
            predecessor_hypothesis_ref=predecessor.hypothesis_id,
            judgment_request=store.get_judgment_request(
                hypothesis_ref=predecessor.hypothesis_id, hypothesis_hash=predecessor.hypothesis_hash),
        )
        if successor.hypothesis_hash == predecessor.hypothesis_hash:
            raise ValueError("runtime observation correction must change the hypothesis")
        return store, predecessor, event, successor

    @staticmethod
    def _validate_event(*, event: LearningEventEnvelope, predecessor: Any, snapshot: Any) -> None:
        if event.event_type is not LearningEventType.HUMAN_EDIT or (
            event.human_label is None
            or event.human_label.decision is not HumanDecision.EDIT
        ):
            raise ValueError("runtime observation correction requires a human edit event")
        if not event.actor_ref:
            raise ValueError("runtime observation correction requires an actor")
        if event.case_id != predecessor.hypothesis_id:
            raise ValueError("grounding event case does not bind predecessor hypothesis")
        if event.source_document_hash != predecessor.source_binding.source_hash:
            raise ValueError("grounding event source does not bind predecessor hypothesis")
        if tuple(event.source_anchor_refs) != predecessor.source_anchor_refs:
            raise ValueError("grounding event anchors do not bind predecessor hypothesis")
        if event.graph_version != predecessor.accepted_graph_release_ref:
            raise ValueError("grounding event graph does not bind predecessor hypothesis")
        if event.rule_version not in snapshot.ordered_governed_rule_version_refs:
            raise ValueError("grounding event rule is outside predecessor retrieval authority")
        outcome = event.downstream_outcome
        if not isinstance(outcome, Mapping) or set(outcome) != _OUTCOME_FIELDS or (
            outcome.get("schema_version") != "runtime_observation_correction.v1"
        ):
            raise ValueError("human edit event requires an exact runtime observation correction")
        indexes = outcome.get("selected_candidate_frame_indexes")
        if not isinstance(indexes, list) or not indexes or any(
            type(index) is not int or index < 0 or index >= len(predecessor.candidate_frames)
            for index in indexes
        ):
            raise ValueError("selected candidate frame indexes are invalid")
        if indexes != sorted(set(indexes)):
            raise ValueError("selected candidate frame indexes must be unique and ordered")
        evidence_ids = {item.evidence_ref for item in predecessor.evidence_refs}
        supporting = outcome.get("supporting_evidence_refs")
        counter = outcome.get("counter_evidence_refs")
        if not isinstance(supporting, list) or not isinstance(counter, list):
            raise ValueError("corrected evidence refs must be arrays")
        if len(supporting) != len(set(supporting)) or len(counter) != len(set(counter)):
            raise ValueError("corrected evidence refs contain duplicate identities")
        if not set(supporting + counter).issubset(evidence_ids):
            raise ValueError("corrected evidence refs are outside predecessor evidence")
        if set(supporting) & set(counter):
            raise ValueError("supporting and counter evidence refs must be disjoint")
        if not isinstance(outcome.get("uncertainty"), Mapping):
            raise ValueError("corrected uncertainty must be an object")


__all__ = ["RuntimeObservationCorrectionService"]
