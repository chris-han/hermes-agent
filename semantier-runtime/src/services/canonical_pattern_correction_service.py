"""Finalize and replay reviewed canonical-pattern-only release successors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contracts.canonical_pattern_correction import (
    CanonicalPatternCorrectionFinalizationV1,
    CanonicalPatternCorrectionReservationV1,
)
from contracts.knowledge_feature_pack import LearningEventEnvelope
from contracts.schema_learning_feedback_v2 import SchemaLearningFeedbackV2
from contracts.semantic_release_retest import SemanticReleaseRetestV1
from eos.artifact_hashing import content_hash
from services.accepted_semantic_graph_materialization_service import (
    AcceptedSemanticGraphMaterializationService,
    ResolveMaterializationInputsCommand,
    ResolveReviewedPatternInputsCommand,
)
from services.graph_schema_discovery_service import GraphSchemaDiscoveryService
from services.semantic_impact_attribution_service import (
    SemanticImpactAttributionService,
)
from storage.atomic_json_artifact import publish_immutable_json
from storage.canonical_pattern_correction_store import (
    CanonicalPatternCorrectionStore,
    _CANONICAL_PATTERN_CORRECTION_AUTHORITY,
)
from storage.schema_learning_feedback_store import SchemaLearningFeedbackStore


_CANONICAL_PATTERN_FINALIZATION_CAPABILITY = object()


class CanonicalPatternCorrectionService:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
        discovery_service: GraphSchemaDiscoveryService,
        materializer: AcceptedSemanticGraphMaterializationService,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.discovery = discovery_service
        self.materializer = materializer
        self.feedback = SchemaLearningFeedbackStore(
            self.workspace_root, workspace_id=workspace_id
        )
        self.store = CanonicalPatternCorrectionStore(
            self.workspace_root, workspace_id=workspace_id
        )

    def _complete_reserved(
        self,
        *,
        reservation: CanonicalPatternCorrectionReservationV1,
        reviewed: ResolveReviewedPatternInputsCommand,
        predecessor: dict[str, Any],
    ) -> dict[str, Any]:
        feedback_payload = self.feedback.get(reservation.feedback_ref)
        if "sha256:" + content_hash(feedback_payload) != reservation.feedback_hash:
            raise ValueError("canonical-pattern reserved feedback hash mismatch")
        successor = self.materializer.materialize_accepted_graph_release_v2(reviewed)
        release = successor["release"]
        predecessor_pairs = {
            (item["pattern_ref"], item["pattern_hash"])
            for item in predecessor["release"]["pattern_membership"]
        }
        changed_membership = tuple(
            item for item in release["pattern_membership"]
            if (item["pattern_ref"], item["pattern_hash"]) not in predecessor_pairs
        )
        if len(changed_membership) != 1:
            raise ValueError("canonical-pattern completion requires one changed pattern")
        changed = tuple(
            item for item in successor["canonical_restriction_patterns"]
            if item["pattern_id"] == changed_membership[0]["pattern_ref"]
            and item["pattern_hash"] == changed_membership[0]["pattern_hash"]
        )
        if len(changed) != 1:
            raise ValueError("canonical-pattern changed artifact is missing")
        retest = SemanticReleaseRetestV1.create(
            {
                "schema_version": "semantic_release_retest.v1",
                "organization_id": self.organization_id,
                "workspace_id": self.workspace_id,
                "baseline_release_ref": reservation.predecessor_release_ref,
                "baseline_release_hash": reservation.predecessor_release_hash,
                "successor_release_ref": release["graph_version"],
                "successor_release_hash": release["release_hash"],
                "schema_feedback_retest_ref": reservation.feedback_ref,
                "schema_feedback_retest_hash": reservation.feedback_hash,
                "pattern_membership": {
                    "before": predecessor["release"]["pattern_membership"],
                    "after": release["pattern_membership"],
                },
                "relation_membership": {
                    "before": predecessor["release"]["relation_membership"],
                    "after": release["relation_membership"],
                },
            }
        )
        retest = SemanticReleaseRetestV1.from_dict(
            publish_immutable_json(
                self.store.root.parent / "release-retests"
                / f"{retest.evaluation_id.replace(':', '-')}.json",
                retest.model_dump(mode="json"),
                conflict_error="SEMANTIC_RELEASE_RETEST_ARTIFACT_CONFLICT",
            )
        )
        evidence = {
            evidence_ref["evidenceRef"]: evidence_ref
            for pattern in changed
            for evidence_ref in pattern["source_metadata"]["evidence_refs"]
        }
        attribution = SemanticImpactAttributionService(
            self.workspace_root, workspace_id=self.workspace_id
        ).record(
            feedback_ref=reservation.feedback_ref,
            feedback_hash=reservation.feedback_hash,
            semantic_impact_layer="canonical_pattern",
            target_artifact_ref=changed_membership[0]["pattern_ref"],
            target_artifact_hash=changed_membership[0]["pattern_hash"],
            decision_evidence_refs=tuple(evidence.values()),
            actor_ref=reservation.created_by,
            predecessor_refs=(
                *(
                    item["pattern_ref"]
                    for item in predecessor["release"]["pattern_membership"]
                ),
                reservation.predecessor_release_ref,
            ),
            successor_refs=(
                *(item["pattern_ref"] for item in release["pattern_membership"]),
                release["graph_version"],
                retest.evaluation_id,
            ),
        )
        finalization = self.store.complete(
            CanonicalPatternCorrectionFinalizationV1.create(
                {
                    "schema_version": "canonical_pattern_correction_finalization.v1",
                    "organization_id": self.organization_id,
                    "workspace_id": self.workspace_id,
                    "reservation_ref": reservation.reservation_ref,
                    "reservation_hash": reservation.reservation_hash,
                    "successor_release_ref": release["graph_version"],
                    "successor_release_hash": release["release_hash"],
                    "semantic_release_retest_ref": retest.evaluation_id,
                    "semantic_release_retest_hash": retest.evaluation_hash,
                    "attribution_ref": attribution.attribution_id,
                    "attribution_hash": attribution.attribution_hash,
                }
            ),
            _authority=_CANONICAL_PATTERN_CORRECTION_AUTHORITY,
        )
        return {
            "reservation": reservation.model_dump(mode="json"),
            "finalization": finalization.model_dump(mode="json"),
            "feedback": feedback_payload,
            "accepted_graph_release": successor,
            "release_comparison": retest.model_dump(mode="json"),
            "semantic_impact_attribution": attribution.model_dump(mode="json"),
        }

    def finalize(
        self,
        *,
        predecessor_release_ref: str,
        predecessor_release_hash: str,
        candidate_graph_id: str,
        human_label_event_id: str,
        requesting_actor_ref: str,
        _authority: object | None = None,
    ) -> dict[str, Any]:
        if _authority is not _CANONICAL_PATTERN_FINALIZATION_CAPABILITY:
            raise PermissionError("canonical-pattern correction is capability-sealed")
        predecessor = self.materializer.resolve_accepted_graph_release_v2(
            predecessor_release_ref, release_hash=predecessor_release_hash
        )
        lineage = predecessor["lineage"]
        existing = self.store.find_completed_review(
            predecessor_release_ref=predecessor_release_ref,
            predecessor_release_hash=predecessor_release_hash,
            candidate_graph_id=candidate_graph_id,
            human_label_event_ref=human_label_event_id,
        )
        if existing is not None:
            reservation, finalization = existing
            self.replay(
                reservation.reservation_ref,
                reservation_hash=reservation.reservation_hash,
                _authority=_CANONICAL_PATTERN_FINALIZATION_CAPABILITY,
            )
            successor = self.materializer.resolve_accepted_graph_release_v2(
                finalization.successor_release_ref,
                release_hash=finalization.successor_release_hash,
            )
            retest_path = (
                self.store.root.parent / "release-retests"
                / f"{finalization.semantic_release_retest_ref.replace(':', '-')}.json"
            )
            retest = SemanticReleaseRetestV1.model_validate_json(
                retest_path.read_text(encoding="utf-8")
            )
            attribution = SemanticImpactAttributionService(
                self.workspace_root, workspace_id=self.workspace_id
            ).get(
                finalization.attribution_ref,
                attribution_hash=finalization.attribution_hash,
            )
            return {
                "reservation": reservation.model_dump(mode="json"),
                "finalization": finalization.model_dump(mode="json"),
                "feedback": self.feedback.get(reservation.feedback_ref),
                "accepted_graph_release": successor,
                "release_comparison": retest.model_dump(mode="json"),
                "semantic_impact_attribution": attribution.model_dump(mode="json"),
            }
        intent = self.discovery.graph_intent_store.get(
            lineage["graph_intent_ref"],
            graph_intent_hash=lineage["graph_intent_hash"],
        )
        command = ResolveMaterializationInputsCommand(
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            source_identity_ref=intent.source_binding.source_identity_ref,
            lineage_id=intent.source_binding.lineage_id,
            graph_intent_ref=lineage["graph_intent_ref"],
            graph_intent_hash=lineage["graph_intent_hash"],
            schema_candidate_ref=lineage["schema_candidate_ref"],
            schema_candidate_hash=lineage["schema_candidate_hash"],
            schema_decision_ref=lineage["schema_decision_ref"],
            schema_decision_hash=lineage["schema_decision_hash"],
            extract_result_ref=lineage["schema_validation_extract_run_ref"],
            extract_result_hash=lineage["schema_validation_extract_result_hash"],
            predecessor_release_ref=predecessor_release_ref,
            predecessor_release_hash=predecessor_release_hash,
        )
        reviewed = ResolveReviewedPatternInputsCommand(
            materialization=command,
            candidate_graph_id=candidate_graph_id,
            human_label_event_ids=(human_label_event_id,),
            canonical_pattern_predecessor_membership=tuple(
                (item["pattern_ref"], item["pattern_hash"])
                for item in predecessor["release"]["pattern_membership"]
            ),
            canonical_pattern_unchanged_relation_membership=tuple(
                (
                    item["relation_instance_ref"],
                    item["relation_instance_hash"],
                    item["occurrence_grounding_ref"],
                    item["occurrence_grounding_hash"],
                )
                for item in predecessor["release"]["relation_membership"]
            ),
        )
        pending = self.store.find_reservation_for_review(
            predecessor_release_ref=predecessor_release_ref,
            predecessor_release_hash=predecessor_release_hash,
            candidate_graph_id=candidate_graph_id,
            human_label_event_ref=human_label_event_id,
        )
        if pending is not None:
            if pending.created_by != requesting_actor_ref:
                raise ValueError("canonical-pattern correction actor mismatch")
            return self._complete_reserved(
                reservation=pending,
                reviewed=reviewed,
                predecessor=predecessor,
            )
        resolved_review = self.materializer.resolve_reviewed_pattern_inputs(reviewed)
        selection = resolved_review.reviewed_selection
        event_payload = self.materializer._grounding_store.get_learning_event(  # noqa: SLF001
            human_label_event_id
        )
        if event_payload is None:
            raise ValueError("canonical-pattern Human Grounding event is missing")
        stored_event_hash = event_payload.get("event_hash")
        event = LearningEventEnvelope.model_validate(
            {key: value for key, value in event_payload.items() if key != "event_hash"}
        )
        if stored_event_hash != event.event_hash():
            raise ValueError("canonical-pattern Human Grounding event hash mismatch")
        if event.actor_ref != requesting_actor_ref:
            raise ValueError("canonical-pattern correction actor mismatch")
        patterns = self.materializer.derive_accepted_canonical_patterns(reviewed)
        predecessor_patterns = {
            (item["pattern_ref"], item["pattern_hash"])
            for item in predecessor["release"]["pattern_membership"]
        }
        successor_patterns = {(item.pattern_ref, item.pattern_hash) for item in patterns}
        changed = tuple(
            item for item in patterns
            if (item.pattern_ref, item.pattern_hash) not in predecessor_patterns
        )
        if successor_patterns == predecessor_patterns or len(changed) != 1:
            raise ValueError(
                "canonical-pattern correction requires exactly one changed pattern"
            )
        decision = self.discovery.get_schema_decision_contract(
            lineage["schema_decision_ref"]
        )
        created_at = event.occurred_at.isoformat().replace("+00:00", "Z")
        feedback = SchemaLearningFeedbackV2.create(
            {
                "source_binding": decision.source_binding,
                "schema_decision_ref": decision.schema_decision_id,
                "candidate_ref": lineage["schema_candidate_ref"],
                "extraction_run_ref": lineage["schema_validation_extract_run_ref"],
                "grounding_ref": event.event_id,
                "feedback_scope": "schema_understanding",
                "produced_by": "user_chat_clarification",
                "operating_mode": "assisted",
                "policy_ref": None,
                "rationale_ref": "rationale:canonical-pattern-human-review",
                "decision_evidence_refs": list(decision.source_evidence_refs),
                "fallback_reason": None,
                "operation_ref": None,
                "before_state_ref": predecessor_release_ref,
                "after_state_ref": candidate_graph_id,
                "attribution": {
                    "graph_intent": False,
                    "source_structure": False,
                    "ontology_induction": False,
                    "synthesis": True,
                    "extraction_plan": False,
                    "execution": False,
                },
                "findings": [{
                    "finding_kind": "canonical_pattern_correction",
                    "candidate_graph_ref": candidate_graph_id,
                    "candidate_graph_hash": selection["candidate_graph_hash"],
                    "predecessor_pattern_membership": sorted(
                        predecessor["release"]["pattern_membership"],
                        key=lambda item: item["pattern_ref"].encode(),
                    ),
                    "successor_pattern_refs": [item.pattern_ref for item in patterns],
                }],
                "proposed_candidate_delta_ref": None,
                "created_at": created_at,
                "created_by": requesting_actor_ref,
            }
        )
        feedback_payload = self.feedback.put(feedback)
        feedback_hash = "sha256:" + content_hash(feedback_payload)
        reservation = self.store.reserve(
            CanonicalPatternCorrectionReservationV1.create(
                {
                    "schema_version": "canonical_pattern_correction_reservation.v1",
                    "organization_id": self.organization_id,
                    "workspace_id": self.workspace_id,
                    "feedback_ref": feedback.feedback_id,
                    "feedback_hash": feedback_hash,
                    "graph_intent_ref": lineage["graph_intent_ref"],
                    "graph_intent_hash": lineage["graph_intent_hash"],
                    "schema_candidate_ref": lineage["schema_candidate_ref"],
                    "schema_candidate_hash": lineage["schema_candidate_hash"],
                    "schema_decision_ref": lineage["schema_decision_ref"],
                    "schema_decision_hash": lineage["schema_decision_hash"],
                    "extract_result_ref": lineage["schema_validation_extract_run_ref"],
                    "extract_result_hash": lineage["schema_validation_extract_result_hash"],
                    "predecessor_release_ref": predecessor_release_ref,
                    "predecessor_release_hash": predecessor_release_hash,
                    "candidate_graph_id": candidate_graph_id,
                    "candidate_graph_hash": selection["candidate_graph_hash"],
                    "human_label_event_ref": event.event_id,
                    "human_label_event_hash": event.event_hash(),
                    "created_at": created_at,
                    "created_by": requesting_actor_ref,
                }
            ),
            _authority=_CANONICAL_PATTERN_CORRECTION_AUTHORITY,
        )
        return self._complete_reserved(
            reservation=reservation,
            reviewed=reviewed,
            predecessor=predecessor,
        )

    def replay(
        self,
        reservation_ref: str,
        *,
        reservation_hash: str,
        _authority: object | None = None,
    ) -> CanonicalPatternCorrectionFinalizationV1:
        """Reconstruct every authority binding for one completed correction."""
        if _authority is not _CANONICAL_PATTERN_FINALIZATION_CAPABILITY:
            raise PermissionError("canonical-pattern correction replay is capability-sealed")
        if type(self.materializer) is not AcceptedSemanticGraphMaterializationService:
            raise ValueError("canonical-pattern replay requires the exact materializer")
        reservation, finalization = self.store.get_verified_pair(
            reservation_ref, reservation_hash=reservation_hash
        )
        if (
            reservation.organization_id != self.organization_id
            or reservation.workspace_id != self.workspace_id
            or finalization.organization_id != self.organization_id
            or finalization.workspace_id != self.workspace_id
        ):
            raise ValueError("canonical-pattern correction tenant mismatch")

        predecessor = self.materializer.resolve_accepted_graph_release_v2(
            reservation.predecessor_release_ref,
            release_hash=reservation.predecessor_release_hash,
        )
        expected_lineage = {
            "graph_intent_ref": reservation.graph_intent_ref,
            "graph_intent_hash": reservation.graph_intent_hash,
            "schema_candidate_ref": reservation.schema_candidate_ref,
            "schema_candidate_hash": reservation.schema_candidate_hash,
            "schema_decision_ref": reservation.schema_decision_ref,
            "schema_decision_hash": reservation.schema_decision_hash,
            "schema_validation_extract_run_ref": reservation.extract_result_ref,
            "schema_validation_extract_result_hash": reservation.extract_result_hash,
        }
        if any(
            predecessor["lineage"].get(field) != value
            for field, value in expected_lineage.items()
        ):
            raise ValueError("canonical-pattern predecessor lineage mismatch")
        self.discovery.graph_intent_store.get(
            reservation.graph_intent_ref,
            graph_intent_hash=reservation.graph_intent_hash,
        )
        graph = self.materializer._grounding_store.get_candidate_graph(  # noqa: SLF001
            reservation.candidate_graph_id
        )
        if graph.graph_hash() != reservation.candidate_graph_hash:
            raise ValueError("canonical-pattern candidate graph binding mismatch")
        event_payload = self.materializer._grounding_store.get_learning_event(  # noqa: SLF001
            reservation.human_label_event_ref
        )
        if event_payload is None:
            raise ValueError("canonical-pattern Human Grounding event is missing")
        stored_event_hash = event_payload.get("event_hash")
        event = LearningEventEnvelope.model_validate(
            {key: value for key, value in event_payload.items() if key != "event_hash"}
        )
        if (
            stored_event_hash != reservation.human_label_event_hash
            or event.event_hash() != reservation.human_label_event_hash
            or event.actor_ref != reservation.created_by
            or event.occurred_at.isoformat().replace("+00:00", "Z")
            != reservation.created_at
        ):
            raise ValueError("canonical-pattern Human Grounding binding mismatch")

        successor = self.materializer.resolve_accepted_graph_release_v2(
            finalization.successor_release_ref,
            release_hash=finalization.successor_release_hash,
        )
        release = successor["release"]
        predecessor_membership = predecessor["release"]["pattern_membership"]
        predecessor_patterns = {
            (item["pattern_ref"], item["pattern_hash"])
            for item in predecessor_membership
        }
        successor_membership = release["pattern_membership"]
        successor_patterns = {
            (item["pattern_ref"], item["pattern_hash"])
            for item in successor_membership
        }
        changed = tuple(
            item for item in successor_membership
            if (item["pattern_ref"], item["pattern_hash"]) not in predecessor_patterns
        )
        removed = predecessor_patterns - successor_patterns
        if len(changed) != 1 or len(removed) != 1:
            raise ValueError("canonical-pattern replay requires one exact pattern replacement")

        feedback_payload = self.feedback.get(reservation.feedback_ref)
        if "sha256:" + content_hash(feedback_payload) != reservation.feedback_hash:
            raise ValueError("canonical-pattern feedback hash mismatch")
        feedback = SchemaLearningFeedbackV2.from_dict(feedback_payload)
        expected_findings = [{
            "finding_kind": "canonical_pattern_correction",
            "candidate_graph_ref": reservation.candidate_graph_id,
            "candidate_graph_hash": reservation.candidate_graph_hash,
            "predecessor_pattern_membership": sorted(
                predecessor_membership,
                key=lambda item: item["pattern_ref"].encode(),
            ),
            "successor_pattern_refs": [item["pattern_ref"] for item in successor_membership],
        }]
        if (
            feedback.grounding_ref != reservation.human_label_event_ref
            or feedback.schema_decision_ref != reservation.schema_decision_ref
            or feedback.candidate_ref != reservation.schema_candidate_ref
            or feedback.extraction_run_ref != reservation.extract_result_ref
            or feedback.before_state_ref != reservation.predecessor_release_ref
            or feedback.after_state_ref != reservation.candidate_graph_id
            or feedback.created_by != reservation.created_by
            or feedback.created_at != reservation.created_at
            or feedback.feedback_scope != "schema_understanding"
            or feedback.produced_by != "user_chat_clarification"
            or feedback.operating_mode != "assisted"
            or feedback.attribution != {
                "graph_intent": False,
                "source_structure": False,
                "ontology_induction": False,
                "synthesis": True,
                "extraction_plan": False,
                "execution": False,
            }
            or feedback.findings != tuple(expected_findings)
        ):
            raise ValueError("canonical-pattern feedback authority mismatch")

        if (
            release.get("parent_graph_version") != reservation.predecessor_release_ref
            or release.get("candidate_graph_id") != reservation.candidate_graph_id
            or tuple(release.get("human_label_event_ids") or ())
            != (reservation.human_label_event_ref,)
            or any(
                successor["lineage"].get(field) != value
                for field, value in {
                    **expected_lineage,
                    "candidate_graph_materialization_extract_run_ref": (
                        reservation.extract_result_ref
                    ),
                }.items()
            )
            or release["relation_membership"]
            != predecessor["release"]["relation_membership"]
        ):
            raise ValueError("canonical-pattern child release authority mismatch")

        retest_path = (
            self.store.root.parent / "release-retests"
            / f"{finalization.semantic_release_retest_ref.replace(':', '-')}.json"
        )
        retest = SemanticReleaseRetestV1.model_validate_json(
            retest_path.read_text(encoding="utf-8")
        )
        if (
            retest.evaluation_id != finalization.semantic_release_retest_ref
            or retest.evaluation_hash != finalization.semantic_release_retest_hash
            or retest.baseline_release_ref != reservation.predecessor_release_ref
            or retest.baseline_release_hash != reservation.predecessor_release_hash
            or retest.successor_release_ref != finalization.successor_release_ref
            or retest.successor_release_hash != finalization.successor_release_hash
            or retest.schema_feedback_retest_ref != reservation.feedback_ref
            or retest.schema_feedback_retest_hash != reservation.feedback_hash
            or retest.pattern_membership.model_dump(mode="json") != {
                "before": predecessor_membership,
                "after": release["pattern_membership"],
            }
            or retest.relation_membership.model_dump(mode="json") != {
                "before": predecessor["release"]["relation_membership"],
                "after": release["relation_membership"],
            }
        ):
            raise ValueError("canonical-pattern semantic retest mismatch")

        attribution = SemanticImpactAttributionService(
            self.workspace_root, workspace_id=self.workspace_id
        ).get(
            finalization.attribution_ref,
            attribution_hash=finalization.attribution_hash,
        )
        expected_predecessors = (
            *(item["pattern_ref"] for item in predecessor_membership),
            reservation.predecessor_release_ref,
        )
        expected_successors = (
            *(item["pattern_ref"] for item in release["pattern_membership"]),
            finalization.successor_release_ref,
            finalization.semantic_release_retest_ref,
        )
        if (
            attribution.feedback_ref != reservation.feedback_ref
            or attribution.feedback_hash != reservation.feedback_hash
            or attribution.semantic_impact_layer != "canonical_pattern"
            or attribution.mapped_legacy_cause != "synthesis"
            or attribution.target_artifact_ref != changed[0]["pattern_ref"]
            or attribution.target_artifact_hash != changed[0]["pattern_hash"]
            or attribution.actor_ref != reservation.created_by
            or attribution.predecessor_refs != expected_predecessors
            or attribution.successor_refs != expected_successors
        ):
            raise ValueError("canonical-pattern attribution lineage mismatch")
        return finalization


__all__ = [
    "CanonicalPatternCorrectionService",
    "_CANONICAL_PATTERN_FINALIZATION_CAPABILITY",
]
