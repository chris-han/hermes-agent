"""Finalize a neutral source-structure correction as an accepted child release."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contracts.corpus_graph_schema import CorpusGraphSchemaCandidate
from contracts.evidence_location import EvidenceRef
from contracts.schema_learning_feedback_v2 import SchemaLearningFeedbackV2
from contracts.semantic_release_retest import SemanticReleaseRetestV1
from contracts.semantic_impact_attribution import (
    create_semantic_impact_attribution_v1,
)
from contracts.source_structure_correction_continuation import (
    SourceStructureCorrectionContinuationV1,
)
from contracts.source_structure_release_finalization import (
    SourceStructureReleaseFinalizationV1,
    SourceStructureReleaseReservationV1,
)
from eos.artifact_hashing import content_hash
from services.accepted_semantic_graph_materialization_service import (
    AcceptedSemanticGraphMaterializationService,
    ResolveMaterializationInputsCommand,
    ResolveReviewedPatternInputsCommand,
    _SOURCE_STRUCTURE_MATERIALIZATION_AUTHORITY,
)
from services.graph_schema_discovery_service import GraphSchemaDiscoveryService
from services.schema_learning_service import (
    SchemaLearningService,
    _SOURCE_STRUCTURE_RETEST_AUTHORITY,
)
from services.semantic_impact_attribution_service import SemanticImpactAttributionService
from storage.atomic_json_artifact import publish_immutable_json
from storage.schema_learning_feedback_store import SchemaLearningFeedbackStore
from storage.source_structure_release_finalization_store import (
    SourceStructureReleaseFinalizationStore,
    _SOURCE_STRUCTURE_RELEASE_AUTHORITY,
)
from storage.source_structure_semantic_validation_store import (
    SourceStructureSemanticValidationStore,
)


_SOURCE_STRUCTURE_FINALIZATION_CAPABILITY = object()


class SourceStructureCorrectionService:
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
        self.finalizations = SourceStructureReleaseFinalizationStore(
            self.workspace_root, workspace_id=workspace_id
        )

    def _comparison(self, continuation: SourceStructureCorrectionContinuationV1) -> dict[str, Any]:
        comparison = SchemaLearningService(
            self.workspace_root,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            discovery_service=self.discovery,
        ).comparable_retest(
            organization_id=self.organization_id,
            baseline_schema_decision_ref=continuation.predecessor_schema_decision_ref,
            successor_schema_decision_ref=continuation.successor_schema_decision_ref,
            requested_by=continuation.created_by,
            source_structure_validation_ref=continuation.semantic_validation_ref,
            source_structure_validation_hash=continuation.semantic_validation_hash,
            _source_structure_authority=_SOURCE_STRUCTURE_RETEST_AUTHORITY,
        )
        if (
            comparison["comparison_id"] != continuation.comparison_ref
            or comparison["comparison_hash"] != continuation.comparison_hash
            or comparison["baseline_extract_ref"] != continuation.baseline_extract_ref
            or comparison["baseline_extract_hash"] != continuation.baseline_extract_hash
            or comparison["successor_extract_ref"] != continuation.successor_extract_ref
            or comparison["successor_extract_hash"] != continuation.successor_extract_hash
        ):
            raise ValueError("source-structure continuation comparison mismatch")
        return comparison

    def finalize(
        self,
        continuation: SourceStructureCorrectionContinuationV1,
        *,
        predecessor_release_ref: str,
        predecessor_release_hash: str,
        candidate_graph_id: str,
        human_label_event_ids: tuple[str, ...],
        requesting_actor_ref: str,
        _authority: object | None = None,
    ) -> dict[str, Any]:
        if _authority is not _SOURCE_STRUCTURE_FINALIZATION_CAPABILITY:
            raise PermissionError("source-structure finalization is capability-sealed")
        if continuation.organization_id != self.organization_id or continuation.workspace_id != self.workspace_id:
            raise ValueError("source-structure continuation tenant mismatch")
        if continuation.created_by != requesting_actor_ref:
            raise PermissionError("source-structure continuation belongs to another actor")
        comparison = self._comparison(continuation)
        semantic_validation = SourceStructureSemanticValidationStore(
            self.workspace_root, workspace_id=self.workspace_id
        ).get(
            continuation.semantic_validation_ref,
            validation_hash=continuation.semantic_validation_hash,
        )
        predecessor_candidate = CorpusGraphSchemaCandidate.from_dict(
            self.discovery.candidate_store.get_by_id(continuation.predecessor_candidate_ref)
        )
        successor_candidate = CorpusGraphSchemaCandidate.from_dict(
            self.discovery.candidate_store.get_by_id(continuation.successor_candidate_ref)
        )
        predecessor_decision = self.discovery.get_schema_decision_contract(
            continuation.predecessor_schema_decision_ref
        )
        successor_decision = self.discovery.get_schema_decision_contract(
            continuation.successor_schema_decision_ref
        )
        successor_intent = self.discovery.graph_intent_store.get(
            continuation.successor_graph_intent_ref,
            graph_intent_hash=continuation.successor_graph_intent_hash,
        )
        predecessor = self.materializer.resolve_accepted_graph_release_v2(
            predecessor_release_ref, release_hash=predecessor_release_hash
        )
        expected_predecessor = {
            "graph_intent_ref": continuation.predecessor_graph_intent_ref,
            "graph_intent_hash": continuation.predecessor_graph_intent_hash,
            "schema_candidate_ref": continuation.predecessor_candidate_ref,
            "schema_candidate_hash": continuation.predecessor_candidate_hash,
            "schema_decision_ref": continuation.predecessor_schema_decision_ref,
            "schema_decision_hash": continuation.predecessor_schema_decision_hash,
        }
        if any(predecessor["lineage"].get(key) != value for key, value in expected_predecessor.items()):
            raise ValueError("predecessor release does not bind source-structure continuation")
        reservation = self.finalizations.reserve(
            SourceStructureReleaseReservationV1.create(
                {
                    "schema_version": "source_structure_release_reservation.v1",
                    "organization_id": self.organization_id,
                    "workspace_id": self.workspace_id,
                    "continuation_ref": continuation.continuation_ref,
                    "continuation_hash": continuation.continuation_hash,
                    "successor_graph_intent_ref": successor_intent.graph_intent_id,
                    "successor_graph_intent_hash": successor_intent.graph_intent_hash,
                    "successor_candidate_ref": successor_candidate.candidate_id,
                    "successor_candidate_hash": successor_candidate.candidate_hash,
                    "successor_schema_decision_ref": successor_decision.schema_decision_id,
                    "successor_schema_decision_hash": successor_decision.schema_decision_hash,
                    "successor_extract_ref": continuation.successor_extract_ref,
                    "successor_extract_hash": continuation.successor_extract_hash,
                    "predecessor_release_ref": predecessor_release_ref,
                    "predecessor_release_hash": predecessor_release_hash,
                    "candidate_graph_id": candidate_graph_id,
                    "human_label_event_ids": human_label_event_ids,
                }
            ),
            _authority=_SOURCE_STRUCTURE_RELEASE_AUTHORITY,
        )
        command = ResolveMaterializationInputsCommand(
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            source_identity_ref=successor_intent.source_binding.source_identity_ref,
            lineage_id=successor_intent.source_binding.lineage_id,
            graph_intent_ref=successor_intent.graph_intent_id,
            graph_intent_hash=successor_intent.graph_intent_hash,
            schema_candidate_ref=successor_candidate.candidate_id,
            schema_candidate_hash=successor_candidate.candidate_hash,
            schema_decision_ref=successor_decision.schema_decision_id,
            schema_decision_hash=successor_decision.schema_decision_hash,
            extract_result_ref=continuation.successor_extract_ref,
            extract_result_hash=continuation.successor_extract_hash,
            predecessor_release_ref=predecessor_release_ref,
            predecessor_release_hash=predecessor_release_hash,
            source_structure_validation_ref=continuation.semantic_validation_ref,
            source_structure_validation_hash=continuation.semantic_validation_hash,
            _source_structure_authority=_SOURCE_STRUCTURE_MATERIALIZATION_AUTHORITY,
        )
        release_bundle = self.materializer.materialize_accepted_graph_release_v2(
            ResolveReviewedPatternInputsCommand(
                materialization=command,
                candidate_graph_id=candidate_graph_id,
                human_label_event_ids=human_label_event_ids,
            )
        )
        successor_release = release_bundle["release"]
        predecessor_release = predecessor["release"]
        release_retest = SemanticReleaseRetestV1.create(
            {
                "schema_version": "semantic_release_retest.v1",
                "organization_id": self.organization_id,
                "workspace_id": self.workspace_id,
                "baseline_release_ref": predecessor_release_ref,
                "baseline_release_hash": predecessor_release_hash,
                "successor_release_ref": successor_release["graph_version"],
                "successor_release_hash": successor_release["release_hash"],
                "schema_feedback_retest_ref": continuation.comparison_ref,
                "schema_feedback_retest_hash": continuation.comparison_hash,
                "pattern_membership": {"before": predecessor_release["pattern_membership"], "after": successor_release["pattern_membership"]},
                "relation_membership": {"before": predecessor_release["relation_membership"], "after": successor_release["relation_membership"]},
            }
        )
        release_retest = SemanticReleaseRetestV1.from_dict(publish_immutable_json(
            self.workspace_root / "workspaces" / self.workspace_id / "artifacts/schema-discovery/release-retests" / f"{release_retest.evaluation_id.replace(':', '-')}.json",
            release_retest.model_dump(mode="json"), conflict_error="SEMANTIC_RELEASE_RETEST_ARTIFACT_CONFLICT",
        ))
        evidence = tuple(
            EvidenceRef.model_validate(item)
            for relation in successor_candidate.proposed_relation_types
            for item in relation["relation_type_candidate"]["evidence_refs"]
        )
        feedback = SchemaLearningFeedbackV2.create(
            {
                "source_binding": predecessor_decision.source_binding,
                "schema_decision_ref": predecessor_decision.schema_decision_id,
                "candidate_ref": predecessor_candidate.candidate_id,
                "extraction_run_ref": continuation.successor_extract_ref,
                "grounding_ref": reservation.human_label_event_ids[0],
                "feedback_scope": "ontology_or_higher_layer",
                "produced_by": "user_chat_clarification",
                "operating_mode": "assisted",
                "policy_ref": None,
                "rationale_ref": continuation.semantic_validation_ref,
                "decision_evidence_refs": [item.evidence_ref for item in evidence],
                "fallback_reason": None,
                "operation_ref": None,
                "before_state_ref": continuation.predecessor_graph_intent_ref,
                "after_state_ref": continuation.successor_graph_intent_ref,
                "attribution": {"graph_intent": False, "source_structure": True, "ontology_induction": False, "synthesis": False, "extraction_plan": False, "execution": False},
                "findings": [{"finding_kind": "neutral_source_structure_correction", "validation_ref": continuation.semantic_validation_ref, "validation_hash": continuation.semantic_validation_hash}],
                "proposed_candidate_delta_ref": None,
                "created_at": continuation.created_at,
                "created_by": continuation.created_by,
            }
        )
        feedback_payload = self.feedback.put(feedback)
        feedback_hash = "sha256:" + content_hash(feedback_payload)
        attribution = SemanticImpactAttributionService(
            self.workspace_root, workspace_id=self.workspace_id
        ).record(
            feedback_ref=feedback.feedback_id,
            feedback_hash=feedback_hash,
            semantic_impact_layer="source_structure",
            target_artifact_ref=successor_intent.source_binding.source_identity_ref,
            target_artifact_hash=semantic_validation.successor_source_identity_hash,
            decision_evidence_refs=evidence,
            actor_ref=continuation.created_by,
            predecessor_refs=(continuation.predecessor_graph_intent_ref, continuation.predecessor_candidate_ref, continuation.predecessor_schema_decision_ref, predecessor_release_ref),
            successor_refs=(successor_intent.source_binding.source_identity_ref, continuation.semantic_validation_ref, continuation.successor_graph_intent_ref, continuation.successor_candidate_ref, continuation.successor_schema_decision_ref, successor_release["graph_version"], release_retest.evaluation_id),
        )
        finalization = self.finalizations.complete(
            SourceStructureReleaseFinalizationV1.create(
                {
                    "schema_version": "source_structure_release_finalization.v1",
                    "organization_id": self.organization_id,
                    "workspace_id": self.workspace_id,
                    "continuation_ref": continuation.continuation_ref,
                    "continuation_hash": continuation.continuation_hash,
                    "reservation_ref": reservation.reservation_ref,
                    "reservation_hash": reservation.reservation_hash,
                    "feedback_ref": feedback.feedback_id,
                    "feedback_hash": feedback_hash,
                    "successor_release_ref": successor_release["graph_version"],
                    "successor_release_hash": successor_release["release_hash"],
                    "semantic_release_retest_ref": release_retest.evaluation_id,
                    "semantic_release_retest_hash": release_retest.evaluation_hash,
                    "attribution_ref": attribution.attribution_id,
                    "attribution_hash": attribution.attribution_hash,
                }
            ),
            _authority=_SOURCE_STRUCTURE_RELEASE_AUTHORITY,
        )
        return {"outcome": "accepted", "feedback": feedback_payload, "accepted_graph_release": release_bundle, "release_comparison": release_retest.model_dump(mode="json"), "semantic_impact_attribution": attribution.model_dump(mode="json"), "release_finalization": finalization.model_dump(mode="json")}

    def replay(
        self,
        continuation: SourceStructureCorrectionContinuationV1,
        *,
        _authority: object | None = None,
    ) -> SourceStructureReleaseFinalizationV1:
        if _authority is not _SOURCE_STRUCTURE_FINALIZATION_CAPABILITY:
            raise PermissionError("source-structure finalization replay is capability-sealed")
        if (
            continuation.organization_id != self.organization_id
            or continuation.workspace_id != self.workspace_id
        ):
            raise ValueError("source-structure continuation tenant mismatch")
        self._comparison(continuation)
        reservation = self.finalizations.get_reservation(
            continuation.continuation_ref,
            continuation_hash=continuation.continuation_hash,
        )
        finalization = self.finalizations.get_finalization(
            continuation.continuation_ref,
            continuation_hash=continuation.continuation_hash,
        )
        fixed = {
            "successor_graph_intent_ref": continuation.successor_graph_intent_ref,
            "successor_graph_intent_hash": continuation.successor_graph_intent_hash,
            "successor_candidate_ref": continuation.successor_candidate_ref,
            "successor_candidate_hash": continuation.successor_candidate_hash,
            "successor_schema_decision_ref": continuation.successor_schema_decision_ref,
            "successor_schema_decision_hash": continuation.successor_schema_decision_hash,
            "successor_extract_ref": continuation.successor_extract_ref,
            "successor_extract_hash": continuation.successor_extract_hash,
        }
        if any(getattr(reservation, key) != value for key, value in fixed.items()):
            raise ValueError("source-structure reservation continuation mismatch")
        feedback_payload = self.feedback.get(finalization.feedback_ref)
        if "sha256:" + content_hash(feedback_payload) != finalization.feedback_hash:
            raise ValueError("source-structure finalization feedback mismatch")
        feedback = SchemaLearningFeedbackV2.from_dict(feedback_payload)
        predecessor_candidate = CorpusGraphSchemaCandidate.from_dict(
            self.discovery.candidate_store.get_by_id(
                continuation.predecessor_candidate_ref
            )
        )
        successor_candidate = CorpusGraphSchemaCandidate.from_dict(
            self.discovery.candidate_store.get_by_id(
                continuation.successor_candidate_ref
            )
        )
        predecessor_decision = self.discovery.get_schema_decision_contract(
            continuation.predecessor_schema_decision_ref
        )
        evidence = tuple(
            EvidenceRef.model_validate(item)
            for relation in successor_candidate.proposed_relation_types
            for item in relation["relation_type_candidate"]["evidence_refs"]
        )
        expected_feedback = SchemaLearningFeedbackV2.create(
            {
                "source_binding": predecessor_decision.source_binding,
                "schema_decision_ref": predecessor_decision.schema_decision_id,
                "candidate_ref": predecessor_candidate.candidate_id,
                "extraction_run_ref": continuation.successor_extract_ref,
                "grounding_ref": reservation.human_label_event_ids[0],
                "feedback_scope": "ontology_or_higher_layer",
                "produced_by": "user_chat_clarification",
                "operating_mode": "assisted",
                "policy_ref": None,
                "rationale_ref": continuation.semantic_validation_ref,
                "decision_evidence_refs": [item.evidence_ref for item in evidence],
                "fallback_reason": None,
                "operation_ref": None,
                "before_state_ref": continuation.predecessor_graph_intent_ref,
                "after_state_ref": continuation.successor_graph_intent_ref,
                "attribution": {
                    "graph_intent": False,
                    "source_structure": True,
                    "ontology_induction": False,
                    "synthesis": False,
                    "extraction_plan": False,
                    "execution": False,
                },
                "findings": [
                    {
                        "finding_kind": "neutral_source_structure_correction",
                        "validation_ref": continuation.semantic_validation_ref,
                        "validation_hash": continuation.semantic_validation_hash,
                    }
                ],
                "proposed_candidate_delta_ref": None,
                "created_at": continuation.created_at,
                "created_by": continuation.created_by,
            }
        )
        if feedback != expected_feedback:
            raise ValueError("source-structure finalization feedback binding mismatch")
        successor = self.materializer.resolve_accepted_graph_release_v2(
            finalization.successor_release_ref,
            release_hash=finalization.successor_release_hash,
        )
        predecessor = self.materializer.resolve_accepted_graph_release_v2(
            reservation.predecessor_release_ref,
            release_hash=reservation.predecessor_release_hash,
        )
        release = successor["release"]
        lineage = successor["lineage"]
        if (
            release.get("parent_graph_version") != reservation.predecessor_release_ref
            or release.get("candidate_graph_id") != reservation.candidate_graph_id
            or tuple(release.get("human_label_event_ids") or ())
            != reservation.human_label_event_ids
            or any(
                lineage.get(key) != value
                for key, value in {
                    "graph_intent_ref": reservation.successor_graph_intent_ref,
                    "graph_intent_hash": reservation.successor_graph_intent_hash,
                    "schema_candidate_ref": reservation.successor_candidate_ref,
                    "schema_candidate_hash": reservation.successor_candidate_hash,
                    "schema_decision_ref": reservation.successor_schema_decision_ref,
                    "schema_decision_hash": reservation.successor_schema_decision_hash,
                    "schema_validation_extract_run_ref": reservation.successor_extract_ref,
                    "schema_validation_extract_result_hash": reservation.successor_extract_hash,
                }.items()
            )
        ):
            raise ValueError("source-structure finalization child release mismatch")
        retest_path = (
            self.finalizations.root.parent
            / "release-retests"
            / f"{finalization.semantic_release_retest_ref.replace(':', '-')}.json"
        )
        retest = SemanticReleaseRetestV1.model_validate_json(
            retest_path.read_text(encoding="utf-8")
        )
        if (
            retest.evaluation_hash != finalization.semantic_release_retest_hash
            or retest.baseline_release_ref != reservation.predecessor_release_ref
            or retest.baseline_release_hash != reservation.predecessor_release_hash
            or retest.successor_release_ref != finalization.successor_release_ref
            or retest.successor_release_hash != finalization.successor_release_hash
            or retest.schema_feedback_retest_ref != continuation.comparison_ref
            or retest.schema_feedback_retest_hash != continuation.comparison_hash
            or retest.pattern_membership.model_dump(mode="json")
            != {
                "before": predecessor["release"]["pattern_membership"],
                "after": release["pattern_membership"],
            }
            or retest.relation_membership.model_dump(mode="json")
            != {
                "before": predecessor["release"]["relation_membership"],
                "after": release["relation_membership"],
            }
        ):
            raise ValueError("source-structure finalization release retest mismatch")
        validation = SourceStructureSemanticValidationStore(
            self.workspace_root, workspace_id=self.workspace_id
        ).get(
            continuation.semantic_validation_ref,
            validation_hash=continuation.semantic_validation_hash,
        )
        attribution = SemanticImpactAttributionService(
            self.workspace_root, workspace_id=self.workspace_id
        ).get(
            finalization.attribution_ref,
            attribution_hash=finalization.attribution_hash,
        )
        expected_attribution = create_semantic_impact_attribution_v1(
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            feedback_ref=feedback.feedback_id,
            feedback_hash=finalization.feedback_hash,
            semantic_impact_layer="source_structure",
            target_artifact_ref=validation.successor_source_identity_ref,
            target_artifact_hash=validation.successor_source_identity_hash,
            decision_evidence_refs=evidence,
            actor_ref=continuation.created_by,
            predecessor_refs=(
                continuation.predecessor_graph_intent_ref,
                continuation.predecessor_candidate_ref,
                continuation.predecessor_schema_decision_ref,
                reservation.predecessor_release_ref,
            ),
            successor_refs=(
                validation.successor_source_identity_ref,
                continuation.semantic_validation_ref,
                continuation.successor_graph_intent_ref,
                continuation.successor_candidate_ref,
                continuation.successor_schema_decision_ref,
                finalization.successor_release_ref,
                finalization.semantic_release_retest_ref,
            ),
        )
        if attribution != expected_attribution:
            raise ValueError("source-structure finalization attribution mismatch")
        return finalization


__all__ = ["SourceStructureCorrectionService", "_SOURCE_STRUCTURE_FINALIZATION_CAPABILITY"]
