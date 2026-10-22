"""Governed writer/replay service for semantic-impact attribution."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from contracts.evidence_location import EvidenceRef
from contracts.schema_learning_feedback_v2 import SchemaLearningFeedbackV2
from contracts.semantic_impact_attribution import (
    SemanticImpactAttributionV1,
    create_semantic_impact_attribution_v1,
    legacy_cause_for_semantic_impact,
)
from eos.artifact_hashing import content_hash
from storage.schema_learning_feedback_store import SchemaLearningFeedbackStore
from storage.semantic_impact_attribution_store import SemanticImpactAttributionStore
from storage.semantic_impact_attribution_store import _ATTRIBUTION_PUBLICATION_AUTHORITY


class SemanticImpactAttributionService:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        workspace_id: str,
    ) -> None:
        self.workspace_id = workspace_id
        self.store = SemanticImpactAttributionStore(
            workspace_root, workspace_id=workspace_id
        )
        self.feedback_store = SchemaLearningFeedbackStore(
            workspace_root, workspace_id=workspace_id
        )

    def record(
        self,
        *,
        feedback_ref: str,
        feedback_hash: str,
        semantic_impact_layer: str,
        target_artifact_ref: str,
        target_artifact_hash: str,
        decision_evidence_refs: Sequence[EvidenceRef | Mapping],
        actor_ref: str,
        predecessor_refs: Sequence[str],
        successor_refs: Sequence[str],
    ) -> SemanticImpactAttributionV1:
        feedback = SchemaLearningFeedbackV2.from_dict(
            self.feedback_store.get(feedback_ref)
        )
        actual_feedback_hash = "sha256:" + content_hash(feedback.as_dict())
        if feedback_hash != actual_feedback_hash:
            raise ValueError("schema learning feedback hash mismatch")
        organization_id = str(feedback.source_binding.get("organization_id") or "")
        feedback_workspace_id = str(feedback.source_binding.get("workspace_id") or "")
        if not organization_id or feedback_workspace_id != self.workspace_id:
            raise ValueError(
                "schema learning feedback lacks governed organization/workspace binding"
            )
        mapped_cause = legacy_cause_for_semantic_impact(semantic_impact_layer)
        if feedback.attribution[mapped_cause] is not True:
            raise ValueError(
                "schema learning feedback does not select mapped legacy cause"
            )
        attribution = create_semantic_impact_attribution_v1(
            organization_id=organization_id,
            workspace_id=self.workspace_id,
            feedback_ref=feedback.feedback_id,
            feedback_hash=actual_feedback_hash,
            semantic_impact_layer=semantic_impact_layer,
            target_artifact_ref=target_artifact_ref,
            target_artifact_hash=target_artifact_hash,
            decision_evidence_refs=decision_evidence_refs,
            actor_ref=actor_ref,
            predecessor_refs=predecessor_refs,
            successor_refs=successor_refs,
        )
        return self.store._publish(
            attribution, _authority=_ATTRIBUTION_PUBLICATION_AUTHORITY
        )

    def get(
        self, attribution_id: str, *, attribution_hash: str | None = None
    ) -> SemanticImpactAttributionV1:
        attribution = self.store.get(
            attribution_id, attribution_hash=attribution_hash
        )
        if attribution.workspace_id != self.workspace_id:
            raise ValueError("semantic impact attribution tenant scope mismatch")
        feedback = SchemaLearningFeedbackV2.from_dict(
            self.feedback_store.get(attribution.feedback_ref)
        )
        actual_feedback_hash = "sha256:" + content_hash(feedback.as_dict())
        if actual_feedback_hash != attribution.feedback_hash:
            raise ValueError("semantic impact attribution feedback binding mismatch")
        if (
            attribution.organization_id
            != str(feedback.source_binding.get("organization_id") or "")
            or attribution.workspace_id
            != str(feedback.source_binding.get("workspace_id") or "")
        ):
            raise ValueError("semantic impact attribution tenant binding mismatch")
        mapped_cause = legacy_cause_for_semantic_impact(
            attribution.semantic_impact_layer
        )
        if feedback.attribution[mapped_cause] is not True:
            raise ValueError(
                "semantic impact attribution feedback cause binding mismatch"
            )
        return attribution


__all__ = ["SemanticImpactAttributionService"]
