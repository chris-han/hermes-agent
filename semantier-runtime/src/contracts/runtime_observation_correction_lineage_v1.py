"""Append-only authority lineage for a Human-Grounding-corrected hypothesis."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.knowledge_feature_pack import LearningEventEnvelope
from contracts.observed_constraint_hypothesis import ObservedConstraintHypothesisV1, ObservedConstraintHypothesisV3
from eos.artifact_hashing import content_hash


_ASCII = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
_SHA = r"^sha256:[0-9a-f]{64}$"


def _ascii(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.isascii() or _ASCII.fullmatch(value) is None:
        raise ValueError(f"{field} must be an ASCII-stable reference")
    return value


class RuntimeObservationCorrectionLineageV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["runtime_observation_correction_lineage.v1"] = Field(
        default="runtime_observation_correction_lineage.v1", alias="schemaVersion"
    )
    correction_id: str = Field(alias="correctionId")
    correction_hash: str = Field(alias="correctionHash", pattern=_SHA)
    organization_id: str = Field(alias="organizationId")
    workspace_id: str = Field(alias="workspaceId")
    predecessor_hypothesis_ref: str = Field(alias="predecessorHypothesisRef")
    predecessor_hypothesis_hash: str = Field(alias="predecessorHypothesisHash", pattern=_SHA)
    grounding_event_ref: str = Field(alias="groundingEventRef")
    grounding_event_hash: str = Field(alias="groundingEventHash", pattern=_SHA)
    successor_hypothesis_ref: str = Field(alias="successorHypothesisRef")
    successor_hypothesis_hash: str = Field(alias="successorHypothesisHash", pattern=_SHA)
    actor_ref: str = Field(alias="actorRef")
    source_identity_ref: str = Field(alias="sourceIdentityRef")
    source_hash: str = Field(alias="sourceHash", pattern=_SHA)
    retrieval_snapshot_ref: str = Field(alias="retrievalSnapshotRef")
    retrieval_snapshot_hash: str = Field(alias="retrievalSnapshotHash", pattern=_SHA)
    accepted_graph_release_ref: str = Field(alias="acceptedGraphReleaseRef")
    accepted_graph_release_hash: str = Field(alias="acceptedGraphReleaseHash", pattern=_SHA)

    @model_validator(mode="after")
    def validate_identity(self) -> "RuntimeObservationCorrectionLineageV1":
        for field in (
            "correction_id", "organization_id", "workspace_id",
            "predecessor_hypothesis_ref", "grounding_event_ref",
            "successor_hypothesis_ref", "actor_ref", "source_identity_ref",
            "retrieval_snapshot_ref", "accepted_graph_release_ref",
        ):
            _ascii(getattr(self, field), field)
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.correction_hash != expected:
            raise ValueError("runtime observation correction lineage hash mismatch")
        if self.correction_id != "rocl_" + expected[7:31]:
            raise ValueError("runtime observation correction lineage id mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("correctionId", None)
        payload.pop("correctionHash", None)
        return payload


def create_runtime_observation_correction_lineage_v1(
    *,
    predecessor: ObservedConstraintHypothesisV1,
    grounding_event: LearningEventEnvelope,
    grounding_event_hash: str,
    successor: ObservedConstraintHypothesisV1,
) -> RuntimeObservationCorrectionLineageV1:
    if (
        type(predecessor) not in (ObservedConstraintHypothesisV1, ObservedConstraintHypothesisV3)
        or type(successor) is not type(predecessor)
    ):
        raise TypeError(
            "lineage hypotheses must be exact ObservedConstraintHypothesisV1 or V3 values of the same version"
        )
    if type(grounding_event) is not LearningEventEnvelope:
        raise TypeError("grounding_event must be an exact LearningEventEnvelope")
    if grounding_event.event_hash() != grounding_event_hash:
        raise ValueError("grounding event hash mismatch")
    if successor.predecessor_hypothesis_ref != predecessor.hypothesis_id:
        raise ValueError("successor does not bind predecessor hypothesis")
    immutable_fields = (
        "organization_id", "workspace_id", "run_id", "source_binding",
        "evidence_refs", "source_anchor_refs", "retrieved_pattern_refs",
        "retrieved_pattern_hashes", "provider_ref", "model_ref",
        "prompt_asset_ref", "prompt_asset_hash", "schema_ref",
        "retrieval_snapshot_ref", "retrieval_snapshot_hash",
        "accepted_graph_release_ref", "accepted_graph_release_hash",
        "governed_rule_projection_ref", "governed_rule_projection_hash",
        "activation_set_snapshot_ref", "activation_set_snapshot_hash",
        "effective_context_resolver_hash",
    )
    if type(predecessor) is ObservedConstraintHypothesisV3:
        immutable_fields += ("decision_projection_ref", "decision_projection_hash", "judgment_request_hash", "observation_shape")
    if any(
        getattr(predecessor, field) != getattr(successor, field)
        for field in immutable_fields
    ):
        raise ValueError("runtime observation successor changes immutable authority pins")
    payload = {
        "schemaVersion": "runtime_observation_correction_lineage.v1",
        "organizationId": successor.organization_id,
        "workspaceId": successor.workspace_id,
        "predecessorHypothesisRef": predecessor.hypothesis_id,
        "predecessorHypothesisHash": predecessor.hypothesis_hash,
        "groundingEventRef": grounding_event.event_id,
        "groundingEventHash": grounding_event_hash,
        "successorHypothesisRef": successor.hypothesis_id,
        "successorHypothesisHash": successor.hypothesis_hash,
        "actorRef": grounding_event.actor_ref,
        "sourceIdentityRef": successor.source_binding.source_identity_ref,
        "sourceHash": successor.source_binding.source_hash,
        "retrievalSnapshotRef": successor.retrieval_snapshot_ref,
        "retrievalSnapshotHash": successor.retrieval_snapshot_hash,
        "acceptedGraphReleaseRef": successor.accepted_graph_release_ref,
        "acceptedGraphReleaseHash": successor.accepted_graph_release_hash,
    }
    digest = "sha256:" + content_hash(payload)
    return RuntimeObservationCorrectionLineageV1.model_validate(
        {**payload, "correctionId": "rocl_" + digest[7:31], "correctionHash": digest}
    )


__all__ = [
    "RuntimeObservationCorrectionLineageV1",
    "create_runtime_observation_correction_lineage_v1",
]
