"""Append-only human correction companion for a governed frame finding."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from contracts.evidence_location import EvidenceRef
from contracts.governed_frame_finding_v1 import GovernedFrameFindingV1
from contracts.knowledge_feature_pack import HumanDecision, LearningEventEnvelope, LearningEventType
from eos.artifact_hashing import content_hash


_ASCII = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
_SHA = r"^sha256:[0-9a-f]{64}$"


def _ascii(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.isascii() or _ASCII.fullmatch(value) is None:
        raise ValueError(f"{field} must be an ASCII-stable reference")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class GovernedFrameFindingCorrectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["governed_frame_finding_correction.v1"] = Field(
        default="governed_frame_finding_correction.v1", alias="schemaVersion"
    )
    correction_id: str = Field(alias="correctionId")
    correction_hash: str = Field(alias="correctionHash", pattern=_SHA)
    organization_id: str = Field(alias="organizationId")
    workspace_id: str = Field(alias="workspaceId")
    predecessor_finding_ref: str = Field(alias="predecessorFindingRef")
    predecessor_finding_hash: str = Field(alias="predecessorFindingHash", pattern=_SHA)
    grounding_event_ref: str = Field(alias="groundingEventRef")
    grounding_event_hash: str = Field(alias="groundingEventHash", pattern=_SHA)
    actor_ref: str = Field(alias="actorRef")
    corrected_decision_status: Literal["violation", "acceptable", "uncertain", "escalate"] = Field(
        alias="correctedDecisionStatus"
    )
    reason_code: str = Field(alias="reasonCode")
    rationale: str
    source_identity_ref: str = Field(alias="sourceIdentityRef")
    source_hash: str = Field(alias="sourceHash", pattern=_SHA)
    evidence_refs: tuple[Any, ...] = Field(alias="evidenceRefs", min_length=1)
    source_anchor_refs: tuple[str, ...] = Field(alias="sourceAnchorRefs")
    canonical_pattern_ref: str = Field(alias="canonicalPatternRef")
    canonical_pattern_hash: str = Field(alias="canonicalPatternHash", pattern=_SHA)
    accepted_graph_release_ref: str = Field(alias="acceptedGraphReleaseRef")
    accepted_graph_release_hash: str = Field(alias="acceptedGraphReleaseHash", pattern=_SHA)
    governed_rule_version_ref: str = Field(alias="governedRuleVersionRef")
    governed_rule_version_hash: str = Field(alias="governedRuleVersionHash", pattern=_SHA)
    activation_set_snapshot_ref: str = Field(alias="activationSetSnapshotRef")
    activation_set_snapshot_hash: str = Field(alias="activationSetSnapshotHash", pattern=_SHA)
    activation_resolver_policy_version: str = Field(alias="activationResolverPolicyVersion")
    effective_context_resolver_hash: str = Field(alias="effectiveContextResolverHash", pattern=_SHA)
    resolved_activation_set_hash: str = Field(alias="resolvedActivationSetHash", pattern=_SHA)

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def freeze_evidence(cls, value: Any) -> tuple[Any, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("evidenceRefs must be a nonempty array")
        return tuple(
            _freeze(EvidenceRef.model_validate(_thaw(item)).model_dump(mode="json", by_alias=True))
            for item in value
        )

    @field_validator("source_anchor_refs", mode="before")
    @classmethod
    def validate_anchors(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("sourceAnchorRefs must be an array")
        refs = tuple(_ascii(item, "sourceAnchorRefs[]") for item in value)
        if len(refs) != len(set(refs)):
            raise ValueError("sourceAnchorRefs contains duplicate identities")
        return refs

    @field_serializer("evidence_refs")
    def serialize_evidence(self, value: tuple[Any, ...]) -> list[Any]:
        return [_thaw(item) for item in value]

    @model_validator(mode="after")
    def validate_identity(self) -> "GovernedFrameFindingCorrectionV1":
        for field in (
            "correction_id", "organization_id", "workspace_id",
            "predecessor_finding_ref", "grounding_event_ref", "actor_ref",
            "reason_code", "source_identity_ref", "canonical_pattern_ref",
            "accepted_graph_release_ref", "governed_rule_version_ref",
            "activation_set_snapshot_ref", "activation_resolver_policy_version",
        ):
            _ascii(getattr(self, field), field)
        for item in self.evidence_refs:
            evidence = EvidenceRef.model_validate(_thaw(item))
            if (evidence.organization_id, evidence.workspace_id, evidence.source_identity_ref) != (
                self.organization_id, self.workspace_id, self.source_identity_ref
            ):
                raise ValueError("correction EvidenceRef scope/source mismatch")
            if evidence.provenance.get("source_hash") != self.source_hash:
                raise ValueError("correction EvidenceRef source hash mismatch")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.correction_hash != expected:
            raise ValueError("correction hash mismatch")
        if self.correction_id != "gffc_" + expected[7:31]:
            raise ValueError("correction id mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("correctionId", None)
        payload.pop("correctionHash", None)
        return payload


def create_governed_frame_finding_correction_v1(
    *,
    finding: GovernedFrameFindingV1,
    grounding_event: LearningEventEnvelope,
    grounding_event_hash: str,
) -> GovernedFrameFindingCorrectionV1:
    if type(finding) is not GovernedFrameFindingV1:
        raise TypeError("finding must be an exact GovernedFrameFindingV1")
    if type(grounding_event) is not LearningEventEnvelope:
        raise TypeError("grounding_event must be an exact LearningEventEnvelope")
    if grounding_event.event_type is not LearningEventType.HUMAN_EDIT or (
        grounding_event.human_label is None
        or grounding_event.human_label.decision is not HumanDecision.EDIT
    ):
        raise ValueError("finding correction requires a human edit event")
    if grounding_event.event_hash() != grounding_event_hash:
        raise ValueError("grounding event hash mismatch")
    if grounding_event.case_id != finding.finding_id:
        raise ValueError("grounding event case does not bind predecessor finding")
    if grounding_event.source_document_hash != finding.source_hash:
        raise ValueError("grounding event source does not bind predecessor finding")
    if tuple(grounding_event.source_anchor_refs) != finding.source_anchor_refs:
        raise ValueError("grounding event anchors do not bind predecessor finding")
    if grounding_event.graph_version != finding.accepted_graph_release_ref:
        raise ValueError("grounding event graph version does not bind predecessor finding")
    if grounding_event.rule_version != finding.governed_rule_version_ref:
        raise ValueError("grounding event rule version does not bind predecessor finding")
    if not grounding_event.actor_ref:
        raise ValueError("grounding event actor is required")
    outcome = grounding_event.downstream_outcome
    if not isinstance(outcome, dict) or set(outcome) != {
        "schema_version", "corrected_decision_status"
    } or outcome.get("schema_version") != "governed_frame_finding_disposition.v1":
        raise ValueError("human edit event requires a structured corrected disposition")
    corrected_decision_status = outcome.get("corrected_decision_status")
    if corrected_decision_status not in {
        "violation", "acceptable", "uncertain", "escalate"
    }:
        raise ValueError("human edit event has an invalid corrected disposition")

    payload = {
        "schemaVersion": "governed_frame_finding_correction.v1",
        "organizationId": finding.organization_id,
        "workspaceId": finding.workspace_id,
        "predecessorFindingRef": finding.finding_id,
        "predecessorFindingHash": finding.finding_hash,
        "groundingEventRef": grounding_event.event_id,
        "groundingEventHash": grounding_event_hash,
        "actorRef": grounding_event.actor_ref,
        "correctedDecisionStatus": corrected_decision_status,
        "reasonCode": grounding_event.human_label.reason_code,
        "rationale": grounding_event.human_label.justification,
        "sourceIdentityRef": finding.source_identity_ref,
        "sourceHash": finding.source_hash,
        "evidenceRefs": finding.model_dump(mode="json", by_alias=True)["evidenceRefs"],
        "sourceAnchorRefs": list(finding.source_anchor_refs),
        "canonicalPatternRef": finding.canonical_pattern_ref,
        "canonicalPatternHash": finding.canonical_pattern_hash,
        "acceptedGraphReleaseRef": finding.accepted_graph_release_ref,
        "acceptedGraphReleaseHash": finding.accepted_graph_release_hash,
        "governedRuleVersionRef": finding.governed_rule_version_ref,
        "governedRuleVersionHash": finding.governed_rule_version_hash,
        "activationSetSnapshotRef": finding.activation_set_snapshot_ref,
        "activationSetSnapshotHash": finding.activation_set_snapshot_hash,
        "activationResolverPolicyVersion": finding.activation_resolver_policy_version,
        "effectiveContextResolverHash": finding.effective_context_resolver_hash,
        "resolvedActivationSetHash": finding.resolved_activation_set_hash,
    }
    digest = "sha256:" + content_hash(payload)
    return GovernedFrameFindingCorrectionV1.model_validate(
        {**payload, "correctionId": "gffc_" + digest[7:31], "correctionHash": digest}
    )


__all__ = [
    "GovernedFrameFindingCorrectionV1",
    "create_governed_frame_finding_correction_v1",
]
