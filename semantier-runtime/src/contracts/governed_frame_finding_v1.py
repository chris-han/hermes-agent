"""Authority-pinned companion finding derived from a governed frame match."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from contracts.frame_match_result_v2 import FrameMatchResultDTOV2
from contracts.evidence_location import EvidenceRef
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_identity


_SHA = r"^sha256:[0-9a-f]{64}$"
_ASCII = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")


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


class FrameFindingDecisionV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["candidate", "acceptable", "uncertain"]
    structural_match: bool
    semantic_relation: Literal["FRAME_EQUIVALENT", "FRAME_ENTAILS", "NO_MATCH", "UNRESOLVED"]
    frame_match_state: Literal["MATCH", "NO_MATCH", "UNKNOWN"]
    match_result_ref: str
    match_result_hash: str = Field(pattern=_SHA)
    rationale: str

    @model_validator(mode="after")
    def exact_mapping(self) -> "FrameFindingDecisionV2":
        expected_state = {
            "FRAME_EQUIVALENT": "MATCH",
            "FRAME_ENTAILS": "MATCH",
            "NO_MATCH": "NO_MATCH",
            "UNRESOLVED": "UNKNOWN",
        }[self.semantic_relation]
        if self.frame_match_state != expected_state:
            raise ValueError("frame finding semantic relation/state mismatch")
        expected = {
            "MATCH": ("candidate", True),
            "NO_MATCH": ("acceptable", False),
            "UNKNOWN": ("uncertain", False),
        }[self.frame_match_state]
        if (self.status, self.structural_match) != expected:
            raise ValueError("frame finding decision mapping mismatch")
        _ascii(self.match_result_ref, "match_result_ref")
        return self


class GovernedFrameFindingV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["governed_frame_finding.v1"] = Field(
        default="governed_frame_finding.v1", alias="schemaVersion"
    )
    finding_id: str = Field(alias="findingId")
    finding_hash: str = Field(alias="findingHash", pattern=_SHA)
    organization_id: str = Field(alias="organizationId")
    workspace_id: str = Field(alias="workspaceId")
    decision_status: Literal["candidate", "violation", "acceptable", "uncertain", "escalate"] = Field(alias="decisionStatus")
    structural_match: bool = Field(alias="structuralMatch")
    semantic_relation: Literal["FRAME_EQUIVALENT", "FRAME_ENTAILS", "NO_MATCH", "UNRESOLVED"] = Field(alias="semanticRelation")
    frame_match_state: Literal["MATCH", "NO_MATCH", "UNKNOWN"] = Field(alias="frameMatchState")
    frame_match_result_ref: str = Field(alias="frameMatchResultRef")
    frame_match_result_hash: str = Field(alias="frameMatchResultHash", pattern=_SHA)
    observed_hypothesis_ref: str = Field(alias="observedHypothesisRef")
    observed_hypothesis_hash: str = Field(alias="observedHypothesisHash", pattern=_SHA)
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
    rationale: str

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def freeze_evidence(cls, value: Any) -> tuple[Any, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("evidenceRefs must be a nonempty array")
        return tuple(
            _freeze(EvidenceRef.model_validate(item).model_dump(mode="json", by_alias=True))
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
    def validate_identity(self) -> "GovernedFrameFindingV1":
        expected_axes = {
            "FRAME_EQUIVALENT": ("MATCH", "candidate", True),
            "FRAME_ENTAILS": ("MATCH", "candidate", True),
            "NO_MATCH": ("NO_MATCH", "acceptable", False),
            "UNRESOLVED": ("UNKNOWN", "uncertain", False),
        }[self.semantic_relation]
        if (self.frame_match_state, self.decision_status, self.structural_match) != expected_axes:
            raise ValueError("finding semantic/match/decision mapping mismatch")
        for field in (
            "organization_id", "workspace_id", "frame_match_result_ref",
            "observed_hypothesis_ref", "source_identity_ref", "canonical_pattern_ref",
            "accepted_graph_release_ref", "governed_rule_version_ref",
            "activation_set_snapshot_ref", "activation_resolver_policy_version",
        ):
            _ascii(getattr(self, field), field)
        evidence_ids: list[str] = []
        for item in self.evidence_refs:
            evidence = EvidenceRef.model_validate(_thaw(item))
            if (evidence.organization_id, evidence.workspace_id, evidence.source_identity_ref) != (
                self.organization_id, self.workspace_id, self.source_identity_ref
            ):
                raise ValueError("finding EvidenceRef scope/source mismatch")
            if evidence.provenance.get("source_hash") != self.source_hash:
                raise ValueError("finding EvidenceRef source hash mismatch")
            if evidence.selector_hash != evidence.selector.selector_hash():
                raise ValueError("finding EvidenceRef selector hash mismatch")
            if evidence.selector_schema_version != evidence.selector.schema_version:
                raise ValueError("finding EvidenceRef selector schema mismatch")
            if evidence.created_at.tzinfo is None or evidence.created_at.utcoffset() is None or evidence.created_at.utcoffset().total_seconds() != 0:
                raise ValueError("finding EvidenceRef.createdAt must be timezone-aware UTC")
            expected_ref = evidence_ref_identity(
                organization_id=evidence.organization_id, workspace_id=evidence.workspace_id,
                source_identity_ref=evidence.source_identity_ref,
                selector_hash=evidence.selector_hash, evidence_role=evidence.evidence_role,
            )
            if evidence.evidence_ref != expected_ref:
                raise ValueError("finding EvidenceRef identity mismatch")
            evidence_ids.append(evidence.evidence_ref)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("finding evidenceRefs contains a duplicate identity")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.finding_hash != expected:
            raise ValueError("finding hash mismatch")
        if self.finding_id != "gff_" + expected[7:31]:
            raise ValueError("finding id mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("findingId", None)
        payload.pop("findingHash", None)
        payload.pop("sourceAnchorRefs", None)
        return payload


def create_governed_frame_finding_v1(*, frame_match: FrameMatchResultDTOV2, decision: FrameFindingDecisionV2) -> GovernedFrameFindingV1:
    if type(decision) is not FrameFindingDecisionV2:
        raise TypeError("decision must be an exact FrameFindingDecisionV2")
    if (decision.match_result_ref, decision.match_result_hash) != (
        frame_match.result_id, frame_match.result_hash
    ):
        raise ValueError("finding decision does not bind the frame match result")
    if (decision.semantic_relation, decision.frame_match_state) != (
        frame_match.semantic_relation, frame_match.frame_match_state
    ):
        raise ValueError("finding decision axes differ from frame match result")
    payload = {
        "schemaVersion": "governed_frame_finding.v1",
        "organizationId": frame_match.organization_id,
        "workspaceId": frame_match.workspace_id,
        "decisionStatus": decision.status,
        "structuralMatch": decision.structural_match,
        "semanticRelation": frame_match.semantic_relation,
        "frameMatchState": frame_match.frame_match_state,
        "frameMatchResultRef": frame_match.result_id,
        "frameMatchResultHash": frame_match.result_hash,
        "observedHypothesisRef": frame_match.observed_hypothesis_ref,
        "observedHypothesisHash": frame_match.observed_hypothesis_hash,
        "sourceIdentityRef": frame_match.source_identity_ref,
        "sourceHash": frame_match.source_hash,
        "evidenceRefs": frame_match.model_dump(mode="json", by_alias=True)["evidenceRefs"],
        "sourceAnchorRefs": list(frame_match.source_anchor_refs),
        "canonicalPatternRef": frame_match.canonical_pattern_ref,
        "canonicalPatternHash": frame_match.canonical_pattern_hash,
        "acceptedGraphReleaseRef": frame_match.accepted_graph_release_ref,
        "acceptedGraphReleaseHash": frame_match.accepted_graph_release_hash,
        "governedRuleVersionRef": frame_match.governed_rule_version_ref,
        "governedRuleVersionHash": frame_match.governed_rule_version_hash,
        "activationSetSnapshotRef": frame_match.activation_set_snapshot_ref,
        "activationSetSnapshotHash": frame_match.activation_set_snapshot_hash,
        "activationResolverPolicyVersion": frame_match.activation_resolver_policy_version,
        "effectiveContextResolverHash": frame_match.effective_context_resolver_hash,
        "resolvedActivationSetHash": frame_match.resolved_activation_set_hash,
        "rationale": decision.rationale,
    }
    digest = "sha256:" + content_hash({
        key: value for key, value in payload.items() if key != "sourceAnchorRefs"
    })
    return GovernedFrameFindingV1.model_validate({
        **payload, "findingId": "gff_" + digest[7:31], "findingHash": digest
    })


__all__ = ["FrameFindingDecisionV2", "GovernedFrameFindingV1", "create_governed_frame_finding_v1"]
