"""Content-addressed, authority-pinned deterministic frame-match result v2."""

from __future__ import annotations

from datetime import datetime
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from contracts.evidence_location import EvidenceRef
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_identity


_SHA = r"^sha256:[0-9a-f]{64}$"
_ASCII = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")


def _ascii(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.isascii() or _ASCII.fullmatch(value) is None:
        raise ValueError(f"{field} must be an ASCII-stable reference")
    return value


def _json(value: Any, field: str) -> None:
    if isinstance(value, float):
        raise ValueError(f"{field} contains a floating-point value")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or re.fullmatch(r"[a-z][a-z0-9_]*", key) is None:
                raise ValueError(f"{field} contains a non-ASCII-stable schema key")
            _json(nested, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _json(nested, field)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(nested) for key, nested in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(nested) for nested in value]
    return value


class FrameMatchResultDTOV2(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    schema_version: Literal["frame_match_result_dto.v2"] = Field(
        default="frame_match_result_dto.v2", alias="schemaVersion"
    )
    result_id: str = Field(alias="resultId")
    result_hash: str = Field(alias="resultHash", pattern=_SHA)
    organization_id: str = Field(alias="organizationId")
    workspace_id: str = Field(alias="workspaceId")
    semantic_relation: Literal[
        "FRAME_EQUIVALENT", "FRAME_ENTAILS", "NO_MATCH", "UNRESOLVED"
    ] = Field(alias="semanticRelation")
    frame_match_state: Literal["MATCH", "NO_MATCH", "UNKNOWN"] = Field(
        alias="frameMatchState"
    )
    observed_hypothesis_ref: str = Field(alias="observedHypothesisRef")
    observed_hypothesis_hash: str = Field(alias="observedHypothesisHash", pattern=_SHA)
    source_identity_ref: str = Field(alias="sourceIdentityRef")
    source_hash: str = Field(alias="sourceHash", pattern=_SHA)
    evidence_refs: tuple[Any, ...] = Field(alias="evidenceRefs", min_length=1)
    source_anchor_refs: tuple[str, ...] = Field(default=(), alias="sourceAnchorRefs")
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
    slot_diagnostics: tuple[Any, ...] = Field(alias="slotDiagnostics")
    slot_diagnostics_hash: str = Field(alias="slotDiagnosticsHash", pattern=_SHA)
    hard_negative_diagnostics: tuple[Any, ...] = Field(alias="hardNegativeDiagnostics")
    exception_diagnostics: tuple[Any, ...] = Field(alias="exceptionDiagnostics")
    derivation_hash: str = Field(alias="derivationHash", pattern=_SHA)

    @field_validator("source_anchor_refs", mode="before")
    @classmethod
    def validate_anchors(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("sourceAnchorRefs must be an array")
        refs = tuple(_ascii(item, "sourceAnchorRefs[]") for item in value)
        if len(refs) != len(set(refs)):
            raise ValueError("sourceAnchorRefs contains duplicate identities")
        return refs

    @field_validator(
        "slot_diagnostics",
        "hard_negative_diagnostics",
        "exception_diagnostics",
        mode="before",
    )
    @classmethod
    def freeze_diagnostics(cls, value: Any) -> tuple[Any, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("diagnostics must be an array")
        if any(not isinstance(item, Mapping) for item in value):
            raise ValueError("each diagnostic must be an object")
        return tuple(_freeze_json(item) for item in value)

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def freeze_evidence(cls, value: Any) -> tuple[Any, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("evidenceRefs must be a nonempty array")
        return tuple(
            _freeze_json(
                EvidenceRef.model_validate(item).model_dump(mode="json", by_alias=True)
            )
            for item in value
        )

    @field_serializer(
        "slot_diagnostics", "hard_negative_diagnostics", "exception_diagnostics"
    )
    def serialize_diagnostics(self, value: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [_thaw_json(item) for item in value]

    @field_serializer("evidence_refs")
    def serialize_evidence(self, value: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [_thaw_json(item) for item in value]

    @model_validator(mode="after")
    def validate_contract(self) -> "FrameMatchResultDTOV2":
        expected_state = {
            "FRAME_EQUIVALENT": "MATCH",
            "FRAME_ENTAILS": "MATCH",
            "NO_MATCH": "NO_MATCH",
            "UNRESOLVED": "UNKNOWN",
        }[self.semantic_relation]
        if self.frame_match_state != expected_state:
            raise ValueError("semantic relation/frame match state mapping mismatch")
        for field in (
            "organization_id",
            "workspace_id",
            "observed_hypothesis_ref",
            "source_identity_ref",
            "canonical_pattern_ref",
            "accepted_graph_release_ref",
            "governed_rule_version_ref",
            "activation_set_snapshot_ref",
            "activation_resolver_policy_version",
        ):
            _ascii(getattr(self, field), field)
        evidence_ids = []
        for frozen_evidence in self.evidence_refs:
            evidence = EvidenceRef.model_validate(_thaw_json(frozen_evidence))
            if (evidence.organization_id, evidence.workspace_id) != (
                self.organization_id,
                self.workspace_id,
            ):
                raise ValueError("EvidenceRef organization/workspace scope mismatch")
            if evidence.source_identity_ref != self.source_identity_ref:
                raise ValueError("EvidenceRef SourceIdentity mismatch")
            if evidence.provenance.get("source_hash") != self.source_hash:
                raise ValueError("EvidenceRef source hash provenance mismatch")
            if evidence.selector_hash != evidence.selector.selector_hash():
                raise ValueError("EvidenceRef selector hash mismatch")
            if evidence.selector_schema_version != evidence.selector.schema_version:
                raise ValueError("EvidenceRef selector schema mismatch")
            if (
                not isinstance(evidence.created_at, datetime)
                or evidence.created_at.tzinfo is None
                or evidence.created_at.utcoffset() is None
                or evidence.created_at.utcoffset().total_seconds() != 0
            ):
                raise ValueError("EvidenceRef.createdAt must be timezone-aware UTC")
            expected = evidence_ref_identity(
                organization_id=evidence.organization_id,
                workspace_id=evidence.workspace_id,
                source_identity_ref=evidence.source_identity_ref,
                selector_hash=evidence.selector_hash,
                evidence_role=evidence.evidence_role,
            )
            if evidence.evidence_ref != expected:
                raise ValueError("EvidenceRef identity mismatch")
            evidence_ids.append(evidence.evidence_ref)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidenceRefs contains a duplicate identity")
        for field in (
            "slot_diagnostics",
            "hard_negative_diagnostics",
            "exception_diagnostics",
        ):
            _json(getattr(self, field), field)
        expected_slot_hash = "sha256:" + content_hash(
            [_thaw_json(item) for item in self.slot_diagnostics]
        )
        if self.slot_diagnostics_hash != expected_slot_hash:
            raise ValueError("slot diagnostics hash mismatch")
        if self.derivation_hash != "sha256:" + content_hash(self.derivation_payload()):
            raise ValueError("derivation hash mismatch")
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.result_hash != expected_hash:
            raise ValueError("result hash mismatch")
        if self.result_id != "fmr_" + expected_hash[7:31]:
            raise ValueError("result id mismatch")
        return self

    def derivation_payload(self) -> dict[str, Any]:
        inputs = self.model_dump(mode="json", by_alias=False)
        for field in ("result_id", "result_hash", "source_anchor_refs", "derivation_hash"):
            inputs.pop(field, None)
        return {"schema_version": "frame_match_derivation.v2", "inputs": inputs}

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("resultId", None)
        payload.pop("resultHash", None)
        payload.pop("sourceAnchorRefs", None)
        return payload


def create_frame_match_result_v2(**values: Any) -> FrameMatchResultDTOV2:
    slot_diagnostics = [dict(item) for item in values.pop("slot_diagnostics")]
    hard_negative_diagnostics = [
        dict(item) for item in values.pop("hard_negative_diagnostics")
    ]
    exception_diagnostics = [dict(item) for item in values.pop("exception_diagnostics")]
    slot_hash = "sha256:" + content_hash(slot_diagnostics)
    payload = {
        "schemaVersion": "frame_match_result_dto.v2",
        **{
            FrameMatchResultDTOV2.model_fields[key].alias: value
            for key, value in values.items()
        },
        "slotDiagnostics": slot_diagnostics,
        "slotDiagnosticsHash": slot_hash,
        "hardNegativeDiagnostics": hard_negative_diagnostics,
        "exceptionDiagnostics": exception_diagnostics,
    }
    provisional_values = {
            name: payload.get(field.alias)
            for name, field in FrameMatchResultDTOV2.model_fields.items()
            if name not in {"result_id", "result_hash", "derivation_hash"}
        }
    for name in (
        "source_anchor_refs",
        "slot_diagnostics",
        "hard_negative_diagnostics",
        "exception_diagnostics",
    ):
        provisional_values[name] = tuple(provisional_values[name])
    provisional_values["evidence_refs"] = tuple(
        _freeze_json(item) for item in provisional_values["evidence_refs"]
    )
    provisional = FrameMatchResultDTOV2.model_construct(
        **provisional_values,
        result_id="",
        result_hash="sha256:" + "0" * 64,
        derivation_hash="sha256:" + "0" * 64,
    )
    payload = provisional.model_dump(mode="json", by_alias=True)
    payload["derivationHash"] = "sha256:" + content_hash(provisional.derivation_payload())
    hash_payload = dict(payload)
    hash_payload.pop("resultId", None)
    hash_payload.pop("resultHash", None)
    hash_payload.pop("sourceAnchorRefs", None)
    result_hash = "sha256:" + content_hash(hash_payload)
    return FrameMatchResultDTOV2.model_validate(
        {
            **payload,
            "resultId": "fmr_" + result_hash[7:31],
            "resultHash": result_hash,
        }
    )


__all__ = ["FrameMatchResultDTOV2", "create_frame_match_result_v2"]
