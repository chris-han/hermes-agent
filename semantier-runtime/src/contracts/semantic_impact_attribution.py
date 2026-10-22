"""Additive semantic-impact axis for governed schema-learning feedback."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from contracts.evidence_location import EvidenceRef
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_identity


SemanticImpactLayer = Literal[
    "source_structure",
    "graph_intent",
    "schema",
    "canonical_pattern",
    "runtime_observation",
    "instance",
]
LegacyFeedbackCause = Literal[
    "source_structure",
    "graph_intent",
    "ontology_induction",
    "synthesis",
    "extraction_plan",
    "execution",
]

SEMANTIC_IMPACT_TO_LEGACY_CAUSE: Mapping[str, str] = MappingProxyType(
    {
        "source_structure": "source_structure",
        "graph_intent": "graph_intent",
        "schema": "ontology_induction",
        "canonical_pattern": "synthesis",
        "runtime_observation": "extraction_plan",
        "instance": "execution",
    }
)

_SUCCESSOR_REQUIREMENTS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "source_structure": frozenset(
            {
                "source_structure",
                "source_semantic_validation",
                "graph_intent",
                "schema",
                "canonical_pattern",
                "accepted_release",
                "evaluation",
            }
        ),
        "graph_intent": frozenset(
            {
                "graph_intent",
                "schema",
                "canonical_pattern",
                "accepted_release",
                "evaluation",
            }
        ),
        "schema": frozenset(
            {"schema", "canonical_pattern", "accepted_release", "evaluation"}
        ),
        "canonical_pattern": frozenset(
            {"canonical_pattern", "accepted_release", "evaluation"}
        ),
        "runtime_observation": frozenset(
            {
                "observed_hypothesis",
                "frame_match_result",
                "finding_instance",
                "evaluation",
            }
        ),
        "instance": frozenset({"finding_instance", "evaluation"}),
    }
)

_SHA = r"^sha256:[0-9a-f]{64}$"
_ASCII = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")


def _ascii_ref(value: str, field: str) -> str:
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


def legacy_cause_for_semantic_impact(layer: str) -> str:
    try:
        return SEMANTIC_IMPACT_TO_LEGACY_CAUSE[layer]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unmapped semantic impact layer: {layer!r}") from exc


def successor_requirements_for_semantic_impact(
    layer: str, *, vocabulary_definition_defect: bool = False
) -> tuple[str, ...]:
    legacy_cause_for_semantic_impact(layer)
    if vocabulary_definition_defect and layer != "schema":
        raise ValueError(
            "vocabulary_definition_defect is only valid for schema impact"
        )
    requirements = set(_SUCCESSOR_REQUIREMENTS[layer])
    if vocabulary_definition_defect:
        requirements.add("taxonomy_or_profile")
    return tuple(sorted(requirements))


def validate_semantic_successor_effects(
    layer: str,
    *,
    actual_effects: Sequence[str],
    vocabulary_definition_defect: bool = False,
) -> tuple[str, ...]:
    expected = successor_requirements_for_semantic_impact(
        layer,
        vocabulary_definition_defect=vocabulary_definition_defect,
    )
    actual = tuple(sorted(actual_effects))
    if len(actual) != len(set(actual)) or actual != expected:
        raise ValueError(
            f"semantic successor effects mismatch: expected {expected}, got {actual}"
        )
    return actual


class SemanticImpactAttributionV1(BaseModel):
    """Content-addressed companion; never participates in feedback identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["semantic_impact_attribution.v1"] = (
        "semantic_impact_attribution.v1"
    )
    attribution_id: str
    attribution_hash: str = Field(pattern=_SHA)
    organization_id: str
    workspace_id: str
    feedback_ref: str
    feedback_hash: str = Field(pattern=_SHA)
    semantic_impact_layer: SemanticImpactLayer
    mapped_legacy_cause: LegacyFeedbackCause
    target_artifact_ref: str
    target_artifact_hash: str = Field(pattern=_SHA)
    decision_evidence_refs: tuple[Any, ...] = Field(min_length=1)
    actor_ref: str
    predecessor_refs: tuple[str, ...]
    successor_refs: tuple[str, ...]

    @field_validator("decision_evidence_refs", mode="before")
    @classmethod
    def freeze_evidence(cls, value: Any) -> tuple[Any, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("decision_evidence_refs must be a nonempty array")
        return tuple(
            _freeze(EvidenceRef.model_validate(item).model_dump(mode="json", by_alias=True))
            for item in value
        )

    @field_validator("predecessor_refs", "successor_refs", mode="before")
    @classmethod
    def validate_lineage_refs(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("lineage refs must be an array")
        refs = tuple(_ascii_ref(item, "lineage ref") for item in value)
        if len(refs) != len(set(refs)):
            raise ValueError("lineage refs contain duplicate identities")
        return refs

    @field_serializer("decision_evidence_refs")
    def serialize_evidence(self, value: tuple[Any, ...]) -> list[Any]:
        return [_thaw(item) for item in value]

    @model_validator(mode="after")
    def validate_mapping_scope_and_identity(self) -> "SemanticImpactAttributionV1":
        expected_cause = legacy_cause_for_semantic_impact(self.semantic_impact_layer)
        if self.mapped_legacy_cause != expected_cause:
            raise ValueError("mapped legacy cause does not match semantic impact layer")
        for field in (
            "organization_id",
            "workspace_id",
            "feedback_ref",
            "target_artifact_ref",
            "actor_ref",
        ):
            _ascii_ref(getattr(self, field), field)
        evidence_ids: list[str] = []
        for item in self.decision_evidence_refs:
            evidence = EvidenceRef.model_validate(_thaw(item))
            if (evidence.organization_id, evidence.workspace_id) != (
                self.organization_id,
                self.workspace_id,
            ):
                raise ValueError("decision EvidenceRef scope mismatch")
            if evidence.selector_hash != evidence.selector.selector_hash():
                raise ValueError("decision EvidenceRef selector hash mismatch")
            if evidence.selector_schema_version != evidence.selector.schema_version:
                raise ValueError("decision EvidenceRef selector schema mismatch")
            if (
                evidence.created_at.tzinfo is None
                or evidence.created_at.utcoffset() is None
                or evidence.created_at.utcoffset().total_seconds() != 0
            ):
                raise ValueError("decision EvidenceRef.createdAt must be UTC")
            expected_ref = evidence_ref_identity(
                organization_id=evidence.organization_id,
                workspace_id=evidence.workspace_id,
                source_identity_ref=evidence.source_identity_ref,
                selector_hash=evidence.selector_hash,
                evidence_role=evidence.evidence_role,
            )
            if evidence.evidence_ref != expected_ref:
                raise ValueError("decision EvidenceRef identity mismatch")
            evidence_ids.append(evidence.evidence_ref)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("decision_evidence_refs contain duplicate identities")
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.attribution_hash != expected_hash:
            raise ValueError("attribution hash mismatch")
        if self.attribution_id != "sia_" + expected_hash[7:31]:
            raise ValueError("attribution id mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("attribution_id", None)
        payload.pop("attribution_hash", None)
        return payload


def create_semantic_impact_attribution_v1(
    *,
    organization_id: str,
    workspace_id: str,
    feedback_ref: str,
    feedback_hash: str,
    semantic_impact_layer: str,
    target_artifact_ref: str,
    target_artifact_hash: str,
    decision_evidence_refs: Sequence[EvidenceRef | Mapping[str, Any]],
    actor_ref: str,
    predecessor_refs: Sequence[str],
    successor_refs: Sequence[str],
) -> SemanticImpactAttributionV1:
    mapped_cause = legacy_cause_for_semantic_impact(semantic_impact_layer)
    payload = {
        "schema_version": "semantic_impact_attribution.v1",
        "organization_id": organization_id,
        "workspace_id": workspace_id,
        "feedback_ref": feedback_ref,
        "feedback_hash": feedback_hash,
        "semantic_impact_layer": semantic_impact_layer,
        "mapped_legacy_cause": mapped_cause,
        "target_artifact_ref": target_artifact_ref,
        "target_artifact_hash": target_artifact_hash,
        "decision_evidence_refs": [
            EvidenceRef.model_validate(item).model_dump(mode="json", by_alias=True)
            for item in decision_evidence_refs
        ],
        "actor_ref": actor_ref,
        "predecessor_refs": list(predecessor_refs),
        "successor_refs": list(successor_refs),
    }
    digest = "sha256:" + content_hash(payload)
    return SemanticImpactAttributionV1.model_validate(
        {
            **payload,
            "attribution_id": "sia_" + digest[7:31],
            "attribution_hash": digest,
        }
    )


__all__ = [
    "LegacyFeedbackCause",
    "SEMANTIC_IMPACT_TO_LEGACY_CAUSE",
    "SemanticImpactAttributionV1",
    "SemanticImpactLayer",
    "create_semantic_impact_attribution_v1",
    "legacy_cause_for_semantic_impact",
    "successor_requirements_for_semantic_impact",
    "validate_semantic_successor_effects",
]
