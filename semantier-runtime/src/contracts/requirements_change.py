"""Minimum owner-native requirements baseline/admission contracts for the W8 pilot."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from contracts.governed_delta import ArtifactVersionPinV1
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import _ContentAddressed, _ordered_unique_pins


class RequirementBaselineV1(_ContentAddressed):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ARTIFACT_KIND = "requirement_baseline"

    schema_version: Literal["requirement_baseline.v1"] = "requirement_baseline.v1"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    owner_ref: str = Field(min_length=1)
    baseline_ref: str = Field(min_length=1)
    predecessor_baseline: ArtifactPinV1 | None = None
    member_versions: tuple[ArtifactVersionPinV1, ...]
    recorded_at: datetime

    @field_validator("member_versions", mode="before")
    @classmethod
    def canonical_members(cls, value: object) -> object:
        if not isinstance(value, (tuple, list)) or not value:
            raise ValueError("member_versions must be non-empty")
        members = tuple(
            item if isinstance(item, ArtifactVersionPinV1) else ArtifactVersionPinV1.model_validate(item)
            for item in value
        )
        keys = tuple((item.artifact_type, item.artifact_id, item.version_id, item.content_hash) for item in members)
        if len(keys) != len(set(keys)):
            raise ValueError("baseline member_versions must be unique")
        return tuple(sorted(members, key=lambda item: (item.artifact_type, item.artifact_id, item.version_id, item.content_hash)))


class RequirementQualificationReceiptV1(_ContentAddressed):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ARTIFACT_KIND = "requirement_change_qualification"

    schema_version: Literal["requirement_change_qualification.v1"] = "requirement_change_qualification.v1"
    organization_id: str
    workspace_id: str
    owner_ref: str
    candidate: ArtifactPinV1
    qualification_refs: tuple[ArtifactPinV1, ...]
    disposition: Literal["QUALIFIED"] = "QUALIFIED"
    recorded_at: datetime

    @field_validator("qualification_refs", mode="before")
    @classmethod
    def qualification_nonempty(cls, value: object) -> object:
        refs = _ordered_unique_pins(value)
        if not isinstance(refs, tuple) or not refs:
            raise ValueError("qualification_refs must be non-empty")
        return refs


class RequirementAdmissionReceiptV1(_ContentAddressed):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ARTIFACT_KIND = "requirement_change_admission"

    schema_version: Literal["requirement_change_admission.v1"] = "requirement_change_admission.v1"
    organization_id: str
    workspace_id: str
    owner_ref: str
    candidate: ArtifactPinV1
    qualification: ArtifactPinV1
    predecessor_baseline: ArtifactPinV1
    successor_baseline: ArtifactPinV1
    disposition: Literal["ADMITTED"] = "ADMITTED"
    recorded_at: datetime


class RequirementActivationReceiptV1(_ContentAddressed):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ARTIFACT_KIND = "requirement_baseline_activation"

    schema_version: Literal["requirement_baseline_activation.v1"] = "requirement_baseline_activation.v1"
    organization_id: str
    workspace_id: str
    owner_ref: str
    baseline: ArtifactPinV1
    admission: ArtifactPinV1
    effective_scope: ArtifactPinV1
    recorded_at: datetime


class RequirementContextProjectionV1(_ContentAddressed):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ARTIFACT_KIND = "requirement_context_projection"

    schema_version: Literal["requirement_context_projection.v1"] = "requirement_context_projection.v1"
    organization_id: str
    workspace_id: str
    baseline: ArtifactPinV1
    member_versions: tuple[ArtifactVersionPinV1, ...]
    projected_at: datetime


__all__ = [
    "RequirementActivationReceiptV1",
    "RequirementAdmissionReceiptV1",
    "RequirementBaselineV1",
    "RequirementContextProjectionV1",
    "RequirementQualificationReceiptV1",
]
