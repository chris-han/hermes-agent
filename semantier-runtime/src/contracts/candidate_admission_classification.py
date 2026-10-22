"""Non-authoritative routing classification for assessed learning candidates.

This contract records deterministic admission-boundary routing evidence only.
It is not an assessment, owner admission, authorization, activation, or effective state.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.artifact_pin import ArtifactPinV1


CandidateAdmissionDisposition: TypeAlias = Literal[
    "PROCEDURAL_SAFE_TO_PROJECT",
    "SEMANTIC_CHANGE_REQUIRES_ADMISSION",
    "AUTHORITY_VIOLATION",
    "INSUFFICIENT_EVIDENCE",
]

AdmissionDimension: TypeAlias = Literal[
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
]

_DIMENSION_ORDER = {f"D{index}": index for index in range(1, 10)}


class CandidateAdmissionClassificationV1(BaseModel):
    """Derived routing evidence; deliberately carries no admission authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: ArtifactPinV1
    disposition: CandidateAdmissionDisposition
    changed_dimensions: tuple[AdmissionDimension, ...] = ()
    classifier_version: str = Field(min_length=1)
    classifier_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    evidence_refs: tuple[ArtifactPinV1, ...] = ()
    reason_code: str = Field(min_length=1)

    @field_validator("changed_dimensions")
    @classmethod
    def validate_changed_dimensions(
        cls,
        value: tuple[AdmissionDimension, ...],
    ) -> tuple[AdmissionDimension, ...]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate admission dimension")
        if tuple(sorted(value, key=_DIMENSION_ORDER.__getitem__)) != value:
            raise ValueError("changed_dimensions must use canonical D1-D9 order")
        return value

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(
        cls,
        value: tuple[ArtifactPinV1, ...],
    ) -> tuple[ArtifactPinV1, ...]:
        identities = [
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
            for pin in value
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate evidence ref")
        return value

    @model_validator(mode="after")
    def validate_candidate_boundary(self) -> "CandidateAdmissionClassificationV1":
        if self.candidate.artifact_kind != "change_candidate":
            raise ValueError("candidate must pin change_candidate")
        return self


__all__ = [
    "AdmissionDimension",
    "CandidateAdmissionClassificationV1",
    "CandidateAdmissionDisposition",
]
