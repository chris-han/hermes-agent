"""Content-addressed Stage-B routing receipts, not qualification or authority."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.project_memory_state import (
    EpistemicState,
    ProjectStateDerivationV1,
    SHA256_PATTERN,
    StateDimension,
)
from eos.artifact_hashing import content_hash


class ResidualSourceConstraintsV1(BaseModel):
    """Snapshot of candidate metadata; never a source-eligibility attestation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_role: str = Field(min_length=1)
    evidence_class: Literal[
        "SOURCE_RECORD",
        "AUTHORITY_RECORD",
        "EXECUTION_RECEIPT",
        "PROJECT_CHAT",
        "DERIVED_MEMORY",
    ]
    epistemic_state: EpistemicState
    eligible_frontier: tuple[str, ...]
    basis: Literal["CANDIDATE_ASSERTED_METADATA_NON_AUTHORITY"] = (
        "CANDIDATE_ASSERTED_METADATA_NON_AUTHORITY"
    )

    @field_validator("source_role")
    @classmethod
    def ascii_role(cls, value: str) -> str:
        if not value.isascii():
            raise ValueError("source_role must be an ASCII-stable identifier")
        return value

    @field_validator("eligible_frontier")
    @classmethod
    def genuine_choice(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) < 2 or len(value) != len(set(value)):
            raise ValueError(
                "semantic residual must have at least two distinct eligible states"
            )
        if "UNKNOWN" not in value or any(not x.isascii() for x in value):
            raise ValueError(
                "residual frontier must be ASCII-stable and preserve UNKNOWN"
            )
        return value


class _ResidualBasis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["residual_stage_b_input.v1"] = "residual_stage_b_input.v1"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    memory_candidate: ArtifactPinV1
    stage_a_projection: ArtifactPinV1
    compiler_profile: ArtifactPinV1
    target_dimension: StateDimension
    stage_a_derivation: ProjectStateDerivationV1
    unresolved_reason: Literal["LOWERER_UNRESOLVED", "REQUESTED_DIMENSION_UNRESOLVED"]
    source_constraints: ResidualSourceConstraintsV1
    input_text_hash: str = Field(pattern=SHA256_PATTERN)
    purpose: Literal["ROUTING_ONLY_NOT_QUALIFICATION"] = (
        "ROUTING_ONLY_NOT_QUALIFICATION"
    )

    @model_validator(mode="after")
    def require_unresolved_basis(self) -> Self:
        for name, kind in (
            ("memory_candidate", "memory_candidate"),
            ("stage_a_projection", "project_state_projection"),
            ("compiler_profile", "project_memory_residual_profile"),
        ):
            if getattr(self, name).artifact_kind != kind:
                raise ValueError(f"{name} must pin artifact kind {kind}")
        if self.stage_a_derivation.dimension != self.target_dimension:
            raise ValueError("Stage-A derivation must match target dimension")
        if self.stage_a_derivation.method != "UNRESOLVED":
            raise ValueError(
                "only an exact UNRESOLVED Stage-A derivation can route to Stage B"
            )
        if self.stage_a_derivation.decision_receipt is not None:
            raise ValueError("Stage-A residual cannot contain a model decision")
        allowed = {
            "epistemic": {"OBSERVED", "CLAIMED", "INFERRED"},
            "authority": {"PROPOSED", "UNAPPROVED", "APPROVED", "ACTIVATED", "UNKNOWN"},
            "execution": {"NOT_RUN", "RUNNING", "SUCCEEDED", "FAILED", "UNKNOWN"},
            "verification": {"NOT_RUN", "PASSED", "FAILED", "UNKNOWN"},
        }[self.target_dimension]
        if not set(self.source_constraints.eligible_frontier) <= allowed:
            raise ValueError(
                "residual frontier contains states outside the target dimension"
            )
        return self


class ResidualStageBInputV1(_ResidualBasis):
    """A replayable eligibility-to-evaluate receipt. It never enables Stage B.

    Shape/hash validation is necessary, not sufficient: consumers must replay
    the router against the exact candidate and current compiler-profile pin.
    """

    residual_ref: str
    residual_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def identity_matches(self) -> Self:
        payload = self.model_dump(
            mode="json", exclude={"residual_ref", "residual_hash"}
        )
        digest = content_hash(payload)
        if (
            self.residual_hash != "sha256:" + digest
            or self.residual_ref != "residual-stage-b:" + digest[:24]
        ):
            raise ValueError("residual receipt identity mismatch")
        return self

    @classmethod
    def create(cls, **fields: Any) -> Self:
        payload = _ResidualBasis.model_validate(fields).model_dump(mode="json")
        digest = content_hash(payload)
        return cls.model_validate(
            {
                **payload,
                "residual_ref": "residual-stage-b:" + digest[:24],
                "residual_hash": "sha256:" + digest,
            }
        )
