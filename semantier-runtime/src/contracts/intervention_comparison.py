"""Frozen W4 comparability and intervention manifests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.artifact_pin import ArtifactPinV1
from contracts.candidate_stage_binding import OwningStage
from eos.artifact_hashing import content_hash

_SHA = r"^sha256:[0-9a-f]{64}$"
ComparisonMode = Literal["OBSERVATIONAL", "REPLAY", "INTERVENTION"]


def _digest(value) -> str:
    return "sha256:" + content_hash(value)


class ComparisonManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["comparison_manifest.v1"] = "comparison_manifest.v1"
    manifest_ref: str
    manifest_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)

    candidate: ArtifactPinV1
    owning_stage: OwningStage
    evaluator: ArtifactPinV1
    obligation_profile: ArtifactPinV1
    comparison_policy: ArtifactPinV1
    case_panel: tuple[ArtifactPinV1, ...]
    protected_holdout: tuple[ArtifactPinV1, ...]
    dgp: ArtifactPinV1 | None = None
    representation_contract: ArtifactPinV1 | None = None
    serializer_policy: ArtifactPinV1 | None = None
    context_envelope: ArtifactPinV1 | None = None
    runtime_pins: tuple[ArtifactPinV1, ...] = ()
    provider_pins: tuple[ArtifactPinV1, ...] = ()
    config_pins: tuple[ArtifactPinV1, ...] = ()
    mode: ComparisonMode
    declared_intervention: ArtifactPinV1 | None = None

    @model_validator(mode="after")
    def verify(self):
        if self.candidate.artifact_kind != "change_candidate":
            raise ValueError("COMPARISON_MANIFEST_REQUIRES_CANDIDATE")
        if not self.case_panel:
            raise ValueError("COMPARISON_MANIFEST_REQUIRES_CASE_PANEL")
        if self.mode == "INTERVENTION" and self.declared_intervention is None:
            raise ValueError("INTERVENTION_REQUIRES_DECLARED_INTERVENTION")
        if self.mode != "INTERVENTION" and self.declared_intervention is not None:
            raise ValueError("NON_INTERVENTION_CANNOT_DECLARE_INTERVENTION")
        if any(pin not in self.case_panel for pin in self.protected_holdout):
            raise ValueError("PROTECTED_HOLDOUT_MUST_BE_IN_CASE_PANEL")
        digest = _digest(self.hash_payload())
        if (self.manifest_ref, self.manifest_hash) != (
            "comparison-manifest:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("comparison manifest identity mismatch")
        return self

    def hash_payload(self):
        return self.model_dump(mode="json", exclude={"manifest_ref", "manifest_hash"})

    @classmethod
    def create(cls, **body):
        for field in (
            "case_panel",
            "protected_holdout",
            "runtime_pins",
            "provider_pins",
            "config_pins",
        ):
            values = tuple(sorted(
                set(body.get(field, ())),
                key=lambda p: (p.artifact_kind, p.artifact_ref, p.artifact_hash),
            ))
            body = {**body, field: values}
        provisional = cls.model_construct(
            manifest_ref="comparison-manifest:v1:pending",
            manifest_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "manifest_ref": "comparison-manifest:v1:" + digest[7:],
            "manifest_hash": digest,
        })

    def pin(self):
        return ArtifactPinV1(
            artifact_kind="comparison_manifest",
            artifact_ref=self.manifest_ref,
            artifact_hash=self.manifest_hash,
        )


class ComparisonCompatibilityV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["comparison_compatibility.v1"] = "comparison_compatibility.v1"
    receipt_ref: str
    receipt_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    left_manifest: ArtifactPinV1
    right_manifest: ArtifactPinV1
    status: Literal["COMPARABLE", "INCOMPARABLE"]
    causal_claim_permitted: bool
    differing_dimensions: tuple[str, ...] = ()
    declared_intervention_dimension: str | None = None

    @model_validator(mode="after")
    def verify(self):
        if self.left_manifest.artifact_kind != "comparison_manifest" or self.right_manifest.artifact_kind != "comparison_manifest":
            raise ValueError("COMPARISON_COMPATIBILITY_REQUIRES_MANIFESTS")
        if self.status == "COMPARABLE" and self.differing_dimensions:
            if (
                len(self.differing_dimensions) != 1
                or self.declared_intervention_dimension != self.differing_dimensions[0]
            ):
                raise ValueError("COMPARABLE_DIFFERENCE_MUST_EQUAL_DECLARED_INTERVENTION")
        if self.status == "INCOMPARABLE" and not self.differing_dimensions:
            raise ValueError("INCOMPARABLE_REQUIRES_DIFFERENCE")
        if self.causal_claim_permitted and (
            self.status != "COMPARABLE" or self.declared_intervention_dimension is None
        ):
            raise ValueError("CAUSAL_CLAIM_REQUIRES_EXPLICIT_COMPARABLE_INTERVENTION")
        digest = _digest(self.hash_payload())
        if (self.receipt_ref, self.receipt_hash) != (
            "comparison-compatibility:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("comparison compatibility identity mismatch")
        return self

    def hash_payload(self):
        return self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"})

    @classmethod
    def create(cls, **body):
        body = {
            **body,
            "differing_dimensions": tuple(sorted(
                set(body.get("differing_dimensions", ())),
                key=lambda v: v.encode("utf-8"),
            )),
        }
        provisional = cls.model_construct(
            receipt_ref="comparison-compatibility:v1:pending",
            receipt_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "receipt_ref": "comparison-compatibility:v1:" + digest[7:],
            "receipt_hash": digest,
        })
