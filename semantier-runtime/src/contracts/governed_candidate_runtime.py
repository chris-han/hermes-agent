"""Derived W5 owner-admission and accepted-only runtime resolution contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.artifact_pin import ArtifactPinV1
from contracts.candidate_stage_binding import OwningStage
from eos.artifact_hashing import content_hash

_SHA = r"^sha256:[0-9a-f]{64}$"
AdmissionState = Literal["ADMITTED", "ACTIVE", "SUPERSEDED"]


def _digest(value) -> str:
    return "sha256:" + content_hash(value)


class CandidateAdmissionBindingV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["candidate_admission_binding.v1"] = "candidate_admission_binding.v1"
    binding_ref: str
    binding_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    candidate: ArtifactPinV1
    owning_stage: OwningStage
    owner_ref: str = Field(min_length=1)
    predecessor_basis: ArtifactPinV1
    successor_input: ArtifactPinV1
    owner_receipt: ArtifactPinV1
    state: AdmissionState
    predecessor_binding: ArtifactPinV1 | None = None
    successor_binding: ArtifactPinV1 | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def verify(self):
        if self.candidate.artifact_kind != "change_candidate":
            raise ValueError("ADMISSION_BINDING_REQUIRES_CHANGE_CANDIDATE")
        if self.owner_receipt.artifact_kind not in {
            "decision_record",
            "context_graph",
            "context_admission_receipt",
            "ontology_admission_receipt",
            "policy_admission_receipt",
            "decision_policy_admission_receipt",
            "commitment_policy_admission_receipt",
        }:
            raise ValueError("OWNER_ADMISSION_RECEIPT_REQUIRED")
        if self.predecessor_binding is not None and self.predecessor_binding.artifact_kind != "candidate_admission_binding":
            raise ValueError("INVALID_PREDECESSOR_BINDING_KIND")
        if self.successor_binding is not None and self.successor_binding.artifact_kind != "candidate_admission_binding":
            raise ValueError("INVALID_SUCCESSOR_BINDING_KIND")
        if self.state == "ADMITTED":
            if self.predecessor_binding is not None or self.successor_binding is not None:
                raise ValueError("ADMITTED_BINDING_CANNOT_HAVE_CHAIN_LINK")
        if self.state == "ACTIVE":
            if self.predecessor_binding is None:
                raise ValueError("ACTIVE_BINDING_REQUIRES_ADMITTED_PREDECESSOR")
            if self.successor_binding is not None:
                raise ValueError("ACTIVE_BINDING_CANNOT_HAVE_SUCCESSOR_BINDING")
        if self.state == "SUPERSEDED" and self.predecessor_binding is None:
            raise ValueError("TERMINAL_BINDING_REQUIRES_PREDECESSOR")
        if self.state == "SUPERSEDED" and self.successor_binding is None:
            raise ValueError("SUPERSEDED_BINDING_REQUIRES_SUCCESSOR")
        if self.state == "SUPERSEDED" and not self.reason:
            raise ValueError("TERMINAL_BINDING_REQUIRES_REASON")
        if self.state in {"ADMITTED", "ACTIVE"} and self.reason is not None:
            raise ValueError("NON_TERMINAL_BINDING_CANNOT_HAVE_REASON")
        digest = _digest(self.hash_payload())
        if (self.binding_ref, self.binding_hash) != (
            "candidate-admission-binding:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("candidate admission binding identity mismatch")
        return self

    def hash_payload(self):
        return self.model_dump(mode="json", exclude={"binding_ref", "binding_hash"})

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            binding_ref="candidate-admission-binding:v1:pending",
            binding_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "binding_ref": "candidate-admission-binding:v1:" + digest[7:],
            "binding_hash": digest,
        })

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="candidate_admission_binding",
            artifact_ref=self.binding_ref,
            artifact_hash=self.binding_hash,
        )


class RuntimeCandidateResolutionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runtime_candidate_resolution.v1"] = "runtime_candidate_resolution.v1"
    resolution_ref: str
    resolution_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    owning_stage: OwningStage
    owner_ref: str = Field(min_length=1)
    active_binding: ArtifactPinV1
    candidate: ArtifactPinV1
    successor_input: ArtifactPinV1

    @model_validator(mode="after")
    def verify(self):
        if self.active_binding.artifact_kind != "candidate_admission_binding":
            raise ValueError("RUNTIME_RESOLUTION_REQUIRES_ACTIVE_BINDING")
        if self.candidate.artifact_kind != "change_candidate":
            raise ValueError("RUNTIME_RESOLUTION_REQUIRES_CANDIDATE")
        digest = _digest(self.hash_payload())
        if (self.resolution_ref, self.resolution_hash) != (
            "runtime-candidate-resolution:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("runtime candidate resolution identity mismatch")
        return self

    def hash_payload(self):
        return self.model_dump(mode="json", exclude={"resolution_ref", "resolution_hash"})

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            resolution_ref="runtime-candidate-resolution:v1:pending",
            resolution_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "resolution_ref": "runtime-candidate-resolution:v1:" + digest[7:],
            "resolution_hash": digest,
        })
