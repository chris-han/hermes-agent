"""Derived non-authoritative runtime competency projections."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.decision_local_projection import DecisionContractV1
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import WorkingContext
from eos.artifact_hashing import content_hash

_SHA = r"^sha256:[0-9a-f]{64}$"


def _digest(value) -> str:
    return "sha256:" + content_hash(value)


def _pin_key(pin: ArtifactPinV1):
    return (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GovernedCompetencyBindingV1(_Frozen):
    """Bounded transport binding; not a registry or authority object."""
    competency_key: str = Field(min_length=1)
    competency_basis: ArtifactPinV1
    decision_contract: DecisionContractV1
    applicability: Literal["APPLICABLE", "NOT_APPLICABLE", "UNRESOLVED"]
    basis_refs: tuple[ArtifactPinV1, ...] = ()

    @field_validator("basis_refs", mode="before")
    @classmethod
    def canonicalize_basis(cls, value):
        pins = tuple(ArtifactPinV1.model_validate(x) for x in value)
        keys = tuple(_pin_key(x) for x in pins)
        if len(keys) != len(set(keys)):
            raise ValueError("runtime competency basis contains duplicate pins")
        return tuple(sorted(pins, key=_pin_key))


class RuntimeCompetencyProjectionV1(_Frozen):
    schema_version: Literal["runtime_competency_projection.v1"] = "runtime_competency_projection.v1"
    projection_ref: str
    projection_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    working_context_ref: str = Field(min_length=1)
    working_context_hash: str = Field(pattern=_SHA)
    competency_key: str = Field(min_length=1)
    competency_basis: ArtifactPinV1
    decision_contract_ref: str = Field(min_length=1)
    decision_contract_hash: str = Field(pattern=_SHA)
    applicability: Literal["APPLICABLE", "NOT_APPLICABLE", "UNRESOLVED"]
    basis_refs: tuple[ArtifactPinV1, ...] = ()
    derivation_policy_ref: Literal["runtime-competency-triage-policy:v1"] = "runtime-competency-triage-policy:v1"
    derivation_policy_hash: str = Field(pattern=_SHA)
    authority_class: Literal["DERIVED_NON_AUTHORITY"] = "DERIVED_NON_AUTHORITY"
    mutation_forbidden: Literal[True] = True

    @field_validator("basis_refs", mode="before")
    @classmethod
    def canonicalize_basis(cls, value):
        pins = tuple(ArtifactPinV1.model_validate(x) for x in value)
        keys = tuple(_pin_key(x) for x in pins)
        if len(keys) != len(set(keys)):
            raise ValueError("runtime competency basis contains duplicate pins")
        return tuple(sorted(pins, key=_pin_key))

    @model_validator(mode="after")
    def verify_identity(self):
        digest = _digest(self.hash_payload())
        if (self.projection_ref, self.projection_hash) != (
            "runtime-competency-projection:v1:" + digest[7:], digest
        ):
            raise ValueError("runtime competency projection identity mismatch")
        return self

    def hash_payload(self):
        return self.model_dump(mode="json", exclude={"projection_ref", "projection_hash"})

    @classmethod
    def create(cls, *, working_context: WorkingContext, binding: GovernedCompetencyBindingV1):
        policy_hash = _digest({
            "policy": "runtime-competency-triage",
            "version": "v1",
            "authority": "DERIVED_NON_AUTHORITY",
            "broadening": "forbidden",
        })
        body = dict(
            organization_id=working_context.organization_id,
            workspace_id=working_context.workspace_id,
            working_context_ref=working_context.value_ref,
            working_context_hash=working_context.value_hash,
            competency_key=binding.competency_key,
            competency_basis=binding.competency_basis,
            decision_contract_ref=binding.decision_contract.contract_ref,
            decision_contract_hash=binding.decision_contract.contract_hash,
            applicability=binding.applicability,
            basis_refs=binding.basis_refs,
            derivation_policy_hash=policy_hash,
        )
        draft = cls.model_construct(
            projection_ref="runtime-competency-projection:v1:pending",
            projection_hash="sha256:" + "0"*64,
            **body,
        )
        digest = _digest(draft.hash_payload())
        return cls.model_validate({
            **body,
            "projection_ref": "runtime-competency-projection:v1:" + digest[7:],
            "projection_hash": digest,
        })
