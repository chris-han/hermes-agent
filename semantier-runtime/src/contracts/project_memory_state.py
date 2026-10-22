"""Derived project-memory state contracts.

These artifacts are advisory/rebuildable projections over existing
MemoryCandidateV1 lineage. They never own evidence, source identity, time, or
institutional authority.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"

EpistemicState = Literal["OBSERVED", "CLAIMED", "INFERRED"]
AuthorityState = Literal["PROPOSED", "UNAPPROVED", "APPROVED", "ACTIVATED", "UNKNOWN"]
ExecutionState = Literal["NOT_RUN", "RUNNING", "SUCCEEDED", "FAILED", "UNKNOWN"]
VerificationState = Literal["NOT_RUN", "PASSED", "FAILED", "UNKNOWN"]
EffectivityState = Literal["ACTIVE", "HISTORICAL", "SUPERSEDED", "FUTURE", "UNKNOWN"]
StateDimension = Literal["epistemic", "authority", "execution", "verification"]
DerivationMethod = Literal[
    "EXACT_STRUCTURED_FIELD",
    "EXPLICIT_SOURCE_TOKEN",
    "SOURCE_ROLE_CONSTRAINT",
    "BOUNDED_TYPED_DISCRIMINATOR",
    "UNRESOLVED",
]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _ascii(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise ValueError(f"{field} must be a non-empty ASCII-stable identifier")
    return value


def _sorted_unique_ascii(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    items = tuple(_ascii(str(item), f"{field}[]") for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{field} contains duplicate identities")
    return tuple(sorted(items))


def _hash_payload(model: BaseModel, *identity_fields: str) -> dict[str, Any]:
    payload = model.model_dump(mode="json")
    for field in identity_fields:
        payload.pop(field, None)
    return payload


def _require_pin_kind(pin: ArtifactPinV1, expected: str, field: str) -> ArtifactPinV1:
    if pin.artifact_kind != expected:
        raise ValueError(f"{field} must pin artifact kind {expected}")
    return pin


class ProjectStateDerivationV1(_Contract):
    dimension: StateDimension
    method: DerivationMethod
    source_locator_refs: tuple[str, ...] = ()
    decision_receipt: ArtifactPinV1 | None = None
    confidence_ppm: int | None = Field(default=None, ge=0, le=1_000_000)

    @field_validator("source_locator_refs", mode="before")
    @classmethod
    def canonical_locators(cls, value: Any) -> tuple[str, ...]:
        return _sorted_unique_ascii(value, "source_locator_refs")

    @model_validator(mode="after")
    def validate_method_receipt(self) -> "ProjectStateDerivationV1":
        if self.method == "BOUNDED_TYPED_DISCRIMINATOR" and self.decision_receipt is None:
            raise ValueError("bounded discriminator derivation requires decision_receipt")
        if self.method != "BOUNDED_TYPED_DISCRIMINATOR" and self.decision_receipt is not None:
            raise ValueError("decision_receipt is only valid for bounded discriminator derivation")
        if self.method == "UNRESOLVED" and self.confidence_ppm is not None:
            raise ValueError("unresolved derivation cannot carry confidence")
        return self


class TypedStateDecisionV1(_Contract):
    schema_version: Literal["typed_state_decision.v1"] = "typed_state_decision.v1"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    decision_ref: str
    decision_hash: str = Field(pattern=SHA256_PATTERN)
    dimension: StateDimension
    selected_state: str
    abstained: bool
    confidence_ppm: int | None = Field(default=None, ge=0, le=1_000_000)
    provider_ref: str
    model_ref: str
    profile_ref: str
    input_hash: str = Field(pattern=SHA256_PATTERN)
    frontier_hash: str = Field(pattern=SHA256_PATTERN)

    @field_validator("provider_ref", "model_ref", "profile_ref")
    @classmethod
    def stable_refs(cls, value: str, info) -> str:
        return _ascii(value, info.field_name)

    @model_validator(mode="after")
    def validate_state_and_identity(self) -> "TypedStateDecisionV1":
        allowed = {
            "epistemic": {"OBSERVED", "CLAIMED", "INFERRED"},
            "authority": {"PROPOSED", "UNAPPROVED", "APPROVED", "ACTIVATED", "UNKNOWN"},
            "execution": {"NOT_RUN", "RUNNING", "SUCCEEDED", "FAILED", "UNKNOWN"},
            "verification": {"NOT_RUN", "PASSED", "FAILED", "UNKNOWN"},
        }[self.dimension]
        if self.selected_state not in allowed:
            raise ValueError("selected_state is outside the typed frontier")
        if self.abstained != (self.selected_state == "UNKNOWN"):
            raise ValueError("abstained must match UNKNOWN selection")
        expected = "sha256:" + content_hash(_hash_payload(self, "decision_ref", "decision_hash"))
        if self.decision_hash != expected or self.decision_ref != "typed-state:" + expected[7:31]:
            raise ValueError("typed state decision identity mismatch")
        return self


class ProjectStateProjectionV1(_Contract):
    schema_version: Literal["project_state_projection.v1"] = "project_state_projection.v1"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    projection_ref: str
    projection_hash: str = Field(pattern=SHA256_PATTERN)
    memory_candidate: ArtifactPinV1
    epistemic_state: EpistemicState
    authority_state: AuthorityState | None = None
    execution_state: ExecutionState | None = None
    verification_state: VerificationState | None = None
    derivations: tuple[ProjectStateDerivationV1, ...]
    semantic_subject_refs: tuple[str, ...] = ()
    semantic_predicate_refs: tuple[str, ...] = ()
    discriminator_receipts: tuple[ArtifactPinV1, ...] = ()
    diagnostics: tuple[str, ...] = ()
    projection_version: Literal["project-memory-compiler.v1"] = "project-memory-compiler.v1"

    @field_validator("memory_candidate")
    @classmethod
    def candidate_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "memory_candidate", "memory_candidate")

    @field_validator(
        "semantic_subject_refs",
        "semantic_predicate_refs",
        "diagnostics",
        mode="before",
    )
    @classmethod
    def canonical_refs(cls, value: Any, info) -> tuple[str, ...]:
        return _sorted_unique_ascii(value, info.field_name)

    @field_validator("discriminator_receipts", mode="before")
    @classmethod
    def canonical_receipts(cls, value: Any) -> tuple[ArtifactPinV1, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("discriminator_receipts must be an array")
        items = tuple(
            item if isinstance(item, ArtifactPinV1) else ArtifactPinV1.model_validate(item)
            for item in value
        )
        keys = tuple((item.artifact_kind, item.artifact_ref, item.artifact_hash) for item in items)
        if len(keys) != len(set(keys)):
            raise ValueError("discriminator_receipts contains duplicates")
        for item in items:
            _require_pin_kind(item, "typed_state_decision", "discriminator_receipts[]")
        return tuple(sorted(items, key=lambda item: (item.artifact_ref, item.artifact_hash)))

    @model_validator(mode="after")
    def validate_derivations_and_identity(self) -> "ProjectStateProjectionV1":
        derivations = {item.dimension: item for item in self.derivations}
        if len(derivations) != len(self.derivations):
            raise ValueError("projection contains duplicate dimension derivations")
        if "epistemic" not in derivations:
            raise ValueError("epistemic derivation is required")

        states = {
            "epistemic": self.epistemic_state,
            "authority": self.authority_state,
            "execution": self.execution_state,
            "verification": self.verification_state,
        }
        for dimension, state in states.items():
            derivation = derivations.get(dimension)
            if state is None:
                if derivation is not None and derivation.method != "UNRESOLVED":
                    raise ValueError("non-null derivation cannot exist for non-applicable dimension")
            elif derivation is None:
                raise ValueError(f"{dimension} state requires a derivation")

        receipt_pins = {
            (pin.artifact_ref, pin.artifact_hash) for pin in self.discriminator_receipts
        }
        for derivation in self.derivations:
            if derivation.decision_receipt is not None:
                key = (
                    derivation.decision_receipt.artifact_ref,
                    derivation.decision_receipt.artifact_hash,
                )
                if key not in receipt_pins:
                    raise ValueError("derivation decision receipt must be retained by projection")

        expected = "sha256:" + content_hash(_hash_payload(self, "projection_ref", "projection_hash"))
        if self.projection_hash != expected or self.projection_ref != "project-state:" + expected[7:31]:
            raise ValueError("project state projection identity mismatch")
        return self


class EffectivityViewV1(_Contract):
    schema_version: Literal["effectivity_view.v1"] = "effectivity_view.v1"
    projection: ArtifactPinV1
    as_of: datetime
    state: EffectivityState
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    superseded: bool = False
    retracted: bool = False
    supported: bool = True

    @field_validator("projection")
    @classmethod
    def projection_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "project_state_projection", "projection")


def bind_typed_state_decision_identity(payload: dict[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value.pop("decision_ref", None)
    value.pop("decision_hash", None)
    digest = content_hash(value)
    value["decision_ref"] = "typed-state:" + digest[:24]
    value["decision_hash"] = "sha256:" + digest
    return value


def bind_project_state_projection_identity(payload: dict[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value.pop("projection_ref", None)
    value.pop("projection_hash", None)
    value.setdefault("schema_version", "project_state_projection.v1")
    value.setdefault("authority_state", None)
    value.setdefault("execution_state", None)
    value.setdefault("verification_state", None)
    value.setdefault("semantic_subject_refs", [])
    value.setdefault("semantic_predicate_refs", [])
    value.setdefault("discriminator_receipts", [])
    value.setdefault("diagnostics", [])
    value.setdefault("projection_version", "project-memory-compiler.v1")
    value["derivations"] = [
        ProjectStateDerivationV1.model_validate(item).model_dump(mode="json")
        for item in value.get("derivations", ())
    ]
    digest = content_hash(value)
    value["projection_ref"] = "project-state:" + digest[:24]
    value["projection_hash"] = "sha256:" + digest
    return value
