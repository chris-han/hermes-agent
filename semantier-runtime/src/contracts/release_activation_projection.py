"""Derived release/activation projections over owner-native authority receipts.

These contracts compose exact owner-native pins. They do not admit, release,
activate, or replace any owner-native store/receipt.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import _ContentAddressed, _ordered_unique_pins


class _ProjectionPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _exact_pin(pin: ArtifactPinV1, field: str) -> ArtifactPinV1:
    if pin.artifact_ref.strip().lower() == "latest":
        raise ValueError(f"{field} cannot use latest")
    return pin


def _exact_pins(value: object, field: str, *, allow_empty: bool = False) -> tuple[ArtifactPinV1, ...]:
    pins = _ordered_unique_pins(value)
    if not isinstance(pins, tuple):
        return pins
    if not pins and not allow_empty:
        raise ValueError(f"{field} must be non-empty")
    if any(pin.artifact_ref.strip().lower() == "latest" for pin in pins):
        raise ValueError(f"{field} cannot use latest")
    return pins


class ReleaseSetMemberV1(_ProjectionPart):
    owner_ref: str = Field(min_length=1)
    candidate: ArtifactPinV1
    admission_receipt: ArtifactPinV1
    owner_release_receipt: ArtifactPinV1
    carried_forward_from: ArtifactPinV1 | None = None

    @model_validator(mode="after")
    def exact_member(self) -> "ReleaseSetMemberV1":
        if self.candidate.artifact_kind != "change_candidate":
            raise ValueError("release member candidate must pin change_candidate")
        for name in ("candidate", "admission_receipt", "owner_release_receipt"):
            _exact_pin(getattr(self, name), name)
        if self.carried_forward_from is not None:
            _exact_pin(self.carried_forward_from, "carried_forward_from")
        return self


class ReleaseSetManifestV1(_ContentAddressed):
    """Immutable composition manifest; publication is not activation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    ARTIFACT_KIND = "release_set_manifest"

    schema_version: Literal["release_set_manifest.v1"] = "release_set_manifest.v1"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    release_set_ref: str = Field(min_length=1)
    parent_release_set: ArtifactPinV1 | None = None
    members: tuple[ReleaseSetMemberV1, ...]
    compatibility_refs: tuple[ArtifactPinV1, ...] = ()
    effective_from: datetime
    effective_to: datetime | None = None
    published_at: datetime

    @field_validator("members", mode="before")
    @classmethod
    def canonicalize_members(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        members = tuple(
            item if isinstance(item, ReleaseSetMemberV1) else ReleaseSetMemberV1.model_validate(item)
            for item in value
        )
        if not members:
            raise ValueError("members must be non-empty")
        keys = tuple(
            (
                item.owner_ref,
                item.candidate.artifact_ref,
                item.candidate.artifact_hash,
            )
            for item in members
        )
        if len(keys) != len(set(keys)):
            raise ValueError("release set members must be unique")
        return tuple(
            sorted(
                members,
                key=lambda item: (
                    item.owner_ref,
                    item.candidate.artifact_ref,
                    item.candidate.artifact_hash,
                ),
            )
        )

    @field_validator("compatibility_refs", mode="before")
    @classmethod
    def exact_compatibility_refs(cls, value: object) -> object:
        return _exact_pins(value, "compatibility_refs", allow_empty=True)

    @model_validator(mode="after")
    def validate_manifest(self) -> "ReleaseSetManifestV1":
        if self.parent_release_set is not None:
            if self.parent_release_set.artifact_kind != "release_set_manifest":
                raise ValueError("parent_release_set must pin release_set_manifest")
            _exact_pin(self.parent_release_set, "parent_release_set")
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("release effectivity interval is invalid")
        return self


class ActivationOwnerBindingV1(_ProjectionPart):
    owner_ref: str = Field(min_length=1)
    activation_receipt: ArtifactPinV1
    effective_scope: ArtifactPinV1
    runtime_binding_refs: tuple[ArtifactPinV1, ...] = ()
    context_binding_refs: tuple[ArtifactPinV1, ...] = ()

    @field_validator("runtime_binding_refs", "context_binding_refs", mode="before")
    @classmethod
    def exact_optional_refs(cls, value: object, info):
        return _exact_pins(value, info.field_name, allow_empty=True)

    @model_validator(mode="after")
    def exact_binding(self) -> "ActivationOwnerBindingV1":
        _exact_pin(self.activation_receipt, "activation_receipt")
        _exact_pin(self.effective_scope, "effective_scope")
        return self


class ActivationBindingProjectionV1(_ContentAddressed):
    """Derived view over real owner activation receipts."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    ARTIFACT_KIND = "activation_binding_projection"

    schema_version: Literal["activation_binding_projection.v1"] = (
        "activation_binding_projection.v1"
    )
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    release_set: ArtifactPinV1
    owner_bindings: tuple[ActivationOwnerBindingV1, ...]
    release_effective_from: datetime
    release_effective_to: datetime | None = None
    effective_from: datetime
    effective_to: datetime | None = None
    projected_at: datetime
    predecessor_projection: ArtifactPinV1 | None = None
    projection_reason: str | None = None

    @field_validator("owner_bindings", mode="before")
    @classmethod
    def canonicalize_owner_bindings(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        bindings = tuple(
            item if isinstance(item, ActivationOwnerBindingV1) else ActivationOwnerBindingV1.model_validate(item)
            for item in value
        )
        if not bindings:
            raise ValueError("owner_bindings must be non-empty")
        keys = tuple(
            (item.owner_ref, item.activation_receipt.artifact_ref, item.activation_receipt.artifact_hash)
            for item in bindings
        )
        if len(keys) != len(set(keys)):
            raise ValueError("owner activation bindings must be unique")
        return tuple(sorted(bindings, key=lambda item: (item.owner_ref, item.activation_receipt.artifact_ref)))

    @model_validator(mode="after")
    def validate_projection(self) -> "ActivationBindingProjectionV1":
        if self.release_set.artifact_kind != "release_set_manifest":
            raise ValueError("release_set must pin release_set_manifest")
        _exact_pin(self.release_set, "release_set")
        if self.predecessor_projection is not None:
            if self.predecessor_projection.artifact_kind != "activation_binding_projection":
                raise ValueError("predecessor_projection must pin activation_binding_projection")
            _exact_pin(self.predecessor_projection, "predecessor_projection")
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("activation effectivity interval is invalid")
        if self.release_effective_to is not None and self.release_effective_to <= self.release_effective_from:
            raise ValueError("release effectivity interval is invalid")
        if self.effective_from < self.release_effective_from:
            raise ValueError("activation is outside release effectivity scope")
        if self.release_effective_to is not None and (
            self.effective_from >= self.release_effective_to
            or (self.effective_to is not None and self.effective_to > self.release_effective_to)
        ):
            raise ValueError("activation is outside release effectivity scope")
        if self.predecessor_projection is not None and not self.projection_reason:
            raise ValueError("projection revision requires projection_reason")
        return self


__all__ = [
    "ActivationBindingProjectionV1",
    "ActivationOwnerBindingV1",
    "ReleaseSetManifestV1",
    "ReleaseSetMemberV1",
]
