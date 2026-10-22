"""Comparable extraction result across a validated source-structure successor."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eos.artifact_hashing import content_hash


_SHA = r"^sha256:[0-9a-f]{64}$"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _SourceBinding(_FrozenModel):
    organization_id: str
    workspace_id: str
    lineage_id: str
    source_identity_ref: str
    source_hash: str = Field(pattern=_SHA)
    source_version: str | None
    canonical_source_ir_ref: str
    canonical_source_ir_hash: str = Field(pattern=_SHA)

    @field_validator(
        "organization_id",
        "workspace_id",
        "lineage_id",
        "source_identity_ref",
        "canonical_source_ir_ref",
        "source_version",
    )
    @classmethod
    def ascii_values(cls, value: str | None, info) -> str | None:
        if value is not None and (not value or not value.isascii()):
            raise ValueError(f"{info.field_name} must be null or non-empty ASCII")
        return value


class _SourceBindings(_FrozenModel):
    before: _SourceBinding
    after: _SourceBinding


class _GraphIntentBinding(_FrozenModel):
    graph_intent_ref: str
    graph_intent_hash: str = Field(pattern=_SHA)
    review_state: Literal["accepted", "edited"]
    source_identity_ref: str

    @field_validator("graph_intent_ref", "source_identity_ref")
    @classmethod
    def ascii_ref(cls, value: str) -> str:
        if not value or not value.isascii():
            raise ValueError("graph_intent_ref must be non-empty ASCII")
        return value


class _GraphIntentBindings(_FrozenModel):
    before: _GraphIntentBinding
    after: _GraphIntentBinding


class _CountDelta(_FrozenModel):
    before: int = Field(ge=0)
    after: int = Field(ge=0)


class _Metrics(_FrozenModel):
    nodes: _CountDelta
    edges: _CountDelta
    events: _CountDelta
    frames: _CountDelta


class _OutputHashDelta(_FrozenModel):
    before: str = Field(pattern=_SHA)
    after: str = Field(pattern=_SHA)
    changed: bool

    @model_validator(mode="after")
    def verify_change(self) -> "_OutputHashDelta":
        if self.changed != (self.before != self.after):
            raise ValueError("output hash changed flag mismatch")
        return self


class _OutputHashes(_FrozenModel):
    nodes: _OutputHashDelta
    edges: _OutputHashDelta
    events: _OutputHashDelta
    frames: _OutputHashDelta


class SourceStructureRetestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["source_structure_retest.v1"]
    comparison_id: str = Field(pattern=r"^source-structure-retest-[0-9a-f]{24}$")
    comparison_hash: str = Field(pattern=_SHA)
    semantic_validation_ref: str
    semantic_validation_hash: str = Field(pattern=_SHA)
    baseline_schema_decision_ref: str
    baseline_schema_decision_hash: str = Field(pattern=_SHA)
    successor_schema_decision_ref: str
    successor_schema_decision_hash: str = Field(pattern=_SHA)
    source_bindings: _SourceBindings
    baseline_extract_ref: str
    baseline_extract_hash: str = Field(pattern=_SHA)
    successor_extract_ref: str
    successor_extract_hash: str = Field(pattern=_SHA)
    graph_intent_bindings: _GraphIntentBindings
    metrics: _Metrics
    evidence_binding_count: _CountDelta
    output_hashes: _OutputHashes

    @field_validator(
        "semantic_validation_ref",
        "baseline_schema_decision_ref",
        "successor_schema_decision_ref",
        "baseline_extract_ref",
        "successor_extract_ref",
    )
    @classmethod
    def ascii_refs(cls, value: str, info) -> str:
        if not value or not value.isascii():
            raise ValueError(f"{info.field_name} must be non-empty ASCII")
        return value

    @model_validator(mode="after")
    def verify_identity(self) -> "SourceStructureRetestV1":
        before = self.source_bindings.before
        after = self.source_bindings.after
        if (
            before.organization_id != after.organization_id
            or before.workspace_id != after.workspace_id
            or before.lineage_id != after.lineage_id
            or before.source_hash != after.source_hash
            or before.source_identity_ref == after.source_identity_ref
            or before.canonical_source_ir_hash == after.canonical_source_ir_hash
            or self.graph_intent_bindings.before.graph_intent_ref
            == self.graph_intent_bindings.after.graph_intent_ref
            or self.graph_intent_bindings.before.source_identity_ref
            != before.source_identity_ref
            or self.graph_intent_bindings.after.source_identity_ref
            != after.source_identity_ref
        ):
            raise ValueError("source-structure retest source lineage mismatch")
        if (
            self.baseline_schema_decision_ref == self.successor_schema_decision_ref
            or self.baseline_extract_ref == self.successor_extract_ref
        ):
            raise ValueError("source-structure retest requires decision/extract successors")
        digest = "sha256:" + content_hash(self.hash_payload())
        if self.comparison_hash != digest:
            raise ValueError("source-structure retest hash mismatch")
        if self.comparison_id != "source-structure-retest-" + digest[7:31]:
            raise ValueError("source-structure retest identity mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("comparison_id", None)
        payload.pop("comparison_hash", None)
        return payload

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "SourceStructureRetestV1":
        digest = "sha256:" + content_hash(payload)
        return cls.model_validate(
            {
                **payload,
                "comparison_id": "source-structure-retest-" + digest[7:31],
                "comparison_hash": digest,
            }
        )


__all__ = ["SourceStructureRetestV1"]
