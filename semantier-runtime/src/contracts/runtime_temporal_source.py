"""Closed contracts for durable native temporal source intake."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.canonical_source_ir import SourceLocator
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_context_reconstruction import CapturedJsonV1
from contracts.runtime_working_context import RuntimeWorldScopeV1
from eos.artifact_hashing import content_hash


_NonEmptyStr = Annotated[str, Field(strict=True, min_length=1)]
_Count = Annotated[int, Field(strict=True, ge=0)]
_Sha256 = Annotated[str, Field(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]


def _mapping(value: BaseModel | Mapping[str, object]) -> dict[str, object]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


def _basis(
    value: BaseModel | Mapping[str, object], ref_field: str, hash_field: str
) -> dict[str, object]:
    result = _mapping(value)
    result.pop(ref_field, None)
    result.pop(hash_field, None)
    return result


def runtime_temporal_source_archive_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    return _basis(value, "archive_ref", "archive_hash")


def runtime_temporal_normalization_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    return _basis(value, "normalization_ref", "normalization_hash")


def runtime_temporal_source_graph_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    return _basis(value, "graph_ref", "graph_hash")


def runtime_temporal_context_selection_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    return _basis(value, "selection_ref", "selection_hash")


class _SourceContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def require_utc(self) -> "_SourceContract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class _AddressedSource(_SourceContract):
    ARTIFACT_KIND: ClassVar[str]
    REF_FIELD: ClassVar[str]
    HASH_FIELD: ClassVar[str]
    REF_PREFIX: ClassVar[str]

    @classmethod
    def _hash_basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return _basis(value, cls.REF_FIELD, cls.HASH_FIELD)

    @model_validator(mode="after")
    def verify_address(self) -> "_AddressedSource":
        digest = content_hash(type(self)._hash_basis(self))
        if (
            getattr(self, self.REF_FIELD),
            getattr(self, self.HASH_FIELD),
        ) != (f"{self.REF_PREFIX}:{digest}", f"sha256:{digest}"):
            raise ValueError("addressed identity does not bind exact content")
        return self

    @classmethod
    def create(cls, **values: object):
        provisional = cls.model_construct(
            **values,
            **{
                cls.REF_FIELD: "pending",
                cls.HASH_FIELD: "sha256:" + "0" * 64,
            },
        )
        digest = content_hash(cls._hash_basis(provisional))
        return cls(
            **values,
            **{
                cls.REF_FIELD: f"{cls.REF_PREFIX}:{digest}",
                cls.HASH_FIELD: f"sha256:{digest}",
            },
        )

    def artifact_pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind=self.ARTIFACT_KIND,
            artifact_ref=getattr(self, self.REF_FIELD),
            artifact_hash=getattr(self, self.HASH_FIELD),
        )


class RuntimeTemporalSourceArchiveV1(_AddressedSource):
    ARTIFACT_KIND = "runtime_temporal_source_archive"
    REF_FIELD = "archive_ref"
    HASH_FIELD = "archive_hash"
    REF_PREFIX = "runtime-temporal-source-archive"

    schema_version: Literal["runtime_temporal_source_archive.v1"]
    archive_ref: _NonEmptyStr
    archive_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    session_ref: _NonEmptyStr
    media_type: Literal["application/x-ipynb+json"]
    content_base64: _NonEmptyStr
    content_hash: _Sha256
    size_bytes: _Count
    actor_id: _NonEmptyStr
    request_id: _NonEmptyStr
    request_hash: _Sha256
    recorded_at: datetime

    @classmethod
    def _hash_basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_temporal_source_archive_hash_basis(value)

    @model_validator(mode="after")
    def verify_content(self) -> "RuntimeTemporalSourceArchiveV1":
        try:
            decoded = base64.b64decode(self.content_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("content_base64 must be strict standard base64") from exc
        import hashlib

        if len(decoded) != self.size_bytes:
            raise ValueError("size_bytes must equal decoded content length")
        if self.content_hash != "sha256:" + hashlib.sha256(decoded).hexdigest():
            raise ValueError("content_hash must bind exact archived bytes")
        return self


class RuntimeTemporalRowProvenanceV1(_SourceContract):
    collection: Literal["entities", "relationships"]
    row_index: _Count
    object_id: _NonEmptyStr
    locator: SourceLocator
    original_row: CapturedJsonV1
    original_temporal_fields: CapturedJsonV1


class RuntimeTemporalNormalizationV1(_AddressedSource):
    ARTIFACT_KIND = "runtime_temporal_normalization"
    REF_FIELD = "normalization_ref"
    HASH_FIELD = "normalization_hash"
    REF_PREFIX = "runtime-temporal-normalization"

    schema_version: Literal["runtime_temporal_normalization.v1"]
    normalization_ref: _NonEmptyStr
    normalization_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    source: ArtifactPinV1
    source_cell_index: _Count
    source_cell_hash: _Sha256
    normalizer_id: Literal["semantier.notebook_temporal.v1"]
    implementation_hash: _Sha256
    rows: tuple[RuntimeTemporalRowProvenanceV1, ...]

    @classmethod
    def _hash_basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_temporal_normalization_hash_basis(value)

    @field_validator("source")
    @classmethod
    def source_is_archive(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        if value.artifact_kind != "runtime_temporal_source_archive":
            raise ValueError("source must pin runtime_temporal_source_archive")
        return value


class RuntimeTemporalEntityV1(_SourceContract):
    id: _NonEmptyStr
    properties: CapturedJsonV1


class RuntimeTemporalRelationshipV1(_SourceContract):
    id: _NonEmptyStr
    source: _NonEmptyStr
    target: _NonEmptyStr
    type: _NonEmptyStr
    valid_from: datetime
    valid_until: datetime
    timestamp: datetime | None
    properties: CapturedJsonV1

    @model_validator(mode="after")
    def valid_interval(self) -> "RuntimeTemporalRelationshipV1":
        if self.valid_from > self.valid_until:
            raise ValueError("valid_from must be before or equal to valid_until")
        if self.timestamp is not None and not (
            self.timestamp == self.valid_from == self.valid_until
        ):
            raise ValueError("timestamp requires equal interval bounds")
        return self


class RuntimeTemporalSourceGraphV1(_AddressedSource):
    ARTIFACT_KIND = "runtime_temporal_source_graph"
    REF_FIELD = "graph_ref"
    HASH_FIELD = "graph_hash"
    REF_PREFIX = "runtime-temporal-source-graph"

    schema_version: Literal["runtime_temporal_source_graph.v1"]
    graph_ref: _NonEmptyStr
    graph_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    session_ref: _NonEmptyStr
    source: ArtifactPinV1
    normalization: ArtifactPinV1
    ontology: ArtifactPinV1
    evidence: tuple[ArtifactPinV1, ...]
    entities: tuple[RuntimeTemporalEntityV1, ...]
    relationships: tuple[RuntimeTemporalRelationshipV1, ...]
    actor_id: _NonEmptyStr
    request_id: _NonEmptyStr
    request_hash: _Sha256
    recorded_at: datetime

    @classmethod
    def _hash_basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_temporal_source_graph_hash_basis(value)

    @model_validator(mode="after")
    def validate_graph(self) -> "RuntimeTemporalSourceGraphV1":
        if self.source.artifact_kind != "runtime_temporal_source_archive":
            raise ValueError("source must pin runtime_temporal_source_archive")
        if self.normalization.artifact_kind != "runtime_temporal_normalization":
            raise ValueError("normalization must pin runtime_temporal_normalization")
        if self.ontology.artifact_kind != "ontology":
            raise ValueError("ontology must pin kernel ontology")
        if not self.evidence or any(pin.artifact_kind != "evidence" for pin in self.evidence):
            raise ValueError("evidence must contain kernel Evidence pins")
        evidence_keys = [
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
            for pin in self.evidence
        ]
        if evidence_keys != sorted(set(evidence_keys)):
            raise ValueError("evidence pins must be sorted unique")
        entity_ids = [row.id for row in self.entities]
        relationship_ids = [row.id for row in self.relationships]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity ids must be unique")
        if len(relationship_ids) != len(set(relationship_ids)):
            raise ValueError("relationship ids must be unique")
        known = set(entity_ids)
        if any(
            row.source not in known or row.target not in known
            for row in self.relationships
        ):
            raise ValueError("relationship endpoint is missing")
        return self


class RuntimeTemporalContextSelectionV1(_AddressedSource):
    ARTIFACT_KIND = "runtime_temporal_context_selection"
    REF_FIELD = "selection_ref"
    HASH_FIELD = "selection_hash"
    REF_PREFIX = "runtime-temporal-context-selection"

    schema_version: Literal["runtime_temporal_context_selection.v1"]
    selection_ref: _NonEmptyStr
    selection_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    profile_id: Literal["explicit_temporal_selection_v1"]
    actor_id: _NonEmptyStr
    session_ref: _NonEmptyStr
    request_id: _NonEmptyStr
    goal: ArtifactPinV1
    source_graph: ArtifactPinV1
    world: RuntimeWorldScopeV1
    semantic_pins: tuple[ArtifactPinV1, ...] = ()
    selected_entity_ids: tuple[_NonEmptyStr, ...]

    @classmethod
    def _hash_basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_temporal_context_selection_hash_basis(value)

    @field_validator("selected_entity_ids")
    @classmethod
    def canonical_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("selected_entity_ids must be unique")
        return tuple(sorted(value))

    @field_validator("semantic_pins")
    @classmethod
    def canonical_semantic_pins(
        cls, value: tuple[ArtifactPinV1, ...]
    ) -> tuple[ArtifactPinV1, ...]:
        keys = [(pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) for pin in value]
        if len(keys) != len(set(keys)):
            raise ValueError("semantic_pins must be unique")
        return tuple(sorted(value, key=lambda pin: (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)))

    @model_validator(mode="after")
    def validate_closure(self) -> "RuntimeTemporalContextSelectionV1":
        if self.goal.artifact_kind != "graph_intent_brief":
            raise ValueError("goal must pin graph_intent_brief")
        if self.source_graph.artifact_kind != "runtime_temporal_source_graph":
            raise ValueError("source_graph must pin runtime_temporal_source_graph")
        if self.world.source_graph != self.source_graph:
            raise ValueError("world must pin the selected source graph")
        return self


RUNTIME_TEMPORAL_SOURCE_MODEL_BY_SCHEMA = MappingProxyType(
    {
        "runtime_temporal_source_archive.v1": RuntimeTemporalSourceArchiveV1,
        "runtime_temporal_normalization.v1": RuntimeTemporalNormalizationV1,
        "runtime_temporal_source_graph.v1": RuntimeTemporalSourceGraphV1,
        "runtime_temporal_context_selection.v1": RuntimeTemporalContextSelectionV1,
    }
)


__all__ = [
    "RuntimeTemporalContextSelectionV1",
    "RuntimeTemporalEntityV1",
    "RuntimeTemporalNormalizationV1",
    "RuntimeTemporalRelationshipV1",
    "RuntimeTemporalRowProvenanceV1",
    "RuntimeTemporalSourceArchiveV1",
    "RuntimeTemporalSourceGraphV1",
    "RUNTIME_TEMPORAL_SOURCE_MODEL_BY_SCHEMA",
    "runtime_temporal_context_selection_hash_basis",
    "runtime_temporal_normalization_hash_basis",
    "runtime_temporal_source_archive_hash_basis",
    "runtime_temporal_source_graph_hash_basis",
]
