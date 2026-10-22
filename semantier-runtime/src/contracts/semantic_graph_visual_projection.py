"""Deterministic renderer-independent semantic graph projection contract."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _ascii(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise ValueError(f"{field} must be a non-empty ASCII-stable identifier")
    return value


def _sorted_unique(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    items = tuple(_ascii(item, f"{field}[]") for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{field} contains a duplicate identity")
    return tuple(sorted(items, key=lambda item: item.encode("utf-8")))


def _sort_objects(value: Any, field: str, identity: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    identities = [
        item.get(identity) if isinstance(item, dict) else getattr(item, identity)
        for item in value
    ]
    for item in identities:
        _ascii(item, f"{field}[].{identity}")
    if len(identities) != len(set(identities)):
        raise ValueError(f"{field} contains a duplicate {identity}")
    return tuple(
        item for _, item in sorted(
            zip(identities, value), key=lambda pair: pair[0].encode("utf-8")
        )
    )


class SemanticGraphProjectionNodeV1(_Contract):
    id: str
    label: str
    object_type_id: str
    role_id: str | None
    x: float
    y: float
    z_semantic: float | None
    cluster_id: str
    importance: float
    properties: dict[str, Any]

    @field_validator("id", "object_type_id", "cluster_id")
    @classmethod
    def stable_ids(cls, value: str, info) -> str:
        return _ascii(value, info.field_name)

    @field_validator("role_id")
    @classmethod
    def optional_stable_id(cls, value: str | None) -> str | None:
        return None if value is None else _ascii(value, "role_id")

    @field_validator("x", "y", "z_semantic", "importance")
    @classmethod
    def finite_numbers(cls, value: float | None, info) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite")
        return value


class SemanticGraphProjectionEdgeV1(_Contract):
    id: str
    relation_instance_ref: str
    relation_instance_hash: str = Field(pattern=SHA256_PATTERN)
    source: str
    target: str
    relation_type_id: str
    relation_family_id: str
    label: str
    representation_kind: Literal["binary_edge", "frame_role", "derived_projection"]
    frame_ref: str | None
    projection_source_refs: tuple[str, ...]
    evidence_refs: tuple[dict[str, Any], ...]
    properties: dict[str, Any]

    @field_validator(
        "id", "relation_instance_ref", "source", "target", "relation_type_id",
        "relation_family_id",
    )
    @classmethod
    def stable_ids(cls, value: str, info) -> str:
        return _ascii(value, info.field_name)

    @field_validator("frame_ref")
    @classmethod
    def optional_stable_ref(cls, value: str | None) -> str | None:
        return None if value is None else _ascii(value, "frame_ref")

    @field_validator("projection_source_refs", mode="before")
    @classmethod
    def canonical_projection_sources(cls, value: Any) -> tuple[str, ...]:
        return _sorted_unique(value, "projection_source_refs")


class RelationFamilyCatalogEntryV1(_Contract):
    family_id: str
    label: str

    @field_validator("family_id")
    @classmethod
    def stable_id(cls, value: str) -> str:
        return _ascii(value, "family_id")


class RelationTypeCatalogEntryV1(_Contract):
    relation_type_id: str
    family_id: str
    label: str

    @field_validator("relation_type_id", "family_id")
    @classmethod
    def stable_ids(cls, value: str, info) -> str:
        return _ascii(value, info.field_name)


class SemanticGraphRelationCatalogV1(_Contract):
    families: tuple[RelationFamilyCatalogEntryV1, ...]
    relation_types: tuple[RelationTypeCatalogEntryV1, ...]
    counts_by_family: dict[str, int]
    counts_by_type: dict[str, int]

    @model_validator(mode="before")
    @classmethod
    def canonicalize(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        value = dict(payload)
        value["families"] = _sort_objects(value.get("families", ()), "families", "family_id")
        value["relation_types"] = _sort_objects(
            value.get("relation_types", ()), "relation_types", "relation_type_id"
        )
        value["counts_by_family"] = dict(sorted(value.get("counts_by_family", {}).items()))
        value["counts_by_type"] = dict(sorted(value.get("counts_by_type", {}).items()))
        return value


class SemanticGraphVisualProjectionV1(_Contract):
    schema_version: Literal["semantic_graph_visual_projection.v1"] = (
        "semantic_graph_visual_projection.v1"
    )
    projection_id: str
    projection_version: Literal[1] = 1
    projection_hash: str = Field(pattern=SHA256_PATTERN)
    graph_release_ref: str
    graph_release_hash: str = Field(pattern=SHA256_PATTERN)
    taxonomy_ref: str
    taxonomy_hash: str = Field(pattern=SHA256_PATTERN)
    profile_ref: str | None
    profile_hash: str | None = Field(pattern=SHA256_PATTERN)
    layout_profile_ref: str
    layout_profile_hash: str = Field(pattern=SHA256_PATTERN)
    layout_engine_version: str
    coordinate_seed: str
    coordinate_hash: str = Field(pattern=SHA256_PATTERN)
    plane: Literal["knowledge"]
    diagnostics: dict[str, Any]
    nodes: tuple[SemanticGraphProjectionNodeV1, ...]
    edges: tuple[SemanticGraphProjectionEdgeV1, ...]
    relation_catalog: SemanticGraphRelationCatalogV1

    @model_validator(mode="before")
    @classmethod
    def canonicalize(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        value = dict(payload)
        value["nodes"] = _sort_objects(value.get("nodes", ()), "nodes", "id")
        value["edges"] = _sort_objects(
            value.get("edges", ()), "edges", "relation_instance_ref"
        )
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "SemanticGraphVisualProjectionV1":
        for field in (
            "projection_id", "graph_release_ref", "taxonomy_ref", "layout_profile_ref",
            "layout_engine_version", "coordinate_seed",
        ):
            _ascii(getattr(self, field), field)
        if (self.profile_ref is None) != (self.profile_hash is None):
            raise ValueError("profile_ref and profile_hash must both be null or non-null")
        expected_coordinates = "sha256:" + content_hash(_coordinate_payload(self.nodes))
        if self.coordinate_hash != expected_coordinates:
            raise ValueError("coordinate_hash does not match canonical coordinates")
        expected_projection = "sha256:" + content_hash(self._hash_payload())
        if self.projection_hash != expected_projection:
            raise ValueError("projection_hash does not match canonical projection content")
        if self.projection_id != "sgvp_" + expected_projection[7:31]:
            raise ValueError("projection_id does not match projection_hash")
        return self

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("projection_id")
        payload.pop("projection_hash")
        return payload

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _coordinate_payload(nodes: Any) -> list[dict[str, Any]]:
    return [
        {"id": node.id, "x": node.x, "y": node.y, "z_semantic": node.z_semantic}
        for node in nodes
    ]


def create_semantic_graph_visual_projection(
    content: dict[str, Any],
) -> SemanticGraphVisualProjectionV1:
    """Create a content-addressed canonical projection from derived content."""
    raw = {
        "schema_version": "semantic_graph_visual_projection.v1",
        "projection_version": 1,
        **dict(content),
    }
    raw.pop("projection_id", None)
    raw.pop("projection_hash", None)
    raw.pop("coordinate_hash", None)
    canonical = SemanticGraphVisualProjectionV1.canonicalize(raw)
    nodes = tuple(SemanticGraphProjectionNodeV1.model_validate(item) for item in canonical["nodes"])
    raw["nodes"] = nodes
    raw["edges"] = canonical["edges"]
    raw["coordinate_hash"] = "sha256:" + content_hash(_coordinate_payload(nodes))
    provisional = SemanticGraphVisualProjectionV1.model_construct(
        **{
            **raw,
            "relation_catalog": SemanticGraphRelationCatalogV1.model_validate(raw["relation_catalog"]),
            "edges": tuple(SemanticGraphProjectionEdgeV1.model_validate(item) for item in raw["edges"]),
            "projection_id": "pending",
            "projection_hash": "sha256:" + "0" * 64,
        }
    )
    projection_hash = "sha256:" + content_hash(provisional._hash_payload())
    return SemanticGraphVisualProjectionV1.model_validate({
        **provisional.as_dict(),
        "projection_id": "sgvp_" + projection_hash[7:31],
        "projection_hash": projection_hash,
    })


def filter_projection_edges(
    projection: SemanticGraphVisualProjectionV1,
    *,
    relation_family_id: str | None = None,
    relation_type_id: str | None = None,
) -> tuple[str, ...]:
    """Return ordered canonical assertion refs without changing projection state."""
    family = None if relation_family_id in (None, "ALL") else relation_family_id
    relation_type = None if relation_type_id in (None, "ALL") else relation_type_id
    type_families = {
        item.relation_type_id: item.family_id
        for item in projection.relation_catalog.relation_types
    }
    if relation_type is not None and relation_type not in type_families:
        raise ValueError("relation type is not present in projection catalog")
    if relation_type is not None and family is not None and type_families[relation_type] != family:
        raise ValueError("relation type does not belong to selected family")
    return tuple(
        edge.relation_instance_ref
        for edge in projection.edges
        if (family is None or edge.relation_family_id == family)
        and (relation_type is None or edge.relation_type_id == relation_type)
    )
