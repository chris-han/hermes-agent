"""Strict relation/pattern-aware AcceptedGraphRelease v2 contracts.

The historical unversioned :class:`AcceptedGraphRelease` is deliberately not
changed here.  This module is the additive, explicitly dispatched successor.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.knowledge_graph_phase1 import AcceptedGraphRelease
from eos.artifact_hashing import canonical_json, content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, dict) or isinstance(value, MappingProxyType):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _required_ascii(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise ValueError(f"{field} must be a non-empty ASCII-stable reference")
    return value


def _sort_members(items: Any, fields: tuple[str, ...], label: str) -> tuple[Any, ...]:
    if not isinstance(items, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    values = tuple(items)
    identities = [
        tuple(
            str(
                (item if isinstance(item, dict) else item.model_dump()).get(
                    field, ""
                )
            )
            for field in fields
        )
        for item in values
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{label} contains a duplicate membership tuple")
    return tuple(
        item
        for _, item in sorted(
            zip(identities, values),
            key=lambda pair: tuple(part.encode("utf-8") for part in pair[0]),
        )
    )


def _reject_duplicate_member_field(
    items: tuple[Any, ...], field: str, label: str
) -> None:
    values = [
        (item if isinstance(item, dict) else item.model_dump(mode="json"))[field]
        for item in items
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains a duplicate {field}")


def _sort_objects(items: Any, id_field: str, label: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(items, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    copied = tuple(dict(item) for item in items)
    ids = [
        _required_ascii(item.get(id_field), f"{label}[].{id_field}")
        for item in copied
    ]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains a duplicate {id_field}")
    return tuple(
        sorted(
            copied,
            key=lambda item: (
                item[id_field].encode("utf-8"),
                canonical_json(item).encode("utf-8"),
            ),
        )
    )


class RelationReleaseMembership(_Contract):
    relation_instance_ref: str
    relation_instance_hash: str = Field(pattern=SHA256_PATTERN)
    occurrence_grounding_ref: str
    occurrence_grounding_hash: str = Field(pattern=SHA256_PATTERN)

    @field_validator("relation_instance_ref", "occurrence_grounding_ref")
    @classmethod
    def refs_are_ascii(cls, value: str, info) -> str:
        return _required_ascii(value, info.field_name)


class PatternReleaseMembership(_Contract):
    pattern_ref: str
    pattern_hash: str = Field(pattern=SHA256_PATTERN)

    @field_validator("pattern_ref")
    @classmethod
    def ref_is_ascii(cls, value: str) -> str:
        return _required_ascii(value, "pattern_ref")


class AcceptedGraphReleaseV2(_Contract):
    """Deeply immutable canonical DTO; authority requires store materialization."""

    schema_version: Literal["accepted_graph_release.v2"] = "accepted_graph_release.v2"
    graph_version: str = Field(pattern=r"^KG_v[1-9]\d*$")
    release_hash: str = Field(pattern=SHA256_PATTERN)
    parent_graph_version: str | None
    candidate_graph_id: str
    graph_delta_id: str | None
    human_label_event_ids: tuple[str, ...]
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    rules: tuple[dict[str, Any], ...]
    relation_membership: tuple[RelationReleaseMembership, ...]
    pattern_membership: tuple[PatternReleaseMembership, ...]
    backend: Literal["falkordb"] = "falkordb"

    @model_validator(mode="before")
    @classmethod
    def canonicalize(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        value = dict(payload)
        events = tuple(
            _required_ascii(item, "human_label_event_ids[]")
            for item in value.get("human_label_event_ids", ())
        )
        if len(events) != len(set(events)):
            raise ValueError("human_label_event_ids contains a duplicate identity")
        value["human_label_event_ids"] = tuple(
            sorted(events, key=lambda item: item.encode("utf-8"))
        )
        value["nodes"] = _sort_objects(value.get("nodes", ()), "node_id", "nodes")
        value["edges"] = _sort_objects(value.get("edges", ()), "edge_id", "edges")
        value["rules"] = _sort_objects(value.get("rules", ()), "rule_id", "rules")
        value["relation_membership"] = _sort_members(
            value.get("relation_membership", ()),
            (
                "relation_instance_ref",
                "relation_instance_hash",
                "occurrence_grounding_ref",
                "occurrence_grounding_hash",
            ),
            "relation_membership",
        )
        _reject_duplicate_member_field(
            value["relation_membership"],
            "relation_instance_ref",
            "relation_membership",
        )
        _reject_duplicate_member_field(
            value["relation_membership"],
            "occurrence_grounding_ref",
            "relation_membership",
        )
        value["pattern_membership"] = _sort_members(
            value.get("pattern_membership", ()),
            ("pattern_ref", "pattern_hash"),
            "pattern_membership",
        )
        _reject_duplicate_member_field(
            value["pattern_membership"], "pattern_ref", "pattern_membership"
        )
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "AcceptedGraphReleaseV2":
        object.__setattr__(self, "nodes", _deep_freeze(self.nodes))
        object.__setattr__(self, "edges", _deep_freeze(self.edges))
        object.__setattr__(self, "rules", _deep_freeze(self.rules))
        for field in ("candidate_graph_id",):
            _required_ascii(getattr(self, field), field)
        for field in ("parent_graph_version", "graph_delta_id"):
            value = getattr(self, field)
            if value is not None:
                _required_ascii(value, field)
        expected = f"sha256:{content_hash(self._hash_payload())}"
        if self.release_hash != expected:
            raise ValueError("release hash does not match accepted graph release content")
        return self

    @property
    def accepted_graph_release_ref(self) -> str:
        return self.graph_version

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("release_hash")
        return payload

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_version": self.graph_version,
            "release_hash": self.release_hash,
            "parent_graph_version": self.parent_graph_version,
            "candidate_graph_id": self.candidate_graph_id,
            "graph_delta_id": self.graph_delta_id,
            "human_label_event_ids": list(self.human_label_event_ids),
            "nodes": _deep_thaw(self.nodes),
            "edges": _deep_thaw(self.edges),
            "rules": _deep_thaw(self.rules),
            "relation_membership": [
                item.model_dump(mode="json") for item in self.relation_membership
            ],
            "pattern_membership": [
                item.model_dump(mode="json") for item in self.pattern_membership
            ],
            "backend": self.backend,
        }

    @classmethod
    def create_unverified(cls, content: dict[str, Any]) -> "AcceptedGraphReleaseV2":
        """Build a canonical DTO; this does not grant accepted-graph authority."""
        return _create_release(cls, content)

    @classmethod
    def parse_unverified(cls, payload: dict[str, Any]) -> "AcceptedGraphReleaseV2":
        """Parse canonical bytes without resolving semantic/store authority."""
        return cls.model_validate(payload)


def _create_release(cls, content: dict[str, Any]) -> AcceptedGraphReleaseV2:
    raw = {"schema_version": "accepted_graph_release.v2", **dict(content)}
    raw.pop("release_hash", None)
    canonical = cls.canonicalize(raw)
    payload = dict(canonical)
    payload["release_hash"] = f"sha256:{content_hash(payload)}"
    return cls.model_validate(payload)


class OccurrenceGroundingManifestV1(_Contract):
    """Unverified canonical manifest DTO for later store-backed resolution."""

    schema_version: Literal["occurrence_grounding_manifest.v1"] = (
        "occurrence_grounding_manifest.v1"
    )
    manifest_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    schema_decision_ref: str
    schema_decision_hash: str = Field(pattern=SHA256_PATTERN)
    extract_result_ref: str
    extract_result_hash: str = Field(pattern=SHA256_PATTERN)
    relation_membership: tuple[RelationReleaseMembership, ...]

    @model_validator(mode="before")
    @classmethod
    def canonicalize(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        value = dict(payload)
        value["relation_membership"] = _sort_members(
            value.get("relation_membership", ()),
            (
                "relation_instance_ref",
                "relation_instance_hash",
                "occurrence_grounding_ref",
                "occurrence_grounding_hash",
            ),
            "relation_membership",
        )
        _reject_duplicate_member_field(
            value["relation_membership"],
            "relation_instance_ref",
            "relation_membership",
        )
        _reject_duplicate_member_field(
            value["relation_membership"],
            "occurrence_grounding_ref",
            "relation_membership",
        )
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "OccurrenceGroundingManifestV1":
        for field in (
            "organization_id",
            "workspace_id",
            "schema_decision_ref",
            "extract_result_ref",
        ):
            _required_ascii(getattr(self, field), field)
        if not self.relation_membership:
            raise ValueError("relation_membership must not be empty")
        expected = f"sha256:{content_hash(self._hash_payload())}"
        if self.manifest_hash != expected:
            raise ValueError("manifest hash does not match content")
        return self

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("manifest_hash")
        return payload

    @property
    def manifest_ref(self) -> str:
        return f"occurrence-grounding-manifest:{self.manifest_hash[7:31]}"

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def assert_matches_release(self, release: AcceptedGraphReleaseV2) -> None:
        if self.relation_membership != release.relation_membership:
            raise ValueError("relation membership mismatch")

    @classmethod
    def create_unverified(
        cls, content: dict[str, Any]
    ) -> "OccurrenceGroundingManifestV1":
        return _create_manifest(cls, content)

    @classmethod
    def parse_unverified(
        cls, payload: dict[str, Any]
    ) -> "OccurrenceGroundingManifestV1":
        return cls.model_validate(payload)


class CanonicalPatternManifestV1(_Contract):
    """Unverified canonical manifest DTO for later store-backed resolution."""

    schema_version: Literal["canonical_pattern_manifest.v1"] = (
        "canonical_pattern_manifest.v1"
    )
    manifest_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    pattern_membership: tuple[PatternReleaseMembership, ...]

    @model_validator(mode="before")
    @classmethod
    def canonicalize(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        value = dict(payload)
        value["pattern_membership"] = _sort_members(
            value.get("pattern_membership", ()),
            ("pattern_ref", "pattern_hash"),
            "pattern_membership",
        )
        _reject_duplicate_member_field(
            value["pattern_membership"], "pattern_ref", "pattern_membership"
        )
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "CanonicalPatternManifestV1":
        for field in ("organization_id", "workspace_id"):
            _required_ascii(getattr(self, field), field)
        if not self.pattern_membership:
            raise ValueError("pattern_membership must not be empty")
        expected = f"sha256:{content_hash(self._hash_payload())}"
        if self.manifest_hash != expected:
            raise ValueError("manifest hash does not match content")
        return self

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("manifest_hash")
        return payload

    @property
    def manifest_ref(self) -> str:
        return f"canonical-pattern-manifest:{self.manifest_hash[7:31]}"

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def assert_matches_release(self, release: AcceptedGraphReleaseV2) -> None:
        if self.pattern_membership != release.pattern_membership:
            raise ValueError("pattern membership mismatch")

    @classmethod
    def create_unverified(
        cls, content: dict[str, Any]
    ) -> "CanonicalPatternManifestV1":
        return _create_manifest(cls, content)

    @classmethod
    def parse_unverified(
        cls, payload: dict[str, Any]
    ) -> "CanonicalPatternManifestV1":
        return cls.model_validate(payload)


def _create_manifest(cls, content: dict[str, Any]):
    raw = {"schema_version": cls.model_fields["schema_version"].default, **dict(content)}
    raw.pop("manifest_hash", None)
    canonical = cls.canonicalize(raw)
    payload = dict(canonical)
    payload["manifest_hash"] = f"sha256:{content_hash(payload)}"
    return cls.model_validate(payload)


def parse_accepted_graph_release_unverified(
    payload: dict[str, Any],
) -> AcceptedGraphRelease | AcceptedGraphReleaseV2:
    """Version-dispatch canonical DTO parsing without authority resolution."""
    version = payload.get("schema_version")
    if version == "accepted_graph_release.v2":
        return AcceptedGraphReleaseV2.parse_unverified(payload)
    if version is not None:
        raise ValueError(f"unsupported accepted graph release schema_version: {version}")
    legacy = dict(payload)
    supplied_hash = legacy.pop("release_hash", None)
    release = AcceptedGraphRelease.model_validate(legacy)
    if supplied_hash is not None and release.release_hash() != supplied_hash:
        raise ValueError("legacy accepted release hash mismatch")
    return release


__all__ = [
    "AcceptedGraphReleaseV2",
    "CanonicalPatternManifestV1",
    "OccurrenceGroundingManifestV1",
    "PatternReleaseMembership",
    "RelationReleaseMembership",
    "parse_accepted_graph_release_unverified",
]
