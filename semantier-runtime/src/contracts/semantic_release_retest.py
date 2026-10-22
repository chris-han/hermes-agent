"""Strict deterministic comparison of two accepted semantic graph releases."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.accepted_graph_release_v2 import (
    PatternReleaseMembership,
    RelationReleaseMembership,
)
from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
T = TypeVar("T")


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MembershipComparison(_Contract, Generic[T]):
    before: tuple[T, ...]
    after: tuple[T, ...]


class SemanticReleaseRetestV1(_Contract):
    schema_version: Literal["semantic_release_retest.v1"] = (
        "semantic_release_retest.v1"
    )
    evaluation_id: str
    evaluation_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    baseline_release_ref: str = Field(pattern=r"^KG_v[1-9]\d*$")
    baseline_release_hash: str = Field(pattern=SHA256_PATTERN)
    successor_release_ref: str = Field(pattern=r"^KG_v[1-9]\d*$")
    successor_release_hash: str = Field(pattern=SHA256_PATTERN)
    schema_feedback_retest_ref: str
    schema_feedback_retest_hash: str = Field(pattern=SHA256_PATTERN)
    pattern_membership: MembershipComparison[PatternReleaseMembership]
    relation_membership: MembershipComparison[RelationReleaseMembership]

    @model_validator(mode="before")
    @classmethod
    def canonicalize_membership(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        value = dict(payload)
        for field, keys in (
            ("pattern_membership", ("pattern_ref", "pattern_hash")),
            (
                "relation_membership",
                (
                    "relation_instance_ref",
                    "relation_instance_hash",
                    "occurrence_grounding_ref",
                    "occurrence_grounding_hash",
                ),
            ),
        ):
            comparison = dict(value.get(field) or {})
            for side in ("before", "after"):
                items = comparison.get(side, ())
                if not isinstance(items, (list, tuple)):
                    continue
                comparison[side] = sorted(
                    items,
                    key=lambda item: tuple(
                        str(
                            (item if isinstance(item, dict) else item.model_dump())[
                                key
                            ]
                        ).encode("utf-8")
                        for key in keys
                    ),
                )
            value[field] = comparison
        return value

    @field_validator(
        "evaluation_id",
        "organization_id",
        "workspace_id",
        "schema_feedback_retest_ref",
    )
    @classmethod
    def ascii_identity(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value or not value.isascii():
            raise ValueError(f"{info.field_name} must be non-empty ASCII")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "SemanticReleaseRetestV1":
        if self.baseline_release_ref == self.successor_release_ref:
            raise ValueError("release retest requires distinct releases")
        for label, comparison in (
            ("pattern_membership", self.pattern_membership),
            ("relation_membership", self.relation_membership),
        ):
            for side in ("before", "after"):
                members = getattr(comparison, side)
                identities = tuple(
                    tuple(sorted(item.model_dump(mode="json").items()))
                    for item in members
                )
                if len(identities) != len(set(identities)):
                    raise ValueError(f"{label}.{side} contains duplicate membership")
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.evaluation_hash != expected_hash:
            raise ValueError("semantic release retest hash mismatch")
        if self.evaluation_id != "semantic-release-retest:" + expected_hash[7:31]:
            raise ValueError("semantic release retest identity mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("evaluation_id", None)
        payload.pop("evaluation_hash", None)
        return payload

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "SemanticReleaseRetestV1":
        content = dict(payload)
        content.pop("evaluation_id", None)
        content.pop("evaluation_hash", None)
        content = cls.canonicalize_membership(content)
        # Validate and canonicalize nested membership without accepting a caller hash.
        patterns = MembershipComparison[PatternReleaseMembership].model_validate(
            content["pattern_membership"]
        )
        relations = MembershipComparison[RelationReleaseMembership].model_validate(
            content["relation_membership"]
        )
        content["pattern_membership"] = patterns.model_dump(mode="json")
        content["relation_membership"] = relations.model_dump(mode="json")
        digest = "sha256:" + content_hash(content)
        return cls.model_validate(
            {
                **content,
                "evaluation_id": "semantic-release-retest:" + digest[7:31],
                "evaluation_hash": digest,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SemanticReleaseRetestV1":
        return cls.model_validate(payload)


__all__ = ["SemanticReleaseRetestV1"]
