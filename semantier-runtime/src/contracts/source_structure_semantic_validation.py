"""Neutral semantic validation for a source-structure-only successor."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eos.artifact_hashing import content_hash


_SHA = r"^sha256:[0-9a-f]{64}$"


class SourceStructureSemanticValidationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["source_structure_semantic_validation.v1"]
    validation_ref: str
    validation_hash: str = Field(pattern=_SHA)
    organization_id: str
    workspace_id: str
    predecessor_graph_intent_ref: str
    predecessor_graph_intent_hash: str = Field(pattern=_SHA)
    successor_graph_intent_ref: str
    successor_graph_intent_hash: str = Field(pattern=_SHA)
    predecessor_source_identity_ref: str
    predecessor_source_identity_hash: str = Field(pattern=_SHA)
    successor_source_identity_ref: str
    successor_source_identity_hash: str = Field(pattern=_SHA)
    predecessor_canonical_source_ir_ref: str
    predecessor_canonical_source_ir_hash: str = Field(pattern=_SHA)
    successor_canonical_source_ir_ref: str
    successor_canonical_source_ir_hash: str = Field(pattern=_SHA)
    validation_method: Literal["block_content_identity.v1"]
    predecessor_block_content_inventory_hash: str = Field(pattern=_SHA)
    successor_block_content_inventory_hash: str = Field(pattern=_SHA)
    competency_question_refs: tuple[str, ...]
    hard_negative_question_refs: tuple[str, ...]
    outcome: Literal["passed"]
    created_at: str
    created_by: str

    @field_validator(
        "validation_ref",
        "organization_id",
        "workspace_id",
        "predecessor_graph_intent_ref",
        "successor_graph_intent_ref",
        "predecessor_source_identity_ref",
        "successor_source_identity_ref",
        "predecessor_canonical_source_ir_ref",
        "successor_canonical_source_ir_ref",
        "created_by",
    )
    @classmethod
    def ascii_ref(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value or not value.isascii():
            raise ValueError(f"{info.field_name} must be non-empty ASCII")
        return value

    @field_validator("competency_question_refs", "hard_negative_question_refs")
    @classmethod
    def canonical_refs(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        if any(not item or not item.isascii() for item in value):
            raise ValueError(f"{info.field_name} must contain ASCII refs")
        ordered = tuple(sorted(value, key=lambda item: item.encode("utf-8")))
        if len(ordered) != len(set(ordered)):
            raise ValueError(f"{info.field_name} contains duplicates")
        return ordered

    @model_validator(mode="after")
    def validate_authority(self) -> "SourceStructureSemanticValidationV1":
        if self.predecessor_source_identity_ref == self.successor_source_identity_ref:
            raise ValueError("source-structure validation requires a source successor")
        if self.predecessor_graph_intent_ref == self.successor_graph_intent_ref:
            raise ValueError("source-structure validation requires a GraphIntent successor")
        if (
            self.predecessor_canonical_source_ir_ref
            == self.successor_canonical_source_ir_ref
            or self.predecessor_canonical_source_ir_hash
            == self.successor_canonical_source_ir_hash
        ):
            raise ValueError("source-structure validation requires a changed canonical IR")
        if (
            self.predecessor_block_content_inventory_hash
            != self.successor_block_content_inventory_hash
        ):
            raise ValueError("source-structure content inventory must remain identical")
        parsed = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("created_at must be timezone-aware UTC")
        digest = "sha256:" + content_hash(self.hash_payload())
        if self.validation_hash != digest:
            raise ValueError("source-structure validation hash mismatch")
        if self.validation_ref != "ssv_" + digest[7:31]:
            raise ValueError("source-structure validation identity mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("validation_ref", None)
        payload.pop("validation_hash", None)
        return payload

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "SourceStructureSemanticValidationV1":
        content = dict(payload)
        content.pop("validation_ref", None)
        content.pop("validation_hash", None)
        content["competency_question_refs"] = sorted(
            content.get("competency_question_refs") or [], key=lambda item: item.encode()
        )
        content["hard_negative_question_refs"] = sorted(
            content.get("hard_negative_question_refs") or [], key=lambda item: item.encode()
        )
        digest = "sha256:" + content_hash(content)
        return cls.model_validate(
            {
                **content,
                "validation_ref": "ssv_" + digest[7:31],
                "validation_hash": digest,
            }
        )


__all__ = ["SourceStructureSemanticValidationV1"]
