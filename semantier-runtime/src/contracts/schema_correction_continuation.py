"""Exact pre-release continuation for one governed schema-only correction."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class SchemaCorrectionContinuationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["schema_correction_continuation.v1"]
    continuation_ref: str
    continuation_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    feedback_ref: str
    feedback_hash: str = Field(pattern=SHA256_PATTERN)
    correction_material_ref: str
    correction_material_hash: str = Field(pattern=SHA256_PATTERN)
    graph_intent_ref: str
    graph_intent_hash: str = Field(pattern=SHA256_PATTERN)
    predecessor_candidate_ref: str
    predecessor_candidate_hash: str = Field(pattern=SHA256_PATTERN)
    predecessor_schema_decision_ref: str
    predecessor_schema_decision_hash: str = Field(pattern=SHA256_PATTERN)
    successor_candidate_ref: str
    successor_candidate_hash: str = Field(pattern=SHA256_PATTERN)
    successor_schema_decision_ref: str
    successor_schema_decision_hash: str = Field(pattern=SHA256_PATTERN)
    positive_manifest_ref: str
    positive_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    positive_count: int = Field(ge=1)
    hard_negative_manifest_ref: str
    hard_negative_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    hard_negative_count: int = Field(ge=1)
    comparison_ref: str
    comparison_hash: str = Field(pattern=SHA256_PATTERN)
    created_at: str
    created_by: str

    @field_validator(
        "continuation_ref", "organization_id", "workspace_id", "feedback_ref",
        "correction_material_ref", "graph_intent_ref", "predecessor_candidate_ref",
        "predecessor_schema_decision_ref", "successor_candidate_ref",
        "successor_schema_decision_ref", "positive_manifest_ref",
        "hard_negative_manifest_ref", "comparison_ref", "created_by",
    )
    @classmethod
    def ascii_refs(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value or not value.isascii():
            raise ValueError(f"{info.field_name} must be non-empty ASCII")
        return value

    @model_validator(mode="after")
    def verify_identity(self) -> "SchemaCorrectionContinuationV1":
        parsed = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("created_at must be timezone-aware UTC")
        if self.predecessor_candidate_ref == self.successor_candidate_ref:
            raise ValueError("schema correction requires a candidate successor")
        if self.predecessor_schema_decision_ref == self.successor_schema_decision_ref:
            raise ValueError("schema correction requires a decision successor")
        material_match = re.fullmatch(
            r"artifacts/material-correction-material/([0-9a-f]{64})\.json",
            self.correction_material_ref,
        )
        if (
            material_match is None
            or material_match.group(1)
            != self.correction_material_hash.removeprefix("sha256:")
        ):
            raise ValueError("correction_material_ref must match its governed hash path")
        digest = "sha256:" + content_hash(self.hash_payload())
        if self.continuation_hash != digest:
            raise ValueError("schema correction continuation hash mismatch")
        if self.continuation_ref != "scc_" + digest[7:31]:
            raise ValueError("schema correction continuation identity mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("continuation_ref", None)
        payload.pop("continuation_hash", None)
        return payload

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "SchemaCorrectionContinuationV1":
        digest = "sha256:" + content_hash(payload)
        return cls.model_validate({
            **payload,
            "continuation_ref": "scc_" + digest[7:31],
            "continuation_hash": digest,
        })


__all__ = ["SchemaCorrectionContinuationV1"]
