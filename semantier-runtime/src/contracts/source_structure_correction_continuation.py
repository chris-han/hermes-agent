"""Exact pre-release continuation for one neutral source-structure correction."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eos.artifact_hashing import content_hash


_SHA = r"^sha256:[0-9a-f]{64}$"


class SourceStructureCorrectionContinuationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["source_structure_correction_continuation.v1"]
    continuation_ref: str = Field(pattern=r"^sscc_[0-9a-f]{24}$")
    continuation_hash: str = Field(pattern=_SHA)
    organization_id: str
    workspace_id: str
    semantic_validation_ref: str
    semantic_validation_hash: str = Field(pattern=_SHA)
    predecessor_graph_intent_ref: str
    predecessor_graph_intent_hash: str = Field(pattern=_SHA)
    successor_graph_intent_ref: str
    successor_graph_intent_hash: str = Field(pattern=_SHA)
    predecessor_candidate_ref: str
    predecessor_candidate_hash: str = Field(pattern=_SHA)
    successor_candidate_ref: str
    successor_candidate_hash: str = Field(pattern=_SHA)
    predecessor_schema_decision_ref: str
    predecessor_schema_decision_hash: str = Field(pattern=_SHA)
    successor_schema_decision_ref: str
    successor_schema_decision_hash: str = Field(pattern=_SHA)
    positive_manifest_ref: str
    positive_manifest_hash: str = Field(pattern=_SHA)
    positive_count: Literal[32]
    hard_negative_manifest_ref: str
    hard_negative_manifest_hash: str = Field(pattern=_SHA)
    hard_negative_count: Literal[6]
    comparison_ref: str
    comparison_hash: str = Field(pattern=_SHA)
    baseline_extract_ref: str
    baseline_extract_hash: str = Field(pattern=_SHA)
    successor_extract_ref: str
    successor_extract_hash: str = Field(pattern=_SHA)
    created_at: str
    created_by: str

    @field_validator(
        "organization_id",
        "workspace_id",
        "semantic_validation_ref",
        "predecessor_graph_intent_ref",
        "successor_graph_intent_ref",
        "predecessor_candidate_ref",
        "successor_candidate_ref",
        "predecessor_schema_decision_ref",
        "successor_schema_decision_ref",
        "positive_manifest_ref",
        "hard_negative_manifest_ref",
        "comparison_ref",
        "baseline_extract_ref",
        "successor_extract_ref",
        "created_by",
    )
    @classmethod
    def ascii_refs(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value or not value.isascii():
            raise ValueError(f"{info.field_name} must be non-empty ASCII")
        return value

    @model_validator(mode="after")
    def verify_identity(self) -> "SourceStructureCorrectionContinuationV1":
        for predecessor, successor, label in (
            (self.predecessor_graph_intent_ref, self.successor_graph_intent_ref, "GraphIntent"),
            (self.predecessor_candidate_ref, self.successor_candidate_ref, "candidate"),
            (
                self.predecessor_schema_decision_ref,
                self.successor_schema_decision_ref,
                "SchemaDecision",
            ),
            (self.baseline_extract_ref, self.successor_extract_ref, "ExtractResult"),
        ):
            if predecessor == successor:
                raise ValueError(f"source-structure correction requires a {label} successor")
        parsed = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("created_at must be timezone-aware UTC")
        digest = "sha256:" + content_hash(self.hash_payload())
        if self.continuation_hash != digest:
            raise ValueError("source-structure continuation hash mismatch")
        if self.continuation_ref != "sscc_" + digest[7:31]:
            raise ValueError("source-structure continuation identity mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("continuation_ref", None)
        payload.pop("continuation_hash", None)
        return payload

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "SourceStructureCorrectionContinuationV1":
        digest = "sha256:" + content_hash(payload)
        return cls.model_validate(
            {
                **payload,
                "continuation_ref": "sscc_" + digest[7:31],
                "continuation_hash": digest,
            }
        )


__all__ = ["SourceStructureCorrectionContinuationV1"]
