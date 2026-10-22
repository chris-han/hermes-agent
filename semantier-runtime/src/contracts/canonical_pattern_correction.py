"""Append-only authority for one reviewed canonical-pattern correction."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="before")
    @classmethod
    def ascii_strings(cls, value: Any) -> Any:
        if isinstance(value, str) and (not value or not value.isascii()):
            raise ValueError("canonical-pattern correction refs must be non-empty ASCII")
        return value


class CanonicalPatternCorrectionReservationV1(_Contract):
    schema_version: Literal["canonical_pattern_correction_reservation.v1"]
    reservation_ref: str
    reservation_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    feedback_ref: str
    feedback_hash: str = Field(pattern=SHA256_PATTERN)
    graph_intent_ref: str
    graph_intent_hash: str = Field(pattern=SHA256_PATTERN)
    schema_candidate_ref: str
    schema_candidate_hash: str = Field(pattern=SHA256_PATTERN)
    schema_decision_ref: str
    schema_decision_hash: str = Field(pattern=SHA256_PATTERN)
    extract_result_ref: str
    extract_result_hash: str = Field(pattern=SHA256_PATTERN)
    predecessor_release_ref: str = Field(pattern=r"^KG_v[1-9]\d*$")
    predecessor_release_hash: str = Field(pattern=SHA256_PATTERN)
    candidate_graph_id: str
    candidate_graph_hash: str = Field(pattern=SHA256_PATTERN)
    human_label_event_ref: str
    human_label_event_hash: str = Field(pattern=SHA256_PATTERN)
    created_at: str
    created_by: str

    @model_validator(mode="after")
    def verify_identity(self) -> "CanonicalPatternCorrectionReservationV1":
        parsed = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("created_at must be timezone-aware UTC")
        digest = "sha256:" + content_hash(self.hash_payload())
        if self.reservation_hash != digest:
            raise ValueError("canonical-pattern correction reservation hash mismatch")
        if self.reservation_ref != "cpcr_" + digest[7:31]:
            raise ValueError("canonical-pattern correction reservation identity mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        value = self.model_dump(mode="json")
        value.pop("reservation_ref", None)
        value.pop("reservation_hash", None)
        return value

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "CanonicalPatternCorrectionReservationV1":
        digest = "sha256:" + content_hash(payload)
        return cls.model_validate({
            **payload,
            "reservation_ref": "cpcr_" + digest[7:31],
            "reservation_hash": digest,
        })


class CanonicalPatternCorrectionFinalizationV1(_Contract):
    schema_version: Literal["canonical_pattern_correction_finalization.v1"]
    finalization_ref: str
    finalization_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    reservation_ref: str
    reservation_hash: str = Field(pattern=SHA256_PATTERN)
    successor_release_ref: str = Field(pattern=r"^KG_v[1-9]\d*$")
    successor_release_hash: str = Field(pattern=SHA256_PATTERN)
    semantic_release_retest_ref: str
    semantic_release_retest_hash: str = Field(pattern=SHA256_PATTERN)
    attribution_ref: str
    attribution_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def verify_identity(self) -> "CanonicalPatternCorrectionFinalizationV1":
        digest = "sha256:" + content_hash(self.hash_payload())
        if self.finalization_hash != digest:
            raise ValueError("canonical-pattern correction finalization hash mismatch")
        if self.finalization_ref != "cpcf_" + digest[7:31]:
            raise ValueError("canonical-pattern correction finalization identity mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        value = self.model_dump(mode="json")
        value.pop("finalization_ref", None)
        value.pop("finalization_hash", None)
        return value

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "CanonicalPatternCorrectionFinalizationV1":
        digest = "sha256:" + content_hash(payload)
        return cls.model_validate({
            **payload,
            "finalization_ref": "cpcf_" + digest[7:31],
            "finalization_hash": digest,
        })


__all__ = [
    "CanonicalPatternCorrectionFinalizationV1",
    "CanonicalPatternCorrectionReservationV1",
]
