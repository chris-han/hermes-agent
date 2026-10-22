"""Reservation and terminal evidence for a source-structure release successor."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eos.artifact_hashing import content_hash


_SHA = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="before")
    @classmethod
    def ascii_strings(cls, value: Any) -> Any:
        if isinstance(value, str) and (not value or not value.isascii()):
            raise ValueError("source-structure release refs must be non-empty ASCII")
        return value


class SourceStructureReleaseReservationV1(_Contract):
    schema_version: Literal["source_structure_release_reservation.v1"]
    reservation_ref: str
    reservation_hash: str = Field(pattern=_SHA)
    organization_id: str
    workspace_id: str
    continuation_ref: str = Field(pattern=r"^sscc_[0-9a-f]{24}$")
    continuation_hash: str = Field(pattern=_SHA)
    successor_graph_intent_ref: str
    successor_graph_intent_hash: str = Field(pattern=_SHA)
    successor_candidate_ref: str
    successor_candidate_hash: str = Field(pattern=_SHA)
    successor_schema_decision_ref: str
    successor_schema_decision_hash: str = Field(pattern=_SHA)
    successor_extract_ref: str
    successor_extract_hash: str = Field(pattern=_SHA)
    predecessor_release_ref: str
    predecessor_release_hash: str = Field(pattern=_SHA)
    candidate_graph_id: str
    human_label_event_ids: tuple[str, ...]

    @model_validator(mode="before")
    @classmethod
    def canonicalize_events(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload["human_label_event_ids"] = tuple(
            sorted(payload.get("human_label_event_ids", ()), key=str.encode)
        )
        return payload

    @model_validator(mode="after")
    def verify_identity(self) -> "SourceStructureReleaseReservationV1":
        if not self.human_label_event_ids or len(self.human_label_event_ids) != len(
            set(self.human_label_event_ids)
        ):
            raise ValueError("source-structure reservation Grounding events are invalid")
        digest = "sha256:" + content_hash(self.hash_payload())
        if self.reservation_hash != digest or self.reservation_ref != "ssrr_" + digest[7:31]:
            raise ValueError("source-structure release reservation identity mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("reservation_ref", None)
        payload.pop("reservation_hash", None)
        return payload

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "SourceStructureReleaseReservationV1":
        content = cls.canonicalize_events(dict(payload))
        digest = "sha256:" + content_hash(content)
        return cls.model_validate(
            {**content, "reservation_ref": "ssrr_" + digest[7:31], "reservation_hash": digest}
        )


class SourceStructureReleaseFinalizationV1(_Contract):
    schema_version: Literal["source_structure_release_finalization.v1"]
    finalization_ref: str
    finalization_hash: str = Field(pattern=_SHA)
    organization_id: str
    workspace_id: str
    continuation_ref: str = Field(pattern=r"^sscc_[0-9a-f]{24}$")
    continuation_hash: str = Field(pattern=_SHA)
    reservation_ref: str
    reservation_hash: str = Field(pattern=_SHA)
    feedback_ref: str
    feedback_hash: str = Field(pattern=_SHA)
    successor_release_ref: str
    successor_release_hash: str = Field(pattern=_SHA)
    semantic_release_retest_ref: str
    semantic_release_retest_hash: str = Field(pattern=_SHA)
    attribution_ref: str
    attribution_hash: str = Field(pattern=_SHA)

    @model_validator(mode="after")
    def verify_identity(self) -> "SourceStructureReleaseFinalizationV1":
        digest = "sha256:" + content_hash(self.hash_payload())
        if self.finalization_hash != digest or self.finalization_ref != "ssrf_" + digest[7:31]:
            raise ValueError("source-structure release finalization identity mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("finalization_ref", None)
        payload.pop("finalization_hash", None)
        return payload

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "SourceStructureReleaseFinalizationV1":
        digest = "sha256:" + content_hash(payload)
        return cls.model_validate(
            {**payload, "finalization_ref": "ssrf_" + digest[7:31], "finalization_hash": digest}
        )


__all__ = ["SourceStructureReleaseFinalizationV1", "SourceStructureReleaseReservationV1"]
