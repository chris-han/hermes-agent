"""Append-only reservation and completion for one GraphIntent release successor."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="before")
    @classmethod
    def reject_non_ascii_strings(cls, value: Any) -> Any:
        if isinstance(value, str) and (not value or not value.isascii()):
            raise ValueError("finalization identities must be non-empty ASCII")
        return value


class GraphIntentReleaseReservationV1(_Contract):
    schema_version: Literal["graph_intent_release_reservation.v1"]
    reservation_ref: str
    reservation_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    workspace_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    rerun_ref: str = Field(pattern=r"^material-correction-rerun:[0-9a-f]{24}$")
    rerun_hash: str = Field(pattern=SHA256_PATTERN)
    adjudication_ref: str
    adjudication_hash: str = Field(pattern=SHA256_PATTERN)
    graph_intent_ref: str
    graph_intent_hash: str = Field(pattern=SHA256_PATTERN)
    schema_candidate_ref: str
    schema_candidate_hash: str = Field(pattern=SHA256_PATTERN)
    schema_decision_ref: str
    schema_decision_hash: str = Field(pattern=SHA256_PATTERN)
    extract_result_ref: str
    extract_result_hash: str = Field(pattern=SHA256_PATTERN)
    predecessor_release_ref: str
    predecessor_release_hash: str = Field(pattern=SHA256_PATTERN)
    candidate_graph_id: str
    human_label_event_ids: tuple[str, ...]

    @model_validator(mode="before")
    @classmethod
    def canonicalize_events(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        value = dict(payload)
        value["human_label_event_ids"] = tuple(
            sorted(value.get("human_label_event_ids", ()), key=str.encode)
        )
        return value

    @model_validator(mode="after")
    def verify_identity(self) -> "GraphIntentReleaseReservationV1":
        if len(self.human_label_event_ids) != len(set(self.human_label_event_ids)):
            raise ValueError("reservation Human Grounding events must be unique")
        digest = "sha256:" + content_hash(self.hash_payload())
        if self.reservation_hash != digest:
            raise ValueError("GraphIntent release reservation hash mismatch")
        if self.reservation_ref != "girr_" + digest[7:31]:
            raise ValueError("GraphIntent release reservation identity mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("reservation_ref", None)
        payload.pop("reservation_hash", None)
        return payload

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "GraphIntentReleaseReservationV1":
        content = cls.canonicalize_events(dict(payload))
        digest = "sha256:" + content_hash(content)
        return cls.model_validate(
            {**content, "reservation_ref": "girr_" + digest[7:31], "reservation_hash": digest}
        )


class GraphIntentReleaseFinalizationV1(_Contract):
    schema_version: Literal["graph_intent_release_finalization.v1"]
    finalization_ref: str
    finalization_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    workspace_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    rerun_ref: str = Field(pattern=r"^material-correction-rerun:[0-9a-f]{24}$")
    rerun_hash: str = Field(pattern=SHA256_PATTERN)
    reservation_ref: str
    reservation_hash: str = Field(pattern=SHA256_PATTERN)
    accepted_rerun_ref: str
    accepted_rerun_hash: str = Field(pattern=SHA256_PATTERN)
    successor_release_ref: str
    successor_release_hash: str = Field(pattern=SHA256_PATTERN)
    schema_feedback_retest_ref: str
    schema_feedback_retest_hash: str = Field(pattern=SHA256_PATTERN)
    semantic_release_retest_ref: str
    semantic_release_retest_hash: str = Field(pattern=SHA256_PATTERN)
    attribution_ref: str
    attribution_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def verify_identity(self) -> "GraphIntentReleaseFinalizationV1":
        digest = "sha256:" + content_hash(self.hash_payload())
        if self.finalization_hash != digest:
            raise ValueError("GraphIntent release finalization hash mismatch")
        if self.finalization_ref != "girf_" + digest[7:31]:
            raise ValueError("GraphIntent release finalization identity mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("finalization_ref", None)
        payload.pop("finalization_hash", None)
        return payload

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "GraphIntentReleaseFinalizationV1":
        digest = "sha256:" + content_hash(payload)
        return cls.model_validate(
            {**payload, "finalization_ref": "girf_" + digest[7:31], "finalization_hash": digest}
        )


__all__ = ["GraphIntentReleaseFinalizationV1", "GraphIntentReleaseReservationV1"]
