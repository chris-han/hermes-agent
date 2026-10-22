"""Append-only bitemporal companion state for immutable relation instances."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eos.artifact_hashing import canonical_json, content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


def parse_utc(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be timezone-aware UTC ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must be timezone-aware UTC ISO-8601")
    return parsed


class RelationTemporalStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["relation_temporal_state.v1"] = (
        "relation_temporal_state.v1"
    )
    state_ref: str
    state_hash: str = Field(pattern=SHA256_PATTERN)
    relation_instance_ref: str
    relation_instance_hash: str = Field(pattern=SHA256_PATTERN)
    valid_from: str | None
    valid_to: str | None
    observed_at: str
    admitted_at: str
    supersedes_state_ref: str | None
    supersedes_state_hash: str | None = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_contract(self) -> "RelationTemporalStateV1":
        if not self.relation_instance_ref.isascii():
            raise ValueError("relation_instance_ref must be ASCII-stable")
        observed = parse_utc(self.observed_at, "observed_at")
        admitted = parse_utc(self.admitted_at, "admitted_at")
        if observed > admitted:
            raise ValueError("observed_at must not be after admitted_at")
        valid_from = (
            parse_utc(self.valid_from, "valid_from") if self.valid_from else None
        )
        valid_to = parse_utc(self.valid_to, "valid_to") if self.valid_to else None
        if valid_from and valid_to and valid_to <= valid_from:
            raise ValueError("valid_to must be after valid_from")
        if (self.supersedes_state_ref is None) != (
            self.supersedes_state_hash is None
        ):
            raise ValueError("supersedes state ref/hash must be both null or both present")
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.state_hash != expected_hash:
            raise ValueError("state hash does not match canonical content")
        if self.state_ref != "relation-temporal-state:" + expected_hash[7:31]:
            raise ValueError("state ref does not match state hash")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("state_ref")
        payload.pop("state_hash")
        return payload

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


def create_relation_temporal_state(
    *,
    relation_instance_ref: str,
    relation_instance_hash: str,
    valid_from: str | None,
    valid_to: str | None,
    observed_at: str,
    admitted_at: str,
    supersedes_state_ref: str | None,
    supersedes_state_hash: str | None,
) -> RelationTemporalStateV1:
    payload = {
        "schema_version": "relation_temporal_state.v1",
        "relation_instance_ref": relation_instance_ref,
        "relation_instance_hash": relation_instance_hash,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "observed_at": observed_at,
        "admitted_at": admitted_at,
        "supersedes_state_ref": supersedes_state_ref,
        "supersedes_state_hash": supersedes_state_hash,
    }
    state_hash = "sha256:" + content_hash(payload)
    return RelationTemporalStateV1.model_validate(
        {
            **payload,
            "state_ref": "relation-temporal-state:" + state_hash[7:31],
            "state_hash": state_hash,
        }
    )


__all__ = ["RelationTemporalStateV1", "create_relation_temporal_state", "parse_utc"]
