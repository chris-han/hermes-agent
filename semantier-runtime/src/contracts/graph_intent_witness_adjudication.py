"""Immutable authority for one GraphIntent witness-set adjudication."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


class WitnessManifestBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_ref: str = Field(min_length=1)
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    artifact_ref: str = Field(pattern=r"^artifacts/")


class GraphIntentWitnessAdjudicationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["graph_intent_witness_adjudication.v1"]
    adjudication_ref: str = Field(pattern=r"^graph-intent-witness-adjudication:[0-9a-f]{24}$")
    adjudication_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    rerun_ref: str = Field(min_length=1)
    rerun_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    proposed_positive_manifest: WitnessManifestBinding
    proposed_hard_negative_manifest: WitnessManifestBinding
    accepted_positive_manifest: WitnessManifestBinding
    accepted_hard_negative_manifest: WitnessManifestBinding
    grounding_event_ref: str = Field(min_length=1)
    grounding_event_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    actor_ref: str = Field(min_length=1)
    decided_at: datetime

    @field_validator("decided_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decided_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def verify_identity(self) -> "GraphIntentWitnessAdjudicationV1":
        payload = self.model_dump(mode="json", exclude={"adjudication_hash"})
        if _hash(payload) != self.adjudication_hash:
            raise ValueError("GraphIntent witness adjudication hash mismatch")
        return self

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "GraphIntentWitnessAdjudicationV1":
        body = {"schema_version": "graph_intent_witness_adjudication.v1", **payload}
        if isinstance(body.get("decided_at"), datetime):
            body["decided_at"] = (
                body["decided_at"]
                .astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        return cls.model_validate({**body, "adjudication_hash": _hash(body)})

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


__all__ = ["GraphIntentWitnessAdjudicationV1", "WitnessManifestBinding"]
