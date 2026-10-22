"""Sealed retry binding for a source-structure successor identity and IR."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eos.artifact_hashing import content_hash


_SHA = r"^sha256:[0-9a-f]{64}$"


class SourceStructureSuccessorReservationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["source_structure_successor_reservation.v1"]
    reservation_ref: str = Field(pattern=r"^ssr_[0-9a-f]{24}$")
    reservation_hash: str = Field(pattern=_SHA)
    organization_id: str
    workspace_id: str
    lineage_id: str
    predecessor_graph_intent_ref: str
    predecessor_graph_intent_hash: str = Field(pattern=_SHA)
    predecessor_source_identity_ref: str
    successor_source_identity_ref: str
    corrected_canonical_source_ir_ref: str
    corrected_canonical_source_ir_hash: str = Field(pattern=_SHA)
    block_content_inventory_hash: str = Field(pattern=_SHA)
    requesting_actor_ref: str
    created_at: str

    @field_validator(
        "organization_id",
        "workspace_id",
        "lineage_id",
        "predecessor_graph_intent_ref",
        "predecessor_source_identity_ref",
        "successor_source_identity_ref",
        "corrected_canonical_source_ir_ref",
        "requesting_actor_ref",
    )
    @classmethod
    def ascii_refs(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value or not value.isascii():
            raise ValueError(f"{info.field_name} must be non-empty ASCII")
        return value

    @model_validator(mode="after")
    def validate_authority(self) -> "SourceStructureSuccessorReservationV1":
        if self.predecessor_source_identity_ref == self.successor_source_identity_ref:
            raise ValueError("source-structure reservation requires a source successor")
        ir_match = re.fullmatch(
            r"artifacts/canonical-source-ir/([0-9a-f]{64})/canonical_source_ir\.v1\.json",
            self.corrected_canonical_source_ir_ref,
        )
        if (
            ir_match is None
            or ir_match.group(1)
            != self.corrected_canonical_source_ir_hash.removeprefix("sha256:")
        ):
            raise ValueError("corrected CanonicalSourceIR ref/hash mismatch")
        parsed = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("created_at must be timezone-aware UTC")
        digest = "sha256:" + content_hash(self.hash_payload())
        if self.reservation_hash != digest:
            raise ValueError("source-structure successor reservation hash mismatch")
        if self.reservation_ref != "ssr_" + digest[7:31]:
            raise ValueError("source-structure successor reservation identity mismatch")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("reservation_ref", None)
        payload.pop("reservation_hash", None)
        return payload

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "SourceStructureSuccessorReservationV1":
        content = dict(payload)
        content.pop("reservation_ref", None)
        content.pop("reservation_hash", None)
        digest = "sha256:" + content_hash(content)
        return cls.model_validate(
            {
                **content,
                "reservation_ref": "ssr_" + digest[7:31],
                "reservation_hash": digest,
            }
        )


__all__ = ["SourceStructureSuccessorReservationV1"]
