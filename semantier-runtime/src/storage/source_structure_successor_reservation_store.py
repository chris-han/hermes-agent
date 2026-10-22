"""Append-only identity-to-IR reservations for source-structure successors."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from contracts.source_structure_successor_reservation import (
    SourceStructureSuccessorReservationV1,
)
from storage.atomic_json_artifact import publish_immutable_json


_SOURCE_STRUCTURE_SUCCESSOR_RESERVATION_AUTHORITY = object()


class SourceStructureSuccessorReservationStore:
    """Prevent one successor SourceIdentity from naming multiple corrected IRs."""

    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if (
            not isinstance(workspace_id, str)
            or not workspace_id
            or workspace_id in {".", ".."}
            or "/" in workspace_id
            or "\\" in workspace_id
        ):
            raise ValueError("workspace_id must be one safe path segment")
        workspaces = Path(workspace_root).resolve() / "workspaces"
        self.workspace_id = workspace_id
        self.root = (
            workspaces
            / workspace_id
            / "artifacts/schema-discovery/source-structure-successor-reservations"
        ).resolve()
        if not self.root.is_relative_to(workspaces.resolve()):
            raise ValueError("workspace_id escapes governed workspace root")

    @staticmethod
    def _reservation_key(payload: dict[str, Any]) -> str:
        identity_key = {
            "organization_id": payload["organization_id"],
            "workspace_id": payload["workspace_id"],
            "predecessor_source_identity_ref": payload[
                "predecessor_source_identity_ref"
            ],
            "successor_source_identity_ref": payload["successor_source_identity_ref"],
        }
        encoded = json.dumps(
            identity_key, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(encoded).hexdigest()[:24]

    def publish(
        self,
        reservation: SourceStructureSuccessorReservationV1,
        *,
        _authority: object,
    ) -> SourceStructureSuccessorReservationV1:
        if _authority is not _SOURCE_STRUCTURE_SUCCESSOR_RESERVATION_AUTHORITY:
            raise PermissionError("source-structure successor reservation is sealed")
        if type(reservation) is not SourceStructureSuccessorReservationV1:
            raise TypeError("source-structure successor reservation must be exact")
        if reservation.workspace_id != self.workspace_id:
            raise ValueError("source-structure successor reservation workspace mismatch")
        payload = reservation.model_dump(mode="json")
        key = self._reservation_key(payload)

        def equivalent(existing: dict[str, Any], proposed: dict[str, Any]) -> bool:
            # A later exact retry may supply a new clock value. The winner's
            # creation time remains authoritative; every semantic binding must match.
            retry_variant_fields = {
                "created_at",
                "reservation_ref",
                "reservation_hash",
            }
            return {
                key: value
                for key, value in existing.items()
                if key not in retry_variant_fields
            } == {
                key: value
                for key, value in proposed.items()
                if key not in retry_variant_fields
            }

        published = publish_immutable_json(
            self.root / f"ssr_{key}.json",
            payload,
            conflict_error="SOURCE_STRUCTURE_SUCCESSOR_RESERVATION_CONFLICT",
            equivalent=equivalent,
        )
        return SourceStructureSuccessorReservationV1.model_validate(published)

    def get(
        self,
        *,
        organization_id: str,
        predecessor_source_identity_ref: str,
        successor_source_identity_ref: str,
        reservation_ref: str,
        reservation_hash: str,
    ) -> SourceStructureSuccessorReservationV1:
        key = self._reservation_key(
            {
                "organization_id": organization_id,
                "workspace_id": self.workspace_id,
                "predecessor_source_identity_ref": predecessor_source_identity_ref,
                "successor_source_identity_ref": successor_source_identity_ref,
            }
        )
        path = self.root / f"ssr_{key}.json"
        if not path.is_file():
            raise ValueError("source-structure successor reservation is missing")
        reservation = SourceStructureSuccessorReservationV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if (
            reservation.organization_id != organization_id
            or reservation.workspace_id != self.workspace_id
            or reservation.predecessor_source_identity_ref
            != predecessor_source_identity_ref
            or reservation.successor_source_identity_ref != successor_source_identity_ref
            or reservation.reservation_ref != reservation_ref
            or reservation.reservation_hash != reservation_hash
        ):
            raise ValueError("source-structure successor reservation binding mismatch")
        return reservation


__all__ = [
    "SourceStructureSuccessorReservationStore",
    "_SOURCE_STRUCTURE_SUCCESSOR_RESERVATION_AUTHORITY",
]
