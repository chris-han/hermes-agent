"""Sealed append-only source-structure release finalization store."""

from __future__ import annotations

from pathlib import Path
import re

from contracts.source_structure_release_finalization import (
    SourceStructureReleaseFinalizationV1,
    SourceStructureReleaseReservationV1,
)
from storage.atomic_json_artifact import publish_immutable_json


_SOURCE_STRUCTURE_RELEASE_AUTHORITY = object()
_REF = re.compile(r"^sscc_[0-9a-f]{24}$")
_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")


class SourceStructureReleaseFinalizationStore:
    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if not workspace_id or workspace_id in {".", ".."} or "/" in workspace_id or "\\" in workspace_id:
            raise ValueError("workspace_id must be one safe path segment")
        workspaces = Path(workspace_root).resolve() / "workspaces"
        self.workspace_id = workspace_id
        self.root = (workspaces / workspace_id / "artifacts/schema-discovery/source-structure-release-finalizations").resolve()
        if not self.root.is_relative_to(workspaces.resolve()):
            raise ValueError("workspace_id escapes governed workspace root")

    def _path(self, continuation_ref: str, suffix: str) -> Path:
        if _REF.fullmatch(continuation_ref) is None:
            raise ValueError("continuation_ref must be exact")
        return self.root / f"{continuation_ref}.{suffix}.json"

    def reserve(self, value: SourceStructureReleaseReservationV1, *, _authority: object) -> SourceStructureReleaseReservationV1:
        if _authority is not _SOURCE_STRUCTURE_RELEASE_AUTHORITY:
            raise PermissionError("source-structure release reservation is sealed")
        if type(value) is not SourceStructureReleaseReservationV1 or value.workspace_id != self.workspace_id:
            raise ValueError("source-structure release reservation scope mismatch")
        return SourceStructureReleaseReservationV1.model_validate(publish_immutable_json(
            self._path(value.continuation_ref, "reservation"), value.model_dump(mode="json"),
            conflict_error="SOURCE_STRUCTURE_RELEASE_RESERVATION_CONFLICT",
        ))

    def get_reservation(self, continuation_ref: str, *, continuation_hash: str) -> SourceStructureReleaseReservationV1:
        if _SHA.fullmatch(continuation_hash) is None:
            raise ValueError("continuation_hash must be exact")
        path = self._path(continuation_ref, "reservation")
        if not path.is_file():
            raise ValueError("source-structure release reservation is missing")
        value = SourceStructureReleaseReservationV1.model_validate_json(path.read_text(encoding="utf-8"))
        if value.continuation_ref != continuation_ref or value.continuation_hash != continuation_hash or value.workspace_id != self.workspace_id:
            raise ValueError("source-structure release reservation binding mismatch")
        return value

    def complete(self, value: SourceStructureReleaseFinalizationV1, *, _authority: object) -> SourceStructureReleaseFinalizationV1:
        if _authority is not _SOURCE_STRUCTURE_RELEASE_AUTHORITY:
            raise PermissionError("source-structure release finalization is sealed")
        reservation = self.get_reservation(value.continuation_ref, continuation_hash=value.continuation_hash)
        if value.organization_id != reservation.organization_id or value.workspace_id != reservation.workspace_id or value.reservation_ref != reservation.reservation_ref or value.reservation_hash != reservation.reservation_hash:
            raise ValueError("source-structure release finalization reservation mismatch")
        return SourceStructureReleaseFinalizationV1.model_validate(publish_immutable_json(
            self._path(value.continuation_ref, "finalization"), value.model_dump(mode="json"),
            conflict_error="SOURCE_STRUCTURE_RELEASE_FINALIZATION_CONFLICT",
        ))

    def get_finalization(self, continuation_ref: str, *, continuation_hash: str) -> SourceStructureReleaseFinalizationV1:
        reservation = self.get_reservation(continuation_ref, continuation_hash=continuation_hash)
        path = self._path(continuation_ref, "finalization")
        if not path.is_file():
            raise ValueError("source-structure release finalization is missing")
        value = SourceStructureReleaseFinalizationV1.model_validate_json(path.read_text(encoding="utf-8"))
        if value.continuation_ref != continuation_ref or value.continuation_hash != continuation_hash or value.reservation_ref != reservation.reservation_ref or value.reservation_hash != reservation.reservation_hash:
            raise ValueError("source-structure release finalization binding mismatch")
        return value


__all__ = ["SourceStructureReleaseFinalizationStore", "_SOURCE_STRUCTURE_RELEASE_AUTHORITY"]
