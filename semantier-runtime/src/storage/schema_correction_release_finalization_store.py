"""Sealed append-only storage for schema-correction release evidence."""

from __future__ import annotations

from pathlib import Path
import re

from contracts.schema_correction_release_finalization import (
    SchemaCorrectionReleaseFinalizationV1,
    SchemaCorrectionReleaseReservationV1,
)
from storage.atomic_json_artifact import publish_immutable_json


_SCHEMA_RELEASE_FINALIZATION_AUTHORITY = object()
_CONTINUATION_REF = re.compile(r"^scc_[0-9a-f]{24}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _safe_root(workspace_root: str | Path, workspace_id: str) -> Path:
    if (
        not isinstance(workspace_id, str)
        or not workspace_id
        or workspace_id in {".", ".."}
        or "/" in workspace_id
        or "\\" in workspace_id
    ):
        raise ValueError("workspace_id must be one safe path segment")
    workspaces = Path(workspace_root).resolve() / "workspaces"
    root = (workspaces / workspace_id / "artifacts").resolve()
    if not root.is_relative_to(workspaces.resolve()):
        raise ValueError("workspace_id escapes governed workspace root")
    return root


def _require_continuation_ref(value: str) -> str:
    if _CONTINUATION_REF.fullmatch(value) is None:
        raise ValueError("continuation_ref must be exact")
    return value


class SchemaCorrectionReleaseFinalizationStore:
    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self.root = (
            _safe_root(workspace_root, workspace_id)
            / "schema-discovery/schema-correction-release-finalizations"
        )

    def _path(self, continuation_ref: str, suffix: str) -> Path:
        return self.root / f"{_require_continuation_ref(continuation_ref)}.{suffix}.json"

    def reserve(
        self,
        reservation: SchemaCorrectionReleaseReservationV1,
        *,
        _authority: object,
    ) -> SchemaCorrectionReleaseReservationV1:
        if _authority is not _SCHEMA_RELEASE_FINALIZATION_AUTHORITY:
            raise PermissionError("schema release reservation publication is sealed")
        if reservation.workspace_id != self.workspace_id:
            raise ValueError("schema release reservation workspace mismatch")
        path = self._path(reservation.continuation_ref, "reservation")
        if path.exists():
            existing = SchemaCorrectionReleaseReservationV1.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if existing != reservation:
                raise ValueError("schema continuation already has a different reservation")
            return existing
        return SchemaCorrectionReleaseReservationV1.model_validate(
            publish_immutable_json(
                path,
                reservation.model_dump(mode="json"),
                conflict_error="SCHEMA_RELEASE_RESERVATION_CONFLICT",
            )
        )

    def get_reservation(
        self, continuation_ref: str, *, continuation_hash: str
    ) -> SchemaCorrectionReleaseReservationV1:
        if _SHA256.fullmatch(continuation_hash) is None:
            raise ValueError("continuation_hash must be exact")
        path = self._path(continuation_ref, "reservation")
        if not path.is_file():
            raise ValueError("schema release reservation is missing")
        reservation = SchemaCorrectionReleaseReservationV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if (
            reservation.workspace_id != self.workspace_id
            or reservation.continuation_ref != continuation_ref
            or reservation.continuation_hash != continuation_hash
        ):
            raise ValueError("schema release reservation path binding mismatch")
        return reservation

    def complete(
        self,
        continuation_ref: str,
        finalization: SchemaCorrectionReleaseFinalizationV1,
        *,
        _authority: object,
    ) -> SchemaCorrectionReleaseFinalizationV1:
        if _authority is not _SCHEMA_RELEASE_FINALIZATION_AUTHORITY:
            raise PermissionError("schema release finalization publication is sealed")
        reservation = self.get_reservation(
            continuation_ref, continuation_hash=finalization.continuation_hash
        )
        if (
            finalization.workspace_id != self.workspace_id
            or finalization.organization_id != reservation.organization_id
            or finalization.continuation_ref != reservation.continuation_ref
            or finalization.reservation_ref != reservation.reservation_ref
            or finalization.reservation_hash != reservation.reservation_hash
        ):
            raise ValueError("schema release finalization reservation mismatch")
        return SchemaCorrectionReleaseFinalizationV1.model_validate(
            publish_immutable_json(
                self._path(continuation_ref, "finalization"),
                finalization.model_dump(mode="json"),
                conflict_error="SCHEMA_RELEASE_FINALIZATION_CONFLICT",
            )
        )

    def get_finalization(
        self, continuation_ref: str, *, continuation_hash: str
    ) -> SchemaCorrectionReleaseFinalizationV1:
        reservation = self.get_reservation(
            continuation_ref, continuation_hash=continuation_hash
        )
        path = self._path(continuation_ref, "finalization")
        if not path.is_file():
            raise ValueError("schema release finalization is missing")
        finalization = SchemaCorrectionReleaseFinalizationV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if (
            finalization.workspace_id != self.workspace_id
            or finalization.organization_id != reservation.organization_id
            or finalization.continuation_ref != continuation_ref
            or finalization.continuation_hash != continuation_hash
            or finalization.reservation_ref != reservation.reservation_ref
            or finalization.reservation_hash != reservation.reservation_hash
        ):
            raise ValueError("schema release finalization path binding mismatch")
        return finalization


__all__ = [
    "SchemaCorrectionReleaseFinalizationStore",
    "_SCHEMA_RELEASE_FINALIZATION_AUTHORITY",
]
