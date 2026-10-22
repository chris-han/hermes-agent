"""Sealed append-only persistence for canonical-pattern correction evidence."""

from __future__ import annotations

from pathlib import Path
import re

from contracts.canonical_pattern_correction import (
    CanonicalPatternCorrectionFinalizationV1,
    CanonicalPatternCorrectionReservationV1,
)
from storage.atomic_json_artifact import publish_immutable_json


_CANONICAL_PATTERN_CORRECTION_AUTHORITY = object()
_RESERVATION_REF = re.compile(r"^cpcr_[0-9a-f]{24}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _root(workspace_root: str | Path, workspace_id: str) -> Path:
    if (
        not isinstance(workspace_id, str)
        or not workspace_id
        or workspace_id in {".", ".."}
        or "/" in workspace_id
        or "\\" in workspace_id
    ):
        raise ValueError("workspace_id must be one safe path segment")
    workspaces = Path(workspace_root).resolve() / "workspaces"
    result = (
        workspaces / workspace_id
        / "artifacts/schema-discovery/canonical-pattern-corrections"
    ).resolve()
    if not result.is_relative_to(workspaces.resolve()):
        raise ValueError("workspace_id escapes governed workspace root")
    return result


class CanonicalPatternCorrectionStore:
    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self.root = _root(workspace_root, workspace_id)

    def _path(self, reservation_ref: str, suffix: str) -> Path:
        if _RESERVATION_REF.fullmatch(reservation_ref) is None:
            raise ValueError("reservation_ref must be exact")
        return self.root / f"{reservation_ref}.{suffix}.json"

    def reserve(
        self,
        reservation: CanonicalPatternCorrectionReservationV1,
        *,
        _authority: object,
    ) -> CanonicalPatternCorrectionReservationV1:
        if _authority is not _CANONICAL_PATTERN_CORRECTION_AUTHORITY:
            raise PermissionError("canonical-pattern correction publication is sealed")
        if reservation.workspace_id != self.workspace_id:
            raise ValueError("canonical-pattern correction workspace mismatch")
        path = self._path(reservation.reservation_ref, "reservation")
        if path.exists():
            existing = CanonicalPatternCorrectionReservationV1.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if existing != reservation:
                raise ValueError("canonical-pattern reservation conflict")
            return existing
        return CanonicalPatternCorrectionReservationV1.model_validate(
            publish_immutable_json(
                path,
                reservation.model_dump(mode="json"),
                conflict_error="CANONICAL_PATTERN_CORRECTION_RESERVATION_CONFLICT",
            )
        )

    def complete(
        self,
        finalization: CanonicalPatternCorrectionFinalizationV1,
        *,
        _authority: object,
    ) -> CanonicalPatternCorrectionFinalizationV1:
        if _authority is not _CANONICAL_PATTERN_CORRECTION_AUTHORITY:
            raise PermissionError("canonical-pattern correction publication is sealed")
        reservation = self.get_reservation(
            finalization.reservation_ref,
            reservation_hash=finalization.reservation_hash,
        )
        if (
            finalization.organization_id != reservation.organization_id
            or finalization.workspace_id != self.workspace_id
        ):
            raise ValueError("canonical-pattern finalization reservation mismatch")
        return CanonicalPatternCorrectionFinalizationV1.model_validate(
            publish_immutable_json(
                self._path(finalization.reservation_ref, "finalization"),
                finalization.model_dump(mode="json"),
                conflict_error="CANONICAL_PATTERN_CORRECTION_FINALIZATION_CONFLICT",
            )
        )

    def get_reservation(
        self, reservation_ref: str, *, reservation_hash: str
    ) -> CanonicalPatternCorrectionReservationV1:
        if _SHA256.fullmatch(reservation_hash) is None:
            raise ValueError("reservation_hash must be exact")
        path = self._path(reservation_ref, "reservation")
        if not path.is_file():
            raise ValueError("canonical-pattern correction reservation is missing")
        value = CanonicalPatternCorrectionReservationV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if (
            value.reservation_ref != reservation_ref
            or value.reservation_hash != reservation_hash
            or value.workspace_id != self.workspace_id
        ):
            raise ValueError("canonical-pattern reservation path binding mismatch")
        return value

    def get_verified_pair(
        self, reservation_ref: str, *, reservation_hash: str
    ) -> tuple[
        CanonicalPatternCorrectionReservationV1,
        CanonicalPatternCorrectionFinalizationV1,
    ]:
        reservation = self.get_reservation(
            reservation_ref, reservation_hash=reservation_hash
        )
        path = self._path(reservation_ref, "finalization")
        if not path.is_file():
            raise ValueError("canonical-pattern correction finalization is missing")
        finalization = CanonicalPatternCorrectionFinalizationV1.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if (
            finalization.reservation_ref != reservation_ref
            or finalization.reservation_hash != reservation_hash
            or finalization.organization_id != reservation.organization_id
            or finalization.workspace_id != reservation.workspace_id
        ):
            raise ValueError("canonical-pattern finalization path binding mismatch")
        return reservation, finalization

    def find_completed_review(
        self,
        *,
        predecessor_release_ref: str,
        predecessor_release_hash: str,
        candidate_graph_id: str,
        human_label_event_ref: str,
    ) -> tuple[
        CanonicalPatternCorrectionReservationV1,
        CanonicalPatternCorrectionFinalizationV1,
    ] | None:
        """Find the one immutable completion for an exact retry key."""
        matches: list[
            tuple[
                CanonicalPatternCorrectionReservationV1,
                CanonicalPatternCorrectionFinalizationV1,
            ]
        ] = []
        for path in sorted(self.root.glob("cpcr_*.reservation.json")):
            reservation = CanonicalPatternCorrectionReservationV1.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if (
                reservation.predecessor_release_ref == predecessor_release_ref
                and reservation.predecessor_release_hash == predecessor_release_hash
                and reservation.candidate_graph_id == candidate_graph_id
                and reservation.human_label_event_ref == human_label_event_ref
            ):
                finalization_path = self._path(
                    reservation.reservation_ref, "finalization"
                )
                if finalization_path.is_file():
                    matches.append(
                        self.get_verified_pair(
                            reservation.reservation_ref,
                            reservation_hash=reservation.reservation_hash,
                        )
                    )
        if len(matches) > 1:
            raise ValueError("canonical-pattern correction retry is ambiguous")
        return matches[0] if matches else None

    def find_reservation_for_review(
        self,
        *,
        predecessor_release_ref: str,
        predecessor_release_hash: str,
        candidate_graph_id: str,
        human_label_event_ref: str,
    ) -> CanonicalPatternCorrectionReservationV1 | None:
        matches: list[CanonicalPatternCorrectionReservationV1] = []
        for path in sorted(self.root.glob("cpcr_*.reservation.json")):
            reservation = CanonicalPatternCorrectionReservationV1.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if (
                reservation.predecessor_release_ref == predecessor_release_ref
                and reservation.predecessor_release_hash == predecessor_release_hash
                and reservation.candidate_graph_id == candidate_graph_id
                and reservation.human_label_event_ref == human_label_event_ref
            ):
                matches.append(reservation)
        if len(matches) > 1:
            raise ValueError("canonical-pattern correction retry is ambiguous")
        return matches[0] if matches else None


__all__ = [
    "CanonicalPatternCorrectionStore",
    "_CANONICAL_PATTERN_CORRECTION_AUTHORITY",
]
