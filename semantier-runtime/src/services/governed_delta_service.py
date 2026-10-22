"""Deterministic shared mechanics for governed delta envelopes.

This service validates exact version pins and canonicalizes adapter-owned lens
results. It never creates ChangeCandidate, admission, activation, or execution
authority.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, Protocol

from contracts.governed_delta import (
    AffectedArtifactV1,
    AffectedInvariantV1,
    ArtifactVersionPinV1,
    DeltaAncestryV1,
    DeltaEnvelopeV1,
    DeltaLens,
    DeltaLensResultV1,
    DeltaLensStatus,
    DeltaLifecycleState,
    DeltaMateriality,
    DeltaTargetType,
    DeltaTransitionReceiptV1,
    ImpactProjectionV1,
    MaterialChangeSummaryV1,
    RequiredQualificationV1,
)
from eos.artifact_hashing import canonical_json, content_hash


class GovernedDeltaError(ValueError):
    pass


class GovernedDeltaAdapter(Protocol):
    target_type: DeltaTargetType

    def normalize_version(self, source: object) -> tuple[ArtifactVersionPinV1, Any]: ...

    def compute_lenses(
        self,
        *,
        predecessor: object,
        successor: object,
    ) -> Mapping[DeltaLens, DeltaLensResultV1]: ...


def canonicalize_setlike(values: Iterable[Any]) -> tuple[Any, ...]:
    """Return a deterministic tuple for a semantically unordered collection."""
    keyed = [(canonical_json(value), value) for value in values]
    keyed.sort(key=lambda pair: pair[0].encode("utf-8"))
    if any(keyed[index][0] == keyed[index - 1][0] for index in range(1, len(keyed))):
        raise GovernedDeltaError("DUPLICATE_CANONICAL_ITEM")
    return tuple(value for _, value in keyed)


def assert_pin_matches_payload(pin: ArtifactVersionPinV1, payload: Any) -> None:
    actual_hash = "sha256:" + content_hash(payload)
    if pin.content_hash != actual_hash:
        raise GovernedDeltaError("ARTIFACT_CONTENT_HASH_MISMATCH")


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise GovernedDeltaError("INVALID_OCCURRED_AT") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise GovernedDeltaError("INVALID_OCCURRED_AT")
    return parsed


def _model_sort_key(value: Any) -> bytes:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return canonical_json(payload).encode("utf-8")


def _canonicalize_lens_result(result: DeltaLensResultV1) -> DeltaLensResultV1:
    operations = tuple(sorted(result.operations, key=lambda item: item.operation_id.encode("utf-8")))
    basis_refs = tuple(sorted(set(result.basis_refs), key=lambda item: item.encode("utf-8")))
    return result.model_copy(update={"operations": operations, "basis_refs": basis_refs})


def _material_summary(lenses: tuple[DeltaLensResultV1, ...]) -> MaterialChangeSummaryV1:
    material: list[DeltaLens] = []
    unchanged: list[DeltaLens] = []
    unavailable: list[DeltaLens] = []
    for item in lenses:
        if item.status is not DeltaLensStatus.AVAILABLE:
            unavailable.append(item.lens)
            continue
        if any(op.materiality in {DeltaMateriality.MATERIAL, DeltaMateriality.UNKNOWN} for op in item.operations):
            material.append(item.lens)
        else:
            unchanged.append(item.lens)
    return MaterialChangeSummaryV1(
        material_lenses=tuple(material),
        unchanged_lenses=tuple(unchanged),
        unavailable_lenses=tuple(unavailable),
    )


class GovernedDeltaService:
    """Build and validate one deterministic cross-artifact delta envelope."""

    @staticmethod
    def build_envelope(
        *,
        delta_id: str,
        target_type: DeltaTargetType,
        predecessor: ArtifactVersionPinV1,
        successor_candidate: ArtifactVersionPinV1,
        lens_results: Mapping[DeltaLens, DeltaLensResultV1],
        ancestry: DeltaAncestryV1 | None = None,
    ) -> DeltaEnvelopeV1:
        if not delta_id or not delta_id.isascii():
            raise GovernedDeltaError("INVALID_DELTA_ID")
        if predecessor.artifact_type != successor_candidate.artifact_type:
            raise GovernedDeltaError("ARTIFACT_TYPE_MISMATCH")
        if predecessor.artifact_id != successor_candidate.artifact_id:
            raise GovernedDeltaError("ARTIFACT_ID_MISMATCH")

        missing = [lens for lens in DeltaLens if lens not in lens_results]
        extra = [lens for lens in lens_results if lens not in set(DeltaLens)]
        if missing or extra:
            raise GovernedDeltaError("INCOMPLETE_LENS_SET")

        ordered: list[DeltaLensResultV1] = []
        for lens in DeltaLens:
            result = lens_results[lens]
            if result.lens is not lens:
                raise GovernedDeltaError("LENS_IDENTITY_MISMATCH")
            ordered.append(_canonicalize_lens_result(result))
        lenses = tuple(ordered)
        summary = _material_summary(lenses)

        payload = {
            "schema_version": "governed_delta_envelope.v1",
            "delta_id": delta_id,
            "target_type": target_type.value,
            "predecessor": predecessor.model_dump(mode="json"),
            "successor_candidate": successor_candidate.model_dump(mode="json"),
            "ancestry": (ancestry or DeltaAncestryV1()).model_dump(mode="json"),
            "lenses": [item.model_dump(mode="json") for item in lenses],
            "material_change_summary": summary.model_dump(mode="json"),
        }
        envelope_hash = "sha256:" + content_hash(payload)
        return DeltaEnvelopeV1.model_validate({**payload, "content_hash": envelope_hash})

    @staticmethod
    def build_impact_projection(
        *,
        envelope: DeltaEnvelopeV1,
        affected_artifacts: Iterable[AffectedArtifactV1] = (),
        affected_invariants: Iterable[AffectedInvariantV1] = (),
        required_qualification: Iterable[RequiredQualificationV1] = (),
    ) -> ImpactProjectionV1:
        artifacts = tuple(sorted(tuple(affected_artifacts), key=_model_sort_key))
        invariants = tuple(sorted(tuple(affected_invariants), key=_model_sort_key))
        qualifications = tuple(sorted(tuple(required_qualification), key=_model_sort_key))
        payload = {
            "schema_version": "governed_delta_impact_projection.v1",
            "delta_ref": envelope.delta_id,
            "delta_hash": envelope.content_hash,
            "affected_artifacts": [item.model_dump(mode="json") for item in artifacts],
            "affected_invariants": [item.model_dump(mode="json") for item in invariants],
            "required_qualification": [item.model_dump(mode="json") for item in qualifications],
        }
        projection_hash = "sha256:" + content_hash(payload)
        return ImpactProjectionV1.model_validate({**payload, "content_hash": projection_hash})

    @staticmethod
    def build_transition_receipt(
        *,
        envelope: DeltaEnvelopeV1,
        lifecycle_state: DeltaLifecycleState,
        owner_ref: str,
        occurred_at: str,
        change_candidate_ref: str | None = None,
        assessment_ref: str | None = None,
        admission_receipt_ref: str | None = None,
        activation_receipt_ref: str | None = None,
    ) -> DeltaTransitionReceiptV1:
        _parse_utc(occurred_at)
        if lifecycle_state is DeltaLifecycleState.CHANGE_ASSESSED and (
            change_candidate_ref is None or assessment_ref is None
        ):
            raise GovernedDeltaError("CHANGE_ASSESSED_REQUIRES_CANDIDATE_AND_ASSESSMENT")
        payload = {
            "schema_version": "governed_delta_transition_receipt.v1",
            "delta_ref": envelope.delta_id,
            "predecessor_ref": f"{envelope.predecessor.artifact_id}@{envelope.predecessor.version_id}",
            "successor_ref": (
                f"{envelope.successor_candidate.artifact_id}@"
                f"{envelope.successor_candidate.version_id}"
            ),
            "change_candidate_ref": change_candidate_ref,
            "assessment_ref": assessment_ref,
            "admission_receipt_ref": admission_receipt_ref,
            "activation_receipt_ref": activation_receipt_ref,
            "lifecycle_state": lifecycle_state.value,
            "owner_ref": owner_ref,
            "occurred_at": occurred_at,
        }
        receipt_hash = "sha256:" + content_hash(payload)
        return DeltaTransitionReceiptV1.model_validate({**payload, "content_hash": receipt_hash})

    @staticmethod
    def order_transition_receipts(
        receipts: Iterable[DeltaTransitionReceiptV1],
    ) -> tuple[DeltaTransitionReceiptV1, ...]:
        items = tuple(receipts)
        hashes = tuple(item.content_hash for item in items)
        if len(hashes) != len(set(hashes)):
            raise GovernedDeltaError("DUPLICATE_TRANSITION_RECEIPT")
        for item in items:
            _parse_utc(item.occurred_at)
        return tuple(
            sorted(
                items,
                key=lambda item: (
                    _parse_utc(item.occurred_at),
                    item.content_hash.encode("utf-8"),
                ),
            )
        )


__all__ = [
    "GovernedDeltaAdapter",
    "GovernedDeltaError",
    "GovernedDeltaService",
    "assert_pin_matches_payload",
    "canonicalize_setlike",
]
