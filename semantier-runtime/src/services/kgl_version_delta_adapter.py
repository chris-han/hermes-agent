"""Read-only governed delta adapter for immutable KGL knowledge versions."""

from __future__ import annotations

from typing import Any

from contracts.governed_delta import (
    ArtifactVersionPinV1,
    DeltaLens,
    DeltaLensResultV1,
    DeltaLensStatus,
    DeltaMateriality,
    DeltaOperationKind,
    DeltaOperationV1,
    DeltaTargetType,
)
from eos.artifact_hashing import content_hash
from services.governed_delta_service import (
    GovernedDeltaError,
    GovernedDeltaService,
    canonicalize_setlike,
)


_REQUIRED_KV_FIELDS = {
    "knowledge_version_id",
    "active_artifacts_json",
    "source_hashes_json",
    "extracted_claims_json",
    "extracted_constraints_json",
    "extracted_projection_rules_json",
    "effective_from",
    "effective_to",
    "content_hash",
}


def _owner_hash(value: str) -> str:
    raw = str(value or "").strip()
    if raw.startswith("sha256:") and len(raw) == 71:
        return raw
    if len(raw) == 64 and all(ch in "0123456789abcdef" for ch in raw):
        return "sha256:" + raw
    raise GovernedDeltaError("INVALID_KV_OWNER_CONTENT_HASH")


def _stable_collection(values: Any, field: str) -> tuple[Any, ...]:
    if not isinstance(values, (list, tuple)):
        raise GovernedDeltaError(f"INVALID_KV_{field.upper()}")
    return canonicalize_setlike(values)


def _normalize_authority_manifest(row: dict[str, Any]) -> tuple[Any, ...] | None:
    if "authority_source_manifest_json" not in row or row["authority_source_manifest_json"] is None:
        return None
    return _stable_collection(row["authority_source_manifest_json"], "authority_source_manifest")


def _comparison_payload(row: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(_REQUIRED_KV_FIELDS - set(row))
    if missing:
        raise GovernedDeltaError("K_V_FIELDS_MISSING:" + ",".join(missing))
    version_id = str(row["knowledge_version_id"] or "").strip()
    if not version_id or not version_id.isascii():
        raise GovernedDeltaError("INVALID_KV_VERSION_ID")
    return {
        "knowledge_version_id": version_id,
        "owner_content_hash": _owner_hash(str(row["content_hash"])),
        "active_artifacts": _stable_collection(row["active_artifacts_json"], "active_artifacts"),
        "source_hashes": _stable_collection(row["source_hashes_json"], "source_hashes"),
        "authority_source_manifest": _normalize_authority_manifest(row),
        "extracted_claims": _stable_collection(row["extracted_claims_json"], "extracted_claims"),
        "extracted_constraints": _stable_collection(
            row["extracted_constraints_json"], "extracted_constraints"
        ),
        "extracted_projection_rules": _stable_collection(
            row["extracted_projection_rules_json"], "extracted_projection_rules"
        ),
        "effective_from": row["effective_from"],
        "effective_to": row["effective_to"],
    }


def _value_ref(prefix: str, value: Any) -> str:
    return f"{prefix}:sha256:{content_hash(value)}"


def _operation(
    *,
    lens: DeltaLens,
    subject_ref: str,
    before: Any,
    after: Any,
    changed_fields: tuple[str, ...],
    materiality: DeltaMateriality = DeltaMateriality.MATERIAL,
) -> DeltaOperationV1:
    identity_payload = {
        "lens": lens.value,
        "subject_ref": subject_ref,
        "before": before,
        "after": after,
        "changed_fields": changed_fields,
    }
    op_hash = content_hash(identity_payload)
    return DeltaOperationV1(
        operation_id=f"delta-op:{op_hash[:24]}",
        operation_kind=DeltaOperationKind.MODIFY,
        lens=lens,
        subject_ref=subject_ref,
        before_ref=_value_ref("before", before),
        after_ref=_value_ref("after", after),
        changed_fields=tuple(sorted(changed_fields)),
        materiality=materiality,
    )


class KglVersionDeltaAdapter:
    """Compare two owner-native KGL version rows without mutating KGL state."""

    target_type = DeltaTargetType.KNOWLEDGE_GRAPH

    def __init__(
        self,
        *,
        artifact_id: str,
        owner_ref: str = "kgl-store",
        organization_id: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        if not artifact_id or not artifact_id.isascii():
            raise GovernedDeltaError("INVALID_KV_ARTIFACT_ID")
        self.artifact_id = artifact_id
        self.owner_ref = owner_ref
        self.organization_id = organization_id
        self.workspace_id = workspace_id

    def normalize_version(self, row: dict[str, Any]) -> tuple[ArtifactVersionPinV1, dict[str, Any]]:
        normalized = _comparison_payload(row)
        pin_hash = "sha256:" + content_hash(normalized)
        pin = ArtifactVersionPinV1(
            artifact_type="knowledge_graph",
            artifact_id=self.artifact_id,
            version_id=normalized["knowledge_version_id"],
            content_hash=pin_hash,
            owner_ref=self.owner_ref,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            effective_from=normalized["effective_from"],
            effective_to=normalized["effective_to"],
        )
        return pin, normalized

    def validate_pin(self, row: dict[str, Any], pin: ArtifactVersionPinV1) -> None:
        expected, _ = self.normalize_version(row)
        if pin != expected:
            raise GovernedDeltaError("K_V_COMPARISON_PIN_MISMATCH")

    def compute_lenses(
        self,
        *,
        predecessor: dict[str, Any],
        successor: dict[str, Any],
        historical_binding_version_id: str | None = None,
    ) -> dict[DeltaLens, DeltaLensResultV1]:
        _, before = self.normalize_version(predecessor)
        _, after = self.normalize_version(successor)

        structural_fields = ("active_artifacts",)
        semantic_fields = (
            "extracted_claims",
            "extracted_constraints",
            "extracted_projection_rules",
        )
        authority_fields = (
            "authority_source_manifest",
            "effective_from",
            "effective_to",
        )
        evidence_fields = ("source_hashes", "active_artifacts")

        lenses: dict[DeltaLens, DeltaLensResultV1] = {}

        def available_diff(lens: DeltaLens, fields: tuple[str, ...]) -> DeltaLensResultV1:
            changed = tuple(field for field in fields if before[field] != after[field])
            operations: tuple[DeltaOperationV1, ...] = ()
            if changed:
                operations = (
                    _operation(
                        lens=lens,
                        subject_ref=f"kgl:{self.artifact_id}",
                        before={field: before[field] for field in changed},
                        after={field: after[field] for field in changed},
                        changed_fields=changed,
                    ),
                )
            return DeltaLensResultV1(
                lens=lens,
                status=DeltaLensStatus.AVAILABLE,
                operations=operations,
                basis_refs=tuple(
                    sorted(
                        (
                            f"kgl-version:{before['knowledge_version_id']}",
                            f"kgl-version:{after['knowledge_version_id']}",
                        )
                    )
                ),
            )

        lenses[DeltaLens.STRUCTURAL] = available_diff(DeltaLens.STRUCTURAL, structural_fields)
        lenses[DeltaLens.SEMANTIC] = available_diff(DeltaLens.SEMANTIC, semantic_fields)

        if before["authority_source_manifest"] is None or after["authority_source_manifest"] is None:
            lenses[DeltaLens.AUTHORITY] = DeltaLensResultV1(
                lens=DeltaLens.AUTHORITY,
                status=DeltaLensStatus.UNAVAILABLE,
                operations=(),
                basis_refs=(),
            )
        else:
            lenses[DeltaLens.AUTHORITY] = available_diff(DeltaLens.AUTHORITY, authority_fields)

        lenses[DeltaLens.BEHAVIOR] = DeltaLensResultV1(
            lens=DeltaLens.BEHAVIOR,
            status=DeltaLensStatus.NOT_APPLICABLE,
            operations=(),
            basis_refs=(),
        )
        lenses[DeltaLens.EVIDENCE] = available_diff(DeltaLens.EVIDENCE, evidence_fields)

        binding_before = historical_binding_version_id or before["knowledge_version_id"]
        binding_after = after["knowledge_version_id"]
        binding_ops: tuple[DeltaOperationV1, ...] = ()
        if binding_before != binding_after:
            binding_ops = (
                _operation(
                    lens=DeltaLens.EXECUTION_BINDING,
                    subject_ref=f"kgl-binding:{self.artifact_id}",
                    before=binding_before,
                    after=binding_after,
                    changed_fields=("knowledge_version_id",),
                ),
            )
        lenses[DeltaLens.EXECUTION_BINDING] = DeltaLensResultV1(
            lens=DeltaLens.EXECUTION_BINDING,
            status=DeltaLensStatus.AVAILABLE,
            operations=binding_ops,
            basis_refs=tuple(
                sorted(
                    (
                        f"kgl-version:{before['knowledge_version_id']}",
                        f"kgl-version:{after['knowledge_version_id']}",
                    )
                )
            ),
        )
        lenses[DeltaLens.PRESENTATION] = DeltaLensResultV1(
            lens=DeltaLens.PRESENTATION,
            status=DeltaLensStatus.NOT_APPLICABLE,
            operations=(),
            basis_refs=(),
        )
        return lenses

    def build_envelope(
        self,
        *,
        delta_id: str,
        predecessor: dict[str, Any],
        successor: dict[str, Any],
        historical_binding_version_id: str | None = None,
    ):
        predecessor_pin, _ = self.normalize_version(predecessor)
        successor_pin, _ = self.normalize_version(successor)
        lenses = self.compute_lenses(
            predecessor=predecessor,
            successor=successor,
            historical_binding_version_id=historical_binding_version_id,
        )
        return GovernedDeltaService.build_envelope(
            delta_id=delta_id,
            target_type=self.target_type,
            predecessor=predecessor_pin,
            successor_candidate=successor_pin,
            lens_results=lenses,
        )


__all__ = ["KglVersionDeltaAdapter"]
