from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eos.authority_validation import (
    parse_aware_iso,
    validate_effectivity_window,
    validate_source_provenance,
)
from semantier.sensitive_governance_contracts.hashing import canonical_hash


@dataclass(frozen=True)
class LegalAuthorityRegistry:
    artifact_id: str
    jurisdiction: str
    effective_from: str
    effective_to: str | None
    references: list[dict[str, Any]]
    provenance_hash: str
    registry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "jurisdiction": self.jurisdiction,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "references": self.references,
            "provenance_hash": self.provenance_hash,
            "registry_hash": self.registry_hash,
        }


def _parse_aware_iso(value: str, *, field: str) -> datetime:
    return parse_aware_iso(value, field=field)


def _validate_effectivity(payload: dict[str, Any], *, prefix: str) -> tuple[str, str | None]:
    return validate_effectivity_window(payload, prefix=prefix)


def _validate_provenance(payload: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    return validate_source_provenance(payload, prefix=prefix)


def load_t2_legal_authority_registry(
    *,
    artifact: dict[str, Any],
    bundle_manifest: dict[str, Any],
) -> LegalAuthorityRegistry:
    if artifact.get("tier") != "T2":
        raise ValueError("T2 legal authority artifact required")
    if artifact.get("lifecycle_state") != "certified":
        raise ValueError("T2 legal authority artifact is not certified")
    jurisdiction = str(artifact.get("jurisdiction") or "").strip()
    if not jurisdiction:
        raise ValueError("T2 legal authority jurisdiction is required")
    if jurisdiction != str(bundle_manifest.get("jurisdiction") or "").strip():
        raise ValueError("T2 legal authority jurisdiction does not match bundle")
    effective_from, effective_to = _validate_effectivity(
        artifact,
        prefix="T2 legal authority",
    )
    manifest_from, manifest_to = _validate_effectivity(
        bundle_manifest,
        prefix="sensitive governance bundle",
    )
    artifact_from_dt = _parse_aware_iso(effective_from, field="T2 effective_from")
    manifest_from_dt = _parse_aware_iso(manifest_from, field="bundle effective_from")
    if artifact_from_dt > manifest_from_dt:
        raise ValueError("T2 legal authority is not effective when bundle activates")
    if effective_to and manifest_to:
        artifact_to_dt = _parse_aware_iso(effective_to, field="T2 effective_to")
        manifest_to_dt = _parse_aware_iso(manifest_to, field="bundle effective_to")
        if artifact_to_dt < manifest_to_dt:
            raise ValueError("T2 legal authority expires before bundle effectivity ends")
    provenance = _validate_provenance(artifact, prefix="T2 legal authority")
    references = artifact.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("T2 legal authority references are required")
    for reference in references:
        if not isinstance(reference, dict):
            raise ValueError("T2 legal authority reference must be an object")
        if not str(reference.get("authority_ref") or "").strip():
            raise ValueError("T2 legal authority reference missing authority_ref")
    registry_payload = {
        "artifact_id": artifact["artifact_id"],
        "jurisdiction": jurisdiction,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "references": references,
        "provenance_hash": canonical_hash(provenance),
    }
    return LegalAuthorityRegistry(
        artifact_id=str(artifact["artifact_id"]),
        jurisdiction=jurisdiction,
        effective_from=effective_from,
        effective_to=effective_to,
        references=references,
        provenance_hash=registry_payload["provenance_hash"],
        registry_hash=canonical_hash(registry_payload),
    )
