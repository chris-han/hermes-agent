"""Typed governed semantic source-region contracts."""

from __future__ import annotations

from datetime import timedelta
from hashlib import sha256
import json
from typing import Annotated, Any, Literal
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.canonical_source_ir import SourceLocator
from contracts.evidence_location import EvidenceRef
from contracts.ontology_building import ArtifactPinV1


Sha256 = Annotated[str, Field(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeSourceBindingV1(_Contract):
    schema_version: Literal["runtime_source_binding.v1"]
    source_identity_ref: str = Field(min_length=1)
    source_hash: Sha256
    canonical_source_ir: ArtifactPinV1

    @model_validator(mode="after")
    def canonical_pin_is_source_ir(self) -> "RuntimeSourceBindingV1":
        if self.canonical_source_ir.artifact_kind != "canonical_source_ir":
            raise ValueError(
                "canonical_source_ir pin must have canonical_source_ir kind"
            )
        return self


class RuntimeSourceAnchorV1(_Contract):
    evidence_ref: EvidenceRef
    locator: SourceLocator
    text_hash: Sha256

    @model_validator(mode="after")
    def binds_native_evidence(self) -> "RuntimeSourceAnchorV1":
        evidence = self.evidence_ref
        exact_quote = evidence.selector.exact_quote
        if not isinstance(exact_quote, str) or not exact_quote.strip():
            raise ValueError("source anchor requires a nonblank exact quote")
        if evidence.selector_schema_version != evidence.selector.schema_version:
            raise ValueError("source anchor selector schema version mismatch")
        if evidence.selector_hash != evidence.selector.selector_hash():
            raise ValueError("source anchor selector hash mismatch")
        created_at = evidence.created_at
        if created_at.tzinfo is None or created_at.utcoffset() != timedelta(0):
            raise ValueError("source anchor EvidenceRef created_at must be UTC")
        identity_payload = {
            "organization_id": evidence.organization_id,
            "workspace_id": evidence.workspace_id,
            "source_identity_ref": evidence.source_identity_ref,
            "selector_hash": evidence.selector_hash,
            "evidence_role": evidence.evidence_role,
        }
        expected_ref = (
            "evidence:"
            + sha256(
                json.dumps(
                    identity_payload, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        )
        if evidence.evidence_ref != expected_ref:
            raise ValueError("source anchor EvidenceRef identity mismatch")
        expected_locator = self.locator.model_dump(mode="json", by_alias=True)
        if evidence.provenance.get("source_locator") != expected_locator:
            raise ValueError("source anchor locator provenance mismatch")
        if self.locator.source_identity_ref != evidence.source_identity_ref:
            raise ValueError("source anchor locator SourceIdentity mismatch")
        if self.locator.source_hash != evidence.provenance.get("source_hash"):
            raise ValueError("source anchor locator source hash mismatch")
        expected_text_hash = "sha256:" + sha256(exact_quote.encode("utf-8")).hexdigest()
        if self.text_hash != expected_text_hash:
            raise ValueError("source anchor text hash mismatch")
        return self


_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")


class RuntimeMaterialDependencyV1(_Contract):
    """Explicit source-grounded declaration of evidence required by a decision."""

    dependency_id: str = Field(min_length=1)
    required_evidence_role: str = Field(min_length=1)
    basis_evidence_refs: tuple[str, ...] = Field(min_length=1)
    min_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def stable_and_unique_identity(self) -> "RuntimeMaterialDependencyV1":
        for field in ("dependency_id", "required_evidence_role", *self.basis_evidence_refs):
            if _STABLE_ID.fullmatch(field) is None:
                raise ValueError("material dependency identifiers and evidence refs must be ASCII-stable")
        if len(self.basis_evidence_refs) != len(set(self.basis_evidence_refs)):
            raise ValueError("material dependency basis refs must be unique")
        return self


class RuntimeSemanticSourceRegionV1(_Contract):
    schema_version: Literal["runtime_semantic_source_region.v1"]
    source_region_ref: str = Field(min_length=1)
    text: str = Field(min_length=1)
    text_hash: Sha256
    source_binding: RuntimeSourceBindingV1
    evidence: tuple[RuntimeSourceAnchorV1, ...] = Field(min_length=1)
    material_dependencies: tuple[RuntimeMaterialDependencyV1, ...] = ()

    @model_validator(mode="after")
    def binds_region_evidence(self) -> "RuntimeSemanticSourceRegionV1":
        if not self.source_region_ref.isascii():
            raise ValueError("source_region_ref must be an ASCII-stable reference")
        if not self.text.strip():
            raise ValueError("source region text is required")
        quotes: list[str] = []
        for anchor in self.evidence:
            exact_quote = anchor.evidence_ref.selector.exact_quote
            if exact_quote is None:
                raise ValueError("source anchor requires a nonblank exact quote")
            quotes.append(exact_quote.strip())
        expected_text = "\n".join(quotes)
        if self.text != expected_text:
            raise ValueError("source region text does not match anchor exact quotes")
        expected_hash = "sha256:" + sha256(self.text.encode("utf-8")).hexdigest()
        if self.text_hash != expected_hash:
            raise ValueError("source region text hash mismatch")
        identities = tuple(anchor.evidence_ref.evidence_ref for anchor in self.evidence)
        if len(identities) != len(set(identities)):
            raise ValueError("source region evidence contains a duplicate identity")
        dependencies = self.material_dependencies
        dependency_ids = tuple(item.dependency_id for item in dependencies)
        if len(dependency_ids) != len(set(dependency_ids)):
            raise ValueError("source region material dependency IDs must be unique")
        dependency_definitions = tuple(
            (item.required_evidence_role, tuple(sorted(item.basis_evidence_refs)), item.min_count)
            for item in dependencies
        )
        if len(dependency_definitions) != len(set(dependency_definitions)):
            raise ValueError("source region contains a duplicate material dependency requirement")
        available = set(identities)
        anchors_by_ref = {anchor.evidence_ref.evidence_ref: anchor for anchor in self.evidence}
        for dependency in dependencies:
            if not set(dependency.basis_evidence_refs) <= available:
                raise ValueError("material dependency basis evidence must exist in source region")
            if any(
                anchors_by_ref[ref].evidence_ref.evidence_role == dependency.required_evidence_role
                for ref in dependency.basis_evidence_refs
            ):
                raise ValueError("material dependency basis cannot also satisfy its required evidence role")

        first = self.evidence[0].evidence_ref
        for anchor in self.evidence:
            evidence = anchor.evidence_ref
            if (
                evidence.organization_id != first.organization_id
                or evidence.workspace_id != first.workspace_id
            ):
                raise ValueError("source region evidence scope mismatch")
            if evidence.source_identity_ref != self.source_binding.source_identity_ref:
                raise ValueError("source region EvidenceRef SourceIdentity mismatch")
            provenance = evidence.provenance
            if (
                anchor.locator.source_hash != self.source_binding.source_hash
                or provenance.get("source_hash") != self.source_binding.source_hash
            ):
                raise ValueError("source region source hash binding mismatch")
            if (
                provenance.get("canonical_source_ir_ref")
                != self.source_binding.canonical_source_ir.artifact_ref
                or provenance.get("canonical_source_ir_hash")
                != self.source_binding.canonical_source_ir.artifact_hash
            ):
                raise ValueError("source region canonical IR provenance mismatch")
        return self


def source_region_to_legacy_dict(
    region: RuntimeSemanticSourceRegionV1,
) -> dict[str, Any]:
    """Project one typed source region into the existing inspection contract."""
    return {
        "source_region_ref": region.source_region_ref,
        "text": region.text,
        "source_binding": {
            "sourceIdentitySchemaVersion": "semantier.source_identity.v1",
            "sourceIdentityRef": region.source_binding.source_identity_ref,
            "sourceHash": region.source_binding.source_hash,
            "canonicalSourceIrRef": region.source_binding.canonical_source_ir.artifact_ref,
            "canonicalSourceIrHash": region.source_binding.canonical_source_ir.artifact_hash,
        },
        "evidence_refs": [
            item.evidence_ref.model_dump(mode="json", by_alias=True)
            for item in region.evidence
        ],
        "material_dependencies": [item.model_dump(mode="json") for item in region.material_dependencies],
    }


__all__ = [
    "RuntimeMaterialDependencyV1",
    "RuntimeSemanticSourceRegionV1",
    "RuntimeSourceAnchorV1",
    "RuntimeSourceBindingV1",
    "source_region_to_legacy_dict",
]
