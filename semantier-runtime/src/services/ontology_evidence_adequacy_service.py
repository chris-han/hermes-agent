"""Deterministic S0 canonical-evidence adequacy assessment."""

from __future__ import annotations

from datetime import datetime

from contracts.ontology_building import (
    CanonicalEvidenceAdequacyAssessmentV1,
    CanonicalEvidenceBoundaryV1,
    EvidenceAdequacyFindingV1,
)
from eos.artifact_hashing import content_hash


class OntologyEvidenceAdequacyService:
    _CRITICAL_REGIONS = frozenset({"scope", "rule", "qualification", "exception", "effectivity"})

    def assess(
        self,
        *,
        boundary: CanonicalEvidenceBoundaryV1,
        source_authority_role: str,
        available_definition_regions: tuple[str, ...],
        missing_normative_dependencies: tuple[str, ...],
        search_evidence_refs: tuple[str, ...],
        assessed_at: datetime,
        authoritative_dependency_refs: tuple[str, ...] = (),
    ) -> CanonicalEvidenceAdequacyAssessmentV1:
        if not search_evidence_refs:
            raise ValueError("adequacy assessment requires exact source-search evidence")
        findings: list[EvidenceAdequacyFindingV1] = []
        direct_authority = source_authority_role in {
            "governed_internal_policy",
            "authoritative_source",
        }
        delegated_authority = (
            source_authority_role == "interpretive_with_authoritative_dependencies"
            and bool(authoritative_dependency_refs)
        )
        if not (direct_authority or delegated_authority):
            findings.append(EvidenceAdequacyFindingV1(
                finding_id="adequacy:unresolved-authority-role",
                finding_type="unresolved_source_authority_role", severity="blocking",
                message="The canonical source authority role is unresolved.",
                evidence_refs=(boundary.source_identity_ref,), search_evidence_refs=search_evidence_refs,
            ))
        for dependency in sorted(set(missing_normative_dependencies)):
            findings.append(EvidenceAdequacyFindingV1(
                finding_id=f"adequacy:missing-dependency:{dependency}",
                finding_type="missing_normative_dependency", severity="blocking",
                message=f"A declared normative dependency is unavailable: {dependency}",
                evidence_refs=(boundary.source_identity_ref,), search_evidence_refs=search_evidence_refs,
            ))
        missing_regions = sorted(self._CRITICAL_REGIONS - set(available_definition_regions))
        for region in missing_regions:
            findings.append(EvidenceAdequacyFindingV1(
                finding_id=f"adequacy:missing-definition:{region}",
                finding_type="missing_critical_definition_region", severity="blocking",
                message=f"The canonical source has no evidenced {region} region.",
                evidence_refs=(boundary.canonical_source_ir_ref,), search_evidence_refs=search_evidence_refs,
            ))

        basis = {
            "organization_id": boundary.organization_id,
            "workspace_id": boundary.workspace_id,
            "boundary_ref": boundary.boundary_ref,
            "boundary_hash": boundary.boundary_hash,
            "findings": [item.model_dump(mode="json") for item in findings],
        }
        digest = content_hash(basis)
        return CanonicalEvidenceAdequacyAssessmentV1(
            assessment_ref=f"evidence-adequacy:{digest}",
            assessment_hash=f"sha256:{digest}",
            assessed_at=assessed_at,
            **basis,
        )


__all__ = ["OntologyEvidenceAdequacyService"]
