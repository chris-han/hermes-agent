"""Content-addressed projection of ontology build artifacts and stage gates."""

from __future__ import annotations

from datetime import datetime

from contracts.ontology_building import (
    ArtifactPinV1,
    CanonicalEvidenceAdequacyAssessmentV1,
    OntologyBuildManifestV1,
    OntologyOperation,
    OntologyStageGateV1,
)
from contracts.ontology_certification import OntologyCertificationReadinessV1
from eos.artifact_hashing import content_hash


class OntologyBuildCoordinator:
    def project_manifest(
        self,
        *,
        build_ref: str,
        organization_id: str,
        workspace_id: str,
        assurance_profile: str,
        assessment: CanonicalEvidenceAdequacyAssessmentV1,
        additional_artifacts: tuple[ArtifactPinV1, ...],
        created_at: datetime,
        predecessor_build_ref: str | None = None,
        predecessor_build_hash: str | None = None,
    ) -> OntologyBuildManifestV1:
        if (assessment.organization_id, assessment.workspace_id) != (
            organization_id,
            workspace_id,
        ):
            raise PermissionError("assessment is outside the build tenant scope")
        assessment_pin = ArtifactPinV1(
            artifact_kind="canonical_evidence_adequacy_assessment",
            artifact_ref=assessment.assessment_ref,
            artifact_hash=assessment.assessment_hash,
        )
        artifacts = (assessment_pin, *additional_artifacts)
        evidence_blockers = tuple(
            finding.finding_id
            for finding in assessment.findings
            if finding.severity == "blocking"
        )
        gates = tuple(
            assessment.gate_for(operation)
            if operation is OntologyOperation.EXPLORE or evidence_blockers
            else OntologyStageGateV1(
                operation=operation,
                decision="PAUSE",
                blocker_refs=("readiness:certification-evidence-required",),
                assessment_ref=assessment.assessment_ref,
                assessment_hash=assessment.assessment_hash,
            )
            for operation in OntologyOperation
        )
        basis = {
            "build_ref": build_ref,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "assurance_profile": assurance_profile,
            "child_artifacts": [item.model_dump(mode="json") for item in artifacts],
            "gates": [item.model_dump(mode="json") for item in gates],
        }
        if (predecessor_build_ref is None) != (predecessor_build_hash is None):
            raise ValueError('predecessor build ref/hash must be paired')
        if predecessor_build_ref is not None:
            basis.update(predecessor_build_ref=predecessor_build_ref, predecessor_build_hash=predecessor_build_hash)
        digest = content_hash(basis)
        return OntologyBuildManifestV1(
            build_ref=build_ref,
            build_hash=f"sha256:{digest}",
            organization_id=organization_id,
            workspace_id=workspace_id,
            assurance_profile=assurance_profile,
            child_artifacts=artifacts,
            gates=gates,
            created_at=created_at,
            predecessor_build_ref=predecessor_build_ref, predecessor_build_hash=predecessor_build_hash,
        )

    def project_certification_manifest(
        self,
        *,
        build_ref: str,
        organization_id: str,
        workspace_id: str,
        assurance_profile: str,
        assessment: CanonicalEvidenceAdequacyAssessmentV1,
        readiness: OntologyCertificationReadinessV1,
        additional_artifacts: tuple[ArtifactPinV1, ...],
        created_at: datetime,
        predecessor_build_ref: str,
        predecessor_build_hash: str,
    ) -> OntologyBuildManifestV1:
        if assurance_profile != "A1_INTERNAL_POLICY_MVL":
            raise ValueError("bounded certification projection requires A1_INTERNAL_POLICY_MVL")
        if (assessment.organization_id, assessment.workspace_id) != (
            organization_id,
            workspace_id,
        ) or (readiness.organization_id, readiness.workspace_id) != (
            organization_id,
            workspace_id,
        ):
            raise PermissionError("certification inputs are outside the build tenant scope")
        if (readiness.origin_build_ref, readiness.origin_build_hash) != (
            predecessor_build_ref,
            predecessor_build_hash,
        ):
            raise ValueError("certification readiness does not bind the predecessor build")
        evidence_blockers = tuple(
            finding.finding_id
            for finding in assessment.findings
            if finding.severity == "blocking"
        )
        compile_blockers = tuple(
            dict.fromkeys((*evidence_blockers, *readiness.compile_certifiable_blockers))
        )
        certify_blockers = tuple(
            dict.fromkeys((*compile_blockers, *readiness.certify_blockers))
        )
        admit_blockers = tuple(
            dict.fromkeys((*certify_blockers, *readiness.admit_blockers))
        )
        activation_blockers = tuple(
            dict.fromkeys((*admit_blockers, *readiness.activation_blockers))
        )
        blockers_by_operation = {
            OntologyOperation.EXPLORE: (),
            OntologyOperation.COMPILE_CERTIFIABLE: compile_blockers,
            OntologyOperation.CERTIFY: certify_blockers,
            OntologyOperation.ADMIT: admit_blockers,
            OntologyOperation.ACTIVATE: activation_blockers,
        }
        gates = tuple(
            OntologyStageGateV1(
                operation=operation,
                decision="PAUSE" if blockers_by_operation[operation] else "PROCEED",
                blocker_refs=blockers_by_operation[operation],
                assessment_ref=(
                    assessment.assessment_ref
                    if operation is OntologyOperation.EXPLORE
                    else readiness.readiness_ref
                ),
                assessment_hash=(
                    assessment.assessment_hash
                    if operation is OntologyOperation.EXPLORE
                    else readiness.readiness_hash
                ),
            )
            for operation in OntologyOperation
        )
        assessment_pin = ArtifactPinV1(
            artifact_kind="canonical_evidence_adequacy_assessment",
            artifact_ref=assessment.assessment_ref,
            artifact_hash=assessment.assessment_hash,
        )
        readiness_pin = ArtifactPinV1(
            artifact_kind="ontology_certification_readiness",
            artifact_ref=readiness.readiness_ref,
            artifact_hash=readiness.readiness_hash,
        )
        artifacts = (assessment_pin, readiness_pin, *additional_artifacts)
        basis = {
            "build_ref": build_ref,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "assurance_profile": assurance_profile,
            "child_artifacts": [item.model_dump(mode="json") for item in artifacts],
            "gates": [item.model_dump(mode="json") for item in gates],
            "predecessor_build_ref": predecessor_build_ref,
            "predecessor_build_hash": predecessor_build_hash,
        }
        digest = content_hash(basis)
        return OntologyBuildManifestV1(
            build_ref=build_ref,
            build_hash=f"sha256:{digest}",
            organization_id=organization_id,
            workspace_id=workspace_id,
            assurance_profile=assurance_profile,
            child_artifacts=artifacts,
            gates=gates,
            created_at=created_at,
            predecessor_build_ref=predecessor_build_ref,
            predecessor_build_hash=predecessor_build_hash,
        )


__all__ = ["OntologyBuildCoordinator"]
