"""Map validation findings back to the smallest governed semantic region."""

from __future__ import annotations

from datetime import datetime

from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.ontology_repair import OntologyDiagnosticV1
from contracts.ontology_quality import QualityRequirementResultV1
from contracts.artifact_pin import ArtifactPinV1
from contracts.validation_contract import ValidationContractV1
from eos.artifact_hashing import content_hash


class OntologyDiagnosisService:
    def from_quality_result(
        self,
        *,
        result: QualityRequirementResultV1,
        validation_contract: ValidationContractV1,
        candidate_set: NormativeSemanticCandidateSetV1,
        created_at: datetime,
    ) -> OntologyDiagnosticV1:
        result = QualityRequirementResultV1.model_validate(result)
        if result.gate != "FITNESS" or result.status != "BLOCK":
            raise ValueError("a blocking FITNESS result is required")
        if result.subject != ArtifactPinV1(artifact_kind="normative_semantic_candidate_set",
            artifact_ref=candidate_set.candidate_set_ref, artifact_hash=candidate_set.candidate_set_hash):
            raise ValueError("diagnosis quality subject mismatch")
        if (candidate_set.organization_id, candidate_set.workspace_id) != (
            validation_contract.organization_id, validation_contract.workspace_id):
            raise PermissionError("diagnosis inputs cross tenant scope")
        obligations = {item.obligation_id: item for item in validation_contract.obligations}
        findings = tuple(item for item in result.findings if item.code == "missing_source_obligation")
        refs = tuple(sorted({ref for finding in findings for ref in finding.subject_refs if ref in obligations}))
        if not refs:
            raise ValueError("diagnosis requires attributable missing source obligations")
        selected = [obligations[ref] for ref in refs]
        finding_refs = tuple(item.finding_ref for item in findings)
        dimensions = {item.semantic_dimension for item in selected}
        if len(dimensions) != 1:
            raise ValueError("diagnosis must be clustered to one semantic dimension")
        dimension = next(iter(dimensions))
        competency_refs = tuple(
            dict.fromkeys(cq for item in selected for cq in item.competency_ids)
        )
        source_refs = tuple(
            dict.fromkeys(ref for item in selected for ref in item.evidence_refs)
        )
        basis = {
            "candidate_set_ref": candidate_set.candidate_set_ref,
            "candidate_set_hash": candidate_set.candidate_set_hash,
            "finding_refs": finding_refs,
            "semantic_dimension": dimension,
            "competency_refs": competency_refs,
            "source_evidence_refs": source_refs,
        }
        digest = content_hash(basis)
        return OntologyDiagnosticV1(
            diagnostic_ref=f"ontology-diagnostic:{digest}",
            diagnostic_hash=f"sha256:{digest}",
            organization_id=candidate_set.organization_id,
            workspace_id=candidate_set.workspace_id,
            candidate_set_ref=candidate_set.candidate_set_ref,
            candidate_set_hash=candidate_set.candidate_set_hash,
            owning_stage="S3",
            semantic_dimension=dimension,
            finding_refs=finding_refs,
            obligation_refs=refs,
            competency_refs=competency_refs,
            source_evidence_refs=source_refs,
            ir_ast_paths=(f"/frames/{dimension}",),
            allowed_typed_actions=("OPEN_EVIDENCE", "PROPOSE_BOUNDED_REPAIR"),
            created_at=created_at,
        )


__all__ = ["OntologyDiagnosisService"]
