"""Server-owned ontology workbench projection over manifest-pinned artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.business_answers_receivables import ReceivablesReportV1
from contracts.ontology_admission import FailureAttributionV1, LearningEvidenceV1
from contracts.epistemic_readiness import BoundedAutonomyTransitionV1, EpistemicReadinessV1
from contracts.normative_schema import NormativeSchemaContractV1, SchemaAdequacyReportV1
from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.ontology_building import (
    CandidateGraphDiffProjectionV1,
    CandidateGraphLineageProjectionV1,
    ExploratoryGraphSnapshotV1,
    OntologyBuildManifestV1,
    CompetencyContractV1,
)
from contracts.ontology_validation import (
    RuntimeSemanticEvaluationBundleV1,
)
from contracts.ontology_seed_alignment import OntologySeedComparisonV1
from contracts.validation_contract import ValidationContractV1
from contracts.ontology_quality import QualityProfileV1, QualityEvaluationRunV1
from eos.artifact_hashing import content_hash


class OntologyComparisonBaselineV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    build_ref: str = Field(min_length=1)
    build_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    snapshot_ref: str = Field(min_length=1)
    snapshot_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    selection_reason: Literal["earliest_certification_ancestor"] = (
        "earliest_certification_ancestor"
    )


class OntologyWorkbenchReadModelV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["ontology_workbench_read_model.v1"] = (
        "ontology_workbench_read_model.v1"
    )
    read_model_ref: str = Field(min_length=1)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    manifest: OntologyBuildManifestV1
    exploratory_snapshot: ExploratoryGraphSnapshotV1 | None
    quality_profile: QualityProfileV1 | None
    lineage: CandidateGraphLineageProjectionV1 | None
    diff: CandidateGraphDiffProjectionV1 | None
    receivables_verification: ReceivablesReportV1 | None = None
    comparison_baseline: OntologyComparisonBaselineV1 | None = None
    seed_comparison: OntologySeedComparisonV1 | None = None
    competency_contract: CompetencyContractV1 | None = None
    normative_schema: NormativeSchemaContractV1 | None = None
    schema_adequacy: SchemaAdequacyReportV1 | None = None
    semantic_candidate: NormativeSemanticCandidateSetV1 | None = None
    validation_contract: ValidationContractV1 | None = None
    quality_evaluation: QualityEvaluationRunV1 | None
    runtime_evaluations: tuple[RuntimeSemanticEvaluationBundleV1, ...]
    learning_evidence: tuple[LearningEvidenceV1, ...]
    failure_attributions: tuple[FailureAttributionV1, ...]
    generated_at: datetime
    epistemic_readiness: EpistemicReadinessV1 | None = None
    transition_recommendation: BoundedAutonomyTransitionV1 | None = None

    @model_validator(mode="after")
    def exact_manifest_projection(self) -> "OntologyWorkbenchReadModelV1":
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() != timedelta(0):
            raise ValueError("generated_at must be timezone-aware UTC")
        tenant = (self.organization_id, self.workspace_id)
        if tenant != (self.manifest.organization_id, self.manifest.workspace_id):
            raise ValueError("read model tenant must match manifest")
        if self.seed_comparison is not None and tenant != (
            self.seed_comparison.organization_id,
            self.seed_comparison.workspace_id,
        ):
            raise ValueError("seed comparison tenant must match read model")
        pins = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
            for pin in self.manifest.child_artifacts
        }
        if self.receivables_verification is not None and tenant != (
            self.receivables_verification.organization_id, self.receivables_verification.workspace_id
        ):
            raise ValueError("receivables verification tenant must match read model")
        if self.receivables_verification is not None:
            report = self.receivables_verification
            if report.build_ref != self.manifest.build_ref:
                raise ValueError("receivables verification must bind the displayed build")
            if (report.snapshot_pin.artifact_kind, report.snapshot_pin.artifact_ref, report.snapshot_pin.artifact_hash) not in pins:
                raise ValueError("receivables verification snapshot must be manifest-pinned")
            if (report.source_pin.artifact_kind, report.source_pin.artifact_ref, report.source_pin.artifact_hash) not in pins:
                raise ValueError("receivables verification source must be manifest-pinned")
        artifacts = (
            ("receivables_report", self.receivables_verification, "artifact_ref", "artifact_hash"),
            ("exploratory_graph_snapshot", self.exploratory_snapshot, "snapshot_ref", "snapshot_hash"),
            ("quality_profile", self.quality_profile, "profile_ref", "profile_hash"),
            ("candidate_graph_lineage_projection", self.lineage, "lineage_ref", "lineage_hash"),
            ("candidate_graph_diff_projection", self.diff, "diff_ref", "diff_hash"),
            ("ontology_seed_comparison", self.seed_comparison, "comparison_ref", "comparison_hash"),
            ("competency_contract", self.competency_contract, "competency_contract_ref", "competency_contract_hash"),
            ("normative_schema_contract", self.normative_schema, "schema_ref", "schema_hash"),
            ("schema_adequacy_report", self.schema_adequacy, "report_ref", "report_hash"),
            ("normative_semantic_candidate_set", self.semantic_candidate, "candidate_set_ref", "candidate_set_hash"),
            ("validation_contract", self.validation_contract, "validation_contract_ref", "validation_contract_hash"),
            ("quality_evaluation_run", self.quality_evaluation, "run_ref", "run_hash"),
        )
        repeated = (
            ("runtime_semantic_evaluation_bundle", self.runtime_evaluations, "bundle_ref", "bundle_hash"),
            ("learning_evidence", self.learning_evidence, "learning_evidence_ref", "learning_evidence_hash"),
            ("failure_attribution", self.failure_attributions, "attribution_ref", "attribution_hash"),
        )
        for kind, artifact, ref_field, hash_field in artifacts:
            if artifact is not None and (
                kind, getattr(artifact, ref_field), getattr(artifact, hash_field)
            ) not in pins:
                raise ValueError(f"{kind} must be manifest-pinned")
        if self.quality_evaluation is not None:
            run = self.quality_evaluation
            if self.quality_profile is None or run.quality_profile != self.quality_profile.pin():
                raise ValueError("quality evaluation must bind the displayed profile")
            for pin in (run.subject, run.validation):
                if (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) not in pins:
                    raise ValueError("quality evaluation basis must be manifest-pinned")
        if self.seed_comparison is not None:
            comparison = self.seed_comparison
            if self.exploratory_snapshot is None:
                raise ValueError("seed comparison requires the candidate snapshot")
            candidate = comparison.candidate_snapshot
            if (
                candidate.artifact_ref != self.exploratory_snapshot.snapshot_ref
                or candidate.artifact_hash != self.exploratory_snapshot.snapshot_hash
                or candidate.payload_hash
                != "sha256:"
                + content_hash(self.exploratory_snapshot.model_dump(mode="json"))
            ):
                raise ValueError("seed comparison must bind the exact candidate snapshot")
            seed = comparison.seed_snapshot
            if (
                "ontology_hub_snapshot",
                seed.artifact_ref,
                seed.artifact_hash,
            ) not in pins:
                raise ValueError("seed comparison must bind the manifest seed snapshot")
            if (
                "source_grounded_seed_delta",
                comparison.delta_ref,
                comparison.delta_hash,
            ) not in pins:
                raise ValueError("seed comparison must bind the manifest seed delta")
        for kind, values, ref_field, hash_field in repeated:
            for artifact in values:
                if (kind, getattr(artifact, ref_field), getattr(artifact, hash_field)) not in pins:
                    raise ValueError(f"{kind} must be manifest-pinned")
        return self


__all__ = ["OntologyComparisonBaselineV1", "OntologyWorkbenchReadModelV1"]
