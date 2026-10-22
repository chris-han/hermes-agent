"""Content-addressed O0/O1/O2 evaluation evidence for ontology MVL closure."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import SHA256_PATTERN
from contracts.ontology_quality import QualityDisposition, QualityRequirementStatus


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OntologyExperimentResultV1(_Contract):
    namespace: Literal["O0", "O1", "O2_BEFORE", "O2_AFTER"]
    constraint_status: QualityRequirementStatus
    quality_disposition: QualityDisposition | None
    mutation_detection_count: int = Field(ge=0)
    mutation_total: int = Field(ge=0)
    critical_dimension_coverage: float = Field(ge=0, le=1)
    admission_ready: bool
    artifact_refs: tuple[str, ...]
    artifact_hashes: tuple[str, ...]

    @model_validator(mode="after")
    def consistent_counts(self) -> "OntologyExperimentResultV1":
        if self.mutation_detection_count > self.mutation_total:
            raise ValueError("mutation detection cannot exceed mutation total")
        return self


class OntologyMVLEvaluationReportV1(_Contract):
    schema_version: Literal["ontology_mvl_evaluation_report.v1"] = "ontology_mvl_evaluation_report.v1"
    report_ref: str = Field(min_length=1)
    report_hash: str = Field(pattern=SHA256_PATTERN)
    fixture_ref: str = Field(min_length=1)
    fixture_hash: str = Field(pattern=SHA256_PATTERN)
    assurance_profile: Literal["A1_INTERNAL_POLICY_MVL"]
    semantic_compiler_ref: str
    validation_compiler_ref: str
    evaluator_ref: str
    samples: int = Field(ge=5)
    cold_latency_ms: float = Field(ge=0)
    warm_p50_latency_ms: float = Field(ge=0)
    warm_p95_latency_ms: float = Field(ge=0)
    experiments: tuple[OntologyExperimentResultV1, ...]
    bounded_improvement: Literal["restore_source_grounded_exception_frame"]
    changed_frame_refs: tuple[str, ...] = Field(min_length=1, max_length=1)
    invalidated_result_kinds: tuple[str, ...]
    comparable_inputs_preserved: bool
    runtime_bundle_complete: bool
    learning_attribution_complete: bool
    decision: Literal["GO", "ITERATE", "NO_GO"]
    environment: dict[str, str]
    generated_at: datetime

    @model_validator(mode="after")
    def closure_invariants(self) -> "OntologyMVLEvaluationReportV1":
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() != timedelta(0):
            raise ValueError("generated_at must be timezone-aware UTC")
        names = tuple(item.namespace for item in self.experiments)
        if names != ("O0", "O1", "O2_BEFORE", "O2_AFTER"):
            raise ValueError("evaluation must contain ordered O0/O1/O2 before/after evidence")
        before, after = self.experiments[2:]
        if self.decision == "GO" and not (
            before.quality_disposition == "NOT_QUALIFIED"
            and after.quality_disposition in {"QUALIFIED", "QUALIFIED_WITH_WARNINGS"}
            and after.mutation_detection_count == after.mutation_total
            and after.admission_ready
            and self.comparable_inputs_preserved
            and self.runtime_bundle_complete
            and self.learning_attribution_complete
        ):
            raise ValueError("GO requires the complete measured MVL closure")
        return self


__all__ = ["OntologyExperimentResultV1", "OntologyMVLEvaluationReportV1"]
