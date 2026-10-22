"""Typed runtime-repair paired-evaluation subjects and comparison views."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_repair import RuntimeRepairMetricResultV1
from contracts.ontology_validation import RuntimeSemanticEvaluationBundleV1
from eos.artifact_hashing import content_hash


class _PairedContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="after")
    def verify_scope_and_time(self):
        for value in self.model_dump(mode="python").values():
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError("paired-evaluation timestamps must be timezone-aware UTC")
        return self


class RuntimeRepairPairV1(_PairedContract):
    schema_version: Literal["runtime_repair_pair.v1"]
    pair_ref: str = Field(min_length=1)
    pair_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    predecessor_manifest: ArtifactPinV1
    successor_manifest: ArtifactPinV1
    predecessor_snapshot: ArtifactPinV1
    successor_snapshot: ArtifactPinV1
    predecessor_candidate_set: ArtifactPinV1
    successor_candidate_set: ArtifactPinV1
    accepted_review: ArtifactPinV1
    learning: ArtifactPinV1
    attribution: ArtifactPinV1
    repair_input: ArtifactPinV1
    repair_impact: ArtifactPinV1
    repair_step_receipt: ArtifactPinV1
    validation_contract: ArtifactPinV1
    source_identity: ArtifactPinV1
    canonical_source_ir: ArtifactPinV1
    graph_intent: ArtifactPinV1

    @model_validator(mode="after")
    def verify_identity(self):
        payload = self.model_dump(mode="json", exclude={"pair_ref", "pair_hash"})
        digest = content_hash(payload)
        if self.pair_ref != "runtime-repair-pair:" + digest or self.pair_hash != "sha256:" + digest:
            raise ValueError("runtime repair pair identity mismatch")
        return self


class RuntimeRepairObligationCaseV1(_PairedContract):
    case_id: str = Field(min_length=1)
    dimension: Literal["COMPETENCY", "VALIDATION", "EVIDENCE_FIDELITY"]
    expected_outcome_pin: ArtifactPinV1
    expected_outcome: Literal["MATCH", "NO_MATCH", "UNKNOWN", "PASS", "FAIL"]
    observed_facts_pin: ArtifactPinV1
    competency_pin: ArtifactPinV1
    validation_obligation_pin: ArtifactPinV1
    baseline_plan_pin: ArtifactPinV1 | None
    successor_plan_pin: ArtifactPinV1 | None
    evidence_pins: tuple[ArtifactPinV1, ...]
    critical: bool


class RuntimeRepairObligationSuiteV1(_PairedContract):
    schema_version: Literal["runtime_repair_obligation_suite.v1"]
    suite_ref: str = Field(min_length=1)
    suite_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    source_identity: ArtifactPinV1
    canonical_source_ir: ArtifactPinV1
    graph_intent: ArtifactPinV1
    oracle_manifest: ArtifactPinV1
    reviewer_ref: str = Field(min_length=1)
    review_decision: ArtifactPinV1
    reviewed_at: datetime
    cases: tuple[RuntimeRepairObligationCaseV1, ...]

    @model_validator(mode="after")
    def unique_case_ids(self):
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("runtime repair obligation case ids must be unique")
        return self


class RuntimeRepairPairedEvaluationRunV1(_PairedContract):
    schema_version: Literal["runtime_repair_paired_evaluation_run.v1"]
    run_ref: str = Field(min_length=1)
    run_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    side: Literal["baseline", "successor"]
    pair_pin: ArtifactPinV1
    candidate_set_pin: ArtifactPinV1
    suite_pin: ArtifactPinV1
    evaluator_bundle_pin: ArtifactPinV1
    environment_pin: ArtifactPinV1
    case_result_pins: tuple[ArtifactPinV1, ...]
    runtime_trace_pins: tuple[ArtifactPinV1, ...]
    evaluated_at: datetime


class RuntimeRepairPairedEvaluationReportV1(_PairedContract):
    schema_version: Literal["runtime_repair_paired_evaluation_report.v1"]
    report_ref: str = Field(min_length=1)
    report_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    pair_pin: ArtifactPinV1
    suite_pin: ArtifactPinV1 | None
    evaluator_bundle_pin: ArtifactPinV1
    environment_pin: ArtifactPinV1
    baseline_run_pin: ArtifactPinV1 | None
    successor_run_pin: ArtifactPinV1 | None
    metric_results: tuple[RuntimeRepairMetricResultV1, ...]
    verdict: Literal["IMPROVED", "REGRESSED", "CHANGED", "INCONCLUSIVE"]
    reason_code: str = Field(min_length=1)
    evaluated_at: datetime


class RuntimeRepairPairedEvaluationResultV1(_PairedContract):
    schema_version: Literal["runtime_repair_paired_evaluation_result.v1"]
    result_ref: str = Field(min_length=1)
    result_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    baseline_run: RuntimeRepairPairedEvaluationRunV1 | None
    successor_run: RuntimeRepairPairedEvaluationRunV1 | None
    report: RuntimeRepairPairedEvaluationReportV1
    runtime_traces: tuple[RuntimeSemanticEvaluationBundleV1, ...]

    @model_validator(mode="after")
    def verify_complete_result_closure(self):
        if (self.organization_id, self.workspace_id) != (
            self.report.organization_id, self.report.workspace_id
        ):
            raise ValueError("paired result scope must match report")
        if (self.baseline_run is None) != (self.successor_run is None):
            raise ValueError("paired result must contain both runs or neither")
        if self.baseline_run is None:
            if self.report.baseline_run_pin is not None or self.report.successor_run_pin is not None:
                raise ValueError("report run pins require exact returned runs")
        else:
            if (self.baseline_run.side, self.successor_run.side) != ("baseline", "successor"):
                raise ValueError("paired result run sides are reversed")
            expected = (
                self.baseline_run.run_ref, self.baseline_run.run_hash,
                self.successor_run.run_ref, self.successor_run.run_hash,
            )
            actual = (
                self.report.baseline_run_pin.artifact_ref if self.report.baseline_run_pin else None,
                self.report.baseline_run_pin.artifact_hash if self.report.baseline_run_pin else None,
                self.report.successor_run_pin.artifact_ref if self.report.successor_run_pin else None,
                self.report.successor_run_pin.artifact_hash if self.report.successor_run_pin else None,
            )
            if actual != expected:
                raise ValueError("report does not bind exact paired runs")
        expected_traces = {
            (pin.artifact_ref, pin.artifact_hash)
            for run in (self.baseline_run, self.successor_run) if run is not None
            for pin in run.runtime_trace_pins
        }
        actual_traces = {(trace.bundle_ref, trace.bundle_hash) for trace in self.runtime_traces}
        if actual_traces != expected_traces or len(actual_traces) != len(self.runtime_traces):
            raise ValueError("runtime traces must exactly resolve paired run pins")
        return self


class RuntimeRepairPairedComparisonViewV1(_PairedContract):
    schema_version: Literal["runtime_repair_paired_comparison_view.v1"]
    comparison_ref: str = Field(min_length=1)
    comparison_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    subject: RuntimeRepairPairV1
    baseline_run_pin: ArtifactPinV1 | None
    successor_run_pin: ArtifactPinV1 | None
    suite_pin: ArtifactPinV1 | None
    evaluator_bundle_pin: ArtifactPinV1
    environment_pin: ArtifactPinV1
    runtime_trace_pins: tuple[ArtifactPinV1, ...]
    metric_results: tuple[RuntimeRepairMetricResultV1, ...]
    verdict: Literal["IMPROVED", "REGRESSED", "CHANGED", "INCONCLUSIVE"]
    reason_code: str = Field(min_length=1)
    audit_input_pins: tuple[ArtifactPinV1, ...]
    evaluated_at: datetime

    @model_validator(mode="after")
    def verify_identity_and_scope(self):
        if (self.organization_id, self.workspace_id) != (
            self.subject.organization_id,
            self.subject.workspace_id,
        ):
            raise ValueError("paired comparison scope must match subject")
        payload = self.model_dump(mode="json", exclude={"comparison_ref", "comparison_hash"})
        digest = content_hash(payload)
        if self.comparison_ref != "runtime-repair-comparison:" + digest or self.comparison_hash != "sha256:" + digest:
            raise ValueError("runtime repair comparison identity mismatch")
        return self


__all__ = [
    "RuntimeRepairMetricResultV1",
    "RuntimeRepairPairV1",
    "RuntimeRepairObligationCaseV1",
    "RuntimeRepairObligationSuiteV1",
    "RuntimeRepairPairedEvaluationRunV1",
    "RuntimeRepairPairedEvaluationReportV1",
    "RuntimeRepairPairedEvaluationResultV1",
    "RuntimeRepairPairedComparisonViewV1",
]
