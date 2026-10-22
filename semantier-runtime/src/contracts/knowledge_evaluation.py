from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")


Layer = Literal["extraction", "graph", "reasoning"]
ExecutionMode = Literal["real", "recorded", "in_memory"]


class KnowledgeBenchmarkRunRequestV1(StrictDTO):
    schemaVersion: Literal["knowledge_benchmark_run_request.v1"]
    benchmarkFamilyRef: str
    profileId: str
    profileVersion: str
    layers: tuple[Layer, ...]
    executionMode: ExecutionMode
    providers: tuple[str, ...] = ()
    referenceReleaseId: str | None = None
    caseFilters: dict[str, Any] | None = None
    challengeTags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def layers_are_unique(self) -> "KnowledgeBenchmarkRunRequestV1":
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must be non-empty and unique")
        return self


class BenchmarkArtifactRefV1(StrictDTO):
    schema_version: Literal["benchmark_artifact_ref.v1"] = "benchmark_artifact_ref.v1"
    artifact_ref: str
    artifact_kind: str
    content_hash: str
    media_type: str
    layer: Literal["extraction", "graph", "reasoning", "canonical_evidence"]
    created_at: str


class KnowledgeBenchmarkRunV1(StrictDTO):
    schema_version: Literal["knowledge_benchmark_run.v1"] = "knowledge_benchmark_run.v1"
    benchmark_orchestration_id: str
    organization_id: str
    workspace_id: str | None
    layer: Layer
    evaluation_run_id: str
    evaluation_evidence_id: str | None
    benchmark_family_ref: str
    scenario_ref: str
    subject_manifest_ref: str
    environment_manifest_ref: str
    evaluator_bundle_ref: str
    profile_id: str
    profile_version: str
    execution_mode: ExecutionMode
    operational_status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "PARTIAL"]
    run_validity: Literal["VALID", "INVALID_EVALUATOR", "INVALID_ENVIRONMENT", "INVALID_FIXTURE"] | None
    certification_result: Literal["INVALID_RUN", "FAIL_CRITICAL", "NOT_EVALUABLE", "PASS_WITH_WAIVER", "PASS"] | None
    critical_gate_results: tuple[dict[str, Any], ...] = ()
    waiver_refs: tuple[str, ...] = ()
    reference_release_id: str | None = None
    reference_release_hash: str | None = None
    artifact_refs: tuple[BenchmarkArtifactRefV1, ...] = ()
    created_at: str
    completed_at: str | None
    content_hash: str


class KnowledgeBenchmarkOrchestrationV1(StrictDTO):
    schema_version: Literal["knowledge_benchmark_orchestration.v1"] = "knowledge_benchmark_orchestration.v1"
    benchmark_orchestration_id: str
    organization_id: str
    workspace_id: str | None
    benchmark_family_ref: str
    profile_id: str
    profile_version: str
    requested_layers: tuple[Layer, ...]
    execution_mode: ExecutionMode
    operational_status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "PARTIAL"]
    child_run_refs: tuple[dict[str, Any], ...]
    created_at: str
    completed_at: str | None
    warnings: tuple[dict[str, Any] | str, ...] = ()
    content_hash: str
