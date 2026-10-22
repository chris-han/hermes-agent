from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import validate_hash, validate_identifier


class EffectiveContextEvalMode(StrEnum):
    PERSISTENT = "persistent"
    REFERENCE_CHECKPOINT = "reference_checkpoint"
    BRANCHED_CHECKPOINT = "branched_checkpoint"
    COMPACTED_CHECKPOINT = "compacted_checkpoint"
    MEMORY_AUGMENTED_CHECKPOINT = "memory_augmented_checkpoint"


class EvaluationFailureLabel(StrEnum):
    MISSED_ACTIVE_REQUIREMENT = "MISSED_ACTIVE_REQUIREMENT"
    STALE_SUPERSEDED_REQUIREMENT = "STALE_SUPERSEDED_REQUIREMENT"
    KNOWLEDGE_DISCLOSURE_LOSS = "KNOWLEDGE_DISCLOSURE_LOSS"
    AUTHORITY_PIN_DRIFT = "AUTHORITY_PIN_DRIFT"
    ARTIFACT_STATE_REGRESSION = "ARTIFACT_STATE_REGRESSION"
    BRANCH_BOUNDARY_LEAK = "BRANCH_BOUNDARY_LEAK"
    CHECKPOINT_CLOSURE_INCOMPLETE = "CHECKPOINT_CLOSURE_INCOMPLETE"
    COMPACTION_INFORMATION_LOSS = "COMPACTION_INFORMATION_LOSS"
    MEMORY_CONTAMINATION = "MEMORY_CONTAMINATION"
    ENVIRONMENT_OR_TOOLING_FAILURE = "ENVIRONMENT_OR_TOOLING_FAILURE"
    UNKNOWN_REVIEW_REQUIRED = "UNKNOWN_REVIEW_REQUIRED"


class EffectiveContextMetric(StrEnum):
    RESTORATION_FIDELITY = "restoration_fidelity"
    BRANCH_EQUIVALENCE = "branch_equivalence"
    ACTIVE_CONTRACT_RETENTION = "active_contract_retention"
    KNOWLEDGE_CLOSURE_RECALL = "knowledge_closure_recall"
    AUTHORITY_PIN_FIDELITY = "authority_pin_fidelity"
    COMPACTION_FIDELITY = "compaction_fidelity"
    MEMORY_UTILITY = "memory_utility"
    CONTAMINATION = "contamination"


class EffectiveContextEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    case_id: str
    benchmark_ref: str
    mode: EffectiveContextEvalMode
    organization_id: str
    workspace_id: str
    session_id: str
    reference_checkpoint_id: str | None = None
    round_boundaries: list[str] = Field(default_factory=list)
    active_requirement_labels: list[str] = Field(default_factory=list)
    released_task_semantics_hash: str
    fixtures: dict[str, Any] = Field(default_factory=dict)

    _ids = field_validator("case_id", "benchmark_ref", "organization_id", "workspace_id", "session_id", mode="after")(validate_identifier)
    _hash = field_validator("released_task_semantics_hash", mode="after")(validate_hash)


class EffectiveContextEvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    result_id: str
    case_id: str
    mode: EffectiveContextEvalMode
    checkpoint_id: str | None = None
    trajectory_ref: str
    checkpoint_lineage_ref: str | None = None
    first_failed_round: int | None = None
    failure_labels: list[EvaluationFailureLabel] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    fail_stop: bool = False
    harbor_round_accounting: dict[str, Any] = Field(default_factory=dict)

    _ids = field_validator("result_id", "case_id", "trajectory_ref", "checkpoint_lineage_ref", mode="after")(lambda v: validate_identifier(v) if v is not None else v)

    @field_validator("failure_labels")
    @classmethod
    def unique_failure_labels(cls, value: list[EvaluationFailureLabel]) -> list[EvaluationFailureLabel]:
        return sorted(set(value), key=lambda item: item.value)

    @field_validator("metrics")
    @classmethod
    def valid_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        allowed = {metric.value for metric in EffectiveContextMetric}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown effective-context metrics: {sorted(unknown)}")
        for name, score in value.items():
            if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
                raise ValueError(f"metric {name} must be between 0 and 1")
        return {name: float(score) for name, score in value.items()}
