"""Pinned output package for source-derived validation; never activation authority."""
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_paired_evaluation import ArtifactPayloadPinV1, EvaluatorCodeUnitV1
from eos.artifact_hashing import content_hash


class SourceValidationExecutionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["source_validation_execution.v1"] = "source_validation_execution.v1"
    execution_ref: str
    execution_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str
    workspace_id: str
    candidate_pin: ArtifactPayloadPinV1
    review_pin: ArtifactPayloadPinV1
    validation_pin: ArtifactPayloadPinV1
    profile_pin: ArtifactPayloadPinV1
    authority_pin: ArtifactPayloadPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    predecessor_pin: ArtifactPayloadPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    run_pin: ArtifactPayloadPinV1
    plan_pin: ArtifactPayloadPinV1 | None
    evaluation_pins: tuple[ArtifactPayloadPinV1, ...]
    compilation_error: str | None
    code_units: tuple[EvaluatorCodeUnitV1, ...] = Field(min_length=1)
    environment: dict[str, str] = Field(min_length=1)
    executed_at: datetime

    @model_validator(mode="after")
    def exact_content(self):
        if self.executed_at.utcoffset() != timedelta(0):
            raise ValueError("source validation execution requires UTC time")
        refs = [unit.code_ref for unit in self.code_units]
        if len(refs) != len(set(refs)):
            raise ValueError("source validation code references must be unique")
        if self.plan_pin is None and self.evaluation_pins:
            raise ValueError("validation outputs require their compiled plan")
        if (self.plan_pin is None) != bool(self.compilation_error):
            raise ValueError("uncompiled validation requires its rejection diagnostic")
        digest = content_hash(self.model_dump(mode="json", exclude={"execution_ref", "execution_hash"}))
        if (self.execution_ref, self.execution_hash) != ("source-validation-execution:" + digest, "sha256:" + digest):
            raise ValueError("source validation execution hash mismatch")
        return self

    @classmethod
    def create(cls, **values):
        probe = cls.model_construct(execution_ref="pending", execution_hash="sha256:" + "0" * 64, **values)
        digest = content_hash(probe.model_dump(mode="json", exclude={"execution_ref", "execution_hash"}))
        return cls(execution_ref="source-validation-execution:" + digest, execution_hash="sha256:" + digest, **values)
