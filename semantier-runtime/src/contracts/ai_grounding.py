"""Candidate-only contracts for structured AI grounding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol, Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AI_GROUNDING_BATCH_SCHEMA_VERSION = "semantier.ai_grounding_batch.v1"
AI_GROUNDING_REQUEST_SCHEMA_VERSION = "knowledge_builder_ai_grounding_request.v2"


class AiGroundingStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    NEEDS_EDIT = "needs_edit"
    PROVIDER_ERROR = "provider_error"


class AiGroundingSplitProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offsets: tuple[int, ...] = ()
    segments: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)

    @field_validator("offsets")
    @classmethod
    def offsets_are_strictly_increasing(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(offset < 1 for offset in value):
            raise ValueError("split offsets must be positive")
        if list(value) != sorted(set(value)):
            raise ValueError("split offsets must be strictly increasing")
        return value

    @model_validator(mode="after")
    def require_segments_for_offsets(self) -> "AiGroundingSplitProposal":
        if self.offsets and len(self.segments) != len(self.offsets) + 1:
            raise ValueError("split proposals require one more segment than offset")
        return self


class AiGroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assertion_id: str = Field(min_length=1)
    status: AiGroundingStatus
    confidence: float = Field(ge=0.0, le=1.0)
    subject_text: str | None = None
    predicate_text: str | None = None
    object_text: str | None = None
    evidence_anchor_refs: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)
    issues: tuple[str, ...] = ()
    split_proposals: tuple[AiGroundingSplitProposal, ...] = ()

    @field_validator("evidence_anchor_refs")
    @classmethod
    def unique_anchor_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("evidence anchor refs must be unique")
        return value


class AiGroundingBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    results: tuple[AiGroundingResult, ...]

    @field_validator("schema_version")
    @classmethod
    def valid_schema_version(cls, value: str) -> str:
        if value != AI_GROUNDING_BATCH_SCHEMA_VERSION:
            raise ValueError(f"{AI_GROUNDING_BATCH_SCHEMA_VERSION} required")
        return value


class AiGroundingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str = Field(alias="schemaVersion")
    extraction_run_id: str = Field(alias="extractionRunId", min_length=1)
    candidate_ids: tuple[str, ...] | None = Field(default=None, alias="candidateIds")

    @field_validator("schema_version")
    @classmethod
    def valid_request_version(cls, value: str) -> str:
        if value != AI_GROUNDING_REQUEST_SCHEMA_VERSION:
            raise ValueError(f"{AI_GROUNDING_REQUEST_SCHEMA_VERSION} required")
        return value

    @field_validator("candidate_ids")
    @classmethod
    def valid_candidate_ids(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        if not 1 <= len(value) <= 500:
            raise ValueError("candidateIds must contain between 1 and 500 ids")
        if len(set(value)) != len(value):
            raise ValueError("candidateIds must be unique")
        return value


class BatchAcceptItem(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    assertion_id: str = Field(alias="assertionId", min_length=1)
    evidence_anchor_refs: tuple[str, ...] = Field(alias="evidenceAnchorRefs", min_length=1)


class BatchAcceptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: str = Field(alias="schemaVersion")
    batch_idempotency_key: str = Field(alias="batchIdempotencyKey", min_length=1, max_length=200)
    decision: str
    items: tuple[BatchAcceptItem, ...] = Field(min_length=1, max_length=100)
    certainty: str = Field(min_length=1)
    reason_code: str = Field(alias="reasonCode", min_length=1)
    justification: str = Field(min_length=1)

    @field_validator("schema_version")
    @classmethod
    def valid_batch_schema(cls, value: str) -> str:
        if value != "learning_event_grounding_batch_request.v1":
            raise ValueError("learning_event_grounding_batch_request.v1 required")
        return value

    @field_validator("decision")
    @classmethod
    def accept_only(cls, value: str) -> str:
        if value != "accept":
            raise ValueError("batch grounding supports accept only")
        return value

    @field_validator("items")
    @classmethod
    def unique_assertions(cls, value: tuple[BatchAcceptItem, ...]) -> tuple[BatchAcceptItem, ...]:
        ids = [item.assertion_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("batch assertion ids must be unique")
        return value

@dataclass(frozen=True)
class StructuredGenerationResult:
    parsed: Mapping[str, Any]
    text: str
    provider: str
    model: str
    agent_id: str
    usage: Mapping[str, Any]
    audit: Mapping[str, Any]


class StructuredGenerator(Protocol):
    async def generate_structured(
        self, *, instructions: str, input_text: str, json_schema: Mapping[str, Any]
    ) -> StructuredGenerationResult: ...


class CompatibleStructuredGenerator(Protocol):
    @property
    def compatibility_execution_identity(self) -> dict: ...

    async def generate_compatible_structured(
        self, *, instructions: str, input_text: str, json_schema: Mapping[str, Any],
        record: Callable[[dict], None], schema_name: str = "runtime_semantic_judgment_v1",
        purpose: str = "runtime_semantic_judgment",
    ) -> StructuredGenerationResult: ...


class AiGroundingContractError(ValueError):
    """Structured provider output failed candidate/evidence validation."""


class StructuredGenerationError(RuntimeError):
    """The active provider failed to produce locally validated JSON."""

    def __init__(self, message: str, *, raw_response: Mapping[str, str] | None = None):
        super().__init__(message)
        self.raw_response = dict(raw_response) if raw_response is not None else None


class NoCompatibleModelError(StructuredGenerationError):
    code = "NO_COMPATIBLE_MODEL"
    retryable = True
