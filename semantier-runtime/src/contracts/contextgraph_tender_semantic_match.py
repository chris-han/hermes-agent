"""Candidate-only semantic match contracts for tender findings."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SemanticRelation = Literal[
    "synonym",
    "paraphrase",
    "narrower_than",
    "broader_than",
    "implication",
]
QualificationContext = Literal[
    "requirement",
    "scoring",
    "description",
    "negation",
    "unknown",
]
Polarity = Literal["positive", "negative", "unclear"]
Modality = Literal["mandatory", "optional", "prohibited", "unknown"]


class TenderSemanticMatchCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    reference_expression: str = Field(min_length=1)
    observed_expression: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1)
    evidence_anchor_refs: tuple[str, ...] = Field(min_length=1)

    @field_validator("evidence_anchor_refs")
    @classmethod
    def unique_anchor_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("evidence anchor refs must be unique")
        return value


class TenderSemanticMatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(min_length=1)
    semantic_relation: SemanticRelation
    meaning_match: bool
    relation_match: bool
    polarity: Polarity
    modality: Modality
    qualification_context: QualificationContext
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_anchor_refs: tuple[str, ...]
    rationale: str = Field(min_length=1)

    @field_validator("evidence_anchor_refs")
    @classmethod
    def unique_anchor_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("evidence anchor refs must be unique")
        return value


class TenderSemanticMatchBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["contextgraph_tender_semantic_match_batch.v1"]
    results: tuple[TenderSemanticMatchResult, ...]


class TenderSemanticMatchSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["contextgraph_tender_semantic_match_suggestion.v1"]
    suggestion_id: str = Field(min_length=1)
    finding_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    observed_expression: str = Field(min_length=1)
    semantic_relation: SemanticRelation | None = None
    meaning_match: bool | None = None
    relation_match: bool | None = None
    polarity: Polarity | None = None
    modality: Modality | None = None
    qualification_context: QualificationContext | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_anchor_refs: tuple[str, ...] = ()
    rationale: str | None = None
    provider: str | None = None
    model: str | None = None
    agent_id: str | None = None
    prompt_ref: str
    prompt_hash: str
    latency_ms: int
    token_usage: dict[str, Any]
    failure_reason: str | None = None
    grounding_status: Literal["ai_suggested", "provider_error"] = "ai_suggested"
