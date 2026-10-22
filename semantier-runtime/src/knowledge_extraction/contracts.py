"""Versioned provider-neutral source-extraction contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from contracts.knowledge_feature_pack import StrictContract
from contracts.knowledge_reasoning_phase1 import SemanticOccurrence


class SourceExtractionRequest(StrictContract):
    schema_version: Literal["source_extraction_request.v1"] = "source_extraction_request.v1"
    extraction_run_id: str
    document_id: str
    canonical_document_ref: str
    canonical_document_hash: str
    text: str = Field(min_length=1)
    extraction_profile: str
    ontology_hint_version: str | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)


class SourceExtractionResult(StrictContract):
    schema_version: Literal["source_extraction_result.v1"] = "source_extraction_result.v1"
    extraction_run_id: str
    provider: str
    provider_version: str
    model_ref: str
    prompt_ref: str
    raw_output_ref: str
    raw_output_hash: str
    occurrences: tuple[SemanticOccurrence, ...]
    warnings: tuple[str, ...] = ()
    capability_state: Literal["healthy", "partial", "unavailable"] = "healthy"
