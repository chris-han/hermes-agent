"""Semantica-oriented candidate grounding over a provider-neutral generator."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from contracts.ai_grounding import (
    AI_GROUNDING_BATCH_SCHEMA_VERSION,
    AiGroundingBatch,
    AiGroundingContractError,
    AiGroundingResult,
    StructuredGenerationResult,
    StructuredGenerator,
)
from prompts.loader import PromptAsset


@dataclass(frozen=True)
class GroundingCandidateInput:
    assertion_id: str
    subject_text: str
    predicate_text: str
    object_text: str
    evidence: tuple[Mapping[str, Any], ...]
    source_hash: str
    candidate_graph_version: str

    @property
    def anchor_ids(self) -> set[str]:
        return {str(item["anchor_id"]) for item in self.evidence}


@dataclass(frozen=True)
class GroundingBatchOutcome:
    results: tuple[AiGroundingResult, ...]
    generations: tuple[StructuredGenerationResult, ...]


class SemanticaGroundingAssistant:
    def __init__(
        self, llm: StructuredGenerator, *, batch_size: int = 12, max_concurrency: int = 4
    ) -> None:
        if batch_size < 1 or max_concurrency < 1:
            raise ValueError("batch size and concurrency must be positive")
        self._llm = llm
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency

    async def ground(
        self, *, candidates: Sequence[GroundingCandidateInput], prompt: PromptAsset
    ) -> GroundingBatchOutcome:
        batches = [candidates[index:index + self._batch_size]
                   for index in range(0, len(candidates), self._batch_size)]
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def run_batch(batch: Sequence[GroundingCandidateInput]):
            async with semaphore:
                payload = {"candidates": [self._candidate_payload(item) for item in batch]}
                generated = await self._llm.generate_structured(
                    instructions=prompt.text,
                    input_text=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    json_schema=AiGroundingBatch.model_json_schema(),
                )
                parsed = AiGroundingBatch.model_validate(
                    self._normalize_provider_envelope(generated.parsed)
                )
                self._validate_batch(batch, parsed.results)
                return parsed.results, generated

        completed = await asyncio.gather(*(run_batch(batch) for batch in batches))
        results = tuple(result for results, _ in completed for result in results)
        expected_ids = {item.assertion_id for item in candidates}
        returned_ids = [item.assertion_id for item in results]
        if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != expected_ids:
            raise AiGroundingContractError("result coverage mismatch")
        return GroundingBatchOutcome(
            results=results,
            generations=tuple(generation for _, generation in completed),
        )

    @staticmethod
    def _normalize_provider_envelope(parsed: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(parsed)
        if normalized.get("schema_version") == "semantier_ai_grounding_batch_v1":
            normalized["schema_version"] = AI_GROUNDING_BATCH_SCHEMA_VERSION
        return normalized

    @staticmethod
    def _candidate_payload(candidate: GroundingCandidateInput) -> dict[str, Any]:
        return {
            "assertion_id": candidate.assertion_id,
            "subject_text": candidate.subject_text,
            "predicate_text": candidate.predicate_text,
            "object_text": candidate.object_text,
            "evidence": [dict(item) for item in candidate.evidence],
            "source_hash": candidate.source_hash,
            "candidate_graph_version": candidate.candidate_graph_version,
        }

    @staticmethod
    def _validate_batch(
        candidates: Sequence[GroundingCandidateInput], results: Sequence[AiGroundingResult]
    ) -> None:
        expected = {item.assertion_id: item for item in candidates}
        returned = [item.assertion_id for item in results]
        if len(returned) != len(set(returned)) or set(returned) != set(expected):
            raise AiGroundingContractError("result coverage mismatch")
        for result in results:
            if not set(result.evidence_anchor_refs).issubset(expected[result.assertion_id].anchor_ids):
                raise AiGroundingContractError("result references noncanonical evidence")
