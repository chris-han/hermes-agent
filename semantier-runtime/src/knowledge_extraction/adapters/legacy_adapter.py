"""Compatibility extraction adapter for the locked deterministic path."""

from __future__ import annotations

from typing import Callable

from knowledge_extraction.adapters.langextract_adapter import LangExtractAdapter
from knowledge_extraction.contracts import SourceExtractionRequest, SourceExtractionResult


LEGACY_MODEL_REF = "legacy.deterministic.v1"
LEGACY_PROMPT_REF = "legacy.match_terms.v1"


class LegacyExtractionAdapter:
    provider = "legacy"

    def __init__(self, *, artifact_writer: Callable[[str, str], str]) -> None:
        self._artifact_writer = artifact_writer

    def capability(self) -> dict[str, str]:
        return {"provider": self.provider, "state": "healthy", "detail": "legacy.v1"}

    def extract(self, request: SourceExtractionRequest) -> SourceExtractionResult:
        terms = [str(item) for item in request.provider_options.get("match_terms") or []]
        extractions = []
        for term in terms:
            start = request.text.find(term)
            while start >= 0:
                extractions.append(
                    _LegacyExtraction(
                        extraction_text=term,
                        start=start,
                        end=start + len(term),
                        attributes=dict(request.provider_options.get("attributes") or {}),
                        group_index=_clause_index(request.text, start),
                    )
                )
                start = request.text.find(term, start + len(term))
        result = LangExtractAdapter(
            artifact_writer=self._artifact_writer,
            extract_fn=lambda **_: _LegacyResult(extractions),
            provider_version="legacy.v1",
            provider_name="legacy",
        ).extract(request)
        occurrences = tuple(
            occurrence.model_copy(
                update={"model_ref": LEGACY_MODEL_REF, "prompt_ref": LEGACY_PROMPT_REF}
            )
            for occurrence in result.occurrences
        )
        return result.model_copy(
            update={
                "provider": self.provider,
                "provider_version": "legacy.v1",
                "model_ref": LEGACY_MODEL_REF,
                "prompt_ref": LEGACY_PROMPT_REF,
                "occurrences": occurrences,
            }
        )


class _Interval:
    def __init__(self, start: int, end: int) -> None:
        self.start_pos = start
        self.end_pos = end


class _LegacyExtraction:
    extraction_class = "legacy_occurrence"
    alignment_status = "exact"
    group_index = None
    extraction_index = None

    def __init__(self, *, extraction_text: str, start: int, end: int, attributes: dict, group_index: int) -> None:
        self.extraction_text = extraction_text
        self.char_interval = _Interval(start, end)
        self.attributes = attributes
        self.group_index = group_index


class _LegacyResult:
    warnings: tuple[str, ...] = ()

    def __init__(self, extractions: list[_LegacyExtraction]) -> None:
        self.extractions = extractions


def _clause_index(text: str, offset: int) -> int:
    return sum(text[:offset].count(mark) for mark in "。！？；\n")
