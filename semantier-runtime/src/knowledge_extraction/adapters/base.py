from __future__ import annotations

from typing import Protocol

from knowledge_extraction.contracts import SourceExtractionRequest, SourceExtractionResult


class SourceExtractionAdapter(Protocol):
    def extract(self, request: SourceExtractionRequest) -> SourceExtractionResult: ...
