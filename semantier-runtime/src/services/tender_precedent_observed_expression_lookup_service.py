from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Protocol

from contracts.tender_sensitive_term_detection_modeling_core import validate_ascii_identifier
from runtime_paths import sqlite_db_path


_GOVERNED_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class ObservedExpressionStore(Protocol):
    def list_human_accepted_observed_expressions(
        self,
        *,
        concept_ids: list[str] | None = None,
        accepted_graph_release_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...


class TenderPrecedentObservedExpressionLookupService(Protocol):
    def list_human_accepted_observed_expressions(
        self,
        *,
        concept_ids: list[str] | None = None,
        accepted_graph_release_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...


def _default_db_path() -> Path:
    return Path(
        os.environ.get(
            "SEMANTIER_EOS_DB_PATH",
            str(sqlite_db_path("eos.db", ".semantier/eos.db")),
        )
    )


def _normalize_governed_ids(values: list[str] | None, *, field: str) -> list[str] | None:
    if not values:
        return None
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            return None
        try:
            identifier = validate_ascii_identifier(value, field)
        except ValueError:
            return None
        if _GOVERNED_ID_PATTERN.fullmatch(identifier) is None:
            return None
        normalized.add(identifier)
    return sorted(normalized) or None


class StoreBackedTenderPrecedentObservedExpressionLookupService:
    """Read human-accepted observed expressions used as tender precedents."""

    def __init__(
        self,
        knowledge_store: ObservedExpressionStore | None = None,
        *,
        db_path: str | Path | None = None,
    ) -> None:
        if knowledge_store is None:
            from storage.knowledge_extraction_store import KnowledgeExtractionStore

            knowledge_store = KnowledgeExtractionStore(
                Path(db_path) if db_path is not None else _default_db_path()
            )
        self._knowledge_store = knowledge_store

    def list_human_accepted_observed_expressions(
        self,
        *,
        concept_ids: list[str] | None = None,
        accepted_graph_release_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_concept_ids = _normalize_governed_ids(concept_ids, field="concept_id")
        normalized_release_ids = _normalize_governed_ids(
            accepted_graph_release_ids,
            field="accepted_graph_release_id",
        )
        if normalized_concept_ids is None or normalized_release_ids is None:
            return []
        return self._knowledge_store.list_human_accepted_observed_expressions(
            concept_ids=normalized_concept_ids,
            accepted_graph_release_ids=normalized_release_ids,
        )
