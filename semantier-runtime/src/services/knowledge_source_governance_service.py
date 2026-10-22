from __future__ import annotations

from storage.hosted_postgres import sql_query, integrity_errors

from pathlib import Path
from typing import Any, Protocol

from contracts.tender_sensitive_term_detection_modeling_core import RuntimeContextDTO


class KnowledgeSourceGovernanceService(Protocol):
    def register_controlled_document(
        self,
        *,
        knowledge_source_id: str,
        source_ref: str,
        source_hash: str,
        semantic_purpose: str,
        context: RuntimeContextDTO,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def get_admissible_source(
        self,
        knowledge_source_id: str,
        organization_id: str,
    ) -> dict[str, Any]: ...

    def assert_source_routable(self, source_kind: str, semantic_purpose: str) -> None: ...


class StoreBackedKnowledgeSourceGovernanceService:
    def __init__(self, db_path: str | Path | None = None) -> None:
        from storage.knowledge_source_activation_store import KnowledgeSourceActivationStore

        self._store = KnowledgeSourceActivationStore(db_path=db_path)

    def register_controlled_document(
        self,
        *,
        knowledge_source_id: str,
        source_ref: str,
        source_hash: str,
        semantic_purpose: str,
        context: RuntimeContextDTO,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return self._store.register_knowledge_source(
                knowledge_source_id=knowledge_source_id,
                source_kind="controlled_document",
                semantic_purpose=semantic_purpose,
                semantic_tier="T4",
                authority_role="interpretive",
                source_lifecycle_status="approved",
                source_ref=source_ref,
                source_hash=source_hash,
                source_version=(metadata or {}).get("source_version"),
            )
        except integrity_errors():
            return self.get_admissible_source(
                knowledge_source_id,
                context.organization_id,
                require_admissible=False,
            )

    def get_admissible_source(
        self,
        knowledge_source_id: str,
        organization_id: str,
        *,
        require_admissible: bool = True,
    ) -> dict[str, Any]:
        row = self._store._conn.execute(
            sql_query("SELECT * FROM knowledge_source WHERE knowledge_source_id = ?", 'SELECT * FROM knowledge_source WHERE knowledge_source_id = %s'),
            (knowledge_source_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("KNOWLEDGE_SOURCE_NOT_FOUND")
        source = dict(row)
        if require_admissible and source["source_lifecycle_status"] not in {"approved", "validated"}:
            raise RuntimeError("KNOWLEDGE_SOURCE_NOT_ADMISSIBLE")
        return source

    def assert_source_routable(self, source_kind: str, semantic_purpose: str) -> None:
        if source_kind != "controlled_document" or semantic_purpose != "sensitive_lexicon":
            raise RuntimeError("SENSITIVE_LEXICON_SOURCE_REQUIRED")

    def resolve_controlled_document_compiler_route(self, knowledge_source_id: str) -> str:
        return self._store.resolve_controlled_document_compiler_route(knowledge_source_id)
