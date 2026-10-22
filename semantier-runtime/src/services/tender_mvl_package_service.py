"""Build the governed canonical package used by the tender MVL source."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contracts.knowledge_reasoning_phase1 import KnowledgeScope, TenderPackageDocument
from eos.artifact_hashing import content_hash
from knowledge_extraction.service import KnowledgeExtractionService
from semantier.knowledge_feature_packs.source_elements import source_elements
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.knowledge_extraction_store import KnowledgeExtractionStore


def register_institutional_tender_source(
    *,
    db_path: str | Path,
    docx_path: str | Path,
    organization_id: str,
    workspace_id: str | None,
    package_id: str,
) -> dict[str, Any]:
    parsed = source_elements(docx_path)
    elements = list(parsed["elements"])
    if {element.get("family_id") for element in elements if element.get("family_id")} != set(range(1, 16)):
        raise ValueError("institutional tender source must preserve all 15 family IDs")
    normalized_text = "\n".join(str(element["text"]) for element in elements)
    builder = KnowledgeBuilderStore(db_path)
    source = builder.create_source(
        organization_id=organization_id,
        workspace_id=workspace_id,
        source_kind="file",
        source_ref=str(Path(docx_path).resolve()),
        source_text=normalized_text,
        source_metadata={
            "semantic_purpose": "sensitive_lexicon",
            "source_file_hash": parsed["document_hash"],
            "element_manifest_hash": parsed["element_manifest_hash"],
        },
    )
    rebound_elements = []
    for element in elements:
        rebound = dict(element)
        rebound["source_file_hash"] = parsed["document_hash"]
        rebound["source_id"] = source["source_id"]
        anchor = dict(rebound["anchor"])
        anchor["document_hash"] = source["source_hash"]
        rebound["anchor"] = anchor
        rebound["document_hash"] = source["source_hash"]
        rebound_elements.append(rebound)
    discovery = builder.create_discovery_run(
        organization_id=organization_id,
        workspace_id=workspace_id,
        source_id=source["source_id"],
        source_metadata={"package_id": package_id, "source_file_hash": parsed["document_hash"]},
    )
    scope = KnowledgeScope(documents=(TenderPackageDocument(
        document_id=f"institutional-source:{source['source_id']}",
        document_hash=source["source_hash"],
        role="main_tender",
        elements=tuple(rebound_elements),
    ),), package_id=package_id)
    KnowledgeExtractionStore(db_path).put_knowledge_scope(
        discovery_run_id=discovery["discovery_run_id"],
        organization_id=organization_id,
        workspace_id=workspace_id,
        scope=scope,
    )
    return {"source": source, "discovery": discovery, "scope": scope.model_dump(mode="json"), "source_hash": content_hash(normalized_text)}


def extract_institutional_tender_source(**kwargs: Any) -> dict[str, Any]:
    registered = register_institutional_tender_source(**kwargs)
    service = KnowledgeExtractionService(db_path=kwargs["db_path"])
    run = service.start(
        organization_id=kwargs["organization_id"],
        workspace_id=kwargs["workspace_id"],
        discovery_run_id=registered["discovery"]["discovery_run_id"],
        document_id=registered["scope"]["documents"][0]["document_id"],
        package_id=registered["scope"]["package_id"],
        provider="semantica",
        profile="tender_sensitive_v1",
    )
    return {**registered, "extraction_run": run}
