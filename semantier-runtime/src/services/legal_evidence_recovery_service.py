"""Compose source availability observations through existing governed owners."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.legal_evidence_recovery import (
    LegalEvidencePassageV1, LegalEvidenceRequestV1, LegalEvidenceResultV1,
    LegalEvidenceSourceV1, RequirementObservationV1, text_hash,
)
from services.runtime_semantic_source_region_service import RuntimeSemanticSourceRegionService
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.knowledge_extraction_store import KnowledgeExtractionStore

class LegalEvidenceRecoveryService:
    """Explicit source access only; no retrieval, interpretation, or authority transition."""

    def __init__(self, *, runtime_root: str | Path, db_path: str | Path):
        self._root = Path(runtime_root).resolve()
        self._db = Path(db_path)

    def recover(
        self, request: LegalEvidenceRequestV1, *,
        observations: tuple[RequirementObservationV1, ...],
        produced_at: datetime | None = None,
    ) -> LegalEvidenceResultV1:
        request = LegalEvidenceRequestV1.model_validate(request.model_dump(mode="json"))
        observations = tuple(RequirementObservationV1.model_validate(o.model_dump(mode="json")) for o in observations)
        resolver = RuntimeSemanticSourceRegionService(runtime_root=self._root, db_path=self._db)
        builder = KnowledgeBuilderStore(self._db, workspace_root=self._root)
        extraction = KnowledgeExtractionStore(self._db)
        sources, passages = [], []
        try:
            for ref in request.normalized_document_refs:
                regions = resolver.resolve(organization_id=request.organization_id,
                    workspace_id=request.workspace_id, normalized_document_artifact_ref=ref)
                edge = extraction.resolve_active_source_representation(artifact_ref=ref,
                    organization_id=request.organization_id, workspace_id=request.workspace_id)
                identities = builder.list_source_identities(organization_id=request.organization_id,
                    workspace_id=request.workspace_id, lineage_id=str(edge["source_id"]))
                matches = [i for i in identities if i.source_identity_ref == regions[0].source_binding.source_identity_ref]
                if len(matches) != 1:
                    raise ValueError("source identity must resolve uniquely")
                identity = matches[0]
                canonical_path = resolver._canonical_path(
                    regions[0].source_binding.canonical_source_ir.artifact_ref,
                    workspace_id=request.workspace_id)
                document = CanonicalSourceDocument.model_validate_json(canonical_path.read_text(encoding="utf-8"))
                # Verify original bytes within the same governed workspace; never host-path fallback.
                workspace = (self._root / "workspaces" / request.workspace_id).resolve()
                artifact = Path(identity.artifact_ref)
                original = (workspace / artifact).resolve()
                if artifact.is_absolute() or not original.is_relative_to(workspace):
                    raise ValueError("original source artifact escapes workspace")
                from hashlib import sha256
                if "sha256:" + sha256(original.read_bytes()).hexdigest() != identity.source_hash:
                    raise ValueError("original source content hash mismatch")
                source = LegalEvidenceSourceV1(normalized_document_ref=ref, identity=identity,
                    canonical_document=document, regions=regions)
                sources.append(source)
                for region in regions:
                    for anchor in region.evidence:
                        quote = anchor.evidence_ref.selector.exact_quote
                        passages.append(LegalEvidencePassageV1(
                            evidence_ref=anchor.evidence_ref.evidence_ref,
                            source_identity_ref=identity.source_identity_ref,
                            source_version=identity.source_version,
                            quote_text_hash=text_hash(quote), start_codepoint=0,
                            end_codepoint=len(quote), exact_quote=quote))
        finally:
            builder.close()
            extraction._conn.close()
        return LegalEvidenceResultV1.create(request=request, sources=tuple(sources),
            passages=tuple(passages), observations=observations,
            produced_at=produced_at or datetime.now(timezone.utc))
