"""Resolve authenticated normalized documents into governed semantic source regions."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import re
from typing import Any

from contracts.canonical_source_ir import SourceLocator
from contracts.evidence_location import EvidenceRef
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_semantic_source_region import (
    RuntimeSemanticSourceRegionV1,
    RuntimeSourceAnchorV1,
    RuntimeSourceBindingV1,
)
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import (
    load_canonical_source_ir,
)
from semantier.knowledge_feature_packs.evidence_selector_builder import (
    evidence_ref_for_block,
    build_evidence_selectors,
)
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.knowledge_extraction_store import KnowledgeExtractionStore


class RuntimeSemanticSourceRegionService:
    """Derive target regions from exact server-resolved source authority."""

    def __init__(
        self,
        *,
        runtime_root: str | Path,
        db_path: str | Path,
        max_region_chars: int = 12_000,
    ) -> None:
        self._runtime_root = Path(runtime_root).resolve()
        self._db_path = Path(db_path)
        if (
            not isinstance(max_region_chars, int)
            or isinstance(max_region_chars, bool)
            or max_region_chars < 1
        ):
            raise ValueError("max_region_chars must be a positive integer")
        self._max_region_chars = max_region_chars

    def resolve(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        normalized_document_artifact_ref: str,
    ) -> tuple[RuntimeSemanticSourceRegionV1, ...]:
        edge = KnowledgeExtractionStore(
            self._db_path
        ).resolve_active_source_representation(
            artifact_ref=normalized_document_artifact_ref,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        builder = KnowledgeBuilderStore(
            self._db_path, workspace_root=self._runtime_root
        )
        source = builder.get_source(str(edge["source_id"]))
        if (
            source is None
            or source.get("organization_id") != organization_id
            or source.get("workspace_id") != workspace_id
        ):
            raise ValueError("governed source is outside the authenticated scope")
        builder._verify_source_row(source)  # noqa: SLF001 - store-owned integrity rule
        canonical_ref = str(
            (source.get("source_metadata") or {}).get("canonical_ir_ref") or ""
        )
        canonical_path = self._canonical_path(canonical_ref, workspace_id=workspace_id)
        try:
            canonical_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("CanonicalSourceIR content hash mismatch") from exc
        canonical_hash = "sha256:" + content_hash(canonical_payload)
        if canonical_path.parent.name != canonical_hash.removeprefix("sha256:"):
            raise ValueError("CanonicalSourceIR content hash mismatch")
        document = load_canonical_source_ir(str(canonical_path))
        identities = builder.list_source_identities(
            organization_id=organization_id,
            workspace_id=workspace_id,
            lineage_id=str(edge["source_id"]),
        )
        identity = next(
            (
                item
                for item in identities
                if item.source_identity_ref == document.source_identity_ref
            ),
            None,
        )
        if identity is None:
            raise ValueError(
                "CanonicalSourceIR SourceIdentity is not governed by this source"
            )
        text_blocks = [
            block
            for block in document.blocks
            if isinstance(block.content, str) and block.content.strip()
        ]
        if not text_blocks:
            raise ValueError("CanonicalSourceIR contains no semantic source regions")
        groups: list[list[Any]] = []
        current: list[Any] = []
        current_chars = 0
        for block in text_blocks:
            size = len(block.content.strip()) + (1 if current else 0)
            if current and current_chars + size > self._max_region_chars:
                groups.append(current)
                current, current_chars = [], 0
            current.append(block)
            current_chars += size
        if current:
            groups.append(current)

        binding = RuntimeSourceBindingV1(
            schema_version="runtime_source_binding.v1",
            source_identity_ref=identity.source_identity_ref,
            source_hash=identity.source_hash,
            canonical_source_ir=ArtifactPinV1(
                artifact_kind="canonical_source_ir",
                artifact_ref=canonical_ref,
                artifact_hash=canonical_hash,
            ),
        )
        # Reuse the existing batch owner; preserve every canonical selector.
        selectors = build_evidence_selectors(document, text_blocks)
        regions: list[RuntimeSemanticSourceRegionV1] = []
        for index, blocks in enumerate(groups, start=1):
            evidence: list[RuntimeSourceAnchorV1] = []
            for block in blocks:
                item = evidence_ref_for_block(
                    document,
                    block,
                    source_identity=identity,
                    selector=selectors[block.block_id],
                    evidence_role="target",
                    created_at=identity.created_at,
                )
                item = EvidenceRef.model_validate(
                    item.model_dump(mode="json", by_alias=True)
                    | {
                        "provenance": dict(item.provenance)
                        | {
                            "canonical_source_ir_ref": canonical_ref,
                            "canonical_source_ir_hash": canonical_hash,
                        }
                    }
                )
                exact_quote = item.selector.exact_quote
                if exact_quote is None:
                    raise ValueError("source anchor requires a nonblank exact quote")
                evidence.append(
                    RuntimeSourceAnchorV1(
                        evidence_ref=item,
                        locator=SourceLocator.model_validate(
                            item.provenance["source_locator"]
                        ),
                        text_hash="sha256:"
                        + sha256(exact_quote.encode("utf-8")).hexdigest(),
                    )
                )
            quotes: list[str] = []
            for anchor in evidence:
                exact_quote = anchor.evidence_ref.selector.exact_quote
                if exact_quote is None:
                    raise ValueError("source anchor requires a nonblank exact quote")
                quotes.append(exact_quote.strip())
            text = "\n".join(quotes)
            regions.append(
                RuntimeSemanticSourceRegionV1(
                    schema_version="runtime_semantic_source_region.v1",
                    source_region_ref=(
                        f"source-region:{identity.source_identity_ref}:{index}"
                    ),
                    text=text,
                    text_hash="sha256:" + sha256(text.encode("utf-8")).hexdigest(),
                    source_binding=binding,
                    evidence=tuple(evidence),
                )
            )
        return tuple(regions)

    def _canonical_path(self, artifact_ref: str, *, workspace_id: str) -> Path:
        if not artifact_ref or Path(artifact_ref).is_absolute():
            raise ValueError(
                "canonical_ir_ref must be a governed workspace-relative ref"
            )
        if (
            re.fullmatch(
                r"artifacts/canonical-source-ir/[0-9a-f]{64}/canonical_source_ir\.v1\.json",
                artifact_ref,
            )
            is None
        ):
            raise ValueError(
                "canonical_ir_ref must use the governed CanonicalSourceIR layout"
            )
        workspace = (self._runtime_root / "workspaces" / workspace_id).resolve()
        candidate = (workspace / artifact_ref).resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("canonical_ir_ref escapes the governed workspace") from exc
        if not candidate.is_file():
            raise ValueError("canonical_source_ir artifact not found")
        return candidate


__all__ = ["RuntimeSemanticSourceRegionService"]
