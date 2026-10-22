"""CanonicalSourceIR utilities over the qualified native source-evidence providers."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from contracts.canonical_source_ir import CanonicalSourceDocument, SourceLocator, content_hash as content_hash
from semantier.knowledge_feature_packs.canonical_source_ir_native_adapter import adapt_docx_bytes as adapt_docx_bytes


class AmbiguousSourceMapping(ValueError):
    """Raised when an evidence lookup cannot identify one structural source."""

def resolve_unique_mapping(document: CanonicalSourceDocument, block_id: str) -> SourceLocator:
    matches = [locator for mapping in document.source_map.mappings if block_id in mapping.block_ids for locator in mapping.source_locators]
    if len(matches) != 1:
        raise AmbiguousSourceMapping(f"expected one locator for {block_id}, found {len(matches)}")
    return matches[0]


def persist_canonical_source_ir(document: CanonicalSourceDocument, root: str | Path, *,
                                organization_id: str, workspace_id: str) -> str:
    """Persist an append-only, tenant/workspace-scoped IR artifact and return its ref."""
    if not organization_id or not workspace_id:
        raise ValueError("organization_id and workspace_id are required")
    payload = document.model_dump(mode="json", by_alias=True, exclude_none=False)
    artifact_hash = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    base = Path(root).resolve()
    workspace_root = (
        base
        if base.name == workspace_id and base.parent.name == "workspaces"
        else base / "workspaces" / workspace_id
    )
    target = workspace_root / "artifacts" / "canonical-source-ir" / artifact_hash
    target.mkdir(parents=True, exist_ok=True)
    path = target / "canonical_source_ir.v1.json"
    if path.exists() and path.read_text(encoding="utf-8") != json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2):
        raise ValueError("CANONICAL_SOURCE_IR_ARTIFACT_CONFLICT")
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return f"artifacts/canonical-source-ir/{artifact_hash}/canonical_source_ir.v1.json"


def load_canonical_source_ir(artifact_ref: str) -> CanonicalSourceDocument:
    """Load a previously persisted CanonicalSourceIR artifact.

    The artifact path follows the convention used by
    :func:`persist_canonical_source_ir` (root / workspaces / ws /
    artifacts / canonical-source-ir / hash / file.json).  Other layouts are
    rejected so the extraction boundary cannot be tricked into consuming
    foreign JSON.
    """
    if not artifact_ref or "canonical-source-ir" not in artifact_ref:
        raise ValueError("canonical_ir_ref must point to a canonical-source-ir artifact")
    candidate = Path(artifact_ref)
    if candidate.is_absolute():
        path = candidate
        if not path.is_file():
            raise ValueError(f"canonical_source_ir artifact not found: {artifact_ref}")
    else:
        from runtime_paths import sqlite_db_path  # type: ignore
        environ = __import__("os").environ
        workspace_root = Path(
            environ.get(
                "SEMANTIER_WORKSPACES_ROOT",
                str(
                    Path(
                        environ.get(
                            "SEMANTIER_WORKSPACE_ROOT",
                            str(sqlite_db_path("workspace", ".semantier/workspace")),
                        )
                    )
                    / "workspaces"
                ),
            )
        )
        rel = candidate.as_posix()
        # Mirror the persist() layout: artifact lives under
        # workspace_root/workspaces/<ws>/artifacts/canonical-source-ir/...
        path = None
        workspaces_dir = workspace_root
        if workspaces_dir.is_dir():
            for ws_dir in sorted(workspaces_dir.iterdir()):
                candidate_path = (ws_dir / rel).resolve()
                if candidate_path.is_file():
                    path = candidate_path
                    break
        if path is None:
            raise ValueError(f"canonical_source_ir artifact not found: {artifact_ref}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CanonicalSourceDocument.model_validate(payload)


def canonical_elements_from_ir(document: CanonicalSourceDocument) -> list[dict[str, Any]]:
    """Adapt :class:`CanonicalSourceDocument` blocks into the canonical_elements shape.

    The returned list mirrors the existing knowledge-scope element contract
    (anchor_id / document_hash / start_offset / end_offset / exact_text) so
    the downstream :func:`_attach_canonical_evidence` flow does not change.
    Each element carries its ``SourceLocator`` provenance through the anchor
    metadata, so R2 EvidenceRefs built later still round-trip to the original
    DOCX structure.
    """
    elements: list[dict[str, Any]] = []
    source_hash = None
    for block in document.blocks:
        source_hash = source_hash or resolve_unique_mapping(document, block.block_id).source_hash
    for block in document.blocks:
        # The IR carries both container blocks (table) and content blocks
        # (paragraph / heading / list_item / cell / legal_article). Container
        # blocks have no span to anchor. Legal articles retain their original
        # source locator and text; do not silently drop them before induction.
        if block.block_type not in {"paragraph", "heading", "list_item", "cell", "legal_article"}:
            continue
        locator = resolve_unique_mapping(document, block.block_id)
        text = str(block.content) if isinstance(block.content, str) else (
            json.dumps(block.content, ensure_ascii=False) if block.content else ""
        )
        is_table_cell = bool(locator.table_id and locator.row_id and locator.column_id)
        # SourceLocator uses strings like "t1", "t1.r1", "t1.c1"; SourceAnchor
        # needs integer indexes for table/row/cell.  Extract the numeric tail
        # regardless of the prefix letter(s).
        def _int_from(value, *prefixes):
            if not value:
                return None
            for prefix in prefixes:
                if value.startswith(prefix):
                    tail = value[len(prefix):]
                    break
            else:
                tail = value.split(".", 1)[-1]
            digits = "".join(ch for ch in tail if ch.isdigit())
            try:
                return int(digits)
            except ValueError:
                return None

        def _paragraph_index(value):
            if not value:
                return None
            tail = value.lstrip("p")
            try:
                return int(tail)
            except ValueError:
                return None

        paragraph_index = _paragraph_index(locator.paragraph_id)
        anchor = {
            "anchor_id": block.block_id,
            "document_hash": locator.source_hash,
            "document_part": "table_cell" if is_table_cell else "paragraph",
            # CanonicalSourceIR locator ids are human-readable and one-based
            # (t1/r1/c1); the established Semantica element contract is
            # zero-based. Keep identity in ``source_locator`` and normalize
            # only this compatibility projection.
            "table_index": (_int_from(locator.table_id, "t") - 1) if is_table_cell else None,
            "row_index": (_int_from(locator.row_id, "t1.r") - 1) if is_table_cell else None,
            "cell_index": (_int_from(locator.column_id, "t1.c") - 1) if is_table_cell else None,
            "paragraph_index": paragraph_index,
            "start_offset": 0,
            "end_offset": len(text),
            "exact_text": text,
            "local_text_hash": block.content_hash,
        }
        family_id = 1 if block.block_type == "cell" else 2
        family = "table_cell_evidence" if block.block_type == "cell" else "structural_text_evidence"
        elements.append({
            "anchor_id": block.block_id,
            "anchor": anchor,
            "text": text,
            "family_id": family_id,
            "family": family,
            "document_hash": locator.source_hash or source_hash,
            "source_id": None,
            "source_block_id": block.block_id,
            "source_locator": locator.model_dump(mode="json", by_alias=True),
            "parser_trace_adapter_version": document.parser_trace.adapter_version,
            "canonical_ir_version": document.canonical_ir_version,
        })
    return elements


def exact_snapshot_text(document: CanonicalSourceDocument) -> tuple[str, dict]:
    """Read an explicitly lossless whole-text block without display normalization.

    Only the declared one-block snapshot mapping is supported. Other source IR
    shapes require their own qualified reversible mapping, never guessed joins.
    """
    if len(document.blocks) != 1:
        raise ValueError('SOURCE_MAPPING_NOT_READY: whole-text block required')
    block = document.blocks[0]
    locator = resolve_unique_mapping(document, block.block_id)
    text = block.content
    mapping = locator.native_metadata.get('source_text_mapping')
    if not isinstance(text, str) or mapping != {
        'version': 'unicode-codepoint-identity/1', 'start': 0, 'end': len(text),
        'text_sha256': sha256(text.encode('utf-8')).hexdigest(),
    }:
        raise ValueError('SOURCE_MAPPING_NOT_READY: exact Unicode mapping required')
    if block.content_hash != content_hash(text):
        raise ValueError('source block content hash mismatch')
    return text, mapping
