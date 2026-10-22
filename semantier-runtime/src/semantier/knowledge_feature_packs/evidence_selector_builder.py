"""Build representation-independent evidence identity from CanonicalSourceIR."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from hashlib import sha256
import re

from contracts.canonical_source_ir import CanonicalSourceDocument, SourceBlock, SourceLocator
from contracts.evidence_location import EvidenceRef, EvidenceSelector, SourceIdentity, TableSelector
from contracts.ontology_building import ArtifactPinV1
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import resolve_unique_mapping


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _locator_for(document: CanonicalSourceDocument, block: SourceBlock) -> SourceLocator:
    return resolve_unique_mapping(document, block.block_id)


def _selector(block: SourceBlock, locator: SourceLocator, parent: SourceBlock | None, headers: list[str]) -> EvidenceSelector:
    value = block.content if isinstance(block.content, str) else ""
    path = [f"block:{block.block_type}", f"ordinal:{block.ordinal}"]
    for key in ("sectionId", "paragraphId", "tableId", "rowId", "columnId"):
        attr = key[0].lower() + key[1:]
        raw = getattr(locator, attr, None)
        if raw:
            path.append(f"{key}:{raw}")
    path.append(f"localContentHash:{block.content_hash}")
    if block.parent_block_id:
        if parent:
            path.append(f"parentContentHash:{parent.content_hash}")

    table = None
    if block.block_type == "cell":
        table = TableSelector(
            tableOrdinal=block.structure.get("tableOrdinal"), headingPath=[], headerPath=headers,
            rowKey=str(locator.row_id) if locator.row_id else None,
            columnKey=str(locator.column_id) if locator.column_id else None,
            cellText=value,
        )
    return EvidenceSelector(
        schemaVersion="semantier.evidence_selector.v1",
        selectorKind="table_cell" if table else "composite",
        exactQuote=value or None, prefix=None, suffix=None,
        normalizedQuote=_normalise(value) if value else None,
        structuralPath=path, table=table,
        sourceElementRef=locator.native_id,
    )


def build_evidence_selector(document: CanonicalSourceDocument, block: SourceBlock) -> EvidenceSelector:
    locator = _locator_for(document, block)
    parent = next((candidate for candidate in document.blocks if candidate.block_id == block.parent_block_id), None) if block.parent_block_id else None
    headers = []
    if block.block_type == "cell":
        table_id = locator.table_id
        cells = [b for b in document.blocks if b.block_type == "cell" and _locator_for(document, b).table_id == table_id]
        headers = [str(b.content) for b in sorted(cells, key=lambda b: b.ordinal)
                   if _locator_for(document, b).row_id == f"{table_id}.r1"]
    return _selector(block, locator, parent, headers)


def build_evidence_selectors(document: CanonicalSourceDocument, blocks: list[SourceBlock]) -> dict[str, EvidenceSelector]:
    """Batch the same native selectors; indexes are local to this exact IR read.

    No truncation/cache/persistence or alternate identity rule is introduced.
    Ambiguous/missing mappings retain the single-selector owner's refusal.
    """
    from semantier.knowledge_feature_packs.canonical_source_ir_adapter import AmbiguousSourceMapping
    mappings = {}
    for mapping in document.source_map.mappings:
        for block_id in mapping.block_ids:
            mappings.setdefault(block_id, []).extend(mapping.source_locators)
    def locator_for(block_id):
        matches = mappings.get(block_id, ())
        if len(matches) != 1:
            raise AmbiguousSourceMapping(f"expected one locator for {block_id}, found {len(matches)}")
        return matches[0]
    parents = {}
    for block in document.blocks:
        parents.setdefault(block.block_id, block)
    headers = {}
    if any(block.block_type == "cell" for block in blocks):
        for cell in sorted((block for block in document.blocks if block.block_type == "cell"), key=lambda block: block.ordinal):
            locator = locator_for(cell.block_id)
            if locator.row_id == f"{locator.table_id}.r1":
                headers.setdefault(locator.table_id, []).append(str(cell.content))
    return {block.block_id: _selector(block, locator_for(block.block_id), parents.get(block.parent_block_id),
                                     headers.get(locator_for(block.block_id).table_id, [])) for block in blocks}


def evidence_ref_identity(*, organization_id: str, workspace_id: str,
                          source_identity_ref: str, selector_hash: str,
                          evidence_role: str) -> str:
    """Return the canonical EvidenceRef identity for frozen selector material."""
    identity_payload = {
        "organization_id": organization_id,
        "workspace_id": workspace_id,
        "source_identity_ref": source_identity_ref,
        "selector_hash": selector_hash,
        "evidence_role": evidence_role,
    }
    return "evidence:" + sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evidence_ref_pin(evidence: EvidenceRef) -> ArtifactPinV1:
    """Pin native evidence identity, not rendering metadata or creation time.

    This verifies identity consistency only; source access and membership remain
    the responsibility of the governed source/object owners.
    """
    evidence = EvidenceRef.model_validate(evidence.model_dump(mode="json", by_alias=True))
    selector_hash = evidence.selector.selector_hash()
    expected_ref = evidence_ref_identity(
        organization_id=evidence.organization_id, workspace_id=evidence.workspace_id,
        source_identity_ref=evidence.source_identity_ref, selector_hash=selector_hash,
        evidence_role=evidence.evidence_role,
    )
    if evidence.selector_hash != selector_hash or evidence.evidence_ref != expected_ref:
        raise ValueError("evidence identity mismatch")
    return ArtifactPinV1(artifact_kind="evidence_ref", artifact_ref=expected_ref,
                         artifact_hash="sha256:" + expected_ref.removeprefix("evidence:"))


def evidence_ref_for_block(document: CanonicalSourceDocument, block: SourceBlock, *, source_identity: SourceIdentity,
                           evidence_role: str = "source", created_at: datetime | None = None,
                           selector: EvidenceSelector | None = None) -> EvidenceRef:
    selector = selector or build_evidence_selector(document, block)
    evidence_ref = evidence_ref_identity(
        organization_id=source_identity.organization_id,
        workspace_id=source_identity.workspace_id,
        source_identity_ref=source_identity.source_identity_ref,
        selector_hash=selector.selector_hash(),
        evidence_role=evidence_role,
    )
    locator = _locator_for(document, block)
    provenance = {
        "source_block_id": block.block_id,
        "source_hash": locator.source_hash,
        "source_locator": locator.model_dump(mode="json", by_alias=True),
        "artifact_ref": source_identity.artifact_ref,
        "canonical_ir_version": document.canonical_ir_version,
    }
    return EvidenceRef(
        schemaVersion="semantier.evidence_ref.v1", evidenceRef=evidence_ref,
        organizationId=source_identity.organization_id, workspaceId=source_identity.workspace_id,
        sourceIdentityRef=source_identity.source_identity_ref,
        selectorSchemaVersion="semantier.evidence_selector.v1", selectorHash=selector.selector_hash(),
        selector=selector, evidenceRole=evidence_role, provenance=provenance,
        createdAt=created_at or datetime.now(timezone.utc),
    )
