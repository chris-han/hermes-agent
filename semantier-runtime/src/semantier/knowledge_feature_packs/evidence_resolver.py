"""Deterministic DOCX evidence resolution and source lineage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from contracts.canonical_source_ir import CanonicalSourceDocument, SourceBlock, SourceLocator
from contracts.evidence_location import EvidenceRef, ResolvedLocation
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import AmbiguousSourceMapping, resolve_unique_mapping
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_pin

RESOLVER_VERSION = "semantier.docx-evidence-resolver.v1"


@dataclass(frozen=True)
class EvidenceLineage:
    evidence_ref: str
    source_identity_ref: str
    source_block_id: str
    source_map: dict[str, Any]
    source_locator: dict[str, Any]
    pinned_original_docx: str


def lineage_for_evidence_ref(evidence_ref: EvidenceRef, document: CanonicalSourceDocument) -> EvidenceLineage:
    block_id = str(evidence_ref.provenance.get("source_block_id") or "")
    block = next((item for item in document.blocks if item.block_id == block_id), None)
    if block is None and evidence_ref.selector.source_element_ref:
        candidates = []
        for candidate in document.blocks:
            try:
                candidate_locator = resolve_unique_mapping(document, candidate.block_id)
            except AmbiguousSourceMapping:
                continue
            if candidate_locator.native_id == evidence_ref.selector.source_element_ref:
                candidates.append(candidate)
        if len(candidates) == 1:
            block = candidates[0]
            block_id = block.block_id
        elif len(candidates) > 1:
            raise AmbiguousSourceMapping("relocated source element is not unique")
    if block is None:
        raise ValueError("SOURCE_BLOCK_NOT_FOUND")
    locator = resolve_unique_mapping(document, block.block_id)
    mapping = next(mapping for mapping in document.source_map.mappings if block.block_id in mapping.block_ids)
    return EvidenceLineage(
        evidence_ref=evidence_ref.evidence_ref, source_identity_ref=evidence_ref.source_identity_ref,
        source_block_id=block.block_id,
        source_map=mapping.model_dump(mode="json", by_alias=True),
        source_locator=locator.model_dump(mode="json", by_alias=True),
        pinned_original_docx=str(evidence_ref.provenance.get("artifact_ref") or ""),
    )


def _location(evidence_ref: EvidenceRef, document: CanonicalSourceDocument, block: SourceBlock,
              locator: SourceLocator, *, status: str, artifact_ref: str) -> ResolvedLocation:
    content = block.content if isinstance(block.content, str) else ""
    if locator.format == "tabular":
        metadata = locator.native_metadata
        return ResolvedLocation(
            schemaVersion="semantier.resolved_location.v1", evidenceRef=evidence_ref.evidence_ref,
            sourceIdentityRef=evidence_ref.source_identity_ref, representationType="tabular_canonical",
            representationRef=artifact_ref, representationHash=locator.source_hash,
            locationType="table_cell", status=status,
            location={"kind": "tabular_cell", "sheetName": metadata["sheet_name"],
                "rowNumber": metadata["row_number"], "columnNumber": metadata["column_number"],
                "headerRow": metadata["header_row"], "cellText": content},
            matchedText=content, resolverVersion="semantier.tabular-evidence-resolver.v1",
            resolvedAt=datetime.now(timezone.utc),
        )
    exact = evidence_ref.selector.exact_quote or content
    requested_start = evidence_ref.provenance.get("start_offset")
    start_offset = requested_start if isinstance(requested_start, int) and content.startswith(exact, requested_start) else content.find(exact)
    if start_offset < 0:
        start_offset = 0
        exact = content
    structural_path = list(locator.ordinal_path)
    if not structural_path:
        structural_path = [locator.paragraph_id or locator.native_id or block.block_id]
    if block.block_type == "cell":
        location_type = "table_cell"
        location = {"kind": "docx_structural_range", "structuralPath": structural_path,
                    "startOffset": start_offset, "endOffset": start_offset + len(exact), "exactText": exact}
    else:
        location_type = "character_range"
        location = {"kind": "docx_structural_range", "structuralPath": structural_path,
                    "startOffset": start_offset, "endOffset": start_offset + len(exact), "exactText": exact}
    return ResolvedLocation(
        schemaVersion="semantier.resolved_location.v1", evidenceRef=evidence_ref.evidence_ref,
        sourceIdentityRef=evidence_ref.source_identity_ref, representationType="docx_structural",
        representationRef=artifact_ref, representationHash=locator.source_hash,
        locationType=location_type, status=status, location=location,
        matchedText=exact,
        resolverVersion=RESOLVER_VERSION, resolvedAt=datetime.now(timezone.utc),
    )


def _pdf_location(evidence_ref: EvidenceRef, block: SourceBlock, locator: SourceLocator,
                  *, artifact_ref: str) -> ResolvedLocation | None:
    metadata = locator.native_metadata or {}
    native = metadata.get("locator") if isinstance(metadata.get("locator"), dict) else metadata
    if native.get("kind") != "pdf_bbox":
        return None
    content = block.content if isinstance(block.content, str) else ""
    exact = evidence_ref.selector.exact_quote or ""
    requested_start = evidence_ref.provenance.get("start_offset")
    if (("start_offset" in evidence_ref.provenance and type(requested_start) is not int)
            or ("end_offset" in evidence_ref.provenance and
                (type(evidence_ref.provenance["end_offset"]) is not int or type(requested_start) is not int))):
        return _structured(evidence_ref, "unresolved", artifact_ref, "PDF explicit text span is malformed")
    if not isinstance(requested_start, int):
        occurrences = [index for index in range(len(content)) if content.startswith(exact, index)] if exact else []
        if len(occurrences) != 1:
            return _structured(evidence_ref, "unresolved", artifact_ref, "PDF text span is ambiguous")
        requested_start = occurrences[0]
    requested_end = evidence_ref.provenance.get("end_offset")
    if not isinstance(requested_end, int):
        requested_end = requested_start + len(exact)
    if requested_start < 0 or requested_end <= requested_start or content[requested_start:requested_end] != exact:
        return _structured(evidence_ref, "unresolved", artifact_ref, "PDF text span does not match")
    spans = [span for span in native.get("text_spans", [])
             if isinstance(span, dict) and type(span.get("start_offset")) is int
             and type(span.get("end_offset")) is int
             and 0 <= span["start_offset"] < span["end_offset"] <= len(content)
             and span["end_offset"] > requested_start and span["start_offset"] < requested_end]
    covered_until = requested_start
    for span in sorted(spans, key=lambda item: item["start_offset"]):
        if span["start_offset"] > covered_until:
            gap = content[covered_until:span["start_offset"]]
            if not gap or not gap.isspace():
                break
        covered_until = max(covered_until, span["end_offset"])
    bbox: list[float]
    precision = "block"
    status = "block"
    if spans and covered_until >= requested_end:
        boxes = [span.get("bbox") for span in spans]
        if all(isinstance(box, list) and len(box) == 4 for box in boxes):
            bbox = [min(float(box[0]) for box in boxes), min(float(box[1]) for box in boxes),
                    max(float(box[2]) for box in boxes), max(float(box[3]) for box in boxes)]
            precision = "span"
            status = "exact"
        else:
            bbox = list(native.get("bbox") or [])
    else:
        bbox = list(native.get("bbox") or [])
    if len(bbox) != 4:
        return ResolvedLocation(schemaVersion="semantier.resolved_location.v1", evidenceRef=evidence_ref.evidence_ref,
            sourceIdentityRef=evidence_ref.source_identity_ref, representationType="pdf_render",
            representationRef=artifact_ref, representationHash=locator.source_hash, locationType="other",
            status="text_fallback", location={"kind": "text_fallback", "exactText": exact,
                "page": native.get("page"), "reason": "historical_pdf_without_geometry"}, matchedText=exact,
            resolverVersion=RESOLVER_VERSION, resolvedAt=datetime.now(timezone.utc))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    return ResolvedLocation(schemaVersion="semantier.resolved_location.v1", evidenceRef=evidence_ref.evidence_ref,
        sourceIdentityRef=evidence_ref.source_identity_ref, representationType="pdf_render",
        representationRef=artifact_ref, representationHash=locator.source_hash, locationType="pdf_text_rects",
        status=status, location={"kind": "pdf_page_rect", "page": int(native.get("page") or 1),
            "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "unit": "pdf-point",
            "origin": "top-left", "precision": precision, "exactText": exact if precision == "span" else None},
        matchedText=exact, resolverVersion=RESOLVER_VERSION, resolvedAt=datetime.now(timezone.utc))


def resolve_evidence_ref(evidence_ref: EvidenceRef, document: CanonicalSourceDocument, *, artifact_ref: str | None = None,
                         selector_hash: str | None = None) -> ResolvedLocation:
    artifact = artifact_ref or str(evidence_ref.provenance.get("artifact_ref") or "artifact:canonical-source")
    if selector_hash is not None and selector_hash != evidence_ref.selector_hash:
        return _structured(evidence_ref, "unresolved", artifact, "selector hash mismatch")
    if selector_hash is not None:
        try:
            evidence_ref_pin(evidence_ref)
        except ValueError:
            return _structured(evidence_ref, "unresolved", artifact, "evidence identity mismatch")
    if document.source_identity_ref != evidence_ref.source_identity_ref:
        return _structured(evidence_ref, "source_changed", artifact, "source identity mismatch")
    if evidence_ref.provenance.get("source_hash") and evidence_ref.provenance["source_hash"] != _document_hash(document):
        return _structured(evidence_ref, "source_changed", artifact, "source bytes changed")
    block_id = str(evidence_ref.provenance.get("source_block_id") or "")
    block = next((item for item in document.blocks if item.block_id == block_id), None)
    status = "exact"
    if block is None and evidence_ref.selector.source_element_ref:
        candidates = []
        for candidate in document.blocks:
            try:
                candidate_locator = resolve_unique_mapping(document, candidate.block_id)
            except AmbiguousSourceMapping:
                continue
            if candidate_locator.native_id == evidence_ref.selector.source_element_ref:
                candidates.append(candidate)
        if len(candidates) == 1:
            block = candidates[0]
            block_id = block.block_id
            status = "relocated_exact"
        elif len(candidates) > 1:
            return _structured(evidence_ref, "ambiguous", artifact, "relocated source element is not unique")
    if block is None:
        return _structured(evidence_ref, "unresolved", artifact, "source block unavailable")
    try:
        locator = resolve_unique_mapping(document, block_id)
    except AmbiguousSourceMapping:
        return _structured(evidence_ref, "ambiguous", artifact, "source mapping is not unique")
    pdf_location = _pdf_location(evidence_ref, block, locator, artifact_ref=artifact)
    if pdf_location is not None:
        return pdf_location
    exact = evidence_ref.selector.exact_quote
    content = block.content if isinstance(block.content, str) else ""
    if "start_offset" in evidence_ref.provenance or "end_offset" in evidence_ref.provenance:
        start = evidence_ref.provenance.get("start_offset")
        end = evidence_ref.provenance.get("end_offset", start + len(exact) if type(start) is int and exact else None)
        if (type(start) is not int or type(end) is not int or not exact
                or start < 0 or end <= start or end > len(content)
                or content[start:end] != exact):
            return _structured(evidence_ref, "unresolved", artifact, "DOCX explicit text span does not match")
        return _location(evidence_ref, document, block, locator, status=status, artifact_ref=artifact)
    if exact is not None:
        occurrences = [index for index in range(len(content)) if content.startswith(exact, index)]
        requested_start = evidence_ref.provenance.get("start_offset")
        if isinstance(requested_start, int) and requested_start in occurrences:
            return _location(evidence_ref, document, block, locator, status=status, artifact_ref=artifact)
        if len(occurrences) == 1:
            return _location(evidence_ref, document, block, locator, status=status, artifact_ref=artifact)
        if len(occurrences) > 1:
            return _structured(evidence_ref, "unresolved", artifact, "quote is ambiguous within canonical block")
    if exact is not None and evidence_ref.selector.normalized_quote == " ".join(exact.split()).casefold() \
            and evidence_ref.selector.normalized_quote == " ".join(content.split()).casefold():
        return _location(evidence_ref, document, block, locator, status="normalized_match", artifact_ref=artifact)
    return _structured(evidence_ref, "unresolved", artifact, "quote does not match canonical block")


def _document_hash(document: CanonicalSourceDocument) -> str | None:
    hashes = {mapping.source_locators[0].source_hash for mapping in document.source_map.mappings if mapping.source_locators}
    return next(iter(hashes)) if len(hashes) == 1 else None


def _structured(evidence_ref: EvidenceRef, status: str, artifact_ref: str, reason: str) -> ResolvedLocation:
    return ResolvedLocation(schemaVersion="semantier.resolved_location.v1", evidenceRef=evidence_ref.evidence_ref,
        sourceIdentityRef=evidence_ref.source_identity_ref, representationType="docx_structural",
        representationRef=artifact_ref, representationHash=None, locationType="other", status=status,
        location={"kind": "unresolved", "reason": reason}, matchedText=None, resolverVersion=RESOLVER_VERSION,
        resolvedAt=datetime.now(timezone.utc))
