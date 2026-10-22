"""Governed Phase-1 source/evidence identity contracts.

Coordinates returned by a renderer are deliberately excluded from EvidenceRef
identity.  They are resolved presentation state and may change when a source is
re-rendered or relocated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SourceIdentity(_Contract):
    schema_version: Literal["semantier.source_identity.v1"] = Field(alias="schemaVersion")
    source_identity_ref: str = Field(alias="sourceIdentityRef", min_length=1)
    organization_id: str = Field(alias="organizationId", min_length=1)
    workspace_id: str = Field(alias="workspaceId", min_length=1)
    source_hash: str = Field(alias="sourceHash", pattern=r"^sha256:[0-9a-f]{64}$")
    media_type: str = Field(alias="mediaType", min_length=1)
    original_name: str = Field(alias="originalName", min_length=1)
    artifact_ref: str = Field(alias="artifactRef", min_length=1)
    source_version: str | None = Field(alias="sourceVersion")
    supersedes_source_identity_ref: str | None = Field(alias="supersedesSourceIdentityRef")
    created_at: datetime = Field(alias="createdAt")

    @model_validator(mode="after")
    def utc_timestamp(self) -> "SourceIdentity":
        if self.created_at.tzinfo is None:
            raise ValueError("createdAt must be timezone-aware")
        return self


class TableSelector(_Contract):
    table_ordinal: int | None = Field(alias="tableOrdinal")
    heading_path: list[str] = Field(alias="headingPath")
    header_path: list[str] = Field(alias="headerPath")
    row_key: str | None = Field(alias="rowKey")
    column_key: str | None = Field(alias="columnKey")
    cell_text: str | None = Field(alias="cellText")


class EvidenceSelector(_Contract):
    schema_version: Literal["semantier.evidence_selector.v1"] = Field(alias="schemaVersion")
    selector_kind: Literal["quote", "structure", "table_cell", "composite"] = Field(alias="selectorKind")
    exact_quote: str | None = Field(alias="exactQuote")
    prefix: str | None
    suffix: str | None
    normalized_quote: str | None = Field(alias="normalizedQuote")
    structural_path: list[str] = Field(alias="structuralPath")
    table: TableSelector | None
    source_element_ref: str | None = Field(alias="sourceElementRef")

    @model_validator(mode="after")
    def has_locator(self) -> "EvidenceSelector":
        if not (self.exact_quote or self.structural_path or self.table or self.source_element_ref):
            raise ValueError("EvidenceSelector requires a locating mechanism")
        return self

    def selector_hash(self) -> str:
        payload = self.model_dump(mode="json", by_alias=True, exclude_none=False)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + sha256(encoded).hexdigest()


class EvidenceRef(_Contract):
    schema_version: Literal["semantier.evidence_ref.v1"] = Field(alias="schemaVersion")
    evidence_ref: str = Field(alias="evidenceRef", min_length=1)
    organization_id: str = Field(alias="organizationId", min_length=1)
    workspace_id: str = Field(alias="workspaceId", min_length=1)
    source_identity_ref: str = Field(alias="sourceIdentityRef", min_length=1)
    selector_schema_version: Literal["semantier.evidence_selector.v1"] = Field(alias="selectorSchemaVersion")
    selector_hash: str = Field(alias="selectorHash", pattern=r"^sha256:[0-9a-f]{64}$")
    selector: EvidenceSelector
    evidence_role: str = Field(alias="evidenceRole")
    provenance: dict[str, Any]
    created_at: datetime = Field(alias="createdAt")


class DocxStructuralRange(_Contract):
    kind: Literal["docx_structural_range"]
    structural_path: list[int | str] = Field(alias="structuralPath", min_length=1)
    start_offset: int = Field(alias="startOffset", ge=0)
    end_offset: int = Field(alias="endOffset", gt=0)
    exact_text: str = Field(alias="exactText", min_length=1)

    @model_validator(mode="after")
    def ordered_offsets(self) -> "DocxStructuralRange":
        if self.end_offset <= self.start_offset:
            raise ValueError("endOffset must be greater than startOffset")
        return self


class HtmlTextRange(DocxStructuralRange):
    kind: Literal["html_text_range"]
    structural_path: list[int | str] = Field(default_factory=list, alias="structuralPath")
    source_hash: str = Field(alias="sourceHash", pattern=r"^sha256:[0-9a-f]{64}$")
    projection_hash: str = Field(alias="projectionHash", pattern=r"^sha256:[0-9a-f]{64}$")
    quote_hash: str = Field(alias="quoteHash", pattern=r"^sha256:[0-9a-f]{64}$")
    mapping_version: Literal["original-html-text/1"] = Field(alias="mappingVersion")


class TabularCellLocation(_Contract):
    kind: Literal["tabular_cell"]
    sheet_name: str = Field(alias="sheetName", min_length=1)
    row_number: int = Field(alias="rowNumber", ge=1)
    column_number: int = Field(alias="columnNumber", ge=1)
    header_row: int = Field(alias="headerRow", ge=1)
    cell_text: str = Field(alias="cellText")


class PdfPageRect(_Contract):
    kind: Literal["pdf_page_rect"]
    page: int = Field(ge=1)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    unit: Literal["pdf-point"]
    origin: Literal["top-left"]
    precision: Literal["span", "block"]
    exact_text: str | None = Field(default=None, alias="exactText")

    @model_validator(mode="after")
    def finite_geometry(self) -> "PdfPageRect":
        if not all(isfinite(value) for value in (self.x, self.y, self.width, self.height)):
            raise ValueError("PDF geometry must be finite")
        if self.precision == "span" and not self.exact_text:
            raise ValueError("span precision requires exactText")
        return self


class TextFallbackLocation(_Contract):
    kind: Literal["text_fallback"]
    exact_text: str = Field(alias="exactText", min_length=1)
    page: int | None = Field(default=None, ge=1)
    reason: str = Field(min_length=1)


class UnresolvedLocation(_Contract):
    kind: Literal["unresolved"] = "unresolved"
    reason: str = Field(min_length=1)

class ResolvedLocation(_Contract):
    schema_version: Literal["semantier.resolved_location.v1"] = Field(alias="schemaVersion")
    evidence_ref: str = Field(alias="evidenceRef", min_length=1)
    source_identity_ref: str = Field(alias="sourceIdentityRef", min_length=1)
    representation_type: str = Field(alias="representationType")
    representation_ref: str = Field(alias="representationRef", min_length=1)
    representation_hash: str | None = Field(alias="representationHash")
    location_type: str = Field(alias="locationType")
    status: Literal["exact", "relocated_exact", "normalized_match", "block", "text_fallback", "ambiguous", "unresolved", "source_changed"]
    location: dict[str, Any]
    matched_text: str | None = Field(alias="matchedText")
    resolver_version: str = Field(alias="resolverVersion", min_length=1)
    resolved_at: datetime = Field(alias="resolvedAt")

    @model_validator(mode="after")
    def validated_location(self) -> "ResolvedLocation":
        kind = self.location.get("kind")
        if kind == "docx_structural_range":
            DocxStructuralRange.model_validate(self.location)
        elif kind == "html_text_range":
            HtmlTextRange.model_validate(self.location)
        elif kind == "tabular_cell":
            TabularCellLocation.model_validate(self.location)
        elif kind == "pdf_page_rect":
            PdfPageRect.model_validate(self.location)
        elif kind == "text_fallback":
            TextFallbackLocation.model_validate(self.location)
        elif kind == "unresolved":
            UnresolvedLocation.model_validate(self.location)
        elif self.location_type != "other":
            raise ValueError("location.kind is required for resolved locations")
        return self


__all__ = ["TabularCellLocation", "DocxStructuralRange", "EvidenceRef", "EvidenceSelector", "PdfPageRect", "ResolvedLocation", "SourceIdentity", "TableSelector", "TextFallbackLocation", "UnresolvedLocation"]
