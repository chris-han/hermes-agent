"""Lossless, representation-independent source structure contracts.

Canonical block identity is ``sha256(canonical_json(source_hash, locator,
structural_role))``.  The payload intentionally excludes Markdown offsets and
parser versions; those belong to the parser trace and presentation layer.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SourceLocator(_Contract):
    source_identity_ref: str = Field(alias="sourceIdentityRef", min_length=1)
    source_hash: str = Field(alias="sourceHash", pattern=r"^sha256:[0-9a-f]{64}$")
    media_type: str = Field(alias="mediaType", min_length=1)
    native_provider: str | None = Field(default=None, alias="nativeProvider")
    format: str | None = None
    package_part: str | None = Field(default=None, alias="packagePart")
    native_id: str | None = Field(default=None, alias="nativeId")
    ordinal_path: list[int | str] = Field(default_factory=list, alias="ordinalPath")
    native_metadata: dict[str, Any] = Field(default_factory=dict, alias="nativeMetadata")
    section_id: str | None = Field(default=None, alias="sectionId")
    paragraph_id: str | None = Field(default=None, alias="paragraphId")
    table_id: str | None = Field(default=None, alias="tableId")
    row_id: str | None = Field(default=None, alias="rowId")
    column_id: str | None = Field(default=None, alias="columnId")


class SourceBlock(_Contract):
    block_id: str = Field(alias="blockId", min_length=1)
    block_type: str = Field(alias="blockType", min_length=1)
    content: Any = None
    parent_block_id: str | None = Field(default=None, alias="parentBlockId")
    ordinal: int = Field(ge=0)
    structure: dict[str, Any] = Field(default_factory=dict)
    content_hash: str = Field(alias="contentHash", pattern=r"^sha256:[0-9a-f]{64}$")


TransformationKind = Literal["one_to_one", "aggregation", "split", "structural_normalization"]


class Transformation(_Contract):
    kind: TransformationKind
    transform_ref: str = Field(alias="transformRef", min_length=1)


class SourceMapping(_Contract):
    source_locators: list[SourceLocator] = Field(alias="sourceLocators", min_length=1)
    block_ids: list[str] = Field(alias="blockIds", min_length=1)
    transformation: Transformation


class SourceMap(_Contract):
    mappings: list[SourceMapping] = Field(default_factory=list)


class ParserTrace(_Contract):
    provider: str
    upstream_revision: str | None = Field(default=None, alias="upstreamRevision")
    upstream_package_version: str | None = Field(default=None, alias="upstreamPackageVersion")
    semantier_patch_series_version: str | None = Field(default=None, alias="semantierPatchSeriesVersion")
    adapter_version: str = Field(alias="adapterVersion")
    canonical_ir_version: Literal["canonical_source_ir.v1"] = Field(alias="canonicalIrVersion")
    warnings: list[str] = Field(default_factory=list)


class CanonicalSourceDocument(_Contract):
    source_identity_ref: str = Field(alias="sourceIdentityRef", min_length=1)
    blocks: list[SourceBlock]
    source_map: SourceMap = Field(alias="sourceMap")
    assets: list[dict[str, Any]] = Field(default_factory=list)
    parser_trace: ParserTrace = Field(alias="parserTrace")
    canonical_ir_version: Literal["canonical_source_ir.v1"] = Field(
        default="canonical_source_ir.v1", alias="canonicalIrVersion"
    )


def source_block_identity(source_hash: str, locator: SourceLocator, structural_role: str) -> str:
    """Return stable identity ``H(source_hash, canonical_locator, role)``.

    ``native_*`` fields are deliberately excluded from the canonical locator
    projection, allowing parser upgrades to preserve identity when structure
    and source semantics remain unchanged.
    """
    canonical_locator = {
        "sourceIdentityRef": locator.source_identity_ref,
        "sourceHash": source_hash,
        "mediaType": locator.media_type,
        "sectionId": locator.section_id,
        "paragraphId": locator.paragraph_id,
        "tableId": locator.table_id,
        "rowId": locator.row_id,
        "columnId": locator.column_id,
    }
    payload = json.dumps(
        {"source_hash": source_hash, "canonical_source_locator": canonical_locator, "structural_role": structural_role},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + sha256(payload).hexdigest()


def content_hash(content: Any) -> str:
    payload = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + sha256(payload).hexdigest()


__all__ = [
    "CanonicalSourceDocument", "ParserTrace", "SourceBlock", "SourceLocator",
    "SourceMap", "SourceMapping", "Transformation", "content_hash", "source_block_identity",
]
