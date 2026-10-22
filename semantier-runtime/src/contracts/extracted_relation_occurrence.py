"""Representation-neutral candidate relation occurrences in ExtractResult."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from contracts.evidence_location import EvidenceRef
from contracts.knowledge_relation_taxonomy import RepresentationKind
from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import (
    evidence_ref_identity,
)


_ASCII_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_OCCURRENCE_REF = re.compile(r"^ero_[0-9a-f]{24}$")


def _ascii_ref(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or _ASCII_REF.fullmatch(value) is None
    ):
        raise ValueError(f"{field} must be an ASCII-stable identifier/reference")
    return value


def _canonical_projection_refs(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("projection_source_refs must be an array")
    refs = tuple(_ascii_ref(item, "projection_source_refs[]") for item in value)
    if len(refs) != len(set(refs)):
        raise ValueError("projection_source_refs contains a duplicate identity")
    return tuple(sorted(refs, key=lambda item: item.encode("utf-8")))


def _evidence_identity(item: EvidenceRef) -> str:
    expected = evidence_ref_identity(
        organization_id=item.organization_id,
        workspace_id=item.workspace_id,
        source_identity_ref=item.source_identity_ref,
        selector_hash=item.selector_hash,
        evidence_role=item.evidence_role,
    )
    if item.evidence_ref != expected:
        raise ValueError("EvidenceRef identity mismatch")
    if item.selector_schema_version != item.selector.schema_version:
        raise ValueError("EvidenceRef selector schema mismatch")
    if item.selector_hash != item.selector.selector_hash():
        raise ValueError("EvidenceRef selector hash mismatch")
    return _ascii_ref(item.evidence_ref, "EvidenceRef.evidenceRef")


def _canonical_evidence(value: Any) -> tuple[EvidenceRef, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("evidence_refs must be an array")
    items = tuple(
        item if isinstance(item, EvidenceRef) else EvidenceRef.model_validate(item)
        for item in value
    )
    identities = [_evidence_identity(item) for item in items]
    if len(identities) != len(set(identities)):
        raise ValueError("evidence_refs contains a duplicate evidenceRef identity")
    return tuple(
        sorted(
            items,
            key=lambda item: (
                item.evidence_ref.encode("utf-8"),
                canonical_json(
                    item.model_dump(mode="json", by_alias=True)
                ).encode("utf-8"),
            ),
        )
    )


class ExtractedRelationOccurrenceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["extracted_relation_occurrence.v1"] = (
        "extracted_relation_occurrence.v1"
    )
    occurrence_ref: str
    occurrence_hash: str
    organization_id: str
    workspace_id: str
    relation_type_id: str
    representation_kind: RepresentationKind
    source_node_id: str | None
    target_node_id: str | None
    frame_ref: str | None
    projection_source_refs: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    authority_state: Literal["candidate"] = "candidate"

    @field_validator("organization_id", "workspace_id", "relation_type_id")
    @classmethod
    def validate_required_refs(cls, value: str, info) -> str:
        return _ascii_ref(value, info.field_name)

    @field_validator("source_node_id", "target_node_id", "frame_ref")
    @classmethod
    def validate_optional_refs(cls, value: str | None, info) -> str | None:
        if value is not None:
            return _ascii_ref(value, info.field_name)
        return value

    @field_validator("occurrence_ref")
    @classmethod
    def validate_occurrence_ref(cls, value: str) -> str:
        if _OCCURRENCE_REF.fullmatch(value) is None:
            raise ValueError("occurrence_ref must be ero_<24 lowercase hex>")
        return value

    @field_validator("occurrence_hash")
    @classmethod
    def validate_occurrence_hash(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("occurrence_hash must be an exact sha256 ref")
        return value

    @field_validator("projection_source_refs", mode="before")
    @classmethod
    def canonicalize_projection_refs(cls, value: Any) -> tuple[str, ...]:
        return _canonical_projection_refs(value)

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def canonicalize_evidence_refs(cls, value: Any) -> tuple[EvidenceRef, ...]:
        return _canonical_evidence(value)

    @model_validator(mode="after")
    def validate_integrity_and_cardinality(self) -> "ExtractedRelationOccurrenceV1":
        for evidence in self.evidence_refs:
            if evidence.organization_id != self.organization_id:
                raise ValueError("EvidenceRef organization scope mismatch")
            if evidence.workspace_id != self.workspace_id:
                raise ValueError("EvidenceRef workspace scope mismatch")
        endpoints = self.source_node_id is not None, self.target_node_id is not None
        if endpoints[0] != endpoints[1]:
            raise ValueError("source and target IDs must be both null or both non-null")
        if self.representation_kind == "binary_edge":
            if (
                not endpoints[0]
                or self.frame_ref is not None
                or self.projection_source_refs
            ):
                raise ValueError("binary_edge occurrence cardinality is invalid")
        elif self.representation_kind == "frame_role":
            if (
                endpoints[0]
                or self.frame_ref is None
                or self.projection_source_refs
            ):
                raise ValueError("frame_role occurrence cardinality is invalid")
        elif self.representation_kind == "derived_projection":
            if not self.projection_source_refs:
                raise ValueError("derived_projection requires projection_source_refs")
            if endpoints[0] == (self.frame_ref is not None):
                raise ValueError("derived_projection must bind endpoints or one frame")
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.occurrence_hash != expected_hash:
            raise ValueError("occurrence hash does not match canonical content")
        if self.occurrence_ref != "ero_" + expected_hash[7:31]:
            raise ValueError("occurrence ref does not match occurrence hash")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("occurrence_ref")
        payload.pop("occurrence_hash")
        return payload


def create_extracted_relation_occurrence(
    *,
    organization_id: str,
    workspace_id: str,
    relation_type_id: str,
    representation_kind: RepresentationKind,
    projection_shape: str,
    source_node_id: str | None,
    target_node_id: str | None,
    frame_ref: str | None,
    projection_source_refs: Sequence[str],
    evidence_refs: Sequence[EvidenceRef | dict[str, Any]],
    authority_state: str = "candidate",
) -> ExtractedRelationOccurrenceV1:
    if representation_kind == "binary_edge" and projection_shape != "binary":
        raise ValueError("binary_edge projection shape is invalid")
    if representation_kind == "frame_role" and projection_shape != "frame":
        raise ValueError("frame_role projection shape is invalid")
    payload: dict[str, Any] = {
        "schema_version": "extracted_relation_occurrence.v1",
        "organization_id": organization_id,
        "workspace_id": workspace_id,
        "relation_type_id": relation_type_id,
        "representation_kind": representation_kind,
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "frame_ref": frame_ref,
        "projection_source_refs": list(
            _canonical_projection_refs(projection_source_refs)
        ),
        "evidence_refs": [
            item.model_dump(mode="json", by_alias=True)
            for item in _canonical_evidence(evidence_refs)
        ],
        "authority_state": authority_state,
    }
    if representation_kind == "derived_projection":
        endpoints_present = source_node_id is not None and target_node_id is not None
        if projection_shape == "binary" and (
            not endpoints_present or frame_ref is not None
        ):
            raise ValueError("binary derived_projection cardinality is invalid")
        if projection_shape == "frame" and (
            endpoints_present or frame_ref is None
        ):
            raise ValueError("frame derived_projection cardinality is invalid")
    digest = "sha256:" + content_hash(payload)
    return ExtractedRelationOccurrenceV1.model_validate(
        {
            **payload,
            "occurrence_ref": "ero_" + digest[7:31],
            "occurrence_hash": digest,
        }
    )


__all__ = [
    "ExtractedRelationOccurrenceV1",
    "create_extracted_relation_occurrence",
]
