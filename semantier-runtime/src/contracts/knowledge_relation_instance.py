"""Strict release-independent knowledge-relation assertion contracts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.evidence_location import EvidenceRef
from contracts.knowledge_relation_taxonomy import (
    KnowledgeRelationAdmission,
    RepresentationKind,
)
from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_identity


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ASCII_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
_RELATION_INSTANCE_ID = re.compile(r"^kri_[0-9a-f]{24}$")
_BINDING_ID = re.compile(r"^arb_[0-9a-f]{24}$")

class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)


def _require_ascii_ref(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.isascii() or _ASCII_REF.fullmatch(value) is None:
        raise ValueError(f"{field} must be an ASCII-stable identifier/reference")
    return value


def _canonical_refs(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    items = tuple(_require_ascii_ref(item, f"{field}[]") for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{field} contains a duplicate identity")
    return tuple(sorted(items, key=lambda item: item.encode("utf-8")))


def _evidence_identity(item: EvidenceRef) -> str:
    _require_ascii_ref(item.evidence_ref, "EvidenceRef.evidenceRef")
    _require_ascii_ref(item.organization_id, "EvidenceRef.organizationId")
    _require_ascii_ref(item.workspace_id, "EvidenceRef.workspaceId")
    _require_ascii_ref(item.source_identity_ref, "EvidenceRef.sourceIdentityRef")
    _require_ascii_ref(item.evidence_role, "EvidenceRef.evidenceRole")
    if item.created_at.tzinfo is None:
        raise ValueError("EvidenceRef.createdAt must be timezone-aware")
    if item.selector_schema_version != item.selector.schema_version:
        raise ValueError("EvidenceRef selector schema mismatch")
    if item.selector_hash != item.selector.selector_hash():
        raise ValueError("EvidenceRef selector hash mismatch")
    expected = evidence_ref_identity(
        organization_id=item.organization_id,
        workspace_id=item.workspace_id,
        source_identity_ref=item.source_identity_ref,
        selector_hash=item.selector_hash,
        evidence_role=item.evidence_role,
    )
    if item.evidence_ref != expected:
        raise ValueError("EvidenceRef identity mismatch")
    return item.evidence_ref


def _canonical_evidence(value: Any) -> tuple[EvidenceRef, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("evidenceRefs must be an array")
    items = tuple(
        item if isinstance(item, EvidenceRef) else EvidenceRef.model_validate(item)
        for item in value
    )
    identities = [_evidence_identity(item) for item in items]
    if len(identities) != len(set(identities)):
        raise ValueError("evidenceRefs contains a duplicate evidenceRef identity")
    return tuple(
        sorted(
            items,
            key=lambda item: (
                item.evidence_ref.encode("utf-8"),
                canonical_json(item.model_dump(mode="json", by_alias=True)).encode("utf-8"),
            ),
        )
    )


class KnowledgeRelationInstanceV1(_Contract):
    schema_version: Literal["knowledge_relation_instance.v1"] = Field(
        default="knowledge_relation_instance.v1", alias="schemaVersion"
    )
    relation_instance_id: str = Field(alias="relationInstanceId")
    relation_instance_hash: str = Field(alias="relationInstanceHash", pattern=SHA256_PATTERN)
    organization_id: str = Field(alias="organizationId")
    workspace_id: str = Field(alias="workspaceId")
    relation_type_id: str = Field(alias="relationTypeId")
    representation_kind: RepresentationKind = Field(alias="representationKind")
    taxonomy_ref: str = Field(alias="taxonomyRef")
    taxonomy_hash: str = Field(alias="taxonomyHash", pattern=SHA256_PATTERN)
    profile_ref: str | None = Field(alias="profileRef")
    profile_hash: str | None = Field(alias="profileHash", pattern=SHA256_PATTERN)
    profile_relation_catalog_hash: str | None = Field(
        alias="profileRelationCatalogHash", pattern=SHA256_PATTERN
    )
    source_node_id: str | None = Field(alias="sourceNodeId")
    target_node_id: str | None = Field(alias="targetNodeId")
    frame_ref: str | None = Field(alias="frameRef")
    projection_source_refs: tuple[str, ...] = Field(alias="projectionSourceRefs")
    evidence_refs: tuple[EvidenceRef, ...] = Field(alias="evidenceRefs")

    @field_validator(
        "organization_id", "workspace_id", "relation_type_id", "taxonomy_ref"
    )
    @classmethod
    def validate_required_ref(cls, value: str, info) -> str:
        return _require_ascii_ref(value, info.field_name)

    @field_validator("profile_ref", "source_node_id", "target_node_id", "frame_ref")
    @classmethod
    def validate_optional_ref(cls, value: str | None, info) -> str | None:
        if value is not None:
            return _require_ascii_ref(value, info.field_name)
        return None

    @field_validator("relation_instance_id")
    @classmethod
    def validate_instance_id(cls, value: str) -> str:
        if _RELATION_INSTANCE_ID.fullmatch(value) is None:
            raise ValueError("relationInstanceId must be kri_<24 lowercase hex>")
        return value

    @field_validator("projection_source_refs", mode="before")
    @classmethod
    def canonicalize_projection_sources(cls, value: Any) -> tuple[str, ...]:
        return _canonical_refs(value, "projectionSourceRefs")

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def canonicalize_evidence(cls, value: Any) -> tuple[EvidenceRef, ...]:
        return _canonical_evidence(value)

    @model_validator(mode="after")
    def validate_local_integrity(self) -> "KnowledgeRelationInstanceV1":
        profile_pins = (self.profile_ref, self.profile_hash, self.profile_relation_catalog_hash)
        if not (all(pin is None for pin in profile_pins) or all(pin is not None for pin in profile_pins)):
            raise ValueError("profile fields must be either all null or all non-null")
        for evidence in self.evidence_refs:
            if evidence.organization_id != self.organization_id:
                raise ValueError("EvidenceRef organization scope mismatch")
            if evidence.workspace_id != self.workspace_id:
                raise ValueError("EvidenceRef workspace scope mismatch")
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.relation_instance_hash != expected_hash:
            raise ValueError("relation instance hash does not match canonical content")
        if self.relation_instance_id != "kri_" + expected_hash[7:31]:
            raise ValueError("relation instance id does not match relation instance hash")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("relationInstanceId")
        payload.pop("relationInstanceHash")
        return payload

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json", by_alias=True))

    @property
    def relation_instance_ref(self) -> str:
        return self.relation_instance_id


def _resolve_exact_admission(
    instance: KnowledgeRelationInstanceV1,
    taxonomy_service: Any,
) -> KnowledgeRelationAdmission:
    _require_exact_taxonomy_authority(taxonomy_service)
    admission = taxonomy_service.resolve_relation_type(
        relation_type_id=instance.relation_type_id,
        taxonomy_ref=instance.taxonomy_ref,
        taxonomy_hash=instance.taxonomy_hash,
        profile_ref=instance.profile_ref,
        profile_hash=instance.profile_hash,
        profile_relation_catalog_hash=instance.profile_relation_catalog_hash,
        for_new_write=True,
    )
    if not isinstance(admission, KnowledgeRelationAdmission):
        raise ValueError("relation type resolver returned invalid authority")
    expected = (
        admission.relation_type.relation_type_id,
        admission.relation_type.representation_kind,
        admission.taxonomy_ref,
        admission.taxonomy_hash,
        admission.profile_ref,
        admission.profile_hash,
        admission.profile_relation_catalog_hash,
    )
    actual = (
        instance.relation_type_id,
        instance.representation_kind,
        instance.taxonomy_ref,
        instance.taxonomy_hash,
        instance.profile_ref,
        instance.profile_hash,
        instance.profile_relation_catalog_hash,
    )
    if actual != expected:
        raise ValueError("relation instance does not match resolved admission authority")
    if admission.relation_type.lifecycle_state != "admitted":
        raise ValueError("relation type must be admitted for a new relation instance")
    _validate_cardinality(instance, admission)
    return admission


def _require_exact_taxonomy_authority(taxonomy_service: Any) -> None:
    from services.corpus_graph_schema_profile_registry import (
        CorpusGraphSchemaProfileRegistry,
    )
    from services.knowledge_relation_taxonomy_service import (
        KnowledgeRelationTaxonomyService,
    )

    if type(taxonomy_service) is not KnowledgeRelationTaxonomyService:
        raise ValueError(
            "relation instance requires the exact governed taxonomy service"
        )
    if type(taxonomy_service.profile_registry) is not CorpusGraphSchemaProfileRegistry:
        raise ValueError(
            "relation instance requires the exact governed profile registry"
        )


def _validate_cardinality(
    instance: KnowledgeRelationInstanceV1,
    admission: KnowledgeRelationAdmission,
) -> None:
    kind = instance.representation_kind
    shape = admission.relation_type.projection_shape
    source_present = instance.source_node_id is not None
    target_present = instance.target_node_id is not None
    if source_present != target_present:
        raise ValueError("source and target node IDs must be both null or both non-null")
    if kind == "binary_edge":
        if shape != "binary" or not source_present or instance.frame_ref is not None or instance.projection_source_refs:
            raise ValueError("binary_edge cardinality does not match admitted relation type")
    elif kind == "frame_role":
        if shape != "frame" or source_present or instance.frame_ref is None or instance.projection_source_refs:
            raise ValueError("frame_role cardinality does not match admitted relation type")
    elif kind == "derived_projection":
        if not instance.projection_source_refs:
            raise ValueError("derived_projection requires projectionSourceRefs")
        if shape == "binary" and not source_present:
            raise ValueError("binary derived_projection requires both endpoints")
        if shape == "frame" and (source_present or instance.frame_ref is None):
            raise ValueError("frame derived_projection requires only frameRef")


def create_knowledge_relation_instance(
    *,
    taxonomy_service: Any,
    organization_id: str,
    workspace_id: str,
    relation_type_id: str,
    representation_kind: RepresentationKind,
    taxonomy_ref: str,
    taxonomy_hash: str,
    profile_ref: str | None,
    profile_hash: str | None,
    profile_relation_catalog_hash: str | None,
    source_node_id: str | None,
    target_node_id: str | None,
    frame_ref: str | None,
    projection_source_refs: Sequence[str],
    evidence_refs: Sequence[EvidenceRef | dict[str, Any]],
) -> KnowledgeRelationInstanceV1:
    _require_exact_taxonomy_authority(taxonomy_service)
    admission = taxonomy_service.resolve_relation_type(
        relation_type_id=relation_type_id,
        taxonomy_ref=taxonomy_ref,
        taxonomy_hash=taxonomy_hash,
        profile_ref=profile_ref,
        profile_hash=profile_hash,
        profile_relation_catalog_hash=profile_relation_catalog_hash,
        for_new_write=True,
    )
    payload: dict[str, Any] = {
        "schemaVersion": "knowledge_relation_instance.v1",
        "organizationId": organization_id,
        "workspaceId": workspace_id,
        "relationTypeId": relation_type_id,
        "representationKind": representation_kind,
        "taxonomyRef": taxonomy_ref,
        "taxonomyHash": taxonomy_hash,
        "profileRef": profile_ref,
        "profileHash": profile_hash,
        "profileRelationCatalogHash": profile_relation_catalog_hash,
        "sourceNodeId": source_node_id,
        "targetNodeId": target_node_id,
        "frameRef": frame_ref,
        "projectionSourceRefs": list(projection_source_refs),
        "evidenceRefs": list(evidence_refs),
    }
    canonical_projection_refs = _canonical_refs(payload["projectionSourceRefs"], "projectionSourceRefs")
    canonical_evidence_refs = _canonical_evidence(payload["evidenceRefs"])
    payload["projectionSourceRefs"] = list(canonical_projection_refs)
    payload["evidenceRefs"] = [item.model_dump(mode="json", by_alias=True) for item in canonical_evidence_refs]
    instance = _build_instance_after_verified_authority(payload)
    _compare_admission_and_cardinality(instance, admission)
    return instance


def _build_instance_after_verified_authority(
    payload: dict[str, Any],
) -> KnowledgeRelationInstanceV1:
    """Hash and construct only after the caller has resolved governed authority."""
    digest = "sha256:" + content_hash(payload)
    return KnowledgeRelationInstanceV1.model_validate(
        {
            **payload,
            "relationInstanceId": "kri_" + digest[7:31],
            "relationInstanceHash": digest,
        }
    )


def _compare_admission_and_cardinality(
    instance: KnowledgeRelationInstanceV1,
    admission: KnowledgeRelationAdmission,
) -> None:
    if not isinstance(admission, KnowledgeRelationAdmission):
        raise ValueError("governed taxonomy service returned invalid authority")
    expected = (
        admission.relation_type.relation_type_id,
        admission.relation_type.representation_kind,
        admission.taxonomy_ref,
        admission.taxonomy_hash,
        admission.profile_ref,
        admission.profile_hash,
        admission.profile_relation_catalog_hash,
    )
    actual = (
        instance.relation_type_id,
        instance.representation_kind,
        instance.taxonomy_ref,
        instance.taxonomy_hash,
        instance.profile_ref,
        instance.profile_hash,
        instance.profile_relation_catalog_hash,
    )
    if actual != expected:
        raise ValueError("relation instance does not match resolved admission authority")
    if admission.relation_type.lifecycle_state != "admitted":
        raise ValueError("relation type must be admitted for a new relation instance")
    _validate_cardinality(instance, admission)


def parse_knowledge_relation_instance(
    payload: dict[str, Any], *, taxonomy_service: Any
) -> KnowledgeRelationInstanceV1:
    instance = KnowledgeRelationInstanceV1.model_validate(payload)
    _resolve_exact_admission(instance, taxonomy_service)
    return instance


class AcceptedRelationBindingV1(_Contract):
    schema_version: Literal["accepted_relation_binding.v1"] = Field(
        default="accepted_relation_binding.v1", alias="schemaVersion"
    )
    binding_id: str = Field(alias="bindingId")
    binding_hash: str = Field(alias="bindingHash", pattern=SHA256_PATTERN)
    organization_id: str = Field(alias="organizationId")
    workspace_id: str = Field(alias="workspaceId")
    relation_instance_ref: str = Field(alias="relationInstanceRef")
    relation_instance_hash: str = Field(alias="relationInstanceHash", pattern=SHA256_PATTERN)
    accepted_graph_release_ref: str = Field(alias="acceptedGraphReleaseRef")
    accepted_graph_release_hash: str = Field(alias="acceptedGraphReleaseHash", pattern=SHA256_PATTERN)

    @field_validator(
        "organization_id", "workspace_id", "relation_instance_ref"
    )
    @classmethod
    def validate_ref(cls, value: str, info) -> str:
        return _require_ascii_ref(value, info.field_name)

    @field_validator("accepted_graph_release_ref")
    @classmethod
    def validate_release_ref(cls, value: str) -> str:
        if re.fullmatch(r"KG_v[1-9][0-9]*", value) is None:
            raise ValueError("acceptedGraphReleaseRef must be exact KG_v<positive integer>")
        return value

    @field_validator("binding_id")
    @classmethod
    def validate_binding_id(cls, value: str) -> str:
        if _BINDING_ID.fullmatch(value) is None:
            raise ValueError("bindingId must be arb_<24 lowercase hex>")
        return value

    @model_validator(mode="after")
    def validate_hash_and_id(self) -> "AcceptedRelationBindingV1":
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.binding_hash != expected_hash:
            raise ValueError("binding hash does not match canonical content")
        if self.binding_id != "arb_" + expected_hash[7:31]:
            raise ValueError("binding id does not match binding hash")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("bindingId")
        payload.pop("bindingHash")
        return payload


def parse_unverified_accepted_relation_binding(
    payload: dict[str, Any],
) -> AcceptedRelationBindingV1:
    """Parse hash integrity only; release/instance authority is deliberately unverified.

    Public authority-valid construction remains unavailable until the strict
    accepted_graph_release.v2 contract and resolver land.
    """
    return AcceptedRelationBindingV1.model_validate(payload)


__all__ = [
    "AcceptedRelationBindingV1",
    "KnowledgeRelationInstanceV1",
    "create_knowledge_relation_instance",
    "parse_knowledge_relation_instance",
    "parse_unverified_accepted_relation_binding",
]
