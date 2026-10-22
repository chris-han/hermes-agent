"""Strict content-addressed knowledge-relation taxonomy contracts."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eos.artifact_hashing import canonical_json, content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_TAXONOMY_REF = re.compile(
    r"^knowledge-relation-taxonomy:(?P<taxonomy_id>[a-z][a-z0-9_.-]*)@v(?P<version>[1-9][0-9]*)$"
)

RelationFamilyId = Literal[
    "structural",
    "descriptive",
    "semantic",
    "dependency",
    "causal_behavioral",
    "temporal",
    "spatial_contextual",
]
DefinitionScope = Literal["canonical", "profile_defined"]
RepresentationKind = Literal["binary_edge", "frame_role", "derived_projection"]
ProjectionShape = Literal["binary", "frame"]
LifecycleState = Literal["proposed", "admitted", "retired"]
Transitivity = bool | Literal["unknown"]

STABLE_FAMILY_IDS = frozenset(
    {
        "structural",
        "descriptive",
        "semantic",
        "dependency",
        "causal_behavioral",
        "temporal",
        "spatial_contextual",
    }
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)


def _require_identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.isascii() or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{field} must be an ASCII-stable lowercase identifier matching "
            "[a-z][a-z0-9_.-]*"
        )
    return value


def _sorted_unique_identifiers(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    items = tuple(_require_identifier(item, f"{field}[]") for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{field} contains a duplicate identity")
    return tuple(sorted(items, key=lambda item: item.encode("utf-8")))


class KnowledgeRelationFamily(_Contract):
    family_id: RelationFamilyId = Field(alias="familyId")
    display_name: str = Field(alias="displayName", min_length=1)
    description: str = Field(min_length=1)


class KnowledgeRelationType(_Contract):
    relation_type_id: str = Field(alias="relationTypeId")
    family_id: RelationFamilyId = Field(alias="familyId")
    namespace: str
    definition_scope: DefinitionScope = Field(alias="definitionScope")
    display_name: str = Field(alias="displayName", min_length=1)
    description: str = Field(min_length=1)
    representation_kind: RepresentationKind = Field(alias="representationKind")
    projection_shape: ProjectionShape = Field(alias="projectionShape")
    source_type_refs: tuple[str, ...] = Field(alias="sourceTypeRefs")
    target_type_refs: tuple[str, ...] = Field(alias="targetTypeRefs")
    source_role_refs: tuple[str, ...] = Field(alias="sourceRoleRefs")
    target_role_refs: tuple[str, ...] = Field(alias="targetRoleRefs")
    directed: bool
    symmetric: bool
    transitive: Transitivity
    inverse_relation_type_id: str | None = Field(alias="inverseRelationTypeId")
    aliases: tuple[str, ...]
    lifecycle_state: LifecycleState = Field(alias="lifecycleState")

    @field_validator("directed", "symmetric", mode="before")
    @classmethod
    def validate_strict_boolean(cls, value: Any, info) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be a boolean")
        return value

    @field_validator("transitive", mode="before")
    @classmethod
    def validate_transitivity(cls, value: Any) -> bool | Literal["unknown"]:
        if not isinstance(value, bool) and value != "unknown":
            raise ValueError("transitive must be true, false, or unknown")
        return value

    @field_validator("relation_type_id", "namespace")
    @classmethod
    def validate_identifier(cls, value: str, info) -> str:
        return _require_identifier(value, info.field_name)

    @field_validator("inverse_relation_type_id")
    @classmethod
    def validate_optional_identifier(cls, value: str | None) -> str | None:
        if value is not None:
            _require_identifier(value, "inverse_relation_type_id")
        return value

    @field_validator(
        "source_type_refs",
        "target_type_refs",
        "source_role_refs",
        "target_role_refs",
        "aliases",
        mode="before",
    )
    @classmethod
    def canonicalize_set_values(cls, value: Any, info) -> tuple[str, ...]:
        return _sorted_unique_identifiers(value, info.field_name)

    @model_validator(mode="after")
    def validate_semantics(self) -> "KnowledgeRelationType":
        if self.definition_scope == "canonical" and self.namespace != "core":
            raise ValueError("canonical relation types require namespace=core")
        if self.definition_scope == "profile_defined" and self.namespace == "core":
            raise ValueError("profile-defined relation types require a non-core namespace")
        if not self.relation_type_id.startswith(f"{self.namespace}."):
            raise ValueError("relation_type_id must be qualified by its namespace")
        if self.representation_kind == "binary_edge" and self.projection_shape != "binary":
            raise ValueError("binary_edge requires projection_shape=binary")
        if self.representation_kind == "frame_role" and self.projection_shape != "frame":
            raise ValueError("frame_role requires projection_shape=frame")
        if self.symmetric:
            if self.directed:
                raise ValueError("symmetric relation types require directed=false")
            if self.inverse_relation_type_id is not None:
                raise ValueError("symmetric relation types require inverse_relation_type_id=null")
        return self


class KnowledgeRelationTaxonomyV1(_Contract):
    schema_version: Literal["knowledge_relation_taxonomy.v1"] = Field(
        default="knowledge_relation_taxonomy.v1", alias="schemaVersion"
    )
    taxonomy_id: str = Field(alias="taxonomyId")
    taxonomy_ref: str = Field(alias="taxonomyRef")
    version: int = Field(gt=0, strict=True)
    taxonomy_hash: str = Field(alias="taxonomyHash", pattern=SHA256_PATTERN)
    families: tuple[KnowledgeRelationFamily, ...]
    relation_types: tuple[KnowledgeRelationType, ...] = Field(alias="relationTypes")

    @field_validator("taxonomy_id")
    @classmethod
    def validate_taxonomy_id(cls, value: str) -> str:
        return _require_identifier(value, "taxonomy_id")

    @field_validator("families", mode="before")
    @classmethod
    def canonicalize_families(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            raise ValueError("families must be an array")
        identities = [
            item.get("familyId", item.get("family_id"))
            if isinstance(item, dict)
            else item.family_id
            for item in value
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("families contains a duplicate family_id")
        return sorted(value, key=lambda item: (
            item.get("familyId", item.get("family_id"))
            if isinstance(item, dict)
            else item.family_id
        ).encode("utf-8"))

    @field_validator("relation_types", mode="before")
    @classmethod
    def canonicalize_relation_types(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            raise ValueError("relationTypes must be an array")
        identities = [
            item.get("relationTypeId", item.get("relation_type_id"))
            if isinstance(item, dict)
            else item.relation_type_id
            for item in value
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("relationTypes contains a duplicate relation_type_id")
        return sorted(value, key=lambda item: (
            item.get("relationTypeId", item.get("relation_type_id"))
            if isinstance(item, dict)
            else item.relation_type_id
        ).encode("utf-8"))

    @model_validator(mode="after")
    def validate_identity_and_hash(self) -> "KnowledgeRelationTaxonomyV1":
        match = _TAXONOMY_REF.fullmatch(self.taxonomy_ref)
        if (
            match is None
            or match.group("taxonomy_id") != self.taxonomy_id
            or int(match.group("version")) != self.version
        ):
            raise ValueError("taxonomy_ref must exactly bind taxonomy_id and version")
        if {family.family_id for family in self.families} != STABLE_FAMILY_IDS:
            raise ValueError("families must contain the exact stable seven-family set")
        if any(
            relation_type.definition_scope != "canonical"
            or relation_type.namespace != "core"
            for relation_type in self.relation_types
        ):
            raise ValueError(
                "canonical taxonomy relation types require definition_scope=canonical "
                "and namespace=core"
            )
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.taxonomy_hash != expected:
            raise ValueError("taxonomy hash does not match canonical taxonomy content")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("taxonomyHash")
        return payload

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json", by_alias=True))


class KnowledgeRelationProfileCatalogV1(_Contract):
    schema_version: Literal["knowledge_relation_profile_catalog.v1"] = Field(
        default="knowledge_relation_profile_catalog.v1", alias="schemaVersion"
    )
    namespace: str
    taxonomy_ref: str = Field(alias="taxonomyRef")
    taxonomy_hash: str = Field(alias="taxonomyHash", pattern=SHA256_PATTERN)
    relation_types: tuple[KnowledgeRelationType, ...] = Field(alias="relationTypes")

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, value: str) -> str:
        value = _require_identifier(value, "namespace")
        if value == "core":
            raise ValueError("profile catalog namespace cannot be core")
        return value

    @field_validator("relation_types", mode="before")
    @classmethod
    def canonicalize_relation_types(cls, value: Any) -> Any:
        return KnowledgeRelationTaxonomyV1.canonicalize_relation_types(value)

    @model_validator(mode="after")
    def validate_profile_definitions(self) -> "KnowledgeRelationProfileCatalogV1":
        for relation_type in self.relation_types:
            if relation_type.definition_scope != "profile_defined":
                raise ValueError("profile catalogs require profile-defined relation types")
            if relation_type.namespace != self.namespace:
                raise ValueError("profile relation namespace must match catalog namespace")
        return self

    def catalog_hash(self) -> str:
        return "sha256:" + content_hash(self.model_dump(mode="json", by_alias=True))


class KnowledgeRelationAdmission(_Contract):
    relation_type: KnowledgeRelationType = Field(alias="relationType")
    taxonomy_ref: str = Field(alias="taxonomyRef")
    taxonomy_hash: str = Field(alias="taxonomyHash", pattern=SHA256_PATTERN)
    profile_ref: str | None = Field(alias="profileRef")
    profile_hash: str | None = Field(alias="profileHash", pattern=SHA256_PATTERN)
    profile_relation_catalog_hash: str | None = Field(
        alias="profileRelationCatalogHash", pattern=SHA256_PATTERN
    )

    @model_validator(mode="after")
    def validate_scope_pins(self) -> "KnowledgeRelationAdmission":
        profile_pins = (
            self.profile_ref,
            self.profile_hash,
            self.profile_relation_catalog_hash,
        )
        if self.relation_type.definition_scope == "canonical":
            if any(item is not None for item in profile_pins):
                raise ValueError("canonical definitions require all profile fields null")
        elif any(item is None for item in profile_pins):
            raise ValueError("profile-defined definitions require all three profile fields")
        return self


__all__ = [
    "DefinitionScope",
    "KnowledgeRelationAdmission",
    "KnowledgeRelationFamily",
    "KnowledgeRelationProfileCatalogV1",
    "KnowledgeRelationTaxonomyV1",
    "KnowledgeRelationType",
    "LifecycleState",
    "ProjectionShape",
    "RelationFamilyId",
    "RepresentationKind",
    "STABLE_FAMILY_IDS",
]
