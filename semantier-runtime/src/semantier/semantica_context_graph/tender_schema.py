"""Governed, candidate-only tender ContextGraph schema.

The schema deliberately separates stable machine identifiers from localized
labels and keeps every extraction item tied to an exact source anchor.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from contracts.knowledge_feature_pack import StrictContract


ASCII_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")


class TenderLayer(str, Enum):
    CATEGORY = "category"
    CONCEPT = "concept"
    ALIAS = "alias"
    INSTANCE = "instance"
    CLAUSE = "clause"
    ATTRIBUTE = "attribute"
    CONSTRAINT = "constraint"
    RISK = "risk"


class TenderRelationType(str, Enum):
    ALIAS_OF = "alias_of"
    MODIFIER_OF = "modifier_of"
    BROADER_THAN = "broader_than"
    IN_CATEGORY = "in_category"
    IMPOSES_REQUIREMENT = "imposes_requirement"
    APPLIES_TO = "applies_to"
    HAS_ATTRIBUTE = "has_attribute"
    HAS_THRESHOLD = "has_threshold"
    REQUIRES = "requires"
    PROHIBITS = "prohibits"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    CREATES_RISK = "creates_risk"
    SUPPORTED_BY = "supported_by"


class TenderEvidence(StrictContract):
    anchor_id: str
    source_hash: str
    locator: str
    quote: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_span(self) -> "TenderEvidence":
        if self.end_offset <= self.start_offset:
            raise ValueError("evidence span must be non-empty")
        return self


class _TenderItem(StrictContract):
    item_id: str
    machine_identifier: str
    display_label: str = Field(min_length=1)
    layer: TenderLayer
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_run_id: str = Field(min_length=1)
    mapper_version: str = Field(min_length=1)
    evidence: tuple[TenderEvidence, ...]

    @model_validator(mode="after")
    def stable_identifier_and_evidence(self) -> "_TenderItem":
        if not ASCII_ID.fullmatch(self.machine_identifier):
            raise ValueError("machine_identifier must be ASCII-stable")
        if not self.evidence:
            raise ValueError("every tender item requires source evidence")
        return self


class TenderEntity(_TenderItem):
    normalized_value: str = Field(min_length=1)


class TenderRelation(StrictContract):
    relation_id: str
    source_item_id: str
    target_item_id: str
    relation_type: TenderRelationType
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_run_id: str = Field(min_length=1)
    mapper_version: str = Field(min_length=1)
    evidence: tuple[TenderEvidence, ...]

    @model_validator(mode="after")
    def valid_relation(self) -> "TenderRelation":
        if not ASCII_ID.fullmatch(self.relation_id):
            raise ValueError("relation_id must be ASCII-stable")
        if self.source_item_id == self.target_item_id:
            raise ValueError("tender relations cannot be self edges")
        if not self.evidence:
            raise ValueError("every tender relation requires source evidence")
        return self


class TenderConstraint(StrictContract):
    constraint_id: str
    target_item_id: str
    operator: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_run_id: str = Field(min_length=1)
    mapper_version: str = Field(min_length=1)
    evidence: tuple[TenderEvidence, ...]

    @model_validator(mode="after")
    def valid_constraint(self) -> "TenderConstraint":
        if not ASCII_ID.fullmatch(self.constraint_id):
            raise ValueError("constraint_id must be ASCII-stable")
        if not self.evidence:
            raise ValueError("every constraint requires source evidence")
        return self


class TenderExtractionResult(StrictContract):
    schema_version: str = "semantier.tender_extraction.v1"
    extraction_run_id: str
    mapper_version: str
    provider: str
    provider_commit: str
    source_hash: str
    entities: tuple[TenderEntity, ...]
    relations: tuple[TenderRelation, ...]
    constraints: tuple[TenderConstraint, ...]
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def references_are_valid(self) -> "TenderExtractionResult":
        known = {item.item_id for item in self.entities}
        for relation in self.relations:
            if relation.source_item_id not in known or relation.target_item_id not in known:
                raise ValueError("tender relation references an unknown item")
        for constraint in self.constraints:
            if constraint.target_item_id not in known:
                raise ValueError("constraint target is not an extracted entity")
        return self


__all__ = [
    "TenderConstraint", "TenderEntity", "TenderEvidence", "TenderExtractionResult",
    "TenderLayer", "TenderRelation", "TenderRelationType",
]
