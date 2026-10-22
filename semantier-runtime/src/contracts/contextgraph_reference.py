"""ContextGraph reference concept metadata contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from contracts.knowledge_feature_pack import StrictContract


GroundingState = Literal[
    "candidate",
    "human_accepted",
    "human_edited",
    "human_rejected",
]
SemanticRelation = Literal[
    "synonym",
    "paraphrase",
    "narrower_than",
    "broader_than",
    "implication",
]


class ReferenceConcept(StrictContract):
    schema_version: Literal["contextgraph_reference_concept.v1"]
    concept_id: str = Field(min_length=1)
    candidate_graph_hash: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    canonical_label: str = Field(min_length=1)
    category: str | None = None
    semantic_role: str | None = None
    prohibition_type: str | None = None
    applies_to: tuple[str, ...] = ()
    source_anchor_refs: tuple[str, ...] = Field(min_length=1)
    grounding_state: GroundingState

    @field_validator("applies_to", "source_anchor_refs")
    @classmethod
    def require_unique_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("reference concept refs must be unique")
        return value


class CanonicalExpression(StrictContract):
    schema_version: Literal["contextgraph_canonical_expression.v1"]
    expression_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: str = Field(min_length=1)
    expression_kind: Literal["source_defined", "example", "definition"]
    source_anchor_refs: tuple[str, ...] = Field(min_length=1)

    @field_validator("source_anchor_refs")
    @classmethod
    def require_unique_source_anchors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("source anchor refs must be unique")
        return value


class ObservedExpressionObservation(StrictContract):
    schema_version: Literal["contextgraph_observed_expression_observation.v1"]
    observation_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    semantic_relation: SemanticRelation
    source_anchor_refs: tuple[str, ...] = Field(min_length=1)
    source_document_hash: str = Field(min_length=1)
    finding_run_id: str = Field(min_length=1)
    finding_id: str = Field(min_length=1)
    reference_concept_version_ref: str = Field(min_length=1)
    accepted_graph_release_id: str = Field(min_length=1)
    human_learning_event_id: str = Field(min_length=1)
    semantic_relation_decision_hash: str = Field(min_length=1)
    grounding_state: Literal["candidate", "human_accepted", "human_rejected"]

    @field_validator("source_anchor_refs")
    @classmethod
    def require_unique_source_anchors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("source anchor refs must be unique")
        return value


class ObservedExpression(StrictContract):
    schema_version: Literal["contextgraph_observed_expression.v1"]
    expression_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    semantic_relation: SemanticRelation
    reference_concept_version_ref: str = Field(min_length=1)
    accepted_graph_release_id: str = Field(min_length=1)
    latest_observation_id: str = Field(min_length=1)
    grounding_state: Literal["candidate", "human_accepted", "human_rejected"]
    observation_count: int = Field(ge=1)
