"""Human-grounded tender graph contracts for Delivery Phase 1."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from contracts.knowledge_feature_pack import MachineProposal, StrictContract, content_hash
from contracts.evidence_location import EvidenceRef


class SourceAnchor(StrictContract):
    anchor_id: str
    document_hash: str
    document_part: Literal["paragraph", "table_cell"]
    paragraph_index: int | None = Field(default=None, ge=0)
    table_index: int | None = Field(default=None, ge=0)
    row_index: int | None = Field(default=None, ge=0)
    cell_index: int | None = Field(default=None, ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    exact_text: str
    local_text_hash: str

    @model_validator(mode="after")
    def require_shape_and_exact_span(self) -> "SourceAnchor":
        if self.end_offset <= self.start_offset:
            raise ValueError("source anchor end_offset must be greater than start_offset")
        if self.document_part == "paragraph" and self.paragraph_index is None:
            raise ValueError("paragraph anchors require paragraph_index")
        if self.document_part == "table_cell" and None in {
            self.table_index,
            self.row_index,
            self.cell_index,
        }:
            raise ValueError("table-cell anchors require table, row, and cell indexes")
        return self


class CandidateNodeType(str, Enum):
    TERM = "term"
    CONCEPT = "concept"
    CATEGORY = "category"
    CONDITION = "condition"
    EXCEPTION = "exception"
    RULE = "rule"
    ALIAS = "alias"
    INSTANCE = "instance"
    CLAUSE = "clause"
    ATTRIBUTE = "attribute"
    CONSTRAINT = "constraint"
    RISK = "risk"


class CandidateRelationType(str, Enum):
    ALIAS_OF = "alias_of"
    RELATED_TO = "related_to"
    MODIFIER_OF = "modifier_of"
    BROADER_THAN = "broader_than"
    NARROWER_THAN = "narrower_than"
    IN_CATEGORY = "in_category"
    CONDITIONED_BY = "conditioned_by"
    EXCEPTION_TO = "exception_to"
    SUPPORTS_RULE = "supports_rule"
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


class CandidateNode(StrictContract):
    node_id: str
    node_type: CandidateNodeType
    canonical_label: str
    display_name_zh: str | None = None
    normalized_value: str
    rationale: str
    source_anchors: tuple[SourceAnchor, ...]
    extraction_run_id: str
    machine_proposal_id: str
    machine_confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: tuple[EvidenceRef, ...] = ()

    @field_validator("source_anchors")
    @classmethod
    def require_source_anchor(cls, value: tuple[SourceAnchor, ...]) -> tuple[SourceAnchor, ...]:
        if not value:
            raise ValueError("candidate nodes require source anchors")
        return value


class CandidateEdge(StrictContract):
    edge_id: str
    source_node_id: str
    target_node_id: str
    relation_type: CandidateRelationType
    rationale: str
    source_anchors: tuple[SourceAnchor, ...]
    extraction_run_id: str
    machine_proposal_id: str
    machine_confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: tuple[EvidenceRef, ...] = ()

    @model_validator(mode="after")
    def reject_self_identity(self) -> "CandidateEdge":
        if self.source_node_id == self.target_node_id:
            raise ValueError("candidate edges cannot be self edges")
        return self


class CandidateRule(StrictContract):
    rule_id: str
    family_id: int = Field(ge=1, le=15)
    condition_node_ids: tuple[str, ...] = ()
    exception_node_ids: tuple[str, ...] = ()
    reason_code: str
    source_anchors: tuple[SourceAnchor, ...]
    extraction_run_id: str
    machine_proposal_id: str
    evidence_refs: tuple[EvidenceRef, ...] = ()
    frame_pattern: dict[str, Any] | None = None


class SensitiveTermCandidateGraph(StrictContract):
    candidate_graph_id: str
    source_document_hash: str
    extraction_run_id: str
    schema_version: str
    ontology_version: str
    machine_proposal: MachineProposal
    nodes: tuple[CandidateNode, ...]
    edges: tuple[CandidateEdge, ...]
    rules: tuple[CandidateRule, ...]

    @model_validator(mode="after")
    def validate_graph_references(self) -> "SensitiveTermCandidateGraph":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("candidate node IDs must be unique")
        if len([edge.edge_id for edge in self.edges]) != len({edge.edge_id for edge in self.edges}):
            raise ValueError("candidate edge IDs must be unique")
        known = set(node_ids)
        dangling = [
            edge.edge_id
            for edge in self.edges
            if edge.source_node_id not in known or edge.target_node_id not in known
        ]
        if dangling:
            raise ValueError(f"candidate edges have dangling references: {dangling}")
        if bool(self.nodes) != bool(self.rules):
            raise ValueError("candidate graph nodes and rules must both be populated or both be empty")
        return self

    def graph_hash(self) -> str:
        payload = self.model_dump(mode="json")
        for rule in payload["rules"]:
            if rule.get("frame_pattern") is None:
                rule.pop("frame_pattern", None)
        return content_hash(payload)


class GraphDeltaAction(str, Enum):
    NODE_ADD = "node_add"
    NODE_EDIT = "node_edit"
    NODE_MERGE = "node_merge"
    NODE_SPLIT = "node_split"
    EDGE_ADD = "edge_add"
    EDGE_EDIT = "edge_edit"
    EDGE_REVERSE = "edge_reverse"
    EDGE_DELETE = "edge_delete"
    ALIAS_EDIT = "alias_edit"
    EVIDENCE_EDIT = "evidence_edit"
    RULE_EDIT = "rule_edit"
    CONDITION_EDIT = "condition_edit"
    EXCEPTION_EDIT = "exception_edit"


class GraphDeltaOperation(StrictContract):
    operation_id: str
    action: GraphDeltaAction
    target_refs: tuple[str, ...]
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    source_anchor_refs: tuple[str, ...]

    @model_validator(mode="after")
    def require_change(self) -> "GraphDeltaOperation":
        if not self.target_refs:
            raise ValueError("graph delta operations require targets")
        if self.before is None and self.after is None:
            raise ValueError("graph delta operations require before or after state")
        return self


class GraphDelta(StrictContract):
    graph_delta_id: str
    candidate_graph_id: str
    from_graph_version: str | None
    to_graph_version: str
    human_label_event_id: str
    justification: str = Field(min_length=1)
    operations: tuple[GraphDeltaOperation, ...]

    @field_validator("operations")
    @classmethod
    def require_operations(
        cls, value: tuple[GraphDeltaOperation, ...]
    ) -> tuple[GraphDeltaOperation, ...]:
        if not value:
            raise ValueError("GraphDelta requires at least one operation")
        return value

    def delta_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class AcceptedGraphRelease(StrictContract):
    graph_version: str = Field(pattern=r"^KG_v[1-9]\d*$")
    parent_graph_version: str | None
    candidate_graph_id: str
    graph_delta_id: str | None
    human_label_event_ids: tuple[str, ...]
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    rules: tuple[dict[str, Any], ...]
    backend: Literal["falkordb"] = "falkordb"

    def release_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))
