"""Document grounding and deterministic graph-reasoning contracts for FP2."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from contracts.knowledge_feature_pack import MachineProposal, StrictContract, content_hash
from contracts.evidence_location import EvidenceRef
from contracts.knowledge_graph_phase1 import SourceAnchor


class GroundingState(str, Enum):
    GROUNDED = "grounded"
    UNRESOLVED = "unresolved"


class SourceAlignmentStatus(str, Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    GREATER = "greater"
    LESSER = "lesser"
    UNGROUNDED = "ungrounded"


class SourceAlignment(StrictContract):
    provider: str
    status: SourceAlignmentStatus
    provider_start: int | None = Field(default=None, ge=0)
    provider_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_provider_interval(self) -> "SourceAlignment":
        if (self.provider_start is None) != (self.provider_end is None):
            raise ValueError("provider alignment offsets must both be present or absent")
        if self.provider_start is not None and self.provider_end <= self.provider_start:
            raise ValueError("provider alignment end must be greater than start")
        if self.status == SourceAlignmentStatus.UNGROUNDED and self.provider_start is not None:
            raise ValueError("ungrounded alignment cannot carry provider offsets")
        return self


class SemanticOccurrence(StrictContract):
    occurrence_id: str
    extraction_run_id: str
    document_id: str
    source_anchor: SourceAnchor
    extraction_class: str
    extraction_attributes: dict[str, Any]
    alignment: SourceAlignment
    machine_confidence: float | None = Field(default=None, ge=0, le=1)
    model_ref: str
    prompt_ref: str
    raw_output_ref: str
    group_ref: str | None = None
    extraction_index: int | None = Field(default=None, ge=0)

    @property
    def accepted_evidence_ready(self) -> bool:
        return self.alignment.status != SourceAlignmentStatus.UNGROUNDED


class NormalizedAssertion(StrictContract):
    assertion_type: str | None = None
    subject: dict[str, Any] | None = None
    predicate: str | None = None
    object: dict[str, Any] | None = None
    modality: str | None = None
    quantifier: dict[str, Any] | None = None
    condition: dict[str, Any] | None = None
    exception: dict[str, Any] | None = None
    consequence: dict[str, Any] | None = None
    document_role: str | None = None
    candidate_family_id: int | None = Field(default=None, ge=1, le=15)
    candidate_concept_ref: str | None = None
    candidate_relation_id: str | None = None


class DocumentCandidateAssertion(StrictContract):
    assertion_id: str
    candidate_graph_id: str
    candidate_graph_hash: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    source_document_hash: str
    source_anchors: tuple[SourceAnchor, ...]
    occurrence_count: int = Field(ge=1)
    proposal_source: Literal["deterministic_exact", "deterministic_alias", "llm_proposal"]
    machine_proposal: MachineProposal | None = None
    confidence: float = Field(ge=0, le=1)
    grounding_state: GroundingState
    accepted_graph_node_id: str | None = None
    accepted_graph_edge_id: str | None = None
    occurrence_refs: tuple[str, ...] = ()
    normalized_assertion: NormalizedAssertion = Field(default_factory=NormalizedAssertion)
    evidence_refs: tuple[EvidenceRef, ...] = ()

    @model_validator(mode="after")
    def enforce_explicit_grounding(self) -> "DocumentCandidateAssertion":
        if self.grounding_state == GroundingState.GROUNDED and not self.accepted_graph_node_id:
            raise ValueError("grounded assertion requires explicit accepted graph node ID")
        if self.grounding_state == GroundingState.UNRESOLVED and self.accepted_graph_node_id:
            raise ValueError("unresolved assertion cannot bind an accepted graph node")
        if self.proposal_source == "llm_proposal" and self.machine_proposal is None:
            raise ValueError("LLM assertion requires machine proposal provenance")
        if self.occurrence_refs and len(self.occurrence_refs) != self.occurrence_count:
            raise ValueError("occurrence_count must match unique occurrence_refs")
        if len(set(self.occurrence_refs)) != len(self.occurrence_refs):
            raise ValueError("occurrence_refs must be unique")
        return self


class TenderPackageDocument(StrictContract):
    document_id: str
    document_hash: str
    role: Literal[
        "main_tender", "technical_specification", "contract_attachment",
        "scoring_table", "engineering_specification", "other_companion",
    ]
    elements: tuple[dict[str, Any], ...]


class KnowledgeScope(StrictContract):
    package_id: str
    documents: tuple[TenderPackageDocument, ...]

    @field_validator("documents")
    @classmethod
    def require_documents(cls, value):
        if not value:
            raise ValueError("TenderPackage requires documents")
        if len({item.document_id for item in value}) != len(value):
            raise ValueError("TenderPackage document IDs must be unique")
        if len({item.document_hash for item in value}) != len(value):
            raise ValueError("TenderPackage document hashes must be unique")
        if len({item.role for item in value}) != len(value):
            raise ValueError("TenderPackage document roles must be unique")
        if sum(item.role == "main_tender" for item in value) != 1:
            raise ValueError("TenderPackage requires exactly one main_tender document")
        return value


class EvidenceObligation(StrictContract):
    obligation_id: str
    obligation_type: Literal[
        "domain_necessity", "legal_basis", "technical_basis", "exception",
        "qualification_scope", "document_role", "scoring_context",
    ]
    primary_anchor_ref: str
    allowed_companion_roles: tuple[str, ...]
    query_terms: tuple[str, ...]
    assertion_id: str | None = None
    status: Literal["open", "satisfied", "unresolved"] = "open"
    supporting_anchor_refs: tuple[str, ...] = ()


class ViolationEvidenceAssembly(StrictContract):
    assembly_id: str
    assertion_id: str
    primary_source_anchors: tuple[SourceAnchor, ...]
    supporting_context_anchors: tuple[SourceAnchor, ...]
    accepted_graph_node_ids: tuple[str, ...]
    accepted_graph_edge_ids: tuple[str, ...]
    condition_node_ids: tuple[str, ...]
    exception_node_ids: tuple[str, ...]
    reasoning_path: tuple[str, ...]
    triggered_rule_id: str | None
    rule_inputs: dict[str, Any]
    graph_version: str
    graph_release_hash: str

    def assembly_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class VerdictState(str, Enum):
    CONFIRMED_VIOLATION = "confirmed_violation"
    CONFIRMED_NON_VIOLATION = "confirmed_non_violation"
    UNRESOLVED = "unresolved"


class DeterministicGraphVerdict(StrictContract):
    verdict_id: str
    state: VerdictState
    assertion_id: str
    assembly_hash: str
    source_anchor_refs: tuple[str, ...]
    graph_path: tuple[str, ...]
    triggered_rule_id: str | None
    reason_code: str
    graph_version: str
    coverage_warnings: tuple[str, ...] = ()
