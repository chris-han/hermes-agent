from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from contracts.semantic_constraint_frame import (
    ComposedFramePatternV01,
    ConstraintFrameV01,
    FramePatternV01,
    FrameValueConstraintV01,
    NormalizedQuantityV01,
)
from eos.artifact_hashing import canonical_json

CONTRACT_VERSION = "tender_sensitive_term_detection_modeling.core.v1"


def sha256_canonical(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_ascii_identifier(value: str, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be ASCII-stable") from exc
    return text


@dataclass(frozen=True)
class RuntimeContextDTO:
    organization_id: str
    workspace_id: str | None
    actor_id: str
    actor_role: str | None = None
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class GovernedArtifactDTO:
    artifact_ref: str
    content_hash: str
    media_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class GovernedRuleVersionDTO:
    rule_version_id: str
    organization_id: str
    rule_family: str
    content_hash: str
    applicability_scope: dict[str, Any]
    governed_rule_text: str
    severity: str | None = None
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class DetectionRunCreateDTO:
    tender_document_id: str
    activation_set_snapshot_id: str
    resolved_activation_set_hash: str
    rule_version_set_hash: str
    target_artifact_ref: str | None = None
    target_artifact_hash: str | None = None
    idempotency_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class DetectionRunDTO:
    run_id: str
    status: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class OccurrenceFindingDTO:
    finding_id: str
    rule_version_id: str
    target_anchor_ref: str
    start_offset: int
    end_offset: int
    occurrence_index: int
    element_hash: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class CandidateFindingDraftDTO:
    finding_id: str
    tender_document_id: str
    source_anchor_refs: tuple[str, ...]
    matched_sensitive_concept_id: str
    observed_expression: str
    detection_method: Literal["exact", "regex", "semantic", "precedent"]
    semantic_relation: Literal[
        "exact",
        "synonym",
        "paraphrase",
        "narrower_than",
        "broader_than",
        "implication",
    ]
    evidence_text: str
    confidence: float
    precedent_refs: tuple[str, ...]
    decision_status: Literal["candidate", "violation", "acceptable", "uncertain", "escalate"]
    grounding_status: Literal[
        "ungrounded",
        "ai_suggested",
        "human_accepted",
        "human_edited",
        "human_rejected",
    ]
    schema_version: Literal["contextgraph_candidate_finding_draft.v2"] = (
        "contextgraph_candidate_finding_draft.v2"
    )
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class CandidateFindingV2DTO:
    finding_id: str
    run_id: str
    tender_document_id: str
    source_anchor_refs: tuple[str, ...]
    matched_sensitive_concept_id: str
    observed_expression: str
    detection_method: Literal["exact", "regex", "semantic", "precedent"]
    semantic_relation: Literal[
        "exact",
        "synonym",
        "paraphrase",
        "narrower_than",
        "broader_than",
        "implication",
    ]
    evidence_text: str
    confidence: float
    precedent_refs: tuple[str, ...]
    decision_status: Literal["candidate", "violation", "acceptable", "uncertain", "escalate"]
    grounding_status: Literal[
        "ungrounded",
        "ai_suggested",
        "human_accepted",
        "human_edited",
        "human_rejected",
    ]
    schema_version: Literal["contextgraph_candidate_finding.v2"] = (
        "contextgraph_candidate_finding.v2"
    )
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class TenderFindingDecisionContextDTO:
    finding_id: str
    concept_id: str
    meaning_match: bool
    relation_match: bool
    qualification_context: Literal[
        "requirement",
        "scoring",
        "description",
        "negation",
        "unknown",
    ]
    polarity: Literal["positive", "negative", "unclear"]
    modality: Literal["mandatory", "optional", "prohibited", "unknown"]
    rationale: str
    schema_version: Literal["contextgraph_tender_finding_decision_context.v1"] = (
        "contextgraph_tender_finding_decision_context.v1"
    )
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class RuntimeConflictDTO:
    conflict_id: str
    conflict_type: str
    target_anchor_ref: str
    rule_version_ids: tuple[str, ...]
    start_offset: int
    end_offset: int
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class FindingSetDTO:
    run_id: str
    finding_count: int
    finding_set_hash: str
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class RuntimeConflictSetDTO:
    run_id: str
    conflict_count: int
    conflict_set_hash: str
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class DetectionEvidenceDTO:
    run: DetectionRunDTO
    findings: tuple[OccurrenceFindingDTO, ...]
    runtime_conflicts: tuple[RuntimeConflictDTO, ...]
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class ProcurementCaseFactsV01:
    jurisdiction: str | None = None
    purchaser_type: str | None = None
    subject_matter: str | None = None
    procurement_method: str | None = None
    as_of_date: str | None = None
    schema_version: Literal["procurement_case_facts.v0_1"] = (
        "procurement_case_facts.v0_1"
    )


@dataclass(frozen=True)
class RegimeDeterminationV01:
    regime: Literal[
        "tender_bidding", "government_procurement", "ambiguous", "unresolved"
    ] = "unresolved"
    basis: Literal[
        "rule_matched", "llm_inferred", "human_reviewed", "not_evaluated"
    ] = "not_evaluated"
    rationale: str | None = None
    schema_version: Literal["regime_determination.v0_1"] = (
        "regime_determination.v0_1"
    )


@dataclass(frozen=True)
class FrameSlotDiagnosticV01:
    slot_name: str
    matched: bool
    observed_value: Any
    canonical_value: Any
    rule_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameMatchResultDTO:
    semantic_relation: Literal[
        "LEXICAL_MATCH", "FRAME_EQUIVALENT", "FRAME_ENTAILS", "NO_MATCH"
    ]
    derivation_shape: Literal["single_frame", "composed_frame_set"]
    matched_pattern_ref: str | None
    constituent_matches: tuple["FrameMatchResultDTO", ...]
    slot_diagnostics: tuple[FrameSlotDiagnosticV01, ...]
    semantic_profile_id: str
    schema_version: Literal["contextgraph_frame_match_result.v1"] = (
        "contextgraph_frame_match_result.v1"
    )

    def __post_init__(self) -> None:
        validate_ascii_identifier(self.semantic_profile_id, "semantic_profile_id")
        if self.semantic_relation == "NO_MATCH" and self.matched_pattern_ref is not None:
            raise ValueError("NO_MATCH cannot carry a matched_pattern_ref")
        if self.semantic_relation != "NO_MATCH" and not self.matched_pattern_ref:
            raise ValueError("a successful frame match requires matched_pattern_ref")
