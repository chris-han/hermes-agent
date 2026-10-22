"""Typed contracts for governed investment opportunity sourcing.

These DTOs deliberately keep observations, derived metrics, and authority
decisions separate.  Marketplace plugins may construct them, but core owns
the lifecycle and persistence of candidates and configuration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal

from contracts.knowledge_classification import RuntimeContextDTO
from eos.artifact_hashing import content_hash

CONTRACT_VERSION = "investment_opportunity.v1"

RADAR_CAPABILITIES = (
    "investment.radar.candidate.read",
    "investment.radar.candidate.submit",
    "investment.radar.opportunity.review",
    "investment.radar.action.propose",
    "investment.radar.configuration.propose",
    "investment.radar.configuration.activate",
    "investment.radar.schedule.create",
)

ActionScope = Literal[
    "OPPORTUNITY_QUALIFICATION", "OPPORTUNITY_REJECTION", "OPPORTUNITY_DEFERRAL",
    "OPPORTUNITY_OWNER_ASSIGNMENT", "OPPORTUNITY_ENRICHMENT_REQUEST",
    "OPPORTUNITY_PRELIMINARY_DILIGENCE_REQUEST", "OPPORTUNITY_MONITORING_TRIGGER_CREATE",
    "OPPORTUNITY_ENTITY_RESOLUTION_CORRECTION", "OPPORTUNITY_FALSE_POSITIVE_RECORD",
    "RADAR_CONFIGURATION_PROPOSAL", "RADAR_CONFIGURATION_ACTIVATION",
]

OperatorOutcomeType = Literal[
    "qualified", "deferred", "rejected", "false_positive", "contacted", "diligence_opened"
]


class OpportunityReasonCode(StrEnum):
    GITHUB_SOURCE_UNAVAILABLE = "GITHUB_SOURCE_UNAVAILABLE"
    GITHUB_RATE_LIMITED = "GITHUB_RATE_LIMITED"
    GITHUB_RESPONSE_INVALID = "GITHUB_RESPONSE_INVALID"
    MISSING_REQUIRED_DATA = "MISSING_REQUIRED_DATA"
    ENTITY_RESOLUTION_AMBIGUOUS = "ENTITY_RESOLUTION_AMBIGUOUS"
    SCORE_SUPPRESSED_INCOMPLETE = "SCORE_SUPPRESSED_INCOMPLETE"
    MANDATE_MISMATCH = "MANDATE_MISMATCH"
    DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
    TRUSTED_RUNTIME_CONTEXT_REQUIRED = "TRUSTED_RUNTIME_CONTEXT_REQUIRED"
    CORE_BINDING_UNAVAILABLE = "CORE_BINDING_UNAVAILABLE"


@dataclass(frozen=True)
class GitHubObservationDTO:
    observation_id: str
    source: str
    source_ref: str
    observed_at: str
    repository: dict[str, Any]
    counters: dict[str, int]
    sampled_activity: dict[str, int | None]
    provenance: dict[str, Any]
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class FeatureSnapshotDTO:
    feature_snapshot_id: str
    target_ref: str
    observation_refs: tuple[str, ...]
    feature_spec_version: str
    window: str
    features: dict[str, float | None]
    quality: dict[str, Any]
    content_hash: str
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class OpportunityScorecardDTO:
    scorecard_id: str
    target_ref: str
    scoring_profile_ref: str
    scoring_profile_version: str
    feature_snapshot_ref: str
    dimensions: dict[str, dict[str, Any]]
    composite_score: float | None
    confidence: float
    missing_evidence: tuple[str, ...]
    disqualifiers: tuple[str, ...]
    calculation_hash: str
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class OpportunityCandidateDTO:
    candidate_id: str
    organization_id: str
    target: dict[str, Any]
    candidate_type: str
    observations: tuple[str, ...]
    scorecard_ref: str
    signal_claims: tuple[dict[str, Any], ...]
    semantic_tier: str = "T6"
    governance_state: str = "proposed"
    contract_version: str = CONTRACT_VERSION


def _hash(value: Any) -> str:
    return "sha256:" + content_hash(value)


@dataclass(frozen=True)
class OpportunityObservationDTO:
    observation_id: str
    source: str
    source_ref: str
    observed_at: str
    payload: dict[str, Any]
    response_hash: str
    adapter_version: str
    missing_fields: tuple[str, ...] = ()
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class OpportunityCandidateSubmitDTO:
    organization_id: str
    candidate_type: str
    target: dict[str, Any]
    detected_at: str
    observation_refs: tuple[str, ...]
    scorecard_ref: str
    signal_claims: tuple[dict[str, Any], ...]
    missing_evidence: tuple[str, ...] = ()
    proposed_next_actions: tuple[str, ...] = ()
    justification: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    contract_version: str = CONTRACT_VERSION

    def hashed(self) -> "OpportunityCandidateSubmitDTO":
        payload = asdict(self)
        payload["content_hash"] = ""
        return OpportunityCandidateSubmitDTO(**{**asdict(self), "content_hash": _hash(payload)})


@dataclass(frozen=True)
class OpportunityCandidateStateDTO:
    candidate_id: str
    organization_id: str
    governance_state: str
    semantic_tier: str
    candidate: dict[str, Any]
    transitions: tuple[dict[str, Any], ...] = ()
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class OpportunityCandidateActionDTO:
    candidate_id: str
    action_scope: ActionScope
    reason: str
    trigger: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class RadarUniverseCandidateDTO:
    universe_candidate_id: str
    organization_id: str
    name: str
    scope: str
    owner_ref: str
    source_filters: dict[str, Any]
    exclusions: dict[str, Any] = field(default_factory=dict)
    operational_limits: dict[str, int] = field(default_factory=dict)
    semantic_tier: str = "T5"
    exploratory: bool = False
    content_hash: str = ""


@dataclass(frozen=True)
class RadarScoringProfileCandidateDTO:
    profile_candidate_id: str
    organization_id: str
    name: str
    scope: str
    owner_ref: str
    feature_spec_version: str
    required_features: tuple[str, ...]
    dimension_weights: dict[str, float]
    thresholds: dict[str, float]
    null_policy: str = "suppress_score"
    semantic_tier: str = "T5"
    exploratory: bool = False
    content_hash: str = ""


@dataclass(frozen=True)
class RadarConfigurationStateDTO:
    candidate_id: str
    name: str
    category: str
    status: str
    content_hash: str
    content: dict[str, Any]


@dataclass(frozen=True)
class OpportunityReplayEnvelopeDTO:
    candidate_id: str
    observation_refs: tuple[str, ...]
    observation_hashes: tuple[str, ...]
    feature_spec_version: str
    feature_snapshot_hash: str
    scoring_profile_version: str
    scoring_profile_hash: str
    entity_resolution_version: str
    resolver_trace_refs: tuple[str, ...] = ()
    replay_binding_ref: str | None = None
    contract_version: str = CONTRACT_VERSION
