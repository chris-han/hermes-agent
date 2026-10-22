"""Build and validate the v7.3 Explainability Surface from recorded traces only."""

# Patent evidence C007: explainability surface constructed from the replay
# binding and recorded traces, exposing winning/suppressed claims, semantic tier
# trace, authority sources and source guarantees, referenced in the
# admissibility-governance method claims.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from eos.db import get_justification, get_latest_validation, list_ledger_projections_for_rea
from eos.projection_context_store import load_snapshot, verify_snapshot_hash
from eos.replay_binding_store import get_replay_binding, verify_replay_binding

_HASH_PREFIX = "sha256:"
_ALLOWED_TIERS = {"T0", "T1", "T2", "T3", "T4", "T5", "T6"}
_ALLOWED_AUTHORITY_ORIGINS = {
    "official_source",
    "interpretive_source",
    "internal_governance",
    "runtime_observation",
    "ontology_governance",
}
_ALLOWED_REASON_CODES = {
    "not_in_active_K_v",
    "violates_O_v",
    "violates_O_tag_v",
    "wrong_authority_domain",
    "deprecated_source",
    "unapproved_source",
    "contains_forbidden_financial_fact",
    "attempts_to_bypass_R_v",
    "hash_mismatch",
    "other",
}


def _normalize_hash(value: str) -> str:
    if value.startswith(_HASH_PREFIX):
        return value
    return f"{_HASH_PREFIX}{value}"


def _reason_code(raw_reason: Optional[str]) -> str:
    if not raw_reason:
        return "other"
    mapping = {
        "not_in_active_k_v": "not_in_active_K_v",
        "reject_not_in_active_k_v": "not_in_active_K_v",
        "context_violates_o_v": "violates_O_v",
        "context_violates_o_tag_v": "violates_O_tag_v",
        "outside_authority_domain": "wrong_authority_domain",
        "context_outside_authority_domain": "wrong_authority_domain",
        "reject_domain_mismatch": "wrong_authority_domain",
        "not_approved": "unapproved_source",
        "reject_not_approved": "unapproved_source",
        "source_status_deprecated": "deprecated_source",
        "source_status_rejected": "deprecated_source",
        "reject_governance_state": "deprecated_source",
        "forbidden_projection_context_fields": "contains_forbidden_financial_fact",
        "projection_context_hash_mismatch": "hash_mismatch",
        "projection_context_hash_mismatch_with_binding": "hash_mismatch",
    }
    normalized = raw_reason.lower()
    return mapping.get(normalized, raw_reason if raw_reason in _ALLOWED_REASON_CODES else "other")


def _claim_source_ref(claim: Dict[str, Any]) -> str:
    return (
        claim.get("source_ref")
        or claim.get("source_id")
        or claim.get("source_type")
        or "trace_unavailable"
    )


def _claim_ref(claim: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "claim_id": claim.get("id") or claim.get("claim_id") or "trace_unavailable",
        "authority_domain": claim.get("authority_domain") or "trace_unavailable",
        "semantic_tier": claim.get("semantic_tier") or "T6",
        "source_ref": _claim_source_ref(claim),
    }


def _suppressed_claim_ref(claim: Dict[str, Any], winning_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    matching_winner = next(
        (winner for winner in winning_claims if winner.get("key") == claim.get("key")),
        None,
    )
    suppression_reason = claim.get("suppression_reason") or "trace_unavailable"
    return {
        **_claim_ref(claim),
        "suppression_reason": suppression_reason,
        "suppressed_by_claim_id": matching_winner.get("id") if matching_winner else None,
        "suppressed_by_rule_ref": None,
    }


def _excluded_context_ref(item: Dict[str, Any]) -> Dict[str, Any]:
    raw_reason = item.get("reason") or item.get("reason_code")
    return {
        "source_ref": item.get("source_ref") or item.get("source_id") or "trace_unavailable",
        "reason_code": _reason_code(raw_reason),
        "reason": raw_reason or "trace_unavailable",
    }


class ExcludedContextRefModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(min_length=1)
    reason_code: str
    reason: str = Field(min_length=1)

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str) -> str:
        if value not in _ALLOWED_REASON_CODES:
            raise ValueError("invalid_reason_code")
        return value


class ClaimRefModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    authority_domain: str = Field(min_length=1)
    semantic_tier: str
    source_ref: str = Field(min_length=1)

    @field_validator("semantic_tier")
    @classmethod
    def validate_tier(cls, value: str) -> str:
        if value not in _ALLOWED_TIERS:
            raise ValueError("invalid_semantic_tier")
        return value


class SuppressedClaimRefModel(ClaimRefModel):
    suppression_reason: str = Field(min_length=1)
    suppressed_by_claim_id: Optional[str] = None
    suppressed_by_rule_ref: Optional[str] = None


class VersionPinsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    O_v: str = Field(min_length=1)
    C_v: str = Field(min_length=1)
    O_tag_v: str = Field(min_length=1)
    K_v: str = Field(min_length=1)
    Pi_v: str = Field(min_length=1)
    ConstraintBundle_v: str = Field(min_length=1)
    PrecedenceGraph_v: str = Field(min_length=1)
    ResolutionRules_v: str = Field(min_length=1)


class SemanticTierTraceModel(BaseModel):
    evaluated_tiers: List[str]
    winning_tier: Optional[str] = None
    priority_rule_refs: List[str]
    model_config = ConfigDict(extra="allow")

    @field_validator("evaluated_tiers")
    @classmethod
    def validate_evaluated_tiers(cls, value: List[str]) -> List[str]:
        invalid = [tier for tier in value if tier not in _ALLOWED_TIERS]
        if invalid:
            raise ValueError(f"invalid_evaluated_tiers:{','.join(invalid)}")
        return value

    @field_validator("winning_tier")
    @classmethod
    def validate_winning_tier(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in _ALLOWED_TIERS:
            raise ValueError("invalid_winning_tier")
        return value


class GeneratedFromModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    projection_context_snapshot_ref: str = Field(min_length=1)
    resolution_result_ref: str = Field(min_length=1)
    replay_binding_ref: str = Field(min_length=1)
    projection_result_ref: str = Field(min_length=1)


class AuthoritySourceModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    artifact_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    authority_domain: str = Field(min_length=1)
    tag_authority_level: str = Field(min_length=1)
    semantic_tier: str
    authority_origin: str
    source_ref: str = Field(min_length=1)
    source_hash: str = Field(min_length=1)
    ingestion_status: str = Field(min_length=1)
    jurisdiction: Optional[str] = None
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    source_version: Optional[str] = None

    @field_validator("semantic_tier")
    @classmethod
    def validate_semantic_tier(cls, value: str) -> str:
        if value not in _ALLOWED_TIERS:
            raise ValueError("invalid_semantic_tier")
        return value

    @field_validator("authority_origin")
    @classmethod
    def validate_authority_origin(cls, value: str) -> str:
        if value not in _ALLOWED_AUTHORITY_ORIGINS:
            raise ValueError("invalid_authority_origin")
        return value


class SourceGuaranteesModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uses_recorded_trace_only: bool
    no_live_llm: bool
    no_live_ocr: bool
    no_live_document_parser: bool
    no_live_retrieval: bool
    retrieval_provider_ref: str

    @field_validator("uses_recorded_trace_only", "no_live_llm", "no_live_ocr", "no_live_document_parser", "no_live_retrieval")
    @classmethod
    def validate_true_flags(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("source_guarantee_must_be_true")
        return value

    @field_validator("retrieval_provider_ref")
    @classmethod
    def validate_provider_ref(cls, value: str) -> str:
        if value != "hermes-agent/plugins/memory/holographic":
            raise ValueError("invalid_retrieval_provider_ref")
        return value


class ExplainabilitySurfaceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    event_id: str = Field(min_length=1)
    projection_id: str = Field(min_length=1)
    selected_context: Dict[str, Any]
    excluded_context_refs: List[ExcludedContextRefModel]
    selection_reason: str = Field(min_length=1)
    winning_claims: List[ClaimRefModel]
    suppressed_claims: List[SuppressedClaimRefModel]
    escalation_reasons: List[str] = []
    semantic_tier_trace: SemanticTierTraceModel
    authority_sources: List[AuthoritySourceModel]
    authority_source_manifest_hash: str
    pins: VersionPinsModel
    projection_context_ref: str = Field(min_length=1)
    projection_context_hash: str
    projection_context_hash_verified: bool
    resolution_trace_hash: str
    effect_hash: str
    event_hash: str
    replay_verified: bool
    generated_from: GeneratedFromModel
    source_guarantees: SourceGuaranteesModel

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != "explainability_surface.v1":
            raise ValueError("invalid_schema_version")
        return value

    @field_validator(
        "projection_context_hash",
        "resolution_trace_hash",
        "effect_hash",
        "event_hash",
        "authority_source_manifest_hash",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not value.startswith(_HASH_PREFIX) or len(value) != len(_HASH_PREFIX) + 64:
            raise ValueError("invalid_hash_format")
        return value


def validate_explainability_surface(surface: Dict[str, Any]) -> Dict[str, Any]:
    try:
        model = ExplainabilitySurfaceModel.model_validate(surface)
    except ValidationError as exc:
        return {
            "status": "INVALID_EXPLANATION_SURFACE",
            "errors": exc.errors(),
        }
    return {
        "status": "OK",
        "surface": model.model_dump(mode="json"),
    }


def build_explainability_surface(event_id: str) -> Dict[str, Any]:
    binding = get_replay_binding(event_id)
    if not binding:
        return {"status": "NOT_FOUND", "reason": "replay_binding_not_found"}

    justification_row = get_justification(event_id)
    if not justification_row:
        return {"status": "NOT_FOUND", "reason": "justification_not_found"}

    validation_row = get_latest_validation(event_id)
    if not validation_row:
        return {"status": "NOT_FOUND", "reason": "validation_not_found"}

    replay_verification = verify_replay_binding(event_id)
    if replay_verification.get("status") != "OK":
        return {
            "status": "INVALID_EXPLANATION_SURFACE",
            "reason": replay_verification.get("reason", "replay_binding_verification_failed"),
            "binding_verification": replay_verification,
        }

    projection_context_ref = binding["projection_context_ref"]
    snapshot = load_snapshot(projection_context_ref)
    if not snapshot:
        return {"status": "NOT_FOUND", "reason": "projection_context_snapshot_missing"}

    context_verification = verify_snapshot_hash(projection_context_ref)
    if context_verification.get("status") != "OK":
        return {
            "status": "INVALID_EXPLANATION_SURFACE",
            "reason": "projection_context_hash_mismatch",
            "context_verification": context_verification,
        }

    justification = justification_row["justification"]
    validation_result = validation_row.get("result") or {}
    resolution_result = (
        validation_result.get("resolution_result")
        or justification.get("resolution_result")
        or {}
    )

    winning_claims = resolution_result.get("winning_claims") or []
    suppressed_claims = resolution_result.get("suppressed_claims") or []
    resolution_trace = resolution_result.get("resolution_trace") or {}
    projections = list_ledger_projections_for_rea(justification_row["rea_id"])
    if not projections:
        return {"status": "NOT_FOUND", "reason": "projection_result_not_found"}

    projection_id = projections[0]["ledger_id"]
    governance_artifacts = justification.get("governance_artifacts") or {}
    resolution_rules = governance_artifacts.get("ResolutionRules_v") or {}
    rule_refs = [
        rule.get("id")
        for rule in (resolution_rules.get("rules") or [])
        if isinstance(rule, dict) and rule.get("id")
    ]
    surface = {
        "schema_version": "explainability_surface.v1",
        "event_id": event_id,
        "projection_id": projection_id,
        "selected_context": snapshot.get("selected_context") or {},
        "excluded_context_refs": [
            _excluded_context_ref(item)
            for item in (snapshot.get("excluded_context_refs") or [])
        ],
        "selection_reason": snapshot.get("selection_reason") or "trace_unavailable",
        "winning_claims": [_claim_ref(claim) for claim in winning_claims],
        "suppressed_claims": [
            _suppressed_claim_ref(claim, winning_claims)
            for claim in suppressed_claims
        ],
        "escalation_reasons": resolution_result.get("escalation_reasons") or [],
        "semantic_tier_trace": {
            "evaluated_tiers": [
                claim.get("semantic_tier")
                for claim in winning_claims + suppressed_claims
                if claim.get("semantic_tier") in _ALLOWED_TIERS
            ],
            "winning_tier": winning_claims[0].get("semantic_tier") if winning_claims else None,
            "priority_rule_refs": rule_refs,
            "ordered_claim_ids": resolution_trace.get("ordered_claim_ids") or [],
            "winning_claim_ids": resolution_trace.get("winning_claim_ids") or [],
            "suppressed_claim_ids": resolution_trace.get("suppressed_claim_ids") or [],
        },
        "authority_sources": binding.get("authority_source_manifest_json") or [],
        "authority_source_manifest_hash": _normalize_hash(
            binding.get("authority_source_manifest_hash") or ""
        ),
        "pins": {
            "O_v": binding.get("ontology_version") or "trace_unavailable",
            "C_v": binding.get("contract_version") or "trace_unavailable",
            "O_tag_v": binding.get("tag_ontology_version") or "trace_unavailable",
            "K_v": binding.get("knowledge_version") or "trace_unavailable",
            "Pi_v": binding.get("projection_bundle_version") or "trace_unavailable",
            "ConstraintBundle_v": binding.get("constraint_bundle_version") or "trace_unavailable",
            "PrecedenceGraph_v": binding.get("precedence_graph_version") or "trace_unavailable",
            "ResolutionRules_v": binding.get("resolution_rules_version") or "trace_unavailable",
        },
        "projection_context_ref": projection_context_ref,
        "projection_context_hash": _normalize_hash(binding["projection_context_hash"]),
        "projection_context_hash_verified": True,
        "resolution_trace_hash": _normalize_hash(binding["resolution_trace_hash"]),
        "effect_hash": _normalize_hash(binding["effect_hash"]),
        "event_hash": _normalize_hash(binding["event_hash"]),
        "replay_verified": True,
        "generated_from": {
            "projection_context_snapshot_ref": projection_context_ref,
            "resolution_result_ref": validation_row["validation_id"],
            "replay_binding_ref": binding["event_id"],
            "projection_result_ref": projection_id,
        },
        "source_guarantees": {
            "uses_recorded_trace_only": True,
            "no_live_llm": True,
            "no_live_ocr": True,
            "no_live_document_parser": True,
            "no_live_retrieval": True,
            "retrieval_provider_ref": snapshot.get("provider_ref") or "hermes-agent/plugins/memory/holographic",
        },
    }
    validation = validate_explainability_surface(surface)
    if validation.get("status") != "OK":
        return {
            "status": "INVALID_EXPLANATION_SURFACE",
            "surface": surface,
            "errors": validation.get("errors", []),
        }
    return {
        "status": "OK",
        "surface": validation["surface"],
    }
