"""CQ v2 engine — deterministic, stateless computation of CQ feature vectors,
CQ scores, trust-state transitions, insurance eligibility, and risk quotes.

Architecture ref: docs/architecture_v8_cq_v2_appendix.md §18.4–18.12, §18.15, §18.19

IMPORTANT: All functions in this module are pure (no DB calls, no live
retrieval, no LLM calls).  They only consume explicitly passed pinned
artifacts and return computed results that callers may then persist.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.trial_balance_view import ProjectionTrustState


# ---------------------------------------------------------------------------
# Default CQ contract constants (§18.4)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "validity": 0.20,
    "direction_quality": 0.12,
    "variance_quality": 0.12,
    "reconciliation_quality": 0.15,
    "evidence_completeness": 0.15,
    "replayability": 0.13,
    "governance_status": 0.08,
    "override_penalty": 0.05,
}

_DEFAULT_GATES: dict[str, float] = {
    "projection_trusted_min": 0.95,
    "period_close_min": 0.96,
    "external_export_min": 0.98,
    "insurance_eligible_min": 0.97,
}


def _extract_weights(cq_contract: dict) -> dict[str, float]:
    features_cfg = cq_contract.get("features", {})
    weights: dict[str, float] = {}
    for name, defaults in _DEFAULT_WEIGHTS.items():
        weights[name] = features_cfg.get(name, {}).get("weight", defaults)
    return weights


def _extract_gates(cq_contract: dict) -> dict[str, float]:
    gates_cfg = cq_contract.get("gates", {})
    result: dict[str, float] = {}
    for k, v in _DEFAULT_GATES.items():
        result[k] = gates_cfg.get(k, v)
    return result


# ---------------------------------------------------------------------------
# Helpers: derive per-feature scores from raw validation artifacts
# ---------------------------------------------------------------------------

def _score_validity(validation_result: dict) -> float:
    """1.0 if overall validation status is 'passed', else 0.0."""
    return 1.0 if validation_result.get("status") == "passed" else 0.0


def _score_checks(checks: list[dict]) -> float:
    """Fraction of checks that passed."""
    if not checks:
        return 1.0
    passed = sum(1 for c in checks if c.get("status") == "passed")
    return passed / len(checks)


def _score_evidence_completeness(evidence_refs: list[str]) -> float:
    """Clamp presence score: ≥3 refs → 1.0, 0 refs → 0.0."""
    n = len(evidence_refs)
    if n == 0:
        return 0.0
    return min(1.0, n / 3.0)


def _score_replayability(replay_binding: dict | None) -> float:
    """1.0 if replay_binding is present and binding is valid, else 0.0."""
    if replay_binding is None:
        return 0.0
    # Binding is considered valid when all four hash fields are non-empty.
    required = [
        "projection_context_hash",
        "resolution_trace_hash",
        "effect_hash",
        "event_hash",
    ]
    return 1.0 if all(replay_binding.get(f) for f in required) else 0.0


def _score_governance_status(governance_decisions: list[dict]) -> float:
    """1.0 if no open or rejected decisions; fraction penalised otherwise."""
    if not governance_decisions:
        return 1.0
    accepted = sum(
        1 for d in governance_decisions
        if d.get("decision_status") in {"accept", "waive"}
    )
    return accepted / len(governance_decisions)


def _score_override_penalty(override_signals: list[dict]) -> float:
    """Penalty: 1.0 means no overrides (no penalty), 0.0 means all overrides."""
    if not override_signals:
        return 0.0  # no override → no penalty contribution
    return float(len(override_signals))  # raw count; weight handles scaling


# ---------------------------------------------------------------------------
# Public: compute_feature_vector
# ---------------------------------------------------------------------------


def compute_feature_vector(
    projection_id: str,
    tenant_id: str,
    cq_contract: dict,
    validation_result: dict,
    direction_checks: list[dict],
    variance_checks: list[dict],
    reconciliation_checks: list[dict],
    justification_evidence_refs: list[str],
    replay_binding: dict | None,
    governance_decisions: list[dict],
    override_signals: list[dict],
    trial_balance_view_id: str | None = None,
) -> dict:
    """Produce a ComplianceQualityFeatureVector_t from pinned validation artifacts.

    All inputs are recorded artifacts.  This function does NOT make DB calls,
    live retrieval calls, or LLM calls.
    """
    features: dict[str, float] = {
        "validity": _score_validity(validation_result),
        "direction_quality": _score_checks(direction_checks),
        "variance_quality": _score_checks(variance_checks),
        "reconciliation_quality": _score_checks(reconciliation_checks),
        "evidence_completeness": _score_evidence_completeness(justification_evidence_refs),
        "replayability": _score_replayability(replay_binding),
        "governance_status": _score_governance_status(governance_decisions),
        "override_penalty": min(1.0, _score_override_penalty(override_signals)),
    }
    # Clamp all feature values to [0.0, 1.0]
    features = {k: max(0.0, min(1.0, v)) for k, v in features.items()}

    source_artifacts: dict[str, str] = {
        "validation_result_hash": validation_result.get("content_hash", ""),
        "replay_binding_hash": (replay_binding or {}).get("content_hash", ""),
    }

    payload = {
        "feature_vector_id": f"cqfv_{uuid4().hex}",
        "projection_id": projection_id,
        "trial_balance_view_id": trial_balance_view_id,
        "tenant_id": tenant_id,
        "cq_contract_version": cq_contract.get("version", "v2.0.0"),
        "features": features,
        "source_artifacts": source_artifacts,
    }
    payload["content_hash"] = content_hash(payload)
    return payload


# ---------------------------------------------------------------------------
# Public: compute_cq_score
# ---------------------------------------------------------------------------


def compute_cq_score(
    feature_vector: dict,
    cq_contract: dict,
    calibration_model: dict,
) -> dict:
    """Compute a ComplianceQualityScore_t from a feature vector.

    The score is a deterministic weighted sum; the override_penalty feature
    is a penalty (subtracted after weighting).  Calibration_status is
    preserved verbatim from the calibration model — it is NEVER silently
    upgraded (§18.17).
    """
    weights = _extract_weights(cq_contract)
    gates = _extract_gates(cq_contract)
    features = feature_vector["features"]

    # Weighted sum, excluding the penalty from positive accumulation
    score = 0.0
    for name, weight in weights.items():
        value = features.get(name, 0.0)
        if name == "override_penalty":
            score -= weight * value  # penalty reduces score
        else:
            score += weight * value

    score = max(0.0, min(1.0, score))

    # Apply calibration model multiplier if tiered_multiplier type
    if calibration_model.get("model_type") == "tiered_multiplier":
        multipliers = calibration_model.get("parameters", {}).get("cq_multipliers", [])
        for tier in multipliers:
            if tier["min"] <= score <= tier["max"]:
                # Multiplier is applied to expected loss in quote, not the score itself
                break

    gate_results = {
        "projection_trusted": score >= gates["projection_trusted_min"],
        "period_close_ready": score >= gates["period_close_min"],
        "external_export_ready": score >= gates["external_export_min"],
        "insurance_eligible_candidate": score >= gates["insurance_eligible_min"],
    }

    primary_strengths = [
        k for k, v in sorted(features.items(), key=lambda x: -x[1])
        if k != "override_penalty" and v >= 0.95
    ][:3]
    primary_weaknesses = [
        k for k, v in sorted(features.items(), key=lambda x: x[1])
        if k != "override_penalty" and v < 0.95
    ][:3]

    cq_score_id = f"cqs_{uuid4().hex}"
    payload = {
        "cq_score_id": cq_score_id,
        "feature_vector_id": feature_vector["feature_vector_id"],
        "projection_id": feature_vector["projection_id"],
        "tenant_id": feature_vector["tenant_id"],
        "cq_contract_version": cq_contract.get("version", "v2.0.0"),
        "calibration_model_version": calibration_model.get("model_version", "v1"),
        # Preserved verbatim — never silently upgraded
        "calibration_status": calibration_model["calibration_status"],
        "score": round(score, 6),
        "gates": gate_results,
        "explanation": {
            "primary_strengths": primary_strengths,
            "primary_weaknesses": primary_weaknesses,
        },
    }
    payload["content_hash"] = content_hash(payload)
    return payload


# ---------------------------------------------------------------------------
# Public: evaluate_trust_transition
# ---------------------------------------------------------------------------


def evaluate_trust_transition(
    validation_result_status: str,
    replay_binding_valid: bool,
    governance_satisfied: bool,
    cq_score: float,
    cq_contract: dict,
) -> str:
    """Map validation + CQ signals to a ProjectionTrustState.

    CQ may explain a trust state but must not bypass validation, replay, or
    governance (§18.7).
    """
    gates = _extract_gates(cq_contract)

    if validation_result_status in {"failed"} or not replay_binding_valid:
        return ProjectionTrustState.PROJECTION_BLOCKED

    if validation_result_status == "requires_governance" or not governance_satisfied:
        return ProjectionTrustState.PROJECTION_REQUIRES_GOVERNANCE

    if cq_score >= gates["projection_trusted_min"] and validation_result_status in {"passed", "warning"}:
        return ProjectionTrustState.PROJECTION_TRUSTED

    if validation_result_status == "warning":
        return ProjectionTrustState.PROJECTION_WARNING

    return ProjectionTrustState.PROJECTION_VALIDATED


# ---------------------------------------------------------------------------
# Public: evaluate_insurance_eligibility
# ---------------------------------------------------------------------------

_DEFAULT_LIABILITY_BASIS = {
    "input_error": "partner",
    "projection_error_under_active_bundle": "platform",
    "manual_override": "actor",
}

_MANDATORY_EXCLUSIONS_MAP = {
    "input_error": lambda signals: any(s.get("type") == "input_error" for s in signals),
    "manual_override": lambda signals: any(s.get("override", False) for s in signals),
}


def evaluate_insurance_eligibility(
    projection_id: str,
    tenant_id: str,
    cq_score_record: dict,
    trust_state: str,
    insurance_contract: dict,
    override_signals: list[dict],
) -> dict:
    """Produce an InsuranceEligibilityResult_t.

    Insurance eligibility MUST NOT imply that the REA fact is correct.
    It only covers the defined projected representation under pinned artifacts
    and contract exclusions (§18.10).
    """
    gates = _extract_gates(insurance_contract.get("cq_requirements", {})) or _extract_gates({})
    eligible_min = insurance_contract.get("cq_requirements", {}).get(
        "insurance_eligible_min",
        _DEFAULT_GATES["insurance_eligible_min"],
    )
    score = cq_score_record.get("score", 0.0)
    contract_exclusions: list[str] = insurance_contract.get("exclusions", [])

    active_exclusions: list[str] = []

    # Hard requirement: must be PROJECTION_TRUSTED
    if trust_state != ProjectionTrustState.PROJECTION_TRUSTED:
        active_exclusions.append("projection_not_trusted")

    # Hard requirement: CQ score threshold
    if score < eligible_min:
        active_exclusions.append("cq_score_below_threshold")

    # Evaluate dynamic exclusions from signals
    if "manual_override" in contract_exclusions:
        if any(s.get("override", False) for s in override_signals):
            active_exclusions.append("manual_override")

    if "missing_evidence" in contract_exclusions:
        if cq_score_record.get("features", {}).get("evidence_completeness", 1.0) < 0.5:
            active_exclusions.append("missing_evidence")

    if "invalid_replay_binding" in contract_exclusions:
        if cq_score_record.get("features", {}).get("replayability", 1.0) < 1.0:
            active_exclusions.append("invalid_replay_binding")

    if not active_exclusions:
        eligibility_state = "ELIGIBLE"
    elif set(active_exclusions) - {"manual_override", "missing_evidence", "invalid_replay_binding"}:
        # Any hard exclusion → DECLINED
        eligibility_state = "DECLINED"
    else:
        eligibility_state = "ELIGIBLE_WITH_EXCLUSIONS"

    liability_basis = insurance_contract.get("liability", _DEFAULT_LIABILITY_BASIS)

    insurance_eligibility_id = f"inselig_{uuid4().hex}"
    payload = {
        "insurance_eligibility_id": insurance_eligibility_id,
        "projection_id": projection_id,
        "tenant_id": tenant_id,
        "insurance_contract_version": insurance_contract.get("version", "v1.0.0"),
        "cq_score_id": cq_score_record.get("cq_score_id", ""),
        "eligibility_state": eligibility_state,
        "exclusions": active_exclusions,
        "liability_basis": liability_basis,
    }
    payload["content_hash"] = content_hash(payload)
    return payload


# ---------------------------------------------------------------------------
# Public: compute_risk_quote
# ---------------------------------------------------------------------------


def compute_risk_quote(
    projection_id: str,
    tenant_id: str,
    exposure_amount: float,
    cq_score_record: dict,
    insurance_contract: dict,
    calibration_model: dict,
) -> dict:
    """Compute a ComplianceQualityRiskQuote_t from CQ score and calibration model.

    Quotes generated from governance_prior models are provisional (§18.12).
    """
    score = cq_score_record.get("score", 0.0)
    params = calibration_model.get("parameters", {})

    base_frequency: float = params.get("base_frequency", 0.015)
    base_severity_rate: float = params.get("base_severity_rate", 0.08)

    # Apply CQ-tiered multiplier to frequency
    frequency_multiplier = 1.0
    for tier in params.get("cq_multipliers", []):
        if tier["min"] <= score <= tier["max"]:
            frequency_multiplier = tier["multiplier"]
            break

    expected_frequency = round(base_frequency * frequency_multiplier, 8)
    expected_severity_rate = base_severity_rate
    expected_loss = round(exposure_amount * expected_frequency * expected_severity_rate, 4)
    pure_premium = expected_loss

    pricing = insurance_contract.get("pricing", {})
    expense_load = pricing.get("expense_load", 0.20)
    risk_margin = pricing.get("risk_margin", 0.10)
    capital_charge = pricing.get("capital_charge", 0.10)
    gross_premium = round(pure_premium * (1 + expense_load + risk_margin + capital_charge), 4)

    liability_cfg = insurance_contract.get("liability_limits", {})
    liability_cap = liability_cfg.get("cap", exposure_amount * 0.10)
    deductible = liability_cfg.get("deductible", 500.0)

    eligibility_state = (
        cq_score_record.get("gates", {}).get("insurance_eligible_candidate", False)
        and True
    )
    elig = "ELIGIBLE" if eligibility_state else "REQUIRES_REVIEW"

    quote_id = f"cqrq_{uuid4().hex}"
    payload = {
        "quote_id": quote_id,
        "projection_id": projection_id,
        "tenant_id": tenant_id,
        "cq_score": score,
        "cq_contract_version": cq_score_record.get("cq_contract_version", "v2.0.0"),
        "calibration_model_version": calibration_model.get("model_version", "v1"),
        "calibration_status": calibration_model["calibration_status"],
        "exposure_amount": exposure_amount,
        "expected_frequency": expected_frequency,
        "expected_severity_rate": expected_severity_rate,
        "expected_loss": expected_loss,
        "pure_premium": pure_premium,
        "gross_premium": gross_premium,
        "liability_cap": liability_cap,
        "deductible": deductible,
        "eligibility_state": elig,
    }
    payload["content_hash"] = content_hash(payload)
    return payload
