"""CQ v2 store — ComplianceQualityFeatureVector_t, ComplianceQualityScore_t,
ComplianceQualityOutcome_t, ComplianceQualityCalibrationModel_v.

Architecture ref: docs/architecture_v8_cq_v2_appendix.md §18.5–18.9, §18.14

All tables are append-only.  No UPDATE or DELETE operations are defined here.
Historical CQ scores must not be overwritten; recalculation under a new contract
creates a new score record with new version pins.
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, insert_json, now_iso


# ---------------------------------------------------------------------------
# CQ Feature Vectors
# ---------------------------------------------------------------------------


def insert_cq_feature_vector(
    projection_id: str,
    tenant_id: str,
    cq_contract_version: str,
    features: dict[str, float],
    source_artifacts: dict[str, str],
    trial_balance_view_id: str | None = None,
) -> dict:
    """Persist a ComplianceQualityFeatureVector_t and return the full record."""
    feature_vector_id = f"cqfv_{uuid4().hex}"
    features_json = canonical_json(features)
    source_artifacts_json = canonical_json(source_artifacts)
    payload = {
        "feature_vector_id": feature_vector_id,
        "projection_id": projection_id,
        "trial_balance_view_id": trial_balance_view_id,
        "tenant_id": tenant_id,
        "cq_contract_version": cq_contract_version,
        "features": features,
        "source_artifacts": source_artifacts,
    }
    ch = content_hash(payload)
    record = {
        "feature_vector_id": feature_vector_id,
        "projection_id": projection_id,
        "trial_balance_view_id": trial_balance_view_id,
        "tenant_id": tenant_id,
        "cq_contract_version": cq_contract_version,
        "features_json": features_json,
        "source_artifacts_json": source_artifacts_json,
        "content_hash": ch,
        "created_at": now_iso(),
    }
    insert_json("cq_feature_vectors", record)
    return {**record, "features": features, "source_artifacts": source_artifacts}


def get_cq_feature_vector(feature_vector_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM cq_feature_vectors WHERE feature_vector_id = ?", 'SELECT * FROM cq_feature_vectors WHERE feature_vector_id = %s'),
            (feature_vector_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["features"] = json.loads(d["features_json"])
    d["source_artifacts"] = json.loads(d["source_artifacts_json"])
    return d


# ---------------------------------------------------------------------------
# CQ Scores
# ---------------------------------------------------------------------------


def insert_cq_score(
    feature_vector_id: str,
    projection_id: str,
    tenant_id: str,
    cq_contract_version: str,
    calibration_model_version: str,
    calibration_status: str,
    score: float,
    gates: dict,
    explanation: dict | None = None,
) -> dict:
    """Persist a ComplianceQualityScore_t and return the full record."""
    if calibration_status not in {"governance_prior", "empirically_calibrated"}:
        raise ValueError(f"Invalid calibration_status: {calibration_status!r}")
    cq_score_id = f"cqs_{uuid4().hex}"
    payload = {
        "cq_score_id": cq_score_id,
        "feature_vector_id": feature_vector_id,
        "projection_id": projection_id,
        "tenant_id": tenant_id,
        "cq_contract_version": cq_contract_version,
        "calibration_model_version": calibration_model_version,
        "calibration_status": calibration_status,
        "score": score,
        "gates": gates,
        "explanation": explanation,
    }
    ch = content_hash(payload)
    record = {
        "cq_score_id": cq_score_id,
        "feature_vector_id": feature_vector_id,
        "projection_id": projection_id,
        "tenant_id": tenant_id,
        "cq_contract_version": cq_contract_version,
        "calibration_model_version": calibration_model_version,
        "calibration_status": calibration_status,
        "score": score,
        "gates_json": canonical_json(gates),
        "explanation_json": canonical_json(explanation) if explanation else None,
        "content_hash": ch,
        "created_at": now_iso(),
    }
    insert_json("cq_scores", record)
    return {**record, "gates": gates, "explanation": explanation}


def get_cq_score(cq_score_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM cq_scores WHERE cq_score_id = ?", 'SELECT * FROM cq_scores WHERE cq_score_id = %s'),
            (cq_score_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["gates"] = json.loads(d["gates_json"])
    d["explanation"] = json.loads(d["explanation_json"]) if d.get("explanation_json") else None
    return d


def get_latest_cq_score_for_projection(projection_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM cq_scores WHERE projection_id = ? ORDER BY created_at DESC LIMIT 1", 'SELECT * FROM cq_scores WHERE projection_id = %s ORDER BY created_at DESC LIMIT 1'),
            (projection_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["gates"] = json.loads(d["gates_json"])
    d["explanation"] = json.loads(d["explanation_json"]) if d.get("explanation_json") else None
    return d


# ---------------------------------------------------------------------------
# CQ Outcomes
# ---------------------------------------------------------------------------

_VALID_OUTCOME_TYPES = {
    "audit_adjustment",
    "tax_penalty",
    "insurance_claim",
    "manual_correction",
    "customer_dispute",
    "regulator_notice",
    "external_audit_finding",
    "advisor_override",
    "reversal_after_close",
}


def insert_cq_outcome(
    projection_id: str,
    tenant_id: str,
    outcome_type: str,
    outcome_date: str,
    *,
    claim_occurred: bool = False,
    claim_amount: float = 0.0,
    audit_adjustment_amount: float = 0.0,
    tax_penalty_amount: float = 0.0,
    correction_cost: float = 0.0,
    time_to_resolution_days: int | None = None,
    source_type: str,
    source_ref: str | None = None,
    evidence_hash: str | None = None,
) -> dict:
    """Persist a ComplianceQualityOutcome_t.

    Outcomes are append-only and must not rewrite CQ scores historically.
    They may only train future calibration models.
    """
    if outcome_type not in _VALID_OUTCOME_TYPES:
        raise ValueError(f"Invalid outcome_type: {outcome_type!r}")
    outcome_id = f"cqout_{uuid4().hex}"
    payload = {
        "outcome_id": outcome_id,
        "projection_id": projection_id,
        "tenant_id": tenant_id,
        "outcome_type": outcome_type,
        "outcome_date": outcome_date,
        "claim_occurred": claim_occurred,
        "claim_amount": claim_amount,
        "audit_adjustment_amount": audit_adjustment_amount,
        "tax_penalty_amount": tax_penalty_amount,
        "correction_cost": correction_cost,
        "time_to_resolution_days": time_to_resolution_days,
        "source_type": source_type,
        "source_ref": source_ref,
        "evidence_hash": evidence_hash,
    }
    ch = content_hash(payload)
    record = {
        **payload,
        "claim_occurred": 1 if claim_occurred else 0,
        "content_hash": ch,
        "created_at": now_iso(),
    }
    insert_json("cq_outcomes", record)
    return record


# ---------------------------------------------------------------------------
# CQ Calibration Models
# ---------------------------------------------------------------------------

_VALID_MODEL_TYPES = {
    "tiered_multiplier",
    "logistic",
    "poisson",
    "gamma",
    "lognormal",
    "empirical_percentile",
}

_VALID_GOV_STATES = {"candidate", "reviewed", "certified", "active", "deprecated"}


def insert_cq_calibration_model(
    model_id: str,
    model_version: str,
    model_type: str,
    calibration_status: str,
    feature_schema_hash: str,
    parameters: dict,
    governance_state: str,
    *,
    training_window_start: str | None = None,
    training_window_end: str | None = None,
    outcome_population_hash: str | None = None,
    performance: dict | None = None,
) -> dict:
    """Persist a ComplianceQualityCalibrationModel_v."""
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(f"Invalid model_type: {model_type!r}")
    if calibration_status not in {"governance_prior", "empirically_calibrated"}:
        raise ValueError(f"Invalid calibration_status: {calibration_status!r}")
    if governance_state not in _VALID_GOV_STATES:
        raise ValueError(f"Invalid governance_state: {governance_state!r}")
    payload = {
        "model_id": model_id,
        "model_version": model_version,
        "model_type": model_type,
        "calibration_status": calibration_status,
        "training_window_start": training_window_start,
        "training_window_end": training_window_end,
        "feature_schema_hash": feature_schema_hash,
        "outcome_population_hash": outcome_population_hash,
        "parameters": parameters,
        "performance": performance,
        "governance_state": governance_state,
    }
    ch = content_hash(payload)
    record = {
        "model_id": model_id,
        "model_version": model_version,
        "model_type": model_type,
        "calibration_status": calibration_status,
        "training_window_start": training_window_start,
        "training_window_end": training_window_end,
        "feature_schema_hash": feature_schema_hash,
        "outcome_population_hash": outcome_population_hash,
        "parameters_json": canonical_json(parameters),
        "performance_json": canonical_json(performance) if performance else None,
        "governance_state": governance_state,
        "content_hash": ch,
        "created_at": now_iso(),
    }
    insert_json("cq_calibration_models", record)
    return {**record, "parameters": parameters, "performance": performance}


def get_active_calibration_model(model_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM cq_calibration_models WHERE model_id = ? AND governance_state = 'active' LIMIT 1", "SELECT * FROM cq_calibration_models WHERE model_id = %s AND governance_state = 'active' LIMIT 1"),
            (model_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["parameters"] = json.loads(d["parameters_json"])
    d["performance"] = json.loads(d["performance_json"]) if d.get("performance_json") else None
    return d
