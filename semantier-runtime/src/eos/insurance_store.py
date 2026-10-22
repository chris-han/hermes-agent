"""Insurance store — InsuranceEligibilityResult_t, ComplianceQualityRiskQuote_t,
ReinsurancePortfolioManifest_t.

Architecture ref: docs/architecture_v8_cq_v2_appendix.md §18.10–18.13, §18.14

All tables are append-only.
Insurance eligibility MUST NOT imply that the REA fact is correct.
It only covers the defined projected representation under pinned artifacts and exclusions.
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, insert_json, now_iso

_VALID_ELIGIBILITY_STATES = {"ELIGIBLE", "ELIGIBLE_WITH_EXCLUSIONS", "REQUIRES_REVIEW", "DECLINED"}


# ---------------------------------------------------------------------------
# Insurance Eligibility Results
# ---------------------------------------------------------------------------


def insert_insurance_eligibility_result(
    projection_id: str,
    tenant_id: str,
    insurance_contract_version: str,
    cq_score_id: str,
    eligibility_state: str,
    liability_basis: dict,
    exclusions: list[str] | None = None,
) -> dict:
    """Persist an InsuranceEligibilityResult_t."""
    if eligibility_state not in _VALID_ELIGIBILITY_STATES:
        raise ValueError(f"Invalid eligibility_state: {eligibility_state!r}")
    insurance_eligibility_id = f"inselig_{uuid4().hex}"
    payload = {
        "insurance_eligibility_id": insurance_eligibility_id,
        "projection_id": projection_id,
        "tenant_id": tenant_id,
        "insurance_contract_version": insurance_contract_version,
        "cq_score_id": cq_score_id,
        "eligibility_state": eligibility_state,
        "exclusions": exclusions or [],
        "liability_basis": liability_basis,
    }
    ch = content_hash(payload)
    record = {
        "insurance_eligibility_id": insurance_eligibility_id,
        "projection_id": projection_id,
        "tenant_id": tenant_id,
        "insurance_contract_version": insurance_contract_version,
        "cq_score_id": cq_score_id,
        "eligibility_state": eligibility_state,
        "exclusions_json": canonical_json(exclusions or []),
        "liability_basis_json": canonical_json(liability_basis),
        "content_hash": ch,
        "created_at": now_iso(),
    }
    insert_json("insurance_eligibility_results", record)
    return {**record, "exclusions": exclusions or [], "liability_basis": liability_basis}


def get_insurance_eligibility_result(insurance_eligibility_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM insurance_eligibility_results WHERE insurance_eligibility_id = ?", 'SELECT * FROM insurance_eligibility_results WHERE insurance_eligibility_id = %s'),
            (insurance_eligibility_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["exclusions"] = json.loads(d["exclusions_json"]) if d.get("exclusions_json") else []
    d["liability_basis"] = json.loads(d["liability_basis_json"])
    return d


# ---------------------------------------------------------------------------
# CQ Risk Quotes
# ---------------------------------------------------------------------------


def insert_cq_risk_quote(
    projection_id: str,
    tenant_id: str,
    cq_score: float,
    cq_contract_version: str,
    calibration_model_version: str,
    calibration_status: str,
    exposure_amount: float,
    eligibility_state: str,
    *,
    expected_frequency: float | None = None,
    expected_severity_rate: float | None = None,
    expected_loss: float | None = None,
    pure_premium: float | None = None,
    gross_premium: float | None = None,
    liability_cap: float | None = None,
    deductible: float | None = None,
) -> dict:
    """Persist a ComplianceQualityRiskQuote_t.

    A quote generated from governance_prior MUST be marked provisional unless
    an insurer-approved rating plan says otherwise.
    """
    if calibration_status not in {"governance_prior", "empirically_calibrated"}:
        raise ValueError(f"Invalid calibration_status: {calibration_status!r}")
    if eligibility_state not in _VALID_ELIGIBILITY_STATES:
        raise ValueError(f"Invalid eligibility_state: {eligibility_state!r}")
    quote_id = f"cqrq_{uuid4().hex}"
    payload = {
        "quote_id": quote_id,
        "projection_id": projection_id,
        "tenant_id": tenant_id,
        "cq_score": cq_score,
        "cq_contract_version": cq_contract_version,
        "calibration_model_version": calibration_model_version,
        "calibration_status": calibration_status,
        "exposure_amount": exposure_amount,
        "expected_frequency": expected_frequency,
        "expected_severity_rate": expected_severity_rate,
        "expected_loss": expected_loss,
        "pure_premium": pure_premium,
        "gross_premium": gross_premium,
        "liability_cap": liability_cap,
        "deductible": deductible,
        "eligibility_state": eligibility_state,
    }
    ch = content_hash(payload)
    record = {**payload, "content_hash": ch, "created_at": now_iso()}
    insert_json("cq_risk_quotes", record)
    return record


def get_cq_risk_quote(quote_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM cq_risk_quotes WHERE quote_id = ?", 'SELECT * FROM cq_risk_quotes WHERE quote_id = %s'),
            (quote_id,),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# Reinsurance Portfolio Manifests
# ---------------------------------------------------------------------------


def insert_reinsurance_portfolio_manifest(
    portfolio_id: str,
    period_start: str,
    period_end: str,
    cq_contract_version: str,
    calibration_model_version: str,
    insured_projection_count: int,
    total_exposure_amount: float,
    cq_distribution: dict,
    reinsurance_layers: dict,
    *,
    expected_loss_total: float | None = None,
    gross_premium_total: float | None = None,
    external_verification_manifest_refs: list[str] | None = None,
) -> dict:
    """Persist a ReinsurancePortfolioManifest_t.

    Reinsurance manifests are portfolio-level views.  They do not modify REA
    events, projections, CQ scores, or insurance eligibility records.
    """
    manifest_id = f"reins_manifest_{uuid4().hex}"
    payload = {
        "manifest_id": manifest_id,
        "portfolio_id": portfolio_id,
        "period_start": period_start,
        "period_end": period_end,
        "cq_contract_version": cq_contract_version,
        "calibration_model_version": calibration_model_version,
        "insured_projection_count": insured_projection_count,
        "total_exposure_amount": total_exposure_amount,
        "cq_distribution": cq_distribution,
        "expected_loss_total": expected_loss_total,
        "gross_premium_total": gross_premium_total,
        "reinsurance_layers": reinsurance_layers,
        "external_verification_manifest_refs": external_verification_manifest_refs or [],
    }
    ch = content_hash(payload)
    record = {
        "manifest_id": manifest_id,
        "portfolio_id": portfolio_id,
        "period_start": period_start,
        "period_end": period_end,
        "cq_contract_version": cq_contract_version,
        "calibration_model_version": calibration_model_version,
        "insured_projection_count": insured_projection_count,
        "total_exposure_amount": total_exposure_amount,
        "cq_distribution_json": canonical_json(cq_distribution),
        "expected_loss_total": expected_loss_total,
        "gross_premium_total": gross_premium_total,
        "reinsurance_layers_json": canonical_json(reinsurance_layers),
        "external_verification_manifest_refs_json": canonical_json(
            external_verification_manifest_refs or []
        ),
        "content_hash": ch,
        "created_at": now_iso(),
    }
    insert_json("reinsurance_portfolio_manifests", record)
    return {
        **record,
        "cq_distribution": cq_distribution,
        "reinsurance_layers": reinsurance_layers,
        "external_verification_manifest_refs": external_verification_manifest_refs or [],
    }


def get_reinsurance_portfolio_manifest(manifest_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM reinsurance_portfolio_manifests WHERE manifest_id = ?", 'SELECT * FROM reinsurance_portfolio_manifests WHERE manifest_id = %s'),
            (manifest_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["cq_distribution"] = json.loads(d["cq_distribution_json"])
    d["reinsurance_layers"] = json.loads(d["reinsurance_layers_json"])
    d["external_verification_manifest_refs"] = json.loads(
        d["external_verification_manifest_refs_json"] or "[]"
    )
    return d
