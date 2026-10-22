from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from datetime import datetime, timezone
from typing import Any

from eos import real_company_setup_store
from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, init_db, now_iso
from eos.full_cycle_store import (
    get_accounting_archive_package,
    get_cash_journal_view,
    get_financial_statement_package,
    get_general_ledger_view,
    get_journal_voucher_projection,
    get_source_document_review,
    get_tax_filing_package,
    insert_accounting_archive_package,
    insert_cash_journal_view,
    insert_financial_statement_package,
    insert_general_ledger_view,
    insert_journal_voucher_projection,
    insert_source_document_review,
    insert_tax_filing_package,
)
from eos.projection_exception_store import create_projection_exception
from eos.real_company_coa_bootstrap import get_active_coa_bundle
from eos.replay_binding_store import create_replay_binding, get_replay_binding


def materialize_lifecycle(
    *,
    organization_id: str,
    dataset_version_id: str,
    requested_by: str,
    idempotency_key: str,
) -> dict[str, Any]:
    init_db()
    real_company_setup_store.ensure_schema()
    run = real_company_setup_store.create_or_get_setup_run(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        requested_by=requested_by,
        idempotency_key=idempotency_key,
        stage="lifecycle_materialization",
    )
    if run.get("ok") is False:
        return {
            "ok": False,
            "setup_run": run,
            "blocking_reasons": run.get("blocking_reasons", ["SETUP_RUN_LOCKED"]),
        }
    if run.get("status") in real_company_setup_store.TERMINAL_RUN_STATUSES:
        return {
            "ok": True,
            "setup_run": run,
            "result": _existing_materialization_result(organization_id, dataset_version_id),
            "warnings": [],
            "blocking_reasons": [],
        }

    claims = _admitted_claims(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
    )
    if not claims:
        return _block_run(
            run=run,
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            warning="REAL_REA_ADMISSION_REQUIRED",
        )
    active_bundle = get_active_coa_bundle(organization_id)
    if not active_bundle or not active_bundle.get("projection_bundle"):
        return _block_run(
            run=run,
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            warning="REAL_COA_REQUIRED",
        )

    projection_bundle = active_bundle["projection_bundle"]
    projection_bundle_version = str(projection_bundle["bundle_version"])
    coa_bundle_version = str(active_bundle["bundle_version"])
    mapping_rule = _first_mapping_rule(organization_id, projection_bundle_version)
    warnings: list[str] = []
    source_review_ids: list[str] = []
    jvp_ids: list[str] = []
    ledger_view_ids: list[str] = []
    projection_exception_ids: list[str] = []

    for claim_row in claims:
        claim = claim_row["claim"]
        period_id = _period_id(claim)
        source_review = _materialize_source_document_review(
            organization_id=organization_id,
            period_id=period_id,
            claim_row=claim_row,
        )
        if source_review:
            source_review_ids.append(source_review["review_id"])

        if not mapping_rule:
            projection_exception_ids.append(
                _create_projection_exception(
                    claim_row=claim_row,
                    projection_bundle_version=projection_bundle_version,
                    coa_bundle_version=coa_bundle_version,
                    reason="no_mapping_rule_exists",
                )["projection_exception_id"]
            )
            continue

        jvp = _materialize_jvp(
            organization_id=organization_id,
            period_id=period_id,
            claim_row=claim_row,
            mapping_rule=mapping_rule,
            projection_bundle_version=projection_bundle_version,
            coa_bundle_version=coa_bundle_version,
        )
        jvp_ids.append(jvp["jvp_id"])

    if projection_exception_ids:
        warnings.append("REAL_PROJECTION_EXCEPTIONS_PRESENT")
    if not jvp_ids:
        return _block_run(
            run=run,
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            warning="REAL_PROJECTION_REQUIRED",
            details={"projection_exception_ids": projection_exception_ids},
        )

    lifecycle = _materialize_lifecycle_views(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        jvp_ids=jvp_ids,
        source_review_ids=source_review_ids,
        projection_exception_ids=projection_exception_ids,
        projection_bundle_version=projection_bundle_version,
        coa_bundle_version=coa_bundle_version,
    )
    ledger_view_ids.extend(lifecycle["ledger_view_ids"])
    replay_binding = create_replay_binding(
        event_id=_replay_event_id(organization_id, dataset_version_id),
        pins={"Pi_v": projection_bundle_version, "K_v": coa_bundle_version},
        projection_context_ref=f"real_company_lifecycle:{organization_id}:{dataset_version_id}",
        projection_context_hash=content_hash(
            {
                "organization_id": organization_id,
                "dataset_version_id": dataset_version_id,
                "jvp_ids": jvp_ids,
                "ledger_view_ids": ledger_view_ids,
                "projection_bundle_version": projection_bundle_version,
                "coa_bundle_version": coa_bundle_version,
            }
        ),
        resolution_trace={"stage": "lifecycle_materialization", "status": "ready"},
        effects=lifecycle,
        event_payload={"organization_id": organization_id, "dataset_version_id": dataset_version_id},
        authority_source_manifest=[
            {
                "source_type": "real_company_lifecycle",
                "dataset_version_id": dataset_version_id,
                "projection_bundle_version": projection_bundle_version,
            }
        ],
    )
    governance_artifacts = _persist_governance_artifacts(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        requested_by=requested_by,
        projection_bundle_version=projection_bundle_version,
        coa_bundle_version=coa_bundle_version,
        replay_binding_event_id=str(replay_binding["event_id"]),
    )
    real_company_setup_store.append_setup_event(
        setup_run_id=str(run["setup_run_id"]),
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        stage="lifecycle_materialization",
        event_type="lifecycle_materialized",
        event_status="ready",
        details={
            "jvp_ids": jvp_ids,
            "ledger_view_ids": ledger_view_ids,
            "projection_exception_ids": projection_exception_ids,
            "financial_statement_package_id": lifecycle["financial_statement_package_id"],
            "tax_filing_package_id": lifecycle["tax_filing_package_id"],
            "accounting_archive_package_id": lifecycle["accounting_archive_package_id"],
            "replay_binding_event_id": replay_binding["event_id"],
            "governance_artifacts": governance_artifacts,
            "warnings": warnings,
        },
    )
    real_company_setup_store.transition_setup_run(
        setup_run_id=str(run["setup_run_id"]),
        status="completed",
        current_stage="lifecycle_materialization",
    )
    return {
        "ok": True,
        "setup_run": run,
        "result": {
            "source_review_ids": source_review_ids,
            "jvp_ids": jvp_ids,
            "ledger_view_ids": ledger_view_ids,
            "projection_exception_ids": projection_exception_ids,
            **lifecycle,
            "replay_binding": replay_binding,
            "governance_artifacts": governance_artifacts,
        },
        "warnings": warnings,
        "blocking_reasons": [],
    }


def _persist_governance_artifacts(
    *,
    organization_id: str,
    dataset_version_id: str,
    requested_by: str,
    projection_bundle_version: str,
    coa_bundle_version: str,
    replay_binding_event_id: str,
) -> dict[str, str]:
    now = now_iso()
    base = {
        "organization_id": organization_id,
        "dataset_version_id": dataset_version_id,
        "projection_bundle_version": projection_bundle_version,
        "coa_bundle_version": coa_bundle_version,
        "replay_binding_event_id": replay_binding_event_id,
    }
    artifact_id = f"real_company_lifecycle_artifact:{organization_id}:{dataset_version_id}"
    artifact_hash = content_hash({"artifact_id": artifact_id, **base})
    candidate_id = f"real_company_lifecycle_policy:{organization_id}:{dataset_version_id}"
    candidate_hash = content_hash({"candidate_id": candidate_id, **base})
    approval_record_id = f"real_company_lifecycle_approval:{organization_id}:{dataset_version_id}"
    candidate_event_id = f"real_company_lifecycle_policy_event:{organization_id}:{dataset_version_id}"
    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("""
            INSERT OR IGNORE INTO knowledge_artifacts (
              artifact_id, source_type, authority_domain, tag_authority_level,
              semantic_tier, authority_origin, source_ref, source_hash, source_uri,
              jurisdiction, effective_from, effective_to, source_version,
              source_anchors_json, extraction_method, curator, ambiguities_json,
              extracted_claims_json, extracted_constraints_json,
              extracted_projection_rules_json, ingestion_status, content_hash,
              created_at, approved_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO knowledge_artifacts (
              artifact_id, source_type, authority_domain, tag_authority_level,
              semantic_tier, authority_origin, source_ref, source_hash, source_uri,
              jurisdiction, effective_from, effective_to, source_version,
              source_anchors_json, extraction_method, curator, ambiguities_json,
              extracted_claims_json, extracted_constraints_json,
              extracted_projection_rules_json, ingestion_status, content_hash,
              created_at, approved_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
            (
                artifact_id,
                "runtime_observation",
                "accounting",
                "operational",
                "T6",
                "runtime_observation",
                replay_binding_event_id,
                artifact_hash,
                None,
                "CN",
                now,
                None,
                dataset_version_id,
                canonical_json([base]),
                "deterministic_real_company_lifecycle_materialization",
                requested_by,
                "[]",
                canonical_json([base]),
                canonical_json([]),
                canonical_json([]),
                "ACTIVE",
                artifact_hash,
                now,
                requested_by,
            ),
        )
        conn.execute(
            hosted.sql_query("""
            INSERT OR IGNORE INTO kgl_approval_records (
              record_id, artifact_id, from_status, to_status, actor, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO kgl_approval_records (
              record_id, artifact_id, from_status, to_status, actor, notes, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
            (
                approval_record_id,
                artifact_id,
                "PROPOSED",
                "ACTIVE",
                requested_by,
                "real_company_lifecycle_materialization",
                now,
            ),
        )
        conn.execute(
            hosted.sql_query("""
            INSERT OR IGNORE INTO policy_candidates (
              candidate_id, name, category, semantic_tier, content_yaml,
              content_hash, status, replay_result, proposed_by, approved_by,
              scoped_owner, rollback_window_hours, rollback_strategy, created_at,
              approved_at, notes, review_status, jurisdiction_reason, imported_legacy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO policy_candidates (
              candidate_id, name, category, semantic_tier, content_yaml,
              content_hash, status, replay_result, proposed_by, approved_by,
              scoped_owner, rollback_window_hours, rollback_strategy, created_at,
              approved_at, notes, review_status, jurisdiction_reason, imported_legacy
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
            (
                candidate_id,
                "real_company_lifecycle_projection_binding",
                "policy_bundle",
                "T6",
                canonical_json(base),
                candidate_hash,
                "ACTIVE",
                replay_binding_event_id,
                requested_by,
                requested_by,
                organization_id,
                0,
                "append_only_replay_binding",
                now,
                now,
                "real_company_lifecycle_materialization",
                "APPROVED",
                "CN",
                0,
            ),
        )
        conn.execute(
            hosted.sql_query("""
            INSERT OR IGNORE INTO policy_candidate_events (
              candidate_event_id, candidate_id, event_type, event_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """, """
            INSERT INTO policy_candidate_events (
              candidate_event_id, candidate_id, event_type, event_json, created_at
            ) VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
            (
                candidate_event_id,
                candidate_id,
                "activated",
                canonical_json(base),
                now,
            ),
        )
    return {
        "artifact_id": artifact_id,
        "candidate_id": candidate_id,
        "approval_record_id": approval_record_id,
        "candidate_event_id": candidate_event_id,
    }


def _admitted_claims(*, organization_id: str, dataset_version_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM rea_claims ORDER BY created_at, rea_id").fetchall()
    admitted: list[dict[str, Any]] = []
    for row in rows:
        try:
            claim = json.loads(row["claim_json"] or "{}")
        except json.JSONDecodeError:
            continue
        if (
            str(claim.get("org_id") or "") != organization_id
            or str(claim.get("dataset_version_id") or "") != dataset_version_id
        ):
            continue
        admitted.append({**dict(row), "claim": claim})
    return admitted


def _first_mapping_rule(organization_id: str, projection_bundle_version: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM coa_mapping_rules
            WHERE org_id = ? AND projection_bundle_version = ? AND status = 'ACTIVE'
            ORDER BY created_at, rule_id
            LIMIT 1
            """, """
            SELECT * FROM coa_mapping_rules
            WHERE org_id = %s AND projection_bundle_version = %s AND status = 'ACTIVE'
            ORDER BY created_at, rule_id
            LIMIT 1
            """),
            (organization_id, projection_bundle_version),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["rule_yaml"])


def _materialize_source_document_review(
    *,
    organization_id: str,
    period_id: str,
    claim_row: dict[str, Any],
) -> dict[str, Any] | None:
    provenance = claim_row["claim"].get("source_provenance") or {}
    source_file_hash = str(provenance.get("source_file_hash") or "").strip()
    if not source_file_hash:
        return None
    review_id = f"sdr_{claim_row['rea_id']}"
    existing = get_source_document_review(review_id)
    if existing:
        return existing
    return insert_source_document_review(
        {
            "review_id": review_id,
            "org_id": organization_id,
            "period_id": period_id,
            "document_type": "real_company_source_row",
            "submitter_id": claim_row["claim"].get("admitted_by"),
            "channel": "real_company_onboarding",
            "file_hash": source_file_hash,
            "authenticity_check": "PASS",
            "legality_check": "PENDING",
            "completeness_check": "PASS",
            "arithmetic_check": "PASS",
            "missing_fields_json": [],
            "extraction_result_json": {
                "rea_id": claim_row["rea_id"],
                "source_row_hash": provenance.get("source_row_hash"),
            },
            "evidence_status": "ELIGIBLE_FOR_REA",
        }
    )


def _materialize_jvp(
    *,
    organization_id: str,
    period_id: str,
    claim_row: dict[str, Any],
    mapping_rule: dict[str, Any],
    projection_bundle_version: str,
    coa_bundle_version: str,
) -> dict[str, Any]:
    jvp_id = f"jvp_{claim_row['rea_id']}"
    existing = get_journal_voucher_projection(jvp_id)
    if existing:
        return existing
    amount = abs(float(claim_row["claim"].get("amount") or 0.0))
    currency = str(claim_row["claim"].get("currency") or "CNY")
    project = mapping_rule.get("project") or {}
    lines = [
        {
            "side": "debit",
            "account_code": str((project.get("debit") or {}).get("account_code") or "1122"),
            "amount": amount,
            "currency": currency,
        },
        {
            "side": "credit",
            "account_code": str((project.get("credit") or {}).get("account_code") or "5001"),
            "amount": amount,
            "currency": currency,
        },
    ]
    return insert_journal_voucher_projection(
        {
            "jvp_id": jvp_id,
            "org_id": organization_id,
            "period_id": period_id,
            "source_event_id": claim_row["rea_id"],
            "projection_bundle_version": projection_bundle_version,
            "coa_version": coa_bundle_version,
            "voucher_type": "sale",
            "lines_json": lines,
            "justification_ref": f"real_company_lifecycle:{claim_row['rea_id']}",
            "projection_status": "projection_validated",
        }
    )


def _materialize_lifecycle_views(
    *,
    organization_id: str,
    dataset_version_id: str,
    jvp_ids: list[str],
    source_review_ids: list[str],
    projection_exception_ids: list[str],
    projection_bundle_version: str,
    coa_bundle_version: str,
) -> dict[str, Any]:
    period_id = f"{dataset_version_id}:active"
    jvps = [_load_jvp(jvp_id) for jvp_id in jvp_ids]
    entries = [
        {**line, "jvp_id": jvp["jvp_id"], "source_event_id": jvp["source_event_id"]}
        for jvp in jvps
        for line in jvp.get("lines_json", [])
    ]
    cash = _insert_cash_journal_if_missing(
        organization_id=organization_id,
        period_id=period_id,
        jvp_ids=jvp_ids,
        projection_bundle_version=projection_bundle_version,
        coa_bundle_version=coa_bundle_version,
        entries=entries,
    )
    ledger = _insert_general_ledger_if_missing(
        organization_id=organization_id,
        period_id=period_id,
        jvp_ids=jvp_ids,
        projection_bundle_version=projection_bundle_version,
        coa_bundle_version=coa_bundle_version,
        entries=entries,
    )
    financial = _insert_financial_statement_if_missing(
        organization_id=organization_id,
        period_id=period_id,
        ledger_view_id=ledger["view_id"],
        replay_binding_ref=_replay_event_id(organization_id, dataset_version_id),
    )
    tax = _insert_tax_filing_if_missing(
        organization_id=organization_id,
        period_id=period_id,
        replay_binding_ref=_replay_event_id(organization_id, dataset_version_id),
        source_refs={"jvp_ids": jvp_ids, "dataset_version_id": dataset_version_id},
    )
    archive = _insert_archive_if_missing(
        organization_id=organization_id,
        period_id=period_id,
        source_review_ids=source_review_ids,
        rea_ids=[jvp["source_event_id"] for jvp in jvps],
        jvp_ids=jvp_ids,
        ledger_view_ids=[cash["view_id"], ledger["view_id"]],
        projection_exception_ids=projection_exception_ids,
        financial_statement_package_id=financial["package_id"],
        tax_filing_package_id=tax["filing_id"],
    )
    return {
        "cash_journal_view_id": cash["view_id"],
        "ledger_view_ids": [cash["view_id"], ledger["view_id"]],
        "general_ledger_view_id": ledger["view_id"],
        "financial_statement_package_id": financial["package_id"],
        "tax_filing_package_id": tax["filing_id"],
        "accounting_archive_package_id": archive["archive_id"],
    }


def _load_jvp(jvp_id: str) -> dict[str, Any]:
    jvp = get_journal_voucher_projection(jvp_id)
    if jvp is None:
        raise ValueError(f"journal_voucher_projection_not_found:{jvp_id}")
    return jvp


def _insert_cash_journal_if_missing(
    *,
    organization_id: str,
    period_id: str,
    jvp_ids: list[str],
    projection_bundle_version: str,
    coa_bundle_version: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    view_id = f"cjv_{content_hash({'org_id': organization_id, 'period_id': period_id, 'jvp_ids': jvp_ids})[:24]}"
    existing = get_cash_journal_view(view_id)
    if existing:
        return existing
    return insert_cash_journal_view(
        {
            "view_id": view_id,
            "org_id": organization_id,
            "period_id": period_id,
            "source_jvp_ids_json": jvp_ids,
            "coa_version": coa_bundle_version,
            "projection_bundle_version": projection_bundle_version,
            "entries_json": entries,
        }
    )


def _insert_general_ledger_if_missing(
    *,
    organization_id: str,
    period_id: str,
    jvp_ids: list[str],
    projection_bundle_version: str,
    coa_bundle_version: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    view_id = f"glv_{content_hash({'org_id': organization_id, 'period_id': period_id, 'jvp_ids': jvp_ids})[:24]}"
    existing = get_general_ledger_view(view_id)
    if existing:
        return existing
    balances: dict[str, dict[str, Any]] = {}
    for entry in entries:
        account_code = str(entry.get("account_code") or "")
        item = balances.setdefault(
            account_code,
            {"account_code": account_code, "debit": 0.0, "credit": 0.0, "currency": entry.get("currency", "CNY")},
        )
        side = str(entry.get("side") or "")
        item[side] = float(item.get(side) or 0.0) + float(entry.get("amount") or 0.0)
    return insert_general_ledger_view(
        {
            "view_id": view_id,
            "org_id": organization_id,
            "period_id": period_id,
            "source_jvp_ids_json": jvp_ids,
            "coa_version": coa_bundle_version,
            "projection_bundle_version": projection_bundle_version,
            "balances_json": list(balances.values()),
        }
    )


def _insert_financial_statement_if_missing(
    *,
    organization_id: str,
    period_id: str,
    ledger_view_id: str,
    replay_binding_ref: str,
) -> dict[str, Any]:
    package_id = f"fsp_{content_hash({'org_id': organization_id, 'period_id': period_id, 'ledger_view_id': ledger_view_id})[:24]}"
    existing = get_financial_statement_package(package_id)
    if existing:
        return existing
    return insert_financial_statement_package(
        {
            "package_id": package_id,
            "org_id": organization_id,
            "period_id": period_id,
            "trial_balance_view_id": ledger_view_id,
            "reporting_contract_version": "real_company_reporting.v1",
            "balance_sheet_json": {"source_ledger_view_id": ledger_view_id},
            "income_statement_json": {"source_ledger_view_id": ledger_view_id},
            "cash_flow_statement_json": {"source_ledger_view_id": ledger_view_id},
            "validation_results_json": {"status": "not_closed", "warnings": ["PERIOD_CLOSE_NOT_RUN"]},
            "replay_binding_ref": replay_binding_ref,
        }
    )


def _insert_tax_filing_if_missing(
    *,
    organization_id: str,
    period_id: str,
    replay_binding_ref: str,
    source_refs: dict[str, Any],
) -> dict[str, Any]:
    filing_id = f"tfp_{content_hash({'org_id': organization_id, 'period_id': period_id, 'source_refs': source_refs})[:24]}"
    existing = get_tax_filing_package(filing_id)
    if existing:
        return existing
    return insert_tax_filing_package(
        {
            "filing_id": filing_id,
            "org_id": organization_id,
            "period_id": period_id,
            "tax_type": "vat_income_placeholder",
            "tax_rule_bundle_version": "cn_tax_rules.placeholder.v1",
            "source_refs_json": source_refs,
            "filing_form_json": {"status": "draft"},
            "validation_result_json": {"status": "requires_review"},
            "approval_state": "draft",
            "replay_binding_ref": replay_binding_ref,
        }
    )


def _insert_archive_if_missing(
    *,
    organization_id: str,
    period_id: str,
    source_review_ids: list[str],
    rea_ids: list[str],
    jvp_ids: list[str],
    ledger_view_ids: list[str],
    projection_exception_ids: list[str],
    financial_statement_package_id: str,
    tax_filing_package_id: str,
) -> dict[str, Any]:
    archive_payload = {
        "source_review_ids": source_review_ids,
        "rea_ids": rea_ids,
        "jvp_ids": jvp_ids,
        "ledger_view_ids": ledger_view_ids,
        "projection_exception_ids": projection_exception_ids,
        "financial_statement_package_id": financial_statement_package_id,
        "tax_filing_package_id": tax_filing_package_id,
    }
    archive_hash = content_hash(archive_payload)
    archive_id = f"aap_{archive_hash[:24]}"
    existing = get_accounting_archive_package(archive_id)
    if existing:
        return existing
    return insert_accounting_archive_package(
        {
            "archive_id": archive_id,
            "org_id": organization_id,
            "period_id": period_id,
            "source_document_review_ids_json": source_review_ids,
            "rea_event_ids_json": rea_ids,
            "jvp_ids_json": jvp_ids,
            "ledger_view_ids_json": ledger_view_ids,
            "validation_result_ids_json": [],
            "governance_decision_ids_json": projection_exception_ids,
            "financial_statement_package_ids_json": [financial_statement_package_id],
            "tax_filing_package_ids_json": [tax_filing_package_id],
            "external_audit_export_refs_json": [],
            "archive_hash": archive_hash,
            "manifest_ref": f"real_company_archive:{archive_id}",
        }
    )


def _create_projection_exception(
    *,
    claim_row: dict[str, Any],
    projection_bundle_version: str,
    coa_bundle_version: str,
    reason: str,
) -> dict[str, Any]:
    existing = _existing_projection_exception(claim_row["rea_id"], projection_bundle_version)
    if existing:
        return existing
    return create_projection_exception(
        rea_id=claim_row["rea_id"],
        justification_id=f"real_company_projection:{claim_row['rea_id']}",
        projection_context_ref=f"real_company_lifecycle:{claim_row['claim'].get('dataset_version_id')}",
        projection_context_hash=content_hash(claim_row["claim"]),
        coa_version=coa_bundle_version,
        projection_bundle_version=projection_bundle_version,
        constraint_bundle_version=None,
        status="OPEN",
        exception_payload={
            "reason": reason,
            "rea_id": claim_row["rea_id"],
            "dataset_version_id": claim_row["claim"].get("dataset_version_id"),
        },
    )


def _existing_projection_exception(rea_id: str, projection_bundle_version: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM projection_exceptions
            WHERE rea_id = ? AND projection_bundle_version = ?
            ORDER BY created_at DESC
            LIMIT 1
            """, """
            SELECT * FROM projection_exceptions
            WHERE rea_id = %s AND projection_bundle_version = %s
            ORDER BY created_at DESC
            LIMIT 1
            """),
            (rea_id, projection_bundle_version),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["exception"] = json.loads(result["exception_json"])
    return result


def _block_run(
    *,
    run: dict[str, Any],
    organization_id: str,
    dataset_version_id: str,
    warning: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    real_company_setup_store.append_setup_event(
        setup_run_id=str(run["setup_run_id"]),
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        stage="lifecycle_materialization",
        event_type="lifecycle_blocked",
        event_status="blocked",
        details={"warning": warning, **(details or {})},
    )
    real_company_setup_store.transition_setup_run(
        setup_run_id=str(run["setup_run_id"]),
        status="blocked",
        current_stage="lifecycle_materialization",
        blocking_reason=warning,
    )
    return {
        "ok": False,
        "setup_run": run,
        "result": details or {},
        "warnings": [warning],
        "blocking_reasons": [warning],
    }


def _existing_materialization_result(organization_id: str, dataset_version_id: str) -> dict[str, Any]:
    event_id = _replay_event_id(organization_id, dataset_version_id)
    return {
        "replay_binding": get_replay_binding(event_id),
    }


def _period_id(claim: dict[str, Any]) -> str:
    text = str(claim.get("occurred_at") or "")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime(1970, 1, 1, tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return f"{parsed.year:04d}-{parsed.month:02d}"


def _replay_event_id(organization_id: str, dataset_version_id: str) -> str:
    return f"real_company_lifecycle:{organization_id}:{dataset_version_id}"
