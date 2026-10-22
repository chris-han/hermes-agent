from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any

from eos import real_company_setup_store
from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, init_db, now_iso
from eos.replay_binding_store import create_replay_binding, get_replay_binding


DEFAULT_COA_BUNDLE_ID = "china_default_sme_coa"
DEFAULT_COA_BUNDLE_VERSION_PREFIX = "china_default_sme_coa"


DEFAULT_CHINA_COA_ENTRIES: tuple[dict[str, Any], ...] = (
    {
        "account_code": "1001",
        "account_name": "库存现金",
        "account_class": "asset",
        "normal_balance": "debit",
    },
    {
        "account_code": "1002",
        "account_name": "银行存款",
        "account_class": "asset",
        "normal_balance": "debit",
    },
    {
        "account_code": "1122",
        "account_name": "应收账款",
        "account_class": "asset",
        "normal_balance": "debit",
    },
    {
        "account_code": "2202",
        "account_name": "应付账款",
        "account_class": "liability",
        "normal_balance": "credit",
    },
    {
        "account_code": "2221",
        "account_name": "应交税费",
        "account_class": "liability",
        "normal_balance": "credit",
    },
    {
        "account_code": "4001",
        "account_name": "实收资本",
        "account_class": "equity",
        "normal_balance": "credit",
    },
    {
        "account_code": "5001",
        "account_name": "主营业务收入",
        "account_class": "revenue",
        "normal_balance": "credit",
    },
    {
        "account_code": "6001",
        "account_name": "主营业务成本",
        "account_class": "expense",
        "normal_balance": "debit",
    },
    {
        "account_code": "6602",
        "account_name": "管理费用",
        "account_class": "expense",
        "normal_balance": "debit",
    },
)


DEFAULT_MAPPING_RULES: tuple[dict[str, Any], ...] = (
    {
        "rule_id": "source_event_default_revenue",
        "when": {"rea_event_type": "source_event"},
        "project": {
            "debit": {"account_code": "1122", "dimensions": ["counterparty"]},
            "credit": {"account_code": "5001", "dimensions": []},
        },
        "requires": {
            "account_codes_exist_in_bundle": True,
            "debit_credit_balanced": True,
            "normal_balance_compatible": True,
        },
        "governance_state": "ACTIVE",
    },
)


def bootstrap_default_coa(
    *,
    organization_id: str,
    dataset_version_id: str,
    requested_by: str,
    idempotency_key: str,
    uploaded_coa_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    init_db()
    real_company_setup_store.ensure_schema()
    run = real_company_setup_store.create_or_get_setup_run(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        requested_by=requested_by,
        idempotency_key=idempotency_key,
        stage="coa_projection",
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
            "bundle": get_active_coa_bundle(organization_id),
            "replay_binding": get_replay_binding(_replay_event_id(organization_id, dataset_version_id)),
            "blocking_reasons": [],
        }

    now = now_iso()
    bundle = _bundle_payload(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        uploaded_coa_candidate=uploaded_coa_candidate,
    )
    bundle_hash = content_hash(bundle)
    projection_bundle_version = str(bundle["projection_bundle_version"])
    coa_version_id = f"coa_v_{bundle['bundle_version']}"
    projection_bundle_id = f"proj_bundle_{projection_bundle_version}"

    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("UPDATE coa_projection_bundles SET status = 'DEPRECATED' WHERE org_id = ? AND status = 'ACTIVE'", "UPDATE coa_projection_bundles SET status = 'DEPRECATED' WHERE org_id = %s AND status = 'ACTIVE'"),
            (organization_id,),
        )
        conn.execute(
            hosted.sql_query("""
            INSERT OR REPLACE INTO coa_projection_bundles (
              bundle_version, org_id, bundle_id, jurisdiction, accounting_standard,
              status, content_yaml, content_hash, source_refs, created_at, activated_at
            ) VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?)
            """, """
            INSERT INTO coa_projection_bundles (
              bundle_version, org_id, bundle_id, jurisdiction, accounting_standard,
              status, content_yaml, content_hash, source_refs, created_at, activated_at
            ) VALUES (%s, %s, %s, %s, %s, 'ACTIVE', %s, %s, %s, %s, %s) ON CONFLICT (bundle_version) DO UPDATE SET org_id=excluded.org_id, bundle_id=excluded.bundle_id, jurisdiction=excluded.jurisdiction, accounting_standard=excluded.accounting_standard, status=excluded.status, content_yaml=excluded.content_yaml, content_hash=excluded.content_hash, source_refs=excluded.source_refs, created_at=excluded.created_at, activated_at=excluded.activated_at"""),
            (
                bundle["bundle_version"],
                organization_id,
                DEFAULT_COA_BUNDLE_ID,
                "CN",
                "china_small_business_accounting_standards",
                canonical_json(bundle),
                bundle_hash,
                canonical_json(bundle["source_refs"]),
                now,
                now,
            ),
        )
        conn.execute(
            hosted.sql_query("""
            INSERT OR IGNORE INTO coa_versions (
              coa_version_id, org_id, bundle_version, version_label,
              content_json, content_hash, status, created_at, activated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
            """, """
            INSERT INTO coa_versions (
              coa_version_id, org_id, bundle_version, version_label,
              content_json, content_hash, status, created_at, activated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, 'ACTIVE', %s, %s) ON CONFLICT DO NOTHING"""),
            (
                coa_version_id,
                organization_id,
                bundle["bundle_version"],
                bundle["bundle_version"],
                canonical_json(bundle),
                bundle_hash,
                now,
                now,
            ),
        )
        projection_payload = {
            "bundle_version": projection_bundle_version,
            "coa_bundle_version": bundle["bundle_version"],
            "mapping_rules": bundle["mapping_rules"],
            "dataset_version_id": dataset_version_id,
            "source_refs": bundle["source_refs"],
        }
        conn.execute(
            hosted.sql_query("""
            INSERT OR IGNORE INTO projection_bundles (
              projection_bundle_id, org_id, bundle_version, coa_version_id,
              content_json, content_hash, status, created_at, activated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
            """, """
            INSERT INTO projection_bundles (
              projection_bundle_id, org_id, bundle_version, coa_version_id,
              content_json, content_hash, status, created_at, activated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, 'ACTIVE', %s, %s) ON CONFLICT DO NOTHING"""),
            (
                projection_bundle_id,
                organization_id,
                projection_bundle_version,
                coa_version_id,
                canonical_json(projection_payload),
                content_hash(projection_payload),
                now,
                now,
            ),
        )
        for entry in bundle["entries"]:
            conn.execute(
                hosted.sql_query("""
                INSERT OR IGNORE INTO coa_entries (
                  coa_entry_id, org_id, bundle_version, account_code, account_name,
                  account_class, normal_balance, parent_code, auxiliary_dimensions,
                  authority_domain, tag_authority_level, semantic_tier, source_refs,
                  status, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
                """, """
                INSERT INTO coa_entries (
                  coa_entry_id, org_id, bundle_version, account_code, account_name,
                  account_class, normal_balance, parent_code, auxiliary_dimensions,
                  authority_domain, tag_authority_level, semantic_tier, source_refs,
                  status, content_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'ACTIVE', %s, %s) ON CONFLICT DO NOTHING"""),
                (
                    f"coa_{bundle['bundle_version']}_{entry['account_code']}",
                    organization_id,
                    bundle["bundle_version"],
                    entry["account_code"],
                    entry["account_name"],
                    entry["account_class"],
                    entry["normal_balance"],
                    entry.get("parent_code"),
                    canonical_json(entry.get("auxiliary_dimensions", [])),
                    "accounting",
                    "accounting",
                    "T3",
                    canonical_json(entry.get("source_refs", [])),
                    content_hash(entry),
                    now,
                ),
            )
        for rule in bundle["mapping_rules"]:
            conn.execute(
                hosted.sql_query("""
                INSERT OR IGNORE INTO coa_mapping_rules (
                  rule_id, org_id, projection_bundle_version, coa_bundle_version,
                  rule_yaml, content_hash, status, source_refs, created_at, activated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)
                """, """
                INSERT INTO coa_mapping_rules (
                  rule_id, org_id, projection_bundle_version, coa_bundle_version,
                  rule_yaml, content_hash, status, source_refs, created_at, activated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, 'ACTIVE', %s, %s, %s) ON CONFLICT DO NOTHING"""),
                (
                    f"{projection_bundle_version}:{rule['rule_id']}",
                    organization_id,
                    projection_bundle_version,
                    bundle["bundle_version"],
                    canonical_json(rule),
                    content_hash(rule),
                    canonical_json(rule.get("source_refs", [])),
                    now,
                    now,
                ),
            )

    replay_binding = create_replay_binding(
        event_id=_replay_event_id(organization_id, dataset_version_id),
        pins={"Pi_v": projection_bundle_version, "K_v": bundle["bundle_version"]},
        projection_context_ref=f"real_company_coa:{organization_id}:{dataset_version_id}",
        projection_context_hash=content_hash(
            {
                "organization_id": organization_id,
                "dataset_version_id": dataset_version_id,
                "projection_bundle_version": projection_bundle_version,
                "coa_bundle_version": bundle["bundle_version"],
            }
        ),
        resolution_trace={"stage": "coa_projection", "status": "ready"},
        effects={"bundle_version": bundle["bundle_version"]},
        event_payload={"organization_id": organization_id, "dataset_version_id": dataset_version_id},
        authority_source_manifest=bundle["source_refs"],
    )
    real_company_setup_store.append_setup_event(
        setup_run_id=str(run["setup_run_id"]),
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        stage="coa_projection",
        event_type="coa_projection_ready",
        event_status="ready",
        details={
            "bundle_version": bundle["bundle_version"],
            "projection_bundle_version": projection_bundle_version,
            "replay_binding_event_id": replay_binding["event_id"],
            "content_hash": bundle_hash,
        },
    )
    real_company_setup_store.transition_setup_run(
        setup_run_id=str(run["setup_run_id"]),
        status="completed",
        current_stage="coa_projection",
    )
    return {
        "ok": True,
        "setup_run": run,
        "bundle": get_active_coa_bundle(organization_id),
        "replay_binding": replay_binding,
        "blocking_reasons": [],
    }


def get_active_coa_bundle(organization_id: str) -> dict[str, Any] | None:
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM coa_projection_bundles
            WHERE org_id = ? AND status = 'ACTIVE'
            ORDER BY activated_at DESC, created_at DESC
            LIMIT 1
            """, """
            SELECT * FROM coa_projection_bundles
            WHERE org_id = %s AND status = 'ACTIVE'
            ORDER BY activated_at DESC, created_at DESC
            LIMIT 1
            """),
            (organization_id,),
        ).fetchone()
        if row is None:
            return None
        entries = conn.execute(
            hosted.sql_query("""
            SELECT * FROM coa_entries
            WHERE org_id = ? AND bundle_version = ? AND status = 'ACTIVE'
            ORDER BY account_code
            """, """
            SELECT * FROM coa_entries
            WHERE org_id = %s AND bundle_version = %s AND status = 'ACTIVE'
            ORDER BY account_code
            """),
            (organization_id, row["bundle_version"]),
        ).fetchall()
        projection = conn.execute(
            hosted.sql_query("""
            SELECT * FROM projection_bundles
            WHERE org_id = ? AND status = 'ACTIVE'
            ORDER BY activated_at DESC, created_at DESC
            LIMIT 1
            """, """
            SELECT * FROM projection_bundles
            WHERE org_id = %s AND status = 'ACTIVE'
            ORDER BY activated_at DESC, created_at DESC
            LIMIT 1
            """),
            (organization_id,),
        ).fetchone()
    return {
        **dict(row),
        "entries": [dict(entry) for entry in entries],
        "projection_bundle": dict(projection) if projection else None,
    }


def _bundle_payload(
    *,
    organization_id: str,
    dataset_version_id: str,
    uploaded_coa_candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    entries = _candidate_entries(uploaded_coa_candidate) or [
        dict(entry) for entry in DEFAULT_CHINA_COA_ENTRIES
    ]
    bundle_version = f"{DEFAULT_COA_BUNDLE_VERSION_PREFIX}:{organization_id}:{dataset_version_id}:1"
    projection_bundle_version = f"{bundle_version}:projection"
    return {
        "bundle_id": DEFAULT_COA_BUNDLE_ID,
        "bundle_version": bundle_version,
        "projection_bundle_version": projection_bundle_version,
        "org_id": organization_id,
        "dataset_version_id": dataset_version_id,
        "jurisdiction": "CN",
        "accounting_standard": "china_small_business_accounting_standards",
        "entries": entries,
        "mapping_rules": [dict(rule) for rule in DEFAULT_MAPPING_RULES],
        "source_refs": [
            {
                "source_type": "repo_default",
                "source_ref": "src/eos/real_company_coa_bootstrap.py",
                "dataset_version_id": dataset_version_id,
            }
        ],
    }


def _candidate_entries(candidate: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not candidate:
        return []
    entries = candidate.get("entries")
    if not isinstance(entries, list):
        return []
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        account_code = str(entry.get("account_code") or "").strip()
        if not account_code or not account_code.isascii():
            continue
        normalized.append(
            {
                "account_code": account_code,
                "account_name": str(entry.get("account_name") or account_code),
                "account_class": str(entry.get("account_class") or "expense"),
                "normal_balance": str(entry.get("normal_balance") or "debit"),
                "source_refs": entry.get("source_refs") if isinstance(entry.get("source_refs"), list) else [],
            }
        )
    return normalized


def _replay_event_id(organization_id: str, dataset_version_id: str) -> str:
    return f"real_company_coa:{organization_id}:{dataset_version_id}"
