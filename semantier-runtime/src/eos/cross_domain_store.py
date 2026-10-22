"""Cross-domain trust store for v8.2.

Provides append-only helpers for:
  - cross_domain_policies
  - cross_domain_resolver_contracts
  - cross_domain_order_results
  - cross_domain_replay_bindings
  - cross_domain_governance_decisions

All writes are append-only. No UPDATE or DELETE operations are exposed.
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash as compute_hash
from eos.db import get_conn, now_iso


def _serialize_json_fields(record: dict, json_fields: list[str]) -> dict:
    out = dict(record)
    for field in json_fields:
        if field in out and not isinstance(out[field], str) and out[field] is not None:
            out[field] = json.dumps(out[field], ensure_ascii=False)
    return out


def _deserialize_json_fields(row: dict, json_fields: list[str]) -> dict:
    out = dict(row)
    for field in json_fields:
        if field in out and out[field] is not None:
            out[field] = json.loads(out[field])
    return out


def _insert(table: str, record: dict) -> dict:
    keys = list(record.keys())
    placeholders = ", ".join([hosted.sql_query("?", '%s')] * len(keys))
    columns = ", ".join(keys)
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            [record[k] for k in keys],
        )
    return record


def insert_cross_domain_policy(record: dict) -> dict:
    row = dict(record)
    row.setdefault("policy_id", f"xpol_{uuid4().hex}")
    row.setdefault("created_at", now_iso())
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))

    row = _serialize_json_fields(
        row,
        [
            "domain_set_json",
            "compatibility_rules_json",
            "action_policies_json",
            "disclosure_requirements_json",
            "limit_policies_json",
        ],
    )
    _insert("cross_domain_policies", row)
    return _deserialize_json_fields(
        row,
        [
            "domain_set_json",
            "compatibility_rules_json",
            "action_policies_json",
            "disclosure_requirements_json",
            "limit_policies_json",
        ],
    )


def insert_cross_domain_resolver_contract(record: dict) -> dict:
    row = dict(record)
    row.setdefault("resolver_contract_id", f"xrc_{uuid4().hex}")
    row.setdefault("created_at", now_iso())
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))

    row = _serialize_json_fields(
        row,
        ["semantics_json", "precedence_bindings_json", "deterministic_rules_json"],
    )
    _insert("cross_domain_resolver_contracts", row)
    return _deserialize_json_fields(
        row,
        ["semantics_json", "precedence_bindings_json", "deterministic_rules_json"],
    )


def insert_cross_domain_order_result(record: dict) -> dict:
    row = dict(record)
    row.setdefault("xorder_id", f"xord_{uuid4().hex}")
    row.setdefault("created_at", now_iso())
    row.setdefault("disclosure_required", 0)
    row.setdefault("blocked_action", 0)
    row.setdefault("blocked_export", 0)
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))

    row = _serialize_json_fields(row, ["limits_json", "reason_codes_json"])
    _insert("cross_domain_order_results", row)
    return _deserialize_json_fields(row, ["limits_json", "reason_codes_json"])


def insert_cross_domain_replay_binding(record: dict) -> dict:
    row = dict(record)
    row.setdefault("cross_replay_id", f"xreplay_{uuid4().hex}")
    row.setdefault("created_at", now_iso())
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))
    _insert("cross_domain_replay_bindings", row)
    return row


def insert_cross_domain_governance_decision(record: dict) -> dict:
    row = dict(record)
    row.setdefault("cross_decision_id", f"xgov_{uuid4().hex}")
    row.setdefault("created_at", now_iso())
    if "content_hash" not in row:
        payload = {k: v for k, v in row.items() if k not in ("content_hash", "created_at")}
        row["content_hash"] = compute_hash(canonical_json(payload))

    row = _serialize_json_fields(row, ["approved_limits_json"])
    _insert("cross_domain_governance_decisions", row)
    return _deserialize_json_fields(row, ["approved_limits_json"])


def get_cross_domain_order_result(xorder_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("SELECT * FROM cross_domain_order_results WHERE xorder_id = ?", 'SELECT * FROM cross_domain_order_results WHERE xorder_id = %s'),
            (xorder_id,),
        ).fetchone()
    if row is None:
        return None
    return _deserialize_json_fields(dict(row), ["limits_json", "reason_codes_json"])
