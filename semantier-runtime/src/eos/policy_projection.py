"""Projection engine for governed COA projection bundles.

Implements Π_v(rea_event, projection_context) → ledger_projections.
Supports legacy `codes` bundles and governed `entries` + `mapping_rules`.
"""

from __future__ import annotations

from storage import hosted_postgres as hosted

from typing import Any, Dict, List, Optional
from uuid import uuid4

from eos.db import get_conn, now_iso

_VALID_ACCOUNT_CLASSES = {"asset", "liability", "equity", "revenue", "expense"}
_VALID_NORMAL_BALANCES = {"debit", "credit"}


def _normalize_projection_views(projection_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    views = projection_bundle.get("views")
    if isinstance(views, dict) and views:
        normalized = []
        for view_name, config in views.items():
            config = config or {}
            normalized.append(
                {
                    "name": view_name,
                    "projection_scope": config.get("projection_scope", view_name),
                    "bundle": config,
                }
            )
        return normalized

    return [
        {
            "name": projection_bundle.get("name", "default"),
            "projection_scope": projection_bundle.get(
                "projection_scope",
                projection_bundle.get("bundle_id", "default"),
            ),
            "bundle": projection_bundle,
        }
    ]


def _bundle_status(bundle: Dict[str, Any]) -> str:
    return bundle.get("governance_state") or bundle.get("status") or "ACTIVE"


def _normalize_entries(bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    entries = bundle.get("entries")
    if isinstance(entries, list) and entries:
        normalized = {}
        for entry in entries:
            code = entry.get("account_code")
            if code:
                normalized[code] = entry
        return normalized

    legacy_codes = bundle.get("codes", {})
    if isinstance(legacy_codes, dict):
        normalized = {}
        for resource_type, account_code in legacy_codes.items():
            normalized[str(account_code)] = {
                "account_code": str(account_code),
                "account_name": str(account_code),
                "account_class": _infer_account_class(resource_type),
                "normal_balance": _infer_normal_balance(resource_type),
                "governance_state": "ACTIVE",
            }
        return normalized

    return {}


def _infer_account_class(resource_type: str) -> str:
    if resource_type in {"cash", "asset", "receivable"}:
        return "asset"
    if resource_type in {"liability", "payable", "tax"}:
        return "liability"
    if resource_type in {"revenue", "income"}:
        return "revenue"
    return "expense"


def _infer_normal_balance(resource_type: str) -> str:
    account_class = _infer_account_class(resource_type)
    return "credit" if account_class in {"liability", "equity", "revenue"} else "debit"


def _validate_entry(entry: Dict[str, Any]) -> Optional[str]:
    if entry.get("account_class") not in _VALID_ACCOUNT_CLASSES:
        return f"invalid_account_class:{entry.get('account_code')}"
    if entry.get("normal_balance") not in _VALID_NORMAL_BALANCES:
        return f"invalid_normal_balance:{entry.get('account_code')}"
    if _bundle_status(entry) != "ACTIVE":
        return f"inactive_coa_entry:{entry.get('account_code')}"
    if entry.get("source_type") == "agent":
        return f"llm_defined_coa_entry:{entry.get('account_code')}"
    return None


def _validate_bundle_metadata(bundle: Dict[str, Any]) -> Optional[str]:
    if bundle.get("source_type") == "agent":
        return "llm_defined_coa_bundle"
    if _bundle_status(bundle) != "ACTIVE":
        return "inactive_coa_bundle"
    if bundle.get("authority_domain") not in (None, "accounting"):
        return "invalid_coa_bundle_authority_domain"
    if bundle.get("tag_authority_level") not in (None, "accounting"):
        return "invalid_coa_bundle_authority_level"
    return None


def _claim_index(semantic_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {claim.get("key"): claim.get("value") for claim in semantic_claims if claim.get("key")}


def _rule_matches(
    rule: Dict[str, Any],
    claim_values: Dict[str, Any],
    projection_context: Dict[str, Any],
) -> bool:
    if _bundle_status(rule) != "ACTIVE":
        return False
    if rule.get("source_type") == "agent":
        return False

    applies_to = rule.get("applies_to", {})
    if applies_to.get("authority_domain") not in (None, "accounting"):
        return False

    when = rule.get("when", {})
    expected_event = when.get("rea_event_type")
    if expected_event and projection_context.get("rea_event_type") != expected_event:
        return False

    expected_tags = when.get("tags", {})
    for key, expected_value in expected_tags.items():
        if claim_values.get(key) != expected_value:
            return False
    return True


def _legacy_rule_from_resources(resources: List[Dict[str, Any]], bundle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    codes = bundle.get("codes", {})
    if not isinstance(codes, dict):
        return None
    if len(resources) < 2:
        return None
    debit_resource = next((resource for resource in resources if float(resource.get("delta", 0)) > 0), None)
    credit_resource = next((resource for resource in resources if float(resource.get("delta", 0)) < 0), None)
    if not debit_resource or not credit_resource:
        return None
    debit_code = codes.get(debit_resource.get("type"))
    credit_code = codes.get(credit_resource.get("type"))
    if not debit_code or not credit_code:
        return None
    return {
        "rule_id": f"legacy:{bundle.get('bundle_id', bundle.get('name', 'default'))}",
        "project": {
            "debit": {"account_code": str(debit_code), "dimensions": []},
            "credit": {"account_code": str(credit_code), "dimensions": []},
        },
        "requires": {
            "account_codes_exist_in_bundle": True,
            "debit_credit_balanced": True,
            "normal_balance_compatible": True,
        },
        "governance_state": "ACTIVE",
    }


def _select_mapping_rule(
    resources: List[Dict[str, Any]],
    semantic_claims: List[Dict[str, Any]],
    projection_context: Dict[str, Any],
    bundle: Dict[str, Any],
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    mapping_rules = bundle.get("mapping_rules", [])
    if isinstance(mapping_rules, list) and mapping_rules:
        claim_values = _claim_index(semantic_claims)
        for rule in mapping_rules:
            if _rule_matches(rule, claim_values, projection_context):
                return rule, None
        return None, "no_mapping_rule_exists"
    return None, "no_mapping_rule_exists"


def _legacy_resource_rows(
    rea_id: str,
    resources: List[Dict[str, Any]],
    bundle: Dict[str, Any],
    projection_scope: str,
    projection_context_hash: str,
    projection_pins: Dict[str, Any],
    timestamp: str,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    rows: List[Dict[str, Any]] = []
    codes = bundle.get("codes", {})
    for resource in resources:
        resource_type = resource.get("type")
        account_code = codes.get(resource_type)
        if not account_code:
            return [], f"no_coa_code_for_resource_type_{resource_type}:scope={projection_scope}"
        amount = float(resource.get("delta", 0))
        rows.append(
            {
                "ledger_id": f"ledg_{uuid4().hex}",
                "rea_id": rea_id,
                "coa_code": str(account_code),
                "amount": amount,
                "currency": resource.get("currency", "CNY"),
                "debit": max(0.0, amount),
                "credit": max(0.0, -amount),
                "projection_status": "PENDING",
                "projection_context_hash": projection_context_hash,
                "projection_scope": projection_scope,
                "created_at": timestamp,
                "projected_at": None,
                **projection_pins,
            }
        )
    return rows, None


def _normal_balance_compatible(entry: Dict[str, Any], side: str) -> bool:
    del side
    account_class = entry.get("account_class")
    normal_balance = entry.get("normal_balance")
    if account_class in {"asset", "expense"}:
        return normal_balance == "debit"
    if account_class in {"liability", "equity", "revenue"}:
        return normal_balance == "credit"
    return False


def _required_dimensions_missing(
    dimensions: List[str],
    entry: Dict[str, Any],
    projection_context: Dict[str, Any],
) -> List[str]:
    available_dimensions = projection_context.get("dimensions", {}) or {}
    waived_dimensions = set(projection_context.get("waived_dimensions", []) or [])
    entry_dimensions = set(entry.get("auxiliary_dimensions", []) or [])
    required_dimensions = set(dimensions or []) | entry_dimensions
    return sorted(
        dimension
        for dimension in required_dimensions
        if dimension not in available_dimensions and dimension not in waived_dimensions
    )


def _build_projection_pins(pin_versions: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pin_versions = pin_versions or {}
    return {
        "projection_bundle_version": pin_versions.get("projection_bundle_version"),
        "coa_bundle_version": pin_versions.get("coa_bundle_version"),
        "knowledge_context_version": pin_versions.get("knowledge_context_version"),
        "constraint_bundle_version": pin_versions.get("constraint_bundle_version"),
        "tag_ontology_version": pin_versions.get("tag_ontology_version"),
        "precedence_graph_version": pin_versions.get("precedence_graph_version"),
        "resolution_rules_version": pin_versions.get("resolution_rules_version"),
    }


# Patent evidence C003: REA-to-ledger projection engine that maps admitted REA
# facts to domain-specific journal-entry projections under a pinned projection
# bundle and semantic claims, referenced in the admissibility-governance method
# claims.

def project_rea_to_ledger(
    rea_id: str,
    rea_claim: Dict[str, Any],
    active_coa: Dict[str, Any],
    projection_context_hash: str = "",
    semantic_claims: Optional[List[Dict[str, Any]]] = None,
    projection_context: Optional[Dict[str, Any]] = None,
    pin_versions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Project an admitted REA claim into ledger_projections rows."""
    resources = rea_claim.get("resources", [])
    if not resources:
        return {"status": "EXCEPTION", "reason": "no_resources_in_claim", "projections": []}

    projection_context = projection_context or {}
    semantic_claims = semantic_claims or []
    bundle_error = _validate_bundle_metadata(active_coa)
    if bundle_error:
        return {"status": "EXCEPTION", "reason": bundle_error, "projections": []}

    projections = []
    timestamp = now_iso()
    projection_views = _normalize_projection_views(active_coa)
    projection_pins = _build_projection_pins(pin_versions)

    for view in projection_views:
        view_bundle = dict(active_coa)
        view_bundle.update(view.get("bundle", {}))
        entries = _normalize_entries(view_bundle)
        for entry in entries.values():
            entry_error = _validate_entry(entry)
            if entry_error:
                return {"status": "EXCEPTION", "reason": entry_error, "projections": []}

        if not view_bundle.get("mapping_rules") and isinstance(view_bundle.get("codes"), dict):
            legacy_rows, legacy_error = _legacy_resource_rows(
                rea_id,
                resources,
                view_bundle,
                view["projection_scope"],
                projection_context_hash,
                projection_pins,
                timestamp,
            )
            if legacy_error:
                return {
                    "status": "EXCEPTION",
                    "reason": legacy_error,
                    "projections": [],
                }
            projections.extend(legacy_rows)
            continue

        rule, rule_error = _select_mapping_rule(
            resources,
            semantic_claims,
            {
                **projection_context,
                "projection_scope": view["projection_scope"],
            },
            view_bundle,
        )
        if rule_error:
            return {
                "status": "ESCALATE",
                "reason": rule_error,
                "projections": [],
                "projection_scope": view["projection_scope"],
            }
        if rule.get("source_type") == "agent":
            return {
                "status": "EXCEPTION",
                "reason": f"llm_defined_mapping_rule:{rule.get('rule_id')}",
                "projections": [],
            }

        rule_project = rule.get("project", {})
        line_specs = [("debit", rule_project.get("debit", {})), ("credit", rule_project.get("credit", {}))]
        totals = {"debit": 0.0, "credit": 0.0}
        for side, line_spec in line_specs:
            account_code = line_spec.get("account_code")
            entry = entries.get(str(account_code))
            if not entry:
                return {
                    "status": "REJECT",
                    "reason": f"account_code_missing_from_bundle:{account_code}",
                    "projections": [],
                }

            if not _normal_balance_compatible(entry, side):
                return {
                    "status": "REJECT",
                    "reason": f"normal_balance_incompatible:{account_code}:{side}",
                    "projections": [],
                }

            missing_dimensions = _required_dimensions_missing(
                line_spec.get("dimensions", []),
                entry,
                projection_context,
            )
            if missing_dimensions:
                severity = (view_bundle.get("dimension_policy") or {}).get("missing_dimension_severity", "ESCALATE")
                return {
                    "status": severity,
                    "reason": f"missing_auxiliary_dimensions:{account_code}:{','.join(missing_dimensions)}",
                    "projections": [],
                }

            amount = sum(
                max(0.0, float(resource.get("delta", 0))) if side == "debit"
                else max(0.0, -float(resource.get("delta", 0)))
                for resource in resources
            )
            totals[side] = amount

            projection_row = {
                "ledger_id": f"ledg_{uuid4().hex}",
                "rea_id": rea_id,
                "coa_code": str(account_code),
                "amount": amount if side == "debit" else -amount,
                "currency": projection_context.get("currency") or resources[0].get("currency", "CNY"),
                "debit": amount if side == "debit" else 0.0,
                "credit": amount if side == "credit" else 0.0,
                "projection_status": "PENDING",
                "projection_context_hash": projection_context_hash,
                "projection_scope": view["projection_scope"],
                "created_at": timestamp,
                "projected_at": None,
                **projection_pins,
            }
            projections.append(projection_row)

        if rule.get("requires", {}).get("debit_credit_balanced", True) and round(totals["debit"], 8) != round(totals["credit"], 8):
            return {"status": "REJECT", "reason": "debit_credit_imbalance", "projections": []}

    try:
        with get_conn() as conn:
            for proj in projections:
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO ledger_projections
                      (ledger_id, rea_id, coa_code, amount, currency,
                       debit, credit, projection_status, projection_context_hash,
                       projection_scope, projection_bundle_version, coa_bundle_version,
                       knowledge_context_version, constraint_bundle_version,
                       tag_ontology_version, precedence_graph_version,
                       resolution_rules_version, created_at, projected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO ledger_projections
                      (ledger_id, rea_id, coa_code, amount, currency,
                       debit, credit, projection_status, projection_context_hash,
                       projection_scope, projection_bundle_version, coa_bundle_version,
                       knowledge_context_version, constraint_bundle_version,
                       tag_ontology_version, precedence_graph_version,
                       resolution_rules_version, created_at, projected_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """),
                    (
                        proj["ledger_id"],
                        proj["rea_id"],
                        proj["coa_code"],
                        proj["amount"],
                        proj["currency"],
                        proj["debit"],
                        proj["credit"],
                        proj["projection_status"],
                        proj["projection_context_hash"],
                        proj["projection_scope"],
                        proj["projection_bundle_version"],
                        proj["coa_bundle_version"],
                        proj["knowledge_context_version"],
                        proj["constraint_bundle_version"],
                        proj["tag_ontology_version"],
                        proj["precedence_graph_version"],
                        proj["resolution_rules_version"],
                        proj["created_at"],
                        proj["projected_at"],
                    ),
                )
            conn.commit()
    except Exception as e:
        return {
            "status": "EXCEPTION",
            "reason": f"projection_write_failed: {str(e)}",
            "projections": [],
        }

    try:
        timestamp_update = now_iso()
        with get_conn() as conn:
            for proj in projections:
                conn.execute(
                    hosted.sql_query("""
                    UPDATE ledger_projections
                    SET projection_status = 'PROJECTED', projected_at = ?
                    WHERE ledger_id = ?
                    """, """
                    UPDATE ledger_projections
                    SET projection_status = 'PROJECTED', projected_at = %s
                    WHERE ledger_id = %s
                    """),
                    (timestamp_update, proj["ledger_id"]),
                )
                proj["projection_status"] = "PROJECTED"
                proj["projected_at"] = timestamp_update
            conn.commit()
    except Exception as e:
        return {
            "status": "EXCEPTION",
            "reason": f"projection_status_update_failed: {str(e)}",
            "projections": projections,
        }

    return {
        "status": "ALLOW",
        "projections": projections,
        "projection_views": [view["projection_scope"] for view in projection_views],
    }


def project_policy(ontology, context):
    allowed_entities = context.get("allowed_entities", [])
    constraints = context.get("constraints", {})

    rules = []
    for entity in allowed_entities:
        if entity not in ontology.get("entities", []):
            return {"status": "REJECT", "reason": "projection_outside_ontology"}
        rules.append({"entity": entity, "constraint": constraints})

    return {"status": "ALLOW", "rules": rules}


def validate_policy_against_ontology(policy, ontology):
    for rule in policy.get("rules", []):
        if rule.get("entity") not in ontology.get("entities", []):
            return {"status": "REJECT", "reason": "policy_outside_ontology"}
    return {"status": "ALLOW"}
