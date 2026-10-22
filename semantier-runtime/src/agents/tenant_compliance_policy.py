from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "policies" / "tenant_compliance_policy.v1.json"


def load_policy() -> dict[str, Any]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def validate_policy(policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if str(policy.get("schema_version") or "") != "tenant-compliance-policy.v1":
        errors.append("schema_version must be tenant-compliance-policy.v1")

    retention = policy.get("retention_classes")
    if not isinstance(retention, dict) or not retention:
        errors.append("retention_classes must be a non-empty object")
    else:
        for name, rule in retention.items():
            if not isinstance(rule, dict):
                errors.append(f"retention_classes.{name} must be an object")
                continue
            days = rule.get("min_retention_days")
            if not isinstance(days, int) or days <= 0:
                errors.append(f"retention_classes.{name}.min_retention_days must be positive int")
            append_only = rule.get("append_only")
            if not isinstance(append_only, bool):
                errors.append(f"retention_classes.{name}.append_only must be boolean")

    residency = policy.get("residency_rules")
    if not isinstance(residency, dict) or not residency:
        errors.append("residency_rules must be a non-empty object")
    else:
        for region, allowed in residency.items():
            if not isinstance(allowed, list) or not allowed or not all(isinstance(item, str) for item in allowed):
                errors.append(f"residency_rules.{region} must be non-empty list of strings")

    audit_surfaces = policy.get("tenant_admin_audit_surfaces")
    if not isinstance(audit_surfaces, list) or not audit_surfaces:
        errors.append("tenant_admin_audit_surfaces must be a non-empty array")
    else:
        for idx, surface in enumerate(audit_surfaces):
            if not isinstance(surface, dict):
                errors.append(f"tenant_admin_audit_surfaces[{idx}] must be object")
                continue
            for required in ("surface", "schema_version", "purpose"):
                if not str(surface.get(required) or "").strip():
                    errors.append(f"tenant_admin_audit_surfaces[{idx}].{required} is required")

    return errors


def retention_rule_for_artifact(policy: dict[str, Any], artifact_class: str) -> dict[str, Any] | None:
    retention = policy.get("retention_classes")
    if not isinstance(retention, dict):
        return None
    rule = retention.get(artifact_class)
    return dict(rule) if isinstance(rule, dict) else None


def is_storage_region_allowed(policy: dict[str, Any], *, tenant_region: str, storage_region: str) -> bool:
    rules = policy.get("residency_rules")
    if not isinstance(rules, dict):
        return False
    allowed = rules.get(tenant_region)
    if not isinstance(allowed, list):
        return False
    return storage_region in [str(item) for item in allowed]


def tenant_admin_audit_surfaces(policy: dict[str, Any]) -> list[dict[str, Any]]:
    surfaces = policy.get("tenant_admin_audit_surfaces")
    if not isinstance(surfaces, list):
        return []
    result: list[dict[str, Any]] = []
    for item in surfaces:
        if isinstance(item, dict):
            result.append(dict(item))
    return result
