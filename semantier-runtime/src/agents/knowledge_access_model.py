from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

_TIERS = ("T1", "T2", "T3", "T4", "T5", "T6")
_UI_ACTIONS = ("view", "propose", "review")

_ROLE_PRECEDENCE = {
    "owner": 3,
    "admin": 2,
    "member": 1,
}

_BUNDLE_LABELS = tuple(_ROLE_PRECEDENCE.keys())

# Phase-1 UI projection defaults. These are bundle presets, not the canonical
# authority model. Runtime responses derive from these presets into a fixed
# contract shape until first-class grants are implemented.
_BUNDLE_UI_PROJECTION: dict[str, dict[str, dict[str, str]]] = {
    "owner": {
        "T1": {"view": "allow", "propose": "deny", "review": "deny"},
        "T2": {"view": "allow", "propose": "deny", "review": "deny"},
        "T3": {"view": "allow", "propose": "allow_with_review", "review": "allow"},
        "T4": {"view": "allow", "propose": "allow_with_review", "review": "allow"},
        "T5": {"view": "allow", "propose": "allow", "review": "allow"},
        "T6": {"view": "allow", "propose": "allow", "review": "deny"},
    },
    "admin": {
        "T1": {"view": "allow", "propose": "deny", "review": "deny"},
        "T2": {"view": "allow", "propose": "deny", "review": "deny"},
        "T3": {"view": "allow", "propose": "allow_with_review", "review": "deny"},
        "T4": {"view": "allow", "propose": "allow_with_review", "review": "allow"},
        "T5": {"view": "allow", "propose": "allow", "review": "allow"},
        "T6": {"view": "allow", "propose": "allow", "review": "deny"},
    },
    "member": {
        "T1": {"view": "allow", "propose": "deny", "review": "deny"},
        "T2": {"view": "allow", "propose": "deny", "review": "deny"},
        "T3": {"view": "allow", "propose": "deny", "review": "deny"},
        "T4": {"view": "allow", "propose": "deny", "review": "deny"},
        "T5": {"view": "allow", "propose": "allow", "review": "deny"},
        "T6": {"view": "allow", "propose": "allow", "review": "deny"},
    },
}

_BUNDLE_CAPABILITIES: dict[str, tuple[dict[str, Any], ...]] = {
    "owner": (
        {
            "capability": "context.read",
            "decision": "allow",
            "scope": {
                "semantic_tier_ceiling": "T6",
                "authority_domains": ["management", "accounting", "compliance", "internal_control"],
                "resource_classes": ["fact_record", "contextual_memory", "active_knowledge_artifact"],
            },
            "governance_conditions": {
                "review_required": False,
                "approver_roles": [],
                "validator_required": False,
                "replay_pin_required": False,
                "activation_required": False,
            },
        },
        {
            "capability": "knowledge.propose",
            "decision": "allow_with_review",
            "scope": {
                "semantic_tier_ceiling": "T4",
                "authority_domains": ["management", "accounting", "internal_control"],
                "resource_classes": ["interpretation_candidate", "policy_candidate", "mapping_candidate"],
            },
            "governance_conditions": {
                "review_required": True,
                "approver_roles": ["policy_owner", "governance_chair"],
                "validator_required": False,
                "replay_pin_required": False,
                "activation_required": True,
            },
        },
        {
            "capability": "knowledge.activate_t5_user",
            "decision": "allow",
            "scope": {
                "semantic_tier_ceiling": "T5",
                "authority_domains": ["management"],
                "resource_classes": ["active_knowledge_artifact"],
            },
            "governance_conditions": {
                "review_required": False,
                "approver_roles": [],
                "validator_required": False,
                "replay_pin_required": False,
                "activation_required": True,
            },
        },
        {
            "capability": "knowledge.activate_t5_org",
            "decision": "allow_with_review",
            "scope": {
                "semantic_tier_ceiling": "T5",
                "authority_domains": ["management", "internal_control"],
                "resource_classes": ["active_knowledge_artifact"],
            },
            "governance_conditions": {
                "review_required": True,
                "approver_roles": ["scoped_owner"],
                "validator_required": False,
                "replay_pin_required": False,
                "activation_required": True,
            },
        },
        {
            "capability": "knowledge.review_t4",
            "decision": "allow",
            "scope": {
                "semantic_tier_ceiling": "T4",
                "authority_domains": ["management", "internal_control"],
                "resource_classes": ["policy_candidate"],
            },
            "governance_conditions": {
                "review_required": True,
                "approver_roles": ["policy_owner", "governance_chair"],
                "validator_required": False,
                "replay_pin_required": True,
                "activation_required": True,
            },
        },
        {
            "capability": "execution.run_scoped",
            "decision": "allow_with_review",
            "scope": {
                "semantic_tier_ceiling": "T5",
                "authority_domains": ["accounting", "management"],
                "workflow_set": ["ap", "expense", "close"],
                "resource_classes": ["execution_intent"],
            },
            "governance_conditions": {
                "review_required": True,
                "approver_roles": ["scoped_owner"],
                "validator_required": True,
                "replay_pin_required": True,
                "activation_required": False,
            },
        },
        {
            "capability": "validation.attest",
            "decision": "allow_with_review",
            "scope": {
                "semantic_tier_ceiling": "T4",
                "authority_domains": ["accounting", "internal_control"],
                "resource_classes": ["validation_attestation"],
            },
            "governance_conditions": {
                "review_required": True,
                "approver_roles": ["validator"],
                "validator_required": True,
                "replay_pin_required": True,
                "activation_required": False,
            },
        },
    ),
    "admin": (
        {
            "capability": "context.read",
            "decision": "allow",
            "scope": {
                "semantic_tier_ceiling": "T6",
                "authority_domains": ["management", "accounting", "internal_control"],
                "resource_classes": ["fact_record", "contextual_memory", "active_knowledge_artifact"],
            },
            "governance_conditions": {
                "review_required": False,
                "approver_roles": [],
                "validator_required": False,
                "replay_pin_required": False,
                "activation_required": False,
            },
        },
        {
            "capability": "knowledge.propose",
            "decision": "allow_with_review",
            "scope": {
                "semantic_tier_ceiling": "T4",
                "authority_domains": ["management", "internal_control"],
                "resource_classes": ["interpretation_candidate", "policy_candidate"],
            },
            "governance_conditions": {
                "review_required": True,
                "approver_roles": ["policy_owner", "governance_chair"],
                "validator_required": False,
                "replay_pin_required": False,
                "activation_required": True,
            },
        },
        {
            "capability": "knowledge.activate_t5_user",
            "decision": "allow",
            "scope": {
                "semantic_tier_ceiling": "T5",
                "authority_domains": ["management"],
                "resource_classes": ["active_knowledge_artifact"],
            },
            "governance_conditions": {
                "review_required": False,
                "approver_roles": [],
                "validator_required": False,
                "replay_pin_required": False,
                "activation_required": True,
            },
        },
        {
            "capability": "knowledge.activate_t5_org",
            "decision": "allow_with_review",
            "scope": {
                "semantic_tier_ceiling": "T5",
                "authority_domains": ["management"],
                "resource_classes": ["active_knowledge_artifact"],
            },
            "governance_conditions": {
                "review_required": True,
                "approver_roles": ["scoped_owner"],
                "validator_required": False,
                "replay_pin_required": False,
                "activation_required": True,
            },
        },
        {
            "capability": "knowledge.review_t4",
            "decision": "allow",
            "scope": {
                "semantic_tier_ceiling": "T4",
                "authority_domains": ["management", "internal_control"],
                "resource_classes": ["policy_candidate"],
            },
            "governance_conditions": {
                "review_required": True,
                "approver_roles": ["policy_owner", "governance_chair"],
                "validator_required": False,
                "replay_pin_required": True,
                "activation_required": True,
            },
        },
    ),
    "member": (
        {
            "capability": "context.read",
            "decision": "allow",
            "scope": {
                "semantic_tier_ceiling": "T6",
                "authority_domains": ["management", "accounting"],
                "resource_classes": ["fact_record", "contextual_memory", "active_knowledge_artifact"],
            },
            "governance_conditions": {
                "review_required": False,
                "approver_roles": [],
                "validator_required": False,
                "replay_pin_required": False,
                "activation_required": False,
            },
        },
        {
            "capability": "knowledge.propose",
            "decision": "allow",
            "scope": {
                "semantic_tier_ceiling": "T5",
                "authority_domains": ["management"],
                "resource_classes": ["interpretation_candidate"],
            },
            "governance_conditions": {
                "review_required": False,
                "approver_roles": [],
                "validator_required": False,
                "replay_pin_required": False,
                "activation_required": False,
            },
        },
        {
            "capability": "knowledge.activate_t5_user",
            "decision": "allow_with_review",
            "scope": {
                "semantic_tier_ceiling": "T5",
                "authority_domains": ["management"],
                "resource_classes": ["active_knowledge_artifact"],
            },
            "governance_conditions": {
                "review_required": True,
                "approver_roles": ["scoped_owner"],
                "validator_required": False,
                "replay_pin_required": False,
                "activation_required": True,
            },
        },
    ),
}


def normalize_role(role: str | None, *, can_change_settings: bool = False) -> str:
    if isinstance(role, str):
        normalized = role.strip().lower()
        if normalized in _BUNDLE_UI_PROJECTION:
            return normalized
    if can_change_settings:
        return "admin"
    return "member"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _deep_copy_projection(projection: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        tier: {action: decision for action, decision in actions.items()}
        for tier, actions in projection.items()
    }


def _project_bundle_preview() -> dict[str, Any]:
    return {
        label: {
            "tiers": _deep_copy_projection(_BUNDLE_UI_PROJECTION[label]),
        }
        for label in _BUNDLE_LABELS
    }


def _capabilities_for_bundle(
    *,
    bundle_label: str,
    organization_id: str | None,
) -> list[dict[str, Any]]:
    org_id = (organization_id or "").strip()
    capabilities: list[dict[str, Any]] = []
    for entry in _BUNDLE_CAPABILITIES.get(bundle_label, ()):
        scope = dict(entry["scope"])
        scope.setdefault("team_ids", [])
        scope.setdefault("workflow_set", [])
        scope.setdefault("authority_domains", [])
        scope.setdefault("resource_classes", [])
        scope["organization_id"] = org_id or None
        capability_payload = {
            "capability": entry["capability"],
            "decision": entry["decision"],
            "scope": scope,
            "governance_conditions": dict(entry.get("governance_conditions") or {}),
            "source": {
                "bundle_labels": [bundle_label],
                "direct_grant_ids": [],
            },
        }
        capabilities.append(capability_payload)
    return capabilities


def _organization_display_source(*, organization_name: str | None) -> str:
    return "seed_default" if (organization_name or "").strip() else "seed_default"


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _claim_value(claims: list[Any], key: str) -> Any:
    for claim in claims:
        if isinstance(claim, dict) and claim.get("key") == key:
            return claim.get("value")
    return None


def _active_knowledge_artifact_payload(limit: int = 12) -> dict[str, Any]:
    try:
        from storage.kgl_store import KGLStore

        rows = KGLStore().list_artifacts(status="ACTIVE")
    except (sqlite3.Error, OSError, ValueError):
        rows = []

    artifacts: list[dict[str, Any]] = []
    seen_source_refs: set[str] = set()
    for row in rows:
        source_ref = str(row.get("source_ref") or "")
        if source_ref in seen_source_refs:
            continue
        seen_source_refs.add(source_ref)

        claims = _json_list(row.get("extracted_claims_json"))
        anchors = _json_list(row.get("source_anchors_json"))
        ambiguities = _json_list(row.get("ambiguities_json"))
        evidence_chain = [
            {
                "stage": "curation_upload",
                "ref": _claim_value(claims, "evidence.source_upload_ref") or source_ref,
                "hash": _claim_value(claims, "evidence.source_pdf_hash"),
            },
            {
                "stage": "normalized_artifact",
                "ref": _claim_value(claims, "evidence.normalized_artifact_ref"),
                "hash": _claim_value(claims, "evidence.normalized_artifact_hash"),
            },
            {
                "stage": "governed_kgl_artifact",
                "ref": row.get("artifact_id"),
                "hash": row.get("source_hash"),
            },
            {
                "stage": "active_authority_record",
                "ref": row.get("artifact_id"),
                "hash": row.get("content_hash"),
            },
        ]
        artifacts.append(
            {
                "artifact_id": row.get("artifact_id"),
                "source_ref": source_ref,
                "semantic_tier": row.get("semantic_tier"),
                "authority_domain": row.get("authority_domain"),
                "tag_authority_level": row.get("tag_authority_level"),
                "source_type": row.get("source_type"),
                "authority_origin": row.get("authority_origin"),
                "ingestion_status": row.get("ingestion_status"),
                "effective_from": row.get("effective_from"),
                "source_version": row.get("source_version"),
                "extraction_method": row.get("extraction_method"),
                "curator": row.get("curator"),
                "claim_count": len(claims),
                "anchors": anchors[:3],
                "ambiguities": ambiguities[:3],
                "evidence_chain": evidence_chain,
            }
        )
        if len(artifacts) >= limit:
            break

    return {
        "items": artifacts,
        "count": len(artifacts),
        "source_surface": "knowledge_artifacts",
        "lifecycle_contract": "docs/canonical/institutional-lifecycle-runtime-spec.md#8-evidence-continuity-and-deterministic-replay",
    }


def knowledge_tier_access_payload(
    *,
    role: str | None,
    can_change_settings: bool = False,
    user_id: str | None = None,
    organization_id: str | None = None,
    organization_name: str | None = None,
    workspace_id: str | None = None,
    membership_status: str | None = None,
) -> dict[str, Any]:
    bundle_label = normalize_role(role, can_change_settings=can_change_settings)
    assigned_bundles = [bundle_label]
    ui_projection = {
        "tiers": _deep_copy_projection(_BUNDLE_UI_PROJECTION[bundle_label]),
        "action_mapping": {
            "view": "can_read_context",
            "propose": "can_initiate_path",
            "review": "can_participate_in_required_gate",
        },
        "decision_legend": {
            "allow": "principal may complete within current contract scope",
            "allow_with_review": "principal may initiate but cannot complete alone",
            "deny": "no eligible path from current contract",
        },
    }
    return {
        "schema_version": "phase1.v1",
        "generated_at": _utc_timestamp(),
        "principal": {
            "user_id": (user_id or "").strip(),
            "organization_id": (organization_id or "").strip(),
            "assigned_bundles": assigned_bundles,
            "membership_status": (membership_status or "unassigned").strip() or "unassigned",
        },
        "organization_context": {
            "organization_id": (organization_id or "").strip(),
            "official_display_name": (organization_name or "").strip(),
            "display_name_source": _organization_display_source(organization_name=organization_name),
            "workspace_id": (workspace_id or "").strip() or None,
        },
        "effective_capabilities": _capabilities_for_bundle(
            bundle_label=bundle_label,
            organization_id=organization_id,
        ),
        "ui_projection": ui_projection,
        "bundle_preview": _project_bundle_preview(),
        "active_knowledge_artifacts": _active_knowledge_artifact_payload(),
        "evaluation_metadata": {
            "resolver_version": "phase1.v1",
            "policy_mode": "static_phase1_bundle_projection",
            "derived_from": {
                "assigned_bundle_labels": assigned_bundles,
                "direct_grants_present": False,
            },
            "warnings": ["direct grants not yet enabled in this environment"],
        },
    }
