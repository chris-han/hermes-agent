from __future__ import annotations

import json
import hashlib
import os
import re
import secrets
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from agents.auth_db import (
    delete_feishu_bot_config_for_user,
    delete_weixin_runtime_accounts_for_user,
    ensure_auth_db,
    get_feishu_bot_config_for_workspace,
    load_feishu_link_states,
    load_gateway_correlations,
    load_organization_events,
    load_organizations as load_organizations_db,
    load_users,
    load_weixin_runtime_accounts,
    load_weixin_login_states,
    save_feishu_link_states,
    save_feishu_bot_config,
    save_gateway_correlations,
    save_organization_events,
    save_organizations as save_organizations_db,
    save_users,
    insert_password_user,
    is_postgresql,
    identity_write_transaction,
    retain_snapshot,
    save_weixin_login_states,
)
from agents.runtime_memory_boundary import sanitize_user_memory_profile
from runtime_paths import (
    platform_runtime_root,
    workspace_runtime_home_path,
    workspace_root_path,
    workspace_sessions_root,
)

ARGON2_MEMORY_COST = 65536
ARGON2_TIME_COST = 3
ARGON2_PARALLELISM = 4
WEIXIN_LOGIN_STATE_TTL_SECONDS = 600
FEISHU_LINK_STATE_TTL_SECONDS = 600
ORG_MEMBERSHIP_STATUSES = frozenset({"unassigned", "pending", "active", "suspended", "revoked"})
ORG_MEMBER_ROLES = frozenset({"owner", "admin", "member", "governance_security_reviewer"})
ORG_DATASET_TYPES = frozenset({"DEMO", "REAL", "DEVELOPMENT", "DEFAULT_REALISTIC_SAMPLE"})
T6_MATERIALIZATION_MODES = frozenset({"AUTO", "APPROVAL"})

_WRITE_LOCK = threading.RLock()
_PASSWORD_HASHER = PasswordHasher(
    memory_cost=ARGON2_MEMORY_COST,
    time_cost=ARGON2_TIME_COST,
    parallelism=ARGON2_PARALLELISM,
)


def _auth_root() -> Path:
    path = platform_runtime_root() / "auth"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _env_workspace_root_path(workspace_id: str) -> Path | None:
    raw_local_state_dir = os.environ.get("SEMANTIER_LOCAL_STATE_DIR")
    if not raw_local_state_dir:
        return None
    normalized = str(workspace_id or "").strip()
    if not normalized or normalized != Path(normalized).name:
        raise ValueError("workspace_id must be a single path segment")
    return Path(raw_local_state_dir).expanduser().resolve().parent / "workspaces" / normalized


def resolve_workspace_paths(workspace_id: str) -> tuple[Path, Path]:
    """Resolve governed workspace locations without provisioning or config writes."""
    env_workspace_root = _env_workspace_root_path(workspace_id)
    if env_workspace_root is None:
        workspace_root = workspace_root_path(workspace_id)
        runtime_home = workspace_runtime_home_path(workspace_id)
    else:
        workspace_root = env_workspace_root
        runtime_home = env_workspace_root

    return workspace_root, runtime_home


def ensure_workspace_paths(workspace_id: str) -> tuple[Path, Path]:
    workspace_root, runtime_home = resolve_workspace_paths(workspace_id)
    sessions_root = workspace_root / "sessions"

    from agents.launcher import bootstrap_workspace_runtime_home

    workspace_root.mkdir(parents=True, exist_ok=True)
    bootstrap_workspace_runtime_home(
        runtime_home,
        shared_skills_external_dir=platform_runtime_root() / "skills",
        shared_plugins_external_dir=platform_runtime_root() / "plugins",
        workspace_id=workspace_id,
    )
    (workspace_root / "swarm" / "runs").mkdir(parents=True, exist_ok=True)
    sessions_root.mkdir(exist_ok=True)
    (runtime_home / "memories").mkdir(exist_ok=True)
    (runtime_home / "skills").mkdir(exist_ok=True)
    (runtime_home / "profiles").mkdir(exist_ok=True)
    (runtime_home / "cron").mkdir(exist_ok=True)
    (runtime_home / "logs").mkdir(exist_ok=True)
    (runtime_home / "home").mkdir(exist_ok=True)
    sanitize_user_memory_profile(runtime_home)

    return workspace_root, runtime_home


def _users_path() -> Path:
    return _auth_root() / "users.json"


def _gateway_correlations_path() -> Path:
    return _auth_root() / "gateway-correlations.json"


def _weixin_login_states_path() -> Path:
    return _auth_root() / "weixin-login-states.json"


def _feishu_link_states_path() -> Path:
    return _auth_root() / "feishu-link-states.json"


def _organizations_path() -> Path:
    return _auth_root() / "organizations.json"


def _organization_events_path() -> Path:
    return _auth_root() / "organization-events.json"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _legacy_auth_json_loader(kind: str) -> Any:
    if kind == "users":
        return _load_json(_users_path(), {})
    if kind == "gateway_correlations":
        return _load_json(_gateway_correlations_path(), [])
    if kind == "organizations":
        return _load_json(_organizations_path(), {})
    if kind == "organization_events":
        return _load_json(_organization_events_path(), [])
    if kind == "weixin_login_states":
        return _load_json(_weixin_login_states_path(), {})
    if kind == "feishu_link_states":
        return _load_json(_feishu_link_states_path(), {})
    return None


def _ensure_auth_store() -> None:
    ensure_auth_db(json_loader=_legacy_auth_json_loader)


def _slugify(raw: str) -> str:
    value = "".join(ch.lower() if ch.isalnum() else "-" for ch in raw.strip())
    compact = "-".join(part for part in value.split("-") if part)
    return compact[:32] or "user"


def get_user_record(user_id: str) -> dict[str, Any] | None:
    _ensure_auth_store()
    users = load_users()
    if not isinstance(users, dict):
        return None
    record = users.get(user_id)
    return record if isinstance(record, dict) else None


def list_user_records() -> list[dict[str, Any]]:
    _ensure_auth_store()
    users = load_users()
    if not isinstance(users, dict):
        return []
    return [record for record in users.values() if isinstance(record, dict)]


def get_user_record_by_workspace_id(workspace_id: str) -> dict[str, Any] | None:
    normalized = _normalize_optional_text(workspace_id)
    if normalized is None:
        return None
    for user in list_user_records():
        if str(user.get("workspace_id") or "").strip() == normalized:
            return user
    return None


def has_password_login_enabled() -> bool:
    # Password registration is available even before the first user exists.
    return True


def _slug_login(raw: str) -> str:
    value = _slugify(raw)
    return value[:48] or "user"


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _slug_org_id(raw: str) -> str:
    value = "".join(ch.lower() if ch.isalnum() else "-" for ch in raw.strip())
    compact = "-".join(part for part in value.split("-") if part)
    return compact[:64] or "org"


def _normalize_membership_status(value: str | None) -> str:
    normalized = _normalize_optional_text(value) or "unassigned"
    return normalized if normalized in ORG_MEMBERSHIP_STATUSES else "unassigned"


def _normalize_member_role(value: str | None) -> str | None:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    return normalized if normalized in ORG_MEMBER_ROLES else "member"


def _normalize_dataset_type(value: str | None) -> str | None:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    upper = normalized.upper()
    return upper if upper in ORG_DATASET_TYPES else None


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        trimmed = item.strip()
        if not trimmed:
            continue
        normalized.append(trimmed)
    deduped = sorted({item for item in normalized})
    return deduped


def _normalize_confidence_thresholds(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        threshold_key = key.strip()
        if not threshold_key:
            continue
        try:
            threshold_value = float(raw)
        except (TypeError, ValueError):
            continue
        normalized[threshold_key] = threshold_value
    return dict(sorted(normalized.items(), key=lambda item: item[0]))


def _normalize_escalation_rules(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        rule_key = key.strip()
        if not rule_key:
            continue
        normalized[rule_key] = raw
    return dict(sorted(normalized.items(), key=lambda item: item[0]))


def _default_t6_materialization_policy() -> dict[str, Any]:
    return {
        "default_mode": "AUTO",
        "auto_allowed_claim_classes": [],
        "approval_required_claim_classes": [],
        "confidence_thresholds": {},
        "escalation_rules": {},
    }


def _materialization_policy_hash(policy: dict[str, Any]) -> str:
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_t6_materialization_policy(
    policy: dict[str, Any] | None,
    *,
    existing_policy: dict[str, Any] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    updates = policy if isinstance(policy, dict) else {}
    base = dict(existing_policy) if isinstance(existing_policy, dict) else _default_t6_materialization_policy()
    requested_mode = _normalize_optional_text(str(updates.get("default_mode") or ""))
    if requested_mode is not None:
        requested_mode = requested_mode.upper()
    if requested_mode in {"AUTOAPPROVE", "AUTO_APPROVE"}:
        requested_mode = "AUTO"
    default_mode = (
        requested_mode
        if requested_mode in T6_MATERIALIZATION_MODES
        else str(base.get("default_mode") or "AUTO")
    )
    if default_mode not in T6_MATERIALIZATION_MODES:
        default_mode = "AUTO"

    auto_allowed_claim_classes = (
        _normalize_string_list(updates.get("auto_allowed_claim_classes"))
        if "auto_allowed_claim_classes" in updates
        else _normalize_string_list(base.get("auto_allowed_claim_classes"))
    )
    approval_required_claim_classes = (
        _normalize_string_list(updates.get("approval_required_claim_classes"))
        if "approval_required_claim_classes" in updates
        else _normalize_string_list(base.get("approval_required_claim_classes"))
    )
    confidence_thresholds = (
        _normalize_confidence_thresholds(updates.get("confidence_thresholds"))
        if "confidence_thresholds" in updates
        else _normalize_confidence_thresholds(base.get("confidence_thresholds"))
    )
    escalation_rules = (
        _normalize_escalation_rules(updates.get("escalation_rules"))
        if "escalation_rules" in updates
        else _normalize_escalation_rules(base.get("escalation_rules"))
    )

    normalized_without_meta = {
        "default_mode": default_mode,
        "auto_allowed_claim_classes": auto_allowed_claim_classes,
        "approval_required_claim_classes": approval_required_claim_classes,
        "confidence_thresholds": confidence_thresholds,
        "escalation_rules": escalation_rules,
    }
    normalized = dict(normalized_without_meta)
    normalized["policy_version_hash"] = _materialization_policy_hash(normalized_without_meta)
    normalized["updated_at"] = updated_at or _normalize_optional_text(str(base.get("updated_at") or "")) or _now_iso()
    return normalized


def _load_seeded_organization_profiles() -> tuple[dict[str, Any], ...]:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from bootstrap.industry_simulator import list_seeded_organization_profiles

    return tuple(dict(profile) for profile in list_seeded_organization_profiles())


def get_default_sample_organization_profile() -> dict[str, Any] | None:
    profiles = _load_seeded_organization_profiles()
    for profile in profiles:
        if bool(profile.get("default_sample")):
            return dict(profile)
    for profile in profiles:
        if str(profile.get("dataset_type") or "").strip().upper() == "DEFAULT_REALISTIC_SAMPLE":
            return dict(profile)
    return None


def _legacy_seeded_organization_ids() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for profile in _load_seeded_organization_profiles():
        canonical_id = str(profile.get("organization_id") or "").strip()
        if not canonical_id:
            continue
        for legacy_id in profile.get("legacy_organization_ids") or ():
            normalized = _normalize_optional_text(str(legacy_id))
            if normalized:
                aliases[normalized] = canonical_id
    return aliases


def _seeded_profile_by_org_id(organization_id: str) -> dict[str, Any] | None:
    organization_id = _canonical_organization_id(organization_id) or organization_id
    for seed in _load_seeded_organization_profiles():
        if str(seed.get("organization_id") or "").strip() == organization_id:
            return dict(seed)
    return None


def _canonical_seeded_display_name(organization_id: str, current_name: str | None) -> str | None:
    seed = _seeded_profile_by_org_id(organization_id)
    if seed is None:
        return current_name
    seeded_name = _normalize_optional_text(str(seed.get("display_name") or ""))
    if seeded_name is None:
        return current_name
    normalized_current = _normalize_optional_text(current_name)
    if normalized_current is None:
        return seeded_name
    legacy_names = tuple(
        _normalize_optional_text(str(item)) or ""
        for item in (seed.get("legacy_display_names") or ())
    )
    if normalized_current == organization_id or normalized_current in legacy_names:
        return seeded_name
    return normalized_current


def _canonical_organization_id(value: str | None) -> str | None:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    return _legacy_seeded_organization_ids().get(normalized, normalized)


def _load_organizations() -> dict[str, dict[str, Any]]:
    _ensure_auth_store()
    payload = load_organizations_db()
    if not isinstance(payload, dict):
        return {}
    organizations: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        org_id = _canonical_organization_id(str(value.get("organization_id") or key))
        if org_id is None:
            continue
        display_name = _normalize_optional_text(str(value.get("display_name") or "")) or org_id
        display_name = _canonical_seeded_display_name(org_id, display_name) or org_id
        seed = _seeded_profile_by_org_id(org_id) or {}
        organizations[org_id] = {
            **value,
            "organization_id": org_id,
            "display_name": display_name,
            "allow_self_join": bool(value.get("allow_self_join")),
            "created_at": str(value.get("created_at") or ""),
            "updated_at": str(value.get("updated_at") or ""),
            "created_by": _normalize_optional_text(str(value.get("created_by") or "")),
            "dataset_type": _normalize_dataset_type(str(value.get("dataset_type") or ""))
            or _normalize_dataset_type(str(seed.get("dataset_type") or "")),
            "industry_code": _normalize_optional_text(str(value.get("industry_code") or ""))
            or _normalize_optional_text(str(seed.get("industry_code") or "")),
            "dataset_key": _normalize_optional_text(str(value.get("dataset_key") or ""))
            or _normalize_optional_text(str(seed.get("dataset_key") or "")),
            "dataset_version": _normalize_optional_text(str(value.get("dataset_version") or ""))
            or _normalize_optional_text(str(seed.get("dataset_version") or "")),
            "active_dataset_version_id": _normalize_optional_text(
                str(value.get("active_dataset_version_id") or "")
            )
            or _normalize_optional_text(str(seed.get("active_dataset_version_id") or "")),
            "bootstrap_source": _normalize_optional_text(str(value.get("bootstrap_source") or ""))
            or _normalize_optional_text(str(seed.get("bootstrap_source") or "")),
            "demo_prompt_profile": _normalize_optional_text(
                str(value.get("demo_prompt_profile") or "")
            )
            or _normalize_optional_text(str(seed.get("demo_prompt_profile") or "")),
            "t6_materialization_policy": _normalize_t6_materialization_policy(
                value.get("t6_materialization_policy")
                if isinstance(value.get("t6_materialization_policy"), dict)
                else None,
                existing_policy=_default_t6_materialization_policy(),
                updated_at=_normalize_optional_text(
                    str(
                        (
                            value.get("t6_materialization_policy") or {}
                        ).get("updated_at")
                        if isinstance(value.get("t6_materialization_policy"), dict)
                        else ""
                    )
                ),
            ),
        }
    return retain_snapshot(payload, organizations, normalized=True)


def _write_organizations(organizations: dict[str, dict[str, Any]]) -> None:
    _ensure_auth_store()
    save_organizations_db(organizations)


def _load_organization_events() -> list[dict[str, Any]]:
    _ensure_auth_store()
    payload = load_organization_events()
    if not isinstance(payload, list):
        return []
    return retain_snapshot(payload, [item for item in payload if isinstance(item, dict)])


def _append_organization_event(event: dict[str, Any]) -> None:
    events = _load_organization_events()
    events.append(event)
    _ensure_auth_store()
    save_organization_events(events)


def _organization_memberships(user: dict[str, Any]) -> list[dict[str, Any]]:
    raw_memberships = user.get("organization_memberships")
    organizations = _load_organizations()
    normalized: list[dict[str, Any]] = []
    if isinstance(raw_memberships, list):
        for item in raw_memberships:
            if not isinstance(item, dict):
                continue
            organization_id = _canonical_organization_id(str(item.get("organization_id") or ""))
            if organization_id is None:
                continue
            organization = organizations.get(organization_id) or {}
            status = _normalize_membership_status(str(item.get("membership_status") or ""))
            role = _normalize_member_role(str(item.get("member_role") or ""))
            normalized.append(
                {
                    "organization_id": organization_id,
                    "organization_name": _canonical_seeded_display_name(
                        organization_id,
                        _normalize_optional_text(
                            str(organization.get("display_name") or "")
                        )
                        or _normalize_optional_text(
                        str(item.get("organization_name") or "")
                        ),
                    ),
                    "membership_status": status,
                    "member_role": role,
                    "sharing_enabled": bool(item.get("sharing_enabled")),
                    "dataset_type": _normalize_dataset_type(str(organization.get("dataset_type") or ""))
                    or _normalize_dataset_type(str(item.get("dataset_type") or "")),
                    "industry_code": _normalize_optional_text(str(organization.get("industry_code") or ""))
                    or _normalize_optional_text(str(item.get("industry_code") or "")),
                    "dataset_key": _normalize_optional_text(str(organization.get("dataset_key") or ""))
                    or _normalize_optional_text(str(item.get("dataset_key") or "")),
                    "active_dataset_version_id": _normalize_optional_text(
                        str(organization.get("active_dataset_version_id") or "")
                    )
                    or _normalize_optional_text(str(item.get("active_dataset_version_id") or "")),
                    "fiscal_year_start_month": organization.get("fiscal_year_start_month")
                    or item.get("fiscal_year_start_month"),
                    "local_currency": _normalize_optional_text(str(organization.get("local_currency") or ""))
                    or _normalize_optional_text(str(item.get("local_currency") or "")),
                    "joined_at": _normalize_optional_text(str(item.get("joined_at") or "")),
                    "updated_at": _normalize_optional_text(str(item.get("updated_at") or "")),
                }
            )
    legacy_organization_id = _canonical_organization_id(str(user.get("organization_id") or ""))
    if legacy_organization_id is not None and not any(
        item["organization_id"] == legacy_organization_id for item in normalized
    ):
        organization = organizations.get(legacy_organization_id) or {}
        normalized.append(
            {
                "organization_id": legacy_organization_id,
                "organization_name": _canonical_seeded_display_name(
                    legacy_organization_id,
                    _normalize_optional_text(str(organization.get("display_name") or ""))
                    or _normalize_optional_text(str(user.get("organization_name") or "")),
                ),
                "membership_status": _normalize_membership_status(
                    str(user.get("membership_status") or "active")
                ),
                "member_role": _normalize_member_role(str(user.get("member_role") or "member")),
                "sharing_enabled": bool(user.get("sharing_enabled")),
                "dataset_type": _normalize_dataset_type(str(organization.get("dataset_type") or ""))
                or _normalize_dataset_type(str(user.get("dataset_type") or "")),
                "industry_code": _normalize_optional_text(str(organization.get("industry_code") or ""))
                or _normalize_optional_text(str(user.get("industry_code") or "")),
                "dataset_key": _normalize_optional_text(str(organization.get("dataset_key") or ""))
                or _normalize_optional_text(str(user.get("dataset_key") or "")),
                "active_dataset_version_id": _normalize_optional_text(
                    str(organization.get("active_dataset_version_id") or "")
                )
                or _normalize_optional_text(str(user.get("active_dataset_version_id") or "")),
                "fiscal_year_start_month": organization.get("fiscal_year_start_month")
                or user.get("fiscal_year_start_month"),
                "local_currency": _normalize_optional_text(str(organization.get("local_currency") or ""))
                or _normalize_optional_text(str(user.get("local_currency") or "")),
                "joined_at": _normalize_optional_text(str(user.get("joined_at") or "")),
                "updated_at": _normalize_optional_text(str(user.get("updated_at") or "")),
            }
        )
    return normalized


def _resolve_current_membership(user: dict[str, Any]) -> dict[str, Any]:
    memberships = _organization_memberships(user)
    default_org_id = _canonical_organization_id(str(user.get("default_organization_id") or ""))
    if default_org_id is not None:
        for membership in memberships:
            if membership["organization_id"] == default_org_id:
                return membership
    if memberships:
        return memberships[0]
    return {
        "organization_id": None,
        "organization_name": None,
        "membership_status": "unassigned",
        "member_role": None,
        "sharing_enabled": False,
        "joined_at": None,
        "updated_at": None,
    }


def _membership_capabilities(membership: dict[str, Any]) -> tuple[bool, bool]:
    # MVL controls depend on an existing governed organization association,
    # not membership status or role. Preserve those values as metadata.
    available = bool(membership.get("organization_id"))
    return available, available


def _resolve_knowledge_base_dataset_context(
    *,
    organization_id: str | None,
    workspace_id: str | None,
    legacy_membership: dict[str, Any],
) -> dict[str, Any]:
    if not organization_id:
        return {
            "dataset_type": None,
            "dataset_key": None,
            "active_dataset_version_id": None,
            "dataset_context_source": "knowledge_base_activation_required",
            "dataset_context_deprecated": False,
            "dataset_context_handoff": "Knowledge Base -> Dataset",
            "dataset_migration_status": "cutover_no_legacy_fallback",
            "dataset_migration_error_code": "GOVERNED_ACTIVATION_REQUIRED",
        }
    store = None
    try:
        from storage.knowledge_source_activation_store import KnowledgeSourceActivationStore

        store = KnowledgeSourceActivationStore()
        active = store.get_active_dataset_binding(
            organization_id=str(organization_id),
            workspace_id=workspace_id,
        )
        if active is not None:
            legacy_dataset_type = _normalize_dataset_type(
                str(legacy_membership.get("dataset_type") or "")
            )
            legacy_dataset_key = _normalize_optional_text(
                str(legacy_membership.get("dataset_key") or "")
            )
            legacy_dataset_version_id = _normalize_optional_text(
                str(legacy_membership.get("active_dataset_version_id") or "")
            )
            governed_dataset_type = _normalize_dataset_type(str(active.get("dataset_type") or ""))
            governed_dataset_key = _normalize_optional_text(str(active.get("dataset_key") or ""))
            governed_dataset_version_id = _normalize_optional_text(
                str(active.get("dataset_version_id") or "")
            )
            conflicts = [
                legacy_dataset_type
                and governed_dataset_type
                and legacy_dataset_type != governed_dataset_type,
                legacy_dataset_key
                and governed_dataset_key
                and legacy_dataset_key != governed_dataset_key,
                legacy_dataset_version_id
                and governed_dataset_version_id
                and legacy_dataset_version_id != governed_dataset_version_id,
            ]
            if any(conflicts):
                return {
                    "dataset_type": None,
                    "dataset_key": None,
                    "active_dataset_version_id": None,
                    "dataset_context_source": "knowledge_base_legacy_conflict",
                    "dataset_context_deprecated": False,
                    "dataset_context_handoff": "Knowledge Base -> Dataset",
                    "knowledge_source_activation_id": active.get("activation_id"),
                    "dataset_asset_version_id": active.get("dataset_asset_version_id"),
                    "dataset_usage_role": active.get("dataset_usage_role"),
                    "dataset_authority_level": active.get("authority_level"),
                    "dataset_readiness_status": active.get("readiness_status"),
                }
            return {
                "dataset_type": active.get("dataset_type"),
                "dataset_key": active.get("dataset_key"),
                "active_dataset_version_id": active.get("dataset_version_id"),
                "dataset_context_source": "knowledge_base_activation",
                "dataset_context_deprecated": False,
                "dataset_context_handoff": "Knowledge Base -> Dataset",
                "knowledge_source_activation_id": active.get("activation_id"),
                "dataset_asset_version_id": active.get("dataset_asset_version_id"),
                "dataset_usage_role": active.get("dataset_usage_role"),
                "dataset_authority_level": active.get("authority_level"),
                "dataset_readiness_status": active.get("readiness_status"),
            }

        migration_state = store.get_migration_state(str(organization_id))
        if migration_state and migration_state.get("migration_status") in {
            "activation_written",
            "complete",
        }:
            return {
                "dataset_type": None,
                "dataset_key": None,
                "active_dataset_version_id": None,
                "dataset_context_source": "knowledge_base_activation_missing",
                "dataset_context_deprecated": False,
                "dataset_context_handoff": "Knowledge Base -> Dataset",
                "dataset_migration_status": migration_state.get("migration_status"),
                "dataset_migration_activation_id": migration_state.get("activation_id"),
            }

        if _normalize_dataset_type(str(legacy_membership.get("dataset_type") or "")) is not None:
            return {
                "dataset_type": None,
                "dataset_key": None,
                "active_dataset_version_id": None,
                "dataset_context_source": "knowledge_base_activation_required",
                "dataset_context_deprecated": False,
                "dataset_context_handoff": "Knowledge Base -> Dataset",
                "dataset_migration_status": "cutover_no_legacy_fallback",
                "dataset_migration_error_code": "GOVERNED_ACTIVATION_REQUIRED",
            }
    except Exception:
        # Dataset context is optional enrichment. A transient store failure
        # (e.g. a SQLite lock while the governed dataset tool is querying the
        # same activation store) must degrade to "resolution_failed" and must
        # never bubble up to invalidate the caller's authenticated session.
        return {
            "dataset_type": None,
            "dataset_key": None,
            "active_dataset_version_id": None,
            "dataset_context_source": "knowledge_base_resolution_failed",
            "dataset_context_deprecated": False,
            "dataset_context_handoff": "Knowledge Base -> Dataset",
        }
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass

    return {
        "dataset_type": None,
        "dataset_key": None,
        "active_dataset_version_id": None,
        "dataset_context_source": "knowledge_base_activation_required",
        "dataset_context_deprecated": False,
        "dataset_context_handoff": "Knowledge Base -> Dataset",
        "dataset_migration_status": "cutover_no_legacy_fallback",
        "dataset_migration_error_code": "GOVERNED_ACTIVATION_REQUIRED",
    }


def _sync_user_membership_fields(user: dict[str, Any]) -> dict[str, Any]:
    memberships = _organization_memberships(user)
    current = _resolve_current_membership(user)
    organization_id = current.get("organization_id")
    organization_name = current.get("organization_name")
    membership_status = current.get("membership_status") or "unassigned"
    member_role = current.get("member_role")
    sharing_enabled = bool(current.get("sharing_enabled"))
    dataset_type = current.get("dataset_type")
    industry_code = current.get("industry_code")
    dataset_key = current.get("dataset_key")
    active_dataset_version_id = current.get("active_dataset_version_id")
    if organization_id is None:
        user.pop("default_organization_id", None)
        user.pop("organization_id", None)
        user.pop("organization_name", None)
        user["membership_status"] = "unassigned"
        user.pop("member_role", None)
        user["sharing_enabled"] = False
        user.pop("dataset_type", None)
        user.pop("industry_code", None)
        user.pop("dataset_key", None)
        user.pop("active_dataset_version_id", None)
    else:
        user["default_organization_id"] = organization_id
        user["organization_id"] = organization_id
        user["organization_name"] = organization_name
        user["membership_status"] = membership_status
        user["member_role"] = member_role
        user["sharing_enabled"] = sharing_enabled
        user["dataset_type"] = dataset_type
        user["industry_code"] = industry_code
        user["dataset_key"] = dataset_key
        user["active_dataset_version_id"] = active_dataset_version_id
    user["organization_memberships"] = memberships
    can_invite_members, can_change_settings = _membership_capabilities(current)
    user["can_invite_members"] = can_invite_members
    user["can_change_settings"] = can_change_settings
    return user


def get_user_organization_context(user_id: str) -> dict[str, Any] | None:
    user = get_user_record(user_id)
    if not isinstance(user, dict):
        return None
    user = _sync_user_membership_fields(dict(user))
    current = _resolve_current_membership(user)
    dataset_context = _resolve_knowledge_base_dataset_context(
        organization_id=(
            str(current.get("organization_id"))
            if current.get("organization_id")
            else None
        ),
        workspace_id=(str(user.get("workspace_id")) if user.get("workspace_id") else None),
        legacy_membership=current,
    )
    can_invite_members, can_change_settings = _membership_capabilities(current)
    memberships = _organization_memberships(user)
    return {
        "organization_id": current.get("organization_id"),
        "organization_name": current.get("organization_name"),
        "membership_status": current.get("membership_status") or "unassigned",
        "member_role": current.get("member_role"),
        "sharing_enabled": bool(current.get("sharing_enabled")),
        "dataset_type": dataset_context.get("dataset_type"),
        "industry_code": current.get("industry_code"),
        "dataset_key": dataset_context.get("dataset_key"),
        "active_dataset_version_id": dataset_context.get("active_dataset_version_id"),
        "dataset_context_source": dataset_context.get("dataset_context_source"),
        "dataset_context_deprecated": dataset_context.get("dataset_context_deprecated"),
        "dataset_context_handoff": dataset_context.get("dataset_context_handoff"),
        "knowledge_source_activation_id": dataset_context.get("knowledge_source_activation_id"),
        "dataset_asset_version_id": dataset_context.get("dataset_asset_version_id"),
        "dataset_usage_role": dataset_context.get("dataset_usage_role"),
        "dataset_authority_level": dataset_context.get("dataset_authority_level"),
        "dataset_readiness_status": dataset_context.get("dataset_readiness_status"),
        "dataset_migration_status": dataset_context.get("dataset_migration_status"),
        "dataset_migration_activation_id": dataset_context.get("dataset_migration_activation_id"),
        "dataset_migration_error_code": dataset_context.get("dataset_migration_error_code"),
        "fiscal_year_start_month": current.get("fiscal_year_start_month"),
        "local_currency": current.get("local_currency"),
        "can_invite_members": can_invite_members,
        "can_change_settings": can_change_settings,
        "memberships": memberships,
    }


def list_organization_members(organization_id: str) -> list[dict[str, Any]]:
    normalized_org_id = _canonical_organization_id(organization_id)
    if normalized_org_id is None:
        return []
    members: list[dict[str, Any]] = []
    for user in list_user_records():
        memberships = _organization_memberships(user)
        target = next(
            (membership for membership in memberships if membership["organization_id"] == normalized_org_id),
            None,
        )
        if target is None:
            continue
        can_invite_members, can_change_settings = _membership_capabilities(target)
        members.append(
            {
                "user_id": str(user.get("user_id") or ""),
                "name": str(user.get("display_name") or user.get("name") or user.get("workspace_slug") or ""),
                "workspace_id": str(user.get("workspace_id") or ""),
                "workspace_slug": str(user.get("workspace_slug") or ""),
                "membership_status": target.get("membership_status") or "unassigned",
                "member_role": target.get("member_role"),
                "sharing_enabled": bool(target.get("sharing_enabled")),
                "dataset_type": target.get("dataset_type"),
                "industry_code": target.get("industry_code"),
                "can_invite_members": can_invite_members,
                "can_change_settings": can_change_settings,
            }
        )
    members.sort(key=lambda item: (item["membership_status"] != "active", item["name"], item["user_id"]))
    return members


def list_organization_events(organization_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
    normalized_org_id = _canonical_organization_id(organization_id)
    if normalized_org_id is None:
        return []
    events = [
        event
        for event in _load_organization_events()
        if str(event.get("organization_id") or "").strip() == normalized_org_id
    ]
    # limit=0 means return all events (used by export paths that must not truncate).
    return events if limit == 0 else events[-limit:]


def get_pending_role_notifications_for_user(
    organization_id: str,
    user_id: str,
) -> list[dict[str, Any]]:
    """Return unacknowledged owner-displacement notifications for a user."""
    normalized_org_id = _canonical_organization_id(organization_id)
    if normalized_org_id is None:
        return []
    results = []
    for event in _load_organization_events():
        if str(event.get("organization_id") or "").strip() != normalized_org_id:
            continue
        if event.get("event_type") != "membership_role_updated":
            continue
        if str(event.get("subject_user_id") or "").strip() != str(user_id).strip():
            continue
        detail = event.get("detail") or {}
        if detail.get("notification_pending"):
            results.append(event)
    return results


def _record_organization_event(
    *,
    organization_id: str,
    actor_user_id: str,
    event_type: str,
    subject_user_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    _append_organization_event(
        {
            "event_id": uuid.uuid4().hex,
            "organization_id": organization_id,
            "actor_user_id": actor_user_id,
            "subject_user_id": subject_user_id or actor_user_id,
            "event_type": event_type,
            "detail": detail or {},
            "created_at": _now_iso(),
        }
    )


def ensure_organization(
    *,
    organization_id: str | None = None,
    display_name: str,
    created_by: str,
    allow_self_join: bool = False,
    dataset_type: str | None = None,
    industry_code: str | None = None,
    dataset_key: str | None = None,
    dataset_version: str | None = None,
    bootstrap_source: str | None = None,
    demo_prompt_profile: str | None = None,
    active_dataset_version_id: str | None = None,
    fiscal_year_start_month: int | None = None,
    local_currency: str | None = None,
    t6_materialization_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_name = _normalize_optional_text(display_name)
    if normalized_name is None:
        raise ValueError("organization display_name required")
    normalized_org_id = _canonical_organization_id(organization_id)
    if normalized_org_id is None:
        normalized_org_id = _slug_org_id(normalized_name)
    normalized_dataset_type = _normalize_dataset_type(dataset_type)
    normalized_industry_code = _normalize_optional_text(industry_code)
    normalized_dataset_key = _normalize_optional_text(dataset_key)
    normalized_dataset_version = _normalize_optional_text(dataset_version)
    normalized_bootstrap_source = _normalize_optional_text(bootstrap_source)
    normalized_demo_prompt_profile = _normalize_optional_text(demo_prompt_profile)
    normalized_active_dataset_version_id = _normalize_optional_text(active_dataset_version_id)
    normalized_local_currency = _normalize_optional_text(local_currency)
    normalized_fiscal_year_start_month = (
        int(fiscal_year_start_month)
        if isinstance(fiscal_year_start_month, int) and 1 <= int(fiscal_year_start_month) <= 12
        else None
    )
    with _WRITE_LOCK, identity_write_transaction():
        organizations = _load_organizations()
        existing = organizations.get(normalized_org_id)
        if existing is not None:
            updated = dict(existing)
            updated["display_name"] = normalized_name
            updated["allow_self_join"] = bool(allow_self_join)
            if normalized_dataset_type is not None:
                updated["dataset_type"] = normalized_dataset_type
            if normalized_industry_code is not None:
                updated["industry_code"] = normalized_industry_code
            if normalized_dataset_key is not None:
                updated["dataset_key"] = normalized_dataset_key
            if normalized_dataset_version is not None:
                updated["dataset_version"] = normalized_dataset_version
            if normalized_bootstrap_source is not None:
                updated["bootstrap_source"] = normalized_bootstrap_source
            if normalized_demo_prompt_profile is not None:
                updated["demo_prompt_profile"] = normalized_demo_prompt_profile
            if normalized_active_dataset_version_id is not None:
                updated["active_dataset_version_id"] = normalized_active_dataset_version_id
            if normalized_fiscal_year_start_month is not None:
                updated["fiscal_year_start_month"] = normalized_fiscal_year_start_month
            if normalized_local_currency is not None:
                updated["local_currency"] = normalized_local_currency
            updated["t6_materialization_policy"] = _normalize_t6_materialization_policy(
                t6_materialization_policy,
                existing_policy=updated.get("t6_materialization_policy"),
                updated_at=_now_iso(),
            )
            updated["updated_at"] = _now_iso()
            if updated != existing:
                organizations[normalized_org_id] = updated
                _write_organizations(organizations)
            return updated
        now = _now_iso()
        organization = {
            "organization_id": normalized_org_id,
            "display_name": normalized_name,
            "allow_self_join": bool(allow_self_join),
            "created_at": now,
            "updated_at": now,
            "created_by": created_by,
            "dataset_type": normalized_dataset_type,
            "industry_code": normalized_industry_code,
            "dataset_key": normalized_dataset_key,
            "dataset_version": normalized_dataset_version,
            "bootstrap_source": normalized_bootstrap_source,
            "demo_prompt_profile": normalized_demo_prompt_profile,
            "active_dataset_version_id": normalized_active_dataset_version_id,
            "fiscal_year_start_month": normalized_fiscal_year_start_month,
            "local_currency": normalized_local_currency,
            "t6_materialization_policy": _normalize_t6_materialization_policy(
                t6_materialization_policy,
                existing_policy=None,
                updated_at=now,
            ),
        }
        organizations[normalized_org_id] = organization
        _write_organizations(organizations)
        _record_organization_event(
            organization_id=normalized_org_id,
            actor_user_id=created_by,
            event_type="organization_created",
            detail={"display_name": normalized_name, "allow_self_join": bool(allow_self_join)},
        )
        return organization


def get_organization_materialization_policy(organization_id: str) -> dict[str, Any]:
    normalized_org_id = _canonical_organization_id(organization_id)
    if normalized_org_id is None:
        raise ValueError("organization_id required")
    organizations = _load_organizations()
    organization = organizations.get(normalized_org_id)
    if organization is None:
        raise ValueError("organization not found")
    policy = organization.get("t6_materialization_policy")
    return _normalize_t6_materialization_policy(
        policy if isinstance(policy, dict) else None,
        existing_policy=_default_t6_materialization_policy(),
    )


def update_organization_materialization_policy(
    *,
    organization_id: str,
    policy_patch: dict[str, Any],
    actor_user_id: str,
) -> dict[str, Any]:
    normalized_org_id = _canonical_organization_id(organization_id)
    if normalized_org_id is None:
        raise ValueError("organization_id required")
    if not isinstance(policy_patch, dict):
        raise ValueError("policy_patch must be an object")

    with _WRITE_LOCK, identity_write_transaction():
        organizations = _load_organizations()
        organization = organizations.get(normalized_org_id)
        if not isinstance(organization, dict):
            raise ValueError("organization not found")

        current_policy = organization.get("t6_materialization_policy")
        normalized_policy = _normalize_t6_materialization_policy(
            policy_patch,
            existing_policy=current_policy if isinstance(current_policy, dict) else None,
            updated_at=_now_iso(),
        )

        organization["t6_materialization_policy"] = normalized_policy
        organization["updated_at"] = _now_iso()
        organizations[normalized_org_id] = organization
        _write_organizations(organizations)
        _record_organization_event(
            organization_id=normalized_org_id,
            actor_user_id=actor_user_id,
            event_type="organization_materialization_policy_updated",
            detail={
                "default_mode": normalized_policy.get("default_mode"),
                "policy_version_hash": normalized_policy.get("policy_version_hash"),
            },
        )
        return dict(normalized_policy)


def set_organization_active_dataset_version(
    *,
    organization_id: str,
    dataset_version_id: str,
    actor_user_id: str,
) -> dict[str, Any]:
    normalized_org_id = _canonical_organization_id(organization_id)
    normalized_dataset_version_id = _normalize_optional_text(dataset_version_id)
    if normalized_org_id is None:
        raise ValueError("organization_id required")
    if normalized_dataset_version_id is None:
        raise ValueError("dataset_version_id required")

    with _WRITE_LOCK, identity_write_transaction():
        now = _now_iso()
        organizations = _load_organizations()
        organization = organizations.get(normalized_org_id)
        if not isinstance(organization, dict):
            raise ValueError("organization not found")
        organization["active_dataset_version_id"] = normalized_dataset_version_id
        organization["updated_at"] = now
        organizations[normalized_org_id] = organization
        _write_organizations(organizations)
        users, correlations = _load_user_state()
        for user_id, user in list(users.items()):
            if not isinstance(user, dict):
                continue
            memberships = _organization_memberships(user)
            changed = False
            for membership in memberships:
                if membership.get("organization_id") != normalized_org_id:
                    continue
                membership["active_dataset_version_id"] = normalized_dataset_version_id
                membership["updated_at"] = now
                changed = True
            if not changed:
                continue
            user["organization_memberships"] = memberships
            user["updated_at"] = now
            users[user_id] = _sync_user_membership_fields(user)
        _save_user_state(users, correlations)
        _record_organization_event(
            organization_id=normalized_org_id,
            actor_user_id=actor_user_id,
            event_type="organization_active_dataset_version_updated",
            detail={"active_dataset_version_id": normalized_dataset_version_id},
        )
        return dict(organization)


def get_user_membership_for_organization(
    *,
    user_id: str,
    organization_id: str,
) -> dict[str, Any] | None:
    normalized_org_id = _canonical_organization_id(organization_id)
    if normalized_org_id is None:
        return None
    user = get_user_record(user_id)
    if not isinstance(user, dict):
        return None
    memberships = _organization_memberships(user)
    for membership in memberships:
        if membership.get("organization_id") == normalized_org_id:
            return membership
    return None


def count_active_organization_owners(organization_id: str) -> int:
    normalized_org_id = _canonical_organization_id(organization_id)
    if normalized_org_id is None:
        return 0
    owners = 0
    for user in list_user_records():
        memberships = _organization_memberships(user)
        for membership in memberships:
            if membership.get("organization_id") != normalized_org_id:
                continue
            if membership.get("membership_status") != "active":
                continue
            if membership.get("member_role") != "owner":
                continue
            owners += 1
    return owners


def update_membership_role(
    *,
    target_user_id: str,
    organization_id: str,
    member_role: str,
    actor_user_id: str,
) -> dict[str, Any]:
    normalized_org_id = _canonical_organization_id(organization_id)
    if normalized_org_id is None:
        raise ValueError("organization_id required")
    normalized_role = _normalize_member_role(member_role)
    if normalized_role is None:
        raise ValueError("member_role required")

    with _WRITE_LOCK, identity_write_transaction():
        users, correlations = _load_user_state()
        user = users.get(target_user_id)
        if not isinstance(user, dict):
            raise KeyError(target_user_id)

        memberships = _organization_memberships(user)
        target = None
        for membership in memberships:
            if membership["organization_id"] == normalized_org_id:
                target = membership
                break
        if target is None:
            raise ValueError("membership not found")

        previous_role = target.get("member_role")
        target["member_role"] = normalized_role
        target["updated_at"] = _now_iso()
        user["organization_memberships"] = memberships
        user["updated_at"] = _now_iso()
        users[target_user_id] = _sync_user_membership_fields(user)

        # Enforce exactly one active owner per org: if this write sets a new owner,
        # demote all other active owners to member atomically and emit a
        # notification_pending audit event for each displaced owner.
        demoted_user_ids: list[str] = []
        if normalized_role == "owner":
            for uid, other_user in users.items():
                if uid == target_user_id:
                    continue
                if not isinstance(other_user, dict):
                    continue
                other_memberships = _organization_memberships(other_user)
                for other_membership in other_memberships:
                    if other_membership.get("organization_id") != normalized_org_id:
                        continue
                    if other_membership.get("membership_status") != "active":
                        continue
                    if other_membership.get("member_role") != "owner":
                        continue
                    other_membership["member_role"] = "member"
                    other_membership["updated_at"] = _now_iso()
                    other_user["organization_memberships"] = other_memberships
                    other_user["updated_at"] = _now_iso()
                    users[uid] = _sync_user_membership_fields(other_user)
                    demoted_user_ids.append(uid)

        _save_user_state(users, correlations)
        _record_organization_event(
            organization_id=normalized_org_id,
            actor_user_id=actor_user_id,
            subject_user_id=target_user_id,
            event_type="membership_role_updated",
            detail={"previous_role": previous_role, "member_role": normalized_role},
        )
        for demoted_uid in demoted_user_ids:
            _record_organization_event(
                organization_id=normalized_org_id,
                actor_user_id=actor_user_id,
                subject_user_id=demoted_uid,
                event_type="membership_role_updated",
                detail={
                    "previous_role": "owner",
                    "member_role": "member",
                    "notification_pending": True,
                },
            )
        return users[target_user_id]


def list_seeded_demo_organization_profiles() -> list[dict[str, Any]]:
    organizations = _load_organizations()
    seeded_profiles: list[dict[str, Any]] = []
    for seed in _load_seeded_organization_profiles():
        org_id = str(seed["organization_id"])
        current = organizations.get(org_id) or {}
        seeded_profiles.append(
            {
                "organization_id": org_id,
                "organization_name": current.get("display_name") or seed.get("display_name") or org_id,
                "dataset_type": current.get("dataset_type") or seed.get("dataset_type"),
                "industry_code": current.get("industry_code") or seed.get("industry_code"),
                "dataset_key": current.get("dataset_key") or seed.get("dataset_key"),
                "dataset_version": current.get("dataset_version") or seed.get("dataset_version"),
                "bootstrap_source": current.get("bootstrap_source") or seed.get("bootstrap_source"),
                "demo_prompt_profile": current.get("demo_prompt_profile")
                or seed.get("demo_prompt_profile"),
                "seeded": bool(current),
            }
        )
    return seeded_profiles


def _repair_user_seeded_organization_cache() -> int:
    users, correlations = _load_user_state()
    organizations = _load_organizations()
    repaired = 0

    for user_id, user in users.items():
        if not isinstance(user, dict):
            continue

        original = json.dumps(user, sort_keys=True, ensure_ascii=False)
        memberships = _organization_memberships(user)
        changed = False

        for membership in memberships:
            org_id = str(membership.get("organization_id") or "").strip()
            if not org_id:
                continue
            organization = organizations.get(org_id) or {}
            canonical_name = _canonical_seeded_display_name(
                org_id,
                _normalize_optional_text(str(organization.get("display_name") or ""))
                or _normalize_optional_text(str(membership.get("organization_name") or "")),
            )
            if canonical_name and membership.get("organization_name") != canonical_name:
                membership["organization_name"] = canonical_name
                changed = True

            dataset_type = _normalize_dataset_type(str(organization.get("dataset_type") or "")) or membership.get(
                "dataset_type"
            )
            industry_code = _normalize_optional_text(str(organization.get("industry_code") or "")) or membership.get(
                "industry_code"
            )
            if membership.get("dataset_type") != dataset_type:
                membership["dataset_type"] = dataset_type
                changed = True
            if membership.get("industry_code") != industry_code:
                membership["industry_code"] = industry_code
                changed = True

        if changed:
            user["organization_memberships"] = memberships

        synced = _sync_user_membership_fields(user)
        updated = json.dumps(synced, sort_keys=True, ensure_ascii=False)
        if updated != original:
            users[user_id] = synced
            repaired += 1

    if repaired:
        _save_user_state(users, correlations)
    return repaired


def ensure_seeded_demo_organizations(*, actor_user_id: str = "bootstrap") -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for seed in _load_seeded_organization_profiles():
        profiles.append(
            ensure_organization(
                organization_id=str(seed["organization_id"]),
                display_name=str(seed["display_name"]),
                created_by=actor_user_id,
                allow_self_join=bool(seed.get("allow_self_join")),
                dataset_type=str(seed.get("dataset_type") or ""),
                industry_code=str(seed.get("industry_code") or ""),
                dataset_key=str(seed.get("dataset_key") or ""),
                dataset_version=str(seed.get("dataset_version") or ""),
                bootstrap_source=str(seed.get("bootstrap_source") or ""),
                demo_prompt_profile=str(seed.get("demo_prompt_profile") or ""),
            )
        )
    _repair_user_seeded_organization_cache()
    return profiles


def resolve_seeded_demo_organization_for_industry(industry_code: str) -> dict[str, Any] | None:
    normalized_industry_code = _normalize_optional_text(industry_code)
    if normalized_industry_code is None:
        return None
    organizations = _load_organizations()
    for organization in organizations.values():
        if organization.get("industry_code") == normalized_industry_code:
            return organization
    return None


def attach_user_to_seeded_demo_organization(
    *,
    user_id: str,
    actor_user_id: str,
    industry_code: str,
) -> dict[str, Any]:
    organization = resolve_seeded_demo_organization_for_industry(industry_code)
    if organization is None:
        raise ValueError("demo organization not found for industry_code")
    return join_organization(
        user_id=user_id,
        actor_user_id=actor_user_id,
        organization_id=str(organization["organization_id"]),
        create=False,
    )


def join_organization(
    *,
    user_id: str,
    actor_user_id: str,
    organization_id: str | None = None,
    organization_name: str | None = None,
    dataset_type: str | None = None,
    industry_code: str | None = None,
    fiscal_year_start_month: int | None = None,
    local_currency: str | None = None,
    create: bool = False,
) -> dict[str, Any]:
    normalized_org_id = _canonical_organization_id(organization_id)
    normalized_org_name = _normalize_optional_text(organization_name)
    with _WRITE_LOCK, identity_write_transaction():
        users, correlations = _load_user_state()
        user = users.get(user_id)
        if not isinstance(user, dict):
            raise KeyError(user_id)
        organizations = _load_organizations()
        organization: dict[str, Any] | None = None
        if create:
            if normalized_org_name is None and normalized_org_id is None:
                raise ValueError("organization_name required when creating organization")
            requested_org_id = normalized_org_id or _slug_org_id(normalized_org_name or "organization")
            if requested_org_id in organizations:
                raise ValueError("organization already exists")
            organization = ensure_organization(
                organization_id=requested_org_id,
                display_name=normalized_org_name or normalized_org_id or "organization",
                created_by=actor_user_id,
                allow_self_join=False,
                dataset_type=dataset_type,
                industry_code=industry_code,
                fiscal_year_start_month=fiscal_year_start_month,
                local_currency=local_currency,
            )
            status = "active"
            role = "owner"
            sharing_enabled = True
        else:
            if normalized_org_id is None:
                raise ValueError("organization_id required")
            organization = organizations.get(normalized_org_id)
            if organization is None:
                raise ValueError("organization not found")
            public_join = bool(organization.get("allow_self_join"))
            if is_postgresql() and organization.get("dataset_type") not in {"DEMO", "DEFAULT_REALISTIC_SAMPLE"}:
                public_join = False
            status = "active" if public_join else "pending"
            role = "member"
            sharing_enabled = False
        memberships = _organization_memberships(user)
        now = _now_iso()
        replaced = False
        for membership in memberships:
            if membership["organization_id"] != organization["organization_id"]:
                continue
            membership["organization_name"] = organization["display_name"]
            membership["membership_status"] = status
            membership["member_role"] = role
            membership["sharing_enabled"] = sharing_enabled
            membership["dataset_type"] = organization.get("dataset_type")
            membership["industry_code"] = organization.get("industry_code")
            membership["joined_at"] = membership.get("joined_at") or now
            membership["updated_at"] = now
            replaced = True
            break
        if not replaced:
            memberships.append(
                {
                    "organization_id": organization["organization_id"],
                    "organization_name": organization["display_name"],
                    "membership_status": status,
                    "member_role": role,
                    "sharing_enabled": sharing_enabled,
                    "dataset_type": organization.get("dataset_type"),
                    "industry_code": organization.get("industry_code"),
                    "joined_at": now,
                    "updated_at": now,
                }
            )
        user["default_organization_id"] = organization["organization_id"]
        user["organization_memberships"] = memberships
        user["updated_at"] = now
        users[user_id] = _sync_user_membership_fields(user)
        _save_user_state(users, correlations)
        _record_organization_event(
            organization_id=organization["organization_id"],
            actor_user_id=actor_user_id,
            subject_user_id=user_id,
            event_type="organization_join_requested" if status == "pending" else "organization_joined",
            detail={"membership_status": status, "member_role": role, "sharing_enabled": sharing_enabled},
        )
        return users[user_id]


def switch_user_organization(
    *,
    user_id: str,
    organization_id: str,
    actor_user_id: str,
) -> dict[str, Any]:
    normalized_org_id = _canonical_organization_id(organization_id)
    if normalized_org_id is None:
        raise ValueError("organization_id required")
    with _WRITE_LOCK, identity_write_transaction():
        users, correlations = _load_user_state()
        user = users.get(user_id)
        if not isinstance(user, dict):
            raise KeyError(user_id)
        memberships = _organization_memberships(user)
        if not any(item["organization_id"] == normalized_org_id for item in memberships):
            raise ValueError("membership not found")
        target = next(item for item in memberships if item["organization_id"] == normalized_org_id)
        user["default_organization_id"] = normalized_org_id
        user["updated_at"] = _now_iso()
        users[user_id] = _sync_user_membership_fields(user)
        _save_user_state(users, correlations)
        _record_organization_event(
            organization_id=normalized_org_id,
            actor_user_id=actor_user_id,
            subject_user_id=user_id,
            event_type="organization_switched",
        )
        return users[user_id]


def set_user_sharing_enabled(
    *,
    user_id: str,
    organization_id: str,
    sharing_enabled: bool,
    actor_user_id: str,
) -> dict[str, Any]:
    normalized_org_id = _canonical_organization_id(organization_id)
    if normalized_org_id is None:
        raise ValueError("organization_id required")
    with _WRITE_LOCK, identity_write_transaction():
        users, correlations = _load_user_state()
        user = users.get(user_id)
        if not isinstance(user, dict):
            raise KeyError(user_id)
        memberships = _organization_memberships(user)
        target = None
        for membership in memberships:
            if membership["organization_id"] == normalized_org_id:
                target = membership
                break
        if target is None:
            raise ValueError("membership not found")
        target["sharing_enabled"] = bool(sharing_enabled)
        target["updated_at"] = _now_iso()
        user["organization_memberships"] = memberships
        user["updated_at"] = _now_iso()
        users[user_id] = _sync_user_membership_fields(user)
        _save_user_state(users, correlations)
        _record_organization_event(
            organization_id=normalized_org_id,
            actor_user_id=actor_user_id,
            subject_user_id=user_id,
            event_type="organization_sharing_updated",
            detail={"sharing_enabled": bool(sharing_enabled)},
        )
        return users[user_id]


def update_membership_status(
    *,
    target_user_id: str,
    organization_id: str,
    membership_status: str,
    member_role: str | None,
    actor_user_id: str,
) -> dict[str, Any]:
    normalized_org_id = _normalize_optional_text(organization_id)
    if normalized_org_id is None:
        raise ValueError("organization_id required")
    normalized_status = _normalize_membership_status(membership_status)
    normalized_role = _normalize_member_role(member_role)
    with _WRITE_LOCK, identity_write_transaction():
        users, correlations = _load_user_state()
        user = users.get(target_user_id)
        if not isinstance(user, dict):
            raise KeyError(target_user_id)
        memberships = _organization_memberships(user)
        target = None
        for membership in memberships:
            if membership["organization_id"] == normalized_org_id:
                target = membership
                break
        if target is None:
            raise ValueError("membership not found")
        target["membership_status"] = normalized_status
        target["member_role"] = normalized_role
        if normalized_status != "active":
            target["sharing_enabled"] = False
        target["updated_at"] = _now_iso()
        user["organization_memberships"] = memberships
        user["updated_at"] = _now_iso()
        users[target_user_id] = _sync_user_membership_fields(user)
        _save_user_state(users, correlations)
        _record_organization_event(
            organization_id=normalized_org_id,
            actor_user_id=actor_user_id,
            subject_user_id=target_user_id,
            event_type=f"membership_{normalized_status}",
            detail={"member_role": normalized_role},
        )
        return users[target_user_id]


def _load_user_state() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _ensure_auth_store()
    users = load_users()
    if not isinstance(users, dict):
        users = {}
    correlations = load_gateway_correlations()
    if not isinstance(correlations, list):
        correlations = []
    return users, correlations


def _prune_weixin_login_states(states: dict[str, Any]) -> dict[str, Any]:
    cutoff = time.time() - WEIXIN_LOGIN_STATE_TTL_SECONDS
    pruned: dict[str, Any] = {}
    for key, value in states.items():
        if not isinstance(value, dict):
            continue
        created_at = value.get("created_at")
        created_ts = float(created_at) if isinstance(created_at, (int, float)) else 0.0
        if created_ts and created_ts < cutoff:
            continue
        pruned[key] = value
    return retain_snapshot(states, pruned)


def _save_user_state(users: dict[str, Any], correlations: list[dict[str, Any]]) -> None:
    _ensure_auth_store()
    save_users(users)
    save_gateway_correlations(correlations)


def _profile_completed(user: dict[str, Any]) -> bool:
    value = user.get("profile_completed_at")
    return isinstance(value, str) and bool(value.strip())


def _has_saved_weixin_account(user: dict[str, Any]) -> bool:
    workspace_id = _normalize_optional_text(str(user.get("workspace_id") or ""))
    weixin_user_id = _normalize_optional_text(str(user.get("weixin_user_id") or ""))
    if workspace_id is None or weixin_user_id is None:
        return False

    for payload in load_weixin_runtime_accounts():
        saved_user_id = _normalize_optional_text(
            str(payload.get("external_user_id") or payload.get("user_id") or "")
        )
        saved_token = _normalize_optional_text(str(payload.get("token") or ""))
        owner_workspace_id = _normalize_optional_text(
            str(payload.get("owner_workspace_id") or payload.get("workspace_id") or "")
        )
        if (
            saved_user_id == weixin_user_id
            and saved_token is not None
            and (owner_workspace_id is None or owner_workspace_id == workspace_id)
        ):
            return True

    return False


def find_resumable_weixin_user() -> dict[str, Any] | None:
    with _WRITE_LOCK, identity_write_transaction():
        users, _ = _load_user_state()
        candidates = [
            user
            for user in users.values()
            if isinstance(user, dict)
            and str(user.get("auth_provider") or "").strip() == "weixin"
            and not _profile_completed(user)
            and _has_saved_weixin_account(user)
        ]
        if len(candidates) != 1:
            return None
        return candidates[0]


def hash_local_password(password: str) -> str:
    return _PASSWORD_HASHER.hash(password)


def verify_local_password_hash(password_hash: str, password: str) -> bool:
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def register_password_user(*, login_name: str, password: str) -> dict[str, Any]:
    login = login_name.strip().lower()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", login) or not 3 <= len(login) <= 32:
        raise ValueError("Use 3–32 letters, numbers, or single hyphens for the login name")
    if not 8 <= len(password.strip()) <= 256:
        raise ValueError("Password must contain 8–256 characters")
    password_hash = hash_local_password(password.strip())
    with _WRITE_LOCK, identity_write_transaction():
        _ensure_auth_store()
        user_id = uuid.uuid4().hex
        now = _now_iso()
        user = {
            "user_id": user_id, "workspace_id": user_id,
            "workspace_slug": f"user-{user_id}", "name": login,
            "display_name": None, "password_login_name": login,
            "password_hash": password_hash, "profile_completed_at": None,
            "auth_provider": "password", "weixin_user_id": None,
            "feishu_open_id": "", "created_at": now, "updated_at": now,
        }
        if is_postgresql():
            from agents.auth_db_postgres import provision_personal_signup
            provision_personal_signup(user)
        else:
            insert_password_user(user)
    # Filesystem preparation is repairable after the SQL authority commits.
    ensure_workspace_paths(user["workspace_id"])
    return user


def repair_personal_billing_account(*, user_id: str) -> dict[str, Any]:
    """Authenticated POST repair for one canonical user, never a GET side effect."""
    with _WRITE_LOCK:
        user=get_user_record(user_id)
        if not isinstance(user,dict):raise KeyError(user_id)
        if is_postgresql():
            from agents.auth_db_postgres import provision_personal_signup
            provision_personal_signup(user,insert_user=False)
    ensure_workspace_paths(user['workspace_id'])
    return user


def account_provisioning_for_user(*, user_id: str, organization_id=None, workspace_id=None,
                                  account_id=None, currency=None):
    """Bounded read-only readiness; no materialization or synchronization wake-up."""
    if not is_postgresql():return None
    from agents.auth_db_postgres import connection
    from agents.model_gateway.account_sync_store import account_provisioning
    user=get_user_record(user_id)
    if not isinstance(user,dict):
        return {'local_status':'REPAIR_REQUIRED','inference_ready':False,'reason':'NO_ACCESS'}
    workspace_id=workspace_id or user.get('workspace_id')
    if organization_id is None:
        organization_id=_resolve_current_membership(user).get('organization_id')
    with connection() as c:
        result=account_provisioning(c,user_id=user_id,organization_id=organization_id,
            workspace_id=workspace_id,billing_schema=os.environ.get('SEMANTIER_MODEL_GATEWAY_BILLING_SCHEMA','semantier_billing'),
            account_id=account_id,currency=currency)
    root,home=resolve_workspace_paths(str(workspace_id))
    if result.get('local_status')=='READY' and (not root.is_dir() or not (home/'config.yaml').is_file()):
        result={**result,'local_status':'REPAIR_REQUIRED','inference_ready':False,'reason':'WORKSPACE_SETUP_REQUIRED'}
    return result


def complete_user_profile(
    *,
    user_id: str,
    display_name: str | None = None,
    login_name: str | None = None,
    password: str | None = None,
) -> dict[str, Any]:
    normalized_display_name = _normalize_optional_text(display_name)
    normalized_login_name = _normalize_optional_text(login_name)
    normalized_password = _normalize_optional_text(password)

    with _WRITE_LOCK, identity_write_transaction():
        users, correlations = _load_user_state()
        user = users.get(user_id)
        if not isinstance(user, dict):
            raise KeyError(user_id)

        if normalized_display_name is not None:
            user["name"] = normalized_display_name
            user["display_name"] = normalized_display_name

        if normalized_login_name is not None:
            slug_login_name = _slug_login(normalized_login_name)
            for candidate_id, candidate in users.items():
                if candidate_id == user_id or not isinstance(candidate, dict):
                    continue
                if str(candidate.get("password_login_name") or "").strip() == slug_login_name:
                    raise ValueError("login name already in use")
            user["password_login_name"] = slug_login_name

        if normalized_password is not None:
            user["password_hash"] = hash_local_password(normalized_password)

        user["profile_completed_at"] = _now_iso()
        user["updated_at"] = _now_iso()
        users[user_id] = user
        _save_user_state(users, correlations)
        return user


def authenticate_password_user(
    *,
    login_name: str | None,
    password: str,
) -> dict[str, Any] | None:
    normalized_login_name = _normalize_optional_text(login_name)
    users = list_user_records()
    candidates: list[dict[str, Any]] = []
    if normalized_login_name is not None:
        expected = _slug_login(normalized_login_name)
        for user in users:
            if str(user.get("password_login_name") or "").strip() == expected:
                candidates.append(user)
    else:
        candidates = [
            user
            for user in users
            if str(user.get("password_hash") or "").strip()
        ]
        if len(candidates) != 1:
            return None

    for user in candidates:
        password_hash = str(user.get("password_hash") or "").strip()
        if password_hash and verify_local_password_hash(password_hash, password):
            return user
    return None


def resolve_weixin_owner_correlation(
    *,
    account_id: str | None = None,
    external_user_id: str | None = None,
    chat_id: str | None = None,
    platform_session_key: str | None = None,
) -> dict[str, Any] | None:
    identifiers = {
        "linked_account_id": _normalize_optional_text(account_id),
        "linked_external_user_id": _normalize_optional_text(external_user_id),
        "chat_id": _normalize_optional_text(chat_id),
        "platform_session_key": _normalize_optional_text(platform_session_key),
    }
    if not any(identifiers.values()):
        raise ValueError("weixin owner resolution requires at least one identifier")

    _, correlations = _load_user_state()
    matched_rows: list[dict[str, Any]] = []
    for entry in correlations:
        if not isinstance(entry, dict):
            continue
        if entry.get("platform") != "weixin":
            continue
        if str(entry.get("status") or "").strip() != "active":
            continue
        if not any(
            identifiers[field] is not None and str(entry.get(field) or "").strip() == identifiers[field]
            for field in identifiers
        ):
            continue
        matched_rows.append(entry)

    if not matched_rows:
        return None

    owner_ids = {
        str(entry.get("owner_user_id") or "").strip()
        for entry in matched_rows
        if str(entry.get("owner_user_id") or "").strip()
    }
    if len(owner_ids) > 1:
        raise ValueError("weixin owner resolution conflict: identifiers map to multiple active owners")

    return matched_rows[0]


def resolve_feishu_owner_correlation(
    *,
    open_id: str | None = None,
    union_id: str | None = None,
    chat_id: str | None = None,
    platform_session_key: str | None = None,
) -> dict[str, Any] | None:
    identity_identifiers = {
        "linked_external_user_id": _normalize_optional_text(open_id),
        "linked_external_union_id": _normalize_optional_text(union_id),
    }
    session_identifiers = {
        "chat_id": _normalize_optional_text(chat_id),
        "platform_session_key": _normalize_optional_text(platform_session_key),
    }
    if not any(identity_identifiers.values()) and not any(session_identifiers.values()):
        raise ValueError("feishu owner resolution requires at least one identifier")

    _, correlations = _load_user_state()
    matched_rows: list[dict[str, Any]] = []
    for entry in correlations:
        if not isinstance(entry, dict):
            continue
        if entry.get("platform") != "feishu":
            continue
        if str(entry.get("status") or "").strip() != "active":
            continue

        if any(identity_identifiers.values()):
            if not any(
                value is not None and str(entry.get(field) or "").strip() == value
                for field, value in identity_identifiers.items()
            ):
                continue
            if any(
                value is not None
                and str(entry.get(field) or "").strip()
                and str(entry.get(field) or "").strip() != value
                for field, value in session_identifiers.items()
            ):
                continue
        elif not any(
            value is not None and str(entry.get(field) or "").strip() == value
            for field, value in session_identifiers.items()
        ):
            continue

        matched_rows.append(entry)

    if not matched_rows:
        return None

    owner_ids = {
        str(entry.get("owner_user_id") or "").strip()
        for entry in matched_rows
        if str(entry.get("owner_user_id") or "").strip()
    }
    if len(owner_ids) > 1:
        raise ValueError("feishu owner resolution conflict: identifiers map to multiple active owners")

    return matched_rows[0]


def get_active_weixin_correlation_for_user(user_id: str) -> dict[str, Any] | None:
    owner_id = _normalize_optional_text(user_id)
    if owner_id is None:
        return None

    _, correlations = _load_user_state()
    matched_rows: list[dict[str, Any]] = []
    for entry in correlations:
        if not isinstance(entry, dict):
            continue
        if entry.get("platform") != "weixin":
            continue
        if str(entry.get("status") or "").strip() != "active":
            continue
        if str(entry.get("owner_user_id") or "").strip() != owner_id:
            continue
        matched_rows.append(entry)

    if not matched_rows:
        return None
    if len(matched_rows) > 1:
        raise ValueError("multiple active Weixin correlations found for owner")
    return matched_rows[0]


def unlink_weixin_owner(*, user_id: str) -> dict[str, Any]:
    """Revoke a user's Weixin association while retaining inactive link history."""
    with _WRITE_LOCK, identity_write_transaction():
        users, correlations = _load_user_state()
        user = users.get(user_id)
        if not isinstance(user, dict):
            raise KeyError(user_id)
        now = _now_iso()
        user["weixin_user_id"] = None
        user["updated_at"] = now
        for entry in correlations:
            if (entry.get("platform") == "weixin"
                    and entry.get("owner_user_id") == user_id
                    and str(entry.get("status") or "active") == "active"):
                entry["status"] = "unlinked"
                entry["updated_at"] = now
        _save_user_state(users, correlations)
        delete_weixin_runtime_accounts_for_user(user_id)
        return user


def upsert_weixin_owner(
    *,
    account_id: str,
    external_user_id: str,
    bind_user_id: str | None = None,
) -> dict[str, Any]:
    account_id = account_id.strip()
    external_user_id = external_user_id.strip()
    if not account_id and not external_user_id:
        raise ValueError("weixin owner correlation requires account_id or external_user_id")

    with _WRITE_LOCK, identity_write_transaction():
        users, correlations = _load_user_state()

        binding_user = users.get(bind_user_id) if bind_user_id else None
        if bind_user_id and not isinstance(binding_user, dict):
            raise ValueError("Binding account no longer exists")
        if binding_user and binding_user.get("weixin_user_id") not in (None, "", external_user_id):
            raise ValueError("This account already has a different Weixin identity")
        if bind_user_id:
            for entry in correlations:
                if entry.get("platform") == "weixin" and str(entry.get("status") or "active") == "active" and (
                    (account_id and entry.get("linked_account_id") == account_id)
                    or (external_user_id and entry.get("linked_external_user_id") == external_user_id)
                ) and entry.get("owner_user_id") != bind_user_id:
                    raise ValueError("This Weixin identity is already linked to another account")

        account_match: dict[str, Any] | None = None
        external_match: dict[str, Any] | None = None
        for entry in correlations:
            if entry.get("platform") != "weixin" or str(entry.get("status") or "active") != "active":
                continue
            if account_id and entry.get("linked_account_id") == account_id:
                account_match = entry
            if external_user_id and entry.get("linked_external_user_id") == external_user_id:
                external_match = entry

        existing = account_match or external_match
        if (
            account_match is not None
            and external_match is not None
            and str(account_match.get("owner_user_id") or "").strip()
            != str(external_match.get("owner_user_id") or "").strip()
        ):
            raise ValueError("weixin owner identity conflict: account_id and external_user_id map to different owners")

        if existing is not None:
            existing["linked_account_id"] = account_id or existing.get("linked_account_id")
            existing["linked_external_user_id"] = external_user_id or existing.get("linked_external_user_id")
            existing["status"] = "active"
            existing["updated_at"] = _now_iso()
            owner_user_id = str(existing.get("owner_user_id") or "").strip()
            user = users.get(owner_user_id)
            if not isinstance(user, dict):
                # Legacy auth state can contain orphaned gateway correlations
                # whose owner user row was removed or never migrated. Repair
                # those rows deterministically and recreate a fresh owner.
                repaired_correlations: list[dict[str, Any]] = []
                for entry in correlations:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("platform") != "weixin":
                        repaired_correlations.append(entry)
                        continue
                    if str(entry.get("owner_user_id") or "").strip() != owner_user_id:
                        repaired_correlations.append(entry)
                        continue
                    linked_account_id = str(entry.get("linked_account_id") or "").strip()
                    linked_external_user_id = str(entry.get("linked_external_user_id") or "").strip()
                    if account_id and linked_account_id == account_id:
                        continue
                    if external_user_id and linked_external_user_id == external_user_id:
                        continue
                    repaired_correlations.append(entry)
                correlations = retain_snapshot(correlations, repaired_correlations)
                existing = None
            else:
                if external_user_id:
                    bound_external_user_id = str(user.get("weixin_user_id") or "").strip()
                    if bound_external_user_id and bound_external_user_id != external_user_id:
                        raise ValueError("weixin owner identity conflict: semantier user already bound to another Weixin owner")
                    user["weixin_user_id"] = external_user_id
                user["updated_at"] = _now_iso()
                users[owner_user_id] = user
                _save_user_state(users, correlations)
                if is_postgresql() and binding_user is None:
                    # A retained QR identity can predate personal billing, just
                    # like a retained password identity. Reuse the canonical
                    # repair owner inside this identity transaction.
                    from agents.auth_db_postgres import provision_personal_signup
                    provision_personal_signup(user, insert_user=False)
                return user

        owner_user_id = bind_user_id or uuid.uuid4().hex
        workspace_id = str(binding_user["workspace_id"]) if binding_user else owner_user_id
        preferred = external_user_id or account_id or owner_user_id
        workspace_slug = f"weixin-{_slugify(preferred)[-12:]}"
        now = _now_iso()
        ensure_workspace_paths(workspace_id)
        user = {
            "user_id": owner_user_id,
            "workspace_id": workspace_id,
            "workspace_slug": workspace_slug,
            "name": external_user_id or account_id or workspace_slug,
            "display_name": None,
            "password_login_name": None,
            "password_hash": None,
            "profile_completed_at": None,
            "auth_provider": "weixin",
            "weixin_user_id": external_user_id or None,
            "feishu_open_id": "",
            "created_at": now,
            "updated_at": now,
        }
        if binding_user is not None:
            user = {**binding_user, "weixin_user_id": external_user_id, "updated_at": now}
        users[owner_user_id] = user
        correlations.append(
            {
                "platform": "weixin",
                "owner_user_id": owner_user_id,
                "owner_workspace_id": workspace_id,
                "linked_external_user_id": external_user_id or None,
                "linked_external_union_id": None,
                "linked_account_id": account_id or None,
                "platform_session_key": None,
                "chat_id": None,
                "status": "active",
                "created_at": now,
                "updated_at": now,
            }
        )
        _save_user_state(users, correlations)
        if is_postgresql() and binding_user is None:
            from agents.auth_db_postgres import provision_personal_signup
            provision_personal_signup(user, insert_user=False)
        return user


def create_weixin_login_state(
    *,
    qrcode: str,
    qrcode_url: str,
    qr_scan_data: str,
    base_url: str,
    bot_type: str,
    bind_user_id: str | None = None,
    prelogin_session_nonce_hash: str | None = None,
    request_user_agent_hash: str | None = None,
) -> dict[str, Any]:
    with _WRITE_LOCK, identity_write_transaction():
        _ensure_auth_store()
        states = load_weixin_login_states()
        if not isinstance(states, dict):
            states = {}
        states = _prune_weixin_login_states(states)
        state = secrets.token_urlsafe(24)
        now = time.time()
        payload = {
            "state": state,
            "qrcode": qrcode,
            "qrcode_url": qrcode_url,
            "qr_scan_data": qr_scan_data,
            "base_url": base_url,
            "redirect_base_url": "",
            "bot_type": bot_type,
            "status": "wait",
            "bind_user_id": bind_user_id,
            "prelogin_session_nonce_hash": prelogin_session_nonce_hash or "",
            "request_user_agent_hash": request_user_agent_hash or "",
            "consumed_at": None,
            "consumed_reason": "",
            "created_at": now,
            "updated_at": now,
        }
        states[state] = payload
        save_weixin_login_states(states)
        return payload


def get_weixin_login_state(state: str) -> dict[str, Any] | None:
    with _WRITE_LOCK, identity_write_transaction():
        _ensure_auth_store()
        states = load_weixin_login_states()
        if not isinstance(states, dict):
            return None
        pruned = _prune_weixin_login_states(states)
        if pruned != states:
            save_weixin_login_states(pruned)
        payload = pruned.get(state)
        return payload if isinstance(payload, dict) else None


def update_weixin_login_state(state: str, **changes: Any) -> dict[str, Any] | None:
    with _WRITE_LOCK, identity_write_transaction():
        _ensure_auth_store()
        states = load_weixin_login_states()
        if not isinstance(states, dict):
            return None
        states = _prune_weixin_login_states(states)
        payload = states.get(state)
        if not isinstance(payload, dict):
            return None
        payload.update(changes)
        payload["updated_at"] = time.time()
        states[state] = payload
        save_weixin_login_states(states)
        return payload


def consume_weixin_login_state(
    state: str,
    *,
    status: str,
    consumed_reason: str,
    **changes: Any,
) -> dict[str, Any] | None:
    with _WRITE_LOCK, identity_write_transaction():
        _ensure_auth_store()
        states = load_weixin_login_states()
        if not isinstance(states, dict):
            return None
        states = _prune_weixin_login_states(states)
        payload = states.get(state)
        if not isinstance(payload, dict):
            return None
        if payload.get("consumed_at"):
            return payload
        payload.update(changes)
        payload["status"] = status
        payload["consumed_reason"] = consumed_reason
        payload["consumed_at"] = time.time()
        payload["updated_at"] = time.time()
        states[state] = payload
        save_weixin_login_states(states)
        return payload


def delete_weixin_login_state(state: str) -> None:
    with _WRITE_LOCK, identity_write_transaction():
        _ensure_auth_store()
        states = load_weixin_login_states()
        if not isinstance(states, dict):
            return
        states = _prune_weixin_login_states(states)
        if state in states:
            del states[state]
            save_weixin_login_states(states)


def _prune_feishu_link_states(states: dict[str, Any]) -> dict[str, Any]:
    cutoff = time.time() - FEISHU_LINK_STATE_TTL_SECONDS
    pruned: dict[str, Any] = {}
    for key, value in states.items():
        if not isinstance(value, dict):
            continue
        status = str(value.get("status") or "").strip()
        created_at = value.get("created_at")
        created_ts = float(created_at) if isinstance(created_at, (int, float)) else 0.0
        if status == "linked" and value.get("link_mode") != "bot_message":
            continue
        if created_ts and created_ts < cutoff:
            continue
        pruned[key] = value
    return retain_snapshot(states, pruned)


def create_feishu_link_state(
    *,
    authorize_url: str,
    device_code: str,
    domain: str,
    interval: int,
    expire_in: int,
    owner_user_id: str | None = None,
) -> dict[str, Any]:
    with _WRITE_LOCK, identity_write_transaction():
        _ensure_auth_store()
        states = load_feishu_link_states()
        if not isinstance(states, dict):
            states = {}
        states = _prune_feishu_link_states(states)
        state = secrets.token_urlsafe(24)
        now = time.time()
        payload = {
            "state": state,
            "authorize_url": authorize_url,
            "device_code": device_code,
            "domain": domain,
            "interval": interval,
            "expire_in": expire_in,
            "owner_user_id": _normalize_optional_text(owner_user_id),
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }
        states[state] = payload
        save_feishu_link_states(states)
        return payload


def get_feishu_link_state(state: str) -> dict[str, Any] | None:
    with _WRITE_LOCK, identity_write_transaction():
        _ensure_auth_store()
        states = load_feishu_link_states()
        if not isinstance(states, dict):
            return None
        pruned = _prune_feishu_link_states(states)
        if pruned != states:
            save_feishu_link_states(pruned)
        payload = pruned.get(state)
        return payload if isinstance(payload, dict) else None


def update_feishu_link_state(state: str, **changes: Any) -> dict[str, Any] | None:
    with _WRITE_LOCK, identity_write_transaction():
        _ensure_auth_store()
        states = load_feishu_link_states()
        if not isinstance(states, dict):
            return None
        states = _prune_feishu_link_states(states)
        payload = states.get(state)
        if not isinstance(payload, dict):
            return None
        payload.update(changes)
        payload["updated_at"] = time.time()
        states[state] = payload
        save_feishu_link_states(states)
        return payload


def delete_feishu_link_state(state: str) -> None:
    with _WRITE_LOCK, identity_write_transaction():
        _ensure_auth_store()
        states = load_feishu_link_states()
        if not isinstance(states, dict):
            return
        states = _prune_feishu_link_states(states)
        if state in states:
            del states[state]
            save_feishu_link_states(states)


def consume_feishu_message_link(*, state: str, app_id: str, open_id: str, union_id: str | None = None) -> dict[str, Any]:
    """Correlate the signed-in owner with a sender authenticated by the configured bot."""
    from agents.auth_db import claim_feishu_message_link

    if not str(open_id or "").startswith("ou_"):
        raise ValueError("Feishu sender identity is missing")
    with _WRITE_LOCK, identity_write_transaction():
        _ensure_auth_store()
        payload = claim_feishu_message_link(state, app_id, time.time())
        if payload is None:
            raise ValueError("Link code is invalid, expired, already used, or belongs to another bot")
        try:
            owner = str(payload.get("owner_user_id") or "")
            user = load_users().get(owner)
            if not user:
                raise ValueError("Link owner is unavailable")
            existing = str(user.get("feishu_open_id") or "")
            if existing and existing != open_id:
                raise ValueError("Unlink the existing Feishu account before linking another")
            user = link_feishu_open_id(user_id=owner, open_id=open_id, union_id=union_id)
        except (KeyError, ValueError) as exc:
            message = str(exc) if isinstance(exc, ValueError) else "Account link failed. Generate a new code in Settings."
            update_feishu_link_state(state, status="failed", message=message)
            raise
        update_feishu_link_state(state, status="linked", feishu_open_id=open_id)
        return user


def link_feishu_open_id(*, user_id: str, open_id: str, union_id: str | None = None) -> dict[str, Any]:
    normalized_open_id = _normalize_optional_text(open_id)
    normalized_union_id = _normalize_optional_text(union_id)
    if normalized_open_id is None:
        raise ValueError("feishu open_id is required")

    with _WRITE_LOCK, identity_write_transaction():
        users, correlations = _load_user_state()
        user = users.get(user_id)
        if not isinstance(user, dict):
            raise KeyError(user_id)

        workspace_id = str(user.get("workspace_id") or "").strip()
        if not workspace_id:
            raise ValueError("feishu link requires workspace_id")

        for candidate_id, candidate in users.items():
            if candidate_id == user_id or not isinstance(candidate, dict):
                continue
            if str(candidate.get("feishu_open_id") or "").strip() == normalized_open_id:
                raise ValueError("feishu open_id already linked to another workspace")
            if (
                normalized_union_id
                and str(candidate.get("feishu_union_id") or "").strip() == normalized_union_id
            ):
                raise ValueError("feishu union_id already linked to another workspace")

        for entry in correlations:
            if not isinstance(entry, dict) or entry.get("platform") != "feishu":
                continue
            if str(entry.get("status") or "").strip() != "active":
                continue
            if str(entry.get("owner_user_id") or "").strip() == user_id:
                continue
            if str(entry.get("linked_external_user_id") or "").strip() == normalized_open_id:
                raise ValueError("feishu open_id already linked to another workspace")
            if (
                normalized_union_id
                and str(entry.get("linked_external_union_id") or "").strip() == normalized_union_id
            ):
                raise ValueError("feishu union_id already linked to another workspace")

        user["feishu_open_id"] = normalized_open_id
        if normalized_union_id:
            user["feishu_union_id"] = normalized_union_id
        user["updated_at"] = _now_iso()
        users[user_id] = user

        now = _now_iso()
        replaced = False
        for entry in correlations:
            if not isinstance(entry, dict) or entry.get("platform") != "feishu":
                continue
            if str(entry.get("owner_user_id") or "").strip() != user_id:
                continue
            entry["owner_workspace_id"] = workspace_id
            entry["linked_external_user_id"] = normalized_open_id
            entry["linked_external_union_id"] = normalized_union_id
            entry["status"] = "active"
            entry["updated_at"] = now
            replaced = True
        if not replaced:
            correlations.append(
                {
                    "platform": "feishu",
                    "owner_user_id": user_id,
                    "owner_workspace_id": workspace_id,
                    "linked_external_user_id": normalized_open_id,
                    "linked_external_union_id": normalized_union_id,
                    "linked_account_id": None,
                    "chat_id": None,
                    "platform_session_key": None,
                    "status": "active",
                    "created_at": now,
                    "updated_at": now,
                }
            )
        _save_user_state(users, correlations)
        return user


def unlink_feishu_open_id(*, user_id: str) -> dict[str, Any]:
    with _WRITE_LOCK, identity_write_transaction():
        users, correlations = _load_user_state()
        user = users.get(user_id)
        if not isinstance(user, dict):
            raise KeyError(user_id)

        previous_open_id = _normalize_optional_text(str(user.get("feishu_open_id") or ""))
        previous_union_id = _normalize_optional_text(str(user.get("feishu_union_id") or ""))
        user["feishu_open_id"] = ""
        user["feishu_union_id"] = ""
        user["updated_at"] = _now_iso()
        users[user_id] = user

        now = _now_iso()
        for entry in correlations:
            if not isinstance(entry, dict) or entry.get("platform") != "feishu":
                continue
            if str(entry.get("owner_user_id") or "").strip() != user_id:
                continue
            if str(entry.get("status") or "").strip() != "active":
                continue
            entry_open_id = _normalize_optional_text(str(entry.get("linked_external_user_id") or ""))
            entry_union_id = _normalize_optional_text(str(entry.get("linked_external_union_id") or ""))
            if (
                previous_open_id is not None
                and entry_open_id is not None
                and entry_open_id != previous_open_id
            ):
                continue
            if (
                previous_union_id is not None
                and entry_union_id is not None
                and entry_union_id != previous_union_id
            ):
                continue
            entry["status"] = "unlinked"
            entry["updated_at"] = now

        _save_user_state(users, correlations)
        return user


def _workspace_feishu_bot_config_path(workspace_id: str) -> Path:
    """Legacy workspace file path retained only for cleanup of stale files."""
    runtime_home = _env_workspace_root_path(workspace_id) or workspace_runtime_home_path(workspace_id)
    return runtime_home / "feishu_bot_config.json"


def _delete_legacy_workspace_feishu_bot_config(workspace_id: str) -> None:
    path = _workspace_feishu_bot_config_path(workspace_id)
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def read_workspace_feishu_bot_config(workspace_id: str) -> dict[str, Any] | None:
    """Read workspace-specific Feishu bot configuration (credentials)."""
    _ensure_auth_store()
    normalized_workspace_id = str(workspace_id or "").strip()
    if not normalized_workspace_id:
        return None
    payload = get_feishu_bot_config_for_workspace(normalized_workspace_id)
    if isinstance(payload, dict) and payload:
        return payload

    return None


def save_workspace_feishu_bot_config(
    workspace_id: str,
    *,
    app_id: str,
    app_secret: str,
    domain: str,
    connection_mode: str,
    bot_open_id: str | None = None,
    bot_name: str | None = None,
    home_channel: str | None = None,
    validated_at: str | None = None,
    last_error: str | None = None,
    manager_user_id: str | None = None,
) -> dict[str, Any]:
    """Save workspace-specific Feishu bot configuration."""
    _ensure_auth_store()
    normalized_workspace_id = str(workspace_id or "").strip()
    if not normalized_workspace_id:
        raise ValueError("feishu bot config requires workspace_id")
    owner_user_id = _normalize_optional_text(manager_user_id)
    if owner_user_id is None:
        user = get_user_record_by_workspace_id(normalized_workspace_id)
        if isinstance(user, dict):
            owner_user_id = _normalize_optional_text(str(user.get("user_id") or ""))
    if owner_user_id is None:
        raise ValueError("feishu bot config requires manager_user_id")
    now = _now_iso()
    existing = get_feishu_bot_config_for_workspace(normalized_workspace_id)
    if not isinstance(existing, dict):
        existing = {}
    created_at = str(existing.get("created_at") or now) if isinstance(existing, dict) else now
    payload = {
        "app_id": app_id.strip(),
        "app_secret": app_secret.strip(),
        "domain": (domain or "feishu").strip() or "feishu",
        "connection_mode": (connection_mode or "websocket").strip() or "websocket",
        "bot_open_id": _normalize_optional_text(bot_open_id),
        "bot_name": _normalize_optional_text(bot_name),
        "home_channel": (
            _normalize_optional_text(home_channel)
            or _normalize_optional_text(str(existing.get("home_channel") or ""))
        ),
        "manager_user_id": owner_user_id,
        "owner_user_id": owner_user_id,
        "owner_workspace_id": normalized_workspace_id,
        "created_at": created_at,
        "updated_at": now,
        "saved_at": now,
        "validated_at": validated_at,
        "last_error": _normalize_optional_text(last_error),
    }
    save_feishu_bot_config(payload)
    _delete_legacy_workspace_feishu_bot_config(normalized_workspace_id)
    return payload


def delete_workspace_feishu_bot_config(workspace_id: str) -> None:
    """Delete workspace-specific Feishu bot configuration."""
    _ensure_auth_store()
    normalized_workspace_id = str(workspace_id or "").strip()
    if not normalized_workspace_id:
        return
    user = get_user_record_by_workspace_id(normalized_workspace_id)
    if isinstance(user, dict):
        owner_user_id = _normalize_optional_text(str(user.get("user_id") or ""))
        if owner_user_id is not None:
            delete_feishu_bot_config_for_user(owner_user_id)
    _delete_legacy_workspace_feishu_bot_config(normalized_workspace_id)
