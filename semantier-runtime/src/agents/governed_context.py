from __future__ import annotations

from datetime import date, datetime, time, timezone
import importlib
import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from agents.temporal_resolution import current_turn_anchor_utc, resolve_relative_period_context
from eos.artifact_hashing import content_hash

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "agents"
_PROMPT_CATALOG_PATH = _PROMPTS_DIR / "governed_context.json"
_DIGITAL_CFO_METRICS_PATH = _PROMPTS_DIR / "digital_cfo_metrics.json"


def _load_prompt_catalog() -> dict[str, Any]:
    loaded = json.loads(_PROMPT_CATALOG_PATH.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _load_metrics_catalog() -> dict[str, Any]:
    loaded = json.loads(_DIGITAL_CFO_METRICS_PATH.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


_PROMPT_CATALOG = _load_prompt_catalog()
_METRICS_CATALOG = _load_metrics_catalog()


def _prompt_marker_group(name: str) -> tuple[str, ...]:
    return tuple(str(item) for item in _PROMPT_CATALOG.get(name) or ())


_ANALYTICS_QUERY_MARKERS = _prompt_marker_group("analytics_query_markers")
_BUSINESS_INTELLIGENCE_MARKERS = _prompt_marker_group("business_intelligence_markers")
_MANAGERIAL_ACCOUNTING_MARKERS = _prompt_marker_group("managerial_accounting_markers")
_VARIANCE_ANALYSIS_MARKERS = _prompt_marker_group("variance_analysis_markers")
_OPERATIONAL_BI_MARKERS = _prompt_marker_group("operational_bi_markers")
_EXPLICIT_COMPANY_MARKERS = tuple(str(item) for item in _PROMPT_CATALOG.get("explicit_company_markers") or ())
_THREE_YEAR_QUERY_MARKERS = tuple(str(item) for item in _PROMPT_CATALOG.get("three_year_query_markers") or ())
_RELATIVE_PERIOD_QUERY_MARKERS = tuple(
    str(item) for item in _PROMPT_CATALOG.get("relative_period_query_markers") or ()
)
_THREE_YEAR_GUIDANCE = str(_PROMPT_CATALOG.get("three_year_guidance") or "").strip()
_RELATIVE_PERIOD_GUIDANCE_TEMPLATE = str(
    _PROMPT_CATALOG.get("relative_period_guidance_template") or ""
).strip()
_METRIC_INTENT_ALIASES = {
    str(key).strip().lower(): str(value).strip()
    for key, value in (_PROMPT_CATALOG.get("metric_intent_aliases") or {}).items()
    if str(key).strip() and str(value).strip()
}
_DIMENSION_MARKERS = {
    str(key).strip(): tuple(str(item).strip().lower() for item in value or () if str(item).strip())
    for key, value in (_PROMPT_CATALOG.get("dimension_markers") or {}).items()
    if str(key).strip()
}
_METRIC_CONTEXT_TEMPLATE = str(_METRICS_CATALOG.get("metric_context_template") or "").strip()
_DIGITAL_CFO_METRICS = {
    str(key).strip(): value
    for key, value in (_METRICS_CATALOG.get("metrics") or {}).items()
    if str(key).strip() and isinstance(value, dict)
}


def _gateway_identity_module():
    agents_pkg = importlib.import_module("agents")
    module = getattr(agents_pkg, "gateway_identity", None)
    if module is not None:
        return module
    return importlib.import_module("agents.gateway_identity")


def _load_prompt_template(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


def _default_analytics_sample_profile() -> dict[str, Any]:
    return _gateway_identity_module().get_default_sample_organization_profile() or {}


def resolve_user_id_for_workspace(workspace_id: str | None) -> str | None:
    normalized = (workspace_id or "").strip()
    if not normalized:
        return None
    record = _gateway_identity_module().get_user_record_by_workspace_id(normalized)
    if not isinstance(record, dict):
        return None
    user_id = str(record.get("user_id") or "").strip()
    return user_id or None


def _is_analytics_query(message: str) -> bool:
    lowered = (message or "").strip().lower()
    if not lowered:
        return False
    combined_markers = (
        _ANALYTICS_QUERY_MARKERS
        + _BUSINESS_INTELLIGENCE_MARKERS
        + _MANAGERIAL_ACCOUNTING_MARKERS
        + _VARIANCE_ANALYSIS_MARKERS
        + _OPERATIONAL_BI_MARKERS
    )
    return any(marker in lowered for marker in combined_markers)


def is_analytics_query_message(message: str) -> bool:
    """Return whether a user message should be treated as analytics intent.

    This is a small public wrapper so runtime layers can reuse the governed
    marker contract from `governed_context.json` without duplicating marker
    lists in gateway code.
    """
    return _is_analytics_query(message)


def _mentions_explicit_company_selection(message: str) -> bool:
    lowered = (message or "").strip().lower()
    return bool(lowered) and any(marker in lowered for marker in _EXPLICIT_COMPANY_MARKERS)


def _is_three_year_query(message: str) -> bool:
    lowered = (message or "").strip().lower()
    return bool(lowered) and any(marker in lowered for marker in _THREE_YEAR_QUERY_MARKERS)


def _has_relative_period_query(message: str) -> bool:
    lowered = (message or "").strip().lower()
    return bool(lowered) and any(marker in lowered for marker in _RELATIVE_PERIOD_QUERY_MARKERS)


def _matched_metric_contract_ids(message: str) -> list[str]:
    lowered = (message or "").strip().lower()
    if not lowered:
        return []
    matched: list[str] = []
    for alias, metric_id in _METRIC_INTENT_ALIASES.items():
        if alias in lowered and metric_id in _DIGITAL_CFO_METRICS and metric_id not in matched:
            matched.append(metric_id)
    for metric_id, metric in _DIGITAL_CFO_METRICS.items():
        for label in metric.get("labels") or ():
            normalized = str(label).strip().lower()
            if normalized and normalized in lowered and metric_id not in matched:
                matched.append(metric_id)
                break
    return matched[:2]


def _detected_dimensions(message: str) -> list[str]:
    lowered = (message or "").strip().lower()
    if not lowered:
        return []
    matched: list[str] = []
    for dimension, markers in _DIMENSION_MARKERS.items():
        if any(marker in lowered for marker in markers):
            matched.append(dimension)
    return matched


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _quarter(month: int) -> int:
    return ((month - 1) // 3) + 1


def _month_shift(value: date, offset: int) -> date:
    month_index = (value.year * 12 + (value.month - 1)) + offset
    year = month_index // 12
    month = (month_index % 12) + 1
    return date(year, month, 1)


def _quarter_start(value: date) -> date:
    return date(value.year, ((_quarter(value.month) - 1) * 3) + 1, 1)


def _year_start(value: date) -> date:
    return date(value.year, 1, 1)


def _relative_period_guidance(
    as_of: date | None = None,
    *,
    anchor_utc: datetime | None = None,
    timezone_name: str = "Asia/Shanghai",
) -> str:
    local_tz = ZoneInfo(timezone_name)
    anchor = anchor_utc or current_turn_anchor_utc()
    if as_of is not None:
        local_midnight = datetime.combine(as_of, time.min, tzinfo=local_tz)
        anchor = local_midnight.astimezone(timezone.utc)
    elif anchor is None:
        today = _utc_today()
        local_midnight = datetime.combine(today, time.min, tzinfo=local_tz)
        anchor = local_midnight.astimezone(timezone.utc)
    context = resolve_relative_period_context(
        anchor_utc=anchor,
        timezone_name=timezone_name,
    )
    if not _RELATIVE_PERIOD_GUIDANCE_TEMPLATE:
        return ""
    return _RELATIVE_PERIOD_GUIDANCE_TEMPLATE.format(**context.__dict__)


def _metric_contract_system_message(user_message: str) -> str | None:
    if not _is_analytics_query(user_message):
        return None
    metric_ids = _matched_metric_contract_ids(user_message)
    if not metric_ids or not _METRIC_CONTEXT_TEMPLATE:
        return None
    metric_lines: list[str] = []
    for metric_id in metric_ids:
        metric = _DIGITAL_CFO_METRICS.get(metric_id) or {}
        labels = ", ".join(str(label) for label in metric.get("labels") or ())
        source_views = ", ".join(str(view) for view in metric.get("source_views") or ())
        comparison_modes = ", ".join(str(mode) for mode in metric.get("comparison_modes") or ())
        dimension_candidates = ", ".join(str(dim) for dim in metric.get("dimension_candidates") or ())
        metric_lines.append(
            "\n".join(
                [
                    f"- Metric `{metric_id}`",
                    f"  Labels: {labels}",
                    f"  Definition: {metric.get('definition') or ''}",
                    f"  Preferred views: {source_views}",
                    f"  Default grain: {metric.get('default_grain') or ''}",
                    f"  Comparison modes: {comparison_modes}",
                    f"  Candidate dimensions: {dimension_candidates}",
                    f"  Guidance: {metric.get('guidance') or ''}",
                ]
            )
        )
    dimensions = _detected_dimensions(user_message)
    dimension_guidance = (
        "Detected business dimensions in the user question: " + ", ".join(dimensions)
        if dimensions
        else "No explicit business dimension marker was detected; default to organization-level analysis unless the user specifies a project, customer, region, product, supplier, or department lens."
    )
    metric_context = _METRIC_CONTEXT_TEMPLATE.format(
        metric_contracts="\n\n".join(metric_lines),
        dimension_guidance=dimension_guidance,
    )
    return _load_prompt_template("DIGITAL_CFO_METRIC_CONTEXT.md").format(
        metric_context=metric_context,
    )


def _active_organization_context_system_message(org_context: dict[str, Any]) -> str | None:
    org_name = str(org_context.get("organization_name") or "").strip()
    if not org_name:
        return None
    return _load_prompt_template("ACTIVE_ORGANIZATION_CONTEXT.md").format(
        organization_name=org_name,
        real_company_setup_context=_real_company_setup_context(org_context),
    )


def _real_company_setup_context(org_context: dict[str, Any]) -> str:
    if str(org_context.get("dataset_type") or "").strip().upper() != "REAL":
        return ""
    try:
        from eos.real_company_setup_state import real_company_setup_status_payload
    except Exception:
        return ""
    payload = real_company_setup_status_payload(
        organization_id=str(org_context.get("organization_id") or "").strip() or None,
        dataset_type="REAL",
        active_dataset_version_id=str(org_context.get("active_dataset_version_id") or "").strip() or None,
    )
    setup_status = str(payload.get("setup_status") or "").strip()
    if not setup_status:
        return ""
    blocking_reasons = payload.get("blocking_reasons") or []
    next_actions = payload.get("next_actions") or []
    return (
        f"setup_status={setup_status}; "
        f"blocking_reasons={','.join(str(item) for item in blocking_reasons)}; "
        f"next_actions={','.join(str(item) for item in next_actions)}"
    )


def _default_analytics_sample_system_message(org_context: dict[str, Any], user_message: str) -> str | None:
    default_profile = _default_analytics_sample_profile()
    default_org_id = str(default_profile.get("organization_id") or "").strip()
    if not default_org_id:
        return None
    if str(org_context.get("organization_id") or "").strip() != default_org_id:
        return None
    if _mentions_explicit_company_selection(user_message):
        return None
    if not _is_analytics_query(user_message):
        return None
    org_name = str(org_context.get("organization_name") or "").strip() or str(
        default_profile.get("display_name") or ""
    )
    time_bounded_guidance = []
    if _is_three_year_query(user_message):
        time_bounded_guidance.append(_THREE_YEAR_GUIDANCE)
    if _has_relative_period_query(user_message):
        time_bounded_guidance.append(_relative_period_guidance())
    return _load_prompt_template("DEFAULT_ANALYTICS_SAMPLE_CONTEXT.md").format(
        organization_name=org_name,
        time_bounded_guidance=(" ".join(item.strip() for item in time_bounded_guidance if item.strip()) + " ").strip(),
    )


def _organization_tool_routing_system_message(org_context: dict[str, Any], user_message: str) -> str | None:
    if not _is_analytics_query(user_message):
        return None
    org_id = str(org_context.get("organization_id") or "").strip()
    if not org_id:
        return None
    default_org_id = str(_default_analytics_sample_profile().get("organization_id") or "").strip()
    if org_id == default_org_id:
        return None
    time_bounded_guidance = []
    if _is_three_year_query(user_message):
        time_bounded_guidance.append(_THREE_YEAR_GUIDANCE)
    if _has_relative_period_query(user_message):
        time_bounded_guidance.append(_relative_period_guidance())
    return _load_prompt_template("ORGANIZATION_TOOL_ROUTING_CONTEXT.md").format(
        organization_id=org_id,
        time_bounded_guidance=(" ".join(item.strip() for item in time_bounded_guidance if item.strip()) + " ").strip(),
    )


def _demo_dataset_system_message(org_context: dict[str, Any], user_message: str) -> str | None:
    if str(org_context.get("dataset_type") or "").strip().upper() != "DEMO":
        return None
    if not _is_analytics_query(user_message):
        return None
    org_name = str(org_context.get("organization_name") or "").strip() or "the active organization"
    return _load_prompt_template("DEMO_DATASET_CONTEXT.md").format(
        organization_name=org_name,
    )


_ACTIVATION_RESOLVER_POLICY_VERSION = "knowledge_activation_resolver.v1"
SAME_SCOPE_DUPLICATE = "SAME_SCOPE_DUPLICATE"
CROSS_SCOPE_AMBIGUITY = "CROSS_SCOPE_AMBIGUITY"
OVERRIDE_NOT_PERMITTED = "OVERRIDE_NOT_PERMITTED"
DATASET_ROLE_CONFLICT = "DATASET_ROLE_CONFLICT"
STALE_DATASET_READINESS = "STALE_DATASET_READINESS"
MIGRATION_STATE_CONFLICT = "MIGRATION_STATE_CONFLICT"

_OPTIONAL_WORKSPACE_OVERLAY_POLICIES = {
    "retrieval_only",
    "prompt_context_only",
    "query_context_only",
    "optional_context",
}


def _context_dimensions_for_control(control_level: str) -> set[str]:
    if control_level == "prompt_context":
        return {"retrieval", "prompt_context"}
    if control_level == "query_context":
        return {"query_context"}
    if control_level == "full_optional_context":
        return {"retrieval", "prompt_context", "query_context"}
    return set()


def _binding_effective(row: dict[str, Any]) -> bool:
    tier = str(row.get("semantic_tier") or "").strip()
    authority_role = str(row.get("authority_role") or "").strip()
    if tier in {"T1", "T2", "T3", "T4"} and authority_role == "binding":
        return True
    if str(row.get("source_kind") or "") == "dataset":
        usage_role = str(row.get("dataset_usage_role") or "")
        authority_level = str(row.get("dataset_authority_level") or "")
        return usage_role == "primary_analytics" and authority_level in {
            "binding",
            "governed_default",
        }
    return False


def _metadata_evidence_payload(
    row: dict[str, Any],
    *,
    organization_id: str,
) -> dict[str, Any]:
    source_kind = str(row.get("source_kind") or "").strip()
    source_hash = str(row.get("source_hash") or "").strip() or None
    source_ref = str(row.get("source_ref") or "").strip() or None
    source_version = str(row.get("source_version") or "").strip() or None
    if source_kind == "dataset":
        dataset_asset_version_id = str(row.get("dataset_asset_version_id") or "").strip()
        return {
            "metadata_plane": "semantier_native",
            "asset_identity": dataset_asset_version_id,
            "asset_kind": "dataset",
            "owner_ref": f"organization:{organization_id}",
            "domain": str(row.get("dataset_type") or "").strip() or "dataset",
            "version_ref": row.get("dataset_version_id") or row.get("dataset_key") or dataset_asset_version_id,
            "lifecycle_state": row.get("source_lifecycle_status"),
            "classification_state": row.get("semantic_tier"),
            "contract_state": "contract_recorded" if row.get("schema_contract_hash") else "contract_missing",
            "lineage_state": "lineage_recorded" if row.get("lineage_manifest_hash") else "lineage_missing",
            "quality_state": row.get("dataset_readiness_status") or "unknown",
            "schema_contract_hash": row.get("schema_contract_hash"),
            "lineage_manifest_hash": row.get("lineage_manifest_hash"),
            "quality_evidence_hash": row.get("quality_evidence_hash"),
            "source_anchor_refs": [item for item in [dataset_asset_version_id, source_ref] if item],
            "source_hash": source_hash,
        }
    return {
        "metadata_plane": "semantier_native",
        "asset_identity": str(row.get("knowledge_source_id") or "").strip(),
        "asset_kind": source_kind or "knowledge_source",
        "owner_ref": f"organization:{organization_id}",
        "domain": row.get("semantic_tier"),
        "version_ref": source_version or source_ref,
        "lifecycle_state": row.get("source_lifecycle_status"),
        "classification_state": row.get("semantic_tier"),
        "contract_state": "source_contract_recorded" if source_hash else "source_contract_missing",
        "lineage_state": "source_lineage_recorded",
        "quality_state": "metadata_present",
        "schema_contract_hash": None,
        "lineage_manifest_hash": None,
        "quality_evidence_hash": None,
        "source_anchor_refs": [item for item in [source_ref, source_version] if item],
        "source_hash": source_hash,
    }


def _migration_state_conflict_evidence(
    migration_state: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not migration_state:
        return []
    migration_status = str(migration_state.get("migration_status") or "").strip()
    if migration_status not in {"activation_written", "complete"}:
        return []
    expected_activation_id = str(migration_state.get("activation_id") or "").strip()
    if not expected_activation_id:
        return [
            {
                "activation_id": None,
                "reason_code": MIGRATION_STATE_CONFLICT,
                "edge_type": "blocked_by_resolver",
                "migration_status": migration_status,
            }
        ]
    active_ids = {str(row.get("activation_id") or "").strip() for row in rows}
    if expected_activation_id in active_ids:
        return []
    return [
        {
            "activation_id": row.get("activation_id"),
            "reason_code": MIGRATION_STATE_CONFLICT,
            "edge_type": "blocked_by_resolver",
            "migration_activation_id": expected_activation_id,
            "migration_status": migration_status,
        }
        for row in rows
    ] or [
        {
            "activation_id": expected_activation_id,
            "reason_code": MIGRATION_STATE_CONFLICT,
            "edge_type": "blocked_by_resolver",
            "migration_status": migration_status,
        }
    ]


def _resolver_overlay_key(row: dict[str, Any]) -> tuple[str, ...]:
    source_kind = str(row.get("source_kind") or "").strip()
    if source_kind == "dataset":
        return ("dataset", str(row.get("dataset_usage_role") or "").strip())
    return (
        "source",
        str(row.get("knowledge_source_id") or "").strip(),
        str(row.get("semantic_tier") or "").strip(),
        str(row.get("authority_role") or "").strip(),
    )


def _resolver_exclusion(
    row: dict[str, Any],
    *,
    reason_code: str,
    edge_type: str = "blocked_by_resolver",
) -> dict[str, Any]:
    return {
        "activation_id": str(row.get("activation_id") or "").strip(),
        "reason_code": reason_code,
        "edge_type": edge_type,
        "scope_key": str(row.get("scope_key") or "").strip(),
        "knowledge_source_id": str(row.get("knowledge_source_id") or "").strip(),
        "dataset_usage_role": row.get("dataset_usage_role"),
    }


def _resolve_overlay_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_resolver_overlay_key(row), []).append(row)

    resolved: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for key, candidates in grouped.items():
        by_scope: dict[str, list[dict[str, Any]]] = {}
        for row in candidates:
            by_scope.setdefault(str(row.get("scope_key") or "").strip(), []).append(row)
        duplicate_scopes = [
            scope_rows for scope_rows in by_scope.values() if len(scope_rows) > 1
        ]
        if duplicate_scopes:
            reason_code = DATASET_ROLE_CONFLICT if key[0] == "dataset" else SAME_SCOPE_DUPLICATE
            for scope_rows in duplicate_scopes:
                excluded.extend(
                    _resolver_exclusion(row, reason_code=reason_code) for row in scope_rows
                )
            for scope, scope_rows in by_scope.items():
                if len(scope_rows) == 1:
                    excluded.append(
                        _resolver_exclusion(scope_rows[0], reason_code=CROSS_SCOPE_AMBIGUITY)
                    )
            continue

        organization_rows = by_scope.get("organization") or []
        workspace_rows = [
            row
            for scope, scope_rows in by_scope.items()
            if scope != "organization"
            for row in scope_rows
        ]
        if organization_rows and workspace_rows:
            organization_row = organization_rows[0]
            workspace_row = workspace_rows[0]
            workspace_policy = str(workspace_row.get("workspace_override_policy") or "none").strip()
            if workspace_policy == "governed_replacement":
                resolved.append(workspace_row)
                excluded.append(
                    _resolver_exclusion(
                        organization_row,
                        reason_code="GOVERNED_REPLACEMENT",
                        edge_type="excluded_by_policy",
                    )
                )
                continue
            if (
                workspace_policy in _OPTIONAL_WORKSPACE_OVERLAY_POLICIES
                and not _binding_effective(organization_row)
                and not _binding_effective(workspace_row)
            ):
                resolved.extend(candidates)
                continue
            reason_code = (
                CROSS_SCOPE_AMBIGUITY
                if workspace_policy == "none"
                else OVERRIDE_NOT_PERMITTED
            )
            excluded.extend(
                _resolver_exclusion(row, reason_code=reason_code) for row in candidates
            )
            continue

        resolved.extend(candidates)

    return resolved, excluded


def resolve_governed_activation_context(
    *,
    user_id: str | None = None,
    workspace_id: str | None = None,
    effective_at: str | None = None,
    persist_snapshot: bool = False,
) -> dict[str, Any] | None:
    resolved_user_id = (user_id or "").strip() or resolve_user_id_for_workspace(workspace_id)
    if not resolved_user_id:
        return None
    org_context = _gateway_identity_module().get_user_organization_context(resolved_user_id)
    if not isinstance(org_context, dict):
        return None
    organization_id = str(org_context.get("organization_id") or "").strip()
    if not organization_id:
        return None
    workspace_ref = (workspace_id or str(org_context.get("workspace_id") or "")).strip() or None
    role_id = str(org_context.get("member_role") or "").strip() or None
    try:
        from storage.knowledge_source_activation_store import KnowledgeSourceActivationStore

        store = KnowledgeSourceActivationStore()
        rows = store.list_active_bindings(
            organization_id=organization_id,
            workspace_id=workspace_ref,
        )
        migration_state = store.get_migration_state(organization_id)
    except (OSError, ValueError, sqlite3.Error):
        return None
    migration_conflicts = _migration_state_conflict_evidence(migration_state, rows)
    if migration_conflicts:
        rows = []
    elif not rows:
        store.close()
        return None

    rows, overlay_exclusions = _resolve_overlay_rows(rows)
    seen_source_scope: set[tuple[str, str]] = set()
    seen_dataset_scope_role: set[tuple[str, str]] = set()
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = [*migration_conflicts, *overlay_exclusions]
    for row in rows:
        activation_id = str(row.get("activation_id") or "").strip()
        scope_key = str(row.get("scope_key") or "").strip()
        source_id = str(row.get("knowledge_source_id") or "").strip()
        source_kind = str(row.get("source_kind") or "").strip()
        usage_role = str(row.get("dataset_usage_role") or "").strip()
        if source_kind == "dataset":
            duplicate_key = (scope_key, usage_role)
            if duplicate_key in seen_dataset_scope_role:
                excluded.append(
                    {
                        "activation_id": activation_id,
                        "reason_code": DATASET_ROLE_CONFLICT,
                        "edge_type": "blocked_by_resolver",
                    }
                )
                continue
            seen_dataset_scope_role.add(duplicate_key)
            if str(row.get("dataset_readiness_status") or "").strip() not in {
                "",
                "ready",
            }:
                excluded.append(
                    {
                        "activation_id": activation_id,
                        "reason_code": STALE_DATASET_READINESS,
                        "edge_type": "blocked_by_resolver",
                        "dataset_readiness_status": row.get("dataset_readiness_status"),
                    }
                )
                continue
        else:
            duplicate_key = (scope_key, source_id)
            if duplicate_key in seen_source_scope:
                excluded.append(
                    {
                        "activation_id": activation_id,
                        "reason_code": SAME_SCOPE_DUPLICATE,
                        "edge_type": "blocked_by_resolver",
                    }
                )
                continue
            seen_source_scope.add(duplicate_key)

        control_level = str(row.get("user_context_control_level") or "none")
        allowed_dimensions = _context_dimensions_for_control(control_level)
        preference = store.get_inclusion_preference(
            activation_id=activation_id,
            organization_id=organization_id,
            principal_id=resolved_user_id,
            role_id=role_id,
            workspace_id=workspace_ref,
        )
        binding_effective = _binding_effective(row)
        retrieval_enabled = bool(preference["retrieval_enabled"]) if preference else True
        prompt_context_enabled = (
            bool(preference["prompt_context_enabled"]) if preference else True
        )
        query_context_enabled = bool(preference["query_context_enabled"]) if preference else True
        source_payload = {
            "activation_id": activation_id,
            "knowledge_source_id": source_id,
            "source_kind": source_kind,
            "semantic_tier": row.get("semantic_tier"),
            "authority_role": row.get("authority_role"),
            "activation_scope": row.get("activation_scope"),
            "scope_key": scope_key,
            "user_context_control_level": control_level,
            "workspace_override_policy": row.get("workspace_override_policy"),
            "effective_authority_status": (
                "binding_effective" if binding_effective else "optional_context"
            ),
            "retrieval_included": binding_effective
            or ("retrieval" in allowed_dimensions and retrieval_enabled),
            "prompt_context_included": binding_effective
            or ("prompt_context" in allowed_dimensions and prompt_context_enabled),
            "query_context_included": binding_effective
            or ("query_context" in allowed_dimensions and query_context_enabled),
            "dataset_asset_version_id": row.get("dataset_asset_version_id"),
            "dataset_type": row.get("dataset_type"),
            "dataset_key": row.get("dataset_key"),
            "dataset_version_id": row.get("dataset_version_id"),
            "dataset_usage_role": row.get("dataset_usage_role"),
            "dataset_authority_level": row.get("dataset_authority_level"),
            "dataset_readiness_status": row.get("dataset_readiness_status"),
            "metadata_evidence": _metadata_evidence_payload(
                row,
                organization_id=organization_id,
            ),
            "preference_id": preference.get("preference_id") if preference else None,
        }
        if not source_payload["retrieval_included"] or not source_payload["prompt_context_included"] or not source_payload["query_context_included"]:
            excluded.append(
                {
                    "activation_id": activation_id,
                    "reason_code": "excluded_by_policy",
                    "edge_type": "excluded_by_policy",
                    "retrieval_included": source_payload["retrieval_included"],
                    "prompt_context_included": source_payload["prompt_context_included"],
                    "query_context_included": source_payload["query_context_included"],
                }
            )
        included.append(source_payload)

    now = effective_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    included_activation_ids = [
        str(item["activation_id"]) for item in included if item.get("activation_id")
    ]
    principal_entitlement_ref = f"user:{resolved_user_id}:role:{role_id or 'none'}"
    if persist_snapshot:
        try:
            snapshot = store.materialize_activation_set_snapshot(
                activation_set_snapshot_id=f"kass_{uuid4().hex}",
                organization_id=organization_id,
                workspace_id=workspace_ref,
                principal_entitlement_ref=principal_entitlement_ref,
                activation_resolver_policy_version=_ACTIVATION_RESOLVER_POLICY_VERSION,
                effective_at=now,
                included_activation_ids=included_activation_ids,
                excluded_activation_evidence=excluded,
            )
        except (OSError, ValueError, sqlite3.Error):
            store.close()
            return None
        activation_set_snapshot_id = snapshot["activation_set_snapshot_id"]
        resolved_activation_set_hash = snapshot["resolved_activation_set_hash"]
    else:
        resolver_payload = {
            "organization_id": organization_id,
            "workspace_id": workspace_ref,
            "principal_entitlement_ref": principal_entitlement_ref,
            "activation_resolver_policy_version": _ACTIVATION_RESOLVER_POLICY_VERSION,
            "effective_at": now,
            "included_activation_ids": sorted(included_activation_ids),
            "excluded_activation_evidence": excluded,
            "prior_snapshot_hash": None,
        }
        activation_set_snapshot_id = None
        resolved_activation_set_hash = content_hash(resolver_payload)
    store.close()
    semantic_indexes = _resolve_runtime_semantic_indexes(
        organization_id=organization_id,
        workspace_id=workspace_ref,
    )
    return {
        "organization_id": organization_id,
        "workspace_id": workspace_ref,
        "user_id": resolved_user_id,
        "activation_set_snapshot_id": activation_set_snapshot_id,
        "activation_resolver_policy_version": _ACTIVATION_RESOLVER_POLICY_VERSION,
        "resolved_activation_set_hash": resolved_activation_set_hash,
        "principal_entitlement_ref": principal_entitlement_ref,
        "activation_context_persisted": persist_snapshot,
        "sources": included,
        "excluded_activation_evidence": excluded,
        "runtime_semantic_indexes": semantic_indexes,
    }


def _resolve_runtime_semantic_indexes(
    *,
    organization_id: str,
    workspace_id: str | None,
) -> list[dict[str, Any]]:
    try:
        from storage.runtime_semantic_index_store import RuntimeSemanticIndexStore

        store = RuntimeSemanticIndexStore()
        authorities = store.list_active_authority_versions(
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
    except (OSError, ValueError, sqlite3.Error):
        return []
    indexes: list[dict[str, Any]] = []
    for authority in authorities:
        read_model = store.get_read_model(str(authority["authority_version_id"]))
        indexes.append(
            {
                "authority_version_id": authority["authority_version_id"],
                "canonical_label": authority["canonical_label"],
                "domain": authority["domain"],
                "authority_state": authority["authority_state"],
                "resolver_selected": True,
                "read_model_non_authoritative": True,
                "read_model": read_model,
            }
        )
    return indexes


def build_governed_runtime_context_prompt(
    *,
    user_id: str | None = None,
    workspace_id: str | None = None,
    user_message: str = "",
) -> str | None:
    resolved_user_id = (user_id or "").strip() or resolve_user_id_for_workspace(workspace_id)
    if not resolved_user_id:
        return None
    org_context = _gateway_identity_module().get_user_organization_context(resolved_user_id)
    if not isinstance(org_context, dict):
        return None
    resolve_governed_activation_context(
        user_id=resolved_user_id,
        workspace_id=workspace_id,
    )
    parts = [
        _active_organization_context_system_message(org_context),
        _metric_contract_system_message(user_message),
        _organization_tool_routing_system_message(org_context, user_message),
        _demo_dataset_system_message(org_context, user_message),
        _default_analytics_sample_system_message(org_context, user_message),
    ]
    return "\n\n".join(part.strip() for part in parts if part and part.strip()) or None
