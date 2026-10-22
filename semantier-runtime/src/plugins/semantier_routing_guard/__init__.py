from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Optional


_SKILL_REVIEW_TRIGGER = "consider saving or updating a skill"

_SEMANTIC_MARKERS = (
    "semantic_tier",
    "authority_domain",
    "authority_origin",
    "jurisdiction",
    "approval threshold",
    "approval-threshold",
    "threshold preference",
    "approval preference",
    "classification",
    "categorization",
    "default reimbursement category",
    "local operating rule",
    "policy",
    "compliance",
    "tax",
    "accounting",
    "ledger",
    "t2",
    "t3",
    "t4",
    "t5",
)

_PROCEDURAL_MARKERS = (
    "format",
    "formatting",
    "workflow",
    "prompt",
    "ask for",
    "comment style",
    "comment-style",
    "reporting output",
    "one turn",
)

_SEMANTIC_ROUTE_RE = re.compile(
    r"metadata:\s*(?:\n[^\S\r\n]+.*)*\n[^\S\r\n]*semantier:\s*(?:\n[^\S\r\n]+.*)*\n[^\S\r\n]*route:\s*(?:semantier_governed_t5|semantic)",
    re.IGNORECASE,
)

_PROCEDURAL_ROUTE_RE = re.compile(
    r"metadata:\s*(?:\n[^\S\r\n]+.*)*\n[^\S\r\n]*semantier:\s*(?:\n[^\S\r\n]+.*)*\n[^\S\r\n]*route:\s*procedural_only",
    re.IGNORECASE,
)

_DIRECT_DATA_READ_TOOLS = {"execute_code", "read_file", "search_files", "terminal"}
_GOVERNED_ANALYTICS_TOOLS = {
    "governed_query",
    "duckdb_python",
    "governed_policy_eval",
    "governed_report_export",
}
_REAL_ONBOARDING_TOOLS = {
    "real_company_setup_status",
    "real_company_reconcile_promoted_imports",
    "real_company_legacy_claims_report",
    "real_company_preview_source_rows",
    "real_company_admit_rea_claims",
    "real_company_bootstrap_coa",
    "real_company_materialize_lifecycle",
    "real_company_run_setup",
    "company_refresh_lakehouse",
}
_NON_GOVERNED_ANALYTICS_TOOLS = {
    "execute_code",
    "read_file",
    "search_files",
    "terminal",
    "write_file",
}
_GENERIC_FILE_MUTATION_TOOLS = {"execute_code", "terminal", "write_file", "patch"}
_SQL_EXECUTION_TOOLS = {"execute_sql", "run_sql", "query_sql", "mcp_eos_db_execute_sql"}
_SQL_EXECUTION_TOOL_PREFIXES = ("mcp_eos_db_", "mcp_centralmind_", "mcp_gateway_")
_DEFAULT_SAMPLE_FIXTURE_MARKER_BASE = (
    "bootstrap/industry_simulator/construction_3_year",
    "construction_3_year",
    "analytics_rea_claims",
    "analytics_cost_revenue_yearly",
    "analytics_event_type_yearly",
    "project_lifecycles",
    "quarterly_revenue",
    "renovation",
    "装修",
)
_GOVERNED_STORE_PATH_MARKERS = (
    ".semantier-home/eos.db",
    ".semantier-home/auth.db",
    ".semantier-home/context.db",
    ".semantier-home/state.db",
    ".semantier-home/lakehouse",
    "/lakehouse/",
    "workspaces/",
)
_GOVERNED_STORE_QUERY_MARKERS = (
    "sqlite3.connect",
    "duckdb.connect",
    "read_parquet",
    "attach ",
    "from eos.db import",
    "platform_runtime_root()",
    "sqlite_db_path(",
)
_GOVERNED_STORE_TABLE_MARKERS = (
    "rea_claims",
    "rea_events",
    "knowledge_artifacts",
    "kgl_versions",
    "kgl_approval_records",
    "projection_results",
    "journal_voucher_projections",
    "general_ledger_views",
    "financial_statement_packages",
    "tax_filing_packages",
    "accounting_archive_packages",
    "organizations",
    "users",
    "gateway_correlations",
)
_WORKSPACE_SCRIPT_MUTATION_MARKERS = (
    "/scripts/",
    "/scripts ",
    "scripts/",
    "scripts ",
)
_WRITE_OR_CREATE_COMMAND_MARKERS = (
    "mkdir ",
    "install ",
    "touch ",
    "chmod ",
    "cp ",
    "mv ",
    "cat >",
    "tee ",
    "write_file",
)
_ANALYTICS_INTENT_MARKERS = (
    "analytics",
    "analysis",
    "business report",
    "report generation",
    "write a report",
    "dataset discovery",
    "discover revenue",
    "revenue drivers",
    "inspect dataset",
    "dataset schema",
    "org_rea_claims",
    "org_general_ledger_views",
    "org_financial_statement_packages",
)


def _default_sample_profile() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from bootstrap.industry_simulator import get_default_sample_organization_profile

    return get_default_sample_organization_profile() or {}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _workspace_id_from_hermes_home() -> str | None:
    raw_home = os.environ.get("HERMES_HOME")
    if not raw_home:
        return None
    try:
        hermes_home = Path(raw_home).expanduser().resolve()
    except OSError:
        return None
    workspace_id = hermes_home.name.strip()
    return workspace_id or None


def _load_governed_users() -> dict[str, Any] | None:
    try:
        from agents.auth_db import ensure_auth_db, load_users
    except Exception:
        return None
    try:
        ensure_auth_db()
        users = load_users()
    except Exception:
        return None
    return users if isinstance(users, dict) else None


def _user_record_by_workspace_id(workspace_id: str) -> dict[str, Any] | None:
    users = _load_governed_users()
    if users is None:
        return None
    for user in users.values():
        if not isinstance(user, dict):
            continue
        if str(user.get("workspace_id") or "").strip() == workspace_id:
            return user
    return None


def _organization_memberships(user: dict[str, Any]) -> list[dict[str, Any]]:
    memberships = user.get("organization_memberships")
    if isinstance(memberships, list):
        return [membership for membership in memberships if isinstance(membership, dict)]
    organization_id = str(user.get("organization_id") or "").strip()
    if not organization_id:
        return []
    return [
        {
            "organization_id": organization_id,
            "organization_name": user.get("organization_name"),
            "membership_status": user.get("membership_status") or "unassigned",
            "member_role": user.get("member_role"),
            "sharing_enabled": bool(user.get("sharing_enabled")),
            "dataset_type": user.get("dataset_type"),
            "industry_code": user.get("industry_code"),
            "dataset_key": user.get("dataset_key"),
            "active_dataset_version_id": user.get("active_dataset_version_id"),
            "fiscal_year_start_month": user.get("fiscal_year_start_month"),
            "local_currency": user.get("local_currency"),
        }
    ]


def _active_membership(user: dict[str, Any]) -> dict[str, Any] | None:
    memberships = _organization_memberships(user)
    default_org_id = str(user.get("default_organization_id") or "").strip()
    if default_org_id:
        for membership in memberships:
            if str(membership.get("organization_id") or "").strip() == default_org_id:
                return membership
    return memberships[0] if memberships else None


def _active_workspace_org_context() -> dict[str, Any] | None:
    workspace_id = _workspace_id_from_hermes_home()
    if not workspace_id:
        return None
    user = _user_record_by_workspace_id(workspace_id)
    if not isinstance(user, dict):
        return {
            "authorization_error": "AUTH_CONTEXT_MISSING",
            "workspace_id": workspace_id,
        }
    user_id = str(user.get("user_id") or "").strip()
    membership = _active_membership(user)
    if not isinstance(membership, dict):
        return {
            "authorization_error": "AUTH_CONTEXT_MISSING",
            "workspace_id": workspace_id,
            "user_id": user_id,
        }
    org_context = dict(membership)
    org_context["workspace_id"] = workspace_id
    org_context["user_id"] = user_id
    return org_context


def _is_authorized_for_default_sample_dataset() -> bool:
    org_context = _active_workspace_org_context()
    if org_context is None:
        return True
    if org_context.get("authorization_error"):
        return False
    organization_id = str(org_context.get("organization_id") or "").strip()
    dataset_key = str(org_context.get("dataset_key") or "").strip()
    default_profile = _default_sample_profile()
    default_org_id = str(default_profile.get("organization_id") or "").strip()
    default_dataset_key = str(default_profile.get("dataset_key") or "").strip()
    if organization_id != default_org_id:
        return False
    return not dataset_key or dataset_key == default_dataset_key


def _tool_args_text(args: dict[str, Any]) -> str:
    try:
        return json.dumps(args, ensure_ascii=False, sort_keys=True).lower()
    except Exception:
        return str(args).lower()


def _is_direct_data_read_tool(tool_name: str) -> bool:
    return tool_name in _DIRECT_DATA_READ_TOOLS or _is_sql_execution_tool(tool_name)


def _is_sql_execution_tool(tool_name: str) -> bool:
    return tool_name in _SQL_EXECUTION_TOOLS or tool_name.startswith(_SQL_EXECUTION_TOOL_PREFIXES)


def _has_analytics_intent(text: str) -> bool:
    return any(marker in text for marker in _ANALYTICS_INTENT_MARKERS)


def _is_workspace_script_mutation(tool_name: str, text: str) -> bool:
    """Allow user-owned workspace script creation without treating it as authority access."""
    if tool_name not in {"terminal", "write_file", "patch", "execute_code"}:
        return False
    if not any(marker in text for marker in _WORKSPACE_SCRIPT_MUTATION_MARKERS):
        return False
    return any(marker in text for marker in _WRITE_OR_CREATE_COMMAND_MARKERS)


def _default_sample_fixture_markers() -> tuple[str, ...]:
    profile = _default_sample_profile()
    dataset_key = str(profile.get("dataset_key") or "").strip()
    if dataset_key:
        return (*_DEFAULT_SAMPLE_FIXTURE_MARKER_BASE, dataset_key)
    return _DEFAULT_SAMPLE_FIXTURE_MARKER_BASE


def block_default_sample_fixture_access(*, tool_name: str, args: dict[str, Any], **_: Any) -> Optional[dict[str, str]]:
    if not _is_direct_data_read_tool(tool_name):
        return None
    text = _tool_args_text(args)
    if not any(marker in text for marker in _default_sample_fixture_markers()):
        return None
    if _is_authorized_for_default_sample_dataset():
        return None
    return {
        "action": "block",
        "message": (
            "Blocked by Semantier data authorization: the active governed organization is not "
            "authorized for the default construction sample analytics fixture. Resolve the "
            "request through an organization-scoped governed dataset or return an authorization denial."
        ),
    }


def block_direct_governed_store_access(*, tool_name: str, args: dict[str, Any], **_: Any) -> Optional[dict[str, str]]:
    if not _is_direct_data_read_tool(tool_name):
        return None
    text = _tool_args_text(args)
    matched_store_markers = tuple(marker for marker in _GOVERNED_STORE_PATH_MARKERS if marker in text)
    has_direct_query = any(marker in text for marker in _GOVERNED_STORE_QUERY_MARKERS)
    has_store_table = any(marker in text for marker in _GOVERNED_STORE_TABLE_MARKERS)
    has_store_path = bool(matched_store_markers)
    if (
        matched_store_markers == ("workspaces/",)
        and not has_direct_query
        and not has_store_table
        and _is_workspace_script_mutation(tool_name, text)
    ):
        return None
    if not has_store_path and not (has_direct_query and has_store_table) and not (
        _is_sql_execution_tool(tool_name) and has_store_table
    ):
        return None
    return {
        "action": "block",
        "message": (
            "Blocked by Semantier Law 1 data authorization: general code, file, search, "
            "or terminal tools must not read governed runtime stores directly. Use an "
            "authorized Semantier runtime/API/tool surface that resolves active organization, "
            "membership, capability scope, and dataset binding in code before returning data."
        ),
    }


def block_ungoverned_reimbursement_artifact_writes(
    *, tool_name: str, args: dict[str, Any], **_: Any
) -> Optional[dict[str, str]]:
    if tool_name not in _GENERIC_FILE_MUTATION_TOOLS:
        return None
    text = _tool_args_text(args)
    if "reimbursement" not in text and "reim-" not in text:
        return None

    artifacts_root = os.environ.get("SEMANTIER_WORKSPACE_ARTIFACTS_DIR", "").strip()
    if artifacts_root and str(Path(artifacts_root).expanduser()) in text:
        return None

    return {
        "action": "block",
        "message": (
            "Blocked by Semantier artifact boundary: reimbursement documents must be "
            "created only under the active session SEMANTIER_WORKSPACE_ARTIFACTS_DIR. "
            "Do not use terminal, execute_code, write_file, home directories, or guessed "
            "workspace paths as fallbacks."
        ),
    }


def enforce_governed_analytics_tool_routing(
    *, tool_name: str, args: dict[str, Any], **_: Any
) -> Optional[dict[str, str]]:
    text = _tool_args_text(args)
    if not _has_analytics_intent(text):
        return None
    if tool_name in _GOVERNED_ANALYTICS_TOOLS:
        return None
    if tool_name not in _NON_GOVERNED_ANALYTICS_TOOLS and not _is_sql_execution_tool(tool_name):
        return None
    return {
        "action": "block",
        "message": (
            "Blocked by Semantier analytics routing guard: analytics, dataset discovery, "
            "and report-generation work must use governed analytics surfaces such as "
            "governed_query, duckdb_python, governed_policy_eval, or governed_report_export. "
            "Do not use terminal, execute_code, raw file/search, or direct SQL tools for this intent."
        ),
    }


def block_demo_analytics_real_onboarding_tools(
    *, tool_name: str, args: dict[str, Any], **_: Any
) -> Optional[dict[str, str]]:
    if tool_name not in _REAL_ONBOARDING_TOOLS:
        return None
    text = _tool_args_text(args)
    if not _has_analytics_intent(text):
        return None
    org_context = _active_workspace_org_context()
    if not isinstance(org_context, dict) or org_context.get("authorization_error"):
        return {
            "action": "block",
            "message": (
                "Blocked by Semantier analytics routing guard: REAL onboarding tools require "
                "an active governed workspace organization before analytics or report-generation routing."
            ),
        }
    dataset_type = str(org_context.get("dataset_type") or "").strip().upper()
    if dataset_type in {"DEMO", "DEFAULT_REALISTIC_SAMPLE"}:
        return {
            "action": "block",
            "message": (
                "Blocked by Semantier analytics routing guard: the active dataset activation is DEMO. "
                "Use governed_query for analytics/report-generation intent; do not route DEMO analytics "
                "through REAL onboarding setup or mutation tools."
            ),
        }
    return None


def _extract_skill_candidate_text(args: dict[str, Any]) -> str:
    action = _normalize_text(args.get("action"))
    file_path = _normalize_text(args.get("file_path"))

    if action in {"create", "edit"}:
        return str(args.get("content") or "")
    if action == "patch":
        if file_path and file_path != "skill.md":
            return ""
        return "\n".join(str(part or "") for part in (args.get("old_string"), args.get("new_string")))
    if action == "write_file":
        if file_path and file_path != "skill.md":
            return ""
        return str(args.get("file_content") or "")
    return ""


def classify_skill_candidate(content: str) -> str:
    text = _normalize_text(content)
    if not text:
        return "unknown"
    if _SEMANTIC_ROUTE_RE.search(content):
        return "semantic_bearing"
    procedural_only = bool(_PROCEDURAL_ROUTE_RE.search(content))
    has_semantic = any(marker in text for marker in _SEMANTIC_MARKERS)
    has_procedural = any(marker in text for marker in _PROCEDURAL_MARKERS)
    if has_semantic:
        return "semantic_bearing"
    if procedural_only or has_procedural:
        return "procedural_only"
    return "unknown"


def build_split_routing_context(user_message: str, **_: Any) -> Optional[dict[str, str]]:
    if _SKILL_REVIEW_TRIGGER not in _normalize_text(user_message):
        return None
    return {
        "context": (
            "Semantier routing policy: use Hermes skills only for procedural execution behavior "
            "such as prompting order, formatting, and workflow orchestration. Do not save scoped "
            "semantic defaults, approval preferences, categorization logic, policy meaning, tax, "
            "accounting, or compliance interpretation as a Hermes skill. If such meaning is present, "
            "do not call skill_manage for it; hand it off to Semantier governed T6-to-T5 routing instead. "
            "If one proposal mixes procedural behavior and governed semantic meaning, split it into separate artifacts."
        )
    }


def block_semantic_skill_manage(*, tool_name: str, args: dict[str, Any], **_: Any) -> Optional[dict[str, str]]:
    if tool_name != "skill_manage":
        return None
    action = _normalize_text(args.get("action"))
    if action not in {"create", "edit", "patch", "write_file"}:
        return None
    content = _extract_skill_candidate_text(args)
    if classify_skill_candidate(content) != "semantic_bearing":
        return None
    return {
        "action": "block",
        "message": (
            "Blocked by Semantier routing guard: this skill candidate appears to carry semantic "
            "authority or scoped rule meaning. Hermes skills may encode procedure only. Route this "
            "proposal through Semantier governed T6->T5 activation instead of skill_manage."
        ),
    }


def register(ctx) -> None:
    from services.personal_procedure_capture_service import begin_native_procedure_capture, finish_native_procedure_capture
    ctx.register_hook("pre_tool_call", begin_native_procedure_capture)
    ctx.register_hook("post_tool_call", finish_native_procedure_capture)
    ctx.register_hook("pre_llm_call", build_split_routing_context)
    ctx.register_hook("pre_tool_call", block_semantic_skill_manage)
    ctx.register_hook("pre_tool_call", block_ungoverned_reimbursement_artifact_writes)
    ctx.register_hook("pre_tool_call", block_demo_analytics_real_onboarding_tools)
    ctx.register_hook("pre_tool_call", enforce_governed_analytics_tool_routing)
    ctx.register_hook("pre_tool_call", block_direct_governed_store_access)
    ctx.register_hook("pre_tool_call", block_default_sample_fixture_access)
