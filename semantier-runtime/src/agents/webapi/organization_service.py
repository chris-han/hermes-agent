from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any
import uuid

import httpx
from fastapi import File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from agents.auth_db import (
    auth_db_path,
    get_feishu_bot_config_for_user,
    get_user_settings_for_user,
    get_weixin_runtime_account,
    upsert_user_settings_for_user,
    save_weixin_runtime_account,
    update_weixin_runtime_account_state,
)
from agents.route_policy import (
    RouteAuthorizationClass,
    classify_authorization_class,
    forwarded_upstream_allowed,
)
from agents import company_dataset_imports
from agents.workspace_uploads import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    WorkspaceUploadError,
    save_workspace_upload,
)
from agents.auth_session import (
    RequestContext,
    VT_BROWSER_SESSION_COOKIE,
    VT_SESSION_COOKIE,
    auth_context_payload,
    browser_session_nonce,
    browser_session_nonce_hash,
    create_vt_session_token,
    decode_vt_session_token,
    request_context_from_request,
    request_user_agent_hash,
    session_grace_window_seconds,
    session_cookie_max_age,
    session_issue_time,
)
from eos.db import init_db
from eos.legal_corpus_store import LegalCorpusStore
import importlib
import importlib.util
import sys
import yaml

def _compat_attr(name: str):
    compat_module = sys.modules.get("agents.webapi_gateway")
    if compat_module is not None and name in vars(compat_module):
        return getattr(compat_module, name)
    return globals()[name]

def _load_gateway_identity() -> object:
    required = [
        "attach_user_to_seeded_demo_organization",
        "authenticate_password_user",
        "count_active_organization_owners",
        "complete_user_profile",
        "consume_weixin_login_state",
        "create_feishu_link_state",
        "create_weixin_login_state",
        "delete_feishu_link_state",
        "delete_workspace_feishu_bot_config",
        "get_active_weixin_correlation_for_user",
        "get_organization_materialization_policy",
        "get_pending_role_notifications_for_user",
        "get_user_membership_for_organization",
        "get_feishu_link_state",
        "get_user_organization_context",
        "get_weixin_login_state",
        "join_organization",
        "link_feishu_open_id",
        "list_organization_events",
        "list_organization_members",
        "list_seeded_demo_organization_profiles",
        "read_workspace_feishu_bot_config",
        "resolve_seeded_demo_organization_for_industry",
        "save_workspace_feishu_bot_config",
        "set_user_sharing_enabled",
        "switch_user_organization",
        "unlink_feishu_open_id",
        "update_membership_role",
        "update_membership_status",
        "update_organization_materialization_policy",
        "update_weixin_login_state",
        "update_feishu_link_state",
        "upsert_weixin_owner",
    ]

    try:
        # Try the normal import first (may be provided by an installed package).
        mod = importlib.import_module("agents.gateway_identity")
        if all(hasattr(mod, name) for name in required):
            return mod
    except Exception:
        pass

    # Fallback: explicitly load the repo `src/agents/gateway_identity.py`.
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "src" / "agents" / "gateway_identity.py"
    spec = importlib.util.spec_from_file_location("agents.gateway_identity", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Semantier gateway_identity from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agents.gateway_identity"] = module
    spec.loader.exec_module(module)
    return module


_gateway_identity = _load_gateway_identity()

def _gateway_identity_module() -> object:
    # Keep the identity authority selected by _load_gateway_identity.
    return _gateway_identity


def __getattr_from_gateway_identity(name: str):
    return getattr(_gateway_identity_module(), name)


def attach_user_to_seeded_demo_organization(*args, **kwargs):
    return __getattr_from_gateway_identity("attach_user_to_seeded_demo_organization")(*args, **kwargs)


def authenticate_password_user(*args, **kwargs):
    return __getattr_from_gateway_identity("authenticate_password_user")(*args, **kwargs)


def count_active_organization_owners(*args, **kwargs):
    return __getattr_from_gateway_identity("count_active_organization_owners")(*args, **kwargs)


def complete_user_profile(*args, **kwargs):
    return __getattr_from_gateway_identity("complete_user_profile")(*args, **kwargs)


def consume_weixin_login_state(*args, **kwargs):
    return __getattr_from_gateway_identity("consume_weixin_login_state")(*args, **kwargs)


def create_feishu_link_state(*args, **kwargs):
    return __getattr_from_gateway_identity("create_feishu_link_state")(*args, **kwargs)


def create_weixin_login_state(*args, **kwargs):
    return __getattr_from_gateway_identity("create_weixin_login_state")(*args, **kwargs)


def delete_feishu_link_state(*args, **kwargs):
    return __getattr_from_gateway_identity("delete_feishu_link_state")(*args, **kwargs)


def delete_workspace_feishu_bot_config(*args, **kwargs):
    return __getattr_from_gateway_identity("delete_workspace_feishu_bot_config")(*args, **kwargs)


def get_active_weixin_correlation_for_user(*args, **kwargs):
    return __getattr_from_gateway_identity("get_active_weixin_correlation_for_user")(*args, **kwargs)


def get_organization_materialization_policy(*args, **kwargs):
    return __getattr_from_gateway_identity("get_organization_materialization_policy")(*args, **kwargs)


def get_pending_role_notifications_for_user(*args, **kwargs):
    return __getattr_from_gateway_identity("get_pending_role_notifications_for_user")(*args, **kwargs)


def get_pending_regulatory_notifications_for_user(
    organization_id: str,
    user_id: str,
    member_role: str | None = None,
) -> list[dict[str, Any]]:
    """Project authoritative legal-corpus alerts into the organization payload."""
    if not organization_id:
        return []
    try:
        init_db()
        alerts = LegalCorpusStore().list_regulatory_notifications(
            organization_id=organization_id,
            recipient_user_id=user_id,
            recipient_role=member_role,
            lifecycle_state="OPEN",
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "failed to load regulatory notifications for organization payload",
        )
        return []
    projected: list[dict[str, Any]] = []
    for alert in alerts:
        projected.append(
            {
                "event_id": alert["alert_id"],
                "event_type": alert["event_type"],
                "organization_id": alert["organization_id"],
                "subject_user_id": alert.get("recipient_user_id") or user_id,
                "created_at": alert.get("created_at"),
                "detail": {
                    "notification_kind": "legal_regulatory",
                    "alert_id": alert["alert_id"],
                    "source_id": alert.get("source_id"),
                    "refresh_check_id": alert.get("refresh_check_id"),
                    "candidate_id": alert.get("candidate_id"),
                    "impact_report_ref": alert.get("impact_report_ref"),
                    "severity": alert.get("severity"),
                    "recipient_role": alert.get("recipient_role"),
                    "recipient_user_id": alert.get("recipient_user_id"),
                    "lifecycle_state": alert.get("lifecycle_state"),
                    "notification": alert.get("notification") or {},
                    "acknowledge_route": f"/api/legal-corpus/alerts/{alert['alert_id']}/acknowledge",
                },
            }
        )
    return projected


def get_user_membership_for_organization(*args, **kwargs):
    return __getattr_from_gateway_identity("get_user_membership_for_organization")(*args, **kwargs)


def get_feishu_link_state(*args, **kwargs):
    return __getattr_from_gateway_identity("get_feishu_link_state")(*args, **kwargs)


def get_user_organization_context(*args, **kwargs):
    return __getattr_from_gateway_identity("get_user_organization_context")(*args, **kwargs)


def get_weixin_login_state(*args, **kwargs):
    return __getattr_from_gateway_identity("get_weixin_login_state")(*args, **kwargs)


def get_user_record(*args, **kwargs):
    return __getattr_from_gateway_identity("get_user_record")(*args, **kwargs)


def get_user_record_by_workspace_id(*args, **kwargs):
    return __getattr_from_gateway_identity("get_user_record_by_workspace_id")(*args, **kwargs)


def join_organization(*args, **kwargs):
    return __getattr_from_gateway_identity("join_organization")(*args, **kwargs)


def link_feishu_open_id(*args, **kwargs):
    return __getattr_from_gateway_identity("link_feishu_open_id")(*args, **kwargs)


def list_organization_events(*args, **kwargs):
    return __getattr_from_gateway_identity("list_organization_events")(*args, **kwargs)


def list_organization_members(*args, **kwargs):
    return __getattr_from_gateway_identity("list_organization_members")(*args, **kwargs)


def list_seeded_demo_organization_profiles(*args, **kwargs):
    return __getattr_from_gateway_identity("list_seeded_demo_organization_profiles")(*args, **kwargs)


def read_workspace_feishu_bot_config(*args, **kwargs):
    return __getattr_from_gateway_identity("read_workspace_feishu_bot_config")(*args, **kwargs)


def resolve_seeded_demo_organization_for_industry(*args, **kwargs):
    return __getattr_from_gateway_identity("resolve_seeded_demo_organization_for_industry")(*args, **kwargs)


def save_workspace_feishu_bot_config(*args, **kwargs):
    return __getattr_from_gateway_identity("save_workspace_feishu_bot_config")(*args, **kwargs)


def set_user_sharing_enabled(*args, **kwargs):
    return __getattr_from_gateway_identity("set_user_sharing_enabled")(*args, **kwargs)


def switch_user_organization(*args, **kwargs):
    return __getattr_from_gateway_identity("switch_user_organization")(*args, **kwargs)


def unlink_feishu_open_id(*args, **kwargs):
    return __getattr_from_gateway_identity("unlink_feishu_open_id")(*args, **kwargs)


def update_membership_role(*args, **kwargs):
    return __getattr_from_gateway_identity("update_membership_role")(*args, **kwargs)


def update_membership_status(*args, **kwargs):
    return __getattr_from_gateway_identity("update_membership_status")(*args, **kwargs)


def update_organization_materialization_policy(*args, **kwargs):
    return __getattr_from_gateway_identity("update_organization_materialization_policy")(*args, **kwargs)


def update_weixin_login_state(*args, **kwargs):
    return __getattr_from_gateway_identity("update_weixin_login_state")(*args, **kwargs)


def update_feishu_link_state(*args, **kwargs):
    return __getattr_from_gateway_identity("update_feishu_link_state")(*args, **kwargs)


def upsert_weixin_owner(*args, **kwargs):
    return __getattr_from_gateway_identity("upsert_weixin_owner")(*args, **kwargs)

def _organization_payload_for_context(ctx: RequestContext) -> dict[str, Any]:
    if not ctx.authenticated:
        return {
            "organization": None,
            "memberships": [],
            "members": [],
            "audit_events": [],
        }
    org_context = get_user_organization_context(ctx.user_id) or {}
    memberships = org_context.get("memberships")
    if not isinstance(memberships, list):
        memberships = []
    # DEMO is seeded organization metadata, not a governed knowledge-source
    # activation. Keep the runtime authority context strict, but expose the
    # seeded DEMO label in this organization-management response so onboarding
    # can render the selected organization's mode before activation exists.
    seeded_dataset_type = next(
        (
            str(membership.get("dataset_type"))
            for membership in memberships
            if isinstance(membership, dict)
            and membership.get("organization_id") == ctx.organization_id
            and str(membership.get("dataset_type") or "").strip()
        ),
        None,
    )
    display_dataset_type = org_context.get("dataset_type") or seeded_dataset_type
    members: list[dict[str, Any]] = []
    audit_events: list[dict[str, Any]] = []
    if ctx.organization_id:
        members = list_organization_members(ctx.organization_id)
        audit_events = list_organization_events(ctx.organization_id, limit=50)
    materialization_policy = None
    if ctx.organization_id:
        try:
            materialization_policy = get_organization_materialization_policy(ctx.organization_id)
        except ValueError:
            materialization_policy = None
    setup_state = company_dataset_imports.setup_status_for_context(ctx, org_context)
    authority_state = company_dataset_imports.authority_state_for(
        dataset_type=str(org_context.get("dataset_type") or ""),
        active_dataset_version_id=(
            str(org_context.get("active_dataset_version_id"))
            if org_context.get("active_dataset_version_id")
            else None
        ),
    )
    authority_state_deprecated = False
    authority_state_remove_after = None
    if str(org_context.get("dataset_type") or "").upper() == "REAL":
        authority_state = str(setup_state.get("setup_status") or authority_state)
        authority_state_deprecated = True
        authority_state_remove_after = "2026-07-31"
    role_notifications = (
        get_pending_role_notifications_for_user(ctx.organization_id, ctx.user_id)
        if ctx.organization_id
        else []
    )
    regulatory_notifications = (
        get_pending_regulatory_notifications_for_user(
            ctx.organization_id,
            ctx.user_id,
            ctx.member_role,
        )
        if ctx.organization_id
        else []
    )

    return {
        "organization": {
            "organization_id": ctx.organization_id,
            "organization_name": ctx.organization_name,
            "membership_status": ctx.membership_status,
            "member_role": ctx.member_role,
            "sharing_enabled": ctx.sharing_enabled,
            "dataset_type": display_dataset_type,
            "industry_code": org_context.get("industry_code"),
            "active_dataset_version_id": org_context.get("active_dataset_version_id"),
            "dataset_context_source": org_context.get("dataset_context_source"),
            "dataset_context_deprecated": org_context.get("dataset_context_deprecated"),
            "dataset_context_handoff": org_context.get("dataset_context_handoff"),
            "knowledge_source_activation_id": org_context.get("knowledge_source_activation_id"),
            "dataset_asset_version_id": org_context.get("dataset_asset_version_id"),
            "dataset_usage_role": org_context.get("dataset_usage_role"),
            "dataset_authority_level": org_context.get("dataset_authority_level"),
            "dataset_readiness_status": org_context.get("dataset_readiness_status"),
            "fiscal_year_start_month": org_context.get("fiscal_year_start_month"),
            "local_currency": org_context.get("local_currency"),
            "setup_status": setup_state.get("setup_status"),
            "blocking_reasons": setup_state.get("blocking_reasons", []),
            "setup_counts": setup_state.get("counts", {}),
            "next_actions": setup_state.get("next_actions", []),
            "authority_state": authority_state,
            "authority_state_deprecated": authority_state_deprecated,
            "authority_state_remove_after": authority_state_remove_after,
            "can_invite_members": ctx.can_invite_members,
            "can_change_settings": ctx.can_change_settings,
            "t6_materialization_policy": materialization_policy,
        },
        "memberships": memberships,
        "members": members,
        "audit_events": audit_events,
        "pending_notifications": role_notifications + regulatory_notifications,
    }


def _auth_runtime_account_snapshot(account: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(account, dict):
        return None
    return {
        "account_id": str(account.get("account_id") or "").strip(),
        "external_user_id": str(
            account.get("external_user_id") or account.get("user_id") or ""
        ).strip(),
        "owner_user_id": str(account.get("owner_user_id") or "").strip(),
        "owner_workspace_id": str(
            account.get("owner_workspace_id") or account.get("workspace_id") or ""
        ).strip(),
        "base_url": str(account.get("base_url") or "").strip(),
        "home_channel": str(account.get("home_channel") or "").strip(),
        "runtime_session_state": str(account.get("runtime_session_state") or "").strip(),
        "runtime_session_updated_at": str(
            account.get("runtime_session_updated_at") or ""
        ).strip(),
        "saved_at": str(account.get("saved_at") or "").strip(),
        "token_present": bool(str(account.get("token") or "").strip()),
    }


def _auth_state_payload(ctx: RequestContext) -> dict[str, Any]:
    user = {
        "user_id": ctx.user_id,
        "workspace_id": ctx.workspace_id,
        "workspace_slug": ctx.workspace_slug,
        "organization_id": ctx.organization_id,
        "organization_name": ctx.organization_name,
        "membership_status": ctx.membership_status,
        "member_role": ctx.member_role,
        "auth_provider": ctx.auth_provider,
        "weixin_user_id": ctx.weixin_user_id,
        "feishu_open_id": ctx.feishu_open_id,
        "profile_completed": ctx.profile_completed,
    }
    correlation = get_active_weixin_correlation_for_user(ctx.user_id)
    runtime_account = None
    if isinstance(correlation, dict):
        account_id = str(correlation.get("linked_account_id") or "").strip()
        if account_id:
            runtime_account = get_weixin_runtime_account(account_id)
    return {
        "authDb": {
            "path": str(auth_db_path()),
            "exists": auth_db_path().exists(),
        },
        "user": user,
        "weixin": {
            "correlation": correlation,
            "runtimeAccount": _auth_runtime_account_snapshot(runtime_account),
        },
    }
