from __future__ import annotations

import os
from pathlib import Path

from agents.auth_session import RequestContext


def resolve_authenticated_tool_context() -> RequestContext | None:
    """Rehydrate authenticated authority for Hermes plugin tool dispatch.

    Hermes passes tool arguments separately from the web request context. The
    session identity and workspace binding are authoritative; model arguments
    are never used to construct this context.
    """
    try:
        from gateway.session_context import get_session_env

        get_env = get_session_env
    except ImportError:
        get_env = os.environ.get

    user_id = str(
        get_env("HERMES_SESSION_USER_ID", os.environ.get("SEMANTIER_USER_ID", ""))
    ).strip()
    workspace_id = str(
        get_env("HERMES_SESSION_WORKSPACE_OWNER_ID", os.environ.get("SEMANTIER_WORKSPACE_ID", ""))
    ).strip()
    if not user_id or not workspace_id:
        return None

    try:
        from agents import gateway_identity

        record = gateway_identity.get_user_record(user_id)
        if not isinstance(record, dict):
            return None
        if str(record.get("workspace_id") or "").strip() != workspace_id:
            return None
        organization = gateway_identity.get_user_organization_context(user_id) or {}
        workspace_root, hermes_home = gateway_identity.ensure_workspace_paths(workspace_id)
        return RequestContext(
            authenticated=True,
            user_id=user_id,
            workspace_id=workspace_id,
            workspace_slug=str(record.get("workspace_slug") or workspace_id),
            workspace_root=Path(workspace_root),
            hermes_home=Path(hermes_home),
            organization_id=organization.get("organization_id"),
            organization_name=organization.get("organization_name"),
            dataset_type=organization.get("dataset_type"),
            industry_code=organization.get("industry_code"),
            membership_status=str(organization.get("membership_status") or "unassigned"),
            member_role=organization.get("member_role"),
            sharing_enabled=bool(organization.get("sharing_enabled")),
            can_invite_members=bool(organization.get("can_invite_members")),
            can_change_settings=bool(organization.get("can_change_settings")),
            user_name=record.get("display_name") or record.get("name"),
            login_name=record.get("password_login_name"),
            auth_provider=record.get("auth_provider"),
            profile_completed=bool(record.get("profile_completed_at")),
        )
    except Exception:
        return None
