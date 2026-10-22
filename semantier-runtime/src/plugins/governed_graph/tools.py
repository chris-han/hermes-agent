from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agents.auth_session import RequestContext
from agents.webapi.governed_graph_service import GovernedGraphService


def _session_context() -> RequestContext | None:
    """Resolve the authenticated Hermes turn into Semantier's governed context.

    Hermes tool dispatch intentionally passes tool arguments separately from
    the gateway session context.  Never trust identity fields supplied by the
    model; read the bound session identity/workspace and rehydrate it from the
    governed auth store instead.
    """
    try:
        from gateway.session_context import get_session_env

        get_env = lambda name: get_session_env(name, "") or os.environ.get(name, "")
    except Exception:
        get_env = lambda name: os.environ.get(name, "")
    user_id = str(get_env("HERMES_SESSION_USER_ID") or os.environ.get("SEMANTIER_USER_ID") or "").strip()
    workspace_id = str(get_env("HERMES_SESSION_WORKSPACE_OWNER_ID") or os.environ.get("SEMANTIER_WORKSPACE_ID") or "").strip()
    if not user_id or not workspace_id:
        return None
    try:
        from agents import gateway_identity

        record = gateway_identity.get_user_record(user_id)
        if not isinstance(record, dict) or str(record.get("workspace_id") or "").strip() != workspace_id:
            return None
        org = gateway_identity.get_user_organization_context(user_id) or {}
        # Match password-authenticated request context: a candidate dataset may
        # be declared on the governed active membership before activation.
        dataset_type = org.get("dataset_type")
        if not isinstance(dataset_type, str) or not dataset_type.strip():
            memberships = org.get("memberships")
            dataset_type = next((
                membership.get("dataset_type")
                for membership in (memberships if isinstance(memberships, list) else [])
                if isinstance(membership, dict)
                and membership.get("organization_id") == org.get("organization_id")
                and isinstance(membership.get("dataset_type"), str)
                and membership["dataset_type"].strip()
            ), None)
        workspace_root, hermes_home = gateway_identity.ensure_workspace_paths(workspace_id)
        return RequestContext(
            authenticated=True,
            user_id=user_id,
            workspace_id=workspace_id,
            workspace_slug=str(record.get("workspace_slug") or workspace_id),
            workspace_root=Path(workspace_root),
            hermes_home=Path(hermes_home),
            organization_id=org.get("organization_id"),
            organization_name=org.get("organization_name"),
            dataset_type=dataset_type.strip() if isinstance(dataset_type, str) else None,
            industry_code=org.get("industry_code"),
            membership_status=str(org.get("membership_status") or "unassigned"),
            member_role=org.get("member_role"),
            sharing_enabled=bool(org.get("sharing_enabled")),
            can_invite_members=bool(org.get("can_invite_members")),
            can_change_settings=bool(org.get("can_change_settings")),
            user_name=record.get("display_name") or record.get("name"),
            login_name=record.get("password_login_name"),
            auth_provider=record.get("auth_provider"),
            profile_completed=bool(record.get("profile_completed_at")),
        )
    except Exception:
        return None


def knowledge_graph_query(args: dict[str, Any], **kwargs: Any) -> str:
    ctx = kwargs.get("ctx") or kwargs.get("request_context") or _session_context()
    if ctx is None:
        return json.dumps({"status": "error", "error_code": "GRAPH_AUTHORITY_REQUIRED"}, sort_keys=True)
    try:
        result = GovernedGraphService().query(ctx=ctx, request=args)
        return json.dumps(result, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return json.dumps({"status": "error", "error_code": str(exc) or exc.__class__.__name__}, sort_keys=True)
