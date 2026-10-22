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
import importlib
import importlib.util
import sys
import yaml

from agents.runtime_inventory import (
    list_plugins_inventory as _runtime_list_plugins_inventory,
    list_skills_inventory as _runtime_list_skills_inventory,
    list_toolsets_inventory as _runtime_list_toolsets_inventory,
)
from agents.hermes_bundled_skill_adapter import GovernedDiagnosticInspectionContext
from agents.workspace_session_logs import (
    WorkspaceSessionResolutionError,
    create_workspace_session_log,
    delete_workspace_session_log,
    derive_workspace_session_transport_metadata,
    get_workspace_session_detail,
    get_workspace_session_log_payload,
    get_workspace_session_messages,
    list_workspace_session_trajectory,
    list_workspace_sessions,
    resolve_or_create_workspace_session_id,
    resolve_workspace_session_id,
    update_workspace_session_title,
)
from eos.artifact_hashing import content_hash
def list_plugins_inventory(*args, **kwargs):
    return _runtime_list_plugins_inventory(*args, **kwargs)


def list_skills_inventory(
    *,
    workspace_hermes_home: Path | None,
    governed_diagnostic_context: GovernedDiagnosticInspectionContext | None = None,
    execution_boundary: Any | None = None,
    authenticated_context: Any | None = None,
) -> dict[str, Any]:
    from contextlib import ExitStack
    with ExitStack() as stack:
        if authenticated_context is not None and authenticated_context.authenticated:
            from gateway.execution_boundary import BoundaryPaths, ExecutionBoundary, bind_execution_boundary
            from services.hermes_skill_review_service import bind_runtime_package_resolver
            from hermes_constants import set_hermes_home_override, reset_hermes_home_override
            boundary = ExecutionBoundary(
                source="webapi:/system/skills",
                user_id=authenticated_context.user_id,
                workspace_id=authenticated_context.workspace_id,
                paths=BoundaryPaths(hermes_home=workspace_hermes_home),
            )
            stack.enter_context(bind_execution_boundary(boundary))
            token = set_hermes_home_override(workspace_hermes_home)
            stack.callback(reset_hermes_home_override, token)
            try:
                stack.enter_context(bind_runtime_package_resolver(boundary))
            except (PermissionError, ValueError, OSError):
                pass  # Inventory emits explicit unavailable managed rows.
        payload = _runtime_list_skills_inventory(
            workspace_hermes_home=workspace_hermes_home,
            governed_diagnostic_context=governed_diagnostic_context,
            execution_boundary=execution_boundary,
        )
    if workspace_hermes_home is None:
        return payload
    workspace_plugins_root = (Path(workspace_hermes_home).expanduser() / "plugins").resolve()
    filtered = []
    for item in payload.get("skills", []):
        if not isinstance(item, dict):
            continue
        source_path = str(item.get("sourcePath") or "").strip()
        source_tier = str(item.get("sourceTier") or "").strip()
        try:
            under_workspace_plugin = Path(source_path).expanduser().resolve().is_relative_to(workspace_plugins_root)
        except (OSError, ValueError):
            under_workspace_plugin = False
        next_item = dict(item)
        if source_tier == "workspace" and under_workspace_plugin:
            next_item["enabled"] = False
            next_item["invokable"] = False
            next_item["nonInvokableReason"] = (
                "workspace_plugin_skill_not_dynamic_invocation_surface"
            )
            next_item["sourcePathClass"] = "workspace_plugin"
        else:
            next_item.setdefault("invokable", bool(next_item.get("enabled", True)))
            next_item.setdefault("sourcePathClass", "skill_root")
        filtered.append(next_item)
    payload = dict(payload)
    payload["skills"] = filtered
    payload["total"] = len(filtered)
    return payload


def list_toolsets_inventory(*args, **kwargs):
    return _runtime_list_toolsets_inventory(*args, **kwargs)
