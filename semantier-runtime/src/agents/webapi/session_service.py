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

_LOGGER = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_HERMES_HOME = _REPO_ROOT / ".semantier-home"

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts" / "agents"
_CRON_ERROR_FOLLOWUP_CONTEXT_PROMPT = (
    _PROMPTS_DIR / "CRON_ERROR_FOLLOWUP_CONTEXT.md"
).read_text(encoding="utf-8").strip()
_SHORT_AFFIRMATIVE_FOLLOWUP_CONTEXT_PROMPT = (
    _PROMPTS_DIR / "SHORT_AFFIRMATIVE_FOLLOWUP_CONTEXT.md"
).read_text(encoding="utf-8").strip()
_AFFIRMATIVE_FOLLOWUP_MESSAGES = {
    "yes",
    "y",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "please do",
    "do it",
    "go ahead",
}


from agents.webapi.deps import enforce_rate_limit as _enforce_rate_limit, require_authenticated_context as _require_authenticated_context
from agents.webapi.embedded_proxy_service import _post_embedded_chat_completion, _stream_embedded_chat_completion
