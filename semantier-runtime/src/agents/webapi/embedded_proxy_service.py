from __future__ import annotations
from agents.chat_execution_transport import isolated_chat_turn

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
from agents.hermes_embedded_api import get_embedded_hermes_api_service
from agents.sandbox_scope import SandboxScope, bind_sandbox_scope, sandbox_key_for_request
from agents.webapi.gateway_channel_service import _workspace_id_from_hermes_home_path
from semantier.semantica_context_graph.bridge import (
    TenderExpertBridge,
    TenderExpertRequest,
    TenderExpertResult,
)
import importlib
import importlib.util
import sys
import yaml

def _compat_attr(name: str):
    compat_module = sys.modules.get("agents.webapi_gateway")
    if compat_module is not None and name in vars(compat_module):
        return getattr(compat_module, name)
    return globals()[name]

_LOGGER = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_HERMES_HOME = _REPO_ROOT / ".semantier-home"
_REQUEST_HEADER_EXCLUDE = frozenset({"host", "content-length", "connection"})
_RESPONSE_HEADER_EXCLUDE = frozenset({"content-length", "connection", "transfer-encoding"})
_RUNTIME_EMBED_GATEWAY_ENV = "SEMANTIER_RUNTIME_EMBED_GATEWAY"
_SEMANTIER_SANDBOX_KEY_HEADER = "X-Semantier-Sandbox-Key"
_SEMANTIER_USER_ID_HEADER = "X-Semantier-User-Id"
_SEMANTIER_WORKSPACE_ID_HEADER = "X-Semantier-Workspace-Id"
_SEMANTIER_UPLOAD_SESSION_ID_HEADER = "X-Semantier-Upload-Session-Id"


def _boundary_header_payload(
    *,
    workspace_id: str | None,
    user_id: str | None,
    session_id: str | None,
    hermes_home: Path | None,
    request_metadata: dict[str, str] | None = None,
) -> str:
    payload = {
        "workspace_id": workspace_id,
        "user_id": user_id,
        "session_id": session_id,
        "hermes_home": str(hermes_home.resolve()) if hermes_home is not None else None,
    }
    payload.update({key: value for key, value in (request_metadata or {}).items()
                    if key in ("durable_request_id", "durable_request_hash")})
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")

SUPPORTED_GATEWAYS = ("weixin", "feishu", "web_api")

GATEWAY_PLATFORM_MAP = {
    "weixin": "weixin",
    "feishu": "feishu",
    "web_api": "api_server",
}
MESSAGING_DOCS_URL = {
    "feishu": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/feishu",
    "weixin": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/weixin",
}
WEIXIN_RECONNECT_CONTRACT_VERSION = "weixin-reconnect.v1"
WEIXIN_LOGIN_STATE_VERSION = "weixin-login.v2"

_RATE_LIMIT_STATE_LOCK = threading.Lock()
_RATE_LIMIT_STATE: dict[str, tuple[float, int]] = {}
# (max_requests, window_seconds)
_RATE_LIMITS: dict[str, tuple[int, int]] = {
    "auth_public": (60, 60),
    "weixin_reconnect_user": (12, 60),
    "session_create_user": (30, 60),
    "session_create_tenant": (120, 60),
    "chat_user": (120, 60),
    "chat_tenant": (600, 60),
}



def _ensure_repo_hermes_agent_on_path() -> None:
    hermes_agent_root = _REPO_ROOT / "hermes-agent"
    if not hermes_agent_root.exists():
        return
    normalized_root = str(hermes_agent_root.resolve())
    sys_path = []
    for existing in os.sys.path:
        try:
            if Path(existing).resolve() == hermes_agent_root.resolve():
                continue
        except OSError:
            pass
        sys_path.append(existing)
    os.sys.path[:] = [normalized_root, *sys_path]

def _temporary_hermes_home(hermes_home: str):
    """Temporarily bind HERMES_HOME to a workspace for embedded gateway calls."""
    target_home = str(hermes_home or "").strip()
    if not target_home:
        yield
        return
    context_token = None
    reset_active_hermes_home = None
    try:
        from hermes_constants import reset_active_hermes_home as _reset_home, set_active_hermes_home

        context_token = set_active_hermes_home(Path(target_home))
        reset_active_hermes_home = _reset_home
    except Exception:
        context_token = None
        reset_active_hermes_home = None

    # Cross-gateway env contract (HERMES_HOME + SEMANTIER_WORKSPACE_RUNS_DIR)
    # is owned by runtime_paths.bind_workspace_env so every adapter binds
    # identically and downstream file-IO tools see a consistent context.
    from runtime_paths import bind_workspace_env

    try:
        try:
            binding = bind_workspace_env(target_home)
        except ValueError:
            binding = None
        if binding is not None:
            with binding:
                yield
        elif _compat_attr("_workspace_id_from_hermes_home_path")(Path(target_home)):
            previous = {
                "HERMES_HOME": os.environ.get("HERMES_HOME"),
                "SEMANTIER_WORKSPACE_ID": os.environ.get("SEMANTIER_WORKSPACE_ID"),
                "SEMANTIER_WRITE_SAFE_ROOT": os.environ.get("SEMANTIER_WRITE_SAFE_ROOT"),
                "SEMANTIER_TERMINAL_CWD": os.environ.get("SEMANTIER_TERMINAL_CWD"),
            }
            resolved_home = Path(target_home).expanduser().resolve()
            os.environ["HERMES_HOME"] = str(resolved_home)
            os.environ["SEMANTIER_WORKSPACE_ID"] = resolved_home.name
            os.environ["SEMANTIER_WRITE_SAFE_ROOT"] = str(resolved_home)
            os.environ["SEMANTIER_TERMINAL_CWD"] = str(resolved_home)
            try:
                yield
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        else:
            raise ValueError("not a workspace runtime home")
    finally:
        if context_token is not None and reset_active_hermes_home is not None:
            try:
                reset_active_hermes_home(context_token)
            except Exception:
                pass


@contextmanager
def _compat_hermes_home_bound(target_home: Path, *, session_id: str | None = None):
    from agents.semantier_agent import _hermes_home_bound

    try:
        with _hermes_home_bound(target_home, session_id=session_id):
            yield
    except ValueError:
        resolved_home = Path(target_home).expanduser().resolve()
        previous = {
            "HERMES_HOME": os.environ.get("HERMES_HOME"),
            "SEMANTIER_WORKSPACE_ID": os.environ.get("SEMANTIER_WORKSPACE_ID"),
            "SEMANTIER_WRITE_SAFE_ROOT": os.environ.get("SEMANTIER_WRITE_SAFE_ROOT"),
            "SEMANTIER_TERMINAL_CWD": os.environ.get("SEMANTIER_TERMINAL_CWD"),
        }
        context_token = None
        reset_active_hermes_home = None
        try:
            from hermes_constants import reset_active_hermes_home as _reset_home, set_active_hermes_home

            context_token = set_active_hermes_home(resolved_home)
            reset_active_hermes_home = _reset_home
        except Exception:
            context_token = None
            reset_active_hermes_home = None
        os.environ["HERMES_HOME"] = str(resolved_home)
        os.environ["SEMANTIER_WORKSPACE_ID"] = resolved_home.name
        os.environ["SEMANTIER_WRITE_SAFE_ROOT"] = str(resolved_home)
        os.environ["SEMANTIER_TERMINAL_CWD"] = str(resolved_home)
        try:
            yield
        finally:
            if context_token is not None and reset_active_hermes_home is not None:
                try:
                    reset_active_hermes_home(context_token)
                except Exception:
                    pass
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

def _filtered_headers(
    headers: dict[str, str] | Any,
    *,
    excluded: frozenset[str],
) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in excluded
    }


async def _forward_embedded_request(request: Request, upstream_path: str) -> Response:
    service = get_embedded_hermes_api_service()
    await service.ensure_started()

    url = f"{service.base_url}/{upstream_path.lstrip('/')}"
    headers = _filtered_headers(request.headers, excluded=_REQUEST_HEADER_EXCLUDE)
    headers.setdefault("Authorization", f"Bearer {service.api_key}")
    try:
        body = await request.body()
    except ClientDisconnect:
        return Response(status_code=499)
    timeout = httpx.Timeout(connect=5.0, read=300.0, write=60.0, pool=60.0)

    try:
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False)
        upstream_request = client.build_request(
            request.method,
            url,
            params=request.query_params,
            headers=headers,
            content=body,
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Embedded Hermes API unavailable: {exc.__class__.__name__}",
        ) from exc

    response_headers = _filtered_headers(
        upstream_response.headers,
        excluded=_RESPONSE_HEADER_EXCLUDE,
    )
    content_type = upstream_response.headers.get("content-type", "")

    if content_type.startswith("text/event-stream"):
        async def stream_body():
            try:
                async for chunk in upstream_response.aiter_raw():
                    yield chunk
            finally:
                await upstream_response.aclose()
                await client.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    try:
        content = await upstream_response.aread()
    finally:
        await upstream_response.aclose()
        await client.aclose()

    return Response(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


@isolated_chat_turn
async def _post_embedded_chat_completion(
    *,
    session_id: str,
    upload_session_id: str | None = None,
    message: str,
    model: str | None = None,
    system_message: str | None = None,
    hermes_home: Path | None = None,
    user_id: str | None = None,
    workspace_id: str | None = None,
    request_metadata: dict[str, str] | None = None,
    _execution_service=None,
) -> tuple[dict[str, Any], str]:
    service = _execution_service or get_embedded_hermes_api_service()
    await service.ensure_started()

    headers = {
        "Authorization": f"Bearer {service.api_key}",
        "Content-Type": "application/json",
        "X-Hermes-Session-Id": session_id,
    }
    if hermes_home is not None:
        from agents.default_workspace_plugins import ensure_default_workspace_plugins
        ensure_default_workspace_plugins(hermes_home)
        headers["X-Semantier-Hermes-Home"] = str(hermes_home.resolve())
    if user_id:
        headers[_SEMANTIER_USER_ID_HEADER] = user_id
    if workspace_id:
        headers[_SEMANTIER_WORKSPACE_ID_HEADER] = workspace_id
    headers[_SEMANTIER_UPLOAD_SESSION_ID_HEADER] = upload_session_id or session_id
    boundary_session_id = upload_session_id or session_id
    boundary_workspace_id = workspace_id or _workspace_id_from_hermes_home_path(hermes_home)
    headers["X-Hermes-Execution-Boundary"] = _boundary_header_payload(
        workspace_id=boundary_workspace_id,
        user_id=user_id,
        session_id=boundary_session_id,
        hermes_home=hermes_home,
        request_metadata=request_metadata,
    )
    headers["X-Hermes-Execution-Boundary-Trust"] = "semantier-internal"
    scope = None
    resolved_workspace_id = _workspace_id_from_hermes_home_path(hermes_home)
    if resolved_workspace_id:
        scope = SandboxScope(
            workspace_id=resolved_workspace_id,
            lane="interactive_session",
            scope_id=session_id,
            adapter_key=None,
            network_enabled=False,
        )
        headers[_SEMANTIER_SANDBOX_KEY_HEADER] = sandbox_key_for_request(scope)
    messages: list[dict[str, Any]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": message})
    payload: dict[str, Any] = {
        "messages": messages,
        "stream": False,
        "_semantier_request_metadata": request_metadata or {},
    }
    if model:
        payload["model"] = model

    timeout = httpx.Timeout(connect=30.0 if _execution_service is not None else 5.0, read=300.0, write=60.0, pool=60.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False) as client:
            with bind_sandbox_scope(scope):
                upstream_response = await client.post(
                    f"{service.base_url}/v1/chat/completions", headers=headers, json=payload,
                )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Embedded Hermes API unavailable: {exc.__class__.__name__}",
        ) from exc

    if upstream_response.status_code >= 400:
        detail = ""
        try:
            error_payload = upstream_response.json()
            if isinstance(error_payload, dict):
                error = error_payload.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message") or "").strip()
                elif isinstance(error_payload.get("detail"), str):
                    detail = str(error_payload["detail"]).strip()
        except Exception:
            detail = ""
        if not detail:
            detail = upstream_response.text.strip() or "Embedded Hermes API request failed"
        raise HTTPException(status_code=upstream_response.status_code, detail=detail)

    try:
        data = upstream_response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Embedded Hermes API returned invalid JSON") from exc

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Embedded Hermes API returned unexpected payload")

    response_session_id = upstream_response.headers.get("X-Hermes-Session-Id", "").strip() or session_id
    return data, response_session_id


_TENDER_RUNTIME_EXPERT_BRIDGE = TenderExpertBridge()


async def _post_tender_runtime_expert_completion(
    *,
    request: TenderExpertRequest,
    hermes_home: Path,
    model: str | None = None,
) -> TenderExpertResult:
    """Route only the bounded Runtime Expert request to its isolated profile."""
    profile_home = hermes_home / "profiles" / "tender-runtime-expert"

    async def invoke(current: TenderExpertRequest) -> TenderExpertResult:
        pin_message = json.dumps(
            {
                "graph_ref": current.graph_ref,
                "graph_version": current.graph_version,
                "graph_hash": current.graph_hash,
                "semantica_commit": current.semantica_commit,
                "mapper_version": current.mapper_version,
                "tool_contract_version": current.tool_contract_version,
                "run_mode": current.run_mode,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        payload, _ = await _post_embedded_chat_completion(
            session_id=current.specialist_session_id,
            message=current.message,
            model=model,
            system_message=pin_message,
            hermes_home=profile_home,
            user_id=current.organization_id,
            workspace_id=current.workspace_id,
        )
        return TenderExpertResult.from_chat_payload(current, payload)

    return await _TENDER_RUNTIME_EXPERT_BRIDGE.run(request, invoke)


@isolated_chat_turn
async def _stream_embedded_chat_completion(
    *,
    session_id: str,
    upload_session_id: str | None = None,
    message: str,
    model: str | None = None,
    system_message: str | None = None,
    hermes_home: Path | None = None,
    user_id: str | None = None,
    workspace_id: str | None = None,
    request_metadata: dict[str, str] | None = None,
    _execution_service=None,
) -> Response:
    service = _execution_service or get_embedded_hermes_api_service()
    await service.ensure_started()

    headers = {
        "Authorization": f"Bearer {service.api_key}",
        "Content-Type": "application/json",
        "X-Hermes-Session-Id": session_id,
    }
    if hermes_home is not None:
        from agents.default_workspace_plugins import ensure_default_workspace_plugins
        ensure_default_workspace_plugins(hermes_home)
        headers["X-Semantier-Hermes-Home"] = str(hermes_home.resolve())
    if user_id:
        headers[_SEMANTIER_USER_ID_HEADER] = user_id
    if workspace_id:
        headers[_SEMANTIER_WORKSPACE_ID_HEADER] = workspace_id
    headers[_SEMANTIER_UPLOAD_SESSION_ID_HEADER] = upload_session_id or session_id
    boundary_session_id = upload_session_id or session_id
    boundary_workspace_id = workspace_id or _workspace_id_from_hermes_home_path(hermes_home)
    headers["X-Hermes-Execution-Boundary"] = _boundary_header_payload(
        workspace_id=boundary_workspace_id,
        user_id=user_id,
        session_id=boundary_session_id,
        hermes_home=hermes_home,
        request_metadata=request_metadata,
    )
    headers["X-Hermes-Execution-Boundary-Trust"] = "semantier-internal"
    scope = None
    resolved_workspace_id = _workspace_id_from_hermes_home_path(hermes_home)
    if resolved_workspace_id:
        scope = SandboxScope(
            workspace_id=resolved_workspace_id,
            lane="interactive_session",
            scope_id=session_id,
            adapter_key=None,
            network_enabled=False,
        )
        headers[_SEMANTIER_SANDBOX_KEY_HEADER] = sandbox_key_for_request(scope)
    messages: list[dict[str, Any]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": message})
    payload: dict[str, Any] = {
        "messages": messages,
        "stream": True,
        "_semantier_request_metadata": request_metadata or {},
    }
    if model:
        payload["model"] = model

    timeout = httpx.Timeout(connect=30.0 if _execution_service is not None else 5.0, read=300.0, write=60.0, pool=60.0)
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False)
    try:
        upstream_request = client.build_request(
            "POST",
            f"{service.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        with bind_sandbox_scope(scope):
            upstream_response = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail=f"Embedded Hermes API unavailable: {exc.__class__.__name__}",
        ) from exc

    if upstream_response.status_code >= 400:
        try:
            detail = ""
            try:
                error_payload = await upstream_response.aread()
                if error_payload:
                    try:
                        parsed = httpx.Response(
                            status_code=upstream_response.status_code,
                            content=error_payload,
                            headers=upstream_response.headers,
                        ).json()
                        if isinstance(parsed, dict):
                            error = parsed.get("error")
                            if isinstance(error, dict):
                                detail = str(error.get("message") or "").strip()
                            elif isinstance(parsed.get("detail"), str):
                                detail = str(parsed["detail"]).strip()
                    except Exception:
                        detail = error_payload.decode(errors="ignore").strip()
            finally:
                await upstream_response.aclose()
                await client.aclose()
        except Exception:
            detail = ""
        if not detail:
            detail = "Embedded Hermes API request failed"
        raise HTTPException(status_code=upstream_response.status_code, detail=detail)

    response_headers = _filtered_headers(
        upstream_response.headers,
        excluded=_RESPONSE_HEADER_EXCLUDE,
    )

    async def stream_body():
        try:
            async for chunk in upstream_response.aiter_raw():
                yield chunk
        finally:
            await upstream_response.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=upstream_response.status_code,
        headers=response_headers,
    )
