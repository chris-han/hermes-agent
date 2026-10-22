from __future__ import annotations
from storage import hosted_postgres as _hosted

def _storage_statement(*,sqlite,postgresql):
    return _hosted.sql_query(sqlite,postgresql)

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
from agents.webapi.deps import ensure_repo_hermes_agent_on_path as _ensure_repo_hermes_agent_on_path
from agents.webapi.gateway_channel_service import _workspace_id_from_hermes_home_path
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

def _normalize_job_deliver_for_api(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _normalize_job_deliver_for_cron(value: Any) -> str | None:
    parts = _normalize_job_deliver_for_api(value)
    return ",".join(parts) if parts else None


def _normalize_job_repeat_for_cron(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("times")
        if value is None:
            return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_job_schedule_for_api(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        expr = str(value.get("expr") or value.get("display") or "* * * * *").strip()
        return {
            "kind": str(value.get("kind") or "cron").strip() or "cron",
            "expr": expr or "* * * * *",
            "display": str(value.get("display") or expr or "* * * * *").strip(),
        }
    expr = str(value or "").strip() or "* * * * *"
    return {"kind": "cron", "expr": expr, "display": expr}


def _cron_job_for_api(job: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job)
    job_id = str(payload.get("id") or payload.get("job_id") or "").strip()
    if job_id:
        payload["id"] = job_id
    schedule = payload.get("schedule")
    if isinstance(schedule, dict):
        display = str(
            payload.get("schedule_display")
            or schedule.get("display")
            or schedule.get("expr")
            or ""
        ).strip()
        payload["schedule_display"] = display
    elif schedule is not None:
        payload["schedule"] = _normalize_job_schedule_for_api(schedule)
        payload["schedule_display"] = payload["schedule"]["display"]
    else:
        payload["schedule"] = _normalize_job_schedule_for_api(payload.get("schedule_display"))
        payload["schedule_display"] = payload["schedule"]["display"]
    payload["enabled"] = bool(payload.get("enabled", True))
    payload["state"] = str(payload.get("state") or ("scheduled" if payload["enabled"] else "paused"))
    payload.setdefault("prompt", "")
    payload.setdefault("skills", [])
    payload["deliver"] = _normalize_job_deliver_for_api(payload.get("deliver"))
    last_status = str(payload.get("last_status") or "").strip().lower()
    if last_status == "ok":
        payload["last_run_success"] = True
    elif last_status in {"error", "failed", "failure"}:
        payload["last_run_success"] = False
    else:
        payload.setdefault("last_run_success", None)
    payload["source"] = "cron"
    return payload


def _workspace_cron_scope(ctx: RequestContext) -> str:
    return f"workspace:{ctx.workspace_id}"


def _list_workspace_cron_jobs(ctx: RequestContext, *, include_disabled: bool = True) -> list[dict[str, Any]]:
    from agents import cron_store

    jobs = [
        _cron_job_for_api(job)
        for job in cron_store.list_jobs(_workspace_cron_scope(ctx))
    ]
    if include_disabled:
        return jobs
    return [job for job in jobs if bool(job.get("enabled", True))]


def _get_workspace_cron_job(ctx: RequestContext, job_id: str) -> dict[str, Any] | None:
    for job in _list_workspace_cron_jobs(ctx, include_disabled=True):
        if str(job.get("id") or "") == str(job_id):
            return job
    return None


@contextmanager
def _workspace_cron_runtime(ctx: RequestContext):
    from agents.cron_storage_provider import SemantierCronStorageProvider
    from cron.jobs import use_cron_store
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    home = Path(ctx.hermes_home).expanduser().resolve()
    if SemantierCronStorageProvider._scope(home) != _workspace_cron_scope(ctx):
        raise RuntimeError("Cron runtime home does not match the authenticated workspace")
    token = set_hermes_home_override(home)
    try:
        with use_cron_store(home):
            yield
    finally:
        reset_hermes_home_override(token)


def _create_workspace_cron_job(ctx: RequestContext, body: dict[str, Any]) -> dict[str, Any]:
    from agents.cron_home_provider import ensure_semantier_cron_storage_provider
    ensure_semantier_cron_storage_provider()
    from agents.cron_store import CronStorageConflict

    try:
        with _workspace_cron_runtime(ctx):
            from cron.jobs import create_job as create_cron_job

            job = create_cron_job(
                prompt=body.get("prompt"),
                schedule=str(body.get("schedule") or "* * * * *"),
                name=body.get("name"),
                repeat=_normalize_job_repeat_for_cron(body.get("repeat")),
                deliver=_normalize_job_deliver_for_cron(body.get("deliver")),
                skills=body.get("skills") if isinstance(body.get("skills"), list) else None,
                script=body.get("script") if isinstance(body.get("script"), str) else None,
                no_agent=bool(body.get("no_agent", False)),
                model=body.get("model") if isinstance(body.get("model"), str) else None,
                provider=body.get("provider") if isinstance(body.get("provider"), str) else None,
                base_url=body.get("base_url") if isinstance(body.get("base_url"), str) else None,
            )
    except CronStorageConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    persisted = _get_workspace_cron_job(ctx, str(job["id"]))
    if persisted is None:
        raise RuntimeError("Created cron job is unavailable in the authoritative store")
    return persisted


def _update_workspace_cron_job(ctx: RequestContext, job_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    from agents.cron_home_provider import ensure_semantier_cron_storage_provider
    ensure_semantier_cron_storage_provider()
    cron_updates = dict(updates)
    if "deliver" in cron_updates:
        cron_updates["deliver"] = _normalize_job_deliver_for_cron(cron_updates.get("deliver"))
    if "repeat" in cron_updates:
        cron_updates["repeat"] = _normalize_job_repeat_for_cron(cron_updates.get("repeat"))
    from agents.cron_store import CronStorageConflict

    try:
        with _workspace_cron_runtime(ctx):
            from cron.jobs import update_job as update_cron_job

            job = update_cron_job(job_id, cron_updates)
    except CronStorageConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not isinstance(job, dict):
        return None
    persisted = _get_workspace_cron_job(ctx, str(job["id"]))
    if persisted is None:
        raise RuntimeError("Updated cron job is unavailable in the authoritative store")
    return persisted


def _delete_workspace_cron_job(ctx: RequestContext, job_id: str) -> bool:
    from agents.cron_home_provider import ensure_semantier_cron_storage_provider
    ensure_semantier_cron_storage_provider()
    from agents.cron_store import CronStorageConflict

    try:
        with _workspace_cron_runtime(ctx):
            from cron.jobs import remove_job

            return remove_job(job_id)
    except CronStorageConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _job_output_response(ctx: RequestContext, job_id: str, *, limit: int = 10) -> dict[str, Any]:
    from agents import cron_store

    outputs = cron_store.list_outputs(f"workspace:{ctx.workspace_id}", job_id, limit=limit)
    return {"ok": True, "outputs": outputs, "runs": outputs}

_KANBAN_BOARD_COLUMNS = (
    "triage",
    "todo",
    "scheduled",
    "ready",
    "running",
    "blocked",
    "review",
    "done",
)


@contextmanager
def _kanban_connection(ctx: RequestContext, *, board: str | None = None):
    _ensure_repo_hermes_agent_on_path()
    from hermes_cli import kanban_db
    from hermes_cli.kanban_db_connect import connect

    with _temporary_hermes_home(str(ctx.hermes_home)):
        kanban_db.init_db(board=board)
        conn = connect(board=board)
        try:
            yield kanban_db, conn
        finally:
            conn.close()


def _kanban_int(value: Any, *, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"invalid integer value: {value!r}")


def _kanban_task_dict(task: Any) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "body": task.body,
        "assignee": task.assignee,
        "status": task.status,
        "priority": task.priority,
        "tenant": task.tenant,
        "user_id": getattr(task, "user_id", None),
        "workspace_id": getattr(task, "workspace_id", None),
        "workspace_kind": task.workspace_kind,
        "workspace_path": task.workspace_path,
        "created_by": task.created_by,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "result": task.result,
        "skills": task.skills or [],
        "current_run_id": task.current_run_id,
        "model_override": task.model_override,
    }


def _kanban_comment_dict(comment: Any) -> dict[str, Any]:
    return {
        "id": comment.id,
        "task_id": comment.task_id,
        "author": comment.author,
        "body": comment.body,
        "created_at": comment.created_at,
    }


def _kanban_event_dict(event: Any) -> dict[str, Any]:
    return {
        "id": event.id,
        "task_id": event.task_id,
        "kind": event.kind,
        "payload": event.payload,
        "created_at": event.created_at,
        "run_id": event.run_id,
    }


def _kanban_links_for(conn: Any, task_id: str) -> dict[str, list[str]]:
    from hermes_cli import kanban_db

    return {
        "parents": kanban_db.parent_ids(conn, task_id),
        "children": kanban_db.child_ids(conn, task_id),
    }


def _set_kanban_status_direct(conn: Any, task_id: str, status: str) -> bool:
    now = int(time.time())
    from hermes_cli import kanban_db

    with kanban_db.write_txn(conn):
        cur = conn.execute(
            _storage_statement(sqlite="UPDATE tasks SET status = ?, claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL WHERE id = ?",postgresql='UPDATE tasks SET status = %s, claim_lock = NULL, claim_expires = NULL, worker_pid = NULL WHERE id = %s'),
            (status, task_id),
        )
        if cur.rowcount != 1:
            return False
        conn.execute(
            _storage_statement(sqlite="INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'edited', ?, ?)",postgresql="INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (%s, 'edited', %s, %s) RETURNING id"),
            (task_id, json.dumps({"status": status}), now),
        )
    return True


def _kanban_assignee_payload(conn: Any) -> dict[str, Any]:
    from hermes_cli import kanban_db

    return {"assignees": kanban_db.known_assignees(conn)}

@contextmanager
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
