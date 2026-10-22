from __future__ import annotations
from storage import hosted_postgres as _hosted

def _storage_statement(*,sqlite,postgresql):
    return _hosted.sql_query(sqlite,postgresql)

from fastapi import APIRouter

import json
import time
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone

from agents.coordination_association import CoordinationAssociationStore, WorkRef
from typing import Any

from fastapi import HTTPException, Query, Request
from fastapi.responses import JSONResponse

from agents.webapi.deps import (
    ensure_repo_hermes_agent_on_path as _ensure_repo_hermes_agent_on_path,
    require_authenticated_context as _require_authenticated_context,
)
from agents.webapi.job_service import (
    _KANBAN_BOARD_COLUMNS,
    _create_workspace_cron_job,
    _cron_job_for_api,
    _get_workspace_cron_job,
    _job_output_response,
    _kanban_assignee_payload,
    _kanban_comment_dict,
    _kanban_connection,
    _kanban_event_dict,
    _kanban_int,
    _kanban_links_for,
    _kanban_task_dict,
    _list_workspace_cron_jobs,
    _set_kanban_status_direct,
    _temporary_hermes_home,
    _update_workspace_cron_job,
)

router = APIRouter(tags=["webapi-gateway"])


@router.post("/api/start-hermes")
async def start_hermes():
    return JSONResponse(
        {
            "ok": True,
            "status": "already-running",
            "message": "semantier webapi is the required backend surface on :8899.",
        }
    )


@router.api_route("/api/jobs", methods=["GET", "POST"])
async def api_jobs(
    request: Request,
    include_disabled: bool = Query(default=True),
) -> Any:
    ctx = _require_authenticated_context(request)
    if request.method == "GET":
        jobs = _list_workspace_cron_jobs(ctx, include_disabled=include_disabled)
        return {"ok": True, "items": jobs, "jobs": jobs}
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object body required")
    execution_goal = body.get("execution_goal")
    if execution_goal is not None:
        if not isinstance(execution_goal, str) or not execution_goal.strip() or len(execution_goal) > 2000:
            raise HTTPException(status_code=400, detail="execution_goal must contain 1–2000 characters")
        from agents.webapi.deps import require_route_authorization
        from agents.route_policy import RouteAuthorizationClass
        ctx = require_route_authorization(request,
            authorization_class=RouteAuthorizationClass.TENANT_MEMBER, route_hint="POST /api/jobs")
    try:
        job = _create_workspace_cron_job(ctx, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result = {"ok": True, "job": job}
    if execution_goal is not None:
        source_version = hashlib.sha256(str(job.get("prompt") or "").encode()).hexdigest()
        goal_revision = hashlib.sha256(execution_goal.strip().encode()).hexdigest()
        context = CoordinationAssociationStore(ctx.hermes_home / "state.db").ensure_context(
            workspace_id=ctx.workspace_id, work_ref=WorkRef("job", str(job["id"]), source_version),
            goal_ref=f"job:{job['id']}:goal", goal_revision=goal_revision,
            principal_ref=f"user:{ctx.user_id}", opened_at=datetime.now(timezone.utc).isoformat(),
        )
        result["coordination_context"] = asdict(context)
    return result


@router.api_route("/api/jobs/{job_id}", methods=["GET", "POST", "PATCH", "DELETE", "PUT"])
async def api_job_detail(
    job_id: str,
    request: Request,
    action: str | None = Query(default=None),
    limit: int = Query(default=10),
) -> Any:
    ctx = _require_authenticated_context(request)
    normalized_action = str(action or "").strip()

    if request.method == "GET":
        if normalized_action in {"output", "runs"}:
            return _job_output_response(ctx, job_id, limit=limit)
        job = _get_workspace_cron_job(ctx, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"ok": True, "job": job}

    if request.method == "DELETE":
        with _temporary_hermes_home(str(ctx.hermes_home)):
            from cron.jobs import remove_job

            removed = remove_job(job_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"ok": True}

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    if request.method == "PATCH":
        job = _update_workspace_cron_job(ctx, job_id, body)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"ok": True, "job": job}

    updates = body
    if "updates" in body and isinstance(body["updates"], dict):
        updates = body["updates"]
    job = _update_workspace_cron_job(ctx, job_id, updates)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True, "job": job}


@router.api_route("/api/jobs/{job_id}/{action}", methods=["GET", "POST"])
async def api_job_action(
    job_id: str,
    action: str,
    request: Request,
    limit: int = Query(default=10),
) -> Any:
    ctx = _require_authenticated_context(request)
    normalized_action = action.strip().lower()
    if request.method == "GET":
        if normalized_action in {"output", "runs"}:
            return _job_output_response(ctx, job_id, limit=limit)
        raise HTTPException(status_code=405, detail="Job action requires POST")
    if normalized_action == "pause":
        with _temporary_hermes_home(str(ctx.hermes_home)):
            from cron.jobs import pause_job

            job = pause_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"ok": True, "job": _cron_job_for_api(job)}
    if normalized_action == "resume":
        with _temporary_hermes_home(str(ctx.hermes_home)):
            from cron.jobs import resume_job

            job = resume_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"ok": True, "job": _cron_job_for_api(job)}
    if normalized_action in {"trigger", "run", "run-if-due"}:
        with _temporary_hermes_home(str(ctx.hermes_home)):
            from cron.jobs import trigger_job

            job = trigger_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"ok": True, "job": _cron_job_for_api(job)}
    raise HTTPException(status_code=404, detail="Unknown job action")


@router.api_route(
    "/api/model/info",
    methods=["GET"],
)
async def api_model_info() -> Any:
    _ensure_repo_hermes_agent_on_path()
    from hermes_cli.web_routers.models import get_model_info

    return get_model_info()


@router.get("/api/plugins/kanban/board")
async def api_kanban_board(
    request: Request,
    tenant: str | None = Query(default=None),
    include_archived: bool = Query(default=False),
    board: str | None = Query(default=None),
) -> Any:
    ctx = _require_authenticated_context(request)
    with _kanban_connection(ctx, board=board) as (kanban_db, conn):
        tasks = kanban_db.list_tasks(
            conn,
            tenant=tenant,
            include_archived=include_archived,
        )
        columns: dict[str, list[dict[str, Any]]] = {name: [] for name in _KANBAN_BOARD_COLUMNS}
        if include_archived:
            columns["archived"] = []
        for task in tasks:
            column = task.status if task.status in columns else "todo"
            columns[column].append(_kanban_task_dict(task))
        tenants = [
            row["tenant"]
            for row in conn.execute(
                "SELECT DISTINCT tenant FROM tasks WHERE tenant IS NOT NULL ORDER BY tenant"
            ).fetchall()
        ]
        assignees = [
            row["assignee"]
            for row in conn.execute(
                "SELECT DISTINCT assignee FROM tasks WHERE assignee IS NOT NULL "
                "AND status != 'archived' ORDER BY assignee"
            ).fetchall()
        ]
        latest_event = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS latest_event_id FROM task_events"
        ).fetchone()
        return {
            "columns": [{"name": name, "tasks": column_tasks} for name, column_tasks in columns.items()],
            "tenants": tenants,
            "assignees": assignees,
            "latest_event_id": int(latest_event["latest_event_id"] if latest_event else 0),
            "now": int(time.time()),
        }


@router.get("/api/plugins/kanban/tasks/{task_id}")
async def api_kanban_task_detail(
    task_id: str,
    request: Request,
    board: str | None = Query(default=None),
) -> Any:
    ctx = _require_authenticated_context(request)
    with _kanban_connection(ctx, board=board) as (kanban_db, conn):
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        return {
            "task": _kanban_task_dict(task),
            "comments": [_kanban_comment_dict(c) for c in kanban_db.list_comments(conn, task_id)],
            "events": [_kanban_event_dict(e) for e in kanban_db.list_events(conn, task_id)],
            "links": _kanban_links_for(conn, task_id),
            "runs": [],
        }


@router.post("/api/plugins/kanban/tasks")
async def api_kanban_task_create(
    request: Request,
    board: str | None = Query(default=None),
) -> Any:
    ctx = _require_authenticated_context(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object body required")
    title = str(body.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    with _kanban_connection(ctx, board=board) as (kanban_db, conn):
        try:
            task_id = kanban_db.create_task(
                conn,
                title=title,
                body=body.get("body") if isinstance(body.get("body"), str) else None,
                assignee=body.get("assignee") if isinstance(body.get("assignee"), str) else None,
                created_by=str(ctx.user_id),
                workspace_kind=str(body.get("workspace_kind") or "scratch"),
                workspace_path=body.get("workspace_path") if isinstance(body.get("workspace_path"), str) else None,
                tenant=str(ctx.workspace_id),
                priority=_kanban_int(body.get("priority"), default=0),
                parents=body.get("parents") if isinstance(body.get("parents"), list) else (),
                triage=bool(body.get("triage", False)),
                idempotency_key=body.get("idempotency_key") if isinstance(body.get("idempotency_key"), str) else None,
                max_runtime_seconds=(
                    _kanban_int(body.get("max_runtime_seconds"))
                    if body.get("max_runtime_seconds") is not None
                    else None
                ),
                skills=body.get("skills") if isinstance(body.get("skills"), list) else None,
                board=board,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        task = kanban_db.get_task(conn, task_id)
        return {"task": _kanban_task_dict(task) if task else None}


@router.patch("/api/plugins/kanban/tasks/{task_id}")
async def api_kanban_task_update(
    task_id: str,
    request: Request,
    board: str | None = Query(default=None),
) -> Any:
    ctx = _require_authenticated_context(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object body required")
    with _kanban_connection(ctx, board=board) as (kanban_db, conn):
        if kanban_db.get_task(conn, task_id) is None:
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        if "assignee" in body:
            try:
                if not kanban_db.assign_task(
                    conn,
                    task_id,
                    body.get("assignee") if isinstance(body.get("assignee"), str) else None,
                ):
                    raise HTTPException(status_code=404, detail=f"task {task_id} not found")
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc))
        status = str(body.get("status") or "").strip()
        if status:
            ok = True
            if status == "done":
                ok = kanban_db.complete_task(
                    conn,
                    task_id,
                    result=body.get("result") if isinstance(body.get("result"), str) else None,
                    summary=body.get("summary") if isinstance(body.get("summary"), str) else None,
                    metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
                )
            elif status == "blocked":
                ok = kanban_db.block_task(
                    conn,
                    task_id,
                    reason=body.get("block_reason") if isinstance(body.get("block_reason"), str) else None,
                )
            elif status == "ready":
                current = kanban_db.get_task(conn, task_id)
                if current and current.status in {"blocked", "scheduled"}:
                    ok = kanban_db.unblock_task(conn, task_id)
                else:
                    ok = _set_kanban_status_direct(conn, task_id, "ready")
            elif status == "archived":
                ok = kanban_db.archive_task(conn, task_id)
            elif status in {"todo", "triage", "scheduled"}:
                ok = _set_kanban_status_direct(conn, task_id, status)
            elif status == "running":
                raise HTTPException(status_code=400, detail="Cannot set status to 'running' directly")
            else:
                raise HTTPException(status_code=400, detail=f"unknown status: {status}")
            if not ok:
                raise HTTPException(
                    status_code=409,
                    detail=f"status transition to {status!r} not valid from current state",
                )
        if "priority" in body:
            priority = _kanban_int(body.get("priority"), default=0)
            with kanban_db.write_txn(conn):
                conn.execute(
                    _storage_statement(sqlite="UPDATE tasks SET priority = ? WHERE id = ?",postgresql='UPDATE tasks SET priority = %s WHERE id = %s'),
                    (priority, task_id),
                )
                conn.execute(
                    _storage_statement(sqlite="INSERT INTO task_events (task_id, kind, payload, created_at) "
                    "VALUES (?, 'reprioritized', ?, ?)",postgresql="INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (%s, 'reprioritized', %s, %s) RETURNING id"),
                    (task_id, json.dumps({"priority": priority}), int(time.time())),
                )
        if "title" in body or "body" in body:
            with kanban_db.write_txn(conn):
                sets: list[str] = []
                values: list[Any] = []
                if "title" in body:
                    title = str(body.get("title") or "").strip()
                    if not title:
                        raise HTTPException(status_code=400, detail="title cannot be empty")
                    sets.append(_storage_statement(sqlite="title = ?",postgresql='title = %s'))
                    values.append(title)
                if "body" in body:
                    sets.append(_storage_statement(sqlite="body = ?",postgresql='body = %s'))
                    values.append(body.get("body") if isinstance(body.get("body"), str) else None)
                if sets:
                    values.append(task_id)
                    conn.execute(_storage_statement(sqlite=f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?",postgresql=f"UPDATE tasks SET {', '.join(sets)} WHERE id = %s"), tuple(values))
                    conn.execute(
                        _storage_statement(sqlite="INSERT INTO task_events (task_id, kind, payload, created_at) "
                        "VALUES (?, 'edited', NULL, ?)",postgresql="INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (%s, 'edited', NULL, %s) RETURNING id"),
                        (task_id, int(time.time())),
                    )
        task = kanban_db.get_task(conn, task_id)
        return {"task": _kanban_task_dict(task) if task else None}


@router.delete("/api/plugins/kanban/tasks/{task_id}")
async def api_kanban_task_delete(
    task_id: str,
    request: Request,
    board: str | None = Query(default=None),
) -> Any:
    ctx = _require_authenticated_context(request)
    with _kanban_connection(ctx, board=board) as (kanban_db, conn):
        if not kanban_db.delete_task(conn, task_id):
            raise HTTPException(status_code=404, detail=f"task {task_id} not found")
        return {"deleted": True, "task_id": task_id}


@router.get("/api/plugins/kanban/assignees")
async def api_kanban_assignees(
    request: Request,
    board: str | None = Query(default=None),
) -> Any:
    ctx = _require_authenticated_context(request)
    with _kanban_connection(ctx, board=board) as (_kanban_db, conn):
        return _kanban_assignee_payload(conn)
