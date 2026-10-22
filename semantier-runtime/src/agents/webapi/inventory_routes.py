"""Runtime inventory and read-only skill/plugin diagnostics routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from pathlib import Path

from fastapi import APIRouter, Request

from agents.hermes_bundled_skill_adapter import (
    GovernedDiagnosticInspectionContext,
    bundled_skill_exposure_enabled,
)
from agents.webapi.deps import ensure_repo_hermes_agent_on_path, request_context
from agents.webapi.errors import bad_gateway
from runtime_paths import platform_runtime_root


from agents.webapi.inventory_service import (
    list_plugins_inventory,
    list_skills_inventory,
    list_toolsets_inventory,
)

router = APIRouter(tags=["webapi-gateway"])


@router.get("/system/skills")
async def system_skills(request: Request):
    ctx = request_context(request)
    workspace_hermes_home = ctx.hermes_home if ctx.authenticated else None
    diagnostic_context = None
    if ctx.authenticated:
        request_id = str(request.headers.get("X-Request-Id") or "").strip()
        if bundled_skill_exposure_enabled() and ctx.organization_id and request_id:
            diagnostic_context = GovernedDiagnosticInspectionContext(
                user_id=ctx.user_id,
                workspace_id=ctx.workspace_id,
                organization_id=ctx.organization_id,
                role=str(ctx.member_role or ""),
                request_id=request_id,
                inspected_at_utc=datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                audit_actor=ctx.user_id,
                audit_source="webapi:/system/skills",
            )
    payload = list_skills_inventory(
        workspace_hermes_home=workspace_hermes_home,
        governed_diagnostic_context=diagnostic_context,
        authenticated_context=ctx if ctx.authenticated else None,
    )
    repo_application_skills_root = (Path(__file__).resolve().parents[2] / "skills").resolve()
    visible_skills = []
    for skill in payload.get("skills", []):
        if not isinstance(skill, dict):
            continue
        source_path = str(skill.get("sourcePath") or "")
        try:
            is_repo_application_skill = Path(source_path).resolve().is_relative_to(
                repo_application_skills_root
            )
        except (OSError, ValueError):
            is_repo_application_skill = False
        if skill.get("builtin") or is_repo_application_skill:
            continue
        visible_skills.append(skill)
    payload["skills"] = visible_skills
    payload["total"] = len(visible_skills)
    payload["categories"] = sorted(
        {"All", *[str(skill.get("category") or "general") for skill in visible_skills]},
        key=str.lower,
    )
    return payload


@router.get("/system/skills/check-updates")
async def system_skills_check_updates(request: Request):
    ctx = request_context(request)
    target_home = ctx.hermes_home if ctx.authenticated else platform_runtime_root()
    name = str(request.query_params.get("name") or "").strip() or None

    ensure_repo_hermes_agent_on_path()
    from tools.skills_hub import HubLockFile
    from tools.skills_hub_install import check_for_skill_updates

    lock = HubLockFile(path=target_home / "skills" / ".hub" / "lock.json")
    try:
        update_results = check_for_skill_updates(name=name, lock=lock)
    except Exception as exc:
        raise bad_gateway(f"failed to check skill updates: {exc}") from exc

    sanitized_updates: list[dict[str, Any]] = []
    for item in update_results:
        if not isinstance(item, dict):
            continue
        sanitized_updates.append(
            {
                "name": str(item.get("name") or ""),
                "identifier": str(item.get("identifier") or ""),
                "source": str(item.get("source") or ""),
                "status": str(item.get("status") or "unavailable"),
                "currentHash": str(item.get("current_hash") or ""),
                "latestHash": str(item.get("latest_hash") or ""),
            }
        )

    return {"ok": True, "updates": sanitized_updates}


@router.get("/system/tools")
async def system_tools(request: Request):
    ctx = request_context(request)
    workspace_hermes_home = ctx.hermes_home if ctx.authenticated else None
    return list_toolsets_inventory(workspace_hermes_home=workspace_hermes_home)


@router.get("/system/plugins")
async def system_plugins(request: Request):
    ctx = request_context(request)
    workspace_hermes_home = ctx.hermes_home if ctx.authenticated else None
    return list_plugins_inventory(workspace_hermes_home=workspace_hermes_home)
