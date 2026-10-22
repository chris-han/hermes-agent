from __future__ import annotations

from datetime import datetime, timezone
import uuid

from fastapi import APIRouter

from fastapi import HTTPException, Request

from agents.auth_session import request_context_from_request
from agents.hermes_bundled_skill_adapter import (
    GovernedDiagnosticInspectionContext,
    bundled_skill_exposure_enabled,
)
from runtime_paths import platform_runtime_root
from agents.webapi.deps import ensure_repo_hermes_agent_on_path as _ensure_repo_hermes_agent_on_path
from agents.webapi.inventory_service import list_plugins_inventory, list_skills_inventory
from agents.webapi.system_skill_service import (
    _repo_local_marketplace_root,
    _infer_marketplace_package_type,
    install_repo_local_plugin_transaction,
    _plugin_name_from_marketplace_path,
    _resolve_repo_local_plugin_source,
    uninstall_repo_local_plugin_transaction,
)

router = APIRouter(tags=["webapi-gateway"])


@router.post("/system/skills/install")
async def system_skills_install(request: Request):
    ctx = request_context_from_request(request)
    target_home = ctx.hermes_home if ctx.authenticated else platform_runtime_root()
    workspace_hermes_home = ctx.hermes_home if ctx.authenticated else None

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    identifier = str(body.get("identifier") or "").strip()
    if not identifier:
        raise HTTPException(status_code=400, detail="identifier required")

    category = str(body.get("category") or "").strip()
    package_type = str(body.get("package_type") or body.get("packageType") or "").strip()
    package_path = str(body.get("path") or "").strip()
    force = bool(body.get("force"))
    config_payload = body.get("config") if isinstance(body.get("config"), dict) else {}
    resolved_package_type = _infer_marketplace_package_type(identifier, package_type)

    _ensure_repo_hermes_agent_on_path()
    from rich.console import Console
    from agents.semantier_agent import _hermes_home_bound
    from hermes_cli.config import load_config, save_config
    from hermes_cli.skills_hub import do_install

    if resolved_package_type == "plugin":
        source_dir, plugin_name = _resolve_repo_local_plugin_source(identifier, package_path)
        before_inventory = list_plugins_inventory(
            workspace_hermes_home=workspace_hermes_home,
        )
        before_ids = {
            str(item.get("id") or "")
            for item in before_inventory.get("plugins", [])
            if isinstance(item, dict)
        }
        install_repo_local_plugin_transaction(
            target_home=target_home,
            source_dir=source_dir,
            plugin_name=plugin_name,
            force=force,
            workspace_id=ctx.workspace_id,
            identifier=identifier,
            package_path=package_path or f"plugins/{plugin_name}",
        )
        after_inventory = list_plugins_inventory(
            workspace_hermes_home=workspace_hermes_home,
        )
        after_plugins = after_inventory.get("plugins", [])
        new_records = [
            item
            for item in after_plugins
            if isinstance(item, dict) and str(item.get("id") or "") not in before_ids
        ]
        installed_record = next(
            (
                item
                for item in after_plugins
                if isinstance(item, dict) and str(item.get("id") or "") == plugin_name
            ),
            None,
        )
        return {
            "ok": True,
            "identifier": identifier,
            "packageType": "plugin",
            "installed": new_records[0] if new_records else installed_record,
        }

    before_inventory = list_skills_inventory(
        workspace_hermes_home=workspace_hermes_home,
    )
    before_ids = {
        str(item.get("id") or "")
        for item in before_inventory.get("skills", [])
        if isinstance(item, dict)
    }

    console = Console(record=True, force_terminal=False, no_color=True, width=120)
    with _hermes_home_bound(target_home):
        do_install(
            identifier,
            category=category,
            force=force,
            console=console,
            skip_confirm=True,
        )
        if config_payload:
            config = load_config()
            skills_cfg = config.setdefault("skills", {})
            if not isinstance(skills_cfg, dict):
                skills_cfg = {}
                config["skills"] = skills_cfg
            runtime_skill_config = skills_cfg.setdefault("config", {})
            if not isinstance(runtime_skill_config, dict):
                runtime_skill_config = {}
                skills_cfg["config"] = runtime_skill_config
            runtime_skill_config.update(config_payload)
            save_config(config)

    transcript = console.export_text().strip()
    after_inventory = list_skills_inventory(
        workspace_hermes_home=workspace_hermes_home,
    )
    after_skills = after_inventory.get("skills", [])
    new_records = [
        item
        for item in after_skills
        if isinstance(item, dict) and str(item.get("id") or "") not in before_ids
    ]

    if new_records or "Installed:" in transcript:
        return {
            "ok": True,
            "identifier": identifier,
            "installed": new_records[0] if new_records else None,
            "output": transcript,
        }

    if "already installed" in transcript.lower():
        raise HTTPException(status_code=409, detail=transcript or "skill already installed")

    raise HTTPException(
        status_code=400,
        detail=transcript or f"failed to install skill '{identifier}'",
    )


@router.put("/system/skills/toggle")
async def system_skills_toggle(request: Request):
    ctx = request_context_from_request(request)
    target_home = ctx.hermes_home if ctx.authenticated else platform_runtime_root()

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    enabled = bool(body.get("enabled"))
    workspace_hermes_home = ctx.hermes_home if ctx.authenticated else None
    diagnostic_context = None
    if (
        ctx.authenticated
        and bundled_skill_exposure_enabled()
        and ctx.organization_id
    ):
        request_id = str(request.headers.get("X-Request-Id") or uuid.uuid4())
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
            audit_source="webapi:/system/skills/toggle",
        )
    inventory = list_skills_inventory(
        workspace_hermes_home=workspace_hermes_home,
        governed_diagnostic_context=diagnostic_context,
    )
    skill_record = next(
        (
            item
            for item in inventory.get("skills", [])
            if isinstance(item, dict)
            and (
                str(item.get("id") or "") == name
                or str(item.get("name") or "") == name
            )
        ),
        None,
    )
    if skill_record is None:
        raise HTTPException(status_code=404, detail="skill not found")
    if skill_record.get("canModify") is not True:
        raise HTTPException(
            status_code=403,
            detail="Semantier built-in skills cannot be toggled from the skills page",
        )

    _ensure_repo_hermes_agent_on_path()
    from agents.semantier_agent import _hermes_home_bound
    from hermes_cli.skills_config import get_disabled_skills, save_disabled_skills
    from hermes_cli.config import load_config

    with _hermes_home_bound(target_home):
        config = load_config()
        disabled = get_disabled_skills(config)
        if enabled:
            disabled.discard(name)
        else:
            disabled.add(name)
        save_disabled_skills(config, disabled)

    return {"ok": True, "name": name, "enabled": enabled}


@router.post("/system/skills/uninstall")
async def system_skills_uninstall(request: Request):
    ctx = request_context_from_request(request)
    target_home = ctx.hermes_home if ctx.authenticated else platform_runtime_root()
    workspace_hermes_home = ctx.hermes_home if ctx.authenticated else None

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    name = str(body.get("name") or body.get("skillId") or "").strip()
    identifier = str(body.get("identifier") or "").strip()
    package_type = str(body.get("package_type") or body.get("packageType") or "").strip()
    package_path = str(body.get("path") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    resolved_package_type = _infer_marketplace_package_type(
        identifier or name,
        package_type,
    )

    if resolved_package_type == "plugin":
        normalized_plugin_name = name
        if package_path.strip():
            normalized_plugin_name = _plugin_name_from_marketplace_path(package_path)
        before_inventory = list_plugins_inventory(
            workspace_hermes_home=workspace_hermes_home,
        )
        before_ids = {
            str(item.get("id") or "")
            for item in before_inventory.get("plugins", [])
            if isinstance(item, dict)
        }
        uninstall_repo_local_plugin_transaction(
            target_home=target_home,
            plugin_name=normalized_plugin_name,
            workspace_id=ctx.workspace_id,
        )
        after_inventory = list_plugins_inventory(
            workspace_hermes_home=workspace_hermes_home,
        )
        after_ids = {
            str(item.get("id") or "")
            for item in after_inventory.get("plugins", [])
            if isinstance(item, dict)
        }
        if normalized_plugin_name in after_ids:
            raise HTTPException(
                status_code=500,
                detail=f"plugin '{normalized_plugin_name}' still present after uninstall",
            )
        return {
            "ok": True,
            "name": normalized_plugin_name,
            "packageType": "plugin",
            "removed": normalized_plugin_name in before_ids,
        }

    _ensure_repo_hermes_agent_on_path()
    from agents.semantier_agent import _hermes_home_bound
    from tools.skills_hub import uninstall_skill

    with _hermes_home_bound(target_home):
        ok, message = uninstall_skill(name)

    if not ok:
        raise HTTPException(status_code=400, detail=message)

    return {"ok": True, "name": name, "message": message}
