"""Selected task composition's plugin facade; lifecycle stays in pinned Hermes.

Package-source approval and current authority are core concerns. This facade
accepts an approved catalog identifier, never a browser-supplied source URL.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import re
import threading

from fastapi import APIRouter, HTTPException, Request
from agents.hermes_sdk import ProfileSdk, SdkBootstrapRequired


class WorkspacePlugins:
    def __init__(self):
        self._lock = threading.Lock()
        self._profiles = {}

    def worker(self, ctx):
        # The authenticated context's home is resolved by the core DB binding.
        key = (ctx.user_id, ctx.workspace_id, str(ctx.hermes_home.resolve()))
        with self._lock:
            if key not in self._profiles:
                from agents.auth_db import get_workspace_config
                from agents.auth_db_postgres import merge_workspace_config

                profile = Path(ctx.hermes_home) / "sdk"
                snapshot = get_workspace_config(ctx.workspace_id).get(
                    "sdk_profile_config"
                )
                if snapshot is None and (profile / "config.yaml").exists():
                    raise RuntimeError("SDK_CONFIG_MIGRATION_REQUIRED")
                worker = ProfileSdk(profile=profile)
                try:
                    if snapshot is not None:
                        if not isinstance(snapshot, dict):
                            raise ValueError("SDK_CONFIG_INVALID")
                        worker.call("profile.configure", config=snapshot)
                    else:
                        merge_workspace_config(
                            ctx.workspace_id,
                            {"sdk_profile_config": worker.call("profile.config")},
                        )
                except BaseException:
                    worker.close()
                    raise
                self._profiles[key] = worker
            return self._profiles[key]

    def discard(self, ctx):
        key = (ctx.user_id, ctx.workspace_id, str(ctx.hermes_home.resolve()))
        with self._lock:
            worker = self._profiles.pop(key, None)
            if worker:
                worker.close()

    def close(self):
        with self._lock:
            for worker in self._profiles.values():
                worker.close()
            self._profiles.clear()


def plugin_router(*, context, service):
    router = APIRouter(tags=["plugins"])

    def actor(request, *, write=False):
        ctx = context(request)
        if write and ctx.member_role not in {"owner", "admin"}:
            raise HTTPException(403, "Workspace administrator required")
        return ctx

    async def call(ctx, operation, **args):
        try:
            worker = service.worker(ctx)
            result = await asyncio.to_thread(worker.call, operation, **args)
            if operation in {
                "plugins.install",
                "plugins.enable",
                "plugins.update",
                "plugins.remove",
            }:
                from agents.auth_db_postgres import merge_workspace_config

                try:
                    snapshot = await asyncio.to_thread(worker.call, "profile.config")
                    await asyncio.to_thread(
                        merge_workspace_config,
                        ctx.workspace_id,
                        {"sdk_profile_config": snapshot},
                    )
                except BaseException:
                    service.discard(ctx)
                    raise
            return result
        except SdkBootstrapRequired as exc:
            raise HTTPException(
                503, "Pinned upstream SDK dependency bootstrap required"
            ) from exc
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            # Native exception text and configuration never enter browser/log DTOs.
            raise HTTPException(502, "Pinned plugin operation unavailable") from exc

    @router.get("/system/plugins")
    async def inventory(request: Request):
        ctx = actor(request)
        plugins = await call(ctx, "plugins.inventory")
        for plugin in plugins:
            plugin["enabled"] = plugin["status"] == "enabled"
        return {
            "plugins": plugins,
            "total": len(plugins),
            "scope": "workspace",
            "activation_scope": "sdk_profile",
            "task_activation": "not_qualified",
        }

    @router.post("/system/skills/install")
    async def install(request: Request):
        from agents.auth_db import get_workspace_config

        ctx = actor(request, write=True)
        body = await request.json()
        identifier = body.get("identifier") if isinstance(body, dict) else None
        if not isinstance(identifier, str):
            raise HTTPException(400, "Plugin identifier required")
        approved = (
            get_workspace_config(ctx.workspace_id).get("sdk_plugin_sources") or {}
        ).get(identifier)
        if not isinstance(approved, dict) or not re.fullmatch(
            r"[a-f0-9]{40}", str(approved.get("ref", ""))
        ):
            raise HTTPException(403, "Approved exact-ref plugin source required")
        if approved.get("dependency_policy") != "qualified_no_dependencies":
            raise HTTPException(
                409,
                "Dependency-bearing plugin lifecycle remains on the incumbent runtime",
            )
        # The selected composition admits only exact-ref no-dependency packages.
        # Native scan/admission remains authoritative; no second installer exists.
        return await call(
            ctx,
            "plugins.install",
            identifier=approved["source"],
            ref=approved["ref"],
            force=bool(body.get("force")),
            enable=bool(body.get("enable", True)),
            assume_deps_consent=body.get("deps_consent") is True,
        )

    @router.post("/system/plugins/{name}/enabled")
    async def toggle(name: str, request: Request):
        ctx = actor(request, write=True)
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("enabled"), bool):
            raise HTTPException(400, "Boolean enabled required")
        expected = None
        if body["enabled"]:
            from agents.auth_db import get_workspace_config

            sources = (
                get_workspace_config(ctx.workspace_id).get("sdk_plugin_sources") or {}
            )
            refs = {
                row.get("ref")
                for row in sources.values()
                if isinstance(row, dict)
                and row.get("name") == name
                and row.get("dependency_policy") == "qualified_no_dependencies"
                and re.fullmatch(r"[a-f0-9]{40}", str(row.get("ref", "")))
            }
            if len(refs) != 1:
                raise HTTPException(
                    409, "Plugin enablement requires one exact qualified pin"
                )
            expected = refs.pop()
        return await call(
            ctx,
            "plugins.enable",
            name=name,
            enabled=body["enabled"],
            expected_ref=expected,
        )

    @router.post("/system/plugins/{name}/update")
    async def update(name: str, request: Request):
        ctx = actor(request, write=True)
        body = await request.json()
        return await call(
            ctx,
            "plugins.update",
            name=name,
            accept_capabilities=body.get("accept_capabilities") is True,
            force=body.get("force") is True,
        )

    @router.delete("/system/plugins/{name}")
    async def remove(name: str, request: Request):
        ctx = actor(request, write=True)
        return await call(ctx, "plugins.remove", name=name)

    return router
