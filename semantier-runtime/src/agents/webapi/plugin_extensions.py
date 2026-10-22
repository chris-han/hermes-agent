"""Generic web API extension loading for installed/repo plugins."""

from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter


def _repo_plugin_root() -> Path:
    return Path(__file__).resolve().parents[3] / "semantier-skills" / "plugins"


def _repo_core_plugin_root() -> Path:
    return Path(__file__).resolve().parents[2] / "plugins"


def _plugin_roots() -> list[Path]:
    roots: list[Path] = []
    raw = os.environ.get("SEMANTIER_WEBAPI_PLUGIN_ROOTS")
    if raw:
        for item in raw.split(os.pathsep):
            item = item.strip()
            if item:
                roots.append(Path(item).expanduser().resolve())
    roots.append(_repo_plugin_root())
    roots.append(_repo_core_plugin_root())
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def _load_manifest(plugin_dir: Path) -> dict[str, Any]:
    manifest_path = plugin_dir / "plugin.yaml"
    if not manifest_path.exists():
        return {}
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _import_declared_module(plugin_root: Path, module_name: str):
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))
    return importlib.import_module(module_name)


@lru_cache(maxsize=1)
def _webapi_extension_modules() -> tuple[Any, ...]:
    modules: list[Any] = []
    for plugin_root in _plugin_roots():
        if not plugin_root.exists():
            continue
        for plugin_dir in sorted(item for item in plugin_root.iterdir() if item.is_dir()):
            manifest = _load_manifest(plugin_dir)
            webapi = manifest.get("webapi")
            if not isinstance(webapi, dict) or webapi.get("enabled") is not True:
                continue
            module_name = str(webapi.get("module") or "").strip()
            if not module_name:
                continue
            try:
                modules.append(_import_declared_module(plugin_root, module_name))
            except Exception:
                continue
    return tuple(modules)


def include_plugin_webapi_routers(router: APIRouter) -> None:
    for module in _webapi_extension_modules():
        plugin_router = getattr(module, "router", None)
        if isinstance(plugin_router, APIRouter):
            router.include_router(plugin_router)


def plugin_route_policy_entries() -> dict[tuple[str, str], str]:
    entries: dict[tuple[str, str], str] = {}
    for module in _webapi_extension_modules():
        raw = getattr(module, "ROUTE_POLICY_MAP", {})
        if isinstance(raw, dict):
            entries.update(raw)
    return entries


def plugin_route_authz_class_entries() -> dict[tuple[str, str], str]:
    entries: dict[tuple[str, str], str] = {}
    for module in _webapi_extension_modules():
        raw = getattr(module, "ROUTE_AUTHZ_CLASS_MAP", {})
        if isinstance(raw, dict):
            entries.update(raw)
    return entries


__all__ = [
    "include_plugin_webapi_routers",
    "plugin_route_authz_class_entries",
    "plugin_route_policy_entries",
]
