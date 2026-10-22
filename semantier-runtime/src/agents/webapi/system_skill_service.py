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
    save_workspace_config,
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
import importlib
import importlib.util
import sys
import yaml

def _compat_attr(name: str):
    compat_module = sys.modules.get("agents.webapi_gateway")
    if compat_module is not None and name in vars(compat_module):
        return getattr(compat_module, name)
    return globals()[name]

_MARKETPLACE_PLUGIN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_INSTALL_LOCK_RELATIVE_PATH = Path(".semantier") / "install-lock.json"


def _repo_local_marketplace_root() -> Path:
    return Path(__file__).resolve().parents[3] / "semantier-skills"


def _read_yaml_file(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _directory_content_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts:
            continue
        try:
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
        except OSError:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _read_install_lock(target_home: Path) -> dict[str, Any]:
    lock_path = target_home / _INSTALL_LOCK_RELATIVE_PATH
    try:
        loaded = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        loaded = {}
    lock = loaded if isinstance(loaded, dict) else {}
    lock.setdefault("version", 1)
    for section in ("plugins", "skills", "tools"):
        if not isinstance(lock.get(section), dict):
            lock[section] = {}
    return lock


def _write_install_lock(target_home: Path, lock: dict[str, Any]) -> None:
    lock_path = target_home / _INSTALL_LOCK_RELATIVE_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _record_repo_local_plugin_install(
    *,
    target_home: Path,
    plugin_name: str,
    source_dir: Path,
    identifier: str,
    package_path: str,
) -> None:
    manifest = _read_yaml_file(source_dir / "plugin.yaml")
    now = datetime.now(timezone.utc).isoformat()
    lock = _read_install_lock(target_home)
    entry = {
        "type": "plugin",
        "identifier": identifier,
        "source": "repo-local-marketplace",
        "path": package_path.strip().strip("/") or f"plugins/{plugin_name}",
        "version": str(manifest.get("version") or "").strip() or None,
        "content_hash": _directory_content_hash(source_dir),
        "updated_at": now,
    }
    lock["plugins"][plugin_name] = {
        key: value for key, value in entry.items() if value is not None
    }
    _write_install_lock(target_home, lock)


def _remove_repo_local_plugin_install_record(*, target_home: Path, plugin_name: str) -> None:
    lock = _read_install_lock(target_home)
    lock.get("plugins", {}).pop(plugin_name, None)
    _write_install_lock(target_home, lock)


def _infer_marketplace_package_type(identifier: str, package_type: str) -> str:
    normalized = package_type.strip().lower()
    if normalized in {"plugin", "skill"}:
        return normalized
    if "/plugins/" in identifier:
        return "plugin"
    return "skill"


def _resolve_repo_local_plugin_source(identifier: str, package_path: str) -> tuple[Path, str]:
    normalized_path = package_path.strip().strip("/")
    if not normalized_path and "/plugins/" in identifier:
        normalized_path = identifier.split("/", 2)[-1]
    plugin_name = _plugin_name_from_marketplace_path(normalized_path)

    marketplace_root = _repo_local_marketplace_root().resolve()
    source_dir = (marketplace_root / normalized_path).resolve()
    try:
        source_dir.relative_to(marketplace_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="plugin marketplace path escapes repository root") from exc

    if not source_dir.exists() or not source_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"plugin source not found: {normalized_path}")
    if not (source_dir / "plugin.yaml").exists():
        raise HTTPException(status_code=400, detail=f"plugin manifest missing at {normalized_path}")

    return source_dir, plugin_name


def _plugin_name_from_marketplace_path(package_path: str) -> str:
    normalized_path = package_path.strip().strip("/")
    if not normalized_path:
        raise HTTPException(status_code=400, detail="plugin marketplace path required")
    if "\\" in normalized_path:
        raise HTTPException(status_code=400, detail="invalid plugin marketplace path")
    if ".." in normalized_path.split("/"):
        raise HTTPException(status_code=400, detail="invalid plugin marketplace path")
    if not normalized_path.startswith("plugins/"):
        raise HTTPException(status_code=400, detail="plugin marketplace path must start with 'plugins/'")

    plugin_name = normalized_path.rsplit("/", 1)[-1].strip()
    if not plugin_name or not _MARKETPLACE_PLUGIN_ID_RE.fullmatch(plugin_name):
        raise HTTPException(status_code=400, detail="invalid plugin marketplace path")
    return plugin_name


def _install_repo_local_plugin(
    *,
    target_home: Path,
    source_dir: Path,
    plugin_name: str,
    force: bool,
) -> None:
    _ensure_repo_hermes_agent_on_path()
    from agents.launcher import _workspace_bootstrap_lock
    from agents.semantier_agent import _hermes_home_bound
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled, _toggle_plugin_toolset

    with _workspace_bootstrap_lock(target_home):
        target_plugins_dir = target_home / "plugins"
        target_plugins_dir.mkdir(parents=True, exist_ok=True)
        target_dir = target_plugins_dir / plugin_name

        if target_dir.exists():
            if force:
                shutil.rmtree(target_dir)

        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

        with _hermes_home_bound(target_home):
            # The gateway may have discovered plugins before this package existed.
            # Refresh registration so enablement can expose its actual toolset.
            from hermes_cli.plugins import discover_plugins
            discover_plugins(force=True)
            result = dashboard_set_agent_plugin_enabled(plugin_name, enabled=True)
            if not result.get("ok"):
                raise HTTPException(
                    status_code=500,
                    detail=str(result.get("error") or f"failed to enable plugin '{plugin_name}'"),
                )
            # Enablement can admit a package that discovery initially skipped.
            # Register its tools before resolving the actual platform toolset.
            discover_plugins(force=True)
            _toggle_plugin_toolset(plugin_name, enable=True)
            _sync_plugin_skill_external_dir(
                target_home=target_home,
                plugin_dir=target_dir,
                enabled=True,
            )
            _remove_materialized_plugin_skill(target_home=target_home, plugin_dir=target_dir)


def _uninstall_repo_local_plugin(
    *,
    target_home: Path,
    plugin_name: str,
) -> None:
    _ensure_repo_hermes_agent_on_path()
    from agents.launcher import _workspace_bootstrap_lock
    from agents.semantier_agent import _hermes_home_bound
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    with _workspace_bootstrap_lock(target_home):
        target_dir = target_home / "plugins" / plugin_name
        if not target_dir.exists() or not target_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"plugin '{plugin_name}' is not installed")

        with _hermes_home_bound(target_home):
            _remove_materialized_plugin_skill(target_home=target_home, plugin_dir=target_dir)
            _sync_plugin_skill_external_dir(
                target_home=target_home,
                plugin_dir=target_dir,
                enabled=False,
            )
            result = dashboard_set_agent_plugin_enabled(plugin_name, enabled=False)
            if not result.get("ok"):
                raise HTTPException(
                    status_code=500,
                    detail=str(result.get("error") or f"failed to disable plugin '{plugin_name}'"),
                )
        shutil.rmtree(target_dir)


def install_repo_local_plugin_transaction(
    *, target_home: Path, source_dir: Path, plugin_name: str, force: bool,
    workspace_id: str | None, identifier: str, package_path: str,
) -> None:
    from agents.launcher import _workspace_bootstrap_lock

    with _workspace_bootstrap_lock(target_home):
        _install_repo_local_plugin(
            target_home=target_home, source_dir=source_dir,
            plugin_name=plugin_name, force=force,
        )
        _record_repo_local_plugin_install(
            target_home=target_home, plugin_name=plugin_name, source_dir=source_dir,
            identifier=identifier, package_path=package_path,
        )
        persist_runtime_config_to_workspace_db(
            workspace_id=workspace_id, target_home=target_home,
        )


def uninstall_repo_local_plugin_transaction(
    *, target_home: Path, plugin_name: str, workspace_id: str | None,
) -> None:
    from agents.launcher import _workspace_bootstrap_lock

    with _workspace_bootstrap_lock(target_home):
        _uninstall_repo_local_plugin(target_home=target_home, plugin_name=plugin_name)
        _remove_repo_local_plugin_install_record(
            target_home=target_home, plugin_name=plugin_name,
        )
        persist_runtime_config_to_workspace_db(
            workspace_id=workspace_id, target_home=target_home,
        )


def persist_runtime_config_to_workspace_db(*, workspace_id: str | None, target_home: Path) -> None:
    normalized_workspace_id = str(workspace_id or "").strip()
    if not normalized_workspace_id:
        return
    config_path = target_home / "config.yaml"
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        loaded = {}
    config = loaded if isinstance(loaded, dict) else {}
    save_workspace_config(normalized_workspace_id, config)


def _plugin_skill_metadata(plugin_dir: Path) -> tuple[str, Path] | None:
    skill_path = plugin_dir / "SKILL.md"
    if not skill_path.exists():
        return None
    try:
        raw = skill_path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta: dict[str, Any] = {}
    if raw.startswith("---\n"):
        end = raw.find("\n---", 4)
        if end != -1:
            try:
                parsed = yaml.safe_load(raw[4:end]) or {}
            except yaml.YAMLError:
                parsed = {}
            if isinstance(parsed, dict):
                meta = parsed
    skill_name = str(meta.get("name") or "").strip() or skill_path.parent.name
    if not skill_name:
        return None
    return skill_name, skill_path


def _materialized_plugin_skill_dir(target_home: Path, *, skill_name: str) -> Path:
    return target_home / "skills" / "productivity" / skill_name


def _sync_plugin_skill_external_dir(*, target_home: Path, plugin_dir: Path, enabled: bool) -> None:
    config_path = target_home / "config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        config = {}
    if not isinstance(config, dict):
        config = {}

    skills_cfg = config.get("skills")
    if not isinstance(skills_cfg, dict):
        skills_cfg = {}
        config["skills"] = skills_cfg

    external_dirs = skills_cfg.get("external_dirs")
    if isinstance(external_dirs, str):
        external_dirs = [external_dirs]
    if not isinstance(external_dirs, list):
        external_dirs = []

    plugin_skill_dir = str(plugin_dir.resolve())
    normalized: list[str] = []
    for item in external_dirs:
        text = str(item or "").strip()
        if not text:
            continue
        try:
            resolved = str(Path(text).expanduser().resolve())
        except OSError:
            resolved = text
        if resolved == plugin_skill_dir and not enabled:
            continue
        if resolved not in normalized:
            normalized.append(resolved)

    if enabled and (plugin_dir / "SKILL.md").exists() and plugin_skill_dir not in normalized:
        normalized.append(plugin_skill_dir)

    skills_cfg["external_dirs"] = normalized
    target_home.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _remove_materialized_plugin_skill(*, target_home: Path, plugin_dir: Path) -> None:
    metadata = _plugin_skill_metadata(plugin_dir)
    if metadata is None:
        return
    skill_name, _skill_path = metadata
    target_dir = _materialized_plugin_skill_dir(target_home, skill_name=skill_name)
    if target_dir.exists():
        shutil.rmtree(target_dir)
