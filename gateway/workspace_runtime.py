"""Shared workspace runtime binding for gateway agent execution."""

from __future__ import annotations

import contextlib
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator


def _session_segment_for_workspace_home(target_home: Path, session_id: str) -> str:
    segment = str(session_id or "").strip()
    prefix = f"{target_home.name}:"
    if segment.startswith(prefix):
        segment = segment[len(prefix) :]
    if not segment or segment in {".", ".."}:
        raise ValueError("session_id required")
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", segment):
        raise ValueError("session_id must be an ASCII-stable path segment")
    if segment != Path(segment).name or any(sep in segment for sep in ("/", "\\")):
        raise ValueError("session_id must be a safe single path segment")
    return segment


@contextlib.contextmanager
def _bind_explicit_workspace_session_env(
    target_home: Path,
    session_id: str | None = None,
) -> Iterator[None]:
    """Bind active-session IO roots when runtime_paths cannot own the home."""
    if not session_id:
        yield
        return
    segment = _session_segment_for_workspace_home(target_home, session_id)
    session_root = target_home / "sessions" / segment
    uploads_root = session_root / "uploads"
    runs_root = session_root / "runs"
    artifacts_root = session_root / "artifacts"
    for root in (uploads_root, runs_root, artifacts_root):
        root.mkdir(parents=True, exist_ok=True)

    updates = {
        "TERMINAL_CWD": str(target_home),
        "HERMES_WRITE_ALLOWED_ROOTS": ",".join(
            [
                str(runs_root.resolve()),
                str(uploads_root.resolve()),
                str(artifacts_root.resolve()),
            ]
        ),
        "SEMANTIER_WORKSPACE_RUNS_DIR": str(runs_root.resolve()),
        "SEMANTIER_WORKSPACE_ARTIFACTS_DIR": str(artifacts_root.resolve()),
        "SEMANTIER_DISABLE_PROVIDER_FALLBACK": "1",
    }
    previous = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _bind_workspace_env_cm(
    target_home: Path,
    session_id: str | None = None,
) -> Iterator[None]:
    try:
        from runtime_paths import bind_workspace_env, bind_workspace_session_env
    except Exception:
        with _bind_explicit_workspace_session_env(target_home, session_id=session_id):
            yield
        return
    try:
        cm = (
            bind_workspace_session_env(target_home, session_id)
            if session_id
            else bind_workspace_env(target_home)
        )
        with cm:
            yield
    except ValueError:
        with _bind_explicit_workspace_session_env(target_home, session_id=session_id):
            yield


@contextlib.contextmanager
def bound_workspace_hermes_home(
    raw_home: str | os.PathLike[str] | None,
    session_id: str | None = None,
) -> Iterator[Path | None]:
    """Bind HERMES_HOME for one trusted workspace-scoped gateway operation."""
    value = str(raw_home or "").strip()
    if not value:
        yield None
        return

    target_home = Path(value).expanduser().resolve()
    prev_home = os.environ.get("HERMES_HOME")
    prev_runs = os.environ.get("SEMANTIER_WORKSPACE_RUNS_DIR")
    prev_artifacts = os.environ.get("SEMANTIER_WORKSPACE_ARTIFACTS_DIR")
    gateway_run = sys.modules.get("gateway.run")

    shared_root_raw = os.environ.get("SEMANTIER_LOCAL_STATE_DIR")
    shared_runtime_root = (
        Path(shared_root_raw).expanduser().resolve()
        if shared_root_raw
        else target_home
    )
    if gateway_run is None and os.environ.get("SEMANTIER_LOCAL_STATE_DIR"):
        os.environ["HERMES_HOME"] = str(shared_runtime_root)
        try:
            gateway_run = importlib.import_module("gateway.run")
        finally:
            os.environ["HERMES_HOME"] = str(target_home)
    elif gateway_run is None:
        try:
            gateway_run = importlib.import_module("gateway.run")
        except Exception:
            gateway_run = None

    try:
        if gateway_run is not None and shared_runtime_root:
            gateway_run._hermes_home = shared_runtime_root
            gateway_run._env_path = shared_runtime_root / ".env"
            gateway_run._config_path = shared_runtime_root / "config.yaml"
        os.environ["HERMES_HOME"] = str(target_home)
        reload_env = getattr(
            gateway_run,
            "_reload_runtime_env_preserving_config_authority",
            None,
        )
        if callable(reload_env):
            reload_env()
        os.environ["HERMES_HOME"] = str(target_home)
        with _bind_workspace_env_cm(target_home, session_id=session_id):
            yield target_home
    finally:
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home
        if prev_runs is None:
            os.environ.pop("SEMANTIER_WORKSPACE_RUNS_DIR", None)
        else:
            os.environ["SEMANTIER_WORKSPACE_RUNS_DIR"] = prev_runs
        if prev_artifacts is None:
            os.environ.pop("SEMANTIER_WORKSPACE_ARTIFACTS_DIR", None)
        else:
            os.environ["SEMANTIER_WORKSPACE_ARTIFACTS_DIR"] = prev_artifacts


def merge_workspace_tool_config(base_config: dict[str, Any], workspace_config: dict[str, Any]) -> dict[str, Any]:
    """Merge workspace tool/plugin config into gateway config for tool resolution."""
    merged_config = dict(base_config)
    merged_platform_toolsets = dict(base_config.get("platform_toolsets") or {})
    for key, values in (workspace_config.get("platform_toolsets") or {}).items():
        if not isinstance(values, list):
            continue
        existing = merged_platform_toolsets.get(key)
        combined = list(existing) if isinstance(existing, list) else []
        for value in values:
            if value not in combined:
                combined.append(value)
        merged_platform_toolsets[key] = combined
    merged_config["platform_toolsets"] = merged_platform_toolsets

    workspace_mcp_servers = workspace_config.get("mcp_servers")
    if isinstance(workspace_mcp_servers, dict):
        merged_mcp_servers = dict(base_config.get("mcp_servers") or {})
        merged_mcp_servers.update(workspace_mcp_servers)
        merged_config["mcp_servers"] = merged_mcp_servers

    workspace_plugins = workspace_config.get("plugins")
    if isinstance(workspace_plugins, dict):
        merged_plugins = dict(base_config.get("plugins") or {})
        for key, value in workspace_plugins.items():
            if isinstance(value, list):
                existing = merged_plugins.get(key)
                combined = list(existing) if isinstance(existing, list) else []
                for item in value:
                    if item not in combined:
                        combined.append(item)
                merged_plugins[key] = combined
            else:
                merged_plugins[key] = value
        merged_config["plugins"] = merged_plugins

    return merged_config


def discover_workspace_plugins_and_config(
    base_config: dict[str, Any],
    workspace_home: str | os.PathLike[str] | None,
    *,
    session_id: str | None = None,
    merge: bool = True,
) -> dict[str, Any]:
    """Discover workspace plugins under a bound home and return tool config."""
    if not workspace_home:
        return base_config
    with bound_workspace_hermes_home(workspace_home, session_id=session_id):
        try:
            from hermes_cli.plugins import discover_plugins

            discover_plugins(force=True)
        except Exception:
            pass
        try:
            from hermes_cli.config import read_raw_config

            workspace_config = read_raw_config()
        except Exception:
            workspace_config = None
        if not isinstance(workspace_config, dict) or not workspace_config:
            return base_config
        if not merge:
            return workspace_config
        return merge_workspace_tool_config(base_config, workspace_config)

