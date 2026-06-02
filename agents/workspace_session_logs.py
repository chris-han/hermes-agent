"""Thin compatibility wrapper around canonical workspace session storage."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


def _load_upstream_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "src" / "agents" / "workspace_session_logs.py"
    qualified_name = "_semantier_upstream_workspace_session_logs"
    module = sys.modules.get(qualified_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(
            f"Missing upstream workspace_session_logs implementation at {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


_UPSTREAM = _load_upstream_module()

WorkspaceSessionRecord = _UPSTREAM.WorkspaceSessionRecord
WorkspaceSessionResolutionError = _UPSTREAM.WorkspaceSessionResolutionError
create_workspace_session_log = _UPSTREAM.create_workspace_session_log
delete_workspace_session_log = _UPSTREAM.delete_workspace_session_log
get_workspace_session_detail = _UPSTREAM.get_workspace_session_detail
get_workspace_session_log_payload = _UPSTREAM.get_workspace_session_log_payload
get_workspace_session_messages = _UPSTREAM.get_workspace_session_messages
list_workspace_sessions = _UPSTREAM.list_workspace_sessions
list_workspace_session_trajectory = _UPSTREAM.list_workspace_session_trajectory
resolve_workspace_session_id = _UPSTREAM.resolve_workspace_session_id
configure_agent_workspace_session_paths = _UPSTREAM.configure_agent_workspace_session_paths
_sessions_dir = _UPSTREAM._sessions_dir
_session_jsonl_path = _UPSTREAM._session_jsonl_path


def _index_path(workspace_hermes_home: Path) -> Path:
    return _UPSTREAM._session_index_path(workspace_hermes_home)


def _session_snapshot_path(workspace_hermes_home: Path, session_id: str) -> Path:
    return _UPSTREAM._session_log_path(workspace_hermes_home, session_id)


def _workspace_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    try:
        from runtime_paths import _WORKSPACES_ROOT  # type: ignore[attr-defined]

        candidates.append(Path(_WORKSPACES_ROOT))
    except ModuleNotFoundError:
        pass
    candidates.append(Path(__file__).resolve().parents[2] / "workspaces")

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _session_record_from_payload(
    workspace_hermes_home: Path,
    session_id: str,
    flat_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = _UPSTREAM.get_workspace_session_log_payload(workspace_hermes_home, session_id) or {}
    primary_entry = flat_index.get(session_id) if isinstance(flat_index.get(session_id), dict) else {}
    aliases = [
        key
        for key, entry in flat_index.items()
        if key != session_id and isinstance(entry, dict) and str(entry.get("session_id") or "") == session_id
    ]
    session_key = str(payload.get("session_key") or primary_entry.get("session_key") or session_id)
    alias = aliases[0] if aliases else session_key
    updated_at = str(payload.get("last_updated") or primary_entry.get("updated_at") or "")
    workspace_id = str(payload.get("workspace_id") or "").strip()
    if not workspace_id and ":" in session_id:
        workspace_id = session_id.split(":", 1)[0]
    source = payload.get("source")
    if source == "api_server":
        source = None
    return {
        "workspace_id": workspace_id or None,
        "alias": alias,
        "platform_session_key": session_key,
        "chat_id": payload.get("chat_id"),
        "thread_id": payload.get("thread_id"),
        "origin_user_id": payload.get("origin_user_id"),
        "source": source,
        "platform": payload.get("platform"),
        "title": payload.get("title") or "",
        "adapter_key": payload.get("adapter_key"),
        "delivery_adapter_key": payload.get("delivery_adapter_key"),
        "updated_at": updated_at,
    }


def _load_index(workspace_hermes_home: Path) -> dict[str, Any]:
    path = _index_path(workspace_hermes_home)
    if not path.exists():
        return {"sessions": {}, "aliases": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid workspace session index JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Workspace session index must be a JSON object: {path}")
    if "sessions" in raw or "aliases" in raw:
        if not isinstance(raw.get("sessions", {}), dict):
            raise ValueError(f"Workspace session index 'sessions' must be an object: {path}")
        if not isinstance(raw.get("aliases", {}), dict):
            raise ValueError(f"Workspace session index 'aliases' must be an object: {path}")

    flat_index = _UPSTREAM._load_session_index(workspace_hermes_home)
    sessions: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for key, entry in flat_index.items():
        if not isinstance(entry, dict):
            continue
        session_id = str(entry.get("session_id") or key).strip()
        if not session_id:
            continue
        sessions.setdefault(
            session_id,
            _session_record_from_payload(workspace_hermes_home, session_id, flat_index),
        )
        if key != session_id:
            aliases[str(key)] = session_id
    return {"sessions": sessions, "aliases": aliases}


def resolve_or_create_workspace_session_id(
    workspace_hermes_home: Path,
    *,
    workspace_id: str,
    alias: str | None = None,
    preferred_session_id: str | None = None,
    platform_session_key: str | None = None,
    chat_id: str | None = None,
    thread_id: str | None = None,
    origin_user_id: str | None = None,
    source: str | None = None,
    platform: str | None = None,
    adapter_key: str | None = None,
    delivery_adapter_key: str | None = None,
    create_if_missing: bool = True,
) -> str:
    return _UPSTREAM.resolve_or_create_workspace_session_id(
        workspace_hermes_home,
        workspace_id=workspace_id,
        alias=alias,
        preferred_session_id=preferred_session_id,
        platform_session_key=platform_session_key,
        chat_id=chat_id,
        thread_id=thread_id,
        origin_user_id=origin_user_id,
        source=source or "api_server",
        platform=platform or "webchat",
        adapter_key=adapter_key,
        delivery_adapter_key=delivery_adapter_key,
        create_if_missing=create_if_missing,
    )


def update_workspace_session_title(
    workspace_hermes_home: Path,
    session_id: str,
    title: str,
) -> bool:
    return _UPSTREAM.update_workspace_session_title(
        workspace_hermes_home,
        session_id,
        title,
    ) is not None


def resolve_workspace_session_delivery_adapter_key(
    workspace_hermes_home: Path,
    session_id: str,
    platform: str | None = None,
) -> str | None:
    return _UPSTREAM.resolve_workspace_session_delivery_adapter_key(
        workspace_hermes_home,
        session_id,
        platform=platform,
    )


def list_all_workspace_session_index_entries(
    workspaces_root: Path | None = None,
) -> Iterable[dict[str, Any]]:
    if workspaces_root is not None:
        roots = [workspaces_root]
    else:
        roots = _workspace_root_candidates()

    rows: list[dict[str, Any]] = []
    for root in roots:
        for row in _UPSTREAM.list_all_workspace_session_index_entries(workspaces_root=root):
            rows.append(
                {
                    "workspace_id": row.get("workspace_id"),
                    "workspace_hermes_home": row.get("workspace_hermes_home"),
                    "index_key": row.get("index_key"),
                    "canonical_session_id": row.get("session_id"),
                    "session_id": row.get("session_id"),
                    "session_key": row.get("session_key"),
                    "platform": row.get("platform"),
                    "chat_id": row.get("chat_id"),
                    "thread_id": row.get("thread_id"),
                    "origin_user_id": row.get("origin_user_id"),
                    "adapter_key": row.get("adapter_key"),
                    "delivery_adapter_key": row.get("delivery_adapter_key"),
                    "updated_at": row.get("updated_at"),
                    "title": row.get("title"),
                    "display_name": row.get("display_name"),
                    "raw_entry": row.get("raw_entry"),
                    "origin": row.get("origin"),
                }
            )
    rows.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
    return rows


def find_workspace_session_index_matches(
    *,
    canonical_session_id: str | None = None,
    alias: str | None = None,
    platform: str | None = None,
    chat_id: str | None = None,
    thread_id: str | None = None,
    origin_user_id: str | None = None,
    platform_session_key: str | None = None,
    workspaces_root: Path | None = None,
) -> list[dict[str, Any]]:
    if workspaces_root is not None:
        roots = [workspaces_root]
    else:
        roots = _workspace_root_candidates()

    matches: list[dict[str, Any]] = []
    for root in roots:
        matches.extend(
            _UPSTREAM.find_workspace_session_index_matches(
                canonical_session_id=canonical_session_id,
                platform=platform,
                chat_id=chat_id,
                thread_id=thread_id,
                origin_user_id=origin_user_id,
                platform_session_key=platform_session_key or alias,
                workspaces_root=root,
            )
        )

    compat_rows = [
        {
            "workspace_id": row.get("workspace_id"),
            "workspace_hermes_home": row.get("workspace_hermes_home"),
            "index_key": row.get("index_key"),
            "canonical_session_id": row.get("session_id"),
            "session_id": row.get("session_id"),
            "session_key": row.get("session_key"),
            "platform": row.get("platform"),
            "chat_id": row.get("chat_id"),
            "thread_id": row.get("thread_id"),
            "origin_user_id": row.get("origin_user_id"),
            "adapter_key": row.get("adapter_key"),
            "delivery_adapter_key": row.get("delivery_adapter_key"),
            "updated_at": row.get("updated_at"),
            "title": row.get("title"),
            "display_name": row.get("display_name"),
            "raw_entry": row.get("raw_entry"),
            "origin": row.get("origin"),
        }
        for row in matches
    ]
    compat_rows.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
    return compat_rows
