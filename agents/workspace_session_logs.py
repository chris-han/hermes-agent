"""Minimal workspace-session log helpers for non-Semantier checkouts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


def _sessions_dir(workspace_hermes_home: Path) -> Path:
    path = Path(workspace_hermes_home).expanduser().resolve() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_jsonl_path(workspace_hermes_home: Path, session_id: str) -> Path:
    return _sessions_dir(workspace_hermes_home) / f"{quote(str(session_id), safe='-_.')}.jsonl"


def _index_path(workspace_hermes_home: Path) -> Path:
    return _sessions_dir(workspace_hermes_home) / "sessions.json"


def _load_index(workspace_hermes_home: Path) -> dict[str, Any]:
    path = _index_path(workspace_hermes_home)
    if not path.exists():
        return {"sessions": {}, "aliases": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid workspace session index JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Workspace session index must be a JSON object: {path}")
    data.setdefault("sessions", {})
    data.setdefault("aliases", {})
    if not isinstance(data["sessions"], dict):
        raise ValueError(f"Workspace session index 'sessions' must be an object: {path}")
    if not isinstance(data["aliases"], dict):
        raise ValueError(f"Workspace session index 'aliases' must be an object: {path}")
    return data


def _save_index(workspace_hermes_home: Path, data: dict[str, Any]) -> None:
    _index_path(workspace_hermes_home).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _new_session_id(workspace_id: str) -> str:
    return f"{workspace_id}:session_{time.time_ns()}"


def _workspace_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    try:
        from runtime_paths import _WORKSPACES_ROOT  # type: ignore

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


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _session_row(
    workspace_hermes_home: Path,
    session_id: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    origin = {
        "platform": record.get("platform"),
        "chat_id": record.get("chat_id"),
        "thread_id": record.get("thread_id"),
        "user_id": record.get("origin_user_id"),
    }
    return {
        "canonical_session_id": str(session_id),
        "session_id": str(session_id),
        "session_key": str(record.get("platform_session_key") or record.get("alias") or session_id),
        "index_key": str(record.get("alias") or record.get("platform_session_key") or session_id),
        "platform": str(record.get("platform") or "").lower(),
        "chat_id": record.get("chat_id"),
        "thread_id": record.get("thread_id"),
        "origin_user_id": record.get("origin_user_id"),
        "workspace_id": record.get("workspace_id"),
        "workspace_hermes_home": Path(workspace_hermes_home).expanduser().resolve(),
        "updated_at": record.get("updated_at") or "",
        "title": record.get("title") or "",
        "display_name": record.get("title") or "",
        "origin": origin,
        "raw_entry": {
            "origin": origin,
            "chat_type": record.get("chat_type") or "dm",
            "title": record.get("title") or "",
        },
    }


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
    create_if_missing: bool = True,
) -> str:
    data = _load_index(workspace_hermes_home)
    aliases: dict[str, str] = data["aliases"]
    sessions: dict[str, dict[str, Any]] = data["sessions"]
    alias_keys = [x for x in (alias, platform_session_key) if isinstance(x, str) and x.strip()]
    existing_session_id = next((aliases.get(key) for key in alias_keys if aliases.get(key)), "")

    if preferred_session_id and (existing_session_id or str(preferred_session_id).strip() in sessions):
        session_id = str(preferred_session_id).strip()
        if session_id:
            record = sessions.setdefault(session_id, {})
            record.update(
                {
                    "workspace_id": workspace_id,
                    "alias": alias,
                    "platform_session_key": platform_session_key,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "origin_user_id": origin_user_id,
                    "source": source,
                    "platform": platform,
                    "updated_at": _utc_now_iso(),
                }
            )
            for key in alias_keys:
                aliases[key] = session_id
            _save_index(workspace_hermes_home, data)
            return session_id

    if existing_session_id:
        return existing_session_id

    if not create_if_missing:
        return ""

    session_id = _new_session_id(workspace_id)
    sessions[session_id] = {
        "workspace_id": workspace_id,
        "alias": alias,
        "platform_session_key": platform_session_key,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "origin_user_id": origin_user_id,
        "source": source,
        "platform": platform,
        "updated_at": _utc_now_iso(),
    }
    for key in alias_keys:
        aliases[key] = session_id
    _save_index(workspace_hermes_home, data)
    return session_id


def configure_agent_workspace_session_paths(agent: Any, workspace_hermes_home: Path, session_id: str) -> None:
    if hasattr(agent, "_session_db") and getattr(agent, "_session_db", None) is not None:
        register = getattr(agent._session_db, "register_workspace_home", None)
        if callable(register):
            register(session_id, Path(workspace_hermes_home))
    setattr(agent, "_workspace_hermes_home", Path(workspace_hermes_home))
    setattr(agent, "session_id", session_id)


def update_workspace_session_title(
    workspace_hermes_home: Path,
    session_id: str,
    title: str,
) -> bool:
    data = _load_index(workspace_hermes_home)
    record = data["sessions"].setdefault(str(session_id), {})
    record["title"] = str(title or "")
    record["updated_at"] = _utc_now_iso()
    _save_index(workspace_hermes_home, data)
    return True


def resolve_workspace_session_delivery_adapter_key(
    workspace_hermes_home: Path,
    session_id: str,
    platform: str | None = None,
) -> str | None:
    data = _load_index(workspace_hermes_home)
    record = data["sessions"].get(str(session_id)) or {}
    if platform:
        record_platform = str(record.get("platform") or "").strip().lower()
        if record_platform and record_platform != str(platform).strip().lower():
            return None
    return record.get("delivery_adapter_key")


def list_all_workspace_session_index_entries() -> Iterable[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in _workspace_root_candidates():
        if not root.exists():
            continue
        for workspace_hermes_home in root.glob("*/.hermes"):
            data = _load_index(workspace_hermes_home)
            sessions = data.get("sessions") or {}
            if not isinstance(sessions, dict):
                continue
            for session_id, record in sessions.items():
                if not isinstance(record, dict):
                    continue
                rows.append(_session_row(workspace_hermes_home, str(session_id), record))
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
    del workspaces_root
    matches: list[dict[str, Any]] = []
    for row in list_all_workspace_session_index_entries():
        if canonical_session_id and str(row.get("session_id") or "") != str(canonical_session_id):
            continue
        if alias and str(row.get("index_key") or "") != str(alias):
            continue
        if platform_session_key and str(row.get("session_key") or "") != str(platform_session_key):
            continue
        if platform and str(row.get("platform") or "").lower() != str(platform).lower():
            continue
        if chat_id is not None and str(row.get("chat_id") or "") != str(chat_id):
            continue
        if thread_id is not None and str(row.get("thread_id") or "") != str(thread_id):
            continue
        if origin_user_id is not None and str(row.get("origin_user_id") or "") != str(origin_user_id):
            continue
        matches.append(row)

    matches.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
    return matches
