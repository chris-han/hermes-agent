from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import unquote

from agents.workspace_uploads import WorkspaceUploadError, sanitize_upload_session_id


_DIR_MOVES = {
    ".hermes/sessions": "sessions",
    ".hermes/memories": "memories",
    ".hermes/skills": "skills",
    ".hermes/profiles": "profiles",
    ".hermes/cron": "cron",
    ".hermes/logs": "logs",
    ".hermes/home": "home",
    ".swarm/runs": "swarm/runs",
}


def _move_tree(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        for child in source.iterdir():
            destination = target / child.name
            if destination.exists():
                continue
            shutil.move(str(child), str(destination))
        return True
    shutil.move(str(source), str(target))
    return True


def _safe_existing_upload_filename(value: object) -> str | None:
    filename = Path(str(value or "").strip()).name
    if not filename or filename in {".", ".."}:
        return None
    if "/" in filename or "\\" in filename:
        return None
    return filename


def _safe_manifest_session_id(value: object) -> str | None:
    raw = str(value or "").strip()
    if "/" in raw or "\\" in raw:
        return None
    try:
        return sanitize_upload_session_id(raw)
    except WorkspaceUploadError:
        return None


def _migrate_flat_uploads(workspace_root: Path) -> bool:
    uploads = workspace_root / "uploads"
    manifest = uploads / ".upload-session-map.json"
    if not uploads.exists() or not manifest.exists():
        return False
    mapping = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("upload session manifest must be an object")
    migrated = False
    for filename, session_id in mapping.items():
        session_id_safe = _safe_manifest_session_id(session_id)
        if session_id_safe is None:
            continue
        filename_safe = _safe_existing_upload_filename(filename)
        if filename_safe is None:
            continue
        source = uploads / filename_safe
        if not source.is_file():
            continue
        target_dir = workspace_root / "sessions" / session_id_safe / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if target.exists():
            continue
        shutil.move(str(source), str(target))
        migrated = True
    return migrated


def _migrate_session_sibling_uploads(workspace_root: Path) -> bool:
    migrated = False
    sessions_root = workspace_root / "sessions"
    children = workspace_root.iterdir() if workspace_root.exists() else ()
    for child in children:
        if not child.is_dir() or child.name in {
            "sessions",
            "memories",
            "skills",
            "profiles",
            "cron",
            "logs",
            "home",
            "swarm",
            "plugins",
        }:
            continue
        if not child.name.startswith("session_"):
            continue
        uploads = child / "uploads"
        if not uploads.exists():
            continue
        target = sessions_root / sanitize_upload_session_id(child.name) / "uploads"
        migrated = _move_tree(uploads, target) or migrated
        try:
            child.rmdir()
        except OSError:
            pass
    return migrated


def _migrate_prefixed_session_dirs(workspace_root: Path) -> bool:
    sessions_root = workspace_root / "sessions"
    if not sessions_root.exists():
        return False
    migrated = False
    prefix = f"{workspace_root.name}:"
    for child in sessions_root.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        session_id = sanitize_upload_session_id(child.name[len(prefix) :])
        if not session_id:
            continue
        migrated = _move_tree(child, sessions_root / session_id) or migrated
        try:
            child.rmdir()
        except OSError:
            pass
    return migrated


def _migrate_hermes_workspace_session_runs(workspace_root: Path) -> bool:
    legacy_runs_root = workspace_root / ".hermes-workspace" / "runs"
    if not legacy_runs_root.exists():
        return False
    migrated = False
    sessions_root = workspace_root / "sessions"
    for child in legacy_runs_root.iterdir():
        if not child.is_dir() or not child.name.startswith("session_"):
            continue
        target = sessions_root / sanitize_upload_session_id(child.name) / "runs"
        migrated = _move_tree(child, target) or migrated
    return migrated


def _session_id_from_log_filename(name: str) -> str | None:
    if name.startswith("session_") and name.endswith(".trajectory.jsonl"):
        return unquote(name.removeprefix("session_").removesuffix(".trajectory.jsonl"))
    if name.startswith("session_") and name.endswith(".json"):
        return unquote(name.removeprefix("session_").removesuffix(".json"))
    if name.endswith(".jsonl") and not name.endswith(".trajectory.jsonl"):
        return unquote(name.removesuffix(".jsonl"))
    return None


def _normalized_log_filename(source_name: str, local_session_id: str) -> str | None:
    if source_name.endswith(".trajectory.jsonl") or source_name.endswith(
        ".trajectory.jsonl.lock"
    ):
        return f"{local_session_id}.trajectory.jsonl"
    if source_name.endswith(".jsonl") or source_name.endswith(".jsonl.lock"):
        return f"{local_session_id}.jsonl"
    if source_name.endswith(".json") or source_name.endswith(".json.lock"):
        return f"{local_session_id}.json"
    return None


def _local_session_id(workspace_root: Path, session_id: str) -> str | None:
    raw = str(session_id or "").strip()
    prefix = f"{workspace_root.name}:"
    if raw.startswith(prefix):
        raw = raw[len(prefix) :]
    try:
        return sanitize_upload_session_id(raw)
    except WorkspaceUploadError:
        return None


def _migrate_top_level_session_logs(workspace_root: Path) -> bool:
    sessions_root = workspace_root / "sessions"
    if not sessions_root.exists():
        return False
    migrated = False
    for child in list(sessions_root.iterdir()):
        if not child.is_file():
            continue
        session_id = _session_id_from_log_filename(child.name)
        if session_id is None:
            continue
        local_session_id = _local_session_id(workspace_root, session_id)
        if local_session_id is None:
            continue
        target_dir = sessions_root / local_session_id / "logs"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_name = _normalized_log_filename(child.name, local_session_id)
        if target_name is None:
            continue
        target = target_dir / target_name
        if target.exists():
            continue
        shutil.move(str(child), str(target))
        migrated = True

        lock = child.with_suffix(child.suffix + ".lock")
        if lock.exists():
            target_lock = target.with_suffix(target.suffix + ".lock")
            if not target_lock.exists():
                shutil.move(str(lock), str(target_lock))
    return migrated


def migrate_workspace_layout(workspace_root: Path) -> dict[str, Any]:
    root = Path(workspace_root)
    migrated = False
    for source_rel, target_rel in _DIR_MOVES.items():
        migrated = _move_tree(root / source_rel, root / target_rel) or migrated
    migrated = _migrate_flat_uploads(root) or migrated
    migrated = _migrate_session_sibling_uploads(root) or migrated
    migrated = _migrate_prefixed_session_dirs(root) or migrated
    migrated = _migrate_hermes_workspace_session_runs(root) or migrated
    migrated = _migrate_top_level_session_logs(root) or migrated
    return {"status": "ok", "migrated": migrated, "workspace_root": str(root)}
