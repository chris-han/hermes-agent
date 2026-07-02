from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid
from urllib.parse import quote, unquote

try:
    import fcntl
except Exception:  # pragma: no cover - non-POSIX fallback
    fcntl = None


def _sessions_dir(hermes_home: Path) -> Path:
    path = hermes_home / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_index_path(hermes_home: Path) -> Path:
    return _sessions_dir(hermes_home) / "sessions.json"


def _session_index_lock_path(hermes_home: Path) -> Path:
    return _sessions_dir(hermes_home) / "sessions.json.lock"


def _workspace_id_from_hermes_home(hermes_home: Path) -> str | None:
    resolved = Path(hermes_home)
    if resolved.name == ".hermes":
        candidate = resolved.parent.name
    else:
        candidate = resolved.name
    return candidate.strip() or None


def _session_segment(hermes_home: Path, session_id: str) -> str:
    normalized = str(session_id or "").strip()
    if ":" in normalized:
        _prefix, tail = normalized.split(":", 1)
        if tail:
            normalized = tail
    return _session_file_key(normalized)


def _session_id_variants_for_legacy_lookup(hermes_home: Path, session_id: str) -> list[str]:
    variants = [session_id]
    workspace_id = _workspace_id_from_hermes_home(hermes_home)
    raw = str(session_id or "").strip()
    if workspace_id and raw and ":" not in raw:
        variants.append(f"{workspace_id}:{raw}")
    return list(dict.fromkeys(variants))


def _session_logs_dir(hermes_home: Path, session_id: str) -> Path:
    path = workspace_session_dir(hermes_home, session_id) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_dir_path(hermes_home: Path, session_id: str) -> Path:
    return _sessions_dir(hermes_home) / _session_segment(hermes_home, session_id)


def _session_logs_dir_path(hermes_home: Path, session_id: str) -> Path:
    return _session_dir_path(hermes_home, session_id) / "logs"


def _session_log_path(hermes_home: Path, session_id: str) -> Path:
    segment = _session_segment(hermes_home, session_id)
    filename = f"{segment}.json" if segment.startswith("session_") else f"session_{segment}.json"
    return _session_logs_dir(hermes_home, session_id) / filename


def _session_jsonl_path(hermes_home: Path, session_id: str) -> Path:
    return _session_logs_dir(hermes_home, session_id) / f"{_session_segment(hermes_home, session_id)}.jsonl"


def _session_trajectory_path(hermes_home: Path, session_id: str) -> Path:
    segment = _session_segment(hermes_home, session_id)
    filename = f"{segment}.trajectory.jsonl" if segment.startswith("session_") else f"session_{segment}.trajectory.jsonl"
    return _session_logs_dir(hermes_home, session_id) / filename


def _session_file_key(session_id: str) -> str:
    return quote(session_id, safe="-_.")


def workspace_session_dir(hermes_home: Path, session_id: str) -> Path:
    path = _session_dir_path(hermes_home, session_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def workspace_session_artifacts_dir(hermes_home: Path, session_id: str) -> Path:
    path = workspace_session_dir(hermes_home, session_id) / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _legacy_session_log_path(hermes_home: Path, session_id: str) -> Path:
    return _sessions_dir(hermes_home) / f"session_{session_id}.json"


def _legacy_encoded_session_log_path(hermes_home: Path, session_id: str) -> Path:
    return _sessions_dir(hermes_home) / f"session_{_session_file_key(session_id)}.json"


def _legacy_logs_session_log_path(hermes_home: Path, session_id: str, variant: str) -> Path:
    return _session_logs_dir(hermes_home, session_id) / f"session_{variant}.json"


def _legacy_session_jsonl_path(hermes_home: Path, session_id: str) -> Path:
    return _sessions_dir(hermes_home) / f"{session_id}.jsonl"


def _legacy_encoded_session_jsonl_path(hermes_home: Path, session_id: str) -> Path:
    return _sessions_dir(hermes_home) / f"{_session_file_key(session_id)}.jsonl"


def _legacy_logs_session_jsonl_path(hermes_home: Path, session_id: str, variant: str) -> Path:
    return _session_logs_dir(hermes_home, session_id) / f"{variant}.jsonl"


def _legacy_session_trajectory_path(hermes_home: Path, session_id: str) -> Path:
    return _sessions_dir(hermes_home) / f"session_{session_id}.trajectory.jsonl"


def _legacy_encoded_session_trajectory_path(hermes_home: Path, session_id: str) -> Path:
    return _sessions_dir(hermes_home) / f"session_{_session_file_key(session_id)}.trajectory.jsonl"


def _legacy_logs_session_trajectory_path(hermes_home: Path, session_id: str, variant: str) -> Path:
    return _session_logs_dir(hermes_home, session_id) / f"session_{variant}.trajectory.jsonl"


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def _existing_session_log_path(hermes_home: Path, session_id: str) -> Path:
    path = _session_log_path(hermes_home, session_id)
    if not path.exists():
        # For session IDs whose segment starts with "session_", the old code
        # always prepended "session_" producing a double-prefix filename, e.g.
        # session_session_03ccf4aae573.json. Check that as a legacy fallback.
        segment = _session_segment(hermes_home, session_id)
        if segment.startswith("session_"):
            old_double = _session_logs_dir_path(hermes_home, session_id) / f"session_{segment}.json"
            if old_double.exists():
                return old_double
        for variant in _session_id_variants_for_legacy_lookup(hermes_home, session_id):
            for legacy in (
                _legacy_logs_session_log_path(hermes_home, session_id, variant),
                _legacy_logs_session_log_path(hermes_home, session_id, _session_file_key(variant)),
                _legacy_encoded_session_log_path(hermes_home, variant),
                _legacy_session_log_path(hermes_home, variant),
            ):
                if legacy.exists():
                    return legacy
    return path


def _payload_message_count(payload: dict[str, Any] | None) -> int:
    if not isinstance(payload, dict):
        return 0
    raw_count = payload.get("message_count")
    if isinstance(raw_count, int):
        return raw_count
    messages = payload.get("messages")
    return len(messages) if isinstance(messages, list) else 0


def _merged_workspace_session_log_payload(
    session_id: str,
    canonical_payload: dict[str, Any] | None,
    legacy_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    canonical_payload = canonical_payload if isinstance(canonical_payload, dict) else {}
    legacy_payload = legacy_payload if isinstance(legacy_payload, dict) else {}

    canonical_messages = canonical_payload.get("messages")
    legacy_messages = legacy_payload.get("messages")
    if not isinstance(canonical_messages, list):
        canonical_messages = []
    if not isinstance(legacy_messages, list):
        legacy_messages = []
    canonical_message_storage = _normalize_text(canonical_payload.get("message_storage"))
    if canonical_message_storage == "sqlite":
        merged_messages = canonical_messages
    else:
        merged_messages = legacy_messages if len(legacy_messages) > len(canonical_messages) else canonical_messages

    canonical_updated = _iso_to_epoch_seconds(canonical_payload.get("last_updated")) or 0.0
    legacy_updated = _iso_to_epoch_seconds(legacy_payload.get("last_updated")) or 0.0
    merged_last_updated = (
        legacy_payload.get("last_updated")
        if legacy_updated > canonical_updated
        else canonical_payload.get("last_updated")
    ) or legacy_payload.get("last_updated") or canonical_payload.get("last_updated") or _now_iso()

    merged_session_start = (
        canonical_payload.get("session_start")
        or legacy_payload.get("session_start")
        or merged_last_updated
    )

    merged = {
        "session_id": session_id,
        "canonical_session_id": canonical_payload.get("canonical_session_id") or legacy_payload.get("canonical_session_id") or session_id,
        "session_key": canonical_payload.get("session_key") or legacy_payload.get("session_key") or session_id,
        "platform_session_key": canonical_payload.get("platform_session_key") or legacy_payload.get("platform_session_key") or canonical_payload.get("session_key") or legacy_payload.get("session_key") or session_id,
        "title": canonical_payload.get("title") or legacy_payload.get("title"),
        "display_name": (
            canonical_payload.get("display_name")
            or legacy_payload.get("display_name")
            or canonical_payload.get("title")
            or legacy_payload.get("title")
        ),
        "source": canonical_payload.get("source") or legacy_payload.get("source"),
        "platform": canonical_payload.get("platform") or legacy_payload.get("platform"),
        "chat_id": canonical_payload.get("chat_id") or legacy_payload.get("chat_id"),
        "thread_id": canonical_payload.get("thread_id") or legacy_payload.get("thread_id"),
        "origin_user_id": canonical_payload.get("origin_user_id") or legacy_payload.get("origin_user_id"),
        "workspace_id": canonical_payload.get("workspace_id") or legacy_payload.get("workspace_id"),
        "sandbox_key": canonical_payload.get("sandbox_key") or legacy_payload.get("sandbox_key"),
        "model": canonical_payload.get("model") or legacy_payload.get("model"),
        "adapter_key": canonical_payload.get("adapter_key") or legacy_payload.get("adapter_key"),
        "delivery_adapter_key": canonical_payload.get("delivery_adapter_key") or legacy_payload.get("delivery_adapter_key"),
        "workspace_owner_id": canonical_payload.get("workspace_owner_id") or legacy_payload.get("workspace_owner_id"),
        "base_url": canonical_payload.get("base_url") or legacy_payload.get("base_url"),
        "session_start": merged_session_start,
        "last_updated": merged_last_updated,
        "updated_at": canonical_payload.get("updated_at") or legacy_payload.get("updated_at") or merged_last_updated,
        "system_prompt": canonical_payload.get("system_prompt") or legacy_payload.get("system_prompt") or "",
        "tools": canonical_payload.get("tools") or legacy_payload.get("tools") or [],
        "message_storage": canonical_payload.get("message_storage") or legacy_payload.get("message_storage"),
        "messages": merged_messages,
    }
    merged["message_count"] = max(
        _payload_message_count(canonical_payload),
        _payload_message_count(legacy_payload),
        len(merged_messages),
    )
    return merged


def _ensure_canonical_workspace_session_log(hermes_home: Path, session_id: str) -> Path:
    canonical_path = _session_log_path(hermes_home, session_id)
    canonical_payload = _read_json(canonical_path, None)
    legacy_payload = None
    for variant in _session_id_variants_for_legacy_lookup(hermes_home, session_id):
        for legacy_path in (
            _legacy_logs_session_log_path(hermes_home, session_id, variant),
            _legacy_logs_session_log_path(hermes_home, session_id, _session_file_key(variant)),
            _legacy_encoded_session_log_path(hermes_home, variant),
            _legacy_session_log_path(hermes_home, variant),
        ):
            legacy_payload = _read_json(legacy_path, None)
            if isinstance(legacy_payload, dict):
                break
        if isinstance(legacy_payload, dict):
            break

    if not isinstance(canonical_payload, dict) and not isinstance(legacy_payload, dict):
        return canonical_path

    merged = _merged_workspace_session_log_payload(session_id, canonical_payload, legacy_payload)
    _atomic_write_json(canonical_path, merged)
    return canonical_path


def _existing_session_jsonl_path(hermes_home: Path, session_id: str) -> Path:
    path = _session_jsonl_path(hermes_home, session_id)
    if not path.exists():
        for variant in _session_id_variants_for_legacy_lookup(hermes_home, session_id):
            for legacy in (
                _legacy_logs_session_jsonl_path(hermes_home, session_id, variant),
                _legacy_logs_session_jsonl_path(hermes_home, session_id, _session_file_key(variant)),
                _legacy_encoded_session_jsonl_path(hermes_home, variant),
                _legacy_session_jsonl_path(hermes_home, variant),
            ):
                if legacy.exists():
                    return legacy
    return path


def _existing_session_trajectory_path(hermes_home: Path, session_id: str) -> Path:
    path = _session_trajectory_path(hermes_home, session_id)
    if not path.exists():
        # Legacy: old code always prepended "session_" producing double-prefix
        # for session IDs whose segment starts with "session_".
        segment = _session_segment(hermes_home, session_id)
        if segment.startswith("session_"):
            old_double = _session_logs_dir_path(hermes_home, session_id) / f"session_{segment}.trajectory.jsonl"
            if old_double.exists():
                return old_double
        for variant in _session_id_variants_for_legacy_lookup(hermes_home, session_id):
            for legacy in (
                _legacy_logs_session_trajectory_path(hermes_home, session_id, variant),
                _legacy_logs_session_trajectory_path(hermes_home, session_id, _session_file_key(variant)),
                _legacy_encoded_session_trajectory_path(hermes_home, variant),
                _legacy_session_trajectory_path(hermes_home, variant),
            ):
                if legacy.exists():
                    return legacy
    return path


class _AdvisoryFileLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a+", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self._handle

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _iso_to_epoch_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        normalized = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def _normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_ascii_session_alias(value: Any) -> str | None:
    normalized = _normalize_text(value)
    if normalized is None:
        return None
    try:
        normalized.encode("ascii")
    except UnicodeEncodeError:
        return None
    return normalized


def _is_webapi_platform(value: Any) -> bool:
    return (_normalize_text(value) or "").lower() in {"api_server", "webchat"}


def _should_preserve_channel_transport_metadata(
    payload: dict[str, Any],
    *,
    source: str | None,
    platform: str | None,
) -> bool:
    """Return True when an API/webchat refresh targets a channel-owned session."""

    existing_source = _normalize_text(payload.get("source"))
    existing_platform = _normalize_text(payload.get("platform"))
    existing_is_channel = any(
        value is not None and not _is_webapi_platform(value)
        for value in (existing_source, existing_platform)
    )
    incoming_is_webapi = _is_webapi_platform(source) or _is_webapi_platform(platform)
    return existing_is_channel and incoming_is_webapi


def _current_sandbox_key_value() -> str | None:
    try:
        from agents.sandbox_scope import current_sandbox_key

        return _normalize_text(current_sandbox_key())
    except Exception:
        return None


@dataclass(frozen=True)
class WorkspaceSessionRecord:
    session_id: str
    session_key: str | None
    title: str | None
    display_name: str | None
    source: str | None
    model: str | None
    platform: str | None
    chat_id: str | None
    thread_id: str | None
    origin_user_id: str | None
    workspace_id: str | None
    sandbox_key: str | None
    adapter_key: str | None
    delivery_adapter_key: str | None
    workspace_owner_id: str | None
    started_at: float
    last_active: float
    message_count: int
    log_path: Path | None
    jsonl_path: Path | None


@dataclass(frozen=True)
class WorkspaceSessionTransportMetadata:
    platform_session_key: str | None
    chat_id: str | None
    thread_id: str | None
    origin_user_id: str | None
    adapter_key: str | None
    delivery_adapter_key: str | None


def derive_workspace_session_transport_metadata(
    *,
    platform: str | None,
    session_id: str | None = None,
    session_key: str | None = None,
    source: Any | None = None,
    platform_session_key: str | None = None,
    chat_id: str | None = None,
    thread_id: str | None = None,
    origin_user_id: str | None = None,
    adapter_key: str | None = None,
    delivery_adapter_key: str | None = None,
) -> WorkspaceSessionTransportMetadata:
    """Normalize transport identifiers before session mapping/log writes."""

    normalized_platform = (_normalize_text(platform) or "").lower()
    normalized_session_id = _normalize_text(session_id)
    normalized_session_key = _normalize_text(session_key)
    resolved_platform_session_key = (
        _normalize_text(platform_session_key)
        or normalized_session_key
        or normalized_session_id
    )
    resolved_chat_id = _normalize_text(chat_id)
    resolved_thread_id = _normalize_text(thread_id)
    resolved_origin_user_id = _normalize_text(origin_user_id)
    resolved_adapter_key = _normalize_text(adapter_key)
    resolved_delivery_adapter_key = _normalize_text(delivery_adapter_key)

    if source is not None:
        resolved_platform_session_key = (
            _normalize_text(platform_session_key)
            or normalized_session_key
            or _normalize_text(getattr(source, "platform_session_key", None))
            or normalized_session_id
        )
        resolved_chat_id = resolved_chat_id or _normalize_text(getattr(source, "chat_id", None))
        resolved_thread_id = resolved_thread_id or _normalize_text(getattr(source, "thread_id", None))
        resolved_origin_user_id = resolved_origin_user_id or _normalize_text(
            getattr(source, "user_id", None)
        )
        resolved_adapter_key = resolved_adapter_key or _normalize_text(
            getattr(source, "adapter_key", None)
        )
        resolved_delivery_adapter_key = resolved_delivery_adapter_key or _normalize_text(
            getattr(source, "delivery_adapter_key", None)
        )

    session_key_is_session_id = (
        normalized_session_key is None
        or normalized_session_key == normalized_session_id
        or _looks_uuid_like_session_id(normalized_session_key)
        or (
            normalized_session_key is not None
            and ":session_" in normalized_session_key
        )
    )
    if (
        resolved_chat_id is None
        and normalized_platform in {"api_server", "webchat"}
        and session_key_is_session_id
    ):
        resolved_chat_id = normalized_session_id or resolved_platform_session_key

    return WorkspaceSessionTransportMetadata(
        platform_session_key=resolved_platform_session_key,
        chat_id=resolved_chat_id,
        thread_id=resolved_thread_id,
        origin_user_id=resolved_origin_user_id,
        adapter_key=resolved_adapter_key,
        delivery_adapter_key=resolved_delivery_adapter_key,
    )


class WorkspaceSessionResolutionError(RuntimeError):
    pass


_SESSION_UUIDISH_RE = re.compile(r"^session_[0-9a-f]{8,32}$", re.IGNORECASE)
_CANONICAL_SESSION_SUFFIX_RE = re.compile(r"^session_[0-9a-f]{12}$", re.IGNORECASE)
_SESSION_GATEWAY_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{6,32}$", re.IGNORECASE)


def _looks_uuid_like_session_id(value: str | None) -> bool:
    normalized = _normalize_text(value)
    if normalized is None:
        return False
    if _SESSION_UUIDISH_RE.fullmatch(normalized):
        return True
    try:
        uuid.UUID(normalized)
        return True
    except ValueError:
        return False


def _looks_gateway_generated_session_id(value: str | None) -> bool:
    normalized = _normalize_text(value)
    if normalized is None:
        return False
    return bool(_SESSION_GATEWAY_RE.fullmatch(normalized))


def _looks_canonical_workspace_session_suffix(value: str | None) -> bool:
    normalized = _normalize_text(value)
    if normalized is None:
        return False
    return bool(_CANONICAL_SESSION_SUFFIX_RE.fullmatch(normalized))


def _preferred_workspace_session_id(
    workspace_id: str,
    preferred_session_id: str | None,
) -> str | None:
    normalized = _normalize_text(preferred_session_id)
    if normalized is None:
        return None
    prefix = f"{workspace_id}:"
    if normalized.startswith(prefix):
        suffix = normalized[len(prefix) :]
        if _looks_canonical_workspace_session_suffix(suffix):
            return normalized
        return None
    if _looks_canonical_workspace_session_suffix(normalized):
        return f"{workspace_id}:{normalized}"
    return None


def _canonical_identity_alias_target(alias: str, session_id: str) -> bool:
    normalized_alias = _normalize_text(alias)
    normalized_session_id = _normalize_text(session_id)
    if normalized_alias is None or normalized_session_id is None:
        return False
    if normalized_alias == normalized_session_id:
        return True
    if ":" not in normalized_session_id:
        return False
    _, suffix = normalized_session_id.split(":", 1)
    return normalized_alias == suffix


def _load_session_index(hermes_home: Path) -> dict[str, dict[str, Any]]:
    raw = _read_json(_session_index_path(hermes_home), {})
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, dict):
            normalized[key] = value
    return normalized


def _store_session_index(hermes_home: Path, payload: dict[str, dict[str, Any]]) -> None:
    _atomic_write_json(_session_index_path(hermes_home), payload)


def _resolve_workspace_session_id_from_index(
    hermes_home: Path,
    index: dict[str, dict[str, Any]],
    requested: str,
) -> str | None:
    candidate = (requested or "").strip()
    if not candidate:
        return None

    direct_log = _existing_session_log_path(hermes_home, candidate)
    direct_jsonl = _existing_session_jsonl_path(hermes_home, candidate)
    if direct_log.exists():
        payload = _read_json(direct_log, None)
        if isinstance(payload, dict):
            direct_session_id = _normalize_text(payload.get("session_id"))
            if direct_session_id:
                return direct_session_id
        return candidate
    if direct_jsonl.exists():
        return candidate

    entry = index.get(candidate)
    if isinstance(entry, dict):
        resolved = _normalize_text(entry.get("session_id"))
        if resolved:
            return resolved

    matches = []
    for row in list_workspace_sessions(hermes_home):
        session_id = str(row.get("session_id") or "").strip()
        session_key = str(row.get("session_key") or "").strip()
        if session_id.startswith(candidate) or session_key.startswith(candidate):
            matches.append(session_id)
    if len(matches) == 1:
        return matches[0]
    return None


def _build_or_refresh_index_entry_for_session(
    session_id: str,
    payload: dict[str, Any],
    existing_entry: dict[str, Any] | None = None,
    *,
    session_key: str | None = None,
) -> dict[str, Any]:
    created_at = (
        _normalize_text((existing_entry or {}).get("created_at"))
        or _normalize_text(payload.get("session_start"))
        or _now_iso()
    )
    updated_at = (
        _normalize_text(payload.get("last_updated"))
        or _normalize_text((existing_entry or {}).get("updated_at"))
        or created_at
    )
    message_count = payload.get("message_count")
    if not isinstance(message_count, int):
        messages = payload.get("messages")
        message_count = len(messages) if isinstance(messages, list) else 0
    return _build_index_entry(
        session_id,
        session_key=session_key or _normalize_text(payload.get("session_key")) or session_id,
        title=_normalize_text(payload.get("title")),
        display_name=_normalize_text(payload.get("display_name")),
        platform=_normalize_text(payload.get("platform")),
        chat_id=_normalize_text(payload.get("chat_id")),
        thread_id=_normalize_text(payload.get("thread_id")),
        origin_user_id=_normalize_text(payload.get("origin_user_id")),
        workspace_id=_normalize_text(payload.get("workspace_id")),
        sandbox_key=_normalize_text(payload.get("sandbox_key")),
        model=_normalize_text(payload.get("model")),
        adapter_key=_normalize_text(payload.get("adapter_key")),
        delivery_adapter_key=_normalize_text(payload.get("delivery_adapter_key")),
        workspace_owner_id=_normalize_text(payload.get("workspace_owner_id")),
        created_at=created_at,
        updated_at=updated_at,
        message_count=message_count,
    )


def _create_workspace_session_log_unlocked(
    hermes_home: Path,
    index: dict[str, dict[str, Any]],
    *,
    session_id: str,
    title: str | None = None,
    source: str = "api_server",
    platform: str = "webchat",
    session_key: str | None = None,
    sandbox_key: str | None = None,
    chat_id: str | None = None,
    thread_id: str | None = None,
    origin_user_id: str | None = None,
    workspace_id: str | None = None,
    adapter_key: str | None = None,
    delivery_adapter_key: str | None = None,
    workspace_owner_id: str | None = None,
) -> dict[str, Any]:
    now = _now_iso()
    normalized_title = _normalize_text(title)
    normalized_session_key = _normalize_text(session_key) or session_id
    effective_sandbox_key = _normalize_text(sandbox_key) or _current_sandbox_key_value()
    payload = {
        "session_id": session_id,
        "canonical_session_id": session_id,
        "session_key": normalized_session_key,
        "platform_session_key": normalized_session_key,
        "title": normalized_title,
        "display_name": normalized_title,
        "source": source,
        "platform": platform,
        "chat_id": _normalize_text(chat_id),
        "thread_id": _normalize_text(thread_id),
        "origin_user_id": _normalize_text(origin_user_id),
        "workspace_id": _normalize_text(workspace_id),
        "sandbox_key": effective_sandbox_key,
        "model": None,
        "base_url": None,
        "adapter_key": _normalize_text(adapter_key),
        "delivery_adapter_key": _normalize_text(delivery_adapter_key),
        "workspace_owner_id": _normalize_text(workspace_owner_id),
        "session_start": now,
        "last_updated": now,
        "updated_at": now,
        "system_prompt": "",
        "tools": [],
        "message_count": 0,
        "messages": [],
    }
    _atomic_write_json(_session_log_path(hermes_home, session_id), payload)
    existing_entry = index.get(session_id, {})
    if not isinstance(existing_entry, dict):
        existing_entry = {}
    index[session_id] = _build_or_refresh_index_entry_for_session(
        session_id,
        payload,
        existing_entry,
        session_key=normalized_session_key,
    )
    return payload


def _bind_workspace_session_aliases_unlocked(
    hermes_home: Path,
    index: dict[str, dict[str, Any]],
    *,
    aliases: list[str],
    session_id: str,
) -> str:
    resolved_session_id = _resolve_workspace_session_id_from_index(hermes_home, index, session_id) or _normalize_text(session_id)
    if resolved_session_id is None:
        raise ValueError("session_id is required")

    payload = _read_json(_existing_session_log_path(hermes_home, resolved_session_id), None)
    if not isinstance(payload, dict):
        payload = _create_workspace_session_log_unlocked(hermes_home, index, session_id=resolved_session_id)

    canonical_entry = index.get(resolved_session_id, {})
    if not isinstance(canonical_entry, dict):
        canonical_entry = {}
    index[resolved_session_id] = _build_or_refresh_index_entry_for_session(
        resolved_session_id,
        payload,
        canonical_entry,
    )

    for alias in aliases:
        normalized_alias = _normalize_text(alias)
        if normalized_alias is None or normalized_alias == resolved_session_id:
            continue
        existing_alias = index.get(normalized_alias)
        if isinstance(existing_alias, dict):
            existing_session_id = _normalize_text(existing_alias.get("session_id"))
            if (
                existing_session_id is not None
                and existing_session_id != resolved_session_id
                and _canonical_identity_alias_target(normalized_alias, existing_session_id)
            ):
                continue
        index[normalized_alias] = _build_or_refresh_index_entry_for_session(
            resolved_session_id,
            payload,
            existing_alias if isinstance(existing_alias, dict) else None,
            session_key=normalized_alias,
        )
    return resolved_session_id


def _build_index_entry(
    session_id: str,
    *,
    session_key: str | None,
    title: str | None,
    display_name: str | None,
    platform: str | None,
    chat_id: str | None,
    thread_id: str | None,
    origin_user_id: str | None,
    workspace_id: str | None,
    sandbox_key: str | None,
    model: str | None,
    adapter_key: str | None,
    delivery_adapter_key: str | None,
    workspace_owner_id: str | None,
    created_at: str,
    updated_at: str,
    message_count: int,
) -> dict[str, Any]:
    return {
        "session_key": session_key or session_id,
        "session_id": session_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "display_name": display_name,
        "title": title,
        "platform": platform,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "origin_user_id": origin_user_id,
        "workspace_id": workspace_id,
        "sandbox_key": sandbox_key,
        "model": model,
        "adapter_key": adapter_key,
        "delivery_adapter_key": delivery_adapter_key,
        "workspace_owner_id": workspace_owner_id,
        "message_count": message_count,
    }


def _refresh_workspace_session_payload_unlocked(
    hermes_home: Path,
    index: dict[str, dict[str, Any]],
    *,
    session_id: str,
    session_key: str | None = None,
    chat_id: str | None = None,
    thread_id: str | None = None,
    origin_user_id: str | None = None,
    workspace_id: str | None = None,
    source: str | None = None,
    platform: str | None = None,
    title: str | None = None,
    sandbox_key: str | None = None,
    adapter_key: str | None = None,
    delivery_adapter_key: str | None = None,
    workspace_owner_id: str | None = None,
) -> dict[str, Any]:
    payload = _read_json(_existing_session_log_path(hermes_home, session_id), None)
    if not isinstance(payload, dict):
        payload = _create_workspace_session_log_unlocked(
            hermes_home,
            index,
            session_id=session_id,
            title=title,
            source=source or "api_server",
            platform=platform or "webchat",
            session_key=session_key,
            chat_id=chat_id,
            thread_id=thread_id,
            origin_user_id=origin_user_id,
            workspace_id=workspace_id,
            sandbox_key=sandbox_key,
            adapter_key=adapter_key,
            delivery_adapter_key=delivery_adapter_key,
            workspace_owner_id=workspace_owner_id,
        )
    else:
        preserve_channel_transport = _should_preserve_channel_transport_metadata(
            payload,
            source=source,
            platform=platform,
        )
        if session_key is not None and not preserve_channel_transport:
            payload["session_key"] = _normalize_text(session_key) or payload.get("session_key") or session_id
            payload["platform_session_key"] = payload["session_key"]
        if chat_id is not None and not preserve_channel_transport:
            payload["chat_id"] = _normalize_text(chat_id)
        if thread_id is not None:
            payload["thread_id"] = _normalize_text(thread_id)
        if origin_user_id is not None and not preserve_channel_transport:
            payload["origin_user_id"] = _normalize_text(origin_user_id)
        if workspace_id is not None:
            payload["workspace_id"] = _normalize_text(workspace_id)
        if sandbox_key is not None:
            payload["sandbox_key"] = _normalize_text(sandbox_key)
        elif _normalize_text(payload.get("sandbox_key")) is None:
            payload["sandbox_key"] = _current_sandbox_key_value()
        if source is not None and not preserve_channel_transport:
            payload["source"] = source
        if platform is not None and not preserve_channel_transport:
            payload["platform"] = platform
        if title is not None:
            normalized_title = _normalize_text(title)
            payload["title"] = normalized_title
            payload["display_name"] = normalized_title
        if adapter_key is not None and not preserve_channel_transport:
            payload["adapter_key"] = _normalize_text(adapter_key)
        if delivery_adapter_key is not None and not preserve_channel_transport:
            payload["delivery_adapter_key"] = _normalize_text(delivery_adapter_key)
        if workspace_owner_id is not None and not preserve_channel_transport:
            payload["workspace_owner_id"] = _normalize_text(workspace_owner_id)
        payload["session_id"] = session_id
        payload["canonical_session_id"] = session_id
        payload["session_key"] = payload.get("session_key") or session_id
        payload["platform_session_key"] = payload.get("platform_session_key") or payload["session_key"]
        payload["last_updated"] = _now_iso()
        payload["updated_at"] = payload["last_updated"]
        payload.setdefault("session_start", payload["last_updated"])
        payload.setdefault("messages", [])
        payload["message_count"] = (
            payload.get("message_count")
            if isinstance(payload.get("message_count"), int)
            else len(payload["messages"])
        )
        _atomic_write_json(_session_log_path(hermes_home, session_id), payload)

    for candidate_key, value in list(index.items()):
        if not isinstance(value, dict):
            continue
        if _normalize_text(value.get("session_id")) != session_id:
            continue
        index[candidate_key] = _build_or_refresh_index_entry_for_session(
            session_id,
            payload,
            value,
            session_key=candidate_key,
        )
    return payload


def _extract_record(
    *,
    session_id: str,
    index_key: str | None,
    index_entry: dict[str, Any] | None,
    log_payload: dict[str, Any] | None,
    hermes_home: Path,
) -> WorkspaceSessionRecord:
    session_key = _normalize_text(index_key) or _normalize_text(
        (index_entry or {}).get("session_key")
    )
    log_title = _normalize_text((log_payload or {}).get("title"))
    log_display_name = _normalize_text((log_payload or {}).get("display_name"))
    index_title = _normalize_text((index_entry or {}).get("title"))
    index_display_name = _normalize_text((index_entry or {}).get("display_name"))
    title = log_title or index_title or log_display_name or index_display_name
    display_name = log_display_name or index_display_name or title
    platform = _normalize_text((log_payload or {}).get("platform")) or _normalize_text(
        (index_entry or {}).get("platform")
    )
    chat_id = _normalize_text((log_payload or {}).get("chat_id")) or _normalize_text(
        (index_entry or {}).get("chat_id")
    )
    thread_id = _normalize_text((log_payload or {}).get("thread_id")) or _normalize_text(
        (index_entry or {}).get("thread_id")
    )
    origin_user_id = _normalize_text((log_payload or {}).get("origin_user_id")) or _normalize_text(
        (index_entry or {}).get("origin_user_id")
    )
    workspace_id = _normalize_text((log_payload or {}).get("workspace_id")) or _normalize_text(
        (index_entry or {}).get("workspace_id")
    )
    sandbox_key = _normalize_text((log_payload or {}).get("sandbox_key")) or _normalize_text(
        (index_entry or {}).get("sandbox_key")
    )
    source = _normalize_text((log_payload or {}).get("source")) or platform
    model = _normalize_text((log_payload or {}).get("model")) or _normalize_text(
        (index_entry or {}).get("model")
    )
    adapter_key = _normalize_text((log_payload or {}).get("adapter_key")) or _normalize_text(
        (index_entry or {}).get("adapter_key")
    )
    delivery_adapter_key = _normalize_text((log_payload or {}).get("delivery_adapter_key")) or _normalize_text(
        (index_entry or {}).get("delivery_adapter_key")
    )
    workspace_owner_id = _normalize_text((log_payload or {}).get("workspace_owner_id")) or _normalize_text(
        (index_entry or {}).get("workspace_owner_id")
    )
    started_at = (
        _iso_to_epoch_seconds((log_payload or {}).get("session_start"))
        or _iso_to_epoch_seconds((index_entry or {}).get("created_at"))
        or 0.0
    )
    last_active = (
        _iso_to_epoch_seconds((log_payload or {}).get("last_updated"))
        or _iso_to_epoch_seconds((index_entry or {}).get("updated_at"))
        or started_at
    )
    raw_message_count = (log_payload or {}).get("message_count")
    if not isinstance(raw_message_count, int):
        messages = (log_payload or {}).get("messages")
        raw_message_count = len(messages) if isinstance(messages, list) else 0
    message_count = raw_message_count
    log_path = _existing_session_log_path(hermes_home, session_id)
    if not log_path.exists():
        log_path = None
    jsonl_path = _existing_session_jsonl_path(hermes_home, session_id)
    if not jsonl_path.exists():
        jsonl_path = None
    return WorkspaceSessionRecord(
        session_id=session_id,
        session_key=session_key,
        title=title,
        display_name=display_name,
        source=source,
        model=model,
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        origin_user_id=origin_user_id,
        workspace_id=workspace_id,
        sandbox_key=sandbox_key,
        adapter_key=adapter_key,
        delivery_adapter_key=delivery_adapter_key,
        workspace_owner_id=workspace_owner_id,
        started_at=started_at,
        last_active=last_active,
        message_count=message_count,
        log_path=log_path,
        jsonl_path=jsonl_path,
    )


def list_workspace_sessions(hermes_home: Path) -> list[dict[str, Any]]:
    sessions_dir = _sessions_dir(hermes_home)
    index = _load_session_index(hermes_home)
    session_ids: set[str] = set()
    key_by_session_id: dict[str, str] = {}
    for key, entry in index.items():
        session_id = _normalize_text(entry.get("session_id")) or _normalize_text(key)
        if session_id is None:
            continue
        session_ids.add(session_id)
        key_by_session_id.setdefault(session_id, key)

    for path in list(sessions_dir.glob("session_*.json")) + list(
        sessions_dir.glob("*/logs/session_*.json")
    ):
        payload = _read_json(path, None)
        session_id = _normalize_text((payload or {}).get("session_id"))
        if session_id is None:
            session_id = _normalize_text(unquote(path.stem.removeprefix("session_")))
        if session_id:
            session_ids.add(session_id)

    rows: list[dict[str, Any]] = []
    for session_id in session_ids:
        log_payload = _read_json(_existing_session_log_path(hermes_home, session_id), None)
        if log_payload is not None and not isinstance(log_payload, dict):
            log_payload = None
        record = _extract_record(
            session_id=session_id,
            index_key=key_by_session_id.get(session_id),
            index_entry=index.get(key_by_session_id.get(session_id, ""), {}),
            log_payload=log_payload,
            hermes_home=hermes_home,
        )
        if record.log_path is None and record.jsonl_path is None:
            continue
        rows.append(
            {
                "id": record.session_id,
                "session_id": record.session_id,
                "session_key": record.session_key or record.session_id,
                "title": record.title,
                "display_name": record.display_name,
                "source": record.source,
                "model": record.model,
                "platform": record.platform,
                "chat_id": record.chat_id,
                "thread_id": record.thread_id,
                "origin_user_id": record.origin_user_id,
                "workspace_id": record.workspace_id,
                "sandbox_key": record.sandbox_key,
                "adapter_key": record.adapter_key,
                "delivery_adapter_key": record.delivery_adapter_key,
                "workspace_owner_id": record.workspace_owner_id,
                "started_at": record.started_at,
                "last_active": record.last_active,
                "message_count": record.message_count,
                "is_active": False,
            }
        )
    rows.sort(key=lambda item: (-float(item.get("last_active") or 0), str(item.get("session_id") or "")))
    return rows


def iter_workspace_hermes_homes(workspaces_root: Path | None = None) -> list[Path]:
    """Return all direct workspace runtime homes currently present on disk."""
    root = workspaces_root or (Path(__file__).resolve().parents[2] / "workspaces")
    if not root.exists() or not root.is_dir():
        return []

    homes: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if (child / "sessions").is_dir() or (child / "config.yaml").exists():
            homes.append(child)
    homes.sort(key=lambda path: path.name)
    return homes


def list_all_workspace_session_index_entries(
    workspaces_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return flattened session-index entries across all workspace homes."""
    rows: list[dict[str, Any]] = []
    for hermes_home in iter_workspace_hermes_homes(workspaces_root):
        workspace_id = hermes_home.parent.name
        index = _load_session_index(hermes_home)
        for index_key, entry in index.items():
            if not isinstance(entry, dict):
                continue
            session_id = _normalize_text(entry.get("session_id")) or _normalize_text(index_key)
            if session_id is None:
                continue
            rows.append(
                {
                    "workspace_hermes_home": hermes_home,
                    "index_key": str(index_key),
                    "session_id": session_id,
                    "session_key": _normalize_text(entry.get("session_key")) or str(index_key),
                    "platform": _normalize_text(entry.get("platform")),
                    "chat_id": _normalize_text(entry.get("chat_id")),
                    "thread_id": _normalize_text(entry.get("thread_id")),
                    "origin_user_id": _normalize_text(entry.get("origin_user_id")),
                    "display_name": _normalize_text(entry.get("display_name")),
                    "title": _normalize_text(entry.get("title")),
                    "workspace_id": _normalize_text(entry.get("workspace_id")) or workspace_id,
                    "adapter_key": _normalize_text(entry.get("adapter_key")),
                    "delivery_adapter_key": _normalize_text(entry.get("delivery_adapter_key")),
                    "workspace_owner_id": _normalize_text(entry.get("workspace_owner_id")),
                    "updated_at": _normalize_text(entry.get("updated_at")),
                    "origin": entry.get("origin") if isinstance(entry.get("origin"), dict) else None,
                    "raw_entry": entry,
                }
            )

    rows.sort(
        key=lambda row: (
            -(_iso_to_epoch_seconds(row.get("updated_at")) or 0.0),
            str(row.get("session_key") or ""),
        )
    )
    return rows


def find_workspace_session_index_matches(
    *,
    platform: str | None = None,
    chat_id: str | None = None,
    thread_id: str | None = None,
    origin_user_id: str | None = None,
    canonical_session_id: str | None = None,
    platform_session_key: str | None = None,
    workspaces_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Find session index matches across workspaces.

    Match priority is canonical ``session_id`` first, then alias/session-key
    candidates derived from platform/chat/thread/user identifiers.
    """
    candidates = _candidate_session_aliases(
        preferred_session_id=canonical_session_id,
        alias=None,
        platform_session_key=platform_session_key,
        chat_id=chat_id,
        thread_id=thread_id,
        origin_user_id=origin_user_id,
    )
    candidate_set = set(candidates)
    normalized_platform = _normalize_text(platform)
    normalized_session_id = _normalize_text(canonical_session_id)

    best_matches: dict[str, tuple[int, float, dict[str, Any]]] = {}
    for row in list_all_workspace_session_index_entries(workspaces_root):
        row_platform = _normalize_text(row.get("platform"))
        if normalized_platform is not None and row_platform != normalized_platform:
            continue

        score = 0
        if normalized_session_id is not None and row.get("session_id") == normalized_session_id:
            score = 100
        else:
            if row.get("index_key") in candidate_set:
                score = max(score, 80)
            if row.get("session_key") in candidate_set:
                score = max(score, 70)
            if chat_id is not None and _normalize_text(row.get("chat_id")) == _normalize_text(chat_id):
                score = max(score, 60)
            if thread_id is not None and _normalize_text(row.get("thread_id")) == _normalize_text(thread_id):
                score = max(score, 50)
            if origin_user_id is not None and _normalize_text(row.get("origin_user_id")) == _normalize_text(origin_user_id):
                score = max(score, 40)

        if score <= 0:
            continue

        updated_at = _iso_to_epoch_seconds(row.get("updated_at")) or 0.0
        session_id = str(row.get("session_id") or "")
        existing = best_matches.get(session_id)
        candidate = (score, updated_at, row)
        if existing is None or candidate[0] > existing[0] or (
            candidate[0] == existing[0] and candidate[1] > existing[1]
        ):
            best_matches[session_id] = candidate

    matches = list(best_matches.values())
    matches.sort(key=lambda item: (-item[0], -item[1], str(item[2].get("session_key") or "")))
    return [item[2] for item in matches]


def resolve_workspace_session_id(hermes_home: Path, requested: str) -> str | None:
    index = _load_session_index(hermes_home)
    return _resolve_workspace_session_id_from_index(hermes_home, index, requested)


def get_workspace_session_detail(hermes_home: Path, requested: str) -> dict[str, Any] | None:
    session_id = resolve_workspace_session_id(hermes_home, requested)
    if session_id is None:
        return None
    for row in list_workspace_sessions(hermes_home):
        if row.get("session_id") == session_id:
            return row
    return None


def get_workspace_session_log_payload(hermes_home: Path, requested: str) -> dict[str, Any] | None:
    session_id = resolve_workspace_session_id(hermes_home, requested)
    if session_id is None:
        return None
    path = _ensure_canonical_workspace_session_log(hermes_home, session_id)
    payload = _read_json(path, None)
    if not isinstance(payload, dict):
        return None
    payload["session_id"] = session_id
    payload.setdefault("canonical_session_id", session_id)
    payload["session_key"] = payload.get("session_key") or session_id
    payload.setdefault("platform_session_key", payload["session_key"])
    payload["message_count"] = _payload_message_count(payload)
    payload.setdefault("messages", [])
    payload.setdefault("system_prompt", "")
    payload.setdefault("tools", [])
    payload.setdefault("chat_id", None)
    payload.setdefault("thread_id", None)
    payload.setdefault("origin_user_id", None)
    payload.setdefault("workspace_id", None)
    payload.setdefault("sandbox_key", None)
    payload.setdefault("adapter_key", None)
    payload.setdefault("delivery_adapter_key", None)
    payload.setdefault("workspace_owner_id", None)
    payload.setdefault("session_start", payload.get("last_updated") or _now_iso())
    payload.setdefault("last_updated", payload.get("session_start") or _now_iso())
    payload.setdefault("updated_at", payload.get("last_updated"))
    return payload


def resolve_workspace_session_delivery_adapter_key(
    hermes_home: Path,
    requested: str,
    *,
    platform: str | None = None,
    runtime_accounts: list[dict[str, Any]] | None = None,
) -> str | None:
    """Resolve the preferred delivery adapter key for a canonical session.

    Preference order:
    1. persisted ``delivery_adapter_key``
    2. persisted ``adapter_key``
    3. deterministic historical fallback using the unique active runtime account
       for the session's owning workspace

    Ambiguous historical sessions fail closed with ``None``.
    """
    payload = get_workspace_session_log_payload(hermes_home, requested)
    if not isinstance(payload, dict):
        return None

    persisted_delivery = _normalize_text(payload.get("delivery_adapter_key"))
    if persisted_delivery is not None:
        return persisted_delivery

    persisted_adapter = _normalize_text(payload.get("adapter_key"))
    if persisted_adapter is not None:
        return persisted_adapter

    normalized_platform = _normalize_text(platform) or _normalize_text(payload.get("platform"))
    if normalized_platform != "weixin":
        return None

    workspace_owner_id = _normalize_text(payload.get("workspace_owner_id"))
    if workspace_owner_id is None:
        return None

    if runtime_accounts is None:
        try:
            from agents.auth_db import load_weixin_runtime_accounts

            runtime_accounts = load_weixin_runtime_accounts()
        except Exception:
            runtime_accounts = []

    session_workspace_id = _normalize_text(payload.get("session_id"))
    if session_workspace_id and ":" in session_workspace_id:
        session_workspace_id = session_workspace_id.split(":", 1)[0]

    candidates: list[str] = []
    for account in runtime_accounts:
        if not isinstance(account, dict):
            continue
        account_id = _normalize_text(account.get("account_id"))
        owner_workspace_id = _normalize_text(account.get("owner_workspace_id") or account.get("workspace_id"))
        if account_id is None or owner_workspace_id is None:
            continue
        if session_workspace_id is not None and owner_workspace_id != session_workspace_id:
            continue
        candidates.append(f"weixin:{owner_workspace_id}:{account_id}")

    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    return None


def _normalize_jsonl_message(message: dict[str, Any], index: int) -> dict[str, Any]:
    timestamp_raw = message.get("timestamp")
    timestamp_epoch = _iso_to_epoch_seconds(timestamp_raw)
    created_at = timestamp_raw if isinstance(timestamp_raw, str) else None
    if created_at is None and timestamp_epoch is not None:
        created_at = datetime.fromtimestamp(timestamp_epoch, tz=timezone.utc).isoformat()
    normalized: dict[str, Any] = {
        "id": message.get("id") or index,
        "role": message.get("role") or "assistant",
        "content": message.get("content") or "",
    }
    if created_at:
        normalized["created_at"] = created_at
    if timestamp_epoch is not None:
        normalized["timestamp"] = timestamp_epoch
    if isinstance(message.get("tool_calls"), list):
        normalized["tool_calls"] = message["tool_calls"]
    if isinstance(message.get("tool_call_id"), str):
        normalized["tool_call_id"] = message["tool_call_id"]
    tool_name = _normalize_text(message.get("tool_name")) or _normalize_text(message.get("name"))
    if tool_name:
        normalized["tool_name"] = tool_name
    if isinstance(message.get("finish_reason"), str):
        normalized["finish_reason"] = message["finish_reason"]
    if isinstance(message.get("reasoning"), str):
        normalized["reasoning"] = message["reasoning"]
    if isinstance(message.get("reasoning_content"), str):
        normalized["reasoning_content"] = message["reasoning_content"]
    return normalized


def get_workspace_session_messages(hermes_home: Path, requested: str) -> tuple[str, list[dict[str, Any]]] | None:
    session_id = resolve_workspace_session_id(hermes_home, requested)
    if session_id is None:
        return None

    jsonl_path = _existing_session_jsonl_path(hermes_home, session_id)
    if jsonl_path.exists():
        messages: list[dict[str, Any]] = []
        try:
            for index, line in enumerate(_read_lines(jsonl_path), start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                if payload.get("role") == "session_meta":
                    continue
                messages.append(_normalize_jsonl_message(payload, index))
        except (OSError, json.JSONDecodeError):
            messages = []
        if messages:
            return session_id, messages

    log_payload = _read_json(_existing_session_log_path(hermes_home, session_id), None)
    if not isinstance(log_payload, dict):
        return session_id, []
    raw_messages = log_payload.get("messages")
    if not isinstance(raw_messages, list):
        return session_id, []

    # If the canonical log has no messages but a legacy log exists with messages,
    # prefer the legacy log. This can happen when the API creates an empty canonical
    # log file that shadows an existing Hermes-format session log.
    if not raw_messages:
        legacy_path = _legacy_session_log_path(hermes_home, session_id)
        if legacy_path.exists():
            legacy_payload = _read_json(legacy_path, None)
            if isinstance(legacy_payload, dict):
                legacy_messages = legacy_payload.get("messages")
                if isinstance(legacy_messages, list) and legacy_messages:
                    log_payload = legacy_payload
                    raw_messages = legacy_messages

    session_start = _iso_to_epoch_seconds(log_payload.get("session_start")) or 0.0
    fallback_messages: list[dict[str, Any]] = []
    for index, payload in enumerate(raw_messages, start=1):
        if not isinstance(payload, dict):
            continue
        timestamp_epoch = _iso_to_epoch_seconds(payload.get("timestamp")) or (session_start + index * 0.001)
        fallback_messages.append(
            {
                "id": payload.get("id") or index,
                "role": payload.get("role") or "assistant",
                "content": payload.get("content") or "",
                "created_at": datetime.fromtimestamp(timestamp_epoch, tz=timezone.utc).isoformat(),
                "timestamp": timestamp_epoch,
                **({"tool_calls": payload["tool_calls"]} if isinstance(payload.get("tool_calls"), list) else {}),
                **({"tool_call_id": payload["tool_call_id"]} if isinstance(payload.get("tool_call_id"), str) else {}),
                **({"tool_name": payload["tool_name"]} if isinstance(payload.get("tool_name"), str) else {}),
                **({"finish_reason": payload["finish_reason"]} if isinstance(payload.get("finish_reason"), str) else {}),
                **({"reasoning": payload["reasoning"]} if isinstance(payload.get("reasoning"), str) else {}),
                **({"reasoning_content": payload["reasoning_content"]} if isinstance(payload.get("reasoning_content"), str) else {}),
            }
        )
    return session_id, fallback_messages


def create_workspace_session_log(
    hermes_home: Path,
    *,
    session_id: str,
    title: str | None = None,
    source: str = "api_server",
    platform: str = "webchat",
    session_key: str | None = None,
    sandbox_key: str | None = None,
    chat_id: str | None = None,
    thread_id: str | None = None,
    origin_user_id: str | None = None,
    workspace_id: str | None = None,
    adapter_key: str | None = None,
    delivery_adapter_key: str | None = None,
    workspace_owner_id: str | None = None,
) -> dict[str, Any]:
    lock_path = _session_index_lock_path(hermes_home)
    with _AdvisoryFileLock(lock_path):
        index = _load_session_index(hermes_home)
        payload = _create_workspace_session_log_unlocked(
            hermes_home,
            index,
            session_id=session_id,
            title=title,
            source=source,
            platform=platform,
            session_key=session_key,
            sandbox_key=sandbox_key,
            chat_id=chat_id,
            thread_id=thread_id,
            origin_user_id=origin_user_id,
            workspace_id=workspace_id,
            adapter_key=adapter_key,
            delivery_adapter_key=delivery_adapter_key,
            workspace_owner_id=workspace_owner_id,
        )
        _store_session_index(hermes_home, index)
        return payload


def bind_workspace_session_alias(
    hermes_home: Path,
    *,
    alias: str,
    session_id: str,
) -> str:
    normalized_alias = _normalize_text(alias)
    if normalized_alias is None:
        raise ValueError("alias and session_id are required")
    lock_path = _session_index_lock_path(hermes_home)
    with _AdvisoryFileLock(lock_path):
        index = _load_session_index(hermes_home)
        resolved_session_id = _bind_workspace_session_aliases_unlocked(
            hermes_home,
            index,
            aliases=[normalized_alias],
            session_id=session_id,
        )
        _store_session_index(hermes_home, index)
        return resolved_session_id


def resolve_or_create_workspace_session_id(
    hermes_home: Path,
    *,
    workspace_id: str,
    alias: str | None = None,
    preferred_session_id: str | None = None,
    platform_session_key: str | None = None,
    chat_id: str | None = None,
    thread_id: str | None = None,
    origin_user_id: str | None = None,
    title: str | None = None,
    source: str = "api_server",
    platform: str = "webchat",
    adapter_key: str | None = None,
    delivery_adapter_key: str | None = None,
    workspace_owner_id: str | None = None,
    create_if_missing: bool = False,
) -> str:
    normalized_workspace_id = _normalize_text(workspace_id)
    if normalized_workspace_id is None:
        raise ValueError("workspace_id is required")
    normalized_workspace_owner_id = _normalize_text(workspace_owner_id) or normalized_workspace_id

    normalized_preferred_session_id = _normalize_text(preferred_session_id)
    canonical_preferred_session_id = (
        _preferred_workspace_session_id(
            normalized_workspace_id,
            normalized_preferred_session_id,
        )
        if normalized_preferred_session_id is not None
        else None
    )

    alias_candidates = _candidate_session_aliases(
        preferred_session_id=preferred_session_id,
        alias=alias,
        platform_session_key=platform_session_key,
        chat_id=chat_id,
        thread_id=thread_id,
        origin_user_id=origin_user_id,
    )
    primary_alias = next(
        (
            candidate
            for candidate in alias_candidates
            if candidate != _normalize_text(preferred_session_id)
        ),
        None,
    )
    bind_aliases = [candidate for candidate in alias_candidates if candidate != normalized_workspace_id]

    lock_path = _session_index_lock_path(hermes_home)
    with _AdvisoryFileLock(lock_path):
        index = _load_session_index(hermes_home)

        # When gateway already minted a fresh preferred session id (for
        # example after /reset), keep that id authoritative and rebind
        # aliases to it instead of snapping back to an older alias target.
        if create_if_missing and canonical_preferred_session_id is not None:
            preferred_resolved = _resolve_workspace_session_id_from_index(
                hermes_home,
                index,
                canonical_preferred_session_id,
            )
            if preferred_resolved is None:
                for candidate in alias_candidates:
                    normalized_candidate = _normalize_text(candidate)
                    if normalized_candidate is None:
                        continue
                    if not _canonical_identity_alias_target(
                        normalized_candidate,
                        canonical_preferred_session_id,
                    ):
                        continue
                    alias_resolved = _resolve_workspace_session_id_from_index(
                        hermes_home,
                        index,
                        normalized_candidate,
                    )
                    if alias_resolved:
                        preferred_resolved = alias_resolved
                        break
            if preferred_resolved is None:
                _create_workspace_session_log_unlocked(
                    hermes_home,
                    index,
                    session_id=canonical_preferred_session_id,
                    title=title,
                    source=source,
                    platform=platform,
                    session_key=primary_alias,
                    chat_id=chat_id,
                    thread_id=thread_id,
                    origin_user_id=origin_user_id,
                    workspace_id=normalized_workspace_id,
                    adapter_key=adapter_key,
                    delivery_adapter_key=delivery_adapter_key,
                    workspace_owner_id=normalized_workspace_owner_id,
                )
                preferred_resolved = canonical_preferred_session_id
            else:
                _refresh_workspace_session_payload_unlocked(
                    hermes_home,
                    index,
                    session_id=preferred_resolved,
                    session_key=primary_alias,
                    chat_id=chat_id,
                    thread_id=thread_id,
                    origin_user_id=origin_user_id,
                    workspace_id=normalized_workspace_id,
                    source=source,
                    platform=platform,
                    title=title,
                    adapter_key=adapter_key,
                    delivery_adapter_key=delivery_adapter_key,
                    workspace_owner_id=normalized_workspace_owner_id,
                )
            _bind_workspace_session_aliases_unlocked(
                hermes_home,
                index,
                aliases=bind_aliases,
                session_id=preferred_resolved,
            )
            _store_session_index(hermes_home, index)
            return preferred_resolved

        for candidate in alias_candidates:
            normalized_candidate = _normalize_text(candidate)
            if normalized_candidate is None:
                continue
            resolved = _resolve_workspace_session_id_from_index(hermes_home, index, normalized_candidate)
            if resolved:
                _refresh_workspace_session_payload_unlocked(
                    hermes_home,
                    index,
                    session_id=resolved,
                    session_key=primary_alias,
                    chat_id=chat_id,
                    thread_id=thread_id,
                    origin_user_id=origin_user_id,
                    workspace_id=normalized_workspace_id,
                    source=source,
                    platform=platform,
                    title=title,
                    adapter_key=adapter_key,
                    delivery_adapter_key=delivery_adapter_key,
                    workspace_owner_id=normalized_workspace_owner_id,
                )
                _bind_workspace_session_aliases_unlocked(
                    hermes_home,
                    index,
                    aliases=bind_aliases,
                    session_id=resolved,
                )
                _store_session_index(hermes_home, index)
                return resolved

        if not create_if_missing:
            identifiers = {
                "preferred_session_id": _normalize_text(preferred_session_id),
                "alias": _normalize_text(alias),
                "platform_session_key": _normalize_text(platform_session_key),
                "chat_id": _normalize_text(chat_id),
                "thread_id": _normalize_text(thread_id),
                "origin_user_id": _normalize_text(origin_user_id),
            }
            present = {key: value for key, value in identifiers.items() if value is not None}
            detail = ", ".join(f"{key}={value}" for key, value in sorted(present.items())) or "no identifiers"
            raise WorkspaceSessionResolutionError(
                f"workspace session mapping missing for workspace_id={normalized_workspace_id} ({detail})"
            )

        session_id = _preferred_workspace_session_id(
            normalized_workspace_id,
            preferred_session_id,
        ) or f"{normalized_workspace_id}:session_{uuid.uuid4().hex[:12]}"
        _create_workspace_session_log_unlocked(
            hermes_home,
            index,
            session_id=session_id,
            title=title,
            source=source,
            platform=platform,
            session_key=primary_alias,
            chat_id=chat_id,
            thread_id=thread_id,
            origin_user_id=origin_user_id,
            workspace_id=normalized_workspace_id,
            adapter_key=adapter_key,
            delivery_adapter_key=delivery_adapter_key,
            workspace_owner_id=normalized_workspace_owner_id,
        )
        _bind_workspace_session_aliases_unlocked(
            hermes_home,
            index,
            aliases=bind_aliases,
            session_id=session_id,
        )
        _store_session_index(hermes_home, index)
        return session_id


def update_workspace_session_title(hermes_home: Path, requested: str, title: str | None) -> dict[str, Any] | None:
    lock_path = _session_index_lock_path(hermes_home)
    with _AdvisoryFileLock(lock_path):
        index = _load_session_index(hermes_home)
        session_id = _resolve_workspace_session_id_from_index(hermes_home, index, requested)
        if session_id is None:
            return None
        payload = _read_json(_existing_session_log_path(hermes_home, session_id), {})
        if not isinstance(payload, dict):
            payload = {"session_id": session_id}
        normalized_title = _normalize_text(title)
        payload["session_id"] = session_id
        payload["session_key"] = payload.get("session_key") or session_id
        payload["title"] = normalized_title
        payload["display_name"] = normalized_title
        payload["last_updated"] = _now_iso()
        payload.setdefault("session_start", payload["last_updated"])
        payload.setdefault("messages", [])
        payload["message_count"] = (
            payload.get("message_count")
            if isinstance(payload.get("message_count"), int)
            else len(payload["messages"])
        )
        _atomic_write_json(_session_log_path(hermes_home, session_id), payload)

        for candidate_key, value in list(index.items()):
            if not isinstance(value, dict):
                continue
            if _normalize_text(value.get("session_id")) != session_id:
                continue
            index[candidate_key] = _build_or_refresh_index_entry_for_session(
                session_id,
                payload,
                value,
                session_key=candidate_key,
            )
        _store_session_index(hermes_home, index)
        return payload


def update_workspace_session_sandbox_key(
    hermes_home: Path,
    requested: str,
    sandbox_key: str | None,
) -> dict[str, Any] | None:
    lock_path = _session_index_lock_path(hermes_home)
    with _AdvisoryFileLock(lock_path):
        index = _load_session_index(hermes_home)
        session_id = _resolve_workspace_session_id_from_index(hermes_home, index, requested)
        if session_id is None:
            return None
        payload = _refresh_workspace_session_payload_unlocked(
            hermes_home,
            index,
            session_id=session_id,
            sandbox_key=sandbox_key,
        )
        _store_session_index(hermes_home, index)
        return payload


def _related_session_delete_ids(
    hermes_home: Path,
    index: dict[str, dict[str, Any]],
    *,
    requested: str,
    session_id: str,
) -> set[str]:
    related = {
        value
        for value in (_normalize_text(requested), _normalize_text(session_id))
        if value is not None
    }
    workspace_id = _workspace_id_from_hermes_home(hermes_home)
    for value in list(related):
        if ":" in value:
            _prefix, suffix = value.split(":", 1)
            if suffix:
                related.add(suffix)
        elif workspace_id:
            related.add(f"{workspace_id}:{value}")

    changed = True
    while changed:
        changed = False
        for key, entry in index.items():
            if not isinstance(entry, dict):
                continue
            values = {
                _normalize_text(key),
                _normalize_text(entry.get("session_id")),
                _normalize_text(entry.get("canonical_session_id")),
                _normalize_text(entry.get("session_key")),
            }
            if values.isdisjoint(related):
                continue
            before = len(related)
            related.update(value for value in values if value is not None)
            for value in list(related):
                if ":" in value:
                    _prefix, suffix = value.split(":", 1)
                    if suffix:
                        related.add(suffix)
                elif workspace_id:
                    related.add(f"{workspace_id}:{value}")
            changed = changed or len(related) != before
    return related


def _unlink_path_and_lock(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    lock_sidecar = path.with_suffix(path.suffix + ".lock")
    try:
        lock_sidecar.unlink(missing_ok=True)
    except OSError:
        pass


def _legacy_flat_snapshot_matches(path: Path, related_ids: set[str]) -> bool:
    payload = _read_json(path, None)
    if not isinstance(payload, dict):
        return False
    values = {
        _normalize_text(payload.get("session_id")),
        _normalize_text(payload.get("canonical_session_id")),
        _normalize_text(payload.get("session_key")),
        _normalize_text(payload.get("platform_session_key")),
    }
    return not values.isdisjoint(related_ids)


def delete_workspace_session_log(hermes_home: Path, requested: str) -> str | None:
    lock_path = _session_index_lock_path(hermes_home)
    with _AdvisoryFileLock(lock_path):
        index = _load_session_index(hermes_home)
        session_id = _resolve_workspace_session_id_from_index(hermes_home, index, requested)
        if session_id is None:
            return None

        related_ids = _related_session_delete_ids(
            hermes_home,
            index,
            requested=requested,
            session_id=session_id,
        )

        for candidate_id in sorted(related_ids):
            for path in (
                _session_log_path(hermes_home, candidate_id),
                _legacy_encoded_session_log_path(hermes_home, candidate_id),
                _legacy_session_log_path(hermes_home, candidate_id),
                _session_logs_dir_path(hermes_home, candidate_id) / f"{_session_segment(hermes_home, candidate_id)}.jsonl",
                _legacy_encoded_session_jsonl_path(hermes_home, candidate_id),
                _legacy_session_jsonl_path(hermes_home, candidate_id),
                _session_trajectory_path(hermes_home, candidate_id),
                _legacy_encoded_session_trajectory_path(hermes_home, candidate_id),
                _legacy_session_trajectory_path(hermes_home, candidate_id),
            ):
                _unlink_path_and_lock(path)

        sessions_dir = _sessions_dir(hermes_home)
        for path in sessions_dir.glob("session_*.json"):
            if _legacy_flat_snapshot_matches(path, related_ids):
                _unlink_path_and_lock(path)

        for variant in related_ids:
            logs_dir = _session_logs_dir_path(hermes_home, session_id)
            for path in (
                logs_dir / f"session_{variant}.json",
                logs_dir / f"session_{_session_file_key(variant)}.json",
                logs_dir / f"{variant}.jsonl",
                logs_dir / f"{_session_file_key(variant)}.jsonl",
                logs_dir / f"session_{variant}.trajectory.jsonl",
                logs_dir / f"session_{_session_file_key(variant)}.trajectory.jsonl",
            ):
                _unlink_path_and_lock(path)

        for candidate_id in sorted(related_ids):
            session_dir = _session_dir_path(hermes_home, candidate_id)
            try:
                shutil.rmtree(session_dir)
            except OSError:
                pass

        to_delete = [
            key
            for key, value in index.items()
            if key in related_ids
            or _normalize_text(value.get("session_id")) in related_ids
            or _normalize_text(value.get("canonical_session_id")) in related_ids
            or _normalize_text(value.get("session_key")) in related_ids
        ]
        for key in to_delete:
            index.pop(key, None)
        _store_session_index(hermes_home, index)
        return session_id


def append_workspace_session_trajectory(
    hermes_home: Path,
    *,
    session_id: str,
    record: dict[str, Any],
) -> None:
    path = _session_trajectory_path(hermes_home, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _AdvisoryFileLock(lock_path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def list_workspace_session_trajectory(
    hermes_home: Path,
    session_id: str,
) -> list[dict[str, Any]]:
    path = _existing_session_trajectory_path(hermes_home, session_id)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in _read_lines(path):
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def configure_agent_workspace_session_paths(agent: Any, hermes_home: Path, session_id: str | None = None) -> None:
    resolved_session_id = _normalize_text(session_id) or _normalize_text(getattr(agent, "session_id", None))
    if resolved_session_id is None:
        return
    logs_dir = _sessions_dir(hermes_home)
    workspace_root = hermes_home.resolve().parent
    setattr(agent, "session_id", resolved_session_id)
    setattr(agent, "session_cwd", str(workspace_root))
    setattr(agent, "logs_dir", logs_dir)
    setattr(agent, "_workspace_session_log_hermes_home", hermes_home)
    setattr(agent, "_workspace_session_log_metadata_only", True)
    setattr(agent, "session_log_file", _ensure_canonical_workspace_session_log(hermes_home, resolved_session_id))


def _candidate_session_aliases(
    *,
    preferred_session_id: str | None,
    alias: str | None,
    platform_session_key: str | None,
    chat_id: str | None,
    thread_id: str | None,
    origin_user_id: str | None,
) -> list[str]:
    ordered = [
        platform_session_key,
        f"{chat_id}:{thread_id}" if chat_id and thread_id else None,
        f"{chat_id}:{origin_user_id}" if chat_id and origin_user_id else None,
        chat_id,
        origin_user_id,
        alias,
        preferred_session_id,
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in ordered:
        normalized = _normalize_ascii_session_alias(candidate)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
