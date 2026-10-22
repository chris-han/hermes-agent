from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ResolvedSecret:
    key: str
    value: str
    source: str
    updated_at: str | None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_secret(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _parse_iso_utc(value: str | None) -> datetime | None:
    text = _normalize_secret(value)
    if text is None:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def parse_workspace_env_file(hermes_home: Path) -> dict[str, str]:
    env_path = hermes_home / ".env"
    if not env_path.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        result[key] = value.strip()
    return result


def resolve_secret(
    key: str,
    *,
    workspace_env: Mapping[str, str] | None,
    platform_env: Mapping[str, str] | None,
    auth_cache: Mapping[str, Mapping[str, Any]] | None,
) -> ResolvedSecret | None:
    """Resolve secret source deterministically.

    Precedence (P2.3.b):
    1. workspace .env (tenant-scoped explicit override)
    2. platform process environment
    3. auth DB cache row (fallback only)
    """
    normalized_key = _normalize_secret(key)
    if normalized_key is None:
        return None

    if workspace_env is not None:
        value = _normalize_secret(workspace_env.get(normalized_key))
        if value is not None:
            return ResolvedSecret(
                key=normalized_key,
                value=value,
                source="workspace_env",
                updated_at=None,
            )

    if platform_env is not None:
        value = _normalize_secret(platform_env.get(normalized_key))
        if value is not None:
            return ResolvedSecret(
                key=normalized_key,
                value=value,
                source="platform_env",
                updated_at=None,
            )

    if auth_cache is not None:
        cached = auth_cache.get(normalized_key)
        if isinstance(cached, Mapping):
            status = _normalize_secret(str(cached.get("status") or "active")) or "active"
            if status == "revoked":
                return None
            value = _normalize_secret(str(cached.get("value") or ""))
            if value is not None:
                updated_at = _normalize_secret(str(cached.get("updated_at") or ""))
                return ResolvedSecret(
                    key=normalized_key,
                    value=value,
                    source="auth_cache",
                    updated_at=updated_at,
                )

    return None


def build_rotation_plan(
    records: list[Mapping[str, Any]],
    *,
    max_age_days: int = 90,
    warning_age_days: int = 75,
    now_utc: datetime | None = None,
) -> list[dict[str, Any]]:
    now = now_utc or datetime.now(timezone.utc)
    due_cutoff = now - timedelta(days=max_age_days)
    warning_cutoff = now - timedelta(days=warning_age_days)

    plan: list[dict[str, Any]] = []
    for record in records:
        key = _normalize_secret(str(record.get("key") or ""))
        if key is None:
            continue
        rotated_at = _parse_iso_utc(str(record.get("last_rotated_at") or ""))
        if rotated_at is None:
            state = "due"
            reason = "missing_or_invalid_last_rotated_at"
        elif rotated_at <= due_cutoff:
            state = "due"
            reason = "exceeds_max_age"
        elif rotated_at <= warning_cutoff:
            state = "warning"
            reason = "approaching_max_age"
        else:
            state = "ok"
            reason = "within_policy_window"

        plan.append(
            {
                "key": key,
                "state": state,
                "reason": reason,
                "source": _normalize_secret(str(record.get("source") or "")) or "unspecified",
                "last_rotated_at": rotated_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")
                if rotated_at is not None
                else None,
                "evaluated_at_utc": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            }
        )

    return sorted(plan, key=lambda item: (item["state"], item["key"]))


def revoke_auth_cache_entry(
    cache: Mapping[str, Mapping[str, Any]],
    *,
    key: str,
    now_utc: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    normalized_key = _normalize_secret(key)
    if normalized_key is None:
        raise ValueError("key is required")

    now_text = (now_utc or datetime.now(timezone.utc)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    updated: dict[str, dict[str, Any]] = {}
    for existing_key, existing_value in cache.items():
        if not isinstance(existing_value, Mapping):
            continue
        updated[str(existing_key)] = dict(existing_value)

    entry = updated.get(normalized_key, {})
    entry["status"] = "revoked"
    entry["revoked_at"] = now_text
    entry["value"] = ""
    updated[normalized_key] = entry
    return updated
