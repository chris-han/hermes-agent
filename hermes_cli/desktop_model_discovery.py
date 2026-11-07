"""Source-bound snapshots for the desktop's query-only provider inventory.

A snapshot is a discovery observation, not model authorization. Legacy picker cache
rows are intentionally not read: they can contain static or curated fallbacks.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ProviderSnapshot:
    models: list[str] | None
    state: str


_inflight_lock = threading.Lock()
_inflight: dict[str, Future[ProviderSnapshot]] = {}


def read_provider_snapshot(*, provider: str, base_url: str, api_key: str | None,
                           source_scope: str, api_mode: str | None = None,
                           headers: dict[str, str] | None = None,
                           refresh: bool = False) -> ProviderSnapshot:
    """Read a verified same-source snapshot or refresh it through the provider adapter."""
    from hermes_cli import models as owner

    endpoint = (base_url or "").strip().rstrip("/")
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        return ProviderSnapshot(None, "unsupported")
    if (not provider or not source_scope or parsed.scheme not in {"http", "https"}
            or not parsed.hostname or parsed.username or parsed.password):
        return ProviderSnapshot(None, "unsupported")
    from agent.command_token_source import materialize_probe_api_key

    materialized_key = materialize_probe_api_key(api_key)
    if callable(api_key) and not materialized_key:
        return ProviderSnapshot(None, "error")
    principal = owner._custom_endpoint_fingerprint(materialized_key, api_mode, headers)
    identity = json.dumps([str(owner._provider_models_cache_path().resolve()), source_scope, provider, endpoint, principal],
                          ensure_ascii=True, separators=(",", ":"))
    key = "desktop-query-v1:" + hashlib.sha256(identity.encode()).hexdigest()

    def cached():
        row = owner._load_provider_models_cache().get(key)
        if not isinstance(row, dict) or row.get("source") != "provider-query-v1":
            return None
        try:
            fetched_at = datetime.fromisoformat(row["fetched_at"])
            if fetched_at.utcoffset() != timedelta(0):
                return None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        values = row.get("models")
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            return None
        return row

    previous_row = cached()
    previous = list(previous_row["models"]) if previous_row is not None else None
    if not refresh:
        state = "stale" if previous_row and previous_row.get("last_refresh_failed") else "cached"
        return ProviderSnapshot(previous, state if previous is not None else "missing")

    def failed() -> ProviderSnapshot:
        if previous is not None:
            with owner._cache_write_lock:
                current = cached()
                if current is not None:
                    owner._store_cache_entry(key, {**current, "last_refresh_failed": True})
        return ProviderSnapshot(previous, "stale" if previous is not None else "error")
    with _inflight_lock:
        future = _inflight.get(key)
        owns_refresh = future is None
        if future is None:
            future = Future()
            _inflight[key] = future
    if not owns_refresh:
        try:
            return future.result(timeout=15)
        except TimeoutError:
            return ProviderSnapshot(previous, "stale" if previous is not None else "error")
    try:
        try:
            result = owner.probe_api_models(materialized_key, endpoint, api_mode=api_mode, request_headers=headers,
                                            require_complete=True)
            values = result.get("models") if isinstance(result, dict) else None
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                snapshot = failed()
            else:
                values = list(dict.fromkeys(value.strip() for value in values))
                row = {"source": "provider-query-v1", "models": values,
                       "fetched_at": datetime.now(UTC).isoformat()}
                with owner._cache_write_lock:
                    owner._store_cache_entry(key, row)
                snapshot = ProviderSnapshot(values, "fresh")
        except Exception:
            snapshot = failed()
        future.set_result(snapshot)
        return snapshot
    finally:
        with _inflight_lock:
            _inflight.pop(key, None)


def configured_provider_rows(ctx, *, source_scope: str, refresh: bool = False) -> list[dict]:
    """Project explicit configured endpoints; never accept endpoint input from a client."""
    from hermes_cli.config import is_provider_enabled
    from hermes_cli.model_switch import _extra_headers_from_config, _scoped_key_env
    from hermes_cli.model_switch_providers import _discover_flag, _entry_api_mode, _entry_base_url, _entry_credentials

    rows = []
    excluded = {str(value).lower() for value in (ctx.excluded_providers or [])}
    for slug, entry in ctx.user_providers.items():
        if not isinstance(entry, dict) or not is_provider_enabled(entry) or slug.lower() in excluded:
            continue
        endpoint = str(_entry_base_url(entry) or "")
        key, key_env, _ = _entry_credentials(entry, "key_env", "api_key_env")
        key = key or _scoped_key_env(key_env)
        if not _discover_flag(entry):
            snapshot = ProviderSnapshot(None, "unsupported")
        else:
            snapshot = read_provider_snapshot(provider=slug, base_url=endpoint, api_key=key,
                source_scope=source_scope, api_mode=_entry_api_mode(entry),
                headers=_extra_headers_from_config(entry), refresh=refresh)
        rows.append({"slug": slug, "name": str(entry.get("name") or slug),
                     "is_current": slug.lower() == ctx.current_provider.lower(),
                     "models": snapshot.models or [], "total_models": len(snapshot.models or []),
                     "discovery_state": snapshot.state})
    return rows
