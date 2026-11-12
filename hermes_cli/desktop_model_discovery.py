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
from urllib.parse import urlencode, urlsplit


@dataclass(frozen=True)
class ProviderSnapshot:
    models: list[str] | None
    state: str
    source_id: str | None = None
    fetched_at: str | None = None
    status: str | None = None


_inflight_lock = threading.Lock()
_inflight: dict[str, Future[ProviderSnapshot]] = {}


def read_provider_snapshot(*, provider: str, base_url: str, api_key: str | None,
                           source_scope: str, api_mode: str | None = None,
                           headers: dict[str, str] | None = None,
                           refresh: bool = False, adapter: str = "openai", query=None, cache_path=None, model_list_url: str = "") -> ProviderSnapshot:
    """Read a verified same-source snapshot or refresh it through the provider adapter."""
    from hermes_cli import models as owner

    endpoint = (base_url or "").strip().rstrip("/")
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        return ProviderSnapshot(None, "unsupported")
    if (not provider or not source_scope or parsed.scheme not in {"http", "https"}
            or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment
            or adapter not in {"openai", "ollama", "cloud", "anthropic", "gemini", "workspace-byok"}):
        return ProviderSnapshot(None, "unsupported")
    from agent.command_token_source import materialize_probe_api_key

    adapter = _effective_catalog_adapter(adapter, model_list_url)
    materialized_key = materialize_probe_api_key(api_key)
    if callable(api_key) and not materialized_key:
        return ProviderSnapshot(None, "error")
    principal = owner._custom_endpoint_fingerprint(materialized_key, api_mode, headers)
    principal_material = json.dumps([materialized_key, api_mode, headers or {}], sort_keys=True,
                                    ensure_ascii=True, separators=(",", ":"))
    principal_full = hashlib.sha256(principal_material.encode()).hexdigest()
    cache_path = cache_path or owner._provider_models_cache_path()
    identity = json.dumps([str(cache_path.resolve()), source_scope, provider, endpoint,
                           principal, principal_full, adapter, model_list_url],
                          ensure_ascii=True, separators=(",", ":"))
    key = "desktop-query-v1:" + hashlib.sha256(identity.encode()).hexdigest()

    def cache_rows():
        if cache_path == owner._provider_models_cache_path():
            return owner._load_provider_models_cache()
        try:
            return owner._read_json_cache(cache_path) or {}
        except Exception:
            return {}

    def store(row):
        if cache_path == owner._provider_models_cache_path():
            owner._store_cache_entry(key, row)
        else:
            rows = cache_rows()
            rows[key] = row
            try:
                owner._write_json_cache(cache_path, rows, indent=None)
            except Exception:
                pass  # Cache failure never turns a verified query into configuration loss.

    def cached():
        row = cache_rows().get(key)
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
        return ProviderSnapshot(previous, state if previous is not None else "missing", key,
                                previous_row.get("fetched_at") if previous_row else None)

    def failed(status: str = "unknown") -> ProviderSnapshot:
        if previous is not None:
            with owner._cache_write_lock:
                current = cached()
                if current is not None:
                    store({**current, "last_refresh_failed": True})
        return ProviderSnapshot(previous, "stale" if previous is not None else "error", key,
                                previous_row.get("fetched_at") if previous_row else None, status)
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
            return ProviderSnapshot(previous, "stale" if previous is not None else "error", key,
                                    previous_row.get("fetched_at") if previous_row else None)
    try:
        try:
            probe_status = "unknown"
            if adapter == "workspace-byok":
                values = query() if callable(query) else None
            elif adapter == "gemini":
                request_headers = {"User-Agent": owner._HERMES_USER_AGENT, "x-goog-api-key": materialized_key,
                                   **(headers or {})}
                values = None
                seen: set[str] = set()
                collected = []
                cursor = None
                ssl_context = owner._custom_provider_ssl_context(endpoint)
                kwargs = {"ssl_context": ssl_context} if ssl_context is not None else {}
                for _ in range(20):
                    if not materialized_key:
                        break
                    params = {"pageSize": "1000"}
                    if cursor:
                        params["pageToken"] = cursor
                    payload = owner._get_json(endpoint + "/models?" + urlencode(params), timeout=5.0,
                                              headers=request_headers, **kwargs)
                    raw = payload.get("models") if isinstance(payload, dict) else None
                    if (not isinstance(raw, list) or any(not isinstance(item, dict) or
                            not isinstance(item.get("name"), str) or not item["name"].strip() or
                            ("supportedGenerationMethods" in item and
                             not isinstance(item["supportedGenerationMethods"], list)) for item in raw)):
                        break
                    collected.extend(item["name"] for item in raw if "supportedGenerationMethods" not in item or
                                     "generateContent" in item["supportedGenerationMethods"])
                    next_cursor = payload.get("nextPageToken")
                    if next_cursor is None or next_cursor == "":
                        values = collected
                        break
                    if not isinstance(next_cursor, str) or next_cursor in seen:
                        break
                    seen.add(next_cursor)
                    cursor = next_cursor
            elif adapter == "anthropic":
                request_headers = {"User-Agent": owner._HERMES_USER_AGENT, "anthropic-version": "2023-06-01",
                                   "x-api-key": materialized_key, **(headers or {})}
                values = None
                seen: set[str] = set()
                collected = []
                cursor = None
                ssl_context = owner._custom_provider_ssl_context(endpoint)
                kwargs = {"ssl_context": ssl_context} if ssl_context is not None else {}
                for _ in range(owner._ANTHROPIC_MODELS_MAX_PAGES):
                    if not materialized_key:
                        break
                    payload = owner._get_json(owner._anthropic_models_url(endpoint, after_id=cursor),
                                              timeout=5.0, headers=request_headers, **kwargs)
                    raw = payload.get("data") if isinstance(payload, dict) else None
                    if (not isinstance(raw, list) or type(payload.get("has_more")) is not bool or
                            any(not isinstance(item, dict) or not isinstance(item.get("id"), str) or
                                not item["id"].strip() for item in raw)):
                        break
                    collected.extend(item["id"] for item in raw)
                    if not payload["has_more"]:
                        values = collected
                        break
                    cursor = owner._anthropic_next_cursor(payload, seen)
                    if cursor is None:
                        break
            elif adapter == "cloud":
                from agent.semantier_gateway import GatewaySettings, cloud_catalog

                payload = cloud_catalog(refresh=True, settings=GatewaySettings(endpoint, materialized_key))
                raw = payload.get("data") if isinstance(payload, dict) and payload.get("object") == "list" else None
                if not isinstance(raw, list) or any(not isinstance(item, dict) or not
                        isinstance(item.get("id"), str) or not item["id"].strip() for item in raw):
                    values = None
                else:
                    values = [item["id"] for item in raw]
            elif adapter == "ollama":
                from hermes_cli.models_local import _root_for_ollama_native_api

                request_headers = {"User-Agent": owner._HERMES_USER_AGENT, **(headers or {})}
                if materialized_key and not any(k.lower() == "authorization" for k in request_headers):
                    request_headers["Authorization"] = f"Bearer {materialized_key}"
                ssl_context = owner._custom_provider_ssl_context(endpoint)
                kwargs = {"ssl_context": ssl_context} if ssl_context is not None else {}
                payload = owner._get_json(_root_for_ollama_native_api(endpoint) + "/api/tags",
                                          timeout=5.0, headers=request_headers, **kwargs)
                raw = payload.get("models") if isinstance(payload, dict) else None
                if not isinstance(raw, list) or any(not isinstance(item, dict) or not
                        isinstance(item.get("model") or item.get("name"), str) or not
                        (item.get("model") or item.get("name")).strip() for item in raw):
                    values = None
                else:
                    values = [item.get("model") or item["name"] for item in raw]
            else:
                result = _query_openai_snapshot(materialized_key, endpoint, api_mode, headers, model_list_url)
                values = result.get("models") if isinstance(result, dict) else None
                probe_status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                snapshot = failed(probe_status)
            else:
                values = list(dict.fromkeys(value.strip() for value in values))
                row = {"source": "provider-query-v1", "models": values,
                       "fetched_at": datetime.now(UTC).isoformat()}
                with owner._cache_write_lock:
                    store(row)
                snapshot = ProviderSnapshot(values, "fresh", key, row["fetched_at"])
        except Exception as exc:  # health: allow BLE001 -- Provider adapters are external boundaries; only sanitized status leaves this query.
            from hermes_cli.models_probe import probe_failure_status
            from hermes_cli.cloud_device_client import CloudConnectionError
            snapshot = failed(exc.code if isinstance(exc, CloudConnectionError) else probe_failure_status(exc))
        future.set_result(snapshot)
        return snapshot
    finally:
        with _inflight_lock:
            _inflight.pop(key, None)



def read_configured_provider_snapshot(*, provider: str, entry: dict, base_url: str,
                                      api_key: str, source_scope: str,
                                      refresh: bool = False) -> ProviderSnapshot:
    """Use the same native adapter and request identity for discovery and key tests."""
    from hermes_cli.model_switch import _extra_headers_from_config, _scoped_key_env
    from hermes_cli.model_switch_providers import _entry_api_mode

    host = (urlsplit(base_url).hostname or "").lower()
    slug = provider.lower()
    adapter = "openai"
    if slug == "anthropic" and host == "api.anthropic.com":
        adapter = "anthropic"
    elif slug == "gemini" and host == "generativelanguage.googleapis.com" and urlsplit(base_url).path.rstrip("/") in {"/v1", "/v1beta"}:
        adapter = "gemini"
    elif slug in {"ollama", "custom:ollama"} and host != "ollama.com" and not host.endswith(".ollama.com"):
        adapter = "ollama"
    headers = _extra_headers_from_config(entry)
    if slug == "openai-api":
        for header, name in (("OpenAI-Organization", "OPENAI_ORG_ID"), ("OpenAI-Project", "OPENAI_PROJECT_ID")):
            value = _scoped_key_env(name)
            if value and not any(key.lower() == header.lower() for key in headers):
                headers[header] = value
    return read_provider_snapshot(provider=provider, base_url=base_url, api_key=api_key,
                                  source_scope=source_scope, api_mode=_entry_api_mode(entry),
                                  headers=headers, refresh=refresh, adapter=adapter,
                                  model_list_url=str(entry.get("model_list_url") or ""))


def _catalog_key_sources(ctx, excluded, cloud_identity):
    from hermes_cli.provider_catalog import provider_catalog
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.model_switch import _scoped_key_env
    from hermes_cli.route_identity import normalize_route_base_url

    sources = []

    # The provider catalog already owns all API-key provider identities,
    # credentials and endpoint defaults. Treat a profile key as a configured
    # provider without inventing per-vendor model-discovery special cases.
    for descriptor in provider_catalog():
        slug = descriptor.slug
        if descriptor.auth_type != "api_key" or not descriptor.api_key_env_vars:
            continue
        if slug.lower() in excluded or any(name.lower() == slug.lower() for name in ctx.user_providers):
            continue
        registered = PROVIDER_REGISTRY.get(slug)
        if not registered or registered.auth_type != "api_key":
            continue
        key = next((value for name in descriptor.api_key_env_vars if (value := _scoped_key_env(name))), "")
        if not key:
            continue
        endpoint = (_scoped_key_env(descriptor.base_url_env_var) if descriptor.base_url_env_var else "") or registered.inference_base_url
        if cloud_identity and normalize_route_base_url(endpoint) == cloud_identity:
            continue
        sources.append((slug, descriptor.label, endpoint, key, {}, None))

    return sources


def configured_provider_rows(ctx, *, source_scope: str, refresh: bool = False) -> list[dict]:
    """Project explicit configured endpoints; never accept endpoint input from a client."""
    from hermes_cli.config import is_provider_enabled, load_config
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.model_switch import _extra_headers_from_config, _scoped_key_env
    from hermes_cli.model_switch_providers import (
        _discover_flag, _display_prefix, _entry_api_mode, _entry_base_url, _entry_credentials,
    )
    from hermes_cli.providers import custom_provider_aliases, custom_provider_slug
    from hermes_cli.route_identity import normalize_route_base_url
    from agent.semantier_gateway import cloud_connection_config

    sources = []
    excluded = {str(value).lower() for value in (ctx.excluded_providers or [])}
    cloud_config = cloud_connection_config()
    cloud_endpoint = (str(cloud_config.get("base_url") or "") if isinstance(cloud_config, dict) else "")
    if not cloud_endpoint:
        cloud_entry = ctx.user_providers.get("semantier-cloud")
        if isinstance(cloud_entry, dict):
            cloud_endpoint = str(_entry_base_url(cloud_entry) or "")
    cloud_identity = normalize_route_base_url(cloud_endpoint)
    def project(slug, name, endpoint, key, entry, *, aliases=None):
        is_cloud = slug.lower() == "semantier-cloud"
        settings = None
        settings_failure = None
        if is_cloud:
            from agent.semantier_gateway import gateway_settings
            from hermes_cli.cloud_device_client import CloudConnectionError

            try:
                settings = gateway_settings(allow_unselected=True)
            except CloudConnectionError as error:
                settings_failure = error.code
            except ValueError:
                settings_failure = 'cloud_config_invalid'
        if (is_cloud and (not settings or settings.base_url.rstrip("/") != endpoint.rstrip("/"))) or not _discover_flag(entry):
            snapshot = ProviderSnapshot(None, "error", status=settings_failure) if settings_failure else ProviderSnapshot(None, "unsupported")
        else:
            if is_cloud:
                snapshot = read_provider_snapshot(provider=slug, base_url=endpoint, api_key=settings.api_key,
                    source_scope=source_scope, api_mode=_entry_api_mode(entry),
                    headers=_extra_headers_from_config(entry), refresh=refresh, adapter="cloud")
            else:
                snapshot = read_configured_provider_snapshot(provider=slug, entry=entry, base_url=endpoint,
                    api_key=key, source_scope=source_scope, refresh=refresh)
        aliases = sorted(aliases or [])
        current = ctx.current_provider.lower()
        is_current = current in {slug.lower(), *(alias.lower() for alias in aliases)}
        if is_cloud and cloud_identity and normalize_route_base_url(ctx.current_base_url) == cloud_identity:
            is_current = True
        if ctx.current_base_url and is_current:
            is_current = endpoint.rstrip("/") == ctx.current_base_url.rstrip("/")
        from agent.model_metadata import is_local_endpoint
        row = {"slug": slug, "name": name, "is_current": is_current,
               "authenticated": bool(key) or is_local_endpoint(endpoint),
               "is_user_defined": True, "aliases": aliases,
               "models": snapshot.models or [], "total_models": len(snapshot.models or []),
               "manual_models": list(dict.fromkeys(str(value).strip() for value in
                   [*(entry.get("models") or []), entry.get("model") or ""] if str(value).strip())),
               "discovery_state": snapshot.state, "discovery_source_id": snapshot.source_id,
               "discovery_status": snapshot.status,
               "discovery_fetched_at": snapshot.fetched_at}
        # This separate actor-authorized catalog is an existing authorization
        # owner, unlike generic model discovery. Preserve its grant projection.
        if is_cloud:
            row["authenticated"] = settings is not None
            if snapshot.models is not None:
                row["eligible_models"] = list(snapshot.models)
        return row

    for slug, entry in ctx.user_providers.items():
        if not isinstance(entry, dict) or not is_provider_enabled(entry) or slug.lower() in excluded:
            continue
        registered = PROVIDER_REGISTRY.get(slug)
        endpoint = str(_entry_base_url(entry) or "")
        aliases = custom_provider_aliases(str(entry.get("name") or slug), slug)
        if not endpoint and ctx.configured_provider.lower() in {slug.lower(), *(alias.lower() for alias in aliases)}:
            endpoint = ctx.configured_base_url
        if not endpoint and registered and registered.auth_type == "api_key":
            endpoint = (_scoped_key_env(registered.base_url_env_var) if registered.base_url_env_var else "")
            endpoint = endpoint or registered.inference_base_url
        if slug.lower() != "semantier-cloud" and cloud_identity and normalize_route_base_url(endpoint) == cloud_identity:
            continue
        key, key_env, _ = _entry_credentials(entry, "key_env", "api_key_env")
        key = key or _scoped_key_env(key_env)
        if not key and registered and registered.auth_type == "api_key":
            key = next((value for name in registered.api_key_env_vars
                        if (value := _scoped_key_env(name))), "")
        if registered and registered.auth_type != "api_key" and slug.lower() != "semantier-cloud":
            # OAuth/external-process providers require their own account-aware
            # listing adapter; an anonymous OpenAI-style request is not one.
            entry = {**entry, "discover_models": False}
        sources.append((slug, str(entry.get("name") or slug), endpoint, key, entry, aliases))

    sources.extend(_catalog_key_sources(ctx, excluded, cloud_identity))

    groups = {}
    for entry in ctx.custom_providers:
        if not isinstance(entry, dict) or not is_provider_enabled(entry):
            continue
        name = str(entry.get("name") or "").strip()
        endpoint = str(_entry_base_url(entry) or "").strip().rstrip("/")
        if not name or not endpoint:
            continue
        if cloud_identity and normalize_route_base_url(endpoint) == cloud_identity:
            continue
        display = _display_prefix(name)
        slug = custom_provider_slug(display, str(entry.get("provider_key") or ""))
        if slug.lower() in excluded:
            continue
        key, key_env, identity = _entry_credentials(entry, "key_env", "api_key_env")
        key = key or _scoped_key_env(key_env)
        provider_key = str(entry.get("provider_key") or "").strip().lower()
        duplicate = next((source for source in sources if source[0].lower() == provider_key
                          and normalize_route_base_url(source[2]) == normalize_route_base_url(endpoint)
                          and _entry_credentials(source[4], "key_env", "api_key_env")[2] == identity
                          and _entry_api_mode(source[4]) == _entry_api_mode(entry)
                          and _extra_headers_from_config(source[4]) == _extra_headers_from_config(entry)), None)
        if duplicate is not None:
            continue
        group_key = (slug.lower(), endpoint, identity, _entry_api_mode(entry),
                     tuple(sorted(_extra_headers_from_config(entry).items())))
        if group_key not in groups:
            groups[group_key] = {"slug": slug, "name": display, "endpoint": endpoint, "key": key,
                                 "entry": dict(entry), "aliases": set()}
        group = groups[group_key]
        group["entry"]["discover_models"] = _discover_flag(group["entry"]) and _discover_flag(entry)
        group["aliases"].update(custom_provider_aliases(name, str(entry.get("provider_key") or "")))

    used = {source[0].lower() for source in sources}
    if cloud_identity and "semantier-cloud" not in used and "semantier-cloud" not in excluded:
        sources.append(("semantier-cloud", "Semantier Model Provider", cloud_endpoint, "",
                        {"base_url": cloud_endpoint}, None))
        used.add("semantier-cloud")
    for group in groups.values():
        slug = group["slug"]
        if slug.lower() in used:
            suffix = 2
            while f"{slug}-{suffix}".lower() in used:
                suffix += 1
            slug = f"{slug}-{suffix}"
        used.add(slug.lower())
        sources.append((slug, group["name"], group["endpoint"], group["key"], group["entry"], group["aliases"]))

    if ctx.current_provider.lower() == "custom" and "custom" not in excluded and "custom" not in used:
        model = load_config().get("model")
        model = model if isinstance(model, dict) else {}
        endpoint = str(_entry_base_url(model) or "").strip().rstrip("/")
        endpoint_identity = normalize_route_base_url(endpoint)
        represented = any(normalize_route_base_url(source[2]) == endpoint_identity for source in sources)
        if (str(model.get("provider") or "").lower() == "custom" and endpoint_identity and not represented
                and endpoint_identity == normalize_route_base_url(ctx.current_base_url)
                and endpoint_identity != cloud_identity):
            key, key_env, _ = _entry_credentials(model, "key_env", "api_key_env")
            sources.append(("custom", "Custom endpoint", endpoint, key or _scoped_key_env(key_env), model, None))

    def build(source):
        slug, name, endpoint, key, entry, aliases = source
        return project(slug, name, endpoint, key, entry, aliases=aliases)

    if not refresh or len(sources) < 2:
        return [build(source) for source in sources]
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    with ThreadPoolExecutor(max_workers=min(4, len(sources)), thread_name_prefix="desktop-model-query") as pool:
        futures = [pool.submit(copy_context().run, build, source) for source in sources]
        return [future.result() for future in futures]


def validate_catalog_url(url: str, *, check_network: bool = False) -> str:
    """Native endpoints may be local; metadata targets and URL credentials never are valid."""
    from agent.model_metadata import is_local_endpoint
    from tools.url_safety import is_always_blocked_url, is_safe_url

    value = url.strip()
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError('Invalid provider URL.') from exc
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username
            or parsed.password or parsed.fragment or parsed.query or port == 0):
        raise ValueError('Provider URL requires an HTTP(S) host without credentials, query or fragment.')
    if is_always_blocked_url(value) or (check_network and not is_local_endpoint(value) and not is_safe_url(value)):
        raise ValueError('Provider URL is blocked by network security rules.')
    return value


def query_catalog_url(url: str, headers: dict[str, str] | None = None) -> dict:
    """Read only the explicitly configured catalog; never POST or guess another endpoint."""
    import httpx
    from agent.model_metadata import is_local_endpoint
    from tools.url_safety import create_ssrf_safe_client

    def result(status, message, *, reachable=False, models=None):
        return {'ok': status in {'catalog_verified', 'empty'}, 'status': status,
                'reachable': reachable, 'message': message, 'models': models or []}

    try:
        target = validate_catalog_url(url, check_network=True)
        local = is_local_endpoint(target)
        factory = httpx.Client if local else create_ssrf_safe_client
        with factory(timeout=5.0, follow_redirects=False, trust_env=not local) as client:
            with client.stream('GET', target, headers={'Accept': 'application/json', **(headers or {})}) as response:
                if not response.is_success:
                    status = {401: 'invalid_key', 403: 'forbidden', 404: 'unsupported',
                              405: 'unsupported', 429: 'rate_limited'}.get(response.status_code, 'error')
                    return result(status, f'Catalog returned HTTP {response.status_code}.', reachable=True)
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > 2 * 1024 * 1024:
                        return result('error', 'Catalog response is too large.', reachable=True)
        payload = json.loads(body)
        rows = payload.get('data') if isinstance(payload, dict) else None
        if (not isinstance(rows, list) or _catalog_has_more(payload)
                or any(not isinstance(row, dict) or not isinstance(row.get('id'), str)
                       or not row['id'].strip() for row in rows)):
            return result('unsupported', 'No complete supported model catalog; enter a model ID manually.', reachable=True)
        models = list(dict.fromkeys(row['id'].strip() for row in rows))
        return result('catalog_verified' if models else 'empty',
                      'Catalog verified. Inference has not been tested.' if models else
                      'The catalog is empty. You can enter a model ID manually.', reachable=True, models=models)
    except httpx.TimeoutException:
        return result('timeout', 'Catalog request timed out. Manual model entry remains available.')
    except (json.JSONDecodeError, UnicodeDecodeError):
        return result('unsupported', 'Response is not a supported catalog. Enter a model ID manually.', reachable=True)
    except ValueError:
        return result('invalid_url', 'Provider URL is invalid or blocked by network security rules.')
    except httpx.HTTPError:
        return result('unreachable', 'Could not reach the catalog. Manual model entry remains available.')


def _effective_catalog_adapter(adapter: str, model_list_url: str) -> str:
    return "openai" if model_list_url else adapter


def _query_openai_snapshot(api_key, endpoint, api_mode, headers, model_list_url):
    from hermes_cli import models as owner
    if model_list_url:
        request_headers = catalog_request_headers(api_key, api_mode, headers)
        result = query_catalog_url(model_list_url, request_headers)
        return {**result, "models": result["models"] if result["ok"] else None}
    return owner.probe_api_models(api_key, endpoint, api_mode=api_mode, request_headers=headers, require_complete=True)


def _catalog_has_more(payload: dict) -> bool:
    links = payload.get('links')
    return bool(payload.get('has_more') or payload.get('next') or payload.get('next_page')
                or (isinstance(links, dict) and links.get('next')))


def catalog_request_headers(api_key: str | None, api_mode: str | None, headers: dict | None = None) -> dict:
    result = {"Accept": "application/json"}
    if api_key:
        if api_mode == "anthropic_messages":
            result.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
        else:
            result["Authorization"] = f"Bearer {api_key}"
    return {**result, **(headers or {})}
