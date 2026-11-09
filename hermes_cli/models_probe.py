"""Provider model probes; the models facade remains the production patch seam."""
from __future__ import annotations

import time
import urllib.error
import urllib.parse
from typing import Any, Optional


def probe_failure_status(exc: Exception) -> str:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    cause = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    fallback = "timeout" if isinstance(cause, TimeoutError) else "error"
    return {401: "invalid_key", 403: "forbidden", 404: "unsupported", 429: "rate_limited"}.get(code, fallback)


def _reported_result(result: dict, status: str, report: bool) -> dict:
    if report:
        result["status"] = status
    return result


def probe_api_models(
    api_key: Optional[str], base_url: Optional[str], timeout: float = 5.0,
    api_mode: Optional[str] = None, request_headers: Optional[dict[str, str]] = None,
    *, require_complete: bool = False,
) -> dict[str, Any]:
    """Probe a ``/models`` endpoint with light URL heuristics (``base`` then ``base±/v1``).
    ``anthropic_messages`` mode sends ``x-api-key`` + ``anthropic-version`` instead of a bearer; the
    ``data[].id`` response shape is identical. ``models`` is None when no candidate answered."""
    from hermes_cli import models as owner

    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return owner._probe_result(None, None, "")
    if owner._is_github_models_base_url(normalized):
        if require_complete:
            return owner._probe_result(None, owner.COPILOT_MODELS_URL, owner.COPILOT_BASE_URL)
        models = owner._fetch_github_models(api_key=api_key, timeout=timeout)
        return owner._probe_result(models, owner.COPILOT_MODELS_URL, owner.COPILOT_BASE_URL)

    alternate_base = normalized[:-3].rstrip("/") if normalized.endswith("/v1") else normalized + "/v1"
    candidates: list[tuple[str, bool]] = [(normalized, False)]
    if not require_complete and alternate_base and alternate_base != normalized:
        candidates.append((alternate_base, True))

    tried: list[str] = []
    _neg_key = owner._probe_neg_key(normalized)
    if _neg_key is not None and not require_complete:
        _neg_seen = owner._probe_neg_cache.get(_neg_key)
        if _neg_seen is not None and (time.monotonic() - _neg_seen) < owner._PROBE_NEG_TTL:
            return owner._probe_result(
                None, normalized.rstrip("/") + "/models", normalized,
                alternate_base if alternate_base != normalized else None)
    headers: dict[str, str] = {"User-Agent": owner._HERMES_USER_AGENT}
    if urllib.parse.urlparse(normalized).hostname == "generativelanguage.googleapis.com":
        headers["X-Goog-Api-Client"] = f"hermes-agent/{owner.get_version_info().base_version}"
    if api_key and api_mode == "anthropic_messages":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if normalized.startswith(owner.COPILOT_BASE_URL):
        headers.update(owner.copilot_default_headers())
    if isinstance(request_headers, dict):
        # Per-provider custom headers can contain secrets: merge last so endpoint config wins; never log.
        from hermes_cli.config import normalize_extra_headers

        headers.update(normalize_extra_headers(request_headers))

    # Only thread ssl_context when a per-provider TLS override applies; public endpoints keep the
    # original 2-arg call so existing call-seam mocks stay valid.
    _open_kwargs: dict[str, Any] = {}
    _ssl_context = owner._custom_provider_ssl_context(normalized)
    if _ssl_context is not None:
        _open_kwargs["ssl_context"] = _ssl_context
    all_timed_out = True
    probe_status = "unknown"
    for candidate_base, is_fallback in candidates:
        url = candidate_base.rstrip("/") + "/models"
        tried.append(url)
        try:
            data = owner._get_json(url, timeout=timeout, headers=headers, **_open_kwargs)
        except Exception as exc:
            # TLS, authentication and parsing failures must not hide corrected settings.
            cause = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            probe_status = probe_failure_status(exc)
            all_timed_out = all_timed_out and isinstance(cause, TimeoutError)
            continue
        if require_complete:
            # A malformed or paginated response is not an authoritative empty or full list.
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                all_timed_out = False
                continue
            links = data.get("links")
            if (data.get("has_more") or data.get("next") or data.get("next_page")
                    or (isinstance(links, dict) and links.get("next"))):
                all_timed_out = False
                continue
            if any(not isinstance(item, dict) or not isinstance(item.get("id"), str)
                   or not item["id"].strip() for item in data["data"]):
                all_timed_out = False
                continue
            return owner._probe_result([item["id"].strip() for item in data["data"]], url,
                                 candidate_base.rstrip("/"),
                                 alternate_base if alternate_base != candidate_base else normalized, is_fallback)
        if _neg_key is not None:
            owner._probe_neg_cache.pop(_neg_key, None)
        from hermes_cli.chat_catalog import note_catalog_item

        probed = []
        for item in data.get("data", []):
            if isinstance(item, dict) and note_catalog_item(item):
                continue
            probed.append(item.get("id", ""))
        return owner._probe_result(
            probed, url, candidate_base.rstrip("/"),
            alternate_base if alternate_base != candidate_base else normalized, is_fallback)

    if _neg_key is not None and all_timed_out and not require_complete:
        owner._probe_neg_cache[_neg_key] = time.monotonic()
    result = owner._probe_result(
        None, tried[0] if tried else normalized.rstrip("/") + "/models", normalized,
        alternate_base if alternate_base != normalized else None)
    return _reported_result(result, probe_status, require_complete)
