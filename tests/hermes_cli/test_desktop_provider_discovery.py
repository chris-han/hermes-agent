"""Desktop discovery snapshots must contain only source-verified provider responses."""

import json

import pytest

from hermes_cli.desktop_model_discovery import read_provider_snapshot


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    from hermes_cli import models

    path = tmp_path / "provider_models_cache.json"
    monkeypatch.setattr(models, "_provider_models_cache_path", lambda: path)
    return path


def test_success_empty_replaces_prior_snapshot_and_failure_retains_verified_snapshot(cache_path, monkeypatch):
    from hermes_cli import models

    responses = iter([["alpha", "beta"], None, []])
    monkeypatch.setattr(models, "probe_api_models", lambda *args, **kwargs: {"models": next(responses)})
    source = {"provider": "fixture", "base_url": "https://provider.test/v1", "api_key": "fixture-secret", "source_scope": "connection-a"}
    first = read_provider_snapshot(**source, refresh=True)
    assert first.models == ["alpha", "beta"]
    assert first.state == "fresh"
    failed = read_provider_snapshot(**source, refresh=True)
    assert failed.models == ["alpha", "beta"]
    assert failed.state == "stale"
    empty = read_provider_snapshot(**source, refresh=True)
    assert empty.models == []
    assert empty.state == "fresh"
    assert read_provider_snapshot(**source).models == []
    assert "fixture-secret" not in cache_path.read_text()


def test_legacy_cache_and_changed_endpoint_or_principal_are_not_reused(cache_path, monkeypatch):
    from hermes_cli import models

    cache_path.write_text(json.dumps({"fixture": {"fp": "legacy", "at": 1, "models": ["static-only"]}}))
    source = {"provider": "fixture", "base_url": "https://provider.test/v1", "api_key": "fixture-secret", "source_scope": "connection-a"}
    assert read_provider_snapshot(**source).models is None
    monkeypatch.setattr(models, "probe_api_models", lambda *args, **kwargs: {"models": ["verified"]})
    assert read_provider_snapshot(**source, refresh=True).models == ["verified"]
    assert read_provider_snapshot(**{**source, "base_url": "https://other.test/v1"}).models is None
    assert read_provider_snapshot(**{**source, "api_key": "different-secret"}).models is None
    assert read_provider_snapshot(**{**source, "provider": "other"}).models is None
    assert read_provider_snapshot(**{**source, "source_scope": "connection-b"}).models is None


def test_failed_refresh_remains_labelled_after_subsequent_reads(cache_path, monkeypatch):
    from hermes_cli import models

    responses = iter([["alpha"], None])
    monkeypatch.setattr(models, "probe_api_models", lambda *args, **kwargs: {"models": next(responses)})
    source = {"provider": "fixture", "base_url": "https://provider.test/v1", "api_key": "fixture-secret", "source_scope": "connection-a"}
    assert read_provider_snapshot(**source, refresh=True).state == "fresh"
    assert read_provider_snapshot(**source, refresh=True).state == "stale"
    assert read_provider_snapshot(**source).state == "stale"


@pytest.mark.parametrize("response", [{}, {"data": [{"id": "alpha"}], "has_more": True}, {"data": [{"bad": "alpha"}]}])
def test_strict_provider_probe_rejects_malformed_or_partial_lists(monkeypatch, response):
    from hermes_cli import models

    monkeypatch.setattr(models, "_get_json", lambda *args, **kwargs: response)
    result = models.probe_api_models(None, "https://provider.test/v1", require_complete=True)
    assert result["models"] is None


def test_same_source_concurrent_refreshes_are_coalesced(cache_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli import models

    started = threading.Event()
    release = threading.Event()
    calls = []

    def probe(*args, **kwargs):
        calls.append(1)
        started.set()
        assert release.wait(5)
        return {"models": ["alpha"]}

    monkeypatch.setattr(models, "probe_api_models", probe)
    source = {"provider": "fixture", "base_url": "https://provider.test/v1", "api_key": "fixture-secret", "source_scope": "connection-a"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(read_provider_snapshot, **source, refresh=True)
        assert started.wait(5)
        second = pool.submit(read_provider_snapshot, **source, refresh=True)
        # The first call is deliberately held until the second is scheduled.
        threading.Event().wait(0.2)
        release.set()
        assert first.result(timeout=5).models == ["alpha"]
        assert second.result(timeout=5).models == ["alpha"]
    assert len(calls) == 1


def test_existing_cache_file_is_partitioned_by_profile_home(tmp_path, monkeypatch):
    from hermes_cli import models

    home = [tmp_path / "profile-a.json"]
    monkeypatch.setattr(models, "_provider_models_cache_path", lambda: home[0])
    monkeypatch.setattr(models, "probe_api_models", lambda *args, **kwargs: {"models": ["only-a"]})
    source = {"provider": "fixture", "base_url": "https://provider.test/v1", "api_key": "fixture-secret", "source_scope": "same-connection"}
    assert read_provider_snapshot(**source, refresh=True).models == ["only-a"]
    home[0] = tmp_path / "profile-b.json"
    assert read_provider_snapshot(**source).models is None


def test_real_http_provider_refresh_failure_empty_and_pagination(cache_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"status": 200, "body": {"data": [{"id": "alpha"}, {"id": "beta"}]}, "calls": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["calls"] += 1
            if self.headers.get("Authorization") != "Bearer fixture-secret":
                self.send_response(401)
                self.end_headers()
                return
            body = json.dumps(state["body"]).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        source = {"provider": "fixture", "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                  "api_key": "fixture-secret", "source_scope": "connection-a"}
        assert read_provider_snapshot(**source, refresh=True).models == ["alpha", "beta"]
        state.update(status=503, body={})
        failure = read_provider_snapshot(**source, refresh=True)
        assert failure.models == ["alpha", "beta"] and failure.state == "stale"
        state.update(status=200, body={"data": [{"id": "partial"}], "has_more": True})
        partial = read_provider_snapshot(**source, refresh=True)
        assert partial.models == ["alpha", "beta"] and partial.state == "stale"
        state.update(status=200, body={"data": []})
        empty = read_provider_snapshot(**source, refresh=True)
        assert empty.models == [] and empty.state == "fresh"
        assert "fixture-secret" not in cache_path.read_text()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_strict_refresh_does_not_skip_provider_query_due_to_legacy_timeout_cache(monkeypatch):
    import time
    from hermes_cli import models

    endpoint = "https://provider.test/v1"
    key = models._probe_neg_key(endpoint)
    monkeypatch.setitem(models._probe_neg_cache, key, time.monotonic())
    calls = []
    monkeypatch.setattr(models, "_get_json", lambda *args, **kwargs: calls.append(args[0]) or {"data": [{"id": "recovered"}]})
    result = models.probe_api_models(None, endpoint, require_complete=True)
    assert result["models"] == ["recovered"]
    assert calls
    assert key in models._probe_neg_cache


def test_strict_timeout_does_not_poison_legacy_probe_cache(monkeypatch):
    from hermes_cli import models

    endpoint = "https://isolated-provider.test/v1"
    key = models._probe_neg_key(endpoint)
    monkeypatch.delitem(models._probe_neg_cache, key, raising=False)
    monkeypatch.setattr(models, "_get_json", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()))
    assert models.probe_api_models(None, endpoint, require_complete=True)["models"] is None
    assert key not in models._probe_neg_cache


def test_snapshot_without_valid_utc_provenance_is_not_admitted(cache_path, monkeypatch):
    from hermes_cli import models

    monkeypatch.setattr(models, "probe_api_models", lambda *args, **kwargs: {"models": ["alpha"]})
    source = {"provider": "fixture", "base_url": "https://provider.test/v1", "api_key": "fixture-secret", "source_scope": "connection-a"}
    assert read_provider_snapshot(**source, refresh=True).models == ["alpha"]
    data = json.loads(cache_path.read_text())
    row = next(iter(data.values()))
    for bad in (None, "not-a-time", "2026-11-07T12:00:00"):
        row["fetched_at"] = bad
        cache_path.write_text(json.dumps(data))
        assert read_provider_snapshot(**source).models is None


def test_configured_provider_projection_never_uses_saved_or_static_models(cache_path, monkeypatch):
    from hermes_cli import models
    from hermes_cli.desktop_model_discovery import configured_provider_rows
    from hermes_cli.inventory import ConfigContext

    monkeypatch.setattr(models, "probe_api_models", lambda *args, **kwargs: {"models": ["live"]})
    ctx = ConfigContext(current_provider="alpha", current_model="retired", current_base_url="https://alpha.test/v1",
        user_providers={"alpha": {"base_url": "https://alpha.test/v1", "api_key": "secret", "models": ["static-only"]},
                        "disabled": {"enabled": False, "base_url": "https://disabled.test/v1"}},
        custom_providers=[])
    rows = configured_provider_rows(ctx, source_scope="connection-a", refresh=True)
    assert [row["slug"] for row in rows] == ["alpha"]
    assert rows[0]["models"] == ["live"]
    assert rows[0]["discovery_state"] == "fresh"
    assert "secret" not in json.dumps(rows)
    assert "eligible_models" not in rows[0]


def test_command_credentials_are_partitioned_by_materialized_principal(cache_path, monkeypatch):
    from hermes_cli import models
    principal = ["first"]
    monkeypatch.setattr(models, "probe_api_models", lambda *args, **kwargs: {"models": ["private"]})
    source = {"provider": "fixture", "base_url": "https://provider.test/v1",
              "api_key": lambda: principal[0], "source_scope": "connection-a"}
    assert read_provider_snapshot(**source, refresh=True).models == ["private"]
    principal[0] = "second"
    assert read_provider_snapshot(**source).models is None


def test_ollama_native_refresh_queries_tags_and_preserves_empty_failure(cache_path, monkeypatch):
    from hermes_cli import models
    seen = []
    replies = iter([{"models": [{"name": "local-a"}]}, {"models": []}, OSError("offline")])

    def get_json(url, *, timeout, headers=None, **kwargs):
        seen.append((url, headers))
        result = next(replies)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(models, "_get_json", get_json)
    source = {"provider": "ollama", "base_url": "http://localhost:11434/v1",
              "api_key": "local-key", "source_scope": "connection-a", "adapter": "ollama"}
    assert read_provider_snapshot(**source, refresh=True).models == ["local-a"]
    assert read_provider_snapshot(**source, refresh=True).models == []
    failed = read_provider_snapshot(**source, refresh=True)
    assert failed.models == [] and failed.state == "stale"
    assert [item[0] for item in seen] == ["http://localhost:11434/api/tags"] * 3
    assert all(any(k.lower() == "authorization" and v == "Bearer local-key" for k, v in headers.items())
               for _, headers in seen)


def test_ollama_malformed_item_cannot_replace_verified_snapshot(cache_path, monkeypatch):
    from hermes_cli import models
    responses = iter([{"models": [{"name": "verified"}]}, {"models": [{"name": "new"}, {}]}])
    monkeypatch.setattr(models, "_get_json", lambda *args, **kwargs: next(responses))
    source = {"provider": "ollama", "base_url": "http://localhost:11434/v1",
              "api_key": None, "source_scope": "connection-a", "adapter": "ollama"}
    assert read_provider_snapshot(**source, refresh=True).models == ["verified"]
    result = read_provider_snapshot(**source, refresh=True)
    assert result.models == ["verified"] and result.state == "stale"


def test_custom_provider_entries_use_source_query_once_and_never_offer_saved_models(cache_path, monkeypatch):
    from hermes_cli import models
    from hermes_cli.desktop_model_discovery import configured_provider_rows
    from hermes_cli.inventory import ConfigContext

    calls = []
    monkeypatch.setattr(models, "probe_api_models", lambda key, url, **kwargs:
                        calls.append((key, url)) or {"models": ["queried"]})
    ctx = ConfigContext(current_provider="custom:sample", current_model="retired", current_base_url="https://example.test/v1",
        user_providers={}, custom_providers=[
            {"name": "Sample — Old", "provider_key": "sample", "base_url": "https://example.test/v1", "api_key": "key", "model": "old"},
            {"name": "Sample — Retired", "provider_key": "sample", "base_url": "https://example.test/v1", "api_key": "key", "model": "retired"},
        ])
    rows = configured_provider_rows(ctx, source_scope="connection-a", refresh=True)
    assert len(rows) == 1
    assert rows[0]["slug"] == "custom:sample"
    assert rows[0]["is_current"] is True
    assert rows[0]["models"] == ["queried"]
    assert calls == [("key", "https://example.test/v1")]
    assert "key" not in json.dumps(rows)


def test_configured_registered_provider_uses_profile_credential_and_runtime_endpoint(cache_path, monkeypatch):
    from hermes_cli import models, model_switch
    from hermes_cli.desktop_model_discovery import configured_provider_rows
    from hermes_cli.inventory import ConfigContext

    calls = []
    monkeypatch.setattr(model_switch, "_scoped_key_env", lambda name: {"OPENAI_API_KEY": "profile-key"}.get(name, ""))
    monkeypatch.setattr(models, "probe_api_models", lambda key, url, **kwargs:
                        calls.append((key, url)) or {"models": ["live"]})
    ctx = ConfigContext(current_provider="openai-api", current_model="old", current_base_url="https://relay.test/v1",
                        configured_provider="openai-api", configured_base_url="https://relay.test/v1",
                        user_providers={"openai-api": {}}, custom_providers=[])
    rows = configured_provider_rows(ctx, source_scope="connection-a", refresh=True)
    assert rows[0]["models"] == ["live"]
    assert calls == [("profile-key", "https://relay.test/v1")]


def test_keyed_catalog_providers_appear_without_provider_config(cache_path, monkeypatch):
    from dataclasses import replace
    from hermes_cli import models, model_switch
    from hermes_cli.desktop_model_discovery import configured_provider_rows
    from hermes_cli.inventory import ConfigContext
    from agent import semantier_gateway

    monkeypatch.setattr(semantier_gateway, "cloud_connection_config", dict)
    keys = {"OPENAI_API_KEY": "a", "DASHSCOPE_API_KEY": "b", "DEEPSEEK_API_KEY": "c"}
    monkeypatch.setattr(model_switch, "_scoped_key_env", lambda name: keys.get(name, ""))
    monkeypatch.setattr(models, "probe_api_models", lambda value, url, **kwargs: {"models": [url]})
    ctx = ConfigContext(current_provider="", current_model="", current_base_url="",
                        user_providers={}, custom_providers=[])
    rows = configured_provider_rows(ctx, source_scope="profile-a", refresh=True)
    assert {"openai-api", "alibaba", "deepseek"} <= {row["slug"] for row in rows if row["models"]}
    restricted = replace(ctx, user_providers={"openai-api": {"enabled": False}},
                         excluded_providers=["alibaba"])
    restricted_rows = configured_provider_rows(restricted, source_scope="profile-a", refresh=True)
    restricted_slugs = {row["slug"] for row in restricted_rows}
    assert "openai-api" not in restricted_slugs
    assert "alibaba" not in restricted_slugs
    assert "deepseek" in restricted_slugs


def test_refresh_queries_sources_concurrently_with_profile_context(cache_path, monkeypatch):
    import contextvars
    import threading
    from hermes_cli import models
    from hermes_cli.desktop_model_discovery import configured_provider_rows
    from hermes_cli.inventory import ConfigContext

    context = contextvars.ContextVar("test_profile", default="wrong")
    context.set("profile-a")
    barrier = threading.Barrier(2)
    seen = []

    def probe(key, url, **kwargs):
        seen.append((url, context.get()))
        barrier.wait(timeout=2)
        return {"models": ["verified"]}

    monkeypatch.setattr(models, "probe_api_models", probe)
    ctx = ConfigContext(current_provider="", current_model="", current_base_url="", user_providers={
        "a": {"base_url": "https://a.test/v1"}, "b": {"base_url": "https://b.test/v1"}}, custom_providers=[])
    rows = configured_provider_rows(ctx, source_scope="connection-a", refresh=True)
    assert [row["models"] for row in rows] == [["verified"], ["verified"]]
    assert sorted(seen) == [("https://a.test/v1", "profile-a"), ("https://b.test/v1", "profile-a")]


def test_snapshot_exposes_only_opaque_source_and_utc_timestamp(cache_path, monkeypatch):
    from hermes_cli import models
    monkeypatch.setattr(models, "probe_api_models", lambda *args, **kwargs: {"models": ["live"]})
    source = {"provider": "fixture", "base_url": "https://provider.test/v1",
              "api_key": "private-key", "source_scope": "connection-a"}
    fresh = read_provider_snapshot(**source, refresh=True)
    cached = read_provider_snapshot(**source)
    other = read_provider_snapshot(**{**source, "api_key": "another-key"})
    assert fresh.source_id and fresh.source_id == cached.source_id and fresh.source_id != other.source_id
    assert fresh.fetched_at == cached.fetched_at and fresh.fetched_at.endswith("+00:00")
    assert "private-key" not in str(fresh.source_id)


def test_cloud_uses_actor_catalog_and_retains_only_same_actor_snapshot(cache_path, monkeypatch):
    from hermes_cli import models
    from agent import semantier_gateway
    from hermes_cli.desktop_model_discovery import configured_provider_rows
    from hermes_cli.inventory import ConfigContext

    def forbidden(*args, **kwargs):
        pytest.fail("Cloud must use the actor-authorized catalog")
    monkeypatch.setattr(models, "probe_api_models", forbidden)
    principal = ["key-a"]
    monkeypatch.setattr(semantier_gateway, "gateway_settings", lambda **_:
                        semantier_gateway.GatewaySettings("https://cloud.test/v1", principal[0]))
    replies = iter([{"object": "list", "data": [{"id": "actor-a"}]}, ValueError("offline")])
    def catalog(*, refresh, settings):
        assert refresh and settings.api_key == "key-a"
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(semantier_gateway, "cloud_catalog", catalog)
    ctx = ConfigContext(current_provider="semantier-cloud", current_model="old", current_base_url="https://cloud.test/v1",
        user_providers={"semantier-cloud": {"base_url": "https://cloud.test/v1", "api_key": "key"}}, custom_providers=[])
    row = configured_provider_rows(ctx, source_scope="connection-a", refresh=True)[0]
    assert row["models"] == row["eligible_models"] == ["actor-a"]
    row = configured_provider_rows(ctx, source_scope="connection-a", refresh=True)[0]
    assert row["discovery_state"] == "stale" and row["eligible_models"] == ["actor-a"]
    principal[0] = "key-b"
    row = configured_provider_rows(ctx, source_scope="connection-a")[0]
    assert row["models"] == [] and row["eligible_models"] == [] and row["manual_models"] == []


def test_desktop_inventory_uses_separate_query_only_path_and_retains_saved_selection(cache_path, monkeypatch):
    from hermes_cli import inventory, models
    from hermes_cli.inventory import ConfigContext

    monkeypatch.setattr(models, "probe_api_models", lambda *args, **kwargs: {"models": []})
    monkeypatch.setattr(inventory, "build_models_payload", lambda *args, **kwargs: pytest.fail("legacy fallback called"))
    ctx = ConfigContext(current_provider="fixture", current_model="retired", current_base_url="https://fixture.test/v1",
                        user_providers={"fixture": {"base_url": "https://fixture.test/v1", "models": ["static"]}},
                        custom_providers=[])
    payload = inventory.build_desktop_model_options_payload(ctx, source_scope="connection-a", refresh=True)
    assert payload["discovery_contract"] == "provider-query-v1"
    assert payload["provider"] == "fixture" and payload["model"] == "retired"
    assert payload["providers"][0]["models"] == []
    assert payload["providers"][0]["discovery_state"] == "fresh"
    assert payload["providers"][0].get("eligible_models") is None


def test_ollama_native_force_refresh_uses_real_http_and_same_source_cache(cache_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"status": 200, "models": [{"name": "alpha"}], "paths": []}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["paths"].append(self.path)
            body = json.dumps({"models": state["models"]}).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        source = {"provider": "ollama", "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                  "api_key": None, "source_scope": "connection-a", "adapter": "ollama"}
        assert read_provider_snapshot(**source, refresh=True).models == ["alpha"]
        state["models"] = [{"name": "beta"}]
        assert read_provider_snapshot(**source, refresh=True).models == ["beta"]
        state["status"] = 503
        stale = read_provider_snapshot(**source, refresh=True)
        assert stale.models == ["beta"] and stale.state == "stale"
        state.update(status=200, models=[])
        assert read_provider_snapshot(**source, refresh=True).models == []
        assert state["paths"] == ["/api/tags"] * 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_legacy_cloud_route_is_actor_catalog_only_even_when_custom_endpoint_exists(cache_path, monkeypatch):
    from agent import semantier_gateway
    from hermes_cli import models
    from hermes_cli.desktop_model_discovery import configured_provider_rows
    from hermes_cli.inventory import ConfigContext

    monkeypatch.setattr(semantier_gateway, "cloud_connection_config", lambda: {"base_url": "https://cloud.test/v1"})
    monkeypatch.setattr(semantier_gateway, "gateway_settings", lambda **_:
                        semantier_gateway.GatewaySettings("https://cloud.test/v1", "actor-key"))
    monkeypatch.setattr(semantier_gateway, "cloud_catalog", lambda **kwargs: {"object": "list", "data": [{"id": "granted"}]})
    monkeypatch.setattr(models, "probe_api_models", lambda *args, **kwargs: pytest.fail("generic Cloud query"))
    ctx = ConfigContext(current_provider="custom", current_model="granted", current_base_url="https://cloud.test/v1",
                        user_providers={}, custom_providers=[{"name": "Proxy", "base_url": "https://cloud.test/v1"}])
    rows = configured_provider_rows(ctx, source_scope="connection-a", refresh=True)
    assert len(rows) == 1 and rows[0]["slug"] == "semantier-cloud"
    assert rows[0]["models"] == rows[0]["eligible_models"] == ["granted"]
    assert rows[0]["is_current"] is True


def test_registered_oauth_provider_is_disclosed_as_unsupported_without_native_adapter(cache_path, monkeypatch):
    from hermes_cli import models
    from hermes_cli.desktop_model_discovery import configured_provider_rows
    from hermes_cli.inventory import ConfigContext

    monkeypatch.setattr(models, "probe_api_models", lambda *args, **kwargs: pytest.fail("unqualified OAuth probe"))
    ctx = ConfigContext(current_provider="openai-codex", current_model="old", current_base_url="https://service.test/v1",
                        user_providers={"openai-codex": {"base_url": "https://service.test/v1"}}, custom_providers=[])
    row = configured_provider_rows(ctx, source_scope="connection-a", refresh=True)[0]
    assert row["models"] == [] and row["discovery_state"] == "unsupported"


def test_bare_current_custom_endpoint_comes_from_profile_config_only(cache_path, monkeypatch):
    from hermes_cli import config, models
    from hermes_cli.desktop_model_discovery import configured_provider_rows
    from hermes_cli.inventory import ConfigContext

    calls = []
    monkeypatch.setattr(config, "load_config", lambda: {"model": {"provider": "custom",
                        "base_url": "https://configured.test/v1", "api_key": "key"}})
    monkeypatch.setattr(models, "probe_api_models", lambda key, url, **kwargs:
                        calls.append((key, url)) or {"models": ["live"]})
    ctx = ConfigContext(current_provider="custom", current_model="old", current_base_url="https://configured.test/v1",
                        user_providers={}, custom_providers=[])
    rows = configured_provider_rows(ctx, source_scope="connection-a", refresh=True)
    assert len(rows) == 1 and rows[0]["slug"] == "custom" and rows[0]["models"] == ["live"]
    assert calls == [("key", "https://configured.test/v1")]
    calls.clear()
    changed = ctx.with_overrides(current_base_url="https://other.test/v1")
    assert configured_provider_rows(changed, source_scope="connection-a", refresh=True) == []
    assert calls == []


def test_session_model_options_query_scope_is_bound_to_authenticated_actor(monkeypatch, tmp_path):
    import asyncio
    from contextlib import nullcontext
    from types import SimpleNamespace
    from gateway import session_config
    from gateway import run as gateway_run
    from hermes_cli import inventory
    from hermes_state_runtime import RuntimeStoreError

    actor = SimpleNamespace(profile_id=str(tmp_path), subject="actor-one", capabilities={"session:read"})
    connection = SimpleNamespace(actor=actor, authority=SimpleNamespace(profile_id=str(tmp_path)))
    ref = SimpleNamespace(session_id="")
    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", lambda home: nullcontext())
    monkeypatch.setattr(inventory, "load_picker_context", lambda: inventory.ConfigContext(
        current_provider="", current_model="", current_base_url="", user_providers={}, custom_providers=[]))
    scopes = []
    monkeypatch.setattr(inventory, "build_desktop_model_options_payload", lambda ctx, **kwargs:
                        scopes.append(kwargs["source_scope"]) or {"providers": [], "discovery_contract": "provider-query-v1"})
    response = asyncio.run(session_config.model_options(connection, ref, {"query_only": True, "query_scope": "connection-a"}))
    assert response["discovery_contract"] == "provider-query-v1"
    actor.subject = "actor-two"
    asyncio.run(session_config.model_options(connection, ref, {"query_only": True, "query_scope": "connection-a"}))
    assert scopes[0] != scopes[1]
    for params in ({"query_only": True}, {"query_only": True, "query_scope": ""},
                   {"query_only": "true", "query_scope": "connection-a"},
                   {"query_only": True, "query_scope": "x" * 201},
                   {"query_only": True, "query_scope": "connection-a", "base_url": "https://spoof.test"}):
        with pytest.raises(RuntimeStoreError):
            asyncio.run(session_config.model_options(connection, ref, params))


def test_legacy_rpc_query_only_forwards_scoped_contract_without_legacy_builder(monkeypatch):
    from tui_gateway import server
    from hermes_cli import inventory

    ctx = inventory.ConfigContext(current_provider="", current_model="", current_base_url="", user_providers={}, custom_providers=[])
    monkeypatch.setattr(server, "_model_picker_context", lambda agent: ctx)
    monkeypatch.setattr(inventory, "build_model_options_payload", lambda *args, **kwargs: pytest.fail("legacy picker used"))
    seen = []
    monkeypatch.setattr(inventory, "build_desktop_model_options_payload", lambda ctx, **kwargs:
                        seen.append(kwargs) or {"providers": [], "discovery_contract": "provider-query-v1"})
    response = server._methods["model.options"](1, {"query_only": True, "query_scope": "connection-a", "refresh": True})
    assert response["result"]["discovery_contract"] == "provider-query-v1"
    assert seen[0]["refresh"] and "connection-a" in seen[0]["source_scope"]
    assert "error" in server._methods["model.options"](1, {"query_only": True})


def test_dashboard_rest_query_only_uses_profile_scope_and_validates_connection_scope(monkeypatch, tmp_path):
    import asyncio
    from contextlib import contextmanager
    from fastapi import HTTPException
    from hermes_cli import inventory
    from hermes_cli.web_routers import models as router

    seen = []
    @contextmanager
    def profile_scope(profile):
        seen.append(("profile", profile))
        yield
    monkeypatch.setattr(router, "_config_profile_scope", profile_scope)
    monkeypatch.setattr(router, "_dashboard_code_skew_guard", lambda: None)
    monkeypatch.setattr(inventory, "load_picker_context", lambda: object())
    monkeypatch.setattr(inventory, "build_desktop_model_options_payload", lambda ctx, **kwargs:
                        seen.append(("scope", kwargs["source_scope"])) or {"providers": [], "discovery_contract": "provider-query-v1"})
    payload = asyncio.run(router.get_model_options(profile="work", query_only=True, query_scope="connection-a"))
    assert payload["discovery_contract"] == "provider-query-v1"
    assert seen[0] == ("profile", "work") and "connection-a" in seen[1][1]
    with pytest.raises(HTTPException) as exc:
        asyncio.run(router.get_model_options(query_only=True))
    assert exc.value.status_code == 400


def test_bare_custom_assignment_resolves_existing_configured_endpoint(monkeypatch):
    from agent import semantier_gateway
    from hermes_cli import web_server_config

    seen = []
    monkeypatch.setattr(semantier_gateway, "is_gateway_route", lambda _: False)
    monkeypatch.setattr(web_server_config, "_validated_main_model_selection", lambda cfg, provider, model, base_url, key:
                        seen.append((provider, model, base_url, key)) or object())
    cfg = {"model": {"provider": "custom", "base_url": "https://configured.test/v1", "default": "old"}}
    url, _ = web_server_config._prepare_main_assignment(cfg, "custom", "live", "", "")
    assert url == "https://configured.test/v1"
    assert seen == [("custom", "live", url, "")]


def test_anthropic_native_pagination_is_complete_and_failure_retains_prior(cache_path, monkeypatch):
    from hermes_cli import models
    urls = []
    replies = iter([
        {"data": [{"id": "first"}], "has_more": True, "last_id": "cursor-1"},
        {"data": [{"id": "second"}], "has_more": False},
        {"data": [{"id": "partial"}], "has_more": True, "last_id": "repeat"},
        {"data": [{"id": "ignored"}], "has_more": True, "last_id": "repeat"},
    ])
    def get_json(url, *, headers, **kwargs):
        urls.append(url)
        assert headers["x-api-key"] == "secret" and headers["anthropic-version"]
        return next(replies)
    monkeypatch.setattr(models, "_get_json", get_json)
    source = {"provider": "anthropic", "base_url": "https://api.anthropic.com", "api_key": "secret",
              "source_scope": "connection-a", "adapter": "anthropic"}
    assert read_provider_snapshot(**source, refresh=True).models == ["first", "second"]
    assert "after_id=cursor-1" in urls[1]
    stale = read_provider_snapshot(**source, refresh=True)
    assert stale.state == "stale" and stale.models == ["first", "second"]
    assert "secret" not in cache_path.read_text()


def test_gemini_native_listing_follows_pages_and_excludes_non_generation_models(cache_path, monkeypatch):
    from hermes_cli import models
    urls = []
    replies = iter([
        {"models": [{"name": "models/gemini-live", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/embedding", "supportedGenerationMethods": ["embedContent"]}], "nextPageToken": "next"},
        {"models": [{"name": "models/gemini-new", "supportedGenerationMethods": ["generateContent"]}]},
        {"models": [{"name": "models/partial"}], "nextPageToken": "repeat"},
        {"models": [], "nextPageToken": "repeat"},
    ])
    def get_json(url, *, headers, **kwargs):
        urls.append(url)
        assert headers["x-goog-api-key"] == "key"
        return next(replies)
    monkeypatch.setattr(models, "_get_json", get_json)
    source = {"provider": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta",
              "api_key": "key", "source_scope": "connection-a", "adapter": "gemini"}
    first = read_provider_snapshot(**source, refresh=True)
    assert first.models == ["models/gemini-live", "models/gemini-new"]
    assert "pageToken=next" in urls[1]
    stale = read_provider_snapshot(**source, refresh=True)
    assert stale.state == "stale" and stale.models == first.models


def test_cache_scope_retains_full_principal_identity_if_legacy_fingerprint_collides(cache_path, monkeypatch):
    from hermes_cli import models
    monkeypatch.setattr(models, "_custom_endpoint_fingerprint", lambda *args: "legacy-collision")
    monkeypatch.setattr(models, "probe_api_models", lambda *args, **kwargs: {"models": ["private"]})
    source = {"provider": "fixture", "base_url": "https://provider.test/v1", "api_key": "first-key", "source_scope": "connection-a"}
    assert read_provider_snapshot(**source, refresh=True).models == ["private"]
    assert read_provider_snapshot(**{**source, "api_key": "second-key"}).models is None


def test_compatible_provider_projection_does_not_query_the_same_config_twice(cache_path, monkeypatch):
    from hermes_cli import config, models
    from hermes_cli.desktop_model_discovery import configured_provider_rows
    from hermes_cli.inventory import ConfigContext

    calls = []
    monkeypatch.setattr(models, "probe_api_models", lambda key, url, **kwargs:
                        calls.append(url) or {"models": ["live"]})
    entries = {"lab": {"name": "Lab", "base_url": "https://lab.test/v1", "api_key": "secret", "models": ["retired"]}}
    ctx = ConfigContext(current_provider="custom:lab", current_model="retired", current_base_url="https://lab.test/v1",
                        user_providers=entries, custom_providers=config.get_compatible_custom_providers({"providers": entries}))
    rows = configured_provider_rows(ctx, source_scope="connection-a", refresh=True)
    assert len(rows) == 1 and rows[0]["models"] == ["live"]
    assert rows[0]["is_current"] and "custom:lab" in rows[0]["aliases"]
    assert calls == ["https://lab.test/v1"]


def test_live_session_base_url_cannot_redirect_configured_provider_probe(cache_path, monkeypatch):
    from hermes_cli import models, model_switch
    from hermes_cli.desktop_model_discovery import configured_provider_rows
    from hermes_cli.inventory import ConfigContext

    calls = []
    monkeypatch.setattr(model_switch, "_scoped_key_env", lambda name: "secret" if name == "OPENAI_API_KEY" else "")
    monkeypatch.setattr(models, "probe_api_models", lambda key, url, **kwargs: calls.append(url) or {"models": ["live"]})
    ctx = ConfigContext(current_provider="openai-api", current_model="old", current_base_url="https://other.test/v1",
                        configured_provider="openai-api", configured_base_url="https://configured.test/v1", user_providers={"openai-api": {}}, custom_providers=[])
    rows = configured_provider_rows(ctx, source_scope="connection-a", refresh=True)
    assert calls == ["https://configured.test/v1"]
    assert rows[0]["is_current"] is False


def test_provider_without_endpoint_does_not_inherit_another_configured_provider_route(cache_path, monkeypatch):
    from hermes_cli import models, model_switch
    from hermes_cli.desktop_model_discovery import configured_provider_rows
    from hermes_cli.inventory import ConfigContext

    calls = []
    monkeypatch.setattr(model_switch, "_scoped_key_env", lambda name: "secret" if name == "OPENAI_API_KEY" else "")
    monkeypatch.setattr(models, "probe_api_models", lambda key, url, **kwargs: calls.append(url) or {"models": []})
    ctx = ConfigContext(current_provider="openai-api", current_model="old", current_base_url="https://other.test/v1",
                        configured_provider="different", configured_base_url="https://different.test/v1",
                        user_providers={"openai-api": {}}, custom_providers=[])
    configured_provider_rows(ctx, source_scope="connection-a", refresh=True)
    assert calls == ["https://api.openai.com/v1"]


def test_strict_query_never_changes_the_configured_endpoint_on_failure(monkeypatch):
    from hermes_cli import models
    calls = []
    def fail(url, **kwargs):
        calls.append(url)
        raise OSError("offline")
    monkeypatch.setattr(models, "_get_json", fail)
    assert models.probe_api_models("key", "https://provider.test/tenant/v1", require_complete=True)["models"] is None
    assert calls == ["https://provider.test/tenant/v1/models"]
