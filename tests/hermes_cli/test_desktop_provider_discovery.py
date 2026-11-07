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
