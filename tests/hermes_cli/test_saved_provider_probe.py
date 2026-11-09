"""Profile-scoped connectivity checks use saved API keys without disclosing them."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def test_saved_key_probe_uses_profile_scope_and_never_returns_key(monkeypatch):
    from hermes_cli import auth, config, desktop_model_discovery, model_switch, provider_catalog
    from hermes_cli.web_models import SavedProviderProbe
    from hermes_cli.web_routers import config_env

    monkeypatch.setattr(config_env, "_require_token", lambda request: None)
    scopes = []

    async def scoped(profile, action):
        scopes.append(profile)
        return action()

    monkeypatch.setattr(config_env, "scoped_to_thread", scoped)
    monkeypatch.setattr(provider_catalog, "provider_catalog", lambda: [
        SimpleNamespace(slug="openai-api", auth_type="api_key", api_key_env_vars=("OPENAI_API_KEY",),
                        base_url_env_var="OPENAI_BASE_URL")
    ])
    monkeypatch.setattr(config, "load_env", lambda: {"OPENAI_API_KEY": "x"})
    monkeypatch.setattr(config, "load_config", lambda: {"providers": {}})
    monkeypatch.setattr(model_switch, "_scoped_key_env", lambda name: {"OPENAI_ORG_ID": "org-test", "OPENAI_PROJECT_ID": "project-test"}.get(name, ""))
    inspected = []
    def snapshot(**kwargs):
        inspected.append(kwargs)
        return desktop_model_discovery.ProviderSnapshot(["verified"], "fresh")
    monkeypatch.setattr(desktop_model_discovery, "read_provider_snapshot", snapshot)

    body = SavedProviderProbe(provider="openai-api", key="OPENAI_API_KEY", profile="beta")
    result = asyncio.run(config_env.validate_saved_provider_credential(body, object()))
    assert scopes == ["beta"]
    assert result["ok"] and result["reachable"]
    assert inspected[0]["base_url"] == auth.PROVIDER_REGISTRY["openai-api"].inference_base_url
    assert inspected[0]["api_key"] == "x"
    assert inspected[0]["headers"] == {"OpenAI-Organization": "org-test", "OpenAI-Project": "project-test"}
    assert "x" not in json.dumps(result)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(config_env.validate_saved_provider_credential(
            SavedProviderProbe(provider="openai-api", key="DEEPSEEK_API_KEY"), object()))
    assert exc.value.status_code == 400


def test_saved_key_probe_missing_or_failed_returns_unknown_not_success(monkeypatch):
    from hermes_cli import config, provider_catalog
    from hermes_cli.web_models import SavedProviderProbe
    from hermes_cli.web_routers import config_env

    monkeypatch.setattr(config_env, "_require_token", lambda request: None)
    async def scoped(profile, action):
        return action()
    monkeypatch.setattr(config_env, "scoped_to_thread", scoped)
    monkeypatch.setattr(provider_catalog, "provider_catalog", lambda: [
        SimpleNamespace(slug="openai-api", auth_type="api_key", api_key_env_vars=("OPENAI_API_KEY",),
                        base_url_env_var="OPENAI_BASE_URL")
    ])
    monkeypatch.setattr(config, "load_env", dict)
    result = asyncio.run(config_env.validate_saved_provider_credential(
        SavedProviderProbe(provider="openai-api", key="OPENAI_API_KEY"), object()))
    assert result["ok"] is False
    assert result["reachable"] is False


def test_real_saved_probe_routes_auth_scope_and_openai_headers(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hermes_cli import web_server
    from hermes_cli.profiles_desktop import resolve_desktop_startup_profile
    from hermes_cli.web_routers.config_env import router

    owner = tmp_path / 'home'
    monkeypatch.setenv('HERMES_HOME', str(owner))
    resolve_desktop_startup_profile(owner)
    observed = []

    class Provider(BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append((self.path, self.headers.get('Authorization'),
                             self.headers.get('OpenAI-Organization'), self.headers.get('OpenAI-Project')))
            accepted = observed[-1] == ('/v1/models', 'Bearer fixture-key', 'org-test', 'project-test')
            self.send_response(200 if accepted else 401)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"fixture-model"}]}' if accepted else b'{"error":"denied"}')
        def log_message(self, *args):
            pass

    provider = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    worker = Thread(target=provider.serve_forever, daemon=True)
    worker.start()
    try:
        (owner / '.env').write_text('OPENAI_API_KEY=fixture-key\nOPENAI_ORG_ID=org-test\nOPENAI_PROJECT_ID=project-test\n')
        (owner / 'config.yaml').write_text('providers:\n  openai-api:\n    base_url: http://127.0.0.1:' + str(provider.server_port) + '/v1\n')
        previous = {p: p.read_bytes() for p in [owner / '.env', owner / 'config.yaml']}
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        body = {'provider': 'openai-api', 'key': 'OPENAI_API_KEY'}
        assert client.post('/api/providers/validate-saved?profile=default', json=body).status_code == 401
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        result = client.post('/api/providers/validate-saved?profile=default', json=body)
        assert result.status_code == 200, result.text
        assert result.json()['ok'] is True
        assert observed == [('/v1/models', 'Bearer fixture-key', 'org-test', 'project-test')]
        assert 'fixture-key' not in result.text
        assert all(p.read_bytes() == contents for p, contents in previous.items())
        assert client.post('/api/providers/validate-saved?profile=missing', json=body).status_code == 404
    finally:
        provider.shutdown()
        provider.server_close()
        worker.join(timeout=5)
