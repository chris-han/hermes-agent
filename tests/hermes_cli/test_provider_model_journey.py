"""Real synthetic provider loop for model-independent profile-owned setup."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def provider_fixture():
    state = {'status': 200, 'models': ['fixture/alpha'], 'calls': []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state['calls'].append((self.path, self.headers.get('Authorization')))
            status = state['status'] if state.get('keyless') or self.headers.get('Authorization') == 'Bearer fixture-key' else 401
            payload = json.dumps(state.get('payload', {'data': [{'id': m} for m in state['models']]})).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            if state.get('location'):
                self.send_header('Location', state['location'])
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
            state['calls'].append((self.path, payload))
            body = json.dumps({'id': 'fixture-completion', 'object': 'chat.completion', 'created': 1,
                               'model': payload['model'], 'choices': [{'index': 0, 'finish_reason': 'stop',
                               'message': {'role': 'assistant', 'content': 'OK'}}]}).encode()
            accepted_key = state.get('keyless') or self.headers.get('Authorization') == 'Bearer fixture-key'
            self.send_response(state.get('inference_status', 200) if accepted_key else 401)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_save_without_model_preserves_main_and_discovers_exact_list_url(provider_fixture):
    from hermes_cli.web_models import CustomEndpointUpdate
    from hermes_cli.web_routers import config_env
    from hermes_cli.desktop_model_discovery import read_configured_provider_snapshot

    origin, state = provider_fixture
    body = CustomEndpointUpdate(name='Fixture', base_url=origin + '/v1',
                                model_list_url=origin + '/catalog', api_key='fixture-key')
    cfg = {'model': {'provider': 'previous', 'default': 'keep-me'}}
    slug, entry = config_env._write_custom_endpoint(cfg, body)
    assert cfg['model'] == {'provider': 'previous', 'default': 'keep-me'}
    assert entry['model_list_url'] == origin + '/catalog'
    assert 'fixture-key' not in json.dumps(config_env._custom_endpoint_response(cfg))
    result = read_configured_provider_snapshot(provider=slug, entry=entry, base_url=entry['base_url'],
                                              api_key='fixture-key', source_scope='profile-a', refresh=True)
    assert result.models == ['fixture/alpha'] and result.state == 'fresh'
    assert state['calls'] == [('/catalog', 'Bearer fixture-key')]
    probe = asyncio.run(config_env.validate_custom_endpoint(body))
    assert probe['ok'] and probe['status'] == 'catalog_verified'
    assert all(path == '/catalog' for path, _ in state['calls'])


@pytest.mark.parametrize('status,expected', [(401, 'invalid_key'), (403, 'forbidden'),
                                           (404, 'unsupported'), (429, 'rate_limited'), (500, 'error')])
def test_connection_truthful_status_without_inference(provider_fixture, status, expected):
    from hermes_cli.web_models import CustomEndpointUpdate
    from hermes_cli.web_routers.config_env import validate_custom_endpoint

    origin, state = provider_fixture
    state['status'] = status
    result = asyncio.run(validate_custom_endpoint(CustomEndpointUpdate(
        name='Fixture', base_url=origin + '/v1', model_list_url=origin + '/catalog', api_key='fixture-key')))
    assert result['status'] == expected and result['ok'] is False
    assert state['calls'] == [('/catalog', 'Bearer fixture-key')]


def test_scoped_save_reload_main_manual_and_explicit_inference(provider_fixture, tmp_path, monkeypatch):
    from pathlib import Path
    from hermes_cli import profiles
    from hermes_cli.web_models import CustomEndpointUpdate, ModelTestRequest
    from hermes_cli.web_routers import config_env, models
    from hermes_cli.config import atomic_config_replace
    from fastapi import HTTPException

    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / '.hermes'))
    home_a = profiles.create_profile('fixture-a', no_alias=True, no_skills=True)
    profiles.create_profile('fixture-b', no_alias=True, no_skills=True)
    atomic_config_replace(home_a / 'config.yaml', {'model': {'provider': 'previous', 'default': 'keep'}})
    origin, state = provider_fixture
    saved = config_env.upsert_custom_endpoint(CustomEndpointUpdate(
        name='Fixture', base_url=origin + '/v1', model_list_url=origin + '/catalog', api_key='fixture-key'), 'fixture-a')
    slug = saved['id']
    assert config_env.list_custom_endpoints('fixture-a')['endpoints'][0]['id'] == slug
    assert config_env.list_custom_endpoints('fixture-b')['endpoints'] == []
    assert 'fixture-key' not in json.dumps(saved)
    assert config_env.list_custom_endpoints('fixture-a')['current']['model'] == 'keep'
    assert not any(path.endswith('completions') for path, _ in state['calls'])
    # No catalog does not block an explicitly selected manual id.
    state['status'] = 404
    body = ModelTestRequest(provider=slug, model='manual/hidden', accept_cost=True)
    result = asyncio.run(models.test_provider_model(body, 'fixture-a'))
    assert result == {'ok': True, 'status': 'response_observed', 'provider': slug, 'model': 'manual/hidden'}
    posts = [(path, payload) for path, payload in state['calls'] if path.endswith('/chat/completions')]
    assert len(posts) == 1 and posts[0][1]['model'] == 'manual/hidden'
    assert posts[0][1]['max_tokens'] == 32
    assert config_env.list_custom_endpoints('fixture-a')['current']['model'] == 'keep'
    with pytest.raises(HTTPException):
        asyncio.run(models.test_provider_model(ModelTestRequest(provider=slug, model='manual/hidden'), 'fixture-a'))
    before = len(state['calls'])
    with pytest.raises((HTTPException, ValueError)):
        asyncio.run(models.test_provider_model(body, 'fixture-b'))
    assert len(state['calls']) == before


def test_exact_catalog_cache_identity_transitions(provider_fixture, tmp_path):
    from hermes_cli.desktop_model_discovery import read_provider_snapshot
    origin, state = provider_fixture
    source = dict(provider='fixture', base_url=origin + '/v1', api_key='fixture-key', source_scope='source-a/profile-a',
                  model_list_url=origin + '/catalog', cache_path=tmp_path / 'cache.json')
    first = read_provider_snapshot(**source, refresh=True)
    assert first.models == ['fixture/alpha'] and first.state == 'fresh'
    assert read_provider_snapshot(**source).state == 'cached'
    for changes in ({'api_key': 'replacement'}, {'model_list_url': origin + '/other'},
                    {'source_scope': 'source-a/profile-b'}, {'source_scope': 'source-b/profile-a'}):
        value = read_provider_snapshot(**{**source, **changes})
        assert value.state == 'missing' and value.models is None
    state['status'] = 403
    stale = read_provider_snapshot(**source, refresh=True)
    assert stale.state == 'stale' and stale.models == ['fixture/alpha'] and stale.status == 'forbidden'
    assert read_provider_snapshot(**source).state == 'stale'
    state.update(status=200, models=[])
    empty = read_provider_snapshot(**source, refresh=True)
    assert empty.state == 'fresh' and empty.models == []
    assert 'fixture-key' not in (tmp_path / 'cache.json').read_text(encoding="utf-8")


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'http://user:pass@localhost/v1',
                                'http://169.254.169.254/models', 'http://100.100.100.200/models',
                                'https://metadata.google.internal/models', 'http://localhost:bad/models'])
def test_reject_unsafe_provider_urls_before_any_query(url):
    from hermes_cli.desktop_model_discovery import query_catalog_url
    assert query_catalog_url(url)['status'] == 'invalid_url'


def test_catalog_timeout_is_not_authentication_or_inference_success(monkeypatch):
    import httpx
    from hermes_cli.desktop_model_discovery import query_catalog_url
    class Client:
        def __init__(self, **_kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def stream(self, *_args, **_kwargs):
            raise httpx.ReadTimeout('Synthetic timeout')
    monkeypatch.setattr(httpx, 'Client', Client)
    result = query_catalog_url('http://127.0.0.1:1234/catalog')
    assert result['status'] == 'timeout' and not result['ok']


def test_model_list_success_does_not_establish_inference_success(provider_fixture, tmp_path, monkeypatch):
    from pathlib import Path
    from hermes_cli.web_models import CustomEndpointUpdate, ModelTestRequest
    from hermes_cli.web_routers import config_env, models
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / '.hermes'))
    origin, state = provider_fixture
    body = CustomEndpointUpdate(name='Fixture', base_url=origin + '/v1',
                                model_list_url=origin + '/catalog', api_key='fixture-key')
    saved = config_env.upsert_custom_endpoint(body)
    assert asyncio.run(config_env.validate_custom_endpoint(body))['ok']
    assert not any(path.endswith('completions') for path, _ in state['calls'])
    state['inference_status'] = 403
    result = asyncio.run(models.test_provider_model(ModelTestRequest(
        provider=saved['id'], model='fixture/alpha', accept_cost=True)))
    assert result['ok'] is False and result['status'] == 'forbidden'
    assert config_env.list_custom_endpoints()['current']['model'] == ''


@pytest.mark.parametrize('pagination', [{'has_more': True}, {'next': 'cursor'}, {'next_page': 'cursor'}, {'links': {'next': 'https://other.invalid'}}])
def test_custom_catalog_does_not_advertise_partial_lists(provider_fixture, pagination):
    from hermes_cli.desktop_model_discovery import query_catalog_url
    origin, state = provider_fixture
    state['payload'] = {'data': [{'id': 'partial-only'}], **pagination}
    result = query_catalog_url(origin + '/catalog', {'Authorization': 'Bearer fixture-key'})
    assert not result['ok'] and result['models'] == [] and result['status'] == 'unsupported'


def test_custom_catalog_caps_response_and_does_not_follow_redirects(provider_fixture):
    from hermes_cli.desktop_model_discovery import query_catalog_url
    origin, state = provider_fixture
    headers = {'Authorization': 'Bearer fixture-key'}
    state['payload'] = {'data': [{'id': 'x' * (2 * 1024 * 1024)}]}
    assert query_catalog_url(origin + '/catalog', headers)['status'] == 'error'
    state.update(status=302, location=origin + '/stolen')
    assert not query_catalog_url(origin + '/catalog', headers)['ok']
    assert [path for path, _ in state['calls']] == ['/catalog', '/catalog']


def test_explicit_local_keyless_provider_and_manual_model_remain_supported(provider_fixture, tmp_path, monkeypatch):
    from pathlib import Path
    from hermes_cli.web_models import CustomEndpointUpdate, ModelTestRequest
    from hermes_cli.web_routers import config_env, models
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / '.hermes'))
    origin, state = provider_fixture
    state['keyless'] = True
    saved = config_env.upsert_custom_endpoint(CustomEndpointUpdate(name='Local Fixture', base_url=origin + '/v1', api_key=''))
    assert saved['current']['model'] == '' and saved['endpoints'][0]['has_api_key'] is False
    result = asyncio.run(models.test_provider_model(ModelTestRequest(provider=saved['id'], model='manual/local', accept_cost=True)))
    assert result['ok'] and result['model'] == 'manual/local'


def test_remote_provider_cannot_clear_required_credential():
    from fastapi import HTTPException
    from hermes_cli.web_models import CustomEndpointUpdate
    from hermes_cli.web_routers.config_env import _write_custom_endpoint
    with pytest.raises(HTTPException) as denied:
        _write_custom_endpoint({}, CustomEndpointUpdate(name='Remote', base_url='https://api.example.com/v1', api_key=''))
    assert denied.value.status_code == 400


def test_real_profile_catalog_rotates_keys_urls_and_enabled_state(provider_fixture, tmp_path, monkeypatch):
    from pathlib import Path
    from hermes_cli import profiles
    from hermes_cli.web_models import CustomEndpointUpdate
    from hermes_cli.web_routers import config_env, models
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / '.hermes'))
    profiles.create_profile('rotate', no_alias=True, no_skills=True)
    profiles.create_profile('other', no_alias=True, no_skills=True)
    origin, state = provider_fixture
    body = CustomEndpointUpdate(name='Rotation', base_url=origin + '/v1', model_list_url=origin + '/catalog', api_key='fixture-key')
    slug = config_env.upsert_custom_endpoint(body, 'rotate')['id']
    async def catalog(profile='rotate', refresh=False):
        return await models.get_model_options(profile=profile, refresh=refresh, query_only=True, query_scope='fixture-source')
    first = asyncio.run(catalog(refresh=True))['providers'][0]
    assert first['models'] == ['fixture/alpha'] and first['discovery_state'] == 'fresh'
    body.id, body.api_key = slug, 'replacement-key'
    config_env.upsert_custom_endpoint(body, 'rotate')
    changed = asyncio.run(catalog())['providers'][0]
    assert changed['models'] == [] and changed['discovery_state'] == 'missing'
    denied = asyncio.run(catalog(refresh=True))['providers'][0]
    assert denied['models'] == [] and denied['discovery_state'] == 'error'
    body.api_key, body.model_list_url = 'fixture-key', origin + '/new-catalog'
    config_env.upsert_custom_endpoint(body, 'rotate')
    state['models'] = ['only-new-url']
    assert asyncio.run(catalog())['providers'][0]['models'] == []
    assert asyncio.run(catalog(refresh=True))['providers'][0]['models'] == ['only-new-url']
    assert state['calls'][-1][0] == '/new-catalog'
    body.enabled = False
    config_env.upsert_custom_endpoint(body, 'rotate')
    assert asyncio.run(catalog())['providers'] == []
    assert asyncio.run(catalog('other'))['providers'] == []
