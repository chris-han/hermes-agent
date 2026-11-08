"""OpenAI's four connection fields share the existing profile credential owner."""
import json

import pytest
from fastapi.testclient import TestClient
from hermes_cli.web_server import _SESSION_TOKEN, app


@pytest.fixture
def owner(tmp_path, monkeypatch):
    home = tmp_path / 'openai-profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    from hermes_cli.config import invalidate_env_cache
    invalidate_env_cache()
    with TestClient(app) as client:
        yield client, home


def test_openai_card_exposes_four_fields_without_treating_scope_ids_as_api_keys(owner):
    client, _ = owner
    response = client.get('/api/env', headers={'X-Hermes-Session-Token': _SESSION_TOKEN})
    assert response.status_code == 200
    rows = response.json()
    for name in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_ORG_ID', 'OPENAI_PROJECT_ID']:
        assert rows[name]['provider'] == 'openai-api'
        assert rows[name]['provider_label'] == 'OpenAI API'
        assert rows[name]['category'] == 'provider'
        assert rows[name]['is_password'] is (name == 'OPENAI_API_KEY')
    assert rows['OPENAI_ORG_ID']['description'] == 'Organization ID'
    assert rows['OPENAI_PROJECT_ID']['description'] == 'Project ID'


def test_four_openai_parameters_save_reload_and_clear_through_existing_env_owner(owner):
    client, home = owner
    headers = {'X-Hermes-Session-Token': _SESSION_TOKEN}
    values = {'OPENAI_API_KEY': 'fixture-' + 'a' * 24,
              'OPENAI_BASE_URL': 'https://openai.fixture.test/v1',
              'OPENAI_ORG_ID': 'org-fixture', 'OPENAI_PROJECT_ID': 'proj_fixture'}
    for name, value in values.items():
        response = client.put('/api/env', json={'key': name, 'value': value}, headers=headers)
        assert response.status_code == 200
    from dotenv import dotenv_values
    assert all(dotenv_values(home / '.env')[k] == v for k, v in values.items())
    response = client.get('/api/env', headers=headers)
    rows = response.json()
    assert all(rows[k]['is_set'] for k in values)
    assert values['OPENAI_API_KEY'] not in json.dumps(rows)
    response = client.request('DELETE', '/api/env', json={'key': 'OPENAI_PROJECT_ID'}, headers=headers)
    assert response.status_code == 200
    assert not client.get('/api/env', headers=headers).json()['OPENAI_PROJECT_ID']['is_set']
    assert dotenv_values(home / '.env')['OPENAI_API_KEY'] == values['OPENAI_API_KEY']


def test_openai_query_uses_saved_account_scope_and_fences_project_snapshots(owner):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    client, home = owner
    calls = []

    class Supplier(BaseHTTPRequestHandler):
        def do_GET(self):
            scope = (self.headers.get('Authorization'), self.headers.get('OpenAI-Organization'),
                     self.headers.get('OpenAI-Project'))
            calls.append(scope)
            ok = scope[0] == 'Bearer fixture-key' and ((scope[1] == 'org-fixture' and scope[2] in ['proj_a', 'proj_b']) or scope[1:] == (None, None))
            body = json.dumps({'data': [{'id': 'returned-' + (scope[2] or 'optional')}]} if ok else {'error': 'scope mismatch'}).encode()
            self.send_response(200 if ok else 403)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    supplier = ThreadingHTTPServer(('127.0.0.1', 0), Supplier)
    worker = threading.Thread(target=supplier.serve_forever, daemon=True)
    worker.start()
    headers = {'X-Hermes-Session-Token': _SESSION_TOKEN}
    home.joinpath('config.yaml').write_text('model:\n  provider: openai-api\n  default: historical-only\nproviders:\n  openai-api: {}\n')
    values = {'OPENAI_API_KEY': 'fixture-key', 'OPENAI_BASE_URL': f'http://127.0.0.1:{supplier.server_port}/v1',
              'OPENAI_ORG_ID': 'org-fixture', 'OPENAI_PROJECT_ID': 'proj_a'}
    try:
        for key, value in values.items():
            assert client.put('/api/env', json={'key': key, 'value': value}, headers=headers).status_code == 200
        path = '/api/model/options?query_only=1&query_scope=four-parameters'
        response = client.get(path + '&refresh=1', headers=headers)
        assert response.status_code == 200
        row = next(r for r in response.json()['providers'] if r['slug'] == 'openai-api')
        assert row['models'] == ['returned-proj_a']
        assert 'eligible_models' not in row
        source_a = row['discovery_source_id']
        assert client.put('/api/env', json={'key': 'OPENAI_PROJECT_ID', 'value': 'proj_b'}, headers=headers).status_code == 200
        response = client.get(path, headers=headers)
        row = next(r for r in response.json()['providers'] if r['slug'] == 'openai-api')
        assert row['models'] == [] and row['discovery_source_id'] != source_a
        response = client.get(path + '&refresh=1', headers=headers)
        row = next(r for r in response.json()['providers'] if r['slug'] == 'openai-api')
        assert row['models'] == ['returned-proj_b']
        assert calls == [('Bearer fixture-key', 'org-fixture', 'proj_a'), ('Bearer fixture-key', 'org-fixture', 'proj_b')]
        for key in ['OPENAI_ORG_ID', 'OPENAI_PROJECT_ID']:
            assert client.request('DELETE', '/api/env', json={'key': key}, headers=headers).status_code == 200
        response = client.get(path + '&refresh=1', headers=headers)
        row = next(r for r in response.json()['providers'] if r['slug'] == 'openai-api')
        assert row['models'] == ['returned-optional']
        assert calls[-1] == ('Bearer fixture-key', None, None)
        from hermes_cli.config import save_config
        save_config({'providers': {'compatible-fixture': {'base_url': values['OPENAI_BASE_URL'], 'key_env': 'OPENAI_API_KEY'}}})
        for key, value in [('OPENAI_ORG_ID', 'org-private'), ('OPENAI_PROJECT_ID', 'proj_private')]:
            assert client.put('/api/env', json={'key': key, 'value': value}, headers=headers).status_code == 200
        response = client.get(path + '&refresh=1', headers=headers)
        row = next(r for r in response.json()['providers'] if r['slug'] == 'compatible-fixture')
        assert row['models'] == ['returned-optional']
        assert calls[-1] == ('Bearer fixture-key', None, None)
    finally:
        supplier.shutdown()
        supplier.server_close()
        worker.join(timeout=5)


def test_openai_scope_fields_do_not_leak_between_selected_native_profiles(owner, monkeypatch):
    client, home = owner
    headers = {'X-Hermes-Session-Token': _SESSION_TOKEN}
    secondary = home / 'profiles' / 'secondary'
    secondary.mkdir(parents=True)
    (secondary / 'config.yaml').write_text('model: {}\n')
    (secondary / '.env').write_text('')
    monkeypatch.setenv('OPENAI_ORG_ID', 'org-launch')
    assert client.put('/api/env', json={'key': 'OPENAI_ORG_ID', 'value': 'org-primary'}, headers=headers).status_code == 200
    assert not client.get('/api/env?profile=secondary', headers=headers).json()['OPENAI_ORG_ID']['is_set']
    assert client.put('/api/env?profile=secondary', json={'key': 'OPENAI_ORG_ID', 'value': 'org-secondary'}, headers=headers).status_code == 200
    from dotenv import dotenv_values
    assert dotenv_values(home / '.env')['OPENAI_ORG_ID'] == 'org-primary'
    assert dotenv_values(secondary / '.env')['OPENAI_ORG_ID'] == 'org-secondary'
    assert client.request('DELETE', '/api/env?profile=secondary', json={'key': 'OPENAI_ORG_ID'}, headers=headers).status_code == 200
    assert dotenv_values(home / '.env')['OPENAI_ORG_ID'] == 'org-primary'
    assert not client.get('/api/env?profile=secondary', headers=headers).json()['OPENAI_ORG_ID']['is_set']
