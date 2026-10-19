"""Explicit profile-scoped gateway routing and correlation; no inferred billing identity."""
import os
from pathlib import Path

import pytest
import yaml

from agent import semantier_gateway as sg


def configure(tmp_path, monkeypatch, **overrides):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    config = {'billing': {'backend': 'semantier'}, 'semantier_gateway': {
        'enabled': True, 'base_url': 'http://127.0.0.1:8765/v1', 'api_key_env': 'TEST_GATEWAY_TOKEN'}}
    config.update(overrides)
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(config))
    (tmp_path / '.env').write_text('TEST_GATEWAY_TOKEN=profile-secret\n')
    return config


def test_explicit_profile_config_and_no_ambient_mutation(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    before = dict(os.environ)
    settings = sg.gateway_settings()
    assert settings.base_url == 'http://127.0.0.1:8765/v1'
    assert settings.api_key == 'profile-secret'
    assert sg.billing_backend() == 'semantier'
    assert dict(os.environ) == before
    (tmp_path / 'config.yaml').write_text('billing: {backend: nous}\n')
    assert sg.billing_backend() == 'nous'


def test_correlation_only_on_exact_configured_route(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    kwargs = {'extra_headers': {'X-Existing': 'ok'}}
    tagged = sg.attribute_request(kwargs, base_url='http://127.0.0.1:8765/v1/',
                                 session_id='s', run_id='r', purpose='chat')
    assert tagged['extra_headers'].pop('Idempotency-Key')
    assert tagged['extra_headers'] == {'X-Existing': 'ok', 'X-Semantier-Session-Id': 's',
                                      'X-Semantier-Run-Id': 'r', 'X-Semantier-Purpose': 'chat'}
    assert kwargs == {'extra_headers': {'X-Existing': 'ok'}}
    assert sg.attribute_request(kwargs, base_url='https://api.openai.com/v1', purpose='chat') is kwargs
    with pytest.raises(ValueError):
        sg.attribute_request(kwargs, base_url='http://127.0.0.1:8765/v1', purpose='title\r\nevil')


def test_invalid_or_disabled_semantier_never_falls_back_to_nous(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, semantier_gateway={'enabled': False})
    assert sg.billing_backend() == 'semantier'
    result = sg.billing_request('state', schema_version='semantier.billing.v1')
    assert result['ok'] is False and result['backend'] == 'semantier'
    assert result['error'] == 'provider_not_configured'
    assert 'portal_url' not in result


def test_billing_http_contract_and_safe_error(tmp_path, monkeypatch):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    payload = {'schema_version': sg.SCHEMA_VERSION, 'backend': 'semantier',
               'organization': {'id': 'org-a'}, 'billing_account_id': 'ba-a', 'mode': 'shadow',
               'funding_source': 'CUSTOMER_CLOUD', 'balances': [],
               'capabilities': {'can_read_usage': True, 'can_charge': False,
                                'can_change_plan': False, 'can_auto_top_up': False},
               'account': {'available': False},
               'usage': {'known_cost': {'amount': '0.012000', 'currency': 'CNY'},
                         'status': 'ESTIMATED', 'unresolved_attempts': 0, 'period': 'lifetime'},
               'as_of': '2026-10-19T00:00:00Z'}
    calls = []
    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get('Authorization')))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        configure(tmp_path, monkeypatch, semantier_gateway={
            'enabled': True, 'base_url': f'http://127.0.0.1:{server.server_port}/v1',
            'api_key_env': 'TEST_GATEWAY_TOKEN'})
        assert sg.billing_request('state')['error'] == 'unsupported_version'
        assert calls == []
        result = sg.read_billing('state', schema_version=sg.SCHEMA_VERSION)
        assert result == {'ok': True, **payload}
        assert 'balance_usd' not in result and result['balances'] == []
        assert calls == [('/api/billing/state', 'Bearer profile-secret')]
        payload['usage']['known_cost']['amount'] = 0.012
        result = sg.read_billing('state', schema_version=sg.SCHEMA_VERSION)
        assert result['ok'] is False and result['error'] == 'billing_unavailable'
        assert 'profile-secret' not in str(result)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_hosted_route_cannot_escape_to_a_fallback(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agent.chat_completion_helpers import try_activate_fallback
    configure(tmp_path, monkeypatch)
    agent = SimpleNamespace(base_url='http://127.0.0.1:8765/v1')
    assert try_activate_fallback(agent) is False


@pytest.mark.parametrize('purpose', ['compression', 'memory', 'vision', 'plugin', 'subagent', 'cron'])
def test_unsupported_hosted_callers_fail_before_dispatch(tmp_path, monkeypatch, purpose):
    configure(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='supports primary chat and title generation only'):
        sg.attribute_request({}, base_url='http://127.0.0.1:8765/v1', purpose=purpose)
    assert sg.attribute_request({}, base_url='https://direct.example/v1', purpose=purpose) == {}


def test_idempotency_is_stable_per_logical_call_and_unique_across_calls(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    route = dict(base_url='http://127.0.0.1:8765/v1', purpose='chat', session_id='s')
    first = sg.attribute_request({}, **route, request_id='turn:api:1')
    retry = sg.attribute_request({}, **route, request_id='turn:api:1')
    next_call = sg.attribute_request({}, **route, request_id='turn:api:2')
    assert first['extra_headers']['Idempotency-Key'] == retry['extra_headers']['Idempotency-Key']
    assert first['extra_headers']['Idempotency-Key'] != next_call['extra_headers']['Idempotency-Key']
    helper = sg.attribute_request({}, base_url=route['base_url'], purpose='title_generation')
    helper_retry = sg.attribute_request(helper, base_url=route['base_url'], purpose='title_generation')
    other_helper = sg.attribute_request({}, base_url=route['base_url'], purpose='title_generation')
    assert helper['extra_headers']['Idempotency-Key'] == helper_retry['extra_headers']['Idempotency-Key']
    assert helper['extra_headers']['Idempotency-Key'] != other_helper['extra_headers']['Idempotency-Key']
