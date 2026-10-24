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


def test_auto_title_snapshots_native_run_correlation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agent.turn_context import _maybe_title_session_at_turn_start

    configure(tmp_path, monkeypatch)
    snapshots = []
    monkeypatch.setattr('agent.title_generator.maybe_auto_title',
                        lambda *args, **kwargs: snapshots.append(kwargs['main_runtime']))
    owner = SimpleNamespace(platform='desktop', session_id='session-owned',
                            _current_task_id='task-owned', run_id='unrelated-value',
                            model='gpt-5.1', provider='custom', api_key='profile-secret',
                            base_url='http://127.0.0.1:8765/v1', api_mode='chat_completions',
                            _session_db=object(), _session_db_created=True)
    sg.bind_task_context(owner,'task-owned','run-owned')
    _maybe_title_session_at_turn_start(owner, [{'role': 'user', 'content': 'Read evidence'}])
    owner._current_task_id = 'next-task'
    snapshot = snapshots[0]
    headers = sg.attribute_request({}, base_url=snapshot['base_url'], purpose='title_generation',
                                   session_id=snapshot['session_id'],
                                   **{k:snapshot[k] for k in ('task_id','root_run_id','parent_run_id','run_id')})['extra_headers']
    assert headers['X-Semantier-Session-Id'] == 'session-owned'
    assert headers['X-Semantier-Run-Id'].startswith('title-')
    assert headers['X-Semantier-Parent-Run-Id'] == 'run-owned'
    assert headers['X-Semantier-Task-Id'] == headers['X-Semantier-Root-Run-Id'] == 'task-owned'
    assert headers['X-Semantier-Purpose'] == 'title'


def test_cloud_model_options_matches_native_wire_contract(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    from hermes_cli.inventory import ConfigContext, build_model_options_payload
    from tui_gateway.contracts.config_free_tier_control import ModelOptionsResult
    monkeypatch.setattr(sg, 'cloud_catalog', lambda **kwargs: {'data': []})
    payload = build_model_options_payload(ConfigContext('semantier-cloud', '',
        'http://127.0.0.1:8765/v1', {}, []))
    result = ModelOptionsResult.model_validate(payload)
    assert result.provider == 'semantier-cloud'


def test_task_root_and_child_lineage_are_runtime_owned():
    from types import SimpleNamespace
    parent = SimpleNamespace()
    first = sg.bind_task_context(parent, 'task-1', 'run-1')
    assert first == {'task_id':'task-1','root_run_id':'task-1','run_id':'run-1'}
    assert sg.bind_task_context(parent,'task-1','run-2')['root_run_id'] == first['root_run_id']
    child = SimpleNamespace(_semantier_parent_task_context=first)
    assert sg.bind_task_context(child,'child-task','child-run') == {
        'task_id':'task-1','root_run_id':'task-1','run_id':'child-run','parent_run_id':'run-1'}
    assert sg.bind_task_context(parent,'other-task','run-3')['root_run_id'] != first['root_run_id']


def test_auxiliary_runtime_receives_current_root_not_a_global_mirror(monkeypatch):
    from types import SimpleNamespace
    from agent.turn_context import _publish_runtime_main
    from agent.auxiliary_client import _normalize_main_runtime, reset_runtime_main, set_runtime_main
    monkeypatch.setattr('agent.prompt_cache_scope.resolve_prompt_cache_scope_safe',lambda agent:'scope')
    token=set_runtime_main('custom','old',session_id='foreign-session',task_id='foreign-task')
    try:
        owner=SimpleNamespace(provider='custom',model='model',base_url='https://fixture.invalid/v1',api_key='key',session_id='session')
        sg.bind_task_context(owner,'task-owned','run-owned')
        _publish_runtime_main(owner)
        snapshot=_normalize_main_runtime(None)
        assert snapshot['task_id']==snapshot['root_run_id']=='task-owned'
        assert snapshot['run_id']=='run-owned' and snapshot['session_id']=='session'
    finally:
        reset_runtime_main(token)


def test_captured_cloud_route_survives_current_profile_reconfiguration(tmp_path, monkeypatch):
    from types import SimpleNamespace
    configure(tmp_path,monkeypatch)
    agent=SimpleNamespace(base_url='http://127.0.0.1:8765/v1',api_key='profile-secret')
    sg.capture_agent_route(agent)
    bound=sg.agent_cloud_settings(agent)
    (tmp_path/'config.yaml').write_text('model: {provider: custom, base_url: http://127.0.0.1:9999/v1}\n')
    assert not sg.is_gateway_route(agent.base_url)
    result=sg.attribute_request({},base_url=agent.base_url,purpose='chat',task_id='t',root_run_id='root',run_id='run',settings=bound)
    assert result['extra_headers']['X-Semantier-Root-Run-Id']=='root'
    seen=[]
    profile={'context_window':128000,'max_output_tokens':128,'capabilities':{},'request_parameters':{'output':'max_tokens','reasoning_controls':[]},'revision':'account-a-revision','account_id':'account-a'}
    monkeypatch.setattr(sg,'cloud_model_profile',lambda model,**kwargs: seen.append(kwargs['settings']) or profile)
    compiled=sg.configure_cloud_request({'model':'alias','messages':[]},base_url=agent.base_url,settings=bound)
    assert seen==[bound]
    assert compiled['extra_headers']['X-Semantier-Account-Id']=='account-a'
    agent.base_url='http://127.0.0.1:9999/v1'
    assert sg.agent_cloud_settings(agent) is None


def test_real_title_response_format_uses_cloud_capability_compiler(tmp_path,monkeypatch):
    from agent.title_generator import _TITLE_RESPONSE_FORMAT
    configure(tmp_path,monkeypatch)
    profile={'context_window':128000,'max_output_tokens':128,'capabilities':{'structured_output':False},
        'request_parameters':{'output':'max_tokens','reasoning_controls':[]},'revision':'title-revision'}
    monkeypatch.setattr(sg,'cloud_model_profile',lambda *args,**kwargs:profile)
    wire=sg.configure_cloud_request({'model':'title-model','messages':[{'role':'user','content':'Title fixture'}],
        'extra_body':{'response_format':_TITLE_RESPONSE_FORMAT}},base_url='http://127.0.0.1:8765/v1')
    assert 'response_format' not in wire and 'extra_body' not in wire
    profile['capabilities']['structured_output']=True
    structured=sg.configure_cloud_request({'model':'title-model','messages':[{'role':'user','content':'Title fixture'}],
        'extra_body':{'response_format':_TITLE_RESPONSE_FORMAT}},base_url='http://127.0.0.1:8765/v1')
    assert structured['response_format']==_TITLE_RESPONSE_FORMAT


def test_unconnected_cloud_model_options_remain_available_for_first_onboarding(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch)
    from hermes_cli.inventory import ConfigContext, build_model_options_payload
    from tui_gateway.contracts.config_free_tier_control import ModelOptionsResult
    def absent(**kwargs):
        raise ValueError('Semantier Cloud Provider scoped credential is unavailable for this profile.')
    monkeypatch.setattr(sg,'gateway_settings',absent)
    payload=build_model_options_payload(ConfigContext('semantier-cloud','',
        'http://127.0.0.1:8765/v1',{},[]))
    result=ModelOptionsResult.model_validate(payload)
    assert result.providers[0].authenticated is False
    assert result.providers[0].models==[]
