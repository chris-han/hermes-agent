"""Native device lifecycle against the frozen Cloud v1.0.1 external wire seam."""
import json
import pytest
from cryptography.fernet import Fernet

ACTOR = {'user_id': 'user-test', 'workspace_id': 'workspace-test', 'organization_id': 'org-test'}
ACCOUNT_ORIGIN = 'http://127.0.0.1:8765'


@pytest.mark.parametrize(('model', 'allowed'), [
    (None, True), ('', True), ({}, True), ({'provider': 'semantier-cloud'}, True),
    ('local-model', False), ({'provider': 'custom:byok'}, False),
    ({'provider': 'semantier-cloud', 'default': 'chosen-model'}, False),
    ({'base_url': 'http://127.0.0.1:1234'}, False), (42, False),
])
def test_account_default_never_overrides_saved_model_or_endpoint(native, model, allowed):
    from hermes_cli.config import atomic_config_replace
    device, home = native
    config = device._config()
    if model is None:
        config.pop('model', None)
    else:
        config['model'] = model
    atomic_config_replace(home / 'config.yaml', config)
    assert device.account_default_eligibility() == {'allowed': allowed}

def test_first_account_fence_creates_owner_root(native, monkeypatch, tmp_path):
    import hermes_constants
    device, _ = native
    root = tmp_path / 'new-owner'
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda: root)
    device.set_account_fence(None)
    assert device._read_account_fence() == {'version': 1, 'account': None}

@pytest.fixture
def native(tmp_path, monkeypatch):
    from hermes_cli import cloud_device_client as device
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(str(tmp_path))
    (tmp_path/'config.yaml').write_text('model:\n  provider: custom\n  base_url: http://127.0.0.1:8765/v1\nproviders:\n  semantier-cloud:\n    base_url: http://127.0.0.1:8765/v1\n')
    cipher = Fernet(Fernet.generate_key())
    monkeypatch.setattr(device, '_protect', cipher.encrypt)
    monkeypatch.setattr(device, '_unprotect', cipher.decrypt)
    yield device, tmp_path
    reset_hermes_home_override(token)


def test_start_recovers_private_intent_without_renderer_secrets(native, monkeypatch):
    device, home = native
    calls = []
    def wire(origin, operation, body=None, bearer=None):
        calls.append((operation, body))
        return {'contract_version': device.CONTRACT, 'intent_ref': 'intent-test', 'status':'pending',
                'expires_at':'2099-01-01T00:00:00+00:00','retry_after':5,
                'data':{'verification_uri':origin+'/billing/connect?intent_ref=intent-test'}}
    monkeypatch.setattr(device, '_wire', wire)
    first = device.start()
    second = device.start()
    assert first == second
    assert len(calls) == 1
    private = device._read()
    assert 'proof' not in json.dumps(first)
    assert private['proof'] not in (home/'.cloud-device.bin').read_bytes().decode(errors='ignore')
    assert calls[0][1]['possession_challenge'] == device._digest(private['proof'].encode())


def test_first_connect_without_config_preserves_absence_and_reuses_intent(native, monkeypatch):
    device, home = native
    config = home/'config.yaml'
    config.unlink()
    calls = install_wire(device, monkeypatch)
    first = device.start()
    assert device.start() == first
    assert calls == ['start']
    assert not config.exists()
    assert 'proof' not in json.dumps(first)


def test_creating_config_after_first_connect_invalidates_only_that_intent(native, monkeypatch):
    device, home = native
    config = home/'config.yaml'
    config.unlink()
    calls = install_wire(device, monkeypatch)
    started = device.start()
    config.write_bytes(b'')
    assert device.poll(started['session_id'])['status'] == 'cancelled'
    assert config.read_bytes() == b''
    assert not any(call.endswith('/claim') for call in calls)


def test_first_connect_can_claim_and_ack_before_any_preferences_exist(native, monkeypatch):
    device, home = native
    config = home/'config.yaml'
    config.unlink()
    calls = install_wire(device, monkeypatch)
    started = device.start()
    assert not config.exists()
    assert device.poll(started['session_id'])['status'] == 'approved'
    assert calls.count('intent-test/claim') == 1
    assert calls.count('intent-test/ack') == 1
    assert 'private-account-token' not in config.read_text()


@pytest.mark.parametrize('model', [None, 'local-model', {'provider': 'custom', 'base_url': 'http://127.0.0.1:8765/v1'}])
def test_account_preparation_installs_scoped_credential_without_selecting_model(native, monkeypatch, model):
    from hermes_cli.config import atomic_config_replace
    device, home = native
    config = device._config()
    if model is None:
        config.pop('model', None)
    else:
        config['model'] = model
    atomic_config_replace(home/'config.yaml', config)
    install_wire(device, monkeypatch)
    started = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    assert device.poll(started['session_id'])['status'] == 'approved'
    installed = device._config()
    assert installed.get('model') == model
    assert installed['providers']['semantier-cloud']['credential_ref'] == 'token-test'


def test_account_preparation_does_not_adopt_an_existing_manual_intent(native, monkeypatch):
    device, _ = native
    install_wire(device, monkeypatch)
    manual = device.start()
    with pytest.raises(device.CloudConnectionError, match='already in progress'):
        device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    assert device._read()['session_id'] == manual['session_id']


def test_manual_connection_does_not_adopt_an_existing_account_preparation(native, monkeypatch):
    device, _ = native
    install_wire(device, monkeypatch)
    preparation = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    with pytest.raises(device.CloudConnectionError, match='already in progress'):
        device.start()
    assert device._read()['session_id'] == preparation['session_id']


def test_account_preparation_ack_recovery_preserves_model(native, monkeypatch):
    device, _ = native
    before = device._config()['model'].copy()
    calls = install_wire(device, monkeypatch, ack_lost=True)
    started = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    with pytest.raises(ValueError, match='lost ack'):
        device.poll(started['session_id'])
    recovered = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    assert recovered['session_id'] == started['session_id']
    assert recovered['verification_url'] == started['verification_url']
    assert device.poll(started['session_id'])['status'] == 'approved'
    assert calls.count('intent-test/claim') == 1
    assert device._config()['model'] == before


def test_account_preparation_refuses_changed_actor_and_foreign_claim(native, monkeypatch):
    device, home = native
    install_wire(device, monkeypatch)
    started = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    with pytest.raises(device.CloudConnectionError, match='already in progress'):
        device.start(prepare_only=True, expected_actor={**ACTOR, 'user_id': 'other-user'}, expected_origin=ACCOUNT_ORIGIN)
    original = device._wire
    def foreign(origin, operation, body=None, bearer=None):
        reply = original(origin, operation, body, bearer)
        if operation.endswith('/claim'):
            reply['data']['user_id'] = 'other-user'
        return reply
    monkeypatch.setattr(device, '_wire', foreign)
    before = (home/'config.yaml').read_bytes()
    with pytest.raises(device.CloudConnectionError, match='account binding mismatch'):
        device.poll(started['session_id'])
    assert (home/'config.yaml').read_bytes() == before
    assert not device._read().get('credential_ref')


def test_account_preparation_requires_complete_expected_actor(native, monkeypatch):
    device, _ = native
    install_wire(device, monkeypatch)
    for expected in (None, {'user_id': 'u'}, {**ACTOR, 'extra': 'x'}):
        with pytest.raises(device.CloudConnectionError):
            device.start(prepare_only=True, expected_actor=expected, expected_origin=ACCOUNT_ORIGIN)


def test_account_preparation_uses_selected_origin_without_changing_model(native, monkeypatch):
    device, _ = native
    before = device._config()['model'].copy()
    install_wire(device, monkeypatch)
    started = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin='http://127.0.0.1:8767')
    assert started['verification_url'].startswith('http://127.0.0.1:8767/')
    assert device.poll(started['session_id'])['status'] == 'approved'
    assert device._config()['model'] == before
    assert device._config()['providers']['semantier-cloud']['base_url'] == 'http://127.0.0.1:8767/v1'


def test_account_preparation_sends_expected_actor_and_rejects_missing_recovery_scope(native, monkeypatch):
    device, home = native
    install_wire(device, monkeypatch)
    original = device._wire
    captured = []
    def wire(origin, operation, body=None, bearer=None):
        if operation == 'start':
            captured.append(body)
        return original(origin, operation, body, bearer)
    monkeypatch.setattr(device, '_wire', wire)
    started = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    assert captured[0]['expected_actor'] == ACTOR
    state = device._read()
    state.pop('expected_actor')
    device._write(state)
    before = (home/'config.yaml').read_bytes()
    with pytest.raises(device.CloudConnectionError, match='account binding mismatch'):
        device.poll(started['session_id'])
    assert (home/'config.yaml').read_bytes() == before


def test_account_preparation_persists_one_browser_approval_attempt(native, monkeypatch):
    device, _ = native
    install_wire(device, monkeypatch)
    started = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    assert device.mark_browser_approval(started['session_id']) == {'first': True}
    assert device.mark_browser_approval(started['session_id']) == {'first': False}
    assert device._read()['approval_attempted'] is True
    with pytest.raises(device.CloudConnectionError):
        device.mark_browser_approval('cloud-foreign')


def test_account_prepared_credential_requires_matching_device_fence(native, monkeypatch):
    from hermes_cli import cloud_device_client as device
    import hermes_constants
    device, home = native
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda **_: home)
    calls = install_wire(device, monkeypatch)
    started = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    assert device.poll(started['session_id'])['status'] == 'approved'
    base = 'http://127.0.0.1:8765/v1'
    with pytest.raises(device.CloudConnectionError, match='account'):
        device.credential('token-test', base_url=base)
    blocked = device.status()
    assert blocked['account_id'] is None and blocked['account_context_unavailable'] is True
    assert blocked['reauthorization_required'] is False and 'readiness' not in calls
    device.set_account_fence({'origin': 'http://127.0.0.1:8765', **ACTOR})
    assert device.credential('token-test', base_url=base) == 'private-account-token'
    device.set_account_fence(None)
    with pytest.raises(device.CloudConnectionError, match='account'):
        device.credential('token-test', base_url=base)
    assert device.status()['readiness'] is None
    device.set_account_fence({'origin': 'http://127.0.0.1:8765', **{**ACTOR, 'user_id': 'other-user'}})
    with pytest.raises(device.CloudConnectionError, match='account'):
        device.credential('token-test', base_url=base)


def test_existing_manual_credential_is_suspended_without_deleting_it(native, monkeypatch):
    import hermes_constants
    device, home = native
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda **_: home)
    install_wire(device, monkeypatch)
    started = device.start()
    device.poll(started['session_id'])
    base = 'http://127.0.0.1:8765/v1'
    assert device.credential('token-test', base_url=base) == 'private-account-token'
    original = device._read()['credentials']
    device.set_account_fence(None)
    with pytest.raises(device.CloudConnectionError, match='account'):
        device.credential('token-test', base_url=base)
    assert device._read()['credentials'] == original
    device.set_account_fence({'origin': 'http://127.0.0.1:8765', **ACTOR})
    with pytest.raises(device.CloudConnectionError, match='account'):
        device.credential('token-test', base_url=base)


def test_corrupt_account_fence_fails_closed(native, monkeypatch):
    import hermes_constants
    device, home = native
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda **_: home)
    install_wire(device, monkeypatch)
    started = device.start()
    device.poll(started['session_id'])
    (home/'.cloud-account-fence.bin').write_bytes(b'not-protected')
    with pytest.raises(device.CloudConnectionError, match='account'):
        device.credential('token-test', base_url='http://127.0.0.1:8765/v1')


def test_account_fence_local_api_is_authenticated_and_never_returns_credentials(native, monkeypatch):
    import hermes_constants
    from fastapi.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_TOKEN
    device, home = native
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda **_: home)
    client = TestClient(app)
    path = '/api/providers/oauth/semantier-cloud/account-context'
    assert client.put(path, json={'account': None}).status_code == 401
    headers = {'X-Hermes-Session-Token': _SESSION_TOKEN}
    response = client.put(path, headers=headers, json={'account': {'origin': 'http://127.0.0.1:8765', **ACTOR}})
    assert response.status_code == 200 and response.json() == {'ok': True}
    assert b'user-test' not in (home/'.cloud-account-fence.bin').read_bytes()
    assert client.put(path, headers=headers, json={'account': None}).json() == {'ok': True}
    assert device._read_account_fence()['account'] is None


def test_account_default_eligibility_api_requires_native_session_and_returns_no_config(native):
    from fastapi.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_TOKEN
    client = TestClient(app)
    path = '/api/providers/oauth/semantier-cloud/account-default-eligibility'
    assert client.get(path).status_code == 401
    response = client.get(path, headers={'X-Hermes-Session-Token': _SESSION_TOKEN})
    assert response.status_code == 200
    assert response.json() == {'allowed': False}


def test_account_fence_applies_across_local_profiles(native, monkeypatch):
    import hermes_constants
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    device, home = native
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda **_: home)
    install_wire(device, monkeypatch)
    initial_config = (home/'config.yaml').read_text()
    start = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    device.poll(start['session_id'])
    second = home/'profiles'/'second'
    second.mkdir(parents=True)
    (second/'config.yaml').write_text(initial_config)
    token = set_hermes_home_override(str(second))
    try:
        start = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
        device.poll(start['session_id'])
        device.set_account_fence({'origin': 'http://127.0.0.1:8765', **ACTOR})
        assert device.credential('token-test', base_url='http://127.0.0.1:8765/v1') == 'private-account-token'
        device.set_account_fence(None)
        with pytest.raises(device.CloudConnectionError, match='account'):
            device.credential('token-test', base_url='http://127.0.0.1:8765/v1')
    finally:
        reset_hermes_home_override(token)
    with pytest.raises(device.CloudConnectionError, match='account'):
        device.credential('token-test', base_url='http://127.0.0.1:8765/v1')


def test_account_switch_during_readiness_does_not_publish_old_account(native, monkeypatch):
    import hermes_constants
    device, home = native
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda **_: home)
    install_wire(device, monkeypatch)
    started = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    device.poll(started['session_id'])
    device.set_account_fence({'origin': 'http://127.0.0.1:8765', **ACTOR})
    state = device._read()
    state.pop('next_readiness', None)
    device._write(state)
    original = device._wire
    def wire(origin, operation, body=None, bearer=None):
        if operation == 'readiness':
            device.set_account_fence(None)
            return {'account_id': 'account-test', 'currency': 'CNY', 'catalog': {'data': ['old-account-model']}}
        return original(origin, operation, body, bearer)
    monkeypatch.setattr(device, '_wire', wire)
    result = device.status()
    assert result['account_context_unavailable'] is True
    assert result['account_id'] is None and result['readiness'] is None


@pytest.mark.parametrize('model', [
    {'provider': 'custom', 'base_url': 'http://127.0.0.1:8765/v1'},
    {},
    {'provider': 'local', 'default': 'local-model'},
])
def test_captured_agent_route_rechecks_account_fence_before_later_requests(native, monkeypatch, model):
    import hermes_constants
    from types import SimpleNamespace
    from agent import semantier_gateway
    device, home = native
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda **_: home)
    install_wire(device, monkeypatch)
    started = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    device.poll(started['session_id'])
    device.set_account_fence({'origin': 'http://127.0.0.1:8765', **ACTOR})
    from hermes_cli.config import atomic_config_replace
    config = device._config()
    config['model'] = model
    atomic_config_replace(home / 'config.yaml', config)
    agent = SimpleNamespace(base_url='http://127.0.0.1:8765/v1', api_key='private-account-token')
    semantier_gateway.capture_agent_route(agent)
    assert semantier_gateway.agent_cloud_settings(agent).api_key == agent.api_key
    device.set_account_fence(None)
    with pytest.raises(device.CloudConnectionError, match='account'):
        semantier_gateway.agent_cloud_settings(agent)


def test_queued_cloud_title_revalidates_account_fence(native, monkeypatch):
    import hermes_constants
    from types import SimpleNamespace
    from agent import semantier_gateway
    from agent.turn_context import _maybe_title_session_at_turn_start
    device, home = native
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda **_: home)
    install_wire(device, monkeypatch)
    started = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    device.poll(started['session_id'])
    device.set_account_fence({'origin': ACCOUNT_ORIGIN, **ACTOR})
    agent = SimpleNamespace(platform='desktop', session_id='fixture', model='fixture-model', provider='custom',
                            base_url=ACCOUNT_ORIGIN + '/v1', api_key='private-account-token', api_mode='chat_completions',
                            _session_db=object(), _session_db_created=True)
    semantier_gateway.capture_agent_route(agent)
    validators = []
    monkeypatch.setattr('agent.title_generator.maybe_auto_title',
                        lambda *args, **kwargs: validators.append(kwargs['runtime_validator']))
    _maybe_title_session_at_turn_start(agent, [{'role': 'user', 'content': 'fixture'}])
    assert validators[0]() is True
    device.set_account_fence(None)
    assert validators[0]() is False


@pytest.mark.parametrize('legacy_base', ['http://127.0.0.1:9999/v1', ACCOUNT_ORIGIN + '/v1'])
def test_account_agent_is_fenced_when_legacy_gateway_coexists(native, monkeypatch, legacy_base):
    import hermes_constants
    from types import SimpleNamespace
    from agent import semantier_gateway
    from hermes_cli.config import atomic_config_replace
    device, home = native
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda **_: home)
    install_wire(device, monkeypatch)
    started = device.start(prepare_only=True, expected_actor=ACTOR, expected_origin=ACCOUNT_ORIGIN)
    device.poll(started['session_id'])
    device.set_account_fence({'origin': ACCOUNT_ORIGIN, **ACTOR})
    config = device._config()
    config['model'] = {'provider': 'local', 'default': 'local-model'}
    config['semantier_gateway'] = {'enabled': True, 'base_url': legacy_base, 'api_key': 'legacy-secret'}
    atomic_config_replace(home / 'config.yaml', config)
    account_agent = SimpleNamespace(base_url=ACCOUNT_ORIGIN + '/v1', api_key='private-account-token')
    semantier_gateway.capture_agent_route(account_agent)
    assert semantier_gateway.agent_cloud_settings(account_agent) is not None
    legacy_agent = SimpleNamespace(base_url=legacy_base, api_key='legacy-secret')
    semantier_gateway.capture_agent_route(legacy_agent)
    device.set_account_fence(None)
    with pytest.raises(device.CloudConnectionError, match='account'):
        semantier_gateway.agent_cloud_settings(account_agent)
    assert semantier_gateway.agent_cloud_settings(legacy_agent).api_key == 'legacy-secret'


def test_cancel_persists_terminal_before_lost_remote_reply(native, monkeypatch):
    device, _ = native
    monkeypatch.setattr(device, '_wire', lambda origin, operation, body=None, bearer=None: {
        'contract_version':device.CONTRACT,'intent_ref':'intent-test','status':'pending',
        'expires_at':'2099-01-01T00:00:00+00:00','data':{'verification_uri':origin+'/billing/connect?intent_ref=intent-test'}})
    started = device.start()
    def unavailable(*args, **kwargs):
        raise ValueError('unavailable')
    monkeypatch.setattr(device, '_wire', unavailable)
    assert device.cancel(started['session_id'])['remote_cancelled'] is False
    assert device.poll(started['session_id'])['status'] == 'cancelled'
    assert device._read()['status'] == 'cancelled'


def test_foreign_session_and_reconfigured_profile_refuse(native, monkeypatch):
    device, home = native
    monkeypatch.setattr(device, '_wire', lambda origin, operation, body=None, bearer=None: {
        'contract_version':device.CONTRACT,'intent_ref':'intent-test','status':'pending',
        'expires_at':'2099-01-01T00:00:00+00:00','data':{'verification_uri':origin+'/billing/connect?intent_ref=intent-test'}})
    started = device.start()
    with pytest.raises(ValueError, match='session'):
        device.poll('foreign')
    (home/'config.yaml').write_text('model: {provider: nous}\n')
    assert device.poll(started['session_id'])['status'] == 'cancelled'


def install_wire(device, monkeypatch, *, ack_lost=False):
    calls = []
    def wire(origin, operation, body=None, bearer=None):
        calls.append(operation)
        state = device._read()
        binding = {k:state[k] for k in ('runtime_ref','profile_ref','generation')}
        binding.update(account_id='account-test',currency='CNY')
        data = {}
        status = 'pending'
        if operation == 'start':
            data = {'verification_uri':origin+'/billing/connect?intent_ref=intent-test'}
        elif operation.endswith('/poll'):
            status, data = 'approved', binding
        elif operation.endswith('/claim'):
            status = 'claimed'
            data = {**binding, 'organization_id':'org-test', 'claim_id':state['claim_id'],
                'credential':{'token_id':'token-test','secret':'private-account-token',
                              'expires_at':'2099-02-01T00:00:00+00:00'},
                'inference_origin':origin,'billing_origin':origin,
                'catalog_path':'/api/cloud-provider/catalog',
                'readiness_path':'/api/cloud-provider/device/readiness',
                'installation_expires_at':'2099-01-01T00:00:00+00:00',
                'renewal':'reauthorization_required'}
            if state.get('prepare_only'):
                data.update(ACTOR)
        elif operation.endswith('/ack'):
            if ack_lost and calls.count(operation) == 1:
                raise ValueError('lost ack')
            status = 'installed'
            data = {'claim_id':state['claim_id'],'installed_generation':state['generation']}
        return {'contract_version':device.CONTRACT,'intent_ref':'intent-test',
                'expires_at':'2099-01-01T00:00:00+00:00','status':status,'data':data}
    monkeypatch.setattr(device,'_wire',wire)
    monkeypatch.setattr(device,'_catalog',lambda *_:[{'id':'allowed-model','revision':'allowed-revision'}])
    return calls


def test_install_and_ack_loss_recovers_same_generation_without_secret_config(native, monkeypatch):
    from agent.semantier_gateway import gateway_settings
    device, home = native
    calls = install_wire(device,monkeypatch,ack_lost=True)
    start = device.start()
    with pytest.raises(ValueError,match='lost ack'):
        device.poll(start['session_id'])
    assert device._read()['status'] == 'installing'
    # A new read of the disk journal (same as restart) retries ack, not claim.
    result = device.poll(start['session_id'])
    assert result['status'] == 'approved'
    assert calls.count('intent-test/claim') == 1
    assert gateway_settings().api_key == 'private-account-token'
    raw = (home/'config.yaml').read_text()
    assert 'private-account-token' not in raw
    assert 'credential_ref: token-test' in raw
    assert not device._config()['model'].get('default')
    assert 'private-account-token' not in json.dumps(result)


def test_post_publication_pre_journal_crash_recovers(native, monkeypatch):
    device, _ = native
    calls = install_wire(device,monkeypatch,ack_lost=True)
    session = device.start()['session_id']
    with pytest.raises(ValueError):
        device.poll(session)
    state = device._read()
    state.pop('committed_hash')
    device._write(state)
    assert device.poll(session)['status'] == 'approved'
    assert calls.count('intent-test/claim') == 1


def test_cancel_after_commit_blocks_local_credential_even_if_remote_unavailable(native, monkeypatch):
    device, _ = native
    install_wire(device,monkeypatch)
    session = device.start()['session_id']
    assert device.poll(session)['status'] == 'approved'
    def unavailable(*args,**kwargs):
        raise ValueError('network unavailable')
    monkeypatch.setattr(device,'_wire',unavailable)
    assert device.cancel(session)['remote_cancelled'] is False
    with pytest.raises(ValueError, match='unavailable'):
        device.credential('token-test',base_url='http://127.0.0.1:8765/v1')


def test_claim_rejects_late_foreign_account_without_config_publication(native, monkeypatch):
    device, home = native
    install_wire(device,monkeypatch)
    original = device._wire
    def mismatched(origin,operation,body=None,bearer=None):
        result = original(origin,operation,body,bearer)
        if operation.endswith('/claim'):
            result['data']['account_id'] = 'foreign-account'
        return result
    monkeypatch.setattr(device,'_wire',mismatched)
    session = device.start()['session_id']
    before = (home/'config.yaml').read_bytes()
    with pytest.raises(ValueError,match='account binding'):
        device.poll(session)
    assert (home/'config.yaml').read_bytes() == before


def test_installed_ack_is_required_before_connected(native, monkeypatch):
    device, _ = native
    install_wire(device,monkeypatch)
    original = device._wire
    def denied_ack(origin,operation,body=None,bearer=None):
        result = original(origin,operation,body,bearer)
        if operation.endswith('/ack'):
            result['status'] = 'expired'
        return result
    monkeypatch.setattr(device,'_wire',denied_ack)
    session = device.start()['session_id']
    with pytest.raises(ValueError,match='acknowledgment'):
        device.poll(session)
    assert device.status()['logged_in'] is False


def test_restart_connect_recovers_committed_ack_loss_not_new_intent(native, monkeypatch):
    device, _ = native
    calls = install_wire(device,monkeypatch,ack_lost=True)
    start = device.start()
    with pytest.raises(ValueError,match='lost ack'):
        device.poll(start['session_id'])
    recovered = device.start()
    assert recovered['session_id'] == start['session_id']
    assert calls.count('start') == 1
    assert device.poll(recovered['session_id'])['status'] == 'approved'


@pytest.mark.parametrize('terminal',['denied','expired','cancelled','revoked'])
def test_terminal_poll_never_claims_or_publishes(native,monkeypatch,terminal):
    device, home = native
    calls = install_wire(device,monkeypatch)
    session = device.start()['session_id']
    original = device._wire
    def terminal_wire(origin,operation,body=None,bearer=None):
        result = original(origin,operation,body,bearer)
        if operation.endswith('/poll'):
            result.update(status=terminal,data={})
        return result
    monkeypatch.setattr(device,'_wire',terminal_wire)
    before = (home/'config.yaml').read_bytes()
    assert device.poll(session)['status'] == terminal
    assert not any(call.endswith('/claim') for call in calls)
    assert (home/'config.yaml').read_bytes() == before


def test_actual_local_routes_require_auth_and_bind_session_to_original_profile(native,monkeypatch):
    from fastapi.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_TOKEN
    device, _home = native
    install_wire(device,monkeypatch)
    client = TestClient(app)
    headers={'X-Hermes-Session-Token':_SESSION_TOKEN}
    assert client.post('/api/providers/oauth/semantier-cloud/start').status_code == 401
    response = client.post('/api/providers/oauth/semantier-cloud/start',headers=headers)
    assert response.status_code == 200, response.text
    session=response.json()['session_id']
    assert 'proof' not in response.text
    approved = client.get('/api/providers/oauth/semantier-cloud/poll/'+session,headers=headers)
    assert approved.status_code == 200, approved.text
    assert approved.json()['status'] == 'approved'
    assert 'private-account-token' not in approved.text
    foreign=client.get('/api/providers/oauth/semantier-cloud/poll/cloud-foreign',headers=headers)
    assert foreign.status_code == 404
    assert foreign.json()['detail']['code'] == 'cloud_session_unavailable'
    assert 'private-account-token' not in foreign.text


def test_account_preparation_local_route_preserves_model_and_keeps_secrets_native(native, monkeypatch):
    from fastapi.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_TOKEN
    device, _ = native
    before = device._config()['model'].copy()
    install_wire(device, monkeypatch)
    client = TestClient(app)
    path = '/api/providers/oauth/semantier-cloud/start?account_prepare=true'
    assert client.post(path).status_code == 401
    headers = {'X-Hermes-Session-Token': _SESSION_TOKEN}
    started = client.post(path, headers=headers, json={'expected_actor': ACTOR, 'origin': ACCOUNT_ORIGIN})
    assert started.status_code == 200
    assert 'proof' not in started.text
    result = client.get('/api/providers/oauth/semantier-cloud/poll/'+started.json()['session_id'], headers=headers)
    assert result.status_code == 200 and result.json()['status'] == 'approved'
    assert 'private-account-token' not in result.text
    assert device._config()['model'] == before


def test_profile_copy_rejects_private_original_binding(native):
    device, home = native
    state={'proof':'original-private-proof','session_id':'cloud-test','status':'pending'}
    device._write(state)
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    foreign=home/'foreign';foreign.mkdir()
    (foreign/'.cloud-device.bin').write_bytes((home/'.cloud-device.bin').read_bytes())
    token=set_hermes_home_override(str(foreign))
    try:
        with pytest.raises(device.CloudConnectionError) as failure:
            device._read()
        assert failure.value.code == 'cloud_profile_mismatch'
        assert failure.value.status == 409
    finally:
        reset_hermes_home_override(token)


def test_scope_export_classifies_native_proof_and_credential_store_private():
    from hermes_cli.profiles import profile_path_is_private
    assert profile_path_is_private(('.cloud-device.bin',))
    assert profile_path_is_private(('.cloud-device.lock',))
    assert profile_path_is_private(('.cloud-account-fence.bin',))
    assert profile_path_is_private(('.cloud-account-fence.lock',))


def test_runtime_and_auxiliary_resolve_protected_reference_only_for_approved_endpoint(native,monkeypatch):
    from hermes_cli.runtime_provider import _get_named_custom_provider, resolve_runtime_provider
    from agent.auxiliary_client import _named_custom_api_key
    device, home = native
    install_wire(device,monkeypatch)
    session=device.start()['session_id'];device.poll(session)
    entry=_get_named_custom_provider('semantier-cloud')
    assert entry['credential_ref']=='token-test'
    assert _named_custom_api_key(entry,'semantier-cloud',entry['base_url'])=='private-account-token'
    runtime=resolve_runtime_provider(requested='semantier-cloud')
    assert runtime['api_key']=='private-account-token'
    with pytest.raises(ValueError,match='another endpoint'):
        _named_custom_api_key(entry,'semantier-cloud','https://foreign.example/v1')
    assert 'private-account-token' not in (home/'config.yaml').read_text()


def test_portal_navigation_origin_is_distinct_from_authenticated_billing_api_origin(native,monkeypatch):
    device, home=native
    (home/'config.yaml').write_text('model: {provider: custom, base_url: "http://127.0.0.1:8765/v1"}\nproviders:\n  semantier-cloud:\n    base_url: http://127.0.0.1:8765/v1\n    portal_origin: http://127.0.0.1:8766\n')
    install_wire(device,monkeypatch)
    original=device._wire
    def separate_portal(origin,operation,body=None,bearer=None):
        result=original(origin,operation,body,bearer)
        if operation=='start':
            result['data']['verification_uri']='http://127.0.0.1:8766/billing/connect?intent_ref=intent-test'
        return result
    monkeypatch.setattr(device,'_wire',separate_portal)
    started=device.start()
    assert started['verification_url'].startswith('http://127.0.0.1:8766/')
    assert device.poll(started['session_id'])['status']=='approved'
    assert device._config()['billing']['portal_origin']=='http://127.0.0.1:8766'


def test_connect_never_reuses_an_unrelated_provider_origin(native,monkeypatch):
    device,home=native
    (home/'config.yaml').write_text('model: {provider: custom, base_url: https://unrelated.invalid/v1, api_key: prior-secret}\n')
    contacted = []
    def wire(origin, operation, body=None, bearer=None):
        contacted.append((origin, body, bearer))
        return {'contract_version': device.CONTRACT, 'intent_ref': 'public-service',
                'expires_at': '2099-01-01T00:00:00+00:00',
                'data': {'verification_uri': origin + '/billing/connect?intent_ref=public-service'}}
    monkeypatch.setattr(device, '_wire', wire)
    device.start()
    assert contacted[0][0] == device.DEFAULT_CLOUD_ORIGIN
    assert 'prior-secret' not in json.dumps(contacted)
    assert device._config()['model']['base_url'] == 'https://unrelated.invalid/v1'


def test_install_retains_embedded_previous_provider_and_capability_revision(native,monkeypatch):
    device,home=native
    cfg=device._config()
    cfg['model'].update(api_key='old-provider-secret',default='old-model',api_mode='anthropic_messages')
    from hermes_cli.config import atomic_config_replace
    atomic_config_replace(home/'config.yaml',cfg)
    install_wire(device,monkeypatch)
    monkeypatch.setattr(device,'_catalog',lambda *_:[{'id':'allowed-model','revision':'allowed-revision'}])
    started=device.start();device.poll(started['session_id'])
    installed=device._config()
    assert installed['model']==cfg['model']
    from agent.semantier_gateway import gateway_settings
    assert gateway_settings().api_key=='old-provider-secret'
    assert gateway_settings(allow_unselected=True).api_key=='private-account-token'
    assert installed['providers']['semantier-cloud']['credential_ref']=='token-test'
    assert device.credential('token-test',base_url='http://127.0.0.1:8765/v1')=='private-account-token'


def test_slow_down_keeps_intent_pending_and_bounds_poll_frequency(native,monkeypatch):
    device,_=native
    calls=install_wire(device,monkeypatch)
    started=device.start()
    original=device._wire
    def slowed(origin,operation,body=None,bearer=None):
        if operation.endswith('/poll'):
            calls.append(operation)
            return {'contract_version':device.CONTRACT,'intent_ref':'intent-test','status':'slow_down','retry_after':5}
        return original(origin,operation,body,bearer)
    monkeypatch.setattr(device,'_wire',slowed)
    assert device.poll(started['session_id'])['status']=='pending'
    assert device.poll(started['session_id'])['status']=='pending'
    assert calls.count('intent-test/poll')==1
    assert calls.count('intent-test/claim')==0


def test_native_model_confirmation_preserves_provider_and_protected_custody(native, monkeypatch):
    """The real /api/model/set owner must not demote a connected native service."""
    from agent import semantier_gateway as cloud
    from hermes_cli.config import atomic_config_replace
    from hermes_cli.web_server_config import _apply_model_assignment_sync
    device, home = native
    install_wire(device, monkeypatch)
    started = device.start()
    device.poll(started['session_id'])
    cfg = device._config()
    cfg['model'].update(key_env='UNRELATED_PROVIDER_KEY', api_key='stale-inline-key')
    cfg['providers']['other-provider'] = {'base_url': 'https://other.invalid/v1'}
    atomic_config_replace(home/'config.yaml', cfg)
    monkeypatch.setattr(cloud, 'cloud_model_profile', lambda *args, **kwargs: {
        'protocol': 'chat_completions', 'revision': 'allowed-revision', 'capabilities': {}})
    result = _apply_model_assignment_sync('main', 'semantier-cloud', 'allowed-model', '', '')
    saved = device._config()
    assert result['provider'] == saved['model']['provider'] == 'semantier-cloud'
    assert saved['providers']['semantier-cloud']['credential_ref'] == 'token-test'
    assert saved['providers']['other-provider'] == cfg['providers']['other-provider']
    assert not {'api_key', 'key_env', 'api_key_env'} & saved['model'].keys()
    assert 'private-account-token' not in (home/'config.yaml').read_text()
    assert saved['model']['capability_revision'] == 'allowed-revision'


def test_native_model_switch_preserves_logical_provider_and_drops_foreign_keys(native, monkeypatch):
    from agent import semantier_gateway as cloud
    from hermes_cli.config import atomic_config_replace
    from hermes_cli.model_switch import switch_model, apply_model_selection
    device, home = native
    install_wire(device, monkeypatch)
    started = device.start()
    device.poll(started['session_id'])
    cfg = device._config()
    cfg['model'].update(key_env='UNRELATED_PROVIDER_KEY', api_key='stale-inline-key')
    atomic_config_replace(home/'config.yaml', cfg)
    monkeypatch.setattr(cloud, 'cloud_model_profile', lambda *args, **kwargs: {
        'protocol': 'chat_completions', 'revision': 'allowed-revision', 'capabilities': {}})
    result = switch_model('allowed-model', 'semantier-cloud', 'allowed-model',
                          current_base_url=cfg['model']['base_url'], explicit_provider='semantier-cloud')
    assert result.success and result.target_provider == 'semantier-cloud'
    saved = apply_model_selection(cfg['model'], result)
    assert saved['provider'] == 'semantier-cloud'
    assert not {'api_key', 'key_env', 'api_key_env'} & saved.keys()
    assert 'private-account-token' not in json.dumps(saved)


def test_install_preserves_selected_http3_transport(native, monkeypatch):
    from hermes_cli.config import atomic_config_replace
    device, home = native
    config = device._config()
    config['providers']['semantier-cloud']['http_version'] = '3'
    config['providers']['semantier-cloud']['ssl_ca_cert'] = '/fixture/root.crt'
    atomic_config_replace(home / 'config.yaml', config)
    install_wire(device, monkeypatch)
    session = device.start()['session_id']
    assert device.poll(session)['status'] == 'approved'
    assert device._config()['providers']['semantier-cloud']['http_version'] == '3'
    assert device._config()['providers']['semantier-cloud']['ssl_ca_cert'] == '/fixture/root.crt'


def test_start_conflict_rotates_only_the_exact_obsolete_start_and_preserves_credentials(native, monkeypatch):
    device, _ = native
    first = device._new_state({})
    first['credentials'] = {'retained': {'opaque': 'saved'}}
    device._write(first)
    calls = []
    def wire(origin, operation, body=None, bearer=None):
        calls.append(body)
        if len(calls) == 1:
            raise device.CloudDeviceError(409, 'start_conflict')
        return {'contract_version':device.CONTRACT, 'intent_ref':'fresh-intent', 'status':'pending',
                'expires_at':'2099-01-01T00:00:00+00:00',
                'data':{'verification_uri':origin+'/billing/connect?intent_ref=fresh-intent'}}
    monkeypatch.setattr(device, '_wire', wire)
    result = device.start()
    assert len(calls) == 2
    assert calls[0]['idempotency_key'] != calls[1]['idempotency_key']
    assert calls[0]['possession_challenge'] != calls[1]['possession_challenge']
    assert result['session_id'] != first['session_id']
    assert device._read()['credentials'] == first['credentials']


@pytest.mark.parametrize('reason', [None, 'profile_generation_mismatch', 'start_conflict'])
def test_start_conflicts_are_bounded_and_never_hide_other_conflicts(native, monkeypatch, reason):
    device, _ = native
    calls = []
    def conflict(*args, **kwargs):
        calls.append(1)
        raise device.CloudDeviceError(409, reason)
    monkeypatch.setattr(device, '_wire', conflict)
    with pytest.raises(device.CloudDeviceError):
        device.start()
    assert len(calls) == (2 if reason == 'start_conflict' else 1)


def test_catalog_outage_does_not_prevent_install_or_replace_working_model(native, monkeypatch):
    from hermes_cli.config import atomic_config_replace
    device, home = native
    cfg=device._config()
    cfg['model'].update(provider='custom', default='prior-model', base_url='https://prior.invalid/v1', api_key='prior-fixture-only')
    atomic_config_replace(home/'config.yaml',cfg)
    calls=install_wire(device,monkeypatch)
    def outage(*args):
        raise device.CloudConnectionError('cloud_network_unavailable','catalog timeout',503,retryable=True)
    monkeypatch.setattr(device,'_catalog',outage)
    started=device.start()
    assert device.poll(started['session_id'])['status']=='approved'
    assert calls.count('intent-test/claim')==calls.count('intent-test/ack')==1
    installed=device._config()
    assert installed['model']==cfg['model']
    assert installed['providers']['semantier-cloud']['credential_ref']=='token-test'
    assert device.credential('token-test',base_url='http://127.0.0.1:8765/v1')=='private-account-token'


@pytest.mark.parametrize('status,retained',[(401,False),(403,True)])
def test_only_confirmed_authentication_refusal_removes_native_custody(native,monkeypatch,status,retained):
    device,_=native
    install_wire(device,monkeypatch)
    original=device._wire
    def wire(origin,operation,body=None,bearer=None):
        if operation=='readiness':raise device.CloudDeviceError(status)
        return original(origin,operation,body,bearer)
    monkeypatch.setattr(device,'_wire',wire)
    start=device.start()
    result=device.poll(start['session_id'])
    assert ('token-test' in device._read()['credentials']) is retained
    assert result['status']==('approved' if retained else 'revoked')


def test_expired_custody_reaches_desktop_catalog_as_authorization_expiry(native,monkeypatch):
    device,_=native
    install_wire(device,monkeypatch)
    started=device.start();device.poll(started['session_id'])
    state=device._read()
    state['credentials']['token-test']['credential']['expires_at']='2000-01-01T00:00:00+00:00'
    device._write(state)
    from hermes_cli.inventory import load_picker_context,build_desktop_model_options_payload
    payload=build_desktop_model_options_payload(load_picker_context(),source_scope='fixture-expired-custody',refresh=True)
    row=next(row for row in payload['providers'] if row['slug']=='semantier-cloud')
    assert row['discovery_status']=='cloud_credential_expired'
    assert row['models']==[]


def test_settings_can_return_to_installed_cloud_from_byok(native, monkeypatch):
    from agent import semantier_gateway as cloud
    from hermes_cli.config import atomic_config_replace
    from hermes_cli.web_server_config import _apply_model_assignment_sync
    device, home = native
    install_wire(device, monkeypatch)
    started = device.start()
    device.poll(started['session_id'])
    cfg = device._config()
    cfg['model'] = {'provider': 'custom:other', 'default': 'byok-model',
                    'base_url': 'https://other.example/v1', 'key_env': 'OTHER_PROVIDER_KEY'}
    atomic_config_replace(home / 'config.yaml', cfg)
    monkeypatch.setattr(cloud, 'cloud_model_profile', lambda *args, **kwargs: {
        'protocol': 'chat_completions', 'revision': 'allowed-revision', 'capabilities': {}})
    result = _apply_model_assignment_sync('main', 'semantier-cloud', 'allowed-model', '', '')
    saved = device._config()
    assert result['provider'] == saved['model']['provider'] == 'semantier-cloud'
    assert saved['model']['default'] == 'allowed-model'
    assert saved['providers']['semantier-cloud']['credential_ref'] == 'token-test'
    assert not {'api_key', 'key_env', 'api_key_env'} & saved['model'].keys()
    assert 'private-account-token' not in (home / 'config.yaml').read_text()
