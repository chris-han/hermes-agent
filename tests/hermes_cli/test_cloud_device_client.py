"""Native device lifecycle against the frozen Cloud v1.0.1 external wire seam."""
import json
import pytest
from cryptography.fernet import Fernet


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
            data = {**binding,'claim_id':state['claim_id'],'organization_id':'org-test',
                'credential':{'token_id':'token-test','secret':'private-account-token',
                              'expires_at':'2099-02-01T00:00:00+00:00'},
                'inference_origin':origin,'billing_origin':origin,
                'catalog_path':'/api/cloud-provider/catalog',
                'readiness_path':'/api/cloud-provider/device/readiness',
                'installation_expires_at':'2099-01-01T00:00:00+00:00',
                'renewal':'reauthorization_required'}
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
    assert device._config()['model']['default'] == 'allowed-model'
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
    assert foreign.status_code == 409
    assert 'private-account-token' not in foreign.text


def test_profile_copy_rejects_private_original_binding(native):
    device, home = native
    state={'proof':'original-private-proof','session_id':'cloud-test','status':'pending'}
    device._write(state)
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    foreign=home/'foreign';foreign.mkdir()
    (foreign/'.cloud-device.bin').write_bytes((home/'.cloud-device.bin').read_bytes())
    token=set_hermes_home_override(str(foreign))
    try:
        with pytest.raises(ValueError,match='unavailable'):
            device._read()
    finally:
        reset_hermes_home_override(token)


def test_scope_export_classifies_native_proof_and_credential_store_private():
    from hermes_cli.profiles import profile_path_is_private
    assert profile_path_is_private(('.cloud-device.bin',))
    assert profile_path_is_private(('.cloud-device.lock',))


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
    monkeypatch.setattr(device,'_wire',lambda *args,**kwargs:pytest.fail('unrelated endpoint contacted'))
    with pytest.raises(ValueError,match='Cloud deployment origin'):
        device.start()
    assert not (home/'.cloud-device.bin').exists()


def test_install_retains_embedded_previous_provider_and_capability_revision(native,monkeypatch):
    device,home=native
    cfg=device._config()
    cfg['model'].update(api_key='old-provider-secret',default='old-model',api_mode='anthropic_messages')
    from hermes_cli.config import atomic_config_replace
    atomic_config_replace(home/'config.yaml',cfg)
    install_wire(device,monkeypatch)
    monkeypatch.setattr(device,'_catalog',lambda *_:[{'id':'allowed-model','revision':'allowed-revision'}])
    started=device.start();device.poll(started['session_id'])
    from hermes_cli.config_providers import get_compatible_custom_providers
    installed=device._config()
    prior=[row for row in get_compatible_custom_providers(installed) if row.get('api_key')=='old-provider-secret']
    assert len(prior)==1 and prior[0]['base_url']=='http://127.0.0.1:8765/v1'
    assert installed['model']['capability_revision']=='allowed-revision'
    assert installed['model']['api_mode']=='chat_completions'
    assert prior[0]['api_mode']=='anthropic_messages'


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
