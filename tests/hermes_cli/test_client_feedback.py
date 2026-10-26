"""Feedback custody tests: real profile storage, substituted OS crypto/wire only."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"src"))
import pytest
from cryptography.fernet import Fernet

@pytest.fixture
def native(tmp_path,monkeypatch):
    from hermes_cli import client_feedback as owner
    from hermes_constants import set_hermes_home_override,reset_hermes_home_override
    token=set_hermes_home_override(str(tmp_path))
    config='model: {provider: custom, base_url: "http://localhost:4000/v1"}\nclient_feedback: {service_url: "http://127.0.0.1:8927", portal_origin: "http://127.0.0.1:3326"}\n'
    (tmp_path/'config.yaml').write_text(config)
    cipher=Fernet(Fernet.generate_key())
    monkeypatch.setattr(owner.device,'_protect',cipher.encrypt)
    monkeypatch.setattr(owner.device,'_unprotect',cipher.decrypt)
    yield owner,tmp_path,config
    reset_hermes_home_override(token)


def test_feedback_authorization_does_not_touch_model_or_expose_capability(native,monkeypatch):
    owner,home,config=native
    calls=[]
    def wire(origin,secret,method,path,body=None):
        calls.append((origin,secret,method,path,body))
        state=owner._read()
        if path=='/authorizations':
            assert body['proof_challenge']==hashlib.sha256(state['authorization']['proof'].encode()).hexdigest()
            return {'authorization_url':'http://127.0.0.1:3326/client-feedback/authorize?intent_ref='+body['intent_ref'],'expires_at':'2099-01-01T00:00:00+00:00'}
        return {'status':'approved','purpose':'diagnostic','profile_ref':state['profile_ref'],'capability':'feedback-only-secret','capability_expires_at':'2099-01-01T00:00:00+00:00'}
    monkeypatch.setattr(owner,'_request',wire)
    first=owner.authorization_start('diagnostic')
    assert owner.authorization_start('diagnostic')==first
    assert len(calls)==1
    result=owner.authorization_poll()
    assert result=={'status':'approved','purpose':'diagnostic'}
    assert 'feedback-only-secret' not in json.dumps(owner.status())
    assert b'feedback-only-secret' not in (home/'.semantier-feedback.bin').read_bytes()
    assert owner.status()['enabled'] is False
    assert owner.status()['diagnostic_signed_in'] is True
    assert (home/'config.yaml').read_text()==config
    assert not (home/'.cloud-device.bin').exists()
    with pytest.raises(ValueError,match='authorization'):owner.consent(True)


def test_lost_start_reply_reuses_intent_and_proof(native,monkeypatch):
    owner,_,_=native;seen=[]
    def wire(origin,secret,method,path,body=None):
        seen.append(body)
        if len(seen)==1:raise OSError('lost reply')
        return {'authorization_url':'http://127.0.0.1:3326/client-feedback/authorize?intent_ref='+body['intent_ref'],'expires_at':'2099-01-01T00:00:00+00:00'}
    monkeypatch.setattr(owner,'_request',wire)
    with pytest.raises(OSError):owner.authorization_start('reporting')
    owner.authorization_start('reporting')
    assert seen[0]==seen[1]


def test_same_shared_fixture_is_consumed_without_client_extensions():
    from contracts.client_operations import UpdateEvent,FeedbackAuthorization
    from pydantic import ValidationError
    path=Path(__file__).resolve().parents[3]/'src/contracts/client-operations-v1.fixture.json'
    fixture=json.loads(path.read_text())
    assert hashlib.sha256(path.with_name('client-operations-v1.schema.json').read_bytes()).hexdigest()==fixture['schema_sha256']
    for collection,model in [('update_events',UpdateEvent),('feedback_authorizations',FeedbackAuthorization)]:
        for vector in fixture[collection]:
            if vector['expected']=='accepted':model.model_validate(vector['input'])
            else:
                with pytest.raises(ValidationError):model.model_validate(vector['input'])


@pytest.mark.parametrize('status',[401,403,413,422,429,503])
def test_local_feedback_route_preserves_cloud_recovery_status(native,monkeypatch,status):
    from fastapi.testclient import TestClient
    from hermes_cli.web_server import app,_SESSION_TOKEN
    owner,_,_=native
    def wire(*args,**kwargs):
        raise owner.device.CloudDeviceError(status)
    monkeypatch.setattr(owner,'_request',wire)
    with TestClient(app) as client:
        response=client.post('/api/client-feedback/authorization_start',
            headers={'X-Hermes-Session-Token':_SESSION_TOKEN},json={'purpose':'diagnostic'})
    assert response.status_code==status,response.text
    if status==429:
        assert 1<=response.json()['detail']['retry_after_seconds']<=900


def test_expired_diagnostic_authorization_requests_sign_in_without_sending(native,monkeypatch):
    owner,_,_=native
    owner._write({'enabled':False,'queue':[],'dropped':0,'diagnostic_authorization':{
        'origin':'http://127.0.0.1:8927','capability':'expired-secret','expires_at':'2000-01-01T00:00:00+00:00'}})
    calls=[]
    monkeypatch.setattr(owner,'_request',lambda *args,**kwargs:calls.append(args))
    assert owner.status()['diagnostic_signed_in'] is False
    with pytest.raises(owner.device.CloudDeviceError) as error:
        owner._diagnostic_identity()
    assert error.value.status==401
    assert calls==[]


def test_definite_unauthorized_reply_clears_only_diagnostic_authorization(native,monkeypatch):
    owner,home,config=native
    owner._write({'enabled':True,'queue':[],'dropped':0,'capability':'reporting-secret','diagnostic_authorization':{
        'origin':'http://127.0.0.1:8927','capability':'revoked-secret','expires_at':'2099-01-01T00:00:00+00:00'}})
    def wire(*args,**kwargs):raise owner.device.CloudDeviceError(401)
    monkeypatch.setattr(owner,'_request',wire)
    payload=owner.preview({'app_version':'0.0.39.0','note':'Keep this draft'})['payload']
    with pytest.raises(owner.device.CloudDeviceError):owner.submit(payload)
    state=owner._read()
    assert not state.get('diagnostic_authorization')
    assert state['enabled'] and state['capability']=='reporting-secret'
    assert (home/'config.yaml').read_text()==config


@pytest.mark.parametrize('header,expected',[('2',2),('0',1),('9999',900),('invalid',30)])
def test_feedback_wire_bounds_server_retry_delay(native,monkeypatch,header,expected):
    import httpx
    owner,_,_=native
    client=httpx.Client
    transport=httpx.MockTransport(lambda request:httpx.Response(429,headers={'retry-after':header},json={'detail':'limited'}))
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:client(transport=transport,**kwargs))
    with pytest.raises(owner.device.CloudDeviceError) as error:
        owner._request('https://feedback.example.test','','POST','/diagnostics',{})
    assert error.value.status==429
    assert error.value.retry_after_seconds==expected


def test_local_feedback_network_failure_is_recoverable_without_leaking_details(native,monkeypatch):
    import httpx
    from fastapi.testclient import TestClient
    from hermes_cli.web_server import app,_SESSION_TOKEN
    owner,_,_=native
    def wire(*args,**kwargs):raise httpx.ConnectError('private-transport-detail')
    monkeypatch.setattr(owner,'_request',wire)
    with TestClient(app) as client:
        response=client.post('/api/client-feedback/authorization_start',headers={'X-Hermes-Session-Token':_SESSION_TOKEN},json={'purpose':'diagnostic'})
    assert response.status_code==503
    assert 'private-transport-detail' not in response.text


@pytest.mark.parametrize('status',[202,302])
def test_feedback_protocol_refusal_cannot_become_a_local_success(native,monkeypatch,status):
    from fastapi.testclient import TestClient
    from hermes_cli.web_server import app,_SESSION_TOKEN
    owner,_,_=native
    def wire(*args,**kwargs):raise owner.device.CloudDeviceError(status)
    monkeypatch.setattr(owner,'_request',wire)
    with TestClient(app) as client:
        response=client.post('/api/client-feedback/authorization_start',headers={'X-Hermes-Session-Token':_SESSION_TOKEN},json={'purpose':'diagnostic'})
    assert response.status_code==502
