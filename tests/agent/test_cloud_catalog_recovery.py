"""Catalog failures preserve public cause and recover without authorization mutations."""
import json
from types import SimpleNamespace
import pytest
import requests
from agent import semantier_gateway as gateway
from hermes_cli import cloud_device_client as device

@pytest.fixture
def catalog(monkeypatch):
    gateway.clear_cloud_catalog()
    replies=[]
    class Session:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def get(self,*args,**kwargs):
            item=replies.pop(0)
            if isinstance(item,Exception):raise item
            return SimpleNamespace(status_code=item[0],json=lambda:item[1])
    monkeypatch.setattr(device,'_cloud_http',lambda origin:(requests,{}))
    monkeypatch.setattr(requests,'Session',Session)
    return replies,gateway.GatewaySettings('https://fixture.invalid/v1','fixture-only')

@pytest.mark.parametrize('status,code',[(401,'cloud_authorization_required'),(403,'cloud_authorization_denied'),(503,'cloud_server_error')])
def test_http_failure_is_not_an_empty_catalog_and_retry_recovers(catalog,status,code):
    replies,settings=catalog
    replies.extend([(status,{}),(200,{'object':'list','data':[]})])
    with pytest.raises(device.CloudConnectionError) as caught:gateway.cloud_catalog(settings=settings)
    assert caught.value.code==code
    assert caught.value.detail()['http_status']==status
    assert caught.value.detail()['operation']=='catalog'
    assert gateway.cloud_catalog(settings=settings)=={'object':'list','data':[]}


def test_timeout_has_no_invented_http_status(catalog):
    replies,settings=catalog
    replies.append(requests.exceptions.Timeout('private supplier or credential diagnostic'))
    with pytest.raises(device.CloudConnectionError) as caught:gateway.cloud_catalog(settings=settings)
    assert caught.value.code=='cloud_request_timeout'
    assert caught.value.detail()['http_status'] is None
    assert 'private' not in json.dumps(caught.value.detail())


@pytest.mark.parametrize('code',['cloud_metadata_timeout','cloud_metadata_unavailable','cloud_capability_mismatch'])
def test_upstream_metadata_diagnosis_is_allowlisted(catalog,code):
    replies,settings=catalog
    replies.append((503,{'detail':{'code':code,'message':'private diagnostic'}}))
    with pytest.raises(device.CloudConnectionError) as caught:gateway.cloud_catalog(settings=settings)
    assert caught.value.code==code
    assert caught.value.detail()['http_status']==503
    assert 'private diagnostic' not in str(caught.value)

@pytest.mark.parametrize('protocol',['chat','chat_completions','openai_chat_completions'])
def test_main_assignment_accepts_producer_chat_protocol_with_installed_connection(monkeypatch,protocol):
    from hermes_cli.web_server_config import _prepare_main_assignment
    settings=gateway.GatewaySettings('https://fixture.invalid/v1','fixture-only')
    monkeypatch.setattr(gateway,'gateway_settings',lambda **kw: settings if kw.get('allow_unselected') else None)
    monkeypatch.setattr(gateway,'cloud_connection_config',lambda **kw:{'credential_ref':'fixture-ref'})
    monkeypatch.setattr(gateway,'cloud_model_profile',lambda model,**kw: {'protocol':protocol} if kw.get('settings') is settings else {})
    base,result=_prepare_main_assignment({'model':{'provider':'custom','default':'previous'}},'semantier-cloud','permitted-model','','')
    assert result.success and result.target_provider=='semantier-cloud'
    assert base==settings.base_url


def test_main_assignment_rejects_real_protocol_mismatch_with_public_diagnosis(monkeypatch):
    from fastapi import HTTPException
    from hermes_cli.web_server_config import _prepare_main_assignment
    monkeypatch.setattr(gateway,'gateway_settings',lambda **kw:gateway.GatewaySettings('https://fixture.invalid/v1','fixture-only'))
    monkeypatch.setattr(gateway,'cloud_connection_config',lambda **kw:{'credential_ref':'fixture-ref'})
    monkeypatch.setattr(gateway,'cloud_model_profile',lambda *a,**kw:{'protocol':'anthropic_messages'})
    with pytest.raises(HTTPException) as caught:
        _prepare_main_assignment({},'semantier-cloud','permitted-model','','')
    assert caught.value.status_code==422
    assert caught.value.detail['code']=='cloud_capability_mismatch'

@pytest.mark.parametrize('protocol,permitted',[('openai_chat_completions',True),('anthropic_messages',False)])
def test_runtime_readiness_uses_same_protocol_contract(monkeypatch,protocol,permitted):
    from hermes_cli import runtime_provider
    monkeypatch.setattr(runtime_provider,'resolve_requested_provider',lambda p:p)
    monkeypatch.setattr(runtime_provider,'_raise_if_provider_disabled',lambda p:None)
    monkeypatch.setattr(runtime_provider,'_get_model_config',dict)
    monkeypatch.setattr(gateway,'gateway_settings',lambda **kw:gateway.GatewaySettings('https://fixture.invalid/v1','fixture-only'))
    monkeypatch.setattr(gateway,'cloud_model_profile',lambda *a,**kw:{'protocol':protocol})
    if permitted:
        assert runtime_provider.resolve_runtime_provider(requested='semantier-cloud',target_model='permitted-model')['api_mode']=='chat_completions'
    else:
        with pytest.raises(device.CloudConnectionError) as caught:
            runtime_provider.resolve_runtime_provider(requested='semantier-cloud',target_model='permitted-model')
        assert caught.value.code=='cloud_capability_mismatch'
