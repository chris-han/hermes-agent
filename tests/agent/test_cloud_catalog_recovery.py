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
