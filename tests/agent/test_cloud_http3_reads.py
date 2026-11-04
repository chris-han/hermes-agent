"""Windows Cloud catalog and billing reads require the same HTTP/3 transport as sign-in."""
import sys
import types

import pytest


@pytest.fixture
def windows_transport(monkeypatch, tmp_path):
    from hermes_cli import cloud_device_client as device
    calls = []
    class RequestException(Exception): pass
    class Session:
        def __init__(self, **kwargs): calls.append(('session', kwargs))
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            calls.append(('get', url, kwargs))
            if '/catalog' in url:
                return types.SimpleNamespace(status_code=200, json=lambda: {'object': 'list', 'data': []})
            return types.SimpleNamespace(status_code=401, headers={}, json=dict)
    exceptions = types.SimpleNamespace(RequestException=RequestException, SSLError=RequestException,
                                       Timeout=RequestException, ConnectionError=RequestException)
    fake = types.SimpleNamespace(Session=Session, RequestException=RequestException, exceptions=exceptions,
                                 get=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('legacy HTTP client used')))
    monkeypatch.setitem(sys.modules, 'requests', fake)
    monkeypatch.setitem(sys.modules, 'curl_cffi', types.SimpleNamespace(requests=fake, CurlOpt=types.SimpleNamespace(SSL_OPTIONS=1)))
    monkeypatch.setattr(device.sys, 'platform', 'win32')
    monkeypatch.setattr(device, '_config', lambda: {'providers': {'semantier-cloud': {'base_url': 'https://cloud.example/v1'}}})
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    return calls


def test_windows_cloud_catalog_requires_http3(windows_transport):
    from agent import semantier_gateway as gateway
    gateway.clear_cloud_catalog()
    assert gateway.cloud_catalog(settings=gateway.GatewaySettings('https://cloud.example/v1', 'fixture')) == {'object': 'list', 'data': []}
    assert windows_transport[0] == ('session', {'http_version': 'v3only', 'trust_env': False, 'curl_options': {1: 16}})
    assert windows_transport[1][2]['headers'] == {'Authorization': 'Bearer fixture'}
    assert windows_transport[1][2]['allow_redirects'] is False
    gateway.clear_cloud_catalog()


def test_windows_cloud_billing_requires_http3(monkeypatch, windows_transport):
    from agent import semantier_gateway as gateway
    monkeypatch.setattr(gateway, 'gateway_settings', lambda: gateway.GatewaySettings('https://cloud.example/v1', 'fixture'))
    assert gateway.billing_request('state', schema_version=gateway.SCHEMA_VERSION)['error'] == 'unauthenticated'
    assert windows_transport[0] == ('session', {'http_version': 'v3only', 'trust_env': False, 'curl_options': {1: 16}})
    assert windows_transport[1][2]['headers'] == {'Authorization': 'Bearer fixture'}
    assert windows_transport[1][2]['allow_redirects'] is False


def test_billing_v1_accepts_matching_account_provisioning():
    from pydantic import ValidationError
    from tui_gateway.contracts.semantier_billing import SemantierBillingState
    payload = {'schema_version': 'semantier.billing.v1', 'backend': 'semantier',
               'organization': {'id': 'org-a'}, 'billing_account_id': 'acct-a',
               'mode': 'shadow', 'funding_source': 'CUSTOMER_CLOUD', 'balances': [],
               'capabilities': {'can_read_usage': True, 'can_charge': False, 'can_change_plan': False},
               'usage': {'status': 'UNAVAILABLE'}, 'as_of': '2026-11-05T00:00:00Z',
               'account_provisioning': {'local_status': 'ready', 'inference_ready': False,
                                        'organization_id': 'org-a', 'billing_account_id': 'acct-a'}}
    assert SemantierBillingState.model_validate(payload).account_provisioning.organization_id == 'org-a'
    payload['account_provisioning']['organization_id'] = 'other-org'
    with pytest.raises(ValidationError):
        SemantierBillingState.model_validate(payload)
