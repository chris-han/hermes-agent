"""Cloud device requests can require HTTP/3 without weakening legacy transports."""
import sys
import types

import pytest


def test_device_request_uses_http3_when_configured(monkeypatch):
    from hermes_cli import cloud_device_client as device

    calls = []
    class Session:
        def __init__(self, **kwargs):
            calls.append(('session', kwargs))
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, **kwargs):
            calls.append(('post', url, kwargs))
            return types.SimpleNamespace(status_code=200, content=b'{}', json=lambda: {'contract_version': device.CONTRACT})
    class RequestException(Exception): pass
    fake = types.SimpleNamespace(Session=Session, exceptions=types.SimpleNamespace(RequestException=RequestException))
    monkeypatch.setitem(sys.modules, 'curl_cffi', types.SimpleNamespace(requests=fake))
    monkeypatch.setattr(device, '_config', lambda: {'providers': {'semantier-cloud': {'http_version': '3'}}})
    assert device._wire('https://cloud.example', 'start')['contract_version'] == device.CONTRACT
    assert calls[0] == ('session', {'http_version': 'v3only', 'trust_env': False})
    assert calls[1][2]['allow_redirects'] is False


def test_http3_failure_does_not_retry_with_another_protocol(monkeypatch):
    from hermes_cli import cloud_device_client as device
    attempts = []
    class RequestException(Exception): pass
    class Session:
        def __init__(self, **kwargs): attempts.append(kwargs)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, *args, **kwargs): raise RequestException('unavailable')
    fake = types.SimpleNamespace(Session=Session, exceptions=types.SimpleNamespace(RequestException=RequestException))
    monkeypatch.setitem(sys.modules, 'curl_cffi', types.SimpleNamespace(requests=fake))
    monkeypatch.setattr(device, '_config', lambda: {'providers': {'semantier-cloud': {'http_version': '3'}}})
    with pytest.raises(ValueError, match='Cloud connection unavailable'):
        device._wire('https://cloud.example', 'start')
    assert attempts == [{'http_version': 'v3only', 'trust_env': False}]


def test_unknown_transport_does_not_silently_change_protocol(monkeypatch):
    from hermes_cli import cloud_device_client as device
    monkeypatch.setattr(device, '_config', lambda: {'providers': {'semantier-cloud': {'http_version': 'typo'}}})
    with pytest.raises(ValueError, match='Unsupported Cloud HTTP version'):
        device._wire('https://cloud.example', 'start')


def test_catalog_uses_http3_and_keeps_bearer_in_authorization(monkeypatch):
    from hermes_cli import cloud_device_client as device
    calls = []
    class Session:
        def __init__(self, **kwargs): calls.append(('session', kwargs))
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            calls.append(('get', url, kwargs))
            return types.SimpleNamespace(status_code=200, content=b'{}', json=lambda: {'object': 'list', 'data': []})
    class RequestException(Exception): pass
    fake = types.SimpleNamespace(Session=Session, exceptions=types.SimpleNamespace(RequestException=RequestException))
    monkeypatch.setitem(sys.modules, 'curl_cffi', types.SimpleNamespace(requests=fake))
    monkeypatch.setattr(device, '_config', lambda: {'providers': {'semantier-cloud': {'http_version': '3'}}})
    assert device._catalog('https://cloud.example', 'private-token') == []
    assert calls[0] == ('session', {'http_version': 'v3only', 'trust_env': False})
    assert calls[1][2]['headers'] == {'Authorization': 'Bearer private-token'}
    assert calls[1][2]['allow_redirects'] is False


def test_http3_mode_requires_https(monkeypatch):
    from hermes_cli import cloud_device_client as device
    monkeypatch.setattr(device, '_config', lambda: {'providers': {'semantier-cloud': {'http_version': '3'}}})
    with pytest.raises(ValueError, match='HTTP/3 requires HTTPS'):
        device._wire('http://127.0.0.1:1', 'start')


def test_provider_normalization_preserves_http3_setting():
    from hermes_cli.config_providers import _normalize_custom_provider_entry, _custom_provider_entry_to_provider_config
    entry = {'base_url': 'https://cloud.example/v1', 'http_version': '3'}
    assert _normalize_custom_provider_entry(entry, provider_key='semantier-cloud')['http_version'] == '3'
    assert _custom_provider_entry_to_provider_config(entry, provider_key='semantier-cloud')['http_version'] == '3'


def test_http3_uses_explicit_development_ca_without_disabling_verification(monkeypatch):
    from hermes_cli import cloud_device_client as device
    calls = []
    class Session:
        def __init__(self, **kwargs): calls.append(kwargs)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, **kwargs):
            return types.SimpleNamespace(status_code=200, content=b'{}', json=lambda: {'contract_version': device.CONTRACT})
    class RequestException(Exception): pass
    fake = types.SimpleNamespace(Session=Session, exceptions=types.SimpleNamespace(RequestException=RequestException))
    monkeypatch.setitem(sys.modules, 'curl_cffi', types.SimpleNamespace(requests=fake))
    monkeypatch.setattr(device, '_config', lambda: {'providers': {'semantier-cloud': {'http_version': '3', 'ssl_ca_cert': '/fixture/root.crt'}}})
    assert device._wire('https://cloud.example', 'start')['contract_version'] == device.CONTRACT
    assert calls == [{'http_version': 'v3only', 'trust_env': False, 'verify': '/fixture/root.crt'}]


def test_windows_cloud_endpoint_defaults_to_http3(monkeypatch):
    from hermes_cli import cloud_device_client as device

    calls = []
    class Session:
        def __init__(self, **kwargs): calls.append(kwargs)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, **kwargs):
            return types.SimpleNamespace(status_code=200, content=b'{}', json=lambda: {'contract_version': device.CONTRACT})
    class RequestException(Exception): pass
    fake = types.SimpleNamespace(Session=Session, exceptions=types.SimpleNamespace(RequestException=RequestException))
    monkeypatch.setattr(device.sys, 'platform', 'win32')
    monkeypatch.setitem(sys.modules, 'curl_cffi', types.SimpleNamespace(requests=fake, CurlOpt=types.SimpleNamespace(SSL_OPTIONS=1)))
    monkeypatch.setitem(sys.modules, 'requests', fake)
    monkeypatch.setattr(device, '_config', lambda: {'providers': {'semantier-cloud': {'base_url': 'https://maas.semantier.cn/v1'}}})
    assert device._wire('https://maas.semantier.cn', 'start')['contract_version'] == device.CONTRACT
    assert calls == [{'http_version': 'v3only', 'trust_env': False, 'curl_options': {1: 1 << 4}}]
