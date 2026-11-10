"""Safe Cloud sign-in classification; HTTP/status errors must not masquerade as profile409."""
import socket
from types import SimpleNamespace

import pytest
import requests

from tests.hermes_cli.test_cloud_device_client import native as native  # noqa: PLC0414 -- pytest fixture re-export


def test_fresh_default_sign_in_uses_managed_origin_without_changing_model(native, monkeypatch):
    device, home = native
    (home / 'config.yaml').write_text('model: {provider: custom}\n', encoding='utf-8')
    seen = []

    def wire(origin, operation, body=None, bearer=None):
        seen.append(origin)
        return {'contract_version': device.CONTRACT, 'intent_ref': 'synthetic',
                'expires_at': '2099-01-01T00:00:00+00:00',
                'data': {'verification_uri': origin + '/billing/connect?intent_ref=synthetic'}}

    monkeypatch.setattr(device, '_wire', wire)
    assert device.start()['flow'] == 'device_code'
    assert seen == ['https://cloud.semantier.com']
    assert device._config()['model']['provider'] == 'custom'
    assert 'providers' not in device._config()


@pytest.mark.parametrize(('failure', 'code'), [
    (requests.ConnectionError(socket.gaierror(-2, 'PRIVATE HOSTNAME')), 'cloud_dns_failed'),
    (requests.exceptions.SSLError('PRIVATE CERTIFICATE'), 'cloud_tls_failed'),
    (requests.ConnectionError('PRIVATE ENDPOINT'), 'cloud_network_unavailable'),
    (requests.Timeout('PRIVATE ENDPOINT'), 'cloud_network_unavailable'),
])
def test_transport_diagnosis_has_no_raw_endpoint(native, monkeypatch, failure, code):
    device, _ = native

    class Session:
        trust_env = True
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def post(self, *_args, **_kwargs): raise failure

    monkeypatch.setattr(device, '_cloud_http', lambda _origin: (SimpleNamespace(
        Session=Session, exceptions=requests.exceptions), {}))
    with pytest.raises(ValueError) as error:
        device._wire('http://127.0.0.1:1', 'start')
    assert getattr(error.value, 'code', None) == code
    assert 'PRIVATE' not in str(error.value)
    assert getattr(error.value, 'status', None) != 409


@pytest.mark.parametrize('status', [400, 401, 403, 409, 410, 429, 500, 503])
@pytest.mark.asyncio
async def test_api_preserves_real_cloud_http_status(native, monkeypatch, status):
    from fastapi import HTTPException
    from hermes_cli.web_routers import oauth
    device, _ = native

    def fail(): raise device.CloudDeviceError(status)
    async def scoped(profile, fn): return fn()
    monkeypatch.setattr(device, 'start', fail)
    monkeypatch.setattr(oauth, 'scoped_to_thread', scoped)
    with pytest.raises(HTTPException) as error:
        await oauth._cloud_device_request('default', 'start')
    assert error.value.status_code == status
    assert isinstance(error.value.detail, dict)
    assert error.value.detail['code'].startswith('cloud_')


@pytest.mark.asyncio
async def test_unknown_value_error_is_not_claimed_to_be_profile_conflict(native, monkeypatch):
    from fastapi import HTTPException
    from hermes_cli.web_routers import oauth
    device, _ = native
    def fail(): raise ValueError('SECRET internal endpoint')
    async def scoped(profile, fn): return fn()
    monkeypatch.setattr(device, 'start', fail)
    monkeypatch.setattr(oauth, 'scoped_to_thread', scoped)
    with pytest.raises(HTTPException) as error:
        await oauth._cloud_device_request('default', 'start')
    assert error.value.status_code == 500
    assert 'SECRET' not in str(error.value.detail)


def test_actual_connection_refused_is_transport_not_profile(native, monkeypatch):
    device, _ = native
    monkeypatch.setattr(device, '_cloud_http', lambda _: (requests, {}))
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1', 0))
        origin = 'http://127.0.0.1:' + str(reserved.getsockname()[1])
        with pytest.raises(device.CloudConnectionError) as error:
            device._wire(origin, 'start')
    assert error.value.status == 503
    assert error.value.code == 'cloud_network_unavailable'


@pytest.mark.parametrize(('curl_code', 'expected'), [(6, 'cloud_dns_failed'), (60, 'cloud_tls_failed'),
                                                    (95, 'cloud_http3_unavailable'), (96, 'cloud_http3_unavailable')])
def test_curl_transport_codes_are_sanitized(native, curl_code, expected):
    device, _ = native
    error = requests.ConnectionError('SECRET ENDPOINT')
    error.code = curl_code
    public = device._transport_error(error)
    assert public.code == expected
    assert 'SECRET' not in str(public.detail())


def test_http3_missing_transport_is_specific_and_never_downgraded(native, monkeypatch):
    import sys
    device, _ = native
    monkeypatch.setattr(device, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setitem(sys.modules, 'curl_cffi', None)
    with pytest.raises(device.CloudConnectionError) as error:
        device._cloud_http('https://cloud.example.invalid')
    assert error.value.code == 'cloud_http3_unavailable'
    assert error.value.status == 503


def test_invalid_provider_config_is_not_a_conflict(native):
    device, home = native
    (home / 'config.yaml').write_text('providers: {semantier-cloud: invalid}\n', encoding='utf-8')
    with pytest.raises(device.CloudConnectionError) as error:
        device.start()
    assert error.value.status == 422
    assert error.value.code == 'cloud_config_invalid'


@pytest.mark.asyncio
async def test_retained_poll_keeps_transport_diagnosis_and_owner(native, monkeypatch):
    from hermes_cli.web_routers import oauth
    from fastapi import HTTPException
    calls = []
    async def native_request(profile, operation, session):
        calls.append((profile, session))
        raise HTTPException(503, {'code': 'cloud_dns_failed', 'message': 'Check DNS and retry.',
                                  'profile': profile, 'retryable': True, 'recovery': 'network'})
    monkeypatch.setattr(oauth, '_cloud_device_request', native_request)
    monkeypatch.setattr(oauth, '_cloud_polls', {})
    first = await oauth._cloud_poll('default', 'synthetic', 'same-operation')
    second = await oauth._cloud_poll('default', 'synthetic', 'same-operation')
    assert first == second
    assert first['error_code'] == 'cloud_dns_failed'
    assert first['http_status'] == 503
    assert first['profile'] == 'default'
    assert calls == [('default', 'synthetic')]


@pytest.mark.parametrize('tls', [False, True], ids=['real-http-conflict', 'untrusted-tls'])
def test_local_server_failure_keeps_transport_and_http_distinct(native, monkeypatch, tmp_path, tls):
    """Exercise real requests sockets; no external DNS, TLS service or credentials."""
    import http.server
    import ssl
    import subprocess
    import threading

    device, _ = native
    monkeypatch.setattr(device, '_cloud_http', lambda _: (requests, {}))

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(409)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"detail":"PRIVATE server diagnostic"}')

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    if tls:
        certificate, key = tmp_path / 'certificate.pem', tmp_path / 'key.pem'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                        '-keyout', str(key), '-out', str(certificate), '-days', '1',
                        '-subj', '/CN=localhost'], check=True, capture_output=True)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate, key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = ('https' if tls else 'http') + '://127.0.0.1:' + str(server.server_port)
        with pytest.raises(device.CloudConnectionError) as error:
            device._wire(origin, 'start')
        assert error.value.code == ('cloud_tls_failed' if tls else 'cloud_server_conflict')
        assert error.value.status == (502 if tls else 409)
        assert 'PRIVATE' not in str(error.value.detail())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize('custom_ca', [False, True])
def test_windows_http3_retains_strict_version_and_trust_store(native, monkeypatch, custom_ca):
    import sys
    device, _ = native
    monkeypatch.setattr(device, 'sys', SimpleNamespace(platform='win32'))
    settings = {'http_version': '3'}
    if custom_ca:
        settings['ssl_ca_cert'] = 'fixture-ca.pem'
    monkeypatch.setattr(device, '_config', lambda: {'providers': {'semantier-cloud': settings}})
    curl = SimpleNamespace(requests=object(), CurlOpt=SimpleNamespace(SSL_OPTIONS=216))
    monkeypatch.setitem(sys.modules, 'curl_cffi', curl)
    client, options = device._cloud_http('https://fixture.invalid')
    assert client is curl.requests
    assert options['http_version'] == 'v3only'
    assert options['trust_env'] is False
    if custom_ca:
        assert options['verify'] == 'fixture-ca.pem'
        assert 'curl_options' not in options
    else:
        assert options['curl_options'] == {216: 1 << 4}
        assert 'verify' not in options  # Never disable certificate validation.


def test_oversized_success_body_is_invalid_response_not_http_success(native, monkeypatch):
    device, _ = native
    class Session:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def post(self, *_args, **_kwargs):
            return SimpleNamespace(status_code=200, content=b'x' * (1024 * 1024 + 1))
    monkeypatch.setattr(device, '_cloud_http', lambda _: (SimpleNamespace(
        Session=Session, exceptions=requests.exceptions), {}))
    with pytest.raises(device.CloudConnectionError) as error:
        device._wire('http://127.0.0.1:1', 'start')
    assert error.value.status == 502
    assert error.value.code == 'cloud_response_invalid'
