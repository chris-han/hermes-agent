"""Safe Cloud sign-in classification; HTTP/status errors must not masquerade as profile409."""
import socket
from types import SimpleNamespace

import pytest
import requests

from tests.hermes_cli.test_cloud_device_client import install_wire
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
    assert seen == ['https://cloud.semantier.cn']
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
        from datetime import UTC, datetime, timedelta
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
        now = datetime.now(UTC)
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
                .public_key(private_key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1))
                .sign(private_key, hashes.SHA256()))
        key.write_bytes(private_key.private_bytes(serialization.Encoding.PEM,
                        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        certificate.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
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


@pytest.mark.parametrize('status', [401, 403, 404, 410, 409, 503])
def test_retry_replaces_only_definitely_invalid_device_intent_and_keeps_credentials(native, monkeypatch, status):
    device, _ = native
    original = device._new_state({})
    original.update(status='approved', credentials={'retained': {
        'credential': {'secret': 'synthetic-retained', 'expires_at': '2099-01-01T00:00:00+00:00'},
        'inference_origin': original['origin']}})
    device._write(original)
    starts = []
    def wire(origin, operation, body=None, bearer=None):
        if operation != 'start':
            raise device.CloudDeviceError(status)
        starts.append(body)
        ref = 'intent-' + str(len(starts))
        return {'contract_version': device.CONTRACT, 'intent_ref': ref,
                'expires_at': '2099-01-01T00:00:00+00:00',
                'data': {'verification_uri': origin + '/billing/connect?intent_ref=' + ref}}
    monkeypatch.setattr(device, '_wire', wire)
    first = device.start()
    with pytest.raises(device.CloudDeviceError) as error:
        device.poll(first['session_id'])
    assert error.value.status == status
    retry = device.start()
    invalid = status in {401, 403, 404, 410}
    assert (retry['session_id'] != first['session_id']) is invalid
    assert len(starts) == (2 if invalid else 1)
    assert device._read()['credentials'] == original['credentials']
    assert device.credential('retained', base_url=original['origin'] + '/v1') == 'synthetic-retained'


@pytest.mark.parametrize('provider_sections', [
    'providers: null\nbilling: null\n',
    'providers: {semantier-cloud: null}\nbilling: null\n',
])
def test_optional_null_sections_preserve_existing_legacy_cloud_origin(native, monkeypatch, provider_sections):
    device, home = native
    (home / 'config.yaml').write_text(provider_sections +
        'semantier_gateway: {enabled: true, base_url: "http://127.0.0.1:8765/v1"}\n', encoding='utf-8')
    assert device._new_state({})['origin'] == 'http://127.0.0.1:8765'
    install_wire(device, monkeypatch)
    session = device.start()['session_id']
    assert device.poll(session)['status'] == 'approved'
    assert device._config()['model']['provider'] == 'semantier-cloud'


@pytest.mark.parametrize('origin', ['https://cloud.semantier.cn', 'https://cloud.semantier.com', 'http://127.0.0.1:8765'])
def test_explicit_origin_remains_owned_by_profile(native, origin):
    device, home = native
    (home / 'config.yaml').write_text('providers:\n  semantier-cloud:\n    base_url: '+origin+'/v1\n')
    assert device._new_state({})['origin'] == origin


def test_transport_error_never_claims_received_http_status(native):
    device, _ = native
    failure = requests.ConnectionError('PRIVATE')
    failure.code = 7
    public = device._transport_error(failure, transport='http3')
    assert public.code == 'cloud_http3_unavailable'
    assert public.detail()['http_status'] is None
    assert public.detail()['error_source'] == 'transport'
    assert public.detail()['transport_code'] == 7


def test_received_503_and_protocol_422_are_distinct(native):
    device, _ = native
    server = device.CloudDeviceError(503)
    assert server.detail()['http_status'] == 503
    assert server.detail()['error_source'] == 'server'
    protocol = device.CloudDeviceError(422)
    assert protocol.code == 'cloud_protocol_unsupported'
    assert protocol.recovery == 'upgrade_client'


@pytest.mark.parametrize('status', [301, 302, 307, 308])
def test_redirect_is_refused_without_becoming_a_gateway_redirect(native, status):
    device, _ = native
    error = device.CloudDeviceError(status)
    assert error.status == 502
    assert error.code == 'cloud_origin_redirected'
    assert error.detail()['http_status'] == status
    assert error.retryable is False



def test_uncertain_start_retries_same_intent_and_proof(native, monkeypatch):
    device, _ = native
    calls = []
    def wire(origin, operation, body=None, bearer=None):
        calls.append(body.copy())
        if len(calls) == 1:
            raise device.CloudConnectionError('cloud_http3_unavailable', 'HTTP/3 unavailable', 503)
        return {'contract_version': device.CONTRACT, 'intent_ref': 'synthetic',
                'expires_at': '2099-01-01T00:00:00+00:00',
                'data': {'verification_uri': origin + '/billing/connect?intent_ref=synthetic'}}
    monkeypatch.setattr(device, '_wire', wire)
    with pytest.raises(device.CloudConnectionError):
        device.start()
    private = device._read()
    device.start()
    assert calls[0] == calls[1]
    assert device._read()['proof'] == private['proof']
    assert device._read()['generation'] == private['generation']
