import httpx
import pytest
from agent.cloud_http3_transport import CloudHTTP3Transport


class Response:
    status_code = 200
    http_version = 30
    headers = httpx.Headers({'content-type': 'text/event-stream'})
    closed = False
    def iter_content(self):
        yield b'data: first\n\n'
        yield b'data: second\n\n'
    def close(self): self.closed = True


class Session:
    def __init__(self, response): self.response = response; self.calls = []
    def request(self, **kwargs): self.calls.append(kwargs); return self.response
    def close(self): pass


def test_stream_and_scope():
    response = Response(); session = Session(response)
    transport = CloudHTTP3Transport('https://cloud.example/v1', session)
    with httpx.Client(transport=transport) as client:
        with client.stream('POST', 'https://cloud.example/v1/chat/completions', content=b'{}') as result:
            assert list(result.iter_bytes()) == [b'data: first\n\n', b'data: second\n\n']
        assert response.closed
        with pytest.raises(httpx.UnsupportedProtocol):
            client.get('https://other.example/v1/models')
    assert len(session.calls) == 1
    assert session.calls[0]['allow_redirects'] is False


def test_no_protocol_fallback():
    response = Response(); response.http_version = 2
    session = Session(response)
    with httpx.Client(transport=CloudHTTP3Transport('https://cloud.example/v1', session)) as client:
        with pytest.raises(httpx.UnsupportedProtocol):
            client.get('https://cloud.example/v1/models')
    assert response.closed and len(session.calls) == 1


def test_httpx_owns_response_decompression():
    import gzip
    response = Response()
    response.headers = httpx.Headers({'content-encoding': 'gzip', 'content-type': 'application/json'})
    response.iter_content = lambda: iter([gzip.compress(b'{"ok":true}')])
    session = Session(response)
    with httpx.Client(transport=CloudHTTP3Transport('https://cloud.example/v1', session)) as client:
        assert client.get('https://cloud.example/v1/models').json() == {'ok': True}
    assert session.calls[0]['accept_encoding'] is None


def test_connection_error_is_redacted_and_not_retried(monkeypatch):
    import sys
    import types
    class RequestException(Exception): pass
    class Timeout(RequestException): pass
    monkeypatch.setitem(sys.modules, 'curl_cffi.requests', types.SimpleNamespace(exceptions=types.SimpleNamespace(RequestException=RequestException, Timeout=Timeout)))
    session = Session(Response())
    def fail(**kwargs):
        session.calls.append(kwargs)
        raise RequestException('private-token-in-upstream-error')
    session.request = fail
    with httpx.Client(transport=CloudHTTP3Transport('https://cloud.example/v1', session)) as client:
        with pytest.raises(httpx.ConnectError) as error:
            client.get('https://cloud.example/v1/models')
    assert 'private-token' not in str(error.value)
    assert len(session.calls) == 1


def test_stream_timeout_is_redacted_and_closed(monkeypatch):
    import sys
    import types
    class RequestException(Exception): pass
    class Timeout(RequestException): pass
    monkeypatch.setitem(sys.modules, 'curl_cffi.requests', types.SimpleNamespace(exceptions=types.SimpleNamespace(RequestException=RequestException, Timeout=Timeout)))
    response = Response()
    def chunks():
        yield b'data: first\n\n'
        raise Timeout('private-upstream-detail')
    response.iter_content = chunks
    with httpx.Client(transport=CloudHTTP3Transport('https://cloud.example/v1', Session(response))) as client:
        with pytest.raises(httpx.ReadTimeout) as error:
            with client.stream('GET', 'https://cloud.example/v1/models') as result:
                list(result.iter_bytes())
    assert response.closed
    assert 'private-upstream' not in str(error.value)


def test_loop_session_closes_without_starting_requests():
    from agent.cloud_http3_transport import LoopCloudSession
    session = LoopCloudSession()
    session.close()
    assert not session.thread.is_alive()
    session.close()


def test_real_http3_silent_stream_cancels_and_stops_loop():
    import os
    import time
    from agent.cloud_http3_transport import LoopCloudSession
    origin = os.environ.get('SEMANTIER_H3_FIXTURE_ORIGIN')
    ca = os.environ.get('SEMANTIER_H3_FIXTURE_CA')
    if not origin or not ca:
        pytest.skip('Requires the owned Docker HTTP/3 silent-stream fixture')
    import hermes_cli.cloud_device_client as device
    from agent.process_bootstrap import build_keepalive_http_client
    original_config = device._config
    device._config = lambda: {'providers': {'semantier-cloud': {
        'base_url': origin, 'http_version': '3', 'ssl_ca_cert': ca}}}
    try:
        client = build_keepalive_http_client(origin)
    finally:
        device._config = original_config
    session = client._transport.session
    with client:
        with client.stream('GET', origin + '/events') as response:
            assert response.status_code == 200
            assert response.http_version == 'HTTP/3'
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total >= 65544:
                    break
            assert total == 65544
            started = time.monotonic()
            response.close()
            assert time.monotonic() - started < 1
    assert not session.thread.is_alive()


def test_cloud_factory_scope_and_explicit_failure(monkeypatch):
    import agent.cloud_http3_transport as transport
    settings = {'base_url': 'https://cloud.example/v1', 'http_version': '3'}
    assert transport.cloud_http3_client('https://other.example/v1', settings=settings) is None
    with pytest.raises(ValueError, match='HTTPS'):
        transport.cloud_http3_client('http://cloud.example/v1',
            settings={**settings, 'base_url': 'http://cloud.example/v1'})
    def unavailable(**kwargs):
        raise ImportError('missing curl')
    monkeypatch.setattr(transport, 'LoopCloudSession', unavailable)
    with pytest.raises(ImportError):
        transport.cloud_http3_client('https://cloud.example/v1', settings=settings)


def test_bootstrap_cloud_failure_cannot_return_legacy_client(monkeypatch):
    from agent.process_bootstrap import build_keepalive_http_client
    import hermes_cli.cloud_device_client as device
    monkeypatch.setattr(device, '_config', lambda: {'providers': {'semantier-cloud': {
        'base_url': 'https://cloud.example/v1', 'http_version': 'invalid'}}})
    with pytest.raises(ValueError, match='Unsupported'):
        build_keepalive_http_client('https://cloud.example/v1')


def test_async_cloud_factory_owns_and_closes_loop():
    import asyncio
    from agent.cloud_http3_transport import cloud_http3_client
    async def check():
        client = cloud_http3_client('https://cloud.example/v1', async_mode=True,
            settings={'base_url': 'https://cloud.example/v1', 'http_version': '3'})
        session = client._transport.sync.session
        await client.aclose()
        assert not session.thread.is_alive()
    asyncio.run(check())


def test_real_async_http3_stream_closes_during_silence(monkeypatch):
    import asyncio
    import os
    import time
    import hermes_cli.cloud_device_client as device
    from agent.process_bootstrap import build_keepalive_http_client
    origin = os.environ.get('SEMANTIER_H3_FIXTURE_ORIGIN')
    ca = os.environ.get('SEMANTIER_H3_FIXTURE_CA')
    if not origin or not ca:
        pytest.skip('Requires the owned Docker HTTP/3 silent-stream fixture')
    monkeypatch.setattr(device, '_config', lambda: {'providers': {'semantier-cloud': {
        'base_url': origin, 'http_version': '3', 'ssl_ca_cert': ca}}})
    async def check():
        client = build_keepalive_http_client(origin, async_mode=True)
        session = client._transport.sync.session
        async with client:
            async with client.stream('GET', origin + '/events') as response:
                assert response.http_version == 'HTTP/3'
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total >= 65544:
                        break
                assert total == 65544
                started = time.monotonic()
                await response.aclose()
                assert time.monotonic() - started < 1
        assert not session.thread.is_alive()
    asyncio.run(check())
