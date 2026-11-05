"""A pre-header failure must finish native cleanup before surfacing its error."""
import asyncio
import threading
import httpx
import pytest
from agent import cloud_http3_transport as transport


def test_tls_failure_during_watchdog_cleanup_keeps_connection_error(monkeypatch):
    import curl_cffi
    cleanup_started = threading.Event()
    closed = []
    class TLSFailure(Exception):
        code = 60
    class Curl:
        def __init__(self, **kwargs): pass
        def setopt(self, *args): pass
        def close(self): closed.append('curl')
    class Multi:
        def __init__(self, *, loop): self.loop = loop
        def add_handle(self, curl):
            future = self.loop.create_future()
            future.set_exception(TLSFailure('synthetic certificate rejection'))
            return future
        def remove_handle(self, curl): pass
        async def close(self): closed.append('multi')
    original_gather = asyncio.gather
    async def delayed_watchdog_cleanup(*tasks, **kwargs):
        if len(tasks) == 1 and tasks[0].get_coro().__name__ == 'watch_headers':
            cleanup_started.set()
            await asyncio.sleep(.1)
        return await original_gather(*tasks, **kwargs)
    original_close = transport._LoopResponse.close
    def close_during_cleanup(response):
        assert cleanup_started.wait(2), 'Transfer did not enter watchdog cleanup'
        original_close(response)
    monkeypatch.setattr(curl_cffi, 'Curl', Curl)
    monkeypatch.setattr(curl_cffi, 'AsyncCurl', Multi)
    monkeypatch.setattr(asyncio, 'gather', delayed_watchdog_cleanup)
    monkeypatch.setattr(transport._LoopResponse, 'close', close_during_cleanup)
    session = transport.LoopCloudSession()
    try:
        with pytest.raises(httpx.ConnectError, match='transfer failed'):
            session.request(method='GET', url='https://fixture.invalid/', headers=[], data=b'')
        assert closed == ['multi', 'curl']
    finally:
        session.close()
    assert not session.thread.is_alive()
