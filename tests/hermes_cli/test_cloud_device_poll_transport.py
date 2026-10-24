"""A lost local HTTP reply must not release a still-running native install."""
import asyncio
import pytest


@pytest.mark.asyncio
async def test_observation_waits_for_original_install_after_request_cancellation(monkeypatch):
    from hermes_cli.web_routers import oauth
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []
    async def native(profile, operation, session):
        if profile != "default":
            raise ValueError("Foreign native session")
        calls.append((profile, operation, session))
        started.set()
        await release.wait()
        return {'status': 'approved'}
    monkeypatch.setattr(oauth, '_cloud_device_request', native)
    original = asyncio.create_task(oauth._cloud_poll('default', 'session', 'transport-test'))
    await started.wait()
    original.cancel()
    with pytest.raises(asyncio.CancelledError):
        await original
    observed = asyncio.create_task(oauth._cloud_poll('default', 'session', 'transport-test'))
    await asyncio.sleep(0)
    assert not observed.done()
    assert (await oauth._cloud_poll('beta', 'session', 'transport-test'))['status'] == 'error'
    release.set()
    assert (await observed)['status'] == 'approved'
    assert calls == [('default', 'poll', 'session')]
