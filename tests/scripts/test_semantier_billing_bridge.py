"""The browser bootstrap adapts transport only, preserving native owner authentication."""
import asyncio

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidStatus

from scripts.semantier_billing_renderer_bridge import NativeRelay


@pytest.mark.asyncio
async def test_relay_keeps_native_ticket_off_browser_and_origin_off_native_wire():
    protocol = 'hermes-gateway-v1'
    seen = []

    async def native(socket):
        seen.append(dict(socket.request.headers))
        await socket.send(await socket.recv())

    async with serve(native, '127.0.0.1', 0, subprotocols=[protocol]) as upstream:
        upstream_url = f'ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}/api/ws'
        relay = NativeRelay('http://127.0.0.1:5187', lambda index: (upstream_url, [protocol, 'hermes-gateway-ticket.native-private']))
        async with serve(relay.handle, '127.0.0.1', 0, origins=[relay.origin], subprotocols=[protocol]) as server:
            nonce = relay.issue(0)
            url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/api/ws'
            protocols = [protocol, 'hermes-gateway-ticket.' + nonce]
            with pytest.raises(InvalidStatus):
                async with connect(url, origin='http://untrusted.example', subprotocols=protocols, proxy=None):
                    pass
            async with connect(url, origin=relay.origin, subprotocols=protocols, proxy=None) as browser:
                await browser.send('{"jsonrpc":"2.0","method":"billing.state","id":1}')
                assert await browser.recv() == '{"jsonrpc":"2.0","method":"billing.state","id":1}'
            assert 'origin' not in seen[0]
            assert 'native-private' in seen[0]['sec-websocket-protocol']
            assert nonce not in seen[0]['sec-websocket-protocol']
            assert relay.consume(nonce) is None
