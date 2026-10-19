"""Offline browser bridge for the real canonical gateway and production Billing screen.

Only native OS bootstrap is adapted: every RPC goes to the production gateway, whose
profile makes actual Semantier Billing HTTP calls. No BillingApi or authority is mocked.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import secrets
import time
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class NativeRelay:
    """Exact-origin browser leg; native credentials remain in this OS-owned adapter."""
    def __init__(self, origin, native_target):
        self.origin, self.native_target = origin, native_target
        self.pending = {}
        self.lock = threading.Lock()

    def issue(self, source):
        nonce = secrets.token_urlsafe(32)
        with self.lock:
            now = time.monotonic()
            self.pending = {key: value for key, value in self.pending.items() if value[1] > now}
            self.pending[nonce] = (source, now + 30)
        return nonce

    def consume(self, nonce):
        with self.lock:
            value = self.pending.pop(nonce, None)
        return value[0] if value and value[1] > time.monotonic() else None

    async def handle(self, browser):
        from websockets.asyncio.client import connect
        from websockets.exceptions import WebSocketException
        protocols = browser.request.headers.get('Sec-WebSocket-Protocol', '').split(',')
        tickets = [value.strip()[22:] for value in protocols if value.strip().startswith('hermes-gateway-ticket.')]
        source = self.consume(tickets[0]) if len(tickets) == 1 else None
        if source is None or browser.request.path.split('?')[0] != '/api/ws':
            await browser.close(4401, 'Invalid bootstrap grant')
            return
        try:
            url, protocols = await asyncio.to_thread(self.native_target, source)
            async with connect(url, subprotocols=protocols, proxy=None, max_size=8 * 1024 * 1024) as native:
                async def forward(reader, writer):
                    async for message in reader:
                        await writer.send(message)
                tasks = [asyncio.create_task(forward(browser, native)), asyncio.create_task(forward(native, browser))]
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except (OSError, ValueError, RuntimeError, WebSocketException):
            await browser.close(1011, 'Native gateway transport unavailable')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', action='append', type=Path, required=True)
    parser.add_argument('--port', type=int, default=55465)
    parser.add_argument('--renderer-origin', default='http://127.0.0.1:5187')
    parser.add_argument('--stop-on-exit', action='store_true', help='Explicitly stop these disposable owners after the browser.')
    args = parser.parse_args()
    from hermes_cli.gateway_runtime import ensure_gateway_runtime
    from hermes_cli.gateway_client import _session_ticket, gateway_ws_target
    homes = [home.resolve() for home in args.home]
    if any(not (home / '.env').is_file() for home in homes):
        parser.error('Each disposable profile must own its issued credential in .env.')
    # This dedicated acceptance bootstrap launches several profile owners. Their
    # credential files, never a launch-profile token, supply authentication.
    os.environ.pop('SEMANTIER_MODEL_GATEWAY_TOKEN', None)
    endpoints = []
    for home in homes:
        if not (home / 'config.yaml').is_file():
            parser.error('Every disposable profile must already have a configured config.yaml.')
        result = ensure_gateway_runtime(home, timeout=60)
        if result.state != 'ready' or result.endpoint is None:
            raise RuntimeError(f'Native gateway unavailable: {result.state}: {result.detail}')
        endpoints.append(result.endpoint)

    def native_target(index):
        endpoint = endpoints[index]
        return gateway_ws_target(endpoint, _session_ticket(homes[index], endpoint))
    relay = NativeRelay(args.renderer_origin, native_target)
    relay_ready = threading.Event()
    relay_state = {}
    async def serve_relay():
        from websockets.asyncio.server import serve
        async with serve(relay.handle, '127.0.0.1', 0, origins=[args.renderer_origin],
                         subprotocols=['hermes-gateway-v1'], max_size=8 * 1024 * 1024) as socket:
            relay_state.update(port=socket.sockets[0].getsockname()[1], loop=asyncio.get_running_loop(), stop=asyncio.Event())
            relay_ready.set()
            await relay_state['stop'].wait()
    relay_thread = threading.Thread(target=lambda: asyncio.run(serve_relay()), daemon=True)
    relay_thread.start()
    if not relay_ready.wait(10):
        raise RuntimeError('Native transport adapter failed to start')

    class Bridge(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            origin = self.headers.get('Origin')
            if parsed.path != '/connection' or origin not in {None, args.renderer_origin}:
                self.send_error(403)
                return
            query = parse_qs(parsed.query)
            try:
                index = int(query.get('source', ['0'])[0])
                if not 0 <= index < len(homes):
                    raise ValueError('unknown source')
            except ValueError:
                self.send_error(400)
                return
            # Only an expiring adapter nonce crosses into the browser. The actual owner
            # ticket is minted server-side when this browser opens its transport.
            ws_url = f"ws://127.0.0.1:{relay_state['port']}/api/ws?" + urlencode({
                'native_dial': '1', 'ticket': relay.issue(index)})
            endpoint = endpoints[index]
            payload = {'mode': 'remote', 'profile': 'default', 'connectionId': f'witness-{index}',
                       'baseUrl': endpoint.api_origin, 'wsUrl': ws_url}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Access-Control-Allow-Origin', args.renderer_origin)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Bridge)
    def stop(*_args):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(json.dumps({'bridge': f'http://127.0.0.1:{server.server_port}', 'profiles': len(homes),
                      'origins': [endpoint.api_origin for endpoint in endpoints]}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        relay_state['loop'].call_soon_threadsafe(relay_state['stop'].set)
        relay_thread.join(timeout=10)
        if args.stop_on_exit:
            for home in homes:
                env = {**os.environ, 'HERMES_HOME': str(home)}
                stopped = subprocess.run(
                    [sys.executable, '-m', 'hermes_cli.main', '--profile', 'default', 'gateway', 'stop'],
                    cwd=ROOT, env=env, timeout=30, capture_output=True, text=True, check=False)
                print(json.dumps({'native_gateway_stop': stopped.returncode}), flush=True)


if __name__ == '__main__':
    main()
