"""Disposable real-gateway browser acceptance bridge and no-cost provider boundary.

Only Electron's OS transport is adapted. HTTP and WebSocket traffic reaches the
canonical native gateway using fresh private tickets; provider/config/model owners
remain production code. Run with an empty, dedicated HERMES_HOME.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semantier_billing_renderer_bridge import NativeRelay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=55475)
    parser.add_argument('--renderer-origin', default='http://127.0.0.1:5188')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    state = {'status': 200, 'inference_status': 200, 'models': ['fixture/alpha', 'fixture/shared'], 'calls': []}

    class Provider(BaseHTTPRequestHandler):
        def do_GET(self):
            key = self.headers.get('Authorization')
            state['calls'].append({'method': 'GET', 'path': self.path, 'key_matches': key == 'Bearer fixture-key'})
            payload = {'data': [{'id': model} for model in state['models']]}
            code = state['status'] if self.path in {'/catalog', '/v1/models'} else 404
            self.answer(code if key == 'Bearer fixture-key' else 401, payload)
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
            state['calls'].append({'method': 'POST', 'path': self.path, 'model': body.get('model'),
                                   'max_tokens': body.get('max_tokens'), 'key_matches': self.headers.get('Authorization') == 'Bearer fixture-key'})
            if self.path not in {'/v1/chat/completions', '/v1/responses', '/v1/messages'}:
                self.answer(404, {'error': 'Unsupported fixture metadata route'})
                return
            self.answer(state['inference_status'] if self.headers.get('Authorization') == 'Bearer fixture-key' else 401, {'id': 'fixture', 'object': 'chat.completion', 'created': 1,
                'model': body.get('model'), 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'OK'}}]})
        def answer(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *_args):
            pass
    provider = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    homes = [root / 'source-a', root / 'source-b']
    for home in homes:
        home.mkdir(mode=0o700)
        (home / '.env').write_text('# Synthetic fixture only\n', encoding='utf-8')
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        (home / 'config.yaml').write_text(json.dumps({'_config_version': DEFAULT_CONFIG['_config_version'], 'model': {}, 'gateway': {'allow_empty': True}}), encoding='utf-8')
        (home / '.no-bundled-skills').touch()
    from hermes_cli.gateway_runtime import ensure_gateway_runtime
    from hermes_cli.gateway_client import _session_ticket, gateway_ws_target
    endpoints = []
    for home in homes:
        result = ensure_gateway_runtime(home, timeout=60)
        if result.state != 'ready' or result.endpoint is None:
            raise RuntimeError(f'Fixture gateway failed: {result.state}: {result.detail}')
        endpoints.append(result.endpoint)
    def native_target(index):
        return gateway_ws_target(endpoints[index], _session_ticket(homes[index], endpoints[index]))
    relay = NativeRelay(args.renderer_origin, native_target)
    ready = threading.Event()
    relay_state = {}
    async def run_relay():
        from websockets.asyncio.server import serve
        async with serve(relay.handle, '127.0.0.1', 0, origins=[args.renderer_origin],
                         subprotocols=['hermes-gateway-v1']) as server:
            relay_state.update(port=server.sockets[0].getsockname()[1], loop=asyncio.get_running_loop(), stop=asyncio.Event())
            ready.set()
            await relay_state['stop'].wait()
    relay_thread = threading.Thread(target=lambda: asyncio.run(run_relay()), daemon=True)
    relay_thread.start()
    if not ready.wait(10):
        raise RuntimeError('Fixture relay failed')

    class Bridge(BaseHTTPRequestHandler):
        def answer(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', args.renderer_origin)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def permitted(self):
            return self.headers.get('Origin') in {None, args.renderer_origin}
        def do_OPTIONS(self):
            self.send_response(204 if self.permitted() else 403)
            self.send_header('Access-Control-Allow-Origin', args.renderer_origin)
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.end_headers()
        def do_GET(self):
            if not self.permitted():
                self.answer(403, {})
                return
            path = urlsplit(self.path)
            index = int(parse_qs(path.query).get('source', ['0'])[0])
            if path.path == '/connection' and index in range(len(homes)):
                self.answer(200, {'mode': 'remote', 'profile': 'default', 'connectionId': f'journey-{index}',
                    'baseUrl': endpoints[index].api_origin, 'wsUrl': f"ws://127.0.0.1:{relay_state['port']}/api/ws?" +
                    urlencode({'native_dial': '1', 'ticket': relay.issue(index)})})
            elif path.path == '/fixture':
                self.answer(200, {**state, 'provider_origin': f'http://127.0.0.1:{provider.server_port}'})
            else:
                self.answer(404, {})
        def do_POST(self):
            if not self.permitted():
                self.answer(403, {})
                return
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
            if self.path == '/fixture':
                state.update({key: value for key, value in body.items() if key in {'status', 'inference_status', 'models'}})
                self.answer(200, {'ok': True})
                return
            if self.path != '/api':
                self.answer(404, {})
                return
            request = body['request']
            path = request['path']
            index = body.get('source', 0)
            if index not in range(len(homes)) or not path.startswith('/api/') or '://' in path:
                self.answer(400, {})
                return
            import httpx
            url = endpoints[index].api_origin + path
            separator = '&' if '?' in url else '?'
            url += separator + urlencode({'profile': request.get('profile') or 'default'})
            ticket = _session_ticket(homes[index], endpoints[index], purpose='native-http')
            try:
                response = httpx.request(request.get('method') or 'GET', url,
                    json=request.get('body'), headers={'X-Hermes-Gateway-Ticket': ticket}, timeout=60, trust_env=False)
                self.answer(response.status_code, response.json())
            except (httpx.HTTPError, ValueError):
                self.answer(502, {'detail': 'Fixture native transport failed'})
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Bridge)
    def stop(*_args):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(json.dumps({'pid': os.getpid(), 'bridge': f'http://127.0.0.1:{server.server_port}',
                      'provider_origin': f'http://127.0.0.1:{provider.server_port}', 'homes': list(map(str, homes))}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        provider.shutdown()
        provider.server_close()
        relay_state['loop'].call_soon_threadsafe(relay_state['stop'].set)
        relay_thread.join(timeout=10)
        for home in homes:
            result = subprocess.run([sys.executable, '-m', 'hermes_cli.main', '--profile', 'default', 'gateway', 'stop'],
                cwd=ROOT, env={**os.environ, 'HERMES_HOME': str(home)}, capture_output=True, timeout=30, check=False)
            print(json.dumps({'stopped': str(home), 'returncode': result.returncode}), flush=True)


if __name__ == '__main__':
    main()
