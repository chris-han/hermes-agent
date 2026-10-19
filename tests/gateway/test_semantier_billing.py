"""Canonical Billing uses the authenticated authority profile, never the launch profile."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import yaml

from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


def test_canonical_billing_binds_profile_and_negotiates_schema(tmp_path, monkeypatch):
    launch = tmp_path / 'launch'
    served = tmp_path / 'served'
    launch.mkdir()
    served.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(launch))
    (launch / 'config.yaml').write_text('billing: {backend: nous}\n')
    calls = []
    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.headers.get('Authorization'))
            payload = {'schema_version': 'semantier.billing.v1', 'backend': 'semantier',
                       'billing_account_id': 'ba-served', 'status': 'unavailable',
                       'reason': 'commercial subscription integration unavailable',
                       'capabilities': {'can_change_plan': False, 'can_charge': False}}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (served / 'config.yaml').write_text(yaml.safe_dump({
        'billing': {'backend': 'semantier'}, 'semantier_gateway': {
            'enabled': True, 'base_url': f'http://127.0.0.1:{server.server_port}/v1',
            'api_key_env': 'TEST_GATEWAY_TOKEN'}}))
    (served / '.env').write_text('TEST_GATEWAY_TOKEN=served-token\n')
    try:
        with SessionDB(db_path=served / 'state.db') as db:
            epoch = begin_runtime_epoch(db, instance_id='current')
            authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(served),
                                         instance_id='current', db=db, epoch=epoch)
            connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
            async def journey():
                unsupported = await connection.dispatch({'id': 1, 'method': 'subscription.state', 'params': {}})
                assert unsupported['result']['error'] == 'unsupported_version'
                assert calls == []
                response = await connection.dispatch({'id': 2, 'method': 'subscription.state', 'params': {
                    'schema_version': 'semantier.billing.v1'}})
                assert response['result']['backend'] == 'semantier'
                assert response['result']['billing_account_id'] == 'ba-served'
                assert response['result']['status'] == 'unavailable'
                assert calls == ['Bearer served-token']
                foreign = await connection.dispatch({'id': 3, 'method': 'billing.state', 'params': {
                    'profile': 'foreign', 'schema_version': 'semantier.billing.v1'}})
                assert foreign['error']['data']['reason'] == 'profile_mismatch'
                await connection.close()
            asyncio.run(journey())
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
