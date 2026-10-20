"""Negotiated commerce reads retain token/read and portal-origin boundaries."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from tui_gateway.contracts.semantier_billing import SemantierBillingV2


def state():
    money = {'amount': '100.00', 'currency': 'CNY'}
    return {'schema_version': 'semantier.billing.v2', 'backend': 'semantier',
            'billing_account_id': 'account-a', 'organization': {'id': 'org-a', 'display_name': None},
            'currency': 'CNY', 'merchant': None, 'environment': 'DEVELOPMENT', 'subscription': None,
            'credits': {key: money for key in ('allowance', 'purchased', 'held', 'available')},
            'usage': {'known_cost': money, 'period': 'lifetime'},
            'capabilities': {'can_read_usage': True, 'can_manage_billing': False,
                             'can_open_portal': True, 'can_charge': False,
                             'can_change_plan': False, 'can_auto_top_up': False},
            'portal': {'url': 'https://billing.example'}, 'as_of': '2026-10-20T00:00:00Z'}


def test_v2_retains_literal_false_and_rejects_secret_bearing_portal():
    payload = state()
    assert SemantierBillingV2.model_validate(payload).portal.url == 'https://billing.example'
    for unsafe in ('https://user:secret@billing.example', 'https://billing.example/?token=secret',
                   'https://billing.example/#secret', 'https://billing.example/billing', 'http://billing.example'):
        bad = deepcopy(payload)
        bad['portal']['url'] = unsafe
        with pytest.raises(ValidationError):
            SemantierBillingV2.model_validate(bad)
    payload['capabilities']['can_charge'] = True
    with pytest.raises(ValidationError):
        SemantierBillingV2.model_validate(payload)


def test_v2_bridge_uses_read_only_resource_and_bound_account(tmp_path, monkeypatch):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from agent import semantier_gateway as sg
    calls = []

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get('Authorization')))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(state()).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(
        f'billing: {{backend: semantier, account_id: account-a}}\nsemantier_gateway:\n'
        f'  enabled: true\n  base_url: http://127.0.0.1:{server.server_port}/v1\n  api_key_env: TEST_BILLING_TOKEN\n')
    (tmp_path / '.env').write_text('TEST_BILLING_TOKEN=read-secret\n')
    try:
        result = sg.read_billing('state', schema_version='semantier.billing.v2')
        assert result == {'ok': True, **state()}
        assert calls == [('/api/billing/commerce/state?account_id=account-a', 'Bearer read-secret')]
        assert 'read-secret' not in str(result)
        assert sg.billing_request('checkout', schema_version='semantier.billing.v2')['ok'] is False
        assert len(calls) == 1
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
