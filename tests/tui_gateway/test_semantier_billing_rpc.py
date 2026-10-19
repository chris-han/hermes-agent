"""Semantier legacy clients get typed unsupported/recovery, never Nous mutations."""
import pytest

import tui_gateway.server as srv


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('billing: {backend: semantier}\nsemantier_gateway: {enabled: false}\n')


@pytest.mark.parametrize('method,params', [
    ('billing.charge', {'amount_usd': '10'}), ('billing.auto_reload', {'enabled': True}),
    ('billing.charge_status', {'charge_id': 'unknown'}), ('billing.step_up', {}),
    ('subscription.change', {'subscription_type_id': 'tier'}), ('subscription.resume', {}),
    ('subscription.upgrade', {'subscription_type_id': 'tier'}),
    ('subscription.preview', {'subscription_type_id': 'tier'}),
])
def test_semantier_cannot_open_nous_commercial_recovery(configured, method, params):
    response = srv._methods[method](1, params)['result']
    assert response['ok'] is False and response['backend'] == 'semantier'
    assert response['error'] == 'unsupported_capability'
    assert not response.get('portal_url')
    assert not response.get('granted')


def test_legacy_dollar_views_never_coerce_semantier_account(configured):
    assert srv._methods['billing.state'](1, {})['result']['error'] == 'unsupported_version'
    assert srv._methods['subscription.state'](2, {})['result']['error'] == 'unsupported_version'
    assert srv._methods['usage.bars'](3, {})['result'] == {'ok': True, 'available': False}
