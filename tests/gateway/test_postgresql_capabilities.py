"""Reject unprovisioned optional native authority before gateway startup."""
import pytest


def test_configured_telegram_postgresql_rejected_before_gateway_start(monkeypatch):
    from gateway.config import GatewayConfig,Platform,PlatformConfig
    from gateway.run import GatewayRunner
    monkeypatch.setenv('HERMES_STATE_BACKEND','postgresql')
    config=GatewayConfig(platforms={Platform.TELEGRAM:PlatformConfig(enabled=True)})
    with pytest.raises(RuntimeError,match='Telegram.*PostgreSQL'):
        GatewayRunner(config)


@pytest.mark.parametrize('backend,enabled',[('sqlite',True),('postgresql',False)])
def test_desktop_telegram_and_disabled_hosted_telegram_remain_available(monkeypatch,backend,enabled):
    from gateway.config import GatewayConfig,Platform,PlatformConfig
    from gateway.run_startup import validate_postgresql_platforms
    monkeypatch.setenv('HERMES_STATE_BACKEND',backend)
    assert validate_postgresql_platforms(GatewayConfig(platforms={Platform.TELEGRAM:PlatformConfig(enabled=enabled)})) is None
