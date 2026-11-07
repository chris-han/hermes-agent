"""Changing telemetry recipient requires a new opt-in before any send."""
from hermes_cli.observability.shared_metrics_consent import OFFER_VERSION, consent_state, set_answer
from hermes_cli.observability.shared_metrics_send_config import DEFAULT_ENDPOINT, resolve_send_config

NOUS = "https://telemetry.nousresearch.com/v1/telemetry"


def shared(**kwargs):
    return {"telemetry": {"shared_metrics": kwargs}}


def test_legacy_nous_opt_in_must_be_reasked_and_cannot_send():
    cfg = shared(enabled=True, send=True, offer_version=2, endpoint=NOUS)
    assert consent_state(cfg) == {
        "enabled": True, "send": False, "decided": False, "reask": True,
    }
    config = resolve_send_config(cfg)
    assert config.endpoint == DEFAULT_ENDPOINT
    assert config.enabled is True
    assert config.send is False


def test_old_no_thanks_remains_settled_to_avoid_reprompting():
    cfg = shared(enabled=False, send=False, offer_version=2)
    state = consent_state(cfg)
    assert state["reask"] is False
    assert state["decided"] is True
    assert state["send"] is False


def test_new_consent_uses_semantier_and_allows_send():
    assert OFFER_VERSION == 3
    cfg = shared(enabled=True, send=True, offer_version=2, endpoint=NOUS)
    set_answer(cfg, True, True)
    assert consent_state(cfg) == {
        "enabled": True, "send": True, "decided": True, "reask": False,
    }
    target = resolve_send_config(cfg)
    assert target.endpoint == DEFAULT_ENDPOINT
    assert target.send is True


def test_existing_custom_stage_endpoint_remains_allowed():
    cfg = shared(enabled=True, send=True, endpoint="http://127.0.0.1:8080/v1/telemetry")
    resolved = resolve_send_config(cfg)
    assert resolved.send is True
    assert resolved.endpoint.startswith("http://127.0.0.1")


def test_new_default_without_consent_is_off():
    assert resolve_send_config({}).send is False
    cfg = shared(enabled=True, send=True)
    assert resolve_send_config(cfg).send is False
    cfg["telemetry"]["shared_metrics"]["offer_version"] = OFFER_VERSION
    assert resolve_send_config(cfg).send is True


def test_recipient_boundary_excludes_old_closed_windows_and_survives_restart(tmp_path):
    from datetime import UTC, datetime, timedelta
    from hermes_cli.observability.shared_metrics import SharedMetricsStore
    from hermes_cli.observability.shared_metrics_sender import (
        CONSENT_GATE_SQL, reconcile_send_consent, ensure_semantier_consent_boundary,
    )
    from hermes_cli.sqlite_util import write_txn

    store = SharedMetricsStore(database_path=tmp_path / 'metrics.db', outbox_directory=tmp_path / 'out')
    start = datetime(2026, 11, 1, tzinfo=UTC)
    stamp = lambda hours: (start + timedelta(hours=hours)).isoformat().replace('+00:00', 'Z')
    with store._connection() as c, write_txn(c):
        reconcile_send_consent(c, True, now=start)
        reconcile_send_consent(c, True, now=start + timedelta(hours=2))
        reconcile_send_consent(c, False, now=start + timedelta(hours=3))
        for pid, lo, hi in [('nous', 0, 1), ('gap', 3, 4), ('semantier', 5, 6)]:
            c.execute('INSERT INTO package_outbox(package_id, period_start, period_end, payload_json, created_at) VALUES (?, ?, ?, ?, ?)',
                      (pid, stamp(lo), stamp(hi), '{}', stamp(hi)))
        ensure_semantier_consent_boundary(c, now=start + timedelta(hours=5))
        reconcile_send_consent(c, True, now=start + timedelta(hours=5))
        reconcile_send_consent(c, True, now=start + timedelta(hours=7))
    with store._connection() as c, write_txn(c):
        ensure_semantier_consent_boundary(c, now=start + timedelta(hours=8))
        eligible = [r[0] for r in c.execute(f'SELECT package_id FROM package_outbox WHERE {CONSENT_GATE_SQL}')]
        assert eligible == ['semantier']
        assert c.execute('SELECT COUNT(*) FROM send_consent_windows').fetchone()[0] == 2


def test_old_endpoint_variants_cannot_bypass_recipient_migration():
    for endpoint in (NOUS + '/', 'https://TELEMETRY.NOUSRESEARCH.COM:443/v1/telemetry?old=1'):
        target = resolve_send_config(shared(enabled=True, send=True, endpoint=endpoint, offer_version=2))
        assert target.endpoint == DEFAULT_ENDPOINT
        assert target.send is False


def test_prior_declines_without_version_remain_settled():
    for enabled in (True, False):
        assert consent_state(shared(enabled=enabled, send=False)) == {
            'enabled': enabled, 'send': False, 'decided': True, 'reask': False,
        }
