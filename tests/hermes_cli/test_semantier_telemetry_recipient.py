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
