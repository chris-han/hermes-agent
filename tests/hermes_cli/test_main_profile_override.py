import os
import sys

from hermes_cli import main as hermes_main


def test_apply_profile_override_ignores_pytest_plugin_short_flag(monkeypatch, tmp_path):
    """Pytest `-p <plugin>` must not be interpreted as Hermes profile shorthand."""
    monkeypatch.setattr(sys, "argv", ["pytest", "-p", "vscode_pytest", "--collect-only"])
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: tmp_path)

    called = {"value": False}

    def _resolver(_name: str) -> str:
        called["value"] = True
        return "/tmp/should-not-be-used"

    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", _resolver)

    hermes_main._apply_profile_override()

    assert called["value"] is False
    assert "HERMES_HOME" not in os.environ


def test_apply_profile_override_keeps_short_flag_for_hermes(monkeypatch, tmp_path):
    """Hermes CLI still supports `-p <profile>` when invoked as hermes."""
    monkeypatch.setattr(sys, "argv", ["hermes", "-p", "writer", "chat"])
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda name: f"/tmp/{name}")

    hermes_main._apply_profile_override()

    assert os.environ.get("HERMES_HOME") == "/tmp/writer"
    assert sys.argv == ["hermes", "chat"]
