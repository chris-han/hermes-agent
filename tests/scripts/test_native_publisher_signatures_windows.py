"""Native publisher trust checks must not require a Linux path bridge."""
import subprocess

import pytest

from scripts.releases.simple_native_slots import verify_windows_signatures


@pytest.mark.platforms('windows')
def test_native_verifier_uses_windows_paths_without_private_environment(tmp_path, monkeypatch):
    package = tmp_path / 'signed package.msix'
    package.write_bytes(b'fixture')
    monkeypatch.setenv('CLOUDFLARE_R2_SECRET_ACCESS_KEY', 'private-fixture')
    monkeypatch.setenv('SEMANTIER_CLIENT_RECEIPT_KEY', 'private-fixture')
    monkeypatch.setenv('PSModulePath', 'incompatible-parent-modules')
    calls = []

    def forbid_bridge(*args, **kwargs):
        raise AssertionError('Native verification must not execute wslpath')

    def record(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(subprocess, 'check_output', forbid_bridge)
    monkeypatch.setattr(subprocess, 'run', record)
    verify_windows_signatures([package], 'A' * 40)
    assert len(calls) == 1
    command, options = calls[0]
    assert command[0].lower().endswith('powershell.exe')
    assert '-EncodedCommand' in command
    assert options['check'] is True
    environment = {name.upper(): value for name, value in options['env'].items()}
    assert 'CLOUDFLARE_R2_SECRET_ACCESS_KEY' not in environment
    assert 'SEMANTIER_CLIENT_RECEIPT_KEY' not in environment
    assert 'PSMODULEPATH' not in environment
