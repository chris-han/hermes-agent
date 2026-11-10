"""Browser resolution uses installed PM facts without provisioning a browser."""

import os

import pytest

import pm
from pm import paths


@pytest.mark.parametrize("record_executable", [False, True])
def test_chromium_resolves_installed_binary_without_mutation(tmp_path, monkeypatch, record_executable):
    from hermes_cli.browser_runtime import chromium_executable

    home = tmp_path / "home"
    store = home / "tools"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
    lock = pm.Lockfile(paths.lockfile_path())
    target = pm.current_target()
    package = pm.get_package("chromium")
    version = lock.version("chromium")
    assert version is not None
    entry = store / package.store_entry(version, target)
    binary = entry / "browser" / ("chrome.exe" if target.startswith("win32") else "chrome")
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"browser-fixture")
    binary.chmod(0o755)
    env = {"PLAYWRIGHT_BROWSERS_PATH": str(store)}
    if record_executable:
        env["AGENT_BROWSER_EXECUTABLE_PATH"] = str(binary)
    facts = pm.Facts(store / "facts.json")
    facts.record(
        "chromium", version, entry.name, env, store,
        target=target, artifacts=[a["sha256"] for a in lock.artifacts("chromium", target)],
    )
    before = facts.path.read_bytes()
    environment = dict(os.environ)

    assert chromium_executable() == str(binary)
    assert facts.path.read_bytes() == before
    assert dict(os.environ) == environment


def test_chromium_override_wins_without_installing(tmp_path, monkeypatch):
    from hermes_cli.browser_runtime import chromium_executable

    store = tmp_path / "missing-store"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    override = str(tmp_path / "external browser")
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", override)

    def forbidden(*args, **kwargs):
        pytest.fail("browser resolution must not provision or activate packages")

    monkeypatch.setattr(pm, "ensure", forbidden)
    monkeypatch.setattr(pm, "activate", forbidden)
    with monkeypatch.context() as scoped:
        scoped.setattr(pm, "installed_package", forbidden)
        assert chromium_executable() == override
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH")
    assert chromium_executable() is None
    assert not store.exists()
    assert "AGENT_BROWSER_EXECUTABLE_PATH" not in os.environ


def test_builtin_default_and_explicit_system_choice_are_independent_of_discovery(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import browser_runtime, browser_connect, config
    builtin = tmp_path / 'chrome'; builtin.write_bytes(b'fixture')
    monkeypatch.delenv('AGENT_BROWSER_EXECUTABLE_PATH', raising=False)
    monkeypatch.setattr(pm, 'installed_package', lambda name: SimpleNamespace(binary=builtin))
    saved = {'browser': {}}
    monkeypatch.setattr(config, 'load_config', lambda: saved)
    monkeypatch.setattr(browser_runtime, 'system_browsers', lambda: [{'key': 'edge', 'executable': '/system/edge'}])
    monkeypatch.setattr(browser_connect, 'chromium_executable', lambda name: '/system/' + name)
    assert browser_runtime.chromium_executable() == str(builtin)
    saved['browser']['local_browser'] = 'edge'
    monkeypatch.setenv('AGENT_BROWSER_EXECUTABLE_PATH', str(builtin))
    assert browser_runtime.chromium_executable() == '/system/edge'
    assert browser_runtime.chromium_executable() == '/system/edge'
    assert 'use_real_profile' not in saved['browser']
    saved['browser'].clear()
    monkeypatch.delenv('AGENT_BROWSER_EXECUTABLE_PATH')
    builtin.unlink()
    assert browser_runtime.chromium_executable() == '/system/edge'
    monkeypatch.setattr(browser_runtime, 'system_browsers', list)
    assert browser_runtime.chromium_executable() is None
    saved['browser']['local_browser'] = 'edge'
    monkeypatch.setattr(browser_connect, 'chromium_executable', lambda name: None)
    with pytest.raises(RuntimeError, match='unavailable'):
        browser_runtime.chromium_executable()


def test_discovery_picker_persists_only_explicit_choice(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import browser_runtime, tools_config_providers as picker, config
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'profile'))
    monkeypatch.setattr(browser_runtime, 'system_browsers', lambda: [
        {'key': key, 'executable': '/fixture/' + key} for key in ('chrome', 'edge', 'brave', 'brave-origin', 'chromium')])
    monkeypatch.setitem(picker._PLUGIN_ROW_BUILDERS, 'Browser Automation', list)
    local = {'name': 'Local Browser', 'browser_provider': 'local', 'browser_engine': 'auto'}
    category = {'name': 'Browser Automation', 'providers': [local]}
    original = {'browser': {}}
    rows = picker._visible_providers(category, original, features=SimpleNamespace(account_info=None, nous_auth_present=False))
    assert len(rows) == 6
    assert original == {'browser': {}}
    assert picker._browser_provider_active(local, original)
    edge = next(row for row in rows if row.get('local_browser') == 'edge')
    picker._write_provider_config(edge, original, managed_feature=None)
    config.save_config(original)
    reloaded = config.load_config()
    assert reloaded['browser']['local_browser'] == 'edge'
    assert picker._browser_provider_active(edge, reloaded)
    assert not picker._browser_provider_active(local, reloaded)
    assert not reloaded['browser']['use_real_profile']
    # Repeated discovery is read-only and cannot erase the saved choice.
    picker._visible_providers(category, reloaded, features=SimpleNamespace(account_info=None, nous_auth_present=False))
    assert config.load_config()['browser']['local_browser'] == 'edge'


def test_installed_builtin_prevents_implicit_cloud_detection_but_keeps_explicit_cloud(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import config
    from tools import browser_tool_cloud as cloud
    binary = tmp_path / 'chrome'; binary.write_bytes(b'fixture')
    monkeypatch.setattr(pm, 'installed_package', lambda name: SimpleNamespace(binary=binary))
    owner = SimpleNamespace(_cached_cloud_provider=None, _cloud_provider_resolved=False)
    monkeypatch.setattr(cloud, '_origin', lambda: owner)
    raw = {}
    monkeypatch.setattr(config, 'read_raw_config', lambda: raw)
    monkeypatch.setattr(cloud, '_autodetect_cloud_provider', lambda: pytest.fail('installed local browser must beat implicit cloud discovery'))
    assert cloud._resolve_cloud_provider_uncached() is None
    raw['browser'] = {'cloud_provider': 'browserbase'}
    selected = object()
    monkeypatch.setattr(cloud, '_instantiate_explicit_cloud_provider', lambda name: selected)
    assert cloud._resolve_cloud_provider_uncached() is selected
