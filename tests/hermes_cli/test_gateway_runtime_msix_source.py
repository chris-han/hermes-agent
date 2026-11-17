"""A same-commit Gateway in a retired MSIX is not a reusable runtime."""
import pytest

from hermes_cli import gateway_runtime as runtime
from hermes_cli import gateway_runtime_replace as replace

pytestmark = pytest.mark.platforms('windows')


@pytest.fixture
def native_owner(tmp_path, monkeypatch):
    from gateway import status
    home = tmp_path / 'profile'
    home.mkdir()
    store = tmp_path / 'WindowsApps'
    current = store / 'Semantier.Agent.Acceptance_0.0.114.0_x64__publisher'
    old = store / 'Semantier.Agent.Acceptance_0.0.113.0_x64__publisher'
    image = old / 'app/python.exe'
    state = {'image': image, 'birth': 12345, 'family': 'Semantier.Agent.Acceptance_publisher'}

    # Only native OS image/registration reads are substituted. Discovery,
    # ownership validation and replacement policy are production functions.
    def source(executable, *, registered):
        root = current if registered else state['image'].parent.parent
        version = tuple(map(int, root.name.split('_')[1].split('.')))
        return root, state['family'] if not registered else 'Semantier.Agent.Acceptance_publisher', version, 'x64_'

    monkeypatch.setattr(replace, '_windows_package_source', source, raising=False)
    monkeypatch.setattr(replace, '_native_process_source', lambda pid: state['image'], raising=False)
    monkeypatch.setattr(replace, '_install_root', lambda: home)
    monkeypatch.setattr(status, '_pid_exists', lambda pid: pid == 424242)
    monkeypatch.setattr(status, 'get_process_start_time', lambda pid: state['birth'])
    payload = {'runtime_protocol': 1, 'semantier_runtime': 0, 'state': 'ready',
               'kind': 'hermes-gateway', 'pid': 424242, 'start_time': 12345,
               'hermes_home': str(home), 'served_profiles': [{'home': str(home), 'profile_id': str(home)}],
               'capabilities': ['session-authority-v1'], 'authority_epoch': 1,
               'instance_id': 'owned-gateway', 'api_origin': 'http://127.0.0.1:41817',
               'supervisor': 'none', 'code_sha': 'a' * 40}
    from hermes_cli import semantier_runtime
    monkeypatch.setattr(semantier_runtime, 'cloud_enabled', lambda: False)
    return home, current, state, payload


def test_same_commit_gateway_in_retired_msix_is_not_ready(native_owner):
    home, _, _, payload = native_owner
    result = runtime._endpoint(payload, home)
    assert (result.state, result.reason_code) == ('incompatible', 'native_package_generation')


@pytest.mark.parametrize(('case', 'expected'), [
    ('same', 'ready'), ('foreign_family', 'incompatible'), ('foreign_store', 'incompatible'),
    ('newer', 'incompatible'), ('reused_pid', 'inaccessible'), ('foreign_profile', 'incompatible')])
def test_native_reuse_and_refusal_boundaries(native_owner, case, expected):
    home, current, state, payload = native_owner
    changes = {
        'same': {'image': current / 'app/python.exe'},
        'foreign_family': {'family': 'Other.Agent_publisher',
                           'image': current.parent / 'Other.Agent_0.0.113.0_x64__publisher/app/python.exe'},
        'foreign_store': {'image': current.parent.parent / 'foreign' / current.name / 'app/python.exe'},
        'newer': {'image': current.parent / current.name.replace('114', '115') / 'app/python.exe'},
        'reused_pid': {'birth': state['birth'] + 100}, 'foreign_profile': {}}
    state.update(changes[case])
    payload['hermes_home'] = str({'foreign_profile': home.parent / 'foreign-profile'}.get(case, home))
    assert runtime._endpoint(payload, home).state == expected


def test_native_generation_uses_existing_bounded_replacement(native_owner, monkeypatch):
    home, _, _, payload = native_owner
    observations = [runtime._endpoint(payload, home), runtime.GatewayDiscovery('absent')]
    monkeypatch.setattr(runtime, 'discover_gateway_endpoint', lambda *a, **kw: observations.pop(0))
    stopped = []
    monkeypatch.setattr(replace, 'replace_incompatible_owner', lambda h, *, deadline: stopped.append(h))
    assert runtime._observe_replacing_stale(home, 9999999999, set()).reason_code == 'replaced'
    assert stopped == [home]


def test_replacement_rechecks_package_before_stopping(native_owner, monkeypatch):
    import time
    home, current, state, payload = native_owner
    state['image'] = current / 'app/python.exe'
    monkeypatch.setattr(replace, '_identify', lambda *a, **kw: (home, payload))
    stops = []
    monkeypatch.setattr(replace, '_write_planned_stop', lambda *a: stops.append(a))
    monkeypatch.setattr(replace, '_wait_gone', lambda *a: True)
    assert replace.replace_incompatible_owner(home, deadline=time.monotonic() + 5) is None
    assert stops == []


def test_valid_external_profile_reuse_does_not_authorize_replacement(native_owner, monkeypatch):
    home, current, state, payload = native_owner
    monkeypatch.setattr(replace, '_install_root', lambda: home.parent / 'another-install')
    state['image'] = current / 'app/python.exe'
    assert runtime._endpoint(payload, home).state == 'ready'
    state['image'] = current.parent / current.name.replace('114', '113') / 'app/python.exe'
    assert runtime._endpoint(payload, home).reason_code == 'native_package_foreign_install'


def test_native_probe_refuses_an_actual_unrelated_process_without_stopping_it():
    import subprocess
    import sys
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        with pytest.raises(ValueError, match='process role mismatch'):
            replace._native_process_source(process.pid)
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_explicit_package_handoff_drains_same_package_through_existing_owner(native_owner, monkeypatch):
    import time
    home, current, state, payload = native_owner
    state['image'] = current / 'app/python.exe'
    monkeypatch.setattr(replace, '_identify', lambda *a, **kw: (home, payload))
    stops = []
    monkeypatch.setattr(replace, '_write_planned_stop', lambda *a: stops.append(a))
    monkeypatch.setattr(replace, '_wait_gone', lambda *a: True)
    assert replace.replace_incompatible_owner(home, deadline=time.monotonic() + 5, for_package_update=True) is None
    assert len(stops) == 1 and stops[0][1] == payload['pid']


@pytest.mark.parametrize(('refusal', 'state', 'expected'), [
    (None, 'absent', 0), (None, 'ready', 2), ('foreign_install', 'absent', 2)])
def test_package_stop_requires_confirmed_absence(native_owner, monkeypatch, refusal, state, expected):
    from types import SimpleNamespace
    from hermes_cli import gateway_runtime_cli as cli
    import hermes_constants
    home, _, _, _ = native_owner
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: home)
    monkeypatch.setattr(replace, 'replace_incompatible_owner', lambda *a, **kw: refusal)
    monkeypatch.setattr(runtime, 'discover_gateway_endpoint', lambda *a, **kw: runtime.GatewayDiscovery(state))
    if expected:
        with pytest.raises(SystemExit) as exit_info:
            cli.cmd_gateway_package_stop(SimpleNamespace(all=False, system=False))
        assert exit_info.value.code == expected
    else:
        assert cli.cmd_gateway_package_stop(SimpleNamespace(all=False, system=False)) is None


def test_package_stop_does_not_allow_global_sweep(native_owner, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.gateway_runtime_cli import cmd_gateway_package_stop
    stops = []
    monkeypatch.setattr(replace, 'replace_incompatible_owner', lambda *a, **kw: stops.append(a))
    with pytest.raises(SystemExit):
        cmd_gateway_package_stop(SimpleNamespace(all=True, system=False))
    assert stops == []
