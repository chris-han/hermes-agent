from pathlib import Path
import os
import pytest
import hermes_constants as homes
from hermes_cli import profiles

@pytest.fixture(autouse=True)
def reset_owner():
    yield
    if hasattr(homes,'pin_profile_owner_root'):homes.pin_profile_owner_root(None)

def test_explicit_nested_native_root_remains_explicit(monkeypatch,tmp_path):
    native=tmp_path/'native';owner=native/'local-client'
    monkeypatch.setattr(homes,'_get_platform_default_hermes_home',lambda:native)
    monkeypatch.setenv('HERMES_HOME',str(owner))
    assert homes.get_default_hermes_root()==owner

def test_owner_stays_stable_while_active_profile_changes(monkeypatch,tmp_path):
    owner=tmp_path/'local';owner.mkdir()
    monkeypatch.setenv('HERMES_HOME',str(owner))
    homes.pin_profile_owner_root(owner)
    one=profiles.create_profile('first',no_alias=True,no_skills=True)
    monkeypatch.setenv('HERMES_HOME',str(one))
    two=profiles.create_profile('second',no_alias=True,no_skills=True)
    assert two==owner/'profiles/second'
    assert profiles.get_profile_dir('default')==owner
    profiles.set_active_profile('second')
    assert (owner/'active_profile').read_text().strip()=='second'

def test_host_boundary_owner_overrides_process_and_requires_binding(tmp_path):
    from gateway.execution_boundary import BoundaryPaths,BoundaryPolicy,ExecutionBoundary,bind_execution_boundary
    homes.pin_profile_owner_root(tmp_path/'local')
    for name in ['a','b']:
        owner=tmp_path/name
        b=ExecutionBoundary('test',paths=BoundaryPaths(hermes_home=owner,profile_owner_root=owner),policy=BoundaryPolicy(require_boundary=True))
        with bind_execution_boundary(b):
            assert homes.get_default_hermes_root()==owner
            assert profiles.create_profile('shared',no_alias=True,no_skills=True)==owner/'profiles/shared'
    with bind_execution_boundary(ExecutionBoundary('test',policy=BoundaryPolicy(require_boundary=True))):
        with pytest.raises(RuntimeError,match='profile owner'):profiles.create_profile('blocked',no_alias=True,no_skills=True)

def test_ensure_profile_is_idempotent_but_create_still_refuses(monkeypatch,tmp_path):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path/'ambient'))
    owner=tmp_path/'owner'
    one=profiles.ensure_profile(owner,'retained',no_skills=True)
    (one/'memories/MEMORY.md').write_text('keep')
    assert profiles.ensure_profile(owner,'retained',no_skills=True)==one
    assert (one/'memories/MEMORY.md').read_text()=='keep'
    token=homes.set_profile_owner_root_override(owner)
    try:
        with pytest.raises(FileExistsError):profiles.create_profile('retained',no_alias=True,no_skills=True)
    finally:homes.reset_profile_owner_root_override(token)

def test_ensure_refuses_markerless_link_and_foreign_host(monkeypatch, tmp_path):
    owner = tmp_path / 'owner'
    namespace = owner / 'profiles'
    namespace.mkdir(parents=True)
    (namespace / 'invalid').mkdir()
    with pytest.raises(FileExistsError, match='valid identity'):
        profiles.ensure_profile(owner, 'invalid', no_skills=True)
    foreign = tmp_path / 'foreign'; foreign.mkdir()
    (namespace / 'linked').symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        profiles.ensure_profile(owner, 'linked', no_skills=True)
    from gateway.execution_boundary import BoundaryPaths, BoundaryPolicy, ExecutionBoundary, bind_execution_boundary
    with bind_execution_boundary(ExecutionBoundary('host', paths=BoundaryPaths(profile_owner_root=foreign), policy=BoundaryPolicy(require_boundary=True))):
        with pytest.raises(PermissionError, match='host binding'):
            profiles.ensure_profile(owner, 'blocked', no_skills=True)
    assert not (namespace / 'blocked').exists()


def test_arbitrary_profiles_shaped_explicit_root_is_not_a_named_home(monkeypatch, tmp_path):
    root = tmp_path / 'custom/profiles/local-client'
    monkeypatch.setenv('HERMES_HOME', str(root))
    assert homes.get_default_hermes_root() == root
    assert homes.get_default_hermes_root(home=root) == root


def test_existing_child_cli_transports_owner_and_selection(monkeypatch, tmp_path):
    import subprocess, sys
    owner = tmp_path / 'custom/profiles/local-client'
    owner.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(owner))
    profiles.ensure_profile(owner, 'worker', no_skills=True)
    script = """
import sys
sys.argv = ['hermes', '--profile', 'worker', '--help']
import hermes_cli.main
from hermes_constants import get_default_hermes_root, get_hermes_home
from pathlib import Path
owner=Path(__import__('os').environ['EXPECTED_OWNER'])
assert get_default_hermes_root() == owner
assert get_hermes_home() == owner/'profiles/worker'
"""
    env = {**os.environ, 'EXPECTED_OWNER': str(owner), 'PYTHONPATH': str(Path(profiles.__file__).resolve().parents[1])}
    subprocess.run([sys.executable, '-c', script], env=env, check=True, capture_output=True, text=True)

def test_ensure_never_resurrects_tombstoned_profile(monkeypatch, tmp_path):
    from hermes_constants import mark_named_profile_deleted
    owner = tmp_path / 'owner'
    monkeypatch.setenv('HERMES_HOME', str(owner))
    home = profiles.ensure_profile(owner, 'retained', no_skills=True)
    (home / 'memories/MEMORY.md').write_text('retained')
    mark_named_profile_deleted(home)
    with pytest.raises(FileExistsError, match='tombstoned'):
        profiles.ensure_profile(owner, 'retained', no_skills=True)
    assert (home / 'memories/MEMORY.md').read_text() == 'retained'
