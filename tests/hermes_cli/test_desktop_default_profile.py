"""Default Desktop startup never creates or destroys a named identity."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_cli.profiles_desktop import resolve_desktop_startup_profile


def test_cold_and_concurrent_startups_use_only_default(tmp_path, monkeypatch):
    owner = tmp_path / 'home'
    monkeypatch.setenv('HERMES_HOME', str(owner))
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: resolve_desktop_startup_profile(owner), range(4)))
    assert {result['profile'] for result in results} == {'default'}
    assert not (owner / 'profiles/semantier-local').exists()
    assert not (owner / 'profiles/tender-runtime-expert').exists()


def test_default_migration_keeps_root_and_named_profile_bytes(tmp_path, monkeypatch):
    from hermes_cli import profiles
    owner = tmp_path / 'home'
    owner.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(owner))
    for home in [owner, owner / 'profiles/work', owner / 'profiles/tender-runtime-expert']:
        home.mkdir(parents=True, exist_ok=True)
        for filename, content in {'SOUL.md': b'Personal identity', 'state.db': b'opaque sessions',
                                  '.cloud-device.bin': b'protected credentials', '.env': b'API_KEY=secret',
                                  'memories/MEMORY.md': b'user memories', 'config.yaml': b'{}\n'}.items():
            target = home / filename
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(content)
    before = {p: p.read_bytes() for p in owner.rglob('*') if p.is_file()}
    (owner / 'active_profile').write_text('semantier-local')
    result = resolve_desktop_startup_profile(owner, 'semantier-local')
    assert result['profile'] == 'default'
    assert all(path.read_bytes() == content for path, content in before.items())
    assert profiles.get_active_profile(owner) == 'default'
    assert resolve_desktop_startup_profile(owner)['profile'] == 'default'


def test_local_profile_is_migrated_with_real_sessions_and_home_bound_credentials(tmp_path, monkeypatch):
    import json
    from hermes_state import SessionDB
    from hermes_cli import cloud_device_client as device
    owner = tmp_path / 'home'
    local = owner / 'profiles/semantier-local'
    local.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(owner))
    (owner / 'SOUL.md').write_text('Original default persona')
    (local / 'SOUL.md').write_text('My local persona')
    (local / 'config.yaml').write_text('model:\n  default: saved-model\n')
    (local / '.env').write_text('SAVED_API_KEY=fixture-secret\n')
    (local / 'memories').mkdir()
    (local / 'memories/MEMORY.md').write_text('Saved memory')
    db = SessionDB(local / 'state.db')
    db.create_session('migrated-session', source='desktop')
    db.append_message('migrated-session', 'user', 'Preserve this conversation')
    db.close()
    # WSL-only storage seam; this does not qualify Windows DPAPI.
    monkeypatch.setattr(device, '_unprotect', lambda value: value)
    monkeypatch.setattr(device, '_protect', lambda value: value)
    state = {'home':str(local.resolve()),'status':'approved','profile_ref':'original-server-binding',
             'credentials':{'saved':{'credential':{'secret':'fixture-protected','expires_at':'2099-01-01T00:00:00+00:00'},
                                     'inference_origin':'https://cloud.example'}}}
    (local / '.cloud-device.bin').write_bytes(json.dumps(state).encode())
    assert resolve_desktop_startup_profile(owner)['profile'] == 'default'
    assert (owner / 'SOUL.md').read_text() == 'My local persona'
    assert (owner / '.env').read_text() == 'SAVED_API_KEY=fixture-secret\n'
    assert (owner / 'memories/MEMORY.md').read_text() == 'Saved memory'
    db = SessionDB(owner / 'state.db')
    assert db.get_messages('migrated-session')[0]['content'] == 'Preserve this conversation'
    db.close()
    migrated = json.loads((owner / '.cloud-device.bin').read_bytes())
    assert migrated['home'] == str(owner.resolve())
    assert migrated['profile_ref'] == 'original-server-binding'
    assert migrated['credentials'] == state['credentials']
    assert device.credential('saved', base_url='https://cloud.example/v1') == 'fixture-protected'
    assert not local.exists()
    assert (owner / '.desktop-default-migration/previous/SOUL.md').read_text() == 'Original default persona'
    import gateway.status
    monkeypatch.setattr(gateway.status, '_is_gateway_runtime_lock_active_strict', lambda *_: True)
    assert resolve_desktop_startup_profile(owner)['profile'] == 'default'


def test_interrupted_migration_resumes_without_overwriting_the_original_backup(tmp_path, monkeypatch):
    owner = tmp_path / 'home'
    local = owner / 'profiles/semantier-local'
    local.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(owner))
    (owner / '.env').write_text('ROOT_KEY=original')
    (local / '.env').write_text('LOCAL_KEY=retained')
    (local / 'config.yaml').write_text('{}')
    rename = Path.rename
    def interrupted(path, target):
        if path.name == 'config.yaml' and path.parent.name.startswith('staging-'):
            raise OSError('interrupted migration')
        return rename(path, target)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'rename', interrupted)
        with pytest.raises(OSError, match='interrupted'):
            resolve_desktop_startup_profile(owner)
    assert local.is_dir()
    assert (local / '.env').read_text() == 'LOCAL_KEY=retained'
    assert (owner / '.desktop-default-migration/previous/.env').read_text() == 'ROOT_KEY=original'
    assert resolve_desktop_startup_profile(owner)['profile'] == 'default'
    assert (owner / '.env').read_text() == 'LOCAL_KEY=retained'
    assert (owner / '.desktop-default-migration/previous/.env').read_text() == 'ROOT_KEY=original'


@pytest.mark.parametrize('failure', ['linked', 'running', 'protected'])
def test_migration_refuses_unsafe_or_unavailable_inputs_without_changing_either_home(tmp_path, monkeypatch, failure):
    from hermes_cli import cloud_device_client as device
    import gateway.status
    owner = tmp_path / 'home'
    local = owner / 'profiles/semantier-local'
    local.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(owner))
    (owner / 'SOUL.md').write_text('Root user persona')
    (local / 'config.yaml').write_text('{}')
    if failure == 'linked':
        outside = tmp_path / 'outside'
        outside.write_text('User data')
        (local / '.env').symlink_to(outside)
    elif failure == 'running':
        monkeypatch.setattr(gateway.status, '_is_gateway_runtime_lock_active_strict', lambda *_: True)
    else:
        (local / '.cloud-device.bin').write_bytes(b'protected-original')
        def unavailable(*_):
            raise ValueError('Windows protected storage is unavailable')
        monkeypatch.setattr(device, '_unprotect', unavailable)
    with pytest.raises((ValueError, RuntimeError)):
        resolve_desktop_startup_profile(owner)
    assert (owner / 'SOUL.md').read_text() == 'Root user persona'
    assert (local / 'config.yaml').read_text() == '{}'
    assert local.exists()


def test_migration_excludes_gateway_startup_until_publication(tmp_path, monkeypatch):
    from gateway.runtime_ownership import ProfileOwnership, OwnershipConflict
    from hermes_cli import profiles_desktop_migration as migration
    owner = tmp_path / 'home'
    local = owner / 'profiles/semantier-local'
    local.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(owner))
    (local / 'config.yaml').write_text('{}', encoding='utf-8')
    original = migration._snapshot
    checked = []
    def snapshot(source, target):
        contender = ProfileOwnership()
        try:
            with pytest.raises(OwnershipConflict):
                contender.reserve([owner])
            checked.append(True)
        finally:
            contender.close()
        return original(source, target)
    monkeypatch.setattr(migration, '_snapshot', snapshot)
    resolve_desktop_startup_profile(owner)
    assert checked


def test_resume_does_not_adopt_user_changes_as_an_applied_snapshot(tmp_path, monkeypatch):
    owner = tmp_path / 'home'
    local = owner / 'profiles/semantier-local'
    local.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(owner))
    (local / '.env').write_text('LOCAL_KEY=retained', encoding='utf-8')
    (local / 'config.yaml').write_text('{}', encoding='utf-8')
    rename = Path.rename
    def interrupted(path, target):
        if path.name == 'config.yaml' and path.parent.name.startswith('staging-'):
            raise OSError('interrupted')
        return rename(path, target)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'rename', interrupted)
        with pytest.raises(OSError):
            resolve_desktop_startup_profile(owner)
    (owner / '.env').write_text('NEW_USER_KEY=must-survive', encoding='utf-8')
    with pytest.raises(RuntimeError, match='changed|snapshot'):
        resolve_desktop_startup_profile(owner)
    assert local.exists()
    assert (local / '.env').read_text(encoding='utf-8-sig') == 'LOCAL_KEY=retained'
    assert (owner / '.env').read_text(encoding='utf-8-sig') == 'NEW_USER_KEY=must-survive'


def test_migration_preserves_homes_while_an_interactive_profile_writer_is_running(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import psutil
    owner = tmp_path / 'home'
    local = owner / 'profiles/semantier-local'
    local.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(owner))
    (local / 'config.yaml').write_text('{}', encoding='utf-8')
    (owner / 'SOUL.md').write_text('Root persona', encoding='utf-8')
    current = psutil.Process()
    writer = SimpleNamespace(pid=99999999,
        info={'pid': 99999999, 'name': 'hermes', 'username': current.username(),
              'cmdline': ['hermes', '--profile', 'semantier-local', 'chat']},
        environ=lambda: {'HERMES_HOME': str(local)})
    monkeypatch.setattr(psutil, 'process_iter', lambda *_: iter([writer]))
    with pytest.raises(RuntimeError, match='running|owned|writer'):
        resolve_desktop_startup_profile(owner)
    assert local.is_dir()
    assert (owner / 'SOUL.md').read_text(encoding='utf-8-sig') == 'Root persona'


def test_other_home_default_writer_does_not_block_migration(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import psutil
    from hermes_cli.profiles_desktop_migration import _require_no_profile_writer
    current = psutil.Process()
    foreign = tmp_path / 'other-install'
    writer = SimpleNamespace(pid=99999999,
        info={'pid': 99999999, 'name': 'hermes', 'username': current.username(),
              'cmdline': ['hermes', '--profile', 'default', 'gateway', 'run']},
        environ=lambda: {'HERMES_HOME': str(foreign)})
    monkeypatch.setattr(psutil, 'process_iter', lambda *_: iter([writer]))
    _require_no_profile_writer(tmp_path / 'migration-owner')
    with pytest.raises(RuntimeError, match='owned'):
        _require_no_profile_writer(foreign)
