"""Desktop startup resolution uses the real profile lifecycle, before gateway boot."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_cli import profiles


@pytest.fixture
def owner(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def resolve(owner, profile=None, explicit=False):
    from hermes_cli.profiles_desktop import resolve_desktop_startup_profile
    return resolve_desktop_startup_profile(owner, profile, explicit=explicit)


def test_saved_profile_keeps_data_and_credentials(owner):
    home = profiles.create_profile("work", no_alias=True, no_skills=True)
    from hermes_cli.config import atomic_config_replace
    atomic_config_replace(home / "config.yaml", {"model": {"provider": "custom", "default": "retained"}})
    secret = home / ".cloud-device.bin"
    secret.write_bytes(b"opaque protected credential")
    before = (home / "config.yaml").read_bytes()
    assert resolve(owner, "work")["profile"] == "work"
    assert secret.read_bytes() == b"opaque protected credential"
    assert (home / "config.yaml").read_bytes() == before
    assert not (owner / "profiles/semantier-local").exists()
    assert resolve(owner, 'work', explicit=True)['profile'] == 'work'
    profiles.set_active_profile('work')
    assert resolve(owner)['profile'] == 'work'


def test_stale_ordinary_reference_creates_one_fallback_concurrently(owner):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: resolve(owner, "registered-0026"), range(4)))
    assert {r["profile"] for r in results} == {"semantier-local"}
    assert not (owner / "profiles/registered-0026").exists()
    assert [p.name for p in (owner / "profiles").iterdir() if p.is_dir()] == ["semantier-local"]
    assert resolve(owner)["profile"] == "semantier-local"


def test_deliberate_missing_profile_does_not_redirect(owner):
    with pytest.raises(ValueError, match="missing"):
        resolve(owner, "missing", explicit=True)
    assert not (owner / "profiles/semantier-local").exists()


def test_tombstoned_selection_is_not_adopted(owner):
    from hermes_constants import mark_named_profile_deleted
    home = profiles.create_profile("work", no_alias=True, no_skills=True)
    mark_named_profile_deleted(home)
    with pytest.raises((ValueError, FileExistsError), match="tombstone|deleted"):
        resolve(owner, "work")
    assert not (owner / "profiles/semantier-local").exists()


@pytest.mark.parametrize('retired', ['ordinary-retired', 'semantier-local'])
def test_removed_ordinary_reference_can_fall_back_but_retired_fallback_is_never_recreated(owner, retired):
    import shutil
    from hermes_constants import mark_named_profile_deleted, profile_tombstone_path

    home = profiles.create_profile(retired, no_alias=True, no_skills=True)
    mark_named_profile_deleted(home)
    marker = profile_tombstone_path(home)
    previous = marker.read_bytes()
    shutil.rmtree(home)
    if retired == 'semantier-local':
        with pytest.raises(FileExistsError, match='tombstoned'):
            resolve(owner, 'ordinary-retired')
    else:
        assert resolve(owner, retired)['profile'] == 'semantier-local'
    assert marker.read_bytes() == previous
    assert not home.exists()


def test_corrupt_saved_profile_is_not_replaced(owner):
    home = profiles.create_profile("work", no_alias=True, no_skills=True)
    (home / "config.yaml").write_text("[broken", encoding="utf-8")
    with pytest.raises(Exception, match="config|parse|read"):
        resolve(owner, "work")
    assert (home / "config.yaml").read_text() == "[broken"
    assert not (owner / "profiles/semantier-local").exists()


def test_foreign_profile_namespace_is_not_adopted(owner, tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (owner / "profiles").symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match="owner"):
        resolve(owner)
    assert not list(foreign.iterdir())


def test_unreadable_profile_config_is_not_replaced_by_a_fallback(owner):
    from hermes_cli.config import atomic_config_replace
    home = profiles.create_profile('work', no_alias=True, no_skills=True)
    config = home / 'config.yaml'
    atomic_config_replace(config, {'model': {'provider': 'custom', 'default': 'retained'}})
    previous = config.read_bytes()
    config.chmod(0)
    try:
        with pytest.raises(Exception, match='cannot be read'):
            resolve(owner, 'work')
        assert not (owner / 'profiles/semantier-local').exists()
    finally:
        config.chmod(0o600)
    assert config.read_bytes() == previous


def test_generated_internal_profile_is_retired_with_exact_recoverable_data(owner):
    from hermes_cli.config import atomic_config_replace
    legacy = profiles.create_profile('tender-runtime-expert', no_alias=True, no_skills=True)
    atomic_config_replace(legacy / 'config.yaml', {
        'display': {'language': 'en'}, 'model': {'provider': 'custom', 'default': 'saved'},
        'plugins': {'enabled': ['tender_contextgraph_runtime']},
    })
    for name, data in {'sessions/chat.json': b'preserved chat', '.env': b'CUSTOM_API_KEY=fixture',
                       'memories/MEMORY.md': b'preserved memory', 'state.db': b'opaque fixture database', 'gateway.lock': b''}.items():
        target = legacy / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    before = {str(p.relative_to(legacy)): p.read_bytes() for p in legacy.rglob('*') if p.is_file()}
    profiles.set_active_profile('tender-runtime-expert')
    result = resolve(owner, 'tender-runtime-expert')
    assert result['profile'] == 'semantier-local'
    assert not legacy.exists()
    retained = list((owner / '.desktop-retired-profiles').glob('*/tender-runtime-expert'))
    assert len(retained) == 1
    assert {str(p.relative_to(retained[0])): p.read_bytes() for p in retained[0].rglob('*') if p.is_file()} == before
    from hermes_constants import named_profile_is_deleted
    assert named_profile_is_deleted(legacy)
    assert profiles.get_active_profile(owner) == 'semantier-local'
    assert 'tender-runtime-expert' not in profiles.list_profile_names()
    assert profiles._load_yaml_dict(owner / 'profiles/semantier-local/config.yaml')['display']['language'] == 'en'
    assert resolve(owner)['profile'] == 'semantier-local'
    assert len(list((owner / '.desktop-retired-profiles').iterdir())) == 1


def test_retirement_preserves_existing_local_profile_and_its_language(owner):
    from hermes_cli.config import atomic_config_replace
    local = profiles.create_profile('semantier-local', no_alias=True, no_skills=True)
    atomic_config_replace(local / 'config.yaml', {'display': {'language': 'zh'}, 'model': {'default': 'local-choice'}})
    original = (local / 'config.yaml').read_bytes()
    legacy = profiles.create_profile('tender-runtime-expert', no_alias=True, no_skills=True)
    atomic_config_replace(legacy / 'config.yaml', {'display': {'language': 'en'}})
    assert resolve(owner, 'tender-runtime-expert')['profile'] == 'semantier-local'
    assert (local / 'config.yaml').read_bytes() == original
    assert not legacy.exists()


def test_retirement_refuses_a_live_legacy_owner_without_mutation(owner, monkeypatch):
    legacy = profiles.create_profile('tender-runtime-expert', no_alias=True, no_skills=True)
    import gateway.status
    monkeypatch.setattr(gateway.status, '_is_gateway_runtime_lock_active_strict', lambda *a, **kw: True)
    with pytest.raises(RuntimeError, match='running|owned'):
        resolve(owner, 'tender-runtime-expert')
    assert legacy.exists()
    assert not (owner / '.desktop-retired-profiles').exists()


def test_fresh_desktop_discovers_only_its_local_profile_but_keeps_real_default_data(owner):
    assert resolve(owner)['profile'] == 'semantier-local'
    assert [p.name for p in profiles.list_profiles(lazy_skill_count=True)] == ['semantier-local']
    (owner / 'state.db').write_bytes(b'retained default chat store')
    assert resolve(owner, 'default', explicit=True)['profile'] == 'default'
    assert {p.name for p in profiles.list_profiles(lazy_skill_count=True)} == {'default', 'semantier-local'}


def test_fallback_discovery_uses_live_profile_identity(owner):
    from hermes_cli.web_server_profiles import _fallback_profile_dicts
    resolve(owner)
    assert [p['name'] for p in _fallback_profile_dicts(profiles)] == ['semantier-local']
    ghost = owner / 'profiles/tender-runtime-expert'
    ghost.mkdir()
    (ghost / 'logs').mkdir()
    assert [p['name'] for p in _fallback_profile_dicts(profiles)] == ['semantier-local']


@pytest.mark.parametrize('namespace', ['.desktop-retired-profiles', 'profiles/.deleted'])
def test_retirement_refuses_linked_backup_or_marker_namespace(owner, tmp_path, namespace):
    legacy = profiles.create_profile('tender-runtime-expert', no_alias=True, no_skills=True)
    foreign = tmp_path / 'outside'
    foreign.mkdir()
    (owner / namespace).symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match='owner'):
        resolve(owner, 'tender-runtime-expert')
    assert legacy.exists()
    assert not list(foreign.iterdir())


def test_interrupted_retirement_resumes_same_backup(owner, monkeypatch):
    legacy = profiles.create_profile('tender-runtime-expert', no_alias=True, no_skills=True)
    original = Path.rename
    with monkeypatch.context() as patch:
        def interrupted(source, target):
            if source == legacy:
                raise SystemExit('interrupted before directory move')
            return original(source, target)
        patch.setattr(Path, 'rename', interrupted)
        with pytest.raises(SystemExit):
            resolve(owner, 'tender-runtime-expert')
    assert legacy.exists()
    assert resolve(owner, 'tender-runtime-expert')['profile'] == 'semantier-local'
    assert not legacy.exists()
    assert len(list((owner / '.desktop-retired-profiles').iterdir())) == 1


def test_dangling_retirement_marker_cannot_create_a_foreign_file(owner, tmp_path):
    from hermes_constants import profile_tombstone_path
    legacy = profiles.create_profile('tender-runtime-expert', no_alias=True, no_skills=True)
    marker = profile_tombstone_path(legacy)
    marker.parent.mkdir(exist_ok=True)
    foreign = tmp_path / 'foreign-marker'
    marker.symlink_to(foreign)
    with pytest.raises(ValueError, match='owner'):
        resolve(owner, 'tender-runtime-expert')
    assert not foreign.exists()
    assert legacy.exists()


def test_remembered_empty_default_resolves_to_desktop_local(owner):
    assert resolve(owner, 'default')['profile'] == 'semantier-local'
    assert [p.name for p in profiles.list_profiles(lazy_skill_count=True)] == ['semantier-local']


def test_interactive_profile_writer_prevents_retirement(owner):
    import os
    import subprocess
    import sys
    legacy = profiles.create_profile('tender-runtime-expert', no_alias=True, no_skills=True)
    process = subprocess.Popen([sys.executable, '-c', "print('ready', flush=True); input()", "hermes_cli.main"],
                               env={**os.environ, 'HERMES_HOME': str(legacy)},
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        with pytest.raises(RuntimeError, match='owned'):
            resolve(owner, 'tender-runtime-expert')
        assert legacy.exists()
        assert not (owner / '.desktop-retired-profiles').exists()
    finally:
        process.communicate('finish\n', timeout=10)


def test_restart_after_move_repairs_sticky_owner_route(owner):
    profiles.create_profile('tender-runtime-expert', no_alias=True, no_skills=True)
    resolve(owner, 'tender-runtime-expert')
    (owner / 'active_profile').write_text('tender-runtime-expert\n')
    assert resolve(owner)['profile'] == 'semantier-local'
    assert profiles.get_active_profile(owner) == 'semantier-local'


def test_host_bootstrap_files_do_not_create_a_second_desktop_identity(owner):
    from hermes_cli.config import atomic_config_replace
    from hermes_state import SessionDB
    assert resolve(owner)['profile'] == 'semantier-local'
    # The actual packaged host bootstrap creates these after profile resolution.
    atomic_config_replace(owner/'config.yaml', {'model': {'default': 'semantier-runtime'}})
    (owner/'SOUL.md').write_text('Generated host scaffold\n')
    db = SessionDB(owner/'state.db'); db.close()
    before = {name: (owner/name).read_bytes() for name in ['config.yaml', 'SOUL.md', 'state.db']}
    assert [p.name for p in profiles.list_profiles(lazy_skill_count=True)] == ['semantier-local']
    assert resolve(owner, 'semantier-local')['profile'] == 'semantier-local'
    assert [p.name for p in profiles.list_profiles(lazy_skill_count=True)] == ['semantier-local']
    assert {name: (owner/name).read_bytes() for name in before} == before


def test_existing_unclassified_default_data_stays_visible_and_unchanged(owner):
    (owner/'config.yaml').write_text('display:\n  language: en\n')
    (owner/'state.db').write_bytes(b'legacy retained chat store')
    before = (owner/'state.db').read_bytes()
    assert resolve(owner)['profile'] == 'semantier-local'
    assert {p.name for p in profiles.list_profiles(lazy_skill_count=True)} == {'default', 'semantier-local'}
    assert resolve(owner, 'default', explicit=True)['profile'] == 'default'
    assert (owner/'state.db').read_bytes() == before
    assert 'language: en' in (owner/'config.yaml').read_text()


@pytest.mark.parametrize('claim', ['launch', 'sticky'])
def test_deliberate_default_claim_keeps_existing_bytes_and_selection(owner, claim):
    resolve(owner)
    (owner/'config.yaml').write_text('display:\n  language: en\n')
    (owner/'state.db').write_bytes(b'retained default conversations')
    before = {p.name: p.read_bytes() for p in [owner/'config.yaml', owner/'state.db']}
    assert [p.name for p in profiles.list_profiles(lazy_skill_count=True)] == ['semantier-local']
    if claim == 'launch':
        assert resolve(owner, 'default', explicit=True)['profile'] == 'default'
    else:
        profiles.set_active_profile('default')
        assert resolve(owner)['profile'] == 'default'
    assert {p.name for p in profiles.list_profiles(lazy_skill_count=True)} == {'default', 'semantier-local'}
    assert resolve(owner, 'default')['profile'] == 'default'
    assert {name: (owner/name).read_bytes() for name in before} == before


@pytest.mark.parametrize('body', ['{invalid', '{"schema":2,"role":"runtime_owner"}'])
def test_unknown_owner_role_holds_without_rewriting_data(owner, body):
    resolve(owner)
    marker = owner/'profiles/.desktop-owner-root.json'
    marker.write_text(body)
    with pytest.raises(RuntimeError, match='role requires recovery'):
        profiles.list_profiles(lazy_skill_count=True)
    with pytest.raises(RuntimeError, match='role requires recovery'):
        resolve(owner, 'semantier-local')
    assert marker.read_text() == body


def test_linked_owner_role_cannot_reclassify_a_profile(owner, tmp_path):
    resolve(owner)
    marker = owner/'profiles/.desktop-owner-root.json'
    marker.unlink()
    foreign = tmp_path/'foreign-role.json'; foreign.write_text('{"schema":1,"role":"runtime_owner"}')
    marker.symlink_to(foreign)
    with pytest.raises(ValueError, match='escapes its owner'):
        profiles.list_profiles(lazy_skill_count=True)
    assert foreign.read_text() == '{"schema":1,"role":"runtime_owner"}'
