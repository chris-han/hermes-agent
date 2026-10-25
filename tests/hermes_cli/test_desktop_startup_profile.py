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
