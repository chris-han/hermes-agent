"""Desktop startup resolution uses the real profile lifecycle, before gateway boot."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys

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


def test_semantier_desktop_soul_upgrades_only_stock_identity(owner):
    from hermes_cli.default_soul import DEFAULT_SOUL_MD

    local = profiles.create_profile("work", no_alias=True, no_skills=True)
    (local / "SOUL.md").write_text(DEFAULT_SOUL_MD, encoding="utf-8")
    assert resolve(owner, "work", explicit=True)["profile"] == "work"
    # A retained named profile is user-owned; only the default is initialized.
    assert (local / "SOUL.md").read_text(encoding="utf-8") == DEFAULT_SOUL_MD
    (owner / "SOUL.md").write_text(DEFAULT_SOUL_MD, encoding="utf-8")
    resolve(owner)
    actual = (owner / "SOUL.md").read_text(encoding="utf-8")
    from pm.paths import repo_root
    canonical = (repo_root() / "semantier-runtime/src/prompts/agents/SOUL.md").read_text(encoding="utf-8")
    assert actual == canonical
    assert "Nous Research" not in actual
    (local / "SOUL.md").write_text("My own persona", encoding="utf-8")
    resolve(owner, "work", explicit=True)
    assert (local / "SOUL.md").read_text(encoding="utf-8") == "My own persona"

    (owner / "config.yaml").write_text("{}", encoding="utf-8")
    (owner / "SOUL.md").write_text(DEFAULT_SOUL_MD, encoding="utf-8")
    assert resolve(owner, "default", explicit=True)["profile"] == "default"
    assert "You are Semantier Agent" in (owner / "SOUL.md").read_text(encoding="utf-8")



def test_deliberate_missing_profile_does_not_redirect(owner):
    with pytest.raises(ValueError, match="missing"):
        resolve(owner, "missing", explicit=True)
    assert not (owner / "profiles/semantier-local").exists()


def test_tombstoned_selection_is_not_adopted(owner):
    from hermes_constants import mark_named_profile_deleted
    home = profiles.create_profile("work", no_alias=True, no_skills=True)
    mark_named_profile_deleted(home)
    with pytest.raises((ValueError, FileExistsError), match="tombstone|deleted"):
        resolve(owner, "work", explicit=True)
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
    assert resolve(owner, retired)['profile'] == 'default'
    assert marker.read_bytes() == previous
    assert not home.exists()


def test_corrupt_saved_profile_is_not_replaced(owner):
    home = profiles.create_profile("work", no_alias=True, no_skills=True)
    (home / "config.yaml").write_text("[broken", encoding="utf-8")
    with pytest.raises(Exception, match="config|parse|read"):
        resolve(owner, "work", explicit=True)
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
            resolve(owner, 'work', explicit=True)
        assert not (owner / 'profiles/semantier-local').exists()
    finally:
        config.chmod(0o600)
    assert config.read_bytes() == previous




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


def test_actual_semantier_soul_enters_prompt_and_custom_persona_survives(owner):
    from agent.prompt_builder import load_soul_md
    from pm.paths import repo_root
    resolve(owner)
    canonical = (repo_root() / 'semantier-runtime/src/prompts/agents/SOUL.md').read_text(encoding='utf-8')
    prompt = load_soul_md(home_override=owner)
    assert canonical.strip() in prompt
    assert 'Nous Research' not in prompt
    (owner / 'SOUL.md').write_text('My own persona', encoding='utf-8')
    resolve(owner)
    assert 'My own persona' in load_soul_md(home_override=owner)


def test_saved_custom_selection_survives_ordinary_restart(owner):
    custom = profiles.create_profile('work', no_alias=True, no_skills=True)
    before = (custom / 'config.yaml').read_bytes()
    for _ in range(2):
        assert resolve(owner, 'work')['profile'] == 'work'
    assert (custom / 'config.yaml').read_bytes() == before


def test_explicit_legacy_selection_migrates_instead_of_reopening_it(owner):
    legacy = profiles.create_profile('semantier-local', no_alias=True, no_skills=True)
    (legacy / '.env').write_text('KEEP=fixture\n')
    assert resolve(owner, 'semantier-local', explicit=True)['profile'] == 'default'
    assert (owner / '.env').read_text() == 'KEEP=fixture\n'
    assert not legacy.exists()


def test_managed_identity_refresh_keeps_appended_user_instructions(owner):
    from hermes_cli.default_soul import DEFAULT_SOUL_MD
    addition = '\n\n## My instructions\nAlways use my project glossary.\n'
    (owner / 'SOUL.md').write_text(DEFAULT_SOUL_MD + addition)
    resolve(owner)
    actual = (owner / 'SOUL.md').read_text()
    assert 'You are Semantier Agent' in actual
    assert 'by Semantier' in actual
    assert 'Nous Research' not in actual
    assert actual.endswith(addition)
    resolve(owner)
    assert (owner / 'SOUL.md').read_text() == actual
