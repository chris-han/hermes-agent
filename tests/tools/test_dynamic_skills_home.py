import json


VALID_SKILL_CONTENT = """\
---
name: dynamic-skill
description: Dynamic HERMES_HOME regression test.
---

# Dynamic Skill

Use the active Hermes home.
"""


def test_skill_manager_uses_dynamic_hermes_home(monkeypatch, tmp_path):
    from tools import skill_manager_tool as skill_manager

    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(skill_manager, "_GUARD_AVAILABLE", False)

    result = json.loads(
        skill_manager.skill_manage(
            action="create",
            name="dynamic-skill",
            content=VALID_SKILL_CONTENT,
        )
    )

    assert result["success"] is True
    assert (hermes_home / "skills" / "dynamic-skill" / "SKILL.md").exists()


def test_skills_tool_uses_dynamic_hermes_home(monkeypatch, tmp_path):
    from tools.skills_tool import skill_view

    hermes_home = tmp_path / "hermes-home"
    skill_dir = hermes_home / "skills" / "dynamic-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(VALID_SKILL_CONTENT, encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = json.loads(skill_view("dynamic-skill"))

    assert result["success"] is True
    assert "Dynamic Skill" in result["content"]


def test_skills_sync_uses_dynamic_hermes_home(monkeypatch, tmp_path):
    from tools.skills_sync import sync_skills

    hermes_home = tmp_path / "hermes-home"
    bundled_dir = tmp_path / "bundled-skills"
    bundled_skill_dir = bundled_dir / "research" / "dynamic-skill"
    bundled_skill_dir.mkdir(parents=True)
    (bundled_skill_dir / "SKILL.md").write_text(VALID_SKILL_CONTENT, encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundled_dir))

    result = sync_skills(quiet=True)

    assert result["copied"] == ["dynamic-skill"]
    assert (hermes_home / "skills" / "research" / "dynamic-skill" / "SKILL.md").exists()
    assert (hermes_home / "skills" / ".bundled_manifest").exists()


def test_skills_hub_uses_dynamic_hermes_home(monkeypatch, tmp_path):
    from tools.skills_hub import HubLockFile, TapsManager, ensure_hub_dirs

    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    ensure_hub_dirs()

    hub_dir = hermes_home / "skills" / ".hub"
    assert hub_dir.exists()
    assert (hub_dir / "lock.json").exists()
    assert (hub_dir / "taps.json").exists()
    assert HubLockFile().path == hub_dir / "lock.json"
    assert TapsManager().path == hub_dir / "taps.json"