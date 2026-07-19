from pathlib import Path
from unittest.mock import patch

from agent.skill_commands import expand_dynamic_skill_command, scan_skill_commands
from gateway.platforms.base import MessageEvent


def _make_skill(skills_dir: Path, name: str, body: str = "Use this skill.") -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""\
---
name: {name}
description: Test {name}
---

# {name}

{body}
""",
        encoding="utf-8",
    )


def test_weixin_private_chat_command_preserves_shared_dynamic_skill_invocation(tmp_path):
    _make_skill(tmp_path, "llm-wiki", "Read workspace wiki pages.")
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        scan_skill_commands()
        event = MessageEvent(text="/llm-wiki Compare governance pages")
        invocation = expand_dynamic_skill_command(event.text, task_id="weixin-private")

    assert event.get_command() == "llm-wiki"
    assert event.get_command_args() == "Compare governance pages"
    assert invocation is not None
    assert invocation.command_key == "/llm-wiki"
    assert invocation.skill_name == "llm-wiki"
    assert invocation.user_instruction == "Compare governance pages"
    assert "Read workspace wiki pages." in invocation.expanded_message


def test_weixin_group_chat_command_uses_same_dynamic_skill_path(tmp_path):
    _make_skill(tmp_path, "plan", "Plan work in ordered steps.")
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        scan_skill_commands()
        event = MessageEvent(text="/plan Design migration", raw_message={"from_group": True})
        invocation = expand_dynamic_skill_command(event.text, task_id="weixin-group")

    assert event.get_command() == "plan"
    assert invocation is not None
    assert invocation.command_key == "/plan"
    assert invocation.user_instruction == "Design migration"
    assert "Plan work in ordered steps." in invocation.expanded_message


def test_weixin_ordinary_text_is_not_dynamic_skill_invocation(tmp_path):
    _make_skill(tmp_path, "llm-wiki")
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        scan_skill_commands()
        event = MessageEvent(text="Hermes /llm-wiki please")
        invocation = expand_dynamic_skill_command(event.text, task_id="weixin-text")

    assert event.get_command() is None
    assert invocation is None
