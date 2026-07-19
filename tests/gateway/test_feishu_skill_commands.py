import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.skill_commands import expand_dynamic_skill_command, scan_skill_commands


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


def test_feishu_direct_message_command_preserves_shared_dynamic_skill_invocation(tmp_path):
    from plugins.platforms.feishu.adapter import normalize_feishu_message

    _make_skill(tmp_path, "llm-wiki", "Read workspace wiki pages.")
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        scan_skill_commands()
        normalized = normalize_feishu_message(
            message_type="text",
            raw_content=json.dumps({"text": "/llm-wiki Compare governance pages"}),
        )
        invocation = expand_dynamic_skill_command(
            normalized.text_content,
            task_id="feishu-direct",
        )

    assert normalized.text_content == "/llm-wiki Compare governance pages"
    assert invocation is not None
    assert invocation.command_key == "/llm-wiki"
    assert invocation.skill_name == "llm-wiki"
    assert invocation.user_instruction == "Compare governance pages"
    assert "Read workspace wiki pages." in invocation.expanded_message


def test_feishu_group_self_mention_stripping_preserves_leading_skill_command(tmp_path):
    from plugins.platforms.feishu.adapter import (
        FeishuMentionRef,
        _FeishuBotIdentity,
        _strip_edge_self_mentions,
        normalize_feishu_message,
    )

    _make_skill(tmp_path, "plan", "Plan work in ordered steps.")
    bot_mention = SimpleNamespace(
        key="@_user_1",
        id=SimpleNamespace(open_id="ou_bot", user_id=""),
        name="Hermes",
    )
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        scan_skill_commands()
        normalized = normalize_feishu_message(
            message_type="text",
            raw_content=json.dumps({"text": "@_user_1 /plan Design migration"}),
            mentions=[bot_mention],
            bot=_FeishuBotIdentity(open_id="ou_bot"),
        )
        command_text = _strip_edge_self_mentions(
            normalized.text_content,
            [FeishuMentionRef(name="Hermes", open_id="ou_bot", is_self=True)],
        )
        invocation = expand_dynamic_skill_command(command_text, task_id="feishu-group")

    assert normalized.text_content == "@Hermes /plan Design migration"
    assert command_text == "/plan Design migration"
    assert invocation is not None
    assert invocation.command_key == "/plan"
    assert invocation.user_instruction == "Design migration"
    assert "Plan work in ordered steps." in invocation.expanded_message


def test_feishu_ordinary_message_is_not_dynamic_skill_invocation(tmp_path):
    from plugins.platforms.feishu.adapter import normalize_feishu_message

    _make_skill(tmp_path, "llm-wiki")
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        scan_skill_commands()
        normalized = normalize_feishu_message(
            message_type="text",
            raw_content=json.dumps({"text": "please use /llm-wiki"}),
        )
        invocation = expand_dynamic_skill_command(
            normalized.text_content,
            task_id="feishu-text",
        )

    assert normalized.text_content == "please use /llm-wiki"
    assert invocation is None
