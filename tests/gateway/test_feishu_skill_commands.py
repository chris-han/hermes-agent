import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import PlatformConfig

from agent.skill_commands import (
    build_skill_invocation_message,
    resolve_skill_command_key,
    scan_skill_commands,
)


def _make_skill(skills_dir: Path, name: str, body: str = "Use this skill.") -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test {name}\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )


def _expand_current(text: str, task_id: str):
    command, _, instruction = text.partition(" ")
    key = resolve_skill_command_key(command.removeprefix("/"))
    if key is None:
        return None
    return build_skill_invocation_message(key, instruction, task_id=task_id)


def test_feishu_direct_message_command_preserves_shared_dynamic_skill_invocation(tmp_path):
    from plugins.platforms.feishu.adapter import normalize_feishu_message

    _make_skill(tmp_path, "llm-wiki", "Read workspace wiki pages.")
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        scan_skill_commands()
        normalized = normalize_feishu_message(
            message_type="text",
            raw_content=json.dumps({"text": "/llm-wiki Compare governance pages"}),
        )
        invocation = _expand_current(normalized.text_content, task_id="feishu-direct")
    assert normalized.text_content == "/llm-wiki Compare governance pages"
    assert invocation is not None
    assert "llm-wiki" in invocation
    assert "Compare governance pages" in invocation


def test_feishu_group_self_mention_stripping_preserves_leading_skill_command(tmp_path):
    from plugins.platforms.feishu.adapter import (
        FeishuMentionRef,
        _FeishuBotIdentity,
        _strip_edge_self_mentions,
        normalize_feishu_message,
    )

    _make_skill(tmp_path, "migration-plan", "Plan work in ordered steps.")
    bot_mention = SimpleNamespace(
        key="@_user_1", id=SimpleNamespace(open_id="ou_bot", user_id=""), name="Hermes"
    )
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        scan_skill_commands()
        normalized = normalize_feishu_message(
            message_type="text",
            raw_content=json.dumps({"text": "@_user_1 /migration-plan Design migration"}),
            mentions=[bot_mention],
            bot=_FeishuBotIdentity(open_id="ou_bot"),
        )
        command_text = _strip_edge_self_mentions(
            normalized.text_content,
            [FeishuMentionRef(name="Hermes", open_id="ou_bot", is_self=True)],
        )
        invocation = _expand_current(command_text, task_id="feishu-group")
    assert command_text == "/migration-plan Design migration"
    assert invocation is not None
    assert "Design migration" in invocation


def test_feishu_stamps_only_governed_workspace_owner(monkeypatch):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    resolver = SimpleNamespace(
        resolve_feishu_ingress_owner=lambda **_kwargs: SimpleNamespace(
            owner_workspace_id="ws-governed"
        )
    )
    monkeypatch.setitem(sys.modules, "agents.feishu_ingress_identity", resolver)
    source = SimpleNamespace(chat_id="oc-chat", workspace_owner_id=None)
    sender = SimpleNamespace(open_id="ou-user", union_id="on-user")

    stamped = FeishuAdapter(PlatformConfig())._stamp_governed_workspace_owner(
        source,
        sender_id=sender,
        chat_id="oc-chat",
        platform_session_key="om-message",
    )

    assert stamped is source
    assert source.workspace_owner_id == "ws-governed"
