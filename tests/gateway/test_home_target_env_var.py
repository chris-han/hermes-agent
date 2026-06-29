"""Regression tests for /sethome env-var resolution.

The `/sethome` command writes to a platform's home-target env var. Two platforms
don't follow the `{PLATFORM}_HOME_CHANNEL` convention: matrix uses
`MATRIX_HOME_ROOM` and email uses `EMAIL_HOME_ADDRESS`. Before PR #12698
`/sethome` hardcoded the `_HOME_CHANNEL` suffix, so Matrix and Email saves went
to env vars nothing read on startup — the home channel appeared to set
successfully but was lost on every new gateway session.
"""

import os
import sys
import types

from gateway.config import Platform
from gateway.run import (
    _feishu_source_has_user_scoped_home_channel,
    _home_target_env_var,
    _home_thread_env_var,
)
from gateway.session import SessionSource


def test_matrix_home_target_env_var_uses_home_room():
    assert _home_target_env_var("matrix") == "MATRIX_HOME_ROOM"


def test_email_home_target_env_var_uses_home_address():
    assert _home_target_env_var("email") == "EMAIL_HOME_ADDRESS"


def test_telegram_home_target_env_var_uses_home_channel():
    assert _home_target_env_var("telegram") == "TELEGRAM_HOME_CHANNEL"


def test_discord_home_target_env_var_uses_home_channel():
    assert _home_target_env_var("discord") == "DISCORD_HOME_CHANNEL"


def test_unknown_platform_home_target_env_var_falls_back_to_home_channel():
    assert _home_target_env_var("custom") == "CUSTOM_HOME_CHANNEL"


def test_case_insensitive_platform_name():
    assert _home_target_env_var("MATRIX") == "MATRIX_HOME_ROOM"
    assert _home_target_env_var("Email") == "EMAIL_HOME_ADDRESS"


def test_home_thread_env_var_uses_home_target_name_plus_thread_id():
    assert _home_thread_env_var("discord") == "DISCORD_HOME_CHANNEL_THREAD_ID"
    assert _home_thread_env_var("matrix") == "MATRIX_HOME_ROOM_THREAD_ID"
    assert _home_thread_env_var("email") == "EMAIL_HOME_ADDRESS_THREAD_ID"


def test_feishu_auto_home_channel_updates_user_scoped_sqlite_config(monkeypatch):
    saved: dict[str, str] = {}
    monkeypatch.delenv("FEISHU_HOME_CHANNEL", raising=False)
    monkeypatch.delenv("FEISHU_HOME_CHANNEL_THREAD_ID", raising=False)
    auth_db = types.ModuleType("agents.auth_db")
    auth_db.get_feishu_bot_config_for_user = lambda user_id: {
        "owner_user_id": user_id,
        "app_id": "cli_a",
        "app_secret": "sec_b",
    }
    auth_db.save_feishu_bot_config = lambda payload: saved.update(payload)
    gateway_identity = types.ModuleType("agents.gateway_identity")
    gateway_identity.get_user_record_by_workspace_id = lambda workspace_id: {
        "user_id": "user-123",
        "workspace_id": workspace_id,
    }
    agents_pkg = types.ModuleType("agents")
    agents_pkg.auth_db = auth_db
    agents_pkg.gateway_identity = gateway_identity
    monkeypatch.setitem(sys.modules, "agents", agents_pkg)
    monkeypatch.setitem(sys.modules, "agents.auth_db", auth_db)
    monkeypatch.setitem(sys.modules, "agents.gateway_identity", gateway_identity)

    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_home_chat",
        chat_name="Home Chat",
        user_id="ou_user",
        workspace_owner_id="ws-123",
    )

    assert _feishu_source_has_user_scoped_home_channel(source) is True

    assert saved["owner_user_id"] == "user-123"
    assert saved["home_channel"] == "oc_home_chat"
    assert saved["home_channel_thread_id"] == ""
    assert "FEISHU_HOME_CHANNEL" not in os.environ
    assert "FEISHU_HOME_CHANNEL_THREAD_ID" not in os.environ
