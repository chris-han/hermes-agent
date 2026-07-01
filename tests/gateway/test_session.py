"""Tests for gateway session management."""

import builtins
import importlib
import json
import pytest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from gateway.config import Platform, HomeChannel, GatewayConfig, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import (
    SessionSource,
    SessionEntry,
    SessionStore,
    build_session_context,
    build_session_context_prompt,
    build_session_key,
    canonical_whatsapp_identifier,
)

# Legacy name preserved for these tests; product renamed the function to
# canonical_whatsapp_identifier.  Keep the tests referencing the old name
# working without duplicating the suite.
normalize_whatsapp_identifier = canonical_whatsapp_identifier


class TestGatewayConfigTrajectoryStore:
    def test_gateway_config_does_not_default_to_shared_trajectory_store(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config = GatewayConfig()
        assert config.sessions_dir != tmp_path / "trajectories"
        assert config.sessions_dir != tmp_path / "gateway-session-store"
        # The new default is the local sessions subdir under HERMES_HOME.
        assert config.sessions_dir == tmp_path / "sessions"


class TestSessionSourceRoundtrip:
    def test_full_roundtrip(self):
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_name="My Group",
            chat_type="group",
            user_id="99",
            user_name="alice",
            thread_id="t1",
        )
        d = source.to_dict()
        restored = SessionSource.from_dict(d)

        assert restored.platform == Platform.TELEGRAM
        assert restored.chat_id == "12345"
        assert restored.chat_name == "My Group"
        assert restored.chat_type == "group"
        assert restored.user_id == "99"
        assert restored.user_name == "alice"
        assert restored.thread_id == "t1"

    def test_full_roundtrip_with_chat_topic(self):
        """chat_topic should survive to_dict/from_dict roundtrip."""
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="789",
            chat_name="Server / #project-planning",
            chat_type="group",
            user_id="42",
            user_name="bob",
            chat_topic="Planning and coordination for Project X",
        )
        d = source.to_dict()
        assert d["chat_topic"] == "Planning and coordination for Project X"

        restored = SessionSource.from_dict(d)
        assert restored.chat_topic == "Planning and coordination for Project X"
        assert restored.chat_name == "Server / #project-planning"

    def test_roundtrip_preserves_adapter_keys(self):
        source = SessionSource(
            platform=Platform.WEIXIN,
            chat_id="wx-a",
            user_id="wx-a",
            workspace_owner_id="owner-1",
            adapter_key="weixin:ws-123:acct-a",
            delivery_adapter_key="weixin:ws-123:acct-b",
        )

        d = source.to_dict()
        restored = SessionSource.from_dict(d)

        assert d["adapter_key"] == "weixin:ws-123:acct-a"
        assert d["delivery_adapter_key"] == "weixin:ws-123:acct-b"
        assert restored.adapter_key == "weixin:ws-123:acct-a"
        assert restored.delivery_adapter_key == "weixin:ws-123:acct-b"

    def test_minimal_roundtrip(self):
        source = SessionSource(platform=Platform.LOCAL, chat_id="cli")
        d = source.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.platform == Platform.LOCAL
        assert restored.chat_id == "cli"
        assert restored.chat_type == "dm"  # default value preserved

    def test_chat_id_coerced_to_string(self):
        """from_dict should handle numeric chat_id (common from Telegram)."""
        restored = SessionSource.from_dict({
            "platform": "telegram",
            "chat_id": 12345,
        })
        assert restored.chat_id == "12345"
        assert isinstance(restored.chat_id, str)

    def test_missing_optional_fields(self):
        restored = SessionSource.from_dict({
            "platform": "discord",
            "chat_id": "abc",
        })
        assert restored.chat_name is None
        assert restored.user_id is None
        assert restored.user_name is None
        assert restored.thread_id is None
        assert restored.chat_topic is None
        assert restored.chat_type == "dm"

    def test_unknown_platform_rejected_for_bad_names(self):
        """Arbitrary platform names are rejected (no accidental enum pollution).

        Only bundled platform plugins (discovered under ``plugins/platforms/``)
        and runtime-registered plugins get dynamic enum members.
        """
        with pytest.raises(ValueError):
            SessionSource.from_dict({"platform": "nonexistent", "chat_id": "1"})


class TestSessionEntryRoundtrip:
    def test_session_model_override_roundtrip(self):
        from gateway.session import SessionEntry

        entry = SessionEntry(
            session_key="agent:main:weixin:dm:c1",
            session_id="sess-1",
            created_at="2026-05-20T00:00:00Z",
            updated_at="2026-05-20T00:00:01Z",
            platform=Platform.WEIXIN,
            chat_type="dm",
            session_model_override={
                "model": "qwen3.6-plus",
                "provider": "alibaba",
                "api_key": "dashscope-key",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_mode": "chat_completions",
            },
        )

        restored = SessionEntry.from_dict(entry.to_dict())

        assert restored.session_model_override is not None
        assert restored.session_model_override["model"] == "qwen3.6-plus"
        assert restored.session_model_override["provider"] == "alibaba"


class TestSessionSourceDescription:
    def test_local_cli(self):
        source = SessionSource(
            platform=Platform.LOCAL, chat_id="cli",
            chat_name="CLI terminal", chat_type="dm",
        )
        assert source.description == "CLI terminal"

    def test_dm_with_username(self):
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="123",
            chat_type="dm", user_name="bob",
        )
        assert "DM" in source.description
        assert "bob" in source.description

    def test_dm_without_username_falls_back_to_user_id(self):
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="123",
            chat_type="dm", user_id="456",
        )
        assert "456" in source.description

    def test_group_shows_chat_name(self):
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="789",
            chat_type="group", chat_name="Dev Chat",
        )
        assert "group" in source.description
        assert "Dev Chat" in source.description

    def test_channel_type(self):
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="100",
            chat_type="channel", chat_name="Announcements",
        )
        assert "channel" in source.description
        assert "Announcements" in source.description

    def test_thread_id_appended(self):
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="789",
            chat_type="group", chat_name="General",
            thread_id="thread-42",
        )
        assert "thread" in source.description
        assert "thread-42" in source.description

    def test_unknown_chat_type_uses_name(self):
        source = SessionSource(
            platform=Platform.SLACK, chat_id="C01",
            chat_type="forum", chat_name="Questions",
        )
        assert "Questions" in source.description


class TestLocalCliFactory:
    def test_local_cli_defaults(self):
        source = SessionSource(
            platform=Platform.LOCAL, chat_id="cli",
            chat_name="CLI terminal", chat_type="dm",
        )
        assert source.platform == Platform.LOCAL
        assert source.chat_id == "cli"
        assert source.chat_type == "dm"
        assert source.chat_name == "CLI terminal"


class TestBuildSessionContextPrompt:
    def test_telegram_prompt_contains_platform_and_chat(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="fake-token",
                    home_channel=HomeChannel(
                        platform=Platform.TELEGRAM,
                        chat_id="111",
                        name="Home Chat",
                    ),
                ),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="111",
            chat_name="Home Chat",
            chat_type="dm",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Telegram" in prompt
        assert "Home Chat" in prompt

    def test_bluebubbles_prompt_mentions_short_conversational_i_message_format(self):
        config = GatewayConfig(
            platforms={
                Platform.BLUEBUBBLES: PlatformConfig(enabled=True, extra={"server_url": "http://localhost:1234", "password": "secret"}),
            },
        )
        source = SessionSource(
            platform=Platform.BLUEBUBBLES,
            chat_id="iMessage;-;user@example.com",
            chat_name="Ben",
            chat_type="dm",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "responding via iMessage" in prompt
        assert "short and conversational" in prompt
        assert "blank line" in prompt

    def test_discord_prompt(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    token="fake-d...oken",
                ),
            },
        )
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_name="Server",
            chat_type="group",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Discord" in prompt
        assert "cannot search" in prompt.lower() or "do not have access" in prompt.lower()

    def test_slack_prompt_includes_platform_notes(self):
        config = GatewayConfig(
            platforms={
                Platform.SLACK: PlatformConfig(enabled=True, token="fake"),
            },
        )
        source = SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_name="general",
            chat_type="group",
            user_name="bob",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Slack" in prompt
        assert "cannot search" in prompt.lower()
        assert "pin" in prompt.lower()
        assert "current message's slack block/attachment payload" in prompt.lower()

    def test_discord_prompt_with_channel_topic(self):
        """Channel topic should appear in the session context prompt."""
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    token="fake-discord-token",
                ),
            },
        )
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_name="Server / #project-planning",
            chat_type="group",
            user_name="alice",
            chat_topic="Planning and coordination for Project X",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Discord" in prompt
        assert "**Channel Topic:** Planning and coordination for Project X" in prompt

    def test_prompt_omits_channel_topic_when_none(self):
        """Channel Topic line should NOT appear when chat_topic is None."""
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    token="fake-discord-token",
                ),
            },
        )
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_name="Server / #general",
            chat_type="group",
            user_name="alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Channel Topic" not in prompt

    def test_local_prompt_mentions_machine(self):
        config = GatewayConfig()
        source = SessionSource(
            platform=Platform.LOCAL, chat_id="cli",
            chat_name="CLI terminal", chat_type="dm",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Local" in prompt
        assert "machine running this agent" in prompt

    def test_local_delivery_path_uses_display_hermes_home(self):
        config = GatewayConfig()
        source = SessionSource(
            platform=Platform.LOCAL, chat_id="cli",
            chat_name="CLI terminal", chat_type="dm",
        )
        ctx = build_session_context(source, config)

        with patch("hermes_constants.display_hermes_home", return_value="~/.hermes/profiles/coder"):
            prompt = build_session_context_prompt(ctx)

        assert "~/.hermes/profiles/coder/cron/output/" in prompt

    def test_whatsapp_prompt(self):
        config = GatewayConfig(
            platforms={
                Platform.WHATSAPP: PlatformConfig(enabled=True, token=""),
            },
        )
        source = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="15551234567@s.whatsapp.net",
            chat_type="dm",
            user_name="Phone User",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "WhatsApp" in prompt or "whatsapp" in prompt.lower()

    def test_multi_user_thread_prompt(self):
        """Shared thread sessions show multi-user note instead of single user."""
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1002285219667",
            chat_name="Test Group",
            chat_type="group",
            thread_id="17585",
            user_name="Alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Multi-user thread" in prompt
        assert "[sender name]" in prompt
        # Should NOT show a specific **User:** line (would bust cache)
        assert "**User:** Alice" not in prompt

    def test_non_thread_group_shows_user(self):
        """Regular group messages (no thread) still show the user name."""
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1002285219667",
            chat_name="Test Group",
            chat_type="group",
            user_name="Alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "**User:** Alice" in prompt
        assert "Multi-user thread" not in prompt

    def test_shared_non_thread_group_prompt_hides_single_user(self):
        """Shared non-thread group sessions should avoid pinning one user."""
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
            group_sessions_per_user=False,
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1002285219667",
            chat_name="Test Group",
            chat_type="group",
            user_name="Alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "Multi-user session" in prompt
        assert "[sender name]" in prompt
        assert "**User:** Alice" not in prompt

    def test_dm_thread_shows_user_not_multi(self):
        """DM threads are single-user and should show User, not multi-user note."""
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="99",
            chat_type="dm",
            thread_id="topic-1",
            user_name="Alice",
        )
        ctx = build_session_context(source, config)
        prompt = build_session_context_prompt(ctx)

        assert "**User:** Alice" in prompt
        assert "Multi-user thread" not in prompt


class TestSenderPrefixWithBackfill:
    """Regression: sender prefix must not wrap the backfill context block.

    Tests exercise the real GatewayRunner._prepare_inbound_message_text()
    method to ensure the [sender_name] prefix applies only to the trigger
    message, not the channel_context backfill block.
    """

    @pytest.fixture()
    def runner(self):
        from gateway.run import GatewayRunner

        r = GatewayRunner.__new__(GatewayRunner)
        r.config = GatewayConfig(group_sessions_per_user=False)
        r.adapters = {}
        r._model = "test-model"
        r._base_url = ""
        r._has_setup_skill = lambda: False
        return r

    @pytest.fixture()
    def source(self):
        return SessionSource(
            platform=Platform.DISCORD,
            chat_id="c1",
            chat_type="group",
            user_name="Alice",
        )

    @pytest.mark.asyncio
    async def test_plain_message_gets_prefix(self, runner, source):
        """Normal message without backfill gets [sender] prefix."""
        event = MessageEvent(text="hello world", source=source)
        result = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[],
        )
        assert result == "[Alice] hello world"

    @pytest.mark.asyncio
    async def test_backfill_prefix_only_on_trigger(self, runner, source):
        """Backfill context must NOT get the sender prefix."""
        event = MessageEvent(
            text="hello world",
            source=source,
            channel_context="[Recent channel messages]\n[Bob] some context",
        )
        result = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[],
        )
        assert result.startswith("[Recent channel messages]")
        assert "[Alice] [Recent channel messages]" not in result
        assert "[New message]\n[Alice] hello world" in result

    @pytest.mark.asyncio
    async def test_backfill_preserves_context_block(self, runner, source):
        """The backfill block should pass through unchanged — no double-prefixing."""
        context = "[Recent channel messages]\n[Bob] first\n[Charlie [bot]] second"
        event = MessageEvent(
            text="hey everyone", source=source, channel_context=context,
        )
        result = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[],
        )
        assert result.startswith(context)
        assert "[Alice] hey everyone" in result
        assert "[Alice] [Bob]" not in result
        assert "[Alice] [Charlie" not in result
        assert "[Alice] [Recent" not in result


class TestSessionStoreRewriteTranscript:
    """Regression: /retry and /undo must persist truncated history to disk."""

    @pytest.fixture()
    def store(self, tmp_path):
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._db = None  # no SQLite for these tests
        s._loaded = True
        return s

    def test_rewrite_replaces_jsonl(self, store, tmp_path):
        session_id = "test_session_1"
        # Register as workspace-bound so JSONL writes are enabled
        ws_home = tmp_path / "workspaces" / "test_session_1" / ".hermes"
        ws_home.mkdir(parents=True)
        store.register_workspace_home(session_id, ws_home)
        # Write initial transcript
        for msg in [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "undo this"},
            {"role": "assistant", "content": "ok"},
        ]:
            store.append_to_transcript(session_id, msg)

        # Rewrite with truncated history
        store.rewrite_transcript(session_id, [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])

        reloaded = store.load_transcript(session_id)
        assert len(reloaded) == 2
        assert reloaded[0]["content"] == "hello"
        assert reloaded[1]["content"] == "hi"

    def test_rewrite_with_empty_list(self, store, tmp_path):
        session_id = "test_session_2"
        ws_home = tmp_path / "workspaces" / "test_session_2" / ".hermes"
        ws_home.mkdir(parents=True)
        store.register_workspace_home(session_id, ws_home)
        store.append_to_transcript(session_id, {"role": "user", "content": "hi"})

        store.rewrite_transcript(session_id, [])

        reloaded = store.load_transcript(session_id)
        assert reloaded == []


class TestLoadTranscriptCorruptLines:
    """Regression: corrupt JSONL lines (e.g. from mid-write crash) must be
    skipped instead of crashing the entire transcript load.  GH-1193."""

    @pytest.fixture()
    def store(self, tmp_path):
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._db = None
        s._loaded = True
        return s

    def test_corrupt_line_skipped(self, store, tmp_path):
        session_id = "corrupt_test"
        ws_home = tmp_path / "workspaces" / "corrupt_test" / ".hermes"
        ws_home.mkdir(parents=True)
        store.register_workspace_home(session_id, ws_home)
        transcript_path = store.get_transcript_path(session_id)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript_path, "w") as f:
            f.write('{"role": "user", "content": "hello"}\n')
            f.write('{"role": "assistant", "content": "hi th')  # truncated
            f.write("\n")
            f.write('{"role": "user", "content": "goodbye"}\n')

        messages = store.load_transcript(session_id)
        assert len(messages) == 2
        assert messages[0]["content"] == "hello"
        assert messages[1]["content"] == "goodbye"

    def test_all_lines_corrupt_returns_empty(self, store, tmp_path):
        session_id = "all_corrupt"
        ws_home = tmp_path / "workspaces" / "all_corrupt" / ".hermes"
        ws_home.mkdir(parents=True)
        store.register_workspace_home(session_id, ws_home)
        transcript_path = store.get_transcript_path(session_id)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript_path, "w") as f:
            f.write("not json at all\n")
            f.write("{truncated\n")

        messages = store.load_transcript(session_id)
        assert messages == []

    def test_valid_transcript_unaffected(self, store, tmp_path):
        session_id = "valid_test"
        ws_home = tmp_path / "workspaces" / "valid_test" / ".hermes"
        ws_home.mkdir(parents=True)
        store.register_workspace_home(session_id, ws_home)
        store.append_to_transcript(session_id, {"role": "user", "content": "a"})
        store.append_to_transcript(session_id, {"role": "assistant", "content": "b"})

        messages = store.load_transcript(session_id)
        assert len(messages) == 2
        assert messages[0]["content"] == "a"
        assert messages[1]["content"] == "b"


class TestLoadTranscriptPreferLongerSource:
    """Regression: load_transcript must return whichever source (SQLite or JSONL)
    has more messages to prevent silent truncation.  GH-3212."""

    @pytest.fixture()
    def store_with_db(self, tmp_path):
        """SessionStore with both SQLite and JSONL active."""
        from hermes_state import SessionDB

        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._db = SessionDB(db_path=tmp_path / "state.db")
        s._loaded = True
        return s

    def test_jsonl_longer_than_sqlite_returns_jsonl(self, store_with_db, tmp_path):
        """Legacy session: JSONL has full history, SQLite has only recent turn."""
        sid = "legacy_session"
        ws_home = tmp_path / "workspaces" / "legacy_session" / ".hermes"
        ws_home.mkdir(parents=True)
        store_with_db.register_workspace_home(sid, ws_home)
        store_with_db._db.create_session(session_id=sid, source="gateway", model="m")
        # JSONL has 10 messages (legacy history — written before SQLite existed)
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            store_with_db.append_to_transcript(
                sid, {"role": role, "content": f"msg-{i}"}, skip_db=True,
            )
        # SQLite has only 2 messages (recent turn after migration)
        store_with_db._db.append_message(session_id=sid, role="user", content="new-q")
        store_with_db._db.append_message(session_id=sid, role="assistant", content="new-a")

        result = store_with_db.load_transcript(sid)
        assert len(result) == 10
        assert result[0]["content"] == "msg-0"

    def test_sqlite_longer_than_jsonl_returns_sqlite(self, store_with_db):
        """Fully migrated session: SQLite has more (JSONL stopped growing)."""
        sid = "migrated_session"
        store_with_db._db.create_session(session_id=sid, source="gateway", model="m")
        # JSONL has 2 old messages
        store_with_db.append_to_transcript(
            sid, {"role": "user", "content": "old-q"}, skip_db=True,
        )
        store_with_db.append_to_transcript(
            sid, {"role": "assistant", "content": "old-a"}, skip_db=True,
        )
        # SQLite has 4 messages (superset after migration)
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            store_with_db._db.append_message(session_id=sid, role=role, content=f"db-{i}")

        result = store_with_db.load_transcript(sid)
        assert len(result) == 4
        assert result[0]["content"] == "db-0"

    def test_sqlite_empty_falls_back_to_jsonl(self, store_with_db, tmp_path):
        """No SQLite rows — falls back to JSONL (original behavior preserved)."""
        sid = "no_db_rows"
        ws_home = tmp_path / "workspaces" / "no_db_rows" / ".hermes"
        ws_home.mkdir(parents=True)
        store_with_db.register_workspace_home(sid, ws_home)
        store_with_db.append_to_transcript(
            sid, {"role": "user", "content": "hello"}, skip_db=True,
        )
        store_with_db.append_to_transcript(
            sid, {"role": "assistant", "content": "hi"}, skip_db=True,
        )

        result = store_with_db.load_transcript(sid)
        assert len(result) == 2
        assert result[0]["content"] == "hello"

    def test_both_empty_returns_empty(self, store_with_db):
        """Neither source has data — returns empty list."""
        result = store_with_db.load_transcript("nonexistent")
        assert result == []

    def test_equal_length_prefers_sqlite(self, store_with_db):
        """When both have same count, SQLite wins (has richer fields like reasoning)."""
        sid = "equal_session"
        store_with_db._db.create_session(session_id=sid, source="gateway", model="m")
        # Write 2 messages to JSONL only
        store_with_db.append_to_transcript(
            sid, {"role": "user", "content": "jsonl-q"}, skip_db=True,
        )
        store_with_db.append_to_transcript(
            sid, {"role": "assistant", "content": "jsonl-a"}, skip_db=True,
        )
        # Write 2 different messages to SQLite only
        store_with_db._db.append_message(session_id=sid, role="user", content="db-q")
        store_with_db._db.append_message(session_id=sid, role="assistant", content="db-a")

        result = store_with_db.load_transcript(sid)
        assert len(result) == 2
        # Should be the SQLite version (equal count → prefers SQLite)
        assert result[0]["content"] == "db-q"

    def test_unreadable_jsonl_returns_sqlite(self, store_with_db, monkeypatch, tmp_path):
        """Unreadable legacy JSONL must not hide valid SQLite history."""
        sid = "unreadable_jsonl"
        ws_home = tmp_path / "workspaces" / "unreadable_jsonl" / ".hermes"
        ws_home.mkdir(parents=True)
        store_with_db.register_workspace_home(sid, ws_home)
        store_with_db._db.create_session(session_id=sid, source="gateway", model="m")
        store_with_db._db.append_message(session_id=sid, role="user", content="db-q")
        store_with_db._db.append_message(session_id=sid, role="assistant", content="db-a")

        transcript_path = store_with_db.get_transcript_path(sid)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text('{"role": "user", "content": "jsonl-q"}\n', encoding="utf-8")

        real_open = builtins.open

        def raise_for_transcript(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if Path(path) == transcript_path and "r" in mode:
                raise OSError("simulated unreadable transcript")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", raise_for_transcript)

        result = store_with_db.load_transcript(sid)
        assert len(result) == 2
        assert result[0]["content"] == "db-q"
        assert result[1]["content"] == "db-a"


class TestSessionStoreSwitchSession:
    """Regression coverage for gateway /resume session switching semantics."""

    def test_switch_session_reopens_target_session_in_db(self, tmp_path):
        from hermes_state import SessionDB

        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
        db = SessionDB(db_path=tmp_path / "state.db")
        store._db = db
        store._loaded = True

        source = SessionSource(
            platform=Platform.FEISHU,
            chat_id="chat-1",
            chat_type="dm",
            user_id="user-1",
            user_name="tester",
        )
        current_entry = store.get_or_create_session(source)
        current_session_id = current_entry.session_id

        target_session_id = "old_session_abc"
        db.create_session(target_session_id, source="feishu", user_id="user-1")
        db.end_session(target_session_id, end_reason="user_exit")
        assert db.get_session(target_session_id)["ended_at"] is not None

        switched = store.switch_session(current_entry.session_key, target_session_id)

        assert switched is not None
        assert switched.session_id == target_session_id
        assert db.get_session(current_session_id)["end_reason"] == "session_switch"
        resumed = db.get_session(target_session_id)
        assert resumed["ended_at"] is None
        assert resumed["end_reason"] is None
        db.close()


class TestSessionStoreLookupBySessionId:
    @pytest.fixture()
    def store(self, tmp_path):
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._db = None
        s._loaded = True
        return s

    def test_returns_active_entry_for_persisted_session_id(self, store):
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:example.org",
            chat_type="group",
            user_id="@alice:example.org",
        )
        entry = store.get_or_create_session(source)

        assert store.lookup_by_session_id(entry.session_id) is entry
        assert store.lookup_by_session_id("missing") is None
        assert store.lookup_by_session_id("") is None


class TestWhatsAppSessionKeyConsistency:
    """Regression: WhatsApp session keys must collapse JID/LID aliases to a
    single stable identity for both DM chat_ids and group participant_ids."""

    @pytest.fixture()
    def store(self, tmp_path):
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._db = None
        s._loaded = True
        return s

    def test_whatsapp_dm_uses_canonical_identifier(self):
        source = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="15551234567@s.whatsapp.net",
            chat_type="dm",
            user_name="Phone User",
        )
        key = build_session_key(source)
        assert key == "agent:main:whatsapp:dm:15551234567"

    def test_whatsapp_dm_aliases_share_one_session_key(self, tmp_path, monkeypatch):
        tmp_home = tmp_path / "hermes-home"
        mapping_dir = tmp_home / "whatsapp" / "session"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        (mapping_dir / "lid-mapping-999999999999999.json").write_text(
            json.dumps("15551234567@s.whatsapp.net"),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_home))

        lid_source = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="999999999999999@lid",
            chat_type="dm",
            user_name="Phone User",
        )
        phone_source = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="15551234567@s.whatsapp.net",
            chat_type="dm",
            user_name="Phone User",
        )

        assert build_session_key(lid_source) == "agent:main:whatsapp:dm:15551234567"
        assert build_session_key(phone_source) == "agent:main:whatsapp:dm:15551234567"

    def test_whatsapp_group_participant_aliases_share_session_key(self, tmp_path, monkeypatch):
        """With group_sessions_per_user, the same human flipping between
        phone-JID and LID inside a group must not produce two isolated
        per-user sessions."""
        tmp_home = tmp_path / "hermes-home"
        mapping_dir = tmp_home / "whatsapp" / "session"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        (mapping_dir / "lid-mapping-999999999999999.json").write_text(
            json.dumps("15551234567@s.whatsapp.net"),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_home))

        lid_source = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="120363000000000000@g.us",
            chat_type="group",
            user_id="999999999999999@lid",
            user_name="Group Member",
        )
        phone_source = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="120363000000000000@g.us",
            chat_type="group",
            user_id="15551234567@s.whatsapp.net",
            user_name="Group Member",
        )

        expected = "agent:main:whatsapp:group:120363000000000000@g.us:15551234567"
        assert build_session_key(lid_source, group_sessions_per_user=True) == expected
        assert build_session_key(phone_source, group_sessions_per_user=True) == expected

    def test_whatsapp_group_shared_sessions_untouched_by_canonicalisation(self):
        """When group_sessions_per_user is False, participant_id is not in the
        key at all, so canonicalisation is a no-op for this mode."""
        source = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="120363000000000000@g.us",
            chat_type="group",
            user_id="999999999999999@lid",
            user_name="Group Member",
        )
        assert (
            build_session_key(source, group_sessions_per_user=False)
            == "agent:main:whatsapp:group:120363000000000000@g.us"
        )

    def test_store_delegates_to_build_session_key(self, store):
        """SessionStore._generate_session_key must produce the same result."""
        source = SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="15551234567@s.whatsapp.net",
            chat_type="dm",
            user_name="Phone User",
        )
        assert store._generate_session_key(source) == build_session_key(source)

    def test_store_creates_distinct_group_sessions_per_user(self, store):
        first = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_type="group",
            user_id="alice",
            user_name="Alice",
        )
        second = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_type="group",
            user_id="bob",
            user_name="Bob",
        )

        first_entry = store.get_or_create_session(first)
        second_entry = store.get_or_create_session(second)

        assert first_entry.session_key == "agent:main:discord:group:guild-123:alice"
        assert second_entry.session_key == "agent:main:discord:group:guild-123:bob"
        assert first_entry.session_id != second_entry.session_id

    def test_store_shares_group_sessions_when_disabled_in_config(self, store):
        store.config.group_sessions_per_user = False

        first = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_type="group",
            user_id="alice",
            user_name="Alice",
        )
        second = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_type="group",
            user_id="bob",
            user_name="Bob",
        )

        first_entry = store.get_or_create_session(first)
        second_entry = store.get_or_create_session(second)

        assert first_entry.session_key == "agent:main:discord:group:guild-123"
        assert second_entry.session_key == "agent:main:discord:group:guild-123"
        assert first_entry.session_id == second_entry.session_id

    def test_telegram_dm_includes_chat_id(self):
        """Non-WhatsApp DMs should also include chat_id to separate users."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="99",
            chat_type="dm",
        )
        key = build_session_key(source)
        assert key == "agent:main:telegram:dm:99"

    def test_distinct_dm_chat_ids_get_distinct_session_keys(self):
        """Different DM chats must not collapse into one shared session."""
        first = SessionSource(platform=Platform.TELEGRAM, chat_id="99", chat_type="dm")
        second = SessionSource(platform=Platform.TELEGRAM, chat_id="100", chat_type="dm")

        assert build_session_key(first) == "agent:main:telegram:dm:99"
        assert build_session_key(second) == "agent:main:telegram:dm:100"
        assert build_session_key(first) != build_session_key(second)

    def test_dm_without_chat_id_falls_back_to_user_id(self):
        """A DM source missing chat_id must isolate on the sender's user_id
        rather than collapsing into the shared per-platform sink."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="",
            chat_type="dm",
            user_id="jordan",
        )
        assert build_session_key(source) == "agent:main:telegram:dm:jordan"

    def test_dm_without_chat_id_distinct_users_do_not_collide(self):
        """Two different DM senders without chat_id must not share one
        session (the cross-user history-bleed footgun)."""
        first = SessionSource(
            platform=Platform.TELEGRAM, chat_id="", chat_type="dm", user_id="jordan"
        )
        second = SessionSource(
            platform=Platform.TELEGRAM, chat_id="", chat_type="dm", user_id="dima"
        )
        assert build_session_key(first) != build_session_key(second)
        assert build_session_key(first) == "agent:main:telegram:dm:jordan"
        assert build_session_key(second) == "agent:main:telegram:dm:dima"

    def test_dm_without_chat_id_prefers_user_id_alt(self):
        """user_id_alt wins over user_id for the DM fallback, matching the
        group-path participant precedence."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="",
            chat_type="dm",
            user_id="primary",
            user_id_alt="alt",
        )
        assert build_session_key(source) == "agent:main:telegram:dm:alt"

    def test_dm_without_chat_id_or_user_id_falls_back_to_thread_then_sink(self):
        """With neither chat_id nor user identifiers, thread_id is the next
        discriminator; only a completely identifier-less DM hits the sink."""
        threaded = SessionSource(
            platform=Platform.TELEGRAM, chat_id="", chat_type="dm", thread_id="7"
        )
        assert build_session_key(threaded) == "agent:main:telegram:dm:7"

        bare = SessionSource(platform=Platform.TELEGRAM, chat_id="", chat_type="dm")
        assert build_session_key(bare) == "agent:main:telegram:dm"

    def test_discord_group_includes_chat_id(self):
        """Group/channel keys include chat_type and chat_id."""
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_type="group",
        )
        key = build_session_key(source)
        assert key == "agent:main:discord:group:guild-123"

    def test_group_sessions_are_isolated_per_user_when_user_id_present(self):
        first = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_type="group",
            user_id="alice",
        )
        second = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_type="group",
            user_id="bob",
        )

        assert build_session_key(first) == "agent:main:discord:group:guild-123:alice"
        assert build_session_key(second) == "agent:main:discord:group:guild-123:bob"
        assert build_session_key(first) != build_session_key(second)

    def test_group_sessions_can_be_shared_when_isolation_disabled(self):
        first = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_type="group",
            user_id="alice",
        )
        second = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_type="group",
            user_id="bob",
        )

        assert build_session_key(first, group_sessions_per_user=False) == "agent:main:discord:group:guild-123"
        assert build_session_key(second, group_sessions_per_user=False) == "agent:main:discord:group:guild-123"

    def test_group_thread_includes_thread_id(self):
        """Forum-style threads need a distinct session key within one group."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1002285219667",
            chat_type="group",
            thread_id="17585",
        )
        key = build_session_key(source)
        assert key == "agent:main:telegram:group:-1002285219667:17585"

    def test_group_thread_sessions_are_shared_by_default(self):
        """Threads default to shared sessions — user_id is NOT appended."""
        alice = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1002285219667",
            chat_type="group",
            thread_id="17585",
            user_id="alice",
        )
        bob = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1002285219667",
            chat_type="group",
            thread_id="17585",
            user_id="bob",
        )
        assert build_session_key(alice) == "agent:main:telegram:group:-1002285219667:17585"
        assert build_session_key(bob) == "agent:main:telegram:group:-1002285219667:17585"
        assert build_session_key(alice) == build_session_key(bob)

    def test_group_thread_sessions_can_be_isolated_per_user(self):
        """thread_sessions_per_user=True restores per-user isolation in threads."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1002285219667",
            chat_type="group",
            thread_id="17585",
            user_id="42",
        )
        key = build_session_key(source, thread_sessions_per_user=True)
        assert key == "agent:main:telegram:group:-1002285219667:17585:42"

    def test_non_thread_group_sessions_still_isolated_per_user(self):
        """Regular group messages (no thread_id) remain per-user by default."""
        alice = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1002285219667",
            chat_type="group",
            user_id="alice",
        )
        bob = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1002285219667",
            chat_type="group",
            user_id="bob",
        )
        assert build_session_key(alice) == "agent:main:telegram:group:-1002285219667:alice"
        assert build_session_key(bob) == "agent:main:telegram:group:-1002285219667:bob"
        assert build_session_key(alice) != build_session_key(bob)

    def test_discord_thread_sessions_shared_by_default(self):
        """Discord threads are shared across participants by default."""
        alice = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_type="thread",
            thread_id="thread-456",
            user_id="alice",
        )
        bob = SessionSource(
            platform=Platform.DISCORD,
            chat_id="guild-123",
            chat_type="thread",
            thread_id="thread-456",
            user_id="bob",
        )
        assert build_session_key(alice) == build_session_key(bob)
        assert "alice" not in build_session_key(alice)
        assert "bob" not in build_session_key(bob)

    def test_dm_thread_sessions_not_affected(self):
        """DM threads use their own keying logic and are not affected."""
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="99",
            chat_type="dm",
            thread_id="topic-1",
            user_id="42",
        )
        key = build_session_key(source)
        # DM logic: chat_id + thread_id, user_id never included
        assert key == "agent:main:telegram:dm:99:topic-1"


class TestWhatsAppIdentifierPublicHelpers:
    """Contract tests for the public WhatsApp identifier helpers.

    These helpers are part of the public API for plugins that need
    WhatsApp identity awareness. Breaking these contracts is a
    breaking change for downstream plugins.
    """

    def test_normalize_strips_jid_suffix(self):
        assert normalize_whatsapp_identifier("60123456789@s.whatsapp.net") == "60123456789"

    def test_normalize_strips_lid_suffix(self):
        assert normalize_whatsapp_identifier("999999999999999@lid") == "999999999999999"

    def test_normalize_strips_device_suffix(self):
        assert normalize_whatsapp_identifier("60123456789:47@s.whatsapp.net") == "60123456789"

    def test_normalize_strips_leading_plus(self):
        assert normalize_whatsapp_identifier("+60123456789") == "60123456789"

    def test_normalize_handles_bare_numeric(self):
        assert normalize_whatsapp_identifier("60123456789") == "60123456789"

    def test_normalize_handles_empty_and_none(self):
        assert normalize_whatsapp_identifier("") == ""
        assert normalize_whatsapp_identifier(None) == ""  # type: ignore[arg-type]

    def test_canonical_without_mapping_returns_normalized(self, tmp_path, monkeypatch):
        """With no bridge mapping files, the normalized input is returned."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert canonical_whatsapp_identifier("60123456789@lid") == "60123456789"

    def test_canonical_walks_lid_mapping(self, tmp_path, monkeypatch):
        """LID is resolved to its paired phone identity via lid-mapping files."""
        mapping_dir = tmp_path / "whatsapp" / "session"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        (mapping_dir / "lid-mapping-999999999999999.json").write_text(
            json.dumps("15551234567@s.whatsapp.net"),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        canonical = canonical_whatsapp_identifier("999999999999999@lid")
        assert canonical == "15551234567"
        assert canonical_whatsapp_identifier("15551234567@s.whatsapp.net") == "15551234567"

    def test_canonical_empty_input(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert canonical_whatsapp_identifier("") == ""


class TestSessionEntryFromDictTraversalValidation:
    """Regression: from_dict must reject traversal sequences in session_key/session_id."""

    BASE = {
        "session_key": "agent:main:local:dm",
        "session_id": "abc123",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }

    def _entry(self, **overrides):
        from gateway.session import SessionEntry
        return {**self.BASE, **overrides}

    def test_valid_entry_loads(self):
        from gateway.session import SessionEntry
        entry = SessionEntry.from_dict(self._entry())
        assert entry.session_id == "abc123"

    def test_session_id_dotdot_raises(self):
        from gateway.session import SessionEntry
        with pytest.raises(ValueError, match="session_id"):
            SessionEntry.from_dict(self._entry(session_id="../../etc/passwd"))

    def test_session_key_dotdot_raises(self):
        from gateway.session import SessionEntry
        with pytest.raises(ValueError, match="session_key"):
            SessionEntry.from_dict(self._entry(session_key="agent:main:../../secret"))

    def test_session_id_absolute_unix_raises(self):
        from gateway.session import SessionEntry
        with pytest.raises(ValueError, match="session_id"):
            SessionEntry.from_dict(self._entry(session_id="/etc/passwd"))

    def test_session_id_absolute_windows_raises(self):
        from gateway.session import SessionEntry
        with pytest.raises(ValueError, match="session_id"):
            SessionEntry.from_dict(self._entry(session_id="\\windows\\system32\\config"))

    def test_session_id_windows_drive_letter_raises(self):
        from gateway.session import SessionEntry
        with pytest.raises(ValueError, match="session_id"):
            SessionEntry.from_dict(self._entry(session_id="C:/windows/system32"))

    def test_session_id_windows_drive_backslash_raises(self):
        from gateway.session import SessionEntry
        with pytest.raises(ValueError, match="session_id"):
            SessionEntry.from_dict(self._entry(session_id="D:\\path\\to\\file"))

    def test_session_id_non_leading_separator_raises(self):
        """A path separator anywhere — not just leading — must be rejected,
        since a non-leading backslash is still a Windows traversal vector."""
        from gateway.session import SessionEntry
        with pytest.raises(ValueError, match="session_id"):
            SessionEntry.from_dict(self._entry(session_id="good\\..\\bad"))
        with pytest.raises(ValueError, match="session_key"):
            SessionEntry.from_dict(self._entry(session_key="agent:main:good/sub"))


class TestEnsureLoadedSkipsInvalidEntries:
    """Regression: one bad sessions.json entry must not block valid entries from loading."""

    def test_invalid_entry_skipped_valid_entry_loads(self, tmp_path):
        import json
        from gateway.session import SessionStore
        from gateway.config import GatewayConfig

        sessions_file = tmp_path / "sessions.json"
        sessions_file.write_text(json.dumps({
            "bad:key": {
                "session_key": "bad:key",
                "session_id": "../../evil",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
            "agent:main:local:dm": {
                "session_key": "agent:main:local:dm",
                "session_id": "good123",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
        }), encoding="utf-8")

        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        store._ensure_loaded()

        assert "bad:key" not in store._entries
        assert "agent:main:local:dm" in store._entries
        assert store._entries["agent:main:local:dm"].session_id == "good123"


class TestSessionStoreEntriesAttribute:
    """Regression: /reset must access _entries, not _sessions."""

    def test_entries_attribute_exists(self):
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=Path("/tmp"), config=config)
        store._loaded = True
        assert hasattr(store, "_entries")
        assert not hasattr(store, "_sessions")


class TestHasAnySessions:
    """Tests for has_any_sessions() fix (issue #351)."""

    @pytest.fixture
    def store_with_mock_db(self, tmp_path):
        """SessionStore with a mocked database."""
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._loaded = True
        s._entries = {}
        s._db = MagicMock()
        return s

    def test_uses_database_count_when_available(self, store_with_mock_db):
        """has_any_sessions should use database session_count, not len(_entries)."""
        store = store_with_mock_db
        # Simulate single-platform user with only 1 entry in memory
        store._entries = {"telegram:12345": MagicMock()}
        # But database has 3 sessions (current + 2 previous resets)
        store._db.session_count.return_value = 3

        assert store.has_any_sessions() is True
        store._db.session_count.assert_called_once()

    def test_first_session_ever_returns_false(self, store_with_mock_db):
        """First session ever should return False (only current session in DB)."""
        store = store_with_mock_db
        store._entries = {"telegram:12345": MagicMock()}
        # Database has exactly 1 session (the current one just created)
        store._db.session_count.return_value = 1

        assert store.has_any_sessions() is False

    def test_fallback_without_database(self, tmp_path):
        """Should fall back to len(_entries) when DB is not available."""
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._loaded = True
        store._db = None
        store._entries = {"key1": MagicMock(), "key2": MagicMock()}

        # > 1 entries means has sessions
        assert store.has_any_sessions() is True

        store._entries = {"key1": MagicMock()}
        assert store.has_any_sessions() is False


class TestLastPromptTokens:
    """Tests for the last_prompt_tokens field — actual API token tracking."""

    def test_session_entry_default(self):
        """New sessions should have last_prompt_tokens=0."""
        from gateway.session import SessionEntry
        from datetime import datetime
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        assert entry.last_prompt_tokens == 0

    def test_session_entry_roundtrip(self):
        """last_prompt_tokens should survive serialization/deserialization."""
        from gateway.session import SessionEntry
        from datetime import datetime
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_prompt_tokens=42000,
        )
        d = entry.to_dict()
        assert d["last_prompt_tokens"] == 42000
        restored = SessionEntry.from_dict(d)
        assert restored.last_prompt_tokens == 42000

    def test_session_entry_from_old_data(self):
        """Old session data without last_prompt_tokens should default to 0."""
        from gateway.session import SessionEntry
        data = {
            "session_key": "test",
            "session_id": "s1",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            # No last_prompt_tokens — old format
        }
        entry = SessionEntry.from_dict(data)
        assert entry.last_prompt_tokens == 0

    def test_update_session_sets_last_prompt_tokens(self, tmp_path):
        """update_session should store the actual prompt token count."""
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._loaded = True
        store._db = None
        store._save = MagicMock()

        from gateway.session import SessionEntry
        from datetime import datetime
        entry = SessionEntry(
            session_key="k1",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        store._entries = {"k1": entry}

        store.update_session("k1", last_prompt_tokens=85000)
        assert entry.last_prompt_tokens == 85000

    def test_update_session_none_does_not_change(self, tmp_path):
        """update_session with default (None) should not change last_prompt_tokens."""
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._loaded = True
        store._db = None
        store._save = MagicMock()

        from gateway.session import SessionEntry
        from datetime import datetime
        entry = SessionEntry(
            session_key="k1",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_prompt_tokens=50000,
        )
        store._entries = {"k1": entry}

        store.update_session("k1")  # No last_prompt_tokens arg
        assert entry.last_prompt_tokens == 50000  # unchanged

    def test_update_session_zero_resets(self, tmp_path):
        """update_session with last_prompt_tokens=0 should reset the field."""
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._loaded = True
        store._db = None
        store._save = MagicMock()

        from gateway.session import SessionEntry
        from datetime import datetime
        entry = SessionEntry(
            session_key="k1",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            last_prompt_tokens=85000,
        )
        store._entries = {"k1": entry}

        store.update_session("k1", last_prompt_tokens=0)
        assert entry.last_prompt_tokens == 0

class TestRewriteTranscriptPreservesReasoning:
    """rewrite_transcript must not drop reasoning fields from SQLite."""

    def test_reasoning_survives_rewrite(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "test.db")
        session_id = "reasoning-test"
        db.create_session(session_id=session_id, source="cli")

        # Insert a message WITH all three reasoning fields
        db.append_message(
            session_id=session_id,
            role="assistant",
            content="The answer is 42.",
            reasoning="I need to think step by step.",
            reasoning_content="provider scratchpad",
            reasoning_details=[{"type": "summary", "text": "step by step"}],
            codex_reasoning_items=[{"id": "r1", "type": "reasoning"}],
        )

        # Verify all three were stored
        before = db.get_messages_as_conversation(session_id)
        assert before[0].get("reasoning") == "I need to think step by step."
        assert before[0].get("reasoning_content") == "provider scratchpad"
        assert before[0].get("reasoning_details") == [{"type": "summary", "text": "step by step"}]
        assert before[0].get("codex_reasoning_items") == [{"id": "r1", "type": "reasoning"}]

        # Now simulate /retry: build the SessionStore and call rewrite_transcript
        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True

        # rewrite_transcript receives the messages that load_transcript returned
        store.rewrite_transcript(session_id, before)

        # Load again — all three reasoning fields must survive
        after = db.get_messages_as_conversation(session_id)
        assert after[0].get("reasoning") == "I need to think step by step."
        assert after[0].get("reasoning_content") == "provider scratchpad"
        assert after[0].get("reasoning_details") == [{"type": "summary", "text": "step by step"}]
        assert after[0].get("codex_reasoning_items") == [{"id": "r1", "type": "reasoning"}]

    def test_db_rewrite_is_atomic_on_insert_failure(self, tmp_path, monkeypatch):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "test.db")
        session_id = "atomic-rewrite-test"
        db.create_session(session_id=session_id, source="cli")
        db.append_message(session_id=session_id, role="user", content="before user")
        db.append_message(session_id=session_id, role="assistant", content="before assistant")

        config = GatewayConfig()
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True

        # Force the second insert inside replace_messages to fail, simulating
        # any storage-layer error that might abort a multi-row rewrite.
        real_encode = SessionDB._encode_content
        calls = {"n": 0}

        def flaky_encode(cls, content):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated storage failure")
            return real_encode.__func__(cls, content)

        monkeypatch.setattr(SessionDB, "_encode_content", classmethod(flaky_encode))

        replacement = [
            {"role": "user", "content": "after user"},
            {"role": "assistant", "content": "after assistant"},
        ]

        store.rewrite_transcript(session_id, replacement)

        # The rewrite must roll back atomically — original messages preserved.
        after = db.get_messages_as_conversation(session_id)
        assert [msg["content"] for msg in after] == [
            "before user",
            "before assistant",
        ]


class TestWorkspaceScopedTranscriptRouting:
    """Regression: workspace-bound gateway sessions must write JSONL transcripts
    to the workspace-scoped sessions directory, not the shared platform-level
    trajectories directory.

    See docs/canonical/architecture.md §0.4A — workspace-bound session JSONL belongs in
    ``workspaces/<workspace_id>/.hermes/sessions/`` via workspace_session_logs.
    """

    @pytest.fixture()
    def store(self, tmp_path):
        config = GatewayConfig()
        platform_sessions_dir = tmp_path / "trajectories"
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=platform_sessions_dir, config=config)
        s._db = None
        s._loaded = True
        return s, platform_sessions_dir

    def test_unregistered_session_returns_none(self, store, tmp_path):
        s, platform_sessions_dir = store
        session_id = "unbound_session"
        path = s.get_transcript_path(session_id)
        assert path is None, (
            "Non-workspace sessions must not receive a shared filesystem path; "
            "SQLite is the primary store for these sessions."
        )

    def test_registered_workspace_uses_workspace_dir(self, store, tmp_path):
        """After register_workspace_home, transcript path must point to workspace dir."""
        import sys, os
        # Ensure agents package is importable from src/
        src_dir = str(Path(__file__).resolve().parents[3] / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        s, platform_sessions_dir = store
        workspace_id = "abc123"
        workspace_hermes_home = tmp_path / "workspaces" / workspace_id
        workspace_hermes_home.mkdir(parents=True)

        session_id = f"{workspace_id}:session_testdeadbeef"
        s.register_workspace_home(session_id, workspace_hermes_home)

        path = s.get_transcript_path(session_id)
        # Must be inside the workspace sessions dir, NOT the platform dir
        assert platform_sessions_dir not in path.parents
        assert (workspace_hermes_home / "sessions") in path.parents or \
               path.parent == workspace_hermes_home / "sessions"

    def test_workspace_bound_session_never_falls_back_to_shared_trajectory_jsonl(self, tmp_path):
        config = GatewayConfig(sessions_dir=tmp_path / "forbidden-shared")
        s = SessionStore(sessions_dir=config.sessions_dir, config=config)
        s._db = None

        workspace_home = tmp_path / "workspaces" / "ws-123" / ".hermes"
        workspace_home.mkdir(parents=True, exist_ok=True)

        session_id = "ws-123:session_abc"
        s.register_workspace_home(session_id, workspace_home)

        path = s.get_transcript_path(session_id)
        assert "forbidden-shared" not in str(path)
        assert path == (
            workspace_home / "sessions" / "session_abc" / "logs" / "session_abc.jsonl"
        )
        assert not (workspace_home / "sessions" / "ws-123%3Asession_abc.jsonl").exists()

    def test_append_to_transcript_writes_to_workspace_dir(self, store, tmp_path):
        """append_to_transcript for a workspace-bound session must write to workspace dir."""
        import sys
        src_dir = str(Path(__file__).resolve().parents[3] / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        s, platform_sessions_dir = store
        workspace_id = "ws_deadbeef01"
        workspace_hermes_home = tmp_path / "workspaces" / workspace_id / ".hermes"
        workspace_hermes_home.mkdir(parents=True)

        session_id = f"{workspace_id}:session_abc123456789"
        s.register_workspace_home(session_id, workspace_hermes_home)
        s.append_to_transcript(session_id, {"role": "user", "content": "hello from weixin"})

        expected_path = s.get_transcript_path(session_id)
        assert expected_path == (
            workspace_hermes_home
            / "sessions"
            / "session_abc123456789"
            / "logs"
            / "session_abc123456789.jsonl"
        )
        assert expected_path.exists(), f"Expected transcript at {expected_path}"

        # Must NOT have written to the platform-level dir
        platform_path = platform_sessions_dir / f"{session_id}.jsonl"
        assert not platform_path.exists(), \
            f"JSONL must not be in platform dir but found {platform_path}"
        assert not (
            workspace_hermes_home / "sessions" / f"{workspace_id}%3Asession_abc123456789.jsonl"
        ).exists()

        lines = expected_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        import json
        msg = json.loads(lines[0])
        assert msg["content"] == "hello from weixin"

    def test_workspace_session_meta_persists_sandbox_key_log_trail(self, store, tmp_path):
        """A sandbox-bound gateway session must leave its sandbox key in the JSONL audit trail."""
        import sys

        src_dir = str(Path(__file__).resolve().parents[3] / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from agents.sandbox_scope import SandboxScope, bind_sandbox_scope, current_sandbox_key

        s, platform_sessions_dir = store
        workspace_id = "ws_sandboxtrail01"
        workspace_hermes_home = tmp_path / "workspaces" / workspace_id / ".hermes"
        workspace_hermes_home.mkdir(parents=True)

        session_id = f"{workspace_id}:session_abc123456789"
        s.register_workspace_home(session_id, workspace_hermes_home)

        scope = SandboxScope(
            workspace_id=workspace_id,
            lane="interactive_session",
            scope_id=session_id,
            adapter_key=None,
            network_enabled=False,
        )
        with bind_sandbox_scope(scope):
            s.append_to_transcript(
                session_id,
                {
                    "role": "session_meta",
                    "tools": [],
                    "model": "qwen3.5-plus",
                    "platform": "weixin",
                    "sandbox_key": current_sandbox_key(),
                },
                skip_db=True,
            )

        expected_path = s.get_transcript_path(session_id)
        assert expected_path.exists(), f"Expected transcript at {expected_path}"
        assert not (platform_sessions_dir / f"{session_id}.jsonl").exists()

        lines = expected_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        msg = json.loads(lines[0])
        assert msg["role"] == "session_meta"
        assert msg["sandbox_key"] == f"ws:{workspace_id}:session:{session_id}"

    def test_session_meta_prefers_agent_result_sandbox_key(self, monkeypatch):
        gateway_run = importlib.import_module("gateway.run")
        sandbox_key = "ws:ws_gatewaysandbox01:session:ws_gatewaysandbox01:session_test123456"

        monkeypatch.setattr("agents.sandbox_scope.current_sandbox_key", lambda: None)

        assert gateway_run._resolve_session_meta_sandbox_key({"sandbox_key": sandbox_key}) == sandbox_key
        assert gateway_run._resolve_session_meta_sandbox_key({}) is None

    def test_persist_workspace_session_sandbox_key_backfills_canonical_log(self, tmp_path):
        import sys

        src_dir = str(Path(__file__).resolve().parents[3] / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from agents.workspace_session_logs import (
            create_workspace_session_log,
            get_workspace_session_log_payload,
        )

        gateway_run = importlib.import_module("gateway.run")

        workspace_id = "ws_backfilltrail01"
        session_id = f"{workspace_id}:session_abc123456789"
        sandbox_key = f"ws:{workspace_id}:session:{session_id}"
        workspace_hermes_home = tmp_path / "workspaces" / workspace_id / ".hermes"
        workspace_hermes_home.mkdir(parents=True, exist_ok=True)

        create_workspace_session_log(
            workspace_hermes_home,
            session_id=session_id,
            title="Backfill Trail",
            source="weixin",
            platform="weixin",
        )

        gateway_run._persist_workspace_session_sandbox_key(
            workspace_hermes_home,
            session_id,
            sandbox_key,
        )

        payload = get_workspace_session_log_payload(workspace_hermes_home, session_id)
        assert payload is not None
        assert payload["sandbox_key"] == sandbox_key

    @pytest.mark.asyncio
    async def test_handle_message_first_turn_uses_workspace_home_without_nameerror(self, tmp_path, monkeypatch):
        gateway_run = importlib.import_module("gateway.run")
        GatewayRunner = gateway_run.GatewayRunner

        config = GatewayConfig()
        store = SessionStore(sessions_dir=tmp_path / "trajectories", config=config)
        store._db = None
        store._loaded = True

        workspace_id = "ws_nameerrorfix01"
        workspace_home = tmp_path / "workspaces" / workspace_id
        workspace_home.mkdir(parents=True, exist_ok=True)
        session_id = f"{workspace_id}:session_abc123456789"
        sandbox_key = f"ws:{workspace_id}:session:{session_id}"

        entry = SessionEntry(
            session_key="agent:main:workspace:ws_nameerrorfix01:weixin:dm:wx-user",
            session_id=session_id,
            created_at="2026-06-03T10:00:00Z",
            updated_at="2026-06-03T10:00:00Z",
            platform=Platform.WEIXIN,
            chat_type="dm",
        )
        store.register_workspace_home(session_id, workspace_home)
        store.get_or_create_session = lambda _source: entry
        store.load_transcript = lambda _session_id: []
        store.update_session = lambda *_args, **_kwargs: None

        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        runner.config = config
        runner.hooks = SimpleNamespace(loaded_hooks=False, emit=AsyncMock())
        runner._session_db = None
        runner._session_model_overrides = {}
        runner._pending_model_notes = {}
        runner._pending_skills_reload_notes = {}
        runner._cache_session_source = lambda *_args, **_kwargs: None
        runner._is_telegram_topic_lane = lambda *_args, **_kwargs: False
        runner._recover_telegram_topic_thread_id = lambda _source: None
        runner._set_session_env = lambda _context: ()
        runner._clear_session_env = lambda _tokens: None
        runner._adapter_for_source = lambda _source: None
        runner._thread_metadata_for_source = lambda *_args, **_kwargs: None
        runner._reply_anchor_for_event = lambda _event: None
        runner._should_send_voice_reply = lambda *_args, **_kwargs: False
        runner._set_session_reasoning_override = lambda *_args, **_kwargs: None
        runner._evict_cached_agent = lambda *_args, **_kwargs: None
        runner._clear_restart_failure_count = lambda *_args, **_kwargs: None
        runner._is_session_run_current = lambda *_args, **_kwargs: True
        runner._bind_adapter_run_generation = lambda *_args, **_kwargs: None
        runner._prepare_inbound_message_text = AsyncMock(return_value="hello")
        runner._deliver_platform_notice = AsyncMock()
        runner._run_agent = AsyncMock(
            return_value={
                "final_response": "done",
                "messages": [],
                "tools": [],
                "api_calls": 1,
                "sandbox_key": sandbox_key,
                "session_id": session_id,
            }
        )

        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        monkeypatch.setattr(gateway_run, "_workspace_hermes_home_for_source", lambda _source: workspace_home)
        monkeypatch.setattr(gateway_run, "_weixin_source_has_user_scoped_home_channel", lambda _source: False)
        monkeypatch.setattr(gateway_run, "_governed_context_prompt_for_source", lambda *_args: None)

        source = SessionSource(
            platform=Platform.WEIXIN,
            chat_id="wx-user",
            chat_type="dm",
            user_id="wx-user",
            workspace_owner_id=workspace_id,
        )
        event = MessageEvent(text="hello", source=source, message_id="msg-1")

        await GatewayRunner._handle_message_with_agent(
            runner,
            event,
            source,
            "quick-key",
            1,
        )

        transcript_path = store.get_transcript_path(session_id)
        assert transcript_path is not None
        assert transcript_path.exists()
        lines = transcript_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines
        msg = json.loads(lines[0])
        assert msg["role"] == "session_meta"
        assert msg["sandbox_key"] == sandbox_key

    def test_authenticated_gateway_turn_binds_workspace_session_paths_before_first_write(self, tmp_path):
        workspace_home = tmp_path / "workspaces" / "ws-123"
        shared_dir = tmp_path / ".semantier-home" / "trajectories"
        shared_dir.mkdir(parents=True, exist_ok=True)

        config = GatewayConfig(sessions_dir=shared_dir)
        s = SessionStore(sessions_dir=config.sessions_dir, config=config)
        s._db = None

        session_id = "ws-123:session_abc"
        s.register_workspace_home(session_id, workspace_home)
        s.append_to_transcript(session_id, {"role": "user", "content": "hi"}, skip_db=True)

        assert not (shared_dir / f"{session_id}.jsonl").exists()
        assert (
            workspace_home / "sessions" / "session_abc" / "logs" / "session_abc.jsonl"
        ).exists()
        assert not (workspace_home / "sessions" / "ws-123%3Asession_abc.jsonl").exists()

    def test_other_sessions_unaffected_after_registration(self, store, tmp_path):
        """Registering a workspace home for one session must not affect others."""
        import sys
        src_dir = str(Path(__file__).resolve().parents[3] / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        s, platform_sessions_dir = store
        workspace_id = "ws_only_for_one"
        workspace_hermes_home = tmp_path / "workspaces" / workspace_id
        workspace_hermes_home.mkdir(parents=True)

        workspace_session_id = f"{workspace_id}:session_bound"
        other_session_id = "standalone_session"

        s.register_workspace_home(workspace_session_id, workspace_hermes_home)

        # The other session has no workspace registration — must return None
        # (no shared-filesystem fallback is allowed)
        other_path = s.get_transcript_path(other_session_id)
        assert other_path is None, (
            "Unregistered sessions must not receive a shared filesystem path; "
            "got: " + str(other_path)
        )
