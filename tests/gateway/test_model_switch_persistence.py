"""Tests that gateway /model switch persists across messages.

The gateway /model command stores session overrides in
``_session_model_overrides``.  These must:

1. Be applied in ``run_sync()`` so the next agent uses the switched model.
2. Not be mistaken for fallback activation (which evicts the cached agent).
3. Survive across multiple messages until /reset clears them.

Tests exercise the real ``_apply_session_model_override()`` and
``_is_intentional_model_switch()`` methods on ``GatewayRunner``.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_runner():
    """Create a minimal GatewayRunner with stubbed internals."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._effective_model = None
    runner._effective_provider = None
    runner.session_store = MagicMock()
    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._entries = {session_key: session_entry}
    return runner


# ---------------------------------------------------------------------------
# Tests: _apply_session_model_override
# ---------------------------------------------------------------------------


class TestApplySessionModelOverride:
    """Verify _apply_session_model_override replaces config defaults."""

    def test_override_replaces_all_fields(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4-turbo",
            "provider": "openrouter",
            "api_key": "or-key-123",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        }

        model, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"},
        )

        assert model == "gpt-5.4-turbo"
        assert rt["provider"] == "openrouter"
        assert rt["api_key"] == "or-key-123"
        assert rt["base_url"] == "https://openrouter.ai/api/v1"
        assert rt["api_mode"] == "chat_completions"

    def test_no_override_returns_originals(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        orig_model = "anthropic/claude-sonnet-4"
        orig_rt = {"provider": "anthropic", "api_key": "key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"}

        model, rt = runner._apply_session_model_override(sk, orig_model, dict(orig_rt))

        assert model == orig_model
        assert rt == orig_rt

    def test_none_values_do_not_overwrite(self):
        """Override with None api_key/base_url should preserve config defaults."""
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": None,
            "base_url": None,
            "api_mode": "chat_completions",
        }

        model, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"},
        )

        assert model == "gpt-5.4"
        assert rt["provider"] == "openai"
        assert rt["api_key"] == "ant-key"  # preserved — None didn't overwrite
        assert rt["base_url"] == "https://api.anthropic.com"  # preserved
        assert rt["api_mode"] == "chat_completions"  # overwritten (not None)

    def test_empty_string_overwrites(self):
        """Empty string is not None — it should overwrite the config value."""
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "local-model",
            "provider": "custom",
            "api_key": "local-key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        _, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"},
        )

        assert rt["base_url"] == ""  # empty string overwrites

    def test_different_session_key_not_affected(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        other_sk = "other_session"

        runner._session_model_overrides[other_sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        model, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "url", "api_mode": "anthropic_messages"},
        )

        assert model == "anthropic/claude-sonnet-4"  # unchanged — wrong session key

    def test_resolve_session_runtime_restores_persisted_override(self, monkeypatch):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        entry = runner.session_store._entries[sk]
        entry.session_model_override = {
            "model": "qwen3.6-plus",
            "provider": "alibaba",
            "api_key": "dashscope-key",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_mode": "chat_completions",
        }
        runner._session_model_overrides = {}

        def _should_not_run(*_args, **_kwargs):
            raise AssertionError("global runtime resolution should not run")

        monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", _should_not_run)
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda _cfg: "qwen3.5-plus")

        model, rt = runner._resolve_session_agent_runtime(
            session_key=sk,
            user_config={"model": {"default": "qwen3.5-plus", "provider": "alibaba"}},
        )

        assert model == "qwen3.6-plus"
        assert rt["provider"] == "alibaba"
        assert rt["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert runner._session_model_overrides[sk]["model"] == "qwen3.6-plus"

    def test_resolve_session_runtime_incomplete_override_keeps_model(self, monkeypatch):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        runner._session_model_overrides[sk] = {
            "model": "qwen3.6-plus",
            "provider": None,
            "api_key": None,
            "base_url": None,
            "api_mode": None,
        }
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda _cfg: "qwen3.5-plus")
        captured = {}

        def _fake_runtime(*, requested_provider=None, target_model=None):
            captured["requested_provider"] = requested_provider
            captured["target_model"] = target_model
            return {
                "provider": "alibaba",
                "api_key": "dashscope-key",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_mode": "chat_completions",
            }

        monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", _fake_runtime)

        model, rt = runner._resolve_session_agent_runtime(
            session_key=sk,
            user_config={"model": {"default": "qwen3.5-plus", "provider": "alibaba"}},
        )

        assert model == "qwen3.6-plus"
        assert rt["provider"] == "alibaba"
        assert captured["requested_provider"] == "alibaba"
        assert captured["target_model"] == "qwen3.6-plus"


# ---------------------------------------------------------------------------
# Tests: _is_intentional_model_switch
# ---------------------------------------------------------------------------


class TestIsIntentionalModelSwitch:
    """Verify fallback detection respects intentional /model overrides."""

    def test_matches_override(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        assert runner._is_intentional_model_switch(sk, "gpt-5.4") is True

    def test_no_override_returns_false(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        assert runner._is_intentional_model_switch(sk, "gpt-5.4") is False

    def test_different_model_returns_false(self):
        """Agent fell back to a different model than the override."""
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        assert runner._is_intentional_model_switch(sk, "gpt-5.4-mini") is False

    def test_wrong_session_key(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides["other_session"] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        assert runner._is_intentional_model_switch(sk, "gpt-5.4") is False
