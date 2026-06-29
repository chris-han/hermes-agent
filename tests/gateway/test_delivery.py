"""Tests for the delivery routing module."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.session import SessionSource


class TestParseTargetPlatformChat:
    def test_explicit_telegram_chat(self):
        target = DeliveryTarget.parse("telegram:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
        assert target.is_explicit is True

    def test_platform_only_no_chat_id(self):
        target = DeliveryTarget.parse("discord")
        assert target.platform == Platform.DISCORD
        assert target.chat_id is None
        assert target.is_explicit is False

    def test_local_target(self):
        target = DeliveryTarget.parse("local")
        assert target.platform == Platform.LOCAL
        assert target.chat_id is None

    def test_origin_with_source(self):
        origin = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="789",
            thread_id="42",
            adapter_key="telegram:workspace-1:acct-a",
            delivery_adapter_key="telegram:workspace-1:acct-b",
        )
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "789"
        assert target.thread_id == "42"
        assert target.adapter_key == "telegram:workspace-1:acct-a"
        assert target.delivery_adapter_key == "telegram:workspace-1:acct-b"
        assert target.is_origin is True

    def test_origin_without_source(self):
        target = DeliveryTarget.parse("origin")
        assert target.platform == Platform.LOCAL
        assert target.is_origin is True

    def test_unknown_platform(self):
        target = DeliveryTarget.parse("unknown_platform")
        assert target.platform == Platform.LOCAL


class TestTargetToStringRoundtrip:
    def test_origin_roundtrip(self):
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="111", thread_id="42")
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.to_string() == "origin"

    def test_local_roundtrip(self):
        target = DeliveryTarget.parse("local")
        assert target.to_string() == "local"

    def test_platform_only_roundtrip(self):
        target = DeliveryTarget.parse("discord")
        assert target.to_string() == "discord"

    def test_explicit_chat_roundtrip(self):
        target = DeliveryTarget.parse("telegram:999")
        s = target.to_string()
        assert s == "telegram:999"

        reparsed = DeliveryTarget.parse(s)
        assert reparsed.platform == Platform.TELEGRAM
        assert reparsed.chat_id == "999"


class TestCaseSensitiveChatIdParsing:
    """Test that chat IDs preserve their original case (issue #11768)."""
    
    def test_slack_uppercase_chat_id_preserved(self):
        """Slack channel IDs like C123ABC should preserve case."""
        target = DeliveryTarget.parse("slack:C123ABC")
        assert target.platform == Platform.SLACK
        assert target.chat_id == "C123ABC"  # Should NOT be lowercased to c123abc
        assert target.is_explicit is True
    
    def test_slack_chat_id_with_thread_preserved(self):
        """Slack channel:thread IDs should preserve case."""
        target = DeliveryTarget.parse("slack:C123ABC:thread123")
        assert target.platform == Platform.SLACK
        assert target.chat_id == "C123ABC"
        assert target.thread_id == "thread123"
    
    def test_matrix_room_id_preserved(self):
        """Matrix room IDs like !RoomABC:example.org should preserve case.
        
        Note: Matrix room IDs contain colons (e.g., !RoomABC:example.org).
        Due to the platform:chat_id:thread_id format, these are parsed as
        chat_id=!RoomABC and thread_id=example.org. This is a known limitation
        of the current format. The fix preserves case but doesn't change the
        parsing structure.
        """
        target = DeliveryTarget.parse("matrix:!RoomABC:example.org")
        assert target.platform == Platform.MATRIX
        # The room ID is split at the first colon after the platform prefix
        # This is a format limitation - the case is preserved but the structure is split
        assert target.chat_id == "!RoomABC"
        assert target.thread_id == "example.org"
    
    def test_mixed_case_chat_id_roundtrip(self):
        """Mixed-case chat IDs should survive parse-to_string roundtrip."""
        original = "telegram:ChatId123ABC"
        target = DeliveryTarget.parse(original)
        s = target.to_string()
        reparsed = DeliveryTarget.parse(s)
        assert reparsed.chat_id == "ChatId123ABC"


class TestPlatformNameCaseInsensitivity:
    """Test that platform names are case-insensitive."""
    
    def test_uppercase_platform_name(self):
        """Platform names should be case-insensitive."""
        target = DeliveryTarget.parse("TELEGRAM:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
    
    def test_mixed_case_platform_name(self):
        """Mixed-case platform names should work."""
        target = DeliveryTarget.parse("TeleGram:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"


class TestDeliveryRouterAdapterResolution:
    @pytest.mark.asyncio
    async def test_delivery_prefers_delivery_adapter_key_resolution(self):
        config = GatewayConfig()
        fallback_adapter = AsyncMock()
        resolved_adapter = AsyncMock()
        resolved_adapter.send.return_value = {"ok": True}

        resolver_calls = []

        def resolve_adapter(target):
            resolver_calls.append(target.delivery_adapter_key)
            if target.delivery_adapter_key == "weixin:workspace-1:acct-b":
                return resolved_adapter
            return None

        router = DeliveryRouter(
            config,
            adapters={Platform.WEIXIN: fallback_adapter},
            adapter_resolver=resolve_adapter,
        )
        target = DeliveryTarget(
            platform=Platform.WEIXIN,
            chat_id="wx-chat",
            delivery_adapter_key="weixin:workspace-1:acct-b",
        )

        result = await router.deliver("hello", [target])

        assert result[target.to_string()]["success"] is True
        assert resolver_calls == ["weixin:workspace-1:acct-b"]
        resolved_adapter.send.assert_awaited_once_with("wx-chat", "hello", metadata=None)
        fallback_adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_delivery_with_missing_delivery_adapter_key_fails_closed(self):
        config = GatewayConfig()
        fallback_adapter = AsyncMock()

        router = DeliveryRouter(
            config,
            adapters={Platform.WEIXIN: fallback_adapter},
            adapter_resolver=lambda target: None,
        )
        target = DeliveryTarget(
            platform=Platform.WEIXIN,
            chat_id="wx-chat",
            delivery_adapter_key="weixin:workspace-1:missing",
        )

        result = await router.deliver("hello", [target])

        assert result[target.to_string()]["success"] is False
        assert "delivery adapter" in result[target.to_string()]["error"].lower()
        fallback_adapter.send.assert_not_called()


