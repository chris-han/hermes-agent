"""Tests for gateway/mirror.py — session mirroring."""

import json
from unittest.mock import patch, MagicMock

import gateway.mirror as mirror_mod
from gateway.mirror import (
    mirror_to_session,
    _find_session_id,
    _append_to_jsonl,
)


def _session_rows(sessions_data):
    rows = []
    for key, entry in sessions_data.items():
        rows.append(
            {
                "index_key": key,
                "session_key": key,
                "session_id": entry.get("session_id", ""),
                "platform": (entry.get("origin") or {}).get("platform") or entry.get("platform"),
                "updated_at": entry.get("updated_at", ""),
                "origin": entry.get("origin") if isinstance(entry.get("origin"), dict) else {},
                "workspace_hermes_home": None,
            }
        )
    return rows


def _patched_workspace_matcher(rows):
    def _match(
        *,
        platform=None,
        chat_id=None,
        thread_id=None,
        origin_user_id=None,
        canonical_session_id=None,
        platform_session_key=None,
        workspaces_root=None,
    ):
        del platform_session_key, workspaces_root
        filtered = []
        for row in rows:
            row_platform = str(row.get("platform") or "").lower()
            if platform and row_platform != str(platform).lower():
                continue
            if canonical_session_id and str(row.get("session_id") or "") != str(canonical_session_id):
                continue

            origin = row.get("origin") or {}
            if chat_id is not None and str(origin.get("chat_id") or "") != str(chat_id):
                continue
            if thread_id is not None and str(origin.get("thread_id") or "") != str(thread_id):
                continue
            if origin_user_id is not None and str(origin.get("user_id") or "") != str(origin_user_id):
                continue
            filtered.append(row)

        filtered.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return filtered

    return _match


class TestFindSessionId:
    def test_finds_matching_session(self, tmp_path):
        rows = _session_rows({
            "agent:main:telegram:dm": {
                "session_id": "sess_abc",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher(rows)):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_abc"

    def test_returns_most_recent(self, tmp_path):
        rows = _session_rows({
            "old": {
                "session_id": "sess_old",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "new": {
                "session_id": "sess_new",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher(rows)):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_new"

    def test_thread_id_disambiguates_same_chat(self, tmp_path):
        rows = _session_rows({
            "topic_a": {
                "session_id": "sess_topic_a",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "10"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "topic_b": {
                "session_id": "sess_topic_b",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "11"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher(rows)):
            result = _find_session_id("telegram", "-1001", thread_id="10")

        assert result == "sess_topic_a"

    def test_user_id_disambiguates_same_group_chat(self, tmp_path):
        rows = _session_rows({
            "alice": {
                "session_id": "sess_alice",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "alice"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "bob": {
                "session_id": "sess_bob",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "bob"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher(rows)):
            result = _find_session_id("telegram", "-1001", user_id="alice")

        assert result == "sess_alice"

    def test_ambiguous_same_group_chat_without_user_id_returns_none(self, tmp_path):
        rows = _session_rows({
            "alice": {
                "session_id": "sess_alice",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "alice"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "bob": {
                "session_id": "sess_bob",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "bob"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher(rows)):
            result = _find_session_id("telegram", "-1001")

        assert result is None

    def test_no_match_returns_none(self, tmp_path):
        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher([])):
            result = _find_session_id("telegram", "12345")

        assert result is None

    def test_missing_sessions_file(self, tmp_path):
        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher([])):
            result = _find_session_id("telegram", "12345")

        assert result is None

    def test_platform_case_insensitive(self, tmp_path):
        rows = _session_rows({
            "s1": {
                "session_id": "sess_1",
                "origin": {"platform": "Telegram", "chat_id": "123"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher(rows)):
            result = _find_session_id("telegram", "123")

        assert result == "sess_1"


class TestAppendToJsonl:
    def test_appends_message(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        with patch.object(mirror_mod, "_workspace_jsonl_path", side_effect=lambda sid: sessions_dir / f"{sid}.jsonl"):
            _append_to_jsonl("sess_1", {"role": "assistant", "content": "Hello"})

        transcript = sessions_dir / "sess_1.jsonl"
        lines = transcript.read_text().strip().splitlines()
        assert len(lines) == 1
        msg = json.loads(lines[0])
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello"

    def test_appends_multiple_messages(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        with patch.object(mirror_mod, "_workspace_jsonl_path", side_effect=lambda sid: sessions_dir / f"{sid}.jsonl"):
            _append_to_jsonl("sess_1", {"role": "assistant", "content": "msg1"})
            _append_to_jsonl("sess_1", {"role": "assistant", "content": "msg2"})

        transcript = sessions_dir / "sess_1.jsonl"
        lines = transcript.read_text().strip().splitlines()
        assert len(lines) == 2


class TestMirrorToSession:
    def test_successful_mirror(self, tmp_path):
        sessions_data = {
            "s1": {
                "session_id": "sess_abc",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        }
        rows = _session_rows(sessions_data)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher(rows)), \
             patch("gateway.mirror._workspace_jsonl_path", side_effect=lambda sid: sessions_dir / f"{sid}.jsonl"), \
             patch("gateway.mirror._append_to_sqlite"):
            result = mirror_to_session("telegram", "12345", "Hello!", source_label="cli")

        assert result is True

        # Check JSONL was written
        transcript = sessions_dir / "sess_abc.jsonl"
        assert transcript.exists()
        msg = json.loads(transcript.read_text().strip())
        assert msg["content"] == "Hello!"
        assert msg["role"] == "assistant"
        assert msg["mirror"] is True
        assert msg["mirror_source"] == "cli"

    def test_successful_mirror_uses_thread_id(self, tmp_path):
        sessions_data = {
            "topic_a": {
                "session_id": "sess_topic_a",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "10"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "topic_b": {
                "session_id": "sess_topic_b",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "11"},
                "updated_at": "2026-02-01T00:00:00",
            },
        }
        rows = _session_rows(sessions_data)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher(rows)), \
             patch("gateway.mirror._workspace_jsonl_path", side_effect=lambda sid: sessions_dir / f"{sid}.jsonl"), \
             patch("gateway.mirror._append_to_sqlite"):
            result = mirror_to_session("telegram", "-1001", "Hello topic!", source_label="cron", thread_id="10")

        assert result is True
        assert (sessions_dir / "sess_topic_a.jsonl").exists()
        assert not (sessions_dir / "sess_topic_b.jsonl").exists()

    def test_successful_mirror_uses_user_id_for_group_session(self, tmp_path):
        sessions_data = {
            "alice": {
                "session_id": "sess_alice",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "alice"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "bob": {
                "session_id": "sess_bob",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "bob"},
                "updated_at": "2026-02-01T00:00:00",
            },
        }
        rows = _session_rows(sessions_data)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher(rows)), \
             patch("gateway.mirror._workspace_jsonl_path", side_effect=lambda sid: sessions_dir / f"{sid}.jsonl"), \
             patch("gateway.mirror._append_to_sqlite"):
            result = mirror_to_session(
                "telegram",
                "-1001",
                "Hello group!",
                source_label="cli",
                user_id="alice",
            )

        assert result is True
        assert (sessions_dir / "sess_alice.jsonl").exists()
        assert not (sessions_dir / "sess_bob.jsonl").exists()

    def test_no_matching_session(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher([])):
            result = mirror_to_session("telegram", "99999", "Hello!")

        assert result is False

    def test_error_returns_false(self, tmp_path):
        with patch("gateway.mirror._find_session_id", side_effect=Exception("boom")):
            result = mirror_to_session("telegram", "123", "msg")

        assert result is False

    def test_successful_mirror_for_feishu_workspace_session(self, tmp_path):
        sessions_data = {
            "feishu_chat": {
                "session_id": "ws-feishu:session_abc",
                "origin": {"platform": "feishu", "chat_id": "oc_123"},
                "updated_at": "2026-02-01T00:00:00",
            }
        }
        rows = _session_rows(sessions_data)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        with patch("gateway.mirror._workspace_find_matches", side_effect=_patched_workspace_matcher(rows)), \
             patch("gateway.mirror._workspace_jsonl_path", side_effect=lambda sid: sessions_dir / f"{sid}.jsonl"), \
             patch("gateway.mirror._append_to_sqlite"):
            result = mirror_to_session("feishu", "oc_123", "Hello Feishu", source_label="cli")

        assert result is True
        assert (sessions_dir / "ws-feishu:session_abc.jsonl").exists()


class TestAppendToSqlite:
    def test_connection_is_closed_after_use(self, tmp_path):
        """Verify _append_to_sqlite closes the SessionDB connection."""
        from gateway.mirror import _append_to_sqlite
        mock_db = MagicMock()

        with patch("hermes_state.SessionDB", return_value=mock_db):
            _append_to_sqlite("sess_1", {"role": "assistant", "content": "hello"})

        mock_db.append_message.assert_called_once()
        mock_db.close.assert_called_once()

    def test_connection_closed_even_on_error(self, tmp_path):
        """Verify connection is closed even when append_message raises."""
        from gateway.mirror import _append_to_sqlite
        mock_db = MagicMock()
        mock_db.append_message.side_effect = Exception("db error")

        with patch("hermes_state.SessionDB", return_value=mock_db):
            _append_to_sqlite("sess_1", {"role": "assistant", "content": "hello"})

        mock_db.close.assert_called_once()
