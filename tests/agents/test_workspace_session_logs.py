import json
from pathlib import Path

import pytest

from agents.workspace_session_logs import (
    _index_path,
    _load_index,
    _session_snapshot_path,
    find_workspace_session_index_matches,
    resolve_or_create_workspace_session_id,
)


def test_index_path_uses_sessions_json(tmp_path):
    workspace_home = tmp_path / "ws" / ".hermes"
    assert _index_path(workspace_home).name == "sessions.json"


def test_load_index_missing_file_returns_empty_defaults(tmp_path):
    workspace_home = tmp_path / "ws" / ".hermes"
    data = _load_index(workspace_home)
    assert data == {"sessions": {}, "aliases": {}}


def test_load_index_invalid_json_raises(tmp_path):
    workspace_home = tmp_path / "ws" / ".hermes"
    path = _index_path(workspace_home)
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid workspace session index JSON"):
        _load_index(workspace_home)


def test_load_index_invalid_shape_raises(tmp_path):
    workspace_home = tmp_path / "ws" / ".hermes"
    path = _index_path(workspace_home)
    path.write_text(json.dumps({"sessions": [], "aliases": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="sessions"):
        _load_index(workspace_home)


def test_session_record_round_trips_canonical_metadata(tmp_path, monkeypatch):
    workspaces_root = tmp_path / "workspaces"
    monkeypatch.setattr("agents.workspace_session_logs._workspace_root_candidates", lambda: [workspaces_root])

    workspace_home = workspaces_root / "ws-123" / ".hermes"
    session_id = resolve_or_create_workspace_session_id(
        workspace_home,
        workspace_id="ws-123",
        alias="agent:main:workspace:ws-123:weixin:dm:wx-user",
        platform_session_key="agent:main:workspace:ws-123:weixin:dm:wx-user",
        platform="weixin",
        chat_id="wx-user",
        thread_id="thread-1",
        origin_user_id="wx-user",
        create_if_missing=True,
    )

    index = _load_index(workspace_home)
    record = index["sessions"][session_id]
    assert record == {
        "adapter_key": None,
        "alias": "agent:main:workspace:ws-123:weixin:dm:wx-user",
        "chat_id": "wx-user",
        "delivery_adapter_key": None,
        "origin_user_id": "wx-user",
        "platform": "weixin",
        "platform_session_key": "agent:main:workspace:ws-123:weixin:dm:wx-user",
        "source": None,
        "thread_id": "thread-1",
        "title": "",
        "updated_at": record["updated_at"],
        "workspace_id": "ws-123",
    }
    assert record["updated_at"].endswith("Z")

    snapshot_path = _session_snapshot_path(workspace_home, session_id)
    assert snapshot_path == workspace_home / "sessions" / f"session_{session_id.replace(':', '%3A')}.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["session_id"] == session_id
    assert snapshot["canonical_session_id"] == session_id
    assert snapshot["workspace_id"] == "ws-123"
    assert snapshot["platform"] == "weixin"
    assert snapshot["platform_session_key"] == "agent:main:workspace:ws-123:weixin:dm:wx-user"
    assert snapshot["adapter_key"] is None
    assert snapshot["delivery_adapter_key"] is None
    assert snapshot["updated_at"] == record["updated_at"]

    rows = find_workspace_session_index_matches(canonical_session_id=session_id)
    assert len(rows) == 1
    assert rows[0]["workspace_id"] == "ws-123"
    assert rows[0]["platform"] == "weixin"
    assert rows[0]["session_key"] == "agent:main:workspace:ws-123:weixin:dm:wx-user"
