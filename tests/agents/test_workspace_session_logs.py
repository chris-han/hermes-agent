import json

import pytest

from agents.workspace_session_logs import _index_path, _load_index


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
