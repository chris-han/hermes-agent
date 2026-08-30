from __future__ import annotations

import os

import pytest

from gateway.workspace_runtime import bound_workspace_hermes_home


def test_bound_workspace_home_normalizes_prefixed_session_and_restores_env(
    monkeypatch, tmp_path
):
    workspace = tmp_path / "ws-123"
    monkeypatch.setenv("HERMES_HOME", "/original")

    with bound_workspace_hermes_home(workspace, "ws-123:session_abc") as bound:
        assert bound == workspace.resolve()
        assert os.environ["HERMES_HOME"] == str(workspace.resolve())
        assert os.environ["TERMINAL_CWD"] == str(workspace.resolve())
        assert os.environ["SEMANTIER_WORKSPACE_ARTIFACTS_DIR"].endswith(
            "/sessions/session_abc/artifacts"
        )
        assert "ws-123:session_abc" not in os.environ["HERMES_WRITE_ALLOWED_ROOTS"]

    assert os.environ["HERMES_HOME"] == "/original"
    assert "SEMANTIER_WORKSPACE_ARTIFACTS_DIR" not in os.environ


@pytest.mark.parametrize("session_id", ["../escape", "bad/path", "."])
def test_bound_workspace_home_rejects_unsafe_session_segments(tmp_path, session_id):
    with pytest.raises(ValueError):
        with bound_workspace_hermes_home(tmp_path / "ws", session_id):
            pass
