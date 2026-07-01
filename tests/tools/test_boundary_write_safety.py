from __future__ import annotations

import json

from gateway.execution_boundary import (
    BoundaryPaths,
    BoundaryPolicy,
    ExecutionBoundary,
    bind_execution_boundary,
)
from tools.file_tools import write_file_tool
from tools.file_tools import read_file_tool


def _boundary(tmp_path, *, with_roots: bool = True) -> ExecutionBoundary:
    workspace_home = tmp_path / "workspace"
    session_root = workspace_home / "sessions" / "session-a"
    paths = BoundaryPaths(
        hermes_home=workspace_home,
        terminal_cwd=workspace_home,
        runs_root=session_root / "runs" if with_roots else None,
        uploads_root=session_root / "uploads" if with_roots else None,
        artifacts_root=session_root / "artifacts" if with_roots else None,
    )
    return ExecutionBoundary(
        source="api_server",
        session_id="session-a",
        user_id="user-a",
        workspace_id="ws-a",
        paths=paths,
        policy=BoundaryPolicy(provider_fallback_enabled=False, require_boundary=True),
    )


def test_boundary_denies_write_outside_session_roots(tmp_path):
    outside = tmp_path / "outside.md"

    with bind_execution_boundary(_boundary(tmp_path)):
        result = json.loads(write_file_tool(str(outside), "nope\n"))

    assert result["error"].startswith("Write denied:")
    assert not outside.exists()


def test_boundary_roots_control_writes_even_if_env_roots_missing(monkeypatch, tmp_path):
    artifacts_file = tmp_path / "workspace" / "sessions" / "session-a" / "artifacts" / "ok.md"

    with bind_execution_boundary(_boundary(tmp_path)):
        monkeypatch.delenv("HERMES_WRITE_ALLOWED_ROOTS", raising=False)
        result = json.loads(write_file_tool(str(artifacts_file), "ok\n"))

    assert result["bytes_written"] == 3
    assert artifacts_file.read_text() == "ok\n"


def test_boundary_without_roots_fails_closed(tmp_path):
    target = tmp_path / "workspace" / "out.md"

    with bind_execution_boundary(_boundary(tmp_path, with_roots=False)):
        result = json.loads(write_file_tool(str(target), "nope\n"))

    assert result["error"].startswith("Write denied:")
    assert not target.exists()


def test_boundary_denies_read_outside_workspace(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("secret\n")

    with bind_execution_boundary(_boundary(tmp_path)):
        result = json.loads(read_file_tool(str(outside)))

    assert "outside the active Semantier execution boundary" in result["error"]


def test_boundary_allows_workspace_read(tmp_path):
    target = tmp_path / "workspace" / "notes.md"
    target.parent.mkdir(parents=True)
    target.write_text("ok\n")

    with bind_execution_boundary(_boundary(tmp_path)):
        result = json.loads(read_file_tool(str(target)))

    assert "ok" in result["content"]


def test_boundary_allows_reviewed_shared_runtime_asset_reads(monkeypatch, tmp_path):
    shared_root = tmp_path / ".semantier-home"
    plugin_file = shared_root / "plugins" / "demo" / "SKILL.md"
    plugin_file.parent.mkdir(parents=True)
    plugin_file.write_text("shared plugin\n")
    blocked_file = shared_root / "logs" / "debug.log"
    blocked_file.parent.mkdir(parents=True)
    blocked_file.write_text("nope\n")
    monkeypatch.setenv("SEMANTIER_LOCAL_STATE_DIR", str(shared_root))

    with bind_execution_boundary(_boundary(tmp_path)):
        allowed = json.loads(read_file_tool(str(plugin_file)))
        blocked = json.loads(read_file_tool(str(blocked_file)))

    assert "shared plugin" in allowed["content"]
    assert "outside the active Semantier execution boundary" in blocked["error"]
