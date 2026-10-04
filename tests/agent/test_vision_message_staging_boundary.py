"""The provider transport must not stage authenticated image bytes in host scratch."""

import base64
from pathlib import Path
import tempfile

import pytest

from agent.vision_message_prep import VisionMessagePrepMixin
from gateway.execution_boundary import (
    BoundaryPaths, BoundaryPolicy, ExecutionBoundary,
    GovernedExecutionBoundaryRequired, bind_execution_boundary,
    clear_execution_boundary_provider, get_execution_boundary_provider,
    replace_execution_boundary_provider,
)


@pytest.fixture
def staging_scope(tmp_path, monkeypatch):
    previous = get_execution_boundary_provider()
    clear_execution_boundary_provider()
    host = tmp_path / "host-temp"
    host.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(host))
    runs = tmp_path / "session" / "runs"
    runs.mkdir(parents=True)
    boundary = ExecutionBoundary(
        source="semantier_agent",
        session_id="session",
        paths=BoundaryPaths(runs_root=runs, artifacts_root=runs),
        policy=BoundaryPolicy(require_boundary=True, allowed_write_roots=(runs,)),
    )
    try:
        yield host, runs, boundary
    finally:
        clear_execution_boundary_provider()
        if previous is not None:
            replace_execution_boundary_provider(previous)


def test_image_materialization_uses_current_session_scratch(staging_scope):
    host, runs, boundary = staging_scope
    payload = b"image bytes"
    data_url = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
    with bind_execution_boundary(boundary):
        path, cleanup = VisionMessagePrepMixin._materialize_data_url_for_vision(data_url)
    try:
        assert Path(path).is_relative_to(runs)
        assert Path(path).read_bytes() == payload
        assert list(host.iterdir()) == []
    finally:
        cleanup.unlink()


def test_image_materialization_rejects_missing_governed_boundary(staging_scope):
    host, _, _ = staging_scope
    replace_execution_boundary_provider(object())
    with pytest.raises(GovernedExecutionBoundaryRequired):
        VisionMessagePrepMixin._materialize_data_url_for_vision("data:image/png;base64,aGVsbG8=")
    assert list(host.iterdir()) == []
