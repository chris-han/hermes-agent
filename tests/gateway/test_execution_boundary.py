from __future__ import annotations

import os
from pathlib import Path

import pytest

from gateway.execution_boundary import (
    BoundaryPaths,
    BoundaryPolicy,
    ExecutionBoundary,
    ExecutionBoundaryRequest,
    bind_execution_boundary,
    clear_execution_boundary_provider,
    current_execution_boundary,
    get_execution_boundary_provider,
    register_execution_boundary_provider,
    replace_execution_boundary_provider,
    resolve_execution_boundary,
)


class _Provider:
    def resolve(self, request: ExecutionBoundaryRequest) -> ExecutionBoundary:
        return ExecutionBoundary(
            source=request.source,
            session_id=request.session_id,
            user_id="user-a",
            workspace_id="ws-a",
            paths=BoundaryPaths(
                hermes_home=Path("/tmp/ws-a"),
                terminal_cwd=Path("/tmp/ws-a"),
                runs_root=Path("/tmp/ws-a/sessions/session-a/runs"),
                uploads_root=Path("/tmp/ws-a/sessions/session-a/uploads"),
                artifacts_root=Path("/tmp/ws-a/sessions/session-a/artifacts"),
            ),
            policy=BoundaryPolicy(provider_fallback_enabled=False),
            audit_metadata={"authority_source": "test"},
        )


def test_provider_registration_rejects_silent_overwrite():
    clear_execution_boundary_provider()
    provider = _Provider()
    register_execution_boundary_provider(provider)

    with pytest.raises(RuntimeError, match="already registered"):
        register_execution_boundary_provider(_Provider())

    replace_execution_boundary_provider(_Provider())
    assert get_execution_boundary_provider() is not provider


def test_provider_resolution_returns_typed_boundary():
    clear_execution_boundary_provider()
    register_execution_boundary_provider(_Provider())

    boundary = resolve_execution_boundary(
        ExecutionBoundaryRequest(
            source="api_server",
            session_id="session-a",
            headers={"x-test": "1"},
        )
    )

    assert boundary is not None
    assert boundary.workspace_id == "ws-a"
    assert boundary.paths.artifacts_root == Path(
        "/tmp/ws-a/sessions/session-a/artifacts"
    )
    assert boundary.policy.provider_fallback_enabled is False
    assert boundary.audit_metadata["authority_source"] == "test"


def test_bind_execution_boundary_uses_contextvar_first_and_env_for_subprocess(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", "/tmp/original")
    boundary = _Provider().resolve(
        ExecutionBoundaryRequest(source="api_server", session_id="session-a")
    )

    assert current_execution_boundary() is None
    with bind_execution_boundary(boundary):
        assert current_execution_boundary() == boundary
        assert os.environ["HERMES_HOME"] == "/tmp/ws-a"
        assert os.environ["TERMINAL_CWD"] == "/tmp/ws-a"
        assert os.environ["HERMES_WRITE_ALLOWED_ROOTS"] == ",".join(
            [
                "/tmp/ws-a/sessions/session-a/runs",
                "/tmp/ws-a/sessions/session-a/uploads",
                "/tmp/ws-a/sessions/session-a/artifacts",
            ]
        )
        assert (
            os.environ["SEMANTIER_WORKSPACE_ARTIFACTS_DIR"]
            == "/tmp/ws-a/sessions/session-a/artifacts"
        )
        assert os.environ["HERMES_PROVIDER_FALLBACK_ENABLED"] == "0"
        assert "SEMANTIER_WORKSPACE_ID" not in os.environ
        assert "SEMANTIER_DISABLE_PROVIDER_FALLBACK" not in os.environ

    assert current_execution_boundary() is None
    assert os.environ["HERMES_HOME"] == "/tmp/original"
    assert "HERMES_PROVIDER_FALLBACK_ENABLED" not in os.environ
    assert "SEMANTIER_WORKSPACE_ARTIFACTS_DIR" not in os.environ
