from agent.file_safety import get_read_block_error, get_write_denied_error
from gateway import execution_boundary as boundary


def test_bound_roots_reject_escapes_and_keep_credential_guards(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    artifacts = workspace / "artifacts"
    artifacts.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(workspace))
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
    inside = artifacts / "plain.txt"
    outside = tmp_path / "outside.txt"
    escape = artifacts / "escape"
    escape.symlink_to(outside)
    scope = boundary.ExecutionBoundary(
        source="test", paths=boundary.BoundaryPaths(hermes_home=workspace, artifacts_root=artifacts),
        policy=boundary.BoundaryPolicy(require_boundary=True, allowed_read_roots=(workspace,), allowed_write_roots=(artifacts,)),
    )
    with boundary.bind_execution_boundary(scope):
        assert get_read_block_error(str(inside)) is None
        assert get_write_denied_error(str(inside)) is None
        for path in (outside, escape, workspace / ".." / "outside.txt"):
            assert get_read_block_error(str(path)) is not None
            assert get_write_denied_error(str(path)) is not None
        assert get_read_block_error(str(artifacts / ".env")) is not None
        assert get_write_denied_error(str(workspace / "config.yaml")) is not None


def test_registered_governed_runtime_fails_closed_without_session(tmp_path):
    path = str(tmp_path / "plain.txt")
    original = boundary.get_execution_boundary_provider()
    boundary.replace_execution_boundary_provider(object())
    try:
        assert get_read_block_error(path) is not None
        assert get_write_denied_error(path) is not None
    finally:
        boundary.replace_execution_boundary_provider(original)
