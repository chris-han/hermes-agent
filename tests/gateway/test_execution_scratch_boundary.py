from dataclasses import replace
from pathlib import Path
import tempfile

import pytest

from gateway import execution_boundary as boundary


def _scope(root):
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    return boundary.ExecutionBoundary(
        source="test", paths=boundary.BoundaryPaths(artifacts_root=artifacts),
        policy=boundary.BoundaryPolicy(require_boundary=True, allowed_write_roots=(artifacts,)),
    )


def test_scratch_uses_each_active_session_and_preserves_standalone(tmp_path):
    scopes = [_scope(tmp_path / "a"), _scope(tmp_path / "b")]
    original = boundary.get_execution_boundary_provider()
    boundary.clear_execution_boundary_provider()
    try:
        assert boundary.execution_scratch_dir() is None
        for scope in [scopes[0], scopes[1], scopes[0]]:
            with boundary.bind_execution_boundary(scope):
                with tempfile.TemporaryDirectory(dir=boundary.execution_scratch_dir()) as directory:
                    path = Path(directory)
                    assert path.is_relative_to(scope.paths.artifacts_root)
                    (path / "result.txt").write_text("owned")
        assert boundary.execution_scratch_dir() is None
    finally:
        boundary.replace_execution_boundary_provider(original)


@pytest.mark.parametrize("case", ["missing_boundary", "missing_artifacts", "excluded_root", "symlink_escape"])
def test_scratch_never_falls_back_when_governed_scope_is_unavailable(tmp_path, case):
    scope = _scope(tmp_path / "workspace")
    original = boundary.get_execution_boundary_provider()
    boundary.replace_execution_boundary_provider(object())
    try:
        if case == "missing_boundary":
            with pytest.raises(boundary.GovernedExecutionBoundaryRequired):
                boundary.execution_scratch_dir()
            return
        if case == "missing_artifacts":
            scope = replace(scope, paths=boundary.BoundaryPaths())
        elif case == "excluded_root":
            scope = replace(scope, policy=boundary.BoundaryPolicy(allowed_write_roots=(tmp_path / "other",)))
        else:
            outside = tmp_path / "outside"
            outside.mkdir()
            (scope.paths.artifacts_root / "tmp").symlink_to(outside, target_is_directory=True)
        with boundary.bind_execution_boundary(scope):
            with pytest.raises(boundary.BoundaryPathRejected):
                boundary.execution_scratch_dir()
    finally:
        boundary.replace_execution_boundary_provider(original)
