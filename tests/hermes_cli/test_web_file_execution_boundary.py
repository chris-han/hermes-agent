import asyncio
import os

import pytest
from fastapi import HTTPException

from gateway import execution_boundary as boundary
from hermes_cli.web_models import FsWriteText
from hermes_cli.web_routers.files import fs_read_text, fs_write_text
from hermes_cli.web_server_files import _ensure_managed_root


@pytest.mark.parametrize("escape", ["absolute", "traversal", "symlink", "staging"])
def test_dashboard_respects_session_io_roots(tmp_path, escape):
    artifacts = tmp_path / "workspace" / "artifacts"
    artifacts.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("unchanged")
    inside = artifacts / "inside.txt"
    scope = boundary.ExecutionBoundary(
        source="test", paths=boundary.BoundaryPaths(artifacts_root=artifacts),
        policy=boundary.BoundaryPolicy(require_boundary=True, allowed_read_roots=(artifacts,), allowed_write_roots=(artifacts,)),
    )
    target = outside
    if escape == "traversal":
        target = artifacts / ".." / ".." / "outside.txt"
    elif escape == "symlink":
        target = artifacts / "escape.txt"
        target.symlink_to(outside)
    elif escape == "staging":
        target = inside
        target.with_name(f".{target.name}.hermes-tmp-{os.getpid()}").symlink_to(outside)
    with boundary.bind_execution_boundary(scope):
        if escape != "staging":
            asyncio.run(fs_write_text(FsWriteText(path=str(inside), content="inside")))
            assert asyncio.run(fs_read_text(str(inside)))["text"] == "inside"
            with pytest.raises(HTTPException) as denied:
                asyncio.run(fs_read_text(str(target)))
            assert denied.value.status_code == 403
        with pytest.raises(HTTPException) as denied:
            asyncio.run(fs_write_text(FsWriteText(path=str(target), content="denied")))
        assert denied.value.status_code == 403
    assert outside.read_text() == "unchanged"


def test_dashboard_rejects_missing_governed_context_before_io(tmp_path):
    target = tmp_path / "outside.txt"
    target.write_text("unchanged")
    original = boundary.get_execution_boundary_provider()
    boundary.replace_execution_boundary_provider(object())
    try:
        for operation in (
            lambda: asyncio.run(fs_read_text(str(target))),
            lambda: asyncio.run(fs_write_text(FsWriteText(path=str(target), content="denied"))),
            lambda: _ensure_managed_root(tmp_path / "unowned-root"),
        ):
            with pytest.raises(HTTPException) as denied:
                operation()
            assert denied.value.status_code == 403
    finally:
        boundary.replace_execution_boundary_provider(original)
    assert target.read_text() == "unchanged"
    assert not (tmp_path / "unowned-root").exists()


@pytest.mark.parametrize("operation", ["delete_readonly", "list_escape"])
def test_managed_routes_check_actual_io_authority(tmp_path, monkeypatch, operation):
    from fastapi import Request
    from hermes_cli.web_models import ManagedFileDelete
    from hermes_cli.web_routers.files import delete_managed_file, list_managed_files

    workspace = tmp_path / "workspace"
    artifacts = workspace / "artifacts"
    artifacts.mkdir(parents=True)
    readonly = workspace / "readonly.txt"
    readonly.write_text("preserved")
    outside = tmp_path / "outside.txt"
    outside.write_text("private metadata")
    (artifacts / "escape").symlink_to(outside)
    monkeypatch.setenv("HOME", str(workspace))
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    scope = boundary.ExecutionBoundary(
        source="test", paths=boundary.BoundaryPaths(hermes_home=workspace, artifacts_root=artifacts),
        policy=boundary.BoundaryPolicy(require_boundary=True, allowed_read_roots=(workspace,), allowed_write_roots=(artifacts,)),
    )
    request = Request({"type": "http", "headers": []})
    with boundary.bind_execution_boundary(scope):
        with pytest.raises(HTTPException) as denied:
            if operation == "delete_readonly":
                asyncio.run(delete_managed_file(ManagedFileDelete(path=str(readonly)), request))
            else:
                asyncio.run(list_managed_files(request, str(artifacts)))
        assert denied.value.status_code == 403
    assert readonly.read_text() == "preserved"
    assert outside.read_text() == "private metadata"
