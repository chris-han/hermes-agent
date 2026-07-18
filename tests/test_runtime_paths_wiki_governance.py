from __future__ import annotations

import os
from pathlib import Path

import runtime_paths


def test_workspace_binding_exposes_governed_wiki_paths(monkeypatch, tmp_path: Path) -> None:
    workspaces_root = tmp_path / "workspaces"
    monkeypatch.setattr(runtime_paths, "_WORKSPACES_ROOT", workspaces_root)

    sentinels = {
        "WIKI_PATH": "/sentinel/wiki",
        "WIKI_GOVERNANCE_PATH": "/sentinel/governance",
        "WIKI_CONTRACTS_PATH": "/sentinel/contracts",
        "WIKI_REPORTS_PATH": "/sentinel/reports",
        "WIKI_DEPENDENCY_GRAPH_PATH": "/sentinel/dependency-graph.json",
    }
    for key, value in sentinels.items():
        monkeypatch.setenv(key, value)

    workspace_home = runtime_paths.workspace_hermes_home_path("acme")

    with runtime_paths.bind_workspace_env(workspace_home):
        wiki_root = workspaces_root / "acme" / "wiki"
        governance_root = wiki_root / ".governance"

        assert os.environ["WIKI_PATH"] == str(wiki_root.resolve())
        assert os.environ["WIKI_GOVERNANCE_PATH"] == str(governance_root.resolve())
        assert os.environ["WIKI_CONTRACTS_PATH"] == str((governance_root / "contracts").resolve())
        assert os.environ["WIKI_REPORTS_PATH"] == str((governance_root / "reports").resolve())
        assert os.environ["WIKI_DEPENDENCY_GRAPH_PATH"] == str(
            (governance_root / "dependency-graph.json").resolve()
        )

        assert governance_root.is_dir()
        assert (governance_root / "contracts").is_dir()
        assert (governance_root / "reports").is_dir()
        assert "sessions" not in os.environ["WIKI_GOVERNANCE_PATH"]

    for key, value in sentinels.items():
        assert os.environ[key] == value


def test_workspace_wiki_path_helpers_are_workspace_scoped(monkeypatch, tmp_path: Path) -> None:
    workspaces_root = tmp_path / "workspaces"
    monkeypatch.setattr(runtime_paths, "_WORKSPACES_ROOT", workspaces_root)

    assert runtime_paths.workspace_wiki_root("acme") == workspaces_root / "acme" / "wiki"
    assert runtime_paths.workspace_wiki_governance_root("acme") == (
        workspaces_root / "acme" / "wiki" / ".governance"
    )
    assert runtime_paths.workspace_wiki_contracts_root("acme") == (
        workspaces_root / "acme" / "wiki" / ".governance" / "contracts"
    )
    assert runtime_paths.workspace_wiki_reports_root("acme") == (
        workspaces_root / "acme" / "wiki" / ".governance" / "reports"
    )
    assert runtime_paths.workspace_wiki_dependency_graph_path("acme") == (
        workspaces_root / "acme" / "wiki" / ".governance" / "dependency-graph.json"
    )
