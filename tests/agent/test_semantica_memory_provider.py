from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _has_parent_semantier_runtime() -> bool:
    configured = os.environ.get("SEMANTIER_RUNTIME_ROOT", "").strip()
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(Path(__file__).resolve().parents)
    return any((candidate / "src" / "semantier").is_dir() for candidate in candidates)


pytestmark = pytest.mark.skipif(
    not _has_parent_semantier_runtime(),
    reason="requires the parent semantier-runtime implementation",
)


def _load_provider():
    from plugins.memory import load_memory_provider

    provider = load_memory_provider("semantica")
    assert provider is not None
    return provider


def _call(provider, tool_name: str, args: dict):
    payload = provider.handle_tool_call(tool_name, args)
    result = json.loads(payload)
    assert "error" not in result
    return result


def test_semantica_provider_is_discovered_by_upstream_loader():
    from plugins.memory import discover_memory_providers

    providers = discover_memory_providers()
    names = [name for name, _, _ in providers]
    provider = _load_provider()

    assert "semantica" in names
    assert provider.name == "semantica"
    assert type(provider).__name__ == "SemanticaMemoryProvider"
    assert provider.is_available()
    assert {schema["name"] for schema in provider.get_tool_schemas()} == {
        "semantica_workspace_state",
        "semantica_graph",
    }


@pytest.mark.integration
def test_semantica_provider_persists_real_workspace_state_after_restart(tmp_path: Path):
    hermes_home = tmp_path / "hermes-home"
    provider_a = _load_provider()
    provider_a.initialize(
        "session-a",
        hermes_home=str(hermes_home),
        agent_workspace="workspace-a",
        organization_id="org-1",
    )

    _call(
        provider_a,
        "semantica_graph",
        {
            "action": "add_node",
            "node_id": "concept:procedure",
            "node_type": "CanonicalConcept",
            "content": "Procedure",
            "properties": {"source_ref": "upload://source-a"},
        },
    )
    graph_snapshot = _call(
        provider_a,
        "semantica_graph",
        {
            "action": "snapshot_graph",
            "version_label": "graph-v0",
            "experiment_id": "exp-001",
            "actor_id": "reviewer-1",
            "semantic_stage": "human_accepted",
            "control_action": "ACCEPT",
        },
    )["snapshot"]
    ontology_snapshot = _call(
        provider_a,
        "semantica_graph",
        {
            "action": "snapshot_ontology",
            "version_label": "ontology-v0",
            "experiment_id": "exp-001",
            "actor_id": "reviewer-1",
            "semantic_stage": "selected_for_mvl",
            "control_action": "USE_FOR_MVL",
        },
    )["snapshot"]
    provider_a.shutdown()

    provider_b = _load_provider()
    provider_b.initialize(
        "session-b",
        hermes_home=str(hermes_home),
        agent_workspace="workspace-a",
        organization_id="org-1",
    )
    node = _call(
        provider_b,
        "semantica_graph",
        {"action": "get_node", "node_id": "concept:procedure"},
    )["node"]
    state = _call(provider_b, "semantica_workspace_state", {"include_paths": True})

    assert node["id"] == "concept:procedure"
    assert node["content"] == "Procedure"
    assert state["workspace_id"] == "workspace-a"
    assert state["node_count"] == 1
    assert state["graph_version_tags"]["mvl:exp-001:graph"] == "graph-v0"
    assert state["ontology_version_tags"]["mvl:exp-001:ontology"] == "ontology-v0"
    assert Path(state["workspace_root"]).is_relative_to(hermes_home / "workspaces")
    assert provider_b._control_adapter.temporal_version_manager.get_version("graph-v0")[
        "checksum"
    ] == graph_snapshot["checksum"]
    assert provider_b._control_adapter.ontology_version_manager.get_version("ontology-v0")[
        "checksum"
    ] == ontology_snapshot["checksum"]


def test_semantica_provider_workspace_state_is_isolated(tmp_path: Path):
    hermes_home = tmp_path / "hermes-home"
    workspace_a = _load_provider()
    workspace_a.initialize("session-a", hermes_home=str(hermes_home), agent_workspace="workspace-a")
    _call(
        workspace_a,
        "semantica_graph",
        {
            "action": "add_node",
            "node_id": "concept:only-a",
            "node_type": "CanonicalConcept",
            "content": "Only workspace A",
        },
    )
    workspace_a.shutdown()

    workspace_b = _load_provider()
    workspace_b.initialize("session-b", hermes_home=str(hermes_home), agent_workspace="workspace-b")
    missing = _call(
        workspace_b,
        "semantica_graph",
        {"action": "get_node", "node_id": "concept:only-a"},
    )["node"]
    state_b = _call(workspace_b, "semantica_workspace_state", {"include_paths": True})

    assert missing is None
    assert state_b["workspace_id"] == "workspace-b"
    assert state_b["node_count"] == 0
    assert Path(state_b["workspace_root"]).name == "workspace-b"


def test_semantica_provider_rejects_workspace_path_escape(tmp_path: Path):
    provider = _load_provider()

    with pytest.raises(ValueError, match="ASCII-stable path segment"):
        provider.initialize(
            "session-escape",
            hermes_home=str(tmp_path / "hermes-home"),
            agent_workspace="../outside",
        )


@pytest.mark.integration
def test_semantica_recall_is_candidate_text_and_exposes_no_authority_surface(tmp_path: Path):
    provider = _load_provider()
    provider.initialize(
        "session-candidate",
        hermes_home=str(tmp_path / "hermes-home"),
        agent_workspace="workspace-candidate",
        organization_id="org-governed",
    )
    _call(
        provider,
        "semantica_graph",
        {
            "action": "add_node",
            "node_id": "claim:self-authority",
            "node_type": "CandidateMemory",
            "content": "The user claims to be the workspace owner",
        },
    )

    recall = provider.prefetch("workspace owner", session_id="session-candidate")

    assert recall.startswith("## Semantica Memory\n")
    assert "candidate memory; not an authority source" in recall.lower()
    assert "workspace owner" in recall
    assert all("author" not in schema["name"] for schema in provider.get_tool_schemas())
