"""Bundled memory provider backed by workspace-scoped Semantica state."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider


def _semantier_root() -> Path:
    configured = os.environ.get("SEMANTIER_RUNTIME_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = next(
            (
                candidate
                for candidate in Path(__file__).resolve().parents
                if (candidate / "src" / "semantier").is_dir()
            ),
            Path(__file__).resolve().parents[4],
        )
    if not (root / "src" / "semantier").is_dir():
        raise RuntimeError(
            "Semantica memory requires the parent Semantier runtime; set "
            f"SEMANTIER_RUNTIME_ROOT (resolved candidate: {root})"
        )
    return root


_SRC_ROOT = _semantier_root() / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from semantier.knowledge_feature_packs.semantica_adapter import (  # noqa: E402
    _ensure_vendored_semantica_importable,
)

_ensure_vendored_semantica_importable()

from semantica.context import ContextGraph  # noqa: E402
from semantica.provenance import ProvenanceManager  # noqa: E402
from services.semantica_mvl_control_adapter import SemanticaMvlControlAdapter  # noqa: E402


def _error(message: str) -> str:
    return json.dumps({"error": message})


SEMANTICA_STATE_SCHEMA = {
    "name": "semantica_workspace_state",
    "description": "Inspect workspace-scoped Semantica candidate-memory state.",
    "parameters": {
        "type": "object",
        "properties": {
            "include_paths": {
                "type": "boolean",
                "description": "Include local state paths for integration evidence.",
                "default": False,
            }
        },
    },
}

SEMANTICA_GRAPH_SCHEMA = {
    "name": "semantica_graph",
    "description": "Mutate or query the workspace-scoped Semantica ContextGraph.",
    "parameters": {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add_node", "add_edge", "get_node", "snapshot_graph", "snapshot_ontology"],
            },
            "node_id": {"type": "string"},
            "node_type": {"type": "string"},
            "content": {"type": "string"},
            "source_id": {"type": "string"},
            "target_id": {"type": "string"},
            "edge_type": {"type": "string"},
            "version_label": {"type": "string"},
            "experiment_id": {"type": "string"},
            "actor_id": {"type": "string"},
            "properties": {"type": "object"},
            "ontology_payload": {"type": "object"},
        },
    },
}


class SemanticaMemoryProvider(MemoryProvider):
    """Own real Semantica state beneath one governed workspace root."""

    def __init__(self) -> None:
        self._session_id = ""
        self._organization_id = "local"
        self._workspace_id = "default"
        self._workspace_root: Path | None = None
        self._state_root: Path | None = None
        self._graph_path: Path | None = None
        self._graph: Any | None = None
        self._provenance_manager: Any | None = None
        self._control_adapter: SemanticaMvlControlAdapter | None = None

    @property
    def name(self) -> str:
        return "semantica"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        hermes_home = Path(str(kwargs.get("hermes_home") or Path.cwd())).resolve()
        workspace_id = str(
            kwargs.get("workspace_id")
            or kwargs.get("agent_workspace")
            or kwargs.get("user_id")
            or "default"
        )
        if (
            workspace_id in {"", ".", ".."}
            or workspace_id != Path(workspace_id).name
            or not re.fullmatch(r"[A-Za-z0-9._:-]+", workspace_id)
        ):
            raise ValueError("workspace_id must be an ASCII-stable path segment")
        organization_id = str(kwargs.get("organization_id") or "local")
        self._session_id = session_id
        self._organization_id = organization_id
        self._workspace_id = workspace_id
        self._workspace_root = hermes_home / "workspaces" / workspace_id
        self._state_root = self._workspace_root / ".semantica-memory"
        self._state_root.mkdir(parents=True, exist_ok=True)
        self._graph_path = self._state_root / "context_graph.json"

        graph = ContextGraph(graph_id=f"semantica-memory:{workspace_id}")
        if self._graph_path.exists():
            graph.load_from_file(str(self._graph_path))
        self._control_adapter = SemanticaMvlControlAdapter(
            self._workspace_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
            domain_id="semantica-memory",
        )
        self._control_adapter.temporal_version_manager.attach_to_graph(graph)
        self._provenance_manager = ProvenanceManager(
            storage_path=str(self._state_root / "provenance.sqlite3")
        )
        self._graph = graph

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEMANTICA_STATE_SCHEMA, SEMANTICA_GRAPH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if tool_name == "semantica_workspace_state":
            return json.dumps(self._workspace_state(bool(args.get("include_paths", False))))
        if tool_name == "semantica_graph":
            return self._handle_graph(args)
        return _error(f"Unknown tool: {tool_name}")

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._graph or not query:
            return ""
        matches = [
            self._node_payload(node)
            for node in self._graph.nodes.values()
            if query.lower() in str(node.content).lower()
        ][:5]
        if not matches:
            return ""
        lines = [f"- {item.get('id') or item.get('node_id')}: {item.get('content')}" for item in matches]
        return (
            "## Semantica Memory\n"
            "The following is candidate memory; not an authority source.\n"
            + "\n".join(lines)
        )

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        self._session_id = new_session_id

    def shutdown(self) -> None:
        self._persist_graph()
        self._graph = None
        self._provenance_manager = None
        self._control_adapter = None

    def backup_paths(self) -> List[str]:
        # State lives beneath HERMES_HOME and is already captured by core backup.
        return []

    def _handle_graph(self, args: Dict[str, Any]) -> str:
        if not self._graph or not self._control_adapter:
            return _error("Semantica provider is not initialized")
        action = args.get("action")
        properties = args.get("properties") or {}
        try:
            if action == "add_node":
                node_id = str(args["node_id"])
                self._graph.add_node(
                    node_id,
                    str(args.get("node_type") or "concept"),
                    content=args.get("content"),
                    **properties,
                )
                self._persist_graph()
                return json.dumps({"status": "added", "node_id": node_id})
            if action == "add_edge":
                self._graph.add_edge(
                    str(args["source_id"]),
                    str(args["target_id"]),
                    str(args.get("edge_type") or "related_to"),
                    **properties,
                )
                self._persist_graph()
                return json.dumps({"status": "added"})
            if action == "get_node":
                node = self._graph.nodes.get(str(args["node_id"]))
                return json.dumps({"node": self._node_payload(node) if node else None})
            if action == "snapshot_graph":
                label = str(args["version_label"])
                snapshot = self._control_adapter.create_native_graph_version(
                    graph_payload=self._export_graph_payload(),
                    version_label=label,
                    author="semantier-mvl@system.invalid",
                    description=f"Semantica memory graph snapshot {label}.",
                    experiment_id=str(args.get("experiment_id") or "default"),
                    actor_id=str(args.get("actor_id") or "system"),
                    source_ref=args.get("source_ref"),
                    source_hash=args.get("source_hash"),
                    semantic_stage=str(args.get("semantic_stage") or "proposed"),
                    control_action=args.get("control_action"),
                )
                return json.dumps({"snapshot": snapshot})
            if action == "snapshot_ontology":
                label = str(args["version_label"])
                payload = args.get("ontology_payload") or {
                    "uri": f"urn:semantier:ontology:{self._workspace_id}",
                    "version_info": {"version": label},
                    "structure": {"classes": [], "properties": [], "individuals": [], "axioms": []},
                }
                snapshot = self._control_adapter.create_native_ontology_version(
                    ontology_payload=payload,
                    version_label=label,
                    author="semantier-mvl@system.invalid",
                    description=f"Semantica memory ontology snapshot {label}.",
                    experiment_id=str(args.get("experiment_id") or "default"),
                    actor_id=str(args.get("actor_id") or "system"),
                    source_ref=args.get("source_ref"),
                    source_hash=args.get("source_hash"),
                    semantic_stage=str(args.get("semantic_stage") or "proposed"),
                    control_action=args.get("control_action"),
                )
                return json.dumps({"snapshot": snapshot})
            return _error(f"Unknown action: {action}")
        except KeyError as exc:
            return _error(f"Missing required argument: {exc}")
        except Exception as exc:
            return _error(str(exc))

    def _workspace_state(self, include_paths: bool) -> Dict[str, Any]:
        if not self._graph or not self._state_root or not self._control_adapter:
            return {"provider_name": self.name, "initialized": False}
        state = {
            "provider_name": self.name,
            "provider_class": type(self).__name__,
            "workspace_id": self._workspace_id,
            "organization_id": self._organization_id,
            "session_id": self._session_id,
            "graph_id": self._graph.graph_id,
            "node_count": len(self._graph.nodes),
            "edge_count": len(self._graph.edges),
            "graph_version_tags": self._control_adapter.temporal_version_manager.list_tags(),
            "ontology_version_tags": self._control_adapter.ontology_version_manager.storage.list_tags(),
        }
        if include_paths:
            state.update(
                workspace_root=str(self._workspace_root),
                state_root=str(self._state_root),
                graph_path=str(self._graph_path),
                graph_versions_path=str(
                    self._control_adapter.semantica_state_root / "graph_versions.sqlite3"
                ),
                ontology_versions_path=str(
                    self._control_adapter.semantica_state_root / "ontology_versions.sqlite3"
                ),
            )
        return state

    def _persist_graph(self) -> None:
        if self._graph and self._graph_path:
            self._graph.save_to_file(str(self._graph_path))

    def _export_graph_payload(self) -> Dict[str, Any]:
        if not self._graph:
            return {"graph_id": "", "nodes": [], "edges": []}
        return {
            "graph_id": self._graph.graph_id,
            "nodes": [node.to_dict() for node in self._graph.nodes.values()],
            "edges": [edge.to_dict() for edge in self._graph.edges],
        }

    @staticmethod
    def _node_payload(node: Any) -> Dict[str, Any]:
        payload = node.to_dict()
        payload.setdefault("id", getattr(node, "node_id", ""))
        payload.setdefault("content", getattr(node, "content", ""))
        payload.setdefault("type", getattr(node, "node_type", ""))
        return payload


def register(ctx: Any) -> None:
    ctx.register_memory_provider(SemanticaMemoryProvider())
