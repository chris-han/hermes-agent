from __future__ import annotations

from collections import deque
from typing import Any

from semantier.semantica_context_graph.artifact_repository import ContextGraphArtifactRepository


class RuntimeGraphBindingError(ValueError):
    pass


class TenderContextGraphRuntimeService:
    """Read-only graph operations bound to one authenticated artifact pin."""

    def __init__(self, repository: ContextGraphArtifactRepository):
        self.repository = repository

    def _graph(self, args: dict[str, Any]):
        metadata = self.repository.get(str(args["artifact_id"]))
        if metadata.graph_version != args["graph_version"] or metadata.content_hash != args["content_hash"]:
            raise RuntimeGraphBindingError("stale or mismatched graph binding")
        if metadata.artifact_kind == "candidate":
            raise RuntimeGraphBindingError("Runtime Expert cannot read candidate graph")
        return metadata, self.repository.load_graph(metadata.artifact_id)

    @staticmethod
    def _node_payload(node: Any) -> dict[str, Any]:
        return {"node_id": node.node_id, "node_type": node.node_type, "content": node.content, "properties": node.properties}

    def search(self, args: dict[str, Any]) -> dict[str, Any]:
        _, graph = self._graph(args)
        query = str(args.get("query") or "").casefold()
        limit = min(int(args.get("limit", 20)), 50)
        matches = [self._node_payload(node) for node in graph.nodes.values() if not query or query in f"{node.content} {node.node_type} {node.properties}".casefold()]
        return {"nodes": matches[:limit]}

    def get_node(self, args: dict[str, Any]) -> dict[str, Any]:
        _, graph = self._graph(args)
        node = graph.nodes.get(str(args["node_id"]))
        if node is None:
            raise RuntimeGraphBindingError("node is not present in pinned graph")
        return {"node": self._node_payload(node)}

    def expand_neighborhood(self, args: dict[str, Any]) -> dict[str, Any]:
        _, graph = self._graph(args)
        node_id = str(args["node_id"])
        if node_id not in graph.nodes:
            raise RuntimeGraphBindingError("node is not present in pinned graph")
        depth = min(int(args.get("depth", 1)), 3)
        seen = {node_id}
        frontier = deque([(node_id, 0)])
        while frontier:
            current, current_depth = frontier.popleft()
            if current_depth >= depth:
                continue
            for edge in graph.edges:
                neighbor = edge.target_id if edge.source_id == current else edge.source_id if edge.target_id == current else None
                if neighbor is not None and neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append((neighbor, current_depth + 1))
        return {"nodes": [self._node_payload(graph.nodes[item]) for item in sorted(seen)]}

    def get_path_evidence(self, args: dict[str, Any]) -> dict[str, Any]:
        _, graph = self._graph(args)
        source, target = str(args["source_node_id"]), str(args["target_node_id"])
        if source not in graph.nodes or target not in graph.nodes:
            raise RuntimeGraphBindingError("path endpoint is not present in pinned graph")
        queue = deque([source])
        parents: dict[str, tuple[str, Any] | None] = {source: None}
        while queue and target not in parents:
            current = queue.popleft()
            for edge in graph.edges:
                neighbor = edge.target_id if edge.source_id == current else edge.source_id if edge.target_id == current else None
                if neighbor is not None and neighbor not in parents:
                    parents[neighbor] = (current, edge)
                    queue.append(neighbor)
        if target not in parents:
            return {"path": [], "evidence": []}
        path_nodes, evidence, current = [target], [], target
        while parents[current] is not None:
            parent, edge = parents[current]
            path_nodes.append(parent)
            evidence.append({"edge_id": edge.edge_id, "type": edge.edge_type, "properties": edge.metadata})
            current = parent
        path_nodes.reverse()
        evidence.reverse()
        return {"path": path_nodes, "evidence": evidence}

    def review(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.search(args)

    def emit_feedback(self, args: dict[str, Any], *, actor_id: str) -> dict[str, Any]:
        metadata, _ = self._graph(args)
        feedback_id = self.repository.record_feedback(
            artifact_id=metadata.artifact_id,
            graph_version=metadata.graph_version,
            content_hash=metadata.content_hash,
            feedback_type=str(args["feedback_type"]),
            note=str(args["note"]),
            source_refs=[str(value) for value in args.get("source_refs", [])],
            actor_id=actor_id,
        )
        return {"feedback_id": feedback_id, "artifact_id": metadata.artifact_id}
