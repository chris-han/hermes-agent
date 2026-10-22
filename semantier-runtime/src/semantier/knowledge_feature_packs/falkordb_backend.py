"""Immutable-version FalkorDB adapter for accepted Phase 1 graphs."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol


ASCII_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")


class GraphBackend(Protocol):
    def materialize(self, release: dict[str, Any]) -> None: ...
    def verify_release(self, release: dict[str, Any]) -> bool: ...
    def export(self, graph_version: str) -> dict[str, Any]: ...


class FalkorDBGraphBackend:
    def __init__(self, *, client: Any, graph_prefix: str = "tender_sensitive") -> None:
        if not ASCII_ID.fullmatch(graph_prefix):
            raise ValueError("graph prefix must be ASCII-stable")
        self._client = client
        self._graph_prefix = graph_prefix

    @classmethod
    def connect(cls, *, host: str, port: int = 6379) -> "FalkorDBGraphBackend":
        try:
            from falkordb import FalkorDB
        except ImportError as exc:
            raise RuntimeError(
                "FalkorDB backend requires the semantier[knowledge-phase1] extra"
            ) from exc
        return cls(client=FalkorDB(host=host, port=port))

    @classmethod
    def open_lite(
        cls,
        path: str,
        *,
        graph_prefix: str = "tender_sensitive",
    ) -> "FalkorDBGraphBackend":
        """Open a real FalkorDBLite embedded client and inject it into the backend.

        The Lite client ships its own Redis+FalkorDB server, managed in-process.
        The factory reuses the existing materialize/export logic; no parallel
        ``LiteGraphBackend`` semantics implementation is introduced.
        """

        if not isinstance(path, str) or not path.strip():
            raise ValueError("open_lite requires a non-empty persistent DB path")
        try:
            from redislite.falkordb_client import FalkorDB as _LiteFalkorDB
        except ImportError as exc:
            raise RuntimeError(
                "FalkorDBLite backend requires the semantier[knowledge-phase1] extra"
            ) from exc
        return cls(client=_LiteFalkorDB(path), graph_prefix=graph_prefix)

    def materialize(self, release: dict[str, Any]) -> None:
        version = _safe_id(str(release["graph_version"]))
        graph = self._client.select_graph(self._graph_name(version))
        canonical_release = _json(release)
        existing = graph.query(
            "MATCH (m:GraphMetadata) "
            "RETURN m.release_hash, m.release_payload_json LIMIT 1"
        )
        rows = getattr(existing, "result_set", [])
        if rows:
            if rows[0][0] != release["release_hash"]:
                raise ValueError("immutable FalkorDB graph version already has different content")
            stored_payload = rows[0][1]
            if stored_payload is not None and stored_payload != canonical_release:
                raise ValueError(
                    "immutable FalkorDB graph version already has different content"
                )
            if release.get("schema_version") == "accepted_graph_release.v2" and stored_payload is None:
                raise ValueError("v2 FalkorDB graph is missing exact release payload")
            return
        graph.query(
            "CREATE (:GraphMetadata {graph_version:$graph_version, release_hash:$release_hash, "
            "parent_graph_version:$parent_graph_version, "
            "release_payload_json:$release_payload_json})",
            {
                "graph_version": version,
                "release_hash": release["release_hash"],
                "parent_graph_version": release.get("parent_graph_version"),
                "release_payload_json": canonical_release,
            },
        )
        for node in release["nodes"]:
            graph.query(
                "CREATE (:KnowledgeNode {node_id:$node_id, node_type:$node_type, payload_json:$payload_json})",
                {
                    "node_id": _safe_id(str(node["node_id"])),
                    "node_type": str(node["node_type"]),
                    "payload_json": _json(node),
                },
            )

        for edge in release["edges"]:
            graph.query(
                "MATCH (s:KnowledgeNode {node_id:$source}), (t:KnowledgeNode {node_id:$target}) "
                "CREATE (s)-[:KNOWLEDGE_RELATION {edge_id:$edge_id, relation_type:$relation_type, "
                "payload_json:$payload_json}]->(t)",
                {
                    "source": _safe_id(str(edge["source_node_id"])),
                    "target": _safe_id(str(edge["target_node_id"])),
                    "edge_id": _safe_id(str(edge["edge_id"])),
                    "relation_type": str(edge["relation_type"]),
                    "payload_json": _json(edge),
                },
            )
        for rule in release["rules"]:
            graph.query(
                "CREATE (:KnowledgeRule {rule_id:$rule_id, family_id:$family_id, payload_json:$payload_json})",
                {
                    "rule_id": _safe_id(str(rule["rule_id"])),
                    "family_id": int(rule["family_id"]),
                    "payload_json": _json(rule),
                },
            )

    def verify_release(self, release: dict[str, Any]) -> bool:
        version = _safe_id(str(release["graph_version"]))
        graph = self._client.select_graph(self._graph_name(version))
        rows = graph.query("MATCH (m:GraphMetadata) RETURN m.release_hash LIMIT 1").result_set
        if not rows:
            return False
        if rows[0][0] != release["release_hash"]:
            raise ValueError("immutable FalkorDB graph version has different release hash")
        exported = self.export(version)
        keys = tuple(release) if release.get("schema_version") == "accepted_graph_release.v2" else (
            "graph_version", "release_hash", "parent_graph_version", "nodes", "edges", "rules"
        )
        for key in keys:
            if exported[key] != release[key]:
                raise ValueError(f"FalkorDB graph release readback mismatch: {key}")
        return True
    def export(self, graph_version: str) -> dict[str, Any]:
        version = _safe_id(graph_version)
        graph = self._client.select_graph(self._graph_name(version))
        metadata = graph.query(
            "MATCH (m:GraphMetadata) RETURN m.graph_version, m.release_hash, "
            "m.parent_graph_version, m.release_payload_json"
        ).result_set
        if not metadata:
            raise ValueError(f"FalkorDB graph version not found: {version}")
        nodes = graph.query("MATCH (n:KnowledgeNode) RETURN n.payload_json ORDER BY n.node_id").result_set
        edges = graph.query(
            "MATCH (:KnowledgeNode)-[e:KNOWLEDGE_RELATION]->(:KnowledgeNode) "
            "RETURN e.payload_json ORDER BY e.edge_id"
        ).result_set
        rules = graph.query("MATCH (r:KnowledgeRule) RETURN r.payload_json ORDER BY r.rule_id").result_set
        stored_release = (
            json.loads(metadata[0][3]) if metadata[0][3] is not None else {}
        )
        result = (
            stored_release
            if stored_release.get("schema_version") == "accepted_graph_release.v2"
            else {}
        )
        result.update({
            "graph_version": metadata[0][0],
            "release_hash": metadata[0][1],
            "parent_graph_version": metadata[0][2],
            "nodes": [json.loads(row[0]) for row in nodes],
            "edges": [json.loads(row[0]) for row in edges],
            "rules": [json.loads(row[0]) for row in rules],
        })
        return result

    def _graph_name(self, version: str) -> str:
        return f"{self._graph_prefix}__{version}"


class InMemoryGraphBackend:
    """Contract-faithful deterministic backend for unit tests and offline replay."""

    def __init__(self) -> None:
        self._releases: dict[str, dict[str, Any]] = {}

    def materialize(self, release: dict[str, Any]) -> None:
        version = str(release["graph_version"])
        existing = self._releases.get(version)
        if existing and existing["release_hash"] != release["release_hash"]:
            raise ValueError("immutable graph version already has different content")
        self._releases.setdefault(version, json.loads(_json(release)))

    def export(self, graph_version: str) -> dict[str, Any]:
        if graph_version not in self._releases:
            raise ValueError(f"graph version not found: {graph_version}")
        return json.loads(_json(self._releases[graph_version]))

    def verify_release(self, release: dict[str, Any]) -> bool:
        existing = self._releases.get(str(release["graph_version"]))
        if existing is None:
            return False
        if existing != json.loads(_json(release)):
            raise ValueError("immutable graph version has different content")
        return True


def _safe_id(value: str) -> str:
    if not ASCII_ID.fullmatch(value):
        raise ValueError(f"machine identifier must be ASCII-stable: {value!r}")
    return value


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
