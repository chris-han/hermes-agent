from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Literal

from contracts.governed_graph import (
    GraphEdgeV1,
    GraphIndexStatusV1,
    GraphNodeV1,
    GraphRedactionManifestV1,
    GraphSceneV1,
    GraphVisualizationModelV1,
    GraphWarningV1,
)

DisclosureDecision = Literal["allow", "minimize", "omit"]

MAX_NODES = 400
MAX_EDGES = 1200
MAX_SCENES = 32


class GraphProjectionError(ValueError):
    """Fail-closed graph projection error."""


def _require(value: str | None, code: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise GraphProjectionError(code)
    return normalized


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _semantic_node_payload(node: GraphNodeV1) -> dict[str, object]:
    payload = asdict(node)
    payload.pop("position", None)
    return payload


def _minimize_node(node: GraphNodeV1) -> GraphNodeV1:
    return replace(
        node,
        label=f"{node.node_kind}:{node.node_id}",
        display_summary="Content minimized by disclosure policy.",
        source_anchor_refs=(),
        capabilities=("view",),
        detail_ref=None,
        metadata={},
    )


def build_authorized_graph_projection(
    *,
    organization_id: str,
    workspace_id: str | None,
    source_ref: str,
    source_hash: str,
    classification_run_ref: str,
    ontology_version: str,
    sensitivity_ontology_version: str,
    resolver_snapshot_hash: str,
    redaction_policy_version: str,
    nodes: list[GraphNodeV1],
    edges: list[GraphEdgeV1],
    scenes: list[GraphSceneV1],
    disclosure_by_node_id: dict[str, DisclosureDecision],
    omission_reason_by_node_id: dict[str, str] | None = None,
    normalized_artifact_ref: str | None = None,
    normalized_artifact_hash: str | None = None,
    graph_snapshot_ref: str | None = None,
    graph_snapshot_hash: str | None = None,
    assertion_resolver_version: str | None = None,
    as_of: str,
    authorization_context_ref: str,
    freshness: GraphIndexStatusV1,
    warnings: list[GraphWarningV1] | None = None,
    projection_id: str | None = None,
) -> GraphVisualizationModelV1:
    """Build a deterministic browser-safe graph from explicit disclosure decisions.

    This function does not query authority stores. Callers must resolve entitlement
    before invoking it. Missing disclosure decisions fail closed as omission.
    """

    organization_id = _require(organization_id, "GRAPH_ORGANIZATION_REQUIRED")
    source_ref = _require(source_ref, "GRAPH_SOURCE_REF_REQUIRED")
    source_hash = _require(source_hash, "GRAPH_SOURCE_HASH_REQUIRED")
    classification_run_ref = _require(
        classification_run_ref, "GRAPH_CLASSIFICATION_RUN_REQUIRED"
    )
    ontology_version = _require(ontology_version, "GRAPH_ONTOLOGY_VERSION_REQUIRED")
    sensitivity_ontology_version = _require(
        sensitivity_ontology_version,
        "GRAPH_SENSITIVITY_ONTOLOGY_VERSION_REQUIRED",
    )
    resolver_snapshot_hash = _require(
        resolver_snapshot_hash, "GRAPH_RESOLVER_SNAPSHOT_REQUIRED"
    )
    redaction_policy_version = _require(
        redaction_policy_version, "GRAPH_REDACTION_POLICY_REQUIRED"
    )
    as_of = _require(as_of, "GRAPH_AS_OF_REQUIRED")
    authorization_context_ref = _require(
        authorization_context_ref, "GRAPH_AUTHORIZATION_CONTEXT_REQUIRED"
    )

    if len(nodes) > MAX_NODES:
        raise GraphProjectionError("GRAPH_NODE_LIMIT_EXCEEDED")
    if len(edges) > MAX_EDGES:
        raise GraphProjectionError("GRAPH_EDGE_LIMIT_EXCEEDED")
    if len(scenes) > MAX_SCENES:
        raise GraphProjectionError("GRAPH_SCENE_LIMIT_EXCEEDED")

    omission_reason_by_node_id = omission_reason_by_node_id or {}
    visible_nodes: list[GraphNodeV1] = []
    omitted_node_ids: set[str] = set()
    minimized_node_count = 0
    reason_codes: set[str] = set()

    for node in nodes:
        decision = disclosure_by_node_id.get(node.node_id)
        if decision is None:
            omitted_node_ids.add(node.node_id)
            reason_codes.add("GRAPH_DISCLOSURE_DECISION_MISSING_OR_DENIED")
            continue
        if decision == "omit":
            omitted_node_ids.add(node.node_id)
            reason_codes.add(
                omission_reason_by_node_id.get(
                    node.node_id, "GRAPH_DISCLOSURE_DECISION_MISSING_OR_DENIED"
                )
            )
            continue
        if decision == "minimize":
            visible_nodes.append(_minimize_node(node))
            minimized_node_count += 1
            continue
        if decision != "allow":
            raise GraphProjectionError("GRAPH_DISCLOSURE_DECISION_INVALID")
        visible_nodes.append(node)

    visible_node_ids = {node.node_id for node in visible_nodes}
    visible_edges = [
        edge
        for edge in edges
        if edge.source_node_id in visible_node_ids
        and edge.target_node_id in visible_node_ids
    ]
    visible_edge_ids = {edge.edge_id for edge in visible_edges}
    omitted_edge_count = len(edges) - len(visible_edges)

    visible_scenes = tuple(
        replace(
            scene,
            node_ids=tuple(
                node_id for node_id in scene.node_ids if node_id in visible_node_ids
            ),
            edge_ids=tuple(
                edge_id for edge_id in scene.edge_ids if edge_id in visible_edge_ids
            ),
        )
        for scene in scenes
    )

    manifest = GraphRedactionManifestV1(
        policy_version=redaction_policy_version,
        omitted_node_count=len(omitted_node_ids),
        omitted_edge_count=omitted_edge_count,
        minimized_node_count=minimized_node_count,
        reason_codes=tuple(sorted(reason_codes)),
    )

    semantic_payload = {
        "schema_version": "governed_graph_contract.v1",
        "organization_id": organization_id,
        "workspace_id": workspace_id,
        "source_ref": source_ref,
        "source_hash": source_hash,
        "normalized_artifact_ref": normalized_artifact_ref,
        "normalized_artifact_hash": normalized_artifact_hash,
        "graph_snapshot_ref": graph_snapshot_ref,
        "graph_snapshot_hash": graph_snapshot_hash,
        "assertion_resolver_version": assertion_resolver_version,
        "classification_run_ref": classification_run_ref,
        "ontology_version": ontology_version,
        "sensitivity_ontology_version": sensitivity_ontology_version,
        "resolver_snapshot_hash": resolver_snapshot_hash,
        "redaction_policy_version": redaction_policy_version,
        "nodes": [_semantic_node_payload(node) for node in visible_nodes],
        "edges": [asdict(edge) for edge in visible_edges],
        "redaction_manifest": asdict(manifest),
    }
    projection_hash = _canonical_hash(semantic_payload)

    presentation_payload = {
        "schema_version": "governed_graph_contract.v1.presentation",
        "projection_hash": projection_hash,
        "node_positions": {
            node.node_id: asdict(node.position) if node.position else None
            for node in visible_nodes
        },
        "scenes": [asdict(scene) for scene in visible_scenes],
    }
    presentation_hash = _canonical_hash(presentation_payload)

    return GraphVisualizationModelV1(
        projection_id=projection_id or f"kgv-{projection_hash[:16]}",
        organization_id=organization_id,
        workspace_id=workspace_id,
        source_ref=source_ref,
        source_hash=source_hash,
        normalized_artifact_ref=normalized_artifact_ref,
        normalized_artifact_hash=normalized_artifact_hash,
        graph_snapshot_ref=graph_snapshot_ref,
        graph_snapshot_hash=graph_snapshot_hash,
        assertion_resolver_version=assertion_resolver_version,
        classification_run_ref=classification_run_ref,
        ontology_version=ontology_version,
        sensitivity_ontology_version=sensitivity_ontology_version,
        resolver_snapshot_hash=resolver_snapshot_hash,
        redaction_policy_version=redaction_policy_version,
        nodes=tuple(visible_nodes),
        edges=tuple(visible_edges),
        scenes=visible_scenes,
        redaction_manifest=manifest,
        projection_hash=projection_hash,
        presentation_hash=presentation_hash,
        as_of=as_of,
        authorization_context_ref=authorization_context_ref,
        freshness=freshness,
        warnings=tuple(warnings or ()),
    )
