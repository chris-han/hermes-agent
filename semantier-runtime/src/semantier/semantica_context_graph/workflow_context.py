"""Server-side reconstruction and validation of the untrusted workbench hint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .artifact_repository import ArtifactNotFound, ContextGraphArtifactRepository


class WorkflowContextRejected(ValueError):
    """Raised when a browser pointer does not belong to the pinned artifact."""


@dataclass(frozen=True)
class CanonicalMvlWorkflowContext:
    workflow_id: str
    phase_plan_id: str
    stage: str
    stage_state: str
    active_profile_id: str
    graph_ref: str
    graph_version: str
    graph_hash: str
    authority_state: str
    run_mode: str
    learning_decision: str | None
    graph_construction_certification: str = "not_certified"
    reasoning_certification: str = "not_certified"
    governance_status: str = "not_admitted"


def browser_source_anchor(anchor: Mapping[str, Any]) -> dict[str, str | None]:
    """Project native SourceAnchor fields into the stable browser contract."""
    source_ref = str(anchor.get("sourceRef") or anchor.get("source_ref") or anchor.get("anchor_id") or "")
    source_hash = str(anchor.get("sourceHash") or anchor.get("source_hash") or anchor.get("document_hash") or "")
    locator = str(anchor.get("locator") or "")
    if not locator:
        part = str(anchor.get("document_part") or "document")
        if part == "paragraph":
            part = f"paragraph:{anchor.get('paragraph_index')}"
        elif part == "table_cell":
            part = f"table:{anchor.get('table_index')}:row:{anchor.get('row_index')}:cell:{anchor.get('cell_index')}"
        locator = f"{part}:span:{anchor.get('start_offset')}-{anchor.get('end_offset')}"
    return {"sourceRef": source_ref, "sourceHash": source_hash, "locator": locator, "quote": anchor.get("exact_text")}


def resolve_untrusted_workbench_hint(
    repository: ContextGraphArtifactRepository,
    *,
    tenant_id: str,
    domain: str,
    artifact_id: str,
    graph_version: str,
    graph_hash: str,
    browser_hint: Mapping[str, Any] | None = None,
) -> tuple[CanonicalMvlWorkflowContext, dict[str, list[str]]]:
    """Resolve canonical workflow state and validate only graph pointers.

    Workflow stage/decision/certification/governance fields from the browser are
    intentionally ignored.  In this slice the evidence-derived state is
    conservative until a real manifest/evaluation record is present.
    """
    try:
        metadata = repository.get(artifact_id)
    except ArtifactNotFound as exc:
        raise WorkflowContextRejected("unknown graph artifact") from exc
    if (
        metadata.organization_id != tenant_id
        or metadata.domain_id != domain
        or metadata.graph_version != graph_version
        or metadata.content_hash != graph_hash
    ):
        raise WorkflowContextRejected("graph pin does not match authenticated workspace")

    graph = repository.load_graph(artifact_id)
    node_ids = {str(node_id) for node_id in graph.nodes}
    edge_ids = {
        str(getattr(edge, "edge_id", edge.get("id", "")) if isinstance(edge, Mapping) else getattr(edge, "edge_id", ""))
        for edge in graph.edges
    }
    hint = browser_hint or {}
    selected_nodes = [str(value) for value in hint.get("selectedNodeIds", [])]
    selected_edges = [str(value) for value in hint.get("selectedEdgeIds", [])]
    if any(value not in node_ids for value in selected_nodes):
        raise WorkflowContextRejected("selected node is not in the pinned graph")
    if any(value not in edge_ids for value in selected_edges):
        raise WorkflowContextRejected("selected edge is not in the pinned graph")
    allowed_anchor_keys = {
        (
            browser_source_anchor(anchor)["sourceRef"] or "",
            browser_source_anchor(anchor)["sourceHash"] or "",
            browser_source_anchor(anchor)["locator"] or "",
        )
        for node in graph.nodes.values()
        for anchor in (getattr(node, "properties", {}) or {}).get("source_anchors", [])
        if isinstance(anchor, Mapping)
    }
    selected_anchors = hint.get("sourceAnchors", [])
    if not isinstance(selected_anchors, list):
        raise WorkflowContextRejected("source anchors must be a list")
    for anchor in selected_anchors:
        if not isinstance(anchor, Mapping):
            raise WorkflowContextRejected("source anchor must be an object")
        projected = browser_source_anchor(anchor)
        key = (
            projected["sourceRef"] or "",
            projected["sourceHash"] or "",
            projected["locator"] or "",
        )
        if key not in allowed_anchor_keys:
            raise WorkflowContextRejected("source anchor is not in the pinned graph")

    return (
        CanonicalMvlWorkflowContext(
            workflow_id="phase1-001",
            phase_plan_id="phase1-001",
            stage="BUILD",
            stage_state="server_evidence_required",
            active_profile_id="tender-runtime-expert",
            graph_ref=metadata.artifact_id,
            graph_version=metadata.graph_version,
            graph_hash=metadata.content_hash,
            authority_state=metadata.authority_state,
            run_mode=(
                metadata.run_mode
                if metadata.run_mode in {"evaluation_baseline", "authoritative"}
                else "evaluation_baseline"
            ),
            learning_decision=None,
        ),
        {
            "selected_node_ids": selected_nodes,
            "selected_edge_ids": selected_edges,
            "source_anchors": [browser_source_anchor(anchor) for anchor in selected_anchors],
        },
    )
