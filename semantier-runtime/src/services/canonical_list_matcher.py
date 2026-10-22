"""Generic canonical-list runtime matcher.

The matcher consumes profile configuration plus a selected Semantica graph
artifact. It deliberately contains no Tender-specific imports or labels.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from services.semantica_mvl_control_adapter import SemanticaMvlControlAdapter
from services.canonical_skill_registry import CANONICAL_JOURNEY_REF
from storage.canonical_list_profile_store import CanonicalListProfileStore


MATCH_REQUEST_SCHEMA_VERSION = "canonical_list_match_request.v1"
MATCH_SCHEMA_VERSION = "canonical_list_match.v1"
_TEXT_RELATIONS = {"synonym", "paraphrase", "implication", "narrower_than", "broader_than"}


class CanonicalListMatcher:
    """Profile-driven matcher over selected canonical-list graph state."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
        domain_id: str = "canonical-list",
        graph_factory: Any | None = None,
    ) -> None:
        self.control = SemanticaMvlControlAdapter(
            workspace_root,
            organization_id=organization_id,
            workspace_id=workspace_id,
            domain_id=domain_id,
        )
        self.profile_store = CanonicalListProfileStore(workspace_root, workspace_id=workspace_id)
        self.graph_factory = graph_factory

    def match(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        if request.get("schema_version") != MATCH_REQUEST_SCHEMA_VERSION:
            raise ValueError("canonical-list match request schema_version is required")
        profile_ref = request.get("profile_ref")
        profile_version = request.get("profile_version")
        graph_ref = request.get("selected_graph_version_ref")
        if not profile_ref or not profile_version or not graph_ref:
            raise ValueError("profile_ref, profile_version, and selected_graph_version_ref are required")
        input_payload = request.get("input") or {}
        text = input_payload.get("text")
        document_ref = input_payload.get("document_ref")
        if not text and not document_ref:
            raise ValueError("document_ref or text is required")
        if not text:
            return []

        profile = self.profile_store.get_profile_version(
            profile_ref=profile_ref,
            version=profile_version,
        )
        resolved = self.control.resolve_graph_ref(graph_ref)
        graph = self.control.repository.load_graph(
            resolved.graph_metadata.artifact_id,
            graph_factory=self.graph_factory,
        )
        nodes, edges = _graph_payload(graph)
        scenario = ((input_payload.get("source_context") or {}).get("scenario"))
        matches = []
        for concept_id, concept in nodes.items():
            if _node_role(concept) != "CanonicalConcept":
                continue
            relation, observed, score = _match_concept(text, concept, nodes, edges)
            if relation is None or observed is None:
                continue
            matches.append(
                {
                    "schema_version": MATCH_SCHEMA_VERSION,
                    "match_id": f"clm_{uuid.uuid4().hex}",
                    "observed_expression": observed,
                    "semantic_relation": relation,
                    "canonical_concept_ref": concept_id,
                    "canonical_concept_label": _node_label(concept),
                    "classifications": _classifications(concept_id, nodes, edges),
                    "applicability": _applicability(concept_id, scenario, profile.runtime_policy, nodes, edges),
                    "graph_path": _graph_path(concept_id, nodes, edges),
                    "output_suggestions": _output_suggestions(concept_id, nodes, edges),
                    "reference_examples_used": _reference_examples_used(text, concept_id, nodes, edges),
                    "discovery_decision_refs": list(concept.get("properties", {}).get("discovery_decision_refs") or []),
                    "reference_evidence": _canonical_source_ref(concept.get("properties", {}).get("reference_evidence")),
                    "observed_evidence": {
                        **_canonical_source_ref(input_payload.get("observed_evidence")),
                        "source_ref": document_ref,
                        "exact_text": observed,
                    },
                    "profile_ref": profile.profile_ref,
                    "profile_version": profile.version,
                    "graph_version_ref": resolved.selection.graph_version_ref,
                    "ontology_version_ref": request.get("selected_ontology_version_ref"),
                    "skill_ref": request.get("skill_ref") or CANONICAL_JOURNEY_REF,
                    "semantica_commit": resolved.graph_metadata.semantica_commit,
                    "scores": {
                        "similarity_score": score,
                        "evidence_score": 1.0,
                        "confidence": score,
                    },
                }
            )
        return matches


def _graph_payload(graph: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    raw_nodes = getattr(graph, "nodes", {})
    raw_edges = getattr(graph, "edges", [])
    if isinstance(raw_nodes, dict):
        nodes = {
            str(node_id): _normalize_node(str(node_id), node)
            for node_id, node in raw_nodes.items()
        }
    else:
        nodes = {
            str(node.get("id")): _normalize_node(str(node.get("id")), node)
            for node in raw_nodes
        }
    if isinstance(raw_edges, dict):
        edges = [_normalize_edge(str(edge_id), edge) for edge_id, edge in raw_edges.items()]
    else:
        edges = [_normalize_edge(str(edge.get("id", f"edge-{index}")), edge) for index, edge in enumerate(raw_edges)]
    return nodes, edges


def _normalize_node(node_id: str, node: Any) -> dict[str, Any]:
    if isinstance(node, dict):
        payload = dict(node)
    else:
        payload = {
            "id": node_id,
            "type": getattr(node, "type", None),
            "label": getattr(node, "label", None) or getattr(node, "content", None),
            "properties": getattr(node, "properties", {}),
        }
    payload.setdefault("id", node_id)
    payload.setdefault("properties", {})
    return payload


def _normalize_edge(edge_id: str, edge: Any) -> dict[str, Any]:
    payload = dict(edge) if isinstance(edge, dict) else {
        "id": edge_id,
        "source": getattr(edge, "source", None),
        "target": getattr(edge, "target", None),
        "type": getattr(edge, "type", None),
        "properties": getattr(edge, "properties", {}),
    }
    payload.setdefault("id", edge_id)
    payload.setdefault("properties", {})
    return payload


def _node_role(node: dict[str, Any]) -> str:
    return str(node.get("role") or node.get("type") or node.get("properties", {}).get("role") or "")


def _node_label(node: dict[str, Any]) -> str:
    return str(node.get("label") or node.get("content") or node.get("properties", {}).get("label") or node["id"])


def _match_concept(
    text: str,
    concept: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[str | None, str | None, float | None]:
    label = _node_label(concept)
    for candidate in [label, *(concept.get("properties", {}).get("aliases") or [])]:
        if not candidate:
            continue
        relation = "exact" if candidate == label else "synonym"
        observed = _contains(text, str(candidate))
        if observed:
            return relation, observed, 1.0 if relation == "exact" else 0.92
        normalized = _contains(_normalize_text(text), _normalize_text(str(candidate)))
        if normalized:
            return "normalized_exact", str(candidate), 0.96
    for edge in _outgoing(concept["id"], edges):
        relation = str(edge.get("type") or edge.get("relation") or "")
        if relation not in _TEXT_RELATIONS:
            continue
        expression = edge.get("properties", {}).get("expression") or _node_label(nodes.get(str(edge.get("target")), {}))
        observed = _contains(text, str(expression))
        if observed:
            return relation, observed, 0.86
    return None, None, None


def _contains(text: str, candidate: str) -> str | None:
    if not candidate:
        return None
    index = text.find(candidate)
    return text[index : index + len(candidate)] if index >= 0 else None


def _normalize_text(value: str) -> str:
    return re.sub(r"\W+", "", value, flags=re.UNICODE).casefold()


def _outgoing(source: str, edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [edge for edge in edges if str(edge.get("source")) == source]


def _incoming(target: str, edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [edge for edge in edges if str(edge.get("target")) == target]


def _classifications(
    concept_id: str,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edge in _outgoing(concept_id, edges):
        if str(edge.get("type") or edge.get("relation")) not in {"classifiedAs", "classified_as"}:
            continue
        classification_ref = str(edge.get("target"))
        rows.append(_classification_row(classification_ref, nodes, {"kind": "none", "ref": None, "path_refs": []}))
        rows.extend(_classification_ancestors(classification_ref, nodes, edges, [classification_ref]))
    return rows


def _classification_ancestors(
    classification_ref: str,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    path: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edge in _outgoing(classification_ref, edges):
        relation = str(edge.get("type") or edge.get("relation"))
        if relation not in {"skos:broader", "broader", "parent"}:
            continue
        parent_ref = str(edge.get("target"))
        if parent_ref in path:
            continue
        rows.append(
            _classification_row(
                parent_ref,
                nodes,
                {"kind": "hierarchy_node", "ref": classification_ref, "path_refs": [*path, parent_ref]},
            )
        )
        rows.extend(_classification_ancestors(parent_ref, nodes, edges, [*path, parent_ref]))
    return rows


def _classification_row(ref: str, nodes: dict[str, dict[str, Any]], inherited_from: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification_ref": ref,
        "classification_label": _node_label(nodes[ref]),
        "inherited_from": inherited_from,
    }


def _applicability(
    concept_id: str,
    scenario: str | None,
    runtime_policy: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    selector_refs = [
        str(edge.get("target"))
        for edge in _outgoing(concept_id, edges)
        if str(edge.get("type") or edge.get("relation")) in {"constrainedBy", "appliesInScope", "runtime_selector"}
    ]
    if not selector_refs:
        return {"decision": "applicable", "selector_refs": [], "input_scenario": scenario, "reason": "no selector constraint"}
    if not scenario:
        return {"decision": "unknown", "selector_refs": selector_refs, "input_scenario": scenario, "reason": runtime_policy["unknown_selector_behavior"]}
    for selector_ref in selector_refs:
        selector = nodes[selector_ref]
        properties = selector.get("properties", {})
        if properties.get("selector_semantic") == "universal":
            return {"decision": "applicable", "selector_refs": selector_refs, "input_scenario": scenario, "reason": "universal selector"}
        if properties.get("semantics") != "confirmed":
            return {"decision": "unknown", "selector_refs": selector_refs, "input_scenario": scenario, "reason": "selector semantics unconfirmed"}
        values = [str(properties.get("selector_value") or _node_label(selector)), *(properties.get("aliases") or [])]
        if _normalize_text(scenario) in {_normalize_text(value) for value in values}:
            return {"decision": "applicable", "selector_refs": selector_refs, "input_scenario": scenario, "reason": "scenario matches selector"}
    return {"decision": "not_applicable", "selector_refs": selector_refs, "input_scenario": scenario, "reason": "scenario does not match selector"}


def _output_suggestions(concept_id: str, nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for edge in _outgoing(concept_id, edges):
        if str(edge.get("type") or edge.get("relation")) not in {"suggests", "output_suggestion"}:
            continue
        ref = str(edge.get("target"))
        rows.append({"suggestion_ref": ref, "text": _node_label(nodes[ref])})
    return rows


def _reference_examples_used(
    text: str,
    concept_id: str,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, str]]:
    rows = []
    for edge in _outgoing(concept_id, edges):
        if str(edge.get("type") or edge.get("relation")) not in {"illustratedBy", "reference_example"}:
            continue
        ref = str(edge.get("target"))
        example = nodes[ref]
        if _contains(text, _node_label(example)):
            rows.append({"reference_example_ref": ref, "use": str(edge.get("properties", {}).get("use") or "similarity_seed")})
    return rows


def _graph_path(concept_id: str, nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [{"node_or_edge_ref": concept_id, "relation": "match_root", "hop": 0, "reason": _node_role(nodes[concept_id])}]
    for hop, edge in enumerate(_outgoing(concept_id, edges), start=1):
        rows.append({
            "node_or_edge_ref": str(edge.get("id")),
            "relation": str(edge.get("type") or edge.get("relation")),
            "hop": hop,
            "reason": str(edge.get("target")),
        })
    return rows


def _canonical_source_ref(value: Any) -> dict[str, Any]:
    payload = dict(value or {})
    return {
        "source_ref": payload.get("source_ref"),
        "source_hash": payload.get("source_hash"),
        "canonical_source_ir_ref": payload.get("canonical_source_ir_ref"),
        "source_block_ref": payload.get("source_block_ref"),
        "source_row_ref": payload.get("source_row_ref"),
        "source_cell_refs": list(payload.get("source_cell_refs") or []),
        "evidence_ref": payload.get("evidence_ref"),
    }
