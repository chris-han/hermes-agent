"""Build and validate proposal-only tender candidate graphs."""

from __future__ import annotations

from typing import Any

from contracts.knowledge_feature_pack import content_hash
from contracts.knowledge_graph_phase1 import (
    CandidateEdge,
    CandidateNode,
    CandidateRule,
    SensitiveTermCandidateGraph,
    SourceAnchor,
)
from semantier.knowledge_feature_packs.source_elements import anchor_span
from semantier.knowledge_feature_packs.semantica_adapter import CandidateExtractionAdapter


def build_candidate_graph(
    *,
    parsed_document: dict[str, Any],
    extraction_run_id: str,
    adapter: CandidateExtractionAdapter,
) -> SensitiveTermCandidateGraph:
    proposal_result = adapter.extract(
        elements=parsed_document["elements"],
        run_id=extraction_run_id,
    )
    output = proposal_result.structured_output
    # In-process Semantica emits a rich tender mapper projection for direct
    # inspection and a compact candidate projection for the v1 graph contract.
    # Select the latter only at this compatibility boundary so the native
    # artifact retains the complete mapper topology.
    if "semantica" in output:
        from semantier.knowledge_feature_packs.semantica_adapter import _candidate_payload

        output = {
            **output,
            **_candidate_payload(
                output["semantica"]["candidateResolvedEntities"],
                output["semantica"].get("relations", []),
            ),
        }
    elements_by_anchor = {
        element["anchor"]["anchor_id"]: element for element in parsed_document["elements"]
    }

    def anchors(values: list[dict[str, Any]]) -> tuple[SourceAnchor, ...]:
        resolved = []
        for value in values:
            anchor_id = str(value["element_anchor_id"])
            if anchor_id not in elements_by_anchor:
                raise ValueError(f"proposal references unknown source anchor: {anchor_id}")
            resolved.append(
                anchor_span(
                    elements_by_anchor[anchor_id],
                    int(value["start_offset"]),
                    int(value["end_offset"]),
                )
            )
        return tuple(resolved)

    proposal = proposal_result.proposal
    nodes = tuple(
        CandidateNode(
            node_id=str(item["node_id"]),
            node_type=item["node_type"],
            canonical_label=str(item["canonical_label"]),
            display_name_zh=item.get("display_name_zh"),
            normalized_value=str(item["normalized_value"]),
            rationale=str(item["rationale"]),
            source_anchors=anchors(item["source_spans"]),
            extraction_run_id=extraction_run_id,
            machine_proposal_id=proposal.proposal_id,
            machine_confidence=float(item["confidence"]),
        )
        for item in output.get("nodes", [])
    )
    edges = tuple(
        CandidateEdge(
            edge_id=str(item["edge_id"]),
            source_node_id=str(item["source_node_id"]),
            target_node_id=str(item["target_node_id"]),
            relation_type=item["relation_type"],
            rationale=str(item["rationale"]),
            source_anchors=anchors(item["source_spans"]),
            extraction_run_id=extraction_run_id,
            machine_proposal_id=proposal.proposal_id,
            machine_confidence=float(item["confidence"]),
        )
        for item in output.get("edges", [])
    )
    rules = tuple(
        CandidateRule(
            rule_id=str(item["rule_id"]),
            family_id=int(item["family_id"]),
            condition_node_ids=tuple(item.get("condition_node_ids", [])),
            exception_node_ids=tuple(item.get("exception_node_ids", [])),
            reason_code=str(item["reason_code"]),
            source_anchors=anchors(item["source_spans"]),
            extraction_run_id=extraction_run_id,
            machine_proposal_id=proposal.proposal_id,
        )
        for item in output.get("rules", [])
    )
    graph_basis = {
        "source_document_hash": parsed_document["document_hash"],
        "extraction_run_id": extraction_run_id,
        "proposal_id": proposal.proposal_id,
    }
    return SensitiveTermCandidateGraph(
        candidate_graph_id=f"scg_{content_hash(graph_basis)[7:31]}",
        source_document_hash=parsed_document["document_hash"],
        extraction_run_id=extraction_run_id,
        schema_version="sensitive_term_candidate_graph.v1",
        ontology_version="tender_risk_ontology.v1",
        machine_proposal=proposal,
        nodes=nodes,
        edges=edges,
        rules=rules,
    )


def conservative_identity_candidates(
    graph: SensitiveTermCandidateGraph,
) -> dict[str, list[dict[str, str]]]:
    exact_aliases: list[dict[str, str]] = []
    review_required: list[dict[str, str]] = []
    nodes = {node.node_id: node for node in graph.nodes}
    for edge in graph.edges:
        item = {
            "candidate_id": edge.edge_id,
            "edge_id": edge.edge_id,
            "source_node_id": edge.source_node_id,
            "target_node_id": edge.target_node_id,
        }
        if edge.relation_type.value == "alias_of" and (
            nodes[edge.source_node_id].normalized_value
            == nodes[edge.target_node_id].normalized_value
        ):
            exact_aliases.append(item)
        else:
            review_required.append(item)
    return {"exact_aliases": exact_aliases, "review_required": review_required}
