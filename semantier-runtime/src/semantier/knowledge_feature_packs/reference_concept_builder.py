"""Build reference concept metadata from proposal-only candidate graphs."""

from __future__ import annotations

import json
from pathlib import Path

from contracts.contextgraph_reference import CanonicalExpression, ReferenceConcept
from knowledge_extraction.quality_gates_w4 import RoleClassifier
from contracts.knowledge_feature_pack import content_hash
from contracts.knowledge_graph_phase1 import CandidateNodeType, SensitiveTermCandidateGraph


REFERENCE_NODE_TYPES = {
    CandidateNodeType.TERM,
    CandidateNodeType.CONCEPT,
    CandidateNodeType.CATEGORY,
    CandidateNodeType.ALIAS,
}
INTERNAL_REFERENCE_NODE_PREFIXES = ("reference_category_",)


def build_reference_concepts(
    graph: SensitiveTermCandidateGraph,
) -> tuple[tuple[ReferenceConcept, ...], tuple[CanonicalExpression, ...]]:
    profile_path = (
        Path(__file__).resolve().parents[2]
        / "knowledge_extraction"
        / "profiles"
        / "tender_sensitive_v1"
        / "profile.json"
    )
    classifier = RoleClassifier(
        json.loads(profile_path.read_text(encoding="utf-8"))
    )
    concepts: list[ReferenceConcept] = []
    expressions: list[CanonicalExpression] = []
    for node in graph.nodes:
        if node.node_type not in REFERENCE_NODE_TYPES:
            continue
        if node.node_id.startswith(INTERNAL_REFERENCE_NODE_PREFIXES):
            continue
        anchor_refs = tuple(anchor.anchor_id for anchor in node.source_anchors)
        concept_basis = {
            "candidate_graph_id": graph.candidate_graph_id,
            "node_id": node.node_id,
            "label": node.canonical_label,
        }
        concept_id = f"ref_concept_{content_hash(concept_basis)[7:31]}"
        concept = ReferenceConcept(
            schema_version="contextgraph_reference_concept.v1",
            concept_id=concept_id,
            canonical_label=node.canonical_label,
            category=node.node_type.value,
            semantic_role=classifier.classify(
                candidate_span_id=node.node_id,
                exact_text=node.canonical_label,
                source_role=node.node_type.value,
                native_entities=[],
            )["semantic_role"],
            prohibition_type=None,
            applies_to=("tender_qualification",),
            source_anchor_refs=anchor_refs,
            grounding_state="candidate",
        )
        expression_basis = {
            "concept_id": concept_id,
            "text": node.canonical_label,
            "source_anchor_refs": anchor_refs,
        }
        expression = CanonicalExpression(
            schema_version="contextgraph_canonical_expression.v1",
            expression_id=f"canon_expr_{content_hash(expression_basis)[7:31]}",
            concept_id=concept_id,
            text=node.canonical_label,
            language="zh-CN",
            expression_kind="source_defined",
            source_anchor_refs=anchor_refs,
        )
        concepts.append(concept)
        expressions.append(expression)
    return tuple(concepts), tuple(expressions)
