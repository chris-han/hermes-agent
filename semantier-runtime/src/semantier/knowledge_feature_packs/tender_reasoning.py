"""Bounded deterministic graph reasoning and companion-evidence assembly."""

from __future__ import annotations

from typing import Any

from contracts.knowledge_feature_pack import content_hash
from contracts.knowledge_graph_phase1 import SourceAnchor
from contracts.knowledge_reasoning_phase1 import (
    DeterministicGraphVerdict,
    DocumentCandidateAssertion,
    EvidenceObligation,
    GroundingState,
    KnowledgeScope,
    VerdictState,
    ViolationEvidenceAssembly,
)
from semantier.knowledge_feature_packs.source_elements import anchor_span


def retrieve_companion_evidence(
    *, scope: KnowledgeScope, obligation: EvidenceObligation
) -> tuple[SourceAnchor, ...]:
    matches: list[SourceAnchor] = []
    allowed = set(obligation.allowed_companion_roles)
    for document in scope.documents:
        if document.role not in allowed:
            continue
        for element in document.elements:
            text = str(element.get("text") or "")
            if any(term in text for term in obligation.query_terms):
                matches.append(anchor_span(element, 0, len(text)))
    return tuple(sorted(matches, key=lambda item: item.anchor_id))


def assemble_violation_evidence(
    *,
    assertion: DocumentCandidateAssertion,
    graph_release: dict[str, Any],
    supporting_context: tuple[SourceAnchor, ...] = (),
) -> ViolationEvidenceAssembly:
    if assertion.grounding_state == GroundingState.UNRESOLVED:
        node_ids: tuple[str, ...] = ()
        rules: list[dict[str, Any]] = []
    else:
        node_ids = (str(assertion.accepted_graph_node_id),)
        known_nodes = {str(node["node_id"]) for node in graph_release["nodes"]}
        if node_ids[0] not in known_nodes:
            raise ValueError("accepted graph node ID is absent from pinned graph version")
        rules = [
            rule for rule in graph_release["rules"]
            if node_ids[0] in set(rule.get("condition_node_ids") or ())
            or node_ids[0] in set(rule.get("target_node_ids") or ())
        ]
        if not rules:
            rules = [rule for rule in graph_release["rules"] if rule.get("family_id")]
    rule = rules[0] if rules else None
    condition_ids = tuple(str(item) for item in (rule or {}).get("condition_node_ids") or ())
    exception_ids = tuple(str(item) for item in (rule or {}).get("exception_node_ids") or ())
    edge_ids = tuple(
        str(edge["edge_id"])
        for edge in graph_release["edges"]
        if edge["source_node_id"] in node_ids or edge["target_node_id"] in node_ids
    )
    path = (*node_ids, *edge_ids, *((str(rule["rule_id"]),) if rule else ()))
    basis = {"assertion": assertion.assertion_id, "graph": graph_release["graph_version"], "path": path}
    return ViolationEvidenceAssembly(
        assembly_id=f"vea_{content_hash(basis)[7:31]}",
        assertion_id=assertion.assertion_id,
        primary_source_anchors=assertion.source_anchors,
        supporting_context_anchors=supporting_context,
        accepted_graph_node_ids=node_ids,
        accepted_graph_edge_ids=edge_ids,
        condition_node_ids=condition_ids,
        exception_node_ids=exception_ids,
        reasoning_path=path,
        triggered_rule_id=str(rule["rule_id"]) if rule else None,
        rule_inputs={"grounding_state": assertion.grounding_state.value},
        graph_version=str(graph_release["graph_version"]),
        graph_release_hash=str(graph_release["release_hash"]),
    )


def evaluate_assembly(
    *, assembly: ViolationEvidenceAssembly, graph_release: dict[str, Any], llm_available: bool = True
) -> DeterministicGraphVerdict:
    warnings = () if llm_available else ("llm_unavailable_deterministic_only",)
    rules = {str(rule["rule_id"]): rule for rule in graph_release["rules"]}
    nodes = {str(node["node_id"]) for node in graph_release["nodes"]}
    rule = rules.get(str(assembly.triggered_rule_id)) if assembly.triggered_rule_id else None
    if not assembly.accepted_graph_node_ids or rule is None:
        state, reason = VerdictState.UNRESOLVED, "unresolved_grounding_or_rule"
    elif not set(assembly.condition_node_ids).issubset(nodes):
        state, reason = VerdictState.UNRESOLVED, "missing_required_condition"
    elif not set(assembly.exception_node_ids).issubset(nodes):
        state, reason = VerdictState.UNRESOLVED, "missing_required_exception"
    elif rule.get("required_edge_id") not in {None, *assembly.accepted_graph_edge_ids}:
        state, reason = VerdictState.UNRESOLVED, "invalid_or_missing_graph_edge"
    elif rule.get("required_edge_direction"):
        required_edge = next(
            (edge for edge in graph_release["edges"] if edge["edge_id"] == rule.get("required_edge_id")),
            None,
        )
        expected = rule["required_edge_direction"]
        if required_edge is None or (
            required_edge["source_node_id"] != expected["source_node_id"]
            or required_edge["target_node_id"] != expected["target_node_id"]
        ):
            state, reason = VerdictState.UNRESOLVED, "invalid_or_missing_graph_edge"
        elif assembly.supporting_context_anchors and rule.get("context_negates_violation"):
            state, reason = VerdictState.CONFIRMED_NON_VIOLATION, str(rule["reason_code"])
        else:
            state, reason = VerdictState.CONFIRMED_VIOLATION, str(rule["reason_code"])
    elif assembly.supporting_context_anchors and rule.get("context_negates_violation"):
        state, reason = VerdictState.CONFIRMED_NON_VIOLATION, str(rule["reason_code"])
    else:
        state, reason = VerdictState.CONFIRMED_VIOLATION, str(rule["reason_code"])
    basis = {"assembly_hash": assembly.assembly_hash(), "state": state.value, "reason": reason}
    return DeterministicGraphVerdict(
        verdict_id=f"verdict_{content_hash(basis)[7:31]}",
        state=state,
        assertion_id=assembly.assertion_id,
        assembly_hash=assembly.assembly_hash(),
        source_anchor_refs=tuple(
            anchor.anchor_id for anchor in (*assembly.primary_source_anchors, *assembly.supporting_context_anchors)
        ),
        graph_path=assembly.reasoning_path,
        triggered_rule_id=assembly.triggered_rule_id,
        reason_code=reason,
        graph_version=assembly.graph_version,
        coverage_warnings=warnings,
    )


def compare_evaluation_runs(
    *, prior: list[DeterministicGraphVerdict], current: list[DeterministicGraphVerdict],
    graph_delta_id: str,
) -> dict[str, Any]:
    old = {item.assertion_id: item for item in prior}
    new = {item.assertion_id: item for item in current}
    shared = sorted(set(old) & set(new))
    changed = [key for key in shared if old[key].state != new[key].state or old[key].reason_code != new[key].reason_code]
    regressions = [
        key for key in changed
        if old[key].state != VerdictState.CONFIRMED_VIOLATION
        and new[key].state == VerdictState.CONFIRMED_VIOLATION
    ]
    return {
        "prior_graph_version": prior[0].graph_version if prior else None,
        "current_graph_version": current[0].graph_version if current else None,
        "causal_graph_delta_id": graph_delta_id,
        "case_count": len(set(old) | set(new)),
        "changed_assertion_ids": changed,
        "regression_assertion_ids": regressions,
        "comparison_hash": content_hash({
            "prior": [item.model_dump(mode="json") for item in prior],
            "current": [item.model_dump(mode="json") for item in current],
            "graph_delta_id": graph_delta_id,
        }),
    }
