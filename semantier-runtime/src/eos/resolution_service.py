"""Deterministic v5 semantic resolution service (R_v)."""

# Patent evidence C002: semantic conflict resolution using precedence graph and
# resolution rules to produce an execution decision (allow / escalate / reject /
# split projection) referenced in the admissibility-governance method claims.

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


TIER_PRIORITY = {
    "T0": 0,
    "T1": 1,
    "T2": 2,
    "T3": 3,
    "T4": 4,
    "T5": 5,
    "T6": 6,
}


def _build_tier_rank(precedence_graph: Dict[str, Any]) -> Dict[str, int]:
    """Build a deterministic tier ranking.

    Lower rank means higher authority. Graph edges follow the architecture's
    `higher -> lower` interpretation.
    """
    graph: Dict[str, set[str]] = {tier: set() for tier in TIER_PRIORITY}
    indegree: Dict[str, int] = {tier: 0 for tier in TIER_PRIORITY}

    for edge in precedence_graph.get("edges", []):
        higher = edge.get("from")
        lower = edge.get("to")
        if higher not in graph or lower not in graph or lower in graph[higher]:
            continue
        graph[higher].add(lower)
        indegree[lower] += 1

    queue = sorted((tier for tier, degree in indegree.items() if degree == 0), key=TIER_PRIORITY.get)
    ordered: List[str] = []

    while queue:
        tier = queue.pop(0)
        ordered.append(tier)
        for child in sorted(graph[tier], key=TIER_PRIORITY.get):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
                queue.sort(key=TIER_PRIORITY.get)

    if len(ordered) != len(TIER_PRIORITY):
        return dict(TIER_PRIORITY)

    return {tier: index for index, tier in enumerate(ordered)}


def _claim_rank(
    claim: Dict[str, Any],
    tier_rank: Dict[str, int],
    domain_priority: Dict[str, int],
) -> Tuple[int, int, str, str]:
    tier = claim.get("semantic_tier", "T6")
    domain = claim.get("authority_domain", "")
    key = claim.get("key", "")
    claim_id = claim.get("id", "")
    return (
        tier_rank.get(tier, 99),
        domain_priority.get(domain, 99),
        key,
        claim_id,
    )


def _extract_domain_priority(resolution_rules: Dict[str, Any]) -> Dict[str, int]:
    ordered_domains = resolution_rules.get("domain_priority", [])
    if not isinstance(ordered_domains, list):
        return {}
    return {domain: index for index, domain in enumerate(ordered_domains)}


def _applicable_constraints(
    constraint_results: Iterable[Dict[str, Any]],
    scopes: set[str],
) -> List[Dict[str, Any]]:
    return [
        constraint
        for constraint in constraint_results
        if constraint.get("scope") in scopes and constraint.get("decision") in {"REJECT", "ESCALATE"}
    ]


def _projection_bucket(claim: Dict[str, Any]) -> str:
    return (
        claim.get("projection_scope")
        or claim.get("projection_bundle")
        or claim.get("authority_domain")
        or ""
    )


def resolve(
    claims: List[Dict[str, Any]],
    constraint_results: List[Dict[str, Any]],
    precedence_graph: Dict[str, Any],
    resolution_rules: Dict[str, Any],
    scope: str,
    pins: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve semantic conflicts deterministically.

    The implementation is intentionally deterministic: no I/O, no randomness,
    no wall-clock branching.
    """
    tier_rank = _build_tier_rank(precedence_graph)
    domain_priority = _extract_domain_priority(resolution_rules)
    sorted_claims = sorted(claims, key=lambda claim: _claim_rank(claim, tier_rank, domain_priority))

    grouped_claims: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for claim in sorted_claims:
        grouped_claims[claim.get("key") or claim.get("id") or ""].append(claim)

    winning_claims = [grouped[0] for _, grouped in sorted(grouped_claims.items(), key=lambda item: item[0])]
    winning_ids = {claim.get("id") for claim in winning_claims}
    suppressed_claims = [claim for claim in sorted_claims if claim.get("id") not in winning_ids]

    execution_constraints = _applicable_constraints(
        constraint_results,
        {"execution", "governance", "evidence"},
    )
    reject_constraints = [c for c in execution_constraints if c.get("decision") == "REJECT"]
    escalate_constraints = [c for c in execution_constraints if c.get("decision") == "ESCALATE"]

    if reject_constraints:
        execution_decision = "REJECT"
    elif escalate_constraints:
        execution_decision = "ESCALATE"
    else:
        execution_decision = "ALLOW"

    projection_decision = "NONE"
    if scope == "projection":
        projection_scopes = {_projection_bucket(claim) for claim in winning_claims}
        projection_decision = "SPLIT_PROJECTION" if len(projection_scopes) > 1 else "SINGLE_PROJECTION"

    final_decision = execution_decision
    if scope == "projection" and projection_decision == "SPLIT_PROJECTION" and execution_decision == "ALLOW":
        final_decision = "SPLIT_PROJECTION"

    return {
        "final_decision": final_decision,
        "execution_decision": execution_decision,
        "projection_decision": projection_decision,
        "winning_claims": winning_claims,
        "suppressed_claims": suppressed_claims,
        "escalation_reasons": [c.get("reason", "unspecified") for c in escalate_constraints],
        "resolution_trace": {
            "scope": scope,
            "ordered_claim_ids": [c.get("id") for c in sorted_claims],
            "winning_claim_ids": [c.get("id") for c in winning_claims],
            "suppressed_claim_ids": [c.get("id") for c in suppressed_claims],
            "reject_constraints": [c.get("constraint_id") for c in reject_constraints],
            "escalate_constraints": [c.get("constraint_id") for c in escalate_constraints],
            "precedence_graph_version": precedence_graph.get("version"),
            "resolution_rules_version": resolution_rules.get("version"),
            "tier_rank": tier_rank,
            "projection_scopes": [_projection_bucket(claim) for claim in winning_claims],
        },
        "pins": pins,
    }
