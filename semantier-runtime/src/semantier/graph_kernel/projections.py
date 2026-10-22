from __future__ import annotations

from pathlib import Path
from typing import Any

from contracts.normative_conflict import ConflictCoreV1, ConflictResolutionV1
from eos.artifact_hashing import canonical_json, content_hash

from .store import GraphKernelStore


ADAPTER_VERSION = "graph_kernel_projection_adapters.v1"


def project_route_inventory(
    *,
    db_path: str | Path | None,
    organization_id: str,
    routes: list[dict[str, Any]],
) -> dict[str, Any]:
    store = GraphKernelStore() if db_path is None else GraphKernelStore.for_offline_path(db_path)
    service = store.upsert_node(
        organization_id=organization_id,
        node_type="service",
        canonical_key="semantier-webapi",
        display_name="Semantier Web API",
        properties={"source": "runtime_inventory"},
    )
    assertions = []
    for route in routes:
        route_node = store.upsert_node(
            organization_id=organization_id,
            node_type="api_operation",
            canonical_key=f"{route['method']} {route['path']}",
            display_name=f"{route['method']} {route['path']}",
            properties={"authorization_class": route.get("authorization_class")},
        )
        assertions.append(
            store.propose_assertion(
                organization_id=organization_id,
                subject_id=service.node_id,
                predicate="EXPOSES",
                object_id=route_node.node_id,
                assertion_class="structural",
                semantic_tier=None,
                authority_role="structural",
                governance_state="active",
                valid_from=None,
                valid_to=None,
                source_ref="runtime_inventory",
                source_anchor=route_node.canonical_key,
                extraction_method="runtime_observation",
                extractor_version=ADAPTER_VERSION,
            )
        )
    return {
        "adapter_version": ADAPTER_VERSION,
        "node_count": len(routes) + 1,
        "assertion_count": len(assertions),
        "projection_hash": content_hash({"routes": routes, "assertions": [item.assertion_id for item in assertions]}),
    }


def project_semantic_relation_candidates(
    *,
    db_path: str | Path | None,
    organization_id: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    store = GraphKernelStore() if db_path is None else GraphKernelStore.for_offline_path(db_path)
    assertion_ids: list[str] = []
    for candidate in candidates:
        subject = store.upsert_node(
            organization_id=organization_id,
            node_type="semantic_candidate",
            canonical_key=str(candidate["from_node_ref"]),
            display_name=candidate.get("from_node_label") or str(candidate["from_node_ref"]),
            properties={"discovery_run_id": candidate.get("discovery_run_id")},
        )
        obj = store.upsert_node(
            organization_id=organization_id,
            node_type="semantic_candidate",
            canonical_key=str(candidate["to_node_ref"]),
            display_name=candidate.get("to_node_label") or str(candidate["to_node_ref"]),
            properties={"discovery_run_id": candidate.get("discovery_run_id")},
        )
        assertion = store.propose_assertion(
            organization_id=organization_id,
            subject_id=subject.node_id,
            predicate=_semantic_relation_predicate(str(candidate["relation_type"])),
            object_id=obj.node_id,
            assertion_class="semantic",
            semantic_tier=str(candidate.get("semantic_tier") or "T3"),
            authority_role=str(candidate.get("authority_role") or "interpretive"),
            governance_state=_semantic_candidate_state(str(candidate.get("curation_state") or "PROPOSED")),
            valid_from=candidate.get("valid_from"),
            valid_to=candidate.get("valid_to"),
            source_ref=str(candidate.get("source_relation_ref") or candidate.get("semantic_relation_candidate_id")),
            source_anchor=canonical_json(candidate.get("source_anchor_refs") or []),
            extraction_method=str(candidate.get("extraction_method") or "human"),
            extractor_version=ADAPTER_VERSION,
            confidence=candidate.get("confidence"),
        )
        assertion_ids.append(assertion.assertion_id)
    return {
        "adapter_version": ADAPTER_VERSION,
        "assertion_ids": assertion_ids,
        "projection_hash": content_hash({"candidates": candidates, "assertion_ids": assertion_ids}),
    }


def project_normative_conflict_core(
    *,
    db_path: str | Path | None,
    organization_id: str,
    core: ConflictCoreV1,
) -> dict[str, Any]:
    """Project one verified N-ary conflict core without inventing pairwise conflicts."""
    store = GraphKernelStore() if db_path is None else GraphKernelStore.for_offline_path(db_path)
    conflict_node = store.upsert_node(
        organization_id=organization_id,
        node_type="normative_conflict_core",
        canonical_key=core.conflict_core_ref,
        display_name=core.conflict_core_ref,
        properties={
            "conflict_core_hash": core.conflict_core_hash,
            "compatibility_result_ref": core.compatibility_result_ref,
            "normative_set_ref": core.normative_set_ref,
            "context_ref": core.context_ref,
            "minimality": core.minimality,
            "evidence_refs": list(core.evidence_refs),
        },
    )
    assertion_ids: list[str] = []
    rule_node_ids: list[str] = []
    for rule_ref in core.member_rule_refs:
        rule_node = store.upsert_node(
            organization_id=organization_id,
            node_type="normative_rule",
            canonical_key=rule_ref,
            display_name=rule_ref,
            properties={"rule_ref": rule_ref},
        )
        rule_node_ids.append(rule_node.node_id)
        assertion = store.propose_assertion(
            organization_id=organization_id,
            subject_id=conflict_node.node_id,
            predicate="MEMBER_OF_CONFLICT",
            object_id=rule_node.node_id,
            assertion_class="structural",
            semantic_tier=None,
            authority_role="structural",
            governance_state="active",
            valid_from=None,
            valid_to=None,
            source_ref=core.conflict_core_ref,
            source_anchor=rule_ref,
            extraction_method="parser",
            extractor_version=ADAPTER_VERSION,
            confidence=None,
        )
        assertion_ids.append(assertion.assertion_id)
    return {
        "adapter_version": ADAPTER_VERSION,
        "conflict_node_id": conflict_node.node_id,
        "conflict_core_hash": core.conflict_core_hash,
        "normative_set_ref": core.normative_set_ref,
        "context_ref": core.context_ref,
        "rule_node_ids": rule_node_ids,
        "membership_assertion_ids": assertion_ids,
        "projection_hash": content_hash({
            "conflict_core_ref": core.conflict_core_ref,
            "conflict_core_hash": core.conflict_core_hash,
            "conflict_node_id": conflict_node.node_id,
            "rule_node_ids": rule_node_ids,
            "membership_assertion_ids": assertion_ids,
        }),
    }


def project_normative_conflict_resolution(
    *,
    db_path: str | Path | None,
    organization_id: str,
    core: ConflictCoreV1,
    resolution: ConflictResolutionV1,
) -> dict[str, Any]:
    """Project a resolution as an append-only view over an existing conflict core."""
    if resolution.conflict_core_ref != core.conflict_core_ref:
        raise ValueError("conflict resolution/core mismatch")
    member_rules = set(core.member_rule_refs)
    if not set(resolution.controlling_rule_refs).issubset(member_rules):
        raise ValueError("conflict resolution controlling rule is not a core member")

    store = GraphKernelStore() if db_path is None else GraphKernelStore.for_offline_path(db_path)
    core_projection = project_normative_conflict_core(
        db_path=db_path,
        organization_id=organization_id,
        core=core,
    )
    resolution_node = store.upsert_node(
        organization_id=organization_id,
        node_type="normative_conflict_resolution",
        canonical_key=resolution.resolution_ref,
        display_name=resolution.resolution_ref,
        properties={
            "resolution_hash": resolution.resolution_hash,
            "conflict_core_ref": resolution.conflict_core_ref,
            "outcome": resolution.outcome,
            "resolution_evidence_refs": list(resolution.resolution_evidence_refs),
            "resolved_at": resolution.resolved_at.isoformat(),
        },
    )
    resolved_by = store.propose_assertion(
        organization_id=organization_id,
        subject_id=core_projection["conflict_node_id"],
        predicate="RESOLVED_BY",
        object_id=resolution_node.node_id,
        assertion_class="structural",
        semantic_tier=None,
        authority_role="structural",
        governance_state="active",
        valid_from=None,
        valid_to=None,
        source_ref=resolution.resolution_ref,
        source_anchor=core.conflict_core_ref,
        extraction_method="parser",
        extractor_version=ADAPTER_VERSION,
        confidence=None,
    )
    controlling_assertion_ids: list[str] = []
    for rule_ref in resolution.controlling_rule_refs:
        rule_node = store.upsert_node(
            organization_id=organization_id,
            node_type="normative_rule",
            canonical_key=rule_ref,
            display_name=rule_ref,
            properties={"rule_ref": rule_ref},
        )
        assertion = store.propose_assertion(
            organization_id=organization_id,
            subject_id=resolution_node.node_id,
            predicate="CONTROLLING_RULE",
            object_id=rule_node.node_id,
            assertion_class="structural",
            semantic_tier=None,
            authority_role="structural",
            governance_state="active",
            valid_from=None,
            valid_to=None,
            source_ref=resolution.resolution_ref,
            source_anchor=rule_ref,
            extraction_method="parser",
            extractor_version=ADAPTER_VERSION,
            confidence=None,
        )
        controlling_assertion_ids.append(assertion.assertion_id)

    return {
        "adapter_version": ADAPTER_VERSION,
        "conflict_node_id": core_projection["conflict_node_id"],
        "resolution_node_id": resolution_node.node_id,
        "resolution_hash": resolution.resolution_hash,
        "resolved_by_assertion_id": resolved_by.assertion_id,
        "controlling_rule_assertion_ids": controlling_assertion_ids,
        "projection_hash": content_hash({
            "conflict_core_ref": core.conflict_core_ref,
            "resolution_ref": resolution.resolution_ref,
            "resolution_hash": resolution.resolution_hash,
            "conflict_node_id": core_projection["conflict_node_id"],
            "resolution_node_id": resolution_node.node_id,
            "resolved_by_assertion_id": resolved_by.assertion_id,
            "controlling_rule_assertion_ids": controlling_assertion_ids,
        }),
    }


def _semantic_relation_predicate(relation_type: str) -> str:
    return {
        "not_same_as": "CONFLICTS_WITH",
        "conflicts_with": "CONFLICTS_WITH",
        "allowed_context_for": "APPLIES_TO",
        "prohibited_context_for": "CONSTRAINS",
        "exception_to": "SUPERSEDES",
        "projects_to": "PROJECTS_TO",
        "synonym_of": "INTERPRETS",
        "variant_of": "INTERPRETS",
    }.get(relation_type, "INTERPRETS")


def _semantic_candidate_state(curation_state: str) -> str:
    return {
        "CURATED": "candidate",
        "PROPOSED": "candidate",
        "REJECTED": "rejected",
        "APPROVED": "approved",
        "ACTIVATED": "active",
    }.get(curation_state.upper(), "candidate")
