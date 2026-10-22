"""Governed reconstruction of ontology candidate context for chat transport."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import ValidationError

from contracts.ontology_building import (
    ExploratoryGraphSnapshotV1,
    OntologyBuildManifestV1,
)
from storage.ontology_build_store import OntologyBuildStore
from contracts.knowledge_workbench_context import UserIntentV4


class OntologyCandidateContextRejected(ValueError):
    """Raised when an untrusted candidate hint does not match governed state."""


def _string_list(hint: Mapping[str, Any], key: str) -> list[str]:
    value = hint.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise OntologyCandidateContextRejected(f"{key} must be a string list")
    return value


def _bounded_active_function(hint: Mapping[str, Any]) -> dict[str, str]:
    value = hint.get("activeFunction")
    if not isinstance(value, Mapping):
        raise OntologyCandidateContextRejected("ontology activeFunction is required")
    surface = value.get("surface")
    function = value.get("function")
    tab = value.get("tab")
    if surface not in {"ontology-workbench", "ontology-workbench2"}:
        raise OntologyCandidateContextRejected("unsupported ontology workbench surface")
    if not isinstance(tab, str) or tab not in {"graph", "evidence", "lineage", "compare", "evaluate"}:
        raise OntologyCandidateContextRejected("invalid ontology activeFunction")
    allowed_functions = {f"{tab}_review"}
    if tab == "graph":
        allowed_functions.update(
            f"graph_{view}_review"
            for view in ("knowledge-graph", "ontology", "semantic-network", "temporal", "embedding", "analytics")
        )
    if not isinstance(function, str) or function not in allowed_functions:
        raise OntologyCandidateContextRejected("invalid ontology activeFunction")
    return {"surface": surface, "function": function, "tab": tab}


def resolve_ontology_candidate_context(
    store: OntologyBuildStore,
    *,
    organization_id: str,
    workspace_id: str,
    candidate_ref: str,
    candidate_hash: str,
    browser_hint: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve an exact manifest-pinned exploratory snapshot and bounded pointers."""

    try:
        snapshot = store.get(
            organization_id=organization_id,
            workspace_id=workspace_id,
            artifact_ref=candidate_ref,
            contract_type=ExploratoryGraphSnapshotV1,
        )
    except (PermissionError, ValueError) as exc:
        raise OntologyCandidateContextRejected("candidate is outside governed scope") from exc
    if snapshot is None or snapshot.snapshot_hash != candidate_hash:
        raise OntologyCandidateContextRejected("candidate snapshot pin is stale or invalid")

    try:
        manifest = store.get(
            organization_id=organization_id,
            workspace_id=workspace_id,
            artifact_ref=snapshot.build_ref,
            contract_type=OntologyBuildManifestV1,
        )
    except (PermissionError, ValueError) as exc:
        raise OntologyCandidateContextRejected("candidate build is outside governed scope") from exc
    pins = () if manifest is None else tuple(
        pin
        for pin in manifest.child_artifacts
        if pin.artifact_kind == "exploratory_graph_snapshot"
    )
    if len(pins) != 1 or (
        pins[0].artifact_ref != snapshot.snapshot_ref
        or pins[0].artifact_hash != snapshot.snapshot_hash
    ):
        raise OntologyCandidateContextRejected("candidate is not exactly manifest-pinned")

    selected_candidate = browser_hint.get("selectedCandidateId")
    if selected_candidate not in (None, candidate_ref):
        raise OntologyCandidateContextRejected("selected candidate does not match candidate pin")
    selected_nodes = _string_list(browser_hint, "selectedNodeIds")
    selected_edges = _string_list(browser_hint, "selectedEdgeIds")
    selected_rules = _string_list(browser_hint, "selectedRuleIds")
    selected_evidence = _string_list(browser_hint, "selectedEvidenceRefs")
    node_ids = {node.node_id for node in snapshot.nodes}
    rule_ids = {
        node.node_id for node in snapshot.nodes if node.node_kind == "rule"
    }
    edge_ids = {edge.edge_id for edge in snapshot.edges}
    evidence_refs = {
        evidence_ref
        for item in (*snapshot.nodes, *snapshot.edges)
        for evidence_ref in item.evidence_refs
    }
    if any(value not in node_ids for value in selected_nodes):
        raise OntologyCandidateContextRejected("selected node is not in candidate")
    if any(value not in rule_ids for value in selected_rules):
        raise OntologyCandidateContextRejected("selected rule is not a candidate rule")
    if any(value not in edge_ids for value in selected_edges):
        raise OntologyCandidateContextRejected("selected edge is not in candidate")
    if any(value not in evidence_refs for value in selected_evidence):
        raise OntologyCandidateContextRejected("selected evidence is not in candidate")
    requested_source = browser_hint.get("activeSourceIdentityRef")
    if requested_source not in (None, snapshot.source_identity_ref):
        raise OntologyCandidateContextRejected("source identity does not match candidate")
    if _string_list(browser_hint, "sourceAnchors"):
        raise OntologyCandidateContextRejected(
            "candidate snapshot does not govern source anchors"
        )

    raw_intent = browser_hint.get("userIntent")
    user_intent = None
    if raw_intent is not None:
        try:
            parsed_intent = UserIntentV4.model_validate(raw_intent)
        except ValidationError as exc:
            raise OntologyCandidateContextRejected("invalid ontology user intent") from exc
        allowed_targets_by_type = {
            "candidate": {candidate_ref},
            "node": set(selected_nodes),
            "edge": set(selected_edges),
            "evidence": set(selected_evidence),
            "finding": set(),
            "release": set(),
            None: set(),
        }
        allowed_intent_targets = allowed_targets_by_type[parsed_intent.target_type]
        if any(item not in allowed_intent_targets for item in parsed_intent.target_ids):
            raise OntologyCandidateContextRejected(
                "user intent target type does not match governed candidate context"
            )
        user_intent = parsed_intent.model_dump(mode="json", by_alias=True)

    return {
        "schemaVersion": "knowledge_workbench_context.v2",
        "activeFunction": _bounded_active_function(browser_hint),
        "userIntent": user_intent,
        "graphRef": None,
        "graphVersion": None,
        "graphHash": snapshot.snapshot_hash,
        "authorityState": "candidate",
        "runMode": None,
        "candidateGraphId": snapshot.snapshot_ref,
        "selectedCandidateId": snapshot.snapshot_ref,
        "acceptedReleaseId": None,
        "acceptedReleaseVersion": None,
        "selectedNodeIds": selected_nodes,
        "selectedEdgeIds": selected_edges,
        "selectedRuleIds": selected_rules,
        "selectedEvidenceRefs": selected_evidence,
        "activeSourceIdentityRef": snapshot.source_identity_ref,
        "sourceAnchors": [],
        "governanceState": "candidate",
        "hasAcceptedRelease": False,
        "extractionRunId": None,
        "providerRef": "ontology-building-mvl",
        "providerCommit": None,
    }


__all__ = [
    "OntologyCandidateContextRejected",
    "resolve_ontology_candidate_context",
]
