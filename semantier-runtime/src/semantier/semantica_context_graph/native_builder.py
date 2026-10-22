"""Controlled Phase1-001 build path for the fixed tender DOCX fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from contracts.accepted_graph_release_v2 import (
    AcceptedGraphReleaseV2,
    parse_accepted_graph_release_unverified,
)
from contracts.accepted_graph_release_lineage import AcceptedGraphReleaseLineageV2
from contracts.canonical_restriction_pattern import (
    canonical_pattern_bytes_unverified,
    parse_canonical_restriction_pattern_unverified,
)
from contracts.knowledge_relation_instance import (
    AcceptedRelationBindingV1,
    KnowledgeRelationInstanceV1,
)
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.source_elements import source_elements
from semantier.knowledge_feature_packs.semantica_adapter import (
    InProcessSemanticaAdapter,
    SEMANTICA_COMMIT,
    _ensure_vendored_semantica_importable,
    build_semantica_candidate_graph,
)

from .artifact_repository import ArtifactMetadata, ContextGraphArtifactRepository


FIXED_SOURCE = Path("docs/招投标法规/POC测试敏感词汇总.docx")
GRAPH_ID = "tender-sensitive"


@dataclass(frozen=True)
class V0CandidateBuildResult:
    metadata: ArtifactMetadata
    proposal_result: Any
    source_hash: str


def _json_artifact_writer(raw: str, digest: str) -> str:
    """Keep native provider output in memory for this controlled build path."""

    if not raw or not digest.startswith("sha256:"):
        raise ValueError("native provider artifact must be hashed")
    return f"memory://phase1-001/native/{digest}"


def _native_graph(candidate: Any) -> Any:
    from semantica.context import ContextGraph

    graph = ContextGraph(advanced_analytics=False)
    graph.graph_id = GRAPH_ID
    nodes = []
    for node in candidate.nodes:
        anchors = [anchor.model_dump(mode="json") for anchor in node.source_anchors]
        nodes.append(
            {
                "id": node.node_id,
                "type": node.node_type.value,
                "content": node.canonical_label,
                "properties": {
                    "normalized_value": node.normalized_value,
                    "display_name_zh": node.display_name_zh,
                    "rationale": node.rationale,
                    "source_anchors": anchors,
                    "evidence_refs": [ref.model_dump(mode="json") for ref in node.evidence_refs],
                    "extraction_run_id": node.extraction_run_id,
                    "machine_proposal_id": node.machine_proposal_id,
                    "machine_confidence": node.machine_confidence,
                },
            }
        )
    for rule in candidate.rules:
        anchors = [anchor.model_dump(mode="json") for anchor in rule.source_anchors]
        nodes.append(
            {
                "id": rule.rule_id,
                "type": "rule",
                "content": rule.reason_code,
                "properties": {
                    "family_id": rule.family_id,
                    "condition_node_ids": list(rule.condition_node_ids),
                    "exception_node_ids": list(rule.exception_node_ids),
                    "source_anchors": anchors,
                    "evidence_refs": [ref.model_dump(mode="json") for ref in rule.evidence_refs],
                    "extraction_run_id": rule.extraction_run_id,
                    "machine_proposal_id": rule.machine_proposal_id,
                },
            }
        )
    edges = [
        {
            "id": edge.edge_id,
            "source": edge.source_node_id,
            "target": edge.target_node_id,
            "type": edge.relation_type.value,
            "properties": {
                "rationale": edge.rationale,
                "source_anchors": [anchor.model_dump(mode="json") for anchor in edge.source_anchors],
                "evidence_refs": [ref.model_dump(mode="json") for ref in edge.evidence_refs],
                "extraction_run_id": edge.extraction_run_id,
                "machine_proposal_id": edge.machine_proposal_id,
                "machine_confidence": edge.machine_confidence,
            },
        }
        for edge in candidate.edges
    ]
    graph.add_nodes(nodes)
    graph.add_edges(edges)
    return graph


def native_graph_from_accepted_release(release: dict[str, Any]) -> Any:
    """Build topology only after explicit legacy/v2 contract dispatch."""

    release_contract = parse_accepted_graph_release_unverified(release)
    _ensure_vendored_semantica_importable()
    from semantica.context import ContextGraph

    graph = ContextGraph(advanced_analytics=False)
    graph.graph_id = GRAPH_ID
    nodes = []
    for node in release.get("nodes", []):
        nodes.append(
            {
                "id": node["node_id"],
                "type": str(node.get("node_type") or "concept"),
                "content": str(node.get("canonical_label") or node["node_id"]),
                "properties": {
                    "normalized_value": node.get("normalized_value"),
                    "display_name_zh": node.get("display_name_zh"),
                    "rationale": node.get("rationale"),
                    "source_anchors": node.get("source_anchors", []),
                    "evidence_refs": node.get("evidence_refs", []),
                    "extraction_run_id": node.get("extraction_run_id"),
                    "machine_proposal_id": node.get("machine_proposal_id"),
                    "machine_confidence": node.get("machine_confidence"),
                    "grounding_state": "accepted",
                    "accepted_graph_release_id": release["graph_version"],
                    "accepted_graph_release_hash": release["release_hash"],
                },
            }
        )
    for rule in release.get("rules", []):
        properties = {
            "family_id": rule.get("family_id"),
            "condition_node_ids": list(rule.get("condition_node_ids", [])),
            "exception_node_ids": list(rule.get("exception_node_ids", [])),
            "source_anchors": rule.get("source_anchors", []),
            "evidence_refs": rule.get("evidence_refs", []),
            "extraction_run_id": rule.get("extraction_run_id"),
            "machine_proposal_id": rule.get("machine_proposal_id"),
            "grounding_state": "accepted",
            "accepted_graph_release_id": release["graph_version"],
            "accepted_graph_release_hash": release["release_hash"],
        }
        if isinstance(release_contract, AcceptedGraphReleaseV2):
            properties["frame_pattern"] = rule.get("frame_pattern")
        nodes.append(
            {
                "id": rule["rule_id"],
                "type": "rule",
                "content": str(rule.get("reason_code") or rule["rule_id"]),
                "properties": properties,
            }
        )
    if isinstance(release_contract, AcceptedGraphReleaseV2):
        occupied = {str(item["id"]) for item in nodes}
        for member in release_contract.relation_membership:
            if member.relation_instance_ref in occupied:
                raise ValueError("v2 semantic membership identity collides with topology")
            occupied.add(member.relation_instance_ref)
            nodes.append({
                "id": member.relation_instance_ref,
                "type": "knowledge_relation_instance",
                "content": member.relation_instance_ref,
                "properties": {
                    "relation_instance_hash": member.relation_instance_hash,
                    "occurrence_grounding_ref": member.occurrence_grounding_ref,
                    "occurrence_grounding_hash": member.occurrence_grounding_hash,
                    "grounding_state": "accepted",
                    "accepted_graph_release_id": release_contract.graph_version,
                    "accepted_graph_release_hash": release_contract.release_hash,
                },
            })
        for member in release_contract.pattern_membership:
            if member.pattern_ref in occupied:
                raise ValueError("v2 semantic membership identity collides with topology")
            occupied.add(member.pattern_ref)
            nodes.append({
                "id": member.pattern_ref,
                "type": "canonical_restriction_pattern",
                "content": member.pattern_ref,
                "properties": {
                    "pattern_hash": member.pattern_hash,
                    "grounding_state": "accepted",
                    "accepted_graph_release_id": release_contract.graph_version,
                    "accepted_graph_release_hash": release_contract.release_hash,
                },
            })
    edges = [
        {
            "id": edge["edge_id"],
            "source": edge["source_node_id"],
            "target": edge["target_node_id"],
            "type": str(edge.get("relation_type") or "related_to"),
            "properties": {
                "rationale": edge.get("rationale"),
                "source_anchors": edge.get("source_anchors", []),
                "evidence_refs": edge.get("evidence_refs", []),
                "extraction_run_id": edge.get("extraction_run_id"),
                "machine_proposal_id": edge.get("machine_proposal_id"),
                "machine_confidence": edge.get("machine_confidence"),
                "grounding_state": "accepted",
                "accepted_graph_release_id": release["graph_version"],
                "accepted_graph_release_hash": release["release_hash"],
            },
        }
        for edge in release.get("edges", [])
    ]
    graph.add_nodes(nodes)
    graph.add_edges(edges)
    return graph


def native_graph_from_accepted_release_bundle(bundle: dict[str, Any]) -> Any:
    """Project an exact replayed v2 bundle into the native Semantica graph."""

    if not isinstance(bundle, dict) or not isinstance(bundle.get("release"), dict):
        raise ValueError("accepted semantic graph bundle is invalid")
    release = AcceptedGraphReleaseV2.parse_unverified(bundle["release"])
    if not isinstance(bundle.get("lineage"), dict):
        raise ValueError("accepted semantic graph bundle lineage is missing")
    lineage = AcceptedGraphReleaseLineageV2.parse_unverified(bundle["lineage"])
    if (
        lineage.accepted_graph_release_ref != release.graph_version
        or lineage.accepted_graph_release_hash != release.release_hash
    ):
        raise ValueError("semantic bundle lineage release mismatch")
    authority_context = (lineage.organization_id, lineage.workspace_id)
    graph = native_graph_from_accepted_release(release.as_dict())

    relation_payloads = bundle.get("knowledge_relation_instances")
    pattern_payloads = bundle.get("canonical_restriction_patterns")
    relation_bindings = bundle.get("accepted_relation_bindings")
    pattern_bindings = bundle.get("accepted_pattern_bindings")
    if not all(
        isinstance(items, list)
        for items in (
            relation_payloads, pattern_payloads,
            relation_bindings, pattern_bindings,
        )
    ):
        raise ValueError("accepted semantic graph bundle artifacts are missing")

    relations = [
        KnowledgeRelationInstanceV1.model_validate(item)
        for item in relation_payloads
    ]
    if any(
        (item.organization_id, item.workspace_id) != authority_context
        for item in relations
    ):
        raise ValueError("semantic bundle relation authority-context mismatch")
    relation_by_identity = {
        (item.relation_instance_id, item.relation_instance_hash): item
        for item in relations
    }
    expected_relations = {
        (item.relation_instance_ref, item.relation_instance_hash)
        for item in release.relation_membership
    }
    if set(relation_by_identity) != expected_relations or len(relations) != len(
        expected_relations
    ):
        raise ValueError("semantic bundle relation membership mismatch")

    raw_patterns = {
        str(item.get("pattern_id")): item for item in pattern_payloads
        if isinstance(item, dict)
    }
    if len(raw_patterns) != len(pattern_payloads):
        raise ValueError("semantic bundle contains duplicate pattern identity")
    parsed_patterns = {}
    visiting: set[str] = set()

    def resolve_pattern(pattern_ref: str):
        if pattern_ref in parsed_patterns:
            return parsed_patterns[pattern_ref]
        if pattern_ref in visiting:
            raise ValueError("semantic bundle pattern composition cycle")
        payload = raw_patterns.get(pattern_ref)
        if payload is None:
            raise ValueError("semantic bundle pattern component is missing")
        visiting.add(pattern_ref)
        component_refs: set[str] = set()
        pending = list(
            payload.get("composition", {}).get("member_pattern_refs", [])
        )
        while pending:
            component_ref = str(pending.pop())
            if component_ref in component_refs:
                continue
            component_refs.add(component_ref)
            component = resolve_pattern(component_ref)
            pending.extend(component.composition.member_pattern_refs)
        components = [parsed_patterns[ref] for ref in sorted(component_refs)]
        pattern = parse_canonical_restriction_pattern_unverified(
            payload,
            component_artifacts=tuple(
                (canonical_pattern_bytes_unverified(item), item.pattern_hash)
                for item in components
            ),
        )
        visiting.remove(pattern_ref)
        parsed_patterns[pattern_ref] = pattern
        return pattern

    for member in release.pattern_membership:
        resolve_pattern(member.pattern_ref)
    expected_patterns = {
        (item.pattern_ref, item.pattern_hash) for item in release.pattern_membership
    }
    actual_patterns = {
        (item.pattern_id, item.pattern_hash) for item in parsed_patterns.values()
    }
    if actual_patterns != expected_patterns or len(parsed_patterns) != len(
        expected_patterns
    ):
        raise ValueError("semantic bundle pattern membership mismatch")
    if any(
        (item.organization_id, item.workspace_id) != authority_context
        for item in parsed_patterns.values()
    ):
        raise ValueError("semantic bundle pattern authority-context mismatch")

    relation_binding_by_ref = {}
    for payload in relation_bindings:
        binding = AcceptedRelationBindingV1.model_validate(payload)
        key = binding.relation_instance_ref
        if key in relation_binding_by_ref:
            raise ValueError("semantic bundle contains duplicate relation binding")
        if (
            binding.accepted_graph_release_ref != release.graph_version
            or binding.accepted_graph_release_hash != release.release_hash
        ):
            raise ValueError("semantic bundle relation binding release mismatch")
        if (binding.organization_id, binding.workspace_id) != authority_context:
            raise ValueError(
                "semantic bundle relation binding authority-context mismatch"
            )
        relation_binding_by_ref[key] = binding
    pattern_binding_by_ref = {}
    pattern_binding_fields = {
        "schema_version", "binding_id", "binding_hash", "organization_id",
        "workspace_id", "pattern_ref", "pattern_hash",
        "accepted_graph_release_ref", "accepted_graph_release_hash",
    }
    for payload in pattern_bindings:
        if not isinstance(payload, dict) or set(payload) != pattern_binding_fields:
            raise ValueError(
                "semantic bundle pattern binding must contain exact fields"
            )
        hash_payload = dict(payload)
        binding_id = hash_payload.pop("binding_id", None)
        binding_hash = hash_payload.pop("binding_hash", None)
        expected_hash = "sha256:" + content_hash(hash_payload)
        if (
            binding_hash != expected_hash
            or binding_id != f"apb_{expected_hash[7:31]}"
            or payload.get("schema_version") != "accepted_pattern_binding.v1"
            or payload.get("accepted_graph_release_ref") != release.graph_version
            or payload.get("accepted_graph_release_hash") != release.release_hash
        ):
            raise ValueError("semantic bundle pattern binding is invalid")
        if (
            payload.get("organization_id"), payload.get("workspace_id")
        ) != authority_context:
            raise ValueError(
                "semantic bundle pattern binding authority-context mismatch"
            )
        pattern_ref = str(payload.get("pattern_ref"))
        if pattern_ref in pattern_binding_by_ref:
            raise ValueError("semantic bundle contains duplicate pattern binding")
        pattern_binding_by_ref[pattern_ref] = payload
    if set(relation_binding_by_ref) != {item[0] for item in expected_relations}:
        raise ValueError("semantic bundle relation binding completeness mismatch")
    if set(pattern_binding_by_ref) != {item[0] for item in expected_patterns}:
        raise ValueError("semantic bundle pattern binding completeness mismatch")

    for relation in relations:
        binding = relation_binding_by_ref[relation.relation_instance_id]
        if binding.relation_instance_hash != relation.relation_instance_hash:
            raise ValueError("semantic bundle relation binding target mismatch")
        node = graph.nodes[relation.relation_instance_id]
        semantic_properties = {
            "organization_id": relation.organization_id,
            "workspace_id": relation.workspace_id,
            "relation_type_id": relation.relation_type_id,
            "representation_kind": relation.representation_kind,
            "source_node_id": relation.source_node_id,
            "target_node_id": relation.target_node_id,
            "frame_ref": relation.frame_ref,
            "projection_source_refs": list(relation.projection_source_refs),
            "evidence_refs": [
                item.model_dump(mode="json", by_alias=True)
                for item in relation.evidence_refs
            ],
            "accepted_relation_binding_id": binding.binding_id,
            "accepted_relation_binding_hash": binding.binding_hash,
        }
        node.properties.update(semantic_properties)
        node.metadata.update(semantic_properties)
    for pattern in parsed_patterns.values():
        binding = pattern_binding_by_ref[pattern.pattern_id]
        if binding["pattern_hash"] != pattern.pattern_hash:
            raise ValueError("semantic bundle pattern binding target mismatch")
        node = graph.nodes[pattern.pattern_id]
        semantic_properties = {
            "organization_id": pattern.organization_id,
            "workspace_id": pattern.workspace_id,
            "pattern_kind": pattern.pattern_kind,
            "frames": list(pattern.frames),
            "composition": pattern.composition.model_dump(mode="json"),
            "semantic_dimensions": pattern.semantic_dimensions.model_dump(mode="json"),
            "source_metadata": pattern.source_metadata.model_dump(
                mode="json", by_alias=True
            ),
            "witness_metadata": pattern.witness_metadata.model_dump(mode="json"),
            "lexical_evidence_refs": list(pattern.lexical_evidence_refs),
            "lineage_metadata": pattern.lineage_metadata.model_dump(mode="json"),
            "authority_metadata": pattern.authority_metadata.model_dump(mode="json"),
            "accepted_pattern_binding_id": binding["binding_id"],
            "accepted_pattern_binding_hash": binding["binding_hash"],
        }
        node.properties.update(semantic_properties)
        node.metadata.update(semantic_properties)
    return graph


def build_v0_candidate(
    repository: ContextGraphArtifactRepository,
    *,
    lineage_id: str,
    source_path: str | Path = FIXED_SOURCE,
    source_manifest_hash: str | None = None,
    created_by_user_id: str = "phase1-001-build",
) -> ArtifactMetadata:
    """Build and persist V0 from the one repository-owned DOCX fixture.

    This is deliberately a controlled build/test operation. It is not exposed
    through Hermes, a browser route, or the Runtime Expert plugin.
    """

    return build_v0_candidate_with_result(
        repository,
        lineage_id=lineage_id,
        source_path=source_path,
        source_manifest_hash=source_manifest_hash,
        created_by_user_id=created_by_user_id,
    ).metadata


def build_v0_candidate_with_result(
    repository: ContextGraphArtifactRepository,
    *,
    lineage_id: str,
    source_path: str | Path = FIXED_SOURCE,
    source_manifest_hash: str | None = None,
    created_by_user_id: str = "phase1-001-build",
) -> V0CandidateBuildResult:
    """Build and persist V0 while returning the real Semantica proposal result."""

    path = Path(source_path)
    if path != FIXED_SOURCE:
        raise ValueError("Phase1-001 build path accepts only the fixed DOCX fixture")
    if not path.is_file():
        raise FileNotFoundError(path)
    parsed = source_elements(path)
    source_hash = source_manifest_hash or parsed["document_hash"]
    result = build_semantica_candidate_graph(
        parsed_document=parsed,
        extraction_run_id="phase1-001-v0",
        adapter=InProcessSemanticaAdapter(artifact_writer=_json_artifact_writer),
    )
    native_graph = _native_graph(result.graph)
    metadata = repository.create_checkpoint(
        native_graph,
        lineage_id=lineage_id,
        graph_id=GRAPH_ID,
        graph_version="v0001-candidate-r001",
        artifact_kind="candidate",
        semantica_commit=SEMANTICA_COMMIT,
        source_manifest_hash=source_hash,
        created_by_user_id=created_by_user_id,
        artifact_id="phase1-001-v0-candidate",
    )
    return V0CandidateBuildResult(
        metadata=metadata,
        proposal_result=result,
        source_hash=source_hash,
    )


def graph_json_digest(repository: ContextGraphArtifactRepository, artifact_id: str) -> str:
    path = repository.artifact_path(artifact_id)
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def canonical_graph_summary(repository: ContextGraphArtifactRepository, artifact_id: str) -> dict[str, Any]:
    metadata = repository.get(artifact_id)
    graph = repository.load_graph(artifact_id)
    return {
        "artifact_id": metadata.artifact_id,
        "graph_id": metadata.graph_id,
        "graph_version": metadata.graph_version,
        "content_hash": metadata.content_hash,
        "node_ids": sorted(str(node_id) for node_id in graph.nodes),
        "edge_ids": sorted(str(edge.edge_id) for edge in graph.edges),
        "semantica_commit": metadata.semantica_commit,
    }
