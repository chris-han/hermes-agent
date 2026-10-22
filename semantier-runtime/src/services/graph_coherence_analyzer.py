"""Deterministic advisory analysis over an exact accepted projection."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.graph_coherence_analysis import create_graph_coherence_analysis
from contracts.semantic_graph_visual_projection import SemanticGraphVisualProjectionV1
from eos.artifact_hashing import content_hash


class GraphCoherenceLimitExceeded(ValueError):
    """Raised when configured deterministic analyzer bounds are exceeded."""


class GraphCoherencePolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["graph_coherence_policy.v1"]
    policy_ref: str
    policy_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    analyzer_version: str
    max_nodes: int = Field(gt=0)
    max_edges: int = Field(gt=0)
    high_significance_threshold: float = Field(ge=0, le=1)
    isolated_component_min_nodes: int = Field(ge=2)
    weak_evidence_max_refs: int = Field(ge=0)
    frame_min_projection_sources: int = Field(ge=1)
    unmodeled_pattern_min_occurrences: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_hash(self) -> "GraphCoherencePolicyV1":
        payload = self.model_dump(mode="json")
        payload.pop("policy_hash")
        if self.policy_hash != "sha256:" + content_hash(payload):
            raise ValueError("policy_hash does not match policy content")
        return self


def load_default_graph_coherence_policy() -> GraphCoherencePolicyV1:
    path = Path(__file__).resolve().parents[1] / "catalogs" / "graph_coherence_policy.v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["policy_hash"] = "sha256:" + content_hash(payload)
    return GraphCoherencePolicyV1.model_validate(payload)


class GraphCoherenceAnalyzer:
    """Emit investigation signals without changing accepted semantic state."""

    def __init__(self, policy: GraphCoherencePolicyV1) -> None:
        if type(policy) is not GraphCoherencePolicyV1:
            raise TypeError("analyzer requires an exact versioned policy")
        self.policy = policy

    def analyze(
        self,
        *,
        accepted_release: dict[str, Any],
        projection: SemanticGraphVisualProjectionV1,
    ):
        if type(projection) is not SemanticGraphVisualProjectionV1:
            raise TypeError("analyzer requires a canonical visual projection")
        if (
            accepted_release.get("graph_version") != projection.graph_release_ref
            or accepted_release.get("release_hash") != projection.graph_release_hash
        ):
            raise ValueError("accepted release and projection pins do not match")
        if len(projection.nodes) > self.policy.max_nodes or len(projection.edges) > self.policy.max_edges:
            raise GraphCoherenceLimitExceeded("graph exceeds coherence analyzer limits")

        diagnostics: dict[str, list[dict[str, Any]]] = {
            "identity_coherence": [],
            "structural_coherence": [],
            "schema_coherence": [],
            "evidence_coherence": [],
        }
        self._identity(projection, diagnostics["identity_coherence"])
        self._structural(projection, diagnostics["structural_coherence"])
        self._schema(projection, diagnostics["schema_coherence"])
        self._evidence(projection, diagnostics["evidence_coherence"])
        candidates = [self._candidate(item) for family in diagnostics.values() for item in family]
        return create_graph_coherence_analysis({
            "graph_release_ref": projection.graph_release_ref,
            "graph_release_hash": projection.graph_release_hash,
            "projection_ref": projection.projection_id,
            "projection_hash": projection.projection_hash,
            "analyzer_version": self.policy.analyzer_version,
            "analyzer_policy_ref": self.policy.policy_ref,
            "analyzer_policy_hash": self.policy.policy_hash,
            "diagnostics": diagnostics,
            "repair_candidates": candidates,
        })

    def _identity(self, projection, output) -> None:
        labels: dict[str, list[str]] = defaultdict(list)
        for node in projection.nodes:
            normalized = re.sub(r"[^0-9a-z]+", "", node.label.casefold())
            if normalized:
                labels[normalized].append(node.id)
        for normalized, node_ids in sorted(labels.items()):
            if len(node_ids) > 1:
                output.append(self._diagnostic(
                    "POSSIBLE_ENTITY_FRAGMENTATION",
                    affected_node_ids=node_ids,
                    diagnostic_basis={"normalized_label": normalized},
                ))

    def _structural(self, projection, output) -> None:
        neighbors = {node.id: set() for node in projection.nodes}
        importance_by_id = {node.id: node.importance for node in projection.nodes}
        for edge in projection.edges:
            neighbors[edge.source].add(edge.target)
            neighbors[edge.target].add(edge.source)
        for node in projection.nodes:
            if not neighbors[node.id] and node.importance >= self.policy.high_significance_threshold:
                output.append(self._diagnostic(
                    "HIGH_SIGNIFICANCE_ORPHAN_NODE",
                    affected_node_ids=[node.id],
                    diagnostic_basis={"importance": node.importance},
                ))
        components: list[list[str]] = []
        unseen = set(neighbors)
        while unseen:
            pending = [min(unseen)]
            component: list[str] = []
            while pending:
                node_id = pending.pop()
                if node_id not in unseen:
                    continue
                unseen.remove(node_id)
                component.append(node_id)
                pending.extend(sorted(neighbors[node_id] & unseen, reverse=True))
            components.append(sorted(component))
        if len(components) <= 1:
            return
        for component in sorted(components):
            if len(component) < self.policy.isolated_component_min_nodes:
                continue
            mean_importance = round(
                sum(importance_by_id[node_id] for node_id in component) / len(component),
                12,
            )
            if mean_importance >= self.policy.high_significance_threshold:
                output.append(self._diagnostic(
                    "HIGH_SIGNIFICANCE_ISOLATED_COMPONENT",
                    affected_node_ids=component,
                    diagnostic_basis={
                        "component_size": len(component),
                        "mean_importance": mean_importance,
                    },
                ))

    def _schema(self, projection, output) -> None:
        for edge in projection.edges:
            if (
                edge.representation_kind == "frame_role"
                and len(edge.projection_source_refs) < self.policy.frame_min_projection_sources
            ):
                output.append(self._diagnostic(
                    "FRAME_ROLE_INCOMPLETENESS",
                    affected_node_ids=[edge.source, edge.target],
                    affected_relation_instance_refs=[edge.relation_instance_ref],
                    evidence_refs=_evidence_refs(edge.evidence_refs),
                    diagnostic_basis={
                        "frame_ref": edge.frame_ref,
                        "projection_source_count": len(edge.projection_source_refs),
                    },
                ))
        patterns: dict[str, list[str]] = defaultdict(list)
        for node in projection.nodes:
            pattern = node.properties.get("unmodeled_relation_pattern")
            if isinstance(pattern, str) and pattern and pattern.isascii():
                patterns[pattern].append(node.id)
        for pattern, node_ids in sorted(patterns.items()):
            if len(node_ids) >= self.policy.unmodeled_pattern_min_occurrences:
                output.append(self._diagnostic(
                    "RECURRING_UNMODELED_RELATION_PATTERN",
                    affected_node_ids=node_ids,
                    diagnostic_basis={
                        "pattern_id": pattern,
                        "occurrence_count": len(node_ids),
                    },
                ))

    def _evidence(self, projection, output) -> None:
        for edge in projection.edges:
            evidence_refs = _evidence_refs(edge.evidence_refs)
            roles = {
                str(item.get("evidenceRole") or item.get("evidence_role") or "").casefold()
                for item in edge.evidence_refs
            }
            if len(evidence_refs) <= self.policy.weak_evidence_max_refs:
                output.append(self._diagnostic(
                    "WEAK_EVIDENCE_NEIGHBORHOOD",
                    affected_node_ids=[edge.source, edge.target],
                    affected_relation_instance_refs=[edge.relation_instance_ref],
                    evidence_refs=evidence_refs,
                    diagnostic_basis={"evidence_ref_count": len(evidence_refs)},
                ))
            if roles & {"support", "supports"} and roles & {"contradict", "contradicts", "conflict"}:
                output.append(self._diagnostic(
                    "CONTRADICTORY_EVIDENCE_NEIGHBORHOOD",
                    affected_node_ids=[edge.source, edge.target],
                    affected_relation_instance_refs=[edge.relation_instance_ref],
                    evidence_refs=evidence_refs,
                    diagnostic_basis={"evidence_roles": sorted(roles)},
                ))

    @staticmethod
    def _diagnostic(
        code: str,
        *,
        affected_node_ids=(),
        affected_entity_refs=(),
        affected_relation_instance_refs=(),
        evidence_refs=(),
        diagnostic_basis=None,
    ) -> dict[str, Any]:
        content = {
            "diagnostic_code": code,
            "affected_node_ids": sorted(set(affected_node_ids)),
            "affected_entity_refs": sorted(set(affected_entity_refs)),
            "affected_relation_instance_refs": sorted(set(affected_relation_instance_refs)),
            "evidence_refs": sorted(set(evidence_refs)),
            "diagnostic_basis": diagnostic_basis or {},
        }
        digest = content_hash(content)
        return {"diagnostic_ref": "gcd_" + digest[:24], **content}

    @staticmethod
    def _candidate(diagnostic: dict[str, Any]) -> dict[str, Any]:
        code = diagnostic["diagnostic_code"]
        kind = {
            "POSSIBLE_ENTITY_FRAGMENTATION": "entity_resolution",
            "HIGH_SIGNIFICANCE_ISOLATED_COMPONENT": "missing_relation",
            "HIGH_SIGNIFICANCE_ORPHAN_NODE": "missing_relation",
            "FRAME_ROLE_INCOMPLETENESS": "schema_gap",
            "RECURRING_UNMODELED_RELATION_PATTERN": "schema_gap",
            "WEAK_EVIDENCE_NEIGHBORHOOD": "evidence_gap",
            "CONTRADICTORY_EVIDENCE_NEIGHBORHOOD": "evidence_gap",
        }[code]
        return {
            "candidate_kind": kind,
            "diagnostic_code": code,
            "affected_node_ids": diagnostic["affected_node_ids"],
            "affected_entity_refs": diagnostic["affected_entity_refs"],
            "affected_relation_instance_refs": diagnostic["affected_relation_instance_refs"],
            "evidence_refs": diagnostic["evidence_refs"],
            "diagnostic_basis": diagnostic["diagnostic_basis"],
            "confidence": None,
            "authority": "advisory",
        }


def _evidence_refs(items) -> list[str]:
    return sorted({
        str(item.get("evidenceRef") or item.get("evidence_ref"))
        for item in items
        if item.get("evidenceRef") or item.get("evidence_ref")
    })
