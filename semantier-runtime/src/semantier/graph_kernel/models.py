from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


AssertionClass = Literal["structural", "semantic", "governance", "lineage", "inferred"]
AuthorityRole = Literal["structural", "binding", "interpretive", "empirical"]
GovernanceState = Literal["candidate", "validated", "approved", "active", "deprecated", "rejected"]


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    organization_id: str
    node_type: str
    canonical_key: str
    display_name: str | None
    properties: dict[str, Any]
    content_hash: str

    def to_record(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class GraphAssertion:
    assertion_id: str
    organization_id: str
    subject_id: str
    predicate: str
    object_id: str
    assertion_class: AssertionClass
    semantic_tier: str | None
    authority_role: AuthorityRole
    governance_state: GovernanceState
    valid_from: str | None
    valid_to: str | None
    source_ref: str
    source_anchor: str | None
    extraction_method: str
    extractor_version: str
    confidence: float | None
    content_hash: str

    def to_record(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class GraphSnapshot:
    graph_version_id: str
    organization_id: str
    ontology_version: str
    resolver_version: str
    active_assertion_set_hash: str
    assertion_ids: list[str]
    snapshot_hash: str

    def to_record(self) -> dict[str, Any]:
        return self.__dict__.copy()

