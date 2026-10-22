from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

GraphCapabilityV1 = Literal[
    "view",
    "inspect_evidence",
    "inspect_authority",
    "trace_lineage",
    "trace_impact",
    "compare",
    "export_evidence",
    "validate",
    "approve",
    "activate",
    "reject",
    "deprecate",
    "escalate",
    "propose_supersession",
    "request_source_correction",
    "request_reindex",
]
GraphDirectionV1 = Literal["forward", "reverse", "bidirectional"]
GraphGovernanceStateV1 = Literal[
    "candidate",
    "validated",
    "approved",
    "active",
    "deprecated",
    "rejected",
]
GraphAuthorityRoleV1 = Literal[
    "structural",
    "binding",
    "interpretive",
    "empirical",
    "contextual",
    "conflict",
]
GraphFreshnessStatusV1 = Literal[
    "fresh",
    "stale",
    "indexing",
    "failed_retryable",
    "failed_terminal",
]
GraphNodeKindV1 = Literal[
    "source",
    "source_span",
    "record",
    "field",
    "entity",
    "relation",
    "classification",
    "claim",
    "rule",
    "policy",
    "authority",
    "conflict",
    "derivative",
    "actor",
    "derived_result",
]
GraphSceneLensV1 = Literal[
    "overview",
    "evidence",
    "sensitivity",
    "authority",
    "conflict",
    "lineage",
    "replay",
    "impact",
]
GraphLayoutProfileV1 = Literal[
    "evidence_chain",
    "authority_hierarchy",
    "conflict_comparison",
    "lineage",
    "impact_neighborhood",
    "replay_chain",
]


@dataclass(frozen=True)
class GraphPositionV1:
    x: float
    y: float


@dataclass(frozen=True)
class GraphIndexStatusV1:
    status: GraphFreshnessStatusV1
    source_ref: str
    checked_at: str
    index_fingerprint: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class GraphWarningV1:
    code: str
    message: str


@dataclass(frozen=True)
class GraphNodeV1:
    node_id: str
    node_kind: GraphNodeKindV1
    label: str
    display_summary: str
    content_hash: str
    governance_state: GraphGovernanceStateV1
    semantic_tier: str
    authority_role: GraphAuthorityRoleV1
    jurisdiction: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    contextual_only: bool = False
    capabilities: tuple[GraphCapabilityV1, ...] = ("view",)
    detail_ref: str | None = None
    source_anchor_refs: tuple[str, ...] = ()
    sensitivity_class: str | None = None
    position: GraphPositionV1 | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdgeV1:
    edge_id: str
    source_node_id: str
    target_node_id: str
    predicate: str
    predicate_label: str
    predicate_description: str
    direction: GraphDirectionV1
    content_hash: str
    governance_state: GraphGovernanceStateV1
    semantic_tier: str
    authority_role: GraphAuthorityRoleV1
    contextual_only: bool = False
    capabilities: tuple[GraphCapabilityV1, ...] = ("view",)
    detail_ref: str | None = None
    evidence_anchor_refs: tuple[str, ...] = ()
    assertion_version_ref: str | None = None
    assertion_hash: str | None = None
    source_assertion_refs: tuple[str, ...] = ()
    governed_event_refs: tuple[str, ...] = ()
    effective_from: str | None = None
    effective_to: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphSceneV1:
    scene_id: str
    title: str
    lens: GraphSceneLensV1
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...] = ()
    layout_profile: GraphLayoutProfileV1 = "evidence_chain"
    focus_node_id: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class GraphRedactionManifestV1:
    policy_version: str
    omitted_node_count: int = 0
    omitted_edge_count: int = 0
    minimized_node_count: int = 0
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphPathSummaryV1:
    path_id: str
    object_refs: tuple[str, ...]
    explanation: str
    incomplete: bool = False


@dataclass(frozen=True)
class GraphSearchRequestV1:
    query: str
    lens: GraphSceneLensV1 | None = None
    node_kind: GraphNodeKindV1 | None = None
    semantic_tier: str | None = None
    authority_role: GraphAuthorityRoleV1 | None = None
    governance_state: GraphGovernanceStateV1 | None = None
    effective_at: str | None = None
    source_ref: str | None = None
    limit: int = 20


@dataclass(frozen=True)
class GraphSearchResultV1:
    object_ref: str
    label: str
    node_kind: GraphNodeKindV1
    summary: str
    rank: int
    lens_hint: GraphSceneLensV1
    path_summary: GraphPathSummaryV1 | None = None


@dataclass(frozen=True)
class GraphConflictV1:
    conflict_ref: str
    conflict_type: Literal[
        "precedence",
        "temporal",
        "jurisdictional",
        "semantic",
        "duplicate",
        "scope_ambiguity",
    ]
    left_assertion_ref: str
    right_assertion_ref: str
    status: Literal["unresolved", "resolved"]
    resolver_rationale: str
    required_approver_roles: tuple[str, ...] = ()
    affected_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphImpactSummaryV1:
    artifact_type: str
    active_count: int
    historical_count: int
    path_summaries: tuple[GraphPathSummaryV1, ...] = ()
    incomplete: bool = False


@dataclass(frozen=True)
class GraphGovernanceCommandV1:
    command: Literal[
        "accept_for_review",
        "validate",
        "approve",
        "activate",
        "reject",
        "deprecate",
        "escalate",
        "propose_supersession",
        "request_source_correction",
        "request_ontology_correction",
        "request_extraction_correction",
    ]
    target_object_ref: str
    expected_graph_snapshot_ref: str
    expected_object_hash: str | None
    justification: str
    idempotency_key: str


@dataclass(frozen=True)
class GraphGovernanceCommandResultV1:
    status: Literal["accepted", "completed", "stale_snapshot", "rejected"]
    target_object_ref: str
    graph_snapshot_ref: str
    governed_event_refs: tuple[str, ...] = ()
    message: str | None = None


@dataclass(frozen=True)
class GraphVisualizationModelV1:
    projection_id: str
    organization_id: str
    workspace_id: str | None
    source_ref: str
    source_hash: str
    classification_run_ref: str
    ontology_version: str
    sensitivity_ontology_version: str
    resolver_snapshot_hash: str
    redaction_policy_version: str
    nodes: tuple[GraphNodeV1, ...]
    edges: tuple[GraphEdgeV1, ...]
    scenes: tuple[GraphSceneV1, ...]
    redaction_manifest: GraphRedactionManifestV1
    projection_hash: str
    presentation_hash: str
    as_of: str
    authorization_context_ref: str
    freshness: GraphIndexStatusV1
    warnings: tuple[GraphWarningV1, ...] = ()
    normalized_artifact_ref: str | None = None
    normalized_artifact_hash: str | None = None
    graph_snapshot_ref: str | None = None
    graph_snapshot_hash: str | None = None
    assertion_resolver_version: str | None = None
    schema_version: str = "governed_graph_contract.v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
