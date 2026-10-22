"""Strict additive contracts for the ontology-building minimum viable loop.

Historical source, intent, schema, witness, release, and workbench contracts are
deliberately not imported or widened here.  These companion artifacts bind to
their exact ref/hash identities and can therefore evolve without rewriting
accepted historical payloads.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.artifact_pin import ArtifactPinV1 as ArtifactPinV1
from contracts.ontology_quality import QualityRequirementDeltaV1


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def persisted_timestamps_are_utc(self) -> "_Contract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime):
                if value.tzinfo is None or value.utcoffset() != timedelta(0):
                    raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class OntologyOperation(str, Enum):
    EXPLORE = "EXPLORE"
    COMPILE_CERTIFIABLE = "COMPILE_CERTIFIABLE"
    CERTIFY = "CERTIFY"
    ADMIT = "ADMIT"
    ACTIVATE = "ACTIVATE"


class ExploratoryGraphNodeV1(_Contract):
    node_id: str = Field(min_length=1)
    node_kind: str = Field(min_length=1)
    label: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)


StructuralRelationRole = Literal[
    "authority", "specialization", "containment", "sequence", "dependency",
    "lineage", "cross_reference", "evidence", "other",
]


class ExploratoryGraphEdgeV1(_Contract):
    edge_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    relation_type: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    # Producers declare roles from canonical relation semantics before pinning
    # the snapshot. This wire contract never infers them from relation text.
    # Unannotated edges retain their exact serialized artifact identity.
    structural_roles: tuple[StructuralRelationRole, ...] = Field(
        default=(), exclude_if=lambda roles: not roles,
    )


class ExploratoryGraphSnapshotV1(_Contract):
    schema_version: Literal["exploratory_graph_snapshot.v1"] = (
        "exploratory_graph_snapshot.v1"
    )
    snapshot_ref: str = Field(min_length=1)
    snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    build_ref: str = Field(min_length=1)
    source_identity_ref: str = Field(min_length=1)
    source_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_source_ir_ref: str = Field(min_length=1)
    canonical_source_ir_hash: str = Field(pattern=SHA256_PATTERN)
    authority_state: Literal["provisional", "certified"] = "provisional"
    completeness_state: Literal["incomplete", "bounded_complete"] = "incomplete"
    nodes: tuple[ExploratoryGraphNodeV1, ...]
    edges: tuple[ExploratoryGraphEdgeV1, ...]
    unresolved_gap_ids: tuple[str, ...]
    predecessor_snapshot_ref: str | None = None
    predecessor_snapshot_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    change_cause: Literal[
        "initial_exploration",
        "schema_rebuild",
        "bounded_repair",
        "runtime_feedback",
        "controlled_trial_projection",
    ]
    triggering_ref: str | None = None
    created_at: datetime

    @model_validator(mode="after")
    def complete_lineage(self) -> "ExploratoryGraphSnapshotV1":
        if (self.predecessor_snapshot_ref is None) != (
            self.predecessor_snapshot_hash is None
        ):
            raise ValueError("predecessor ref/hash must both be present or absent")
        if self.change_cause != "initial_exploration" and self.predecessor_snapshot_ref is None:
            raise ValueError("successor snapshots require predecessor ref/hash")
        node_ids = {node.node_id for node in self.nodes}
        for edge in self.edges:
            if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
                raise ValueError("edges must reference nodes in the same snapshot")
        return self


class CandidateGraphDiffProjectionV1(_Contract):
    schema_version: Literal["candidate_graph_diff_projection.v1"] = (
        "candidate_graph_diff_projection.v1"
    )
    diff_ref: str = Field(min_length=1)
    diff_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    baseline_snapshot_ref: str = Field(min_length=1)
    baseline_snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    successor_snapshot_ref: str = Field(min_length=1)
    successor_snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    added_node_ids: tuple[str, ...]
    removed_node_ids: tuple[str, ...]
    added_edge_ids: tuple[str, ...]
    removed_edge_ids: tuple[str, ...]
    quality_changes: tuple[QualityRequirementDeltaV1, ...]
    comparison_verdict: Literal["IMPROVED", "CHANGED", "REGRESSED", "INCONCLUSIVE"]
    verdict_evidence_refs: tuple[str, ...]
    change_cause: Literal["schema_rebuild", "bounded_repair", "runtime_feedback"]
    triggering_ref: str | None
    created_at: datetime


class CandidateGraphLineageNodeV1(_Contract):
    snapshot_ref: str = Field(min_length=1)
    snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    build_ref: str = Field(min_length=1)
    authority_state: Literal["provisional", "certified"]
    completeness_state: Literal["incomplete", "bounded_complete"]


class CandidateGraphLineageEdgeV1(_Contract):
    edge_kind: Literal["artifact_successor"] = "artifact_successor"
    predecessor_snapshot_ref: str = Field(min_length=1)
    predecessor_snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    successor_snapshot_ref: str = Field(min_length=1)
    successor_snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    branch_ref: str = Field(min_length=1)
    change_cause: Literal["schema_rebuild", "bounded_repair", "runtime_feedback"]
    triggering_ref: str | None


class CandidateGraphLineageProjectionV1(_Contract):
    schema_version: Literal["candidate_graph_lineage_projection.v1"] = (
        "candidate_graph_lineage_projection.v1"
    )
    lineage_ref: str = Field(min_length=1)
    lineage_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    nodes: tuple[CandidateGraphLineageNodeV1, ...] = Field(min_length=1)
    edges: tuple[CandidateGraphLineageEdgeV1, ...]
    created_at: datetime


class CanonicalEvidenceBoundaryV1(_Contract):
    schema_version: Literal["canonical_evidence_boundary.v1"] = (
        "canonical_evidence_boundary.v1"
    )
    boundary_ref: str = Field(min_length=1)
    boundary_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    source_identity_ref: str = Field(min_length=1)
    source_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_source_ir_ref: str = Field(min_length=1)
    canonical_source_ir_hash: str = Field(pattern=SHA256_PATTERN)
    intended_domain: str = Field(min_length=1)
    intended_use: str = Field(min_length=1)
    target_time: datetime
    jurisdiction: str = Field(min_length=1)
    explicit_exclusions: tuple[str, ...]
    created_at: datetime


class EvidenceAdequacyFindingV1(_Contract):
    finding_id: str = Field(min_length=1)
    finding_type: Literal[
        "missing_normative_dependency",
        "unresolved_source_authority_role",
        "unresolved_domain_or_competency_target",
        "target_time_outside_effectivity",
        "missing_critical_definition_region",
        "incompatible_cross_source_jurisdiction",
        "incompatible_cross_source_version_overlap",
    ]
    severity: Literal["warning", "blocking"]
    message: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    search_evidence_refs: tuple[str, ...]

    @model_validator(mode="after")
    def exact_search_evidence_required(self) -> "EvidenceAdequacyFindingV1":
        if not self.search_evidence_refs:
            raise ValueError("exact search evidence is required for every adequacy finding")
        return self


class OntologyStageGateV1(_Contract):
    schema_version: Literal["ontology_stage_gate.v1"] = "ontology_stage_gate.v1"
    operation: OntologyOperation
    decision: Literal["PROCEED", "ASK", "PAUSE", "ESCALATE"]
    blocker_refs: tuple[str, ...]
    assessment_ref: str = Field(min_length=1)
    assessment_hash: str = Field(pattern=SHA256_PATTERN)


class CanonicalEvidenceAdequacyAssessmentV1(_Contract):
    schema_version: Literal["canonical_evidence_adequacy_assessment.v1"] = (
        "canonical_evidence_adequacy_assessment.v1"
    )
    assessment_ref: str = Field(min_length=1)
    assessment_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    boundary_ref: str = Field(min_length=1)
    boundary_hash: str = Field(pattern=SHA256_PATTERN)
    findings: tuple[EvidenceAdequacyFindingV1, ...]
    assessed_at: datetime

    def gate_for(self, operation: OntologyOperation) -> OntologyStageGateV1:
        blockers = tuple(
            finding.finding_id
            for finding in self.findings
            if finding.severity == "blocking"
        )
        if operation is OntologyOperation.EXPLORE:
            decision = "PROCEED"
            blockers = ()
        else:
            decision = "PAUSE" if blockers else "PROCEED"
        return OntologyStageGateV1(
            operation=operation,
            decision=decision,
            blocker_refs=blockers,
            assessment_ref=self.assessment_ref,
            assessment_hash=self.assessment_hash,
        )


class CompetencyDimensionApplicabilityV1(_Contract):
    dimension: str
    state: Literal["REQUIRED", "NOT_APPLICABLE", "UNRESOLVED_APPLICABILITY"]
    rationale: str = Field(min_length=1)
    evidence_refs: tuple[str, ...]

    @model_validator(mode="after")
    def reviewed_basis(self):
        if not self.rationale.strip() or (self.state != "UNRESOLVED_APPLICABILITY" and not self.evidence_refs):
            raise ValueError("applicability requires rationale and exact source evidence")
        return self


class CompetencyQuestionV1(_Contract):
    competency_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    criticality: Literal["proposed_critical", "critical", "noncritical", "excluded"]
    review_state: Literal["unresolved", "confirmed", "rejected"]
    required_semantic_dimensions: tuple[str, ...] = Field(min_length=1)
    acceptance_obligations: tuple[str, ...] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    provenance_kind: Literal["source_derived", "user_confirmed", "system_confirmed"]
    dimension_applicability: tuple[CompetencyDimensionApplicabilityV1, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def complete_applicability(self):
        if self.dimension_applicability is not None:
            from contracts.normative_schema import NormativeSchemaContractV1
            rows = self.dimension_applicability
            if len(rows) != len(NormativeSchemaContractV1.REQUIRED_DIMENSIONS) or {
                r.dimension for r in rows} != set(NormativeSchemaContractV1.REQUIRED_DIMENSIONS):
                raise ValueError("applicability must classify all nine dimensions exactly once")
            if set(self.required_semantic_dimensions) != {r.dimension for r in rows if r.state == "REQUIRED"}:
                raise ValueError("required dimensions must match applicability projection")
        return self

    @property
    def applicability_resolved(self):
        return self.dimension_applicability is None or all(
            row.state != "UNRESOLVED_APPLICABILITY" for row in self.dimension_applicability)


class CompetencyContractV1(_Contract):
    schema_version: Literal["competency_contract.v1"] = "competency_contract.v1"
    competency_contract_ref: str = Field(min_length=1)
    competency_contract_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    boundary_ref: str = Field(min_length=1)
    boundary_hash: str = Field(pattern=SHA256_PATTERN)
    graph_intent_brief_ref: str = Field(min_length=1)
    graph_intent_brief_hash: str = Field(pattern=SHA256_PATTERN)
    intended_domain: str = Field(min_length=1)
    in_scope: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    questions: tuple[CompetencyQuestionV1, ...] = Field(min_length=1)
    applicability_review_pin: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    predecessor_ref: str | None = None
    predecessor_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    created_at: datetime

    @model_validator(mode="after")
    def complete_predecessor(self) -> "CompetencyContractV1":
        if (self.predecessor_ref is None) != (self.predecessor_hash is None):
            raise ValueError("predecessor ref/hash must both be present or absent")
        if any(q.dimension_applicability is not None for q in self.questions):
            if (self.applicability_review_pin is None or self.predecessor_ref is None
                or self.applicability_review_pin.artifact_kind != "conversation_decision"
                or any(q.dimension_applicability is None for q in self.questions)):
                raise ValueError("applicability requires a complete governed successor review")
        return self


class ConversationDecisionV1(_Contract):
    schema_version: Literal["conversation_decision.v1"] = "conversation_decision.v1"
    decision_ref: str = Field(min_length=1)
    decision_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    actor_type: Literal["human", "system"]
    governed_actor_ref: str = Field(min_length=1)
    channel: Literal["chat", "direct_ui"]
    decision_kind: Literal[
        "confirm_criticality", "downgrade_criticality", "exclude_competency", "define_boundary"
    ]
    target_ref: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    decided_at: datetime

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"channel"})


class OntologyBuildManifestV1(_Contract):
    schema_version: Literal["ontology_build_manifest.v1"] = "ontology_build_manifest.v1"
    build_ref: str = Field(min_length=1)
    build_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    assurance_profile: Literal[
        "A0_EXPLORATORY",
        "A1_INTERNAL_POLICY_MVL",
        "A2_GOVERNED_ENTERPRISE",
        "A3_REGULATORY_HIGH_ASSURANCE",
    ]
    child_artifacts: tuple[ArtifactPinV1, ...]
    gates: tuple[OntologyStageGateV1, ...]
    predecessor_build_ref: str | None = None
    predecessor_build_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    created_at: datetime

    @model_validator(mode="after")
    def manifest_is_complete(self) -> "OntologyBuildManifestV1":
        if (self.predecessor_build_ref is None) != (self.predecessor_build_hash is None):
            raise ValueError("predecessor build ref/hash must both be present or absent")
        kinds = [item.artifact_kind for item in self.child_artifacts]
        if len(kinds) != len(set(kinds)):
            raise ValueError("child artifact kinds must be unique")
        operations = [gate.operation for gate in self.gates]
        if len(operations) != len(set(operations)):
            raise ValueError("gate operations must be unique")
        return self


__all__ = [
    "ArtifactPinV1",
    "CanonicalEvidenceAdequacyAssessmentV1",
    "CanonicalEvidenceBoundaryV1",
    "CandidateGraphDiffProjectionV1",
    "CandidateGraphLineageEdgeV1",
    "CandidateGraphLineageNodeV1",
    "CandidateGraphLineageProjectionV1",
    "CompetencyContractV1",
    "CompetencyQuestionV1",
    "ConversationDecisionV1",
    "EvidenceAdequacyFindingV1",
    "ExploratoryGraphEdgeV1",
    "ExploratoryGraphNodeV1",
    "ExploratoryGraphSnapshotV1",
    "OntologyBuildManifestV1",
    "OntologyOperation",
    "OntologyStageGateV1",
]
