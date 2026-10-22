"""Closed comparison inputs; parsing establishes neither compatibility nor authority."""

from typing import Annotated, Literal
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_repair import RuntimeRepairMetricResultV1
from contracts.runtime_working_context import V2UtcDateTime
from eos.artifact_hashing import content_hash

_Id = Annotated[str, Field(min_length=1, pattern=r"^[\x20-\x7e]+$")]
_UtcTime = Annotated[V2UtcDateTime, Field(strict=False)]


class _ComparisonContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="after")
    def verify_pin_identifiers(self):
        identities = {}

        def check(value):
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError("comparison timestamps must be timezone-aware UTC")
            if isinstance(value, ArtifactPinV1):
                if (
                    not value.artifact_kind.isascii()
                    or not value.artifact_ref.isascii()
                ):
                    raise ValueError("comparison pin identifiers must be ASCII-stable")
                key = (value.artifact_kind, value.artifact_ref)
                previous = identities.setdefault(key, value.artifact_hash)
                if previous != value.artifact_hash:
                    raise ValueError(
                        "comparison has conflicting hashes for one artifact"
                    )
            elif isinstance(value, tuple):
                for item in value:
                    check(item)

        for field in type(self).model_fields:
            check(getattr(self, field))
        return self


class VerifiedComparisonInputsV1(_ComparisonContract):
    """Server-verified, pure-projector inputs; never an authority grant."""

    schema_version: Literal["verified_comparison_inputs.v1"]
    organization_id: _Id
    workspace_id: _Id
    subject_payload_hash: _Id
    resolved_pins: tuple[ArtifactPinV1, ...]
    compatibility_evidence: tuple[ArtifactPinV1, ...]
    origin_handoff_pin: ArtifactPinV1
    navigation_anchors: tuple[_Id, ...]
    source_times: tuple[_UtcTime, ...]


DecisionStructureFamilyV1 = Literal[
    "RULE",
    "CONSTRAINT",
    "SCOPE",
    "QUALIFICATION",
    "EXCEPTION",
    "PRECEDENCE",
    "TEMPORAL",
    "EVIDENCE_BINDING",
    "ENTITY_DISTINCTION",
]


class DecisionStructureProjectionV1(_ComparisonContract):
    """Pure bounded ContextGraph projection used only as empirical learning input.

    It is content-addressed for reproducibility but is not a persisted authority,
    graph store, release, admission, or activation artifact.
    """

    schema_version: Literal["decision_structure_projection.v1"] = "decision_structure_projection.v1"
    projection_ref: _Id
    projection_hash: _Id
    organization_id: _Id
    workspace_id: _Id
    source_snapshot: ArtifactPinV1
    context_pins: tuple[ArtifactPinV1, ...]
    node_ids: tuple[_Id, ...]
    edge_ids: tuple[_Id, ...]
    evidence_refs: tuple[_Id, ...]
    semantic_families: tuple[DecisionStructureFamilyV1, ...]

    @model_validator(mode="after")
    def verify_identity(self):
        digest = content_hash(
            self.model_dump(mode="json", exclude={"projection_ref", "projection_hash"})
        )
        if (
            self.projection_ref != "decision-structure-projection:" + digest
            or self.projection_hash != "sha256:" + digest
        ):
            raise ValueError("decision structure projection identity mismatch")
        if len(self.node_ids) != len(set(self.node_ids)):
            raise ValueError("decision structure node ids must be unique")
        if len(self.edge_ids) != len(set(self.edge_ids)):
            raise ValueError("decision structure edge ids must be unique")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("decision structure evidence refs must be unique")
        return self


StructuralLearningFailureClassV1 = Literal[
    "CAPABILITY", "STEERING", "STRUCTURE", "ENVIRONMENT"
]


class BuildCandidatePairV1(_ComparisonContract):
    schema_version: Literal["build_candidate_pair.v1"]
    pair_kind: Literal["BUILD_CANDIDATES"]
    organization_id: _Id
    workspace_id: _Id
    baseline_manifest: ArtifactPinV1
    candidate_manifest: ArtifactPinV1
    baseline_snapshot: ArtifactPinV1
    candidate_snapshot: ArtifactPinV1
    subtype: Literal["SEED", "RUNTIME_REPAIR"]
    lineage_receipt: ArtifactPinV1
    repair_or_alignment_receipt: ArtifactPinV1


class WorldSnapshotPairV1(_ComparisonContract):
    schema_version: Literal["world_snapshot_pair.v1"]
    pair_kind: Literal["WORLD_SNAPSHOTS"]
    organization_id: _Id
    workspace_id: _Id
    baseline_world: ArtifactPinV1
    candidate_world: ArtifactPinV1
    semantic_basis: ArtifactPinV1
    baseline_valid_time: ArtifactPinV1
    candidate_valid_time: ArtifactPinV1
    baseline_source_facts: tuple[ArtifactPinV1, ...]
    candidate_source_facts: tuple[ArtifactPinV1, ...]
    query_pins: tuple[ArtifactPinV1, ...]


class ReasoningVersionPairV1(_ComparisonContract):
    schema_version: Literal["reasoning_version_pair.v1"]
    pair_kind: Literal["REASONING_VERSIONS"]
    organization_id: _Id
    workspace_id: _Id
    baseline_context: ArtifactPinV1
    candidate_context: ArtifactPinV1
    stream_ref: _Id
    root_context: ArtifactPinV1
    branch_relationship: ArtifactPinV1 | None
    baseline_world: ArtifactPinV1
    candidate_world: ArtifactPinV1
    transition_pins: tuple[ArtifactPinV1, ...]
    proof_pins: tuple[ArtifactPinV1, ...]
    provider_pins: tuple[ArtifactPinV1, ...]


class RuntimeRerunPairV1(_ComparisonContract):
    schema_version: Literal["runtime_rerun_pair.v1"]
    pair_kind: Literal["RUNTIME_RERUNS"]
    organization_id: _Id
    workspace_id: _Id
    original_run: ArtifactPinV1
    rerun: ArtifactPinV1
    original_epoch: ArtifactPinV1
    rerun_epoch: ArtifactPinV1
    original_target: ArtifactPinV1
    rerun_target: ArtifactPinV1
    original_report: ArtifactPinV1 | None
    rerun_report: ArtifactPinV1 | None
    original_result_pins: tuple[ArtifactPinV1, ...]
    rerun_result_pins: tuple[ArtifactPinV1, ...]
    root_run: ArtifactPinV1
    parent_run: ArtifactPinV1
    change_kind: Literal["TARGET_CORRECTION", "SEMANTIC_SUCCESSOR", "BOTH_EXPLICIT"]
    handoff_step: ArtifactPinV1


class MeasuredComparisonEvaluationV1(_ComparisonContract):
    state: Literal["MEASURED"]
    evaluation_pin: ArtifactPinV1
    verdict: Literal["IMPROVED", "REGRESSED", "CHANGED", "INCONCLUSIVE"]


class UnmeasuredComparisonEvaluationV1(_ComparisonContract):
    state: Literal["UNMEASURED"]
    reason_code: Literal[
        "EVALUATION_NOT_REQUESTED", "EVALUATION_NOT_AVAILABLE", "INCOMPATIBLE_SUBJECT"
    ]


ComparisonEvaluationBindingV1 = Annotated[
    MeasuredComparisonEvaluationV1 | UnmeasuredComparisonEvaluationV1,
    Field(discriminator="state"),
]


class CompatibleComparisonV1(_ComparisonContract):
    state: Literal["COMPATIBLE"]
    evidence: tuple[ArtifactPinV1, ...]


class IncompatibleComparisonV1(_ComparisonContract):
    state: Literal["INCOMPATIBLE"]
    code: Literal[
        "INCOMPATIBLE_WORLD",
        "INCOMPATIBLE_LINEAGE",
        "INCOMPATIBLE_TYPE",
        "INCOMPATIBLE_SCOPE",
        "INCOMPATIBLE_SOURCE_BASIS",
    ]
    evidence: tuple[ArtifactPinV1, ...]


ComparisonCompatibilityV1 = Annotated[
    CompatibleComparisonV1 | IncompatibleComparisonV1,
    Field(discriminator="state"),
]


ComparisonSubjectV1 = Annotated[
    BuildCandidatePairV1
    | WorldSnapshotPairV1
    | ReasoningVersionPairV1
    | RuntimeRerunPairV1,
    Field(discriminator="pair_kind"),
]


class _Projection(_ComparisonContract):
    @model_validator(mode="after")
    def verify_identity_and_scope(self):
        if (self.organization_id, self.workspace_id) != (
            self.subject.organization_id,
            self.subject.workspace_id,
        ):
            raise ValueError("projection scope must match subject")
        digest = content_hash(
            self.model_dump(mode="json", exclude={"projection_ref", "projection_hash"})
        )
        if (
            self.projection_hash != "sha256:" + digest
            or self.projection_ref != "runtime-comparison-projection:" + digest
        ):
            raise ValueError("comparison projection identity mismatch")
        return self


class BuildCandidateDimensionsV1(_ComparisonContract):
    candidate_delta_pins: tuple[ArtifactPinV1, ...]
    competency_metrics: tuple[RuntimeRepairMetricResultV1, ...]
    validation_metrics: tuple[RuntimeRepairMetricResultV1, ...]
    evidence_fidelity_metrics: tuple[RuntimeRepairMetricResultV1, ...]

    @model_validator(mode="after")
    def metrics_match_dimension(self):
        for field, dimension in (
            ("competency_metrics", "COMPETENCY"),
            ("validation_metrics", "VALIDATION"),
            ("evidence_fidelity_metrics", "EVIDENCE_FIDELITY"),
        ):
            if any(metric.dimension != dimension for metric in getattr(self, field)):
                raise ValueError("metric must remain in its declared dimension")
        return self


class WorldSnapshotDimensionsV1(_ComparisonContract):
    added_fact_pins: tuple[ArtifactPinV1, ...]
    removed_fact_pins: tuple[ArtifactPinV1, ...]
    changed_fact_pairs: tuple[tuple[ArtifactPinV1, ArtifactPinV1], ...]
    added_relation_pins: tuple[ArtifactPinV1, ...]
    removed_relation_pins: tuple[ArtifactPinV1, ...]
    changed_relation_pairs: tuple[tuple[ArtifactPinV1, ArtifactPinV1], ...]
    baseline_interval_pin: ArtifactPinV1
    candidate_interval_pin: ArtifactPinV1


class ReasoningVersionDimensionsV1(_ComparisonContract):
    admitted_operation_pairs: tuple[
        tuple[ArtifactPinV1 | None, ArtifactPinV1 | None], ...
    ]
    active_partition_pairs: tuple[
        tuple[ArtifactPinV1 | None, ArtifactPinV1 | None], ...
    ]
    deactivated_partition_pairs: tuple[
        tuple[ArtifactPinV1 | None, ArtifactPinV1 | None], ...
    ]
    derived_partition_pairs: tuple[
        tuple[ArtifactPinV1 | None, ArtifactPinV1 | None], ...
    ]
    justification_pairs: tuple[tuple[ArtifactPinV1 | None, ArtifactPinV1 | None], ...]
    evidence_pairs: tuple[tuple[ArtifactPinV1 | None, ArtifactPinV1 | None], ...]


class RuntimeRerunDimensionsV1(_ComparisonContract):
    result_decision_pairs: tuple[tuple[ArtifactPinV1 | None, ArtifactPinV1 | None], ...]
    exception_pairs: tuple[tuple[ArtifactPinV1 | None, ArtifactPinV1 | None], ...]
    unknown_pairs: tuple[tuple[ArtifactPinV1 | None, ArtifactPinV1 | None], ...]
    evidence_pairs: tuple[tuple[ArtifactPinV1 | None, ArtifactPinV1 | None], ...]
    target_pair: tuple[ArtifactPinV1, ArtifactPinV1]
    epoch_pair: tuple[ArtifactPinV1, ArtifactPinV1]


class BuildCandidateComparisonProjectionV1(_Projection):
    schema_version: Literal["build_candidate_comparison_projection.v1"]
    projection_kind: Literal["BUILD_CANDIDATES"]
    projection_ref: _Id
    projection_hash: _Id
    organization_id: _Id
    workspace_id: _Id
    subject: BuildCandidatePairV1
    dimensions: BuildCandidateDimensionsV1
    compatibility: ComparisonCompatibilityV1
    evaluation: ComparisonEvaluationBindingV1
    navigation_anchors: tuple[_Id, ...]
    origin_handoff_pin: ArtifactPinV1
    projected_at: _UtcTime


class WorldSnapshotComparisonProjectionV1(_Projection):
    schema_version: Literal["world_snapshot_comparison_projection.v1"]
    projection_kind: Literal["WORLD_SNAPSHOTS"]
    projection_ref: _Id
    projection_hash: _Id
    organization_id: _Id
    workspace_id: _Id
    subject: WorldSnapshotPairV1
    dimensions: WorldSnapshotDimensionsV1
    compatibility: ComparisonCompatibilityV1
    evaluation: ComparisonEvaluationBindingV1
    navigation_anchors: tuple[_Id, ...]
    origin_handoff_pin: ArtifactPinV1
    projected_at: _UtcTime


class ReasoningVersionComparisonProjectionV1(_Projection):
    schema_version: Literal["reasoning_version_comparison_projection.v1"]
    projection_kind: Literal["REASONING_VERSIONS"]
    projection_ref: _Id
    projection_hash: _Id
    organization_id: _Id
    workspace_id: _Id
    subject: ReasoningVersionPairV1
    dimensions: ReasoningVersionDimensionsV1
    compatibility: ComparisonCompatibilityV1
    evaluation: ComparisonEvaluationBindingV1
    navigation_anchors: tuple[_Id, ...]
    origin_handoff_pin: ArtifactPinV1
    projected_at: _UtcTime


class RuntimeRerunComparisonProjectionV1(_Projection):
    schema_version: Literal["runtime_rerun_comparison_projection.v1"]
    projection_kind: Literal["RUNTIME_RERUNS"]
    projection_ref: _Id
    projection_hash: _Id
    organization_id: _Id
    workspace_id: _Id
    subject: RuntimeRerunPairV1
    dimensions: RuntimeRerunDimensionsV1
    compatibility: ComparisonCompatibilityV1
    evaluation: ComparisonEvaluationBindingV1
    navigation_anchors: tuple[_Id, ...]
    origin_handoff_pin: ArtifactPinV1
    projected_at: _UtcTime

    @model_validator(mode="after")
    def verify_ordered_sides_and_verdict(self):
        if self.dimensions.target_pair != (
            self.subject.original_target,
            self.subject.rerun_target,
        ):
            raise ValueError("target dimensions must preserve subject side order")
        if self.dimensions.epoch_pair != (
            self.subject.original_epoch,
            self.subject.rerun_epoch,
        ):
            raise ValueError("epoch dimensions must preserve subject side order")
        if (
            self.subject.change_kind == "TARGET_CORRECTION"
            and isinstance(self.evaluation, MeasuredComparisonEvaluationV1)
            and self.evaluation.verdict == "IMPROVED"
        ):
            raise ValueError(
                "target correction alone cannot establish semantic improvement"
            )
        return self


TypedComparisonProjectionV1 = Annotated[
    BuildCandidateComparisonProjectionV1
    | WorldSnapshotComparisonProjectionV1
    | ReasoningVersionComparisonProjectionV1
    | RuntimeRerunComparisonProjectionV1,
    Field(discriminator="projection_kind"),
]
