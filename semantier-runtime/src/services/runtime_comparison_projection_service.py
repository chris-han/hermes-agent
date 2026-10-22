"""Pure, deterministic comparison projection owner for P5-03."""

from __future__ import annotations

from pydantic import TypeAdapter

from contracts.ontology_building import (
    ArtifactPinV1,
    CandidateGraphDiffProjectionV1,
    ExploratoryGraphSnapshotV1,
)
from contracts.runtime_comparison_projection import (
    BuildCandidateDimensionsV1,
    BuildCandidatePairV1,
    BuildCandidateComparisonProjectionV1,
    ComparisonSubjectV1,
    CompatibleComparisonV1,
    DecisionStructureProjectionV1,
    MeasuredComparisonEvaluationV1,
    ReasoningVersionDimensionsV1,
    ReasoningVersionPairV1,
    ReasoningVersionComparisonProjectionV1,
    RuntimeRerunDimensionsV1,
    RuntimeRerunPairV1,
    RuntimeRerunComparisonProjectionV1,
    UnmeasuredComparisonEvaluationV1,
    VerifiedComparisonInputsV1,
    WorldSnapshotDimensionsV1,
    WorldSnapshotPairV1,
    WorldSnapshotComparisonProjectionV1,
)
from eos.artifact_hashing import content_hash


_DECISION_STRUCTURE_NODE_KINDS = frozenset({
    "fact",
    "observation",
    "evidence",
    "rule",
    "constraint",
    "scope",
    "qualification",
    "exception",
    "precedence",
    "authority",
    "policy",
    "temporal",
    "temporal_condition",
    "entity",
    "role",
    "entity_distinction",
    "decision",
    "proposed_action",
    "outcome",
    "assessment",
})

_DECISION_STRUCTURE_FAMILIES = {
    "rule": "RULE",
    "constraint": "CONSTRAINT",
    "scope": "SCOPE",
    "qualification": "QUALIFICATION",
    "exception": "EXCEPTION",
    "precedence": "PRECEDENCE",
    "temporal": "TEMPORAL",
    "temporal_condition": "TEMPORAL",
    "evidence": "EVIDENCE_BINDING",
    "entity": "ENTITY_DISTINCTION",
    "role": "ENTITY_DISTINCTION",
    "entity_distinction": "ENTITY_DISTINCTION",
}


class RuntimeComparisonProjectionService:
    """Build typed comparison projections and bounded decision-structure views."""

    def project_decision_structure(
        self,
        *,
        snapshot: ExploratoryGraphSnapshotV1,
        context_pins: tuple[ArtifactPinV1, ...] = (),
    ) -> DecisionStructureProjectionV1:
        selected_nodes = tuple(
            node
            for node in snapshot.nodes
            if node.node_kind.strip().lower() in _DECISION_STRUCTURE_NODE_KINDS
        )
        selected_node_ids = {node.node_id for node in selected_nodes}
        selected_edges = tuple(
            edge
            for edge in snapshot.edges
            if edge.source_node_id in selected_node_ids
            and edge.target_node_id in selected_node_ids
        )
        evidence_refs = tuple(
            sorted(
                {
                    ref
                    for item in (*selected_nodes, *selected_edges)
                    for ref in item.evidence_refs
                }
            )
        )
        families = tuple(
            sorted(
                {
                    _DECISION_STRUCTURE_FAMILIES[kind]
                    for kind in (
                        node.node_kind.strip().lower() for node in selected_nodes
                    )
                    if kind in _DECISION_STRUCTURE_FAMILIES
                }
            )
        )
        payload = {
            "schema_version": "decision_structure_projection.v1",
            "organization_id": snapshot.organization_id,
            "workspace_id": snapshot.workspace_id,
            "source_snapshot": ArtifactPinV1(
                artifact_kind="exploratory_graph_snapshot",
                artifact_ref=snapshot.snapshot_ref,
                artifact_hash=snapshot.snapshot_hash,
            ),
            "context_pins": context_pins,
            "node_ids": tuple(sorted(selected_node_ids)),
            "edge_ids": tuple(sorted(edge.edge_id for edge in selected_edges)),
            "evidence_refs": evidence_refs,
            "semantic_families": families,
        }
        digest = content_hash(TypeAdapter(dict[str, object]).dump_python(payload, mode="json"))
        return DecisionStructureProjectionV1(
            **payload,
            projection_ref="decision-structure-projection:" + digest,
            projection_hash="sha256:" + digest,
        )

    def validate_structural_learning_delta(
        self,
        *,
        failure_class: str,
        baseline: DecisionStructureProjectionV1,
        successor: DecisionStructureProjectionV1,
        diff: CandidateGraphDiffProjectionV1,
    ) -> CandidateGraphDiffProjectionV1:
        if failure_class != "STRUCTURE":
            raise ValueError("only STRUCTURE failures may enter structural learning")
        if (baseline.organization_id, baseline.workspace_id) != (
            successor.organization_id,
            successor.workspace_id,
        ):
            raise ValueError("decision structure projections must share scope")
        if (
            diff.organization_id,
            diff.workspace_id,
            diff.baseline_snapshot_ref,
            diff.baseline_snapshot_hash,
            diff.successor_snapshot_ref,
            diff.successor_snapshot_hash,
        ) != (
            baseline.organization_id,
            baseline.workspace_id,
            baseline.source_snapshot.artifact_ref,
            baseline.source_snapshot.artifact_hash,
            successor.source_snapshot.artifact_ref,
            successor.source_snapshot.artifact_hash,
        ):
            raise ValueError("structural diff must bind exact decision structure snapshots")
        changed_ids = set(diff.added_node_ids) | set(diff.removed_node_ids) | set(diff.added_edge_ids) | set(diff.removed_edge_ids)
        projected_ids = set(baseline.node_ids) | set(successor.node_ids) | set(baseline.edge_ids) | set(successor.edge_ids)
        if not changed_ids or not changed_ids.intersection(projected_ids):
            raise ValueError("structural learning requires a decision-structure graph delta")
        return diff

    def project(
        self,
        *,
        subject: ComparisonSubjectV1,
        inputs: VerifiedComparisonInputsV1,
        evaluation_pin=None,
    ):
        if (subject.organization_id, subject.workspace_id) != (
            inputs.organization_id,
            inputs.workspace_id,
        ):
            raise ValueError("comparison inputs scope must match subject scope")
        subject_hash = content_hash(subject.model_dump(mode="json"))
        if inputs.subject_payload_hash not in {subject_hash, "sha256:" + subject_hash}:
            raise ValueError("comparison inputs subject payload hash mismatch")
        if not inputs.source_times:
            raise ValueError("comparison inputs require source times")

        evaluation = UnmeasuredComparisonEvaluationV1(
            state="UNMEASURED",
            reason_code=(
                "EVALUATION_NOT_AVAILABLE"
                if evaluation_pin is not None
                else "EVALUATION_NOT_REQUESTED"
            ),
        )
        compatibility = CompatibleComparisonV1(
            state="COMPATIBLE", evidence=inputs.compatibility_evidence
        )
        common = {
            "organization_id": subject.organization_id,
            "workspace_id": subject.workspace_id,
            "subject": subject,
            "compatibility": compatibility,
            "evaluation": evaluation,
            "navigation_anchors": inputs.navigation_anchors,
            "origin_handoff_pin": inputs.origin_handoff_pin,
            "projected_at": max(inputs.source_times),
        }
        if isinstance(subject, BuildCandidatePairV1):
            schema = "build_candidate_comparison_projection.v1"
            dimensions = BuildCandidateDimensionsV1(
                candidate_delta_pins=(),
                competency_metrics=(),
                validation_metrics=(),
                evidence_fidelity_metrics=(),
            )
            payload = {
                "schema_version": schema,
                "projection_kind": subject.pair_kind,
                "dimensions": dimensions,
                **common,
            }
            return BuildCandidateComparisonProjectionV1(
                **self._identity(payload),
            )
        if isinstance(subject, WorldSnapshotPairV1):
            schema = "world_snapshot_comparison_projection.v1"
            dimensions = WorldSnapshotDimensionsV1(
                added_fact_pins=(),
                removed_fact_pins=(),
                changed_fact_pairs=(),
                added_relation_pins=(),
                removed_relation_pins=(),
                changed_relation_pairs=(),
                baseline_interval_pin=subject.baseline_valid_time,
                candidate_interval_pin=subject.candidate_valid_time,
            )
            payload = {
                "schema_version": schema,
                "projection_kind": subject.pair_kind,
                "dimensions": dimensions,
                **common,
            }
            return WorldSnapshotComparisonProjectionV1(**self._identity(payload))
        if isinstance(subject, ReasoningVersionPairV1):
            schema = "reasoning_version_comparison_projection.v1"
            dimensions = ReasoningVersionDimensionsV1(
                admitted_operation_pairs=(),
                active_partition_pairs=(),
                deactivated_partition_pairs=(),
                derived_partition_pairs=(),
                justification_pairs=(),
                evidence_pairs=(),
            )
            payload = {
                "schema_version": schema,
                "projection_kind": subject.pair_kind,
                "dimensions": dimensions,
                **common,
            }
            return ReasoningVersionComparisonProjectionV1(**self._identity(payload))
        if isinstance(subject, RuntimeRerunPairV1):
            schema = "runtime_rerun_comparison_projection.v1"
            dimensions = RuntimeRerunDimensionsV1(
                result_decision_pairs=(),
                exception_pairs=(),
                unknown_pairs=(),
                evidence_pairs=(),
                target_pair=(subject.original_target, subject.rerun_target),
                epoch_pair=(subject.original_epoch, subject.rerun_epoch),
            )
            payload = {
                "schema_version": schema,
                "projection_kind": subject.pair_kind,
                "dimensions": dimensions,
                **common,
            }
            return RuntimeRerunComparisonProjectionV1(**self._identity(payload))
        raise TypeError("unsupported comparison subject")

    @staticmethod
    def _identity(payload):
        serialized = TypeAdapter(dict[str, object]).dump_python(payload, mode="json")
        digest = content_hash(serialized)
        return {
            **payload,
            "projection_ref": "runtime-comparison-projection:" + digest,
            "projection_hash": "sha256:" + digest,
        }


__all__ = ["RuntimeComparisonProjectionService"]
