"""Exact candidate-lineage comparison projections."""

from __future__ import annotations

from contracts.ontology_building import (
    CandidateGraphDiffProjectionV1,
    CandidateGraphLineageEdgeV1,
    CandidateGraphLineageNodeV1,
    CandidateGraphLineageProjectionV1,
    ExploratoryGraphSnapshotV1,
)
from contracts.artifact_pin import ArtifactPinV1
from contracts.ontology_quality import QualityEvaluationRunV1, QualityRequirementDeltaV1
from eos.artifact_hashing import content_hash


class OntologyLineageService:
    def project(
        self, *, snapshots: tuple[ExploratoryGraphSnapshotV1, ...]
    ) -> CandidateGraphLineageProjectionV1:
        if not snapshots:
            raise ValueError("lineage requires at least one snapshot")
        tenants = {(item.organization_id, item.workspace_id) for item in snapshots}
        if len(tenants) != 1:
            raise PermissionError("candidate lineage crosses governed tenant scope")
        by_identity = {(item.snapshot_ref, item.snapshot_hash): item for item in snapshots}
        edges: list[CandidateGraphLineageEdgeV1] = []
        for snapshot in snapshots:
            if snapshot.predecessor_snapshot_ref is None:
                continue
            predecessor_identity = (
                snapshot.predecessor_snapshot_ref,
                snapshot.predecessor_snapshot_hash,
            )
            if predecessor_identity not in by_identity:
                raise ValueError("lineage predecessor is missing from projection")
            edges.append(
                CandidateGraphLineageEdgeV1(
                    predecessor_snapshot_ref=snapshot.predecessor_snapshot_ref,
                    predecessor_snapshot_hash=snapshot.predecessor_snapshot_hash,
                    successor_snapshot_ref=snapshot.snapshot_ref,
                    successor_snapshot_hash=snapshot.snapshot_hash,
                    branch_ref=snapshot.build_ref,
                    change_cause=snapshot.change_cause,
                    triggering_ref=snapshot.triggering_ref,
                )
            )
        ordered = tuple(sorted(snapshots, key=lambda item: (item.created_at, item.snapshot_ref)))
        nodes = tuple(
            CandidateGraphLineageNodeV1(
                snapshot_ref=item.snapshot_ref,
                snapshot_hash=item.snapshot_hash,
                build_ref=item.build_ref,
                authority_state=item.authority_state,
                completeness_state=item.completeness_state,
            )
            for item in ordered
        )
        organization_id, workspace_id = next(iter(tenants))
        basis = {
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "nodes": tuple(item.model_dump(mode="json") for item in nodes),
            "edges": tuple(item.model_dump(mode="json") for item in edges),
        }
        digest = content_hash(basis)
        return CandidateGraphLineageProjectionV1(
            lineage_ref=f"candidate-graph-lineage:{digest}",
            lineage_hash=f"sha256:{digest}",
            organization_id=organization_id,
            workspace_id=workspace_id,
            nodes=nodes,
            edges=tuple(edges),
            created_at=max(item.created_at for item in snapshots),
        )

    def compare(
        self,
        *,
        baseline: ExploratoryGraphSnapshotV1,
        successor: ExploratoryGraphSnapshotV1,
        baseline_quality: QualityEvaluationRunV1 | None = None,
        successor_quality: QualityEvaluationRunV1 | None = None,
    ) -> CandidateGraphDiffProjectionV1:
        if (baseline.organization_id, baseline.workspace_id) != (
            successor.organization_id,
            successor.workspace_id,
        ):
            raise PermissionError("candidate comparison crosses governed tenant scope")
        if (
            successor.predecessor_snapshot_ref != baseline.snapshot_ref
            or successor.predecessor_snapshot_hash != baseline.snapshot_hash
        ):
            raise ValueError("successor does not pin the selected baseline predecessor")

        baseline_nodes = {item.node_id for item in baseline.nodes}
        successor_nodes = {item.node_id for item in successor.nodes}
        baseline_edges = {item.edge_id for item in baseline.edges}
        successor_edges = {item.edge_id for item in successor.edges}
        quality_changes = []
        verdict = "INCONCLUSIVE"
        verdict_evidence_refs = ()
        for snapshot, run in ((baseline, baseline_quality), (successor, successor_quality)):
            if run is None:
                continue
            QualityEvaluationRunV1.model_validate(run)
            if (run.organization_id, run.workspace_id) != (snapshot.organization_id, snapshot.workspace_id):
                raise PermissionError("quality comparison crosses governed tenant scope")
            if run.subject != ArtifactPinV1(artifact_kind="exploratory_graph_snapshot",
                artifact_ref=snapshot.snapshot_ref, artifact_hash=snapshot.snapshot_hash):
                raise ValueError("quality evaluation does not pin the selected snapshot")
        if baseline_quality is not None and successor_quality is not None:
            if baseline_quality.quality_profile != successor_quality.quality_profile:
                raise ValueError("quality comparison requires the same frozen profile")
            verdict = "CHANGED"  # Quality status alone cannot prove behavioral improvement.
        if baseline_quality is not None or successor_quality is not None:
            before = {item.requirement_id: item for item in baseline_quality.results} if baseline_quality else {}
            after = {item.requirement_id: item for item in successor_quality.results} if successor_quality else {}
            for name in sorted(before.keys() | after.keys()):
                left, right = before.get(name), after.get(name)
                quality_changes.append(QualityRequirementDeltaV1.create(requirement_id=name, gate=(left or right).gate,
                    before_run=baseline_quality.pin() if baseline_quality else None,
                    after_run=successor_quality.pin() if successor_quality else None,
                    before_result=left.pin() if left else None, after_result=right.pin() if right else None,
                    before_status=left.status if left else "NOT_EVALUATED",
                    after_status=right.status if right else "NOT_EVALUATED"))
            verdict_evidence_refs = tuple(run.run_ref for run in (baseline_quality, successor_quality) if run is not None)
        basis = {
            "organization_id": baseline.organization_id,
            "workspace_id": baseline.workspace_id,
            "baseline_snapshot_ref": baseline.snapshot_ref,
            "baseline_snapshot_hash": baseline.snapshot_hash,
            "successor_snapshot_ref": successor.snapshot_ref,
            "successor_snapshot_hash": successor.snapshot_hash,
            "added_node_ids": tuple(sorted(successor_nodes - baseline_nodes)),
            "removed_node_ids": tuple(sorted(baseline_nodes - successor_nodes)),
            "added_edge_ids": tuple(sorted(successor_edges - baseline_edges)),
            "removed_edge_ids": tuple(sorted(baseline_edges - successor_edges)),
            "quality_changes": [item.model_dump(mode="json") for item in quality_changes],
            "comparison_verdict": verdict,
            "verdict_evidence_refs": verdict_evidence_refs,
            "change_cause": successor.change_cause,
            "triggering_ref": successor.triggering_ref,
        }
        digest = content_hash(basis)
        return CandidateGraphDiffProjectionV1(
            diff_ref=f"candidate-graph-diff:{digest}",
            diff_hash=f"sha256:{digest}",
            created_at=successor.created_at,
            **basis,
        )


__all__ = ["OntologyLineageService"]
