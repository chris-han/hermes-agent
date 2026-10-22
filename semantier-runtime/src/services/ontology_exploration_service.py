"""Bounded, deterministic, non-authoritative ontology exploration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from contracts.ontology_artifact_integrity import (
    ExploratoryGraphSnapshotIntegrityV1,
    create_exploratory_snapshot_integrity,
)
from contracts.ontology_building import (
    ExploratoryGraphEdgeV1,
    ExploratoryGraphNodeV1,
    ExploratoryGraphSnapshotV1,
)
from eos.artifact_hashing import content_hash


@dataclass(frozen=True)
class ExploratoryCompilationResult:
    snapshot: ExploratoryGraphSnapshotV1
    integrity: ExploratoryGraphSnapshotIntegrityV1


class OntologyExplorationService:
    """Compile only a preview graph and its explicit unresolved gaps.

    This service intentionally has no dependency on admission, release, or
    activation services.  Its output type can represent only provisional
    authority, so a caller cannot promote it by changing a request flag.
    """

    def __init__(self, *, compiler_version: str, compiled_at: datetime) -> None:
        if not compiler_version:
            raise ValueError("compiler_version is required")
        if compiled_at.tzinfo is None or compiled_at.utcoffset() != timedelta(0):
            raise ValueError("compiled_at must be timezone-aware UTC")
        self._compiler_version = compiler_version
        self._compiled_at = compiled_at

    def compile(
        self,
        *,
        source: dict[str, Any],
        organization_id: str,
        workspace_id: str,
        build_ref: str,
        source_identity_ref: str,
        source_hash: str,
        canonical_source_ir_ref: str,
        canonical_source_ir_hash: str,
    ) -> ExploratoryCompilationResult:
        sections = source.get("source_sections")
        dimensions = source.get("semantic_dimensions")
        if not isinstance(sections, list) or not sections:
            raise ValueError("readable source_sections are required")
        if not isinstance(dimensions, list) or not dimensions:
            raise ValueError("semantic_dimensions are required for the proving adapter")

        nodes = tuple(
            ExploratoryGraphNodeV1(
                node_id=f"dimension:{item['dimension']}",
                node_kind=str(item["dimension"]),
                label=str(item["dimension"]).replace("_", " ").title(),
                evidence_refs=tuple(str(ref) for ref in item["evidence_refs"]),
            )
            for item in dimensions
        )
        edges = tuple(
            ExploratoryGraphEdgeV1(
                edge_id=f"edge:{nodes[index - 1].node_kind}:{node.node_kind}",
                source_node_id=nodes[index - 1].node_id,
                target_node_id=node.node_id,
                relation_type="semantic_dependency",
                evidence_refs=tuple(
                    sorted(set(nodes[index - 1].evidence_refs + node.evidence_refs))
                ),
            )
            for index, node in enumerate(nodes)
            if index > 0
        )
        unresolved_gaps = ("gap:competency_review", "gap:certification_not_started")
        snapshot_basis = {
            "compiler_version": self._compiler_version,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "build_ref": build_ref,
            "source_identity_ref": source_identity_ref,
            "source_hash": source_hash,
            "canonical_source_ir_ref": canonical_source_ir_ref,
            "canonical_source_ir_hash": canonical_source_ir_hash,
            "nodes": [item.model_dump(mode="json") for item in nodes],
            "edges": [item.model_dump(mode="json") for item in edges],
            "unresolved_gap_ids": unresolved_gaps,
        }
        snapshot_digest = content_hash(snapshot_basis)
        now = self._compiled_at
        snapshot = ExploratoryGraphSnapshotV1(
            snapshot_ref=f"exploratory-snapshot:{snapshot_digest}",
            snapshot_hash=f"sha256:{snapshot_digest}",
            organization_id=organization_id,
            workspace_id=workspace_id,
            build_ref=build_ref,
            source_identity_ref=source_identity_ref,
            source_hash=source_hash,
            canonical_source_ir_ref=canonical_source_ir_ref,
            canonical_source_ir_hash=canonical_source_ir_hash,
            nodes=nodes,
            edges=edges,
            unresolved_gap_ids=unresolved_gaps,
            change_cause="initial_exploration",
            created_at=now,
        )

        integrity = create_exploratory_snapshot_integrity(
            snapshot=snapshot,
            producer_profile="bounded_exploration.v1",
            producer_version=self._compiler_version,
            recorded_at=now,
        )
        return ExploratoryCompilationResult(
            snapshot=snapshot, integrity=integrity
        )


__all__ = ["ExploratoryCompilationResult", "OntologyExplorationService"]
