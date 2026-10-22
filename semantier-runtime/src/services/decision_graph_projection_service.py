"""MVL graph-native projection adapter over the existing DecisionLocalProjectionV2."""

from __future__ import annotations

from dataclasses import dataclass

from contracts.decision_graph_projection import (
    DecisionEvidenceViewV1,
    DecisionGraphProjectionReceiptV1,
    DecisionGraphProjectionV1,
    ProjectionSufficiencyProofV1,
)
from contracts.decision_local_projection_v2 import DecisionLocalProjectionV2
from contracts.typed_decision_model import TypedDecisionFrontierV1


@dataclass(frozen=True)
class DecisionGraphProjectionBundleV1:
    projection: DecisionGraphProjectionV1
    sufficiency: ProjectionSufficiencyProofV1
    evidence_view: DecisionEvidenceViewV1
    receipt: DecisionGraphProjectionReceiptV1


class DecisionGraphProjectionService:
    """Derive a non-authoritative ref-view without rewriting the DLP compiler."""

    @staticmethod
    def _ordered_text(values) -> tuple[str, ...]:
        return tuple(sorted((str(value) for value in values), key=lambda item: item.encode("utf-8")))

    @staticmethod
    def _missing_dependencies(dlp: DecisionLocalProjectionV2) -> tuple[str, ...]:
        return tuple(sorted(
            (
                str(item.dependency_id)
                for item in dlp.legacy.material_dependency_resolutions
                if item.status != "SATISFIED"
            ),
            key=lambda item: item.encode("utf-8"),
        ))

    @staticmethod
    def _structural_missing(dlp: DecisionLocalProjectionV2) -> tuple[str, ...]:
        context = dlp.context
        working = dlp.working_context
        existing_nodes = {item.node_id for item in context.nodes}
        existing_edges = {item.edge_id for item in context.edges}
        existing_assertions = {item.assertion_id for item in context.assertions}
        missing: list[str] = []
        missing.extend(f"node:{item}" for item in working.selected_node_ids if item not in existing_nodes)
        missing.extend(f"edge:{item}" for item in working.selected_edge_ids if item not in existing_edges)
        missing.extend(
            f"assertion:{item}"
            for item in working.selected_assertion_ids
            if item not in existing_assertions
        )
        return tuple(sorted(missing, key=lambda item: item.encode("utf-8")))

    def compile(
        self,
        *,
        dlp: DecisionLocalProjectionV2,
        frontier: TypedDecisionFrontierV1,
    ) -> DecisionGraphProjectionBundleV1:
        if not isinstance(dlp, DecisionLocalProjectionV2):
            raise TypeError("DGP requires DecisionLocalProjectionV2")
        if not isinstance(frontier, TypedDecisionFrontierV1):
            raise TypeError("DGP requires TypedDecisionFrontierV1")
        frontier.verify_against_projection(dlp)

        working = dlp.working_context
        scope = (dlp.legacy.organization_id, dlp.legacy.workspace_id)
        if (working.organization_id, working.workspace_id) != scope:
            raise ValueError("DGP_SCOPE_MISMATCH")

        missing = self._ordered_text((
            *self._missing_dependencies(dlp),
            *self._structural_missing(dlp),
        ))
        evidence = tuple(sorted(
            dlp.legacy.evidence,
            key=lambda pin: (
                pin.artifact_kind.encode("utf-8"),
                pin.artifact_ref.encode("utf-8"),
                pin.artifact_hash.encode("utf-8"),
            ),
        ))
        authority = tuple(sorted(
            dlp.decision_contract.deterministic_constraints,
            key=lambda pin: (
                pin.artifact_kind.encode("utf-8"),
                pin.artifact_ref.encode("utf-8"),
                pin.artifact_hash.encode("utf-8"),
            ),
        ))
        conflicts = self._ordered_text(dlp.context.formation.conflicts)

        proof = ProjectionSufficiencyProofV1.create(
            status="INSUFFICIENT" if missing else "SUFFICIENT",
            missing_dependency_ids=missing,
            frontier_ref=frontier.frontier_ref,
            frontier_hash=frontier.frontier_hash,
            dlp_ref=dlp.projection_ref,
            dlp_hash=dlp.projection_hash,
            required_node_ids=self._ordered_text(working.selected_node_ids),
            required_edge_ids=self._ordered_text(working.selected_edge_ids),
            required_assertion_ids=self._ordered_text(working.selected_assertion_ids),
            evidence=evidence,
            authority_bindings=authority,
            unresolved_conflict_ids=conflicts,
        )

        projection = DecisionGraphProjectionV1.create(
            organization_id=scope[0],
            workspace_id=scope[1],
            source_dlp_ref=dlp.projection_ref,
            source_dlp_hash=dlp.projection_hash,
            source_model_input_hash=dlp.model_input_hash,
            frontier_ref=frontier.frontier_ref,
            frontier_hash=frontier.frontier_hash,
            canonical_context=dlp.context.pin(),
            working_context=working.pin(),
            ontology=dlp.ontology.pin(),
            node_ids=self._ordered_text(working.selected_node_ids),
            edge_ids=self._ordered_text(working.selected_edge_ids),
            assertion_ids=self._ordered_text(working.selected_assertion_ids),
            evidence=evidence,
            authority_bindings=authority,
            unresolved_conflict_ids=conflicts,
            world_time_basis=dlp.legacy.world_time_basis,
            sufficiency_proof=proof.pin(),
        )

        handles = tuple(
            (f"e{index}", pin)
            for index, pin in enumerate(evidence)
        )
        view = DecisionEvidenceViewV1.create(
            decision_graph_projection=projection.pin(),
            source_dlp_ref=dlp.projection_ref,
            source_dlp_hash=dlp.projection_hash,
            model_input_hash=dlp.model_input_hash,
            evidence_handles=handles,
        )
        receipt = DecisionGraphProjectionReceiptV1.create(
            projection=projection.pin(),
            sufficiency_proof=proof.pin(),
            evidence_view=view.pin(),
            source_dlp_ref=dlp.projection_ref,
            source_dlp_hash=dlp.projection_hash,
            frontier_ref=frontier.frontier_ref,
            frontier_hash=frontier.frontier_hash,
        )
        return DecisionGraphProjectionBundleV1(
            projection=projection,
            sufficiency=proof,
            evidence_view=view,
            receipt=receipt,
        )

    def verify(
        self,
        *,
        dlp: DecisionLocalProjectionV2,
        frontier: TypedDecisionFrontierV1,
        bundle: DecisionGraphProjectionBundleV1,
    ) -> None:
        """Replay verification is exact derivation comparison; it performs no retrieval."""
        expected = self.compile(dlp=dlp, frontier=frontier)
        if bundle != expected:
            raise ValueError("DGP_DERIVATION_MISMATCH")
