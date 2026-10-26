"""Derived Tender epistemic-lineage views over existing governed owners.

This module creates no semantic authority, persistence owner, or sixth kernel object.
It derives one deterministic dependency basis from DecisionLocalProjectionV2 and
DecisionRecord so explanation, impact, recomputation, debugging, and replay can
share the same material edges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Iterable

from contracts.decision_local_projection_v2 import DecisionLocalProjectionV2
from contracts.decision_verification import DecisionVerificationV1
from contracts.governed_delta import (
    AffectedArtifactV1,
    DeltaEnvelopeV1,
    DeltaLens,
    ImpactProjectionV1,
    ImpactPopulationScopeV2,
    ImpactRelation,
    ImpactStatus,
)
from contracts.semantic_kernel import DecisionRecord
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_identity


Layer = Literal["EVIDENCE", "MEANING", "STRUCTURE", "DECISION"]
Relation = Literal[
    "SUPPORTS_MEANING",
    "BOUNDS_STRUCTURE",
    "SATISFIES_MATERIAL_DEPENDENCY",
    "VERIFIES_DECISION",
]


def _ordered(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values), key=lambda item: item.encode("utf-8")))


@dataclass(frozen=True)
class EpistemicLineageEdgeV1:
    source_layer: Layer
    source_ref: str
    target_layer: Layer
    target_ref: str
    relation: Relation
    basis_refs: tuple[str, ...]
    edge_ref: str

    @classmethod
    def create(
        cls,
        *,
        source_layer: Layer,
        source_ref: str,
        target_layer: Layer,
        target_ref: str,
        relation: Relation,
        basis_refs: Iterable[str],
    ) -> "EpistemicLineageEdgeV1":
        basis = _ordered(basis_refs)
        payload = {
            "source_layer": source_layer,
            "source_ref": source_ref,
            "target_layer": target_layer,
            "target_ref": target_ref,
            "relation": relation,
            "basis_refs": basis,
        }
        return cls(
            source_layer=source_layer,
            source_ref=source_ref,
            target_layer=target_layer,
            target_ref=target_ref,
            relation=relation,
            basis_refs=basis,
            edge_ref="epistemic-lineage-edge:" + content_hash(payload),
        )


@dataclass(frozen=True)
class TenderEpistemicLineageViewV1:
    projection_ref: str
    projection_hash: str
    decision_ref: str
    decision_hash: str
    decision_result_code: str
    evidence_refs: tuple[str, ...]
    predicate_refs: tuple[str, ...]
    assertion_refs: tuple[str, ...]
    edges: tuple[EpistemicLineageEdgeV1, ...]
    organization_id: str | None = None
    workspace_id: str | None = None

    @property
    def material_basis_refs(self) -> tuple[str, ...]:
        return _ordered(edge.edge_ref for edge in self.edges)


@dataclass(frozen=True)
class TenderEvidenceImpactV1:
    population_scope: ImpactPopulationScopeV2
    evidence_ref: str
    assertion_refs: tuple[str, ...]
    predicate_refs: tuple[str, ...]
    decision_refs: tuple[str, ...]
    basis_refs: tuple[str, ...]


@dataclass(frozen=True)
class TenderBoundaryDiagnosisV1:
    first_invalid_layer: Layer | None
    expected_basis_refs: tuple[str, ...]
    observed_basis_refs: tuple[str, ...]


class TenderEpistemicLineageService:
    """Pure derived traversal over already-authoritative Tender artifacts."""

    @staticmethod
    def _verification(decision: DecisionRecord) -> DecisionVerificationV1:
        invocations = tuple(
            item
            for item in decision.reasoning.invocations
            if item.strategy == "tender_interpreted_predicate_verification"
        )
        if len(invocations) != 1 or not invocations[0].proof_steps:
            raise ValueError("TENDER_DECISION_VERIFICATION_BASIS_UNAVAILABLE")
        try:
            verification = DecisionVerificationV1.model_validate_json(
                invocations[0].proof_steps[0]
            )
        except Exception as exc:  # pydantic supplies the detailed contract error
            raise ValueError("TENDER_DECISION_VERIFICATION_BASIS_INVALID") from exc
        if verification.status != "VERIFIED" or verification.trace is None:
            raise ValueError("TENDER_DECISION_NOT_VERIFIED")
        return verification

    def compile(
        self,
        *,
        projection: DecisionLocalProjectionV2,
        decision: DecisionRecord,
    ) -> TenderEpistemicLineageViewV1:
        verification = self._verification(decision)
        if decision.context != projection.context.pin():
            raise ValueError("LINEAGE_DECISION_CONTEXT_MISMATCH")
        if decision.ontology != projection.ontology.pin():
            raise ValueError("LINEAGE_DECISION_ONTOLOGY_MISMATCH")
        if verification.projection is None or (
            verification.projection.artifact_ref,
            verification.projection.artifact_hash,
        ) != (projection.projection_ref, projection.projection_hash):
            raise ValueError("LINEAGE_VERIFIED_PROJECTION_MISMATCH")

        decision_evidence = {
            (pin.artifact_ref, pin.artifact_hash) for pin in decision.evidence
        }
        projection_evidence = {
            (pin.artifact_ref, pin.artifact_hash) for pin in projection.legacy.evidence
        }
        if decision_evidence != projection_evidence:
            raise ValueError("LINEAGE_DECISION_EVIDENCE_MISMATCH")

        predicate_id = verification.trace.predicate_id
        semantic_predicates = {
            item.predicate_id for item in projection.legacy.semantic_predicates
        }
        if predicate_id not in semantic_predicates:
            raise ValueError("LINEAGE_PREDICATE_NOT_IN_PROJECTION")

        evidence_ref_by_kernel = {
            item.artifact_ref: evidence_ref_identity(
                organization_id=item.organization_id,
                workspace_id=item.workspace_id,
                source_identity_ref=item.source_ref,
                selector_hash=item.selector_hash,
                evidence_role=item.evidence_role,
            )
            for item in projection.evidence
        }
        if set(evidence_ref_by_kernel) != {
            pin.artifact_ref for pin in projection.legacy.evidence
        }:
            raise ValueError("LINEAGE_KERNEL_EVIDENCE_CLOSURE_MISMATCH")

        edges: list[EpistemicLineageEdgeV1] = []
        assertion_ids: list[str] = []
        evidence_refs: list[str] = []

        for item in projection.legacy.evidence_projection:
            assertion_ids.append(item.assertion_id)
            for pin in item.evidence:
                source_evidence_ref = evidence_ref_by_kernel[pin.artifact_ref]
                evidence_refs.append(source_evidence_ref)
                edges.append(
                    EpistemicLineageEdgeV1.create(
                        source_layer="EVIDENCE",
                        source_ref=source_evidence_ref,
                        target_layer="MEANING",
                        target_ref=item.assertion_id,
                        relation="SUPPORTS_MEANING",
                        basis_refs=(
                            source_evidence_ref,
                            pin.artifact_ref,
                            projection.context.artifact_ref,
                            projection.projection_ref,
                            item.assertion_id,
                        ),
                    )
                )
            edges.append(
                EpistemicLineageEdgeV1.create(
                    source_layer="MEANING",
                    source_ref=item.assertion_id,
                    target_layer="STRUCTURE",
                    target_ref=predicate_id,
                    relation="BOUNDS_STRUCTURE",
                    basis_refs=(
                        item.assertion_id,
                        projection.projection_ref,
                        projection.working_context.value_ref,
                        predicate_id,
                    ),
                )
            )

        for resolution in projection.legacy.material_dependency_resolutions:
            if resolution.predicate_id != predicate_id or resolution.status != "SATISFIED":
                continue
            for pin in resolution.matched_evidence:
                source_evidence_ref = evidence_ref_by_kernel[pin.artifact_ref]
                evidence_refs.append(source_evidence_ref)
                edges.append(
                    EpistemicLineageEdgeV1.create(
                        source_layer="EVIDENCE",
                        source_ref=source_evidence_ref,
                        target_layer="STRUCTURE",
                        target_ref=predicate_id,
                        relation="SATISFIES_MATERIAL_DEPENDENCY",
                        basis_refs=(
                            resolution.dependency_id,
                            source_evidence_ref,
                            pin.artifact_ref,
                            predicate_id,
                            projection.projection_ref,
                        ),
                    )
                )

        edges.append(
            EpistemicLineageEdgeV1.create(
                source_layer="STRUCTURE",
                source_ref=predicate_id,
                target_layer="DECISION",
                target_ref=decision.artifact_ref,
                relation="VERIFIES_DECISION",
                basis_refs=(
                    predicate_id,
                    verification.verification_ref,
                    projection.projection_ref,
                    decision.artifact_ref,
                ),
            )
        )

        ordered_edges = tuple(
            sorted(edges, key=lambda item: item.edge_ref.encode("utf-8"))
        )
        return TenderEpistemicLineageViewV1(
            projection_ref=projection.projection_ref,
            projection_hash=projection.projection_hash,
            decision_ref=decision.artifact_ref,
            decision_hash=decision.artifact_hash,
            decision_result_code=decision.result_code,
            evidence_refs=_ordered(evidence_refs),
            predicate_refs=(predicate_id,),
            assertion_refs=_ordered(assertion_ids),
            edges=ordered_edges,
            organization_id=projection.legacy.organization_id,
            workspace_id=projection.legacy.workspace_id,
        )

    @staticmethod
    def explain(
        view: TenderEpistemicLineageViewV1,
    ) -> tuple[EpistemicLineageEdgeV1, ...]:
        """Return the exact decision explanation path, ordered backwards."""
        rank = {"DECISION": 0, "STRUCTURE": 1, "MEANING": 2, "EVIDENCE": 3}
        return tuple(
            sorted(
                view.edges,
                key=lambda item: (
                    rank[item.target_layer],
                    rank[item.source_layer],
                    item.edge_ref,
                ),
            )
        )

    @staticmethod
    def _population(views):
        scopes = {(v.organization_id, v.workspace_id) for v in views}
        if len(scopes) > 1:
            raise ValueError("LINEAGE_POPULATION_SCOPE_MISMATCH")
        scope = next(iter(scopes), (None, None))
        return ImpactPopulationScopeV2(organization_id=scope[0], workspace_id=scope[1],
            member_pins=tuple(sorted({(v.projection_ref, v.projection_hash, v.decision_ref, v.decision_hash) for v in views})))

    @staticmethod
    def impact(
        *,
        evidence_ref: str,
        views: Iterable[TenderEpistemicLineageViewV1],
    ) -> TenderEvidenceImpactV1:
        views = tuple(views)
        population_scope = TenderEpistemicLineageService._population(views)
        assertions: set[str] = set()
        predicates: set[str] = set()
        decisions: set[str] = set()
        basis: set[str] = set()
        for view in views:
            direct_assertions = {
                edge.target_ref
                for edge in view.edges
                if edge.source_layer == "EVIDENCE"
                and edge.source_ref == evidence_ref
                and edge.target_layer == "MEANING"
            }
            direct_predicates = {
                edge.target_ref
                for edge in view.edges
                if edge.source_layer == "EVIDENCE"
                and edge.source_ref == evidence_ref
                and edge.target_layer == "STRUCTURE"
            }
            bounded_predicates = {
                edge.target_ref
                for edge in view.edges
                if edge.source_layer == "MEANING"
                and edge.source_ref in direct_assertions
                and edge.target_layer == "STRUCTURE"
            }
            reached_predicates = direct_predicates | bounded_predicates
            reached_decisions = {
                edge.target_ref
                for edge in view.edges
                if edge.source_layer == "STRUCTURE"
                and edge.source_ref in reached_predicates
                and edge.target_layer == "DECISION"
            }
            if direct_assertions or reached_predicates or reached_decisions:
                assertions |= direct_assertions
                predicates |= reached_predicates
                decisions |= reached_decisions
                basis |= {
                    edge.edge_ref
                    for edge in view.edges
                    if (
                        edge.source_ref == evidence_ref
                        or edge.source_ref in direct_assertions
                        or edge.source_ref in reached_predicates
                    )
                }
        return TenderEvidenceImpactV1(
            population_scope=population_scope,
            evidence_ref=evidence_ref,
            assertion_refs=_ordered(assertions),
            predicate_refs=_ordered(predicates),
            decision_refs=_ordered(decisions),
            basis_refs=_ordered(basis),
        )

    def recomputation_required(
        self,
        *,
        changed_evidence_refs: Iterable[str],
        views: Iterable[TenderEpistemicLineageViewV1],
    ) -> tuple[str, ...]:
        view_tuple = tuple(views)
        self._population(view_tuple)
        decisions: set[str] = set()
        for evidence_ref in changed_evidence_refs:
            decisions.update(
                self.impact(evidence_ref=evidence_ref, views=view_tuple).decision_refs
            )
        return _ordered(decisions)

    def impact_projection(
        self,
        *,
        delta: DeltaEnvelopeV1,
        views: Iterable[TenderEpistemicLineageViewV1],
    ) -> ImpactProjectionV1:
        evidence_lens = next(item for item in delta.lenses if item.lens is DeltaLens.EVIDENCE)
        changed_refs: set[str] = set()
        for operation in evidence_lens.operations:
            changed_refs.update(operation.evidence_refs)
            if operation.before_ref:
                changed_refs.add(operation.before_ref)
            if operation.after_ref:
                changed_refs.add(operation.after_ref)
        view_tuple = tuple(views)
        impacted = self.recomputation_required(
            changed_evidence_refs=changed_refs,
            views=view_tuple,
        )
        affected = []
        for decision_ref in impacted:
            basis = set()
            for evidence_ref in changed_refs:
                impact = self.impact(evidence_ref=evidence_ref, views=view_tuple)
                if decision_ref in impact.decision_refs:
                    basis.update(impact.basis_refs)
            affected.append(
                AffectedArtifactV1(
                    artifact_ref=decision_ref,
                    relation=ImpactRelation.DEPENDS_ON,
                    impact_kind=DeltaLens.EVIDENCE,
                    impact_status=ImpactStatus.TRANSITIVE,
                    basis_refs=_ordered(basis),
                )
            )
        ordered_affected = tuple(
            sorted(affected, key=lambda item: item.artifact_ref.encode("utf-8"))
        )
        hash_payload = {
            "schema_version": "governed_delta_impact_projection.v1",
            "population_scope": self._population(view_tuple).model_dump(mode="json"),
            "delta_ref": delta.delta_id,
            "delta_hash": delta.content_hash,
            "affected_artifacts": [
                item.model_dump(mode="json") for item in ordered_affected
            ],
            "affected_invariants": [],
            "required_qualification": [],
        }
        return ImpactProjectionV1.model_validate(
            {
                **hash_payload,
                "content_hash": "sha256:" + content_hash(hash_payload),
            }
        )

    @staticmethod
    def diagnose(
        *,
        expected: TenderEpistemicLineageViewV1,
        observed: TenderEpistemicLineageViewV1,
    ) -> TenderBoundaryDiagnosisV1:
        if expected.evidence_refs != observed.evidence_refs:
            layer: Layer | None = "EVIDENCE"
        elif expected.assertion_refs != observed.assertion_refs:
            layer = "MEANING"
        elif expected.predicate_refs != observed.predicate_refs:
            layer = "STRUCTURE"
        elif (
            expected.decision_ref != observed.decision_ref
            or expected.decision_hash != observed.decision_hash
            or expected.decision_result_code != observed.decision_result_code
        ):
            layer = "DECISION"
        elif expected.material_basis_refs != observed.material_basis_refs:
            # A dependency changed while node identities did not. Treat the first
            # broken boundary conservatively according to the changed edge source.
            changed = {
                edge.edge_ref: edge for edge in (*expected.edges, *observed.edges)
            }
            missing = set(expected.material_basis_refs) ^ set(observed.material_basis_refs)
            source_layers = {changed[item].source_layer for item in missing if item in changed}
            layer = next(
                (
                    candidate
                    for candidate in ("EVIDENCE", "MEANING", "STRUCTURE", "DECISION")
                    if candidate in source_layers
                ),
                "STRUCTURE",
            )
        else:
            layer = None
        return TenderBoundaryDiagnosisV1(
            first_invalid_layer=layer,
            expected_basis_refs=expected.material_basis_refs,
            observed_basis_refs=observed.material_basis_refs,
        )
