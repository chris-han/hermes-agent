"""Deterministic predicate-local epistemic admissibility routing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from contracts.decision_graph_projection import (
    DecisionGraphProjectionV1,
    EpistemicAdmissibilityRoutingV1,
    PredicateEpistemicAdmissibilityV1,
    PredicateEpistemicRequirementV1,
    ProjectionSufficiencyProofV1,
)


class EpistemicAdmissibilityRoutingService:
    """Narrow a semantic frontier from governed projection sufficiency only."""

    @staticmethod
    def _ordered(values) -> tuple[str, ...]:
        return tuple(sorted((str(value) for value in values), key=lambda item: item.encode("utf-8")))

    def compile(
        self,
        *,
        projection: DecisionGraphProjectionV1,
        sufficiency: ProjectionSufficiencyProofV1,
        requirements: Sequence[PredicateEpistemicRequirementV1],
        available_semantic_dimensions: Mapping[str, Sequence[str]],
    ) -> EpistemicAdmissibilityRoutingV1:
        if projection.sufficiency_proof.artifact_ref != sufficiency.proof_ref or projection.sufficiency_proof.artifact_hash != sufficiency.proof_hash:
            raise ValueError("EPISTEMIC_ADMISSIBILITY_SUFFICIENCY_PROOF_MISMATCH")
        if projection.source_dlp_ref != sufficiency.dlp_ref or projection.source_dlp_hash != sufficiency.dlp_hash:
            raise ValueError("EPISTEMIC_ADMISSIBILITY_DLP_MISMATCH")
        if projection.frontier_ref != sufficiency.frontier_ref or projection.frontier_hash != sufficiency.frontier_hash:
            raise ValueError("EPISTEMIC_ADMISSIBILITY_FRONTIER_MISMATCH")

        partition = self.partition(requirements=requirements, available_semantic_dimensions=available_semantic_dimensions)
        ordered_requirements = sorted(requirements, key=lambda item: item.predicate_id.encode("utf-8"))

        items = []
        for requirement in ordered_requirements:
            required = requirement.required_semantic_dimension_ids
            available, missing = partition[requirement.predicate_id]
            items.append(PredicateEpistemicAdmissibilityV1(
                predicate_id=requirement.predicate_id,
                status="INADMISSIBLE" if missing else "ADMISSIBLE",
                required_semantic_dimension_ids=required,
                available_semantic_dimension_ids=available,
                missing_semantic_dimension_ids=missing,
                projection_sufficiency_proof=sufficiency.pin(),
                disposition="UNKNOWN" if missing else "SCORE_RESIDUAL_FRONTIER",
            ))

        items_tuple = tuple(items)
        eligible = tuple(item.predicate_id for item in items_tuple if item.status == "ADMISSIBLE")
        unknown = tuple(item.predicate_id for item in items_tuple if item.status == "INADMISSIBLE")
        return EpistemicAdmissibilityRoutingV1.create(
            decision_graph_projection=projection.pin(),
            projection_sufficiency_proof=sufficiency.pin(),
            predicate_admissibility=items_tuple,
            scorer_eligible_predicate_ids=eligible,
            structurally_unknown_predicate_ids=unknown,
        )

    def verify(self, **inputs) -> None:
        routing = inputs.pop("routing")
        if routing != self.compile(**inputs):
            raise ValueError("EPISTEMIC_ADMISSIBILITY_DERIVATION_MISMATCH")

    def partition(self, *, requirements, available_semantic_dimensions):
        ids = [r.predicate_id for r in requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("EPISTEMIC_ADMISSIBILITY_DUPLICATE_PREDICATE")
        result = {}
        for requirement in sorted(
            requirements, key=lambda r: r.predicate_id.encode("utf-8")
        ):
            available = self._ordered(
                available_semantic_dimensions.get(requirement.predicate_id, ())
            )
            result[requirement.predicate_id] = (
                available,
                tuple(
                    d
                    for d in requirement.required_semantic_dimension_ids
                    if d not in available
                ),
            )
        return result

    def compile_candidate_review(self, *, projection):
        available = {}
        witnesses = {w.pin: w for w in projection.supporting_evidence}
        for r in projection.requirements:
            dimensions = []
            for d in r.dimensions:
                matches = [
                    w
                    for w in witnesses.values()
                    if d.evidence_role is None or w.evidence_role == d.evidence_role
                ]
                if (
                    set(d.required_evidence_pins) <= set(w.pin for w in matches)
                    and len(matches) >= d.min_count
                ):
                    dimensions.append(d.dimension_id)
            available[r.requirement.predicate_id] = dimensions
        return self.partition(
            requirements=[r.requirement for r in projection.requirements],
            available_semantic_dimensions=available,
        )

    def verify_candidate_review(self, *, projection, routing):
        if routing != self.compile_candidate_review(projection=projection):
            raise ValueError("EPISTEMIC_ADMISSIBILITY_DERIVATION_MISMATCH")
