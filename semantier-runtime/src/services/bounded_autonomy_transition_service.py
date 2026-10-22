"""Pure single-user bounded-autonomy transition recommendation."""
from __future__ import annotations

from contracts.epistemic_readiness import BoundedAutonomyTransitionV1, EpistemicReadinessV1


class BoundedAutonomyTransitionService:
    @staticmethod
    def recommend(
        *,
        readiness: EpistemicReadinessV1,
        actor_authorized: bool,
        intent_requires_transition: bool,
        exploration_only: bool = False,
        repair_available: bool = False,
        inference_available: bool = False,
        recommendation_available: bool = False,
        consequential: bool = False,
        reversible: bool = True,
        within_risk_envelope: bool = True,
    ) -> BoundedAutonomyTransitionV1:
        evidence = readiness.evidence_refs
        if not within_risk_envelope or (consequential and not reversible):
            return BoundedAutonomyTransitionV1(
                disposition="ASK", best_next_action="confirm_responsibility",
                reason_codes=("NON_DELEGATED_CONSEQUENTIAL_TRANSITION",),
                evidence_refs=evidence, user_interruption_required=True,
            )
        if readiness.disposition == "REPAIRABLE" or repair_available:
            return BoundedAutonomyTransitionV1(
                disposition="AUTO_REPAIR", best_next_action="repair_and_reevaluate",
                reason_codes=readiness.reason_codes or ("REPAIR_AVAILABLE",), evidence_refs=evidence,
            )
        if readiness.disposition == "AUTO_QUALIFIED":
            if intent_requires_transition:
                return BoundedAutonomyTransitionV1(
                    disposition="AUTO_TRANSITION", best_next_action="proceed",
                    reason_codes=readiness.reason_codes, evidence_refs=evidence,
                )
            return BoundedAutonomyTransitionV1(
                disposition="RECOMMEND", best_next_action="continue_exploration",
                reason_codes=("NO_TRANSITION_INTENT",), evidence_refs=evidence,
            )
        if exploration_only or not intent_requires_transition:
            return BoundedAutonomyTransitionV1(
                disposition="RECOMMEND", best_next_action="continue_with_visible_uncertainty",
                reason_codes=readiness.reason_codes, evidence_refs=evidence,
            )
        if inference_available:
            return BoundedAutonomyTransitionV1(
                disposition="AUTO_INFER", best_next_action="infer_and_reevaluate",
                reason_codes=readiness.reason_codes, evidence_refs=evidence,
            )
        if recommendation_available:
            return BoundedAutonomyTransitionV1(
                disposition="RECOMMEND", best_next_action="review_recommendation",
                reason_codes=readiness.reason_codes, evidence_refs=evidence,
            )
        return BoundedAutonomyTransitionV1(
            disposition="ASK", best_next_action="resolve_material_uncertainty",
            reason_codes=readiness.reason_codes, evidence_refs=evidence,
            user_interruption_required=True,
        )


__all__ = ["BoundedAutonomyTransitionService"]
