"""Single-user orchestration over existing admission/projection/activation owners."""
from __future__ import annotations

from collections.abc import Callable

from contracts.epistemic_readiness import (
    EpistemicReadinessV1,
    KnowledgeUseOrchestrationResultV1,
)
from services.bounded_autonomy_transition_service import BoundedAutonomyTransitionService


class KnowledgeUseOrchestrationService:
    @staticmethod
    def execute(
        *,
        readiness: EpistemicReadinessV1,
        actor_authorized: bool,
        activate_requested: bool,
        admit: Callable[[], str],
        project: Callable[[], str] | None = None,
        activate: Callable[[], str] | None = None,
        repair: Callable[[], EpistemicReadinessV1] | None = None,
        infer: Callable[[], EpistemicReadinessV1] | None = None,
        consequential: bool = False,
        reversible: bool = True,
        within_risk_envelope: bool = True,
    ) -> KnowledgeUseOrchestrationResultV1:
        current = readiness
        for _ in range(3):
            transition = BoundedAutonomyTransitionService.recommend(
                readiness=current,
                actor_authorized=actor_authorized,
                intent_requires_transition=True,
                repair_available=repair is not None and current.disposition == "REPAIRABLE",
                inference_available=infer is not None,
                consequential=consequential,
                reversible=reversible,
                within_risk_envelope=within_risk_envelope,
            )
            if transition.disposition == "AUTO_REPAIR" and repair is not None:
                current = repair()
                repair = None
                continue
            if transition.disposition == "AUTO_INFER" and infer is not None:
                current = infer()
                infer = None
                continue
            break

        if transition.disposition != "AUTO_TRANSITION":
            return KnowledgeUseOrchestrationResultV1(readiness=current, transition=transition)

        admission_ref = admit()
        if not activate_requested:
            return KnowledgeUseOrchestrationResultV1(
                readiness=current, transition=transition, admission_receipt_ref=admission_ref,
            )
        if project is None or activate is None:
            raise ValueError("activation intent requires existing projection and activation owners")
        projection_ref = project()
        activation_ref = activate()
        return KnowledgeUseOrchestrationResultV1(
            readiness=current,
            transition=transition,
            admission_receipt_ref=admission_ref,
            projection_receipt_ref=projection_ref,
            activation_receipt_ref=activation_ref,
            activated=True,
        )


__all__ = ["KnowledgeUseOrchestrationService"]
