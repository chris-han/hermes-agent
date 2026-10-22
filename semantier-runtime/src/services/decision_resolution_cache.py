from __future__ import annotations

from dataclasses import dataclass, field

from contracts.decision_verification import ConflictControllerBindingV1, ConflictMembershipReceiptV1, ConflictResolutionReceiptV1, content_hash
from contracts.ontology_building import ArtifactPinV1
from services.tender_decision_verification_service import TenderDecisionVerificationService


@dataclass
class DecisionResolutionCache:
    _values: dict[str, ConflictResolutionReceiptV1] = field(default_factory=dict)

    @staticmethod
    def key(*, membership: ConflictMembershipReceiptV1, projection: ArtifactPinV1, composition_policy: ArtifactPinV1, controller: ConflictControllerBindingV1 | None) -> str:
        payload = {
            "membership": membership.pin().model_dump(mode="json"),
            "projection": projection.model_dump(mode="json"),
            "composition_policy": composition_policy.model_dump(mode="json"),
            "controller": None if controller is None else controller.model_dump(mode="json"),
        }
        return "decision-resolution-cache:v1:" + content_hash(payload)

    def resolve(self, *, membership: ConflictMembershipReceiptV1, projection: ArtifactPinV1, composition_policy: ArtifactPinV1, controller: ConflictControllerBindingV1 | None = None) -> tuple[ConflictResolutionReceiptV1, bool]:
        key = self.key(membership=membership, projection=projection, composition_policy=composition_policy, controller=controller)
        cached = self._values.get(key)
        if cached is not None:
            return cached, True
        result = TenderDecisionVerificationService.resolve_conflict(conflict_membership=membership, projection=projection, controller=controller)
        self._values[key] = result
        return result, False
