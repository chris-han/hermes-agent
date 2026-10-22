from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EscalationReviewerAssignment:
    assigned_role: str
    assigned_principal: str


class DeterministicEscalationReviewerResolver:
    assigned_role = "governance_security_reviewer"

    def resolve(self, *, organization_id: str) -> EscalationReviewerAssignment | None:
        if not organization_id:
            return None
        return EscalationReviewerAssignment(
            assigned_role=self.assigned_role,
            assigned_principal=f"{organization_id}:{self.assigned_role}",
        )
