from __future__ import annotations

from typing import Any, Protocol

from contracts.tender_sensitive_term_detection_modeling_core import RuntimeContextDTO


class GovernedRuleActivationService(Protocol):
    def create_activation_atomic(
        self,
        activation_payload: dict[str, Any],
        context: RuntimeContextDTO,
    ) -> dict[str, Any]: ...

    def append_lifecycle_event(
        self,
        activation_id: str,
        event_payload: dict[str, Any],
        context: RuntimeContextDTO,
    ) -> dict[str, Any]: ...

    def resolve_effective_state(self, activation_id: str, effective_at: str) -> dict[str, Any]: ...
