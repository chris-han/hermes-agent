"""Compose deterministic CCC output with existing institutional route authority."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from contracts.consequential_convergence import CCCDecisionReceiptV1, CCCInputV1
from contracts.ontology_building import ArtifactPinV1
from security.sensitive_data.policy.evaluator import ROUTE_ORDER
from services.consequential_convergence_controller import ConsequentialConvergenceController


ExecutionMode = Literal["AUTOPILOT", "ASSISTED"]
HumanResponseKind = Literal["SELECT_ALTERNATIVE", "CHANGE_CONSTRAINT"]


@dataclass(frozen=True)
class CCCInstitutionalRouteResult:
    ccc_receipt: CCCDecisionReceiptV1
    institutional_policy_route: str
    final_route: str
    route_basis: tuple[str, ...]
    mode: ExecutionMode
    requires_human_checkpoint: bool


class CCCInstitutionalRouteService:
    """Monotonic route composition; mode affects surfacing, never authority."""

    def __init__(
        self,
        controller: ConsequentialConvergenceController | None = None,
    ) -> None:
        self._controller = controller or ConsequentialConvergenceController()

    def evaluate(
        self,
        *,
        input_value: CCCInputV1,
        ccc_policy: ArtifactPinV1,
        institutional_policy_route: str,
        mode: ExecutionMode,
        evaluated_at: datetime,
    ) -> CCCInstitutionalRouteResult:
        if institutional_policy_route not in ROUTE_ORDER:
            raise ValueError("invalid institutional_policy_route")
        if mode not in {"AUTOPILOT", "ASSISTED"}:
            raise ValueError("invalid mode")

        receipt = self._controller.evaluate(
            input_value=input_value,
            policy=ccc_policy,
            evaluated_at=evaluated_at,
        )
        ccc_route = receipt.institutional_route or "PASS_AUTO"
        final_route = max(
            (institutional_policy_route, ccc_route),
            key=lambda route: ROUTE_ORDER[route],
        )
        basis = ["institutional_policy"]
        if receipt.institutional_route is not None:
            basis.append("ccc")

        return CCCInstitutionalRouteResult(
            ccc_receipt=receipt,
            institutional_policy_route=institutional_policy_route,
            final_route=final_route,
            route_basis=tuple(basis),
            mode=mode,
            requires_human_checkpoint=(
                receipt.disposition == "HUMAN_CHECKPOINT"
                and final_route != "FAIL_HARD"
            ),
        )

    @staticmethod
    def evaluate_human_response(
        *,
        prior: CCCInstitutionalRouteResult,
        response_kind: HumanResponseKind,
        selected_option_ref: ArtifactPinV1 | None,
    ) -> str:
        if not prior.requires_human_checkpoint:
            raise ValueError("human response requires prior checkpoint")
        if response_kind == "SELECT_ALTERNATIVE":
            if selected_option_ref is None:
                raise ValueError("selected alternative requires exact option ref")
            if selected_option_ref not in prior.ccc_receipt.typed_input.viable_option_refs:
                raise ValueError("selected alternative is outside prior viable frontier")
            return "REVALIDATION_REQUIRED"
        if response_kind == "CHANGE_CONSTRAINT":
            if selected_option_ref is not None:
                raise ValueError("constraint change does not directly select option")
            return "RECONVERGENCE_REQUIRED"
        raise ValueError("unsupported human response kind")


__all__ = ["CCCInstitutionalRouteResult", "CCCInstitutionalRouteService"]
