"""Pure deterministic Consequential Convergence Controller (CCC)."""
from __future__ import annotations

from datetime import datetime

from contracts.consequential_convergence import (
    CCCDecisionReceiptV1,
    CCCGateResultV1,
    CCCInputV1,
)
from contracts.ontology_building import ArtifactPinV1


_IMPACT = {
    "R0_TRANSIENT": 0,
    "R1_TASK": 1,
    "R2_WORKFLOW": 2,
    "R3_ORGANIZATIONAL": 3,
    "R4_EXTERNAL": 4,
    "R5_INSTITUTIONAL": 5,
}
_STAKE = {"LOW": 0, "MATERIAL": 1, "HIGH": 2, "CRITICAL": 3}


class ConsequentialConvergenceController:
    """Evaluate typed commitment context without semantic inference or side effects."""

    def evaluate(
        self,
        *,
        input_value: CCCInputV1,
        policy: ArtifactPinV1,
        evaluated_at: datetime,
    ) -> CCCDecisionReceiptV1:
        value = CCCInputV1.model_validate(input_value)
        gates: list[CCCGateResultV1] = []

        def check(gate: str, matched: bool) -> bool:
            gates.append(CCCGateResultV1(gate=gate, matched=matched))
            return matched

        if check("G0_NO_VIABLE_FRONTIER", not value.viable_option_refs):
            return self._receipt(
                value=value,
                policy=policy,
                gates=gates,
                decisive_gate="G0_NO_VIABLE_FRONTIER",
                disposition="AUTO_CONTINUE",
                route=None,
                selected=(),
                recommended=(),
                evaluated_at=evaluated_at,
            )

        if check(
            "G1_SAFE_INFORMATION_GAIN",
            value.safe_information_gain_available
            and not value.immediate_commitment_required,
        ):
            return self._receipt(
                value=value,
                policy=policy,
                gates=gates,
                decisive_gate="G1_SAFE_INFORMATION_GAIN",
                disposition="AUTO_CONTINUE",
                route=None,
                selected=(),
                recommended=(),
                evaluated_at=evaluated_at,
            )

        if check("G2_HUMAN_RESERVED", value.authority_gap == "HUMAN_RESERVED"):
            return self._checkpoint(
                value, policy, gates, "G2_HUMAN_RESERVED", evaluated_at
            )

        durable = (
            _IMPACT[value.impact_radius] >= _IMPACT["R3_ORGANIZATIONAL"]
            or value.reversibility in {"COMPENSATABLE", "IRREVERSIBLE"}
        )
        if check(
            "G3_UNRESOLVED_DURABLE_AUTHORITY",
            value.authority_gap == "UNRESOLVED_AUTHORITY" and durable,
        ):
            return self._checkpoint(
                value,
                policy,
                gates,
                "G3_UNRESOLVED_DURABLE_AUTHORITY",
                evaluated_at,
            )

        if check(
            "G4_IRREVERSIBLE_MATERIAL",
            value.reversibility == "IRREVERSIBLE"
            and _STAKE[value.stake_value] >= _STAKE["MATERIAL"],
        ):
            return self._checkpoint(
                value, policy, gates, "G4_IRREVERSIBLE_MATERIAL", evaluated_at
            )

        sufficient_authority = value.authority_gap in {"NONE", "ENCODED_DELEGATION"}
        if check(
            "G5_DOMINANT_LAWFUL_PATH",
            value.dominant_option_ref is not None and sufficient_authority,
        ):
            return self._receipt(
                value=value,
                policy=policy,
                gates=gates,
                decisive_gate="G5_DOMINANT_LAWFUL_PATH",
                disposition="AUTO_COMMIT",
                route="PASS_AUTO",
                selected=(value.dominant_option_ref,),
                recommended=(),
                evaluated_at=evaluated_at,
            )

        consequential_tradeoff = (
            len(value.viable_option_refs) > 1
            and value.branch_ambiguity in {"MATERIAL", "UNRESOLVED"}
            and (
                _IMPACT[value.impact_radius] >= _IMPACT["R3_ORGANIZATIONAL"]
                or _STAKE[value.stake_value] >= _STAKE["MATERIAL"]
                or value.reversibility in {"COMPENSATABLE", "IRREVERSIBLE"}
            )
        )
        check("G6_CONSEQUENTIAL_NONDOMINATED_TRADEOFF", consequential_tradeoff)
        if consequential_tradeoff:
            if value.human_decision_value is None:
                raise ValueError("HUMAN_DECISION_VALUE_REQUIRED")
            positive = value.human_decision_value.is_positive()
            check("G7_HUMAN_DECISION_VALUE", positive)
            if positive:
                return self._checkpoint(
                    value,
                    policy,
                    gates,
                    "G7_POSITIVE_HUMAN_DECISION_VALUE",
                    evaluated_at,
                )
            return self._receipt(
                value=value,
                policy=policy,
                gates=gates,
                decisive_gate="G7_NONPOSITIVE_HUMAN_DECISION_VALUE",
                disposition="RECOMMEND",
                route="PASS_AUTO",
                selected=(),
                recommended=value.viable_option_refs,
                evaluated_at=evaluated_at,
            )

        return self._receipt(
            value=value,
            policy=policy,
            gates=gates,
            decisive_gate="G6_LOW_CONSEQUENCE_MULTIBRANCH",
            disposition="RECOMMEND",
            route="PASS_AUTO",
            selected=(),
            recommended=value.viable_option_refs,
            evaluated_at=evaluated_at,
        )

    def _checkpoint(
        self,
        value: CCCInputV1,
        policy: ArtifactPinV1,
        gates: list[CCCGateResultV1],
        decisive_gate: str,
        evaluated_at: datetime,
    ) -> CCCDecisionReceiptV1:
        return self._receipt(
            value=value,
            policy=policy,
            gates=gates,
            decisive_gate=decisive_gate,
            disposition="HUMAN_CHECKPOINT",
            route="ESCALATE",
            selected=(),
            recommended=value.viable_option_refs,
            evaluated_at=evaluated_at,
        )

    @staticmethod
    def _receipt(
        *,
        value: CCCInputV1,
        policy: ArtifactPinV1,
        gates: list[CCCGateResultV1],
        decisive_gate: str,
        disposition: str,
        route: str | None,
        selected: tuple[ArtifactPinV1, ...],
        recommended: tuple[ArtifactPinV1, ...],
        evaluated_at: datetime,
    ) -> CCCDecisionReceiptV1:
        return CCCDecisionReceiptV1.create(
            schema_version="ccc_decision_receipt.v1",
            organization_id=value.organization_id,
            workspace_id=value.workspace_id,
            typed_input=value,
            policy=policy,
            gate_results=tuple(gates),
            decisive_gate=decisive_gate,
            disposition=disposition,
            institutional_route=route,
            selected_option_refs=selected,
            recommended_option_refs=recommended,
            human_checkpoint_ref=None,
            human_response_ref=None,
            evaluated_at=evaluated_at,
        )


__all__ = ["ConsequentialConvergenceController"]
