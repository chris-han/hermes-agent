"""Research-only H5/H6 mapping from bounded assessment to learning consumption.

This module has no admission, activation, runtime mutation, or training authority.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AssessmentKind = Literal[
    "INSUFFICIENT_EVIDENCE",
    "INCOMPARABLE",
    "NO_ACTIONABLE_CHANGE",
    "PENDING_CHANGE",
    "SUSTAINED_CHANGE",
    "TERMINAL_INVARIANT_VIOLATION",
]
EvidenceProvenance = Literal[
    "EXECUTED_PRODUCER_OWNED",
    "DEMONSTRATION",
    "CLIENT_ASSERTED",
    "UNKNOWN",
]
ConsumptionDisposition = Literal["SKIP", "OBSERVE_ONLY", "CANDIDATE_ELIGIBLE"]


class LearningConsumptionInputV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_ref: str = Field(min_length=1)
    assessment_kind: AssessmentKind
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    comparison_basis_ref: str = Field(min_length=1)
    domain_contract_ref: str = Field(min_length=1)
    assessment_trace: tuple[str, ...] = Field(min_length=1)
    provenance: EvidenceProvenance

    learning_objective: str | None = None
    expected_learning_value: str | None = None
    candidate_owner: str | None = None
    candidate_type: str | None = None
    candidate_change_kind: Literal["REPAIR", "MITIGATION", "OTHER"] | None = None


class LearningConsumptionDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: ConsumptionDisposition
    reason_code: str = Field(min_length=1)
    assessment_ref: str
    evidence_refs: tuple[str, ...]
    comparison_basis_ref: str
    domain_contract_ref: str
    assessment_trace: tuple[str, ...]
    learning_objective: str | None = None
    expected_learning_value: str | None = None
    candidate_owner: str | None = None
    candidate_type: str | None = None

    admission_authority: Literal[False] = False
    activation_authority: Literal[False] = False


def map_assessment_to_learning(
    value: LearningConsumptionInputV1,
) -> LearningConsumptionDecisionV1:
    """Apply the frozen H5/H6 state-to-learning mapping."""

    common = dict(
        assessment_ref=value.assessment_ref,
        evidence_refs=value.evidence_refs,
        comparison_basis_ref=value.comparison_basis_ref,
        domain_contract_ref=value.domain_contract_ref,
        assessment_trace=value.assessment_trace,
    )

    if value.provenance != "EXECUTED_PRODUCER_OWNED":
        return LearningConsumptionDecisionV1(
            disposition="SKIP",
            reason_code=f"EMPIRICAL_PROVENANCE_{value.provenance}_NOT_ELIGIBLE",
            **common,
        )

    if value.assessment_kind in {
        "INSUFFICIENT_EVIDENCE",
        "INCOMPARABLE",
        "NO_ACTIONABLE_CHANGE",
    }:
        return LearningConsumptionDecisionV1(
            disposition="SKIP",
            reason_code=f"{value.assessment_kind}_DEFAULT_SKIP",
            **common,
        )

    if value.assessment_kind == "PENDING_CHANGE":
        return LearningConsumptionDecisionV1(
            disposition="OBSERVE_ONLY",
            reason_code="PENDING_CHANGE_REQUIRES_MORE_QUALIFIED_EVIDENCE",
            learning_objective=value.learning_objective,
            expected_learning_value=value.expected_learning_value,
            **common,
        )

    required = (
        value.learning_objective,
        value.expected_learning_value,
        value.candidate_owner,
        value.candidate_type,
    )
    if any(item is None or not str(item).strip() for item in required):
        return LearningConsumptionDecisionV1(
            disposition="SKIP",
            reason_code="CANDIDATE_METADATA_OR_OWNER_INCOMPLETE",
            **common,
        )

    if (
        value.assessment_kind == "TERMINAL_INVARIANT_VIOLATION"
        and value.candidate_change_kind not in {"REPAIR", "MITIGATION"}
    ):
        return LearningConsumptionDecisionV1(
            disposition="SKIP",
            reason_code="TERMINAL_INVARIANT_REQUIRES_REPAIR_OR_MITIGATION",
            **common,
        )

    return LearningConsumptionDecisionV1(
        disposition="CANDIDATE_ELIGIBLE",
        reason_code=(
            "SUSTAINED_CHANGE_LEARNING_ELIGIBLE"
            if value.assessment_kind == "SUSTAINED_CHANGE"
            else "TERMINAL_INVARIANT_REPAIR_ELIGIBLE"
        ),
        learning_objective=value.learning_objective,
        expected_learning_value=value.expected_learning_value,
        candidate_owner=value.candidate_owner,
        candidate_type=value.candidate_type,
        **common,
    )


__all__ = [
    "LearningConsumptionDecisionV1",
    "LearningConsumptionInputV1",
    "map_assessment_to_learning",
]
