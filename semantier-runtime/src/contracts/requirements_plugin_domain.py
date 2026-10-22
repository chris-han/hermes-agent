"""Requirements-plugin domain evidence contracts.

These objects describe bounded review obligations and epistemic status. They
carry no admission, publication, activation, or institutional-route authority.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import ArtifactPinV1


class ReviewObligationEvidenceStatus(StrEnum):
    CONFIRMED = "CONFIRMED"
    POSSIBLE = "POSSIBLE"
    UNAVAILABLE = "UNAVAILABLE"


class RequirementReviewObligationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["requirement_review_obligation.v1"] = (
        "requirement_review_obligation.v1"
    )
    obligation_id: str = Field(min_length=1)
    obligation_kind: Literal[
        "EVALUATION",
        "VERIFICATION",
        "FUNCTION_ALLOCATION",
        "SUBMISSION_EVIDENCE",
        "COMPLIANCE_REVIEW",
    ]
    trigger_requirement_ref: str = Field(min_length=1)
    review_action: str = Field(min_length=1)
    owner_ref: str | None
    evidence_status: ReviewObligationEvidenceStatus
    witness_refs: tuple[ArtifactPinV1, ...]
    coverage_scope_refs: tuple[ArtifactPinV1, ...]
    basis_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence(self) -> "RequirementReviewObligationV1":
        if not self.witness_refs:
            raise ValueError("review obligation requires at least one exact witness")
        if not self.coverage_scope_refs:
            raise ValueError("review obligation requires bounded coverage scope")
        if self.evidence_status == ReviewObligationEvidenceStatus.CONFIRMED:
            if self.owner_ref is None or not self.owner_ref.strip():
                raise ValueError("CONFIRMED review obligation requires owner_ref")
        return self


__all__ = [
    "RequirementReviewObligationV1",
    "ReviewObligationEvidenceStatus",
]
