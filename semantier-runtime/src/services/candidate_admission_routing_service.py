"""Pure post-assessment routing for non-authoritative admission classifications.

This service consumes an already-assessed learning candidate plus a deterministic
CandidateAdmissionClassificationV1. It does not persist, admit, activate, create
owner receipts, or mutate candidate state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from contracts.artifact_pin import ArtifactPinV1
from contracts.candidate_admission_classification import CandidateAdmissionClassificationV1
from contracts.change_candidate_v2 import ChangeCandidateV2, LearningHypothesisProfileV2
from contracts.semantic_kernel import ChangeCandidate


CandidateAdmissionDestination = Literal[
    "PROCEDURAL_PROJECTION",
    "GOVERNED_ADMISSION",
    "REJECTED_NEGATIVE_EVIDENCE",
    "PENDING",
]


@dataclass(frozen=True)
class CandidateAdmissionRouteV1:
    """Non-authoritative routing result over an assessed learning candidate."""

    candidate: ArtifactPinV1
    classification: CandidateAdmissionClassificationV1
    destination: CandidateAdmissionDestination
    projection_eligible: bool
    governed_admission_required: bool
    rejected: bool
    pending: bool
    preserve_negative_evidence: bool
    negative_evidence_candidate: ArtifactPinV1 | None
    negative_evidence_classification: CandidateAdmissionClassificationV1 | None
    reason_code: str


class CandidateAdmissionRoutingService:
    """Route assessed candidates without creating admission or activation authority."""

    def route(
        self,
        *,
        candidate: ChangeCandidate | ChangeCandidateV2,
        classification: CandidateAdmissionClassificationV1,
    ) -> CandidateAdmissionRouteV1:
        candidate_pin = candidate.pin()
        if classification.candidate != candidate_pin:
            raise ValueError("CLASSIFICATION_CANDIDATE_MISMATCH")

        stage, assessment = self._assessment_state(candidate)
        if stage != "ASSESSMENT" or assessment is None:
            raise ValueError("ASSESSMENT_REQUIRED")
        if assessment.disposition != "ACCEPTED_FOR_SUCCESSOR_INPUT":
            raise ValueError("BEHAVIORAL_QUALIFICATION_REQUIRED")

        if classification.disposition == "PROCEDURAL_SAFE_TO_PROJECT":
            return CandidateAdmissionRouteV1(
                candidate=candidate_pin,
                classification=classification,
                destination="PROCEDURAL_PROJECTION",
                projection_eligible=True,
                governed_admission_required=False,
                rejected=False,
                pending=False,
                preserve_negative_evidence=False,
                negative_evidence_candidate=None,
                negative_evidence_classification=None,
                reason_code="PROCEDURAL_PROJECTION_ELIGIBLE",
            )

        if classification.disposition == "SEMANTIC_CHANGE_REQUIRES_ADMISSION":
            return CandidateAdmissionRouteV1(
                candidate=candidate_pin,
                classification=classification,
                destination="GOVERNED_ADMISSION",
                projection_eligible=False,
                governed_admission_required=True,
                rejected=False,
                pending=False,
                preserve_negative_evidence=False,
                negative_evidence_candidate=None,
                negative_evidence_classification=None,
                reason_code="GOVERNED_SEMANTIC_ADMISSION_REQUIRED",
            )

        if classification.disposition == "AUTHORITY_VIOLATION":
            return CandidateAdmissionRouteV1(
                candidate=candidate_pin,
                classification=classification,
                destination="REJECTED_NEGATIVE_EVIDENCE",
                projection_eligible=False,
                governed_admission_required=False,
                rejected=True,
                pending=False,
                preserve_negative_evidence=True,
                negative_evidence_candidate=candidate_pin,
                negative_evidence_classification=classification,
                reason_code="AUTHORITY_VIOLATION_REJECTED",
            )

        if classification.disposition == "INSUFFICIENT_EVIDENCE":
            return CandidateAdmissionRouteV1(
                candidate=candidate_pin,
                classification=classification,
                destination="PENDING",
                projection_eligible=False,
                governed_admission_required=False,
                rejected=False,
                pending=True,
                preserve_negative_evidence=False,
                negative_evidence_candidate=None,
                negative_evidence_classification=None,
                reason_code="INSUFFICIENT_EVIDENCE_PENDING",
            )

        raise ValueError("UNSUPPORTED_ADMISSION_CLASSIFICATION")

    @staticmethod
    def _assessment_state(
        candidate: ChangeCandidate | ChangeCandidateV2,
    ) -> tuple[str | None, object | None]:
        if isinstance(candidate, ChangeCandidate):
            return candidate.stage, candidate.assessment

        profile = candidate.profile
        if not isinstance(profile, LearningHypothesisProfileV2):
            raise ValueError("ASSESSMENT_REQUIRED")
        return profile.stage, profile.assessment


__all__ = [
    "CandidateAdmissionDestination",
    "CandidateAdmissionRouteV1",
    "CandidateAdmissionRoutingService",
]
