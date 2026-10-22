"""Task-local immutable HR review evidence; never an active policy artifact."""

from __future__ import annotations
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from eos.artifact_hashing import content_hash


class ReviewContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JdFindingV1(ReviewContract):
    finding_id: str
    requirement_id: str
    summary: str
    finding_class: str
    evaluation_profile: str = "jd_review_candidate_checks.v1"
    rule_refs: list[str] = Field(default_factory=list)
    content_category: str
    sensitivity_assessment: Literal["CONFIRMED", "CANDIDATE", "UNKNOWN"]
    task_relevance: Literal["RELEVANT", "IRRELEVANT", "UNKNOWN"]
    collection_necessity: Literal["NECESSARY", "UNNECESSARY", "UNKNOWN"]
    legal_applicability: Literal["APPLICABLE", "NOT_APPLICABLE", "UNKNOWN"]
    compliance_status: Literal[
        "RISK_CANDIDATE",
        "POLICY_VIOLATION_SUPPORTED",
        "NEEDS_REVIEW",
        "NO_SUPPORTED_VIOLATION",
    ]
    remediation_status: Literal[
        "AUTO_DRAFT_ELIGIBLE", "REVIEW_REQUIRED", "UNAVAILABLE", "CHANGED_AND_RECHECKED"
    ]
    evidence_ref: dict[str, Any]
    anchor_ref: str
    occurrence_ref: str
    presentation_location: dict[str, Any] | None
    basis_refs: list[str] = Field(default_factory=list)
    basis_evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    information_categories: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    role: str
    proposed_edit: dict[str, Any] | None = None


class JdReviewV1(ReviewContract):
    schema_version: Literal["jd_review.v1"] = "jd_review.v1"
    organization_id: str
    workspace_id: str
    actor_id: str
    session_id: str
    review_id: str
    result_hash: str
    version_id: str
    status: Literal[
        "CHECKED", "PARTIAL", "GENERATION_FAILED", "RECHECK_FAILED", "CANCELLED"
    ]
    recorded_at: datetime
    source: dict[str, Any]
    canonical_input: dict[str, Any]
    requirements_graph: dict[str, Any]
    findings: list[JdFindingV1]
    basis_documents: list[dict[str, Any]] = Field(default_factory=list)
    active_basis_status: Literal["AVAILABLE", "UNAVAILABLE"] = "UNAVAILABLE"
    classification_basis: dict[str, Any] = Field(default_factory=dict)
    classifier_findings: list[dict[str, Any]] = Field(default_factory=list)
    successor: dict[str, Any] | None = None
    coverage: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def integrity(self):
        if self.recorded_at.tzinfo is None:
            raise ValueError("UTC_TIMESTAMP_REQUIRED")
        payload = self.model_dump(
            mode="json", exclude={"result_hash"}, exclude_unset=True
        )
        if self.result_hash != "sha256:" + content_hash(payload):
            raise ValueError("REVIEW_HASH_MISMATCH")
        return self


class JdSuccessorCommandV1(ReviewContract):
    request_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    result_hash: str
    source_hash: str
    version_id: str
    cancel: bool = False


class JdBasisEvidenceV1(ReviewContract):
    schema_version: Literal["jd_basis_evidence.v1"] = "jd_basis_evidence.v1"
    organization_id: str
    workspace_id: str
    actor_id: str
    session_id: str
    basis_ref: str
    basis_hash: str
    source_id: str
    document: dict[str, Any]
    activation_status: Literal["NOT_ACTIVATED"] = "NOT_ACTIVATED"

    @model_validator(mode="after")
    def integrity(self):
        if self.basis_hash != "sha256:" + content_hash(
            self.model_dump(mode="json", exclude={"basis_hash"})
        ):
            raise ValueError("BASIS_HASH_MISMATCH")
        return self


class JdFeedbackCommandV1(ReviewContract):
    request_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    result_hash: str
    version_id: str
    finding_id: str
    reason: Literal[
        "finding_wrong",
        "basis_inapplicable",
        "job_fact_lost",
        "location_wrong",
        "retain_objection",
    ]


class JdFeedbackV1(JdFeedbackCommandV1):
    schema_version: Literal["jd_feedback.v1"] = "jd_feedback.v1"
    organization_id: str
    workspace_id: str
    actor_id: str
    session_id: str
    feedback_ref: str
    feedback_hash: str
    review_id: str
    recorded_at: datetime
    authority_role: Literal["NON_AUTHORITATIVE_USER_FEEDBACK"] = (
        "NON_AUTHORITATIVE_USER_FEEDBACK"
    )

    @model_validator(mode="after")
    def integrity(self):
        if self.recorded_at.tzinfo is None:
            raise ValueError("UTC_TIMESTAMP_REQUIRED")
        if self.feedback_hash != "sha256:" + content_hash(
            self.model_dump(mode="json", exclude={"feedback_hash"})
        ):
            raise ValueError("FEEDBACK_HASH_MISMATCH")
        return self
