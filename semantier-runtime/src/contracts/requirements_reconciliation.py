"""Embedded workflow receipts; no semantic authority or activation capability."""

from datetime import datetime, timedelta
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from contracts.ontology_building import ArtifactPinV1
from contracts.evidence_location import EvidenceRef
from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.normative_conflict import NormativeSetV1, NormativeCompatibilityResultV1
from eos.artifact_hashing import content_hash


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RequirementQuestionV1(Closed):
    question_id: str
    kind: Literal["verified_conflict", "missing_semantics"]
    member_rule_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    dependency_hash: str
    fact_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)
    proof_result_ref: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    member_effect_refs: tuple[str, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    proposed_repair: Literal["clarify_scope_or_submit_successor"] = (
        "clarify_scope_or_submit_successor"
    )


class RequirementDispositionV1(Closed):
    question_id: str
    dependency_hash: str
    decision: Literal["answer", "defer", "reject", "accept_repair"]
    rationale: str = Field(min_length=1)
    actor_ref: str
    successor_candidate_pin: ArtifactPinV1 | None = None


class RequirementsReviewV1(Closed):
    schema_version: Literal["requirements_review.v1"] = "requirements_review.v1"
    review_ref: str
    review_hash: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    organization_id: str
    workspace_id: str
    journey_id: str
    revision: int = Field(ge=1)
    journey_revision: int = Field(ge=0)
    predecessor_pin: ArtifactPinV1 | None
    source_role: Literal[
        "DESIGN_SOURCE", "ACTIVE_RULE_BASIS", "RUNTIME_TARGET", "PREPARED_SENSITIVITY"
    ]
    candidate: NormativeSemanticCandidateSetV1
    source_evidence: tuple[EvidenceRef, ...] = ()
    source_lineage_anchor: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    outcome: Literal["COMPATIBLE", "CONFLICT", "UNKNOWN"]
    coverage: Literal["BOUNDED_COMPLETE", "INCOMPLETE"]
    assessed_sets: tuple[NormativeSetV1, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    assessments: tuple[NormativeCompatibilityResultV1, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    supported_domains: tuple[str, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    supported_predicates: tuple[str, ...] = ("EQ", "IN", "GE", "GT", "LE", "LT")
    unknown_rule_refs: tuple[str, ...]
    questions: tuple[RequirementQuestionV1, ...]
    dispositions: tuple[RequirementDispositionV1, ...]
    superseded_dispositions: tuple[RequirementDispositionV1, ...] = ()
    affected_rule_refs: tuple[str, ...]
    actor_ref: str
    created_at: datetime

    def pin(self):
        return ArtifactPinV1(
            artifact_kind="requirements_review",
            artifact_ref=self.review_ref,
            artifact_hash=self.review_hash,
        )

    @model_validator(mode="after")
    def identity(self):
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != timedelta(
            0
        ):
            raise ValueError("UTC_REQUIRED")
        if (self.organization_id, self.workspace_id) != (
            self.candidate.organization_id,
            self.candidate.workspace_id,
        ):
            raise ValueError("REQUIREMENTS_SCOPE_MISMATCH")
        sets = {s.normative_set_ref: s for s in self.assessed_sets}
        for basis in self.assessed_sets:
            if (basis.organization_id, basis.workspace_id) != (
                self.organization_id,
                self.workspace_id,
            ) or (basis.compiled_rule_plan_ref, basis.compiled_rule_plan_hash) != (
                self.candidate.candidate_set_ref,
                self.candidate.candidate_set_hash,
            ):
                raise ValueError("REQUIREMENTS_PROOF_SCOPE_MISMATCH")
        proofs = {r.result_ref: r for r in self.assessments}
        for result in self.assessments:
            basis = sets.get(result.normative_set_ref)
            if basis is None or basis.normative_set_hash != result.normative_set_hash:
                raise ValueError("REQUIREMENTS_PROOF_BASIS_MISMATCH")
        for question in self.questions:
            if question.proof_result_ref is not None:
                proof = proofs.get(question.proof_result_ref)
                if (
                    proof is None
                    or proof.outcome != "CONFLICT"
                    or question.member_effect_refs != proof.conflicting_effect_refs
                ):
                    raise ValueError("REQUIREMENTS_CONFLICT_PROOF_MISMATCH")
        keys = {(q.question_id, q.dependency_hash) for q in self.questions}
        ids = [(d.question_id, d.dependency_hash) for d in self.dispositions]
        if len(ids) != len(set(ids)) or not set(ids) <= keys:
            raise ValueError("REQUIREMENTS_DISPOSITION_BASIS_MISMATCH")
        digest = content_hash(
            self.model_dump(mode="json", exclude={"review_ref", "review_hash"})
        )
        if (self.review_ref, self.review_hash) != (
            "requirements-review:" + digest,
            "sha256:" + digest,
        ):
            raise ValueError("REQUIREMENTS_HASH_MISMATCH")
        return self


class AssessRequirementsV1(Closed):
    expected_revision: int = Field(ge=0)
    candidate_pin: ArtifactPinV1
    source_role: Literal[
        "DESIGN_SOURCE", "ACTIVE_RULE_BASIS", "RUNTIME_TARGET", "PREPARED_SENSITIVITY"
    ] = "DESIGN_SOURCE"
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)


class ReviewRequirementV1(Closed):
    expected_revision: int = Field(ge=0)
    review_pin: ArtifactPinV1
    question_id: str
    dependency_hash: str
    decision: Literal["answer", "defer", "reject", "accept_repair"]
    rationale: str = Field(min_length=1)
    successor_candidate_pin: ArtifactPinV1 | None = None
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
