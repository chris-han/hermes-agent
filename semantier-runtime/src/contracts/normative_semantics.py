"""Closed, source-grounded normative semantic candidate contracts."""

from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from eos.artifact_hashing import content_hash
from contracts.ontology_building import ArtifactPinV1


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
ASCII_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def timestamps_and_ids(self) -> "_Contract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        frame_id = getattr(self, "frame_id", None)
        if frame_id is not None and ASCII_ID.fullmatch(frame_id) is None:
            raise ValueError("frame_id must be an ASCII-stable machine identifier")
        return self


class SemanticPredicateV1(_Contract):
    predicate_id: str = Field(min_length=1)
    operator: Literal[
        "EQ", "IN", "EXISTS", "NOT", "AND", "OR", "GE", "GT", "LE", "LT", "DATE_ON_OR_AFTER", "DATE_ON_OR_BEFORE",
        "RATIO_LE", "RATIO_GT", "TIMESTAMP_BEFORE", "TIMESTAMP_AFTER", "CALENDAR_YEARS_AFTER"
    ]
    fact_ref: str = Field(min_length=1)
    expected_value: str | int | float | bool | tuple[str, ...] | None = None
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    comparison_fact_ref: str | None = Field(default=None, pattern=ASCII_ID.pattern,
        exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def comparison_shape(self):
        import math
        comparison = self.operator in {"RATIO_LE", "RATIO_GT", "TIMESTAMP_BEFORE", "TIMESTAMP_AFTER", "CALENDAR_YEARS_AFTER"}
        if comparison != (self.comparison_fact_ref is not None):
            raise ValueError("comparison operators require exactly one second observed fact")
        if self.operator in {"RATIO_LE", "RATIO_GT"} and (type(self.expected_value) not in {int, float}
            or not math.isfinite(self.expected_value) or self.expected_value < 0):
            raise ValueError("ratio limit must be a finite nonnegative source constant")
        if self.operator == "CALENDAR_YEARS_AFTER" and (type(self.expected_value) is not int or self.expected_value <= 0):
            raise ValueError("calendar duration must be a positive source integer")
        if self.operator.startswith("TIMESTAMP_") and self.expected_value is not None:
            raise ValueError("timestamp ordering has no literal operand")
        return self


class RuleEffectV1(_Contract):
    """Source-grounded executable normative effect; prose consequence is not authority."""

    effect_id: str = Field(pattern=ASCII_ID.pattern)
    operator: Literal["EQ", "IN", "GE", "GT", "LE", "LT"]
    fact_ref: str = Field(pattern=ASCII_ID.pattern)
    expected_value: str | int | float | bool | tuple[str, ...]
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class _Frame(_Contract):
    frame_id: str = Field(min_length=1)
    frame_hash: str = Field(pattern=SHA256_PATTERN)
    display_name: str = Field(min_length=1)
    display_name_zh: str | None = None
    epistemic_status: Literal["FOUND", "NONE_EVIDENCED", "UNRESOLVED"] = "FOUND"
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    source_region_refs: tuple[str, ...] = Field(min_length=1)


class RuleFrameV1(_Frame):
    frame_kind: Literal["rule"] = "rule"
    modality: Literal["obligation", "prohibition", "permission"]
    actor_ref: str
    object_ref: str
    conditions: tuple[SemanticPredicateV1, ...]
    consequence: str
    rule_effects: tuple[RuleEffectV1, ...] | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    exception_refs: tuple[str, ...]
    scope_refs: tuple[str, ...]
    qualification_refs: tuple[str, ...]
    temporal_condition_refs: tuple[str, ...]
    precedence_refs: tuple[str, ...]
    interpretation_required: str | None = Field(default=None, min_length=1,
        exclude_if=lambda value: value is None)


class ExceptionFrameV1(_Frame):
    frame_kind: Literal["exception"] = "exception"
    applies_to_rule_ref: str
    conditions: tuple[SemanticPredicateV1, ...]


class ScopeFrameV1(_Frame):
    frame_kind: Literal["scope"] = "scope"
    applies_to_rule_ref: str
    actor_scope: tuple[str, ...]
    object_scope: tuple[str, ...]
    conditions: tuple[SemanticPredicateV1, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None)


class QualificationFrameV1(_Frame):
    frame_kind: Literal["qualification"] = "qualification"
    qualified_term: str
    definition: str
    conditions: tuple[SemanticPredicateV1, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None)


class PrecedenceFrameV1(_Frame):
    frame_kind: Literal["precedence"] = "precedence"
    higher_priority_ref: str
    lower_priority_ref: str
    condition_refs: tuple[str, ...]


class TemporalConditionFrameV1(_Frame):
    frame_kind: Literal["temporal_condition"] = "temporal_condition"
    effective_from: str
    effective_until: str
    event_time_ref: str


class RelativeTemporalConditionFrameV1(_Frame):
    frame_kind: Literal["relative_temporal_condition"] = "relative_temporal_condition"
    definition: str = Field(min_length=1)
    conditions: tuple[SemanticPredicateV1, ...] = Field(min_length=1)


class EntityDistinctionRuleEffectV1(_Contract):
    rule_ref: str = Field(pattern=ASCII_ID.pattern)
    effect: Literal["applicability", "input_requirement"]
    conditions: tuple[SemanticPredicateV1, ...] = Field(min_length=1)


class EntityDistinctionFrameV1(_Frame):
    frame_kind: Literal["entity_distinction"] = "entity_distinction"
    left_entity_ref: str
    right_entity_ref: str
    outcome_effect: str
    rule_effects: tuple[EntityDistinctionRuleEffectV1, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None)

    @property
    def conditions(self) -> tuple[SemanticPredicateV1, ...]:
        return tuple(predicate for effect in (self.rule_effects or ()) for predicate in effect.conditions)

    @model_validator(mode="after")
    def distinct_executable_bindings(self):
        if self.rule_effects is not None:
            keys = [(item.rule_ref, item.effect) for item in self.rule_effects]
            if self.left_entity_ref == self.right_entity_ref or len(keys) != len(set(keys)):
                raise ValueError("entity distinction requires distinct endpoints and unique rule effects")
        return self


NormativeFrameV1 = Annotated[
    Union[
        RuleFrameV1,
        ExceptionFrameV1,
        ScopeFrameV1,
        QualificationFrameV1,
        PrecedenceFrameV1,
        TemporalConditionFrameV1,
        RelativeTemporalConditionFrameV1,
        EntityDistinctionFrameV1,
    ],
    Field(discriminator="frame_kind"),
]


class NormativeSemanticCorrectionReviewV1(_Contract):
    """Explicit actor-authored correction; no certification/admission authority."""
    schema_version: Literal["normative_semantic_correction_review.v1"] = "normative_semantic_correction_review.v1"
    review_ref: str
    review_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    actor_ref: str = Field(pattern=r"^user:.+$")
    reviewer_kind: Literal["human", "delegated_ai"]
    predecessor_proposal_ref: str
    predecessor_proposal_hash: str = Field(pattern=SHA256_PATTERN)
    corrected_output: dict
    rationale: str = Field(min_length=1)
    compiler_hash: str = Field(pattern=SHA256_PATTERN)
    reviewed_at: datetime
    predecessor_candidate_pin: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def exact_review_identity(self):
        digest = content_hash(self.model_dump(mode="json", exclude={"review_ref", "review_hash"}))
        if (self.review_ref, self.review_hash) != ("normative-semantic-review:" + digest, "sha256:" + digest):
            raise ValueError("normative semantic correction review hash mismatch")
        return self


class NormativeSemanticCandidateSetV1(_Contract):
    teaching_candidate: Literal[True] | None = Field(default=None, exclude_if=lambda value: value is None)
    schema_version: Literal["normative_semantic_candidate_set.v1"] = (
        "normative_semantic_candidate_set.v1"
    )
    candidate_set_ref: str = Field(min_length=1)
    candidate_set_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    source_identity_ref: str = Field(min_length=1)
    source_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_source_ir_ref: str = Field(min_length=1)
    canonical_source_ir_hash: str = Field(pattern=SHA256_PATTERN)
    competency_contract_ref: str = Field(min_length=1)
    competency_contract_hash: str = Field(pattern=SHA256_PATTERN)
    normative_schema_ref: str = Field(min_length=1)
    normative_schema_hash: str = Field(pattern=SHA256_PATTERN)
    profile_ref: str = Field(min_length=1)
    profile_hash: str = Field(pattern=SHA256_PATTERN)
    compiler_ref: str = Field(min_length=1)
    compiler_hash: str = Field(pattern=SHA256_PATTERN)
    prompt_hash: str = Field(pattern=SHA256_PATTERN)
    provider_ref: str = Field(min_length=1)
    model_ref: str = Field(min_length=1)
    config_hash: str = Field(pattern=SHA256_PATTERN)
    proposal_evidence_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)
    proposal_evidence_hash: str | None = Field(default=None, pattern=SHA256_PATTERN,
                                             exclude_if=lambda value: value is None)
    source_review_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)
    source_review_hash: str | None = Field(default=None, pattern=SHA256_PATTERN, exclude_if=lambda value: value is None)
    frames: tuple[NormativeFrameV1, ...] = Field(min_length=1)
    completeness_state: Literal["COMPLETE", "UNRESOLVED"]
    unresolved_slot_refs: tuple[str, ...]
    predecessor_candidate_set_ref: str | None = None
    predecessor_candidate_set_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    change_cause: Literal["initial_compilation", "bounded_repair", "runtime_feedback"] = "initial_compilation"
    triggering_ref: str | None = None
    compiled_at: datetime

    @model_validator(mode="after")
    def exact_identity_and_ordering(self) -> "NormativeSemanticCandidateSetV1":
        if (self.proposal_evidence_ref is None) != (self.proposal_evidence_hash is None):
            raise ValueError("proposal evidence ref/hash must be supplied together")
        if (self.source_review_ref is None) != (self.source_review_hash is None):
            raise ValueError("source review ref/hash must be supplied together")
        ids = [frame.frame_id for frame in self.frames]
        if len(ids) != len(set(ids)):
            raise ValueError("normative frame ids must be unique")
        if ids != sorted(ids, key=lambda item: item.encode("utf-8")):
            raise ValueError("normative frames must use canonical ordering")
        if self.completeness_state == "COMPLETE" and self.unresolved_slot_refs:
            raise ValueError("complete candidate set cannot retain unresolved slots")
        if (self.predecessor_candidate_set_ref is None) != (
            self.predecessor_candidate_set_hash is None
        ):
            raise ValueError("predecessor candidate ref/hash must be supplied together")
        if self.change_cause != "initial_compilation" and self.predecessor_candidate_set_ref is None:
            raise ValueError("candidate successor requires predecessor ref/hash")
        return self


class NormativeSemanticProposalEvidenceV1(_Contract):
    """Immutable source proposal receipt, without certification authority."""

    schema_version: Literal["normative_semantic_proposal_evidence.v1"] = "normative_semantic_proposal_evidence.v1"
    evidence_ref: str
    evidence_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    instructions: str = Field(min_length=1)
    input_text: str = Field(min_length=1)
    output_schema: dict
    output: dict
    provider_input_text: str | None = Field(default=None, exclude_if=lambda value: value is None)
    rejected_response_text: str | None = Field(default=None, exclude_if=lambda value: value is None)
    usage: dict | None = Field(default=None, exclude_if=lambda value: value is None)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    compiler_ref: str = Field(min_length=1)
    compiler_hash: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime

    @model_validator(mode="after")
    def exact_content_identity(self) -> "NormativeSemanticProposalEvidenceV1":
        digest = content_hash(self.model_dump(mode="json", exclude={"evidence_ref", "evidence_hash"}))
        if (self.evidence_ref, self.evidence_hash) != (
            "normative-semantic-proposal:" + digest, "sha256:" + digest,
        ):
            raise ValueError("normative semantic proposal evidence hash mismatch")
        return self


__all__ = [
    "NormativeSemanticProposalEvidenceV1",
    "EntityDistinctionFrameV1",
    "EntityDistinctionRuleEffectV1",
    "ExceptionFrameV1",
    "NormativeFrameV1",
    "NormativeSemanticCandidateSetV1",
    "PrecedenceFrameV1",
    "QualificationFrameV1",
    "RuleEffectV1",
    "RuleFrameV1",
    "ScopeFrameV1",
    "SemanticPredicateV1",
    "TemporalConditionFrameV1",
    "RelativeTemporalConditionFrameV1",
]
