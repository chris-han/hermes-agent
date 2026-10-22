"""Embedded review values. No kernel family, admission or latest-result state."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from contracts.artifact_pin import ArtifactPinV1
from contracts.decision_graph_projection import PredicateEpistemicRequirementV1
from eos.artifact_hashing import content_hash

Sha = r"^sha256:[0-9a-f]{64}$"


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def aware_utc(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and value.utcoffset() != timedelta(0):
                raise ValueError("review time must be aware UTC")
        return self


class CandidateReviewDimensionV1(_Value):
    dimension_id: str = Field(pattern=r"^[\x21-\x7e]+$")
    required_evidence_pins: tuple[ArtifactPinV1, ...] = ()
    evidence_role: str | None = None
    min_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def declared_support(self):
        if not self.required_evidence_pins and not self.evidence_role:
            raise ValueError("review dimensions require an explicit support definition")
        if any(p.artifact_kind != "evidence" for p in self.required_evidence_pins):
            raise ValueError("support must be Evidence")
        if len(set(self.required_evidence_pins)) != len(self.required_evidence_pins):
            raise ValueError("duplicate evidence support")
        return self


class CandidateReviewRequirementV1(_Value):
    requirement: PredicateEpistemicRequirementV1
    mode: Literal["COMPUTABLE", "INTERPRETED"]
    check_kind: Literal["PREIMAGE", "EVIDENCE", "SEMANTIC"]
    dimensions: tuple[CandidateReviewDimensionV1, ...] = ()
    basis_class: Literal["CORE_MECHANICAL", "DEVELOPMENT", "UNAVAILABLE"]
    evaluator_pin: ArtifactPinV1 | None = None
    qualification_pin: ArtifactPinV1 | None = None

    @model_validator(mode="after")
    def bound_dimensions(self):
        ids = tuple(d.dimension_id for d in self.dimensions)
        if ids != self.requirement.required_semantic_dimension_ids:
            raise ValueError("requirement dimension definition mismatch")
        if not self.requirement.predicate_id.isascii():
            raise ValueError("predicate id must be ASCII")
        if (self.mode == "INTERPRETED") != (self.check_kind == "SEMANTIC"):
            raise ValueError("review check mode mismatch")
        if self.basis_class == "UNAVAILABLE" and self.dimensions:
            raise ValueError("unavailable policy cannot invent dimensions")
        if self.basis_class != "UNAVAILABLE" and not self.dimensions:
            raise ValueError("empty requirements cannot pass")
        return self

    def rule_pin(self):
        digest = content_hash(self.model_dump(mode="json"))
        return ArtifactPinV1(
            artifact_kind="candidate_review_requirement",
            artifact_ref="candidate-review-requirement:" + digest,
            artifact_hash="sha256:" + digest,
        )


class CandidateReviewEvidenceV1(_Value):
    pin: ArtifactPinV1
    evidence_role: str
    source_hash: str = Field(pattern=Sha)
    selector_hash: str = Field(pattern=Sha)


class CandidateReviewMemberV1(_Value):
    kind: Literal["node", "edge", "assertion"]
    object_ref: str
    content_hash: str = Field(pattern=Sha)
    evidence_pins: tuple[ArtifactPinV1, ...] = ()


class CandidateReviewMaterialV1(_Value):
    cursor_ref: str
    head_pin: ArtifactPinV1
    revision: int = Field(ge=0)
    projection_basis_hash: str = Field(pattern=Sha)
    session_id: str
    actor_ref: str
    selection_pin: ArtifactPinV1
    design_binding_set_pin: ArtifactPinV1
    runtime_binding_set_pin: ArtifactPinV1
    preparation_pin: ArtifactPinV1 | None = None

    @classmethod
    def from_head(cls, head, *, preparation_pin=None):
        return cls(
            cursor_ref=head.cursor_ref,
            head_pin=ArtifactPinV1(
                artifact_kind="workbench_projection_cursor_head",
                artifact_ref=head.cursor_head_ref,
                artifact_hash=head.cursor_head_hash,
            ),
            revision=head.revision,
            projection_basis_hash=head.projection_basis_hash,
            session_id=head.session_id,
            actor_ref=head.actor_ref,
            selection_pin=head.selection_pin,
            design_binding_set_pin=head.design_binding_set_pin,
            runtime_binding_set_pin=head.runtime_binding_set_pin,
            preparation_pin=preparation_pin,
        )


class CandidateReviewProjectionV1(_Value):
    schema_version: Literal["candidate_review_projection.v1"] = (
        "candidate_review_projection.v1"
    )
    organization_id: str
    workspace_id: str
    proposal_pin: ArtifactPinV1
    proposal_delta_hash: str = Field(pattern=Sha)
    baseline_graph_pin: ArtifactPinV1
    baseline_state: Literal["PROPOSED", "ADMITTED"]
    ontology_pin: ArtifactPinV1
    hypothesis_role: Literal["HYPOTHESIS"] = "HYPOTHESIS"
    proposed_edit_json: str
    original_material_pins: tuple[ArtifactPinV1, ...]
    original_embedded_hash: str = Field(pattern=Sha)
    selected_evidence_pins: tuple[ArtifactPinV1, ...]
    supporting_evidence: tuple[CandidateReviewEvidenceV1, ...]
    unresolved_evidence_pins: tuple[ArtifactPinV1, ...] = ()
    closure_members: tuple[CandidateReviewMemberV1, ...]
    requirements: tuple[CandidateReviewRequirementV1, ...]
    policy_definition: str
    policy_pin: ArtifactPinV1
    compiler_pin: ArtifactPinV1
    evaluated_material_basis: CandidateReviewMaterialV1 | None = None
    projection_hash: str = Field(pattern=Sha)

    @model_validator(mode="after")
    def identity(self):
        if self.projection_hash != "sha256:" + content_hash(
            self.model_dump(mode="json", exclude={"projection_hash"})
        ):
            raise ValueError("REVIEW_PROJECTION_HASH_MISMATCH")
        ids = tuple(r.requirement.predicate_id for r in self.requirements)
        if not ids or ids != tuple(sorted(set(ids))):
            raise ValueError("review predicates must be nonempty and ordered")
        if self.policy_pin.artifact_hash != "sha256:" + content_hash(
            json.loads(self.policy_definition)
        ):
            raise ValueError("REVIEW_POLICY_HASH_MISMATCH")
        if (
            self.proposal_pin.artifact_kind != "change_candidate"
            or self.baseline_graph_pin.artifact_kind != "context_graph"
            or self.ontology_pin.artifact_kind != "ontology"
        ):
            raise ValueError("review input kinds mismatch")
        return self

    @classmethod
    def create(cls, **values):
        draft = cls.model_construct(projection_hash="sha256:" + "0" * 64, **values)
        payload = draft.model_dump(mode="json", exclude={"projection_hash"})
        return cls.model_validate(
            {**payload, "projection_hash": "sha256:" + content_hash(payload)}
        )


class CandidateReviewPredicateV1(_Value):
    predicate_id: str
    rule_pin: ArtifactPinV1
    required_dimensions: tuple[str, ...]
    available_dimensions: tuple[str, ...]
    missing_dimensions: tuple[str, ...]
    admissibility: Literal["ADMISSIBLE", "INADMISSIBLE", "NOT_ESTABLISHED"]
    execution: Literal["RAN", "NOT_RUN", "FAILED"]
    reason: str
    deterministic_result: Literal["PASS", "FAIL"] | None = None
    semantic_verdict: (
        Literal["SEMANTIC_TRUE", "SEMANTIC_FALSE", "SEMANTIC_UNKNOWN"] | None
    ) = None
    failure_kind: Literal["PROVIDER", "EXECUTION", "UNSUPPORTED_CONTRACT"] | None = None
    invocation_count: int = Field(ge=0)
    evidence_pins: tuple[ArtifactPinV1, ...] = ()

    @model_validator(mode="after")
    def no_false_pass(self):
        if self.missing_dimensions != tuple(
            d for d in self.required_dimensions if d not in self.available_dimensions
        ):
            raise ValueError("review missing dimensions mismatch")
        if self.execution != "RAN" and (
            self.deterministic_result is not None or self.semantic_verdict is not None
        ):
            raise ValueError("unexecuted review cannot carry a verdict")
        if self.execution == "NOT_RUN" and self.invocation_count:
            raise ValueError("NOT_RUN cannot fabricate an invocation")
        if self.admissibility != "ADMISSIBLE" and self.execution != "NOT_RUN":
            raise ValueError("inadmissible review cannot execute")
        if self.execution == "RAN" and (
            self.invocation_count != 1
            or (self.deterministic_result is None) == (self.semantic_verdict is None)
        ):
            raise ValueError("execution requires exactly one actual result")
        if self.failure_kind == "PROVIDER" and self.execution != "FAILED":
            raise ValueError("provider failure is not a semantic verdict")
        return self


class CandidateReviewResultV1(_Value):
    schema_version: Literal["candidate_review_result.v1"] = "candidate_review_result.v1"
    projection: CandidateReviewProjectionV1
    predicates: tuple[CandidateReviewPredicateV1, ...]
    closure_status: Literal["COMPLETE", "INCOMPLETE", "NOT_ESTABLISHED"]
    human_outcome: Literal["ACCEPTED", "REJECTED", "NEEDS_MORE_EVIDENCE"]
    human_rationale: str
    human_role: Literal["HUMAN_STATEMENT"] = "HUMAN_STATEMENT"
    request_hash: str = Field(pattern=Sha)
    envelope_hash: str = Field(pattern=Sha)
    evaluated_at: datetime
    checked_at: datetime
    delivery_status: Literal["CURRENT_AS_OF_PUBLICATION", "SUPERSEDED"]
    superseded_reasons: tuple[str, ...] = ()
    result_hash: str = Field(pattern=Sha)

    @model_validator(mode="after")
    def identity_and_partition(self):
        if self.result_hash != "sha256:" + content_hash(
            self.model_dump(mode="json", exclude={"result_hash"})
        ):
            raise ValueError("REVIEW_RESULT_HASH_MISMATCH")
        if tuple(p.predicate_id for p in self.predicates) != tuple(
            r.requirement.predicate_id for r in self.projection.requirements
        ):
            raise ValueError("review predicate partition mismatch")
        for p, r in zip(self.predicates, self.projection.requirements):
            if (
                p.rule_pin != r.rule_pin()
                or p.required_dimensions
                != r.requirement.required_semantic_dimension_ids
            ):
                raise ValueError("review predicate requirement mismatch")
        expected = (
            "INCOMPLETE"
            if any(p.admissibility == "INADMISSIBLE" for p in self.predicates)
            else "NOT_ESTABLISHED"
            if any(p.admissibility == "NOT_ESTABLISHED" for p in self.predicates)
            else "COMPLETE"
        )
        if self.closure_status != expected or (
            self.delivery_status == "SUPERSEDED"
        ) != bool(self.superseded_reasons):
            raise ValueError("review status mismatch")
        return self

    @classmethod
    def create(cls, **values):
        draft = cls.model_construct(result_hash="sha256:" + "0" * 64, **values)
        payload = draft.model_dump(mode="json", exclude={"result_hash"})
        return cls.model_validate(
            {**payload, "result_hash": "sha256:" + content_hash(payload)}
        )


class CandidateReviewDisclosureV1(_Value):
    decision_pin: ArtifactPinV1
    result: CandidateReviewResultV1
    delivery_status: Literal["CURRENT_AS_OF_PUBLICATION", "SUPERSEDED"]
    superseded_reasons: tuple[str, ...] = ()
    checked_at: datetime

    @model_validator(mode="after")
    def current_use(self):
        if self.decision_pin.artifact_kind != "decision_record" or (
            self.delivery_status == "SUPERSEDED"
        ) != bool(self.superseded_reasons):
            raise ValueError("review disclosure mismatch")
        return self
