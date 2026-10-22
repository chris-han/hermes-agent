"""Versioned ChangeCandidate v2 contract.

This module extends the existing Five-Object Kernel without replacing the
persisted v1 learning contract.  It deliberately reuses the kernel's content
addressing rule and governed-delta algebra.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.governed_delta import DeltaEnvelopeV1
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import (
    ChangeAssessment,
    ChangeCandidate,
    ChangeDelta,
    ChangeObservation,
    ChangeOrigin,
    EvaluationBasis,
    _ContentAddressed,
    _ordered_unique_pins,
    _pin_key,
)


CandidateKindV2: TypeAlias = Literal[
    "LEARNING_HYPOTHESIS", "ATOMIC_CHANGE", "CHANGE_SET"
]


class _Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _ascii_identifier(value: str, field: str) -> str:
    if not value or not value.isascii():
        raise ValueError(f"{field} must be a non-empty ASCII-stable identifier")
    return value


def _version_scope_matches(
    *,
    organization_id: str,
    workspace_id: str,
    delta: DeltaEnvelopeV1,
) -> bool:
    for pin in (delta.predecessor, delta.successor_candidate):
        if pin.organization_id is not None and pin.organization_id != organization_id:
            return False
        if pin.workspace_id is not None and pin.workspace_id != workspace_id:
            return False
    return True


class LearningHypothesisProfileV2(_Profile):
    """The v1 empirical-learning payload nested under a v2 discriminator."""

    stage: Literal["PROPOSAL", "OBSERVATION", "ASSESSMENT"]
    origin: ChangeOrigin
    failure_class: Literal["CAPABILITY", "STEERING", "STRUCTURE", "ENVIRONMENT"]
    target: Literal["ONTOLOGY", "CONTEXT_FORMATION", "REASONING", "WORKFLOW", "POLICY"]
    hypothesis: str = Field(min_length=1)
    evaluation_basis: EvaluationBasis
    delta: ChangeDelta
    matched_cases: tuple[ArtifactPinV1, ...]
    independent_holdout_cases: tuple[ArtifactPinV1, ...]
    observations: tuple[ChangeObservation, ...] = ()
    assessment: ChangeAssessment | None = None
    successor_inputs: tuple[ArtifactPinV1, ...] = ()

    @field_validator(
        "matched_cases",
        "independent_holdout_cases",
        "successor_inputs",
        mode="before",
    )
    @classmethod
    def canonicalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_profile_semantics(self) -> "LearningHypothesisProfileV2":
        matched = {_pin_key(pin) for pin in self.matched_cases}
        holdout = {_pin_key(pin) for pin in self.independent_holdout_cases}
        if matched & holdout:
            raise ValueError("matched and independent holdout cases must be disjoint")
        if self.target == "ONTOLOGY" and self.failure_class != "STRUCTURE":
            raise ValueError("ontology change requires STRUCTURE failure attribution")
        if self.target == "POLICY" and self.failure_class != "STRUCTURE":
            raise ValueError("policy change requires STRUCTURE failure attribution")
        if self.delta.claims_semantic_graph_change and self.failure_class != "STRUCTURE":
            raise ValueError("semantic graph change requires STRUCTURE failure attribution")
        if self.stage == "PROPOSAL":
            if self.observations or self.assessment is not None:
                raise ValueError("proposal cannot contain future observations/assessment")
            if self.successor_inputs:
                raise ValueError("proposal cannot contain future successor inputs")
        elif self.stage == "OBSERVATION":
            if not self.observations:
                raise ValueError("observation revision requires observations")
            if self.assessment is not None:
                raise ValueError("observation revision cannot contain future assessment")
            if self.successor_inputs:
                raise ValueError("observation revision cannot contain future successor inputs")
        else:
            if not self.observations:
                raise ValueError("assessment revision requires prior observations")
            if self.assessment is None:
                raise ValueError("assessment revision requires assessment")
            if self.assessment.verdict == "BETTER":
                if not self.independent_holdout_cases:
                    raise ValueError("generalized BETTER claim requires independent holdout")
                observed = {_pin_key(item.case): item for item in self.observations}
                required = {
                    _pin_key(pin)
                    for pin in (*self.matched_cases, *self.independent_holdout_cases)
                }
                if not required <= observed.keys() or any(
                    observed[key].status != "COMPLETE" for key in required
                ):
                    raise ValueError(
                        "generalized BETTER claim requires complete cases"
                    )
        if (
            self.assessment is not None
            and self.assessment.disposition == "ACCEPTED_FOR_SUCCESSOR_INPUT"
            and not self.successor_inputs
        ):
            raise ValueError("accepted candidate requires successor input")
        if (
            self.assessment is not None
            and self.assessment.disposition != "ACCEPTED_FOR_SUCCESSOR_INPUT"
            and self.successor_inputs
        ):
            raise ValueError("non-accepted candidate cannot bind successor input")
        if any(
            pin.artifact_kind in {"context_graph", "decision_record", "change_candidate"}
            for pin in self.successor_inputs
        ):
            raise ValueError(
                "ChangeCandidate may bind successor inputs, not future semantic outputs"
            )
        return self


class AtomicChangeProfileV1(_Profile):
    owner_ref: str
    authority_domain_ref: str
    rationale: str = Field(min_length=1)
    basis_refs: tuple[ArtifactPinV1, ...]
    delta: DeltaEnvelopeV1

    @field_validator("owner_ref")
    @classmethod
    def owner_is_ascii(cls, value: str) -> str:
        return _ascii_identifier(value, "owner_ref")

    @field_validator("authority_domain_ref")
    @classmethod
    def authority_domain_is_ascii(cls, value: str) -> str:
        return _ascii_identifier(value, "authority_domain_ref")

    @field_validator("basis_refs", mode="before")
    @classmethod
    def canonicalize_basis_refs(cls, value: object) -> object:
        result = _ordered_unique_pins(value)
        if isinstance(result, tuple) and not result:
            raise ValueError("basis_refs must be non-empty")
        return result

    @model_validator(mode="after")
    def delta_owner_matches_profile(self) -> "AtomicChangeProfileV1":
        if (
            self.delta.predecessor.owner_ref != self.owner_ref
            or self.delta.successor_candidate.owner_ref != self.owner_ref
        ):
            raise ValueError("atomic change delta owner must match owner_ref")
        return self


class ChangeSetMemberV1(_Profile):
    member_id: str
    delta: DeltaEnvelopeV1
    rationale_refs: tuple[ArtifactPinV1, ...] = ()

    @field_validator("member_id")
    @classmethod
    def member_id_is_ascii(cls, value: str) -> str:
        return _ascii_identifier(value, "member_id")

    @field_validator("rationale_refs", mode="before")
    @classmethod
    def canonicalize_rationale_refs(cls, value: object) -> object:
        return _ordered_unique_pins(value)


class ChangeSetProfileV1(_Profile):
    owner_ref: str
    authority_domain_ref: str
    rationale: str = Field(min_length=1)
    basis_refs: tuple[ArtifactPinV1, ...]
    coherence_refs: tuple[ArtifactPinV1, ...]
    members: tuple[ChangeSetMemberV1, ...]

    @field_validator("owner_ref")
    @classmethod
    def owner_is_ascii(cls, value: str) -> str:
        return _ascii_identifier(value, "owner_ref")

    @field_validator("authority_domain_ref")
    @classmethod
    def authority_domain_is_ascii(cls, value: str) -> str:
        return _ascii_identifier(value, "authority_domain_ref")

    @field_validator("basis_refs", "coherence_refs", mode="before")
    @classmethod
    def canonicalize_required_refs(cls, value: object) -> object:
        result = _ordered_unique_pins(value)
        if isinstance(result, tuple) and not result:
            raise ValueError("required ChangeSet reference collection must be non-empty")
        return result

    @field_validator("members", mode="before")
    @classmethod
    def canonicalize_members(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        members = tuple(
            item if isinstance(item, ChangeSetMemberV1) else ChangeSetMemberV1.model_validate(item)
            for item in value
        )
        if len(members) < 2:
            raise ValueError("CHANGE_SET requires at least two members")
        ids = tuple(item.member_id for item in members)
        if len(ids) != len(set(ids)):
            raise ValueError("ChangeSet member_id values must be unique")
        delta_keys = tuple((item.delta.delta_id, item.delta.content_hash) for item in members)
        if len(delta_keys) != len(set(delta_keys)):
            raise ValueError("ChangeSet delta envelopes must be unique")
        return tuple(sorted(members, key=lambda item: item.member_id))

    @model_validator(mode="after")
    def member_owners_match_profile(self) -> "ChangeSetProfileV1":
        if any(
            member.delta.predecessor.owner_ref != self.owner_ref
            or member.delta.successor_candidate.owner_ref != self.owner_ref
            for member in self.members
        ):
            raise ValueError(
                "cross-owner CHANGE_SET is invalid; use independently governed "
                "ChangeCandidates plus federation projection"
            )
        return self


ChangeCandidateProfileV2: TypeAlias = (
    LearningHypothesisProfileV2 | AtomicChangeProfileV1 | ChangeSetProfileV1
)


class ChangeCandidateV2(_ContentAddressed):
    """Concrete v2 root registered under the existing change_candidate kernel kind."""

    ARTIFACT_KIND = "change_candidate"

    schema_version: Literal["semantier.change_candidate.v2"] = "semantier.change_candidate.v2"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    predecessor: ArtifactPinV1 | None = None
    candidate_kind: CandidateKindV2
    profile: ChangeCandidateProfileV2
    recorded_at: datetime

    @model_validator(mode="after")
    def validate_v2_boundary(self) -> "ChangeCandidateV2":
        if self.predecessor is not None and self.predecessor.artifact_kind != "change_candidate":
            raise ValueError("predecessor must pin change_candidate")
        if self.revision == 0 and self.predecessor is not None:
            raise ValueError("revision zero requires no predecessor")
        if self.revision > 0 and self.predecessor is None:
            raise ValueError("nonzero revision requires predecessor")

        expected_type = {
            "LEARNING_HYPOTHESIS": LearningHypothesisProfileV2,
            "ATOMIC_CHANGE": AtomicChangeProfileV1,
            "CHANGE_SET": ChangeSetProfileV1,
        }[self.candidate_kind]
        if type(self.profile) is not expected_type:
            raise ValueError("candidate_kind does not match profile")

        if isinstance(self.profile, LearningHypothesisProfileV2):
            # Reuse the existing v1 semantic validator as the compatibility oracle.
            # The constructed v1 identity is discarded; this does not rewrite history.
            ChangeCandidate.create(
                schema_version="semantier.change_candidate.v1",
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                revision=self.revision,
                stage=self.profile.stage,
                predecessor=self.predecessor,
                origin=self.profile.origin,
                failure_class=self.profile.failure_class,
                target=self.profile.target,
                hypothesis=self.profile.hypothesis,
                evaluation_basis=self.profile.evaluation_basis,
                delta=self.profile.delta,
                matched_cases=self.profile.matched_cases,
                independent_holdout_cases=self.profile.independent_holdout_cases,
                observations=self.profile.observations,
                assessment=self.profile.assessment,
                successor_inputs=self.profile.successor_inputs,
                recorded_at=self.recorded_at,
            )
        elif isinstance(self.profile, AtomicChangeProfileV1):
            if not _version_scope_matches(
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                delta=self.profile.delta,
            ):
                raise ValueError("atomic change delta scope mismatch")
        else:
            if any(
                not _version_scope_matches(
                    organization_id=self.organization_id,
                    workspace_id=self.workspace_id,
                    delta=member.delta,
                )
                for member in self.profile.members
            ):
                raise ValueError("ChangeSet member delta scope mismatch")
        return self


def validate_change_candidate_v2_revision(
    previous: ChangeCandidateV2,
    current: ChangeCandidateV2,
) -> None:
    """Validate the common v2 append-only revision boundary.

    Profile-specific semantic equivalence is deliberately qualified in later
    plan stages; W2 freezes only the shared version/profile/scope ownership.
    """

    if current.predecessor != previous.pin():
        raise ValueError("ChangeCandidate v2 revision must pin exact predecessor")
    if current.revision != previous.revision + 1:
        raise ValueError("ChangeCandidate v2 revision must advance exactly once")
    if (
        current.organization_id,
        current.workspace_id,
        current.candidate_kind,
    ) != (
        previous.organization_id,
        previous.workspace_id,
        previous.candidate_kind,
    ):
        raise ValueError("ChangeCandidate v2 revision changed scope or candidate_kind")

    if isinstance(previous.profile, LearningHypothesisProfileV2):
        if not isinstance(current.profile, LearningHypothesisProfileV2):
            raise ValueError("ChangeCandidate v2 revision changed profile")
        if (previous.profile.stage, current.profile.stage) not in {
            ("PROPOSAL", "OBSERVATION"),
            ("OBSERVATION", "ASSESSMENT"),
        }:
            raise ValueError("ChangeCandidate v2 learning stages must advance A -> B -> C")
        immutable_fields = (
            "origin",
            "failure_class",
            "target",
            "hypothesis",
            "evaluation_basis",
            "delta",
            "matched_cases",
            "independent_holdout_cases",
        )
        if any(
            getattr(previous.profile, field) != getattr(current.profile, field)
            for field in immutable_fields
        ):
            raise ValueError(
                "ChangeCandidate v2 learning proposal/evaluation basis is immutable"
            )
        if (
            previous.profile.observations
            and current.profile.observations != previous.profile.observations
        ):
            raise ValueError(
                "recorded ChangeCandidate v2 learning observations cannot be rewritten"
            )
    elif isinstance(previous.profile, (AtomicChangeProfileV1, ChangeSetProfileV1)):
        if not isinstance(current.profile, type(previous.profile)):
            raise ValueError("ChangeCandidate v2 revision changed profile")
        if (
            current.profile.owner_ref,
            current.profile.authority_domain_ref,
        ) != (
            previous.profile.owner_ref,
            previous.profile.authority_domain_ref,
        ):
            raise ValueError("ChangeCandidate v2 revision changed owner or authority domain")


__all__ = [
    "AtomicChangeProfileV1",
    "CandidateKindV2",
    "ChangeCandidateProfileV2",
    "ChangeCandidateV2",
    "ChangeSetMemberV1",
    "ChangeSetProfileV1",
    "LearningHypothesisProfileV2",
    "validate_change_candidate_v2_revision",
]
