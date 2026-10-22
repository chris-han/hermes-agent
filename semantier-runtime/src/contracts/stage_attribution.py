"""Derived W3 stage-attribution and bounded structural-hypothesis contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.artifact_pin import ArtifactPinV1
from contracts.candidate_stage_binding import CandidateStageBindingV1, OwningStage
from contracts.semantic_kernel import ChangeDelta
from eos.artifact_hashing import content_hash

_SHA = r"^sha256:[0-9a-f]{64}$"
FailureClass = Literal["CAPABILITY", "STEERING", "STRUCTURE", "ENVIRONMENT"]


def _digest(value) -> str:
    return "sha256:" + content_hash(value)


class StageAttributionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["stage_attribution.v1"] = "stage_attribution.v1"
    attribution_ref: str
    attribution_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    episode: ArtifactPinV1
    acquisition: ArtifactPinV1
    institutional_adaptation_ingress: ArtifactPinV1 | None = None
    failure_class: FailureClass
    stage_binding: CandidateStageBindingV1
    evidence: tuple[ArtifactPinV1, ...]
    observable: bool
    intervenable: bool
    outcome_sensitive: bool
    cross_episode_reusable: bool
    diagnostics: tuple[str, ...] = ()

    @property
    def hypothesis_eligible(self) -> bool:
        return (
            self.observable
            and self.intervenable
            and self.outcome_sensitive
            and self.cross_episode_reusable
        )

    @model_validator(mode="after")
    def verify(self):
        if self.episode.artifact_kind != "governed_decision_episode":
            raise ValueError("ATTRIBUTION_REQUIRES_GOVERNED_EPISODE")
        if self.acquisition.artifact_kind != "acquisition_decision":
            raise ValueError("ATTRIBUTION_REQUIRES_ACQUISITION_DECISION")
        if (
            self.institutional_adaptation_ingress is not None
            and self.institutional_adaptation_ingress.artifact_kind
            != "institutional_adaptation_ingress"
        ):
            raise ValueError("ATTRIBUTION_ADAPTATION_INGRESS_KIND")
        if self.failure_class != self.stage_binding.failure_class:
            raise ValueError("ATTRIBUTION_FAILURE_CLASS_BINDING_MISMATCH")
        if not self.evidence:
            raise ValueError("ATTRIBUTION_REQUIRES_EVIDENCE")
        if not self.hypothesis_eligible and not self.diagnostics:
            raise ValueError("INELIGIBLE_ATTRIBUTION_REQUIRES_DIAGNOSTICS")
        digest = _digest(self.hash_payload())
        if (self.attribution_ref, self.attribution_hash) != (
            "stage-attribution:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("stage attribution identity mismatch")
        return self

    def hash_payload(self):
        return self.model_dump(mode="json", exclude={"attribution_ref", "attribution_hash"})

    @classmethod
    def create(cls, **body):
        evidence = tuple(sorted(
            set(body.get("evidence", ())),
            key=lambda pin: (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash),
        ))
        diagnostics = tuple(sorted(set(body.get("diagnostics", ())), key=lambda v: v.encode("utf-8")))
        body = {**body, "evidence": evidence, "diagnostics": diagnostics}
        provisional = cls.model_construct(
            attribution_ref="stage-attribution:v1:pending",
            attribution_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "attribution_ref": "stage-attribution:v1:" + digest[7:],
            "attribution_hash": digest,
        })

    def pin(self):
        return ArtifactPinV1(
            artifact_kind="stage_attribution",
            artifact_ref=self.attribution_ref,
            artifact_hash=self.attribution_hash,
        )


class OptimizerProposalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis: str = Field(min_length=1)
    delta: ChangeDelta
    search_lane: Literal["REFINE", "ALTERNATIVE"]
    optimizer_ref: str = Field(min_length=1)
    optimizer_hash: str = Field(pattern=_SHA)
    optimizer_config: ArtifactPinV1
    budget: ArtifactPinV1


class StructuralHypothesisV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["structural_hypothesis.v1"] = "structural_hypothesis.v1"
    hypothesis_ref: str
    hypothesis_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    attribution: ArtifactPinV1
    owning_stage: OwningStage
    stage_binding: CandidateStageBindingV1
    selected_episodes: tuple[ArtifactPinV1, ...]
    optional_human_evidence: tuple[ArtifactPinV1, ...] = ()
    predecessor_candidate: ArtifactPinV1 | None = None
    allowed_candidate_delta_schema: ArtifactPinV1
    proposal: OptimizerProposalV1

    @model_validator(mode="after")
    def verify(self):
        if self.attribution.artifact_kind != "stage_attribution":
            raise ValueError("STRUCTURAL_HYPOTHESIS_REQUIRES_ATTRIBUTION")
        if self.owning_stage != self.stage_binding.owning_stage:
            raise ValueError("STRUCTURAL_HYPOTHESIS_STAGE_MISMATCH")
        if not self.selected_episodes:
            raise ValueError("STRUCTURAL_HYPOTHESIS_REQUIRES_EPISODES")
        if any(pin.artifact_kind != "governed_decision_episode" for pin in self.selected_episodes):
            raise ValueError("STRUCTURAL_HYPOTHESIS_EPISODE_KIND")
        if self.predecessor_candidate is not None and self.predecessor_candidate.artifact_kind != "change_candidate":
            raise ValueError("STRUCTURAL_HYPOTHESIS_PREDECESSOR_KIND")
        if self.allowed_candidate_delta_schema.artifact_kind != "candidate_delta_schema":
            raise ValueError("STRUCTURAL_HYPOTHESIS_DELTA_SCHEMA_KIND")
        digest = _digest(self.hash_payload())
        if (self.hypothesis_ref, self.hypothesis_hash) != (
            "structural-hypothesis:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("structural hypothesis identity mismatch")
        return self

    def hash_payload(self):
        return self.model_dump(mode="json", exclude={"hypothesis_ref", "hypothesis_hash"})

    @classmethod
    def create(cls, **body):
        for field in ("selected_episodes", "optional_human_evidence"):
            values = tuple(sorted(
                set(body.get(field, ())),
                key=lambda pin: (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash),
            ))
            body = {**body, field: values}
        provisional = cls.model_construct(
            hypothesis_ref="structural-hypothesis:v1:pending",
            hypothesis_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "hypothesis_ref": "structural-hypothesis:v1:" + digest[7:],
            "hypothesis_hash": digest,
        })

    def pin(self):
        return ArtifactPinV1(
            artifact_kind="structural_hypothesis",
            artifact_ref=self.hypothesis_ref,
            artifact_hash=self.hypothesis_hash,
        )
