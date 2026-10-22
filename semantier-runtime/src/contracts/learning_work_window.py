"""W6 unified learning-runtime work-window contract across Realtime/Sleeptime/Slow Clock."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.artifact_pin import ArtifactPinV1
from eos.artifact_hashing import content_hash

_SHA = r"^sha256:[0-9a-f]{64}$"

ExecutionMode = Literal["REALTIME", "SLEEPTIME"]
LearningClock = Literal["MEDIUM", "SLOW"]
LearningLoop = Literal["OBJECT_LEARNING", "LEARNING_SYSTEM_ADAPTATION"]
EligibilityBasis = Literal["DURABLE_EVENT", "CONSOLIDATION", "ACCUMULATED_LOOP1"]

_ALLOWED_STAGES = ("CAPTURE", "ACQUIRE", "PRE_LEARNING_TRIAGE", "ATTRIBUTE", "PROPOSE", "ASSESS")
_FORBIDDEN_STAGES = ("ADMIT", "ACTIVATE", "COMMITMENT_TRANSITION")


def _digest(value) -> str:
    return "sha256:" + content_hash(value)


class LearningWorkWindowV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["learning_work_window.v1"] = "learning_work_window.v1"
    window_ref: str
    window_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)

    execution_mode: ExecutionMode
    clock: LearningClock
    loop: LearningLoop
    eligibility_basis: EligibilityBasis

    episodes: tuple[ArtifactPinV1, ...]
    acquisitions: tuple[ArtifactPinV1, ...] = ()
    attributions: tuple[ArtifactPinV1, ...] = ()
    comparison_receipts: tuple[ArtifactPinV1, ...] = ()
    independent_evaluation_basis: ArtifactPinV1 | None = None

    scheduler_wake: ArtifactPinV1 | None = None
    checkpoint: ArtifactPinV1
    resource_policy: ArtifactPinV1
    foreground_decoupled: bool = True

    allowed_stages: tuple[str, ...] = _ALLOWED_STAGES
    forbidden_stages: tuple[str, ...] = _FORBIDDEN_STAGES

    @model_validator(mode="after")
    def verify(self):
        if not self.foreground_decoupled:
            raise ValueError("LEARNING_WINDOW_MUST_BE_FOREGROUND_DECOUPLED")
        if self.allowed_stages != _ALLOWED_STAGES or self.forbidden_stages != _FORBIDDEN_STAGES:
            raise ValueError("UNTRUSTED_LEARNING_TOPOLOGY")
        if not self.episodes:
            raise ValueError("LEARNING_WINDOW_REQUIRES_DURABLE_EPISODE")
        if any(pin.artifact_kind != "governed_decision_episode" for pin in self.episodes):
            raise ValueError("LEARNING_WINDOW_EPISODE_KIND")
        if any(pin.artifact_kind != "acquisition_decision" for pin in self.acquisitions):
            raise ValueError("LEARNING_WINDOW_ACQUISITION_KIND")
        if any(pin.artifact_kind != "stage_attribution" for pin in self.attributions):
            raise ValueError("LEARNING_WINDOW_ATTRIBUTION_KIND")
        if any(pin.artifact_kind != "comparison_compatibility" for pin in self.comparison_receipts):
            raise ValueError("LEARNING_WINDOW_COMPARISON_KIND")

        if self.execution_mode == "REALTIME":
            if self.clock != "MEDIUM" or self.loop != "OBJECT_LEARNING":
                raise ValueError("REALTIME_IS_MEDIUM_OBJECT_LEARNING_ONLY")
            if self.eligibility_basis != "DURABLE_EVENT":
                raise ValueError("REALTIME_REQUIRES_DURABLE_EVENT")
            if self.scheduler_wake is not None:
                raise ValueError("REALTIME_CANNOT_REQUIRE_SCHEDULER_WAKE")

        if self.clock == "SLOW":
            if self.execution_mode != "SLEEPTIME":
                raise ValueError("SLOW_CLOCK_REQUIRES_SLEEPTIME_EXECUTION")
            if self.loop != "LEARNING_SYSTEM_ADAPTATION":
                raise ValueError("SLOW_CLOCK_REQUIRES_LOOP2")
            if self.eligibility_basis != "ACCUMULATED_LOOP1":
                raise ValueError("SLOW_CLOCK_REQUIRES_ACCUMULATED_LOOP1")
            if len(self.episodes) < 2 or len(self.attributions) < 2:
                raise ValueError("SLOW_CLOCK_REQUIRES_REPEATED_LOOP1_EVIDENCE")
            if not self.comparison_receipts:
                raise ValueError("SLOW_CLOCK_REQUIRES_COMPARABLE_HISTORY")
            if self.independent_evaluation_basis is None:
                raise ValueError("SLOW_CLOCK_REQUIRES_INDEPENDENT_EVALUATION_BASIS")
        else:
            if self.loop != "OBJECT_LEARNING":
                raise ValueError("MEDIUM_CLOCK_REQUIRES_LOOP1")
            if self.execution_mode == "SLEEPTIME" and self.eligibility_basis != "CONSOLIDATION":
                raise ValueError("SLEEPTIME_MEDIUM_REQUIRES_CONSOLIDATION")

        digest = _digest(self.hash_payload())
        if (self.window_ref, self.window_hash) != (
            "learning-work-window:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("learning work window identity mismatch")
        return self

    def hash_payload(self):
        return self.model_dump(mode="json", exclude={"window_ref", "window_hash"})

    @classmethod
    def create(cls, **body):
        for field in ("episodes", "acquisitions", "attributions", "comparison_receipts"):
            values = tuple(sorted(
                set(body.get(field, ())),
                key=lambda pin: (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash),
            ))
            body = {**body, field: values}
        provisional = cls.model_construct(
            window_ref="learning-work-window:v1:pending",
            window_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "window_ref": "learning-work-window:v1:" + digest[7:],
            "window_hash": digest,
        })

    def pin(self):
        return ArtifactPinV1(
            artifact_kind="learning_work_window",
            artifact_ref=self.window_ref,
            artifact_hash=self.window_hash,
        )
