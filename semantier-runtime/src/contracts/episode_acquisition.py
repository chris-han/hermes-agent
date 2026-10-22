"""Deterministic W2 acquisition decision over one governed episode."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from eos.artifact_hashing import content_hash

AcquisitionReason = Literal[
    "TYPED_FAILURE",
    "CONFLICT",
    "PROJECTION_INSUFFICIENT",
    "REPRESENTATION_INSUFFICIENT",
    "MODEL_ENVELOPE_INSUFFICIENT",
    "FRONTIER_INSUFFICIENT",
    "NOVELTY",
    "UNCERTAINTY",
    "PROSPECTIVE_IMPACT",
    "COMMITMENT_DIVERGENCE",
    "RANDOM_AUDIT",
]

_SHA = r"^sha256:[0-9a-f]{64}$"


def _digest(value) -> str:
    return "sha256:" + content_hash(value)


class AcquisitionPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["acquisition_policy.v1"] = "acquisition_policy.v1"
    policy_ref: str
    policy_hash: str = Field(pattern=_SHA)
    uncertainty_threshold: float = Field(ge=0.0, le=1.0)
    random_audit_rate: float = Field(ge=0.0, le=1.0)
    random_audit_budget: int = Field(ge=0)
    seed: str = Field(min_length=1)

    @model_validator(mode="after")
    def verify_identity(self):
        digest = _digest(self.model_dump(mode="json", exclude={"policy_ref", "policy_hash"}))
        if (self.policy_ref, self.policy_hash) != (
            "acquisition-policy:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("acquisition policy identity mismatch")
        return self

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            policy_ref="acquisition-policy:v1:pending",
            policy_hash="sha256:" + "0" * 64,
            **body,
        )
        payload = provisional.model_dump(mode="json", exclude={"policy_ref", "policy_hash"})
        digest = _digest(payload)
        return cls.model_validate({
            **body,
            "policy_ref": "acquisition-policy:v1:" + digest[7:],
            "policy_hash": digest,
        })


class AcquisitionSignalSetV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["acquisition_signal_set.v1"] = "acquisition_signal_set.v1"
    signal_ref: str
    signal_hash: str = Field(pattern=_SHA)
    typed_failure: bool = False
    conflict: bool = False
    projection_insufficient: bool = False
    representation_insufficient: bool = False
    model_envelope_insufficient: bool = False
    frontier_insufficient: bool = False
    novelty: bool = False
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    prospective_impact: bool = False
    commitment_divergence: bool = False
    protected_holdout: bool = False
    already_labeled: bool = False

    @model_validator(mode="after")
    def verify_identity(self):
        digest = _digest(self.model_dump(mode="json", exclude={"signal_ref", "signal_hash"}))
        if (self.signal_ref, self.signal_hash) != (
            "acquisition-signals:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("acquisition signal identity mismatch")
        return self

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            signal_ref="acquisition-signals:v1:pending",
            signal_hash="sha256:" + "0" * 64,
            **body,
        )
        payload = provisional.model_dump(mode="json", exclude={"signal_ref", "signal_hash"})
        digest = _digest(payload)
        return cls.model_validate({
            **body,
            "signal_ref": "acquisition-signals:v1:" + digest[7:],
            "signal_hash": digest,
        })


class AcquisitionDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["acquisition_decision.v1"] = "acquisition_decision.v1"
    receipt_ref: str
    receipt_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    episode_ref: str = Field(min_length=1)
    episode_hash: str = Field(pattern=_SHA)
    policy_ref: str = Field(min_length=1)
    policy_hash: str = Field(pattern=_SHA)
    signal_ref: str = Field(min_length=1)
    signal_hash: str = Field(pattern=_SHA)
    audit_index: int = Field(ge=0)
    eligible: bool
    reasons: tuple[AcquisitionReason, ...]
    excluded_reason: Literal["PROTECTED_HOLDOUT", "ALREADY_LABELED"] | None = None
    random_audit_rank: int | None = Field(default=None, ge=0)
    random_audit_selected: bool = False

    @model_validator(mode="after")
    def verify(self):
        canonical = tuple(sorted(set(self.reasons), key=lambda value: value.encode("utf-8")))
        if canonical != self.reasons:
            raise ValueError("acquisition reasons must be unique canonical order")
        if self.excluded_reason is not None and self.eligible:
            raise ValueError("excluded episode cannot be acquisition eligible")
        if self.eligible != bool(self.reasons) and self.excluded_reason is None:
            raise ValueError("acquisition eligibility must match persisted reasons")
        if self.random_audit_selected != ("RANDOM_AUDIT" in self.reasons):
            raise ValueError("random audit receipt/reason mismatch")
        digest = _digest(self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"}))
        if (self.receipt_ref, self.receipt_hash) != (
            "acquisition-decision:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("acquisition decision identity mismatch")
        return self

    @classmethod
    def create(cls, **body):
        body = {
            **body,
            "reasons": tuple(sorted(set(body.get("reasons", ())), key=lambda value: value.encode("utf-8"))),
        }
        provisional = cls.model_construct(
            receipt_ref="acquisition-decision:v1:pending",
            receipt_hash="sha256:" + "0" * 64,
            **body,
        )
        payload = provisional.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"})
        digest = _digest(payload)
        return cls.model_validate({
            **body,
            "receipt_ref": "acquisition-decision:v1:" + digest[7:],
            "receipt_hash": digest,
        })
