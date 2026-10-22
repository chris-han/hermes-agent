"""Derived commitment lifecycle contracts for the responsibility-anchored MVL."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.artifact_pin import ArtifactPinV1
from eos.artifact_hashing import content_hash

_SHA = r"^sha256:[0-9a-f]{64}$"


def _digest(value) -> str:
    return "sha256:" + content_hash(value)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CommitmentProposalV1(_Frozen):
    schema_version: Literal["commitment_proposal.v1"] = "commitment_proposal.v1"
    proposal_ref: str = Field(min_length=1)
    proposal_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    decision: ArtifactPinV1
    commitment_subject: ArtifactPinV1
    responsibility_anchor: ArtifactPinV1
    authority_basis_hash: str = Field(pattern=_SHA)
    scope_ref: str = Field(min_length=1)
    effective_from: datetime
    effective_until: datetime | None = None
    proposed_at: datetime

    @model_validator(mode="after")
    def verify(self):
        if self.decision.artifact_kind != "decision_record":
            raise ValueError("commitment proposal requires decision_record")
        if self.responsibility_anchor.artifact_kind != "human_responsibility_context":
            raise ValueError("commitment proposal requires responsibility anchor")
        if self.effective_until is not None and self.effective_until <= self.effective_from:
            raise ValueError("commitment effective_until must follow effective_from")
        digest = _digest(self.hash_payload())
        if (self.proposal_ref, self.proposal_hash) != (
            "commitment-proposal:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("commitment proposal identity mismatch")
        return self

    def hash_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"proposal_ref", "proposal_hash"})

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            proposal_ref="commitment-proposal:v1:pending",
            proposal_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "proposal_ref": "commitment-proposal:v1:" + digest[7:],
            "proposal_hash": digest,
        })

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="commitment_proposal",
            artifact_ref=self.proposal_ref,
            artifact_hash=self.proposal_hash,
        )


class CommitmentTransitionV1(_Frozen):
    schema_version: Literal["commitment_transition.v1"] = "commitment_transition.v1"
    transition_ref: str = Field(min_length=1)
    transition_hash: str = Field(pattern=_SHA)
    proposal: ArtifactPinV1
    predecessor: ArtifactPinV1 | None = None
    state: Literal["ADMITTED", "ACTIVE", "SUPERSEDED", "REVOKED"]
    decision: ArtifactPinV1
    responsibility_anchor: ArtifactPinV1
    authority_basis_hash: str = Field(pattern=_SHA)
    scope_ref: str = Field(min_length=1)
    effective_from: datetime
    effective_until: datetime | None = None
    replacement_ref: str | None = None
    cause: str | None = None
    transitioned_at: datetime

    @model_validator(mode="after")
    def verify(self):
        if self.proposal.artifact_kind != "commitment_proposal":
            raise ValueError("commitment transition requires commitment proposal")
        if self.predecessor is not None and self.predecessor.artifact_kind != "commitment_transition":
            raise ValueError("commitment predecessor must be commitment_transition")
        if self.state == "ADMITTED" and self.predecessor is not None:
            raise ValueError("ADMITTED must originate directly from proposal")
        if self.state == "ACTIVE" and self.predecessor is None:
            raise ValueError("ACTIVE requires prior ADMITTED transition")
        if self.state == "SUPERSEDED" and not self.replacement_ref:
            raise ValueError("SUPERSEDED requires replacement relation")
        if self.state == "REVOKED" and not self.cause:
            raise ValueError("REVOKED requires cause")
        if self.state not in {"SUPERSEDED"} and self.replacement_ref is not None:
            raise ValueError("replacement relation only valid for SUPERSEDED")
        if self.state != "REVOKED" and self.cause is not None:
            raise ValueError("cause is reserved for REVOKED")
        digest = _digest(self.hash_payload())
        if (self.transition_ref, self.transition_hash) != (
            "commitment-transition:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("commitment transition identity mismatch")
        return self

    def hash_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"transition_ref", "transition_hash"})

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            transition_ref="commitment-transition:v1:pending",
            transition_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "transition_ref": "commitment-transition:v1:" + digest[7:],
            "transition_hash": digest,
        })

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="commitment_transition",
            artifact_ref=self.transition_ref,
            artifact_hash=self.transition_hash,
        )


class CommitmentTransitionReceiptV1(_Frozen):
    schema_version: Literal["commitment_transition_receipt.v1"] = "commitment_transition_receipt.v1"
    receipt_ref: str = Field(min_length=1)
    receipt_hash: str = Field(pattern=_SHA)
    transition: ArtifactPinV1
    decision: ArtifactPinV1
    responsibility_anchor: ArtifactPinV1
    authority_basis_hash: str = Field(pattern=_SHA)
    scope_ref: str = Field(min_length=1)
    recorded_at: datetime

    @model_validator(mode="after")
    def verify(self):
        if self.transition.artifact_kind != "commitment_transition":
            raise ValueError("commitment receipt requires commitment transition")
        digest = _digest(self.hash_payload())
        if (self.receipt_ref, self.receipt_hash) != (
            "commitment-transition-receipt:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("commitment receipt identity mismatch")
        return self

    def hash_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"})

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            receipt_ref="commitment-transition-receipt:v1:pending",
            receipt_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "receipt_ref": "commitment-transition-receipt:v1:" + digest[7:],
            "receipt_hash": digest,
        })
