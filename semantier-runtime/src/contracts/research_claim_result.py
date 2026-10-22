"""Content-addressed research results; reports project these stored receipts."""
from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash


class _ResearchReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    REF_PREFIX: ClassVar[str]
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    receipt_ref: str
    receipt_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    def hash_payload(self):
        return self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"})

    @model_validator(mode="after")
    def exact_identity(self):
        digest = content_hash(self.hash_payload())
        if (self.receipt_ref, self.receipt_hash) != (f"{self.REF_PREFIX}:{digest}", f"sha256:{digest}"):
            raise ValueError("RESEARCH_RESULT_CONTENT_IDENTITY_MISMATCH")
        return self

    @classmethod
    def create(cls, **values):
        draft = cls.model_construct(**values)
        digest = content_hash(draft.hash_payload())
        return cls.model_validate({**values, "receipt_ref": f"{cls.REF_PREFIX}:{digest}", "receipt_hash": f"sha256:{digest}"})

    def pin(self):
        return ArtifactPinV1(artifact_kind=self.schema_version.removesuffix(".v1"),
                             artifact_ref=self.receipt_ref, artifact_hash=self.receipt_hash)


class ResearchReplayReceiptV1(_ResearchReceipt):
    REF_PREFIX = "research-replay"
    schema_version: Literal["research_replay.v1"] = "research_replay.v1"
    run_basis: ArtifactPinV1
    instrumentation: ArtifactPinV1
    reservations: tuple[ArtifactPinV1, ...]
    outcomes: tuple[ArtifactPinV1, ...]
    evidence_refs: tuple[ArtifactPinV1, ...]
    scheduled_count: int = Field(ge=0)
    finalized_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    metrics: dict[str, Any]

    @model_validator(mode="after")
    def accounting(self):
        if self.finalized_count + self.uncertain_count + self.missing_count != self.scheduled_count:
            raise ValueError("RESEARCH_REPLAY_ACCOUNTING_MISMATCH")
        if len(self.reservations) != self.finalized_count + self.uncertain_count or len(self.outcomes) != len(self.reservations):
            raise ValueError("RESEARCH_REPLAY_INVOCATION_EVIDENCE_MISMATCH")
        return self


class ClaimResultV1(_ResearchReceipt):
    REF_PREFIX = "research-claim-result"
    schema_version: Literal["claim_result.v1"] = "claim_result.v1"
    claim_id: Literal["C1", "C2a", "C2b"]
    status: Literal["SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"]
    basis_ref: str
    basis_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    comparison_refs: tuple[ArtifactPinV1, ...]
    metric_results: dict[str, Any]
    applicable_thresholds: dict[str, Any]
    failed_thresholds: tuple[str, ...]
    not_estimable_metrics: tuple[str, ...]
    limitations: tuple[str, ...]
    evidence_refs: tuple[ArtifactPinV1, ...]
    replay_ref: ArtifactPinV1

    @model_validator(mode="after")
    def status_evidence(self):
        if self.status == "SUPPORTED" and (self.failed_thresholds or self.not_estimable_metrics or not self.comparison_refs):
            raise ValueError("RESEARCH_SUPPORTED_CLAIM_REQUIRES_COMPLETE_COMPARISON")
        if self.status == "NOT_SUPPORTED" and not self.failed_thresholds:
            raise ValueError("RESEARCH_REFUTED_CLAIM_REQUIRES_FAILED_THRESHOLD")
        if self.status == "INCONCLUSIVE" and not self.limitations:
            raise ValueError("RESEARCH_INCONCLUSIVE_CLAIM_REQUIRES_LIMITATION")
        return self


class RunDispositionV1(_ResearchReceipt):
    REF_PREFIX = "research-run-disposition"
    schema_version: Literal["research_run_disposition.v1"] = "research_run_disposition.v1"
    run_basis: ArtifactPinV1
    confirmatory_basis: ArtifactPinV1 | None
    disposition: Literal["INCONCLUSIVE_UNCERTAIN_INVOCATION", "HOLD_MECHANICS", "NOT_SUPPORTED", "INCONCLUSIVE_DATA", "SUPPORTED"]
    reason_codes: tuple[str, ...]
    claim_results: dict[Literal["C1", "C2a", "C2b"], ArtifactPinV1]
    evidence_refs: tuple[ArtifactPinV1, ...]
    replay_ref: ArtifactPinV1

    @model_validator(mode="after")
    def all_claims(self):
        if set(self.claim_results) != {"C1", "C2a", "C2b"}:
            raise ValueError("RESEARCH_RUN_REQUIRES_ALL_CLAIM_RECORDS")
        if any(pin.artifact_kind != "claim_result" for pin in self.claim_results.values()):
            raise ValueError("RESEARCH_RUN_CLAIM_KIND_MISMATCH")
        if self.disposition != "SUPPORTED" and not self.reason_codes:
            raise ValueError("RESEARCH_RUN_REQUIRES_REASON")
        return self


class ResearchRunCheckpointV1(_ResearchReceipt):
    REF_PREFIX = "research-run-checkpoint"
    schema_version: Literal["research_run_checkpoint.v1"] = "research_run_checkpoint.v1"
    run_basis: ArtifactPinV1
    instrumentation: ArtifactPinV1
    lifecycle: Literal["FROZEN", "PREFLIGHTED", "COLLECTING", "COLLECTED", "REPLAY_VERIFIED", "REPORTED"]
    predecessor: ArtifactPinV1 | None
    evidence_refs: tuple[ArtifactPinV1, ...]
