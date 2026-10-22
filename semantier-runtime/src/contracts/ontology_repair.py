"""Typed diagnosis, repair proposal, and invalidation contracts."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.normative_semantics import NormativeFrameV1
from contracts.ontology_building import ArtifactPinV1
from contracts import runtime_context_reconstruction as rr


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def utc(self) -> "_Contract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class RuntimeRepairMetricResultV1(_Contract):
    """Stored case outcomes, not a verdict inferred by a presentation projector."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["runtime_repair_metric_result.v1"]
    case_id: str = Field(min_length=1, pattern=r"^[\x20-\x7e]+$")
    dimension: Literal["COMPETENCY", "VALIDATION", "EVIDENCE_FIDELITY"]
    expected_outcome_pin: ArtifactPinV1
    baseline_observed_pin: ArtifactPinV1 | None
    successor_observed_pin: ArtifactPinV1 | None
    baseline_outcome: Literal["PASS", "FAIL", "UNMEASURED"]
    successor_outcome: Literal["PASS", "FAIL", "UNMEASURED"]
    reason_code: str = Field(min_length=1, pattern=r"^[\x20-\x7e]+$")

    @model_validator(mode="after")
    def measured_outcomes_have_observations(self):
        for side in ("baseline", "successor"):
            if (getattr(self, f"{side}_outcome") != "UNMEASURED"
                    and getattr(self, f"{side}_observed_pin") is None):
                raise ValueError("measured outcome requires its observed result pin")
        return self


class OntologyDiagnosticV1(_Contract):
    schema_version: Literal["ontology_diagnostic.v1"] = "ontology_diagnostic.v1"
    diagnostic_ref: str = Field(min_length=1)
    diagnostic_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    candidate_set_ref: str = Field(min_length=1)
    candidate_set_hash: str = Field(pattern=SHA256_PATTERN)
    owning_stage: Literal["S0", "S1", "S2", "S3", "S4", "S5"]
    semantic_dimension: str = Field(min_length=1)
    finding_refs: tuple[str, ...] = Field(min_length=1)
    obligation_refs: tuple[str, ...] = Field(min_length=1)
    competency_refs: tuple[str, ...] = Field(min_length=1)
    source_evidence_refs: tuple[str, ...] = Field(min_length=1)
    ir_ast_paths: tuple[str, ...] = Field(min_length=1)
    allowed_typed_actions: tuple[str, ...] = Field(min_length=1)
    created_at: datetime


class RepairProposalV1(_Contract):
    schema_version: Literal["repair_proposal.v1"] = "repair_proposal.v1"
    repair_ref: str = Field(min_length=1)
    repair_hash: str = Field(pattern=SHA256_PATTERN)
    diagnostic_ref: str = Field(min_length=1)
    predecessor_candidate_set_ref: str = Field(min_length=1)
    predecessor_candidate_set_hash: str = Field(pattern=SHA256_PATTERN)
    operation: Literal["ADD_FRAME", "REPLACE_FRAME", "DELETE_FRAME"]
    target_frame_refs: tuple[str, ...] = Field(min_length=1)
    requires_human_reconfirmation: bool
    created_at: datetime


class RepairImpactV1(_Contract):
    schema_version: Literal["repair_impact.v1"] = "repair_impact.v1"
    impact_ref: str = Field(min_length=1)
    impact_hash: str = Field(pattern=SHA256_PATTERN)
    repair_ref: str = Field(min_length=1)
    predecessor_candidate_set_ref: str = Field(min_length=1)
    successor_candidate_set_ref: str = Field(min_length=1)
    changed_frame_refs: tuple[str, ...] = Field(min_length=1)
    preserved_frame_refs: tuple[str, ...]
    validation_contract_ref: str = Field(min_length=1)
    invalidated_result_kinds: tuple[Literal["quality_evaluation_run"], ...]
    created_at: datetime


def _advisory_outcomes(outcomes: tuple[rr.EpisodeOutcomePinV1, ...]) -> None:
    refs = [
        (pin.artifact_kind, pin.artifact_ref)
        for pair in outcomes
        for pin in (pair.episode, pair.assessment)
    ]
    if len(refs) != len(set(refs)):
        raise ValueError("outcomes contain duplicate or conflicting pins")


class AdvisoryNextProposalRequestV1(rr._ReconstructionContract):
    schema_version: Literal["advisory_next_proposal_request.v1"]
    request_id: rr.AdvisoryRequestId
    formation: ArtifactPinV1
    outcomes: tuple[rr.EpisodeOutcomePinV1, ...] = Field(min_length=1, max_length=8)
    predecessor: ArtifactPinV1
    operation: Literal["ADD_FRAME"]
    frame: NormativeFrameV1
    validation_contract: ArtifactPinV1

    @model_validator(mode="after")
    def exact_proposal_inputs(self) -> "AdvisoryNextProposalRequestV1":
        rr._require_pin_kind(self.formation, "advisory_context_formation", "formation")
        rr._require_pin_kind(
            self.predecessor, "normative_semantic_candidate_set", "predecessor"
        )
        rr._require_pin_kind(
            self.validation_contract, "validation_contract", "validation_contract"
        )
        _advisory_outcomes(self.outcomes)
        return self


class AdvisoryNextProposalCommandV1(rr._WorkspaceIdentityContract):
    schema_version: Literal["advisory_next_proposal_command.v1"]
    command_ref: rr.NonEmptyStr
    command_hash: rr.Sha256
    actor_id: rr.NonEmptyStr
    session_ref: rr.NonEmptyStr
    operation: Literal["PROPOSE_ADVISORY_REPAIR"]
    request: AdvisoryNextProposalRequestV1

    identity_prefix = "advisory-next-proposal-command"
    identity_ref_field = "command_ref"
    identity_hash_field = "command_hash"
    identity_label = "advisory next proposal command"


class AdvisoryRepairProposalV1(rr._WorkspaceIdentityContract):
    schema_version: Literal["advisory_repair_proposal.v1"]
    proposal_ref: rr.NonEmptyStr
    proposal_hash: rr.Sha256
    actor_id: rr.NonEmptyStr
    session_ref: rr.NonEmptyStr
    command: ArtifactPinV1
    formation: ArtifactPinV1
    outcomes: tuple[rr.EpisodeOutcomePinV1, ...] = Field(min_length=1, max_length=8)
    predecessor: ArtifactPinV1
    operation: Literal["ADD_FRAME"]
    frame: NormativeFrameV1
    validation_contract: ArtifactPinV1
    usage: Literal["CONTEXT_ONLY"]
    state: Literal["PROPOSED"]
    recorded_at: rr.UtcDateTime

    identity_prefix = "advisory-repair-proposal"
    identity_ref_field = "proposal_ref"
    identity_hash_field = "proposal_hash"
    identity_label = "advisory repair proposal"

    @model_validator(mode="after")
    def exact_proposal_links(self) -> "AdvisoryRepairProposalV1":
        rr._require_pin_kind(self.command, "advisory_next_proposal_command", "command")
        AdvisoryNextProposalRequestV1.exact_proposal_inputs(self)
        return self


class AdvisoryNextProposalReceiptV1(rr._WorkspaceIdentityContract):
    schema_version: Literal["advisory_next_proposal_receipt.v1"]
    receipt_ref: rr.NonEmptyStr
    receipt_hash: rr.Sha256
    actor_id: rr.NonEmptyStr
    session_ref: rr.NonEmptyStr
    operation: Literal["PROPOSE_ADVISORY_REPAIR"]
    request_id: rr.AdvisoryRequestId
    request_hash: rr.Sha256
    command: ArtifactPinV1
    proposal: ArtifactPinV1
    formation: ArtifactPinV1
    outcomes: tuple[rr.EpisodeOutcomePinV1, ...] = Field(min_length=1, max_length=8)
    usage: Literal["CONTEXT_ONLY"]
    recorded_at: rr.UtcDateTime

    identity_prefix = "advisory-next-proposal-receipt"
    identity_ref_field = "receipt_ref"
    identity_hash_field = "receipt_hash"
    identity_label = "advisory next proposal receipt"

    @model_validator(mode="after")
    def exact_receipt_links(self) -> "AdvisoryNextProposalReceiptV1":
        rr._require_pin_kind(self.command, "advisory_next_proposal_command", "command")
        rr._require_pin_kind(self.proposal, "advisory_repair_proposal", "proposal")
        rr._require_pin_kind(self.formation, "advisory_context_formation", "formation")
        _advisory_outcomes(self.outcomes)
        if self.request_hash != self.command.artifact_hash:
            raise ValueError("request hash mismatch")
        return self


__all__ = [
    "OntologyDiagnosticV1",
    "RepairImpactV1",
    "RepairProposalV1",
    "AdvisoryNextProposalRequestV1",
    "AdvisoryNextProposalCommandV1",
    "AdvisoryRepairProposalV1",
    "AdvisoryNextProposalReceiptV1",
]
