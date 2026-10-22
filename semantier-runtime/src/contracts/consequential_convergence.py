"""Typed contracts for the deterministic Consequential Convergence Controller."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import _ContentAddressed, _ordered_unique_pins


BranchAmbiguity = Literal["NONE", "LOW", "MATERIAL", "UNRESOLVED"]
ImpactRadius = Literal[
    "R0_TRANSIENT",
    "R1_TASK",
    "R2_WORKFLOW",
    "R3_ORGANIZATIONAL",
    "R4_EXTERNAL",
    "R5_INSTITUTIONAL",
]
StakeValue = Literal["LOW", "MATERIAL", "HIGH", "CRITICAL"]
Reversibility = Literal["PURE", "REVERSIBLE", "COMPENSATABLE", "IRREVERSIBLE"]
AuthorityGap = Literal[
    "NONE",
    "ENCODED_DELEGATION",
    "UNRESOLVED_AUTHORITY",
    "HUMAN_RESERVED",
]
CCCDisposition = Literal[
    "AUTO_CONTINUE",
    "AUTO_COMMIT",
    "RECOMMEND",
    "HUMAN_CHECKPOINT",
]
InstitutionalRoute = Literal["PASS_AUTO", "ESCALATE", "FAIL_HARD"]
CostLevel = Literal["LOW", "MATERIAL", "HIGH"]


class _Part(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HumanDecisionValueV1(_Part):
    missing_preference_information: bool
    likely_decision_change: bool
    authority_resolution_needed: bool
    interruption_cost: CostLevel
    delay_cost: CostLevel

    def is_positive(self) -> bool:
        if self.authority_resolution_needed:
            return True
        if self.missing_preference_information and self.likely_decision_change:
            return True
        return False


class CCCInputV1(_Part):
    organization_id: str
    workspace_id: str
    frontier_ref: ArtifactPinV1
    viable_option_refs: tuple[ArtifactPinV1, ...]
    dominant_option_ref: ArtifactPinV1 | None
    branch_ambiguity: BranchAmbiguity
    impact_radius: ImpactRadius
    stake_value: StakeValue
    reversibility: Reversibility
    authority_gap: AuthorityGap
    safe_information_gain_available: bool
    immediate_commitment_required: bool
    human_decision_value: HumanDecisionValueV1 | None

    @field_validator("viable_option_refs", mode="before")
    @classmethod
    def canonical_options(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_frontier(self) -> "CCCInputV1":
        if self.dominant_option_ref is not None and self.dominant_option_ref not in self.viable_option_refs:
            raise ValueError("dominant option must be in viable frontier")
        if len(self.viable_option_refs) > 1 and self.branch_ambiguity == "NONE":
            raise ValueError("multiple viable options cannot have NONE branch ambiguity")
        return self


class CCCGateResultV1(_Part):
    gate: str
    matched: bool


class CCCDecisionReceiptV1(_ContentAddressed):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ARTIFACT_KIND = "ccc_decision_receipt"

    schema_version: Literal["ccc_decision_receipt.v1"] = "ccc_decision_receipt.v1"
    organization_id: str
    workspace_id: str
    typed_input: CCCInputV1
    policy: ArtifactPinV1
    gate_results: tuple[CCCGateResultV1, ...]
    decisive_gate: str
    disposition: CCCDisposition
    institutional_route: InstitutionalRoute | None
    selected_option_refs: tuple[ArtifactPinV1, ...]
    recommended_option_refs: tuple[ArtifactPinV1, ...]
    human_checkpoint_ref: ArtifactPinV1 | None = None
    human_response_ref: ArtifactPinV1 | None = None
    evaluated_at: datetime


__all__ = [
    "AuthorityGap",
    "BranchAmbiguity",
    "CCCDecisionReceiptV1",
    "CCCDisposition",
    "CCCGateResultV1",
    "CCCInputV1",
    "HumanDecisionValueV1",
    "ImpactRadius",
    "InstitutionalRoute",
    "Reversibility",
    "StakeValue",
]
