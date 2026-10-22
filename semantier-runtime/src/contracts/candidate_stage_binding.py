"""Derived owning-stage binding for the existing ChangeCandidate lifecycle."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, model_validator

OwningStage = Literal[
    "DESCRIPTIVE", "NORMATIVE", "EPISTEMIC", "DECISION", "COMMITMENT", "ENVIRONMENT"
]


class CandidateStageBindingV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: Literal["ONTOLOGY", "CONTEXT_FORMATION", "REASONING", "WORKFLOW", "POLICY"]
    failure_class: Literal["CAPABILITY", "STEERING", "STRUCTURE", "ENVIRONMENT"]
    owning_stage: OwningStage
    owner_ref: str

    @model_validator(mode="after")
    def validate_owner_boundary(self):
        if self.owning_stage == "DESCRIPTIVE":
            if self.target != "ONTOLOGY" or self.failure_class != "STRUCTURE":
                raise ValueError("CROSS_STAGE_OWNER_MISMATCH")
        elif self.owning_stage == "NORMATIVE":
            if self.target != "POLICY" or self.failure_class != "STRUCTURE":
                raise ValueError("CROSS_STAGE_OWNER_MISMATCH")
        elif self.owning_stage == "DECISION":
            if self.target not in {"CONTEXT_FORMATION", "REASONING"}:
                raise ValueError("CROSS_STAGE_OWNER_MISMATCH")
        elif self.owning_stage == "COMMITMENT":
            if self.owner_ref != "commitment-policy-owner":
                raise ValueError("COMMITMENT_OWNER_REQUIRED")
        elif self.owning_stage == "EPISTEMIC":
            if self.target != "REASONING":
                raise ValueError("CROSS_STAGE_OWNER_MISMATCH")
        elif self.owning_stage == "ENVIRONMENT":
            if self.failure_class != "ENVIRONMENT":
                raise ValueError("CROSS_STAGE_OWNER_MISMATCH")
        return self
