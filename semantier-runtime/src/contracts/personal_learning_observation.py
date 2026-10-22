"""Producer completion metadata and derived personal observations; no learning authority."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import _ContentAddressed, SHA256_PATTERN


class WorkspaceInputPinV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    relative_path: str
    content_hash: str = Field(pattern=SHA256_PATTERN)


class ProcedureExecutionReceiptV1(_ContentAddressed):
    """Native tool producer facts, retained in the existing session trajectory journal."""

    ARTIFACT_KIND = "procedure_execution_receipt"
    schema_version: Literal["procedure_execution_receipt.v1"] = (
        "procedure_execution_receipt.v1"
    )
    organization_id: str
    workspace_id: str
    owner_ref: str
    session_id: str
    task_id: str
    tool_call_id: str
    skill_id: str
    entrypoint: str
    predecessor: ArtifactPinV1 | None
    predecessor_verified: bool
    input_pins: tuple[WorkspaceInputPinV1, ...]
    inputs_verified: bool
    command_hash: str = Field(pattern=SHA256_PATTERN)
    argument_contract_hash: str = Field(pattern=SHA256_PATTERN)
    result_hash: str = Field(pattern=SHA256_PATTERN)
    exit_code: int | None
    producer_status: str
    synthetic: bool
    started_at: datetime
    completed_at: datetime
    duration_ms: int = Field(ge=0)


class PersonalLearningObservationV1(_ContentAddressed):
    """Reconstructable experiment-eligibility view; not a second execution record."""

    ARTIFACT_KIND = "personal_learning_observation"
    schema_version: Literal["personal_learning_observation.v1"] = (
        "personal_learning_observation.v1"
    )
    organization_id: str
    workspace_id: str
    owner_ref: str
    occurrence_id: str
    family_id: str
    group_id: str | None
    producer_receipt: ArtifactPinV1
    predecessor: ArtifactPinV1 | None
    result_message_id: str | None
    outcome: Literal["SUCCESS", "FAILURE", "UNKNOWN"]
    binding_complete: bool
    eligibility: str
    rejection_reasons: tuple[str, ...]
