"""Workflow-only contracts for the resumable Tender review journey.

These contracts coordinate existing governed artifacts. They are not semantic-kernel
objects and carry no admission, activation, certification, or legal authority.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.validation_contract import CaseRuleExpectations
from contracts.runtime_semantic_inspection_result import (
    RuntimeSemanticInspectionContextV1, RuntimeSemanticInspectionResultV1,
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def persisted_timestamps_are_utc(self) -> "_Contract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime):
                if value.tzinfo is None or value.utcoffset() != timedelta(0):
                    raise ValueError(f"{name} must be timezone-aware UTC")
        return self


TenderJourneyPhase = Literal[
    "NEEDS_POLICY",
    "POLICY_READY",
    "BUILDING_CONTEXT",
    "REVIEW_INTENT",
    "CERTIFYING_CONTEXT",
    "RUNTIME_BINDING_REQUIRED",
    "CONTEXT_READY",
    "NEEDS_RUNTIME_DOCUMENT",
    "FORMING_RUNTIME_CONTEXT",
    "INSPECTING",
    "REVIEW_RESULTS",
]

TenderJourneySurface = Literal["CHAT", "WORKBENCH", "SYSTEM"]
TenderJourneyLifecycle = Literal["ACTIVE", "COMPLETED", "PAUSED", "CANCELLED"]
TenderJourneyBindingRole = Literal[
    "policy_source",
    "canonical_ir",
    "graph_intent",
    "context_build",
    "certification",
    "authority_dependencies",
    "source_search_coverage",
    "competency_contract",
    "normative_schema",
    "normative_semantics",
    "validation_case_proposal",
    "validation_case_review",
    "source_validation_execution",
    "schema_adequacy",
    "runtime_binding",
    "runtime_document",
    "runtime_document_preparation",
    "runtime_context",
    "inspection_result",
    "operation_receipt",
    "other",
]


class JourneyArtifactBindingV1(_Contract):
    schema_version: Literal["tender_journey_artifact_binding.v1"] = (
        "tender_journey_artifact_binding.v1"
    )
    pin: ArtifactPinV1
    owner_kind: str = Field(min_length=1)
    owner_revision: str | None = None
    role: TenderJourneyBindingRole


class TenderJourneyTransitionV1(_Contract):
    schema_version: Literal["tender_journey_transition.v1"] = "tender_journey_transition.v1"
    transition_id: str = Field(min_length=1)
    journey_id: str = Field(min_length=1)
    predecessor_revision: int = Field(ge=0)
    successor_revision: int = Field(ge=1)
    from_phase: TenderJourneyPhase
    to_phase: TenderJourneyPhase
    from_lifecycle_status: TenderJourneyLifecycle | None = None
    to_lifecycle_status: TenderJourneyLifecycle | None = None
    cause_code: str = Field(min_length=1)
    actor_ref: str = Field(min_length=1)
    surface: TenderJourneySurface
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    artifact_bindings: tuple[JourneyArtifactBindingV1, ...] = ()
    blocker_code: str | None = None
    operation_receipt: JourneyArtifactBindingV1 | None = None
    created_at: datetime

    @model_validator(mode="after")
    def validate_revision(self) -> "TenderJourneyTransitionV1":
        if (self.from_lifecycle_status is None) != (self.to_lifecycle_status is None):
            raise ValueError("journey lifecycle transition requires both statuses")
        if self.to_lifecycle_status is not None and self.from_phase != self.to_phase:
            raise ValueError("lifecycle transition cannot change phase")
        if self.successor_revision != self.predecessor_revision + 1:
            raise ValueError("journey transition revision must advance by exactly one")
        if self.from_phase != self.to_phase and self.operation_receipt is None:
            raise ValueError("phase-changing journey transition requires an exact operation receipt")
        return self


class TenderJourneyHeadV1(_Contract):
    schema_version: Literal["tender_journey_head.v1"] = "tender_journey_head.v1"
    journey_id: str = Field(min_length=1)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    current_phase: TenderJourneyPhase
    lifecycle_status: TenderJourneyLifecycle = "ACTIVE"
    artifact_bindings: tuple[JourneyArtifactBindingV1, ...] = ()
    blocker_code: str | None = None
    operation_receipt: JourneyArtifactBindingV1 | None = None
    latest_transition_id: str | None = None
    updated_at: datetime


class TenderJourneyProjectionV1(_Contract):
    schema_version: Literal["tender_journey_projection.v1"] = "tender_journey_projection.v1"
    journey_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    current_phase: TenderJourneyPhase
    lifecycle_status: TenderJourneyLifecycle
    blocker_code: str | None = None
    next_action_code: str
    next_action_requires_user: bool
    artifact_bindings: tuple[JourneyArtifactBindingV1, ...] = ()


class TenderJourneyChatSelectionV1(_Contract):
    journey_id: str = Field(min_length=1, strict=True)
    transition_id: str | None = Field(default=None, min_length=1, strict=True)


class TenderJourneyInspectionEpisodeV1(_Contract):
    schema_version: Literal["tender_journey_inspection_episode.v1"] = "tender_journey_inspection_episode.v1"
    transition_id: str = Field(min_length=1)
    journey: TenderJourneyProjectionV1
    inspection_context: RuntimeSemanticInspectionContextV1
    inspection_result: RuntimeSemanticInspectionResultV1


class TenderJourneyAdvanceRequestV1(_Contract):
    schema_version: Literal["tender_journey_advance_request.v1"] = "tender_journey_advance_request.v1"
    expected_revision: int = Field(ge=0)
    to_phase: TenderJourneyPhase
    cause_code: str = Field(min_length=1)
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    operation_receipt: JourneyArtifactBindingV1
    artifact_bindings: tuple[JourneyArtifactBindingV1, ...] = ()
    surface: TenderJourneySurface = "SYSTEM"


class TenderJourneyCertifyContextRequestV1(_Contract):
    expected_revision: int = Field(ge=0)
    competency_pin: ArtifactPinV1
    authority_pin: ArtifactPinV1
    execution_plan_pin: ArtifactPinV1
    schema_adequacy_pin: ArtifactPinV1
    representative_evaluation_pin: ArtifactPinV1
    paired_suite_pin: ArtifactPinV1 | None = None
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    surface: TenderJourneySurface = "WORKBENCH"


class TenderJourneySelectCertificationRequestV1(_Contract):
    expected_revision: int = Field(ge=0)
    receipt_pin: ArtifactPinV1
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    surface: TenderJourneySurface = "WORKBENCH"


class TenderJourneyDiscoverIntentRequestV1(_Contract):
    expected_revision: int = Field(ge=0)
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    surface: TenderJourneySurface = "WORKBENCH"


class TenderJourneyReviewIntentRequestV1(_Contract):
    expected_revision: int = Field(ge=0)
    graph_intent_pin: ArtifactPinV1
    state: Literal["accepted", "edited", "rejected"]
    relation_edit: dict | None = None
    selected_question_ids: list[str] = Field(default_factory=list)
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    surface: TenderJourneySurface = "WORKBENCH"


class TenderJourneyBuildContextRequestV1(_Contract):
    expected_revision: int = Field(ge=0)
    graph_intent_pin: ArtifactPinV1
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    surface: TenderJourneySurface = "WORKBENCH"


class TenderJourneySelectPolicyRequestV1(_Contract):
    expected_revision: int = Field(ge=0)
    artifact_pin: ArtifactPinV1
    preparation_pin: ArtifactPinV1
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    surface: TenderJourneySurface = "WORKBENCH"


class TenderJourneyRuntimeCommandV1(_Contract):
    expected_revision: int = Field(ge=0)
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    surface: TenderJourneySurface = "WORKBENCH"


class TenderJourneySelectRuntimeDocumentRequestV1(_Contract):
    expected_revision: int = Field(ge=0)
    artifact_pin: ArtifactPinV1
    preparation_pin: ArtifactPinV1
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    surface: TenderJourneySurface = "WORKBENCH"


class TenderJourneyBlockRequestV1(_Contract):
    schema_version: Literal["tender_journey_block_request.v1"] = "tender_journey_block_request.v1"
    expected_revision: int = Field(ge=0)
    blocker_code: str = Field(min_length=1)
    cause_code: str = Field(min_length=1)
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    artifact_bindings: tuple[JourneyArtifactBindingV1, ...] = ()
    surface: TenderJourneySurface = "SYSTEM"


class TenderJourneyV1(_Contract):
    schema_version: Literal["tender_journey.v1"] = "tender_journey.v1"
    journey_id: str = Field(min_length=1)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    journey_kind: Literal["TENDER_REVIEW"] = "TENDER_REVIEW"
    created_by: str = Field(min_length=1)
    lifecycle_status: TenderJourneyLifecycle = "ACTIVE"
    created_at: datetime


class TenderAuthoritySelectionV1(_Contract):
    source_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    artifact_hash: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    section_roles: dict[str, tuple[Literal["rule_authority", "scope_authority", "exception_authority", "qualification_authority", "effectivity_authority", "precedence_authority"], ...]]
    section_hashes: dict[str, tuple[str, str]]


class TenderJourneyPrepareAuthorityRequestV1(_Contract):
    expected_revision: int = Field(ge=0)
    coverage_pin: ArtifactPinV1
    selections: tuple[TenderAuthoritySelectionV1, ...] = Field(min_length=1)
    jurisdiction: str = Field(min_length=1)
    as_of_time: datetime
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    surface: TenderJourneySurface = "WORKBENCH"


class TenderJourneyReviewCompetencyRequestV1(TenderJourneyRuntimeCommandV1):
    competency_pin: ArtifactPinV1
    question_id: str = Field(min_length=1)
    decision_kind: Literal["confirm_criticality", "downgrade_criticality", "exclude_competency", "define_boundary"]
    rationale: str = Field(min_length=1)


class TenderJourneyReviewValidationCasesRequestV1(TenderJourneyRuntimeCommandV1):
    proposal_pin: ArtifactPinV1
    case_review_pins: tuple[ArtifactPinV1, ...] = ()
    case_rule_expectations: CaseRuleExpectations | None = None
    decision: Literal["accepted", "rejected"]
    rationale: str = Field(min_length=1)


class TenderJourneyReviewCoverageRequestV1(TenderJourneyRuntimeCommandV1):
    selections: tuple[TenderAuthoritySelectionV1, ...] = Field(min_length=1)


class TenderJourneyProduceNormativeSchemaRequestV1(TenderJourneyRuntimeCommandV1):
    schema_decision_pin: ArtifactPinV1


class TenderRuntimeCommandCompletionV1(_Contract):
    """Read-only exact retained result; absence makes no execution claim."""
    schema_version: Literal["tender_runtime_command_completion.v1"] = "tender_runtime_command_completion.v1"
    status: Literal["NOT_CONFIRMED", "COMPLETED"]
    journey_id: str
    command_id: str
    idempotency_key: str
    expected_revision: int = Field(ge=0)
    compatible: bool = False
    transition: TenderJourneyTransitionV1 | None = None
    result: TenderJourneyProjectionV1 | None = None
