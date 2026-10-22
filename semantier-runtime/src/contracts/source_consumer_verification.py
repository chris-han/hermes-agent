"""Embedded DEVELOPMENT verification evidence; no business or source authority.

Kept adjacent to validation_contract because HTTP rejection and set output are
not semantic frames or rule outcomes. Persisted only inside a quality run.
"""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from contracts.artifact_pin import ArtifactPinV1
from eos.artifact_hashing import content_hash

H = r'^sha256:[0-9a-f]{64}$'
ID = Annotated[str, Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$')]
Kind = Literal['input_prerequisite', 'input_requirement', 'set_output']
Value = Literal['ACCEPTED', 'REJECTED', 'SATISFIED', 'FAILED', 'UNKNOWN'] | tuple[str, ...]


class Embedded(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, revalidate_instances='always')


class ConsumerSourceRequirementV1(Embedded):
    requirement_id: ID = Field(min_length=1)
    competency_ids: tuple[ID, ...] = Field(min_length=1)
    occurrence_ref: ID = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    byte_start: int = Field(ge=0)
    byte_end: int = Field(gt=0)
    quote: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    target_ids: tuple[str, ...] = Field(min_length=1)


class ConsumerExpectedCaseV1(Embedded):
    case_id: ID = Field(min_length=1)
    # An empty set is meaningful, unlike an empty target inventory.
    expected: Value


class ConsumerTargetV1(Embedded):
    target_id: ID = Field(min_length=1)
    requirement_id: ID = Field(min_length=1)
    kind: Kind
    field: str = Field(min_length=1)
    fact_ref: str | None = None
    cases: tuple[ConsumerExpectedCaseV1, ...] = Field(min_length=1)


class ConsumerBindingV1(Embedded):
    binding_id: ID = Field(min_length=1)
    target_id: ID = Field(min_length=1)
    requirement_id: ID = Field(min_length=1)
    competency_ids: tuple[ID, ...] = Field(min_length=1)
    occurrence_ref: ID = Field(min_length=1)
    source_hash: str = Field(pattern=H)
    kind: Kind
    consumer_ref: ID = Field(min_length=1)
    consumer_hash: str = Field(pattern=H)
    selector: ID = Field(min_length=1)
    fact_ref: str | None = None


class ConsumerCaseEvidenceV1(Embedded):
    case_id: ID = Field(min_length=1)
    session_ref: str = Field(min_length=1)
    request: dict
    request_hash: str = Field(pattern=H)
    http_status: int
    response: dict
    response_hash: str = Field(pattern=H)
    # Original session artifacts keyed by relative ref, frozen as base64 bytes.
    dependencies: dict[str, str]
    current_result: dict | None = None
    draft: dict | None = None

    @model_validator(mode='after')
    def exact_capture(self):
        if self.request_hash != 'sha256:' + content_hash(self.request) or self.response_hash != 'sha256:' + content_hash(self.response):
            raise ValueError('consumer case capture hash mismatch')
        return self


class ConsumerObservationV1(Embedded):
    binding_id: ID
    case_id: ID
    evidence_hash: str = Field(pattern=H)
    actual: Value


class ConsumerTargetAssessmentV1(Embedded):
    requirement_id: ID
    target_id: ID
    mapping_status: Literal['MAPPED', 'MISSING', 'UNKNOWN']
    witness_status: Literal['PASS', 'FAIL', 'UNKNOWN', 'NOT_RUN']
    supporting_binding_ids: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()


class ConsumerRequirementAssessmentV1(Embedded):
    requirement_id: ID
    coverage: Literal['FULL', 'PARTIAL', 'UNKNOWN', 'NONE']
    scope: str


class ConsumerVerificationV1(Embedded):
    schema_version: Literal['source_consumer_verification.v1'] = 'source_consumer_verification.v1'
    exposure: Literal['DEVELOPMENT'] = 'DEVELOPMENT'
    authorship: Literal['ASSISTANT_AUTHORED'] = 'ASSISTANT_AUTHORED'
    completeness: Literal['NOT_QUALIFIED'] = 'NOT_QUALIFIED'
    organization_id: str
    workspace_id: str
    session_ref: str
    source_identity_ref: ID
    source_hash: str = Field(pattern=H)
    source_text: str
    pdf_hash: str = Field(pattern=H)
    # Retained exact source/consumer/plan bytes. Hashes never grant scope authority.
    assets: dict[str, str]
    asset_hashes: dict[str, str]
    plan: ArtifactPinV1
    eligibility: Literal['ELIGIBLE', 'UNRESOLVED']
    requirements: tuple[ConsumerSourceRequirementV1, ...] = Field(min_length=1)
    targets: tuple[ConsumerTargetV1, ...] = Field(min_length=1)
    bindings: tuple[ConsumerBindingV1, ...]
    cases: tuple[ConsumerCaseEvidenceV1, ...]
    observations: tuple[ConsumerObservationV1, ...]
    target_assessments: tuple[ConsumerTargetAssessmentV1, ...]
    requirement_assessments: tuple[ConsumerRequirementAssessmentV1, ...]

    @model_validator(mode='after')
    def retained_verification(self):
        # Pure retained reduction; never loads a live consumer, parser or evaluator.
        from services.source_consumer_verification import verify_basis, assess_consumers
        verify_basis(self)
        targets, requirements = assess_consumers(self)
        if self.target_assessments != targets or self.requirement_assessments != requirements:
            raise ValueError('retained consumer assessment mismatch')
        return self
