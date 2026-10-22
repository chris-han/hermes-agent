"""Immutable auxiliary HR task evidence; no policy or hiring authority."""

from datetime import datetime, timedelta
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator, StrictInt, StrictFloat, StrictBool, StrictStr
from typing_extensions import TypedDict, NotRequired
from contracts.artifact_pin import ArtifactPinV1
from eos.artifact_hashing import content_hash


class HrArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_ref: str = Field(pattern=r"^hr-[a-z-]+:[0-9a-f]{64}$")
    artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str
    workspace_id: str
    actor_id: str
    session_id: str
    recorded_at: datetime

    @model_validator(mode="after")
    def verify(self):
        if self.recorded_at.utcoffset() != timedelta(0):
            raise ValueError("UTC_TIMESTAMP_REQUIRED")
        digest = content_hash(
            self.model_dump(mode="json", exclude={"artifact_ref", "artifact_hash"})
        )
        if (
            self.artifact_hash != "sha256:" + digest
            or self.artifact_ref.rsplit(":", 1)[1] != digest
        ):
            raise ValueError("HR_HASH_MISMATCH")
        return self

    def pin(self):
        return ArtifactPinV1(
            artifact_kind=self.schema_version.split(".")[0],
            artifact_ref=self.artifact_ref,
            artifact_hash=self.artifact_hash,
        )

    @classmethod
    def create(cls, **values):
        values["schema_version"] = cls.model_fields["schema_version"].default
        draft = cls.model_construct(artifact_ref="", artifact_hash="", **values)
        digest = content_hash(
            draft.model_dump(mode="json", exclude={"artifact_ref", "artifact_hash"})
        )
        return cls.model_validate(
            {
                **draft.model_dump(mode="json"),
                "artifact_ref": values["schema_version"].split(".")[0].replace("_", "-")
                + ":"
                + digest,
                "artifact_hash": "sha256:" + digest,
            }
        )


class HrJobModelV1(HrArtifact):
    schema_version: Literal["hr_job_model.v1"] = "hr_job_model.v1"
    job_ref: str = Field(pattern=r"^jdrv_[A-Za-z0-9_-]+$")
    review_pin: ArtifactPinV1
    predecessor_pin: ArtifactPinV1 | None = None
    source: dict[str, Any]
    graph_pin: dict[str, str]
    requirements: list[dict[str, Any]]
    producer_profile: str = "hr_explicit_facts.v1"
    eligibility: Literal["AUXILIARY_SOURCE_REVIEW"] = "AUXILIARY_SOURCE_REVIEW"
    authority: Literal["NOT_ACTIVATED"] = "NOT_ACTIVATED"


class HrScreeningRunV1(HrArtifact):
    schema_version: Literal["hr_screening_run.v1"] = "hr_screening_run.v1"
    job_ref: str
    model_pin: ArtifactPinV1
    resume_pins: list[ArtifactPinV1] = Field(min_length=1, max_length=30)
    request_id: str
    request_hash: str
    as_of: str
    status: Literal["COMPLETED", "PARTIAL", "CANCELLED", "FAILED"]
    candidates: list[dict[str, Any]]
    failure_code: str | None = None
    producer_profile: str = "hr_explicit_facts.v1"
    evaluator_profile: str = "semantier_deterministic_v1"
    model_snapshot: dict[str, Any]

    @model_validator(mode="after")
    def complete_membership(self):
        refs = [p.artifact_ref for p in self.resume_pins]
        if len(set(refs)) != len(refs):
            raise ValueError("DUPLICATE_RESUME")
        if (
            self.status in ("COMPLETED", "PARTIAL")
            and [x["review_pin"]["artifact_ref"] for x in self.candidates] != refs
        ):
            raise ValueError("INCOMPLETE_RUN")
        if self.status in ("CANCELLED", "FAILED") and self.candidates:
            raise ValueError("NONTERMINAL_OUTPUT")
        model = HrJobModelV1.model_validate(self.model_snapshot)
        if model.pin() != self.model_pin or model.job_ref != self.job_ref:
            raise ValueError("JOB_MODEL_MISMATCH")
        return self


class HrSourceBindingV1(HrArtifact):
    schema_version: Literal["hr_source_binding.v1"] = "hr_source_binding.v1"
    workspace_artifact_pin: ArtifactPinV1
    review_pin: ArtifactPinV1
    purpose: Literal["JD", "RESUME"]
    content_hash: str


class HrImportCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_pin: ArtifactPinV1
    purpose: Literal["JD", "RESUME"]


class HrModelCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_id: str
    job_ref: str | None = None
    predecessor_pin: ArtifactPinV1 | None = None


class HrRunCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_ref: str
    model_pin: ArtifactPinV1
    resume_pins: list[ArtifactPinV1] = Field(min_length=1, max_length=30)
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    as_of: str
    cancel: bool = False
    interpretation_pins: list[ArtifactPinV1] = Field(default_factory=list, max_length=90)


class HrRemediateCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_pin: ArtifactPinV1
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")


class HrClauseV2(BaseModel):
    """A source clause and its bounded executable children, never authority."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    requirement_id: str
    text: str
    role: Literal["TITLE", "HEADING", "DUTY", "QUALIFICATION", "INFORMATIONAL", "UNKNOWN", "EXCLUDED"]
    modality: Literal["REQUIRED", "PREFERRED", "UNSPECIFIED"]
    evidence: dict[str, Any]
    structure_evidence: dict[str, Any] | None = None
    predicate: dict[str, Any] | None = None
    unit: str | None = None
    reason: str
    subject: Literal["degree", "experience", "specialist_experience", "major", "skill", "unresolved"] = "unresolved"
    domain: str | None = None
    comparator: Literal["GE", "EQ", "IN"] | None = None
    threshold: float | str | list[str] | None = None
    connective: Literal["AND", "OR"] | None = None
    child_ids: list[str] = Field(default_factory=list)
    parent_requirement_id: str | None = None

    @model_validator(mode="after")
    def executable_role(self):
        if self.role != "QUALIFICATION" and (self.predicate or self.child_ids):
            raise ValueError("NON_QUALIFICATION_NOT_EXECUTABLE")
        if bool(self.child_ids) != bool(self.connective):
            raise ValueError("COMPOUND_CHILDREN_REQUIRED")
        return self

    def __getitem__(self, key):
        # Incumbent projection/evaluator consumes serialized companion rows.
        return self.model_dump(mode="json")[key]


class HrJobModelV2(HrJobModelV1):
    schema_version: Literal["hr_job_model.v2"] = "hr_job_model.v2"
    requirements: list[HrClauseV2]
    producer_profile: str = "hr_source_meaning.v2"
    coverage: dict[str, int]
    interpretation_capability: Literal["AUTOMATION_UNAVAILABLE"] = "AUTOMATION_UNAVAILABLE"

    @model_validator(mode="after")
    def complete_conditions(self):
        by_id = {c.requirement_id: c for c in self.requirements}
        if len(by_id) != len(self.requirements):
            raise ValueError("DUPLICATE_CONDITION")
        for clause in self.requirements:
            for ref in clause.child_ids:
                child = by_id.get(ref)
                if child is None or child.parent_requirement_id != clause.requirement_id:
                    raise ValueError("INCOMPLETE_COMPOUND")
            if clause.parent_requirement_id:
                parent = by_id.get(clause.parent_requirement_id)
                if parent is None or clause.requirement_id not in parent.child_ids:
                    raise ValueError("ORPHAN_CONDITION")
        return self


class HrPinData(TypedDict):
    artifact_kind: str
    artifact_ref: str
    artifact_hash: str


class HrEvidenceData(TypedDict):
    evidence_ref: str
    anchor_ref: str
    quote: str
    source_hash: str
    source_identity_ref: str
    presentation_location: dict[str, Any] | None


class HrConditionResultV2(TypedDict):
    requirement_id: str
    role: Literal["TITLE", "HEADING", "DUTY", "QUALIFICATION", "INFORMATIONAL", "UNKNOWN", "EXCLUDED"]
    modality: Literal["REQUIRED", "PREFERRED", "UNSPECIFIED"]
    outcome: Literal["SUPPORTED_MATCH", "SUPPORTED_GAP", "UNKNOWN", "CONFLICT", "EXCLUDED"]
    observed_value: StrictInt | StrictFloat | StrictBool | StrictStr | None
    expected_value: StrictInt | StrictFloat | StrictBool | StrictStr | None
    unit: str | None
    reason: str
    jd_evidence: HrEvidenceData
    resume_evidence: list[HrEvidenceData]
    parent_requirement_id: str | None
    child_ids: list[str]
    support_basis: Literal["AUTOMATIC", "REVIEWED", "UNRESOLVED"]
    interpretation_pins: NotRequired[list[HrPinData]]


class HrCandidateV2(TypedDict):
    candidate_name: str | None
    review_pin: HrPinData
    source: dict[str, Any]
    results: list[HrConditionResultV2]
    sensitivity_findings: list[dict[str, Any]]
    fit_score: None
    coverage: NotRequired[dict[str, int]]


class HrScreeningRunV2(HrScreeningRunV1):
    schema_version: Literal["hr_screening_run.v2"] = "hr_screening_run.v2"
    candidates: list[HrCandidateV2]
    producer_profile: str = "hr_source_meaning.v2"
    evaluator_profile: str = "hr_strict_conditions.v2"
    interpretation_pins: list[ArtifactPinV1] = Field(default_factory=list)
    interpretation_snapshots: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_membership(self):
        refs = [p.artifact_ref for p in self.resume_pins]
        if len(set(refs)) != len(refs):
            raise ValueError("DUPLICATE_RESUME")
        if self.status in ("COMPLETED", "PARTIAL") and [x["review_pin"]["artifact_ref"] for x in self.candidates] != refs:
            raise ValueError("INCOMPLETE_RUN")
        if self.status in ("CANCELLED", "FAILED") and self.candidates:
            raise ValueError("NONTERMINAL_OUTPUT")
        interpretations = [HrInterpretationV1.model_validate(x) for x in self.interpretation_snapshots]
        if [x.pin() for x in interpretations] != self.interpretation_pins:
            raise ValueError("INTERPRETATION_PIN_MISMATCH")
        for value in interpretations:
            if value.model_pin != self.model_pin or value.resume_pin not in self.resume_pins or value.job_ref != self.job_ref:
                raise ValueError("INTERPRETATION_SCOPE_MISMATCH")
            if any(getattr(value, k) != getattr(self, k) for k in ("organization_id", "workspace_id", "actor_id", "session_id")):
                raise ValueError("INTERPRETATION_SCOPE_MISMATCH")
        model = HrJobModelV2.model_validate(self.model_snapshot)
        if model.pin() != self.model_pin or model.job_ref != self.job_ref:
            raise ValueError("JOB_MODEL_MISMATCH")
        conditions = {c.requirement_id: c for c in model.requirements}
        reviews = {v.artifact_ref: v for v in interpretations}
        for candidate in self.candidates:
            for result in candidate["results"]:
                condition = conditions.get(result["requirement_id"])
                if condition is None:
                    raise ValueError("RESULT_CONDITION_MISMATCH")
                for pin in result.get("interpretation_pins", []):
                    review = reviews.get(pin["artifact_ref"])
                    if review is None or review.pin().model_dump(mode="json") != pin or review.resume_pin.model_dump(mode="json") != candidate["review_pin"]:
                        raise ValueError("RESULT_INTERPRETATION_MISMATCH")
                    if review.requirement_id not in [condition.requirement_id, *condition.child_ids]:
                        raise ValueError("RESULT_INTERPRETATION_MISMATCH")
        return self


class HrEvidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    anchor_ref: str
    source_hash: str
    quote: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)


class HrInterpretationCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_pin: ArtifactPinV1
    resume_pin: ArtifactPinV1
    requirement_id: str
    selections: list[HrEvidenceSelection] = Field(min_length=1, max_length=30)
    interpretation_value: str = Field(min_length=1)
    source_value: str = Field(min_length=1)
    disposition: Literal["PROPOSED", "CONFIRMED", "REJECTED"]
    predecessor_pin: ArtifactPinV1 | None = None
    supports: bool = True


class HrInterpretationV1(HrArtifact):
    schema_version: Literal["hr_interpretation.v1"] = "hr_interpretation.v1"
    job_ref: str
    model_pin: ArtifactPinV1
    resume_pin: ArtifactPinV1
    requirement_id: str
    subject: Literal["major", "skill", "specialist_experience"]
    interpretation_value: str
    source_value: str
    disposition: Literal["PROPOSED", "CONFIRMED", "REJECTED"]
    predecessor_pin: ArtifactPinV1 | None = None
    supports: bool
    source_evidence: list[dict[str, Any]] = Field(min_length=1)
    intervals: list[tuple[str, str]] = Field(default_factory=list)
    eligibility: Literal["AUXILIARY_SOURCE_INTERPRETATION"] = "AUXILIARY_SOURCE_INTERPRETATION"
    authority: Literal["NOT_ACTIVATED"] = "NOT_ACTIVATED"


class HrInterpretationEvidenceCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_pin: ArtifactPinV1
    resume_pin: ArtifactPinV1
    requirement_id: str


class HrOperationV1(BaseModel):
    """Mutable execution projection; immutable terminal artifacts remain separate."""
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["hr_screening_operation.v1"] = "hr_screening_operation.v1"
    request_id: str
    request_hash: str
    command: dict[str, Any]
    status: Literal["RUNNING", "STOP_REQUESTED", "INTERRUPTED", "COMPLETED", "CANCELLED", "FAILED"]
    processed_candidates: int = Field(ge=0)
    result_pin: ArtifactPinV1 | None = None
    stop_disposition: Literal["ACKNOWLEDGED", "TOO_LATE", "FINAL"] | None = None
    started_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def integrity(self):
        if self.request_hash != "sha256:" + content_hash(self.command):
            raise ValueError("REQUEST_CONFLICT")
        if any(t.utcoffset() != timedelta(0) for t in (self.started_at, self.updated_at)):
            raise ValueError("UTC_TIMESTAMP_REQUIRED")
        if (self.status in ("COMPLETED", "CANCELLED", "FAILED")) != (self.result_pin is not None):
            raise ValueError("OPERATION_TERMINAL_MISMATCH")
        if self.processed_candidates > len(self.command["resume_pins"]):
            raise ValueError("INVALID_OPERATION_PROGRESS")
        return self
