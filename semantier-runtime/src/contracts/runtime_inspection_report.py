"""Pure report sections: presentation of retained results is not policy authority."""

from typing import Annotated, Literal
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_working_context import V2UtcDateTime
from eos.artifact_hashing import content_hash


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    return value

_Text = Annotated[str, Field(min_length=1)]
_Id = Annotated[str, Field(min_length=1, pattern=r"^[\x20-\x7e]+$")]


class _ReportContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def require_utc(self):
        for field in type(self).model_fields:
            value = getattr(self, field)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError("report source times must be timezone-aware UTC")
        return self


class RuntimeReportModificationV1(_ReportContract):
    source_event_pin: ArtifactPinV1
    recorded_at: Annotated[V2UtcDateTime, Field(strict=False)]
    change_kind: Literal["TARGET_CORRECTION", "SEMANTIC_CHANGE", "REVIEW_CHANGE"]


class RuntimeReportHistoryV1(_ReportContract):
    modification_events: tuple[RuntimeReportModificationV1, ...]
    predecessor_report_pin: ArtifactPinV1 | None
    successor_link_receipt_pins: tuple[ArtifactPinV1, ...]
    root_run_pin: ArtifactPinV1 | None
    parent_run_pin: ArtifactPinV1 | None
    learning_pins: tuple[ArtifactPinV1, ...]
    attribution_pins: tuple[ArtifactPinV1, ...]
    handoff_pins: tuple[ArtifactPinV1, ...]

    @model_validator(mode="after")
    def preserve_run_lineage(self):
        if self.parent_run_pin is not None and self.root_run_pin is None:
            raise ValueError("rerun parent requires the retained root run")
        return self


class RuntimeReportReviewV1(_ReportContract):
    """Human review projection; no field can replace a retained machine outcome."""

    status: Literal["OPEN", "RESOLVED", "DISMISSED", "DEFERRED", "NEEDS_REVIEW"]
    disposition_pins: tuple[ArtifactPinV1, ...]
    final_user_decision: _Text | None
    review_receipt_pin: ArtifactPinV1 | None
    rationale: _Text | None
    remediation_provenance: Literal["DETERMINISTIC", "AI_ASSISTED", "NONE"]
    remediation_artifact_pin: ArtifactPinV1 | None

    @model_validator(mode="after")
    def require_retained_review_and_remediation(self):
        if (self.final_user_decision is None) != (self.review_receipt_pin is None):
            raise ValueError("final decision and review receipt must be paired")
        if self.status == "RESOLVED" and (
            self.review_receipt_pin is None or self.rationale is None
        ):
            raise ValueError("resolved review requires its receipt and rationale")
        if (self.remediation_provenance == "NONE") != (
            self.remediation_artifact_pin is None
        ):
            raise ValueError(
                "remediation provenance requires an exact persisted artifact"
            )
        return self


class ResolutionRequirementV1(_ReportContract):
    kind: Literal["EVIDENCE", "USER_DECISION", "DOMAIN_DECISION"]
    description: _Text
    requested_evidence_type: _Id | None
    decision_owner: _Id | None
    related_predicate_or_rule_pin: ArtifactPinV1 | None
    evidence_pins: tuple[ArtifactPinV1, ...]

    @model_validator(mode="after")
    def require_actionable_resolution(self):
        if self.kind == "EVIDENCE":
            if self.requested_evidence_type is None or self.decision_owner is not None:
                raise ValueError(
                    "evidence resolution requires an evidence type, not a decision owner"
                )
        elif self.decision_owner is None or self.requested_evidence_type is not None:
            raise ValueError(
                "decision resolution requires an owner, not an evidence type"
            )
        return self


class RuntimeReportUnresolvedV1(_ReportContract):
    result_pin: ArtifactPinV1
    finding_pin: ArtifactPinV1 | None
    missing_evidence_types: tuple[_Id, ...]
    missing_inputs: tuple[_Id, ...]
    competing_interpretations: tuple[_Text, ...]
    indeterminate_rule_pins: tuple[ArtifactPinV1, ...]
    absent_rule_reason: _Text | None
    resolution_requirements: Annotated[
        tuple[ResolutionRequirementV1, ...], Field(min_length=1)
    ]

    @model_validator(mode="after")
    def explain_absent_rule_basis(self):
        if not self.indeterminate_rule_pins and self.absent_rule_reason is None:
            raise ValueError(
                "unresolved result without rule pins requires an explicit reason"
            )
        return self


class ResolvedRuntimeReportPolicyBasisV1(_ReportContract):
    status: Literal["RESOLVED"]
    evidence_pins: Annotated[tuple[ArtifactPinV1, ...], Field(min_length=1)]


class MissingRuntimeReportPolicyBasisV1(_ReportContract):
    status: Literal["MISSING"]
    reason: _Text
    resolution_requirements: Annotated[
        tuple[ResolutionRequirementV1, ...], Field(min_length=1)
    ]


class NotApplicableRuntimeReportPolicyBasisV1(_ReportContract):
    status: Literal["NOT_APPLICABLE"]
    reason: _Text


RuntimeReportPolicyBasisV1 = Annotated[
    ResolvedRuntimeReportPolicyBasisV1
    | MissingRuntimeReportPolicyBasisV1
    | NotApplicableRuntimeReportPolicyBasisV1,
    Field(discriminator="status"),
]


class RuntimeReportSubjectV1(_ReportContract):
    target_pin: ArtifactPinV1
    source_pin: ArtifactPinV1
    document_title: _Text
    run_pin: ArtifactPinV1
    root_run_pin: ArtifactPinV1 | None
    parent_run_pin: ArtifactPinV1 | None
    epoch_pin: ArtifactPinV1 | None
    certified_plan_pin: ArtifactPinV1 | None
    provider_identity_pin: ArtifactPinV1 | None
    graph_pin: ArtifactPinV1 | None
    release_pin: ArtifactPinV1 | None
    normative_ir_pin: ArtifactPinV1 | None
    resolver_pin: ArtifactPinV1 | None
    rule_version_pins: tuple[ArtifactPinV1, ...]
    evaluator_pin: ArtifactPinV1 | None
    context_pin: ArtifactPinV1
    world_pin: ArtifactPinV1
    transition_pin: ArtifactPinV1 | None
    business_effective_at: Annotated[V2UtcDateTime, Field(strict=False)]
    recorded_at: Annotated[V2UtcDateTime, Field(strict=False)]


class RuntimeReportSummaryV1(_ReportContract):
    status: Literal["COMPLETE", "PARTIAL", "BLOCKED"]
    ordered_result_pins: tuple[ArtifactPinV1, ...]
    ordered_finding_pins: tuple[ArtifactPinV1, ...]
    evaluated_competency_pins: tuple[ArtifactPinV1, ...]
    material_count: int = Field(ge=0)
    resolved_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    review_pending_count: int = Field(ge=0)
    denominator_definition: _Text


class RuntimeReportFindingV1(_ReportContract):
    result_pin: ArtifactPinV1
    finding_pin: ArtifactPinV1 | None
    target_evidence_pins: tuple[ArtifactPinV1, ...]
    anchor: _Text | None
    excerpt: _Text | None
    observed_facts: dict[str, object]
    ambiguities: tuple[_Text, ...]
    outcome: Literal["MATCH", "NO_MATCH", "UNKNOWN"]
    candidate_rule_trace_pins: tuple[ArtifactPinV1, ...]
    predicate_result_pins: tuple[ArtifactPinV1, ...]
    exception_result_pins: tuple[ArtifactPinV1, ...]
    missing_inputs: tuple[_Id, ...]
    policy_basis: RuntimeReportPolicyBasisV1


class RuntimeReportReplayInputsV1(_ReportContract):
    schema_version: Literal["runtime_report_replay_inputs.v1"] = (
        "runtime_report_replay_inputs.v1"
    )
    replay_ref: _Text
    replay_hash: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    organization_id: _Text
    workspace_id: _Text
    source_pins: tuple[ArtifactPinV1, ...]
    closure_manifest: tuple[ArtifactPinV1, ...]
    renderer_pin: ArtifactPinV1
    template_pin: ArtifactPinV1
    environment_pin: ArtifactPinV1
    access_policy_pin: ArtifactPinV1 | None
    supported_projections: tuple[_Id, ...]

    @classmethod
    def create(cls, **values: object) -> "RuntimeReportReplayInputsV1":
        basis = _jsonable(values)
        assert isinstance(basis, dict)
        basis.setdefault("schema_version", "runtime_report_replay_inputs.v1")
        basis.pop("replay_ref", None)
        basis.pop("replay_hash", None)
        digest = content_hash(basis)
        return cls(
            **values,
            replay_ref=f"runtime-report-replay-inputs:{digest}",
            replay_hash=f"sha256:{digest}",
        )

    @property
    def replay_pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="runtime_report_replay_inputs",
            artifact_ref=self.replay_ref,
            artifact_hash=self.replay_hash,
        )

    @model_validator(mode="after")
    def identity_is_bound(self):
        basis = _jsonable(self.model_dump(mode="json", exclude={"replay_ref", "replay_hash"}))
        digest = content_hash(basis)
        if (self.replay_ref, self.replay_hash) != (
            f"runtime-report-replay-inputs:{digest}", f"sha256:{digest}"
        ):
            raise ValueError("replay identity does not bind exact content")
        return self


class RuntimeInspectionReportV1(_ReportContract):
    schema_version: Literal["runtime_inspection_report.v1"] = "runtime_inspection_report.v1"
    report_ref: _Text
    report_hash: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    organization_id: _Text
    workspace_id: _Text
    subject: RuntimeReportSubjectV1
    summary: RuntimeReportSummaryV1
    findings: tuple[RuntimeReportFindingV1, ...]
    unresolved: tuple[RuntimeReportUnresolvedV1, ...]
    review: RuntimeReportReviewV1
    history: RuntimeReportHistoryV1
    replay_access: RuntimeReportReplayInputsV1
    projector_pin: ArtifactPinV1
    renderer_pin: ArtifactPinV1
    template_pin: ArtifactPinV1
    environment_pin: ArtifactPinV1

    @classmethod
    def create(cls, **values: object) -> "RuntimeInspectionReportV1":
        basis = _jsonable(values)
        assert isinstance(basis, dict)
        basis.setdefault("schema_version", "runtime_inspection_report.v1")
        basis.pop("report_ref", None)
        basis.pop("report_hash", None)
        digest = content_hash(basis)
        return cls(
            **values,
            report_ref=f"runtime-inspection-report:{digest}",
            report_hash=f"sha256:{digest}",
        )

    @property
    def report_pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="runtime_inspection_report",
            artifact_ref=self.report_ref,
            artifact_hash=self.report_hash,
        )

    @model_validator(mode="after")
    def identity_is_bound(self):
        basis = _jsonable(self.model_dump(mode="json", exclude={"report_ref", "report_hash"}))
        digest = content_hash(basis)
        if (self.report_ref, self.report_hash) != (
            f"runtime-inspection-report:{digest}", f"sha256:{digest}"
        ):
            raise ValueError("report identity does not bind exact content")
        return self


class RuntimeReportReceiptV1(_ReportContract):
    schema_version: Literal["runtime_report_receipt.v1"] = "runtime_report_receipt.v1"
    receipt_ref: _Text
    receipt_hash: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    organization_id: _Text
    workspace_id: _Text
    request_id: _Text
    request_hash: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    report_pin: ArtifactPinV1
    replay_pin: ArtifactPinV1
    predecessor_pin: ArtifactPinV1 | None
    actor_ref: _Text
    recorded_at: Annotated[V2UtcDateTime, Field(strict=False)]

    @classmethod
    def create(cls, **values: object) -> "RuntimeReportReceiptV1":
        basis = _jsonable(values)
        assert isinstance(basis, dict)
        basis.setdefault("schema_version", "runtime_report_receipt.v1")
        digest = content_hash(basis)
        return cls(
            **values,
            receipt_ref=f"runtime-report-receipt:{digest}",
            receipt_hash=f"sha256:{digest}",
        )

    @model_validator(mode="after")
    def identity_is_bound(self):
        basis = _jsonable(self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"}))
        digest = content_hash(basis)
        if (self.receipt_ref, self.receipt_hash) != (
            f"runtime-report-receipt:{digest}", f"sha256:{digest}"
        ):
            raise ValueError("receipt identity does not bind exact content")
        if self.report_pin.artifact_kind != "runtime_inspection_report":
            raise ValueError("receipt report pin kind mismatch")
        if self.replay_pin.artifact_kind != "runtime_report_replay_inputs":
            raise ValueError("receipt replay pin kind mismatch")
        return self


class VerifiedRuntimeReportInputsV1(_ReportContract):
    """Already-resolved P3 closure; projection must not resolve or evaluate it."""

    organization_id: _Text
    workspace_id: _Text
    target_pin: ArtifactPinV1
    source_pin: ArtifactPinV1
    document_title: _Text
    run_pin: ArtifactPinV1
    root_run_pin: ArtifactPinV1 | None
    parent_run_pin: ArtifactPinV1 | None
    epoch_pin: ArtifactPinV1 | None
    certified_plan_pin: ArtifactPinV1 | None
    provider_identity_pin: ArtifactPinV1 | None
    graph_pin: ArtifactPinV1 | None
    release_pin: ArtifactPinV1 | None
    normative_ir_pin: ArtifactPinV1 | None
    resolver_pin: ArtifactPinV1 | None
    rule_version_pins: tuple[ArtifactPinV1, ...]
    evaluator_pin: ArtifactPinV1 | None
    context_pin: ArtifactPinV1
    world_pin: ArtifactPinV1
    transition_pin: ArtifactPinV1 | None
    result_pins: tuple[ArtifactPinV1, ...]
    evaluated_competency_pins: tuple[ArtifactPinV1, ...]
    findings: tuple[RuntimeReportFindingV1, ...]
    unknown_results: tuple[ArtifactPinV1, ...]
    status: Literal["COMPLETE", "PARTIAL", "BLOCKED"]
    resolution_requirements: tuple[ResolutionRequirementV1, ...] = ()
    disposition_pins: tuple[ArtifactPinV1, ...] = ()
    final_user_decision: _Text | None = None
    review_receipt_pin: ArtifactPinV1 | None = None
    review_status: Literal["OPEN", "RESOLVED", "DISMISSED", "DEFERRED", "NEEDS_REVIEW"] = "OPEN"
    review_rationale: _Text | None = None
    remediation_provenance: Literal["DETERMINISTIC", "AI_ASSISTED", "NONE"] = "NONE"
    remediation_artifact_pin: ArtifactPinV1 | None = None
    modification_events: tuple[RuntimeReportModificationV1, ...] = ()
    predecessor_report_pin: ArtifactPinV1 | None = None
    successor_link_receipt_pins: tuple[ArtifactPinV1, ...] = ()
    learning_pins: tuple[ArtifactPinV1, ...] = ()
    attribution_pins: tuple[ArtifactPinV1, ...] = ()
    handoff_pins: tuple[ArtifactPinV1, ...] = ()
    business_effective_at: Annotated[V2UtcDateTime, Field(strict=False)]
    recorded_at: Annotated[V2UtcDateTime, Field(strict=False)]

    @model_validator(mode="after")
    def validate_unknown_resolution(self):
        if self.unknown_results and not self.resolution_requirements:
            raise ValueError("UNKNOWN results require actionable resolution requirements")
        return self
