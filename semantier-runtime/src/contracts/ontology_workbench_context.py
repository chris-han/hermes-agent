"""Strict shared-context and presentation-action contracts for the ontology workbench."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from copy import deepcopy
from datetime import timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from eos.artifact_hashing import content_hash

from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import ContextGraph, Evidence
from contracts.candidate_review import CandidateReviewDisclosureV1
from contracts.workbench_action_errors import (
    WorkbenchActionErrorV6 as WorkbenchActionErrorV6,
    ErrorDetailsV6 as ErrorDetailsV6,
)
from contracts.runtime_inspection_report import RuntimeInspectionReportV1
from contracts.evidence_location import ResolvedLocation
from contracts.ontology_repair import (
    AdvisoryNextProposalReceiptV1,
    AdvisoryNextProposalRequestV1,
)
from contracts.runtime_context_reconstruction import (
    AdvisoryContextReceiptV1,
    AdvisoryContextRequestV1,
    NonEmptyStr,
    Sha256,
)
from contracts.runtime_semantic_evaluation import (
    GovernedTargetPinV1,
    RuntimeEvaluationRunV1,
    RuntimeEvaluationStateV1,
    RuntimeProviderResultV1,
    RuntimeTemporalProviderRequestV1,
    TemporalQueryV1,
)
from contracts.runtime_temporal_source import (
    RuntimeTemporalSourceArchiveV1,
    RuntimeTemporalSourceGraphV1,
)
from contracts.runtime_working_context import V2UtcDateTime


RuntimeSemanticSubtype = Literal[
    "CONFIRMED",
    "FALSE_POSITIVE",
    "FALSE_NEGATIVE",
    "WRONG_REASON",
    "EXCEPTION_CORRECTION",
    "DISPUTED",
    "NEEDS_MORE_EVIDENCE",
]


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


def projection_hash_basis(value: Mapping[str, object]) -> dict[str, object]:
    """Hash basis of the serialized camelCase projection, not a native DTO.

    Call after wire serialization. Do not remove similarly named fields in
    artifact payloads or receipt pins: those are semantic inputs, not echoes.
    """
    basis = deepcopy(dict(value))
    projections = [basis]
    if isinstance(basis.get("knowledge"), dict):
        projections.append(basis["knowledge"])
    for projection in projections:
        projection.pop("projectionHash", None)
        for capability in projection.get("capabilities", []):
            capability.pop("expectedProjectionHash", None)
        evidence = projection.get("evidence")
        if isinstance(evidence, dict):
            for member in evidence.get("members", []):
                if not isinstance(member, dict):
                    continue
                resolved = member.get("location")
                if isinstance(resolved, dict):
                    # Renderer geometry and resolution timestamps are late-bound
                    # presentation state. Canonical EvidenceRef/source/occurrence
                    # identity remains hashed through the surrounding member.
                    resolved.pop("location", None)
                    resolved.pop("resolvedAt", None)
    return basis


def verify_projection_hash(value: Mapping[str, object]) -> str:
    """Check a server projection's digest and every declared mirrored echo.

    This verifies identity only; the V3 model owns shape and common-field
    equality, and the resolver owns current authorization.
    """
    digest = "sha256:" + content_hash(projection_hash_basis(value))
    projections = [value]
    if isinstance(value.get("knowledge"), dict):
        projections.append(value["knowledge"])
    for projection in projections:
        if projection.get("projectionHash") != digest:
            raise ValueError("projection hash does not match canonical basis")
        artifact_workspace = projection.get("artifactWorkspace")
        material_hash = (
            artifact_workspace.get("material_projection_hash")
            if isinstance(artifact_workspace, dict)
            else None
        )
        for capability in projection.get("capabilities", []):
            expected_material_hash = capability.get("expectedProjectionHash")
            if expected_material_hash is not None and expected_material_hash != material_hash:
                raise ValueError("capability material projection hash does not match cursor basis")
    return digest


WorkbenchView = Literal["graph", "evidence", "lineage", "compare", "evaluate"]
WorkbenchCommand = Literal[
    "FOCUS_OBJECT",
    "OPEN_VIEW",
    "SET_GRAPH_LENS",
    "OPEN_EVIDENCE",
    "OPEN_INSPECTOR",
    "OPEN_LINEAGE",
    "SET_COMPARISON_PAIR",
    "OPEN_COMPARE",
]
WorkbenchActionError = Literal[
    "STALE_CONTEXT",
    "UNRESOLVABLE_REF",
    "FORBIDDEN",
    "UNSUPPORTED_VIEW",
    "INCOMPATIBLE_COMPARISON_PAIR",
]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkbenchEvidenceMemberV1(_Contract):
    """One ordered membership result, including an explicit unresolved member."""

    schema_version: Literal["workbench_evidence_member.v1"]
    evidence_ref: NonEmptyStr
    evidence_pin: ArtifactPinV1 | None
    occurrence_ref: NonEmptyStr | None
    location: ResolvedLocation

    @model_validator(mode="after")
    def verify_member(self):
        if self.location.evidence_ref != self.evidence_ref:
            raise ValueError("location must belong to the evidence member")
        if self.evidence_pin is not None and (self.evidence_pin.artifact_kind != "evidence_ref" or self.evidence_pin.artifact_ref != self.evidence_ref):
            raise ValueError("evidence pin must identify the member")
        if self.occurrence_ref is not None and self.evidence_pin is None:
            raise ValueError("an occurrence requires a verified evidence pin")
        if self.evidence_pin is None and self.location.status not in {"unresolved", "ambiguous", "source_changed"}:
            raise ValueError("resolved membership requires an evidence pin")
        if self.evidence_pin is not None and (not self.evidence_ref.startswith("evidence:") or self.evidence_pin.artifact_hash != "sha256:" + self.evidence_ref.removeprefix("evidence:")):
            raise ValueError("evidence pin must preserve native identity hash")
        return self


class EvidenceFocusPayloadV1(_Contract):
    """Proposed focus only; resolving access and locations belongs to the owner."""

    command: Literal["FOCUS_EVIDENCE"]
    object_pin: ArtifactPinV1
    evidence_refs: Annotated[tuple[ArtifactPinV1, ...], Field(min_length=1)]
    selected_evidence_ref: ArtifactPinV1 | None
    canonical_anchor: ArtifactPinV1 | None
    target_view: Literal["EVIDENCE"]
    expected_selection_pin: ArtifactPinV1 | None
    expected_selection_revision: Annotated[int, Field(strict=True, ge=0)]

    @model_validator(mode="after")
    def verify_selected_member(self):
        if (
            self.selected_evidence_ref is not None
            and self.selected_evidence_ref not in self.evidence_refs
        ):
            raise ValueError("selected evidence must match an exact evidence pin")
        return self


class WorkbenchEvidenceFocusV1(_Contract):
    """Immutable navigation fact; creation requires owner-validated locations."""

    schema_version: Literal["workbench_evidence_focus.v1"]
    focus_ref: NonEmptyStr
    focus_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    session_id: NonEmptyStr
    actor_ref: NonEmptyStr
    lifecycle: Literal["DESIGN", "RUNTIME"]
    revision: Annotated[int, Field(strict=True, ge=0)]
    predecessor_pin: ArtifactPinV1 | None
    parent_pin: ArtifactPinV1
    artifact_pin: ArtifactPinV1
    preparation_pin: ArtifactPinV1
    occurrence_ref: NonEmptyStr | None
    focus: EvidenceFocusPayloadV1
    recorded_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_focus_identity(self):
        if self.artifact_pin.artifact_kind != "workspace_artifact" or self.preparation_pin.artifact_kind != "workspace_artifact_preparation":
            raise ValueError("focus requires exact artifact and preparation pins")
        if (self.focus.selected_evidence_ref is None) != (self.occurrence_ref is None):
            raise ValueError("singular evidence focus requires one occurrence ref")
        if self.recorded_at.utcoffset() != timedelta(0):
            raise ValueError("recorded_at must be timezone-aware UTC")
        if (self.revision == 0) != (self.predecessor_pin is None):
            raise ValueError("only genesis focus has no predecessor")
        if self.predecessor_pin is not None and self.predecessor_pin.artifact_kind != "workbench_evidence_focus":
            raise ValueError("invalid evidence focus predecessor kind")
        digest = content_hash(self.model_dump(mode="json", exclude={"focus_ref", "focus_hash"}))
        if (self.focus_ref, self.focus_hash) != (f"workbench-evidence-focus:{digest}", f"sha256:{digest}"):
            raise ValueError("evidence focus identity mismatch")
        return self


ActionCommandV6 = Literal[
    "FOCUS_EVIDENCE", "OPEN_ORIGINATING_RUN", "OPEN_DESIGN_SUCCESSOR",
    "UPLOAD_ARTIFACT", "SELECT_ARTIFACT", "PREPARE_ARTIFACT", "BIND_SOURCE",
    "BIND_TARGET", "CLEAR_SOURCE", "CLEAR_TARGET", "START_RUNTIME", "RUN_EVALUATION",
    "OPEN_EVALUATION", "CREATE_REPORT", "EXPORT_REPORT", "DOWNLOAD_EXPORT",
    "SUBMIT_RUNTIME_CORRECTION", "SUBMIT_SEMANTIC_GRAPH_EDIT", "REVIEW_RUNTIME_CORRECTION",
    "EVALUATE_RUNTIME_REPAIR_PAIR", "OPEN_COMPARISON",
    "REQUEST_RUNTIME_LEARNING_FUTURE_RUN",
]
AsciiWorkbenchId = Annotated[str, Field(strict=True, min_length=1, pattern=r"^[\x20-\x7e]+$")]


class FocusEvidenceActionEnvelopeV6(_Contract):
    """FOCUS_EVIDENCE variant; not the full v6 action union."""

    schema_version: Literal["workbench_action_envelope.v6"]
    command: Literal["FOCUS_EVIDENCE"]
    request_id: AsciiWorkbenchId
    idempotency_key: AsciiWorkbenchId
    expected_material_context_revision: Annotated[int, Field(strict=True, ge=0)]
    cursor_ref: AsciiWorkbenchId
    expected_projection_hash: Sha256
    ingress: Literal["DIRECT", "CHAT"]
    payload: EvidenceFocusPayloadV1

    @model_validator(mode="after")
    def verify_ascii_pins(self):
        pins = (self.payload.object_pin, *self.payload.evidence_refs,
                self.payload.selected_evidence_ref, self.payload.canonical_anchor,
                self.payload.expected_selection_pin)
        if any(pin is not None and (not pin.artifact_ref.isascii() or not pin.artifact_kind.isascii()) for pin in pins):
            raise ValueError("v6 identifiers must be ASCII-stable")
        return self


class WorkbenchActionAcknowledgementV6(_Contract):
    """Content-addressed successful presentation result, not mutation authority."""

    schema_version: Literal["workbench_action_acknowledgement.v6"]
    receipt_ref: AsciiWorkbenchId
    receipt_hash: Sha256
    organization_id: AsciiWorkbenchId
    workspace_id: AsciiWorkbenchId
    session_id: AsciiWorkbenchId
    actor_ref: AsciiWorkbenchId
    command: ActionCommandV6
    accepted: Annotated[bool, Field(strict=True)]
    result_pin: ArtifactPinV1
    cursor_pin: ArtifactPinV1
    material_context_revision: Annotated[int, Field(strict=True, ge=0)]
    projection_hash: Sha256
    request_id: AsciiWorkbenchId
    idempotency_key: AsciiWorkbenchId
    request_hash: Sha256
    ingress: Literal["DIRECT", "CHAT"]
    recorded_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_acknowledgement(self):
        if self.command not in {"FOCUS_EVIDENCE", "OPEN_ORIGINATING_RUN", "OPEN_DESIGN_SUCCESSOR", "OPEN_EVALUATION", "OPEN_COMPARISON"}:
            raise ValueError("mutation requires its owning operation receipt")
        if not self.accepted or self.recorded_at.utcoffset() != timedelta(0):
            raise ValueError("acknowledgement requires success and UTC time")
        if self.cursor_pin.artifact_kind != "workbench_projection_cursor_head":
            raise ValueError("acknowledgement requires a cursor head")
        if self.command == "FOCUS_EVIDENCE" and self.result_pin.artifact_kind != "workbench_evidence_focus":
            raise ValueError("focus acknowledgement requires an exact focus pin")
        if any(not pin.artifact_ref.isascii() or not pin.artifact_kind.isascii() for pin in (self.result_pin, self.cursor_pin)):
            raise ValueError("v6 identifiers must be ASCII-stable")
        digest = content_hash(self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"}))
        if (self.receipt_ref, self.receipt_hash) != ("workbench-action-acknowledgement:" + digest, "sha256:" + digest):
            raise ValueError("acknowledgement identity mismatch")
        return self


class WorkbenchActionReceiptV6(_Contract):
    schema_version: Literal["workbench_action_receipt.v6"] = "workbench_action_receipt.v6"
    receipt_kind: Literal[
        "WORKSPACE_ARTIFACT_OPERATION", "RUNTIME_EVALUATION", "RUNTIME_REPORT",
        "RUNTIME_EXPORT", "CHANGE_CANDIDATE", "CANDIDATE_REVIEW", "CANDIDATE_EVALUATION",
        "ONTOLOGY_ADMISSION", "SEMANTIC_ACTIVATION", "FUTURE_RUN",
        "PRESENTATION_ACKNOWLEDGEMENT",
    ]
    receipt_pin: ArtifactPinV1
    receipt_payload: WorkbenchActionAcknowledgementV6 | None
    candidate_review: CandidateReviewDisclosureV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    operation_status: Literal["PENDING", "COMPLETE", "FAILED", "BLOCKED"]
    request_id: AsciiWorkbenchId
    request_hash: Sha256
    idempotency_key: AsciiWorkbenchId
    recorded_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_presentation_receipt(self):
        if self.candidate_review is not None and (self.receipt_kind != "CANDIDATE_REVIEW" or self.receipt_pin != self.candidate_review.decision_pin):
            raise ValueError("review receipt must pin its DecisionRecord")
        if self.receipt_kind == "PRESENTATION_ACKNOWLEDGEMENT":
            if self.receipt_payload is None or self.receipt_pin != ArtifactPinV1(
                artifact_kind="workbench_action_acknowledgement",
                artifact_ref=self.receipt_payload.receipt_ref,
                artifact_hash=self.receipt_payload.receipt_hash,
            ):
                raise ValueError("presentation receipt must pin its acknowledgement")
            if self.operation_status != "COMPLETE":
                raise ValueError("presentation acknowledgement is complete")
        return self


class CompletedWorkbenchActionResultV6(_Contract):
    schema_version: Literal["workbench_action_result.v6"] = "workbench_action_result.v6"
    status: Literal["COMPLETED"] = "COMPLETED"
    command: ActionCommandV6
    result_pin: ArtifactPinV1
    receipt: WorkbenchActionReceiptV6
    context_projection: "UnifiedWorkbenchContextV3"
    recorded_at: V2UtcDateTime


_V6Revision = Annotated[int, Field(strict=True, ge=0)]
_V6Pins = Annotated[tuple[ArtifactPinV1, ...], Field(min_length=1)]


class _HandoffNavigationPayloadV6(_Contract):
    handoff_pin: ArtifactPinV1
    exact_operand_pins: _V6Pins
    origin_anchor: ArtifactPinV1
    expected_selection_pin: ArtifactPinV1 | None
    expected_selection_revision: _V6Revision


class _OpenOriginatingRunPayloadV6(_HandoffNavigationPayloadV6):
    command: Literal["OPEN_ORIGINATING_RUN"]
    destination: Literal["RUNTIME_EVALUATE", "RUNTIME_EVIDENCE"]


class _OpenDesignSuccessorPayloadV6(_HandoffNavigationPayloadV6):
    command: Literal["OPEN_DESIGN_SUCCESSOR"]
    destination: Literal["DESIGN_GRAPH"]


class _OpenEvaluationPayloadV6(_HandoffNavigationPayloadV6):
    command: Literal["OPEN_EVALUATION"]
    handoff_pin: ArtifactPinV1 | None
    destination: Literal["RUNTIME_EVALUATE"]
    origin_anchor: ArtifactPinV1 | None


class _OpenComparisonPayloadV6(_HandoffNavigationPayloadV6):
    command: Literal["OPEN_COMPARISON"]
    destination: Literal["DESIGN_COMPARE"]


class _UploadArtifactPayloadV6(_Contract):
    # multipart_file is the separate binary HTTP part, never a JSON path/value.
    command: Literal["UPLOAD_ARTIFACT"]
    expected_collection_revision: _V6Revision
    original_name: NonEmptyStr
    declared_media_type: NonEmptyStr
    byte_length: Annotated[int, Field(strict=True, gt=0)]
    byte_hash: Sha256
    ingress: Literal["DIRECT", "CHAT"]


class _ArtifactNavigationPayloadV6(_Contract):
    artifact_pin: ArtifactPinV1
    expected_collection_revision: _V6Revision
    cursor_ref: AsciiWorkbenchId
    expected_projection_hash: Sha256
    expected_material_context_revision: _V6Revision
    ingress: Literal["DIRECT", "CHAT"]


class _SelectArtifactPayloadV6(_ArtifactNavigationPayloadV6):
    command: Literal["SELECT_ARTIFACT"]
    expected_selection_pin: ArtifactPinV1 | None
    expected_selection_revision: _V6Revision


class _PrepareArtifactPayloadV6(_Contract):
    command: Literal["PREPARE_ARTIFACT"]
    artifact_pin: ArtifactPinV1
    purpose: Literal["DESIGN_SOURCE", "RUNTIME_TARGET"]
    expected_preparation_pin: ArtifactPinV1 | None
    expected_preparation_revision: _V6Revision | None
    ingress: Literal["DIRECT", "CHAT"]

    @model_validator(mode="after")
    def paired_preparation(self):
        if (self.expected_preparation_pin is None) != (self.expected_preparation_revision is None):
            raise ValueError("preparation pin and revision must be paired")
        return self


class _BindingPayloadV6(_ArtifactNavigationPayloadV6):
    expected_binding_set_pin: ArtifactPinV1
    expected_binding_set_revision: _V6Revision


class _BindArtifactPayloadV6(_BindingPayloadV6):
    preparation_pin: ArtifactPinV1
    select_after_bind: Annotated[bool, Field(strict=True)]
    expected_selection_pin: ArtifactPinV1 | None
    expected_selection_revision: _V6Revision | None

    @model_validator(mode="after")
    def paired_selection(self):
        if (self.expected_selection_pin is None) != (self.expected_selection_revision is None):
            raise ValueError("selection pin and revision must be paired")
        if self.select_after_bind and self.expected_selection_pin is None:
            raise ValueError("atomic selection requires its expected head")
        return self


class _BindSourcePayloadV6(_BindArtifactPayloadV6):
    command: Literal["BIND_SOURCE"]
    lifecycle: Literal["DESIGN"]
    role: Literal["SOURCE"]
    insert_at: _V6Revision | None


class _BindTargetPayloadV6(_BindArtifactPayloadV6):
    command: Literal["BIND_TARGET"]
    lifecycle: Literal["RUNTIME"]
    role: Literal["TARGET"]
    insert_at: None


class _ClearSourcePayloadV6(_BindingPayloadV6):
    command: Literal["CLEAR_SOURCE"]
    lifecycle: Literal["DESIGN"]
    role: Literal["SOURCE"]


class _ClearTargetPayloadV6(_BindingPayloadV6):
    command: Literal["CLEAR_TARGET"]
    lifecycle: Literal["RUNTIME"]
    role: Literal["TARGET"]


class _StartRuntimePayloadV6(_Contract):
    command: Literal["START_RUNTIME"]
    source_graph_pin: ArtifactPinV1
    ontology_pin: ArtifactPinV1
    evidence_pins: _V6Pins
    target: GovernedTargetPinV1
    selected_entity_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    preparation_pin: ArtifactPinV1
    world_time: V2UtcDateTime

    @model_validator(mode="after")
    def verify_start_runtime(self):
        if self.source_graph_pin.artifact_kind != "runtime_temporal_source_graph":
            raise ValueError("runtime start requires runtime_temporal_source_graph")
        if self.ontology_pin.artifact_kind != "ontology":
            raise ValueError("runtime start requires kernel ontology")
        if any(pin.artifact_kind != "evidence" for pin in self.evidence_pins):
            raise ValueError("runtime start requires kernel evidence")
        if self.preparation_pin.artifact_kind != "workspace_artifact_preparation":
            raise ValueError("runtime start requires workspace_artifact_preparation")
        if self.target.target_type != "graph" or self.target.artifact != self.source_graph_pin:
            raise ValueError("runtime start target must identify the exact source graph")
        if len(self.selected_entity_ids) != len(set(self.selected_entity_ids)):
            raise ValueError("runtime start entity ids must be unique")
        if self.world_time.utcoffset() != timedelta(0):
            raise ValueError("runtime start world_time must be aware UTC")
        return self


class RuntimeEvaluationCommandV1(_Contract):
    command: Literal["RUN_EVALUATION"]
    context_pin: ArtifactPinV1
    context_head_pin: ArtifactPinV1
    target: GovernedTargetPinV1
    certified_plan_pin: ArtifactPinV1
    provider_request_pin: ArtifactPinV1
    business_effective_at: V2UtcDateTime
    parent_run_pin: ArtifactPinV1 | None
    expected_run_head_pin: ArtifactPinV1 | None


class RuntimeReportCommandV1(_Contract):
    command: Literal["CREATE_REPORT"]
    run_pin: ArtifactPinV1
    result_pins: _V6Pins
    disposition_pins: tuple[ArtifactPinV1, ...]
    predecessor_report_pin: ArtifactPinV1 | None
    projector_pin: ArtifactPinV1
    renderer_pin: ArtifactPinV1
    template_pin: ArtifactPinV1
    environment_pin: ArtifactPinV1


class RuntimeExportCommandV1(_Contract):
    command: Literal["EXPORT_REPORT"]
    report_pin: ArtifactPinV1
    replay_inputs_pin: ArtifactPinV1
    format: Literal["HTML", "DOCX", "JSON", "EVIDENCE_PACKAGE", "REPLAY_PACKAGE", "RDF", "TRIPLETS", "QUERY_RESULTS"]
    access_projection: Literal["MEMBER", "AUDIT"]
    render_profile_pin: ArtifactPinV1
    expected_export_head_pin: ArtifactPinV1 | None


class _DownloadExportPayloadV6(_Contract):
    command: Literal["DOWNLOAD_EXPORT"]
    receipt_pin: ArtifactPinV1
    published_artifact_pin: ArtifactPinV1
    expected_content_hash: Sha256
    expected_size_bytes: _V6Revision


class _RenameSemanticNodeV1(_Contract):
    kind: Literal["RENAME_NODE"]
    node_ref: AsciiWorkbenchId
    previous_label: NonEmptyStr
    proposed_label: NonEmptyStr


class _AddSemanticRelationV1(_Contract):
    kind: Literal["ADD_RELATION"]
    source_node_ref: AsciiWorkbenchId
    target_node_ref: AsciiWorkbenchId
    proposed_relation_type: AsciiWorkbenchId


class _ReplaceSemanticRelationV1(_Contract):
    kind: Literal["REPLACE_RELATION"]
    edge_ref: AsciiWorkbenchId
    source_node_ref: AsciiWorkbenchId
    target_node_ref: AsciiWorkbenchId
    previous_relation_type: AsciiWorkbenchId
    proposed_relation_type: AsciiWorkbenchId


class _RemoveSemanticRelationV1(_Contract):
    kind: Literal["REMOVE_RELATION"]
    edge_ref: AsciiWorkbenchId
    source_node_ref: AsciiWorkbenchId
    target_node_ref: AsciiWorkbenchId
    previous_relation_type: AsciiWorkbenchId


class SemanticGraphEditV1(_Contract):
    """Renderer-neutral intent; exact observations are resolved by server owners."""
    schema_version: Literal["semantic_graph_edit.v1"]
    graph: ArtifactPinV1
    operation: Annotated[
        _RenameSemanticNodeV1 | _AddSemanticRelationV1 | _ReplaceSemanticRelationV1 | _RemoveSemanticRelationV1,
        Field(discriminator="kind"),
    ]
    affected_object_refs: tuple[AsciiWorkbenchId, ...] = Field(min_length=1)
    evidence_refs: tuple[AsciiWorkbenchId, ...] = Field(min_length=1)
    rationale: NonEmptyStr | None = Field(default=None, exclude_if=lambda value: value is None)
    source: Literal["DIRECT", "CHAT"]

    @model_validator(mode="before")
    @classmethod
    def reject_null_rationale(cls, value):
        if isinstance(value, dict) and "rationale" in value and value["rationale"] is None:
            raise ValueError("rationale must be a string when supplied")
        return value

    @model_validator(mode="after")
    def verify_edit(self):
        if self.graph.artifact_kind not in {"context_graph", "exploratory_graph_snapshot"}:
            raise ValueError("invalid semantic graph kind")
        for refs in (self.affected_object_refs, self.evidence_refs):
            if len(refs) != len(set(refs)):
                raise ValueError("duplicate semantic refs")
        op = self.operation
        expected = {getattr(op, field) for field in ("node_ref", "edge_ref", "source_node_ref", "target_node_ref") if hasattr(op, field)}
        if set(self.affected_object_refs) != expected:
            raise ValueError("affected refs must exactly identify operation operands")
        for previous, proposed in (("previous_label", "proposed_label"), ("previous_relation_type", "proposed_relation_type")):
            if hasattr(op, previous) and hasattr(op, proposed) and getattr(op, previous) == getattr(op, proposed):
                raise ValueError("NO_CHANGE_REQUIRED")
        return self


class SemanticGraphEditCommandV1(_Contract):
    command: Literal["SUBMIT_SEMANTIC_GRAPH_EDIT"]
    graph_pin: ArtifactPinV1
    edit: SemanticGraphEditV1
    evidence_pins: _V6Pins
    rationale: NonEmptyStr

    @model_validator(mode="after")
    def verify_pins(self):
        if self.graph_pin != self.edit.graph:
            raise ValueError("semantic graph pins must match")
        if any(pin.artifact_kind != "evidence" for pin in self.evidence_pins):
            raise ValueError("semantic edits require Evidence pins")
        if {pin.artifact_ref for pin in self.evidence_pins} != set(self.edit.evidence_refs):
            raise ValueError("semantic evidence refs must match exact pins")
        return self


class RuntimeCorrectionCommandV1(_Contract):
    command: Literal["SUBMIT_RUNTIME_CORRECTION"]
    report_pin: ArtifactPinV1
    result_pin: ArtifactPinV1
    finding_pin: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    evidence_pins: _V6Pins
    correction_kind: RuntimeSemanticSubtype
    rationale: NonEmptyStr
    independent_holdout_context_pin: ArtifactPinV1 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def verify_semantic_correction(self):
        if self.report_pin.artifact_kind != "runtime_inspection_report" or self.result_pin.artifact_kind != "decision_record":
            raise ValueError("semantic correction pins have invalid kinds")
        if self.finding_pin is not None and self.finding_pin.artifact_kind != "runtime_finding":
            raise ValueError("finding pin has invalid kind")
        if (
            self.independent_holdout_context_pin is not None
            and self.independent_holdout_context_pin.artifact_kind != "context_graph"
        ):
            raise ValueError("independent holdout must pin ContextGraph")
        return self


class _ReviewRuntimeCorrectionPayloadV6(_Contract):
    command: Literal["REVIEW_RUNTIME_CORRECTION"]
    change_candidate_pin: ArtifactPinV1
    outcome: Literal["ACCEPTED", "REJECTED", "NEEDS_MORE_EVIDENCE"]
    semantic_subtype: RuntimeSemanticSubtype | None = Field(default=None, exclude_if=lambda value: value is None)
    rationale: NonEmptyStr

    @model_validator(mode="after")
    def verify_review_candidate(self):
        if self.change_candidate_pin.artifact_kind != "change_candidate":
            raise ValueError("review requires exact ChangeCandidate proposal")
        return self


class _EvaluateRuntimeRepairPairPayloadV6(_Contract):
    command: Literal["EVALUATE_RUNTIME_REPAIR_PAIR"]
    change_candidate_pin: ArtifactPinV1
    matched_baseline_decision_pin: ArtifactPinV1
    matched_successor_decision_pin: ArtifactPinV1
    matched_execution_attempt_pin: ArtifactPinV1
    holdout_baseline_decision_pin: ArtifactPinV1
    holdout_successor_decision_pin: ArtifactPinV1
    holdout_execution_attempt_pin: ArtifactPinV1
    successor_input_pin: ArtifactPinV1

    @model_validator(mode="after")
    def verify_candidate_pair(self):
        if self.change_candidate_pin.artifact_kind != "change_candidate":
            raise ValueError("paired evaluation requires exact ChangeCandidate proposal")
        for pin in (
            self.matched_baseline_decision_pin,
            self.matched_successor_decision_pin,
            self.holdout_baseline_decision_pin,
            self.holdout_successor_decision_pin,
        ):
            if pin.artifact_kind != "decision_record":
                raise ValueError("paired evaluation requires exact DecisionRecord operands")
        return self


class _RequestRuntimeLearningFutureRunPayloadV6(_Contract):
    command: Literal["REQUEST_RUNTIME_LEARNING_FUTURE_RUN"]
    change_candidate_pin: ArtifactPinV1
    context_pin: ArtifactPinV1
    target: GovernedTargetPinV1
    provider_request_pin: ArtifactPinV1
    business_effective_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_future_run_candidate(self):
        if self.change_candidate_pin.artifact_kind != "change_candidate":
            raise ValueError("future run requires accepted ChangeCandidate")
        if self.context_pin.artifact_kind != "context_graph":
            raise ValueError("future run requires kernel ContextGraph")
        return self


ActionPayloadV6 = Annotated[
    EvidenceFocusPayloadV1 | _OpenOriginatingRunPayloadV6 | _OpenDesignSuccessorPayloadV6
    | _UploadArtifactPayloadV6 | _SelectArtifactPayloadV6 | _PrepareArtifactPayloadV6
    | _BindSourcePayloadV6 | _BindTargetPayloadV6 | _ClearSourcePayloadV6 | _ClearTargetPayloadV6
    | _StartRuntimePayloadV6 | RuntimeEvaluationCommandV1 | _OpenEvaluationPayloadV6 | RuntimeReportCommandV1
    | RuntimeExportCommandV1 | _DownloadExportPayloadV6 | RuntimeCorrectionCommandV1 | SemanticGraphEditCommandV1
    | _ReviewRuntimeCorrectionPayloadV6 | _EvaluateRuntimeRepairPairPayloadV6
    | _OpenComparisonPayloadV6 | _RequestRuntimeLearningFutureRunPayloadV6,
    Field(discriminator="command"),
]


def _validate_v6_nested_values(model: BaseModel) -> None:
    """Check identifiers/times without applying ASCII restrictions to user prose."""
    from datetime import datetime

    for field in type(model).model_fields:
        value = getattr(model, field)
        if isinstance(value, str) and (field.endswith(("_ref", "_id")) or field == "artifact_kind") and not value.isascii():
            raise ValueError("v6 identifiers must be ASCII-stable")
        if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() != timedelta(0)):
            raise ValueError("v6 timestamps must be aware UTC")
        if isinstance(value, BaseModel):
            _validate_v6_nested_values(value)
        elif isinstance(value, tuple):
            for item in value:
                if isinstance(item, BaseModel):
                    _validate_v6_nested_values(item)


class WorkbenchActionEnvelopeV6(_Contract):
    """All accepted v6 input variants; parsing alone never executes an action."""

    schema_version: Literal["workbench_action_envelope.v6"]
    command: ActionCommandV6
    request_id: AsciiWorkbenchId
    idempotency_key: AsciiWorkbenchId
    expected_material_context_revision: _V6Revision
    cursor_ref: AsciiWorkbenchId
    expected_projection_hash: Sha256
    ingress: Literal["DIRECT", "CHAT"]
    payload: ActionPayloadV6

    @model_validator(mode="after")
    def verify_envelope(self):
        if self.command != self.payload.command:
            raise ValueError("envelope and payload commands must match")
        for field in ("cursor_ref", "expected_material_context_revision", "expected_projection_hash", "ingress"):
            if hasattr(self.payload, field) and getattr(self.payload, field) != getattr(self, field):
                raise ValueError("envelope and payload preconditions must match")
        if isinstance(self.payload, SemanticGraphEditCommandV1) and self.payload.edit.source != self.ingress:
            raise ValueError("semantic edit source must match ingress")
        _validate_v6_nested_values(self)
        return self


class StagedWorkspaceArtifactV1(_Contract):
    schema_version: Literal["staged_workspace_artifact.v1"]
    stage_ref: Annotated[str, Field(pattern=r"^artifacts/workspace-originals/\.staging/[0-9a-f]{32}$")]
    content_hash: Sha256
    byte_size: Annotated[int, Field(strict=True, ge=0)]
    detected_media_type: NonEmptyStr


class _WorkspaceArtifactCommand(_Contract):
    request_id: NonEmptyStr
    idempotency_key: NonEmptyStr
    ingress: Literal["DIRECT", "CHAT"]

    @model_validator(mode="after")
    def verify_request_alias(self):
        if self.request_id != self.idempotency_key:
            raise ValueError("idempotency_key must equal request_id")
        return self


class UploadWorkspaceArtifactCommandV1(_WorkspaceArtifactCommand):
    schema_version: Literal["upload_workspace_artifact_command.v1"]
    expected_collection_revision: Annotated[int, Field(strict=True, ge=0)]
    original_name: NonEmptyStr
    declared_media_type: NonEmptyStr
    byte_length: Annotated[int, Field(strict=True, ge=0)]
    byte_hash: Sha256


class _WorkspaceArtifactPinnedCommand(_WorkspaceArtifactCommand):
    artifact_pin: ArtifactPinV1

    @model_validator(mode="after")
    def verify_artifact_kind(self):
        if self.artifact_pin.artifact_kind != "workspace_artifact":
            raise ValueError("command must pin workspace_artifact")
        return self


class PrepareWorkspaceArtifactCommandV1(_WorkspaceArtifactPinnedCommand):
    schema_version: Literal["prepare_workspace_artifact_command.v1"]
    purpose: Literal["DESIGN_SOURCE", "RUNTIME_TARGET"]
    expected_preparation_pin: ArtifactPinV1 | None
    expected_preparation_revision: Annotated[int, Field(strict=True, ge=0)] | None

    @model_validator(mode="after")
    def verify_preparation_expectation(self):
        if (self.expected_preparation_pin is None) != (self.expected_preparation_revision is None):
            raise ValueError("expected preparation pin and revision must be paired")
        if self.expected_preparation_pin is not None and self.expected_preparation_pin.artifact_kind != "workspace_artifact_preparation":
            raise ValueError("invalid expected preparation kind")
        return self


class _WorkspaceArtifactNavigationCommand(_WorkspaceArtifactPinnedCommand):
    expected_collection_revision: Annotated[int, Field(strict=True, ge=0)]
    cursor_ref: NonEmptyStr
    expected_projection_hash: Sha256
    expected_material_context_revision: Annotated[int, Field(strict=True, ge=0)]


class SelectWorkspaceArtifactCommandV1(_WorkspaceArtifactNavigationCommand):
    schema_version: Literal["select_workspace_artifact_command.v1"]
    expected_selection_pin: ArtifactPinV1
    expected_selection_revision: Annotated[int, Field(strict=True, ge=0)]

    @model_validator(mode="after")
    def verify_selection_kind(self):
        if self.expected_selection_pin.artifact_kind != "workspace_artifact_selection":
            raise ValueError("invalid expected selection kind")
        return self


class _LifecycleArtifactCommand(_WorkspaceArtifactNavigationCommand):
    lifecycle: Literal["DESIGN", "RUNTIME"]
    role: Literal["SOURCE", "TARGET"]
    expected_binding_set_pin: ArtifactPinV1
    expected_binding_set_revision: Annotated[int, Field(strict=True, ge=0)]

    @model_validator(mode="after")
    def verify_binding_expectation(self):
        if (self.lifecycle, self.role) not in (("DESIGN", "SOURCE"), ("RUNTIME", "TARGET")):
            raise ValueError("invalid lifecycle/role")
        if self.expected_binding_set_pin.artifact_kind != "lifecycle_artifact_binding_set":
            raise ValueError("invalid expected binding-set kind")
        return self


class BindLifecycleArtifactCommandV1(_LifecycleArtifactCommand):
    schema_version: Literal["bind_lifecycle_artifact_command.v1"]
    preparation_pin: ArtifactPinV1 | None
    insert_at: Annotated[int, Field(strict=True, ge=0)] | None
    select_after_bind: Annotated[bool, Field(strict=True)]
    expected_selection_pin: ArtifactPinV1 | None
    expected_selection_revision: Annotated[int, Field(strict=True, ge=0)] | None

    @model_validator(mode="after")
    def verify_atomic_selection(self):
        if (self.expected_selection_pin is None) != (self.expected_selection_revision is None):
            raise ValueError("expected selection pin and revision must be paired")
        if self.select_after_bind and self.expected_selection_pin is None:
            raise ValueError("atomic selection requires expected selection head")
        if self.expected_selection_pin is not None and self.expected_selection_pin.artifact_kind != "workspace_artifact_selection":
            raise ValueError("invalid expected selection kind")
        if self.preparation_pin is not None and self.preparation_pin.artifact_kind != "workspace_artifact_preparation":
            raise ValueError("invalid preparation kind")
        return self


class ClearLifecycleArtifactBindingCommandV1(_LifecycleArtifactCommand):
    schema_version: Literal["clear_lifecycle_artifact_binding_command.v1"]


class WorkbenchProjectionCursorHeadV3(_Contract):
    """Current material workbench cursor. Development databases are reseeded on schema change."""

    schema_version: Literal["workbench_projection_cursor_head.v3"]
    cursor_head_ref: NonEmptyStr
    cursor_head_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    session_id: NonEmptyStr
    actor_ref: NonEmptyStr
    cursor_ref: Annotated[str, Field(pattern=r"^workbench-cursor:[0-9a-f]{64}$")]
    revision: Annotated[int, Field(strict=True, ge=0)]
    predecessor_pin: ArtifactPinV1 | None
    selection_pin: ArtifactPinV1
    design_binding_set_pin: ArtifactPinV1
    runtime_binding_set_pin: ArtifactPinV1
    evidence_focus_pin: ArtifactPinV1 | None
    design_snapshot_pin: ArtifactPinV1 | None
    projection_basis_hash: Sha256
    recorded_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_cursor_identity(self):
        if self.recorded_at.utcoffset() != timedelta(0):
            raise ValueError("recorded_at must be timezone-aware UTC")
        if (self.revision == 0) != (self.predecessor_pin is None):
            raise ValueError("only genesis has no predecessor")
        if self.predecessor_pin is not None and self.predecessor_pin.artifact_kind != "workbench_projection_cursor_head":
            raise ValueError("invalid cursor predecessor kind")
        if self.selection_pin.artifact_kind != "workspace_artifact_selection":
            raise ValueError("invalid cursor selection kind")
        if any(pin.artifact_kind != "lifecycle_artifact_binding_set" for pin in (self.design_binding_set_pin, self.runtime_binding_set_pin)):
            raise ValueError("invalid cursor binding-set kind")
        if self.evidence_focus_pin is not None and self.evidence_focus_pin.artifact_kind != "workbench_evidence_focus":
            raise ValueError("invalid cursor evidence focus kind")
        if self.design_snapshot_pin is not None and self.design_snapshot_pin.artifact_kind != "exploratory_graph_snapshot":
            raise ValueError("invalid cursor design snapshot kind")
        material = self.model_dump(mode="json", include={
            "organization_id", "workspace_id", "session_id", "actor_ref",
            "cursor_ref", "revision", "selection_pin", "design_binding_set_pin",
            "runtime_binding_set_pin", "evidence_focus_pin", "design_snapshot_pin",
        })
        if self.projection_basis_hash != f"sha256:{content_hash(material)}":
            raise ValueError("cursor material projection basis mismatch")
        digest = content_hash(self.model_dump(mode="json", exclude={"cursor_head_ref", "cursor_head_hash"}))
        if (self.cursor_head_ref, self.cursor_head_hash) != (f"workbench-projection-cursor-head:{digest}", f"sha256:{digest}"):
            raise ValueError("cursor head identity mismatch")
        return self


class WorkspaceArtifactDescriptorV1(_Contract):
    schema_version: Literal["workspace_artifact_descriptor.v1"]
    artifact_ref: NonEmptyStr
    artifact_hash: Sha256
    content_hash: Sha256
    original_name: NonEmptyStr
    media_type: NonEmptyStr
    byte_size: Annotated[int, Field(strict=True, ge=0)]
    created_by: NonEmptyStr
    created_at: V2UtcDateTime

    @model_validator(mode="after")
    def created_at_is_utc(self):
        if self.created_at.utcoffset() != timedelta(0):
            raise ValueError("created_at must be timezone-aware UTC")
        return self


class WorkspaceArtifactV1(WorkspaceArtifactDescriptorV1):
    """Internal identity; never serialize directly to HTTP or Chat."""

    schema_version: Literal["workspace_artifact.v1"]
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    byte_owner_ref: NonEmptyStr

    @model_validator(mode="after")
    def identity_binds_immutable_payload(self):
        if "\\" in self.byte_owner_ref or ":" in self.byte_owner_ref or "\x00" in self.byte_owner_ref or any(part in ("", ".", "..") for part in self.byte_owner_ref.split("/")):
            raise ValueError("byte_owner_ref must be a normalized relative reference")
        basis = self.model_dump(mode="json", exclude={"artifact_ref", "artifact_hash"})
        digest = content_hash(basis)
        if self.artifact_ref != f"workspace-artifact:{digest}" or self.artifact_hash != f"sha256:{digest}":
            raise ValueError("artifact identity must bind its immutable scoped payload")
        return self


class WorkspaceArtifactPreparationV1(_Contract):
    schema_version: Literal["workspace_artifact_preparation.v1"]
    preparation_ref: NonEmptyStr
    preparation_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    artifact_pin: ArtifactPinV1
    purpose: Literal["DESIGN_SOURCE", "RUNTIME_TARGET"]
    outcome: Literal["READY", "FAILED", "UNSUPPORTED"]
    source_identity_pin: ArtifactPinV1 | None
    canonical_source_ir_pin: ArtifactPinV1 | None
    derived_artifact_pins: tuple[ArtifactPinV1, ...]
    predecessor_pin: ArtifactPinV1 | None
    revision: Annotated[int, Field(strict=True, ge=0)]
    failure_code: NonEmptyStr | None
    prepared_by: NonEmptyStr
    prepared_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_preparation(self):
        if self.prepared_at.utcoffset() != timedelta(0):
            raise ValueError("prepared_at must be timezone-aware UTC")
        if self.artifact_pin.artifact_kind != "workspace_artifact":
            raise ValueError("preparation must pin workspace_artifact")
        if (self.revision == 0) != (self.predecessor_pin is None):
            raise ValueError("only genesis has no predecessor")
        if self.predecessor_pin is not None and self.predecessor_pin.artifact_kind != "workspace_artifact_preparation":
            raise ValueError("invalid preparation predecessor")
        if self.outcome == "READY":
            if self.failure_code is not None:
                raise ValueError("READY has no failure code")
            if self.purpose == "DESIGN_SOURCE" and (self.source_identity_pin is None or self.canonical_source_ir_pin is None):
                raise ValueError("READY DESIGN_SOURCE requires source and IR pins")
        elif self.failure_code is None or self.source_identity_pin is not None or self.canonical_source_ir_pin is not None or self.derived_artifact_pins:
            raise ValueError("failed preparation requires failure code and no authoritative pins")
        keys = [(pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) for pin in self.derived_artifact_pins]
        if keys != sorted(set(keys)):
            raise ValueError("derived pins must be sorted unique")
        digest = content_hash(self.model_dump(mode="json", exclude={"preparation_ref", "preparation_hash"}))
        if (self.preparation_ref, self.preparation_hash) != (f"workspace-artifact-preparation:{digest}", f"sha256:{digest}"):
            raise ValueError("preparation identity mismatch")
        return self


class WorkspaceArtifactPreparationAdapterResultV1(_Contract):
    schema_version: Literal["workspace_artifact_preparation_adapter_result.v1"]
    adapter_result_ref: NonEmptyStr
    adapter_result_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    artifact_pin: ArtifactPinV1
    purpose: Literal["DESIGN_SOURCE", "RUNTIME_TARGET"]
    correlation_ref: NonEmptyStr
    outcome: Literal["READY", "FAILED", "UNSUPPORTED"]
    source_identity_pin: ArtifactPinV1 | None
    canonical_source_ir_pin: ArtifactPinV1 | None
    derived_artifact_pins: tuple[ArtifactPinV1, ...]
    failure_code: NonEmptyStr | None
    completed_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_adapter_result(self):
        if self.completed_at.utcoffset() != timedelta(0):
            raise ValueError("completed_at must be timezone-aware UTC")
        if self.artifact_pin.artifact_kind != "workspace_artifact":
            raise ValueError("adapter result must pin workspace_artifact")
        if self.outcome == "READY":
            if self.failure_code is not None:
                raise ValueError("READY has no failure code")
            if self.purpose == "DESIGN_SOURCE" and (self.source_identity_pin is None or self.canonical_source_ir_pin is None):
                raise ValueError("READY DESIGN_SOURCE requires source and IR pins")
        elif self.failure_code is None or self.source_identity_pin is not None or self.canonical_source_ir_pin is not None or self.derived_artifact_pins:
            raise ValueError("failed preparation requires failure code and no authoritative pins")
        keys = [(pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) for pin in self.derived_artifact_pins]
        if keys != sorted(set(keys)):
            raise ValueError("derived pins must be sorted unique")
        digest = content_hash(self.model_dump(mode="json", exclude={"adapter_result_ref", "adapter_result_hash"}))
        if (self.adapter_result_ref, self.adapter_result_hash) != (f"workspace-artifact-preparation-adapter-result:{digest}", f"sha256:{digest}"):
            raise ValueError("adapter result identity mismatch")
        return self


class WorkspaceArtifactPreparationClaimV1(_Contract):
    schema_version: Literal["workspace_artifact_preparation_claim.v1"]
    claim_ref: NonEmptyStr
    claim_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    artifact_pin: ArtifactPinV1
    purpose: Literal["DESIGN_SOURCE", "RUNTIME_TARGET"]
    expected_preparation_pin: ArtifactPinV1 | None
    expected_preparation_revision: Annotated[int, Field(strict=True, ge=0)] | None
    correlation_ref: NonEmptyStr
    operation_key: NonEmptyStr
    request_id: NonEmptyStr
    request_hash: Sha256
    actor_ref: NonEmptyStr
    state: Literal["PENDING"]
    claimed_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_claim(self):
        if self.claimed_at.utcoffset() != timedelta(0):
            raise ValueError("claimed_at must be timezone-aware UTC")
        if (self.expected_preparation_pin is None) != (self.expected_preparation_revision is None):
            raise ValueError("expected preparation pin and revision must be paired")
        if self.artifact_pin.artifact_kind != "workspace_artifact":
            raise ValueError("claim must pin workspace_artifact")
        if self.expected_preparation_pin is not None and self.expected_preparation_pin.artifact_kind != "workspace_artifact_preparation":
            raise ValueError("invalid expected preparation kind")
        digest = content_hash(self.model_dump(mode="json", exclude={"claim_ref", "claim_hash"}))
        if (self.claim_ref, self.claim_hash) != (f"workspace-artifact-preparation-claim:{digest}", f"sha256:{digest}"):
            raise ValueError("claim identity mismatch")
        return self


class WorkspaceArtifactPreparationCatalogV1(_Contract):
    schema_version: Literal["workspace_artifact_preparation_catalog.v1"]
    catalog_ref: NonEmptyStr
    catalog_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    revision: Annotated[int, Field(strict=True, ge=0)]
    preparation_pins: tuple[ArtifactPinV1, ...]
    predecessor_pin: ArtifactPinV1 | None
    recorded_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_catalog(self):
        if self.recorded_at.utcoffset() != timedelta(0):
            raise ValueError("recorded_at must be timezone-aware UTC")
        if (self.revision == 0) != (self.predecessor_pin is None):
            raise ValueError("only genesis has no predecessor")
        if self.predecessor_pin is not None and self.predecessor_pin.artifact_kind != "workspace_artifact_preparation_catalog":
            raise ValueError("invalid catalog predecessor")
        keys = [(pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) for pin in self.preparation_pins]
        if any(pin.artifact_kind != "workspace_artifact_preparation" for pin in self.preparation_pins) or keys != sorted(set(keys)):
            raise ValueError("catalog pins must be sorted unique preparations")
        digest = content_hash(self.model_dump(mode="json", exclude={"catalog_ref", "catalog_hash"}))
        if (self.catalog_ref, self.catalog_hash) != (f"workspace-artifact-preparation-catalog:{digest}", f"sha256:{digest}"):
            raise ValueError("catalog identity mismatch")
        return self


class WorkspaceArtifactListEntryV1(_Contract):
    schema_version: Literal["workspace_artifact_list_entry.v1"]
    artifact: WorkspaceArtifactDescriptorV1
    access_state: Literal["AVAILABLE", "INACCESSIBLE", "RETIRED"]
    active_preparation_pins: tuple[ArtifactPinV1, ...]


class WorkspaceArtifactCollectionV1(_Contract):
    schema_version: Literal["workspace_artifact_collection.v1"]
    collection_ref: NonEmptyStr
    collection_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    revision: Annotated[int, Field(strict=True, ge=0)]
    artifact_pins: tuple[ArtifactPinV1, ...]
    recorded_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_collection(self):
        if self.recorded_at.utcoffset() != timedelta(0):
            raise ValueError("recorded_at must be timezone-aware UTC")
        keys = [(pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) for pin in self.artifact_pins]
        if any(pin.artifact_kind != "workspace_artifact" for pin in self.artifact_pins) or keys != sorted(set(keys)):
            raise ValueError("collection pins must be sorted unique workspace artifacts")
        digest = content_hash(self.model_dump(mode="json", exclude={"collection_ref", "collection_hash"}))
        if self.collection_ref != f"workspace-artifacts:{digest}" or self.collection_hash != f"sha256:{digest}":
            raise ValueError("collection identity must bind its immutable scoped payload")
        return self


class WorkspaceArtifactErrorV1(_Contract):
    schema_version: Literal["workspace_artifact_error.v1"]
    code: Literal[
        "AUTHORITY_CONTEXT_REQUIRED", "WORKSPACE_REQUIRED", "FORBIDDEN",
        "INVALID_MEDIA_TYPE", "INVALID_BYTE_LENGTH", "ARTIFACT_HASH_MISMATCH",
        "ARTIFACT_METADATA_CONFLICT", "ARTIFACT_NOT_FOUND", "ARTIFACT_INACCESSIBLE",
        "ARTIFACT_RETIRED", "ARTIFACT_SCOPE_MISMATCH", "ARTIFACT_PIN_MISMATCH",
        "STALE_COLLECTION_HEAD", "STALE_PREPARATION_HEAD", "STALE_SELECTION_HEAD",
        "STALE_BINDING_SET_HEAD", "STALE_MATERIAL_CONTEXT", "REQUEST_CONFLICT",
        "INVALID_LIFECYCLE_ROLE", "SOURCE_NOT_PREPARED", "SOURCE_PREPARATION_FAILED",
        "UNSUPPORTED_SOURCE_PREPARATION", "TARGET_UNSUPPORTED", "BINDING_NOT_FOUND",
        "PREPARATION_RECONCILIATION_FAILED", "PERSISTENCE_CONFLICT", "TEMPORARY_STORAGE_FAILURE",
    ]
    retryable: Annotated[bool, Field(strict=True)]
    current_collection_pin: ArtifactPinV1 | None
    current_collection_revision: Annotated[int, Field(strict=True, ge=0)] | None
    current_preparation_pin: ArtifactPinV1 | None
    current_preparation_revision: Annotated[int, Field(strict=True, ge=0)] | None
    current_selection_pin: ArtifactPinV1 | None
    current_selection_revision: Annotated[int, Field(strict=True, ge=0)] | None
    current_binding_set_pin: ArtifactPinV1 | None
    current_binding_set_revision: Annotated[int, Field(strict=True, ge=0)] | None
    current_cursor_pin: ArtifactPinV1 | None
    current_material_projection_hash: Sha256 | None
    current_material_context_revision: Annotated[int, Field(strict=True, ge=0)] | None
    detail: NonEmptyStr

    @model_validator(mode="after")
    def verify_current_heads(self):
        for name, kind in (
            ("collection", "workspace_artifact_collection"),
            ("preparation", "workspace_artifact_preparation"),
            ("selection", "workspace_artifact_selection"),
            ("binding_set", "lifecycle_artifact_binding_set"),
        ):
            pin = getattr(self, f"current_{name}_pin")
            revision = getattr(self, f"current_{name}_revision")
            if (pin is None) != (revision is None) or (pin is not None and pin.artifact_kind != kind):
                raise ValueError("current head requires its exact pin and revision")
        cursor = (self.current_cursor_pin, self.current_material_projection_hash, self.current_material_context_revision)
        if any(v is None for v in cursor) and not all(v is None for v in cursor):
            raise ValueError("current cursor requires pin, basis hash and revision")
        if self.current_cursor_pin is not None and self.current_cursor_pin.artifact_kind != "workbench_projection_cursor_head":
            raise ValueError("invalid current cursor pin")
        if self.retryable and self.code != "TEMPORARY_STORAGE_FAILURE" and not self.code.startswith("STALE_"):
            raise ValueError("this failure is not retryable")
        return self


class LifecycleArtifactBindingEntryV1(_Contract):
    schema_version: Literal["lifecycle_artifact_binding_entry.v1"]
    binding_ref: NonEmptyStr
    binding_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    lifecycle: Literal["DESIGN", "RUNTIME"]
    role: Literal["SOURCE", "TARGET"]
    artifact_pin: ArtifactPinV1
    source_identity_pin: ArtifactPinV1 | None
    canonical_source_ir_pin: ArtifactPinV1 | None
    preparation_pin: ArtifactPinV1 | None
    entry_revision: Annotated[int, Field(strict=True, ge=0)]
    predecessor_entry_pin: ArtifactPinV1 | None
    state: Literal["ACTIVE", "STALE", "INACCESSIBLE"]
    bound_by: NonEmptyStr
    bound_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_binding(self):
        if (self.lifecycle, self.role) not in (("DESIGN", "SOURCE"), ("RUNTIME", "TARGET")):
            raise ValueError("invalid lifecycle/role")
        if self.bound_at.utcoffset() != timedelta(0):
            raise ValueError("bound_at must be timezone-aware UTC")
        if self.artifact_pin.artifact_kind != "workspace_artifact":
            raise ValueError("binding must pin workspace_artifact")
        if self.role == "SOURCE" and any(pin is None for pin in (self.source_identity_pin, self.canonical_source_ir_pin, self.preparation_pin)):
            raise ValueError("SOURCE requires preparation and source/IR pins")
        digest = content_hash(self.model_dump(mode="json", exclude={"binding_ref", "binding_hash"}))
        if (self.binding_ref, self.binding_hash) != (f"workspace-artifact-binding:{digest}", f"sha256:{digest}"):
            raise ValueError("binding identity mismatch")
        return self


class LifecycleArtifactBindingSetV1(_Contract):
    schema_version: Literal["lifecycle_artifact_binding_set.v1"]
    binding_set_ref: NonEmptyStr
    binding_set_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    lifecycle: Literal["DESIGN", "RUNTIME"]
    role: Literal["SOURCE", "TARGET"]
    entries: tuple[LifecycleArtifactBindingEntryV1, ...]
    revision: Annotated[int, Field(strict=True, ge=0)]
    predecessor_pin: ArtifactPinV1 | None
    recorded_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_binding_set(self):
        if (self.lifecycle, self.role) not in (("DESIGN", "SOURCE"), ("RUNTIME", "TARGET")):
            raise ValueError("invalid lifecycle/role")
        if self.recorded_at.utcoffset() != timedelta(0):
            raise ValueError("recorded_at must be timezone-aware UTC")
        if self.role == "TARGET" and len(self.entries) > 1:
            raise ValueError("RUNTIME TARGET is singleton")
        if (self.revision == 0) != (self.predecessor_pin is None):
            raise ValueError("only genesis has no predecessor")
        keys = []
        for entry in self.entries:
            if (entry.organization_id, entry.workspace_id, entry.lifecycle, entry.role) != (self.organization_id, self.workspace_id, self.lifecycle, self.role):
                raise ValueError("binding entry scope mismatch")
            keys.append((entry.artifact_pin.artifact_ref, entry.artifact_pin.artifact_hash))
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate binding artifacts")
        digest = content_hash(self.model_dump(mode="json", exclude={"binding_set_ref", "binding_set_hash"}))
        if (self.binding_set_ref, self.binding_set_hash) != (f"workspace-artifact-binding-set:{digest}", f"sha256:{digest}"):
            raise ValueError("binding set identity mismatch")
        return self


class WorkspaceArtifactSelectionV1(_Contract):
    schema_version: Literal["workspace_artifact_selection.v1"]
    selection_ref: NonEmptyStr
    selection_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    artifact_pin: ArtifactPinV1 | None
    revision: Annotated[int, Field(strict=True, ge=0)]
    predecessor_pin: ArtifactPinV1 | None
    selected_by: NonEmptyStr
    selected_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_selection(self):
        if self.selected_at.utcoffset() != timedelta(0):
            raise ValueError("selected_at must be timezone-aware UTC")
        if (self.revision == 0) != (self.predecessor_pin is None):
            raise ValueError("only genesis selection has no predecessor")
        if self.artifact_pin is not None and self.artifact_pin.artifact_kind != "workspace_artifact":
            raise ValueError("selection must pin workspace_artifact")
        if self.predecessor_pin is not None and self.predecessor_pin.artifact_kind != "workspace_artifact_selection":
            raise ValueError("predecessor must pin workspace_artifact_selection")
        digest = content_hash(self.model_dump(mode="json", exclude={"selection_ref", "selection_hash"}))
        if self.selection_ref != f"workspace-artifact-selection:{digest}" or self.selection_hash != f"sha256:{digest}":
            raise ValueError("selection identity must bind its immutable scoped payload")
        return self


class WorkspaceArtifactOperationReceiptV1(_Contract):
    schema_version: Literal["workspace_artifact_operation_receipt.v1"]
    receipt_ref: NonEmptyStr
    receipt_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    operation: Literal["UPLOAD_ARTIFACT", "SELECT_ARTIFACT", "PREPARE_ARTIFACT", "BIND_SOURCE", "BIND_TARGET", "CLEAR_SOURCE", "CLEAR_TARGET"]
    artifact_pin: ArtifactPinV1
    collection_pin: ArtifactPinV1
    collection_revision: Annotated[int, Field(strict=True, ge=0)]
    preparation_catalog_pin: ArtifactPinV1 | None
    preparation_catalog_revision: Annotated[int, Field(strict=True, ge=0)] | None
    preparation_pin: ArtifactPinV1 | None
    selection_pin: ArtifactPinV1 | None
    selection_revision: Annotated[int, Field(strict=True, ge=0)] | None
    binding_set_pin: ArtifactPinV1 | None
    binding_set_revision: Annotated[int, Field(strict=True, ge=0)] | None
    material_context_revision: Annotated[int, Field(strict=True, ge=0)] | None
    cursor_pin: ArtifactPinV1 | None
    ingress: Literal["DIRECT", "CHAT"]
    request_id: NonEmptyStr
    request_hash: Sha256
    idempotency_key: NonEmptyStr
    actor_ref: NonEmptyStr
    outcome: Literal["CREATED", "REUSED", "SELECTED", "PREPARED", "PREPARATION_FAILED", "BOUND", "CLEARED"]
    recorded_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_receipt(self):
        if self.recorded_at.utcoffset() != timedelta(0):
            raise ValueError("recorded_at must be timezone-aware UTC")
        if self.request_id != self.idempotency_key:
            raise ValueError("idempotency_key must equal request_id")
        for pin, kind in (
            (self.artifact_pin, "workspace_artifact"),
            (self.collection_pin, "workspace_artifact_collection"),
            (self.preparation_catalog_pin, "workspace_artifact_preparation_catalog"),
            (self.preparation_pin, "workspace_artifact_preparation"),
            (self.selection_pin, "workspace_artifact_selection"),
            (self.binding_set_pin, "lifecycle_artifact_binding_set"),
            (self.cursor_pin, "workbench_projection_cursor_head"),
        ):
            if pin is not None and pin.artifact_kind != kind:
                raise ValueError(f"invalid receipt pin kind: {kind}")
        for pin, revision in (
            (self.preparation_catalog_pin, self.preparation_catalog_revision),
            (self.selection_pin, self.selection_revision),
            (self.binding_set_pin, self.binding_set_revision),
            (self.cursor_pin, self.material_context_revision),
        ):
            if (pin is None) != (revision is None):
                raise ValueError("receipt head pin and revision must be paired")
        outcomes = {
            "UPLOAD_ARTIFACT": ("CREATED", "REUSED"),
            "PREPARE_ARTIFACT": ("PREPARED", "PREPARATION_FAILED"),
            "SELECT_ARTIFACT": ("SELECTED",),
            "BIND_SOURCE": ("BOUND",), "BIND_TARGET": ("BOUND",),
            "CLEAR_SOURCE": ("CLEARED",), "CLEAR_TARGET": ("CLEARED",),
        }
        if self.outcome not in outcomes[self.operation]:
            raise ValueError("receipt outcome does not match operation")
        if self.operation in ("UPLOAD_ARTIFACT", "PREPARE_ARTIFACT"):
            if any(pin is not None for pin in (self.selection_pin, self.binding_set_pin, self.cursor_pin)):
                raise ValueError("overlay operation cannot claim material state changes")
            if self.operation == "UPLOAD_ARTIFACT" and (self.preparation_pin is not None or self.preparation_catalog_pin is not None):
                raise ValueError("upload cannot claim preparation")
            if self.operation == "PREPARE_ARTIFACT" and (self.preparation_pin is None or self.preparation_catalog_pin is None):
                raise ValueError("prepare requires terminal preparation and catalog pins")
        else:
            if self.cursor_pin is None:
                raise ValueError("material operation requires resulting cursor")
            if self.operation == "SELECT_ARTIFACT":
                if self.selection_pin is None or self.binding_set_pin is not None:
                    raise ValueError("select requires selection without binding mutation")
            elif self.binding_set_pin is None:
                raise ValueError("bind/clear requires resulting binding set")
        digest = content_hash(self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"}))
        if (self.receipt_ref, self.receipt_hash) != (f"workspace-artifact-receipt:{digest}", f"sha256:{digest}"):
            raise ValueError("receipt identity mismatch")
        return self


class PrepareWorkspaceArtifactResultV1(_Contract):
    schema_version: Literal["prepare_workspace_artifact_result.v1"]
    result_ref: NonEmptyStr
    result_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    artifact_pin: ArtifactPinV1
    preparation_pin: ArtifactPinV1
    preparation_revision: Annotated[int, Field(strict=True, ge=0)]
    preparation_catalog_pin: ArtifactPinV1
    preparation_catalog_revision: Annotated[int, Field(strict=True, ge=0)]
    predecessor_pin: ArtifactPinV1 | None
    receipt_pin: ArtifactPinV1
    outcome: Literal["READY", "FAILED", "UNSUPPORTED"]
    recorded_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_result(self):
        if self.recorded_at.utcoffset() != timedelta(0):
            raise ValueError("recorded_at must be timezone-aware UTC")
        for pin, kind in (
            (self.artifact_pin, "workspace_artifact"),
            (self.preparation_pin, "workspace_artifact_preparation"),
            (self.preparation_catalog_pin, "workspace_artifact_preparation_catalog"),
            (self.receipt_pin, "workspace_artifact_operation_receipt"),
        ):
            if pin.artifact_kind != kind:
                raise ValueError(f"invalid preparation result pin kind: {kind}")
        digest = content_hash(self.model_dump(mode="json", exclude={"result_ref", "result_hash"}))
        if (self.result_ref, self.result_hash) != (f"workspace-artifact-prepare-result:{digest}", f"sha256:{digest}"):
            raise ValueError("preparation result identity mismatch")
        return self


class ArtifactWorkspaceProjectionV1(_Contract):
    schema_version: Literal["artifact_workspace_projection.v1"]
    projection_ref: NonEmptyStr
    projection_hash: Sha256
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    cursor_ref: NonEmptyStr
    material_projection_hash: Sha256
    collection_pin: ArtifactPinV1
    collection_revision: Annotated[int, Field(strict=True, ge=0)]
    preparation_catalog_pin: ArtifactPinV1
    preparation_catalog_revision: Annotated[int, Field(strict=True, ge=0)]
    artifacts: tuple[WorkspaceArtifactListEntryV1, ...]
    preparations: tuple[WorkspaceArtifactPreparationV1, ...]
    selection: WorkspaceArtifactSelectionV1
    design_sources: LifecycleArtifactBindingSetV1
    runtime_target: LifecycleArtifactBindingSetV1
    workspace_selection_head: WorkspaceArtifactSelectionV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    workspace_design_sources_head: LifecycleArtifactBindingSetV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    workspace_runtime_target_head: LifecycleArtifactBindingSetV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    material_context_revision: Annotated[int, Field(strict=True, ge=0)]
    projected_at: V2UtcDateTime

    @model_validator(mode="after")
    def verify_projection(self):
        if self.projected_at.utcoffset() != timedelta(0):
            raise ValueError("projected_at must be timezone-aware UTC")
        if self.collection_pin.artifact_kind != "workspace_artifact_collection":
            raise ValueError("invalid collection kind")
        if self.preparation_catalog_pin.artifact_kind != "workspace_artifact_preparation_catalog":
            raise ValueError("invalid preparation catalog kind")
        for snapshot in (self.selection, self.design_sources, self.runtime_target, self.workspace_selection_head, self.workspace_design_sources_head, self.workspace_runtime_target_head, *self.preparations):
            if snapshot is None:
                continue
            if (snapshot.organization_id, snapshot.workspace_id) != (self.organization_id, self.workspace_id):
                raise ValueError("projection snapshot scope mismatch")
        if (self.design_sources.lifecycle, self.design_sources.role) != ("DESIGN", "SOURCE") or (self.runtime_target.lifecycle, self.runtime_target.role) != ("RUNTIME", "TARGET"):
            raise ValueError("projection lifecycle role mismatch")
        for head, role in ((self.workspace_design_sources_head, ("DESIGN", "SOURCE")), (self.workspace_runtime_target_head, ("RUNTIME", "TARGET"))):
            if head is not None and (head.lifecycle, head.role) != role:
                raise ValueError("workspace write head lifecycle mismatch")
        artifact_keys = [(item.artifact.artifact_ref, item.artifact.artifact_hash) for item in self.artifacts]
        if artifact_keys != sorted(set(artifact_keys)):
            raise ValueError("projection artifacts must follow sorted unique collection order")
        preparation_keys = [
            (item.artifact_pin.artifact_ref, item.purpose, item.revision, item.preparation_ref, item.preparation_hash)
            for item in self.preparations
        ]
        if preparation_keys != sorted(set(preparation_keys)):
            raise ValueError("projection preparations must follow sorted unique order")
        digest = content_hash(self.model_dump(mode="json", exclude={"projection_ref", "projection_hash"}))
        if (self.projection_ref, self.projection_hash) != (f"workspace-artifact-projection:{digest}", f"sha256:{digest}"):
            raise ValueError("workspace projection identity mismatch")
        return self


def serialize_workspace_artifact_descriptor(
    value: WorkspaceArtifactV1,
) -> WorkspaceArtifactDescriptorV1:
    return WorkspaceArtifactDescriptorV1(
        schema_version="workspace_artifact_descriptor.v1",
        artifact_ref=value.artifact_ref,
        artifact_hash=value.artifact_hash,
        content_hash=value.content_hash,
        original_name=value.original_name,
        media_type=value.media_type,
        byte_size=value.byte_size,
        created_by=value.created_by,
        created_at=value.created_at,
    )


WorkbenchArtifactKindV2 = Literal[
    "runtime_temporal_source_archive",
    "runtime_temporal_source_graph",
    "context_graph",
    "evidence",
    "runtime_provider_request",
    "runtime_evaluation_run",
    "runtime_evaluation_state",
    "runtime_provider_result",
    "runtime_inspection_report",
]


class WorkbenchArtifactReadRequestV2(_Contract):
    session_ref: NonEmptyStr
    artifact_kind: WorkbenchArtifactKindV2
    artifact_ref: NonEmptyStr
    artifact_hash: Sha256


WorkbenchArtifactPayloadV2 = Annotated[
    RuntimeTemporalSourceArchiveV1
    | RuntimeTemporalSourceGraphV1
    | ContextGraph
    | Evidence
    | RuntimeTemporalProviderRequestV1
    | RuntimeEvaluationRunV1
    | RuntimeEvaluationStateV1
    | RuntimeProviderResultV1
    | RuntimeInspectionReportV1,
    Field(discriminator="schema_version"),
]


class WorkbenchArtifactProjectionV2(_Contract):
    schema_version: Literal["workbench_artifact_projection.v2"]
    artifact: ArtifactPinV1
    payload: WorkbenchArtifactPayloadV2

    @model_validator(mode="after")
    def artifact_matches_native_payload(self) -> "WorkbenchArtifactProjectionV2":
        payload = self.payload
        if isinstance(payload, (ContextGraph, Evidence)):
            expected = payload.pin()
        elif isinstance(payload, RuntimeInspectionReportV1):
            expected = payload.report_pin
        else:
            expected = payload.artifact_pin()
        if self.artifact != expected:
            raise ValueError("artifact must exactly identify payload")
        return self


class WorkbenchRepresentationV2(_Contract):
    view: WorkbenchView
    lens: str | None = None


class WorkbenchArtifactV2(_Contract):
    build_ref: str | None = None
    source_ref: str | None = None
    candidate_graph_ref: str | None = None
    candidate_graph_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    release_ref: str | None = None
    runtime_evaluation_ref: str | None = None

    @model_validator(mode="after")
    def candidate_identity_is_paired(self) -> "WorkbenchArtifactV2":
        if (self.candidate_graph_ref is None) != (self.candidate_graph_hash is None):
            raise ValueError(
                "candidate_graph_ref and candidate_graph_hash must be paired"
            )
        return self


class WorkbenchComparisonV2(_Contract):
    baseline_candidate_ref: str | None = None
    baseline_candidate_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    candidate_ref: str | None = None
    candidate_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    diff_ref: str | None = None

    @model_validator(mode="after")
    def identities_are_paired(self) -> "WorkbenchComparisonV2":
        for ref, digest in (
            (self.baseline_candidate_ref, self.baseline_candidate_hash),
            (self.candidate_ref, self.candidate_hash),
        ):
            if (ref is None) != (digest is None):
                raise ValueError("comparison candidate ref and hash must be paired")
        return self


class WorkbenchSelectionV2(_Contract):
    object_ref: str | None = None
    node_ref: str | None = None
    edge_ref: str | None = None
    evidence_refs: tuple[str, ...] = ()
    finding_ref: str | None = None
    quality_or_validation_ref: str | None = None


class KnowledgeWorkbenchContextV5(_Contract):
    schema_version: Literal["knowledge_workbench_context.v5"] = (
        "knowledge_workbench_context.v5"
    )
    representation: WorkbenchRepresentationV2
    artifact: WorkbenchArtifactV2
    comparison: WorkbenchComparisonV2
    selection: WorkbenchSelectionV2
    material_context_revision: int = Field(ge=0)


class UnifiedWorkbenchContextV2(_Contract):
    schema_version: Literal["unified_workbench_context.v2"] = (
        "unified_workbench_context.v2"
    )
    knowledge: KnowledgeWorkbenchContextV5
    representation: WorkbenchRepresentationV2
    artifact: WorkbenchArtifactV2
    comparison: WorkbenchComparisonV2
    selection: WorkbenchSelectionV2
    material_context_revision: int = Field(ge=0)

    @model_validator(mode="after")
    def one_shared_projection(self) -> "UnifiedWorkbenchContextV2":
        for field in (
            "representation",
            "artifact",
            "comparison",
            "selection",
            "material_context_revision",
        ):
            if getattr(self, field) != getattr(self.knowledge, field):
                raise ValueError(f"{field} must match knowledge projection")
        return self


class WorkbenchEvidenceSelectionIntentV1(_Contract):
    schema_version: Literal["workbench_evidence_selection_intent.v1"] = (
        "workbench_evidence_selection_intent.v1"
    )
    object_kind: Literal["node", "edge"]
    object_ref: NonEmptyStr
    evidence_ref: NonEmptyStr | None = None
    occurrence_ref: NonEmptyStr | None = None

    @model_validator(mode="after")
    def verify_focus_pair(self):
        if self.occurrence_ref is not None and self.evidence_ref is None:
            raise ValueError("occurrence focus requires an evidence ref")
        return self


class WorkbenchSelectionRequestV1(_Contract):
    schema_version: Literal["workbench_selection_request.v1"] = "workbench_selection_request.v1"
    cursor_ref: AsciiWorkbenchId
    expected_material_context_revision: _V6Revision
    expected_projection_hash: Sha256
    design_snapshot_pin: ArtifactPinV1 | None
    runtime_context_pin: ArtifactPinV1 | None = None
    runtime_run_pin: ArtifactPinV1 | None = None
    runtime_report_pin: ArtifactPinV1 | None = None
    learning_candidate_pin: ArtifactPinV1 | None = None
    evidence_focus: WorkbenchEvidenceSelectionIntentV1 | None = None
    lifecycle: Literal["DESIGN", "RUNTIME"]
    representation: WorkbenchRepresentationV2

    @model_validator(mode="after")
    def verify_selection_pins(self):
        if self.design_snapshot_pin is not None and self.design_snapshot_pin.artifact_kind != "exploratory_graph_snapshot":
            raise ValueError("design snapshot must pin exploratory_graph_snapshot")
        if self.runtime_context_pin is not None and self.runtime_context_pin.artifact_kind not in {"context_graph", "runtime_working_context"}:
            raise ValueError("runtime context must pin context_graph during kernel cutover")
        if self.runtime_run_pin is not None and self.runtime_run_pin.artifact_kind != "runtime_evaluation_run":
            raise ValueError("runtime run must pin runtime_evaluation_run")
        if self.runtime_report_pin is not None and self.runtime_report_pin.artifact_kind != "runtime_inspection_report":
            raise ValueError("runtime report must pin runtime_inspection_report")
        if self.learning_candidate_pin is not None and self.learning_candidate_pin.artifact_kind != "change_candidate":
            raise ValueError("learning candidate must pin change_candidate")
        if self.lifecycle == "DESIGN" and any(pin is not None for pin in (self.runtime_context_pin, self.runtime_run_pin, self.runtime_report_pin)):
            raise ValueError("DESIGN selection cannot carry runtime operands")
        if self.evidence_focus is not None and (
            self.lifecycle != "DESIGN" or self.design_snapshot_pin is None
        ):
            raise ValueError("design evidence focus requires a DESIGN snapshot")
        if self.lifecycle == "RUNTIME" and self.runtime_run_pin is not None and self.runtime_context_pin is None:
            raise ValueError("runtime run requires its exact runtime context pin")
        if self.runtime_report_pin is not None and self.runtime_run_pin is None:
            raise ValueError("runtime report requires its exact runtime run pin")
        return self


class WorkbenchRuntimeStartV1(_Contract):
    source_graph_pin: ArtifactPinV1
    ontology_pin: ArtifactPinV1
    evidence_pins: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    target: GovernedTargetPinV1
    selected_entity_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    preparation_pin: ArtifactPinV1

    @model_validator(mode="after")
    def verify_start_operands(self):
        if self.source_graph_pin.artifact_kind != "runtime_temporal_source_graph":
            raise ValueError("runtime start requires runtime_temporal_source_graph")
        if self.ontology_pin.artifact_kind != "ontology":
            raise ValueError("runtime start requires kernel ontology")
        if any(pin.artifact_kind != "evidence" for pin in self.evidence_pins):
            raise ValueError("runtime start requires kernel evidence")
        evidence_keys = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
            for pin in self.evidence_pins
        }
        if len(evidence_keys) != len(self.evidence_pins):
            raise ValueError("runtime start evidence pins must be unique")
        if self.preparation_pin.artifact_kind != "workspace_artifact_preparation":
            raise ValueError("runtime start requires workspace_artifact_preparation")
        if self.target.target_type != "graph" or self.target.artifact != self.source_graph_pin:
            raise ValueError("runtime start target must identify the exact source graph")
        if len(self.selected_entity_ids) != len(set(self.selected_entity_ids)):
            raise ValueError("runtime start entity ids must be unique")
        return self


class WorkbenchRuntimeV3(_Contract):
    start: WorkbenchRuntimeStartV1 | None = None
    context_ref: str | None = None
    context_version: str | None = None
    context_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    world_scope_ref: str | None = None
    world_scope_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    world_time: str | None = None
    target_ref: str | None = None
    target_version: str | None = None
    target_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    run_ref: str | None = None
    run_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    root_run_ref: str | None = None
    root_run_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    parent_run_ref: str | None = None
    parent_run_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    evaluation_ref: str | None = None
    evaluation_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    report_ref: str | None = None
    report_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    export_replay_inputs_pin: ArtifactPinV1 | None = None
    export_profile_pin: ArtifactPinV1 | None = None
    correction_evidence_pins: tuple[ArtifactPinV1, ...] = ()
    report_projector_pin: ArtifactPinV1 | None = None
    report_renderer_pin: ArtifactPinV1 | None = None
    report_template_pin: ArtifactPinV1 | None = None
    report_environment_pin: ArtifactPinV1 | None = None
    execution_epoch_ref: str | None = None
    semantic_release_ref: str | None = None
    semantic_release_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def verify_pairs(self):
        for ref, digest, label in (
            (self.context_ref, self.context_hash, "context"),
            (self.world_scope_ref, self.world_scope_hash, "world_scope"),
            (self.target_ref, self.target_hash, "target"),
            (self.run_ref, self.run_hash, "run"),
            (self.root_run_ref, self.root_run_hash, "root_run"),
            (self.parent_run_ref, self.parent_run_hash, "parent_run"),
            (self.evaluation_ref, self.evaluation_hash, "evaluation"),
            (self.report_ref, self.report_hash, "report"),
            (self.semantic_release_ref, self.semantic_release_hash, "semantic_release"),
        ):
            if (ref is None) != (digest is None):
                raise ValueError(f"{label} ref/hash must be paired")
        if self.context_ref is None and self.context_version is not None:
            raise ValueError("context_version requires context_ref")
        if self.target_ref is None and self.target_version is not None:
            raise ValueError("target_version requires target_ref")
        profile_pins = (
            self.report_projector_pin,
            self.report_renderer_pin,
            self.report_template_pin,
            self.report_environment_pin,
        )
        if any(pin is None for pin in profile_pins) and not all(pin is None for pin in profile_pins):
            raise ValueError("report profile pins must be all present or all absent")
        if all(pin is not None for pin in profile_pins):
            expected = (
                "runtime_report_projector",
                "runtime_report_renderer",
                "runtime_report_template",
                "runtime_report_environment",
            )
            if tuple(pin.artifact_kind for pin in profile_pins if pin is not None) != expected:
                raise ValueError("report profile pin kinds are invalid")
        export_pins = (self.export_replay_inputs_pin, self.export_profile_pin)
        if any(pin is None for pin in export_pins) and not all(pin is None for pin in export_pins):
            raise ValueError("export pins must be paired")
        if self.export_replay_inputs_pin is not None:
            if self.report_ref is None:
                raise ValueError("export pins require selected report")
            if self.export_replay_inputs_pin.artifact_kind != "runtime_report_replay_inputs" or self.export_profile_pin.artifact_kind != "pinned_render_profile":
                raise ValueError("export pin kinds are invalid")
        return self


class WorkbenchComparisonOperandV3(_Contract):
    ref: NonEmptyStr
    hash: Sha256


class WorkbenchComparisonV3(_Contract):
    subject: Literal["candidate", "world", "reasoning", "rerun"] | None = None
    operands: tuple[WorkbenchComparisonOperandV3, ...] = ()
    diff_ref: str | None = None

    @model_validator(mode="after")
    def verify_operands(self):
        if self.subject is None:
            if self.operands or self.diff_ref is not None:
                raise ValueError("comparison subject is required when comparison is populated")
        elif len(self.operands) != 2:
            raise ValueError("comparison requires exactly two ordered operands")
        return self


class WorkbenchSelectionV3(_Contract):
    object_ref: str | None = None
    node_ref: str | None = None
    edge_ref: str | None = None
    evidence_refs: tuple[str, ...] = ()
    finding_ref: str | None = None
    quality_or_validation_ref: str | None = None
    canonical_anchor_ref: str | None = None
    transition_ref: str | None = None
    proof_ref: str | None = None


class EvidenceSourceGroupV1(_Contract):
    source_identity_ref: NonEmptyStr
    source_hash: Sha256
    artifact_ref: NonEmptyStr
    document_name: NonEmptyStr
    document_kind: Literal["pdf", "docx", "canonical_source_ir", "unknown"]
    evidence_refs: tuple[NonEmptyStr, ...]


class EvidenceActionBindingV1(_Contract):
    binding_kind: Literal["focus_evidence"] = "focus_evidence"
    payload: EvidenceFocusPayloadV1


class EvidenceFocusProjectionV1(_Contract):
    schema_version: Literal["evidence_focus_projection.v1"] = "evidence_focus_projection.v1"
    status: Literal["READY", "BLOCKED"]
    object_pin: ArtifactPinV1
    members: tuple[WorkbenchEvidenceMemberV1, ...]
    source_groups: tuple[EvidenceSourceGroupV1, ...]
    selected_evidence_ref: NonEmptyStr | None = None
    selected_occurrence_ref: NonEmptyStr | None = None
    action_binding: EvidenceActionBindingV1 | None = None
    error_code: NonEmptyStr | None = None

    @model_validator(mode="after")
    def verify_projection(self):
        if (self.selected_evidence_ref is None) != (self.selected_occurrence_ref is None):
            raise ValueError("singular evidence focus requires evidence and occurrence together")
        member_refs = tuple(member.evidence_ref for member in self.members)
        grouped_refs = tuple(ref for group in self.source_groups for ref in group.evidence_refs)
        if grouped_refs != member_refs:
            raise ValueError("source groups must preserve complete evidence membership order")
        if self.action_binding is not None:
            payload = self.action_binding.payload
            if payload.object_pin != self.object_pin:
                raise ValueError("evidence action binding must preserve object pin")
            exact_pins = tuple(
                member.evidence_pin
                for member in self.members
                if member.evidence_pin is not None and member.location.status == "exact"
            )
            if payload.evidence_refs != exact_pins:
                raise ValueError("evidence action binding must preserve exact server membership")
            if self.selected_evidence_ref is None:
                if payload.selected_evidence_ref is not None or payload.canonical_anchor is not None:
                    raise ValueError("aggregate evidence binding cannot fabricate singular focus")
            else:
                if (
                    payload.selected_evidence_ref is None
                    or payload.selected_evidence_ref.artifact_ref != self.selected_evidence_ref
                    or payload.canonical_anchor is None
                    or payload.canonical_anchor.artifact_ref != self.selected_occurrence_ref
                ):
                    raise ValueError("evidence action binding must preserve singular focus")
        if self.status == "READY" and self.action_binding is None:
            raise ValueError("ready evidence projection requires an action binding")
        if self.status == "BLOCKED" and self.error_code is None:
            raise ValueError("blocked evidence projection requires an error code")
        return self


class WorkbenchLearningV1(_Contract):
    candidate_ref: NonEmptyStr
    candidate_hash: str = Field(pattern=SHA256_PATTERN)
    revision: Annotated[int, Field(strict=True, ge=0)]
    stage: Literal["PROPOSAL", "OBSERVATION", "ASSESSMENT"]
    target: Literal["ONTOLOGY", "CONTEXT_FORMATION", "REASONING", "WORKFLOW"]
    failure_class: Literal["CAPABILITY", "STEERING", "STRUCTURE", "ENVIRONMENT"]
    origin_context_ref: str | None = None
    origin_context_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    verdict: Literal["BETTER", "WORSE", "INDETERMINATE", "LOCAL_FIX_VERIFIED"] | None = None
    disposition: Literal[
        "REVIEW_CANDIDATE",
        "ACCEPTED_FOR_SUCCESSOR_INPUT",
        "REJECTED_FOR_USE",
        "NOT_YET_ADMISSIBLE",
    ] | None = None

    @model_validator(mode="after")
    def verify_shape(self):
        if (self.origin_context_ref is None) != (self.origin_context_hash is None):
            raise ValueError("origin context ref/hash must be paired")
        if self.stage == "ASSESSMENT" and (self.verdict is None or self.disposition is None):
            raise ValueError("assessment projection requires verdict/disposition")
        if self.stage != "ASSESSMENT" and (self.verdict is not None or self.disposition is not None):
            raise ValueError("pre-assessment learning projection cannot fabricate verdict/disposition")
        return self


class WorkbenchCapabilityV3(_Contract):
    operation: NonEmptyStr
    allowed: bool
    reason_code: str | None = None
    governing_receipt_pin: ArtifactPinV1 | None = None
    expected_material_context_revision: int = Field(ge=0)
    expected_projection_hash: Sha256 | None = None


class KnowledgeWorkbenchContextV6(_Contract):
    schema_version: Literal["knowledge_workbench_context.v6"] = "knowledge_workbench_context.v6"
    representation: WorkbenchRepresentationV2
    artifact: WorkbenchArtifactV2
    artifact_workspace: ArtifactWorkspaceProjectionV1 | None = None
    lifecycle: Literal["DESIGN", "RUNTIME"]
    runtime: WorkbenchRuntimeV3 | None = None
    learning: WorkbenchLearningV1 | None = None
    evidence: EvidenceFocusProjectionV1 | None = None
    comparison: WorkbenchComparisonV3
    selection: WorkbenchSelectionV3
    capabilities: tuple[WorkbenchCapabilityV3, ...] = ()
    cursor_ref: str | None = None
    projection_hash: Sha256 | None = None
    material_context_revision: int = Field(ge=0)


class UnifiedWorkbenchContextV3(_Contract):
    schema_version: Literal["unified_workbench_context.v3"] = "unified_workbench_context.v3"
    knowledge: KnowledgeWorkbenchContextV6
    representation: WorkbenchRepresentationV2
    artifact: WorkbenchArtifactV2
    artifact_workspace: ArtifactWorkspaceProjectionV1 | None = None
    lifecycle: Literal["DESIGN", "RUNTIME"]
    runtime: WorkbenchRuntimeV3 | None = None
    learning: WorkbenchLearningV1 | None = None
    evidence: EvidenceFocusProjectionV1 | None = None
    comparison: WorkbenchComparisonV3
    selection: WorkbenchSelectionV3
    capabilities: tuple[WorkbenchCapabilityV3, ...] = ()
    cursor_ref: str | None = None
    projection_hash: Sha256 | None = None
    material_context_revision: int = Field(ge=0)

    @model_validator(mode="after")
    def one_shared_projection(self):
        for field in (
            "representation", "artifact", "artifact_workspace", "lifecycle", "runtime", "learning", "evidence",
            "comparison", "selection", "capabilities", "cursor_ref",
            "projection_hash", "material_context_revision",
        ):
            if getattr(self, field) != getattr(self.knowledge, field):
                raise ValueError(f"{field} must match knowledge projection")
        if self.cursor_ref is None:
            if self.projection_hash is not None or self.capabilities or self.artifact_workspace is not None:
                raise ValueError("authority-free V3 projection cannot fabricate cursor/hash/capabilities")
        else:
            if self.artifact_workspace is None:
                raise ValueError("server V3 projection requires artifact workspace cursor basis")
            if (
                self.artifact_workspace.cursor_ref != self.cursor_ref
                or self.artifact_workspace.material_context_revision != self.material_context_revision
            ):
                raise ValueError("artifact workspace cursor must match V3 projection")
        return self


def serialize_unified_workbench_context_v3(value: UnifiedWorkbenchContextV3) -> dict[str, object]:
    def pin(value):
        return value.model_dump(mode="json") if value is not None else None

    def runtime(value):
        if value is None:
            return None
        start = None
        if value.start is not None:
            start = {
                "sourceGraphPin": pin(value.start.source_graph_pin),
                "ontologyPin": pin(value.start.ontology_pin),
                "evidencePins": [pin(item) for item in value.start.evidence_pins],
                "target": value.start.target.model_dump(mode="json"),
                "selectedEntityIds": list(value.start.selected_entity_ids),
                "preparationPin": pin(value.start.preparation_pin),
            }
        return {
            "start": start,
            "contextRef": value.context_ref, "contextVersion": value.context_version, "contextHash": value.context_hash,
            "worldScopeRef": value.world_scope_ref, "worldScopeHash": value.world_scope_hash, "worldTime": value.world_time,
            "targetRef": value.target_ref, "targetVersion": value.target_version, "targetHash": value.target_hash,
            "runRef": value.run_ref, "runHash": value.run_hash,
            "rootRunRef": value.root_run_ref, "rootRunHash": value.root_run_hash,
            "parentRunRef": value.parent_run_ref, "parentRunHash": value.parent_run_hash,
            "evaluationRef": value.evaluation_ref, "evaluationHash": value.evaluation_hash,
            "reportRef": value.report_ref, "reportHash": value.report_hash,
            "exportReplayInputsPin": pin(value.export_replay_inputs_pin),
            "exportProfilePin": pin(value.export_profile_pin),
            "correctionEvidencePins": [pin(item) for item in value.correction_evidence_pins],
            "reportProjectorPin": pin(value.report_projector_pin),
            "reportRendererPin": pin(value.report_renderer_pin),
            "reportTemplatePin": pin(value.report_template_pin),
            "reportEnvironmentPin": pin(value.report_environment_pin),
            "executionEpochRef": value.execution_epoch_ref,
            "semanticReleaseRef": value.semantic_release_ref, "semanticReleaseHash": value.semantic_release_hash,
        }

    def learning(value):
        if value is None:
            return None
        return {
            "candidateRef": value.candidate_ref,
            "candidateHash": value.candidate_hash,
            "revision": value.revision,
            "stage": value.stage,
            "target": value.target,
            "failureClass": value.failure_class,
            "originContextRef": value.origin_context_ref,
            "originContextHash": value.origin_context_hash,
            "verdict": value.verdict,
            "disposition": value.disposition,
        }

    def evidence(value):
        if value is None:
            return None
        return {
            "schemaVersion": value.schema_version,
            "status": value.status,
            "objectPin": pin(value.object_pin),
            "members": [
                {
                    "schemaVersion": member.schema_version,
                    "evidenceRef": member.evidence_ref,
                    "evidencePin": pin(member.evidence_pin),
                    "occurrenceRef": member.occurrence_ref,
                    "location": member.location.model_dump(
                        mode="json", by_alias=True
                    ),
                }
                for member in value.members
            ],
            "sourceGroups": [
                {
                    "sourceIdentityRef": group.source_identity_ref,
                    "sourceHash": group.source_hash,
                    "artifactRef": group.artifact_ref,
                    "documentName": group.document_name,
                    "documentKind": group.document_kind,
                    "evidenceRefs": list(group.evidence_refs),
                }
                for group in value.source_groups
            ],
            "selectedEvidenceRef": value.selected_evidence_ref,
            "selectedOccurrenceRef": value.selected_occurrence_ref,
            "actionBinding": (
                {
                    "bindingKind": value.action_binding.binding_kind,
                    "payload": value.action_binding.payload.model_dump(mode="json"),
                }
                if value.action_binding is not None
                else None
            ),
            "errorCode": value.error_code,
        }

    def shared(context):
        return {
            "representation": context.representation.model_dump(mode="json"),
            "artifact": {
                "buildRef": context.artifact.build_ref,
                "sourceRef": context.artifact.source_ref,
                "candidateGraphRef": context.artifact.candidate_graph_ref,
                "candidateGraphHash": context.artifact.candidate_graph_hash,
                "releaseRef": context.artifact.release_ref,
                "runtimeEvaluationRef": context.artifact.runtime_evaluation_ref,
            },
            "artifactWorkspace": context.artifact_workspace.model_dump(mode="json") if context.artifact_workspace else None,
            "lifecycle": context.lifecycle,
            "runtime": runtime(context.runtime),
            "learning": learning(context.learning),
            "evidence": evidence(context.evidence),
            "comparison": {
                "subject": context.comparison.subject,
                "operands": [item.model_dump(mode="json") for item in context.comparison.operands],
                "diffRef": context.comparison.diff_ref,
            },
            "selection": {
                "objectRef": context.selection.object_ref, "nodeRef": context.selection.node_ref,
                "edgeRef": context.selection.edge_ref, "evidenceRefs": list(context.selection.evidence_refs),
                "findingRef": context.selection.finding_ref,
                "qualityOrValidationRef": context.selection.quality_or_validation_ref,
                "canonicalAnchorRef": context.selection.canonical_anchor_ref,
                "transitionRef": context.selection.transition_ref, "proofRef": context.selection.proof_ref,
            },
            "capabilities": [{
                "operation": item.operation, "allowed": item.allowed, "reasonCode": item.reason_code,
                "governingReceiptPin": pin(item.governing_receipt_pin),
                "expectedMaterialContextRevision": item.expected_material_context_revision,
                "expectedProjectionHash": item.expected_projection_hash,
            } for item in context.capabilities],
            "cursorRef": context.cursor_ref,
            "projectionHash": context.projection_hash,
            "materialContextRevision": context.material_context_revision,
        }

    root = shared(value)
    root["schemaVersion"] = "unified_workbench_context.v3"
    root["knowledge"] = {"schemaVersion": "knowledge_workbench_context.v6", **shared(value.knowledge)}
    return root


def adapt_unified_workbench_v2_to_v3(value: UnifiedWorkbenchContextV2) -> UnifiedWorkbenchContextV3:
    shared = dict(
        representation=value.representation,
        artifact=value.artifact,
        artifact_workspace=None,
        lifecycle="DESIGN",
        runtime=None,
        learning=None,
        evidence=None,
        comparison=WorkbenchComparisonV3(subject=None, operands=(), diff_ref=None),
        selection=WorkbenchSelectionV3(**value.selection.model_dump()),
        capabilities=(),
        cursor_ref=None,
        projection_hash=None,
        material_context_revision=value.material_context_revision,
    )
    return UnifiedWorkbenchContextV3(
        knowledge=KnowledgeWorkbenchContextV6(**shared),
        **shared,
    )


class WorkbenchViewActionV2(_Contract):
    schema_version: Literal["workbench_view_action.v2"] = "workbench_view_action.v2"
    command: WorkbenchCommand
    target_refs: tuple[str, ...]
    material_context_revision: int = Field(ge=0)
    target_view: str | None = None
    lens: str | None = None
    baseline_candidate_ref: str | None = None
    candidate_ref: str | None = None

    @model_validator(mode="after")
    def command_arguments_are_complete(self) -> "WorkbenchViewActionV2":
        if self.command in {"FOCUS_OBJECT", "OPEN_EVIDENCE"} and not self.target_refs:
            raise ValueError(f"{self.command} requires target_refs")
        if self.command == "OPEN_VIEW" and self.target_view is None:
            raise ValueError("OPEN_VIEW requires target_view")
        if self.command == "SET_GRAPH_LENS" and self.lens is None:
            raise ValueError("SET_GRAPH_LENS requires lens")
        if self.command == "SET_COMPARISON_PAIR" and (
            self.baseline_candidate_ref is None or self.candidate_ref is None
        ):
            raise ValueError("SET_COMPARISON_PAIR requires both candidate refs")
        return self


class WorkbenchActionAcknowledgementV2(_Contract):
    schema_version: Literal["workbench_action_acknowledgement.v2"] = (
        "workbench_action_acknowledgement.v2"
    )
    command: WorkbenchCommand
    accepted: bool
    material_context_revision: int = Field(ge=0)
    error_code: WorkbenchActionError | None = None


@dataclass(frozen=True)
class WorkbenchViewActionResultV2:
    context: UnifiedWorkbenchContextV2
    acknowledgement: WorkbenchActionAcknowledgementV2


def _resolvable_refs(context: UnifiedWorkbenchContextV2) -> set[str]:
    values = [
        *context.artifact.model_dump().values(),
        *context.comparison.model_dump().values(),
        *context.selection.model_dump().values(),
    ]
    refs: set[str] = set()
    for value in values:
        if isinstance(value, str) and not value.startswith("sha256:"):
            refs.add(value)
        elif isinstance(value, tuple):
            refs.update(value)
    return refs


def apply_workbench_view_action(
    context: UnifiedWorkbenchContextV2,
    action: WorkbenchViewActionV2,
    *,
    forbidden_commands: set[str] | None = None,
) -> WorkbenchViewActionResultV2:
    error: WorkbenchActionError | None = None
    if action.material_context_revision != context.material_context_revision:
        error = "STALE_CONTEXT"
    elif action.command in (forbidden_commands or set()):
        error = "FORBIDDEN"
    elif action.command == "OPEN_VIEW" and action.target_view not in {
        "graph",
        "evidence",
        "lineage",
        "compare",
        "evaluate",
    }:
        error = "UNSUPPORTED_VIEW"
    elif action.command == "SET_COMPARISON_PAIR" and (
        action.baseline_candidate_ref != context.comparison.baseline_candidate_ref
        or action.candidate_ref != context.comparison.candidate_ref
    ):
        error = "INCOMPATIBLE_COMPARISON_PAIR"
    elif any(ref not in _resolvable_refs(context) for ref in action.target_refs):
        error = "UNRESOLVABLE_REF"

    acknowledgement = WorkbenchActionAcknowledgementV2(
        command=action.command,
        accepted=error is None,
        material_context_revision=context.material_context_revision,
        error_code=error,
    )
    if error is not None:
        return WorkbenchViewActionResultV2(context, acknowledgement)

    payload = context.model_dump()
    representation = dict(payload["representation"])
    selection = dict(payload["selection"])
    if action.command == "OPEN_VIEW":
        representation["view"] = action.target_view
    elif action.command == "SET_GRAPH_LENS":
        representation["view"] = "graph"
        representation["lens"] = action.lens
    elif action.command == "OPEN_EVIDENCE":
        representation["view"] = "evidence"
        selection["evidence_refs"] = action.target_refs
    elif action.command == "OPEN_LINEAGE":
        representation["view"] = "lineage"
    elif action.command == "OPEN_COMPARE":
        representation["view"] = "compare"
    elif action.command == "FOCUS_OBJECT":
        selection["object_ref"] = action.target_refs[0] if action.target_refs else None
    payload["representation"] = representation
    payload["selection"] = selection
    payload["knowledge"] = {
        **payload["knowledge"],
        "representation": representation,
        "selection": selection,
    }
    return WorkbenchViewActionResultV2(
        UnifiedWorkbenchContextV2.model_validate(payload), acknowledgement
    )


AdvisoryActionCommandV1 = Literal[
    "FORM_ADVISORY_CONTEXT",
    "PROPOSE_ADVISORY_REPAIR",
]
AdvisoryActionErrorCodeV1 = Literal[
    "INVALID_STATE",
    "INCOMPLETE_CLOSURE",
    "UNSUPPORTED_SOURCE",
    "UNSUPPORTED_VERIFIER_REGISTRY",
    "HASH_MISMATCH",
    "REQUEST_CONFLICT",
    "RESOURCE_LIMIT_EXCEEDED",
    "NOT_FOUND",
    "FORBIDDEN_SCOPE",
]


class _FormAdvisoryContextPayloadV1(_Contract):
    command: Literal["FORM_ADVISORY_CONTEXT"]
    request: AdvisoryContextRequestV1


class _ProposeAdvisoryRepairPayloadV1(_Contract):
    command: Literal["PROPOSE_ADVISORY_REPAIR"]
    request: AdvisoryNextProposalRequestV1


AdvisoryActionPayloadV1 = Annotated[
    _FormAdvisoryContextPayloadV1 | _ProposeAdvisoryRepairPayloadV1,
    Field(discriminator="command"),
]


class AdvisoryActionEnvelopeV1(_Contract):
    schema_version: Literal["advisory_action_envelope.v1"]
    action_version: Literal["advisory-context.v1"]
    ingress: Literal["direct", "chat"]
    session_ref: NonEmptyStr
    payload: AdvisoryActionPayloadV1


class AdvisoryActionErrorDetailsV1(_Contract):
    request_id: Annotated[str, Field(strict=True)]


class AdvisoryActionErrorV1(_Contract):
    code: AdvisoryActionErrorCodeV1
    details: AdvisoryActionErrorDetailsV1


class _FormAdvisoryContextCompleteResultV1(_Contract):
    schema_version: Literal["advisory_action_result.v1"]
    state: Literal["COMPLETE"]
    command: Literal["FORM_ADVISORY_CONTEXT"]
    receipt: AdvisoryContextReceiptV1


class _ProposeAdvisoryRepairCompleteResultV1(_Contract):
    schema_version: Literal["advisory_action_result.v1"]
    state: Literal["COMPLETE"]
    command: Literal["PROPOSE_ADVISORY_REPAIR"]
    receipt: AdvisoryNextProposalReceiptV1


class _AdvisoryActionFailedResultV1(_Contract):
    schema_version: Literal["advisory_action_result.v1"]
    state: Literal["FAILED"]
    command: AdvisoryActionCommandV1
    error: AdvisoryActionErrorV1


class AdvisoryActionResultV1(
    RootModel[
        _FormAdvisoryContextCompleteResultV1
        | _ProposeAdvisoryRepairCompleteResultV1
        | _AdvisoryActionFailedResultV1
    ]
):
    model_config = ConfigDict(frozen=True)


class _CaptureRuntimeTemporalSourcePayloadV1(_Contract):
    command: Literal["CAPTURE_RUNTIME_TEMPORAL_SOURCE"]
    upload_ref: NonEmptyStr
    upload_hash: Sha256
    request_id: NonEmptyStr


class _ImportRuntimeTemporalSourcePayloadV1(_Contract):
    command: Literal["IMPORT_RUNTIME_TEMPORAL_SOURCE"]
    source: ArtifactPinV1
    source_cell_index: Annotated[int, Field(strict=True, ge=0)]
    request_id: NonEmptyStr


class _ResumeTemporalEvaluationPayloadV1(_Contract):
    command: Literal["RESUME_TEMPORAL_EVALUATION"]
    run: ArtifactPinV1


TemporalActionPayloadV1 = Annotated[
    _CaptureRuntimeTemporalSourcePayloadV1
    | _ImportRuntimeTemporalSourcePayloadV1
    | _ResumeTemporalEvaluationPayloadV1,
    Field(discriminator="command"),
]


class TemporalActionEnvelopeV1(_Contract):
    schema_version: Literal["temporal_action_envelope.v1"]
    session_ref: NonEmptyStr
    payload: TemporalActionPayloadV1


class _TemporalActionResultV1(_Contract):
    schema_version: Literal["temporal_action_result.v1"]


class _TemporalSourceCaptureResultV1(_TemporalActionResultV1):
    command: Literal["CAPTURE_RUNTIME_TEMPORAL_SOURCE"]
    output: RuntimeTemporalSourceArchiveV1
    state: None


class _TemporalSourceImportResultV1(_TemporalActionResultV1):
    command: Literal["IMPORT_RUNTIME_TEMPORAL_SOURCE"]
    output: RuntimeTemporalSourceGraphV1
    state: None


class _TemporalResumeResultV1(_TemporalActionResultV1):
    command: Literal["RESUME_TEMPORAL_EVALUATION"]
    output: RuntimeEvaluationStateV1
    state: RuntimeEvaluationStateV1

    @model_validator(mode="after")
    def state_equals_output(self) -> "_TemporalResumeResultV1":
        if self.state != self.output:
            raise ValueError("state must equal the returned evaluation state")
        return self


class TemporalActionResultV1(
    RootModel[
        Annotated[
            _TemporalSourceCaptureResultV1
            | _TemporalSourceImportResultV1
            | _TemporalResumeResultV1,
            Field(discriminator="command"),
        ]
    ]
):
    model_config = ConfigDict(frozen=True)
