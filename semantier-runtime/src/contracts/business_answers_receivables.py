"""Source-bounded descriptive receivables; these artifacts grant no authority."""

from datetime import date, datetime, timedelta
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from eos.artifact_hashing import content_hash
from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.evidence_location import EvidenceRef
from contracts.ontology_building import ArtifactPinV1


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReceivablesRunRequest(Contract):
    import_id: str = Field(min_length=1)
    file_id: str | None = None
    mapping: dict[str, str] | None = None
    snapshot_date: date | None = None
    confirmation: bool | dict[str, str | bool] | None = None
    predecessor_build_ref: str | None = None


class ReceivablesCheck(Contract):
    check_id: str
    status: Literal["passed", "failed", "unavailable"]
    message: str
    row_numbers: tuple[int, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    blocks: Literal["all", "overdue", "none"] = "none"


class ReceivablesInvoice(Contract):
    invoice_id: str
    customer_id: str
    customer_name: str | None = None
    currency: str
    outstanding_balance: str
    overdue_balance: str | None
    due_date: date | None
    row_number: int
    sheet_name: str
    node_ref: str
    evidence_refs: tuple[str, ...]


class ReceivablesCustomer(Contract):
    customer_id: str
    customer_name: str | None = None
    currency: str
    outstanding_balance: str
    overdue_balance: str | None
    invoice_count: int
    outstanding_invoice_count: int
    contributing_row_refs: tuple[str, ...]
    overdue_contributing_row_refs: tuple[str, ...] = ()
    node_ref: str


class ReceivablesCurrencyTotal(Contract):
    currency: str
    outstanding_balance: str
    overdue_balance: str | None
    outstanding_invoice_count: int
    contributing_row_refs: tuple[str, ...]
    overdue_contributing_row_refs: tuple[str, ...] = ()


class ReceivablesArtifact(Contract):
    artifact_ref: str
    artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str
    workspace_id: str
    created_at: datetime

    @model_validator(mode="after")
    def identity(self):
        if self.created_at.utcoffset() != timedelta(0):
            raise ValueError("UTC timestamp required")
        digest = content_hash(
            self.model_dump(mode="json", exclude={"artifact_ref", "artifact_hash"})
        )
        if (
            self.artifact_hash != "sha256:" + digest
            or self.artifact_ref
            != self.schema_version.removesuffix(".v1")
            + ":"
            + (
                self.request_hash.removeprefix("sha256:")
                if self.schema_version == "receivables_run.v1"
                else digest
            )
        ):
            raise ValueError("RECEIVABLES_ARTIFACT_HASH_MISMATCH")
        return self


class ReceivablesSourceV1(ReceivablesArtifact):
    schema_version: Literal["receivables_source.v1"] = "receivables_source.v1"
    import_id: str
    file_id: str
    source_identity_ref: str
    source_hash: str
    byte_owner_ref: str
    byte_size: int
    original_name: str
    parser_profile: dict
    parser_profile_hash: str
    canonical_source_ir: CanonicalSourceDocument
    canonical_source_ir_pin: ArtifactPinV1
    evidence: tuple[EvidenceRef, ...]
    sheet_name: str
    header_row: int
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    row_numbers: tuple[int, ...]


class ReceivablesReportV1(ReceivablesArtifact):
    schema_version: Literal["receivables_report.v1"] = "receivables_report.v1"
    build_ref: str
    source_pin: ArtifactPinV1
    mapping: dict[str, str]
    mapping_hash: str
    schema_hash: str
    snapshot_pin: ArtifactPinV1
    canonical_source_ir_pin: ArtifactPinV1
    parser_profile_hash: str
    recipe_hash: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    recipe_id: Literal["receivables.v1"] = "receivables.v1"
    recipe_version: Literal["1"] = "1"
    competency_question_version: Literal["receivables_snapshot.v1"] = (
        "receivables_snapshot.v1"
    )
    snapshot_date: date | None
    checks: tuple[ReceivablesCheck, ...]
    business_definition_status: Literal[
        "source_explicit", "confirmed", "needs_confirmation"
    ]
    confirmation_actor_id: str | None = None
    confirmation: dict[str, str | bool] | None = None
    governance_state: Literal["candidate"] = "candidate"
    row_count: int


class ReceivablesAnswerV1(ReceivablesArtifact):
    schema_version: Literal["receivables_answer.v1"] = "receivables_answer.v1"
    run_ref: str
    build_ref: str
    snapshot_ref: str
    build_hash: str
    source_pin: ArtifactPinV1
    report_pin: ArtifactPinV1
    snapshot_date: date
    source_name: str
    governance_state: Literal["candidate"] = "candidate"
    business_definition_status: str
    row_count: int
    outstanding_invoice_count: int
    overdue_available: bool
    customers: tuple[ReceivablesCustomer, ...]
    invoices: tuple[ReceivablesInvoice, ...]
    totals: tuple[ReceivablesCurrencyTotal, ...]


class ReceivablesRunV1(ReceivablesArtifact):
    schema_version: Literal["receivables_run.v1"] = "receivables_run.v1"
    request: ReceivablesRunRequest
    request_hash: str
    state: Literal[
        "needs_confirmation", "needs_correction", "answered", "partially_answered"
    ]
    build_ref: str
    snapshot_ref: str
    build_hash: str
    source_pin: ArtifactPinV1
    report_pin: ArtifactPinV1
    answer_ref: str | None
    mapping: dict[str, str]
    available_columns: tuple[str, ...]
    checks: tuple[ReceivablesCheck, ...]
    stages: tuple[str, ...]
    governance_state: Literal["candidate"] = "candidate"
    business_definition_status: str


RECEIVABLES_MODELS = {
    model.model_fields["schema_version"].default: model
    for model in (
        ReceivablesSourceV1,
        ReceivablesReportV1,
        ReceivablesAnswerV1,
        ReceivablesRunV1,
    )
}


def make_artifact(model, **basis):
    # Populate contract defaults before hashing the complete persisted payload.
    provisional = model.model_construct(
        artifact_ref="pending", artifact_hash="sha256:" + "0" * 64, **basis
    )
    payload = provisional.model_dump(
        mode="json", exclude={"artifact_ref", "artifact_hash"}
    )
    digest = content_hash(payload)
    ref_digest = (
        payload["request_hash"].removeprefix("sha256:")
        if payload["schema_version"] == "receivables_run.v1"
        else digest
    )
    return model.model_validate(
        {
            "artifact_ref": payload["schema_version"].removesuffix(".v1")
            + ":"
            + ref_digest,
            "artifact_hash": "sha256:" + digest,
            **payload,
        }
    )
