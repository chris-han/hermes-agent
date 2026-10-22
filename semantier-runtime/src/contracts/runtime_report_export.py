"""Typed deterministic render outputs; publication and access stay elsewhere."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import WorkspaceScope
from eos.artifact_hashing import content_hash

RuntimeExportFormat = Literal[
    "HTML", "DOCX", "JSON", "EVIDENCE_PACKAGE", "REPLAY_PACKAGE", "RDF", "TRIPLETS", "QUERY_RESULTS"
]
RuntimeAccessProjection = Literal["MEMBER", "AUDIT"]
RuntimeExportState = Literal["PENDING", "COMPLETE", "ABANDONED"]
_Hash = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class _ExportContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def utc_times(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() != timedelta(0)):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class PinnedRenderProfileV1(_ExportContract):
    schema_version: Literal["pinned_render_profile.v1"] = "pinned_render_profile.v1"
    profile_ref: str = Field(min_length=1)
    profile_hash: _Hash
    format: RuntimeExportFormat
    media_type: str = Field(min_length=1)
    renderer: ArtifactPinV1
    template: ArtifactPinV1
    environment: ArtifactPinV1
    redaction_policy: ArtifactPinV1
    canonical_filename: str = Field(min_length=1)
    archive_timestamp: datetime
    options: Mapping[str, str | int | bool]

    @classmethod
    def create(cls, **values: object) -> "PinnedRenderProfileV1":
        basis = dict(values)
        basis.setdefault("schema_version", "pinned_render_profile.v1")
        digest = content_hash(_jsonable(basis))
        return cls(**values, profile_ref=f"pinned-render-profile:{digest}", profile_hash=f"sha256:{digest}")

    @model_validator(mode="after")
    def identity_is_bound(self):
        basis = self.model_dump(mode="json", exclude={"profile_ref", "profile_hash"})
        digest = content_hash(basis)
        if (self.profile_ref, self.profile_hash) != (f"pinned-render-profile:{digest}", f"sha256:{digest}"):
            raise ValueError("render profile identity does not bind exact content")
        return self


class ExportBytesV1(_ExportContract):
    schema_version: Literal["runtime_export_bytes.v1"] = "runtime_export_bytes.v1"
    format: RuntimeExportFormat
    media_type: str = Field(min_length=1)
    canonical_filename: str = Field(min_length=1)
    content: bytes
    content_hash: _Hash
    size_bytes: int = Field(ge=0)
    render_profile: ArtifactPinV1
    access_projection: RuntimeAccessProjection
    source_timestamps: tuple[datetime, ...]


class RuntimeReportExportReceiptV1(_ExportContract):
    schema_version: Literal["runtime_report_export_receipt.v1"] = "runtime_report_export_receipt.v1"
    receipt_ref: str = Field(min_length=1)
    receipt_hash: _Hash
    scope: WorkspaceScope
    report: ArtifactPinV1
    replay_inputs: ArtifactPinV1
    format: RuntimeExportFormat
    access_projection: RuntimeAccessProjection
    render_profile: ArtifactPinV1
    content_hash: _Hash
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1)
    stage_ref: str | None
    published_artifact: ArtifactPinV1 | None
    request_id: str = Field(min_length=1)
    request_hash: _Hash
    state: RuntimeExportState
    predecessor_intent: ArtifactPinV1 | None
    actor_ref: str = Field(min_length=1)
    recorded_at: datetime
    completed_at: datetime | None
    abandoned_at: datetime | None

    @model_validator(mode="after")
    def validate_state(self):
        if self.state == "PENDING":
            if self.stage_ref is None or self.published_artifact is not None or self.completed_at is not None or self.abandoned_at is not None:
                raise ValueError("PENDING export receipt has invalid completion fields")
        elif self.state == "COMPLETE":
            if self.stage_ref is None or self.published_artifact is None or self.completed_at is None or self.abandoned_at is not None:
                raise ValueError("COMPLETE export receipt has invalid completion fields")
            if self.published_artifact.artifact_hash != self.content_hash:
                raise ValueError("published export hash does not match content")
        else:
            if self.abandoned_at is None or self.published_artifact is not None or self.completed_at is not None:
                raise ValueError("ABANDONED export receipt has invalid completion fields")
        if self.report.artifact_kind != "runtime_inspection_report" or self.replay_inputs.artifact_kind != "runtime_report_replay_inputs":
            raise ValueError("export receipt input pin kind mismatch")
        return self

    @model_validator(mode="after")
    def identity_is_bound(self):
        basis = self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"})
        digest = content_hash(basis)
        if (self.receipt_ref, self.receipt_hash) != (
            f"runtime-report-export-receipt:{digest}", f"sha256:{digest}"
        ):
            raise ValueError("export receipt identity does not bind exact content")
        return self

    @classmethod
    def create(cls, **values: object) -> "RuntimeReportExportReceiptV1":
        basis = _jsonable(values)
        assert isinstance(basis, dict)
        basis.setdefault("schema_version", "runtime_report_export_receipt.v1")
        digest = content_hash(basis)
        return cls(**values, receipt_ref=f"runtime-report-export-receipt:{digest}", receipt_hash=f"sha256:{digest}")


class RuntimeOwnerPortErrorV1(_ExportContract):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class RuntimeReportExportSuccessV1(_ExportContract):
    status: Literal["OK"] = "OK"
    receipt: RuntimeReportExportReceiptV1


class RuntimeReportExportFailureV1(_ExportContract):
    status: Literal["ERROR"] = "ERROR"
    error: RuntimeOwnerPortErrorV1


RuntimeReportExportResultV1 = RuntimeReportExportSuccessV1 | RuntimeReportExportFailureV1


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


__all__ = [
    "ExportBytesV1", "PinnedRenderProfileV1", "RuntimeAccessProjection",
    "RuntimeExportFormat", "RuntimeReportExportFailureV1", "RuntimeReportExportReceiptV1",
    "RuntimeReportExportResultV1", "RuntimeReportExportSuccessV1", "RuntimeExportState",
]
