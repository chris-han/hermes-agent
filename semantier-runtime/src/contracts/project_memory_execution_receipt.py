"""Source-owned execution receipt and derived project-memory feature contracts."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
TriState = Literal["YES", "NO", "UNKNOWN"]
FreshnessState = Literal["CURRENT", "STALE", "UNKNOWN"]
TerminalKind = Literal["SUCCESS", "FAILURE", "OTHER", "UNKNOWN"]
FeatureName = Literal[
    "run_identity_present",
    "worker_assignment_present",
    "first_step_time_present",
    "activity_signal_present",
    "activity_signal_freshness",
    "terminal_event_present",
    "terminal_event_kind",
    "terminal_event_time_present",
    "explicit_no_start",
]
FeatureDerivationMethod = Literal[
    "EXACT_STRUCTURED_FIELD",
    "SOURCE_TIME_COMPARISON",
    "EXPLICIT_NEGATIVE_FIELD",
    "UNRESOLVED",
]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def persisted_timestamps_are_utc(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime):
                if value.tzinfo is None or value.utcoffset() != timedelta(0):
                    raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class TerminalExecutionEventV1(_Contract):
    kind: Literal["SUCCESS", "FAILURE", "OTHER"]
    occurred_at: datetime


class ExecutionReceiptSourceV1(_Contract):
    schema_version: Literal["execution_receipt_source.v1"] = (
        "execution_receipt_source.v1"
    )
    producer_schema: Literal["hermes.execution_receipt.v1"]
    source_object_ref: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    observed_at: datetime
    run_id: str | None = None
    worker_id: str | None = None
    started_at: datetime | None = None
    activity_observed_at: datetime | None = None
    activity_freshness_window_seconds: int | None = Field(default=None, gt=0, le=86400)
    terminal_event: TerminalExecutionEventV1 | None = None
    no_start: bool | None = None

    @field_validator("run_id", "worker_id")
    @classmethod
    def nonempty_optional_ids(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(
                "optional execution identity must be non-empty when present"
            )
        return value

    @model_validator(mode="after")
    def validate_execution_consistency(self):
        for field in ("started_at", "activity_observed_at"):
            value = getattr(self, field)
            if value is not None and value > self.observed_at:
                raise ValueError(f"{field} cannot be later than observed_at")
        if (
            self.terminal_event is not None
            and self.terminal_event.occurred_at > self.observed_at
        ):
            raise ValueError(
                "terminal_event.occurred_at cannot be later than observed_at"
            )
        if (self.activity_observed_at is None) != (
            self.activity_freshness_window_seconds is None
        ):
            raise ValueError(
                "activity timestamp and freshness window must both be present or absent"
            )
        if self.no_start is True and any(
            value is not None
            for value in (
                self.run_id,
                self.started_at,
                self.activity_observed_at,
                self.terminal_event,
            )
        ):
            raise ValueError(
                "explicit no_start conflicts with execution-start evidence"
            )
        return self


class ExecutionFeatureDerivationV1(_Contract):
    feature: FeatureName
    method: FeatureDerivationMethod
    source_locator: str = Field(min_length=1)


class StructuredEventFeatureV1(_Contract):
    schema_version: Literal["structured_event_feature.v1"] = (
        "structured_event_feature.v1"
    )
    feature_ref: str
    feature_hash: str = Field(pattern=SHA256_PATTERN)
    memory_candidate: ArtifactPinV1
    producer_schema: Literal["hermes.execution_receipt.v1"]
    source_object_ref: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    observed_at: datetime

    run_identity_present: TriState
    worker_assignment_present: TriState
    first_step_time_present: TriState
    activity_signal_present: TriState
    activity_signal_freshness: FreshnessState
    terminal_event_present: TriState
    terminal_event_kind: TerminalKind
    terminal_event_time_present: TriState
    explicit_no_start: TriState

    derivations: tuple[ExecutionFeatureDerivationV1, ...]
    compiler_profile: Literal["execution-receipt-feature.v1"] = (
        "execution-receipt-feature.v1"
    )

    @field_validator("memory_candidate")
    @classmethod
    def candidate_pin_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        if value.artifact_kind != "memory_candidate":
            raise ValueError("memory_candidate must pin artifact kind memory_candidate")
        return value

    @model_validator(mode="after")
    def validate_feature_identity(self):
        names = [item.feature for item in self.derivations]
        if len(names) != len(set(names)):
            raise ValueError("feature derivations must be unique per feature")
        expected_names = set(FeatureName.__args__)
        if set(names) != expected_names:
            raise ValueError(
                "feature derivations must cover the complete execution feature set"
            )
        payload = self.model_dump(mode="json")
        payload.pop("feature_ref", None)
        payload.pop("feature_hash", None)
        digest = content_hash(payload)
        expected_hash = "sha256:" + digest
        expected_ref = "structured-event-feature:" + digest[:24]
        if self.feature_hash != expected_hash or self.feature_ref != expected_ref:
            raise ValueError("structured event feature identity mismatch")
        return self


def bind_structured_event_feature_identity(payload: dict[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value.pop("feature_ref", None)
    value.pop("feature_hash", None)
    value.setdefault("schema_version", "structured_event_feature.v1")
    value.setdefault("compiler_profile", "execution-receipt-feature.v1")
    if isinstance(value.get("observed_at"), datetime):
        value["observed_at"] = value["observed_at"].isoformat().replace("+00:00", "Z")
    value["derivations"] = [
        ExecutionFeatureDerivationV1.model_validate(item).model_dump(mode="json")
        for item in value.get("derivations", ())
    ]
    digest = content_hash(value)
    value["feature_ref"] = "structured-event-feature:" + digest[:24]
    value["feature_hash"] = "sha256:" + digest
    return value
