"""Strict contracts for pure runtime semantic evaluation and provider queries."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, ClassVar, Literal, Mapping, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_context_reconstruction import CapturedJsonV1
from contracts.ontology_validation import (
    CandidateRuleEvaluationV1,
    RuntimeJustificationV1,
)
from eos.artifact_hashing import content_hash


_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_EvaluationOutcome = Literal["MATCH", "NO_MATCH", "UNKNOWN"]
_StrictCount = Annotated[int, Field(strict=True, ge=0)]
_StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]
_TemporalPatternGapTolerance = Annotated[
    int, Field(strict=True, ge=1576800000, le=1576800000)
]
_NonEmptyStr = Annotated[str, Field(strict=True, min_length=1)]
_AsciiId = Annotated[
    str,
    Field(strict=True, pattern=r"^[A-Za-z][A-Za-z0-9_.:-]*$"),
]
_Sha256 = Annotated[str, Field(strict=True, pattern=_SHA256_PATTERN)]


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        encoded = value.isoformat()
        return (
            encoded.replace("+00:00", "Z")
            if value.utcoffset() == timedelta(0)
            else encoded
        )
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _mapping(value: BaseModel | Mapping[str, object]) -> dict[str, object]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return {str(key): _json_value(item) for key, item in value.items()}


def _pin_key(value: ArtifactPinV1 | Mapping[str, object]) -> tuple[str, str, str]:
    payload = _mapping(value)
    return (
        str(payload["artifact_kind"]),
        str(payload["artifact_ref"]),
        str(payload["artifact_hash"]),
    )


def _canonical_pin_set(
    values: tuple[ArtifactPinV1, ...], label: str
) -> tuple[ArtifactPinV1, ...]:
    keys = [_pin_key(pin) for pin in values]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} contains duplicate pins")
    return tuple(sorted(values, key=_pin_key))


def _ordered_pin_values(value: object) -> object:
    if not isinstance(value, (list, tuple)):
        return value
    ordered = sorted(value, key=_pin_key)
    return tuple(ordered) if isinstance(value, tuple) else ordered


def _hash_basis(
    value: BaseModel | Mapping[str, object], ref_field: str, hash_field: str
) -> dict[str, object]:
    basis = _mapping(value)
    basis.pop(ref_field, None)
    basis.pop(hash_field, None)
    return basis


def runtime_evaluation_run_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    """Return every run field except the envelope's own ref/hash pair."""

    return _hash_basis(value, "run_ref", "run_hash")


def runtime_evaluation_diagnostic_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    """Return the canonical diagnostic content bound by its ref/hash pair."""

    basis = _hash_basis(value, "diagnostic_ref", "diagnostic_hash")
    for field in ("input_pins", "evidence_pins"):
        basis[field] = _ordered_pin_values(basis.get(field, []))
    return basis


def runtime_temporal_query_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    """Return query content with its boundary default and without its identity."""

    basis = _hash_basis(value, "query_ref", "query_hash")
    basis.setdefault("include_boundaries", True)
    return basis


def runtime_provider_descriptor_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    """Return every descriptor field except its own ref/hash pair."""

    return _hash_basis(value, "descriptor_ref", "descriptor_hash")


def runtime_provider_profile_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    """Return every profile field except its own ref/hash pair."""

    return _hash_basis(value, "profile_ref", "profile_hash")


def runtime_provider_request_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    """Return every request field except its own ref/hash pair."""

    return _hash_basis(value, "request_ref", "request_hash")


def runtime_execution_admission_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    return _hash_basis(value, "admission_ref", "admission_hash")


def runtime_temporal_capture_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    return _hash_basis(value, "capture_ref", "capture_hash")


def runtime_temporal_world_snapshot_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    return _hash_basis(value, "snapshot_ref", "snapshot_hash")


def runtime_provider_result_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    return _hash_basis(value, "result_ref", "result_hash")


def runtime_evaluation_state_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    return _hash_basis(value, "state_ref", "state_hash")


def runtime_evaluation_request_identity_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    return _mapping(value)


def runtime_evaluation_request_outcome_hash_basis(
    value: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    return _hash_basis(value, "outcome_ref", "outcome_hash")


class _RuntimeIdentityContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_utc_timestamps(self) -> "_RuntimeIdentityContract":
        for field in type(self).model_fields:
            value = getattr(self, field)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{field} must be timezone-aware UTC")
        return self


class _AddressedRuntimeIdentity(_RuntimeIdentityContract):
    ARTIFACT_KIND: ClassVar[str]
    REF_FIELD: ClassVar[str]
    HASH_FIELD: ClassVar[str]
    REF_PREFIX: ClassVar[str]

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        raise NotImplementedError

    @model_validator(mode="after")
    def validate_addressed_identity(self) -> "_AddressedRuntimeIdentity":
        digest = content_hash(type(self)._basis(self))
        if (
            getattr(self, self.REF_FIELD),
            getattr(self, self.HASH_FIELD),
        ) != (
            f"{self.REF_PREFIX}:{digest}",
            f"sha256:{digest}",
        ):
            raise ValueError("addressed identity does not bind exact content")
        return self

    @classmethod
    def create(cls, **values: object):
        provisional = cls.model_construct(
            **values,
            **{
                cls.REF_FIELD: "pending",
                cls.HASH_FIELD: "sha256:" + "0" * 64,
            },
        )
        digest = content_hash(cls._basis(provisional))
        return cls(
            **values,
            **{
                cls.REF_FIELD: f"{cls.REF_PREFIX}:{digest}",
                cls.HASH_FIELD: f"sha256:{digest}",
            },
        )

    def artifact_pin(self) -> ArtifactPinV1:
        """Return the native pin for this addressed runtime artifact."""

        return ArtifactPinV1(
            artifact_kind=self.ARTIFACT_KIND,
            artifact_ref=getattr(self, self.REF_FIELD),
            artifact_hash=getattr(self, self.HASH_FIELD),
        )


class TemporalQueryV1(_AddressedRuntimeIdentity):
    """Pure addressed query shape for a closed temporal provider."""

    ARTIFACT_KIND = "runtime_temporal_query"
    REF_FIELD = "query_ref"
    HASH_FIELD = "query_hash"
    REF_PREFIX = "runtime-temporal-query"

    schema_version: Literal["runtime_temporal_query.v1"]
    query_ref: _NonEmptyStr
    query_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    operation: Literal[
        "point", "range", "path", "pattern", "evolution", "world_snapshot"
    ]
    source_graph: ArtifactPinV1
    at_time: datetime | None
    start_time: datetime | None
    end_time: datetime | None
    source_id: _NonEmptyStr | None
    target_id: _NonEmptyStr | None
    pattern_type: Literal["sequence", "cycle"] | None
    min_frequency: _StrictPositiveInt | None
    query_text: Annotated[str, Field(strict=True)]
    include_boundaries: Annotated[bool, Field(strict=True)] = True

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_temporal_query_hash_basis(value)

    @model_validator(mode="after")
    def validate_operation_shape(self) -> "TemporalQueryV1":
        if self.operation in ("point", "world_snapshot"):
            if self.at_time is None:
                raise ValueError(f"{self.operation} requires at_time")
            if any(
                value is not None
                for value in (
                    self.start_time,
                    self.end_time,
                    self.source_id,
                    self.target_id,
                    self.pattern_type,
                    self.min_frequency,
                )
            ):
                raise ValueError(f"{self.operation} has irrelevant non-null fields")
        elif self.operation in ("range", "evolution"):
            if (
                self.at_time is not None
                or self.start_time is None
                or self.end_time is None
            ):
                raise ValueError(
                    f"{self.operation} requires start_time/end_time and null at_time"
                )
            if any(
                value is not None
                for value in (
                    self.source_id,
                    self.target_id,
                    self.pattern_type,
                    self.min_frequency,
                )
            ):
                raise ValueError(f"{self.operation} has irrelevant non-null fields")
            if self.operation == "evolution" and not self.include_boundaries:
                raise ValueError("evolution requires include_boundaries=True")
        elif self.operation == "path":
            if (
                self.at_time is not None
                or self.source_id is None
                or self.target_id is None
                or self.pattern_type is not None
                or self.min_frequency is not None
            ):
                raise ValueError("path requires only source_id and target_id")
            if (self.start_time is None) != (self.end_time is None):
                raise ValueError("path bounds must be both null or both present")
            if not self.include_boundaries:
                raise ValueError("path requires include_boundaries=True")
        else:
            if (
                self.at_time is not None
                or self.start_time is not None
                or self.end_time is not None
                or self.source_id is not None
                or self.target_id is not None
                or self.pattern_type is None
                or self.min_frequency is None
            ):
                raise ValueError("pattern requires only pattern_type and min_frequency")
            if not self.include_boundaries:
                raise ValueError("pattern requires include_boundaries=True")

        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time > self.end_time
        ):
            raise ValueError("start_time must be before or equal to end_time")
        return self


class GovernedTargetPinV1(_RuntimeIdentityContract):
    """Pure scoped target pins for a runtime evaluation request."""

    schema_version: Literal["governed_target_pin.v1"]
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    artifact: ArtifactPinV1
    target_type: Literal["document", "graph", "dataset", "query"]
    evidence: tuple[ArtifactPinV1, ...]
    scope_bridge: ArtifactPinV1 | None

    @field_validator("evidence")
    @classmethod
    def canonicalize_evidence(
        cls, value: tuple[ArtifactPinV1, ...]
    ) -> tuple[ArtifactPinV1, ...]:
        return _canonical_pin_set(value, "evidence")

    @field_validator("scope_bridge")
    @classmethod
    def validate_scope_bridge_kind(
        cls, value: ArtifactPinV1 | None
    ) -> ArtifactPinV1 | None:
        if value is not None and value.artifact_kind != "runtime_scope_bridge_receipt":
            raise ValueError("scope_bridge must pin a runtime_scope_bridge_receipt")
        return value


class RuntimeTemporalProviderDescriptorV1(_AddressedRuntimeIdentity):
    """Addressed implementation identity for the closed temporal provider."""

    ARTIFACT_KIND = "runtime_provider_descriptor"
    REF_FIELD = "descriptor_ref"
    HASH_FIELD = "descriptor_hash"
    REF_PREFIX = "runtime-provider-descriptor"

    schema_version: Literal["runtime_provider_descriptor.v1"]
    descriptor_ref: _NonEmptyStr
    descriptor_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    provider_kind: Literal["semantica_temporal"]
    implementation_id: Literal["semantier.runtime_temporal"]
    implementation_version: Literal["v1"]
    implementation_hash: _Sha256

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_provider_descriptor_hash_basis(value)


class RuntimeTemporalProviderConfigV1(_RuntimeIdentityContract):
    """Closed local behavior configuration for the temporal provider."""

    temporal_granularity: Literal["second"]
    pattern_gap_tolerance_seconds: _TemporalPatternGapTolerance


class RuntimeTemporalDependencyV1(_RuntimeIdentityContract):
    """Exact Semantica source declaration bound by a temporal profile."""

    distribution: Literal["semantica"]
    package_metadata_version: Literal["0.7.0"]
    source_revision: Literal["3e4338695dead4bd6c598f491335715fffb02d0d"]


class RuntimeTemporalProviderProfileV1(_AddressedRuntimeIdentity):
    """Addressed closed local profile for the temporal provider."""

    ARTIFACT_KIND = "runtime_provider_profile"
    REF_FIELD = "profile_ref"
    HASH_FIELD = "profile_hash"
    REF_PREFIX = "runtime-provider-profile"

    schema_version: Literal["runtime_provider_profile.v1"]
    profile_ref: _NonEmptyStr
    profile_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    profile_id: Literal["semantica_temporal_local_v1"]
    provider: ArtifactPinV1
    query_schema_version: Literal["runtime_temporal_query.v1"]
    configuration: RuntimeTemporalProviderConfigV1
    dependency: RuntimeTemporalDependencyV1

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_provider_profile_hash_basis(value)

    @field_validator("provider")
    @classmethod
    def validate_provider_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        if value.artifact_kind != "runtime_provider_descriptor":
            raise ValueError("provider must pin a runtime_provider_descriptor")
        return value


class RuntimeTemporalDependencyV2(_RuntimeIdentityContract):
    """Exact qualified successor composition; V1 remains historical-only."""

    distribution: Literal["semantica"]
    package_metadata_version: Literal["0.7.0"]
    source_revision: Literal["701aa6a755899b8b63ce38ee971827b3e2c092e7"]


class RuntimeTemporalProviderProfileV2(RuntimeTemporalProviderProfileV1):
    """Successor profile identity with the unchanged configuration/hash rules."""

    schema_version: Literal["runtime_provider_profile.v2"]
    profile_id: Literal["semantica_temporal_local_v2"]
    dependency: RuntimeTemporalDependencyV2


def runtime_temporal_profile_model(
    schema_version: object,
) -> type[RuntimeTemporalProviderProfileV1] | type[RuntimeTemporalProviderProfileV2]:
    """Closed retained-profile dispatch; decoding never implies executability."""
    if schema_version == "runtime_provider_profile.v1":
        return RuntimeTemporalProviderProfileV1
    if schema_version == "runtime_provider_profile.v2":
        return RuntimeTemporalProviderProfileV2
    raise ValueError("unsupported runtime temporal profile schema")


class RuntimeTemporalRequestSettingsV1(_RuntimeIdentityContract):
    """Closed empty settings for the first temporal request branch."""


class RuntimeTemporalProviderRequestV1(_AddressedRuntimeIdentity):
    """Addressed pure request for the closed temporal provider branch."""

    ARTIFACT_KIND = "runtime_provider_request"
    REF_FIELD = "request_ref"
    HASH_FIELD = "request_hash"
    REF_PREFIX = "runtime-provider-request"

    schema_version: Literal["runtime_provider_request.v1"]
    request_ref: _NonEmptyStr
    request_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    context: ArtifactPinV1
    world: ArtifactPinV1
    target: GovernedTargetPinV1
    provider: ArtifactPinV1
    profile: ArtifactPinV1
    query: ArtifactPinV1
    source_pins: tuple[ArtifactPinV1, ...]
    model_pins: tuple[ArtifactPinV1, ...]
    index_pins: tuple[ArtifactPinV1, ...]
    namespace_pin: ArtifactPinV1 | None
    requested_operation: Literal[
        "point", "range", "path", "pattern", "evolution", "world_snapshot"
    ]
    settings: RuntimeTemporalRequestSettingsV1

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_provider_request_hash_basis(value)

    @model_validator(mode="after")
    def validate_local_request_shape(self) -> "RuntimeTemporalProviderRequestV1":
        if self.context.artifact_kind == "context_graph":
            if self.world != self.context:
                raise ValueError("kernel request world must equal exact ContextGraph pin")
        elif (
            self.context.artifact_kind != "runtime_working_context"
            or self.world.artifact_kind != "runtime_world_scope"
        ):
            raise ValueError("context/world must use one supported runtime context basis")
        expected_kinds = {
            "provider": "runtime_provider_descriptor",
            "profile": "runtime_provider_profile",
            "query": "runtime_temporal_query",
        }
        for field, expected_kind in expected_kinds.items():
            if getattr(self, field).artifact_kind != expected_kind:
                raise ValueError(f"{field} must pin a {expected_kind}")
        if (self.organization_id, self.workspace_id) != (
            self.target.organization_id,
            self.target.workspace_id,
        ):
            raise ValueError("request and target scopes must match")
        if self.target.target_type != "graph":
            raise ValueError("target must have target_type graph")
        if self.source_pins != (self.target.artifact,):
            raise ValueError("source_pins must contain exactly target.artifact")
        if self.model_pins:
            raise ValueError("model_pins must be empty")
        if self.index_pins:
            raise ValueError("index_pins must be empty")
        if self.namespace_pin is not None:
            raise ValueError("namespace_pin must be null")
        return self


class RuntimeEvaluationRunV1(_AddressedRuntimeIdentity):
    """Immutable local-shape identity for one admitted evaluation run."""

    ARTIFACT_KIND = "runtime_evaluation_run"
    REF_FIELD = "run_ref"
    HASH_FIELD = "run_hash"
    REF_PREFIX = "runtime-evaluation-run"

    schema_version: Literal["runtime_evaluation_run.v1"]
    run_ref: _NonEmptyStr
    run_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    admission: ArtifactPinV1
    root_run: ArtifactPinV1 | None
    parent_run: ArtifactPinV1 | None
    request_id: _NonEmptyStr
    created_by: _NonEmptyStr
    created_at: datetime

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_evaluation_run_hash_basis(value)

    @model_validator(mode="before")
    @classmethod
    def reject_local_self_references(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        own_identity = (value.get("run_ref"), value.get("run_hash"))
        for field in ("root_run", "parent_run"):
            pin = value.get(field)
            if isinstance(pin, BaseModel):
                pin_identity = (
                    getattr(pin, "artifact_ref", None),
                    getattr(pin, "artifact_hash", None),
                )
            elif isinstance(pin, Mapping):
                pin_identity = (pin.get("artifact_ref"), pin.get("artifact_hash"))
            else:
                continue
            if pin_identity == own_identity:
                raise ValueError(f"{field} must not reference its own identity")
        return value

    @model_validator(mode="after")
    def validate_local_run_shape(self) -> "RuntimeEvaluationRunV1":
        if self.admission.artifact_kind != "runtime_execution_admission":
            raise ValueError("admission must pin a runtime_execution_admission")
        if (self.root_run is None) != (self.parent_run is None):
            raise ValueError(
                "root_run and parent_run must be both null or both present"
            )
        for field in ("root_run", "parent_run"):
            pin = getattr(self, field)
            if pin is None:
                continue
            if pin.artifact_kind != "runtime_evaluation_run":
                raise ValueError(f"{field} must pin a runtime_evaluation_run")
            if (pin.artifact_ref, pin.artifact_hash) == (self.run_ref, self.run_hash):
                raise ValueError(f"{field} must not reference its own identity")
        return self


class RuntimeEvaluationDiagnosticV1(_AddressedRuntimeIdentity):
    """Immutable diagnostic identity without authority or sanitization claims."""

    ARTIFACT_KIND = "runtime_evaluation_diagnostic"
    REF_FIELD = "diagnostic_ref"
    HASH_FIELD = "diagnostic_hash"
    REF_PREFIX = "runtime-evaluation-diagnostic"

    schema_version: Literal["runtime_evaluation_diagnostic.v1"]
    diagnostic_ref: _NonEmptyStr
    diagnostic_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    run: ArtifactPinV1
    code: _AsciiId
    message: _NonEmptyStr
    input_pins: tuple[ArtifactPinV1, ...]
    evidence_pins: tuple[ArtifactPinV1, ...]
    recorded_at: datetime

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_evaluation_diagnostic_hash_basis(value)

    @field_validator("input_pins", "evidence_pins")
    @classmethod
    def canonicalize_pin_sets(
        cls, value: tuple[ArtifactPinV1, ...], info
    ) -> tuple[ArtifactPinV1, ...]:
        return _canonical_pin_set(value, info.field_name)

    @field_validator("run")
    @classmethod
    def validate_run_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        if value.artifact_kind != "runtime_evaluation_run":
            raise ValueError("run must pin a runtime_evaluation_run")
        return value


TemporalOperation: TypeAlias = Literal[
    "point", "range", "path", "pattern", "evolution", "world_snapshot"
]


class RuntimeExecutionAdmissionV1(_AddressedRuntimeIdentity):
    ARTIFACT_KIND = "runtime_execution_admission"
    REF_FIELD = "admission_ref"
    HASH_FIELD = "admission_hash"
    REF_PREFIX = "runtime-execution-admission"

    schema_version: Literal["runtime_execution_admission.v1"]
    admission_ref: _NonEmptyStr
    admission_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    actor_id: _NonEmptyStr
    session_ref: _NonEmptyStr
    context: ArtifactPinV1
    world: ArtifactPinV1
    provider_request: ArtifactPinV1
    target: GovernedTargetPinV1
    context_stream_ref: _NonEmptyStr
    context_revision: _StrictCount
    certified_plan: ArtifactPinV1 | None
    epoch: ArtifactPinV1 | None
    activation_snapshot: ArtifactPinV1 | None
    activation_events: tuple[ArtifactPinV1, ...]
    business_effective_at: datetime
    admitted_at: datetime

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_execution_admission_hash_basis(value)

    @model_validator(mode="after")
    def validate_nonpolicy_temporal_shape(self) -> "RuntimeExecutionAdmissionV1":
        if self.context.artifact_kind == "context_graph":
            if self.world != self.context:
                raise ValueError("kernel admission world must equal exact ContextGraph pin")
        elif (
            self.context.artifact_kind != "runtime_working_context"
            or self.world.artifact_kind != "runtime_world_scope"
        ):
            raise ValueError("context/world must use one supported runtime context basis")
        if self.provider_request.artifact_kind != "runtime_provider_request":
            raise ValueError("provider_request must pin runtime_provider_request")
        if (self.organization_id, self.workspace_id) != (
            self.target.organization_id,
            self.target.workspace_id,
        ):
            raise ValueError("admission and target scopes must match")
        if any(
            value is not None
            for value in (self.certified_plan, self.epoch, self.activation_snapshot)
        ) or self.activation_events:
            raise ValueError("temporal nonpolicy admission must not assert policy authority")
        return self


class RuntimeTemporalCaptureV1(_AddressedRuntimeIdentity):
    ARTIFACT_KIND = "runtime_temporal_capture"
    REF_FIELD = "capture_ref"
    HASH_FIELD = "capture_hash"
    REF_PREFIX = "runtime-temporal-capture"

    schema_version: Literal["runtime_temporal_capture.v1"]
    capture_ref: _NonEmptyStr
    capture_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    run: ArtifactPinV1
    provider_request: ArtifactPinV1
    source_graph: ArtifactPinV1
    normalization: ArtifactPinV1
    operation: TemporalOperation
    raw_output: CapturedJsonV1
    selected_relationship_ids: tuple[_NonEmptyStr, ...] | None
    world_snapshot: CapturedJsonV1 | None
    limitations: tuple[_AsciiId, ...]
    captured_at: datetime

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_temporal_capture_hash_basis(value)

    @model_validator(mode="after")
    def validate_capture_shape(self) -> "RuntimeTemporalCaptureV1":
        expected = {
            "run": "runtime_evaluation_run",
            "provider_request": "runtime_provider_request",
            "source_graph": "runtime_temporal_source_graph",
            "normalization": "runtime_temporal_normalization",
        }
        for field, kind in expected.items():
            if getattr(self, field).artifact_kind != kind:
                raise ValueError(f"{field} must pin {kind}")
        selection_operations = {"point", "range", "world_snapshot"}
        if (self.selected_relationship_ids is not None) != (
            self.operation in selection_operations
        ):
            raise ValueError("selected_relationship_ids do not match operation")
        if self.selected_relationship_ids is not None:
            if tuple(sorted(set(self.selected_relationship_ids))) != self.selected_relationship_ids:
                raise ValueError("selected_relationship_ids must be sorted unique")
        if (self.world_snapshot is not None) != (self.operation == "world_snapshot"):
            raise ValueError("world_snapshot does not match operation")
        return self


class RuntimeTemporalPathV1(_RuntimeIdentityContract):
    entity_ids: tuple[_NonEmptyStr, ...]
    relationship_ids: tuple[_NonEmptyStr, ...]

    @model_validator(mode="after")
    def validate_path_length(self) -> "RuntimeTemporalPathV1":
        if len(self.entity_ids) != len(self.relationship_ids) + 1:
            raise ValueError("path entity length must equal edge length plus one")
        return self


class RuntimeTemporalOccurrenceV1(_RuntimeIdentityContract):
    occurrence_ref: _NonEmptyStr
    entity_ids: tuple[_NonEmptyStr, ...]
    relationship_ids: tuple[_NonEmptyStr, ...]
    start_time: datetime
    end_time: datetime

    @model_validator(mode="after")
    def validate_occurrence(self) -> "RuntimeTemporalOccurrenceV1":
        if self.start_time > self.end_time:
            raise ValueError("occurrence times are reversed")
        if len(self.entity_ids) != len(self.relationship_ids) + 1:
            raise ValueError("occurrence entity length must equal edge length plus one")
        return self


class RuntimeTemporalPatternV1(_RuntimeIdentityContract):
    pattern_type: Literal["sequence", "cycle"]
    signature: tuple[_NonEmptyStr, ...]
    frequency: _StrictCount
    occurrences: tuple[RuntimeTemporalOccurrenceV1, ...]

    @model_validator(mode="after")
    def validate_frequency(self) -> "RuntimeTemporalPatternV1":
        if self.frequency != len(self.occurrences):
            raise ValueError("pattern frequency must equal occurrence count")
        if any(item.entity_ids != self.signature for item in self.occurrences):
            raise ValueError("pattern signature must equal occurrence entity sequence")
        return self


class RuntimeTemporalMetricsV1(_RuntimeIdentityContract):
    count: _StrictCount
    diversity: _StrictCount
    stability_seconds: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class RuntimeTemporalPayloadV1(_RuntimeIdentityContract):
    operation: TemporalOperation
    entity_ids: tuple[_NonEmptyStr, ...]
    relationship_ids: tuple[_NonEmptyStr, ...]
    paths: tuple[RuntimeTemporalPathV1, ...]
    patterns: tuple[RuntimeTemporalPatternV1, ...]
    metrics: RuntimeTemporalMetricsV1 | None
    world_snapshot: ArtifactPinV1 | None
    normalization: ArtifactPinV1
    proof_state: Literal["query_evidence"]
    proof_pins: tuple[ArtifactPinV1, ...]

    @field_validator("entity_ids", "relationship_ids")
    @classmethod
    def canonical_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("temporal id sets must be sorted unique")
        return value

    @model_validator(mode="after")
    def validate_operation_population(self) -> "RuntimeTemporalPayloadV1":
        if self.normalization.artifact_kind != "runtime_temporal_normalization":
            raise ValueError("normalization must pin runtime_temporal_normalization")
        if self.proof_pins:
            raise ValueError("temporal proof_pins must be empty")
        if self.operation in {"point", "range"}:
            valid = not self.paths and not self.patterns and self.metrics is None and self.world_snapshot is None
        elif self.operation == "path":
            valid = not self.patterns and self.metrics is None and self.world_snapshot is None
        elif self.operation == "pattern":
            valid = not self.paths and self.metrics is None and self.world_snapshot is None
        elif self.operation == "evolution":
            valid = not self.entity_ids and not self.relationship_ids and not self.paths and not self.patterns and self.metrics is not None and self.world_snapshot is None
        else:
            valid = not self.paths and not self.patterns and self.metrics is None and self.world_snapshot is not None
        if not valid:
            raise ValueError("payload population does not match operation")
        if self.world_snapshot is not None and self.world_snapshot.artifact_kind != "runtime_temporal_world_snapshot":
            raise ValueError("world_snapshot must pin runtime_temporal_world_snapshot")
        if self.paths != tuple(
            sorted(self.paths, key=lambda item: (item.entity_ids, item.relationship_ids))
        ):
            raise ValueError("paths must be in canonical order")
        if self.patterns != tuple(
            sorted(self.patterns, key=lambda item: (item.pattern_type, item.signature))
        ):
            raise ValueError("patterns must be in canonical order")
        for pattern in self.patterns:
            expected = tuple(
                sorted(
                    pattern.occurrences,
                    key=lambda item: (
                        item.start_time,
                        item.end_time,
                        item.entity_ids,
                        item.relationship_ids,
                    ),
                )
            )
            if pattern.occurrences != expected:
                raise ValueError("pattern occurrences must be in canonical order")
        return self


class RuntimeTemporalWorldSnapshotV1(_AddressedRuntimeIdentity):
    ARTIFACT_KIND = "runtime_temporal_world_snapshot"
    REF_FIELD = "snapshot_ref"
    HASH_FIELD = "snapshot_hash"
    REF_PREFIX = "runtime-temporal-world-snapshot"

    schema_version: Literal["runtime_temporal_world_snapshot.v1"]
    snapshot_ref: _NonEmptyStr
    snapshot_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    run: ArtifactPinV1
    query: ArtifactPinV1
    source_graph: ArtifactPinV1
    capture: ArtifactPinV1
    graph: CapturedJsonV1

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_temporal_world_snapshot_hash_basis(value)

    @model_validator(mode="after")
    def validate_snapshot_pins(self) -> "RuntimeTemporalWorldSnapshotV1":
        expected = {
            "run": "runtime_evaluation_run",
            "query": "runtime_temporal_query",
            "source_graph": "runtime_temporal_source_graph",
            "capture": "runtime_temporal_capture",
        }
        if any(getattr(self, field).artifact_kind != kind for field, kind in expected.items()):
            raise ValueError("world snapshot pin kind is invalid")
        return self


class RuntimeTemporalSupportedResultV1(_AddressedRuntimeIdentity):
    ARTIFACT_KIND = "runtime_provider_result"
    REF_FIELD = "result_ref"
    HASH_FIELD = "result_hash"
    REF_PREFIX = "runtime-provider-result"

    schema_version: Literal["runtime_provider_result.v1"]
    result_ref: _NonEmptyStr
    result_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    run: ArtifactPinV1
    provider_request: ArtifactPinV1
    provider_output: ArtifactPinV1
    kind: Literal["temporal"]
    capability_state: Literal["supported"]
    payload: RuntimeTemporalPayloadV1
    evidence: tuple[ArtifactPinV1, ...]
    proposal_pins: tuple[ArtifactPinV1, ...]
    limitations: tuple[_AsciiId, ...]
    completed_at: datetime

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_provider_result_hash_basis(value)

    @field_validator("evidence", "proposal_pins")
    @classmethod
    def canonicalize_pin_sets(
        cls, value: tuple[ArtifactPinV1, ...], info
    ) -> tuple[ArtifactPinV1, ...]:
        return _canonical_pin_set(value, info.field_name)

    @model_validator(mode="after")
    def validate_supported(self) -> "RuntimeTemporalSupportedResultV1":
        if self.provider_output.artifact_kind != "runtime_temporal_capture":
            raise ValueError("provider_output must pin runtime_temporal_capture")
        if self.proposal_pins:
            raise ValueError("proposal_pins must be empty")
        return self


class RuntimeTemporalUnavailableResultV1(_AddressedRuntimeIdentity):
    ARTIFACT_KIND = "runtime_provider_result"
    REF_FIELD = "result_ref"
    HASH_FIELD = "result_hash"
    REF_PREFIX = "runtime-provider-result"

    schema_version: Literal["runtime_provider_result.v1"]
    result_ref: _NonEmptyStr
    result_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    run: ArtifactPinV1
    provider_request: ArtifactPinV1
    provider_output: None
    kind: Literal["temporal"]
    capability_state: Literal["unsupported", "dependency_blocked"]
    payload: None
    evidence: tuple[ArtifactPinV1, ...]
    proposal_pins: tuple[ArtifactPinV1, ...]
    limitations: tuple[_AsciiId, ...]
    completed_at: datetime

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_provider_result_hash_basis(value)

    @field_validator("evidence", "proposal_pins")
    @classmethod
    def canonicalize_pin_sets(
        cls, value: tuple[ArtifactPinV1, ...], info
    ) -> tuple[ArtifactPinV1, ...]:
        return _canonical_pin_set(value, info.field_name)

    @model_validator(mode="after")
    def validate_unavailable(self) -> "RuntimeTemporalUnavailableResultV1":
        if self.proposal_pins:
            raise ValueError("proposal_pins must be empty")
        if not self.limitations:
            raise ValueError("unavailable results require a limitation")
        return self


RuntimeProviderResultV1: TypeAlias = Annotated[
    RuntimeTemporalSupportedResultV1 | RuntimeTemporalUnavailableResultV1,
    Field(discriminator="capability_state"),
]


class RuntimeEvaluationStateV1(_AddressedRuntimeIdentity):
    ARTIFACT_KIND = "runtime_evaluation_state"
    REF_FIELD = "state_ref"
    HASH_FIELD = "state_hash"
    REF_PREFIX = "runtime-evaluation-state"

    schema_version: Literal["runtime_evaluation_state.v1"]
    state_ref: _NonEmptyStr
    state_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    run: ArtifactPinV1
    revision: _StrictCount
    predecessor: ArtifactPinV1 | None
    status: Literal["LOADING", "RUNNING", "PARTIAL", "FAILED", "COMPLETE"]
    capture: ArtifactPinV1 | None
    results: tuple[ArtifactPinV1, ...]
    decisions: tuple[ArtifactPinV1, ...] = ()
    diagnostics: tuple[ArtifactPinV1, ...]
    recorded_at: datetime

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_evaluation_state_hash_basis(value)

    @model_validator(mode="after")
    def validate_state_shape(self) -> "RuntimeEvaluationStateV1":
        if self.run.artifact_kind != "runtime_evaluation_run":
            raise ValueError("run must pin runtime_evaluation_run")
        if (self.revision == 0) != (self.predecessor is None):
            raise ValueError("only revision zero has no predecessor")
        if self.predecessor is not None and self.predecessor.artifact_kind != "runtime_evaluation_state":
            raise ValueError("predecessor must pin runtime_evaluation_state")
        if self.capture is not None and self.capture.artifact_kind != "runtime_temporal_capture":
            raise ValueError("capture must pin runtime_temporal_capture")
        if len(self.results) > 1 or any(pin.artifact_kind != "runtime_provider_result" for pin in self.results):
            raise ValueError("state supports one runtime_provider_result")
        if len(self.decisions) > 1 or any(pin.artifact_kind != "decision_record" for pin in self.decisions):
            raise ValueError("state supports one semantic DecisionRecord")
        if any(pin.artifact_kind != "runtime_evaluation_diagnostic" for pin in self.diagnostics):
            raise ValueError("diagnostics must pin runtime_evaluation_diagnostic")
        return self


class RuntimeEvaluationRequestIdentityV1(_RuntimeIdentityContract):
    schema_version: Literal["runtime_evaluation_request_identity.v1"]
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    actor_ref: _NonEmptyStr
    operation: Literal["EVALUATE_RUNTIME_DOCUMENT"]
    request_id: _NonEmptyStr
    context: ArtifactPinV1
    provider_request: ArtifactPinV1
    target: GovernedTargetPinV1
    business_effective_at: datetime
    parent_run: ArtifactPinV1 | None

    @model_validator(mode="after")
    def validate_identity_closure(self) -> "RuntimeEvaluationRequestIdentityV1":
        if self.context.artifact_kind not in {"runtime_working_context", "context_graph"}:
            raise ValueError("context must pin one supported runtime context basis")
        if self.provider_request.artifact_kind != "runtime_provider_request":
            raise ValueError("provider_request must pin runtime_provider_request")
        if self.parent_run is not None and self.parent_run.artifact_kind != "runtime_evaluation_run":
            raise ValueError("parent_run must pin runtime_evaluation_run")
        if (self.organization_id, self.workspace_id) != (
            self.target.organization_id,
            self.target.workspace_id,
        ):
            raise ValueError("identity and target scopes must match")
        return self

    def normalized_request_hash(self) -> str:
        return "sha256:" + content_hash(runtime_evaluation_request_identity_hash_basis(self))


class RuntimeEvaluationRequestOutcomeV1(_AddressedRuntimeIdentity):
    ARTIFACT_KIND = "runtime_evaluation_request_outcome"
    REF_FIELD = "outcome_ref"
    HASH_FIELD = "outcome_hash"
    REF_PREFIX = "runtime-evaluation-request-outcome"

    schema_version: Literal["runtime_evaluation_request_outcome.v1"]
    outcome_ref: _NonEmptyStr
    outcome_hash: _Sha256
    organization_id: _NonEmptyStr
    workspace_id: _NonEmptyStr
    actor_ref: _NonEmptyStr
    operation: Literal["EVALUATE_RUNTIME_DOCUMENT"]
    request_id: _NonEmptyStr
    normalized_request_hash: _Sha256
    run_pin: ArtifactPinV1
    execution_admission_pin: ArtifactPinV1
    state_pin: ArtifactPinV1
    epoch_pin: ArtifactPinV1 | None
    authority_release_receipt_pin: ArtifactPinV1 | None
    result_pins: tuple[ArtifactPinV1, ...]
    recorded_at: datetime

    @classmethod
    def _basis(cls, value: BaseModel | Mapping[str, object]) -> dict[str, object]:
        return runtime_evaluation_request_outcome_hash_basis(value)

    @model_validator(mode="after")
    def validate_outcome_shape(self) -> "RuntimeEvaluationRequestOutcomeV1":
        expected = {
            "run_pin": "runtime_evaluation_run",
            "execution_admission_pin": "runtime_execution_admission",
            "state_pin": "runtime_evaluation_state",
        }
        for field, kind in expected.items():
            if getattr(self, field).artifact_kind != kind:
                raise ValueError(f"{field} must pin {kind}")
        if self.epoch_pin is not None or self.authority_release_receipt_pin is not None:
            raise ValueError("nonpolicy temporal outcome has no policy authority pins")
        if len(self.result_pins) > 1 or any(pin.artifact_kind != "runtime_provider_result" for pin in self.result_pins):
            raise ValueError("outcome supports one runtime_provider_result")
        return self


class RuntimeRuleEvaluationSetV1(BaseModel):
    """One ordered, addressed evaluation for every rule in a compiled plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["runtime_rule_evaluation_set.v1"]
    evaluation_ref: str = Field(min_length=1)
    evaluation_hash: str = Field(pattern=_SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    plan: ArtifactPinV1
    execution_epoch: ArtifactPinV1
    observed_facts: ArtifactPinV1
    target_evidence_refs: tuple[Annotated[str, Field(min_length=1)], ...] = Field(
        min_length=1
    )
    rule_evaluations: tuple[CandidateRuleEvaluationV1, ...] = Field(min_length=1)
    justifications: tuple[RuntimeJustificationV1, ...] = Field(min_length=1)
    counts: dict[_EvaluationOutcome, _StrictCount]
    evaluated_at: datetime

    @model_validator(mode="after")
    def validate_complete_evaluation(self) -> "RuntimeRuleEvaluationSetV1":
        if (
            self.evaluated_at.tzinfo is None
            or self.evaluated_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("evaluated_at must be timezone-aware UTC")
        if self.plan.artifact_kind != "compiled_rule_plan":
            raise ValueError("plan must pin a compiled_rule_plan")
        if self.execution_epoch.artifact_kind != "runtime_execution_epoch":
            raise ValueError("execution_epoch must pin a runtime_execution_epoch")
        if self.observed_facts.artifact_kind != "observed_facts":
            raise ValueError("observed_facts must pin observed_facts")

        outcomes = ("MATCH", "NO_MATCH", "UNKNOWN")
        if set(self.counts) != set(outcomes):
            raise ValueError("counts must contain exactly MATCH, NO_MATCH, and UNKNOWN")
        expected_counts = {
            outcome: sum(
                evaluation.outcome == outcome for evaluation in self.rule_evaluations
            )
            for outcome in outcomes
        }
        if self.counts != expected_counts:
            raise ValueError("counts must equal the complete rule evaluation outcomes")
        if sum(self.counts.values()) != len(self.rule_evaluations):
            raise ValueError("counts must sum to the number of rule evaluations")

        if len(self.rule_evaluations) != len(self.justifications):
            raise ValueError(
                "rule evaluations and justifications must have equal lengths"
            )
        for evaluation, justification in zip(
            self.rule_evaluations, self.justifications, strict=True
        ):
            if (
                justification.result != evaluation.outcome
                or justification.rule_refs != (evaluation.rule_id,)
            ):
                raise ValueError(
                    "each justification must match its result and single matching rule ID"
                )
        return self

    @model_validator(mode="after")
    def validate_identity(self) -> "RuntimeRuleEvaluationSetV1":
        digest = content_hash(
            self.model_dump(mode="json", exclude={"evaluation_ref", "evaluation_hash"})
        )
        if (self.evaluation_ref, self.evaluation_hash) != (
            f"runtime-rule-evaluation-set:{digest}",
            f"sha256:{digest}",
        ):
            raise ValueError("rule evaluation set identity does not bind exact content")
        return self

    @classmethod
    def create(cls, **values: object) -> "RuntimeRuleEvaluationSetV1":
        provisional = cls.model_construct(
            **values,
            evaluation_ref="pending",
            evaluation_hash="sha256:" + "0" * 64,
        )
        digest = content_hash(
            provisional.model_dump(
                mode="json", exclude={"evaluation_ref", "evaluation_hash"}
            )
        )
        return cls(
            **values,
            evaluation_ref=f"runtime-rule-evaluation-set:{digest}",
            evaluation_hash=f"sha256:{digest}",
        )


__all__ = [
    "GovernedTargetPinV1",
    "RuntimeEvaluationDiagnosticV1",
    "RuntimeEvaluationRequestIdentityV1",
    "RuntimeEvaluationRequestOutcomeV1",
    "RuntimeEvaluationRunV1",
    "RuntimeEvaluationStateV1",
    "RuntimeExecutionAdmissionV1",
    "RuntimeProviderResultV1",
    "RuntimeRuleEvaluationSetV1",
    "RuntimeTemporalCaptureV1",
    "RuntimeTemporalDependencyV1",
    "RuntimeTemporalDependencyV2",
    "RuntimeTemporalProviderProfileV2",
    "runtime_temporal_profile_model",
    "RuntimeTemporalMetricsV1",
    "RuntimeTemporalOccurrenceV1",
    "RuntimeTemporalPathV1",
    "RuntimeTemporalPatternV1",
    "RuntimeTemporalPayloadV1",
    "RuntimeTemporalProviderConfigV1",
    "RuntimeTemporalProviderDescriptorV1",
    "RuntimeTemporalProviderProfileV1",
    "RuntimeTemporalProviderRequestV1",
    "RuntimeTemporalRequestSettingsV1",
    "RuntimeTemporalSupportedResultV1",
    "RuntimeTemporalUnavailableResultV1",
    "RuntimeTemporalWorldSnapshotV1",
    "TemporalOperation",
    "TemporalQueryV1",
    "runtime_evaluation_request_identity_hash_basis",
    "runtime_evaluation_request_outcome_hash_basis",
    "runtime_evaluation_state_hash_basis",
    "runtime_execution_admission_hash_basis",
    "runtime_evaluation_diagnostic_hash_basis",
    "runtime_evaluation_run_hash_basis",
    "runtime_provider_descriptor_hash_basis",
    "runtime_provider_profile_hash_basis",
    "runtime_provider_request_hash_basis",
    "runtime_provider_result_hash_basis",
    "runtime_temporal_capture_hash_basis",
    "runtime_temporal_query_hash_basis",
    "runtime_temporal_world_snapshot_hash_basis",
]
