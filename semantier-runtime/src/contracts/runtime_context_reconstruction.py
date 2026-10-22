"""Strict pure contracts for deterministic runtime-context reconstruction."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import gcd
from types import MappingProxyType
from typing import Annotated, Callable, ClassVar, Generic, Literal, Mapping, TypeAlias, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_working_context import (
    RuntimeContextElementRole,
    RuntimeContextTransitionProposalV1,
    RuntimeContextTransitionProposalV2,
    RuntimeContextTransitionReceiptV1,
    RuntimeContextTransitionReceiptV2,
    RuntimeContextTransitionV1,
    RuntimeContextTransitionV2,
    RuntimeTaskContextV1,
    RuntimeTaskContextV2,
    RuntimeWorldScopeV1,
)
from eos.artifact_hashing import canonical_json as encode_canonical_json
from eos.artifact_hashing import content_hash
from eos.artifact_hashing import sha256_hex


NonEmptyStr = Annotated[str, Field(strict=True, min_length=1)]
AsciiId = Annotated[
    str,
    Field(strict=True, min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$"),
]
Sha256 = Annotated[str, Field(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]
StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
StrictPositiveInt = Annotated[int, Field(strict=True, gt=0)]
PartsPerMillion = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]


def _strict_datetime_input(value: object) -> object:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string or datetime")
    try:
        float(value)
    except ValueError:
        return value
    raise ValueError("timestamp must not be a numeric epoch string")


UtcDateTime = Annotated[datetime, BeforeValidator(_strict_datetime_input)]


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        serialized = value.isoformat()
        return serialized.replace("+00:00", "Z") if value.utcoffset() == timedelta(0) else serialized
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _without(value: Mapping[str, object], *excluded: str) -> dict[str, object]:
    return {
        key: _json_value(item)
        for key, item in value.items()
        if key not in excluded
    }


def _pin_key(value: object) -> tuple[str, str, str]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        raise ValueError("artifact pin must be a mapping or model")
    try:
        return (
            str(value["artifact_kind"]),
            str(value["artifact_ref"]),
            str(value["artifact_hash"]),
        )
    except KeyError as exc:
        raise ValueError(f"artifact pin is missing {exc.args[0]}") from exc


def _ordered_unique_pins(value: object) -> object:
    if not isinstance(value, (list, tuple)):
        return value
    ordered = sorted(value, key=_pin_key)
    keys = [_pin_key(item) for item in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("artifact pin set contains duplicate identities")
    refs = [(kind, ref) for kind, ref, _ in keys]
    if len(refs) != len(set(refs)):
        raise ValueError("artifact pin set contains one ref with conflicting hashes")
    return tuple(ordered)


def _ordered_unique(
    value: object,
    *,
    key: Callable[[object], object],
    label: str,
) -> object:
    if not isinstance(value, (list, tuple)):
        return value
    ordered = sorted(value, key=key)
    keys = [key(item) for item in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} contains duplicate identities")
    return tuple(ordered)


def _field(value: object, name: str, default: object = "") -> object:
    if isinstance(value, BaseModel):
        return getattr(value, name, default)
    if isinstance(value, Mapping):
        return value.get(name, default)
    return default


def _verify_identity(
    *,
    prefix: str,
    artifact_ref: str,
    artifact_hash: str,
    basis: Mapping[str, object],
    label: str,
) -> None:
    digest = content_hash(dict(basis))
    if (artifact_ref, artifact_hash) != (
        f"{prefix}:{digest}",
        f"sha256:{digest}",
    ):
        raise ValueError(f"{label} identity does not bind full canonical content")


def bind_envelope_identity(
    value: Mapping[str, object],
    *,
    prefix: str,
    ref_field: str,
    hash_field: str,
) -> dict[str, object]:
    """Return a copy with the common deterministic ref/hash pair populated."""
    result = dict(value)
    basis = _without(result, ref_field, hash_field)
    digest = content_hash(basis)
    result[ref_field] = f"{prefix}:{digest}"
    result[hash_field] = f"sha256:{digest}"
    return result


def _require_pin_kind(value: ArtifactPinV1, kind: str, field: str) -> ArtifactPinV1:
    if value.artifact_kind != kind:
        raise ValueError(f"{field} must pin artifact kind {kind}")
    return value


def _validate_scope_policy_selection(
    scope_policy: ArtifactPinV1,
    semantic_pins: tuple[ArtifactPinV1, ...],
) -> None:
    if scope_policy.artifact_kind != "runtime_context_scope_policy":
        raise ValueError("scope_policy must pin runtime_context_scope_policy")
    scope_policies = tuple(
        pin
        for pin in semantic_pins
        if pin.artifact_kind == "runtime_context_scope_policy"
    )
    if len(scope_policies) != 1:
        raise ValueError("semantic_pins must contain exactly one scope-policy pin")
    if _pin_key(scope_policies[0]) != _pin_key(scope_policy):
        raise ValueError("scope_policy must equal the selected scope-policy pin")


class _ReconstructionContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def persisted_timestamps_are_utc(self) -> "_ReconstructionContract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime):
                if value.tzinfo is None or value.utcoffset() != timedelta(0):
                    raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class _WorkspaceIdentityContract(_ReconstructionContract):
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr

    identity_prefix: ClassVar[str]
    identity_ref_field: ClassVar[str]
    identity_hash_field: ClassVar[str]
    identity_label: ClassVar[str]

    @model_validator(mode="after")
    def identity_binds_full_content(self) -> "_WorkspaceIdentityContract":
        ref_field = type(self).identity_ref_field
        hash_field = type(self).identity_hash_field
        _verify_identity(
            prefix=type(self).identity_prefix,
            artifact_ref=getattr(self, ref_field),
            artifact_hash=getattr(self, hash_field),
            basis=_without(self.model_dump(mode="json"), ref_field, hash_field),
            label=type(self).identity_label,
        )
        return self


class CapturedJsonV1(_ReconstructionContract):
    media_type: Literal["application/json"]
    canonical_json: str
    bytes_hash: Sha256

    @model_validator(mode="after")
    def validate_canonical_bytes(self) -> "CapturedJsonV1":
        def reject_nonfinite(token: str) -> object:
            raise ValueError(f"non-finite JSON number {token} is forbidden")

        try:
            decoded = json.loads(
                self.canonical_json,
                parse_constant=reject_nonfinite,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("canonical_json must contain finite JSON") from exc
        if encode_canonical_json(decoded) != self.canonical_json:
            raise ValueError("canonical_json must be the exact canonical encoding")
        if self.bytes_hash != f"sha256:{sha256_hex(self.canonical_json)}":
            raise ValueError("bytes_hash must bind the exact canonical JSON UTF-8 bytes")
        return self


class SourceObjectV1(_ReconstructionContract):
    source: ArtifactPinV1
    object_ref: NonEmptyStr
    object_kind: AsciiId
    locator: str = Field(strict=True)


class TimeExtentV1(_ReconstructionContract):
    knowledge: Literal["known", "unknown"]
    valid_from: UtcDateTime | None
    valid_to: UtcDateTime | None
    bounds: Literal["closed", "left_closed_right_open", "point", "unknown"]
    recorded_at: UtcDateTime | None

    @model_validator(mode="after")
    def validate_extent(self) -> "TimeExtentV1":
        if self.knowledge == "unknown":
            if self.bounds != "unknown" or self.valid_from is not None or self.valid_to is not None:
                raise ValueError("unknown time extent must not assume endpoints")
            return self
        if self.bounds == "unknown":
            raise ValueError("known time extent requires known bounds")
        if self.bounds == "point":
            if self.valid_from is None or self.valid_to is None or self.valid_from != self.valid_to:
                raise ValueError("point time extent requires equal nonnull endpoints")
        elif (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_from >= self.valid_to
        ):
            raise ValueError("finite time extent endpoints must be ordered")
        return self


class PolicySectionPinV1(_ReconstructionContract):
    container: ArtifactPinV1
    section: Literal["weighting"]
    section_version: AsciiId
    section_hash: Sha256


class ResourceBoundsV1(_ReconstructionContract):
    max_candidates: StrictPositiveInt
    max_items: StrictPositiveInt
    max_payload_bytes: StrictPositiveInt
    max_context_tokens: StrictPositiveInt
    max_event_steps: StrictPositiveInt
    max_closure_bytes: StrictPositiveInt
    max_retrieval_latency_ms: StrictPositiveInt


DiagnosticCode: TypeAlias = Literal[
    "SOURCE_UNAVAILABLE",
    "PROVIDER_UNAVAILABLE",
    "TIMEOUT",
    "TRUNCATED",
    "MISSING_PROVENANCE",
    "UNKNOWN_BASELINE",
    "UNKNOWN_ORDER",
    "UNSUPPORTED_MODE",
    "EXCLUDED_BY_BUDGET",
]


class CandidateDiagnosticV1(_ReconstructionContract):
    code: DiagnosticCode
    subject: ArtifactPinV1 | None
    detail: str = Field(strict=True)


class ElementBindingBodyV1(_ReconstructionContract):
    object: SourceObjectV1
    role: RuntimeContextElementRole
    derived: StrictBool
    evidence: tuple[ArtifactPinV1, ...]
    derivation: ArtifactPinV1 | None

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_derivation(self) -> "ElementBindingBodyV1":
        if self.derived and (not self.evidence or self.derivation is None):
            raise ValueError("derived element bodies require evidence and derivation")
        if not self.derived and self.derivation is not None:
            raise ValueError("non-derived element bodies must not carry derivation")
        return self


class FormationElementBindingV1(_ReconstructionContract):
    reconstruction: ArtifactPinV1
    item_id: AsciiId
    body: ElementBindingBodyV1

    @field_validator("reconstruction")
    @classmethod
    def validate_reconstruction_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "reconstructed_context", "reconstruction")


class WorldQueryRequestV1(_ReconstructionContract):
    source_graph: ArtifactPinV1
    view: Literal["non_temporal", "temporal"]


class SemanticSelectionRequestV1(_ReconstructionContract):
    ontology: ArtifactPinV1
    tag_ontology: ArtifactPinV1
    knowledge: ArtifactPinV1
    constraints: ArtifactPinV1
    precedence: ArtifactPinV1
    resolution: ArtifactPinV1
    projection: ArtifactPinV1
    scope_policy: ArtifactPinV1
    semantic_pins: tuple[ArtifactPinV1, ...]

    @field_validator("semantic_pins", mode="before")
    @classmethod
    def normalize_semantic_pins(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_named_roles(self) -> "SemanticSelectionRequestV1":
        selected = {_pin_key(pin) for pin in self.semantic_pins}
        named = (
            self.ontology,
            self.tag_ontology,
            self.knowledge,
            self.constraints,
            self.precedence,
            self.resolution,
            self.projection,
            self.scope_policy,
        )
        if any(_pin_key(pin) not in selected for pin in named):
            raise ValueError("every named semantic role pin must occur in semantic_pins")
        _validate_scope_policy_selection(self.scope_policy, self.semantic_pins)
        return self


class TemporalCurrentRequestV1(_ReconstructionContract):
    mode: Literal["current"]


class TemporalPointRequestV1(_ReconstructionContract):
    mode: Literal["point_in_time"]
    valid_at: UtcDateTime


class TemporalIntervalRequestV1(_ReconstructionContract):
    mode: Literal["interval"]
    start: UtcDateTime
    end: UtcDateTime
    bounds: Literal["closed"]

    @model_validator(mode="after")
    def validate_window(self) -> "TemporalIntervalRequestV1":
        if self.start >= self.end:
            raise ValueError("temporal interval start must precede end")
        return self


class TemporalPriorStateRequestV1(_ReconstructionContract):
    mode: Literal["prior_state"]
    before_event: SourceObjectV1
    ordering: ArtifactPinV1
    baseline: ArtifactPinV1 | None
    start: UtcDateTime
    end: UtcDateTime

    @model_validator(mode="after")
    def validate_window(self) -> "TemporalPriorStateRequestV1":
        if self.start >= self.end:
            raise ValueError("prior-state start must precede end")
        return self


class TemporalCausalWindowRequestV1(_ReconstructionContract):
    mode: Literal["causal_window"]
    anchor_event: SourceObjectV1
    ordering: ArtifactPinV1
    start: UtcDateTime
    end: UtcDateTime
    bounds: Literal["closed"]

    @model_validator(mode="after")
    def validate_window(self) -> "TemporalCausalWindowRequestV1":
        if self.start >= self.end:
            raise ValueError("causal-window start must precede end")
        return self


class TemporalInheritOriginRequestV1(_ReconstructionContract):
    mode: Literal["inherit_origin"]


TemporalRequestV1: TypeAlias = Annotated[
    TemporalCurrentRequestV1
    | TemporalPointRequestV1
    | TemporalIntervalRequestV1
    | TemporalPriorStateRequestV1
    | TemporalCausalWindowRequestV1
    | TemporalInheritOriginRequestV1,
    Field(discriminator="mode"),
]


class RequestedCurrentInstantV1(_ReconstructionContract):
    kind: Literal["current"]


class RequestedExplicitInstantV1(_ReconstructionContract):
    kind: Literal["at"]
    at: UtcDateTime


RequestedInstantV1: TypeAlias = Annotated[
    RequestedCurrentInstantV1 | RequestedExplicitInstantV1,
    Field(discriminator="kind"),
]


class AvailabilityRequestV1(_ReconstructionContract):
    mode: Literal["as_known", "counterfactual"]
    known_at: RequestedInstantV1
    governance_at: RequestedInstantV1


class ReconstructionQueryRequestV1(_ReconstructionContract):
    schema_version: Literal["reconstruction_query_request.v1"]
    purpose: Literal["create", "branch", "reasoning", "view"]
    origin: ArtifactPinV1 | None
    goal: ArtifactPinV1 | None = None
    world: WorldQueryRequestV1 | None = None
    semantic_selection: SemanticSelectionRequestV1 | None = None
    source_pins: tuple[ArtifactPinV1, ...]
    operational_inputs: tuple[ArtifactPinV1, ...]
    hypotheses: tuple[SourceObjectV1, ...]
    unresolved_questions: tuple[SourceObjectV1, ...]
    temporal: TemporalRequestV1
    availability: AvailabilityRequestV1
    admissibility_policy: ArtifactPinV1
    operator: ArtifactPinV1
    weighting: PolicySectionPinV1
    providers: tuple[ArtifactPinV1, ...]
    profile: AsciiId
    intent: Literal["exploration", "historical_analysis", "comparison", "projection"]
    bounds: ResourceBoundsV1

    @model_validator(mode="before")
    @classmethod
    def validate_override_presence(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        origin = value.get("origin")
        for field in ("goal", "world", "semantic_selection"):
            if field in value and value[field] is None:
                raise ValueError(f"{field} cannot be explicit null")
            if field not in value and origin is None:
                raise ValueError(f"{field} may be omitted only with origin")
        return value

    @field_validator("source_pins", "operational_inputs", "providers", mode="before")
    @classmethod
    def normalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @field_validator("hypotheses", "unresolved_questions", mode="before")
    @classmethod
    def normalize_source_object_sets(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: (
                _pin_key(_field(item, "source")),
                str(_field(item, "object_kind")),
                str(_field(item, "object_ref")),
                str(_field(item, "locator")),
            ),
            label="source object set",
        )

    @model_validator(mode="after")
    def validate_request_relationships(self) -> "ReconstructionQueryRequestV1":
        if not self.providers:
            raise ValueError("providers must be nonempty")
        if _pin_key(self.weighting.container) != _pin_key(self.admissibility_policy):
            raise ValueError("weighting container must equal admissibility_policy")
        if self.availability.mode == "counterfactual" and self.intent != "comparison":
            raise ValueError("counterfactual availability requires comparison intent")
        if self.purpose in {"branch", "reasoning"} and self.origin is None:
            raise ValueError(f"{self.purpose} requires origin")
        if self.purpose == "create" and self.origin is not None:
            raise ValueError("create requires null origin")
        if self.temporal.mode == "inherit_origin" and self.origin is None:
            raise ValueError("inherit_origin temporal mode requires origin")
        return self


T = TypeVar("T")


class InheritOverrideV1(_ReconstructionContract):
    mode: Literal["inherit"]


class ReplaceOverrideV1(_ReconstructionContract, Generic[T]):
    mode: Literal["replace"]
    value: T


GoalOverrideV1: TypeAlias = Annotated[
    InheritOverrideV1 | ReplaceOverrideV1[ArtifactPinV1],
    Field(discriminator="mode"),
]
WorldOverrideV1: TypeAlias = Annotated[
    InheritOverrideV1 | ReplaceOverrideV1[WorldQueryRequestV1],
    Field(discriminator="mode"),
]
SemanticSelectionOverrideV1: TypeAlias = Annotated[
    InheritOverrideV1 | ReplaceOverrideV1[SemanticSelectionRequestV1],
    Field(discriminator="mode"),
]


class NormalizedReconstructionRequestV1(_ReconstructionContract):
    schema_version: Literal["reconstruction_query_request.v1"]
    purpose: Literal["create", "branch", "reasoning", "view"]
    origin: ArtifactPinV1 | None
    goal: GoalOverrideV1
    world: WorldOverrideV1
    semantic_selection: SemanticSelectionOverrideV1
    source_pins: tuple[ArtifactPinV1, ...]
    operational_inputs: tuple[ArtifactPinV1, ...]
    hypotheses: tuple[SourceObjectV1, ...]
    unresolved_questions: tuple[SourceObjectV1, ...]
    temporal: TemporalRequestV1
    availability: AvailabilityRequestV1
    admissibility_policy: ArtifactPinV1
    operator: ArtifactPinV1
    weighting: PolicySectionPinV1
    providers: tuple[ArtifactPinV1, ...]
    profile: AsciiId
    intent: Literal["exploration", "historical_analysis", "comparison", "projection"]
    bounds: ResourceBoundsV1

    @field_validator("source_pins", "operational_inputs", "providers", mode="before")
    @classmethod
    def normalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @field_validator("hypotheses", "unresolved_questions", mode="before")
    @classmethod
    def normalize_source_objects(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: (
                _pin_key(_field(item, "source")),
                str(_field(item, "object_kind")),
                str(_field(item, "object_ref")),
                str(_field(item, "locator")),
            ),
            label="source object set",
        )

    @model_validator(mode="after")
    def validate_normalized_relationships(self) -> "NormalizedReconstructionRequestV1":
        if not self.providers:
            raise ValueError("providers must be nonempty")
        if _pin_key(self.weighting.container) != _pin_key(self.admissibility_policy):
            raise ValueError("weighting container must equal admissibility_policy")
        if self.availability.mode == "counterfactual" and self.intent != "comparison":
            raise ValueError("counterfactual availability requires comparison intent")
        if self.purpose in {"branch", "reasoning"} and self.origin is None:
            raise ValueError(f"{self.purpose} requires origin")
        if self.origin is None and any(
            override.mode == "inherit"
            for override in (self.goal, self.world, self.semantic_selection)
        ):
            raise ValueError("request cannot inherit without an origin")
        if self.purpose == "create":
            if self.origin is not None:
                raise ValueError("create requires null origin")
        if self.temporal.mode == "inherit_origin" and self.origin is None:
            raise ValueError("inherit_origin temporal mode requires origin")
        return self


def normalize_reconstruction_request(
    request: ReconstructionQueryRequestV1 | Mapping[str, object],
) -> NormalizedReconstructionRequestV1:
    public = (
        request
        if isinstance(request, ReconstructionQueryRequestV1)
        else ReconstructionQueryRequestV1.model_validate(request)
    )
    fields_set = public.model_fields_set
    payload = public.model_dump(mode="python")
    for field in ("goal", "world", "semantic_selection"):
        value = payload[field]
        payload[field] = (
            {"mode": "replace", "value": value}
            if field in fields_set
            else {"mode": "inherit"}
        )
    return NormalizedReconstructionRequestV1.model_validate(payload)


class TemporalCurrentQueryV1(_ReconstructionContract):
    mode: Literal["current"]
    valid_at: UtcDateTime


class TemporalPointQueryV1(_ReconstructionContract):
    mode: Literal["point_in_time"]
    valid_at: UtcDateTime


class TemporalIntervalQueryV1(TemporalIntervalRequestV1):
    pass


class TemporalPriorStateQueryV1(TemporalPriorStateRequestV1):
    pass


class TemporalCausalWindowQueryV1(TemporalCausalWindowRequestV1):
    pass


TemporalQueryV1: TypeAlias = Annotated[
    TemporalCurrentQueryV1
    | TemporalPointQueryV1
    | TemporalIntervalQueryV1
    | TemporalPriorStateQueryV1
    | TemporalCausalWindowQueryV1,
    Field(discriminator="mode"),
]


class AvailabilityBasisV1(_ReconstructionContract):
    mode: Literal["as_known", "counterfactual"]
    known_at: UtcDateTime
    governance_at: UtcDateTime
    governing_evidence: tuple[ArtifactPinV1, ...]

    @field_validator("governing_evidence", mode="before")
    @classmethod
    def normalize_governing_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)


class ReconstructionQueryV1(_WorkspaceIdentityContract):
    schema_version: Literal["reconstruction_query.v1"]
    query_ref: NonEmptyStr
    query_hash: Sha256
    session_ref: NonEmptyStr
    principal_id: NonEmptyStr
    authority_domain: NonEmptyStr
    actor_evidence: tuple[ArtifactPinV1, ...]
    goal: ArtifactPinV1
    world: RuntimeWorldScopeV1
    source_pins: tuple[ArtifactPinV1, ...]
    operational_inputs: tuple[ArtifactPinV1, ...]
    reasoning_origin: ArtifactPinV1 | None
    hypotheses: tuple[SourceObjectV1, ...]
    unresolved_questions: tuple[SourceObjectV1, ...]
    temporal: TemporalQueryV1
    availability: AvailabilityBasisV1
    ontology: ArtifactPinV1
    tag_ontology: ArtifactPinV1
    knowledge: ArtifactPinV1
    constraints: ArtifactPinV1
    precedence: ArtifactPinV1
    resolution: ArtifactPinV1
    projection: ArtifactPinV1
    scope_policy: ArtifactPinV1
    admissibility_policy: ArtifactPinV1
    operator: ArtifactPinV1
    semantic_pins: tuple[ArtifactPinV1, ...]
    weighting: PolicySectionPinV1
    providers: tuple[ArtifactPinV1, ...]
    verifier_registry: ArtifactPinV1
    profile: AsciiId
    intent: Literal["exploration", "historical_analysis", "comparison", "projection"]
    bounds: ResourceBoundsV1

    identity_prefix = "reconstruction-query"
    identity_ref_field = "query_ref"
    identity_hash_field = "query_hash"
    identity_label = "reconstruction query"

    @field_validator(
        "actor_evidence",
        "source_pins",
        "operational_inputs",
        "semantic_pins",
        "providers",
        mode="before",
    )
    @classmethod
    def normalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @field_validator("hypotheses", "unresolved_questions", mode="before")
    @classmethod
    def normalize_source_objects(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: (
                _pin_key(_field(item, "source")),
                str(_field(item, "object_kind")),
                str(_field(item, "object_ref")),
                str(_field(item, "locator")),
            ),
            label="source object set",
        )

    @model_validator(mode="after")
    def validate_query_relationships(self) -> "ReconstructionQueryV1":
        if not self.actor_evidence:
            raise ValueError("trusted reconstruction query requires actor_evidence")
        if not self.providers:
            raise ValueError("reconstruction query providers must be nonempty")
        if _pin_key(self.weighting.container) != _pin_key(self.admissibility_policy):
            raise ValueError("weighting container must equal admissibility_policy")
        named = (
            self.ontology,
            self.tag_ontology,
            self.knowledge,
            self.constraints,
            self.precedence,
            self.resolution,
            self.projection,
            self.scope_policy,
        )
        selected = {_pin_key(pin) for pin in self.semantic_pins}
        if any(_pin_key(pin) not in selected for pin in named):
            raise ValueError("named semantic role pins must occur in semantic_pins")
        _validate_scope_policy_selection(self.scope_policy, self.semantic_pins)
        if self.availability.mode == "counterfactual" and self.intent != "comparison":
            raise ValueError("counterfactual availability requires comparison intent")
        return self


class VerifierSelectorV1(_ReconstructionContract):
    registry: ArtifactPinV1
    entry_id: AsciiId

    @field_validator("registry")
    @classmethod
    def validate_registry_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "packaged_verifier_registry", "registry")


class CapturedSourceV1(_ReconstructionContract):
    source: ArtifactPinV1
    verifier: VerifierSelectorV1
    payload: CapturedJsonV1
    dependencies: tuple[ArtifactPinV1, ...]

    @field_validator("dependencies", mode="before")
    @classmethod
    def normalize_dependencies(cls, value: object) -> object:
        return _ordered_unique_pins(value)


def _captured_source_key(value: object) -> tuple[str, str, str]:
    return _pin_key(_field(value, "source"))


def _ordered_unique_captured_sources(value: object, *, label: str) -> object:
    ordered = _ordered_unique(value, key=_captured_source_key, label=label)
    if not isinstance(ordered, tuple):
        return ordered
    refs = [key[:2] for key in map(_captured_source_key, ordered)]
    if len(refs) != len(set(refs)):
        raise ValueError(f"{label} contains one source ref with conflicting hashes")
    return ordered


class CandidateCaptureV1(_WorkspaceIdentityContract):
    schema_version: Literal["candidate_capture.v1"]
    capture_ref: NonEmptyStr
    capture_hash: Sha256
    query: ArtifactPinV1
    provider: ArtifactPinV1
    output: CapturedJsonV1
    sources: tuple[CapturedSourceV1, ...]
    diagnostics: tuple[CandidateDiagnosticV1, ...]
    completeness: Literal["complete", "partial", "unavailable"]

    identity_prefix = "candidate-capture"
    identity_ref_field = "capture_ref"
    identity_hash_field = "capture_hash"
    identity_label = "candidate capture"

    @field_validator("query")
    @classmethod
    def validate_query_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "reconstruction_query", "query")

    @field_validator("sources", mode="before")
    @classmethod
    def normalize_sources(cls, value: object) -> object:
        return _ordered_unique_captured_sources(value, label="captured sources")


class AssertedGovernanceV1(_ReconstructionContract):
    tier: Literal["T1", "T2", "T3", "T4", "T5", "T6"] | None
    domain: str | None
    role: Literal["binding", "interpretive", "empirical", "contextual"] | None
    state: str | None


class MemoryCandidateV1(_WorkspaceIdentityContract):
    schema_version: Literal["memory_candidate.v1"]
    candidate_ref: NonEmptyStr
    candidate_hash: Sha256
    query: ArtifactPinV1
    capture: ArtifactPinV1
    provider: ArtifactPinV1
    clock: Literal["fast", "middle", "slow"]
    item_type: Literal[
        "event",
        "observation",
        "entity_state",
        "pattern",
        "hypothesis",
        "knowledge_artifact",
        "policy",
        "constraint",
        "opaque_context",
    ]
    payload: CapturedJsonV1
    source_objects: tuple[SourceObjectV1, ...]
    temporal: TimeExtentV1
    asserted_governance: AssertedGovernanceV1
    evidence: tuple[ArtifactPinV1, ...]
    provider_relevance_ppm: PartsPerMillion | None

    identity_prefix = "memory-candidate"
    identity_ref_field = "candidate_ref"
    identity_hash_field = "candidate_hash"
    identity_label = "memory candidate"

    @field_validator("query")
    @classmethod
    def validate_query_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "reconstruction_query", "query")

    @field_validator("capture")
    @classmethod
    def validate_capture_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "candidate_capture", "capture")

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @field_validator("source_objects", mode="before")
    @classmethod
    def normalize_source_objects(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: (
                _pin_key(_field(item, "source")),
                str(_field(item, "object_kind")),
                str(_field(item, "object_ref")),
                str(_field(item, "locator")),
            ),
            label="candidate source objects",
        )


class MemoryCandidateSetV1(_WorkspaceIdentityContract):
    schema_version: Literal["memory_candidate_set.v1"]
    candidate_set_ref: NonEmptyStr
    candidate_set_hash: Sha256
    query: ArtifactPinV1
    captures: tuple[ArtifactPinV1, ...]
    candidates: tuple[ArtifactPinV1, ...]
    ordered_event_candidates: tuple[ArtifactPinV1, ...]
    diagnostics: tuple[CandidateDiagnosticV1, ...]
    completeness: Literal["complete", "partial", "unavailable"]

    identity_prefix = "memory-candidate-set"
    identity_ref_field = "candidate_set_ref"
    identity_hash_field = "candidate_set_hash"
    identity_label = "memory candidate set"

    @field_validator("query")
    @classmethod
    def validate_query_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "reconstruction_query", "query")

    @field_validator("captures", "candidates", mode="before")
    @classmethod
    def normalize_pin_sets(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @field_validator("ordered_event_candidates", mode="before")
    @classmethod
    def reject_duplicate_event_candidates(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        keys = [_pin_key(item) for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError("ordered event candidates contain duplicate identities")
        return tuple(value)


class SourceEligibilityV1(_ReconstructionContract):
    source: ArtifactPinV1
    object_ref: str | None
    tier: Literal["T1", "T2", "T3", "T4", "T5", "T6"] | None
    domain: str | None
    permitted_role: Literal["binding", "interpretive", "empirical", "contextual"] | None
    approved: StrictBool
    active: StrictBool
    effective: StrictBool
    membership_known: StrictBool
    ontology_compatible: StrictBool | None
    tag_compatible: StrictBool | None
    evidence: tuple[ArtifactPinV1, ...]

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)


class ReconstructionGovernanceCaptureV1(_WorkspaceIdentityContract):
    schema_version: Literal["reconstruction_governance_capture.v1"]
    governance_ref: NonEmptyStr
    governance_hash: Sha256
    query: ArtifactPinV1
    scope_policy: ArtifactPinV1
    admissibility_policy: ArtifactPinV1
    weighting: PolicySectionPinV1
    mandatory_sources: tuple[CapturedSourceV1, ...]
    eligibility: tuple[SourceEligibilityV1, ...]
    authority_evidence: tuple[CapturedSourceV1, ...]

    identity_prefix = "reconstruction-governance"
    identity_ref_field = "governance_ref"
    identity_hash_field = "governance_hash"
    identity_label = "reconstruction governance capture"

    @field_validator("query")
    @classmethod
    def validate_query_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "reconstruction_query", "query")

    @field_validator("mandatory_sources", "authority_evidence", mode="before")
    @classmethod
    def normalize_sources(cls, value: object) -> object:
        return _ordered_unique_captured_sources(value, label="captured sources")

    @field_validator("eligibility", mode="before")
    @classmethod
    def normalize_eligibility(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: (
                _pin_key(_field(item, "source")),
                str(_field(item, "object_ref")),
            ),
            label="source eligibility",
        )

    @model_validator(mode="after")
    def validate_policy_binding(self) -> "ReconstructionGovernanceCaptureV1":
        if _pin_key(self.weighting.container) != _pin_key(self.admissibility_policy):
            raise ValueError("weighting container must equal admissibility_policy")
        return self


def reconstructed_item_hash_basis(value: Mapping[str, object]) -> dict[str, object]:
    return _without(value, "item_id")


def reconstructed_item_id(value: Mapping[str, object]) -> str:
    return f"item_{content_hash(reconstructed_item_hash_basis(value))}"


class ReconstructedItemV1(_ReconstructionContract):
    item_id: AsciiId
    clock: Literal["fast", "middle", "slow"]
    representation: Literal[
        "source_value",
        "entity_state",
        "pattern",
        "interpretation",
        "constraint_candidate",
        "opaque_context",
    ]
    payload: CapturedJsonV1
    candidate_dependencies: tuple[ArtifactPinV1, ...]
    source_dependencies: tuple[SourceObjectV1, ...]
    temporal: TimeExtentV1
    salience_numerator: StrictNonNegativeInt
    salience_denominator: StrictPositiveInt
    element: ElementBindingBodyV1 | None

    @model_validator(mode="before")
    @classmethod
    def normalize_rational(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        result = dict(value)
        numerator = result.get("salience_numerator")
        denominator = result.get("salience_denominator")
        if (
            isinstance(numerator, int)
            and not isinstance(numerator, bool)
            and isinstance(denominator, int)
            and not isinstance(denominator, bool)
            and numerator >= 0
            and denominator > 0
        ):
            divisor = gcd(numerator, denominator)
            result["salience_numerator"] = numerator // divisor
            result["salience_denominator"] = denominator // divisor
        return result

    @field_validator("candidate_dependencies", mode="before")
    @classmethod
    def normalize_candidate_dependencies(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @field_validator("source_dependencies", mode="before")
    @classmethod
    def normalize_source_dependencies(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: (
                _pin_key(_field(item, "source")),
                str(_field(item, "object_kind")),
                str(_field(item, "object_ref")),
                str(_field(item, "locator")),
            ),
            label="item source dependencies",
        )

    @model_validator(mode="after")
    def validate_item_identity(self) -> "ReconstructedItemV1":
        if not self.candidate_dependencies:
            raise ValueError("reconstructed item requires candidate dependencies")
        expected = reconstructed_item_id(self.model_dump(mode="json"))
        if self.item_id != expected:
            raise ValueError("item_id does not bind the exact reconstructed item body")
        return self


class ReconstructedTransitionV1(_ReconstructionContract):
    event: SourceObjectV1
    before_item_id: str | None
    after_item_id: str | None
    sequence: StrictNonNegativeInt


def reconstruction_conflict_hash_basis(value: Mapping[str, object]) -> dict[str, object]:
    return _without(value, "conflict_id")


def reconstruction_conflict_id(value: Mapping[str, object]) -> str:
    return f"conflict_{content_hash(reconstruction_conflict_hash_basis(value))}"


class ReconstructionConflictV1(_ReconstructionContract):
    conflict_id: AsciiId
    item_ids: tuple[str, ...]
    kind: Literal["factual", "temporal", "normative", "interpretive"]
    resolution: Literal["unresolved"]
    evidence: tuple[ArtifactPinV1, ...]

    @field_validator("item_ids", mode="before")
    @classmethod
    def normalize_item_ids(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: str(item),
            label="conflict item_ids",
        )

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_conflict_identity(self) -> "ReconstructionConflictV1":
        expected = reconstruction_conflict_id(self.model_dump(mode="json"))
        if self.conflict_id != expected:
            raise ValueError("conflict_id does not bind the exact conflict body")
        return self


class ReconstructedContextV1(_WorkspaceIdentityContract):
    schema_version: Literal["reconstructed_context.v1"]
    reconstruction_ref: NonEmptyStr
    reconstruction_hash: Sha256
    query: ArtifactPinV1
    candidate_set: ArtifactPinV1
    governance: ArtifactPinV1
    operator: ArtifactPinV1
    weighting: PolicySectionPinV1
    items: tuple[ReconstructedItemV1, ...]
    transitions: tuple[ReconstructedTransitionV1, ...]
    conflicts: tuple[ReconstructionConflictV1, ...]
    excluded: tuple[CandidateDiagnosticV1, ...]
    completeness: Literal["complete", "partial", "blocked"]
    unknowns: tuple[CandidateDiagnosticV1, ...]
    mandatory_closure: tuple[ArtifactPinV1, ...]

    identity_prefix = "reconstructed-context"
    identity_ref_field = "reconstruction_ref"
    identity_hash_field = "reconstruction_hash"
    identity_label = "reconstructed context"

    @field_validator("query")
    @classmethod
    def validate_query_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "reconstruction_query", "query")

    @field_validator("candidate_set")
    @classmethod
    def validate_candidate_set_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "memory_candidate_set", "candidate_set")

    @field_validator("governance")
    @classmethod
    def validate_governance_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(
            value,
            "reconstruction_governance_capture",
            "governance",
        )

    @field_validator("mandatory_closure", mode="before")
    @classmethod
    def normalize_mandatory_closure(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @field_validator("items", mode="before")
    @classmethod
    def normalize_items(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: str(_field(item, "item_id")),
            label="reconstructed items",
        )

    @model_validator(mode="after")
    def validate_reconstruction_relationships(self) -> "ReconstructedContextV1":
        item_ids = {item.item_id for item in self.items}
        for transition in self.transitions:
            for item_id in (transition.before_item_id, transition.after_item_id):
                if item_id is not None and item_id not in item_ids:
                    raise ValueError("reconstructed transition references an unknown item")
        for conflict in self.conflicts:
            if any(item_id not in item_ids for item_id in conflict.item_ids):
                raise ValueError("reconstruction conflict references an unknown item")
        return self


class WeightProfileV1(_ReconstructionContract):
    profile: AsciiId
    fast_ppm: PartsPerMillion
    middle_ppm: PartsPerMillion
    slow_ppm: PartsPerMillion
    missing_relevance_ppm: PartsPerMillion
    semantic_scorer: ArtifactPinV1
    temporal_scorer: ArtifactPinV1
    purpose_scorer: ArtifactPinV1

    @model_validator(mode="after")
    def validate_clock_weight_sum(self) -> "WeightProfileV1":
        if self.fast_ppm + self.middle_ppm + self.slow_ppm != 1_000_000:
            raise ValueError("fast, middle and slow weights must sum to 1000000 ppm")
        return self


class WeightingPolicyV1(_ReconstructionContract):
    section_version: AsciiId
    profiles: tuple[WeightProfileV1, ...]

    @field_validator("profiles", mode="before")
    @classmethod
    def normalize_profiles(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: str(_field(item, "profile")),
            label="weight profiles",
        )


MemoryCheckName: TypeAlias = Literal[
    "scope_valid",
    "provenance_complete",
    "domain_matches",
    "ontology_compatible",
    "tag_compatible",
    "source_rejected",
    "source_approved",
    "source_active",
    "source_effective",
    "authority_metadata_complete",
    "mandatory_closure_complete",
    "material_conflict",
    "risk_requires_governed_control",
    "contextual_use",
    "partial_optional_only",
]


class MemoryCheckV1(_ReconstructionContract):
    check: MemoryCheckName
    expected: StrictBool


class AllOfRulePredicateV1(_ReconstructionContract):
    kind: Literal["all_of"]
    checks: tuple[MemoryCheckV1, ...] = Field(min_length=1)

    @field_validator("checks", mode="before")
    @classmethod
    def normalize_checks(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: str(_field(item, "check")),
            label="rule checks",
        )


class OtherwiseRulePredicateV1(_ReconstructionContract):
    kind: Literal["otherwise"]


RulePredicateV1: TypeAlias = Annotated[
    AllOfRulePredicateV1 | OtherwiseRulePredicateV1,
    Field(discriminator="kind"),
]


class MemoryAdmissionRuleV1(_ReconstructionContract):
    rule_id: AsciiId
    priority: StrictNonNegativeInt
    predicate: RulePredicateV1
    effect: Literal["REJECT", "BLOCK", "CONTEXT_ONLY", "ADMIT"]
    reason: AsciiId


class MemoryAdmissibilityPolicyV2(_ReconstructionContract):
    schema_version: Literal["memory_admissibility_policy.v2"]
    policy_version: AsciiId
    organization_id: NonEmptyStr
    workspace_id: NonEmptyStr
    supersedes: ArtifactPinV1 | None
    governing_artifacts: tuple[ArtifactPinV1, ...]
    weighting: WeightingPolicyV1
    rules: tuple[MemoryAdmissionRuleV1, ...]

    @field_validator("governing_artifacts", mode="before")
    @classmethod
    def normalize_governing_artifacts(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @field_validator("rules", mode="before")
    @classmethod
    def normalize_rules(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: int(_field(item, "priority", -1)),
            label="memory admission rule priorities",
        )

    @model_validator(mode="after")
    def validate_closed_rule_order(self) -> "MemoryAdmissibilityPolicyV2":
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("memory admission rule_id values must be unique")
        otherwise = [rule for rule in self.rules if rule.predicate.kind == "otherwise"]
        if len(otherwise) != 1 or not self.rules or self.rules[-1] is not otherwise[0]:
            raise ValueError("otherwise must occur exactly once as the last rule")
        if otherwise[0].effect != "REJECT":
            raise ValueError("terminal otherwise rule must reject")
        return self


def memory_admissibility_policy_pin(
    policy: MemoryAdmissibilityPolicyV2,
) -> ArtifactPinV1:
    digest = content_hash(policy.model_dump(mode="json"))
    return ArtifactPinV1(
        artifact_kind="adapter_admissibility_policy",
        artifact_ref=f"admissibility-policy:{policy.policy_version}",
        artifact_hash=f"sha256:{digest}",
    )


def weighting_section_pin(
    policy_pin: ArtifactPinV1,
    weighting: WeightingPolicyV1,
) -> PolicySectionPinV1:
    return PolicySectionPinV1(
        container=policy_pin,
        section="weighting",
        section_version=weighting.section_version,
        section_hash=f"sha256:{content_hash(weighting.model_dump(mode='json'))}",
    )


class CheckEvaluationV1(_ReconstructionContract):
    rule_id: str
    check: MemoryCheckV1
    actual: StrictBool | None
    matched: StrictBool


class MemoryAdmissionDecisionV1(_ReconstructionContract):
    item_id: str
    decision: Literal["ADMIT", "CONTEXT_ONLY", "REJECT", "BLOCK"]
    rule_id: str | None
    reason: AsciiId
    evaluations: tuple[CheckEvaluationV1, ...]
    source_evidence: tuple[ArtifactPinV1, ...]

    @field_validator("source_evidence", mode="before")
    @classmethod
    def normalize_source_evidence(cls, value: object) -> object:
        return _ordered_unique_pins(value)


class AdmittedReconstructedContextV1(_WorkspaceIdentityContract):
    schema_version: Literal["admitted_reconstructed_context.v1"]
    admitted_ref: NonEmptyStr
    admitted_hash: Sha256
    query: ArtifactPinV1
    reconstruction: ArtifactPinV1
    governance: ArtifactPinV1
    policy: ArtifactPinV1
    decisions: tuple[MemoryAdmissionDecisionV1, ...]
    admitted_item_ids: tuple[str, ...]
    contextual_item_ids: tuple[str, ...]
    conflicts: tuple[ReconstructionConflictV1, ...]
    gate: Literal["DEFAULT_PASS", "GOVERNED_CONTROL_REQUIRED", "BLOCKED"]
    completeness: Literal["complete", "partial", "blocked"]

    identity_prefix = "admitted-reconstructed-context"
    identity_ref_field = "admitted_ref"
    identity_hash_field = "admitted_hash"
    identity_label = "admitted reconstructed context"

    @field_validator("query")
    @classmethod
    def validate_query_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "reconstruction_query", "query")

    @field_validator("reconstruction")
    @classmethod
    def validate_reconstruction_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "reconstructed_context", "reconstruction")

    @field_validator("governance")
    @classmethod
    def validate_governance_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(
            value,
            "reconstruction_governance_capture",
            "governance",
        )

    @field_validator("decisions", mode="before")
    @classmethod
    def normalize_decisions(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: str(_field(item, "item_id")),
            label="admission decisions",
        )

    @field_validator("admitted_item_ids", "contextual_item_ids", mode="before")
    @classmethod
    def normalize_item_ids(cls, value: object) -> object:
        return _ordered_unique(value, key=lambda item: str(item), label="selected item ids")

    @model_validator(mode="after")
    def validate_selected_sets(self) -> "AdmittedReconstructedContextV1":
        decision_ids = {decision.item_id for decision in self.decisions}
        if set(self.admitted_item_ids) & set(self.contextual_item_ids):
            raise ValueError("admitted and contextual item sets must be disjoint")
        if (set(self.admitted_item_ids) | set(self.contextual_item_ids)) - decision_ids:
            raise ValueError("selected item ids require decisions")
        derived_admitted = {
            decision.item_id for decision in self.decisions if decision.decision == "ADMIT"
        }
        derived_contextual = {
            decision.item_id
            for decision in self.decisions
            if decision.decision == "CONTEXT_ONLY"
        }
        if set(self.admitted_item_ids) != derived_admitted:
            raise ValueError("admitted_item_ids must derive exactly from decisions")
        if set(self.contextual_item_ids) != derived_contextual:
            raise ValueError("contextual_item_ids must derive exactly from decisions")
        return self


class VerifierTraversalLimitsV1(_ReconstructionContract):
    max_capture_nodes: StrictPositiveInt
    max_dependency_depth: StrictPositiveInt
    max_total_bytes: StrictPositiveInt


class PackagedVerifierEntryV1(_ReconstructionContract):
    entry_id: AsciiId
    artifact_kind: AsciiId
    schema_versions: tuple[AsciiId, ...] = Field(min_length=1)
    verification_mode: Literal[
        "full_canonical_json",
        "self_bound_envelope",
        "native_owner",
    ]
    ref_field: str | None
    hash_field: str | None
    ref_prefix: str | None
    resource_id: AsciiId
    entrypoint: AsciiId

    @field_validator("schema_versions", mode="before")
    @classmethod
    def normalize_schema_versions(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: str(item),
            label="verifier schema_versions",
        )

    @model_validator(mode="after")
    def validate_identity_descriptor(self) -> "PackagedVerifierEntryV1":
        identity = (self.ref_field, self.hash_field, self.ref_prefix)
        if self.verification_mode == "self_bound_envelope":
            if any(value is None or value == "" for value in identity):
                raise ValueError(
                    "self_bound_envelope requires ref_field, hash_field and ref_prefix"
                )
        elif any(value is not None for value in identity):
            raise ValueError(
                "non-self-bound verifier entries must not declare identity fields"
            )
        return self


class PackagedResourceManifestV1(_ReconstructionContract):
    resource_id: AsciiId
    media_type: AsciiId
    byte_length: StrictNonNegativeInt
    bytes_hash: Sha256


class PackagedVerifierRegistryV1(_ReconstructionContract):
    schema_version: Literal["packaged_verifier_registry.v1"]
    registry_version: AsciiId
    kernel_id: Literal["semantier.offline_artifact_integrity.v1"]
    kernel_resource: AsciiId
    limits: VerifierTraversalLimitsV1
    entries: tuple[PackagedVerifierEntryV1, ...]
    resources: tuple[PackagedResourceManifestV1, ...]

    @field_validator("entries", mode="before")
    @classmethod
    def normalize_entries(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: str(_field(item, "entry_id")),
            label="packaged verifier entries",
        )

    @field_validator("resources", mode="before")
    @classmethod
    def normalize_resources(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: str(_field(item, "resource_id")),
            label="packaged resource manifests",
        )

    @model_validator(mode="after")
    def validate_registry_links(self) -> "PackagedVerifierRegistryV1":
        resources = {resource.resource_id for resource in self.resources}
        if self.kernel_resource not in resources:
            raise ValueError("kernel_resource must name a packaged resource")
        if any(entry.resource_id not in resources for entry in self.entries):
            raise ValueError("verifier entry must name a packaged resource")
        selectors: set[tuple[str, str]] = set()
        for entry in self.entries:
            for schema in entry.schema_versions:
                selector = (entry.artifact_kind, schema)
                if selector in selectors:
                    raise ValueError(
                        "artifact kind and schema must resolve to exactly one verifier entry"
                    )
                selectors.add(selector)
        return self


def packaged_verifier_registry_pin(
    registry: PackagedVerifierRegistryV1,
) -> ArtifactPinV1:
    digest = content_hash(registry.model_dump(mode="json"))
    return ArtifactPinV1(
        artifact_kind="packaged_verifier_registry",
        artifact_ref=f"packaged-verifier-registry:{digest}",
        artifact_hash=f"sha256:{digest}",
    )


class RetainedPackageResourceV1(_ReconstructionContract):
    resource_id: AsciiId
    bytes_base64: str = Field(strict=True)

    def decoded_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.bytes_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("bytes_base64 must use strict canonical base64") from exc

    @model_validator(mode="after")
    def validate_base64(self) -> "RetainedPackageResourceV1":
        raw = self.decoded_bytes()
        if base64.b64encode(raw).decode("ascii") != self.bytes_base64:
            raise ValueError("bytes_base64 must use strict canonical base64")
        return self


class _RetainedResourceMissingError(ValueError):
    """A well-shaped package omits a declared retained manifest member."""


class _RetainedResourceLengthError(ValueError):
    """A well-shaped retained member differs from its declared decoded length."""


class RetainedVerifierPackageV1(_ReconstructionContract):
    registry: CapturedJsonV1
    resources: tuple[RetainedPackageResourceV1, ...]

    @field_validator("resources", mode="before")
    @classmethod
    def normalize_resources(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: str(_field(item, "resource_id")),
            label="retained package resources",
        )

    def parsed_registry(self) -> PackagedVerifierRegistryV1:
        return PackagedVerifierRegistryV1.model_validate_json(self.registry.canonical_json)

    @model_validator(mode="after")
    def validate_resource_manifest(self) -> "RetainedVerifierPackageV1":
        registry = self.parsed_registry()
        manifests = {resource.resource_id: resource for resource in registry.resources}
        resources = {resource.resource_id: resource for resource in self.resources}
        if set(resources) - set(manifests):
            raise ValueError("retained resources must match registry manifest one-to-one")
        if set(manifests) - set(resources):
            raise _RetainedResourceMissingError(
                "retained resources must match registry manifest one-to-one"
            )
        for resource_id, resource in resources.items():
            raw = resource.decoded_bytes()
            manifest = manifests[resource_id]
            if len(raw) != manifest.byte_length:
                raise _RetainedResourceLengthError(
                    f"retained resource {resource_id} length mismatch"
                )
            if f"sha256:{hashlib.sha256(raw).hexdigest()}" != manifest.bytes_hash:
                raise ValueError(f"retained resource {resource_id} hash mismatch")
        return self


AdvisoryRequestId = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]


class EpisodeOutcomePinV1(_ReconstructionContract):
    episode: ArtifactPinV1
    assessment: ArtifactPinV1

    @model_validator(mode="after")
    def exact_pair_kinds(self) -> "EpisodeOutcomePinV1":
        _require_pin_kind(self.episode, "empirical_episode", "episode")
        _require_pin_kind(self.assessment, "empirical_episode_assessment", "assessment")
        return self


class AdvisoryContextRequestV1(_ReconstructionContract):
    schema_version: Literal["advisory_context_request.v1"]
    request_id: AdvisoryRequestId
    profile: Literal["context_only_empirical_v1"]
    purpose: Literal["NEXT_PROPOSAL"]
    outcomes: tuple[EpisodeOutcomePinV1, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def distinct_ordered_outcomes(self) -> "AdvisoryContextRequestV1":
        refs = [
            (pin.artifact_kind, pin.artifact_ref)
            for pair in self.outcomes
            for pin in (pair.episode, pair.assessment)
        ]
        if len(refs) != len(set(refs)):
            raise ValueError("outcomes contain duplicate or conflicting pins")
        return self


class AdvisoryContextFormationV1(_WorkspaceIdentityContract):
    schema_version: Literal["advisory_context_formation.v1"]
    formation_ref: NonEmptyStr
    formation_hash: Sha256
    actor_id: NonEmptyStr
    session_ref: NonEmptyStr
    request: AdvisoryContextRequestV1
    profile: ArtifactPinV1
    operator: ArtifactPinV1
    verifier_registry: ArtifactPinV1
    verifier_package: RetainedVerifierPackageV1
    sources: tuple[CapturedSourceV1, ...]
    representation: CapturedJsonV1
    usage: Literal["CONTEXT_ONLY"]
    recorded_at: UtcDateTime

    identity_prefix = "advisory-context-formation"
    identity_ref_field = "formation_ref"
    identity_hash_field = "formation_hash"
    identity_label = "advisory context formation"

    @model_validator(mode="after")
    def exact_formation_links(self) -> "AdvisoryContextFormationV1":
        _require_pin_kind(self.profile, "advisory_context_profile", "profile")
        _require_pin_kind(self.operator, "advisory_context_operator", "operator")
        _require_pin_kind(
            self.verifier_registry, "packaged_verifier_registry", "verifier_registry"
        )
        if (
            packaged_verifier_registry_pin(self.verifier_package.parsed_registry())
            != self.verifier_registry
        ):
            raise ValueError("verifier registry hash mismatch")
        keys = [_pin_key(source.source) for source in self.sources]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("sources must be in unique canonical order")
        for source in self.sources:
            if source.verifier.registry != self.verifier_registry:
                raise ValueError("source registry hash mismatch")
            if any(
                pin.artifact_kind == "advisory_context_formation"
                for pin in (source.source, *source.dependencies)
            ):
                raise ValueError("formation cannot refer to a formation")
            payload = json.loads(source.payload.canonical_json)
            if isinstance(payload, dict):
                for field in ("organization_id", "workspace_id"):
                    if field in payload and payload[field] != getattr(self, field):
                        raise ValueError("source scope mismatch")
        return self


class AdvisoryContextCommandV1(_WorkspaceIdentityContract):
    schema_version: Literal["advisory_context_command.v1"]
    command_ref: NonEmptyStr
    command_hash: Sha256
    actor_id: NonEmptyStr
    session_ref: NonEmptyStr
    operation: Literal["FORM_ADVISORY_CONTEXT"]
    request: AdvisoryContextRequestV1

    identity_prefix = "advisory-context-command"
    identity_ref_field = "command_ref"
    identity_hash_field = "command_hash"
    identity_label = "advisory context command"


class AdvisoryContextReceiptV1(_WorkspaceIdentityContract):
    schema_version: Literal["advisory_context_receipt.v1"]
    receipt_ref: NonEmptyStr
    receipt_hash: Sha256
    actor_id: NonEmptyStr
    session_ref: NonEmptyStr
    operation: Literal["FORM_ADVISORY_CONTEXT"]
    request_id: AdvisoryRequestId
    request_hash: Sha256
    command: ArtifactPinV1
    formation: ArtifactPinV1
    usage: Literal["CONTEXT_ONLY"]
    recorded_at: UtcDateTime

    identity_prefix = "advisory-context-receipt"
    identity_ref_field = "receipt_ref"
    identity_hash_field = "receipt_hash"
    identity_label = "advisory context receipt"

    @model_validator(mode="after")
    def exact_receipt_kinds(self) -> "AdvisoryContextReceiptV1":
        _require_pin_kind(self.command, "advisory_context_command", "command")
        _require_pin_kind(self.formation, "advisory_context_formation", "formation")
        if self.request_hash != self.command.artifact_hash:
            raise ValueError("request hash mismatch")
        return self


class FrozenContextFormationV1(_WorkspaceIdentityContract):
    schema_version: Literal["frozen_context_formation.v1"]
    formation_ref: NonEmptyStr
    formation_hash: Sha256
    query: ArtifactPinV1
    candidate_set: ArtifactPinV1
    governance: ArtifactPinV1
    reconstruction: ArtifactPinV1
    admitted: ArtifactPinV1
    operator: ArtifactPinV1
    providers: tuple[ArtifactPinV1, ...]
    weighting: PolicySectionPinV1
    admissibility_policy: ArtifactPinV1
    verifier_registry: ArtifactPinV1
    verifier_package: RetainedVerifierPackageV1
    closure: tuple[CapturedSourceV1, ...]
    element_bindings: tuple[FormationElementBindingV1, ...]
    retained_element_bindings: tuple[FormationElementBindingV1, ...]
    completeness: Literal["complete", "partial", "blocked"]

    identity_prefix = "frozen-context-formation"
    identity_ref_field = "formation_ref"
    identity_hash_field = "formation_hash"
    identity_label = "frozen context formation"

    @field_validator("query")
    @classmethod
    def validate_query_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "reconstruction_query", "query")

    @field_validator("candidate_set")
    @classmethod
    def validate_candidate_set_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "memory_candidate_set", "candidate_set")

    @field_validator("governance")
    @classmethod
    def validate_governance_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(
            value,
            "reconstruction_governance_capture",
            "governance",
        )

    @field_validator("reconstruction")
    @classmethod
    def validate_reconstruction_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "reconstructed_context", "reconstruction")

    @field_validator("admitted")
    @classmethod
    def validate_admitted_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(
            value,
            "admitted_reconstructed_context",
            "admitted",
        )

    @field_validator("verifier_registry")
    @classmethod
    def validate_registry_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "packaged_verifier_registry", "verifier_registry")

    @field_validator("providers", mode="before")
    @classmethod
    def normalize_providers(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @field_validator("closure", mode="before")
    @classmethod
    def normalize_closure(cls, value: object) -> object:
        return _ordered_unique_captured_sources(value, label="formation closure")

    @field_validator("element_bindings", "retained_element_bindings", mode="before")
    @classmethod
    def normalize_bindings(cls, value: object) -> object:
        return _ordered_unique(
            value,
            key=lambda item: (
                str(
                    _field(
                        _field(_field(item, "body"), "object"),
                        "object_ref",
                    )
                ),
                str(_field(item, "item_id")),
                _pin_key(_field(item, "reconstruction")),
            ),
            label="formation element bindings",
        )

    @model_validator(mode="after")
    def validate_formation_relationships(self) -> "FrozenContextFormationV1":
        if not self.providers:
            raise ValueError("formation providers must be nonempty")
        if _pin_key(self.weighting.container) != _pin_key(self.admissibility_policy):
            raise ValueError("weighting container must equal admissibility_policy")
        registry = packaged_verifier_registry_pin(self.verifier_package.parsed_registry())
        if _pin_key(registry) != _pin_key(self.verifier_registry):
            raise ValueError("verifier_package registry bytes must match verifier_registry")
        if any(
            _pin_key(captured.verifier.registry) != _pin_key(self.verifier_registry)
            for captured in self.closure
        ):
            raise ValueError("every immediate closure verifier must use verifier_registry")
        selected_refs = {
            binding.body.object.object_ref for binding in self.element_bindings
        }
        retained_refs = {
            binding.body.object.object_ref for binding in self.retained_element_bindings
        }
        if selected_refs & retained_refs:
            raise ValueError("selected and retained element bindings must be disjoint")
        return self


class ProjectionReconstructionManifestV1(_ReconstructionContract):
    schema_version: Literal["projection_reconstruction_manifest.v1"]
    formation: FrozenContextFormationV1
    artifacts: tuple[CapturedSourceV1, ...]
    selected_payload_hash: Sha256

    @field_validator("artifacts", mode="before")
    @classmethod
    def normalize_artifacts(cls, value: object) -> object:
        return _ordered_unique_captured_sources(value, label="projection artifacts")


def selected_payload_hash(value: Mapping[str, object]) -> str:
    if "__semantier_reconstruction_v1" in value:
        raise ValueError("ordinary selected payload must not contain the reserved manifest key")
    return f"sha256:{content_hash(_json_value(value))}"


class CreateFromQueryCommandRequestV1(_ReconstructionContract):
    schema_version: Literal["create_from_query_command_request.v1"]
    request_id: NonEmptyStr
    reconstruction_request: ReconstructionQueryRequestV1
    element_receipts: tuple[ArtifactPinV1, ...]

    @field_validator("element_receipts", mode="before")
    @classmethod
    def normalize_element_receipts(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_purpose(self) -> "CreateFromQueryCommandRequestV1":
        if self.reconstruction_request.purpose != "create":
            raise ValueError("create command requires create reconstruction purpose")
        return self


class BranchFromQueryCommandRequestV1(_ReconstructionContract):
    schema_version: Literal["branch_from_query_command_request.v1"]
    request_id: NonEmptyStr
    reconstruction_request: ReconstructionQueryRequestV1
    element_receipts: tuple[ArtifactPinV1, ...]

    @field_validator("element_receipts", mode="before")
    @classmethod
    def normalize_element_receipts(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_purpose(self) -> "BranchFromQueryCommandRequestV1":
        if self.reconstruction_request.purpose != "branch":
            raise ValueError("branch command requires branch reconstruction purpose")
        return self


class QueryCommandFormationIngressV1(_ReconstructionContract):
    kind: Literal["query_command"]


WorldScopeOverrideV1: TypeAlias = Annotated[
    InheritOverrideV1 | ReplaceOverrideV1[RuntimeWorldScopeV1],
    Field(discriminator="mode"),
]
PinTupleOverrideV1: TypeAlias = Annotated[
    InheritOverrideV1 | ReplaceOverrideV1[tuple[ArtifactPinV1, ...]],
    Field(discriminator="mode"),
]


class LegacyBranchFacadeFormationIngressV1(_ReconstructionContract):
    kind: Literal["legacy_branch_facade"]
    goal: GoalOverrideV1
    world: WorldScopeOverrideV1
    semantic_pins: PinTupleOverrideV1
    reconstruction_request_supplied: StrictBool

    @model_validator(mode="before")
    @classmethod
    def normalize_semantic_replacement(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        semantic_pins = value.get("semantic_pins")
        if isinstance(semantic_pins, ReplaceOverrideV1):
            normalized = dict(value)
            normalized["semantic_pins"] = {
                "mode": "replace",
                "value": _ordered_unique_pins(semantic_pins.value),
            }
            return normalized
        if (
            not isinstance(semantic_pins, Mapping)
            or semantic_pins.get("mode") != "replace"
        ):
            return value
        normalized = dict(value)
        normalized_semantic_pins = dict(semantic_pins)
        normalized_semantic_pins["value"] = _ordered_unique_pins(
            normalized_semantic_pins.get("value")
        )
        normalized["semantic_pins"] = normalized_semantic_pins
        return normalized


FormationIngressV1: TypeAlias = Annotated[
    QueryCommandFormationIngressV1 | LegacyBranchFacadeFormationIngressV1,
    Field(discriminator="kind"),
]


class RuntimeContextFormationCommandV1(_WorkspaceIdentityContract):
    schema_version: Literal["runtime_context_formation_command.v1"]
    command_ref: NonEmptyStr
    command_hash: Sha256
    operation: Literal["CREATE", "BRANCH"]
    session_ref: NonEmptyStr
    request_id: NonEmptyStr
    actor_id: NonEmptyStr
    request: NormalizedReconstructionRequestV1
    ingress: FormationIngressV1
    origin: ArtifactPinV1 | None
    element_receipts: tuple[ArtifactPinV1, ...]
    request_resolution_at: UtcDateTime
    resolved_query: ArtifactPinV1

    identity_prefix = "runtime-context-formation-command"
    identity_ref_field = "command_ref"
    identity_hash_field = "command_hash"
    identity_label = "runtime context formation command"

    @field_validator("element_receipts", mode="before")
    @classmethod
    def normalize_element_receipts(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @field_validator("resolved_query")
    @classmethod
    def validate_query_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "reconstruction_query", "resolved_query")

    @model_validator(mode="after")
    def validate_command_relationships(self) -> "RuntimeContextFormationCommandV1":
        purpose = self.operation.lower()
        if self.request.purpose != purpose:
            raise ValueError("formation command operation must match request purpose")
        if self.origin != self.request.origin:
            raise ValueError("formation command origin must equal normalized request origin")
        if self.operation == "CREATE" and self.origin is not None:
            raise ValueError("CREATE formation command requires null origin")
        if self.operation == "BRANCH" and self.origin is None:
            raise ValueError("BRANCH formation command requires origin")
        return self


class RuntimeContextPersistenceManifestV1(_WorkspaceIdentityContract):
    schema_version: Literal["runtime_context_persistence_manifest.v1"]
    manifest_ref: NonEmptyStr
    manifest_hash: Sha256
    actor_id: NonEmptyStr
    request_id: NonEmptyStr
    operation: Literal["CREATE", "BRANCH", "COMMIT"]
    expected_parent: ArtifactPinV1 | None
    expected_version: StrictNonNegativeInt | None
    successor: ArtifactPinV1
    transition: ArtifactPinV1 | None
    proposal: ArtifactPinV1 | None
    formation: ArtifactPinV1
    formation_command: ArtifactPinV1 | None
    related_artifacts: tuple[ArtifactPinV1, ...]

    identity_prefix = "runtime-context-persistence-manifest"
    identity_ref_field = "manifest_ref"
    identity_hash_field = "manifest_hash"
    identity_label = "runtime context persistence manifest"

    @field_validator("successor")
    @classmethod
    def validate_successor_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "runtime_working_context", "successor")

    @field_validator("expected_parent")
    @classmethod
    def validate_expected_parent_kind(
        cls,
        value: ArtifactPinV1 | None,
    ) -> ArtifactPinV1 | None:
        if value is None:
            return None
        return _require_pin_kind(value, "runtime_working_context", "expected_parent")

    @field_validator("transition")
    @classmethod
    def validate_transition_kind(
        cls,
        value: ArtifactPinV1 | None,
    ) -> ArtifactPinV1 | None:
        if value is None:
            return None
        return _require_pin_kind(value, "runtime_context_transition", "transition")

    @field_validator("proposal")
    @classmethod
    def validate_proposal_kind(
        cls,
        value: ArtifactPinV1 | None,
    ) -> ArtifactPinV1 | None:
        if value is None:
            return None
        return _require_pin_kind(value, "runtime_context_transition_proposal", "proposal")

    @field_validator("formation")
    @classmethod
    def validate_formation_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        return _require_pin_kind(value, "frozen_context_formation", "formation")

    @field_validator("formation_command")
    @classmethod
    def validate_formation_command_kind(
        cls,
        value: ArtifactPinV1 | None,
    ) -> ArtifactPinV1 | None:
        if value is None:
            return None
        return _require_pin_kind(
            value,
            "runtime_context_formation_command",
            "formation_command",
        )

    @field_validator("related_artifacts", mode="before")
    @classmethod
    def normalize_related_artifacts(cls, value: object) -> object:
        return _ordered_unique_pins(value)

    @model_validator(mode="after")
    def validate_operation_shape(self) -> "RuntimeContextPersistenceManifestV1":
        if self.operation == "COMMIT":
            if (
                self.expected_parent is None
                or self.expected_version is None
                or self.transition is None
                or self.proposal is None
                or self.formation_command is not None
            ):
                raise ValueError(
                    "COMMIT manifest requires parent/version/transition/proposal and no formation command"
                )
        elif any(
            value is not None
            for value in (
                self.expected_parent,
                self.expected_version,
                self.transition,
                self.proposal,
            )
        ) or self.formation_command is None:
            raise ValueError(
                "CREATE/BRANCH manifest requires formation command and null commit fields"
            )
        forbidden_kinds = {
            "runtime_context_persistence_manifest",
            "runtime_context_transition_receipt",
            "runtime_working_context",
            "runtime_context_transition",
            "runtime_context_transition_proposal",
        }
        if any(pin.artifact_kind in forbidden_kinds for pin in self.related_artifacts):
            raise ValueError("related_artifacts contains a forbidden persistence artifact")
        if sum(_pin_key(pin) == _pin_key(self.formation) for pin in self.related_artifacts) != 1:
            raise ValueError("related_artifacts must include formation exactly once")
        command_count = sum(
            pin.artifact_kind == "runtime_context_formation_command"
            for pin in self.related_artifacts
        )
        if self.operation == "COMMIT" and command_count:
            raise ValueError("COMMIT related_artifacts must not contain a formation command")
        if self.operation in {"CREATE", "BRANCH"}:
            if command_count != 1 or all(
                _pin_key(pin) != _pin_key(self.formation_command)
                for pin in self.related_artifacts
            ):
                raise ValueError(
                    "CREATE/BRANCH related_artifacts must include the exact formation command once"
                )
        return self


def runtime_context_persistence_command_hash_v2(
    manifest: RuntimeContextPersistenceManifestV1 | ArtifactPinV1,
) -> str:
    manifest_pin = (
        manifest
        if isinstance(manifest, ArtifactPinV1)
        else artifact_pin_for(manifest)
    )
    if manifest_pin.artifact_kind != "runtime_context_persistence_manifest":
        raise ValueError("manifest must pin runtime_context_persistence_manifest")
    return "sha256:" + content_hash(
        {
            "schema_version": "runtime_context_persistence_command.v2",
            "manifest": manifest_pin.model_dump(mode="json"),
        }
    )


@dataclass(frozen=True, slots=True)
class ArtifactSchemaRegistration:
    model: type[BaseModel]
    artifact_kind: str
    ref_field: str | None
    hash_field: str | None
    ref_prefix: str

    def hash_basis(self, value: Mapping[str, object]) -> dict[str, object]:
        excluded = tuple(
            field
            for field in (self.ref_field, self.hash_field)
            if field is not None
        )
        return _without(value, *excluded)


_SELF_BOUND_REGISTRATIONS = {
    "runtime_world_scope.v1": ArtifactSchemaRegistration(
        RuntimeWorldScopeV1,
        "runtime_world_scope",
        "world_ref",
        "world_hash",
        "runtime-world",
    ),
    "runtime_working_context.v1": ArtifactSchemaRegistration(
        RuntimeTaskContextV1,
        "runtime_working_context",
        "context_ref",
        "context_hash",
        "runtime-context",
    ),
    "runtime_working_context.v2": ArtifactSchemaRegistration(
        RuntimeTaskContextV2,
        "runtime_working_context",
        "context_ref",
        "context_hash",
        "runtime-context",
    ),
    "runtime_context_transition_proposal.v1": ArtifactSchemaRegistration(
        RuntimeContextTransitionProposalV1,
        "runtime_context_transition_proposal",
        "proposal_ref",
        "proposal_hash",
        "runtime-context-proposal",
    ),
    "runtime_context_transition_proposal.v2": ArtifactSchemaRegistration(
        RuntimeContextTransitionProposalV2,
        "runtime_context_transition_proposal",
        "proposal_ref",
        "proposal_hash",
        "runtime-context-proposal",
    ),
    "runtime_context_transition.v1": ArtifactSchemaRegistration(
        RuntimeContextTransitionV1,
        "runtime_context_transition",
        "transition_ref",
        "transition_hash",
        "runtime-context-transition",
    ),
    "runtime_context_transition.v2": ArtifactSchemaRegistration(
        RuntimeContextTransitionV2,
        "runtime_context_transition",
        "transition_ref",
        "transition_hash",
        "runtime-context-transition",
    ),
    "runtime_context_transition_receipt.v1": ArtifactSchemaRegistration(
        RuntimeContextTransitionReceiptV1,
        "runtime_context_transition_receipt",
        "receipt_ref",
        "receipt_hash",
        "runtime-context-receipt",
    ),
    "runtime_context_transition_receipt.v2": ArtifactSchemaRegistration(
        RuntimeContextTransitionReceiptV2,
        "runtime_context_transition_receipt",
        "receipt_ref",
        "receipt_hash",
        "runtime-context-receipt",
    ),
    "reconstruction_query.v1": ArtifactSchemaRegistration(
        ReconstructionQueryV1,
        "reconstruction_query",
        "query_ref",
        "query_hash",
        "reconstruction-query",
    ),
    "candidate_capture.v1": ArtifactSchemaRegistration(
        CandidateCaptureV1,
        "candidate_capture",
        "capture_ref",
        "capture_hash",
        "candidate-capture",
    ),
    "memory_candidate.v1": ArtifactSchemaRegistration(
        MemoryCandidateV1,
        "memory_candidate",
        "candidate_ref",
        "candidate_hash",
        "memory-candidate",
    ),
    "memory_candidate_set.v1": ArtifactSchemaRegistration(
        MemoryCandidateSetV1,
        "memory_candidate_set",
        "candidate_set_ref",
        "candidate_set_hash",
        "memory-candidate-set",
    ),
    "reconstruction_governance_capture.v1": ArtifactSchemaRegistration(
        ReconstructionGovernanceCaptureV1,
        "reconstruction_governance_capture",
        "governance_ref",
        "governance_hash",
        "reconstruction-governance",
    ),
    "reconstructed_context.v1": ArtifactSchemaRegistration(
        ReconstructedContextV1,
        "reconstructed_context",
        "reconstruction_ref",
        "reconstruction_hash",
        "reconstructed-context",
    ),
    "admitted_reconstructed_context.v1": ArtifactSchemaRegistration(
        AdmittedReconstructedContextV1,
        "admitted_reconstructed_context",
        "admitted_ref",
        "admitted_hash",
        "admitted-reconstructed-context",
    ),
    "frozen_context_formation.v1": ArtifactSchemaRegistration(
        FrozenContextFormationV1,
        "frozen_context_formation",
        "formation_ref",
        "formation_hash",
        "frozen-context-formation",
    ),
    "runtime_context_formation_command.v1": ArtifactSchemaRegistration(
        RuntimeContextFormationCommandV1,
        "runtime_context_formation_command",
        "command_ref",
        "command_hash",
        "runtime-context-formation-command",
    ),
    "runtime_context_persistence_manifest.v1": ArtifactSchemaRegistration(
        RuntimeContextPersistenceManifestV1,
        "runtime_context_persistence_manifest",
        "manifest_ref",
        "manifest_hash",
        "runtime-context-persistence-manifest",
    ),
}

_SELF_BOUND_REGISTRATIONS.update(
    {
        "advisory_context_formation.v1": ArtifactSchemaRegistration(
            AdvisoryContextFormationV1,
            "advisory_context_formation",
            "formation_ref",
            "formation_hash",
            "advisory-context-formation",
        ),
        "advisory_context_command.v1": ArtifactSchemaRegistration(
            AdvisoryContextCommandV1,
            "advisory_context_command",
            "command_ref",
            "command_hash",
            "advisory-context-command",
        ),
        "advisory_context_receipt.v1": ArtifactSchemaRegistration(
            AdvisoryContextReceiptV1,
            "advisory_context_receipt",
            "receipt_ref",
            "receipt_hash",
            "advisory-context-receipt",
        ),
    }
)


from contracts.tender_review import TenderReviewRecordV1, TenderReviewReportV1, TenderReviewReportV2, TenderReviewDocxDerivativeV1
from contracts.tender_legal_compliance import TenderLegalComplianceBasisV1

_SELF_BOUND_REGISTRATIONS.update({
    "tender_review_docx_derivative.v1": ArtifactSchemaRegistration(
        TenderReviewDocxDerivativeV1, "tender_review_docx_derivative", "derivative_ref", "derivative_hash", "tender-review-docx-derivative",
    ),
    "tender_legal_compliance_basis.v1": ArtifactSchemaRegistration(
        TenderLegalComplianceBasisV1, "tender_legal_compliance_basis", "basis_ref", "basis_hash", "tender-legal-compliance-basis",
    ),
    "tender_review_record.v1": ArtifactSchemaRegistration(
        TenderReviewRecordV1, "tender_review_record", "record_ref", "record_hash", "tender-review-record",
    ),
    "tender_review_report.v2": ArtifactSchemaRegistration(
        TenderReviewReportV2, "tender_review_report", "report_ref", "report_hash", "tender-review-report",
    ),
    "tender_review_report.v1": ArtifactSchemaRegistration(
        TenderReviewReportV1, "tender_review_report", "report_ref", "report_hash", "tender-review-report",
    ),
})

ARTIFACT_SCHEMA_REGISTRY = MappingProxyType(
    {
        **_SELF_BOUND_REGISTRATIONS,
        "packaged_verifier_registry.v1": ArtifactSchemaRegistration(
            PackagedVerifierRegistryV1,
            "packaged_verifier_registry",
            None,
            None,
            "packaged-verifier-registry",
        ),
        "memory_admissibility_policy.v2": ArtifactSchemaRegistration(
            MemoryAdmissibilityPolicyV2,
            "adapter_admissibility_policy",
            None,
            None,
            "admissibility-policy",
        ),
    }
)


def artifact_model_for_schema(schema_version: str) -> type[BaseModel]:
    try:
        return ARTIFACT_SCHEMA_REGISTRY[schema_version].model
    except KeyError as exc:
        raise ValueError(f"unsupported artifact schema {schema_version!r}") from exc


def artifact_kind_for_schema(schema_version: str) -> str:
    try:
        return ARTIFACT_SCHEMA_REGISTRY[schema_version].artifact_kind
    except KeyError as exc:
        raise ValueError(f"unsupported artifact schema {schema_version!r}") from exc


def artifact_pin_for(value: BaseModel) -> ArtifactPinV1:
    schema_version = getattr(value, "schema_version", None)
    if not isinstance(schema_version, str):
        raise ValueError("artifact model has no registered schema_version")
    try:
        registration = ARTIFACT_SCHEMA_REGISTRY[schema_version]
    except KeyError as exc:
        raise ValueError(f"unsupported artifact schema {schema_version!r}") from exc
    if not isinstance(value, registration.model):
        raise ValueError("artifact model does not match its registered schema")
    if registration.ref_field is not None and registration.hash_field is not None:
        return ArtifactPinV1(
            artifact_kind=registration.artifact_kind,
            artifact_ref=getattr(value, registration.ref_field),
            artifact_hash=getattr(value, registration.hash_field),
        )
    if isinstance(value, PackagedVerifierRegistryV1):
        return packaged_verifier_registry_pin(value)
    if isinstance(value, MemoryAdmissibilityPolicyV2):
        return memory_admissibility_policy_pin(value)
    raise ValueError(f"schema {schema_version!r} has no artifact identity rule")


FormationArtifactV1: TypeAlias = (
    ReconstructionQueryV1
    | CandidateCaptureV1
    | MemoryCandidateV1
    | MemoryCandidateSetV1
    | ReconstructionGovernanceCaptureV1
    | ReconstructedContextV1
    | AdmittedReconstructedContextV1
)


class CandidateRetrievalV1(_ReconstructionContract):
    captures: tuple[CandidateCaptureV1, ...]
    candidates: tuple[MemoryCandidateV1, ...]
    candidate_set: MemoryCandidateSetV1

    @model_validator(mode="after")
    def validate_retrieval_links(self) -> "CandidateRetrievalV1":
        for value in (*self.captures, *self.candidates, self.candidate_set):
            if (
                value.organization_id != self.candidate_set.organization_id
                or value.workspace_id != self.candidate_set.workspace_id
            ):
                raise ValueError("retrieval artifacts must share one workspace scope")
        capture_pins = tuple(artifact_pin_for(value) for value in self.captures)
        candidate_pins = tuple(artifact_pin_for(value) for value in self.candidates)
        if tuple(sorted(capture_pins, key=_pin_key)) != self.candidate_set.captures:
            raise ValueError("candidate_set captures must match retrieval captures")
        if tuple(sorted(candidate_pins, key=_pin_key)) != self.candidate_set.candidates:
            raise ValueError("candidate_set candidates must match retrieval candidates")
        query = _pin_key(self.candidate_set.query)
        if any(_pin_key(value.query) != query for value in (*self.captures, *self.candidates)):
            raise ValueError("retrieval artifacts must share the exact query")
        captures_by_pin = {
            _pin_key(artifact_pin_for(capture)): capture for capture in self.captures
        }
        for candidate in self.candidates:
            capture = captures_by_pin.get(_pin_key(candidate.capture))
            if capture is None:
                raise ValueError("candidate must pin one supplied capture")
            if _pin_key(candidate.provider) != _pin_key(capture.provider):
                raise ValueError("candidate provider must equal its capture provider")
        candidates_by_pin = {
            _pin_key(artifact_pin_for(candidate)): candidate
            for candidate in self.candidates
        }
        for event_pin in self.candidate_set.ordered_event_candidates:
            event_candidate = candidates_by_pin.get(_pin_key(event_pin))
            if event_candidate is None or event_candidate.item_type != "event":
                raise ValueError(
                    "ordered event candidate must pin a supplied event candidate"
                )
        return self


class FrozenFormationBundleV1(_ReconstructionContract):
    formation: FrozenContextFormationV1
    artifacts: tuple[FormationArtifactV1, ...]

    @model_validator(mode="after")
    def validate_bundle_scope_and_set(self) -> "FrozenFormationBundleV1":
        pins = [artifact_pin_for(value) for value in self.artifacts]
        if len({_pin_key(pin) for pin in pins}) != len(pins):
            raise ValueError("formation bundle contains duplicate artifact identities")
        for artifact in self.artifacts:
            if (
                artifact.organization_id != self.formation.organization_id
                or artifact.workspace_id != self.formation.workspace_id
            ):
                raise ValueError("formation bundle artifact scope mismatch")
        artifacts_by_pin = {
            _pin_key(artifact_pin_for(artifact)): artifact for artifact in self.artifacts
        }
        query = artifacts_by_pin.get(_pin_key(self.formation.query))
        if not isinstance(query, ReconstructionQueryV1):
            raise ValueError("formation bundle must contain its exact resolved query")
        if _pin_key(query.verifier_registry) != _pin_key(self.formation.verifier_registry):
            raise ValueError("formation and resolved query verifier registry must match")
        return self


@dataclass(frozen=True, slots=True)
class ResolvedReconstructionOperatorV1:
    descriptor: ArtifactPinV1
    implementation: Callable[..., object]
    configuration: CapturedJsonV1


@dataclass(frozen=True, slots=True)
class TrustedRetainedVerifierRegistrationV1:
    registry_pin: ArtifactPinV1
    registry_bytes: bytes
    resource_hashes: tuple[tuple[str, str], ...]
    verifiers: tuple[tuple[str, Callable[..., object]], ...]


def validate_admission_decisions(
    admitted: AdmittedReconstructedContextV1,
    reconstructed: ReconstructedContextV1,
) -> None:
    if (
        admitted.organization_id != reconstructed.organization_id
        or admitted.workspace_id != reconstructed.workspace_id
    ):
        raise ValueError("admission and reconstruction scope must match")
    if _pin_key(admitted.query) != _pin_key(reconstructed.query):
        raise ValueError("admission and reconstruction query must match")
    if _pin_key(admitted.governance) != _pin_key(reconstructed.governance):
        raise ValueError("admission and reconstruction governance must match")
    if _pin_key(admitted.reconstruction) != _pin_key(artifact_pin_for(reconstructed)):
        raise ValueError("admission must pin the supplied reconstruction")
    item_ids = {item.item_id for item in reconstructed.items}
    decision_ids = {decision.item_id for decision in admitted.decisions}
    if decision_ids != item_ids:
        raise ValueError("admission decisions must cover every reconstructed item exactly once")
    if admitted.conflicts != reconstructed.conflicts:
        raise ValueError("admission conflicts must preserve reconstruction conflicts")


def validate_formation_element_bindings(
    formation: FrozenContextFormationV1,
    reconstructed: ReconstructedContextV1,
    admitted: AdmittedReconstructedContextV1,
    *,
    origin_bindings: tuple[FormationElementBindingV1, ...] = (),
) -> None:
    reconstruction_pin = artifact_pin_for(reconstructed)
    scopes = {
        (formation.organization_id, formation.workspace_id),
        (reconstructed.organization_id, reconstructed.workspace_id),
        (admitted.organization_id, admitted.workspace_id),
    }
    if len(scopes) != 1:
        raise ValueError("formation association scope must match")
    if not (
        _pin_key(formation.query)
        == _pin_key(reconstructed.query)
        == _pin_key(admitted.query)
    ):
        raise ValueError("formation association query must match")
    if _pin_key(formation.candidate_set) != _pin_key(reconstructed.candidate_set):
        raise ValueError("formation association candidate set must match")
    if not (
        _pin_key(formation.governance)
        == _pin_key(reconstructed.governance)
        == _pin_key(admitted.governance)
    ):
        raise ValueError("formation association governance must match")
    if _pin_key(formation.operator) != _pin_key(reconstructed.operator):
        raise ValueError("formation association operator must match")
    if formation.weighting != reconstructed.weighting:
        raise ValueError("formation association weighting must match")
    if _pin_key(formation.admissibility_policy) != _pin_key(admitted.policy):
        raise ValueError("formation association admissibility policy must match")
    if _pin_key(formation.reconstruction) != _pin_key(reconstruction_pin):
        raise ValueError("formation must pin the supplied reconstruction")
    if _pin_key(formation.admitted) != _pin_key(artifact_pin_for(admitted)):
        raise ValueError("formation must pin the supplied admitted result")
    validate_admission_decisions(admitted, reconstructed)
    items = {item.item_id: item for item in reconstructed.items}
    selected = set(admitted.admitted_item_ids) | set(admitted.contextual_item_ids)
    expected_selected = {
        item_id: items[item_id].element
        for item_id in selected
        if items[item_id].element is not None
    }
    actual_selected: dict[str, ElementBindingBodyV1] = {}
    for binding in formation.element_bindings:
        if _pin_key(binding.reconstruction) != _pin_key(reconstruction_pin):
            raise ValueError("element binding must pin its exact reconstruction")
        if binding.item_id not in expected_selected:
            raise ValueError("element binding selects an unknown or unselected item")
        if binding.body != expected_selected[binding.item_id]:
            raise ValueError("element binding body must equal its reconstructed item body")
        actual_selected[binding.item_id] = binding.body
    if set(actual_selected) != set(expected_selected):
        raise ValueError("formation bindings must cover every selected bindable item")
    origin_by_identity = {
        (binding.item_id, binding.body.object.object_ref): binding
        for binding in origin_bindings
    }
    for binding in formation.retained_element_bindings:
        key = (binding.item_id, binding.body.object.object_ref)
        if origin_by_identity.get(key) != binding:
            raise ValueError("retained binding must equal a verified origin binding")


def validate_persistence_association(
    manifest: RuntimeContextPersistenceManifestV1,
    receipt: RuntimeContextTransitionReceiptV2,
    successor: RuntimeTaskContextV2,
    formation: FrozenContextFormationV1,
    *,
    related_artifacts: tuple[BaseModel, ...],
    transition: RuntimeContextTransitionV2 | None = None,
    proposal: RuntimeContextTransitionProposalV2 | None = None,
    formation_command: RuntimeContextFormationCommandV1 | None = None,
) -> None:
    manifest_pin = artifact_pin_for(manifest)
    successor_pin = artifact_pin_for(successor)
    formation_pin = artifact_pin_for(formation)
    if _pin_key(receipt.persistence_manifest) != _pin_key(manifest_pin):
        raise ValueError("receipt must pin the exact persistence manifest")
    if receipt.request_hash != runtime_context_persistence_command_hash_v2(manifest):
        raise ValueError("receipt request_hash must bind the exact manifest pin")
    for field in ("organization_id", "workspace_id", "actor_id", "request_id", "operation"):
        if getattr(receipt, field) != getattr(manifest, field):
            raise ValueError(f"receipt and manifest {field} must match")
    if _pin_key(receipt.successor) != _pin_key(successor_pin) or _pin_key(manifest.successor) != _pin_key(successor_pin):
        raise ValueError("receipt and manifest must pin the exact successor")
    if _pin_key(successor.formation) != _pin_key(formation_pin) or _pin_key(manifest.formation) != _pin_key(formation_pin):
        raise ValueError("successor and manifest must pin the exact formation")
    if (
        successor.organization_id != manifest.organization_id
        or successor.workspace_id != manifest.workspace_id
        or formation.organization_id != manifest.organization_id
        or formation.workspace_id != manifest.workspace_id
    ):
        raise ValueError("persistence association scope mismatch")
    related_pins = tuple(
        sorted((artifact_pin_for(value) for value in related_artifacts), key=_pin_key)
    )
    for value in related_artifacts:
        if (
            getattr(value, "organization_id", None) != manifest.organization_id
            or getattr(value, "workspace_id", None) != manifest.workspace_id
        ):
            raise ValueError("related artifact scope mismatch")
    if manifest.operation == "COMMIT":
        if transition is None or proposal is None or formation_command is not None:
            raise ValueError("COMMIT association requires transition/proposal and no command")
        transition_pin = artifact_pin_for(transition)
        proposal_pin = artifact_pin_for(proposal)
        if any(
            _pin_key(left) != _pin_key(right)
            for left, right in (
                (manifest.transition, transition_pin),
                (receipt.transition, transition_pin),
                (manifest.proposal, proposal_pin),
                (transition.proposal, proposal_pin),
                (proposal.formation, formation_pin),
                (transition.formation, formation_pin),
            )
        ):
            raise ValueError("COMMIT association linkage mismatch")
        if _pin_key(receipt.transition) != _pin_key(transition_pin):
            raise ValueError("COMMIT receipt must pin the exact transition")
        if not (
            _pin_key(manifest.expected_parent)
            == _pin_key(proposal.parent)
            == _pin_key(transition.parent)
            == _pin_key(successor.parent)
        ):
            raise ValueError("COMMIT expected parent linkage mismatch")
        if successor.version != manifest.expected_version + 1:
            raise ValueError("COMMIT successor version must follow expected_version")
        if _pin_key(successor.transition) != _pin_key(transition_pin):
            raise ValueError("COMMIT successor must pin the exact transition")
        for value in (transition, proposal):
            if (
                value.organization_id != manifest.organization_id
                or value.workspace_id != manifest.workspace_id
            ):
                raise ValueError("COMMIT transition/proposal scope mismatch")
        if transition.committed_by != manifest.actor_id:
            raise ValueError("COMMIT transition actor mismatch")
    else:
        if transition is not None or proposal is not None or formation_command is None:
            raise ValueError("CREATE/BRANCH association requires only a formation command")
        if (
            successor.version != 0
            or successor.parent is not None
            or successor.transition is not None
        ):
            raise ValueError("CREATE/BRANCH association requires a genesis successor")
        command_pin = artifact_pin_for(formation_command)
        if _pin_key(manifest.formation_command) != _pin_key(command_pin):
            raise ValueError("manifest must pin the exact formation command")
        if _pin_key(formation_command.resolved_query) != _pin_key(formation.query):
            raise ValueError("formation command query must equal formation query")
        if sum(
            _pin_key(pin) == _pin_key(formation_command.resolved_query)
            for pin in related_pins
        ) != 1:
            raise ValueError("related artifacts must contain the exact resolved query once")
        if formation_command.organization_id != manifest.organization_id or formation_command.workspace_id != manifest.workspace_id:
            raise ValueError("formation command scope mismatch")
        if formation_command.actor_id != manifest.actor_id or formation_command.request_id != manifest.request_id:
            raise ValueError("formation command actor/request mismatch")
        if formation_command.operation != manifest.operation:
            raise ValueError("formation command operation mismatch")
        if manifest.operation == "CREATE":
            if formation_command.origin is not None or successor.branch_origin is not None:
                raise ValueError("CREATE association requires null origin")
        elif formation_command.origin is None or successor.branch_origin is None or _pin_key(formation_command.origin) != _pin_key(successor.branch_origin):
            raise ValueError("BRANCH association origin mismatch")
        if formation_command.session_ref != successor.session_ref:
            raise ValueError("formation command session must equal successor session")
    if related_pins != manifest.related_artifacts:
        raise ValueError("manifest related_artifacts must equal the complete supplied set")
