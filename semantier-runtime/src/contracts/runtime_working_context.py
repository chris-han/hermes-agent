"""Strict contracts for immutable runtime task-context artifacts.

The Python type names follow ADR-0027: this family is session/stream-scoped task state,
not the decision-relative semantic-kernel ``WorkingContext``. Persisted schema versions
and artifact kinds retain the historical ``runtime_working_context`` wire identifiers for
deterministic replay of existing artifacts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Annotated, Literal, Mapping, TypeAlias

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
RuntimeContextElementRole: TypeAlias = Literal[
    "goal",
    "entity",
    "event",
    "observation",
    "hypothesis",
    "constraint",
    "action",
    "decision",
    "outcome",
    "justification",
]


def _strict_v2_datetime_input(value: object) -> object:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string or datetime")
    try:
        float(value)
    except ValueError:
        return value
    raise ValueError("timestamp must not be a numeric epoch string")


V2UtcDateTime = Annotated[datetime, BeforeValidator(_strict_v2_datetime_input)]


class _RuntimeContextContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def persisted_timestamps_are_utc(self) -> "_RuntimeContextContract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime):
                if value.tzinfo is None or value.utcoffset() != timedelta(0):
                    raise ValueError(f"{name} must be timezone-aware UTC")
        return self


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


def _basis(value: Mapping[str, object], *identity_fields: str) -> dict[str, object]:
    return {
        key: _json_value(item)
        for key, item in value.items()
        if key not in identity_fields
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


def _ordered_pins(values: object) -> object:
    if not isinstance(values, (list, tuple)):
        return values
    ordered = sorted(values, key=_pin_key)
    return tuple(ordered) if isinstance(values, tuple) else ordered


def _normalize_element(value: object) -> dict[str, object]:
    element = dict(value) if isinstance(value, Mapping) else {}
    element["evidence"] = _ordered_pins(element.get("evidence", []))
    return element


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
        raise ValueError(f"{label} identity does not bind full semantic content")


class RuntimeWorldScopeV1(_RuntimeContextContract):
    schema_version: Literal["runtime_world_scope.v1"] = "runtime_world_scope.v1"
    world_ref: str = Field(min_length=1)
    world_hash: str = Field(pattern=SHA256_PATTERN)
    source_graph: ArtifactPinV1
    mode: Literal["non_temporal", "point", "interval"]
    at_time: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    interval_bounds: Literal["closed"] = "closed"

    @model_validator(mode="after")
    def validate_time_shape(self) -> "RuntimeWorldScopeV1":
        if self.mode == "non_temporal":
            if any(value is not None for value in (self.at_time, self.start_time, self.end_time)):
                raise ValueError("non-temporal worlds must not contain times")
        elif self.mode == "point":
            if self.at_time is None or self.start_time is not None or self.end_time is not None:
                raise ValueError("point worlds require only at_time")
        elif (
            self.at_time is not None
            or self.start_time is None
            or self.end_time is None
            or self.start_time >= self.end_time
        ):
            raise ValueError("interval worlds require ordered start/end and no at_time")
        return self

    @model_validator(mode="after")
    def verify_hash_binding(self) -> "RuntimeWorldScopeV1":
        _verify_identity(
            prefix="runtime-world",
            artifact_ref=self.world_ref,
            artifact_hash=self.world_hash,
            basis=world_hash_basis(self.model_dump(mode="json")),
            label="world",
        )
        return self


class RuntimeContextElementV1(_RuntimeContextContract):
    schema_version: Literal["runtime_context_element.v1"] = "runtime_context_element.v1"
    object_ref: str = Field(min_length=1)
    source_artifact: ArtifactPinV1
    role: RuntimeContextElementRole
    admission: Literal["candidate", "admitted"]
    active: bool
    derived: bool
    evidence: tuple[ArtifactPinV1, ...] = ()
    admission_receipt: ArtifactPinV1 | None = None
    derivation: ArtifactPinV1 | None = None

    @field_validator("evidence")
    @classmethod
    def canonically_order_evidence(cls, value: object) -> object:
        return _ordered_pins(value)

    @model_validator(mode="after")
    def validate_state_proofs(self) -> "RuntimeContextElementV1":
        if self.active and self.admission != "admitted":
            raise ValueError("active elements must be admitted")
        if self.active and self.admission_receipt is None:
            raise ValueError("active elements require an admission receipt")
        if self.admission == "admitted" and self.admission_receipt is None:
            raise ValueError("admitted elements require an admission receipt")
        if self.admission == "candidate" and self.admission_receipt is not None:
            raise ValueError("candidate elements must not have an admission receipt")
        if self.derived and not self.evidence:
            raise ValueError("derived elements require evidence")
        if self.derived and self.derivation is None:
            raise ValueError("derived elements require a derivation pin")
        if not self.derived and self.derivation is not None:
            raise ValueError("non-derived elements must not have a derivation pin")
        return self


class AddCandidateOperationV1(_RuntimeContextContract):
    operation: Literal["ADD_CANDIDATE"] = "ADD_CANDIDATE"
    element: RuntimeContextElementV1

    @model_validator(mode="after")
    def require_candidate_input(self) -> "AddCandidateOperationV1":
        if self.element.admission != "candidate" or self.element.active:
            raise ValueError("ADD_CANDIDATE input must be candidate and inactive")
        return self


class AdmitOperationV1(_RuntimeContextContract):
    operation: Literal["ADMIT"] = "ADMIT"
    object_ref: str = Field(min_length=1)
    admission_receipt: ArtifactPinV1


class ActivateOperationV1(_RuntimeContextContract):
    operation: Literal["ACTIVATE"] = "ACTIVATE"
    object_ref: str = Field(min_length=1)


class DeactivateOperationV1(_RuntimeContextContract):
    operation: Literal["DEACTIVATE"] = "DEACTIVATE"
    object_ref: str = Field(min_length=1)


class SuspendOperationV1(_RuntimeContextContract):
    operation: Literal["SUSPEND"] = "SUSPEND"


class CloseOperationV1(_RuntimeContextContract):
    operation: Literal["CLOSE"] = "CLOSE"


RuntimeContextOperationV1: TypeAlias = Annotated[
    AddCandidateOperationV1
    | AdmitOperationV1
    | ActivateOperationV1
    | DeactivateOperationV1
    | SuspendOperationV1
    | CloseOperationV1,
    Field(discriminator="operation"),
]


class RuntimeTaskContextV1(_RuntimeContextContract):
    schema_version: Literal["runtime_working_context.v1"] = "runtime_working_context.v1"
    context_ref: str = Field(min_length=1)
    context_hash: str = Field(pattern=SHA256_PATTERN)
    stream_ref: str = Field(min_length=1)
    version: int = Field(ge=0)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    session_ref: str = Field(min_length=1)
    goal: ArtifactPinV1
    world: RuntimeWorldScopeV1
    semantic_pins: tuple[ArtifactPinV1, ...]
    elements: tuple[RuntimeContextElementV1, ...]
    parent: ArtifactPinV1 | None = None
    branch_origin: ArtifactPinV1 | None = None
    transition: ArtifactPinV1 | None = None
    status: Literal["OPEN", "SUSPENDED", "CLOSED"]
    created_by: str = Field(min_length=1)
    created_at: datetime

    @field_validator("semantic_pins")
    @classmethod
    def canonically_order_semantic_pins(cls, value: object) -> object:
        return _ordered_pins(value)

    @field_validator("elements", mode="before")
    @classmethod
    def canonically_order_elements(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        elements = list(value)

        def object_ref(element: object) -> str:
            if isinstance(element, BaseModel):
                return str(getattr(element, "object_ref", ""))
            if isinstance(element, Mapping):
                return str(element.get("object_ref", ""))
            return ""

        return sorted(elements, key=object_ref)

    @model_validator(mode="after")
    def validate_lineage_and_elements(self) -> "RuntimeTaskContextV1":
        if self.version == 0:
            if self.parent is not None or self.transition is not None:
                raise ValueError("genesis contexts require null parent and transition")
        else:
            if self.parent is None or self.transition is None:
                raise ValueError("later contexts require paired parent and transition")
            if self.branch_origin is not None:
                raise ValueError("only branch genesis may pin branch_origin")
        object_refs = [element.object_ref for element in self.elements]
        if len(object_refs) != len(set(object_refs)):
            raise ValueError("context elements contain duplicate object_ref values")
        return self

    @model_validator(mode="after")
    def verify_hash_binding(self) -> "RuntimeTaskContextV1":
        _verify_identity(
            prefix="runtime-context",
            artifact_ref=self.context_ref,
            artifact_hash=self.context_hash,
            basis=context_hash_basis(self.model_dump(mode="json")),
            label="context",
        )
        return self


class RuntimeContextTransitionProposalV1(_RuntimeContextContract):
    schema_version: Literal["runtime_context_transition_proposal.v1"] = "runtime_context_transition_proposal.v1"
    proposal_ref: str = Field(min_length=1)
    proposal_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    parent: ArtifactPinV1
    trigger: ArtifactPinV1
    provider: ArtifactPinV1
    operations: tuple[RuntimeContextOperationV1, ...] = Field(min_length=1)
    evidence: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    justifications: tuple[ArtifactPinV1, ...]
    required_governance: tuple[ArtifactPinV1, ...]
    proposed_by: str = Field(min_length=1)
    proposed_at: datetime

    @field_validator("evidence", "justifications", "required_governance")
    @classmethod
    def canonically_order_pin_sets(cls, value: object) -> object:
        return _ordered_pins(value)

    @model_validator(mode="after")
    def verify_hash_binding(self) -> "RuntimeContextTransitionProposalV1":
        _verify_identity(
            prefix="runtime-context-proposal",
            artifact_ref=self.proposal_ref,
            artifact_hash=self.proposal_hash,
            basis=proposal_hash_basis(self.model_dump(mode="json")),
            label="proposal",
        )
        return self


class RuntimeContextTransitionV1(_RuntimeContextContract):
    schema_version: Literal["runtime_context_transition.v1"] = "runtime_context_transition.v1"
    transition_ref: str = Field(min_length=1)
    transition_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    parent: ArtifactPinV1
    proposal: ArtifactPinV1
    governance_receipts: tuple[ArtifactPinV1, ...]
    committed_by: str = Field(min_length=1)
    committed_at: datetime

    @field_validator("governance_receipts")
    @classmethod
    def canonically_order_governance_receipts(cls, value: object) -> object:
        return _ordered_pins(value)

    @model_validator(mode="after")
    def verify_hash_binding(self) -> "RuntimeContextTransitionV1":
        _verify_identity(
            prefix="runtime-context-transition",
            artifact_ref=self.transition_ref,
            artifact_hash=self.transition_hash,
            basis=transition_hash_basis(self.model_dump(mode="json")),
            label="transition",
        )
        return self


class RuntimeContextTransitionReceiptV1(_RuntimeContextContract):
    schema_version: Literal["runtime_context_transition_receipt.v1"] = "runtime_context_transition_receipt.v1"
    receipt_ref: str = Field(min_length=1)
    receipt_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    operation: Literal["CREATE", "BRANCH", "COMMIT"]
    request_id: str = Field(min_length=1)
    request_hash: str = Field(pattern=SHA256_PATTERN)
    transition: ArtifactPinV1 | None = None
    successor: ArtifactPinV1
    recorded_at: datetime

    @model_validator(mode="after")
    def validate_operation(self) -> "RuntimeContextTransitionReceiptV1":
        if self.operation == "COMMIT" and self.transition is None:
            raise ValueError("COMMIT receipts require a transition")
        if self.operation in {"CREATE", "BRANCH"} and self.transition is not None:
            raise ValueError("CREATE and BRANCH receipts must not pin a transition")
        return self

    @model_validator(mode="after")
    def verify_hash_binding(self) -> "RuntimeContextTransitionReceiptV1":
        _verify_identity(
            prefix="runtime-context-receipt",
            artifact_ref=self.receipt_ref,
            artifact_hash=self.receipt_hash,
            basis=receipt_hash_basis(self.model_dump(mode="json")),
            label="receipt",
        )
        return self


class RuntimeTaskContextV2(RuntimeTaskContextV1):
    schema_version: Literal["runtime_working_context.v2"]
    version: Annotated[int, Field(strict=True, ge=0)]
    created_at: V2UtcDateTime
    formation: ArtifactPinV1

    @field_validator("formation")
    @classmethod
    def validate_formation_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        if value.artifact_kind != "frozen_context_formation":
            raise ValueError("formation must pin a frozen_context_formation")
        return value


class RuntimeContextTransitionProposalV2(RuntimeContextTransitionProposalV1):
    schema_version: Literal["runtime_context_transition_proposal.v2"]
    proposed_at: V2UtcDateTime
    formation: ArtifactPinV1

    @field_validator("formation")
    @classmethod
    def validate_formation_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        if value.artifact_kind != "frozen_context_formation":
            raise ValueError("formation must pin a frozen_context_formation")
        return value


class RuntimeContextTransitionV2(RuntimeContextTransitionV1):
    schema_version: Literal["runtime_context_transition.v2"]
    committed_at: V2UtcDateTime
    formation: ArtifactPinV1

    @field_validator("formation")
    @classmethod
    def validate_formation_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        if value.artifact_kind != "frozen_context_formation":
            raise ValueError("formation must pin a frozen_context_formation")
        return value


class RuntimeContextTransitionReceiptV2(RuntimeContextTransitionReceiptV1):
    schema_version: Literal["runtime_context_transition_receipt.v2"]
    recorded_at: V2UtcDateTime
    persistence_manifest: ArtifactPinV1

    @field_validator("persistence_manifest")
    @classmethod
    def validate_manifest_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        if value.artifact_kind != "runtime_context_persistence_manifest":
            raise ValueError(
                "persistence_manifest must pin a runtime_context_persistence_manifest"
            )
        return value


class RuntimeTaskContextV3(RuntimeTaskContextV1):
    """Candidate-only context created from one explicit temporal selection."""

    schema_version: Literal["runtime_working_context.v3"] = "runtime_working_context.v3"
    version: Annotated[int, Field(strict=True, ge=0)]
    created_at: V2UtcDateTime
    selection: ArtifactPinV1

    @field_validator("selection")
    @classmethod
    def validate_selection_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        if value.artifact_kind != "runtime_temporal_context_selection":
            raise ValueError("selection must pin runtime_temporal_context_selection")
        return value

    @model_validator(mode="after")
    def validate_temporal_genesis(self) -> "RuntimeTaskContextV3":
        if self.version != 0 or self.parent is not None or self.branch_origin is not None or self.transition is not None:
            raise ValueError("temporal V3 is CREATE-only version zero")
        return self


class RuntimeContextTransitionReceiptV3(RuntimeContextTransitionReceiptV1):
    """Typed receipt for the closed V3 temporal CREATE publisher."""

    schema_version: Literal["runtime_context_transition_receipt.v3"] = "runtime_context_transition_receipt.v3"
    operation: Literal["CREATE"]
    recorded_at: V2UtcDateTime
    transition: None
    selection: ArtifactPinV1

    @field_validator("selection")
    @classmethod
    def validate_selection_kind(cls, value: ArtifactPinV1) -> ArtifactPinV1:
        if value.artifact_kind != "runtime_temporal_context_selection":
            raise ValueError("selection must pin runtime_temporal_context_selection")
        return value


def context_hash_basis(value: Mapping[str, object]) -> dict[str, object]:
    basis = _basis(value, "context_ref", "context_hash")
    basis["semantic_pins"] = _ordered_pins(basis.get("semantic_pins", []))
    elements = basis.get("elements", [])
    basis["elements"] = sorted(
        (_normalize_element(element) for element in elements),
        key=lambda element: str(element.get("object_ref", "")),
    )
    return basis


def world_hash_basis(value: Mapping[str, object]) -> dict[str, object]:
    return _basis(value, "world_ref", "world_hash")


def proposal_hash_basis(value: Mapping[str, object]) -> dict[str, object]:
    basis = _basis(value, "proposal_ref", "proposal_hash")
    for field in ("evidence", "justifications", "required_governance"):
        basis[field] = _ordered_pins(basis.get(field, []))
    operations = basis.get("operations", [])
    normalized_operations: list[object] = []
    for operation_value in operations:
        operation = dict(operation_value) if isinstance(operation_value, Mapping) else {}
        if operation.get("operation") == "ADD_CANDIDATE":
            operation["element"] = _normalize_element(operation.get("element", {}))
        normalized_operations.append(operation)
    basis["operations"] = normalized_operations
    return basis


def transition_hash_basis(value: Mapping[str, object]) -> dict[str, object]:
    basis = _basis(value, "transition_ref", "transition_hash")
    basis["governance_receipts"] = _ordered_pins(
        basis.get("governance_receipts", [])
    )
    return basis


def receipt_hash_basis(value: Mapping[str, object]) -> dict[str, object]:
    return _basis(value, "receipt_ref", "receipt_hash")


RUNTIME_CONTEXT_MODEL_BY_SCHEMA = MappingProxyType(
    {
        "runtime_working_context.v1": RuntimeTaskContextV1,
        "runtime_working_context.v2": RuntimeTaskContextV2,
        "runtime_context_transition_proposal.v1": RuntimeContextTransitionProposalV1,
        "runtime_context_transition_proposal.v2": RuntimeContextTransitionProposalV2,
        "runtime_context_transition.v1": RuntimeContextTransitionV1,
        "runtime_context_transition.v2": RuntimeContextTransitionV2,
        "runtime_context_transition_receipt.v1": RuntimeContextTransitionReceiptV1,
        "runtime_context_transition_receipt.v2": RuntimeContextTransitionReceiptV2,
    }
)

RUNTIME_CONTEXT_ARTIFACT_KIND_BY_SCHEMA = MappingProxyType(
    {
        "runtime_working_context.v1": "runtime_working_context",
        "runtime_working_context.v2": "runtime_working_context",
        "runtime_context_transition_proposal.v1": "runtime_context_transition_proposal",
        "runtime_context_transition_proposal.v2": "runtime_context_transition_proposal",
        "runtime_context_transition.v1": "runtime_context_transition",
        "runtime_context_transition.v2": "runtime_context_transition",
        "runtime_context_transition_receipt.v1": "runtime_context_transition_receipt",
        "runtime_context_transition_receipt.v2": "runtime_context_transition_receipt",
    }
)
