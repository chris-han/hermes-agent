"""Self-bound contracts for runtime-context provider and admission authority."""

from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import Literal, Mapping, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_working_context import RuntimeContextElementRole
from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ASCII_MACHINE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")

RuntimeContextOperationName: TypeAlias = Literal[
    "ADD_CANDIDATE",
    "ADMIT",
    "ACTIVATE",
    "DEACTIVATE",
    "SUSPEND",
    "CLOSE",
]


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        encoded = value.isoformat()
        return encoded.replace("+00:00", "Z") if value.utcoffset() == timedelta(0) else encoded
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


def _rule_key(
    value: "RuntimeContextAdmissionRuleV1" | Mapping[str, object],
) -> tuple[str, str, str]:
    payload = _mapping(value)
    return (
        str(payload["source_artifact_kind"]),
        str(payload["source_object_kind"]),
        str(payload["context_role"]),
    )


def _sorted_strings(values: object) -> object:
    if not isinstance(values, (list, tuple)):
        return values
    ordered = sorted(values)
    return tuple(ordered) if isinstance(values, tuple) else ordered


def _sorted_pins(values: object) -> object:
    if not isinstance(values, (list, tuple)):
        return values
    ordered = sorted(values, key=_pin_key)
    return tuple(ordered) if isinstance(values, tuple) else ordered


def _normalize_rule(value: object) -> object:
    payload = _mapping(value) if isinstance(value, (BaseModel, Mapping)) else value
    if isinstance(payload, dict):
        payload["required_governance_kinds"] = _sorted_strings(
            payload.get("required_governance_kinds", [])
        )
    return payload


def proposal_provider_descriptor_hash_basis(
    descriptor: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    basis = _mapping(descriptor)
    basis.pop("descriptor_ref", None)
    basis.pop("descriptor_hash", None)
    configuration = basis.get("configuration")
    if isinstance(configuration, Mapping):
        normalized = dict(configuration)
        normalized["supported_operations"] = _sorted_strings(
            normalized.get("supported_operations", [])
        )
        basis["configuration"] = normalized
    return basis


def scope_policy_hash_basis(
    policy: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    basis = _mapping(policy)
    basis.pop("policy_ref", None)
    basis.pop("policy_hash", None)
    for field in (
        "allowed_goal_kinds",
        "allowed_world_source_kinds",
        "allowed_element_source_kinds",
        "allowed_trigger_kinds",
        "allowed_evidence_kinds",
        "allowed_justification_kinds",
        "allowed_governance_kinds",
        "allowed_operations",
    ):
        basis[field] = _sorted_strings(basis.get(field, []))
    for field in ("allowed_provider_descriptors", "governing_receipts"):
        basis[field] = _sorted_pins(basis.get(field, []))
    rules = basis.get("admission_rules", [])
    if isinstance(rules, (list, tuple)):
        basis["admission_rules"] = sorted(
            (_normalize_rule(rule) for rule in rules), key=_rule_key
        )
    return basis


def element_admission_hash_basis(
    receipt: BaseModel | Mapping[str, object],
) -> dict[str, object]:
    basis = _mapping(receipt)
    basis.pop("admission_ref", None)
    basis.pop("admission_hash", None)
    for field in ("governing_receipts", "evidence"):
        basis[field] = _sorted_pins(basis.get(field, []))
    return basis


def _verify_identity(
    *, prefix: str, artifact_ref: str, artifact_hash: str, basis: Mapping[str, object]
) -> None:
    digest = content_hash(dict(basis))
    if (artifact_ref, artifact_hash) != (
        f"{prefix}:{digest}",
        f"sha256:{digest}",
    ):
        raise ValueError("artifact identity does not bind full semantic content")


def _ensure_unique(values: tuple[object, ...], label: str) -> None:
    normalized = [
        item.model_dump_json() if isinstance(item, BaseModel) else repr(item)
        for item in values
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} contains duplicate values")


def _ensure_machine_id(value: str, label: str) -> str:
    if _ASCII_MACHINE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a non-empty ASCII-stable machine identifier")
    return value


class _AuthorityContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_utc_timestamps(self) -> "_AuthorityContract":
        for field in type(self).model_fields:
            value = getattr(self, field)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{field} must be timezone-aware UTC")
        return self


class RuntimeContextExplicitOperationsProviderConfigV1(_AuthorityContract):
    operation_schema_version: Literal["runtime_context_operation.v1"]
    supported_operations: tuple[RuntimeContextOperationName, ...] = Field(min_length=1)
    max_operations_per_proposal: int = Field(gt=0)

    @field_validator("supported_operations")
    @classmethod
    def order_operations(cls, value: object) -> object:
        return _sorted_strings(value)

    @model_validator(mode="after")
    def validate_operations(self) -> "RuntimeContextExplicitOperationsProviderConfigV1":
        _ensure_unique(tuple(self.supported_operations), "supported_operations")
        if len(self.supported_operations) > self.max_operations_per_proposal:
            raise ValueError("supported_operations exceeds max_operations_per_proposal")
        return self


class RuntimeContextProposalProviderDescriptorV1(_AuthorityContract):
    schema_version: Literal["runtime_context_proposal_provider_descriptor.v1"] = (
        "runtime_context_proposal_provider_descriptor.v1"
    )
    descriptor_ref: str = Field(min_length=1)
    descriptor_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    provider_kind: Literal["runtime_context_explicit_operations"]
    implementation_id: Literal["semantier.runtime_context.explicit_operations"]
    implementation_version: Literal["v1"]
    implementation_hash: str = Field(pattern=SHA256_PATTERN)
    configuration: RuntimeContextExplicitOperationsProviderConfigV1
    configuration_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_configuration_and_identity(
        self,
    ) -> "RuntimeContextProposalProviderDescriptorV1":
        expected_configuration_hash = "sha256:" + content_hash(
            self.configuration.model_dump(mode="json")
        )
        if self.configuration_hash != expected_configuration_hash:
            raise ValueError("configuration_hash does not bind complete configuration")
        _verify_identity(
            prefix="runtime-context-proposal-provider",
            artifact_ref=self.descriptor_ref,
            artifact_hash=self.descriptor_hash,
            basis=proposal_provider_descriptor_hash_basis(self),
        )
        return self


class RuntimeContextAdmissionRuleV1(_AuthorityContract):
    source_artifact_kind: str = Field(min_length=1)
    source_object_kind: str = Field(min_length=1)
    context_role: RuntimeContextElementRole
    candidate_allowed: bool
    admission_authorization: Literal["TENANT_MEMBER", "TENANT_ADMIN"]
    evidence_required: bool
    required_governance_kinds: tuple[str, ...]

    @field_validator("required_governance_kinds")
    @classmethod
    def order_governance_kinds(cls, value: object) -> object:
        return _sorted_strings(value)

    @field_validator(
        "source_artifact_kind", "source_object_kind", "required_governance_kinds"
    )
    @classmethod
    def validate_machine_ids(cls, value: object) -> object:
        if isinstance(value, tuple):
            _ensure_unique(tuple(value), "required_governance_kinds")
            return tuple(_ensure_machine_id(item, "governance kind") for item in value)
        return _ensure_machine_id(str(value), "rule kind")


class RuntimeContextScopePolicyV1(_AuthorityContract):
    schema_version: Literal["runtime_context_scope_policy.v1"] = (
        "runtime_context_scope_policy.v1"
    )
    policy_ref: str = Field(min_length=1)
    policy_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    allowed_goal_kinds: tuple[str, ...] = Field(min_length=1)
    allowed_world_source_kinds: tuple[str, ...] = Field(min_length=1)
    allowed_element_source_kinds: tuple[str, ...] = Field(min_length=1)
    allowed_trigger_kinds: tuple[str, ...] = Field(min_length=1)
    allowed_provider_descriptors: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    allowed_evidence_kinds: tuple[str, ...] = Field(min_length=1)
    allowed_justification_kinds: tuple[str, ...]
    allowed_governance_kinds: tuple[str, ...] = Field(min_length=1)
    admission_rules: tuple[RuntimeContextAdmissionRuleV1, ...] = Field(min_length=1)
    allowed_operations: tuple[RuntimeContextOperationName, ...] = Field(min_length=1)
    max_elements: int = Field(gt=0)
    max_operations: int = Field(gt=0)
    governing_receipts: tuple[ArtifactPinV1, ...]
    supersedes_policy: ArtifactPinV1 | None
    authority_class: Literal["TENANT_ADMIN"]
    issued_by: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    request_hash: str = Field(pattern=SHA256_PATTERN)
    effective_from: datetime
    effective_to: datetime
    issued_at: datetime

    @field_validator(
        "allowed_goal_kinds",
        "allowed_world_source_kinds",
        "allowed_element_source_kinds",
        "allowed_trigger_kinds",
        "allowed_evidence_kinds",
        "allowed_justification_kinds",
        "allowed_governance_kinds",
        "allowed_operations",
    )
    @classmethod
    def order_string_sets(cls, value: object) -> object:
        return _sorted_strings(value)

    @field_validator("allowed_provider_descriptors", "governing_receipts")
    @classmethod
    def order_pin_sets(cls, value: object) -> object:
        return _sorted_pins(value)

    @field_validator("admission_rules")
    @classmethod
    def order_rules(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        ordered = sorted(value, key=_rule_key)
        return tuple(ordered) if isinstance(value, tuple) else ordered

    @model_validator(mode="after")
    def validate_policy(self) -> "RuntimeContextScopePolicyV1":
        for field in (
            "allowed_goal_kinds",
            "allowed_world_source_kinds",
            "allowed_element_source_kinds",
            "allowed_trigger_kinds",
            "allowed_evidence_kinds",
            "allowed_justification_kinds",
            "allowed_governance_kinds",
            "allowed_operations",
        ):
            values = tuple(getattr(self, field))
            _ensure_unique(values, field)
            if field != "allowed_operations":
                for value in values:
                    _ensure_machine_id(str(value), field)
        _ensure_unique(tuple(self.allowed_provider_descriptors), "allowed_provider_descriptors")
        _ensure_unique(tuple(self.governing_receipts), "governing_receipts")
        for pin in (
            *self.allowed_provider_descriptors,
            *self.governing_receipts,
            *((self.supersedes_policy,) if self.supersedes_policy is not None else ()),
        ):
            _ensure_machine_id(pin.artifact_kind, "artifact kind")
        rule_keys = [_rule_key(rule) for rule in self.admission_rules]
        if len(rule_keys) != len(set(rule_keys)):
            raise ValueError("admission_rules contains duplicate rule keys")
        allowed_sources = set(self.allowed_element_source_kinds)
        if any(rule.source_artifact_kind not in allowed_sources for rule in self.admission_rules):
            raise ValueError("admission rule source must occur in allowed_element_source_kinds")
        if any(
            pin.artifact_kind != "runtime_context_proposal_provider_descriptor"
            for pin in self.allowed_provider_descriptors
        ):
            raise ValueError("allowed provider descriptor pin has the wrong artifact kind")
        if self.supersedes_policy is not None and (
            self.supersedes_policy.artifact_kind != "runtime_context_scope_policy"
        ):
            raise ValueError("supersedes policy pin has the wrong artifact kind")
        if self.effective_from < self.issued_at:
            raise ValueError("effective_from must not predate issued_at")
        if self.effective_to <= self.effective_from:
            raise ValueError("effective_to must follow effective_from")
        _verify_identity(
            prefix="runtime-context-policy",
            artifact_ref=self.policy_ref,
            artifact_hash=self.policy_hash,
            basis=scope_policy_hash_basis(self),
        )
        return self


class RuntimeContextElementAdmissionReceiptV1(_AuthorityContract):
    schema_version: Literal["runtime_context_element_admission.v1"] = (
        "runtime_context_element_admission.v1"
    )
    admission_ref: str = Field(min_length=1)
    admission_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    decision: Literal["ADMIT"]
    object_ref: str = Field(min_length=1)
    source_artifact: ArtifactPinV1
    source_object_kind: str = Field(min_length=1)
    context_role: RuntimeContextElementRole
    scope_policy: ArtifactPinV1
    admission_authorization: Literal["TENANT_MEMBER", "TENANT_ADMIN"]
    governing_receipts: tuple[ArtifactPinV1, ...]
    evidence: tuple[ArtifactPinV1, ...]
    admitted_by: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    request_hash: str = Field(pattern=SHA256_PATTERN)
    admitted_at: datetime

    @field_validator("governing_receipts", "evidence")
    @classmethod
    def order_pin_sets(cls, value: object) -> object:
        return _sorted_pins(value)

    @field_validator("source_object_kind")
    @classmethod
    def validate_source_object_kind(cls, value: str) -> str:
        return _ensure_machine_id(value, "source_object_kind")

    @model_validator(mode="after")
    def validate_admission(self) -> "RuntimeContextElementAdmissionReceiptV1":
        _ensure_unique(tuple(self.governing_receipts), "governing_receipts")
        _ensure_unique(tuple(self.evidence), "evidence")
        for pin in (
            self.source_artifact,
            self.scope_policy,
            *self.governing_receipts,
            *self.evidence,
        ):
            _ensure_machine_id(pin.artifact_kind, "artifact kind")
        if self.scope_policy.artifact_kind != "runtime_context_scope_policy":
            raise ValueError("scope policy pin has the wrong artifact kind")
        _verify_identity(
            prefix="runtime-context-admission",
            artifact_ref=self.admission_ref,
            artifact_hash=self.admission_hash,
            basis=element_admission_hash_basis(self),
        )
        return self


__all__ = [
    "RuntimeContextAdmissionRuleV1",
    "RuntimeContextElementAdmissionReceiptV1",
    "RuntimeContextExplicitOperationsProviderConfigV1",
    "RuntimeContextOperationName",
    "RuntimeContextProposalProviderDescriptorV1",
    "RuntimeContextScopePolicyV1",
    "element_admission_hash_basis",
    "proposal_provider_descriptor_hash_basis",
    "scope_policy_hash_basis",
]
