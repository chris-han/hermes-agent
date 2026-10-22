"""Server-owned issuance of runtime-context policy and element authority."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.knowledge_classification import RuntimeContextDTO
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_context_authority import (
    RuntimeContextAdmissionRuleV1,
    RuntimeContextElementAdmissionReceiptV1,
    RuntimeContextExplicitOperationsProviderConfigV1,
    RuntimeContextOperationName,
    RuntimeContextProposalProviderDescriptorV1,
    RuntimeContextScopePolicyV1,
    element_admission_hash_basis,
    scope_policy_hash_basis,
)
from contracts.runtime_working_context import RuntimeContextElementRole
from eos.artifact_hashing import content_hash
from services.runtime_context_proposal_provider import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    RuntimeContextProposalProviderRegistry,
)
from services.runtime_context_transition_service import (
    ResolvedRuntimeObject,
    RuntimeAuthorityResolver,
)
from storage.runtime_context_authority_store import RuntimeContextAuthorityStore
from storage.runtime_working_context_store import RuntimeContextError


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be timezone-aware UTC")
    return value


def _pin_key(pin: ArtifactPinV1) -> tuple[str, str, str]:
    return pin.artifact_kind, pin.artifact_ref, pin.artifact_hash


def _rule_key(
    rule: RuntimeContextAdmissionRuleV1,
) -> tuple[str, str, RuntimeContextElementRole]:
    return rule.source_artifact_kind, rule.source_object_kind, rule.context_role


class _Command(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "_Command":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime):
                _utc(value, name)
        return self


class IssueRuntimeContextScopePolicyCommandV1(_Command):
    schema_version: Literal["issue_runtime_context_scope_policy_command.v1"] = (
        "issue_runtime_context_scope_policy_command.v1"
    )
    request_id: str = Field(min_length=1)
    allowed_goal_kinds: tuple[str, ...]
    allowed_world_source_kinds: tuple[str, ...]
    allowed_element_source_kinds: tuple[str, ...]
    allowed_trigger_kinds: tuple[str, ...]
    allowed_provider_descriptors: tuple[ArtifactPinV1, ...]
    allowed_evidence_kinds: tuple[str, ...]
    allowed_justification_kinds: tuple[str, ...]
    allowed_governance_kinds: tuple[str, ...]
    admission_rules: tuple[RuntimeContextAdmissionRuleV1, ...]
    allowed_operations: tuple[RuntimeContextOperationName, ...]
    max_elements: int = Field(gt=0)
    max_operations: int = Field(gt=0)
    governing_receipts: tuple[ArtifactPinV1, ...]
    supersedes_policy: ArtifactPinV1 | None
    effective_from: datetime | None
    effective_to: datetime

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
    def canonicalize_string_sets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("policy command sets must contain unique values")
        return tuple(sorted(value))

    @field_validator("allowed_provider_descriptors", "governing_receipts")
    @classmethod
    def canonicalize_pin_sets(
        cls, value: tuple[ArtifactPinV1, ...]
    ) -> tuple[ArtifactPinV1, ...]:
        if len(value) != len(set(_pin_key(pin) for pin in value)):
            raise ValueError("policy command pins must be unique")
        return tuple(sorted(value, key=_pin_key))

    @field_validator("admission_rules")
    @classmethod
    def canonicalize_rules(
        cls, value: tuple[RuntimeContextAdmissionRuleV1, ...]
    ) -> tuple[RuntimeContextAdmissionRuleV1, ...]:
        if len(value) != len(set(_rule_key(rule) for rule in value)):
            raise ValueError("policy command admission rules must be unique")
        return tuple(sorted(value, key=_rule_key))

    @model_validator(mode="after")
    def validate_explicit_interval(self) -> "IssueRuntimeContextScopePolicyCommandV1":
        if self.effective_from is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must follow effective_from")
        return self


class AdmitRuntimeContextElementCommandV1(_Command):
    schema_version: Literal["admit_runtime_context_element_command.v1"] = (
        "admit_runtime_context_element_command.v1"
    )
    request_id: str = Field(min_length=1)
    object_ref: str = Field(min_length=1)
    source_artifact: ArtifactPinV1
    context_role: RuntimeContextElementRole
    scope_policy: ArtifactPinV1
    governing_receipts: tuple[ArtifactPinV1, ...]
    evidence: tuple[ArtifactPinV1, ...]

    @field_validator("governing_receipts", "evidence")
    @classmethod
    def reject_duplicate_pins(
        cls, value: tuple[ArtifactPinV1, ...]
    ) -> tuple[ArtifactPinV1, ...]:
        if len(value) != len(set(_pin_key(pin) for pin in value)):
            raise ValueError("proof pins must be unique")
        return tuple(sorted(value, key=_pin_key))


def _request_hash(payload: dict[str, object]) -> str:
    return "sha256:" + content_hash(payload)


def _policy_request_basis(
    *,
    actor_context: RuntimeContextDTO,
    command: IssueRuntimeContextScopePolicyCommandV1,
) -> dict[str, object]:
    return {
        "schema_version": command.schema_version,
        "organization_id": actor_context.organization_id,
        "workspace_id": actor_context.workspace_id,
        "actor_id": actor_context.actor_id,
        "operation": "ISSUE_SCOPE_POLICY",
        **command.model_dump(
            mode="json", exclude={"schema_version"}, exclude_none=False
        ),
    }


def _admission_request_basis(
    *,
    actor_context: RuntimeContextDTO,
    command: AdmitRuntimeContextElementCommandV1,
    resolved_object: ResolvedRuntimeObject,
    matched_rule: RuntimeContextAdmissionRuleV1,
) -> dict[str, object]:
    return {
        "schema_version": command.schema_version,
        "organization_id": actor_context.organization_id,
        "workspace_id": actor_context.workspace_id,
        "actor_id": actor_context.actor_id,
        "operation": "ADMIT_CONTEXT_ELEMENT",
        "request_id": command.request_id,
        "object_ref": command.object_ref,
        "source_artifact": command.source_artifact.model_dump(mode="json"),
        "source_object_kind": resolved_object.source_object_kind,
        "context_role": command.context_role,
        "scope_policy": command.scope_policy.model_dump(mode="json"),
        "admission_authorization": matched_rule.admission_authorization,
        "governing_receipts": [
            pin.model_dump(mode="json") for pin in command.governing_receipts
        ],
        "evidence": [pin.model_dump(mode="json") for pin in command.evidence],
    }


def _identified_policy(payload: dict[str, object]) -> RuntimeContextScopePolicyV1:
    seed = {**payload, "policy_ref": "pending", "policy_hash": "sha256:" + "0" * 64}
    digest = content_hash(scope_policy_hash_basis(seed))
    seed.update(
        policy_ref=f"runtime-context-policy:{digest}",
        policy_hash=f"sha256:{digest}",
    )
    return RuntimeContextScopePolicyV1.model_validate(seed)


def _identified_admission(
    payload: dict[str, object],
) -> RuntimeContextElementAdmissionReceiptV1:
    seed = {
        **payload,
        "admission_ref": "pending",
        "admission_hash": "sha256:" + "0" * 64,
    }
    digest = content_hash(element_admission_hash_basis(seed))
    seed.update(
        admission_ref=f"runtime-context-admission:{digest}",
        admission_hash=f"sha256:{digest}",
    )
    return RuntimeContextElementAdmissionReceiptV1.model_validate(seed)


class RuntimeContextAuthorityIssuanceService:
    def __init__(
        self,
        *,
        authority_store: RuntimeContextAuthorityStore,
        resolver: RuntimeAuthorityResolver,
        provider_registry: RuntimeContextProposalProviderRegistry,
    ) -> None:
        self._authority_store = authority_store
        self._resolver = resolver
        self._provider_registry = provider_registry

    def ensure_builtin_proposal_provider(
        self,
        *,
        actor_context: RuntimeContextDTO,
        configuration: RuntimeContextExplicitOperationsProviderConfigV1,
    ) -> RuntimeContextProposalProviderDescriptorV1:
        self._resolver.require_current_actor(actor_context=actor_context)
        workspace_id = actor_context.workspace_id
        if not workspace_id:
            raise RuntimeContextError("MISSING_WORKSPACE")
        registration = self._provider_registry.get(
            implementation_id=IMPLEMENTATION_ID,
            implementation_version=IMPLEMENTATION_VERSION,
        )
        descriptor = registration.provider.descriptor(
            organization_id=actor_context.organization_id,
            workspace_id=workspace_id,
            configuration=configuration,
        )

        def verify() -> None:
            current = self._provider_registry.get(
                implementation_id=descriptor.implementation_id,
                implementation_version=descriptor.implementation_version,
            )
            if (
                current.implementation_hash != descriptor.implementation_hash
                or current.provider.descriptor(
                    organization_id=actor_context.organization_id,
                    workspace_id=workspace_id,
                    configuration=descriptor.configuration,
                )
                != descriptor
            ):
                raise RuntimeContextError("INVALID_GOVERNANCE")

        return self._authority_store.put_builtin_provider_descriptor(
            descriptor=descriptor, verify_implementation=verify
        )

    def _validate_current_descriptors(
        self,
        *,
        actor_context: RuntimeContextDTO,
        descriptors: tuple[ArtifactPinV1, ...],
    ) -> None:
        resolved = self._resolver.resolve_pins(
            actor_context=actor_context, pins=descriptors
        )
        workspace_id = actor_context.workspace_id
        if workspace_id is None:
            raise RuntimeContextError("MISSING_WORKSPACE")
        for pin, artifact in zip(descriptors, resolved, strict=True):
            if not isinstance(artifact, RuntimeContextProposalProviderDescriptorV1):
                raise RuntimeContextError("INVALID_GOVERNANCE")
            registration = self._provider_registry.get(
                implementation_id=artifact.implementation_id,
                implementation_version=artifact.implementation_version,
            )
            reproduced = registration.provider.descriptor(
                organization_id=actor_context.organization_id,
                workspace_id=workspace_id,
                configuration=artifact.configuration,
            )
            if reproduced != artifact or pin != ArtifactPinV1(
                artifact_kind="runtime_context_proposal_provider_descriptor",
                artifact_ref=artifact.descriptor_ref,
                artifact_hash=artifact.descriptor_hash,
            ):
                raise RuntimeContextError("INVALID_GOVERNANCE")

    def issue_scope_policy(
        self,
        *,
        actor_context: RuntimeContextDTO,
        command: IssueRuntimeContextScopePolicyCommandV1,
        issued_at: datetime,
    ) -> RuntimeContextScopePolicyV1:
        _utc(issued_at, "issued_at")
        self._resolver.require_current_actor(
            actor_context=actor_context, authorization_class="TENANT_ADMIN"
        )
        prior = self._authority_store.get_request(
            actor_context=actor_context,
            operation="ISSUE_SCOPE_POLICY",
            request_id=command.request_id,
        )
        request_hash = _request_hash(
            _policy_request_basis(actor_context=actor_context, command=command)
        )
        if prior is not None:
            if not isinstance(prior, RuntimeContextScopePolicyV1):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if prior.request_hash != request_hash:
                raise RuntimeContextError("REQUEST_CONFLICT")
            historical = (
                *prior.allowed_provider_descriptors,
                *prior.governing_receipts,
                *((prior.supersedes_policy,) if prior.supersedes_policy else ()),
            )
            self._resolver.resolve_pins(actor_context=actor_context, pins=historical)
            expected_effective = command.effective_from or prior.issued_at
            if (
                prior.effective_from != expected_effective
                or prior.effective_to != command.effective_to
            ):
                raise RuntimeContextError("REQUEST_CONFLICT")
            return prior

        if command.effective_from is not None and command.effective_from < issued_at:
            raise RuntimeContextError("POLICY_NOT_EFFECTIVE")
        if command.effective_to <= (command.effective_from or issued_at):
            raise RuntimeContextError("POLICY_NOT_EFFECTIVE")
        if not command.allowed_provider_descriptors:
            raise RuntimeContextError("MISSING_GOVERNANCE")
        self._validate_current_descriptors(
            actor_context=actor_context,
            descriptors=command.allowed_provider_descriptors,
        )
        historical = (
            *command.governing_receipts,
            *((command.supersedes_policy,) if command.supersedes_policy else ()),
        )
        self._resolver.resolve_pins(actor_context=actor_context, pins=historical)
        if any(
            pin.artifact_kind not in command.allowed_governance_kinds
            for pin in command.governing_receipts
        ):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        effective_from = command.effective_from or issued_at
        policy = _identified_policy(
            {
                "schema_version": "runtime_context_scope_policy.v1",
                "organization_id": actor_context.organization_id,
                "workspace_id": actor_context.workspace_id,
                **command.model_dump(
                    mode="json",
                    exclude={"schema_version", "effective_from"},
                    exclude_none=False,
                ),
                "effective_from": effective_from,
                "authority_class": "TENANT_ADMIN",
                "issued_by": actor_context.actor_id,
                "request_hash": request_hash,
                "issued_at": issued_at,
            }
        )
        result = self._authority_store.issue(
            actor_context=actor_context,
            operation="ISSUE_SCOPE_POLICY",
            request_id=command.request_id,
            request_hash=request_hash,
            artifact=policy,
            verify_before_commit=lambda: self._resolver.require_current_actor(
                actor_context=actor_context, authorization_class="TENANT_ADMIN"
            ),
        )
        if not isinstance(result, RuntimeContextScopePolicyV1):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        return result

    @staticmethod
    def _matched_rule(
        *,
        policy: RuntimeContextScopePolicyV1,
        source_kind: str,
        source_object_kind: str,
        context_role: RuntimeContextElementRole,
    ) -> RuntimeContextAdmissionRuleV1:
        matches = tuple(
            rule
            for rule in policy.admission_rules
            if (
                rule.source_artifact_kind,
                rule.source_object_kind,
                rule.context_role,
            )
            == (source_kind, source_object_kind, context_role)
        )
        if len(matches) != 1:
            raise RuntimeContextError("ROLE_NOT_ELIGIBLE")
        return matches[0]

    def admit_element(
        self,
        *,
        actor_context: RuntimeContextDTO,
        command: AdmitRuntimeContextElementCommandV1,
        admitted_at: datetime,
    ) -> RuntimeContextElementAdmissionReceiptV1:
        _utc(admitted_at, "admitted_at")
        self._resolver.require_current_actor(actor_context=actor_context)
        prior = self._authority_store.get_request(
            actor_context=actor_context,
            operation="ADMIT_CONTEXT_ELEMENT",
            request_id=command.request_id,
        )
        if prior is not None:
            if not isinstance(prior, RuntimeContextElementAdmissionReceiptV1):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            policy_values = self._resolver.resolve_pins(
                actor_context=actor_context, pins=(prior.scope_policy,)
            )
            policy = policy_values[0]
            if not isinstance(policy, RuntimeContextScopePolicyV1):
                raise RuntimeContextError("INVALID_GOVERNANCE")
            rule = self._matched_rule(
                policy=policy,
                source_kind=prior.source_artifact.artifact_kind,
                source_object_kind=prior.source_object_kind,
                context_role=prior.context_role,
            )
            self._resolver.require_current_actor(
                actor_context=actor_context,
                authorization_class=rule.admission_authorization,
            )
            resolved_object = self._resolver.resolve_object(
                actor_context=actor_context,
                source=prior.source_artifact,
                object_ref=prior.object_ref,
            )
            self._resolver.resolve_pins(
                actor_context=actor_context,
                pins=(*prior.governing_receipts, *prior.evidence),
            )
            request_hash = _request_hash(
                _admission_request_basis(
                    actor_context=actor_context,
                    command=command,
                    resolved_object=resolved_object,
                    matched_rule=rule,
                )
            )
            if prior.request_hash != request_hash or (
                prior.object_ref,
                prior.source_artifact,
                prior.context_role,
                prior.scope_policy,
                prior.governing_receipts,
                prior.evidence,
                prior.admission_authorization,
            ) != (
                command.object_ref,
                command.source_artifact,
                command.context_role,
                command.scope_policy,
                command.governing_receipts,
                command.evidence,
                rule.admission_authorization,
            ):
                raise RuntimeContextError("REQUEST_CONFLICT")
            return prior

        policy_values = self._resolver.resolve_pins(
            actor_context=actor_context, pins=(command.scope_policy,)
        )
        policy = policy_values[0]
        if not isinstance(policy, RuntimeContextScopePolicyV1):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        if not (policy.effective_from <= admitted_at <= policy.effective_to):
            raise RuntimeContextError("POLICY_NOT_EFFECTIVE")
        resolved_object = self._resolver.resolve_object(
            actor_context=actor_context,
            source=command.source_artifact,
            object_ref=command.object_ref,
        )
        source_value = self._resolver.resolve_pins(
            actor_context=actor_context, pins=(command.source_artifact,)
        )[0]
        if (
            command.context_role == "goal"
            and isinstance(source_value, GraphIntentBrief)
            and not source_value.compiler_eligible
        ):
            raise RuntimeContextError("ROLE_NOT_ELIGIBLE")
        rule = self._matched_rule(
            policy=policy,
            source_kind=command.source_artifact.artifact_kind,
            source_object_kind=resolved_object.source_object_kind,
            context_role=command.context_role,
        )
        if not rule.candidate_allowed:
            raise RuntimeContextError("ROLE_NOT_ELIGIBLE")
        self._resolver.require_current_actor(
            actor_context=actor_context,
            authorization_class=rule.admission_authorization,
        )
        if rule.evidence_required and not command.evidence:
            raise RuntimeContextError("MISSING_GOVERNANCE")
        if any(
            pin.artifact_kind not in policy.allowed_evidence_kinds
            for pin in command.evidence
        ):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        actual_governance = tuple(pin.artifact_kind for pin in command.governing_receipts)
        if sorted(actual_governance) != sorted(rule.required_governance_kinds):
            raise RuntimeContextError("MISSING_GOVERNANCE")
        if any(kind not in policy.allowed_governance_kinds for kind in actual_governance):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        self._resolver.resolve_pins(
            actor_context=actor_context,
            pins=(*command.governing_receipts, *command.evidence),
        )
        request_hash = _request_hash(
            _admission_request_basis(
                actor_context=actor_context,
                command=command,
                resolved_object=resolved_object,
                matched_rule=rule,
            )
        )
        receipt = _identified_admission(
            {
                "schema_version": "runtime_context_element_admission.v1",
                "organization_id": actor_context.organization_id,
                "workspace_id": actor_context.workspace_id,
                "decision": "ADMIT",
                "object_ref": command.object_ref,
                "source_artifact": command.source_artifact,
                "source_object_kind": resolved_object.source_object_kind,
                "context_role": command.context_role,
                "scope_policy": command.scope_policy,
                "admission_authorization": rule.admission_authorization,
                "governing_receipts": command.governing_receipts,
                "evidence": command.evidence,
                "admitted_by": actor_context.actor_id,
                "request_id": command.request_id,
                "request_hash": request_hash,
                "admitted_at": admitted_at,
            }
        )
        result = self._authority_store.issue(
            actor_context=actor_context,
            operation="ADMIT_CONTEXT_ELEMENT",
            request_id=command.request_id,
            request_hash=request_hash,
            artifact=receipt,
            verify_before_commit=lambda: self._resolver.require_current_actor(
                actor_context=actor_context,
                authorization_class=rule.admission_authorization,
            ),
        )
        if not isinstance(result, RuntimeContextElementAdmissionReceiptV1):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        return result


__all__ = [
    "AdmitRuntimeContextElementCommandV1",
    "IssueRuntimeContextScopePolicyCommandV1",
    "RuntimeContextAuthorityIssuanceService",
]
