"""Frozen provenance for the built-in explicit runtime-context operation adapter."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import TypeAdapter

from contracts.runtime_context_authority import (
    RuntimeContextExplicitOperationsProviderConfigV1,
    RuntimeContextProposalProviderDescriptorV1,
    proposal_provider_descriptor_hash_basis,
)
from contracts.runtime_working_context import RuntimeContextOperationV1
from eos.artifact_hashing import content_hash
from storage.runtime_working_context_store import RuntimeContextError


IMPLEMENTATION_ID = "semantier.runtime_context.explicit_operations"
IMPLEMENTATION_VERSION = "v1"
_OPERATION_TUPLE = TypeAdapter(tuple[RuntimeContextOperationV1, ...])


class ExplicitOperationProposalProvider:
    """Authority-neutral adapter that validates one ordered explicit operation list."""

    def __init__(self, *, implementation_hash: str) -> None:
        self._implementation_hash = implementation_hash

    @property
    def implementation_hash(self) -> str:
        return self._implementation_hash

    def descriptor(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        configuration: RuntimeContextExplicitOperationsProviderConfigV1,
    ) -> RuntimeContextProposalProviderDescriptorV1:
        configuration = RuntimeContextExplicitOperationsProviderConfigV1.model_validate(
            configuration.model_dump(mode="json")
        )
        configuration_hash = "sha256:" + content_hash(
            configuration.model_dump(mode="json")
        )
        payload = {
            "schema_version": "runtime_context_proposal_provider_descriptor.v1",
            "descriptor_ref": "pending",
            "descriptor_hash": "sha256:" + "0" * 64,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "provider_kind": "runtime_context_explicit_operations",
            "implementation_id": IMPLEMENTATION_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "implementation_hash": self._implementation_hash,
            "configuration": configuration.model_dump(mode="json"),
            "configuration_hash": configuration_hash,
        }
        digest = content_hash(proposal_provider_descriptor_hash_basis(payload))
        payload.update(
            descriptor_ref=f"runtime-context-proposal-provider:{digest}",
            descriptor_hash=f"sha256:{digest}",
        )
        return RuntimeContextProposalProviderDescriptorV1.model_validate(payload)

    def construct_operations(
        self,
        *,
        descriptor: RuntimeContextProposalProviderDescriptorV1,
        operations: tuple[RuntimeContextOperationV1, ...],
    ) -> tuple[RuntimeContextOperationV1, ...]:
        descriptor = RuntimeContextProposalProviderDescriptorV1.model_validate(
            descriptor.model_dump(mode="json")
        )
        if (
            descriptor.implementation_id,
            descriptor.implementation_version,
            descriptor.implementation_hash,
        ) != (IMPLEMENTATION_ID, IMPLEMENTATION_VERSION, self._implementation_hash):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        parsed = _OPERATION_TUPLE.validate_python(operations)
        if len(parsed) > descriptor.configuration.max_operations_per_proposal:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        supported = set(descriptor.configuration.supported_operations)
        if any(operation.operation not in supported for operation in parsed):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        return parsed


@dataclass(frozen=True)
class RuntimeContextProposalProviderRegistration:
    implementation_id: Literal["semantier.runtime_context.explicit_operations"]
    implementation_version: Literal["v1"]
    implementation_hash: str
    provider: ExplicitOperationProposalProvider


class RuntimeContextProposalProviderRegistry:
    """Immutable startup-composed registry keyed by the exact implementation tuple."""

    def __init__(
        self,
        registrations: tuple[RuntimeContextProposalProviderRegistration, ...],
    ) -> None:
        indexed: dict[tuple[str, str], RuntimeContextProposalProviderRegistration] = {}
        for registration in registrations:
            key = (registration.implementation_id, registration.implementation_version)
            if (
                key in indexed
                or registration.provider.implementation_hash
                != registration.implementation_hash
            ):
                raise ValueError("invalid or duplicate runtime context provider registration")
            indexed[key] = registration
        self._registrations: Mapping[
            tuple[str, str], RuntimeContextProposalProviderRegistration
        ] = MappingProxyType(indexed)

    @classmethod
    def compose_from_packaged_resources(cls) -> "RuntimeContextProposalProviderRegistry":
        source = resources.files("services").joinpath(
            "runtime_context_proposal_provider.py"
        ).read_bytes()
        implementation_hash = "sha256:" + sha256(source).hexdigest()
        provider = ExplicitOperationProposalProvider(
            implementation_hash=implementation_hash
        )
        return cls(
            (
                RuntimeContextProposalProviderRegistration(
                    implementation_id=IMPLEMENTATION_ID,
                    implementation_version=IMPLEMENTATION_VERSION,
                    implementation_hash=implementation_hash,
                    provider=provider,
                ),
            )
        )

    def get(
        self, *, implementation_id: str, implementation_version: str
    ) -> RuntimeContextProposalProviderRegistration:
        try:
            return self._registrations[(implementation_id, implementation_version)]
        except KeyError as exc:
            raise RuntimeContextError("INVALID_GOVERNANCE") from exc


__all__ = [
    "ExplicitOperationProposalProvider",
    "RuntimeContextProposalProviderRegistration",
    "RuntimeContextProposalProviderRegistry",
]
