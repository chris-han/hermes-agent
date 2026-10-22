"""Atomic workspace persistence for self-bound runtime-context authority."""

from __future__ import annotations

from storage.hosted_postgres import sql_query,begin_write,in_transaction

import json
import re
from typing import Callable, Literal, TypeVar

from pydantic import BaseModel

from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_context_authority import (
    RuntimeContextElementAdmissionReceiptV1,
    RuntimeContextProposalProviderDescriptorV1,
    RuntimeContextScopePolicyV1,
)
from eos.artifact_hashing import canonical_json
from storage.ontology_build_store import OntologyBuildStore
from storage.runtime_working_context_store import RuntimeContextError


AuthorityOperation = Literal["ISSUE_SCOPE_POLICY", "ADMIT_CONTEXT_ELEMENT"]
AuthorityArtifact = RuntimeContextScopePolicyV1 | RuntimeContextElementAdmissionReceiptV1
ReadT = TypeVar("ReadT")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_DESCRIPTOR_KIND = "runtime_context_proposal_provider_descriptor"


class RuntimeContextAuthorityStore:
    def __init__(self, build_store: OntologyBuildStore) -> None:
        if not isinstance(build_store, OntologyBuildStore):
            raise TypeError("build_store must be OntologyBuildStore")
        self._build_store = build_store
        self._connection = build_store._connection

    def close(self) -> None:
        self._build_store.close()

    def _consistent_read(self, read: Callable[[], ReadT]) -> ReadT:
        owns_transaction = not in_transaction(self._connection)
        if owns_transaction:
            self._connection.execute(sql_query("BEGIN","BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        try:
            result = read()
            if owns_transaction:
                self._connection.commit()
            return result
        except BaseException:
            if owns_transaction:
                self._connection.rollback()
            raise

    def _load_artifact(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
        artifact_kind: str,
        schema_version: str,
        contract_type: type[BaseModel],
        ref_field: str,
        hash_field: str,
    ) -> BaseModel:
        pin = ArtifactPinV1.model_validate(pin.model_dump(mode="json"))
        if pin.artifact_kind != artifact_kind:
            raise RuntimeContextError("INVALID_STATE")
        row = self._connection.execute(
            sql_query("""
            SELECT organization_id, workspace_id, schema_version, artifact_hash,
                   payload_json
            FROM ontology_build_artifacts WHERE artifact_ref=?
            """,'\n            SELECT organization_id, workspace_id, schema_version, artifact_hash,\n                   payload_json\n            FROM ontology_build_artifacts WHERE artifact_ref=%s\n            '),
            (pin.artifact_ref,),
        ).fetchone()
        if row is None:
            raise RuntimeContextError("NOT_FOUND")
        stored_org, stored_workspace, stored_schema, stored_hash, encoded = row
        if (stored_org, stored_workspace) != (organization_id, workspace_id):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        if (stored_schema, stored_hash) != (schema_version, pin.artifact_hash):
            raise RuntimeContextError("HASH_MISMATCH")
        try:
            artifact = contract_type.model_validate(json.loads(str(encoded)))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeContextError("HASH_MISMATCH") from exc
        if (
            getattr(artifact, ref_field),
            getattr(artifact, hash_field),
            canonical_json(artifact.model_dump(mode="json")),
        ) != (pin.artifact_ref, pin.artifact_hash, encoded):
            raise RuntimeContextError("HASH_MISMATCH")
        return artifact

    def _descriptor_from_pin_once(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
    ) -> RuntimeContextProposalProviderDescriptorV1:
        loaded = self._load_artifact(
            organization_id=organization_id,
            workspace_id=workspace_id,
            pin=pin,
            artifact_kind=_DESCRIPTOR_KIND,
            schema_version="runtime_context_proposal_provider_descriptor.v1",
            contract_type=RuntimeContextProposalProviderDescriptorV1,
            ref_field="descriptor_ref",
            hash_field="descriptor_hash",
        )
        if not isinstance(loaded, RuntimeContextProposalProviderDescriptorV1):
            raise RuntimeContextError("INVALID_STATE")
        index = self._connection.execute(
            sql_query("""
            SELECT organization_id, workspace_id, descriptor_ref, descriptor_hash,
                   implementation_id, implementation_version, implementation_hash,
                   configuration_hash
            FROM runtime_context_provider_descriptor_index WHERE descriptor_ref=?
            """,'\n            SELECT organization_id, workspace_id, descriptor_ref, descriptor_hash,\n                   implementation_id, implementation_version, implementation_hash,\n                   configuration_hash\n            FROM runtime_context_provider_descriptor_index WHERE descriptor_ref=%s\n            '),
            (loaded.descriptor_ref,),
        ).fetchone()
        if index != (
            loaded.organization_id,
            loaded.workspace_id,
            loaded.descriptor_ref,
            loaded.descriptor_hash,
            loaded.implementation_id,
            loaded.implementation_version,
            loaded.implementation_hash,
            loaded.configuration_hash,
        ):
            raise RuntimeContextError("INVALID_STATE")
        return loaded

    def put_builtin_provider_descriptor(
        self,
        *,
        descriptor: RuntimeContextProposalProviderDescriptorV1,
        verify_implementation: Callable[[], None],
    ) -> RuntimeContextProposalProviderDescriptorV1:
        descriptor = RuntimeContextProposalProviderDescriptorV1.model_validate(
            descriptor.model_dump(mode="json")
        )
        if not callable(verify_implementation):
            raise TypeError("verify_implementation must be callable")
        begin_write(self._connection)
        try:
            artifact_exists = self._connection.execute(
                sql_query("SELECT artifact_ref FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT artifact_ref FROM ontology_build_artifacts WHERE artifact_ref=%s'),
                (descriptor.descriptor_ref,),
            ).fetchone()
            exact_index = self._connection.execute(
                sql_query("SELECT descriptor_ref FROM runtime_context_provider_descriptor_index "
                "WHERE descriptor_ref=?",'SELECT descriptor_ref FROM runtime_context_provider_descriptor_index WHERE descriptor_ref=%s'),
                (descriptor.descriptor_ref,),
            ).fetchone()
            if (artifact_exists is None) != (exact_index is None):
                raise RuntimeContextError("INVALID_STATE")
            verify_implementation()
            if exact_index is not None:
                stored = self._descriptor_from_pin_once(
                    organization_id=descriptor.organization_id,
                    workspace_id=descriptor.workspace_id,
                    pin=ArtifactPinV1(
                        artifact_kind=_DESCRIPTOR_KIND,
                        artifact_ref=descriptor.descriptor_ref,
                        artifact_hash=descriptor.descriptor_hash,
                    ),
                )
                if stored != descriptor:
                    raise RuntimeContextError("DESCRIPTOR_CONFLICT")
                self._connection.commit()
                return stored
            self._build_store._insert_prepared((self._build_store._prepare(descriptor),))
            self._connection.execute(
                sql_query("""
                INSERT INTO runtime_context_provider_descriptor_index (
                    organization_id, workspace_id, descriptor_ref, descriptor_hash,
                    implementation_id, implementation_version, implementation_hash,
                    configuration_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO runtime_context_provider_descriptor_index (\n                    organization_id, workspace_id, descriptor_ref, descriptor_hash,\n                    implementation_id, implementation_version, implementation_hash,\n                    configuration_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    descriptor.organization_id,
                    descriptor.workspace_id,
                    descriptor.descriptor_ref,
                    descriptor.descriptor_hash,
                    descriptor.implementation_id,
                    descriptor.implementation_version,
                    descriptor.implementation_hash,
                    descriptor.configuration_hash,
                ),
            )
            self._connection.commit()
            return descriptor
        except BaseException:
            self._connection.rollback()
            raise

    def get_proposal_provider_descriptor(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
    ) -> RuntimeContextProposalProviderDescriptorV1:
        return self._consistent_read(
            lambda: self._descriptor_from_pin_once(
                organization_id=organization_id,
                workspace_id=workspace_id,
                pin=pin,
            )
        )

    @staticmethod
    def _operation_spec(
        operation: str,
    ) -> tuple[type[BaseModel], str, str, str, str, str]:
        if operation == "ISSUE_SCOPE_POLICY":
            return (
                RuntimeContextScopePolicyV1,
                "runtime_context_scope_policy.v1",
                "runtime_context_scope_policy",
                "policy_ref",
                "policy_hash",
                "issued_by",
            )
        if operation == "ADMIT_CONTEXT_ELEMENT":
            return (
                RuntimeContextElementAdmissionReceiptV1,
                "runtime_context_element_admission.v1",
                "runtime_context_element_admission",
                "admission_ref",
                "admission_hash",
                "admitted_by",
            )
        raise ValueError("unsupported runtime context authority operation")

    def _issued_from_pin_once(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        operation: AuthorityOperation,
        pin: ArtifactPinV1,
    ) -> AuthorityArtifact:
        contract_type, schema, kind, ref_field, hash_field, actor_field = self._operation_spec(
            operation
        )
        loaded = self._load_artifact(
            organization_id=organization_id,
            workspace_id=workspace_id,
            pin=pin,
            artifact_kind=kind,
            schema_version=schema,
            contract_type=contract_type,
            ref_field=ref_field,
            hash_field=hash_field,
        )
        if not isinstance(
            loaded,
            (RuntimeContextScopePolicyV1, RuntimeContextElementAdmissionReceiptV1),
        ):
            raise RuntimeContextError("INVALID_STATE")
        index = self._connection.execute(
            sql_query("""
            SELECT organization_id, workspace_id, actor_id, operation, request_id,
                   request_hash, artifact_kind, artifact_ref, artifact_hash
            FROM runtime_context_authority_request_index WHERE artifact_ref=?
            """,'\n            SELECT organization_id, workspace_id, actor_id, operation, request_id,\n                   request_hash, artifact_kind, artifact_ref, artifact_hash\n            FROM runtime_context_authority_request_index WHERE artifact_ref=%s\n            '),
            (getattr(loaded, ref_field),),
        ).fetchone()
        if index != (
            loaded.organization_id,
            loaded.workspace_id,
            getattr(loaded, actor_field),
            operation,
            loaded.request_id,
            loaded.request_hash,
            kind,
            getattr(loaded, ref_field),
            getattr(loaded, hash_field),
        ):
            raise RuntimeContextError("INVALID_STATE")
        return loaded

    def get_scope_policy(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
    ) -> RuntimeContextScopePolicyV1:
        result = self._consistent_read(
            lambda: self._issued_from_pin_once(
                organization_id=organization_id,
                workspace_id=workspace_id,
                operation="ISSUE_SCOPE_POLICY",
                pin=pin,
            )
        )
        if not isinstance(result, RuntimeContextScopePolicyV1):
            raise RuntimeContextError("INVALID_STATE")
        return result

    def get_admission(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
    ) -> RuntimeContextElementAdmissionReceiptV1:
        result = self._consistent_read(
            lambda: self._issued_from_pin_once(
                organization_id=organization_id,
                workspace_id=workspace_id,
                operation="ADMIT_CONTEXT_ELEMENT",
                pin=pin,
            )
        )
        if not isinstance(result, RuntimeContextElementAdmissionReceiptV1):
            raise RuntimeContextError("INVALID_STATE")
        return result

    @staticmethod
    def _actor_scope(actor_context: RuntimeContextDTO) -> tuple[str, str, str]:
        if not isinstance(actor_context, RuntimeContextDTO) or not actor_context.workspace_id:
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        return (
            actor_context.organization_id,
            actor_context.workspace_id,
            actor_context.actor_id,
        )

    def _request_from_index_once(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        actor_id: str,
        operation: AuthorityOperation,
        request_id: str,
    ) -> AuthorityArtifact | None:
        contract_type, schema, kind, ref_field, hash_field, actor_field = self._operation_spec(
            operation
        )
        row = self._connection.execute(
            sql_query("""
            SELECT request_hash, artifact_kind, artifact_ref, artifact_hash
            FROM runtime_context_authority_request_index
            WHERE organization_id=? AND workspace_id=? AND actor_id=?
              AND operation=? AND request_id=?
            """,'\n            SELECT request_hash, artifact_kind, artifact_ref, artifact_hash\n            FROM runtime_context_authority_request_index\n            WHERE organization_id=%s AND workspace_id=%s AND actor_id=%s\n              AND operation=%s AND request_id=%s\n            '),
            (organization_id, workspace_id, actor_id, operation, request_id),
        ).fetchone()
        if row is None:
            actor_json_field = (
                "issued_by" if operation == "ISSUE_SCOPE_POLICY" else "admitted_by"
            )
            orphan = self._connection.execute(
                sql_query(f"""
                SELECT artifact_ref FROM ontology_build_artifacts
                WHERE organization_id=? AND workspace_id=? AND schema_version=?
                  AND json_extract(payload_json, '$.{actor_json_field}')=?
                  AND json_extract(payload_json, '$.request_id')=? LIMIT 1
                """,f"\n                SELECT artifact_ref FROM ontology_build_artifacts\n                WHERE organization_id=%s AND workspace_id=%s AND schema_version=%s\n                  AND (payload_json::jsonb #>> '{{{actor_json_field}}}')=%s\n                  AND (payload_json::jsonb #>> '{{request_id}}')=%s LIMIT 1\n                "),
                (organization_id, workspace_id, schema, actor_id, request_id),
            ).fetchone()
            if orphan is not None:
                raise RuntimeContextError("INVALID_STATE")
            return None
        request_hash, indexed_kind, artifact_ref, artifact_hash = row
        if indexed_kind != kind:
            raise RuntimeContextError("INVALID_STATE")
        try:
            loaded = self._load_artifact(
                organization_id=organization_id,
                workspace_id=workspace_id,
                pin=ArtifactPinV1(
                    artifact_kind=kind,
                    artifact_ref=str(artifact_ref),
                    artifact_hash=str(artifact_hash),
                ),
                artifact_kind=kind,
                schema_version=schema,
                contract_type=contract_type,
                ref_field=ref_field,
                hash_field=hash_field,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeContextError("INVALID_STATE") from exc
        if (
            getattr(loaded, actor_field),
            loaded.request_id,
            loaded.request_hash,
        ) != (actor_id, request_id, request_hash):
            raise RuntimeContextError("INVALID_STATE")
        if not isinstance(
            loaded,
            (RuntimeContextScopePolicyV1, RuntimeContextElementAdmissionReceiptV1),
        ):
            raise RuntimeContextError("INVALID_STATE")
        return loaded

    def get_request(
        self,
        *,
        actor_context: RuntimeContextDTO,
        operation: AuthorityOperation,
        request_id: str,
    ) -> AuthorityArtifact | None:
        organization_id, workspace_id, actor_id = self._actor_scope(actor_context)
        self._operation_spec(operation)
        return self._consistent_read(
            lambda: self._request_from_index_once(
                organization_id=organization_id,
                workspace_id=workspace_id,
                actor_id=actor_id,
                operation=operation,
                request_id=request_id,
            )
        )

    def issue(
        self,
        *,
        actor_context: RuntimeContextDTO,
        operation: AuthorityOperation,
        request_id: str,
        request_hash: str,
        artifact: AuthorityArtifact,
        verify_before_commit: Callable[[], None],
    ) -> AuthorityArtifact:
        organization_id, workspace_id, actor_id = self._actor_scope(actor_context)
        contract_type, _schema, kind, ref_field, hash_field, actor_field = self._operation_spec(
            operation
        )
        if not isinstance(artifact, contract_type):
            raise RuntimeContextError("INVALID_STATE")
        artifact = contract_type.model_validate(artifact.model_dump(mode="json"))
        if _SHA256.fullmatch(request_hash) is None:
            raise ValueError("request_hash must be a lowercase sha256 digest")
        if (artifact.organization_id, artifact.workspace_id) != (
            organization_id,
            workspace_id,
        ):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        if (getattr(artifact, actor_field), artifact.request_id) != (actor_id, request_id):
            raise RuntimeContextError("INVALID_STATE")
        if artifact.request_hash != request_hash:
            raise RuntimeContextError("REQUEST_CONFLICT")
        if not callable(verify_before_commit):
            raise TypeError("verify_before_commit must be callable")

        begin_write(self._connection)
        try:
            prior = self._request_from_index_once(
                organization_id=organization_id,
                workspace_id=workspace_id,
                actor_id=actor_id,
                operation=operation,
                request_id=request_id,
            )
            if prior is not None:
                if prior.request_hash != request_hash:
                    raise RuntimeContextError("REQUEST_CONFLICT")
                self._connection.commit()
                return prior
            verify_before_commit()
            self._build_store._insert_prepared((self._build_store._prepare(artifact),))
            self._connection.execute(
                sql_query("""
                INSERT INTO runtime_context_authority_request_index (
                    organization_id, workspace_id, actor_id, operation, request_id,
                    request_hash, artifact_kind, artifact_ref, artifact_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO runtime_context_authority_request_index (\n                    organization_id, workspace_id, actor_id, operation, request_id,\n                    request_hash, artifact_kind, artifact_ref, artifact_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    organization_id,
                    workspace_id,
                    actor_id,
                    operation,
                    request_id,
                    request_hash,
                    kind,
                    getattr(artifact, ref_field),
                    getattr(artifact, hash_field),
                ),
            )
            self._connection.commit()
            return artifact
        except BaseException:
            self._connection.rollback()
            raise


__all__ = ["AuthorityOperation", "RuntimeContextAuthorityStore"]
