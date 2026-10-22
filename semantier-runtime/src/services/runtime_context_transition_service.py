"""Closed runtime artifact ownership and operation authorization prerequisites."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel

from agents.auth_db import load_users
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.knowledge_derivation import DerivationRecordV1
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1, ExploratoryGraphSnapshotV1
from contracts.semantic_kernel import (
    AuthorizationBasis,
    ChangeCandidate,
    ContextGraph,
    DecisionRecord,
    Evidence,
    Ontology,
)
from contracts.runtime_context_authority import (
    RuntimeContextAdmissionRuleV1,
    RuntimeContextElementAdmissionReceiptV1,
    RuntimeContextProposalProviderDescriptorV1,
    RuntimeContextScopePolicyV1,
)
from contracts.runtime_working_context import (
    AddCandidateOperationV1,
    AdmitOperationV1,
    ActivateOperationV1,
    DeactivateOperationV1,
    RuntimeContextElementV1,
    RuntimeContextTransitionProposalV1,
    RuntimeContextTransitionReceiptV1,
    RuntimeContextTransitionV1,
    RuntimeTaskContextV1,
)
from contracts.runtime_temporal_source import (
    RuntimeTemporalNormalizationV1,
    RuntimeTemporalSourceArchiveV1,
    RuntimeTemporalSourceGraphV1,
)
from eos.artifact_hashing import canonical_json
from services.runtime_context_proposal_provider import (
    RuntimeContextProposalProviderRegistry,
)
from storage.graph_intent_brief_store import GraphIntentBriefStore
from storage.knowledge_derivation_store import KnowledgeDerivationStore
from storage.ontology_build_store import OntologyBuildStore
from storage.runtime_context_authority_store import RuntimeContextAuthorityStore
from storage.runtime_working_context_store import RuntimeContextError, RuntimeTaskContextStore


_AUTHORITY_RESOLVER_REF = "runtime-authority-resolver:current-actor:v1"
_AUTHORITY_RESOLVER_HASH = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

ResolvedRuntimePin: TypeAlias = BaseModel | GraphIntentBrief


@dataclass(frozen=True)
class ResolvedRuntimeObject:
    source: ArtifactPinV1
    object_ref: str
    source_object_kind: str


class RuntimePinOwner(Protocol):
    def resolve_exact(
        self, *, organization_id: str, workspace_id: str, pin: ArtifactPinV1
    ) -> ResolvedRuntimePin: ...

    def resolve_object(
        self,
        *,
        artifact: ResolvedRuntimePin,
        pin: ArtifactPinV1,
        object_ref: str,
    ) -> ResolvedRuntimeObject: ...


def _raise_owner_error(exc: Exception) -> None:
    if isinstance(exc, RuntimeContextError):
        if exc.code in {
            "FORBIDDEN_SCOPE",
            "HASH_MISMATCH",
            "UNSUPPORTED_ARTIFACT_KIND",
            "OBJECT_NOT_FOUND",
            "INTEGRITY_UNPROVABLE",
        }:
            raise exc
        if exc.code == "NOT_FOUND":
            raise RuntimeContextError("OBJECT_NOT_FOUND") from exc
        raise RuntimeContextError("INTEGRITY_UNPROVABLE") from exc
    if isinstance(exc, PermissionError):
        raise RuntimeContextError("FORBIDDEN_SCOPE") from exc
    message = str(exc)
    if "not found" in message.lower():
        raise RuntimeContextError("OBJECT_NOT_FOUND") from exc
    if "hash" in message.lower() or "SUBJECT_MISMATCH" in message:
        raise RuntimeContextError("HASH_MISMATCH") from exc
    raise RuntimeContextError("INTEGRITY_UNPROVABLE") from exc


class _GraphIntentOwner:
    def __init__(self, store: GraphIntentBriefStore) -> None:
        self._store = store

    def resolve_exact(
        self, *, organization_id: str, workspace_id: str, pin: ArtifactPinV1
    ) -> GraphIntentBrief:
        if (self._store.organization_id, self._store.workspace_id) != (
            organization_id,
            workspace_id,
        ):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        try:
            brief = self._store.get(
                pin.artifact_ref, graph_intent_hash=pin.artifact_hash
            )
        except Exception as exc:
            _raise_owner_error(exc)
        if (brief.graph_intent_id, brief.graph_intent_hash) != (
            pin.artifact_ref,
            pin.artifact_hash,
        ):
            raise RuntimeContextError("HASH_MISMATCH")
        return brief

    def resolve_object(
        self,
        *,
        artifact: ResolvedRuntimePin,
        pin: ArtifactPinV1,
        object_ref: str,
    ) -> ResolvedRuntimeObject:
        if not isinstance(artifact, GraphIntentBrief) or object_ref != artifact.graph_intent_id:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        return ResolvedRuntimeObject(
            source=pin,
            object_ref=object_ref,
            source_object_kind="graph_intent_brief",
        )


class _ExploratorySnapshotOwner:
    def __init__(self, store: OntologyBuildStore) -> None:
        self._store = store

    def resolve_exact(
        self, *, organization_id: str, workspace_id: str, pin: ArtifactPinV1
    ) -> ExploratoryGraphSnapshotV1:
        try:
            raw = self._store.get_raw(
                organization_id=organization_id,
                workspace_id=workspace_id,
                artifact_ref=pin.artifact_ref,
            )
            if raw is None:
                raise RuntimeContextError("OBJECT_NOT_FOUND")
            if raw["artifact_hash"] != pin.artifact_hash:
                raise RuntimeContextError("HASH_MISMATCH")
            snapshot, _integrity = self._store.get_verified_exploratory_snapshot(
                organization_id=organization_id, workspace_id=workspace_id, pin=pin
            )
        except Exception as exc:
            _raise_owner_error(exc)
        return snapshot

    def resolve_object(
        self,
        *,
        artifact: ResolvedRuntimePin,
        pin: ArtifactPinV1,
        object_ref: str,
    ) -> ResolvedRuntimeObject:
        if not isinstance(artifact, ExploratoryGraphSnapshotV1):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        matches = tuple(node for node in artifact.nodes if node.node_id == object_ref)
        if len(matches) != 1:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        return ResolvedRuntimeObject(
            source=pin, object_ref=object_ref, source_object_kind=matches[0].node_kind
        )


class _ExactStoredModelOwner:
    def __init__(
        self,
        store: OntologyBuildStore,
        *,
        contract_type: type[BaseModel],
        ref_field: str,
        hash_field: str,
        schema_version: str | None = None,
    ) -> None:
        self._store = store
        self._contract_type = contract_type
        self._ref_field = ref_field
        self._hash_field = hash_field
        self._schema_version = schema_version

    def resolve_exact(
        self, *, organization_id: str, workspace_id: str, pin: ArtifactPinV1
    ) -> BaseModel:
        try:
            raw = self._store.get_raw(
                organization_id=organization_id,
                workspace_id=workspace_id,
                artifact_ref=pin.artifact_ref,
            )
        except Exception as exc:
            _raise_owner_error(exc)
        if raw is None:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        expected_schema = self._schema_version or (pin.artifact_kind + ".v1")
        if raw["schema_version"] != expected_schema or raw["artifact_hash"] != pin.artifact_hash:
            raise RuntimeContextError("HASH_MISMATCH")
        try:
            artifact = self._contract_type.model_validate(raw["payload"])
        except (TypeError, ValueError) as exc:
            raise RuntimeContextError("HASH_MISMATCH") from exc
        if (
            getattr(artifact, self._ref_field),
            getattr(artifact, self._hash_field),
            canonical_json(artifact.model_dump(mode="json")),
        ) != (
            pin.artifact_ref,
            pin.artifact_hash,
            canonical_json(raw["payload"]),
        ):
            raise RuntimeContextError("HASH_MISMATCH")
        return artifact

    def resolve_object(
        self,
        *,
        artifact: ResolvedRuntimePin,
        pin: ArtifactPinV1,
        object_ref: str,
    ) -> ResolvedRuntimeObject:
        raise RuntimeContextError("OBJECT_NOT_FOUND")


class _UnavailableOwner:
    def resolve_exact(
        self, *, organization_id: str, workspace_id: str, pin: ArtifactPinV1
    ) -> ResolvedRuntimePin:
        raise RuntimeContextError("UNSUPPORTED_ARTIFACT_KIND")

    def resolve_object(
        self,
        *,
        artifact: ResolvedRuntimePin,
        pin: ArtifactPinV1,
        object_ref: str,
    ) -> ResolvedRuntimeObject:
        raise RuntimeContextError("UNSUPPORTED_ARTIFACT_KIND")


class _ContextOwner(_ExactStoredModelOwner):
    def __init__(self, store: RuntimeTaskContextStore) -> None:
        self._context_store = store

    def resolve_exact(
        self, *, organization_id: str, workspace_id: str, pin: ArtifactPinV1
    ) -> RuntimeTaskContextV1:
        try:
            return self._context_store.get_context(
                organization_id=organization_id, workspace_id=workspace_id, pin=pin
            )
        except Exception as exc:
            _raise_owner_error(exc)


class _TemporalGraphOwner(_ExactStoredModelOwner):
    def __init__(self, store: OntologyBuildStore) -> None:
        super().__init__(
            store,
            contract_type=RuntimeTemporalSourceGraphV1,
            ref_field="graph_ref",
            hash_field="graph_hash",
        )

    def resolve_object(
        self,
        *,
        artifact: ResolvedRuntimePin,
        pin: ArtifactPinV1,
        object_ref: str,
    ) -> ResolvedRuntimeObject:
        if not isinstance(artifact, RuntimeTemporalSourceGraphV1):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        entity_matches = [row for row in artifact.entities if row.id == object_ref]
        relationship_matches = [
            row for row in artifact.relationships if row.id == object_ref
        ]
        if len(entity_matches) + len(relationship_matches) != 1:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        return ResolvedRuntimeObject(
            source=pin,
            object_ref=object_ref,
            source_object_kind="entity" if entity_matches else "relationship",
        )


class _AuthorityOwner:
    def __init__(
        self,
        store: RuntimeContextAuthorityStore,
        *,
        method_name: str,
        contract_type: type[BaseModel],
    ) -> None:
        self._store = store
        self._method_name = method_name
        self._contract_type = contract_type

    def resolve_exact(
        self, *, organization_id: str, workspace_id: str, pin: ArtifactPinV1
    ) -> BaseModel:
        try:
            artifact = getattr(self._store, self._method_name)(
                organization_id=organization_id, workspace_id=workspace_id, pin=pin
            )
        except Exception as exc:
            _raise_owner_error(exc)
        if not isinstance(artifact, self._contract_type):
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        return artifact

    def resolve_object(
        self,
        *,
        artifact: ResolvedRuntimePin,
        pin: ArtifactPinV1,
        object_ref: str,
    ) -> ResolvedRuntimeObject:
        raise RuntimeContextError("OBJECT_NOT_FOUND")


class _DerivationOwner:
    def __init__(self, store: KnowledgeDerivationStore) -> None:
        self._store = store

    def resolve_exact(
        self, *, organization_id: str, workspace_id: str, pin: ArtifactPinV1
    ) -> DerivationRecordV1:
        record = self._store.get_by_ref(pin.artifact_ref)
        if record is None:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        if record.derivation_hash != pin.artifact_hash:
            raise RuntimeContextError("HASH_MISMATCH")
        return record

    def resolve_object(
        self,
        *,
        artifact: ResolvedRuntimePin,
        pin: ArtifactPinV1,
        object_ref: str,
    ) -> ResolvedRuntimeObject:
        if not isinstance(artifact, DerivationRecordV1) or artifact.output_ref != object_ref:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        return ResolvedRuntimeObject(
            source=pin,
            object_ref=object_ref,
            source_object_kind="inference" if artifact.derivation_kind == "inference" else artifact.derivation_kind,
        )


class RuntimeAuthorityResolver:
    """Resolve only artifact kinds whose exact owner is registered here."""

    def __init__(
        self,
        *,
        graph_intent_store: GraphIntentBriefStore | None,
        ontology_build_store: OntologyBuildStore,
        runtime_context_store: RuntimeTaskContextStore | None,
        authority_store: RuntimeContextAuthorityStore | None,
        provider_registry: RuntimeContextProposalProviderRegistry,
        derivation_store: KnowledgeDerivationStore | None = None,
    ) -> None:
        owners: dict[str, RuntimePinOwner] = {
            "ontology": _ExactStoredModelOwner(
                ontology_build_store,
                contract_type=Ontology,
                ref_field="artifact_ref",
                hash_field="artifact_hash",
                schema_version="semantier.ontology.v1",
            ),
            "evidence": _ExactStoredModelOwner(
                ontology_build_store,
                contract_type=Evidence,
                ref_field="artifact_ref",
                hash_field="artifact_hash",
                schema_version="semantier.evidence.v1",
            ),
            "context_graph": _ExactStoredModelOwner(
                ontology_build_store,
                contract_type=ContextGraph,
                ref_field="artifact_ref",
                hash_field="artifact_hash",
                schema_version="semantier.context_graph.v1",
            ),
            "decision_record": _ExactStoredModelOwner(
                ontology_build_store,
                contract_type=DecisionRecord,
                ref_field="artifact_ref",
                hash_field="artifact_hash",
                schema_version="semantier.decision_record.v1",
            ),
            "change_candidate": _ExactStoredModelOwner(
                ontology_build_store,
                contract_type=ChangeCandidate,
                ref_field="artifact_ref",
                hash_field="artifact_hash",
                schema_version="semantier.change_candidate.v1",
            ),
            "exploratory_graph_snapshot": _ExploratorySnapshotOwner(ontology_build_store),
            "runtime_working_context": (
                _ContextOwner(runtime_context_store)
                if runtime_context_store is not None
                else _UnavailableOwner()
            ),
            "runtime_temporal_source_archive": _ExactStoredModelOwner(
                ontology_build_store,
                contract_type=RuntimeTemporalSourceArchiveV1,
                ref_field="archive_ref",
                hash_field="archive_hash",
            ),
            "runtime_temporal_normalization": _ExactStoredModelOwner(
                ontology_build_store,
                contract_type=RuntimeTemporalNormalizationV1,
                ref_field="normalization_ref",
                hash_field="normalization_hash",
            ),
            "runtime_temporal_source_graph": _TemporalGraphOwner(
                ontology_build_store
            ),
            "runtime_context_transition_proposal": _ExactStoredModelOwner(
                ontology_build_store,
                contract_type=RuntimeContextTransitionProposalV1,
                ref_field="proposal_ref",
                hash_field="proposal_hash",
            ),
            "runtime_context_transition": _ExactStoredModelOwner(
                ontology_build_store,
                contract_type=RuntimeContextTransitionV1,
                ref_field="transition_ref",
                hash_field="transition_hash",
            ),
            "runtime_context_transition_receipt": _ExactStoredModelOwner(
                ontology_build_store,
                contract_type=RuntimeContextTransitionReceiptV1,
                ref_field="receipt_ref",
                hash_field="receipt_hash",
            ),
            "runtime_context_proposal_provider_descriptor": _AuthorityOwner(
                authority_store,
                method_name="get_proposal_provider_descriptor",
                contract_type=RuntimeContextProposalProviderDescriptorV1,
            ),
            "runtime_context_scope_policy": _AuthorityOwner(
                authority_store,
                method_name="get_scope_policy",
                contract_type=RuntimeContextScopePolicyV1,
            ),
            "runtime_context_element_admission": _AuthorityOwner(
                authority_store,
                method_name="get_admission",
                contract_type=RuntimeContextElementAdmissionReceiptV1,
            ),
        }
        if graph_intent_store is not None:
            owners["graph_intent_brief"] = _GraphIntentOwner(graph_intent_store)
        if derivation_store is not None:
            owners["derivation_record"] = _DerivationOwner(derivation_store)
        self._owners = owners
        self._provider_registry = provider_registry

    @staticmethod
    def _current_identity_context(actor_context: RuntimeContextDTO) -> tuple[dict, dict]:
        users = load_users()
        record = users.get(actor_context.actor_id) if isinstance(users, dict) else None
        if not isinstance(record, dict):
            raise RuntimeContextError("FORBIDDEN_ACTOR")
        memberships = record.get("organization_memberships") or ()
        current = next(
            (
                item
                for item in memberships
                if isinstance(item, dict)
                and str(item.get("organization_id") or "") == actor_context.organization_id
            ),
            None,
        )
        if current is None and str(record.get("organization_id") or "") == actor_context.organization_id:
            current = {
                "organization_id": record.get("organization_id"),
                "membership_status": record.get("membership_status"),
                "member_role": record.get("member_role"),
            }
        if not isinstance(current, dict):
            raise RuntimeContextError("FORBIDDEN_ACTOR")
        role = str(current.get("member_role") or "")
        organization = {
            "organization_id": current.get("organization_id"),
            "membership_status": current.get("membership_status") or "unassigned",
            "member_role": role or None,
            "can_change_settings": (
                str(current.get("membership_status") or "") == "active"
                and role in {"owner", "admin"}
            ),
        }
        return record, organization

    @classmethod
    def require_current_actor(
        cls,
        *,
        actor_context: RuntimeContextDTO,
        authorization_class: Literal["TENANT_MEMBER", "TENANT_ADMIN", "WORKSPACE_SIMULATION", "WORKSPACE_LOCAL"] = "TENANT_MEMBER",
    ) -> None:
        if not isinstance(actor_context, RuntimeContextDTO):
            raise RuntimeContextError("FORBIDDEN_ACTOR")
        if not actor_context.workspace_id:
            raise RuntimeContextError("MISSING_WORKSPACE")
        if authorization_class not in {"TENANT_MEMBER", "TENANT_ADMIN", "WORKSPACE_SIMULATION", "WORKSPACE_LOCAL"}:
            raise RuntimeContextError("FORBIDDEN_ACTOR")
        record, organization = cls._current_identity_context(actor_context)
        if (
            str(record.get("user_id") or "") != actor_context.actor_id
            or str(record.get("workspace_id") or "") != actor_context.workspace_id
            or str(organization.get("organization_id") or "")
            != actor_context.organization_id
        ):
            raise RuntimeContextError("FORBIDDEN_ACTOR")

    def require_current_owner(self, *, actor_context: RuntimeContextDTO) -> None:
        """Exact governed owner permission for explicitly scoped internal adoption.

        This does not establish professional credentials or legal certification.
        """
        if not isinstance(actor_context, RuntimeContextDTO):
            raise RuntimeContextError("FORBIDDEN_ACTOR")
        if not actor_context.workspace_id:
            raise RuntimeContextError("MISSING_WORKSPACE")
        record, organization = self._current_identity_context(actor_context)
        if (str(record.get("user_id") or "") != actor_context.actor_id
            or str(record.get("workspace_id") or "") != actor_context.workspace_id
            or str(organization.get("organization_id") or "") != actor_context.organization_id
        ):
            raise RuntimeContextError("FORBIDDEN_ACTOR")

    def current_actor_authorization_basis(
        self,
        *,
        actor_context: RuntimeContextDTO,
        operation: str,
        subject: ArtifactPinV1,
        authorization_class: Literal["TENANT_MEMBER", "TENANT_ADMIN"] = "TENANT_MEMBER",
    ) -> AuthorizationBasis:
        """Capture the exact current governed membership basis as an embedded value."""

        self.require_current_actor(
            actor_context=actor_context, authorization_class=authorization_class
        )
        _record, organization = self._current_identity_context(actor_context)
        return AuthorizationBasis(
            operation=operation,
            subject=subject,
            scope_ref=f"workspace:{actor_context.workspace_id}",
            actor_id=actor_context.actor_id,
            organization_id=actor_context.organization_id,
            workspace_id=actor_context.workspace_id,
            authorization_class=authorization_class,
            membership_status=str(organization.get("membership_status") or "unassigned"),
            can_change_settings=bool(organization.get("can_change_settings")),
            resolver_ref=_AUTHORITY_RESOLVER_REF,
            resolver_hash=_AUTHORITY_RESOLVER_HASH,
            governing_pins=(),
        )

    def verify_authorization_basis(
        self,
        *,
        actor_context: RuntimeContextDTO,
        basis: AuthorizationBasis,
    ) -> None:
        expected = self.current_actor_authorization_basis(
            actor_context=actor_context,
            operation=basis.operation,
            subject=basis.subject,
            authorization_class=basis.authorization_class,
        )
        if expected != basis:
            raise RuntimeContextError("INVALID_GOVERNANCE")

    def _validate_element_source(
        self,
        *,
        actor_context: RuntimeContextDTO,
        policy: RuntimeContextScopePolicyV1,
        element: RuntimeContextElementV1,
    ) -> tuple[ResolvedRuntimeObject, RuntimeContextAdmissionRuleV1]:
        if element.source_artifact.artifact_kind not in policy.allowed_element_source_kinds:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        resolved = self.resolve_object(
            actor_context=actor_context,
            source=element.source_artifact,
            object_ref=element.object_ref,
        )
        if element.role == "goal" and element.source_artifact.artifact_kind == "graph_intent_brief":
            source = self.resolve_pins(
                actor_context=actor_context, pins=(element.source_artifact,)
            )[0]
            if not isinstance(source, GraphIntentBrief) or not source.compiler_eligible:
                raise RuntimeContextError("ROLE_NOT_ELIGIBLE")
        matches = tuple(
            rule
            for rule in policy.admission_rules
            if (
                rule.source_artifact_kind,
                rule.source_object_kind,
                rule.context_role,
            )
            == (
                element.source_artifact.artifact_kind,
                resolved.source_object_kind,
                element.role,
            )
        )
        if len(matches) != 1:
            raise RuntimeContextError("ROLE_NOT_ELIGIBLE")
        rule = matches[0]
        if not rule.candidate_allowed:
            raise RuntimeContextError("ROLE_NOT_ELIGIBLE")
        if any(
            pin.artifact_kind not in policy.allowed_evidence_kinds
            for pin in element.evidence
        ):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        self.resolve_pins(actor_context=actor_context, pins=element.evidence)
        if element.derivation is not None:
            self.resolve_pins(actor_context=actor_context, pins=(element.derivation,))
        return resolved, rule

    def _validate_admission_receipt(
        self,
        *,
        actor_context: RuntimeContextDTO,
        policy_pin: ArtifactPinV1,
        policy: RuntimeContextScopePolicyV1,
        element: RuntimeContextElementV1,
        receipt_pin: ArtifactPinV1,
    ) -> RuntimeContextElementAdmissionReceiptV1:
        values = self.resolve_pins(actor_context=actor_context, pins=(receipt_pin,))
        receipt = values[0]
        if not isinstance(receipt, RuntimeContextElementAdmissionReceiptV1):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        resolved, rule = self._validate_element_source(
            actor_context=actor_context, policy=policy, element=element
        )
        if (
            receipt.source_artifact,
            receipt.object_ref,
            receipt.source_object_kind,
            receipt.context_role,
            receipt.scope_policy,
            receipt.admission_authorization,
        ) != (
            element.source_artifact,
            element.object_ref,
            resolved.source_object_kind,
            element.role,
            policy_pin,
            rule.admission_authorization,
        ):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        if rule.evidence_required and not receipt.evidence:
            raise RuntimeContextError("MISSING_GOVERNANCE")
        if any(
            pin.artifact_kind not in policy.allowed_evidence_kinds
            for pin in receipt.evidence
        ):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        actual_governance = tuple(
            pin.artifact_kind for pin in receipt.governing_receipts
        )
        if sorted(actual_governance) != sorted(rule.required_governance_kinds):
            raise RuntimeContextError("MISSING_GOVERNANCE")
        if any(
            kind not in policy.allowed_governance_kinds for kind in actual_governance
        ):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        self.resolve_pins(
            actor_context=actor_context,
            pins=(*receipt.governing_receipts, *receipt.evidence),
        )
        return receipt

    def _owner(self, artifact_kind: str) -> RuntimePinOwner:
        try:
            return self._owners[artifact_kind]
        except KeyError as exc:
            raise RuntimeContextError("UNSUPPORTED_ARTIFACT_KIND") from exc

    def resolve_pins(
        self,
        *,
        actor_context: RuntimeContextDTO,
        pins: tuple[ArtifactPinV1, ...],
    ) -> tuple[ResolvedRuntimePin, ...]:
        self.require_current_actor(actor_context=actor_context)
        workspace_id = actor_context.workspace_id
        if workspace_id is None:
            raise RuntimeContextError("MISSING_WORKSPACE")
        resolved: list[ResolvedRuntimePin] = []
        for candidate in pins:
            pin = ArtifactPinV1.model_validate(candidate.model_dump(mode="json"))
            resolved.append(
                self._owner(pin.artifact_kind).resolve_exact(
                    organization_id=actor_context.organization_id,
                    workspace_id=workspace_id,
                    pin=pin,
                )
            )
        return tuple(resolved)

    def resolve_object(
        self,
        *,
        actor_context: RuntimeContextDTO,
        source: ArtifactPinV1,
        object_ref: str,
    ) -> ResolvedRuntimeObject:
        artifact = self.resolve_pins(actor_context=actor_context, pins=(source,))[0]
        return self._owner(source.artifact_kind).resolve_object(
            artifact=artifact, pin=source, object_ref=object_ref
        )

    def verify_operation(
        self,
        *,
        actor_context: RuntimeContextDTO,
        parent: RuntimeTaskContextV1,
        proposal: RuntimeContextTransitionProposalV1,
    ) -> tuple[ArtifactPinV1, ...]:
        self.require_current_actor(actor_context=actor_context)
        workspace_id = actor_context.workspace_id
        if workspace_id is None:
            raise RuntimeContextError("MISSING_WORKSPACE")
        if (
            parent.organization_id,
            parent.workspace_id,
            proposal.organization_id,
            proposal.workspace_id,
        ) != (
            actor_context.organization_id,
            workspace_id,
            actor_context.organization_id,
            workspace_id,
        ):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        expected_parent = ArtifactPinV1(
            artifact_kind="runtime_working_context",
            artifact_ref=parent.context_ref,
            artifact_hash=parent.context_hash,
        )
        if proposal.parent != expected_parent:
            raise RuntimeContextError("HASH_MISMATCH")

        policy_pins = tuple(
            pin
            for pin in proposal.required_governance
            if pin.artifact_kind == "runtime_context_scope_policy"
        )
        if not policy_pins:
            raise RuntimeContextError("MISSING_GOVERNANCE")
        if len(policy_pins) != 1:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        parent_policy_pins = tuple(
            pin
            for pin in parent.semantic_pins
            if pin.artifact_kind == "runtime_context_scope_policy"
        )
        if not parent_policy_pins:
            raise RuntimeContextError("MISSING_GOVERNANCE")
        if len(parent_policy_pins) != 1 or parent_policy_pins[0] != policy_pins[0]:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        policy = self.resolve_pins(actor_context=actor_context, pins=policy_pins)[0]
        if not isinstance(policy, RuntimeContextScopePolicyV1):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        self.resolve_pins(actor_context=actor_context, pins=parent.semantic_pins)
        if not (policy.effective_from <= proposal.proposed_at <= policy.effective_to):
            raise RuntimeContextError("POLICY_NOT_EFFECTIVE")

        goal = self.resolve_pins(actor_context=actor_context, pins=(parent.goal,))[0]
        if parent.goal.artifact_kind not in policy.allowed_goal_kinds:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        if isinstance(goal, GraphIntentBrief) and not goal.compiler_eligible:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        self.resolve_pins(actor_context=actor_context, pins=(parent.world.source_graph,))
        if parent.world.source_graph.artifact_kind not in policy.allowed_world_source_kinds:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        if proposal.trigger.artifact_kind not in policy.allowed_trigger_kinds:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        self.resolve_pins(actor_context=actor_context, pins=(proposal.trigger,))

        descriptor_values = self.resolve_pins(
            actor_context=actor_context, pins=(proposal.provider,)
        )
        descriptor = descriptor_values[0]
        if not isinstance(descriptor, RuntimeContextProposalProviderDescriptorV1):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        if proposal.provider not in policy.allowed_provider_descriptors:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        if self._provider_registry is None:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        registration = self._provider_registry.get(
            implementation_id=descriptor.implementation_id,
            implementation_version=descriptor.implementation_version,
        )
        if registration.provider.descriptor(
            organization_id=actor_context.organization_id,
            workspace_id=workspace_id,
            configuration=descriptor.configuration,
        ) != descriptor:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        constructed_operations = registration.provider.construct_operations(
            descriptor=descriptor, operations=proposal.operations
        )
        if constructed_operations != proposal.operations:
            raise RuntimeContextError("INVALID_GOVERNANCE")

        if not proposal.operations or len(proposal.operations) > policy.max_operations:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        allowed_operations = set(policy.allowed_operations)
        if any(operation.operation not in allowed_operations for operation in proposal.operations):
            raise RuntimeContextError("INVALID_GOVERNANCE")

        elements = {element.object_ref: element for element in parent.elements}
        if len(elements) > policy.max_elements:
            raise RuntimeContextError("INVALID_GOVERNANCE")
        consumed_receipts: list[ArtifactPinV1] = []
        for element in parent.elements:
            self._validate_element_source(
                actor_context=actor_context, policy=policy, element=element
            )
            if element.admission_receipt is not None:
                self._validate_admission_receipt(
                    actor_context=actor_context,
                    policy_pin=policy_pins[0],
                    policy=policy,
                    element=element,
                    receipt_pin=element.admission_receipt,
                )
                consumed_receipts.append(element.admission_receipt)

        for operation in proposal.operations:
            if isinstance(operation, AddCandidateOperationV1):
                if operation.element.object_ref in elements:
                    raise RuntimeContextError("INVALID_GOVERNANCE")
                self._validate_element_source(
                    actor_context=actor_context,
                    policy=policy,
                    element=operation.element,
                )
                elements[operation.element.object_ref] = operation.element
                if len(elements) > policy.max_elements:
                    raise RuntimeContextError("INVALID_GOVERNANCE")
            elif isinstance(operation, AdmitOperationV1):
                element = elements.get(operation.object_ref)
                if element is None or element.admission != "candidate":
                    raise RuntimeContextError("INVALID_GOVERNANCE")
                self._validate_admission_receipt(
                    actor_context=actor_context,
                    policy_pin=policy_pins[0],
                    policy=policy,
                    element=element,
                    receipt_pin=operation.admission_receipt,
                )
                consumed_receipts.append(operation.admission_receipt)
                elements[operation.object_ref] = element.model_copy(
                    update={
                        "admission": "admitted",
                        "admission_receipt": operation.admission_receipt,
                    }
                )
            elif isinstance(operation, ActivateOperationV1):
                element = elements.get(operation.object_ref)
                if (
                    element is None
                    or element.admission != "admitted"
                    or element.admission_receipt is None
                ):
                    raise RuntimeContextError("INVALID_GOVERNANCE")
                self._validate_admission_receipt(
                    actor_context=actor_context,
                    policy_pin=policy_pins[0],
                    policy=policy,
                    element=element,
                    receipt_pin=element.admission_receipt,
                )
                consumed_receipts.append(element.admission_receipt)
                elements[operation.object_ref] = element.model_copy(
                    update={"active": True}
                )
            elif isinstance(operation, DeactivateOperationV1):
                element = elements.get(operation.object_ref)
                if element is None or not element.active:
                    raise RuntimeContextError("INVALID_GOVERNANCE")
                if element.admission_receipt is None:
                    raise RuntimeContextError("INVALID_GOVERNANCE")
                self._validate_admission_receipt(
                    actor_context=actor_context,
                    policy_pin=policy_pins[0],
                    policy=policy,
                    element=element,
                    receipt_pin=element.admission_receipt,
                )
                consumed_receipts.append(element.admission_receipt)
                elements[operation.object_ref] = element.model_copy(
                    update={"active": False}
                )
        for pins, allowed in (
            (proposal.evidence, set(policy.allowed_evidence_kinds)),
            (proposal.justifications, set(policy.allowed_justification_kinds)),
        ):
            if any(pin.artifact_kind not in allowed for pin in pins):
                raise RuntimeContextError("INVALID_GOVERNANCE")
            self.resolve_pins(actor_context=actor_context, pins=pins)
        governance = tuple(
            pin for pin in proposal.required_governance if pin not in policy_pins
        )
        if any(pin.artifact_kind not in policy.allowed_governance_kinds for pin in governance):
            raise RuntimeContextError("INVALID_GOVERNANCE")
        self.resolve_pins(actor_context=actor_context, pins=governance)
        used_governance = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash): pin
            for pin in (*proposal.required_governance, *consumed_receipts)
        }
        return tuple(used_governance[key] for key in sorted(used_governance))


__all__ = [
    "ResolvedRuntimeObject",
    "ResolvedRuntimePin",
    "RuntimeAuthorityResolver",
    "RuntimePinOwner",
]
