"""Atomic SQLite persistence for immutable runtime task-context streams.

The module path and persisted ``runtime_working_context`` identifiers are historical wire
names retained for replay compatibility; live Python ownership is ``RuntimeTaskContext*``
per ADR-0027 and must not be confused with semantic-kernel ``WorkingContext``.
"""

from __future__ import annotations

from storage.hosted_postgres import sql_query,begin_write,in_transaction

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Literal, Mapping, TypeVar, overload

from pydantic import BaseModel, ValidationError

from contracts import runtime_context_reconstruction as rr

from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_working_context import (
    RuntimeContextTransitionProposalV1,
    RuntimeContextTransitionReceiptV1,
    RuntimeContextTransitionV1,
    RuntimeTaskContextV1,
    RuntimeTaskContextV2,
    RuntimeContextTransitionProposalV2,
    RuntimeContextTransitionV2,
    RuntimeContextTransitionReceiptV2,
    RuntimeContextTransitionReceiptV3,
    RuntimeContextElementV1,
    RuntimeTaskContextV3,
    context_hash_basis,
    receipt_hash_basis,
)
from contracts.runtime_temporal_source import RuntimeTemporalContextSelectionV1
from eos.artifact_hashing import canonical_json, content_hash
from storage.ontology_build_store import (
    OntologyBuildStore,
    _IDENTITY_FIELDS,
    _RUNTIME_CONTEXT_AUTHORITY_SCHEMAS,
    _RUNTIME_CONTEXT_CAS_SCHEMAS,
    _SEED_CAS_SCHEMAS,
    _WORKSPACE_REGISTRY_SCHEMAS,
    _FORMATION_EVIDENCE_SCHEMAS,
    _LEGACY_CONTEXT_SCHEMAS,
    _ADVISORY_CONTEXT_SCHEMAS,
    _validate_registered_artifact,
    _TASK909_MODELS,
)


if TYPE_CHECKING:
    from services.runtime_context_transition_service import RuntimeAuthorityResolver
    from services.runtime_working_context_service import GovernedRuntimeSessionResolver


_CONTEXT_KIND = "runtime_working_context"
_PROPOSAL_KIND = "runtime_context_transition_proposal"
_TRANSITION_KIND = "runtime_context_transition"
ReadT = TypeVar("ReadT")
_RuntimeContextErrorOrigin = Literal[
    "ARTIFACT_LOOKUP",
    "AUTHORITY_SESSION",
    "RETAINED_CLOSURE",
]


ContextVersion = RuntimeTaskContextV1 | RuntimeTaskContextV2 | RuntimeTaskContextV3
ProposalVersion = RuntimeContextTransitionProposalV1 | RuntimeContextTransitionProposalV2
TransitionVersion = RuntimeContextTransitionV1 | RuntimeContextTransitionV2
ReceiptVersion = RuntimeContextTransitionReceiptV1 | RuntimeContextTransitionReceiptV2 | RuntimeContextTransitionReceiptV3


@dataclass(frozen=True, slots=True)
class VerifiedRuntimeContextRequestBundleV2:
    receipt: RuntimeContextTransitionReceiptV2
    manifest: rr.RuntimeContextPersistenceManifestV1
    successor: RuntimeTaskContextV2
    transition: RuntimeContextTransitionV2 | None
    proposal: RuntimeContextTransitionProposalV2 | None
    formation: rr.FrozenContextFormationV1
    formation_command: rr.RuntimeContextFormationCommandV1 | None
    related_artifacts: tuple[BaseModel, ...]


from services.runtime_errors import RuntimeContextError


def _with_context_error_origin(
    error: RuntimeContextError,
    origin: _RuntimeContextErrorOrigin,
) -> RuntimeContextError:
    """Attach private transport provenance without changing public error data."""

    if error._origin is not None:
        return error
    return RuntimeContextError(error.code, error.details, _origin=origin)


def _pin(artifact_kind: str, artifact_ref: str, artifact_hash: str) -> ArtifactPinV1:
    return ArtifactPinV1(
        artifact_kind=artifact_kind,
        artifact_ref=artifact_ref,
        artifact_hash=artifact_hash,
    )


def _model_pin(artifact: BaseModel) -> ArtifactPinV1:
    schema = getattr(artifact, "schema_version", "")
    if schema in _TASK909_MODELS and hasattr(artifact, "artifact_pin"):
        return artifact.artifact_pin()
    if schema == "runtime_working_context.v3":
        return _pin("runtime_working_context", artifact.context_ref, artifact.context_hash)
    if schema == "runtime_context_transition_receipt.v3":
        return _pin(
            "runtime_context_transition_receipt",
            artifact.receipt_ref,
            artifact.receipt_hash,
        )
    if schema in _WORKSPACE_REGISTRY_SCHEMAS:
        return rr.artifact_pin_for(_validated(artifact))
    payload = artifact.model_dump(mode="json")
    type(artifact).model_validate(payload)
    try:
        ref_field, hash_field = _IDENTITY_FIELDS[schema]
    except KeyError as exc:
        raise RuntimeContextError("INVALID_STATE") from exc
    return _pin(schema.removesuffix(".v1"), str(payload[ref_field]), str(payload[hash_field]))


def _validated(artifact: BaseModel) -> BaseModel:
    try:
        schema = getattr(artifact, "schema_version", "")
        if schema in _TASK909_MODELS:
            if type(artifact) not in _TASK909_MODELS[schema]:
                raise ValueError("INVALID_STATE")
            return type(artifact).model_validate(artifact.model_dump(mode="json"))
        return _validate_registered_artifact(artifact)
    except (ValueError, TypeError, AttributeError) as exc:
        code = "HASH_MISMATCH" if str(exc) == "HASH_MISMATCH" else "INVALID_STATE"
        raise RuntimeContextError(code) from exc


def _dump_pin(pin: ArtifactPinV1 | None) -> dict[str, str] | None:
    return None if pin is None else pin.model_dump(mode="json")


def runtime_context_persistence_command_hash(
    *,
    organization_id: str,
    workspace_id: str,
    actor_id: str,
    operation: str,
    request_id: str,
    expected_parent: ArtifactPinV1 | None,
    expected_version: int | None,
    successor: RuntimeTaskContextV1,
    transition: RuntimeContextTransitionV1 | None,
    proposal: RuntimeContextTransitionProposalV1 | None,
    related_artifacts: tuple[BaseModel, ...] = (),
) -> str:
    """Hash the complete persistence command independently of receipt metadata.

    Related artifact kinds derive from their ASCII-stable schema discriminants;
    their exact refs and hashes are sorted because the related set is unordered.
    """

    successor = type(successor).model_validate(successor.model_dump(mode="json"))
    if transition is not None:
        transition = RuntimeContextTransitionV1.model_validate(
            transition.model_dump(mode="json")
        )
    if proposal is not None:
        proposal = RuntimeContextTransitionProposalV1.model_validate(
            proposal.model_dump(mode="json")
        )
    related_pins = sorted(
        (_model_pin(artifact).model_dump(mode="json") for artifact in related_artifacts),
        key=lambda item: (
            item["artifact_kind"], item["artifact_ref"], item["artifact_hash"]
        ),
    )
    basis = {
        "organization_id": organization_id,
        "workspace_id": workspace_id,
        "actor_id": actor_id,
        "operation": operation,
        "request_id": request_id,
        "expected_parent": _dump_pin(expected_parent),
        "expected_version": expected_version,
        "successor": _dump_pin(
            _pin(_CONTEXT_KIND, successor.context_ref, successor.context_hash)
        ),
        "transition": None
        if transition is None
        else _dump_pin(
            _pin(
                _TRANSITION_KIND,
                transition.transition_ref,
                transition.transition_hash,
            )
        ),
        "proposal": None
        if proposal is None
        else _dump_pin(_pin(_PROPOSAL_KIND, proposal.proposal_ref, proposal.proposal_hash)),
        "related_artifacts": related_pins,
    }
    return "sha256:" + content_hash(basis)


def _same_pin(left: ArtifactPinV1 | None, right: ArtifactPinV1 | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.model_dump(mode="json") == right.model_dump(mode="json")


class RuntimeTaskContextStore:
    def __init__(self, build_store: OntologyBuildStore) -> None:
        if not isinstance(build_store, OntologyBuildStore):
            raise TypeError("build_store must be OntologyBuildStore")
        self._build_store = build_store
        self._connection = build_store._connection

    def close(self) -> None:
        self._build_store.close()

    def get_advisory_receipt_for_request(
        self,
        *,
        actor_context: RuntimeContextDTO,
        command: rr.AdvisoryContextCommandV1,
    ) -> rr.AdvisoryContextReceiptV1 | None:
        from services.runtime_context_formation_service import (
            _revalidate,
            _verify_advisory_triple,
            _require_bound_advisory_access,
        )

        command = _revalidate(rr.AdvisoryContextCommandV1, command)
        _require_bound_advisory_access(
            build_store=self._build_store,
            actor_context=actor_context,
            session_ref=command.session_ref,
            pins=tuple(
                pin
                for pair in command.request.outcomes
                for pin in (pair.episode, pair.assessment)
            ),
        )
        if any(
            getattr(command, field) != getattr(actor_context, field)
            for field in ("organization_id", "workspace_id", "actor_id")
        ):
            raise RuntimeContextError(
                "FORBIDDEN_SCOPE", _origin="AUTHORITY_SESSION"
            )

        def read():
            row = self._connection.execute(
                sql_query("SELECT request_hash, receipt_ref, receipt_hash FROM workspace_operation_request_index "
                "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash, receipt_ref, receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
                (
                    command.organization_id,
                    command.workspace_id,
                    command.actor_id,
                    command.operation,
                    command.request.request_id,
                ),
            ).fetchone()
            if row is None:
                # Losing the request index does not make a surviving publication
                # new. The deterministic command reference binds this request.
                if self._connection.execute(
                    sql_query("SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref=%s'),
                    (command.command_ref,),
                ).fetchone() is not None:
                    raise RuntimeContextError(
                        "INCOMPLETE_CLOSURE", _origin="RETAINED_CLOSURE"
                    )
                return None
            if row[0] != command.command_hash:
                raise RuntimeContextError("REQUEST_CONFLICT")
            try:
                receipt = self._read_artifact(
                    command.organization_id,
                    command.workspace_id,
                    _pin("advisory_context_receipt", row[1], row[2]),
                    missing="INCOMPLETE_CLOSURE",
                )
            except RuntimeContextError as exc:
                raise _with_context_error_origin(exc, "RETAINED_CLOSURE") from exc
            if type(receipt) is not rr.AdvisoryContextReceiptV1:
                raise RuntimeContextError(
                    "INVALID_STATE", _origin="RETAINED_CLOSURE"
                )
            try:
                saved_command = self._read_artifact(
                    command.organization_id,
                    command.workspace_id,
                    receipt.command,
                    missing="INCOMPLETE_CLOSURE",
                )
                formation = self._get_advisory_formation(
                    actor_context=actor_context,
                    pin=receipt.formation,
                    lookup_origin="RETAINED_CLOSURE",
                )
                _verify_advisory_triple(
                    actor_context=actor_context,
                    command=saved_command,
                    formation=formation,
                    receipt=receipt,
                )
            except RuntimeContextError as exc:
                raise _with_context_error_origin(exc, "RETAINED_CLOSURE") from exc
            if saved_command != command or receipt.request_hash != row[0]:
                raise RuntimeContextError("REQUEST_CONFLICT")
            return receipt

        return self._consistent_read(read)

    def get_advisory_formation(
        self,
        *,
        actor_context: RuntimeContextDTO,
        pin: ArtifactPinV1,
    ) -> rr.AdvisoryContextFormationV1:
        return self._get_advisory_formation(
            actor_context=actor_context,
            pin=pin,
            lookup_origin="ARTIFACT_LOOKUP",
        )

    def _get_advisory_formation(
        self,
        *,
        actor_context: RuntimeContextDTO,
        pin: ArtifactPinV1,
        lookup_origin: _RuntimeContextErrorOrigin,
    ) -> rr.AdvisoryContextFormationV1:
        from services.runtime_context_formation_service import (
            verify_advisory_formation_v1,
            _require_bound_advisory_access,
            _revalidate,
        )

        context = getattr(self._build_store, "_advisory_owner_context", None)
        if context is None:
            raise RuntimeContextError("INVALID_STATE")
        _require_bound_advisory_access(
            build_store=self._build_store,
            actor_context=actor_context,
            session_ref=context.session_ref,
            pins=(),
        )

        pin = _revalidate(ArtifactPinV1, pin)
        if pin.artifact_kind != "advisory_context_formation":
            raise RuntimeContextError("INVALID_STATE")
        row = self._connection.execute(
            sql_query("SELECT organization_id, workspace_id, schema_version, artifact_hash, payload_json "
            "FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT organization_id, workspace_id, schema_version, artifact_hash, payload_json FROM ontology_build_artifacts WHERE artifact_ref=%s'),
            (pin.artifact_ref,),
        ).fetchone()
        if row is None:
            raise RuntimeContextError(
                "INCOMPLETE_CLOSURE", _origin=lookup_origin
            )
        if row[:2] != (actor_context.organization_id, actor_context.workspace_id):
            raise RuntimeContextError("FORBIDDEN_SCOPE", _origin=lookup_origin)
        if row[2] != "advisory_context_formation.v1" or row[3] != pin.artifact_hash:
            raise RuntimeContextError("HASH_MISMATCH")
        try:
            value = verify_advisory_formation_v1(raw=row[4].encode("utf-8"))
        except RuntimeContextError as exc:
            if exc.code == "FORBIDDEN_SCOPE" and exc._origin is None:
                raise _with_context_error_origin(
                    exc, "RETAINED_CLOSURE"
                ) from exc
            raise
        if (
            rr.artifact_pin_for(value) != pin
            or canonical_json(value.model_dump(mode="json")) != row[4]
        ):
            raise RuntimeContextError("HASH_MISMATCH")
        if (value.organization_id, value.workspace_id) != row[:2]:
            raise RuntimeContextError(
                "FORBIDDEN_SCOPE", _origin="RETAINED_CLOSURE"
            )
        if value.actor_id != actor_context.actor_id:
            raise RuntimeContextError("FORBIDDEN_SCOPE", _origin=lookup_origin)
        _require_bound_advisory_access(
            build_store=self._build_store,
            actor_context=actor_context,
            session_ref=context.session_ref,
            pins=tuple(source.source for source in value.sources),
        )
        return value

    def publish_advisory(
        self,
        *,
        actor_context: RuntimeContextDTO,
        command: rr.AdvisoryContextCommandV1,
        formation: rr.AdvisoryContextFormationV1,
        receipt: rr.AdvisoryContextReceiptV1,
        verify_before_commit: Callable[[], None],
    ) -> rr.AdvisoryContextReceiptV1:
        return self._build_store._publish_advisory_context(
            actor_context=actor_context,
            command=command,
            formation=formation,
            receipt=receipt,
            verify_before_commit=verify_before_commit,
        )

    def _consistent_read(self, read: Callable[[], ReadT]) -> ReadT:
        owns_read_transaction = not in_transaction(self._connection)
        if owns_read_transaction:
            self._connection.execute(sql_query("BEGIN","BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        try:
            result = read()
            if owns_read_transaction:
                self._connection.commit()
            return result
        except BaseException:
            if owns_read_transaction:
                self._connection.rollback()
            raise

    @staticmethod
    def _decode_registered(schema: str, encoded: str) -> BaseModel:
        if schema not in _WORKSPACE_REGISTRY_SCHEMAS and schema not in _TASK909_MODELS:
            raise RuntimeContextError("INVALID_STATE")
        try:
            payload = json.loads(encoded)
        except (ValueError, TypeError) as exc:
            raise RuntimeContextError("HASH_MISMATCH") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != schema:
            raise RuntimeContextError("INVALID_STATE")
        if schema in _TASK909_MODELS:
            models = _TASK909_MODELS[schema]
            if len(models) != 1:
                raise RuntimeContextError("INVALID_STATE")
            model = models[0]
        else:
            model = rr.artifact_model_for_schema(schema)
        try:
            value = model.model_validate(payload)
        except ValidationError as exc:
            identity_failure = any("identity" in error["msg"] or "hash mismatch" in error["msg"] for error in exc.errors())
            raise RuntimeContextError("HASH_MISMATCH" if identity_failure else "INVALID_STATE") from exc
        if canonical_json(value.model_dump(mode="json")) != encoded:
            raise RuntimeContextError("HASH_MISMATCH")
        return value

    def _read_artifact(
        self, organization_id: str, workspace_id: str, pin: ArtifactPinV1,
        *, missing: str = "NOT_FOUND",
    ) -> BaseModel:
        row = self._connection.execute(
            sql_query("SELECT organization_id, workspace_id, schema_version, artifact_hash, payload_json "
            "FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT organization_id, workspace_id, schema_version, artifact_hash, payload_json FROM ontology_build_artifacts WHERE artifact_ref=%s'), (pin.artifact_ref,)
        ).fetchone()
        if row is None:
            raise RuntimeContextError(missing)
        org, workspace, schema, digest, encoded = row
        if (org, workspace) != (organization_id, workspace_id):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        value = self._decode_registered(schema, encoded)
        if (value.organization_id, value.workspace_id) != (organization_id, workspace_id):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        if digest != pin.artifact_hash or _model_pin(value) != pin:
            raise RuntimeContextError("HASH_MISMATCH")
        return value

    def _check_existing_bytes(
        self, prepared: tuple[tuple[str, str, str, str, str, str], ...],
    ) -> None:
        for org, workspace, ref, digest, schema, encoded in prepared:
            row = self._connection.execute(
                sql_query("SELECT organization_id, workspace_id, artifact_hash, schema_version, payload_json "
                "FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT organization_id, workspace_id, artifact_hash, schema_version, payload_json FROM ontology_build_artifacts WHERE artifact_ref=%s'), (ref,)
            ).fetchone()
            if row is not None and row != (org, workspace, digest, schema, encoded):
                raise RuntimeContextError("HASH_MISMATCH")

    @staticmethod
    def _validate_parent_basis(parent: ContextVersion, successor: ContextVersion) -> None:
        if parent.schema_version != successor.schema_version:
            raise RuntimeContextError("INVALID_STATE")
        for field in ("organization_id", "workspace_id", "session_ref", "goal", "world", "semantic_pins", "stream_ref"):
            if getattr(parent, field) != getattr(successor, field):
                raise RuntimeContextError("INVALID_STATE")

    def _formation_association(
        self, formation_pin: ArtifactPinV1, organization_id: str, workspace_id: str,
        related: tuple[BaseModel, ...] = (), seen: frozenset[str] = frozenset(),
    ) -> tuple[rr.FrozenContextFormationV1, rr.ReconstructionQueryV1]:
        if formation_pin.artifact_ref in seen:
            raise RuntimeContextError("INVALID_STATE")
        seen = seen | {formation_pin.artifact_ref}
        supplied = {rr.artifact_pin_for(value).artifact_ref: value for value in related}

        def read(pin: ArtifactPinV1) -> BaseModel:
            value = supplied.get(pin.artifact_ref)
            if value is None:
                return self._read_artifact(organization_id, workspace_id, pin)
            if (value.organization_id, value.workspace_id) != (organization_id, workspace_id):
                raise RuntimeContextError("FORBIDDEN_SCOPE")
            if rr.artifact_pin_for(value) != pin:
                raise RuntimeContextError("HASH_MISMATCH")
            return value

        formation = read(formation_pin)
        if type(formation) is not rr.FrozenContextFormationV1:
            raise RuntimeContextError("INVALID_STATE")
        query, candidates, governance, reconstructed, admitted = (
            read(getattr(formation, field)) for field in
            ("query", "candidate_set", "governance", "reconstruction", "admitted")
        )
        if not (type(query) is rr.ReconstructionQueryV1 and type(candidates) is rr.MemoryCandidateSetV1
                and type(governance) is rr.ReconstructionGovernanceCaptureV1
                and type(reconstructed) is rr.ReconstructedContextV1
                and type(admitted) is rr.AdmittedReconstructedContextV1):
            raise RuntimeContextError("INVALID_STATE")
        captures = tuple(read(pin) for pin in candidates.captures)
        candidate_rows = tuple(read(pin) for pin in candidates.candidates)
        for item in reconstructed.items:
            for dependency in item.candidate_dependencies:
                candidate = read(dependency)
                if type(candidate) is not rr.MemoryCandidateV1 or dependency not in candidates.candidates:
                    raise RuntimeContextError("INVALID_STATE")
                read(candidate.capture)
        origin_bindings = ()
        if formation.retained_element_bindings:
            if query.reasoning_origin is None:
                raise RuntimeContextError("INVALID_STATE")
            origin = self.get_context(organization_id=organization_id,
                workspace_id=workspace_id, pin=query.reasoning_origin)
            if origin.schema_version != "runtime_working_context.v2":
                raise RuntimeContextError("INVALID_STATE")
            origin_formation, _ = self._formation_association(
                origin.formation, organization_id, workspace_id, seen=seen)
            origin_bindings = origin_formation.element_bindings + origin_formation.retained_element_bindings
        try:
            rr.FrozenFormationBundleV1(formation=formation,
                artifacts=(query, candidates, governance, reconstructed, admitted, *captures, *candidate_rows))
            rr.CandidateRetrievalV1(captures=captures, candidates=candidate_rows, candidate_set=candidates)
            rr.validate_formation_element_bindings(formation, reconstructed, admitted,
                origin_bindings=origin_bindings)
        except (ValueError, TypeError) as exc:
            raise RuntimeContextError("INVALID_STATE") from exc
        if candidates.query != formation.query or governance.query != formation.query:
            raise RuntimeContextError("INVALID_STATE")
        for field in ("operator", "providers", "weighting", "admissibility_policy", "verifier_registry"):
            if getattr(query, field) != getattr(formation, field):
                raise RuntimeContextError("INVALID_STATE")
        for field in ("scope_policy", "admissibility_policy", "weighting"):
            if getattr(query, field) != getattr(governance, field):
                raise RuntimeContextError("INVALID_STATE")
        return formation, query

    def _validate_proposal_formation(
        self, proposal: ProposalVersion, parent: ContextVersion,
        related: tuple[BaseModel, ...] = (),
    ) -> None:
        if type(parent) is not RuntimeTaskContextV2 or type(proposal) is not RuntimeContextTransitionProposalV2:
            raise RuntimeContextError("INVALID_STATE")
        formation, query = self._formation_association(proposal.formation,
            proposal.organization_id, proposal.workspace_id, related)
        if any(op.operation == "ADD_CANDIDATE" for op in proposal.operations) and proposal.formation == parent.formation:
            raise RuntimeContextError("INVALID_STATE")
        if proposal.formation != parent.formation:
            if query.reasoning_origin != rr.artifact_pin_for(parent):
                raise RuntimeContextError("INVALID_STATE")
            if all(op.operation != "ADD_CANDIDATE" for op in proposal.operations):
                raise RuntimeContextError("INVALID_STATE")
        for field in ("session_ref", "goal", "world", "semantic_pins"):
            if getattr(query, field) != getattr(parent, field):
                raise RuntimeContextError("INVALID_STATE")
        # Only the exact stored parent can supply retained binding lineage.
        parent_formation, _ = self._formation_association(parent.formation,
            parent.organization_id, parent.workspace_id)
        if proposal.formation != parent.formation:
            required = parent_formation.element_bindings + parent_formation.retained_element_bindings
            if any(binding not in formation.retained_element_bindings for binding in required):
                raise RuntimeContextError("INVALID_STATE")
        selected = {binding.body.object.object_ref: binding.body for binding in formation.element_bindings}
        for operation in proposal.operations:
            if operation.operation == "ADD_CANDIDATE":
                element = operation.element
                body = selected.get(element.object_ref)
                if body is None or element.source_artifact != body.object.source or any(getattr(element, field) != getattr(body, field)
                    for field in ("role", "derived", "evidence", "derivation")):
                    raise RuntimeContextError("INVALID_STATE")

    def _validate_v2_association(
        self, manifest: rr.RuntimeContextPersistenceManifestV1,
        receipt: RuntimeContextTransitionReceiptV2, successor: RuntimeTaskContextV2,
        transition: RuntimeContextTransitionV2 | None,
        proposal: RuntimeContextTransitionProposalV2 | None,
        related: tuple[BaseModel, ...],
    ) -> tuple[rr.FrozenContextFormationV1, rr.RuntimeContextFormationCommandV1 | None]:
        by_pin = {rr.artifact_pin_for(value).artifact_ref: value for value in related}
        formation = by_pin.get(manifest.formation.artifact_ref)
        command = None if manifest.formation_command is None else by_pin.get(manifest.formation_command.artifact_ref)
        if type(formation) is not rr.FrozenContextFormationV1:
            raise RuntimeContextError("INVALID_STATE")
        if command is not None and type(command) is not rr.RuntimeContextFormationCommandV1:
            raise RuntimeContextError("INVALID_STATE")
        try:
            rr.validate_persistence_association(manifest, receipt, successor, formation,
                related_artifacts=related, transition=transition, proposal=proposal, formation_command=command)
        except (ValueError, TypeError, AttributeError) as exc:
            raise RuntimeContextError("INVALID_STATE") from exc
        if command is not None:
            query = by_pin.get(formation.query.artifact_ref)
            if type(query) is not rr.ReconstructionQueryV1:
                raise RuntimeContextError("INVALID_STATE")
            for field in ("session_ref", "goal", "world", "semantic_pins"):
                if getattr(query, field) != getattr(successor, field):
                    raise RuntimeContextError("INVALID_STATE")
            if query.reasoning_origin != successor.branch_origin:
                raise RuntimeContextError("INVALID_STATE")
            receipts = tuple(sorted(
                (element.admission_receipt for element in successor.elements if element.admission_receipt is not None),
                key=lambda pin: (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)))
            if command.element_receipts != receipts:
                raise RuntimeContextError("INVALID_STATE")
            bindings = {binding.body.object.object_ref: binding.body
                for binding in formation.element_bindings + formation.retained_element_bindings}
            if set(bindings) != {element.object_ref for element in successor.elements}:
                raise RuntimeContextError("INVALID_STATE")
            for element in successor.elements:
                body = bindings[element.object_ref]
                if element.source_artifact != body.object.source or any(
                    getattr(element, field) != getattr(body, field)
                    for field in ("role", "derived", "evidence", "derivation")):
                    raise RuntimeContextError("INVALID_STATE")
        return formation, command

    def get_request_bundle(self, *, organization_id: str, workspace_id: str,
                           actor_id: str, operation: Literal["CREATE", "BRANCH", "COMMIT"],
                           request_id: str) -> VerifiedRuntimeContextRequestBundleV2 | None:
        return self._consistent_read(lambda: self._get_request_bundle_once(
            organization_id=organization_id, workspace_id=workspace_id,
            actor_id=actor_id, operation=operation, request_id=request_id))

    def _get_request_bundle_once(
        self, *, organization_id: str, workspace_id: str, actor_id: str,
        operation: str, request_id: str,
    ) -> VerifiedRuntimeContextRequestBundleV2 | None:
        key = (organization_id, workspace_id, actor_id, operation, request_id)
        indexes = self._connection.execute(
            sql_query("SELECT request_hash, receipt_ref, receipt_hash FROM runtime_context_request_index "
            "WHERE organization_id=? AND workspace_id=? AND actor_id=? AND operation=? AND request_id=?",'SELECT request_hash, receipt_ref, receipt_hash FROM runtime_context_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_id=%s AND operation=%s AND request_id=%s'), key
        ).fetchmany(2)
        if len(indexes) > 1:
            raise RuntimeContextError("INVALID_STATE")
        if not indexes:
            orphan = self._connection.execute(
                sql_query("SELECT 1 FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? "
                "AND schema_version IN ('runtime_context_formation_command.v1', "
                "'runtime_context_persistence_manifest.v1', 'runtime_context_transition_receipt.v2') "
                "AND json_extract(payload_json, '$.actor_id')=? "
                "AND json_extract(payload_json, '$.operation')=? "
                "AND json_extract(payload_json, '$.request_id')=? LIMIT 1","SELECT 1 FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND schema_version IN ('runtime_context_formation_command.v1', 'runtime_context_persistence_manifest.v1', 'runtime_context_transition_receipt.v2') AND (payload_json::jsonb #>> '{actor_id}')=%s AND (payload_json::jsonb #>> '{operation}')=%s AND (payload_json::jsonb #>> '{request_id}')=%s LIMIT 1"), key
            ).fetchone()
            if orphan:
                raise RuntimeContextError("ORPHAN_REQUEST_EVIDENCE")
            return None
        index = indexes[0]
        request_hash, receipt_ref, receipt_hash = index
        try:
            receipt = self._read_artifact(organization_id, workspace_id,
                _pin("runtime_context_transition_receipt", receipt_ref, receipt_hash))
            if receipt.schema_version == "runtime_context_transition_receipt.v1":
                raise RuntimeContextError("LEGACY_RECEIPT_READ_ONLY")
            if type(receipt) is not RuntimeContextTransitionReceiptV2:
                raise RuntimeContextError("INVALID_STATE")
            if (receipt.organization_id, receipt.workspace_id, receipt.actor_id, receipt.operation, receipt.request_id) != key:
                raise RuntimeContextError("INVALID_STATE")
            if receipt.request_hash != request_hash:
                raise RuntimeContextError("HASH_MISMATCH")
            receipt_indexes = self._connection.execute(
                sql_query("SELECT COUNT(*) FROM runtime_context_request_index WHERE receipt_ref=?",'SELECT COUNT(*) FROM runtime_context_request_index WHERE receipt_ref=%s'),
                (receipt_ref,),
            ).fetchone()[0]
            if receipt_indexes != 1:
                raise RuntimeContextError("INVALID_STATE")
            manifest = self._read_artifact(organization_id, workspace_id, receipt.persistence_manifest)
            if type(manifest) is not rr.RuntimeContextPersistenceManifestV1:
                raise RuntimeContextError("INVALID_STATE")
            if rr.runtime_context_persistence_command_hash_v2(manifest) != request_hash:
                raise RuntimeContextError("HASH_MISMATCH")
            related = tuple(self._read_artifact(organization_id, workspace_id, pin) for pin in manifest.related_artifacts)
            successor = self.get_context(organization_id=organization_id, workspace_id=workspace_id, pin=manifest.successor)
            transition = None if manifest.transition is None else self._read_artifact(organization_id, workspace_id, manifest.transition)
            proposal = None if manifest.proposal is None else self._read_artifact(organization_id, workspace_id, manifest.proposal)
            if type(successor) is not RuntimeTaskContextV2 or (transition is not None and type(transition) is not RuntimeContextTransitionV2) or (proposal is not None and type(proposal) is not RuntimeContextTransitionProposalV2):
                raise RuntimeContextError("INVALID_STATE")
            formation, command = self._validate_v2_association(manifest, receipt, successor, transition, proposal, related)
            self._validate_append_links(successor=successor, expected_parent=manifest.expected_parent,
                expected_version=manifest.expected_version, transition=transition, proposal=proposal, receipt=receipt)
            self._formation_association(manifest.formation, organization_id, workspace_id)
            if successor.parent is not None:
                parent = self.get_context(organization_id=organization_id, workspace_id=workspace_id, pin=successor.parent)
                if parent.version != manifest.expected_version:
                    raise RuntimeContextError("INVALID_STATE")
                self._validate_parent_basis(parent, successor)
                self._validate_proposal_formation(proposal, parent)
            if successor.branch_origin is not None:
                origin = self.get_context(organization_id=organization_id, workspace_id=workspace_id, pin=successor.branch_origin)
                if origin.stream_ref == successor.stream_ref:
                    raise RuntimeContextError("INVALID_STATE")
            return VerifiedRuntimeContextRequestBundleV2(receipt, manifest, successor,
                transition, proposal, formation, command, related)
        except RuntimeContextError as exc:
            if exc.code == "NOT_FOUND":
                raise RuntimeContextError("INCOMPLETE_REQUEST_EVIDENCE") from exc
            raise

    def _context_from_row(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        expected_ref: str,
        expected_hash: str,
        row: tuple[object, ...] | None,
    ) -> ContextVersion:
        if row is None:
            raise RuntimeContextError("NOT_FOUND")
        stored_org, stored_workspace, schema_version, stored_hash, encoded = row
        if (stored_org, stored_workspace) != (organization_id, workspace_id):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        if schema_version not in {"runtime_working_context.v1", "runtime_working_context.v2", "runtime_working_context.v3"}:
            raise RuntimeContextError("INVALID_STATE")
        if stored_hash != expected_hash:
            raise RuntimeContextError("HASH_MISMATCH")
        context = self._decode_registered(schema_version, encoded)
        if (context.organization_id, context.workspace_id) != (organization_id, workspace_id):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        if (
            context.organization_id,
            context.workspace_id,
            context.context_ref,
            context.context_hash,
            canonical_json(context.model_dump(mode="json")),
        ) != (
            organization_id,
            workspace_id,
            expected_ref,
            expected_hash,
            encoded,
        ):
            raise RuntimeContextError("HASH_MISMATCH")
        return context

    def get_context(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
    ) -> ContextVersion:
        return self._consistent_read(
            lambda: self._get_context_once(
                organization_id=organization_id,
                workspace_id=workspace_id,
                pin=pin,
            )
        )

    def _get_context_once(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
    ) -> ContextVersion:
        pin = ArtifactPinV1.model_validate(pin.model_dump(mode="json"))
        if pin.artifact_kind != _CONTEXT_KIND:
            raise RuntimeContextError("INVALID_STATE")
        row = self._connection.execute(
            sql_query("""
            SELECT organization_id, workspace_id, schema_version, artifact_hash,
                   payload_json
            FROM ontology_build_artifacts WHERE artifact_ref = ?
            """,'\n            SELECT organization_id, workspace_id, schema_version, artifact_hash,\n                   payload_json\n            FROM ontology_build_artifacts WHERE artifact_ref = %s\n            '),
            (pin.artifact_ref,),
        ).fetchone()
        context = self._context_from_row(
            organization_id=organization_id,
            workspace_id=workspace_id,
            expected_ref=pin.artifact_ref,
            expected_hash=pin.artifact_hash,
            row=row,
        )
        indexed = self._connection.execute(
            sql_query("""
            SELECT organization_id, workspace_id, stream_ref, version,
                   context_ref, context_hash
            FROM runtime_context_version_index WHERE context_ref=?
            """,'\n            SELECT organization_id, workspace_id, stream_ref, version,\n                   context_ref, context_hash\n            FROM runtime_context_version_index WHERE context_ref=%s\n            '),
            (context.context_ref,),
        ).fetchone()
        if indexed != (
            context.organization_id,
            context.workspace_id,
            context.stream_ref,
            context.version,
            context.context_ref,
            context.context_hash,
        ):
            raise RuntimeContextError("INVALID_STATE")
        return context

    def _validate_stream_projection(
        self, *, organization_id: str, workspace_id: str, stream_ref: str
    ) -> tuple[int, int] | None:
        stats = self._connection.execute(
            sql_query("""
            SELECT COUNT(*), MIN(version), MAX(version)
            FROM runtime_context_version_index
            WHERE organization_id=? AND workspace_id=? AND stream_ref=?
            """,'\n            SELECT COUNT(*), MIN(version), MAX(version)\n            FROM runtime_context_version_index\n            WHERE organization_id=%s AND workspace_id=%s AND stream_ref=%s\n            '),
            (organization_id, workspace_id, stream_ref),
        ).fetchone()
        if stats is None:
            raise RuntimeContextError("INVALID_STATE")
        count, minimum, maximum = stats
        artifact_count = self._connection.execute(
            sql_query("""
            SELECT COUNT(*) FROM ontology_build_artifacts
            WHERE organization_id=? AND workspace_id=?
              AND schema_version IN ('runtime_working_context.v1', 'runtime_working_context.v2', 'runtime_working_context.v3')
              AND json_extract(payload_json, '$.stream_ref')=?
            ""","\n            SELECT COUNT(*) FROM ontology_build_artifacts\n            WHERE organization_id=%s AND workspace_id=%s\n              AND schema_version IN ('runtime_working_context.v1', 'runtime_working_context.v2', 'runtime_working_context.v3')\n              AND (payload_json::jsonb #>> '{stream_ref}')=%s\n            "),
            (organization_id, workspace_id, stream_ref),
        ).fetchone()[0]
        if count == 0:
            if artifact_count != 0:
                raise RuntimeContextError("INVALID_STATE")
            return None
        if (minimum, maximum, artifact_count) != (0, count - 1, count):
            raise RuntimeContextError("INVALID_STATE")
        rows = self._connection.execute(
            sql_query("""
            SELECT i.version, i.context_ref, i.context_hash,
                   a.organization_id, a.workspace_id, a.schema_version,
                   a.artifact_hash, a.payload_json
            FROM runtime_context_version_index AS i
            LEFT JOIN ontology_build_artifacts AS a
              ON a.artifact_ref=i.context_ref
            WHERE i.organization_id=? AND i.workspace_id=? AND i.stream_ref=?
            ORDER BY i.version
            """,'\n            SELECT i.version, i.context_ref, i.context_hash,\n                   a.organization_id, a.workspace_id, a.schema_version,\n                   a.artifact_hash, a.payload_json\n            FROM runtime_context_version_index AS i\n            LEFT JOIN ontology_build_artifacts AS a\n              ON a.artifact_ref=i.context_ref\n            WHERE i.organization_id=%s AND i.workspace_id=%s AND i.stream_ref=%s\n            ORDER BY i.version\n            '),
            (organization_id, workspace_id, stream_ref),
        ).fetchall()
        predecessor: ContextVersion | None = None
        for version, context_ref, context_hash, *artifact_row in rows:
            try:
                context = self._context_from_row(
                    organization_id=organization_id,
                    workspace_id=workspace_id,
                    expected_ref=str(context_ref),
                    expected_hash=str(context_hash),
                    row=tuple(artifact_row),
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeContextError("INVALID_STATE") from exc
            expected_parent = (
                None
                if predecessor is None
                else _pin(
                    _CONTEXT_KIND,
                    predecessor.context_ref,
                    predecessor.context_hash,
                )
            )
            if (
                context.stream_ref,
                context.version,
                context.parent,
            ) != (stream_ref, version, expected_parent):
                raise RuntimeContextError("INVALID_STATE")
            if predecessor is not None and predecessor.schema_version != context.schema_version:
                raise RuntimeContextError("INVALID_STATE")
            predecessor = context
        return int(count), int(maximum)

    def get_head(
        self, *, organization_id: str, workspace_id: str, stream_ref: str
    ) -> ContextVersion | None:
        return self._consistent_read(
            lambda: self._get_head_once(
                organization_id=organization_id,
                workspace_id=workspace_id,
                stream_ref=stream_ref,
            )
        )

    def _get_head_once(
        self, *, organization_id: str, workspace_id: str, stream_ref: str
    ) -> ContextVersion | None:
        stats = self._validate_stream_projection(
            organization_id=organization_id,
            workspace_id=workspace_id,
            stream_ref=stream_ref,
        )
        row = self._connection.execute(
            sql_query("""
            SELECT version, context_ref, context_hash
            FROM runtime_context_heads
            WHERE organization_id=? AND workspace_id=? AND stream_ref=?
            """,'\n            SELECT version, context_ref, context_hash\n            FROM runtime_context_heads\n            WHERE organization_id=%s AND workspace_id=%s AND stream_ref=%s\n            '),
            (organization_id, workspace_id, stream_ref),
        ).fetchone()
        if stats is None:
            if row is not None:
                raise RuntimeContextError("INVALID_STATE")
            return None
        if row is None or row[0] != stats[1]:
            raise RuntimeContextError("INVALID_STATE")
        context = self.get_context(
            organization_id=organization_id,
            workspace_id=workspace_id,
            pin=_pin(_CONTEXT_KIND, str(row[1]), str(row[2])),
        )
        if (context.stream_ref, context.version) != (stream_ref, row[0]):
            raise RuntimeContextError("INVALID_STATE")
        return context

    def _receipt_from_index(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        actor_id: str,
        operation: str,
        request_id: str,
    ) -> tuple[str, ReceiptVersion] | None:
        row = self._connection.execute(
            sql_query("""
            SELECT i.request_hash, i.receipt_ref, i.receipt_hash,
                   a.organization_id, a.workspace_id, a.schema_version,
                   a.artifact_hash, a.payload_json
            FROM runtime_context_request_index AS i
            LEFT JOIN ontology_build_artifacts AS a ON a.artifact_ref=i.receipt_ref
            WHERE i.organization_id=? AND i.workspace_id=? AND i.actor_id=?
              AND i.operation=? AND i.request_id=?
            """,'\n            SELECT i.request_hash, i.receipt_ref, i.receipt_hash,\n                   a.organization_id, a.workspace_id, a.schema_version,\n                   a.artifact_hash, a.payload_json\n            FROM runtime_context_request_index AS i\n            LEFT JOIN ontology_build_artifacts AS a ON a.artifact_ref=i.receipt_ref\n            WHERE i.organization_id=%s AND i.workspace_id=%s AND i.actor_id=%s\n              AND i.operation=%s AND i.request_id=%s\n            '),
            (organization_id, workspace_id, actor_id, operation, request_id),
        ).fetchone()
        if row is None:
            orphan = self._connection.execute(
                sql_query("""
                SELECT artifact_ref FROM ontology_build_artifacts
                WHERE organization_id=? AND workspace_id=?
                  AND schema_version IN ('runtime_context_transition_receipt.v1', 'runtime_context_transition_receipt.v2', 'runtime_context_transition_receipt.v3')
                  AND json_extract(payload_json, '$.actor_id')=?
                  AND json_extract(payload_json, '$.operation')=?
                  AND json_extract(payload_json, '$.request_id')=? LIMIT 1
                ""","\n                SELECT artifact_ref FROM ontology_build_artifacts\n                WHERE organization_id=%s AND workspace_id=%s\n                  AND schema_version IN ('runtime_context_transition_receipt.v1', 'runtime_context_transition_receipt.v2', 'runtime_context_transition_receipt.v3')\n                  AND (payload_json::jsonb #>> '{actor_id}')=%s\n                  AND (payload_json::jsonb #>> '{operation}')=%s\n                  AND (payload_json::jsonb #>> '{request_id}')=%s LIMIT 1\n                "),
                (organization_id, workspace_id, actor_id, operation, request_id),
            ).fetchone()
            if orphan is not None:
                raise RuntimeContextError("INVALID_STATE")
            return None
        (
            request_hash,
            receipt_ref,
            receipt_hash,
            stored_org,
            stored_workspace,
            schema_version,
            stored_hash,
            encoded,
        ) = row
        if (
            stored_org,
            stored_workspace,
            stored_hash,
        ) != (
            organization_id,
            workspace_id,
            receipt_hash,
        ) or schema_version not in {
            "runtime_context_transition_receipt.v1",
            "runtime_context_transition_receipt.v2",
            "runtime_context_transition_receipt.v3",
        } or encoded is None:
            raise RuntimeContextError("INVALID_STATE")
        receipt = self._decode_registered(str(schema_version), str(encoded))
        if not isinstance(
            receipt,
            (
                RuntimeContextTransitionReceiptV1,
                RuntimeContextTransitionReceiptV2,
                RuntimeContextTransitionReceiptV3,
            ),
        ):
            raise RuntimeContextError("INVALID_STATE")
        if (
            receipt.organization_id,
            receipt.workspace_id,
            receipt.actor_id,
            receipt.operation,
            receipt.request_id,
            receipt.request_hash,
            receipt.receipt_ref,
            receipt.receipt_hash,
            canonical_json(receipt.model_dump(mode="json")),
        ) != (
            organization_id,
            workspace_id,
            actor_id,
            operation,
            request_id,
            request_hash,
            receipt_ref,
            receipt_hash,
            encoded,
        ):
            raise RuntimeContextError("HASH_MISMATCH")
        if isinstance(receipt, RuntimeContextTransitionReceiptV3):
            successor = self.get_context(
                organization_id=organization_id,
                workspace_id=workspace_id,
                pin=receipt.successor,
            )
            selection = self._read_artifact(
                organization_id,
                workspace_id,
                receipt.selection,
            )
            if (
                not isinstance(successor, RuntimeTaskContextV3)
                or not isinstance(selection, RuntimeTemporalContextSelectionV1)
                or successor.selection != receipt.selection
            ):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        else:
            self._validate_stored_receipt_links(receipt)
        return str(request_hash), receipt

    def _load_pinned_artifact(
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
        artifact = contract_type.model_validate(json.loads(str(encoded)))
        if (
            getattr(artifact, ref_field),
            getattr(artifact, hash_field),
            canonical_json(artifact.model_dump(mode="json")),
        ) != (pin.artifact_ref, pin.artifact_hash, encoded):
            raise RuntimeContextError("HASH_MISMATCH")
        return artifact

    def _validate_stored_receipt_links(
        self, receipt: RuntimeContextTransitionReceiptV1
    ) -> None:
        successor = self.get_context(
            organization_id=receipt.organization_id,
            workspace_id=receipt.workspace_id,
            pin=receipt.successor,
        )
        if type(successor) is not RuntimeTaskContextV1:
            raise RuntimeContextError("INVALID_STATE")
        indexed = self._connection.execute(
            sql_query("""
            SELECT organization_id, workspace_id, stream_ref, version,
                   context_ref, context_hash
            FROM runtime_context_version_index WHERE context_ref=?
            """,'\n            SELECT organization_id, workspace_id, stream_ref, version,\n                   context_ref, context_hash\n            FROM runtime_context_version_index WHERE context_ref=%s\n            '),
            (successor.context_ref,),
        ).fetchone()
        if indexed != (
            successor.organization_id,
            successor.workspace_id,
            successor.stream_ref,
            successor.version,
            successor.context_ref,
            successor.context_hash,
        ):
            raise RuntimeContextError("INVALID_STATE")
        transition: RuntimeContextTransitionV1 | None = None
        proposal: RuntimeContextTransitionProposalV1 | None = None
        if receipt.transition is not None:
            loaded_transition = self._load_pinned_artifact(
                organization_id=receipt.organization_id,
                workspace_id=receipt.workspace_id,
                pin=receipt.transition,
                artifact_kind=_TRANSITION_KIND,
                schema_version="runtime_context_transition.v1",
                contract_type=RuntimeContextTransitionV1,
                ref_field="transition_ref",
                hash_field="transition_hash",
            )
            if not isinstance(loaded_transition, RuntimeContextTransitionV1):
                raise RuntimeContextError("INVALID_STATE")
            transition = loaded_transition
            loaded_proposal = self._load_pinned_artifact(
                organization_id=receipt.organization_id,
                workspace_id=receipt.workspace_id,
                pin=transition.proposal,
                artifact_kind=_PROPOSAL_KIND,
                schema_version="runtime_context_transition_proposal.v1",
                contract_type=RuntimeContextTransitionProposalV1,
                ref_field="proposal_ref",
                hash_field="proposal_hash",
            )
            if not isinstance(loaded_proposal, RuntimeContextTransitionProposalV1):
                raise RuntimeContextError("INVALID_STATE")
            proposal = loaded_proposal
        self._validate_append_links(
            successor=successor,
            expected_parent=successor.parent,
            expected_version=(successor.version - 1 if successor.version > 0 else None),
            transition=transition,
            proposal=proposal,
            receipt=receipt,
        )
        if successor.parent is not None:
            parent = self.get_context(
                organization_id=receipt.organization_id,
                workspace_id=receipt.workspace_id,
                pin=successor.parent,
            )
            if type(parent) is not RuntimeTaskContextV1:
                raise RuntimeContextError("INVALID_STATE")
            if (
                parent.stream_ref,
                parent.version,
            ) != (successor.stream_ref, successor.version - 1):
                raise RuntimeContextError("INVALID_STATE")
        if successor.branch_origin is not None:
            self.get_context(
                organization_id=receipt.organization_id,
                workspace_id=receipt.workspace_id,
                pin=successor.branch_origin,
            )

    def get_request_receipt(
        self,
        *,
        actor_context: RuntimeContextDTO,
        operation: str,
        request_id: str,
    ) -> ReceiptVersion | None:
        if not isinstance(actor_context, RuntimeContextDTO) or not actor_context.workspace_id:
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        return self._consistent_read(
            lambda: self._get_request_receipt_once(
                actor_context=actor_context,
                operation=operation,
                request_id=request_id,
            )
        )

    def _get_request_receipt_once(
        self,
        *,
        actor_context: RuntimeContextDTO,
        operation: str,
        request_id: str,
    ) -> ReceiptVersion | None:
        workspace_id = actor_context.workspace_id
        if workspace_id is None:
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        found = self._receipt_from_index(
            organization_id=actor_context.organization_id,
            workspace_id=workspace_id,
            actor_id=actor_context.actor_id,
            operation=operation,
            request_id=request_id,
        )
        return None if found is None else found[1]

    def _validate_append_links(
        self,
        *,
        successor: ContextVersion,
        expected_parent: ArtifactPinV1 | None,
        expected_version: int | None,
        transition: TransitionVersion | None,
        proposal: ProposalVersion | None,
        receipt: ReceiptVersion,
    ) -> None:
        scope = (successor.organization_id, successor.workspace_id)
        if (receipt.organization_id, receipt.workspace_id) != scope:
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        successor_pin = _pin(_CONTEXT_KIND, successor.context_ref, successor.context_hash)
        if not _same_pin(receipt.successor, successor_pin):
            raise RuntimeContextError("INVALID_STATE")
        if receipt.operation == "CREATE":
            valid = (
                successor.version == 0
                and successor.parent is None
                and successor.branch_origin is None
                and successor.transition is None
                and expected_parent is None
                and expected_version is None
                and transition is None
                and proposal is None
                and receipt.transition is None
                and successor.created_by == receipt.actor_id
            )
        elif receipt.operation == "BRANCH":
            valid = (
                successor.version == 0
                and successor.parent is None
                and successor.branch_origin is not None
                and successor.transition is None
                and expected_parent is None
                and expected_version is None
                and transition is None
                and proposal is None
                and receipt.transition is None
                and successor.created_by == receipt.actor_id
            )
        else:
            transition_pin = (
                None
                if transition is None
                else _pin(
                    _TRANSITION_KIND,
                    transition.transition_ref,
                    transition.transition_hash,
                )
            )
            proposal_pin = (
                None
                if proposal is None
                else _pin(_PROPOSAL_KIND, proposal.proposal_ref, proposal.proposal_hash)
            )
            valid = (
                expected_parent is not None
                and expected_version is not None
                and transition is not None
                and proposal is not None
                and successor.version == expected_version + 1
                and _same_pin(successor.parent, expected_parent)
                and _same_pin(proposal.parent, expected_parent)
                and _same_pin(transition.parent, expected_parent)
                and _same_pin(transition.proposal, proposal_pin)
                and _same_pin(successor.transition, transition_pin)
                and _same_pin(receipt.transition, transition_pin)
                and transition.committed_by == receipt.actor_id
                and successor.created_by == receipt.actor_id
                and successor.created_at == transition.committed_at
            )
        if not valid:
            raise RuntimeContextError("INVALID_STATE")

    def _insert_request_index(self, receipt: ReceiptVersion) -> None:
        self._connection.execute(
            sql_query("""
            INSERT INTO runtime_context_request_index (
                organization_id, workspace_id, actor_id, operation, request_id,
                request_hash, receipt_ref, receipt_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,'\n            INSERT INTO runtime_context_request_index (\n                organization_id, workspace_id, actor_id, operation, request_id,\n                request_hash, receipt_ref, receipt_hash\n            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            (
                receipt.organization_id,
                receipt.workspace_id,
                receipt.actor_id,
                receipt.operation,
                receipt.request_id,
                receipt.request_hash,
                receipt.receipt_ref,
                receipt.receipt_hash,
            ),
        )

    def publish_temporal_context(
        self,
        *,
        actor_context: RuntimeContextDTO,
        selection: RuntimeTemporalContextSelectionV1,
        authority_resolver: "RuntimeAuthorityResolver",
        session_resolver: "GovernedRuntimeSessionResolver",
    ) -> RuntimeContextTransitionReceiptV3:
        """Create one V3 context through the closed explicit-selection port."""

        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from services.runtime_working_context_service import GovernedRuntimeSessionResolver

        if not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("authority_resolver must be RuntimeAuthorityResolver")
        if not isinstance(session_resolver, GovernedRuntimeSessionResolver):
            raise TypeError("session_resolver must be GovernedRuntimeSessionResolver")
        context_owner = authority_resolver._owners.get("runtime_working_context")
        if (
            session_resolver._authority_resolver is not authority_resolver
            or getattr(getattr(context_owner, "_context_store", None), "_build_store", None)
            is not self._build_store
        ):
            raise RuntimeContextError("INVALID_STATE")
        try:
            selection = RuntimeTemporalContextSelectionV1.model_validate(
                selection.model_dump(mode="json")
            )
        except (ValidationError, ValueError, AttributeError) as exc:
            raise RuntimeContextError("HASH_MISMATCH") from exc
        if not isinstance(actor_context, RuntimeContextDTO) or not actor_context.workspace_id:
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        if (
            selection.organization_id,
            selection.workspace_id,
            selection.actor_id,
            selection.session_ref,
        ) != (
            actor_context.organization_id,
            actor_context.workspace_id,
            actor_context.actor_id,
            actor_context.session_id,
        ):
            raise RuntimeContextError("FORBIDDEN_SCOPE")

        def current_access() -> None:
            authority_resolver.require_current_actor(actor_context=actor_context)
            if session_resolver.resolve(
                actor_context=actor_context, session_ref=selection.session_ref
            ) != selection.session_ref:
                raise RuntimeContextError("FORBIDDEN_SESSION")
            authority_resolver.resolve_pins(
                actor_context=actor_context,
                pins=(selection.goal, selection.source_graph, *selection.semantic_pins),
            )
            for entity_id in selection.selected_entity_ids:
                authority_resolver.resolve_object(
                    actor_context=actor_context,
                    source=selection.source_graph,
                    object_ref=entity_id,
                )

        current_access()
        existing = self._receipt_from_index(
            organization_id=selection.organization_id,
            workspace_id=selection.workspace_id,
            actor_id=selection.actor_id,
            operation="CREATE",
            request_id=selection.request_id,
        )
        if existing is not None:
            _saved_request_hash, saved = existing
            if not isinstance(saved, RuntimeContextTransitionReceiptV3):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if saved.selection != selection.artifact_pin():
                raise RuntimeContextError("REQUEST_CONFLICT")
            saved_context = self.get_context(
                organization_id=selection.organization_id,
                workspace_id=selection.workspace_id,
                pin=saved.successor,
            )
            if (
                not isinstance(saved_context, RuntimeTaskContextV3)
                or saved_context.selection != selection.artifact_pin()
            ):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            current_access()
            return saved
        instant = datetime.now(timezone.utc)
        stream_ref = "runtime-context-stream:" + selection.selection_ref.rsplit(":", 1)[-1]
        elements = tuple(
            RuntimeContextElementV1(
                object_ref=entity_id,
                source_artifact=selection.source_graph,
                role="entity",
                admission="candidate",
                active=False,
                derived=False,
                evidence=(selection.source_graph,),
                admission_receipt=None,
                derivation=None,
            )
            for entity_id in selection.selected_entity_ids
        )
        context_values = {
            "schema_version": "runtime_working_context.v3",
            "stream_ref": stream_ref,
            "version": 0,
            "organization_id": selection.organization_id,
            "workspace_id": selection.workspace_id,
            "session_ref": selection.session_ref,
            "goal": selection.goal,
            "world": selection.world,
            "semantic_pins": selection.semantic_pins,
            "elements": elements,
            "parent": None,
            "branch_origin": None,
            "transition": None,
            "status": "OPEN",
            "created_by": selection.actor_id,
            "created_at": instant,
            "selection": selection.artifact_pin(),
        }
        context_digest = content_hash(context_hash_basis(context_values))
        successor = RuntimeTaskContextV3(
            **context_values,
            context_ref=f"runtime-context:{context_digest}",
            context_hash=f"sha256:{context_digest}",
        )
        request_hash = runtime_context_persistence_command_hash(
            organization_id=selection.organization_id,
            workspace_id=selection.workspace_id,
            actor_id=selection.actor_id,
            operation="CREATE",
            request_id=selection.request_id,
            expected_parent=None,
            expected_version=None,
            successor=successor,
            transition=None,
            proposal=None,
            related_artifacts=(selection,),
        )
        receipt_values = {
            "schema_version": "runtime_context_transition_receipt.v3",
            "organization_id": selection.organization_id,
            "workspace_id": selection.workspace_id,
            "actor_id": selection.actor_id,
            "operation": "CREATE",
            "request_id": selection.request_id,
            "request_hash": request_hash,
            "transition": None,
            "successor": _model_pin(successor),
            "recorded_at": instant,
            "selection": selection.artifact_pin(),
        }
        receipt_digest = content_hash(receipt_hash_basis(receipt_values))
        receipt = RuntimeContextTransitionReceiptV3(
            **receipt_values,
            receipt_ref=f"runtime-context-receipt:{receipt_digest}",
            receipt_hash=f"sha256:{receipt_digest}",
        )

        connection = self._connection
        begin_write(connection)
        try:
            key = (
                selection.organization_id,
                selection.workspace_id,
                selection.actor_id,
                "CREATE",
                selection.request_id,
            )
            index = connection.execute(
                sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM runtime_context_request_index "
                "WHERE organization_id=? AND workspace_id=? AND actor_id=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM runtime_context_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_id=%s AND operation=%s AND request_id=%s'),
                key,
            ).fetchone()
            if index is not None:
                saved = self._read_artifact(
                    selection.organization_id,
                    selection.workspace_id,
                    _pin("runtime_context_transition_receipt", index[1], index[2]),
                )
                if not isinstance(saved, RuntimeContextTransitionReceiptV3):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                if saved.selection != selection.artifact_pin():
                    raise RuntimeContextError("REQUEST_CONFLICT")
                saved_context = self.get_context(
                    organization_id=selection.organization_id,
                    workspace_id=selection.workspace_id,
                    pin=saved.successor,
                )
                if (
                    not isinstance(saved_context, RuntimeTaskContextV3)
                    or saved_context.selection != selection.artifact_pin()
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                current_access()
                connection.commit()
                return saved
            orphan = connection.execute(
                sql_query("SELECT 1 FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? "
                "AND ((schema_version='runtime_temporal_context_selection.v1' AND artifact_ref=?) "
                "OR (schema_version='runtime_context_transition_receipt.v3' "
                "AND json_extract(payload_json,'$.actor_id')=? "
                "AND json_extract(payload_json,'$.operation')='CREATE' "
                "AND json_extract(payload_json,'$.request_id')=?)) LIMIT 1","SELECT 1 FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND ((schema_version='runtime_temporal_context_selection.v1' AND artifact_ref=%s) OR (schema_version='runtime_context_transition_receipt.v3' AND (payload_json::jsonb #>> '{actor_id}')=%s AND (payload_json::jsonb #>> '{operation}')='CREATE' AND (payload_json::jsonb #>> '{request_id}')=%s)) LIMIT 1"),
                (
                    selection.organization_id,
                    selection.workspace_id,
                    selection.selection_ref,
                    selection.actor_id,
                    selection.request_id,
                ),
            ).fetchone()
            if orphan is not None:
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if self._get_head_once(
                organization_id=selection.organization_id,
                workspace_id=selection.workspace_id,
                stream_ref=stream_ref,
            ) is not None:
                raise RuntimeContextError("STALE_PARENT")
            current_access()
            prepared = tuple(
                self._build_store._prepare(item) for item in (selection, successor, receipt)
            )
            self._check_existing_bytes(prepared)
            self._build_store._insert_prepared(prepared)
            connection.execute(
                sql_query("INSERT INTO runtime_context_version_index "
                "(organization_id,workspace_id,stream_ref,version,context_ref,context_hash) "
                "VALUES (?,?,?,?,?,?)",'INSERT INTO runtime_context_version_index (organization_id,workspace_id,stream_ref,version,context_ref,context_hash) VALUES (%s,%s,%s,%s,%s,%s)'),
                (
                    selection.organization_id,
                    selection.workspace_id,
                    stream_ref,
                    0,
                    successor.context_ref,
                    successor.context_hash,
                ),
            )
            self._insert_request_index(receipt)
            connection.execute(
                sql_query("INSERT INTO runtime_context_heads "
                "(organization_id,workspace_id,stream_ref,version,context_ref,context_hash) "
                "VALUES (?,?,?,?,?,?)",'INSERT INTO runtime_context_heads (organization_id,workspace_id,stream_ref,version,context_ref,context_hash) VALUES (%s,%s,%s,%s,%s,%s)'),
                (
                    selection.organization_id,
                    selection.workspace_id,
                    stream_ref,
                    0,
                    successor.context_ref,
                    successor.context_hash,
                ),
            )
            current_access()
            connection.commit()
            return receipt
        except BaseException:
            connection.rollback()
            raise

    def put_proposal(
        self,
        *,
        proposal: RuntimeContextTransitionProposalV2,
        verify_before_commit: Callable[[], None] | None = None,
    ) -> RuntimeContextTransitionProposalV2:
        """Persist one validated candidate without advancing a context stream."""

        proposal = _validated(proposal)
        if proposal.schema_version == "runtime_context_transition_proposal.v1":
            raise RuntimeContextError("LEGACY_WRITE_FORBIDDEN")
        if type(proposal) is not RuntimeContextTransitionProposalV2:
            raise RuntimeContextError("INVALID_STATE")
        if verify_before_commit is not None and not callable(verify_before_commit):
            raise TypeError("verify_before_commit must be callable")
        begin_write(self._connection)
        try:
            parent = self.get_context(
                organization_id=proposal.organization_id,
                workspace_id=proposal.workspace_id,
                pin=proposal.parent,
            )
            self._validate_proposal_formation(proposal, parent)
            prepared = (self._build_store._prepare(proposal),)
            self._check_existing_bytes(prepared)
            if verify_before_commit is not None:
                verify_before_commit()
            self._build_store._insert_prepared(prepared)
            self._connection.commit()
            return proposal
        except BaseException:
            self._connection.rollback()
            raise

    @overload
    def append_context(
        self, *, successor: RuntimeTaskContextV2,
        expected_parent: ArtifactPinV1 | None, expected_version: int | None,
        transition: RuntimeContextTransitionV2 | None,
        proposal: RuntimeContextTransitionProposalV2 | None,
        receipt: RuntimeContextTransitionReceiptV2,
        persistence_manifest: rr.RuntimeContextPersistenceManifestV1,
        related_artifacts: tuple[BaseModel, ...],
        verify_before_commit: Callable[[], None] | None = None,
    ) -> RuntimeContextTransitionReceiptV2: ...

    @overload
    def append_context(
        self, *, successor: RuntimeTaskContextV1,
        expected_parent: ArtifactPinV1 | None, expected_version: int | None,
        transition: RuntimeContextTransitionV1 | None,
        proposal: RuntimeContextTransitionProposalV1 | None,
        receipt: RuntimeContextTransitionReceiptV1,
        related_artifacts: tuple[BaseModel, ...] = (),
        verify_before_commit: Callable[[], None] | None = None,
    ) -> RuntimeContextTransitionReceiptV1: ...

    def append_context(
        self,
        *,
        successor: ContextVersion,
        expected_parent: ArtifactPinV1 | None,
        expected_version: int | None,
        transition: TransitionVersion | None,
        proposal: ProposalVersion | None,
        receipt: ReceiptVersion,
        persistence_manifest: rr.RuntimeContextPersistenceManifestV1 | None = None,
        related_artifacts: tuple[BaseModel, ...] = (),
        verify_before_commit: Callable[[], None] | None = None,
    ) -> ReceiptVersion:
        successor = _validated(successor)
        receipt = _validated(receipt)
        transition = None if transition is None else _validated(transition)
        proposal = None if proposal is None else _validated(proposal)
        v2 = successor.schema_version == "runtime_working_context.v2"
        suffix = ".v2" if v2 else ".v1"
        for value, kind in ((successor, "runtime_working_context"),
                            (receipt, "runtime_context_transition_receipt"),
                            (transition, "runtime_context_transition"),
                            (proposal, "runtime_context_transition_proposal")):
            if value is not None and value.schema_version != kind + suffix:
                raise RuntimeContextError("INVALID_STATE")
        if expected_parent is not None:
            try:
                expected_parent = ArtifactPinV1.model_validate(expected_parent.model_dump(mode="json"))
            except (ValueError, AttributeError) as exc:
                raise RuntimeContextError("INVALID_STATE") from exc
            if expected_parent.artifact_kind != _CONTEXT_KIND:
                raise RuntimeContextError("INVALID_STATE")
        if verify_before_commit is not None and not callable(verify_before_commit):
            raise TypeError("verify_before_commit must be callable")
        if v2:
            persistence_manifest = _validated(persistence_manifest)
            if type(persistence_manifest) is not rr.RuntimeContextPersistenceManifestV1:
                raise RuntimeContextError("INVALID_STATE")
        elif persistence_manifest is not None:
            raise RuntimeContextError("INVALID_STATE")
        self._validate_append_links(
            successor=successor,
            expected_parent=expected_parent,
            expected_version=expected_version,
            transition=transition,
            proposal=proposal,
            receipt=receipt,
        )
        for artifact in related_artifacts:
            schema = getattr(artifact, "schema_version", "")
            if schema in (
                _RUNTIME_CONTEXT_CAS_SCHEMAS
                | _RUNTIME_CONTEXT_AUTHORITY_SCHEMAS
                | _SEED_CAS_SCHEMAS
                | _LEGACY_CONTEXT_SCHEMAS
                | _ADVISORY_CONTEXT_SCHEMAS
                | {
                    "runtime_context_transition_proposal.v2",
                    "runtime_context_persistence_manifest.v1",
                    "exploratory_graph_snapshot_integrity.v1",
                }
            ):
                raise RuntimeContextError("INVALID_STATE")
            if v2 and schema not in _FORMATION_EVIDENCE_SCHEMAS | {"runtime_context_formation_command.v1"}:
                raise RuntimeContextError("INVALID_STATE")
        try:
            prepared_related = tuple(self._build_store._prepare(artifact) for artifact in related_artifacts)
        except (ValueError, TypeError) as exc:
            raise RuntimeContextError("HASH_MISMATCH" if str(exc) == "HASH_MISMATCH" else "INVALID_STATE") from exc
        for item in prepared_related:
            if (item[0], item[1]) != (
                successor.organization_id,
                successor.workspace_id,
            ):
                raise RuntimeContextError("FORBIDDEN_SCOPE")
        if proposal is not None and (
            proposal.organization_id,
            proposal.workspace_id,
        ) != (successor.organization_id, successor.workspace_id):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        if transition is not None and (
            transition.organization_id,
            transition.workspace_id,
        ) != (successor.organization_id, successor.workspace_id):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        if v2:
            if (
                persistence_manifest.organization_id,
                persistence_manifest.workspace_id,
            ) != (successor.organization_id, successor.workspace_id):
                raise RuntimeContextError("FORBIDDEN_SCOPE")
            self._validate_v2_association(persistence_manifest, receipt, successor,
                transition, proposal, related_artifacts)
            if (persistence_manifest.expected_parent, persistence_manifest.expected_version) != (expected_parent, expected_version):
                raise RuntimeContextError("INVALID_STATE")
            command_hash = rr.runtime_context_persistence_command_hash_v2(persistence_manifest)
        else:
            command_hash = runtime_context_persistence_command_hash(
                organization_id=successor.organization_id,
                workspace_id=successor.workspace_id,
                actor_id=receipt.actor_id,
                operation=receipt.operation,
                request_id=receipt.request_id,
                expected_parent=expected_parent,
                expected_version=expected_version,
                successor=successor,
                transition=transition,
                proposal=proposal,
                related_artifacts=related_artifacts,
            )
        if receipt.request_hash != command_hash:
            raise RuntimeContextError("REQUEST_CONFLICT")
        artifacts: tuple[BaseModel, ...] = (
            *((proposal,) if proposal is not None else ()),
            *((transition,) if transition is not None else ()),
            *related_artifacts,
            *((persistence_manifest,) if persistence_manifest is not None else ()),
            successor,
            receipt,
        )
        begin_write(self._connection)
        try:
            key = dict(organization_id=successor.organization_id,
                workspace_id=successor.workspace_id, actor_id=receipt.actor_id,
                operation=receipt.operation, request_id=receipt.request_id)
            if v2:
                bundle = self._get_request_bundle_once(**key)
                prior = None if bundle is None else (bundle.receipt.request_hash, bundle.receipt)
            else:
                prior = self._receipt_from_index(**key)
            if prior is not None:
                if prior[0] != command_hash:
                    raise RuntimeContextError("REQUEST_CONFLICT")
                self._connection.commit()
                return prior[1]
            if not v2:
                raise RuntimeContextError("LEGACY_WRITE_FORBIDDEN")

            self._validate_stream_projection(
                organization_id=successor.organization_id,
                workspace_id=successor.workspace_id,
                stream_ref=successor.stream_ref,
            )
            head = self._connection.execute(
                sql_query("""
                SELECT version, context_ref, context_hash
                FROM runtime_context_heads
                WHERE organization_id=? AND workspace_id=? AND stream_ref=?
                """,'\n                SELECT version, context_ref, context_hash\n                FROM runtime_context_heads\n                WHERE organization_id=%s AND workspace_id=%s AND stream_ref=%s\n                '),
                (
                    successor.organization_id,
                    successor.workspace_id,
                    successor.stream_ref,
                ),
            ).fetchone()
            if expected_parent is None:
                if head is not None:
                    raise RuntimeContextError("STALE_PARENT")
            else:
                if head != (
                    expected_version,
                    expected_parent.artifact_ref,
                    expected_parent.artifact_hash,
                ):
                    raise RuntimeContextError("STALE_PARENT")
                parent = self.get_context(
                    organization_id=successor.organization_id,
                    workspace_id=successor.workspace_id,
                    pin=expected_parent,
                )
                if parent.version != expected_version or parent.stream_ref != successor.stream_ref:
                    raise RuntimeContextError("STALE_PARENT")
                self._validate_parent_basis(parent, successor)
                self._validate_proposal_formation(proposal, parent, related_artifacts)
            if successor.branch_origin is not None:
                self.get_context(
                    organization_id=successor.organization_id,
                    workspace_id=successor.workspace_id,
                    pin=successor.branch_origin,
                )
            self._formation_association(persistence_manifest.formation,
                successor.organization_id, successor.workspace_id, related_artifacts)
            prepared = tuple(self._build_store._prepare(item) for item in artifacts)
            self._check_existing_bytes(prepared)
            if verify_before_commit is not None:
                verify_before_commit()

            self._build_store._insert_prepared(prepared)
            self._connection.execute(
                sql_query("""
                INSERT INTO runtime_context_version_index (
                    organization_id, workspace_id, stream_ref, version,
                    context_ref, context_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO runtime_context_version_index (\n                    organization_id, workspace_id, stream_ref, version,\n                    context_ref, context_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s)\n                '),
                (
                    successor.organization_id,
                    successor.workspace_id,
                    successor.stream_ref,
                    successor.version,
                    successor.context_ref,
                    successor.context_hash,
                ),
            )
            self._insert_request_index(receipt)
            if expected_parent is None:
                self._connection.execute(
                    sql_query("""
                    INSERT INTO runtime_context_heads (
                        organization_id, workspace_id, stream_ref, version,
                        context_ref, context_hash
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,'\n                    INSERT INTO runtime_context_heads (\n                        organization_id, workspace_id, stream_ref, version,\n                        context_ref, context_hash\n                    ) VALUES (%s, %s, %s, %s, %s, %s)\n                    '),
                    (
                        successor.organization_id,
                        successor.workspace_id,
                        successor.stream_ref,
                        successor.version,
                        successor.context_ref,
                        successor.context_hash,
                    ),
                )
            else:
                cursor = self._connection.execute(
                    sql_query("""
                    UPDATE runtime_context_heads
                    SET version=?, context_ref=?, context_hash=?
                    WHERE organization_id=? AND workspace_id=? AND stream_ref=?
                      AND version=? AND context_ref=? AND context_hash=?
                    """,'\n                    UPDATE runtime_context_heads\n                    SET version=%s, context_ref=%s, context_hash=%s\n                    WHERE organization_id=%s AND workspace_id=%s AND stream_ref=%s\n                      AND version=%s AND context_ref=%s AND context_hash=%s\n                    '),
                    (
                        successor.version,
                        successor.context_ref,
                        successor.context_hash,
                        successor.organization_id,
                        successor.workspace_id,
                        successor.stream_ref,
                        expected_version,
                        expected_parent.artifact_ref,
                        expected_parent.artifact_hash,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeContextError("STALE_PARENT")
            self._connection.commit()
            return receipt
        except BaseException:
            self._connection.rollback()
            raise

    def list_stream_contexts(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        stream_ref: str,
        after_version: int = -1,
        limit: int = 100,
    ) -> tuple[ContextVersion, ...]:
        if after_version < -1:
            raise ValueError("after_version must be at least -1")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        return self._consistent_read(
            lambda: self._list_stream_contexts_once(
                organization_id=organization_id,
                workspace_id=workspace_id,
                stream_ref=stream_ref,
                after_version=after_version,
                limit=limit,
            )
        )

    def _list_stream_contexts_once(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        stream_ref: str,
        after_version: int,
        limit: int,
    ) -> tuple[ContextVersion, ...]:
        self._validate_stream_projection(
            organization_id=organization_id,
            workspace_id=workspace_id,
            stream_ref=stream_ref,
        )
        rows = self._connection.execute(
            sql_query("""
            SELECT version, context_ref, context_hash
            FROM runtime_context_version_index
            WHERE organization_id=? AND workspace_id=? AND stream_ref=?
              AND version>? ORDER BY version LIMIT ?
            """,'\n            SELECT version, context_ref, context_hash\n            FROM runtime_context_version_index\n            WHERE organization_id=%s AND workspace_id=%s AND stream_ref=%s\n              AND version>%s ORDER BY version LIMIT %s\n            '),
            (organization_id, workspace_id, stream_ref, after_version, limit),
        ).fetchall()
        contexts = []
        for version, context_ref, context_hash in rows:
            context = self.get_context(
                organization_id=organization_id,
                workspace_id=workspace_id,
                pin=_pin(_CONTEXT_KIND, context_ref, context_hash),
            )
            if (context.stream_ref, context.version) != (stream_ref, version):
                raise RuntimeContextError("INVALID_STATE")
            contexts.append(context)
        return tuple(contexts)


__all__ = [
    "ContextVersion", "ProposalVersion", "TransitionVersion", "ReceiptVersion",
    "VerifiedRuntimeContextRequestBundleV2",
    "RuntimeContextError",
    "RuntimeTaskContextStore",
    "runtime_context_persistence_command_hash",
]
