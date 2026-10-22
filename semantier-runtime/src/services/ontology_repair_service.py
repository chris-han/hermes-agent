"""Append-only bounded repairs and exact downstream invalidation."""

from __future__ import annotations

from storage.hosted_postgres import sql_query,begin_write,in_transaction

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from contracts.normative_semantics import (
    NormativeFrameV1,
    NormativeSemanticCandidateSetV1,
)
from contracts.ontology_repair import RepairImpactV1
from contracts import ontology_repair as ar
from contracts import runtime_context_reconstruction as rr
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import canonical_json, content_hash

if TYPE_CHECKING:
    from services.runtime_context_formation_service import (
        RuntimeContextFormationService,
    )
    from storage.ontology_build_store import OntologyBuildStore


def _proposal_pin(value) -> ArtifactPinV1:
    return ArtifactPinV1(
        artifact_kind=value.schema_version.removesuffix(".v1"),
        artifact_ref=getattr(value, value.identity_ref_field),
        artifact_hash=getattr(value, value.identity_hash_field),
    )


def _proposal_composition(*, actor_context, session_ref, formation_service, store):
    from services.runtime_context_formation_service import (
        RuntimeContextFormationService,
    )
    from storage.ontology_build_store import OntologyBuildStore
    from storage.runtime_working_context_store import RuntimeContextError

    if (
        type(store) is not OntologyBuildStore
        or type(formation_service) is not RuntimeContextFormationService
        or formation_service._store._build_store is not store
        or formation_service._store._connection is not store._connection
        or formation_service._context
        is not getattr(store, "_advisory_owner_context", None)
    ):
        raise RuntimeContextError("INVALID_STATE")
    return formation_service._scope(actor_context, session_ref)


def _proposal_evidence(*, actor_context, command, store):
    """Current owners and the exact saved native profile precede request recovery."""
    from services.runtime_context_formation_service import (
        RuntimeContextFormationService,
        _revalidate,
        _require_bound_advisory_access,
    )
    from services._runtime_context_advisory.compiler import (
        _candidate_value,
        _validation_value,
        _validate_candidate_frame_native_value,
    )
    from storage.runtime_working_context_store import (
        RuntimeContextError,
        RuntimeTaskContextStore,
    )

    command = _revalidate(ar.AdvisoryNextProposalCommandV1, command)
    context = getattr(store, "_advisory_owner_context", None)
    if context is None:
        raise RuntimeContextError("INVALID_STATE")
    service = RuntimeContextFormationService(
        store=RuntimeTaskContextStore(store), context=context
    )
    _proposal_composition(
        actor_context=actor_context,
        session_ref=command.session_ref,
        formation_service=service,
        store=store,
    )
    if any(
        getattr(command, field) != getattr(actor_context, field)
        for field in ("organization_id", "workspace_id", "actor_id")
    ):
        raise RuntimeContextError("FORBIDDEN_SCOPE")
    request = command.request
    _require_bound_advisory_access(
        build_store=store,
        actor_context=actor_context,
        session_ref=command.session_ref,
        pins=(
            request.predecessor,
            request.validation_contract,
            *(
                pin
                for pair in request.outcomes
                for pin in (pair.episode, pair.assessment)
            ),
        ),
    )
    formation = service.get_advisory(actor_context=actor_context, pin=request.formation)
    # A reused request key is compared only after current source access, but
    # before new-proposal membership checks. Changed selection/session conflicts
    # even when that changed input would also be invalid for a fresh proposal.
    _recover_advisory_proposal(
        actor_context=actor_context, command=command, formation=formation, store=store
    )
    if request.outcomes != formation.request.outcomes:
        raise RuntimeContextError("INVALID_STATE")
    closure = {rr._pin_key(source.source): source for source in formation.sources}
    if any(
        rr._pin_key(pin) not in closure
        for pin in (request.predecessor, request.validation_contract)
    ):
        raise RuntimeContextError("INCOMPLETE_CLOSURE")
    # get_advisory authenticated the exact original package/callables and all
    # captured nodes. These are pure retained reads, never current compilation.
    predecessor, producing, _ = _candidate_value(
        closure[rr._pin_key(request.predecessor)], closure
    )
    _, validation_producing, _ = _validation_value(
        closure[rr._pin_key(request.validation_contract)], closure
    )
    if producing.inputs != validation_producing.inputs:
        raise RuntimeContextError("INVALID_STATE")
    input_pin = ArtifactPinV1(
        artifact_kind="advisory_compiler_inputs",
        artifact_ref=producing.inputs.input_ref,
        artifact_hash=producing.inputs.input_hash,
    )
    originals = [
        source
        for source in formation.sources
        if source.source.artifact_kind == "normative_semantic_candidate_set"
        and source.dependencies == (input_pin,)
    ]
    if len(originals) != 1:
        raise RuntimeContextError("INCOMPLETE_CLOSURE")
    original, _, _ = _candidate_value(originals[0], closure)
    try:
        _validate_candidate_frame_native_value(request.frame)
    except ValueError as exc:
        raise RuntimeContextError("HASH_MISMATCH") from exc
    if not any(
        frame.model_dump(mode="json") == request.frame.model_dump(mode="json")
        for frame in original.frames
    ):
        raise RuntimeContextError("HASH_MISMATCH")
    if any(frame.frame_id == request.frame.frame_id for frame in predecessor.frames):
        raise RuntimeContextError("INVALID_STATE")
    return formation


def _verify_advisory_proposal_triple(
    *, actor_context, command, proposal, receipt, formation
):
    from services.runtime_context_formation_service import _revalidate
    from storage.runtime_working_context_store import RuntimeContextError

    command = _revalidate(ar.AdvisoryNextProposalCommandV1, command)
    proposal = _revalidate(ar.AdvisoryRepairProposalV1, proposal)
    receipt = _revalidate(ar.AdvisoryNextProposalReceiptV1, receipt)
    for value in (command, proposal, receipt, formation):
        if any(
            getattr(value, field) != getattr(actor_context, field)
            for field in ("organization_id", "workspace_id", "actor_id")
        ):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
    request = command.request
    if (
        any(value.session_ref != command.session_ref for value in (proposal, receipt))
        or any(value.command != _proposal_pin(command) for value in (proposal, receipt))
        or any(value.formation != request.formation for value in (proposal, receipt))
        or any(value.outcomes != request.outcomes for value in (proposal, receipt))
        or rr.artifact_pin_for(formation) != request.formation
        or formation.request.outcomes != request.outcomes
        or any(
            getattr(proposal, field) != getattr(request, field)
            for field in ("predecessor", "operation", "frame", "validation_contract")
        )
        or receipt.proposal != _proposal_pin(proposal)
        or receipt.request_id != request.request_id
        or receipt.request_hash != command.command_hash
        or receipt.recorded_at != proposal.recorded_at
    ):
        raise RuntimeContextError("INVALID_STATE")
    return command, proposal, receipt


def _read_proposal_record(
    *,
    store,
    actor_context,
    model,
    pin,
    lookup_origin="RETAINED_CLOSURE",
):
    from services.runtime_context_formation_service import _translate
    from storage.runtime_working_context_store import (
        RuntimeContextError,
        _with_context_error_origin,
    )

    row = store._connection.execute(
        sql_query("SELECT organization_id, workspace_id, schema_version, artifact_hash, payload_json "
        "FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT organization_id, workspace_id, schema_version, artifact_hash, payload_json FROM ontology_build_artifacts WHERE artifact_ref=%s'),
        (pin.artifact_ref,),
    ).fetchone()
    if row is None:
        raise RuntimeContextError("INCOMPLETE_CLOSURE", _origin=lookup_origin)
    if row[:2] != (actor_context.organization_id, actor_context.workspace_id):
        raise RuntimeContextError("FORBIDDEN_SCOPE", _origin=lookup_origin)
    if (
        row[2] != model.model_fields["schema_version"].annotation.__args__[0]
        or row[3] != pin.artifact_hash
    ):
        raise RuntimeContextError("HASH_MISMATCH")
    try:
        value = model.model_validate_json(row[4])
    except ValueError as exc:
        raise _with_context_error_origin(
            _translate(exc), "RETAINED_CLOSURE"
        ) from exc
    if (
        _proposal_pin(value) != pin
        or canonical_json(value.model_dump(mode="json")) != row[4]
    ):
        raise RuntimeContextError("HASH_MISMATCH")
    return value


def _recover_advisory_proposal(*, actor_context, command, formation, store):
    from storage.runtime_working_context_store import (
        RuntimeContextError,
        _with_context_error_origin,
    )

    row = store._connection.execute(
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
        # The command pin is deterministic before sampling proposal time. Its
        # survival proves this is incomplete publication state, not a fresh
        # request. Recheck here under both the read and writer transactions.
        if (
            store._connection.execute(
                sql_query("SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref=%s'),
                (command.command_ref,),
            ).fetchone()
            is not None
        ):
            raise RuntimeContextError(
                "INCOMPLETE_CLOSURE", _origin="RETAINED_CLOSURE"
            )
        return None
    if row[0] != command.command_hash:
        raise RuntimeContextError("REQUEST_CONFLICT")
    receipt = _read_proposal_record(
        store=store,
        actor_context=actor_context,
        model=ar.AdvisoryNextProposalReceiptV1,
        pin=ArtifactPinV1(
            artifact_kind="advisory_next_proposal_receipt",
            artifact_ref=row[1],
            artifact_hash=row[2],
        ),
    )
    saved_command = _read_proposal_record(
        store=store,
        actor_context=actor_context,
        model=ar.AdvisoryNextProposalCommandV1,
        pin=receipt.command,
    )
    proposal = _read_proposal_record(
        store=store,
        actor_context=actor_context,
        model=ar.AdvisoryRepairProposalV1,
        pin=receipt.proposal,
    )
    try:
        _verify_advisory_proposal_triple(
            actor_context=actor_context,
            command=saved_command,
            proposal=proposal,
            receipt=receipt,
            formation=formation,
        )
    except RuntimeContextError as exc:
        raise _with_context_error_origin(exc, "RETAINED_CLOSURE") from exc
    if saved_command != command or receipt.request_hash != row[0]:
        raise RuntimeContextError("REQUEST_CONFLICT")
    return receipt


class OntologyRepairService:
    def get_advisory_proposal(
        self,
        *,
        actor_context: RuntimeContextDTO,
        session_ref: str,
        pin: ArtifactPinV1,
        formation_service: RuntimeContextFormationService,
        store: OntologyBuildStore,
    ) -> ar.AdvisoryRepairProposalV1:
        from services.runtime_context_formation_service import _revalidate
        from storage.runtime_working_context_store import (
            RuntimeContextError,
            RuntimeTaskContextStore,
            _with_context_error_origin,
        )

        pin = _revalidate(ArtifactPinV1, pin)
        if pin.artifact_kind != "advisory_repair_proposal":
            raise RuntimeContextError("INVALID_STATE")
        facade = RuntimeTaskContextStore(store)

        def read():
            canonical = _proposal_composition(
                actor_context=actor_context,
                session_ref=session_ref,
                formation_service=formation_service,
                store=store,
            )
            proposal = _read_proposal_record(
                store=store,
                actor_context=actor_context,
                model=ar.AdvisoryRepairProposalV1,
                pin=pin,
                lookup_origin="ARTIFACT_LOOKUP",
            )
            if (
                proposal.actor_id != actor_context.actor_id
                or proposal.session_ref != canonical
            ):
                raise RuntimeContextError(
                    "FORBIDDEN_SCOPE", _origin="ARTIFACT_LOOKUP"
                )
            try:
                command = _read_proposal_record(
                    store=store,
                    actor_context=actor_context,
                    model=ar.AdvisoryNextProposalCommandV1,
                    pin=proposal.command,
                )
                formation = facade._get_advisory_formation(
                    actor_context=actor_context,
                    pin=proposal.formation,
                    lookup_origin="RETAINED_CLOSURE",
                )
                receipt = _recover_advisory_proposal(
                    actor_context=actor_context,
                    command=command,
                    formation=formation,
                    store=store,
                )
            except RuntimeContextError as exc:
                raise _with_context_error_origin(
                    exc, "RETAINED_CLOSURE"
                ) from exc
            if receipt is None or receipt.proposal != pin:
                raise RuntimeContextError(
                    "INCOMPLETE_CLOSURE", _origin="RETAINED_CLOSURE"
                )
            formation_service._scope(actor_context, canonical)
            return proposal

        return facade._consistent_read(read)

    def propose_from_advisory(
        self,
        *,
        actor_context: RuntimeContextDTO,
        session_ref: str,
        request: ar.AdvisoryNextProposalRequestV1,
        formation_service: RuntimeContextFormationService,
        store: OntologyBuildStore,
    ) -> ar.AdvisoryNextProposalReceiptV1:
        from services.runtime_context_formation_service import _bind, _revalidate
        from storage.runtime_working_context_store import RuntimeTaskContextStore

        request = _revalidate(ar.AdvisoryNextProposalRequestV1, request)
        canonical = _proposal_composition(
            actor_context=actor_context,
            session_ref=session_ref,
            formation_service=formation_service,
            store=store,
        )
        identity = dict(
            organization_id=actor_context.organization_id,
            workspace_id=actor_context.workspace_id,
            actor_id=actor_context.actor_id,
            session_ref=canonical,
        )
        command = _bind(
            ar.AdvisoryNextProposalCommandV1,
            schema_version="advisory_next_proposal_command.v1",
            **identity,
            operation="PROPOSE_ADVISORY_REPAIR",
            request=request,
        )

        def read():
            formation = _proposal_evidence(
                actor_context=actor_context, command=command, store=store
            )
            return _recover_advisory_proposal(
                actor_context=actor_context,
                command=command,
                formation=formation,
                store=store,
            )

        saved = RuntimeTaskContextStore(store)._consistent_read(read)
        if saved is not None:
            return saved
        recorded_at = datetime.now(timezone.utc)
        proposal = _bind(
            ar.AdvisoryRepairProposalV1,
            schema_version="advisory_repair_proposal.v1",
            **identity,
            command=_proposal_pin(command),
            formation=request.formation,
            outcomes=request.outcomes,
            predecessor=request.predecessor,
            operation="ADD_FRAME",
            frame=request.frame,
            validation_contract=request.validation_contract,
            usage="CONTEXT_ONLY",
            state="PROPOSED",
            recorded_at=recorded_at,
        )
        receipt = _bind(
            ar.AdvisoryNextProposalReceiptV1,
            schema_version="advisory_next_proposal_receipt.v1",
            **identity,
            operation="PROPOSE_ADVISORY_REPAIR",
            request_id=request.request_id,
            request_hash=command.command_hash,
            command=_proposal_pin(command),
            proposal=_proposal_pin(proposal),
            formation=request.formation,
            outcomes=request.outcomes,
            usage="CONTEXT_ONLY",
            recorded_at=recorded_at,
        )
        return store._publish_advisory_proposal(
            actor_context=actor_context,
            command=command,
            proposal=proposal,
            receipt=receipt,
        )

    def _publish_advisory_proposal(
        self,
        *,
        actor_context,
        command,
        proposal,
        receipt,
        store,
        verify_before_commit=None,
    ):
        """Fixed P5 validation/index callbacks inside the shared workspace UoW."""
        from services.runtime_context_formation_service import (
            _require_bound_advisory_access,
        )
        from storage.runtime_working_context_store import RuntimeContextError

        connection = store._connection
        if not in_transaction(connection):
            raise RuntimeContextError("INVALID_STATE")

        def validate(connection):
            if connection is not store._connection:
                raise RuntimeContextError("INVALID_STATE")
            formation, triple = store._validate_advisory_proposal(
                actor_context=actor_context,
                command=command,
                proposal=proposal,
                receipt=receipt,
            )
            saved = _recover_advisory_proposal(
                actor_context=actor_context,
                command=triple[0],
                formation=formation,
                store=store,
            )
            return formation, triple, saved

        with store._guard_readonly_workspace_callback("validate_before_insert"):
            formation, triple, saved = validate(connection)
        command, proposal, receipt = triple
        if saved is None:
            store._insert_prepared(tuple(store._prepare(value) for value in triple))

            def mutate_indexes(connection):
                connection.execute(
                    sql_query("INSERT INTO workspace_operation_request_index "
                    "(organization_id, workspace_id, actor_ref, operation, request_id, request_hash, receipt_ref, receipt_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",'INSERT INTO workspace_operation_request_index (organization_id, workspace_id, actor_ref, operation, request_id, request_hash, receipt_ref, receipt_hash) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'),
                    (
                        command.organization_id,
                        command.workspace_id,
                        command.actor_id,
                        command.operation,
                        command.request.request_id,
                        command.command_hash,
                        receipt.receipt_ref,
                        receipt.receipt_hash,
                    ),
                )

            with store._guard_private_advisory_index():
                mutate_indexes(connection)

        def current_access():
            _require_bound_advisory_access(
                build_store=store,
                actor_context=actor_context,
                session_ref=command.session_ref,
                pins=tuple(source.source for source in formation.sources),
            )

        with store._guard_readonly_workspace_callback("advisory verify_before_commit"):
            current_access()
            if verify_before_commit is not None:
                verify_before_commit()
            current_access()
        return receipt if saved is None else saved

    def add_missing_frame(
        self,
        *,
        predecessor: NormativeSemanticCandidateSetV1,
        frame: NormativeFrameV1,
        repair_ref: str,
        validation_contract_ref: str,
        created_at: datetime,
    ) -> tuple[NormativeSemanticCandidateSetV1, RepairImpactV1]:
        if any(item.frame_id == frame.frame_id for item in predecessor.frames):
            raise ValueError("repair frame already exists")
        frames = tuple(
            sorted((*predecessor.frames, frame), key=lambda item: item.frame_id.encode("utf-8"))
        )
        basis = {
            "predecessor_candidate_set_ref": predecessor.candidate_set_ref,
            "predecessor_candidate_set_hash": predecessor.candidate_set_hash,
            "repair_ref": repair_ref,
            "frames": [item.model_dump(mode="json") for item in frames],
        }
        digest = content_hash(basis)
        payload = predecessor.model_dump(mode="json")
        payload.update(
            {
                "candidate_set_ref": f"normative-candidate-set:{digest}",
                "candidate_set_hash": f"sha256:{digest}",
                "frames": [item.model_dump(mode="json") for item in frames],
                "predecessor_candidate_set_ref": predecessor.candidate_set_ref,
                "predecessor_candidate_set_hash": predecessor.candidate_set_hash,
                "change_cause": "bounded_repair",
                "triggering_ref": repair_ref,
                "compiled_at": created_at.isoformat(),
            }
        )
        successor = NormativeSemanticCandidateSetV1.model_validate(payload)
        preserved = tuple(item.frame_id for item in predecessor.frames)
        impact_basis = {
            "repair_ref": repair_ref,
            "predecessor": predecessor.candidate_set_ref,
            "successor": successor.candidate_set_ref,
            "changed_frame_refs": (frame.frame_id,),
            "preserved_frame_refs": preserved,
            "validation_contract_ref": validation_contract_ref,
        }
        impact_digest = content_hash(impact_basis)
        impact = RepairImpactV1(
            impact_ref=f"repair-impact:{impact_digest}",
            impact_hash=f"sha256:{impact_digest}",
            repair_ref=repair_ref,
            predecessor_candidate_set_ref=predecessor.candidate_set_ref,
            successor_candidate_set_ref=successor.candidate_set_ref,
            changed_frame_refs=(frame.frame_id,),
            preserved_frame_refs=preserved,
            validation_contract_ref=validation_contract_ref,
            invalidated_result_kinds=("quality_evaluation_run",),
            created_at=created_at,
        )
        return successor, impact

    def delete_frame(
        self,
        *,
        predecessor: NormativeSemanticCandidateSetV1,
        frame_id: str,
        linked_critical_coverage_gap_open: bool,
        governed_scope_successor_ref: str | None,
    ) -> None:
        del predecessor, frame_id
        if linked_critical_coverage_gap_open and governed_scope_successor_ref is None:
            raise PermissionError(
                "source-grounded deletion is forbidden while a critical coverage gap is open"
            )
        raise NotImplementedError("deletion requires a separately governed scope successor")


__all__ = ["OntologyRepairService"]
