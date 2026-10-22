from __future__ import annotations

import threading
from typing import Any

from .models import (
    ActiveSemanticContract,
    AuthorityPins,
    ContextRef,
    EffectiveContextCheckpoint,
    EffectiveContextEvent,
    EventType,
    RefType,
    StateEffect,
    sha256_hex,
)
from .store import CHECKPOINT_AUTHORITY_PROOF_MISSING, ECC_SCOPE_REQUIRED, EffectiveContextStore


class EffectiveContextCheckpointService:
    _boundary_locks: dict[str, threading.RLock] = {}
    _boundary_locks_guard = threading.Lock()

    def __init__(self, store: EffectiveContextStore | None = None, *, full_snapshot_interval: int = 10) -> None:
        self.store = store or EffectiveContextStore()
        if full_snapshot_interval < 1:
            raise ValueError("full_snapshot_interval must be positive")
        self.full_snapshot_interval = full_snapshot_interval
        lock_key = str(getattr(self.store, "_db_path", None) or "default")
        with self._boundary_locks_guard:
            self._boundary_lock = self._boundary_locks.setdefault(lock_key, threading.RLock())


    def append_event(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        session_id: str,
        actor_ref: str,
        event_type: EventType,
        state_effect: StateEffect,
        authority_context_ref: str,
        input_refs: list[ContextRef] | None = None,
        output_refs: list[ContextRef] | None = None,
        cause_event_ids: list[str] | None = None,
    ) -> EffectiveContextEvent:
        self._validate_ref_scope(organization_id, workspace_id, input_refs or [])
        self._validate_ref_scope(organization_id, workspace_id, output_refs or [])
        event = EffectiveContextEvent(
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            actor_ref=actor_ref,
            event_type=event_type,
            state_effect=state_effect,
            input_refs=input_refs or [],
            output_refs=output_refs or [],
            cause_event_ids=cause_event_ids or [],
            authority_context_ref=authority_context_ref,
        )
        return self.store.append_event(event)

    @staticmethod
    def _validate_ref_scope(
        organization_id: str,
        workspace_id: str,
        refs: list[ContextRef],
    ) -> None:
        for ref in refs:
            if ref.organization_id != organization_id or ref.workspace_id != workspace_id:
                raise ValueError(ECC_SCOPE_REQUIRED)

    def record_message_committed(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        session_id: str,
        message_id: str,
        message_content_hash: str,
        actor_ref: str,
        authority_context_ref: str,
        cause_event_ids: list[str] | None = None,
    ) -> EffectiveContextEvent:
        message_ref = ContextRef(
            ref_type=RefType.MESSAGE,
            ref_id=message_id,
            content_hash=message_content_hash,
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            authority_role="contextual",
            admission_state="admitted",
        )
        return self.append_event(
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            actor_ref=actor_ref,
            event_type=EventType.MESSAGE_COMMITTED,
            state_effect=StateEffect.ADMIT,
            authority_context_ref=authority_context_ref,
            output_refs=[message_ref],
            cause_event_ids=cause_event_ids,
        )

    def record_tool_observation_committed(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        session_id: str,
        observation_id: str,
        observation_hash: str,
        actor_ref: str,
        authority_context_ref: str,
        input_refs: list[ContextRef] | None = None,
        cause_event_ids: list[str] | None = None,
    ) -> EffectiveContextEvent:
        observation_ref = ContextRef(
            ref_type=RefType.TOOL_OBSERVATION,
            ref_id=observation_id,
            content_hash=observation_hash,
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            authority_role="operational",
            admission_state="used",
        )
        return self.append_event(
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            actor_ref=actor_ref,
            event_type=EventType.TOOL_OBSERVATION_COMMITTED,
            state_effect=StateEffect.DISCLOSE,
            authority_context_ref=authority_context_ref,
            input_refs=input_refs,
            output_refs=[observation_ref],
            cause_event_ids=cause_event_ids,
        )

    def record_disclosure(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        session_id: str,
        disclosed_ref: ContextRef,
        actor_ref: str,
        authority_context_ref: str,
        cause_event_ids: list[str] | None = None,
    ) -> EffectiveContextEvent:
        event_type_by_ref = {
            RefType.DOCUMENT_SPAN: EventType.DOCUMENT_DISCLOSED,
            RefType.KNOWLEDGE_ITEM: EventType.KNOWLEDGE_RETRIEVED,
            RefType.DATABASE_RESULT: EventType.DATABASE_QUERY_DISCLOSED,
            RefType.ARTIFACT: EventType.ARTIFACT_MATERIALIZED,
            RefType.MEMORY: EventType.MEMORY_ADMITTED,
            RefType.SKILL: EventType.SKILL_LOADED,
        }
        event_type = event_type_by_ref.get(disclosed_ref.ref_type)
        if event_type is None:
            raise ValueError(f"unsupported disclosure ref_type: {disclosed_ref.ref_type}")
        state_effect = (
            StateEffect.MATERIALIZE
            if disclosed_ref.ref_type == RefType.ARTIFACT
            else StateEffect.ADMIT
            if disclosed_ref.ref_type in {RefType.MEMORY, RefType.SKILL}
            else StateEffect.DISCLOSE
        )
        normalized_ref = disclosed_ref.model_copy(
            update={
                "admission_state": disclosed_ref.admission_state or "retrieved",
            }
        )
        return self.append_event(
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            actor_ref=actor_ref,
            event_type=event_type,
            state_effect=state_effect,
            authority_context_ref=authority_context_ref,
            output_refs=[normalized_ref],
            cause_event_ids=cause_event_ids,
        )

    def record_knowledge_admitted(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        session_id: str,
        candidate_ref: ContextRef,
        admission_event_id: str,
        actor_ref: str,
        authority_context_ref: str,
        cause_event_ids: list[str] | None = None,
    ) -> EffectiveContextEvent:
        """Promote a retrieved candidate only through an explicit admission event."""
        if candidate_ref.ref_type != RefType.KNOWLEDGE_ITEM:
            raise ValueError("knowledge admission requires a knowledge_item ref")
        self._validate_ref_scope(organization_id, workspace_id, [candidate_ref])
        admitted_ref = candidate_ref.model_copy(
            update={
                "source_event_id": admission_event_id,
                "admission_state": "admitted",
            }
        )
        return self.append_event(
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            actor_ref=actor_ref,
            event_type=EventType.KNOWLEDGE_ADMITTED,
            state_effect=StateEffect.ADMIT,
            authority_context_ref=authority_context_ref,
            input_refs=[candidate_ref],
            output_refs=[admitted_ref],
            cause_event_ids=cause_event_ids,
        )

    def record_database_result_disclosed(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        session_id: str,
        result_id: str,
        result_hash: str,
        entitlement_context_ref: str,
        actor_ref: str,
        data_snapshot_ref: str | None = None,
        cause_event_ids: list[str] | None = None,
    ) -> EffectiveContextEvent:
        """Record a database result with its original entitlement context pinned."""
        result_ref = ContextRef(
            ref_type=RefType.DATABASE_RESULT,
            ref_id=result_id,
            version_ref=data_snapshot_ref,
            content_hash=result_hash,
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            authority_role="operational",
            admission_state="retrieved",
        )
        return self.append_event(
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            actor_ref=actor_ref,
            event_type=EventType.DATABASE_QUERY_DISCLOSED,
            state_effect=StateEffect.DISCLOSE,
            authority_context_ref=entitlement_context_ref,
            output_refs=[result_ref],
            cause_event_ids=cause_event_ids,
        )

    def materialize_checkpoint(
        self,
        **kwargs: Any,
    ) -> EffectiveContextCheckpoint:
        """Materialize once per semantic boundary under a process boundary lock."""
        with self._boundary_lock:
            return self._materialize_checkpoint(**kwargs)

    def _materialize_checkpoint(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        session_id: str,
        boundary_message_id: str,
        actor_ref: str,
        knowledge_state: dict[str, Any],
        active_contract: ActiveSemanticContract,
        authority_pins: dict[str, Any],
        transcript_snapshot: dict[str, Any],
        parent_checkpoint_id: str | None = None,
        delta: dict[str, Any] | None = None,
        force_full: bool = False,
    ) -> EffectiveContextCheckpoint:
        existing = self.store.get_checkpoint_by_boundary(
            organization_id, workspace_id, session_id, boundary_message_id
        )
        if existing is not None:
            return existing
        events = self.store.list_events(organization_id, workspace_id, session_id)
        if not events:
            raise ValueError("checkpoint requires at least one event")
        boundary_sequence = events[-1].sequence
        try:
            pins = AuthorityPins.model_validate(authority_pins)
        except Exception as exc:
            raise ValueError(CHECKPOINT_AUTHORITY_PROOF_MISSING) from exc
        if pins.organization_ref and pins.organization_ref != organization_id:
            raise ValueError(ECC_SCOPE_REQUIRED)
        if pins.workspace_ref and pins.workspace_ref != workspace_id:
            raise ValueError(ECC_SCOPE_REQUIRED)

        knowledge_ref = self.store.put_payload(organization_id, workspace_id, "knowledge_state", knowledge_state)
        contract_ref = self.store.put_payload(organization_id, workspace_id, "active_contract", active_contract.model_dump(mode="json"))
        pins_ref = self.store.put_payload(organization_id, workspace_id, "authority_pins", pins.model_dump(mode="json"))
        transcript_ref = self.store.put_payload(organization_id, workspace_id, "transcript_snapshot", transcript_snapshot)

        parent = self.store.get_checkpoint(organization_id, workspace_id, parent_checkpoint_id) if parent_checkpoint_id else None
        if parent is not None:
            if parent.session_id != session_id:
                raise ValueError(ECC_SCOPE_REQUIRED)
            if parent.boundary_event_sequence >= boundary_sequence:
                raise ValueError("parent checkpoint must precede child boundary")
        checkpoint_count = self.store.count_checkpoints(organization_id, workspace_id, session_id)
        is_periodic_full = force_full or parent is None or (
            self.full_snapshot_interval > 0 and checkpoint_count % self.full_snapshot_interval == 0
        )
        delta_ref = None if is_periodic_full else (
            self.store.put_payload(organization_id, workspace_id, "checkpoint_delta", delta) if delta else None
        )
        chain_verified = self.store.verify_event_chain(
            organization_id, workspace_id, session_id, through_sequence=boundary_sequence
        )
        checkpoint = EffectiveContextCheckpoint(
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            boundary_message_id=boundary_message_id,
            boundary_event_sequence=boundary_sequence,
            knowledge_state_ref=knowledge_ref,
            active_contract_ref=contract_ref,
            authority_pins_ref=pins_ref,
            transcript_snapshot_ref=transcript_ref,
            trajectory_from_sequence=1 if parent is None else parent.boundary_event_sequence + 1,
            trajectory_to_sequence=boundary_sequence,
            parent_checkpoint_id=parent_checkpoint_id,
            lineage_root_checkpoint_id=parent.lineage_root_checkpoint_id if parent else None,
            delta_ref=delta_ref,
            verification_status="verified" if chain_verified else "failed",
            verification_reason=None if chain_verified else "EVENT_HASH_CHAIN_MISMATCH",
        )
        checkpoint = self.store.insert_checkpoint(checkpoint)
        checkpoint_ref = ContextRef(
            ref_type="checkpoint",
            ref_id=checkpoint.checkpoint_id,
            content_hash=checkpoint.checkpoint_hash or sha256_hex(checkpoint.hash_payload()),
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            authority_role="operational",
            admission_state="active",
        )
        self.append_event(
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            actor_ref=actor_ref,
            event_type=EventType.CHECKPOINT_MATERIALIZED,
            state_effect=StateEffect.MATERIALIZE,
            authority_context_ref=pins.historical_admission_proof_ref,
            output_refs=[checkpoint_ref],
            cause_event_ids=[events[-1].event_id],
        )
        return checkpoint

    def verify_checkpoint(self, organization_id: str, workspace_id: str, checkpoint_id: str) -> bool:
        checkpoint = self.store.get_checkpoint(organization_id, workspace_id, checkpoint_id)
        if checkpoint.checkpoint_hash != sha256_hex(checkpoint.hash_payload()):
            return False
        for ref in (
            checkpoint.knowledge_state_ref,
            checkpoint.active_contract_ref,
            checkpoint.authority_pins_ref,
            checkpoint.transcript_snapshot_ref,
        ):
            self.store.get_payload(organization_id, workspace_id, ref)
        return self.store.verify_event_chain(
            organization_id,
            workspace_id,
            checkpoint.session_id,
            through_sequence=checkpoint.boundary_event_sequence,
        )

    def branch_checkpoint(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        source_checkpoint_id: str,
        child_session_id: str,
        boundary_message_id: str,
        actor_ref: str,
        authority_context_ref: str,
    ) -> EffectiveContextCheckpoint:
        """Create a child checkpoint linked to a verified source checkpoint."""
        with self._boundary_lock:
            source = self.store.get_checkpoint(
                organization_id, workspace_id, source_checkpoint_id
            )
            if source.verification_status != "verified" or not self.verify_checkpoint(
                organization_id, workspace_id, source_checkpoint_id
            ):
                raise ValueError("BRANCH_CONTEXT_INCOMPLETE")
            existing = self.store.get_checkpoint_by_boundary(
                organization_id, workspace_id, child_session_id, boundary_message_id
            )
            if existing is not None:
                return existing

            pins = self.store.get_payload(
                organization_id, workspace_id, source.authority_pins_ref
            )
            proof = str(pins.get("historical_admission_proof_ref") or "").strip()
            if not proof:
                raise ValueError(CHECKPOINT_AUTHORITY_PROOF_MISSING)
            source_ref = ContextRef(
                ref_type=RefType.CHECKPOINT,
                ref_id=source.checkpoint_id,
                content_hash=source.checkpoint_hash or sha256_hex(source.hash_payload()),
                organization_id=organization_id,
                workspace_id=workspace_id,
                session_id=source.session_id,
                authority_role="operational",
                admission_state="active",
            )
            branch_event = self.append_event(
                organization_id=organization_id,
                workspace_id=workspace_id,
                session_id=child_session_id,
                actor_ref=actor_ref,
                event_type=EventType.SESSION_BRANCHED,
                state_effect=StateEffect.MATERIALIZE,
                authority_context_ref=authority_context_ref or proof,
                input_refs=[source_ref],
            )
            checkpoint = EffectiveContextCheckpoint(
                organization_id=organization_id,
                workspace_id=workspace_id,
                session_id=child_session_id,
                boundary_message_id=boundary_message_id,
                boundary_event_sequence=branch_event.sequence,
                knowledge_state_ref=source.knowledge_state_ref,
                active_contract_ref=source.active_contract_ref,
                authority_pins_ref=source.authority_pins_ref,
                transcript_snapshot_ref=source.transcript_snapshot_ref,
                trajectory_from_sequence=1,
                trajectory_to_sequence=branch_event.sequence,
                parent_checkpoint_id=source.checkpoint_id,
                lineage_root_checkpoint_id=(
                    source.lineage_root_checkpoint_id or source.checkpoint_id
                ),
                verification_status="verified",
            )
            checkpoint = self.store.insert_checkpoint(checkpoint)
            self.append_event(
                organization_id=organization_id,
                workspace_id=workspace_id,
                session_id=child_session_id,
                actor_ref=actor_ref,
                event_type=EventType.CHECKPOINT_MATERIALIZED,
                state_effect=StateEffect.MATERIALIZE,
                authority_context_ref=proof,
                output_refs=[ContextRef(
                    ref_type=RefType.CHECKPOINT,
                    ref_id=checkpoint.checkpoint_id,
                    content_hash=checkpoint.checkpoint_hash or sha256_hex(checkpoint.hash_payload()),
                    organization_id=organization_id,
                    workspace_id=workspace_id,
                    session_id=child_session_id,
                    authority_role="operational",
                    admission_state="active",
                )],
                cause_event_ids=[branch_event.event_id],
            )
            return checkpoint

    def restore_read_only(self, organization_id: str, workspace_id: str, checkpoint_id: str) -> dict[str, Any]:
        checkpoint = self.store.get_checkpoint(organization_id, workspace_id, checkpoint_id)
        pins = self.store.get_payload(organization_id, workspace_id, checkpoint.authority_pins_ref)
        if not pins.get("historical_admission_proof_ref"):
            raise ValueError(CHECKPOINT_AUTHORITY_PROOF_MISSING)
        return {
            "checkpoint": checkpoint.model_dump(mode="json"),
            "knowledge_state": self.store.get_payload(organization_id, workspace_id, checkpoint.knowledge_state_ref),
            "active_contract": self.store.get_payload(organization_id, workspace_id, checkpoint.active_contract_ref),
            "authority_pins": pins,
            "transcript_snapshot": self.store.get_payload(organization_id, workspace_id, checkpoint.transcript_snapshot_ref),
        }
