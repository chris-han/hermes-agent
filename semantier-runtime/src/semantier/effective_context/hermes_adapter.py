from __future__ import annotations

from typing import Any

from .models import ActiveSemanticContract, EffectiveContextCheckpoint, EffectiveContextEvent, sha256_hex
from .service import EffectiveContextCheckpointService


class HermesEffectiveContextAdapter:
    """Translate explicit Hermes persistence envelopes into ECC boundaries.

    The adapter deliberately requires organization, workspace, and authority
    inputs from the authenticated caller. A Hermes session id is not an
    authority source and is never parsed to derive tenant identity.
    """

    def __init__(self, service: EffectiveContextCheckpointService) -> None:
        self.service = service

    def record_persisted_message(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        session_id: str,
        message_id: str,
        role: str,
        content: str,
        actor_ref: str,
        authority_context_ref: str,
        checkpoint: bool = False,
        knowledge_state: dict[str, Any] | None = None,
        active_contract: ActiveSemanticContract | None = None,
        authority_pins: dict[str, Any] | None = None,
        transcript_snapshot: dict[str, Any] | None = None,
        parent_checkpoint_id: str | None = None,
        delta: dict[str, Any] | None = None,
    ) -> tuple[EffectiveContextEvent, EffectiveContextCheckpoint | None]:
        if role not in {"user", "assistant", "tool", "system"}:
            raise ValueError("unsupported Hermes message role")
        event = self.service.record_message_committed(
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            message_id=message_id,
            message_content_hash=sha256_hex({"role": role, "content": content}),
            actor_ref=actor_ref,
            authority_context_ref=authority_context_ref,
        )
        if not checkpoint:
            return event, None
        if role != "assistant":
            raise ValueError("only completed assistant messages may materialize a checkpoint")
        if active_contract is None or authority_pins is None or transcript_snapshot is None:
            raise ValueError("assistant checkpoint inputs are required")
        result = self.service.materialize_checkpoint(
            organization_id=organization_id,
            workspace_id=workspace_id,
            session_id=session_id,
            boundary_message_id=message_id,
            actor_ref=actor_ref,
            knowledge_state=knowledge_state or {},
            active_contract=active_contract,
            authority_pins=authority_pins,
            transcript_snapshot=transcript_snapshot,
            parent_checkpoint_id=parent_checkpoint_id,
            delta=delta,
        )
        return event, result
