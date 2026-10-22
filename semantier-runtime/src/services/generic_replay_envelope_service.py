from __future__ import annotations

from typing import Any, Protocol

from contracts.tender_sensitive_term_detection_modeling_core import RuntimeContextDTO


class GenericReplayEnvelopeService(Protocol):
    def create_typed_replay_binding(
        self,
        *,
        binding_type: str,
        payload: dict[str, Any],
        context: RuntimeContextDTO,
    ) -> dict[str, Any]: ...

    def verify_typed_replay_binding(
        self,
        replay_binding_id: str,
        context: RuntimeContextDTO,
    ) -> dict[str, Any]: ...


class StoreBackedGenericReplayEnvelopeService:
    def __init__(self, store: Any | None = None) -> None:
        if store is None:
            from eos.generic_replay_envelope_store import GenericReplayEnvelopeStore

            store = GenericReplayEnvelopeStore()
        self._store = store

    def create_typed_replay_binding(
        self,
        *,
        binding_type: str,
        payload: dict[str, Any],
        context: RuntimeContextDTO,
    ) -> dict[str, Any]:
        return self._store.create_typed_replay_binding(
            binding_type=binding_type,
            organization_id=context.organization_id,
            workspace_id=context.workspace_id,
            payload=payload,
            **{
                key: value
                for key, value in payload.items()
                if key
                in {
                    "replay_binding_id",
                    "binding_schema_version",
                    "subject_type",
                    "subject_id",
                }
            },
        )

    def verify_typed_replay_binding(
        self,
        replay_binding_id: str,
        context: RuntimeContextDTO,
    ) -> dict[str, Any]:
        return self._store.verify_typed_replay_binding(
            replay_binding_id=replay_binding_id,
            expected_organization_id=context.organization_id,
            expected_workspace_id=context.workspace_id,
        )
