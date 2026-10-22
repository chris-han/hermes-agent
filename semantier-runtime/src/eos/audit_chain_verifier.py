"""Verification helpers for governed event streams."""

from __future__ import annotations

from typing import Any, Dict

from storage.event_store import UnifiedEventStore


def verify_governed_stream(store: UnifiedEventStore, stream_id: str) -> Dict[str, Any]:
    verification = store.verify_chain(stream_id)
    return {
        "stream_id": stream_id,
        **verification,
    }
