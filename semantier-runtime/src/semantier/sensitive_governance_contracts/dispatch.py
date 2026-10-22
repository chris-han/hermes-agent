from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DispatchOutboxEntry:
    dispatch_id: str
    aggregate_id: str
    adapter: str
    idempotency_key: str
    status: str
    payload_hash: str


@dataclass(frozen=True)
class DispatchAttemptEvent:
    dispatch_id: str
    attempt_id: str
    adapter: str
    status: str
    idempotency_key: str
