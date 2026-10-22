from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RestrictedPayloadRef:
    ref_id: str
    custody_provider: str
    object_hash: str
    purpose: str


class RestrictedPayloadCustodyProvider(Protocol):
    def put_clear_value(self, *, value: str, purpose: str) -> RestrictedPayloadRef:
        ...

    def get_clear_value(self, *, ref: RestrictedPayloadRef) -> str:
        ...
