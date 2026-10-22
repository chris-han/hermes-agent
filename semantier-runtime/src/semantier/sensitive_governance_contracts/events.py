from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from eos.db import now_iso
from semantier.sensitive_governance_contracts.enums import ReplayPhase
from semantier.sensitive_governance_contracts.hashing import canonical_hash


@dataclass(frozen=True)
class SensitiveGovernanceEvent:
    event_id: str
    event_type: str
    phase: ReplayPhase
    occurred_at: str
    aggregate_id: str
    payload: dict[str, Any]
    causation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["phase"] = self.phase.value
        result["event_hash"] = canonical_hash(
            {
                "event_id": self.event_id,
                "event_type": self.event_type,
                "phase": self.phase.value,
                "occurred_at": self.occurred_at,
                "aggregate_id": self.aggregate_id,
                "payload": self.payload,
                "causation_id": self.causation_id,
            }
        )
        return result


def event_now(
    *,
    event_id: str,
    event_type: str,
    phase: ReplayPhase,
    aggregate_id: str,
    payload: dict[str, Any],
    causation_id: str | None = None,
) -> SensitiveGovernanceEvent:
    return SensitiveGovernanceEvent(
        event_id=event_id,
        event_type=event_type,
        phase=phase,
        occurred_at=now_iso(),
        aggregate_id=aggregate_id,
        payload=payload,
        causation_id=causation_id,
    )
