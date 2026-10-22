from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from semantier.sensitive_governance_contracts.enums import (
    DisplayAction,
    GovernedResponseSegmentKind,
)
from semantier.sensitive_governance_contracts.hashing import canonical_hash
from semantier.sensitive_governance_contracts.pins import SensitiveGovernancePins
from semantier.sensitive_governance_contracts.registry import CURRENT_CONTRACT_VERSION


@dataclass(frozen=True)
class GovernedResponseSegment:
    segment_id: str
    sequence: int
    kind: GovernedResponseSegmentKind
    display_action: DisplayAction
    text: str
    classification: str | None = None
    authority_refs: list[dict[str, Any]] | None = None
    eligible_for_reveal: bool = False
    restricted_payload_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["kind"] = self.kind.value
        result["display_action"] = self.display_action.value
        return result


@dataclass(frozen=True)
class GovernedResponse:
    response_id: str
    run_id: str
    contract_version: str
    segments: list[GovernedResponseSegment]
    governance_summary: dict[str, Any]
    pins: SensitiveGovernancePins
    response_hash: str | None = None

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "run_id": self.run_id,
            "contract_version": self.contract_version,
            "segments": [segment.to_dict() for segment in self.segments],
            "governance_summary": self.governance_summary,
            "pins": self.pins.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["response_hash"] = self.response_hash or canonical_hash(payload)
        return payload


def build_response(
    *,
    response_id: str,
    run_id: str,
    segments: list[GovernedResponseSegment],
    governance_summary: dict[str, Any],
    pins: SensitiveGovernancePins,
) -> GovernedResponse:
    response = GovernedResponse(
        response_id=response_id,
        run_id=run_id,
        contract_version=CURRENT_CONTRACT_VERSION,
        segments=segments,
        governance_summary=governance_summary,
        pins=pins,
    )
    return GovernedResponse(
        response_id=response.response_id,
        run_id=response.run_id,
        contract_version=response.contract_version,
        segments=response.segments,
        governance_summary=response.governance_summary,
        pins=response.pins,
        response_hash=canonical_hash(response.unsigned_dict()),
    )
