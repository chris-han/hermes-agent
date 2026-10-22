from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from eos.artifact_hashing import content_hash


class DriftValidationError(ValueError):
    pass


class DriftFamily(str, Enum):
    PHYSICAL_METADATA = "physical_metadata"
    DATA_CONTRACT = "data_contract"
    LINEAGE = "lineage"
    QUALITY = "quality"
    CLASSIFICATION = "classification"
    SEMANTIC = "semantic"
    AUTHORITY = "authority"
    JURISDICTION = "jurisdiction"
    RUNTIME_BEHAVIOR = "runtime_behavior"
    OUTCOME = "outcome"


class LearningRoute(str, Enum):
    LOOP_A_OPERATIONAL = "LOOP_A_OPERATIONAL"
    LOOP_B_INSTITUTIONAL = "LOOP_B_INSTITUTIONAL"
    INCIDENT_ONLY = "INCIDENT_ONLY"
    INFORMATION_ONLY = "INFORMATION_ONLY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    DUPLICATE_FINDING = "DUPLICATE_FINDING"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class PolicyState(str, Enum):
    DRAFT = "DRAFT"
    PROPOSED = "PROPOSED"
    CERTIFIED = "CERTIFIED"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    SUPERSEDED = "SUPERSEDED"
    REVOKED = "REVOKED"


LOOP_B_FAMILIES = {
    DriftFamily.CLASSIFICATION.value,
    DriftFamily.SEMANTIC.value,
    DriftFamily.AUTHORITY.value,
    DriftFamily.JURISDICTION.value,
    DriftFamily.OUTCOME.value,
}

UTC_ERROR = "timestamp must be timezone-aware UTC"


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise DriftValidationError(UTC_ERROR)
    utc = parsed.astimezone(timezone.utc)
    if utc.utcoffset() != timezone.utc.utcoffset(utc):
        raise DriftValidationError(UTC_ERROR)
    return utc


def canonical_window(start: str, end: str) -> str:
    started = parse_utc(start)
    ended = parse_utc(end)
    if ended <= started:
        raise DriftValidationError("observation window end must be after start")
    return f"{started.isoformat()}/{ended.isoformat()}"


def stable_hash(payload: dict[str, Any]) -> str:
    return content_hash(payload)


@dataclass(frozen=True)
class DriftMonitorPolicy:
    policy_id: str
    organization_id: str
    workspace_id: str | None
    asset_selector: dict[str, Any]
    monitored_dimensions: list[str]
    cadence: str
    baseline_resolution_mode: str
    allowed_self_healing_level: str
    owner_role: str
    reviewer_role: str
    governance_state: str
    policy_version: str
    effective_from: str
    effective_to: str | None
    policy_hash: str

    def to_record(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class DriftObservation:
    observation_id: str
    idempotency_key: str
    policy_id: str
    organization_id: str
    workspace_id: str | None
    asset_ref: str
    observation_window: str
    payload: dict[str, Any]
    source_anchors: list[str]
    collector_version: str
    observation_hash: str

    def to_record(self) -> dict[str, Any]:
        return self.__dict__.copy()

