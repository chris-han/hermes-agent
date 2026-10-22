from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal, Protocol
import hashlib
import json
import uuid


class ItemType(StrEnum):
    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    RULE = "rule"


class Scope(StrEnum):
    USER = "user"
    WORKSPACE = "workspace"
    ORGANIZATION = "organization"


class Status(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    DISABLED = "disabled"
    PENDING = "pending"
    REJECTED = "rejected"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CognitiveItem:
    id: str
    type: ItemType
    scope: Scope
    status: Status
    source_refs: list[str]
    content: dict[str, Any]
    workspace_id: str
    organization_id: str
    user_id: str | None = None
    confidence: float | None = None
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id or not self.workspace_id or not self.organization_id:
            raise ValueError("cognitive item identity and scope are required")
        if self.type == ItemType.KNOWLEDGE and self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.scope == Scope.USER and not self.user_id:
            raise ValueError("user scope requires user_id")

    @classmethod
    def new(cls, *, type: ItemType, scope: Scope, workspace_id: str, organization_id: str,
            content: dict[str, Any], source_refs: list[str], user_id: str | None = None,
            confidence: float | None = None, status: Status = Status.CANDIDATE) -> "CognitiveItem":
        return cls(_id("item"), type, scope, status, list(source_refs), dict(content), workspace_id,
                   organization_id, user_id, confidence)


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    workspace_id: str
    organization_id: str
    session_id: str
    episode_id: str
    turn_id: str
    kind: str
    content: dict[str, Any]
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def new(cls, *, workspace_id: str, organization_id: str, session_id: str, episode_id: str,
            turn_id: str, kind: str, content: dict[str, Any]) -> "EvidenceRecord":
        return cls(_id("evidence"), workspace_id, organization_id, session_id, episode_id, turn_id, kind, dict(content))


@dataclass(frozen=True)
class DecisionContext:
    context_id: str
    request_ref: str
    evidence_refs: list[str]
    knowledge_refs: list[str]
    rule_refs: list[str]
    conflicts: list[dict[str, Any]]
    context_hash: str
    workspace_id: str
    organization_id: str
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def build(cls, *, request_ref: str, evidence_refs: list[str], knowledge_refs: list[str],
              rule_refs: list[str], conflicts: list[dict[str, Any]], workspace_id: str,
              organization_id: str) -> "DecisionContext":
        payload = {"request_ref": request_ref, "evidence_refs": sorted(set(evidence_refs)),
                   "knowledge_refs": sorted(set(knowledge_refs)), "rule_refs": sorted(set(rule_refs)),
                   "conflicts": conflicts}
        return cls(_id("context"), request_ref, payload["evidence_refs"], payload["knowledge_refs"],
                   payload["rule_refs"], list(conflicts), canonical_hash(payload), workspace_id, organization_id)


@dataclass(frozen=True)
class LearningCandidate:
    id: str
    proposed_type: ItemType
    evidence_refs: list[str]
    hypothesis: str
    confidence: float
    proposed_change: dict[str, Any]
    workspace_id: str
    organization_id: str
    status: Status = Status.CANDIDATE
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def new(cls, *, proposed_type: ItemType, evidence_refs: list[str], hypothesis: str,
            confidence: float, proposed_change: dict[str, Any], workspace_id: str,
            organization_id: str) -> "LearningCandidate":
        return cls(_id("candidate"), proposed_type, list(evidence_refs), hypothesis, confidence,
                   dict(proposed_change), workspace_id, organization_id)

    def __post_init__(self) -> None:
        if not self.hypothesis or not self.evidence_refs:
            raise ValueError("learning candidate requires hypothesis and provenance")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class LearningProposal:
    proposed_type: ItemType
    hypothesis: str
    confidence: float
    proposed_change: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.hypothesis or not 0 <= self.confidence <= 1:
            raise ValueError("learning proposal requires a hypothesis and confidence in [0, 1]")


class LearningProposalProvider(Protocol):
    def propose(self, *, evidence: EvidenceRecord, event_kind: str) -> LearningProposal:
        ...


class DecisionStepConflictError(RuntimeError):
    """Same immutable step key was delivered with different canonical contents."""


@dataclass(frozen=True)
class DecisionStep:
    step_id: str
    step_key: str
    organization_id: str
    workspace_id: str
    session_id: str
    turn_id: str | None
    request_ref: str
    decision_context_ref: str
    checkpoint_ref: str | None
    trajectory_ref: str
    action_refs: tuple[str, ...]
    tool_refs: tuple[str, ...]
    outcome_ref: str
    event_kind: str
    logical_step_id: str | None = None
    execution_attempt_ref: str | None = None
    execution_attempt_hash: str | None = None
    attempt_index: int = 0
    lane: Literal["TASK", "RECOVERY", "DIAGNOSTIC"] = "TASK"
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if (self.execution_attempt_ref is None) != (self.execution_attempt_hash is None):
            raise ValueError("execution attempt ref/hash must both be present or absent")
        if self.execution_attempt_hash is not None and (
            len(self.execution_attempt_hash) != 71
            or not self.execution_attempt_hash.startswith("sha256:")
            or any(ch not in "0123456789abcdef" for ch in self.execution_attempt_hash[7:])
        ):
            raise ValueError("execution_attempt_hash must be sha256:<64 lowercase hex>")
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be nonnegative")

    @classmethod
    def build(cls, *, organization_id: str, workspace_id: str, session_id: str,
              turn_id: str | None, request_ref: str, decision_context_ref: str,
              checkpoint_ref: str | None, trajectory_ref: str, action_refs: tuple[str, ...],
              tool_refs: tuple[str, ...], outcome_ref: str, event_kind: str,
              logical_step_id: str | None = None, execution_attempt_ref: str | None = None,
              execution_attempt_hash: str | None = None, attempt_index: int = 0,
              lane: Literal["TASK", "RECOVERY", "DIAGNOSTIC"] = "TASK") -> "DecisionStep":
        if (execution_attempt_ref is None) != (execution_attempt_hash is None):
            raise ValueError("execution attempt ref/hash must both be present or absent")
        if attempt_index < 0:
            raise ValueError("attempt_index must be nonnegative")
        identity = {
            "organization_id": organization_id, "workspace_id": workspace_id,
            "session_id": session_id, "request_ref": request_ref,
            "decision_context_ref": decision_context_ref, "trajectory_ref": trajectory_ref,
            "outcome_ref": outcome_ref, "event_kind": event_kind,
        }
        if (
            logical_step_id is not None
            or execution_attempt_ref is not None
            or execution_attempt_hash is not None
            or attempt_index != 0
            or lane != "TASK"
        ):
            identity.update({
                "logical_step_id": logical_step_id,
                "execution_attempt_ref": execution_attempt_ref,
                "execution_attempt_hash": execution_attempt_hash,
                "attempt_index": attempt_index,
                "lane": lane,
            })
        return cls(_id("step"), canonical_hash(identity), organization_id, workspace_id, session_id,
                   turn_id, request_ref, decision_context_ref, checkpoint_ref, trajectory_ref,
                   tuple(action_refs), tuple(tool_refs), outcome_ref, event_kind, logical_step_id,
                   execution_attempt_ref, execution_attempt_hash, attempt_index, lane)


@dataclass(frozen=True)
class LearningProcess:
    learning_evidence_key: str
    organization_id: str
    workspace_id: str
    step_id: str
    event_kind: str
    outcome_ref: str
    stage: Literal["started", "evidence_captured", "proposal_captured", "candidate_created", "closed"]
    evidence_id: str | None = None
    proposal_schema_version: int | None = None
    proposal_json: str | None = None
    candidate_id: str | None = None
    terminal_status: Literal["active", "pending", "rejected", "ignored"] | None = None
    item_id: str | None = None
    last_error_code: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
