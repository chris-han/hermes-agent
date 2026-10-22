from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UUIDV7_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
SCHEMA_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+$")
RFC3339_MILLIS_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def utc_now_millis() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def uuid7() -> str:
    """Return an RFC 9562-shaped UUIDv7 string without adding a runtime dependency."""
    unix_ms = int(time.time() * 1000)
    rand = uuid.uuid4().int & ((1 << 74) - 1)
    value = (unix_ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= ((rand >> 62) & 0xFFF) << 64
    value |= 0x2 << 62
    value |= rand & ((1 << 62) - 1)
    return str(uuid.UUID(int=value))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_identifier(value: str) -> str:
    if not isinstance(value, str) or not (1 <= len(value.encode("utf-8")) <= 128):
        raise ValueError("identifiers must be UTF-8 strings of 1-128 bytes")
    return value


def validate_timestamp(value: str) -> str:
    if not isinstance(value, str) or not RFC3339_MILLIS_Z_RE.match(value):
        raise ValueError("timestamps must be UTC RFC3339 with millisecond precision and Z suffix")
    return value


def validate_hash(value: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.match(value):
        raise ValueError("hashes must be lowercase SHA-256 hex")
    return value


class StateEffect(StrEnum):
    DISCLOSE = "disclose"
    ADMIT = "admit"
    ACTIVATE = "activate"
    SUPERSEDE = "supersede"
    REVOKE = "revoke"
    MATERIALIZE = "materialize"
    OBSERVE_ONLY = "observe_only"


class EventType(StrEnum):
    MESSAGE_COMMITTED = "MESSAGE_COMMITTED"
    TOOL_CALL_STARTED = "TOOL_CALL_STARTED"
    TOOL_OBSERVATION_COMMITTED = "TOOL_OBSERVATION_COMMITTED"
    DOCUMENT_DISCLOSED = "DOCUMENT_DISCLOSED"
    KNOWLEDGE_RETRIEVED = "KNOWLEDGE_RETRIEVED"
    KNOWLEDGE_ADMITTED = "KNOWLEDGE_ADMITTED"
    DATABASE_QUERY_DISCLOSED = "DATABASE_QUERY_DISCLOSED"
    MEMORY_ADMITTED = "MEMORY_ADMITTED"
    SKILL_LOADED = "SKILL_LOADED"
    SKILL_UNLOADED = "SKILL_UNLOADED"
    ARTIFACT_MATERIALIZED = "ARTIFACT_MATERIALIZED"
    REQUIREMENT_ACTIVATED = "REQUIREMENT_ACTIVATED"
    REQUIREMENT_CORRECTED = "REQUIREMENT_CORRECTED"
    REQUIREMENT_SUPERSEDED = "REQUIREMENT_SUPERSEDED"
    REQUIREMENT_REVOKED = "REQUIREMENT_REVOKED"
    UNRESOLVED_REQUIREMENT_CANDIDATE = "UNRESOLVED_REQUIREMENT_CANDIDATE"
    POLICY_PINNED = "POLICY_PINNED"
    ONTOLOGY_PINNED = "ONTOLOGY_PINNED"
    CONTEXT_COMPACTION_STARTED = "CONTEXT_COMPACTION_STARTED"
    CONTEXT_COMPACTION_COMMITTED = "CONTEXT_COMPACTION_COMMITTED"
    CHECKPOINT_MATERIALIZED = "CHECKPOINT_MATERIALIZED"
    CHECKPOINT_VERIFIED = "CHECKPOINT_VERIFIED"
    CHECKPOINT_RESTORED = "CHECKPOINT_RESTORED"
    SESSION_BRANCHED = "SESSION_BRANCHED"
    LEGACY_BRANCH_IDENTITY_USED = "LEGACY_BRANCH_IDENTITY_USED"


class RefType(StrEnum):
    MESSAGE = "message"
    DOCUMENT_SPAN = "document_span"
    KNOWLEDGE_ITEM = "knowledge_item"
    DATABASE_RESULT = "database_result"
    ARTIFACT = "artifact"
    MEMORY = "memory"
    SKILL = "skill"
    POLICY = "policy"
    ONTOLOGY = "ontology"
    REQUIREMENT = "requirement"
    TOOL_OBSERVATION = "tool_observation"
    COMPACTION = "compaction"
    CHECKPOINT = "checkpoint"


class ContextRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_type: RefType
    ref_id: str
    version_ref: str | None = None
    content_hash: str
    organization_id: str
    workspace_id: str
    session_id: str | None = None
    source_event_id: str | None = None
    effectivity_at: str | None = None
    authority_role: Literal["binding", "interpretive", "contextual", "operational"] | None = None
    admission_state: Literal["retrieved", "admitted", "used", "active", "superseded", "revoked"] | None = None

    _ids = field_validator("ref_id", "organization_id", "workspace_id", "session_id", "source_event_id", mode="after")(lambda v: validate_identifier(v) if v is not None else v)
    _hash = field_validator("content_hash", mode="after")(validate_hash)
    _time = field_validator("effectivity_at", mode="after")(lambda v: validate_timestamp(v) if v is not None else v)

    def sort_key(self) -> tuple[str, str, str, str]:
        return (self.ref_type.value, self.ref_id, self.version_ref or "", self.content_hash)


def normalize_refs(refs: list[ContextRef]) -> list[ContextRef]:
    keyed = {ref.sort_key(): ref for ref in refs}
    return [keyed[key] for key in sorted(keyed)]


class EffectiveContextEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    event_id: str = Field(default_factory=uuid7)
    organization_id: str
    workspace_id: str
    session_id: str
    sequence: int = 0
    occurred_at: str = Field(default_factory=utc_now_millis)
    actor_ref: str
    event_type: EventType
    state_effect: StateEffect
    input_refs: list[ContextRef] = Field(default_factory=list)
    output_refs: list[ContextRef] = Field(default_factory=list)
    cause_event_ids: list[str] = Field(default_factory=list)
    authority_context_ref: str
    content_hash: str | None = None
    prior_event_hash: str | None = None

    _ids = field_validator("organization_id", "workspace_id", "session_id", "actor_ref", "authority_context_ref", mode="after")(validate_identifier)
    _event_id = field_validator("event_id", mode="after")(lambda v: v if UUIDV7_RE.match(v) else (_ for _ in ()).throw(ValueError("event_id must be UUIDv7")))
    _version = field_validator("schema_version", mode="after")(lambda v: v if SCHEMA_VERSION_RE.match(v) and v.split(".", 1)[0] == "1" else (_ for _ in ()).throw(ValueError("unsupported schema version")))
    _occurred = field_validator("occurred_at", mode="after")(validate_timestamp)
    _hash = field_validator("content_hash", "prior_event_hash", mode="after")(lambda v: validate_hash(v) if v is not None else v)

    @field_validator("sequence")
    @classmethod
    def valid_sequence(cls, value: int) -> int:
        if value < 0 or value >= 2**63:
            raise ValueError("sequence must be positive int64 when persisted")
        return value

    @field_validator("cause_event_ids")
    @classmethod
    def sorted_unique_causes(cls, value: list[str]) -> list[str]:
        return sorted(set(validate_identifier(v) for v in value))

    @model_validator(mode="after")
    def normalize(self) -> EffectiveContextEvent:
        self.input_refs = normalize_refs(self.input_refs)
        self.output_refs = normalize_refs(self.output_refs)
        return self

    def hash_payload(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data.pop("content_hash", None)
        return data

    def with_hash(self) -> EffectiveContextEvent:
        return self.model_copy(update={"content_hash": sha256_hex(self.hash_payload())})


class RequirementStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"
    PENDING_CLARIFICATION = "pending_clarification"
    UNRESOLVED_CONFLICT = "unresolved_conflict"


class ActiveRequirementItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    status: RequirementStatus
    statement_hash: str
    source_event_ids: list[str]
    effective_from_sequence: int
    effective_to_sequence: int | None = None
    supersedes_requirement_ids: list[str] = Field(default_factory=list)
    authority_role: Literal["binding", "interpretive", "contextual", "operational"]

    _id = field_validator("requirement_id", mode="after")(validate_identifier)
    _hash = field_validator("statement_hash", mode="after")(validate_hash)

    @field_validator("source_event_ids", "supersedes_requirement_ids")
    @classmethod
    def unique_sorted_ids(cls, value: list[str]) -> list[str]:
        return sorted(set(validate_identifier(v) for v in value))


class ActiveSemanticContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    items: list[ActiveRequirementItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_conflicting_statuses(self) -> ActiveSemanticContract:
        by_id: dict[str, set[RequirementStatus]] = {}
        for item in self.items:
            by_id.setdefault(item.requirement_id, set()).add(item.status)
        for req_id, statuses in by_id.items():
            if RequirementStatus.ACTIVE in statuses and RequirementStatus.SUPERSEDED in statuses:
                raise ValueError(f"{req_id} cannot be both active and superseded")
        return self


class AuthorityPins(BaseModel):
    """Immutable authority/version references captured at checkpoint time.

    Most references are optional because individual adapters do not expose every
    authority surface. The historical admission proof is mandatory for any
    checkpoint that can be restored.
    """

    model_config = ConfigDict(extra="forbid")

    historical_admission_proof_ref: str
    principal_ref: str | None = None
    organization_ref: str | None = None
    workspace_ref: str | None = None
    membership_ref: str | None = None
    entitlement_version_ref: str | None = None
    policy_version_refs: list[str] = Field(default_factory=list)
    ontology_version_refs: list[str] = Field(default_factory=list)
    skill_version_refs: list[str] = Field(default_factory=list)
    model_ref: str | None = None
    context_engine_version_ref: str | None = None
    memory_provider_refs: list[str] = Field(default_factory=list)
    data_snapshot_refs: list[str] = Field(default_factory=list)

    _ids = field_validator(
        "historical_admission_proof_ref",
        "principal_ref",
        "organization_ref",
        "workspace_ref",
        "membership_ref",
        "entitlement_version_ref",
        "model_ref",
        "context_engine_version_ref",
        mode="after",
    )(lambda v: validate_identifier(v) if v is not None else v)

    @field_validator(
        "policy_version_refs",
        "ontology_version_refs",
        "skill_version_refs",
        "memory_provider_refs",
        "data_snapshot_refs",
    )
    @classmethod
    def valid_ref_lists(cls, value: list[str]) -> list[str]:
        return sorted(set(validate_identifier(v) for v in value))


class EffectiveContextCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    checkpoint_id: str = Field(default_factory=uuid7)
    organization_id: str
    workspace_id: str
    session_id: str
    boundary_message_id: str
    boundary_event_sequence: int
    knowledge_state_ref: str
    active_contract_ref: str
    authority_pins_ref: str
    transcript_snapshot_ref: str
    trajectory_from_sequence: int
    trajectory_to_sequence: int
    parent_checkpoint_id: str | None = None
    lineage_root_checkpoint_id: str | None = None
    delta_ref: str | None = None
    checkpoint_hash: str | None = None
    verification_status: Literal["pending", "verified", "degraded", "failed"] = "pending"
    verification_reason: str | None = None
    created_at: str = Field(default_factory=utc_now_millis)

    _ids = field_validator("organization_id", "workspace_id", "session_id", "boundary_message_id", "knowledge_state_ref", "active_contract_ref", "authority_pins_ref", "transcript_snapshot_ref", mode="after")(validate_identifier)
    _checkpoint_id = field_validator("checkpoint_id", mode="after")(lambda v: v if UUIDV7_RE.match(v) else (_ for _ in ()).throw(ValueError("checkpoint_id must be UUIDv7")))
    _parent_ids = field_validator("parent_checkpoint_id", "lineage_root_checkpoint_id", mode="after")(lambda v: v if v is None or UUIDV7_RE.match(v) else (_ for _ in ()).throw(ValueError("checkpoint lineage ids must be UUIDv7")))
    _hash = field_validator("checkpoint_hash", mode="after")(lambda v: validate_hash(v) if v is not None else v)
    _created = field_validator("created_at", mode="after")(validate_timestamp)

    @model_validator(mode="after")
    def validate_ranges(self) -> EffectiveContextCheckpoint:
        if self.boundary_event_sequence <= 0:
            raise ValueError("boundary_event_sequence must be positive")
        if self.trajectory_from_sequence < 1:
            raise ValueError("trajectory_from_sequence must be >= 1")
        if self.trajectory_to_sequence != self.boundary_event_sequence:
            raise ValueError("trajectory_to_sequence must equal boundary_event_sequence")
        if self.trajectory_from_sequence > self.trajectory_to_sequence:
            raise ValueError("trajectory_from_sequence must be <= trajectory_to_sequence")
        if self.lineage_root_checkpoint_id is None:
            self.lineage_root_checkpoint_id = self.checkpoint_id
        return self

    def hash_payload(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data.pop("checkpoint_hash", None)
        data.pop("verification_status", None)
        data.pop("verification_reason", None)
        return data

    def with_hash(self) -> EffectiveContextCheckpoint:
        return self.model_copy(update={"checkpoint_hash": sha256_hex(self.hash_payload())})
