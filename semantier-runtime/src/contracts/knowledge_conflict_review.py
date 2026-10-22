"""Typed conflict authority records and derived Review Queue projections."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.governed_graph import GraphConflictV1
from contracts.knowledge_temporal_state import parse_utc
from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ASCII_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")


class ConflictType(str, Enum):
    SOURCE_CONFLICT = "SOURCE_CONFLICT"
    TEMPORAL_CONFLICT = "TEMPORAL_CONFLICT"
    ENTITY_RESOLUTION_CONFLICT = "ENTITY_RESOLUTION_CONFLICT"
    RELATION_CONFLICT = "RELATION_CONFLICT"
    SCHEMA_CONFLICT = "SCHEMA_CONFLICT"
    ONTOLOGY_CONFLICT = "ONTOLOGY_CONFLICT"
    AUTHORITY_CONFLICT = "AUTHORITY_CONFLICT"
    JURISDICTION_CONFLICT = "JURISDICTION_CONFLICT"
    PROJECTION_CONFLICT = "PROJECTION_CONFLICT"
    RUNTIME_MATCH_CONFLICT = "RUNTIME_MATCH_CONFLICT"


_GRAPH_CONFLICT_MAP = {
    "precedence": ConflictType.AUTHORITY_CONFLICT,
    "temporal": ConflictType.TEMPORAL_CONFLICT,
    "jurisdictional": ConflictType.JURISDICTION_CONFLICT,
    "semantic": ConflictType.RELATION_CONFLICT,
    "duplicate": ConflictType.ENTITY_RESOLUTION_CONFLICT,
    "scope_ambiguity": ConflictType.SOURCE_CONFLICT,
}


def classify_graph_conflict(conflict: GraphConflictV1) -> ConflictType:
    return _GRAPH_CONFLICT_MAP[conflict.conflict_type]


def _canonical_refs(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a non-empty array")
    refs = tuple(value)
    if any(not isinstance(ref, str) or _ASCII_REF.fullmatch(ref) is None for ref in refs):
        raise ValueError(f"{field} must contain ASCII-stable references")
    if len(refs) != len(set(refs)):
        raise ValueError(f"{field} contains a duplicate reference")
    return tuple(sorted(refs, key=lambda ref: ref.encode("utf-8")))


class KnowledgeConflictV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["knowledge_conflict.v1"] = "knowledge_conflict.v1"
    conflict_ref: str
    conflict_hash: str = Field(pattern=SHA256_PATTERN)
    conflict_type: ConflictType
    object_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    detected_at: str
    detected_by: Literal["human", "agent", "system"]
    detector_ref: str
    required_authority: str | None
    proof_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)
    normative_set_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)
    context_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)
    status: Literal["open"] = "open"
    resolution_ref: None = None

    @field_validator("object_refs", mode="before")
    @classmethod
    def objects_are_canonical(cls, value: Any) -> tuple[str, ...]:
        return _canonical_refs(value, "object_refs")

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def evidence_is_canonical(cls, value: Any) -> tuple[str, ...]:
        return _canonical_refs(value, "evidence_refs")

    @model_validator(mode="after")
    def integrity(self) -> "KnowledgeConflictV1":
        parse_utc(self.detected_at, "detected_at")
        for field in (
            "detector_ref", "required_authority", "proof_ref", "normative_set_ref", "context_ref"
        ):
            value = getattr(self, field)
            if value is not None and _ASCII_REF.fullmatch(value) is None:
                raise ValueError(f"{field} must be ASCII-stable")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.conflict_hash != expected:
            raise ValueError("conflict hash does not match canonical content")
        if self.conflict_ref != "knowledge-conflict:" + expected[7:31]:
            raise ValueError("conflict ref does not match conflict hash")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("conflict_ref")
        payload.pop("conflict_hash")
        return payload


class KnowledgeConflictDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["knowledge_conflict_decision.v1"] = (
        "knowledge_conflict_decision.v1"
    )
    decision_ref: str
    decision_hash: str = Field(pattern=SHA256_PATTERN)
    conflict_ref: str
    conflict_hash: str = Field(pattern=SHA256_PATTERN)
    resolver: str
    resolution_authority: str
    outcome: Literal["resolved", "superseded"]
    rationale: str
    decided_at: str

    @model_validator(mode="after")
    def integrity(self) -> "KnowledgeConflictDecisionV1":
        parse_utc(self.decided_at, "decided_at")
        if not self.rationale.strip():
            raise ValueError("rationale is required")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.decision_hash != expected:
            raise ValueError("decision hash does not match canonical content")
        if self.decision_ref != "knowledge-conflict-decision:" + expected[7:31]:
            raise ValueError("decision ref does not match decision hash")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("decision_ref")
        payload.pop("decision_hash")
        return payload


class KnowledgeReviewItemV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["knowledge_review_item.v1"] = "knowledge_review_item.v1"
    review_ref: str
    object_kind: Literal["conflict"] = "conflict"
    object_ref: str
    reason_code: str
    severity: Literal["low", "medium", "high", "blocking"]
    evidence_refs: tuple[str, ...]
    recommended_actions: tuple[str, ...]
    required_authority: str | None
    authority_source_ref: str
    state: Literal["pending", "resolved", "superseded"]
    resolution_ref: str | None


def create_knowledge_conflict(**values: Any) -> KnowledgeConflictV1:
    payload = {
        "schema_version": "knowledge_conflict.v1",
        **values,
        "status": "open",
        "resolution_ref": None,
    }
    payload["object_refs"] = _canonical_refs(payload["object_refs"], "object_refs")
    payload["evidence_refs"] = _canonical_refs(
        payload["evidence_refs"], "evidence_refs"
    )
    conflict_hash = "sha256:" + content_hash(payload)
    return KnowledgeConflictV1.model_validate(
        {
            **payload,
            "conflict_ref": "knowledge-conflict:" + conflict_hash[7:31],
            "conflict_hash": conflict_hash,
        }
    )


def create_conflict_decision(**values: Any) -> KnowledgeConflictDecisionV1:
    payload = {"schema_version": "knowledge_conflict_decision.v1", **values}
    decision_hash = "sha256:" + content_hash(payload)
    return KnowledgeConflictDecisionV1.model_validate(
        {
            **payload,
            "decision_ref": "knowledge-conflict-decision:" + decision_hash[7:31],
            "decision_hash": decision_hash,
        }
    )


def project_conflict_review_item(
    conflict: KnowledgeConflictV1,
    decision: KnowledgeConflictDecisionV1 | None,
) -> KnowledgeReviewItemV1:
    state = decision.outcome if decision else "pending"
    return KnowledgeReviewItemV1(
        review_ref="knowledge-review:" + conflict.conflict_hash[7:31],
        object_ref=conflict.conflict_ref,
        reason_code="conflict_detected",
        severity="blocking"
        if conflict.conflict_type
        in {ConflictType.AUTHORITY_CONFLICT, ConflictType.JURISDICTION_CONFLICT}
        else "high",
        evidence_refs=conflict.evidence_refs,
        recommended_actions=("OPEN_CONFLICT", "PROPOSE_CONFLICT_RESOLUTION"),
        required_authority=conflict.required_authority,
        authority_source_ref=conflict.conflict_ref,
        state=state,
        resolution_ref=decision.decision_ref if decision else None,
    )


__all__ = [
    "ConflictType",
    "KnowledgeConflictDecisionV1",
    "KnowledgeConflictV1",
    "KnowledgeReviewItemV1",
    "classify_graph_conflict",
    "create_conflict_decision",
    "create_knowledge_conflict",
    "project_conflict_review_item",
]
