"""Deterministic read-only Decision Ledger projections over canonical events."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.knowledge_conflict_review import (
    KnowledgeConflictDecisionV1,
    KnowledgeConflictV1,
)
from contracts.knowledge_temporal_state import RelationTemporalStateV1, parse_utc
from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class DecisionLedgerEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["decision_ledger_entry.v1"] = "decision_ledger_entry.v1"
    entry_ref: str
    entry_hash: str = Field(pattern=SHA256_PATTERN)
    source_event_ref: str
    source_event_hash: str = Field(pattern=SHA256_PATTERN)
    event_type: Literal["relation_temporal_admitted", "conflict_resolved"]
    occurred_at: str
    actor_ref: str
    action: str
    object_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    rationale: str | None

    @model_validator(mode="after")
    def integrity(self) -> "DecisionLedgerEntryV1":
        parse_utc(self.occurred_at, "occurred_at")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.entry_hash != expected:
            raise ValueError("ledger entry hash does not match canonical projection")
        if self.entry_ref != "decision-ledger-entry:" + expected[7:31]:
            raise ValueError("ledger entry ref does not match entry hash")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("entry_ref")
        payload.pop("entry_hash")
        return payload


def _entry(**values: Any) -> DecisionLedgerEntryV1:
    payload = {"schema_version": "decision_ledger_entry.v1", **values}
    payload["object_refs"] = tuple(
        sorted(set(payload["object_refs"]), key=lambda ref: ref.encode("utf-8"))
    )
    payload["evidence_refs"] = tuple(
        sorted(set(payload["evidence_refs"]), key=lambda ref: ref.encode("utf-8"))
    )
    entry_hash = "sha256:" + content_hash(payload)
    return DecisionLedgerEntryV1.model_validate(
        {
            **payload,
            "entry_ref": "decision-ledger-entry:" + entry_hash[7:31],
            "entry_hash": entry_hash,
        }
    )


class DecisionLedgerProjector:
    """Pure projector; it intentionally exposes no mutation or persistence API."""

    def project(
        self,
        *,
        conflict_decisions: list[
            tuple[KnowledgeConflictV1, KnowledgeConflictDecisionV1]
        ]
        | tuple[tuple[KnowledgeConflictV1, KnowledgeConflictDecisionV1], ...] = (),
        temporal_states: list[RelationTemporalStateV1]
        | tuple[RelationTemporalStateV1, ...] = (),
        object_ref: str | None = None,
    ) -> tuple[DecisionLedgerEntryV1, ...]:
        entries: list[DecisionLedgerEntryV1] = []
        for state in temporal_states:
            state = RelationTemporalStateV1.model_validate(state.model_dump(mode="json"))
            entries.append(
                _entry(
                    source_event_ref=state.state_ref,
                    source_event_hash=state.state_hash,
                    event_type="relation_temporal_admitted",
                    occurred_at=state.admitted_at,
                    actor_ref="system:temporal-admission",
                    action="admit_relation_temporal_state",
                    object_refs=(state.relation_instance_ref,),
                    evidence_refs=(),
                    rationale=None,
                )
            )
        for conflict, decision in conflict_decisions:
            conflict = KnowledgeConflictV1.model_validate(conflict.model_dump(mode="json"))
            decision = KnowledgeConflictDecisionV1.model_validate(
                decision.model_dump(mode="json")
            )
            if (decision.conflict_ref, decision.conflict_hash) != (
                conflict.conflict_ref,
                conflict.conflict_hash,
            ):
                raise ValueError("decision must bind the exact conflict pin")
            entries.append(
                _entry(
                    source_event_ref=decision.decision_ref,
                    source_event_hash=decision.decision_hash,
                    event_type="conflict_resolved",
                    occurred_at=decision.decided_at,
                    actor_ref=decision.resolution_authority,
                    action=decision.resolver,
                    object_refs=conflict.object_refs,
                    evidence_refs=conflict.evidence_refs,
                    rationale=decision.rationale,
                )
            )
        if object_ref is not None:
            entries = [entry for entry in entries if object_ref in entry.object_refs]
        return tuple(
            sorted(
                entries,
                key=lambda entry: (
                    parse_utc(entry.occurred_at, "occurred_at"),
                    entry.entry_ref.encode("utf-8"),
                ),
            )
        )


__all__ = ["DecisionLedgerEntryV1", "DecisionLedgerProjector"]
