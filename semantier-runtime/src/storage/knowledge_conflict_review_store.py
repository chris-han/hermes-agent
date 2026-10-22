"""Canonical conflict events with a derived, non-authoritative Review Queue."""

from __future__ import annotations

import json
import sqlite3
from storage.hosted_postgres import enabled,is_postgres,require_tables,sql_query,integrity_errors,HostedStorageUnavailable
from storage.transaction_scope import transaction_scope

from contracts.knowledge_conflict_review import (
    KnowledgeConflictDecisionV1,
    KnowledgeConflictV1,
    KnowledgeReviewItemV1,
    project_conflict_review_item,
)


class KnowledgeConflictReviewStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        if enabled() and not is_postgres(connection):raise HostedStorageUnavailable('hosted injected owner requires native PostgreSQL connection')
        self._connection = connection
        if is_postgres(connection):
            require_tables(connection,'knowledge_conflicts','knowledge_conflict_decisions')
            return
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_conflicts (
              conflict_ref TEXT PRIMARY KEY,
              conflict_hash TEXT NOT NULL UNIQUE,
              payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS knowledge_conflict_decisions (
              decision_ref TEXT PRIMARY KEY,
              decision_hash TEXT NOT NULL UNIQUE,
              conflict_ref TEXT NOT NULL UNIQUE,
              payload_json TEXT NOT NULL,
              FOREIGN KEY (conflict_ref) REFERENCES knowledge_conflicts(conflict_ref)
            );
            """
        )

    def append_conflict(self, conflict: KnowledgeConflictV1) -> None:
        conflict = KnowledgeConflictV1.model_validate(conflict.model_dump(mode="json"))
        try:
            with transaction_scope(self._connection):
                self._connection.execute(
                    sql_query("INSERT INTO knowledge_conflicts VALUES (?, ?, ?)",'INSERT INTO knowledge_conflicts VALUES (%s, %s, %s)'),
                    (
                        conflict.conflict_ref,
                        conflict.conflict_hash,
                        conflict.model_dump_json(),
                    ),
                )
        except integrity_errors() as exc:
            raise ValueError("conflict already exists") from exc

    def append_decision(self, decision: KnowledgeConflictDecisionV1) -> None:
        decision = KnowledgeConflictDecisionV1.model_validate(
            decision.model_dump(mode="json")
        )
        conflict = self._get_conflict(decision.conflict_ref)
        if conflict is None or (
            conflict.conflict_ref,
            conflict.conflict_hash,
        ) != (decision.conflict_ref, decision.conflict_hash):
            raise ValueError("decision must bind the exact conflict pin")
        try:
            with transaction_scope(self._connection):
                self._connection.execute(
                    sql_query("INSERT INTO knowledge_conflict_decisions VALUES (?, ?, ?, ?)",'INSERT INTO knowledge_conflict_decisions VALUES (%s, %s, %s, %s)'),
                    (
                        decision.decision_ref,
                        decision.decision_hash,
                        decision.conflict_ref,
                        decision.model_dump_json(),
                    ),
                )
        except integrity_errors() as exc:
            raise ValueError("conflict already has a decision") from exc

    def project_review_item(self, conflict_ref: str) -> KnowledgeReviewItemV1:
        conflict = self._get_conflict(conflict_ref)
        if conflict is None:
            raise KeyError(conflict_ref)
        row = self._connection.execute(
            sql_query("SELECT payload_json FROM knowledge_conflict_decisions WHERE conflict_ref = ?",'SELECT payload_json FROM knowledge_conflict_decisions WHERE conflict_ref = %s'),
            (conflict_ref,),
        ).fetchone()
        decision = (
            KnowledgeConflictDecisionV1.model_validate(json.loads(row[0]))
            if row
            else None
        )
        return project_conflict_review_item(conflict, decision)

    def _get_conflict(self, conflict_ref: str) -> KnowledgeConflictV1 | None:
        row = self._connection.execute(
            sql_query("SELECT payload_json FROM knowledge_conflicts WHERE conflict_ref = ?",'SELECT payload_json FROM knowledge_conflicts WHERE conflict_ref = %s'),
            (conflict_ref,),
        ).fetchone()
        return KnowledgeConflictV1.model_validate(json.loads(row[0])) if row else None


__all__ = ["KnowledgeConflictReviewStore"]
