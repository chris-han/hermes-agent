"""SQLite-backed append-only relation temporal-state history."""

from __future__ import annotations

import json
import sqlite3
from storage.hosted_postgres import enabled,is_postgres,require_tables,sql_query,integrity_errors,HostedStorageUnavailable
from storage.transaction_scope import transaction_scope,StoreMutationLock

from contracts.knowledge_temporal_state import RelationTemporalStateV1, parse_utc


class KnowledgeTemporalStateStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._lock=StoreMutationLock(connection)
        if enabled() and not is_postgres(connection):
            raise HostedStorageUnavailable('hosted knowledge store requires a native PostgreSQL connection')
        if is_postgres(connection):
            require_tables(connection,'relation_temporal_states')
            return
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS relation_temporal_states (
              state_ref TEXT PRIMARY KEY,
              state_hash TEXT NOT NULL UNIQUE,
              relation_instance_ref TEXT NOT NULL,
              relation_instance_hash TEXT NOT NULL,
              admitted_at TEXT NOT NULL,
              supersedes_state_ref TEXT,
              payload_json TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_relation_temporal_states_history
            ON relation_temporal_states (relation_instance_ref, admitted_at, state_ref)
            """
        )

    def append(self, state: RelationTemporalStateV1) -> None:
        state = RelationTemporalStateV1.model_validate(
            state.model_dump(mode="json")
        )
        with self._lock:
            current = self._head(state.relation_instance_ref)
            if current is None:
                if state.supersedes_state_ref is not None:
                    raise ValueError("first temporal state cannot supersede another state")
            else:
                if (
                    state.supersedes_state_ref,
                    state.supersedes_state_hash,
                ) != (current.state_ref, current.state_hash):
                    raise ValueError("successor must supersede the current temporal head")
                if state.relation_instance_hash != current.relation_instance_hash:
                    raise ValueError("temporal successor cannot change relation identity")
                if parse_utc(state.admitted_at, "admitted_at") <= parse_utc(
                    current.admitted_at, "admitted_at"
                ):
                    raise ValueError("successor admitted_at must advance knowledge time")
            try:
                with transaction_scope(self._connection):
                    self._connection.execute(
                        sql_query("""
                        INSERT INTO relation_temporal_states (
                          state_ref, state_hash, relation_instance_ref,
                          relation_instance_hash, admitted_at,
                          supersedes_state_ref, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,'\n                    INSERT INTO relation_temporal_states (\n                      state_ref, state_hash, relation_instance_ref,\n                      relation_instance_hash, admitted_at,\n                      supersedes_state_ref, payload_json\n                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)\n                    '),
                        (
                            state.state_ref,
                            state.state_hash,
                            state.relation_instance_ref,
                            state.relation_instance_hash,
                            state.admitted_at,
                            state.supersedes_state_ref,
                            state.canonical_json(),
                        ),
                    )
            except integrity_errors() as exc:
                raise ValueError("temporal state already exists") from exc

    def history(self, relation_instance_ref: str) -> tuple[RelationTemporalStateV1, ...]:
        rows = self._connection.execute(
            sql_query("""
            SELECT payload_json FROM relation_temporal_states
            WHERE relation_instance_ref = ?
            ORDER BY admitted_at ASC, state_ref ASC
            """,'\n            SELECT payload_json FROM relation_temporal_states\n            WHERE relation_instance_ref = %s\n            ORDER BY admitted_at ASC, state_ref ASC\n            '),
            (relation_instance_ref,),
        ).fetchall()
        return tuple(
            RelationTemporalStateV1.model_validate(json.loads(row[0])) for row in rows
        )

    def as_of(
        self,
        relation_instance_ref: str,
        *,
        valid_at: str,
        known_at: str,
    ) -> RelationTemporalStateV1 | None:
        valid_time = parse_utc(valid_at, "valid_at")
        knowledge_time = parse_utc(known_at, "known_at")
        candidates = [
            state
            for state in self.history(relation_instance_ref)
            if parse_utc(state.admitted_at, "admitted_at") <= knowledge_time
        ]
        if not candidates:
            return None
        state = candidates[-1]
        lower = parse_utc(state.valid_from, "valid_from") if state.valid_from else None
        upper = parse_utc(state.valid_to, "valid_to") if state.valid_to else None
        if lower is not None and valid_time < lower:
            return None
        if upper is not None and valid_time >= upper:
            return None
        return state

    def _head(self, relation_instance_ref: str) -> RelationTemporalStateV1 | None:
        history = self.history(relation_instance_ref)
        return history[-1] if history else None


__all__ = ["KnowledgeTemporalStateStore"]
