"""Append-only SQLite persistence and offline traversal for derivations."""

from __future__ import annotations

import json
import sqlite3
from storage.hosted_postgres import enabled,is_postgres,require_tables,sql_query,integrity_errors,HostedStorageUnavailable
from storage.transaction_scope import transaction_scope,StoreMutationLock

from contracts.knowledge_derivation import DerivationRecordV1


class KnowledgeDerivationStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._lock=StoreMutationLock(connection)
        if enabled() and not is_postgres(connection):
            raise HostedStorageUnavailable('hosted knowledge store requires a native PostgreSQL connection')
        if is_postgres(connection):
            require_tables(connection,'knowledge_derivations')
            return
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_derivations (
              derivation_ref TEXT PRIMARY KEY,
              derivation_hash TEXT NOT NULL UNIQUE,
              output_ref TEXT NOT NULL UNIQUE,
              payload_json TEXT NOT NULL
            )
            """
        )

    def append(self, record: DerivationRecordV1) -> None:
        record = DerivationRecordV1.model_validate(
            record.model_dump(mode="json", by_alias=True)
        )
        try:
            with self._lock, transaction_scope(self._connection):
                self._connection.execute(
                    sql_query("""
                    INSERT INTO knowledge_derivations (
                      derivation_ref, derivation_hash, output_ref, payload_json
                    ) VALUES (?, ?, ?, ?)
                    """,'\n                    INSERT INTO knowledge_derivations (\n                      derivation_ref, derivation_hash, output_ref, payload_json\n                    ) VALUES (%s, %s, %s, %s)\n                    '),
                    (
                        record.derivation_ref,
                        record.derivation_hash,
                        record.output_ref,
                        record.canonical_json(),
                    ),
                )
        except integrity_errors() as exc:
            raise ValueError("output already has a derivation") from exc

    def get_by_output(self, output_ref: str) -> DerivationRecordV1 | None:
        row = self._connection.execute(
            sql_query("SELECT payload_json FROM knowledge_derivations WHERE output_ref = ?",'SELECT payload_json FROM knowledge_derivations WHERE output_ref = %s'),
            (output_ref,),
        ).fetchone()
        if row is None:
            return None
        return DerivationRecordV1.model_validate(json.loads(row[0]))

    def get_by_ref(self, derivation_ref: str) -> DerivationRecordV1 | None:
        row = self._connection.execute(
            sql_query("SELECT payload_json FROM knowledge_derivations WHERE derivation_ref = ?",'SELECT payload_json FROM knowledge_derivations WHERE derivation_ref = %s'),
            (derivation_ref,),
        ).fetchone()
        if row is None:
            return None
        return DerivationRecordV1.model_validate(json.loads(row[0]))

    def trace_output(self, output_ref: str) -> tuple[DerivationRecordV1, ...]:
        ordered: list[DerivationRecordV1] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(ref: str) -> None:
            record = self.get_by_output(ref)
            if record is None or record.derivation_ref in visited:
                return
            if record.derivation_ref in visiting:
                raise ValueError("derivation cycle detected")
            visiting.add(record.derivation_ref)
            ordered.append(record)
            for premise_ref in record.premise_refs:
                visit(premise_ref)
            visiting.remove(record.derivation_ref)
            visited.add(record.derivation_ref)

        visit(output_ref)
        return tuple(ordered)


__all__ = ["KnowledgeDerivationStore"]
