"""Append-only corpus observations with derived governed proposal candidates."""

from __future__ import annotations

import json
import sqlite3
from storage.hosted_postgres import enabled,is_postgres,require_tables,sql_query,integrity_errors,HostedStorageUnavailable
from storage.transaction_scope import transaction_scope

from contracts.corpus_growth import (
    CorpusGrowthCandidateV1,
    CorpusGrowthObservationV1,
    project_corpus_growth_candidate,
)


class CorpusGrowthStore:
    def __init__(self, connection: sqlite3.Connection, *, organization_id: str | None=None, workspace_id: str | None=None) -> None:
        if enabled() and not is_postgres(connection):raise HostedStorageUnavailable('hosted injected owner requires native PostgreSQL connection')
        self._connection = connection
        if is_postgres(connection):
            if not organization_id or not workspace_id:raise HostedStorageUnavailable('hosted corpus requires explicit governed workspace binding')
            self._scope=(organization_id,workspace_id)
            require_tables(connection,'corpus_growth_observations')
            return
        self._scope: tuple[str, str] | None = None
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_growth_observations (
              observation_ref TEXT PRIMARY KEY,
              observation_hash TEXT NOT NULL UNIQUE,
              organization_id TEXT NOT NULL,
              workspace_id TEXT NOT NULL,
              semantic_key TEXT NOT NULL,
              observed_at TEXT NOT NULL,
              payload_json TEXT NOT NULL
            )
            """
        )

    def append_observation(self, observation: CorpusGrowthObservationV1) -> None:
        observation = CorpusGrowthObservationV1.model_validate(
            observation.model_dump(mode="json")
        )
        incoming_scope = (observation.organization_id, observation.workspace_id)
        if self._scope is None:
            self._scope = incoming_scope
        elif self._scope != incoming_scope:
            raise ValueError("observation does not match store workspace scope")
        try:
            with transaction_scope(self._connection):
                self._connection.execute(
                    sql_query("INSERT INTO corpus_growth_observations VALUES (?, ?, ?, ?, ?, ?, ?)",'INSERT INTO corpus_growth_observations VALUES (%s, %s, %s, %s, %s, %s, %s)'),
                    (
                        observation.observation_ref,
                        observation.observation_hash,
                        observation.organization_id,
                        observation.workspace_id,
                        observation.semantic_key,
                        observation.observed_at,
                        observation.model_dump_json(),
                    ),
                )
        except integrity_errors() as exc:
            raise ValueError("corpus growth observation already exists") from exc

    def project_candidate(self, semantic_key: str) -> CorpusGrowthCandidateV1:
        if is_postgres(self._connection):
            rows=self._connection.execute("SELECT payload_json FROM corpus_growth_observations WHERE semantic_key=%s AND organization_id=%s AND workspace_id=%s ORDER BY observed_at ASC,observation_ref ASC",(semantic_key,*self._scope)).fetchall()
            return project_corpus_growth_candidate(tuple(CorpusGrowthObservationV1.model_validate(json.loads(row[0])) for row in rows))
        rows = self._connection.execute(
            sql_query("""
            SELECT payload_json FROM corpus_growth_observations
            WHERE semantic_key = ? ORDER BY observed_at ASC, observation_ref ASC
            """,'\n            SELECT payload_json FROM corpus_growth_observations\n            WHERE semantic_key = %s ORDER BY observed_at ASC, observation_ref ASC\n            '),
            (semantic_key,),
        ).fetchall()
        observations = tuple(
            CorpusGrowthObservationV1.model_validate(json.loads(row[0])) for row in rows
        )
        return project_corpus_growth_candidate(observations)


__all__ = ["CorpusGrowthStore"]
