"""Append-only Phase 1 knowledge feature-pack persistence in eos.db."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contracts.knowledge_feature_pack import (
    EvaluatorConfiguration,
    FeedbackDatasetRelease,
    HistoricalDecisionContext,
    KnowledgeFeaturePackManifest,
    LearningEventEnvelope,
    LearningEventType,
    content_hash,
    utc_iso,
)
from storage.knowledge_builder_evaluation_store import KnowledgeBuilderEvaluationStore
from storage.hosted_postgres import enabled,require_tables,sql_query
from storage.schema_guard import mark_schema_current, schema_is_current


_SCHEMA_NAME = "knowledge_feature_pack_store"
_SCHEMA_VERSION = 1
_SCHEMA_SENTINELS = frozenset(
    {
        "knowledge_feature_pack_manifests",
        "knowledge_learning_events",
        "knowledge_feedback_dataset_releases",
        "knowledge_evaluator_configurations",
    }
)


class KnowledgeFeaturePackStore(KnowledgeBuilderEvaluationStore):
    """Append-only Phase 1 extension of KnowledgeBuilderEvaluationStore."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        super().__init__(db_path=self._db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        if enabled():
            with self._lock:require_tables(self._conn,*_SCHEMA_SENTINELS)
            return
        with self._lock:
            if schema_is_current(
                self._conn,
                schema_name=_SCHEMA_NAME,
                schema_version=_SCHEMA_VERSION,
                sentinels=_SCHEMA_SENTINELS,
            ):
                return
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_feature_pack_manifests (
                  pack_id TEXT NOT NULL,
                  version TEXT NOT NULL,
                  manifest_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  PRIMARY KEY (pack_id, version)
                );
                CREATE TABLE IF NOT EXISTS knowledge_learning_events (
                  event_id TEXT PRIMARY KEY,
                  case_id TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  event_json TEXT NOT NULL,
                  prior_event_hash TEXT,
                  event_hash TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_learning_events_case
                  ON knowledge_learning_events(case_id, occurred_at, event_id);
                CREATE TABLE IF NOT EXISTS knowledge_feedback_dataset_releases (
                  dataset_release_id TEXT PRIMARY KEY,
                  dataset_key TEXT NOT NULL,
                  release_json TEXT NOT NULL,
                  source_event_manifest_hash TEXT NOT NULL,
                  content_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_evaluator_configurations (
                  evaluator_key TEXT NOT NULL,
                  version TEXT NOT NULL,
                  configuration_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  PRIMARY KEY (evaluator_key, version)
                );
                """
            )
            mark_schema_current(
                self._conn,
                schema_name=_SCHEMA_NAME,
                schema_version=_SCHEMA_VERSION,
            )
            self._conn.commit()

    def put_manifest(self, manifest: KnowledgeFeaturePackManifest) -> dict[str, Any]:
        payload = manifest.model_dump(mode="json")
        digest = content_hash(payload)
        self._insert_immutable(
            table="knowledge_feature_pack_manifests",
            identity_where="pack_id = ? AND version = ?",
            identity=(manifest.pack_id, manifest.version),
            insert_sql="""
                INSERT INTO knowledge_feature_pack_manifests
                  (pack_id, version, manifest_json, content_hash)
                VALUES (?, ?, ?, ?)
            """,
            values=(manifest.pack_id, manifest.version, _json(payload), digest),
            expected_hash=digest,

            postgres_select_sql='SELECT content_hash FROM knowledge_feature_pack_manifests WHERE pack_id = %s AND version = %s',
            postgres_insert_sql='\n                INSERT INTO knowledge_feature_pack_manifests\n                  (pack_id, version, manifest_json, content_hash)\n                VALUES (%s, %s, %s, %s)\n            ',
        )
        return {**payload, "content_hash": digest}

    def append_event(self, event: LearningEventEnvelope) -> dict[str, Any]:
        result = self._append_event_uncommitted(event)
        with self._lock:
            self._conn.commit()
        return result

    def _append_event_uncommitted(self, event: LearningEventEnvelope) -> dict[str, Any]:
        payload = event.model_dump(mode="json")
        digest = event.event_hash()
        with self._lock:
            if self._conn.execute(
                sql_query("SELECT 1 FROM knowledge_learning_events WHERE event_id = ?",'SELECT 1 FROM knowledge_learning_events WHERE event_id = %s'), (event.event_id,)
            ).fetchone():
                raise ValueError(f"learning event already exists: {event.event_id}")
            prior = self._conn.execute(
                sql_query("""
                SELECT event_hash FROM knowledge_learning_events
                WHERE case_id = ? ORDER BY occurred_at DESC, event_id DESC LIMIT 1
                """,'\n                SELECT event_hash FROM knowledge_learning_events\n                WHERE case_id = %s ORDER BY occurred_at DESC, event_id DESC LIMIT 1\n                '),
                (event.case_id,),
            ).fetchone()
            expected_prior = prior["event_hash"] if prior else None
            if event.prior_event_hash != expected_prior:
                raise ValueError("learning event prior_event_hash does not match case history")
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_learning_events
                  (event_id, case_id, event_type, occurred_at, event_json,
                   prior_event_hash, event_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO knowledge_learning_events\n                  (event_id, case_id, event_type, occurred_at, event_json,\n                   prior_event_hash, event_hash)\n                VALUES (%s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    event.event_id,
                    event.case_id,
                    event.event_type.value,
                    utc_iso(event.occurred_at),
                    _json(payload),
                    event.prior_event_hash,
                    digest,
                ),
            )
        return {**payload, "event_hash": digest}

    def list_events(self, case_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            sql_query("""
            SELECT event_json, event_hash FROM knowledge_learning_events
            WHERE case_id = ? ORDER BY occurred_at, event_id
            """,'\n            SELECT event_json, event_hash FROM knowledge_learning_events\n            WHERE case_id = %s ORDER BY occurred_at, event_id\n            '),
            (case_id,),
        ).fetchall()
        return [{**json.loads(row["event_json"]), "event_hash": row["event_hash"]} for row in rows]

    def put_dataset_release(self, release: FeedbackDatasetRelease) -> dict[str, Any]:
        payload = release.model_dump(mode="json")
        release_id = release.release_id()
        digest = release.release_hash()
        manifest_hash = release.source_event_manifest_hash()
        self._insert_immutable(
            table="knowledge_feedback_dataset_releases",
            identity_where="dataset_release_id = ?",
            identity=(release_id,),
            insert_sql="""
                INSERT INTO knowledge_feedback_dataset_releases
                  (dataset_release_id, dataset_key, release_json,
                   source_event_manifest_hash, content_hash)
                VALUES (?, ?, ?, ?, ?)
            """,
            values=(release_id, release.dataset_key, _json(payload), manifest_hash, digest),
            expected_hash=digest,

            postgres_select_sql='SELECT content_hash FROM knowledge_feedback_dataset_releases WHERE dataset_release_id = %s',
            postgres_insert_sql='\n                INSERT INTO knowledge_feedback_dataset_releases\n                  (dataset_release_id, dataset_key, release_json,\n                   source_event_manifest_hash, content_hash)\n                VALUES (%s, %s, %s, %s, %s)\n            ',
        )
        return {
            **payload,
            "dataset_release_id": release_id,
            "source_event_manifest_hash": manifest_hash,
            "content_hash": digest,
        }

    def put_evaluator_configuration(self, config: EvaluatorConfiguration) -> dict[str, Any]:
        payload = config.model_dump(mode="json")
        digest = config.configuration_hash()
        self._insert_immutable(
            table="knowledge_evaluator_configurations",
            identity_where="evaluator_key = ? AND version = ?",
            identity=(config.evaluator_key, config.version),
            insert_sql="""
                INSERT INTO knowledge_evaluator_configurations
                  (evaluator_key, version, configuration_json, content_hash)
                VALUES (?, ?, ?, ?)
            """,
            values=(config.evaluator_key, config.version, _json(payload), digest),
            expected_hash=digest,

            postgres_select_sql='SELECT content_hash FROM knowledge_evaluator_configurations WHERE evaluator_key = %s AND version = %s',
            postgres_insert_sql='\n                INSERT INTO knowledge_evaluator_configurations\n                  (evaluator_key, version, configuration_json, content_hash)\n                VALUES (%s, %s, %s, %s)\n            ',
        )
        return {**payload, "content_hash": digest}

    def reconstruct_decision_context(self, case_id: str) -> HistoricalDecisionContext:
        events = self.list_events(case_id)
        if not events:
            raise ValueError(f"learning case not found: {case_id}")
        machine = next(
            (event.get("machine_proposal") for event in events if event.get("machine_proposal")),
            None,
        )
        human = next(
            (event.get("human_label") for event in reversed(events) if event.get("human_label")),
            None,
        )
        graph_delta_ref = next(
            (event.get("graph_delta_ref") for event in reversed(events) if event.get("graph_delta_ref")),
            None,
        )
        first = events[0]
        decision_event = next(
            (
                event
                for event in reversed(events)
                if event["event_type"]
                in {
                    LearningEventType.HUMAN_ACCEPT.value,
                    LearningEventType.HUMAN_EDIT.value,
                    LearningEventType.HUMAN_REJECT.value,
                    LearningEventType.HUMAN_UNCERTAIN.value,
                }
            ),
            first,
        )
        return HistoricalDecisionContext.model_validate(
            {
                "case_id": case_id,
                "source_document_hash": first["source_document_hash"],
                "source_anchor_refs": first["source_anchor_refs"],
                "machine_proposal": machine,
                "graph_version": decision_event["graph_version"],
                "rule_version": decision_event["rule_version"],
                "human_label": human,
                "graph_delta_ref": graph_delta_ref,
                "evaluation_corpus_version": decision_event["evaluation_corpus_version"],
                "event_ids": [event["event_id"] for event in events],
                "event_hashes": [event["event_hash"] for event in events],
            }
        )

    def _insert_immutable(
        self,
        *,
        table: str,
        identity_where: str,
        identity: tuple[Any, ...],
        insert_sql: str,
        values: tuple[Any, ...],
        expected_hash: str,
        postgres_select_sql: str | None = None,
        postgres_insert_sql: str | None = None,
    ) -> None:
        if enabled():
            if not postgres_select_sql or not postgres_insert_sql:
                raise ValueError('explicit PostgreSQL immutable owner statements required')
            query=postgres_select_sql
            insert_sql=postgres_insert_sql
        else:
            query=f"SELECT content_hash FROM {table} WHERE {identity_where}"
        with self._lock:
            row = self._conn.execute(query,identity).fetchone()
            if row:
                if row["content_hash"] != expected_hash:
                    raise ValueError(f"immutable {table} identity already has different content")
                return
            self._conn.execute(insert_sql, values)
            self._conn.commit()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
