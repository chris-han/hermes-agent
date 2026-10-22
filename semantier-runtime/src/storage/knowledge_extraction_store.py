"""Normalized, additive Phase 1 extraction/assertion/package persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from contracts.knowledge_feature_pack import content_hash
from contracts.knowledge_reasoning_phase1 import (
    DocumentCandidateAssertion,
    EvidenceObligation,
    GroundingState,
    KnowledgeScope,
    SemanticOccurrence,
)
from contracts.knowledge_graph_phase1 import SensitiveTermCandidateGraph
from contracts.contextgraph_reference import (
    CanonicalExpression,
    ObservedExpression,
    ObservedExpressionObservation,
    ReferenceConcept,
)
from contracts.accepted_graph_release_lineage import AcceptedGraphReleaseLineage
from storage.accepted_graph_release_origin import record_accepted_release_origin
from storage.knowledge_feature_pack_store import KnowledgeFeaturePackStore
from storage.hosted_postgres import enabled,require_tables,sql_query,begin_write
from storage.schema_guard import mark_schema_current, schema_is_current


_SCHEMA_NAME = "knowledge_extraction_store"
_SCHEMA_VERSION = 2
_SCHEMA_SENTINELS = frozenset(
    {
        "knowledge_builder_extraction_runs",
        "contextgraph_artifact_origins",
        "contextgraph_observed_expressions",
        "lineage_materialization_retirement_events",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class KnowledgeExtractionStore(KnowledgeFeaturePackStore):
    def __init__(self, db_path: str | Path) -> None:
        super().__init__(db_path=db_path)
        self._init_extraction_schema()

    def _init_extraction_schema(self) -> None:
        if enabled():
            with self._lock:require_tables(self._conn,'artifact_lifecycle_events','contextgraph_artifact_origins','contextgraph_canonical_expressions','contextgraph_observed_expression_observations','contextgraph_observed_expressions','contextgraph_reference_concepts','knowledge_builder_ai_grounding_results','knowledge_builder_ai_grounding_runs','knowledge_builder_ai_grounding_suggestions','knowledge_builder_assertion_grounding_bindings','knowledge_builder_assertion_occurrences','knowledge_builder_document_candidate_assertions','knowledge_builder_evidence_obligation_anchors','knowledge_builder_evidence_obligation_resolution_events','knowledge_builder_evidence_obligations','knowledge_builder_extraction_runs','knowledge_builder_grounding_batch_items','knowledge_builder_grounding_batches','knowledge_builder_knowledge_scopes','knowledge_builder_semantic_occurrences','lineage_materialization_retirement_events','lineage_repair_events','sensitive_term_candidate_graphs','source_lifecycle_events','source_representation_edges','source_representation_lifecycle_events','source_retirement_operations','tender_graph_release_materialization_events','tender_graph_releases')
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
                CREATE TABLE IF NOT EXISTS knowledge_builder_extraction_runs (
                  extraction_run_id TEXT PRIMARY KEY,
                  discovery_run_id TEXT NOT NULL,
                  source_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  package_id TEXT,
                  document_id TEXT NOT NULL,
                  source_role TEXT NOT NULL DEFAULT 'legacy_unknown',
                  workflow_kind TEXT NOT NULL DEFAULT 'legacy_candidate_assertion',
                  provider TEXT NOT NULL,
                  provider_version TEXT NOT NULL,
                  provider_ref TEXT,
                  provider_commit TEXT,
                  profile_ref TEXT NOT NULL,
                  source_identity_ref TEXT,
                  parser_trace_id TEXT,
                  run_status TEXT NOT NULL,
                  raw_output_ref TEXT,
                  raw_output_hash TEXT,
                  native_artifact_ref TEXT,
                  native_artifact_hash TEXT,
                  failure_reason TEXT,
                  warnings_json TEXT NOT NULL,
                  boundary_validation_json TEXT NOT NULL DEFAULT '[]',
                  boundary_review_required INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT,
                  started_at TEXT NOT NULL,
                  completed_at TEXT,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(discovery_run_id) REFERENCES knowledge_builder_discovery_runs(discovery_run_id),
                  FOREIGN KEY(source_id) REFERENCES knowledge_builder_sources(source_id)
                );
                CREATE INDEX IF NOT EXISTS idx_kber_org ON knowledge_builder_extraction_runs(organization_id, started_at);
                CREATE TABLE IF NOT EXISTS knowledge_builder_semantic_occurrences (
                  occurrence_id TEXT PRIMARY KEY,
                  extraction_run_id TEXT NOT NULL,
                  document_id TEXT NOT NULL,
                  source_anchor_json TEXT NOT NULL,
                  extraction_class TEXT NOT NULL,
                  extraction_attributes_json TEXT NOT NULL,
                  alignment_json TEXT NOT NULL,
                  machine_confidence REAL,
                  model_ref TEXT NOT NULL,
                  prompt_ref TEXT NOT NULL,
                  raw_output_ref TEXT NOT NULL,
                  occurrence_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(extraction_run_id) REFERENCES knowledge_builder_extraction_runs(extraction_run_id)
                );
                CREATE TABLE IF NOT EXISTS knowledge_builder_document_candidate_assertions (
                  assertion_id TEXT PRIMARY KEY,
                  extraction_run_id TEXT NOT NULL,
                  candidate_graph_id TEXT NOT NULL,
                  source_document_hash TEXT NOT NULL,
                  assertion_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(extraction_run_id) REFERENCES knowledge_builder_extraction_runs(extraction_run_id)
                );
                CREATE TABLE IF NOT EXISTS knowledge_builder_assertion_grounding_bindings (
                  binding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  assertion_id TEXT NOT NULL,
                  human_label_event_id TEXT NOT NULL,
                  graph_version TEXT NOT NULL,
                  graph_release_hash TEXT NOT NULL,
                  accepted_node_ids_json TEXT NOT NULL,
                  accepted_edge_ids_json TEXT NOT NULL,
                  accepted_rule_ids_json TEXT NOT NULL,
                  materialized_at TEXT NOT NULL,
                  FOREIGN KEY(assertion_id) REFERENCES knowledge_builder_document_candidate_assertions(assertion_id),
                  FOREIGN KEY(graph_version) REFERENCES tender_graph_releases(graph_version),
                  FOREIGN KEY(human_label_event_id) REFERENCES knowledge_learning_events(event_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_kbagb_event_release
                  ON knowledge_builder_assertion_grounding_bindings(human_label_event_id, graph_version);
                CREATE INDEX IF NOT EXISTS idx_kbagb_assertion_latest
                  ON knowledge_builder_assertion_grounding_bindings(assertion_id, binding_id);
                CREATE TABLE IF NOT EXISTS knowledge_builder_ai_grounding_suggestions (
                  assertion_id TEXT PRIMARY KEY,
                  extraction_run_id TEXT NOT NULL,
                  suggestion_status TEXT NOT NULL CHECK(suggestion_status IN
                    ('ready_for_review','low_confidence','missing_evidence')),
                  confidence REAL NOT NULL,
                  evidence_anchor_count INTEGER NOT NULL,
                  provider TEXT NOT NULL,
                  provider_version TEXT NOT NULL,
                  threshold REAL NOT NULL,
                  rationale TEXT NOT NULL,
                  suggested_at TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(assertion_id) REFERENCES knowledge_builder_document_candidate_assertions(assertion_id),
                  FOREIGN KEY(extraction_run_id) REFERENCES knowledge_builder_extraction_runs(extraction_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_kbags_run_status
                  ON knowledge_builder_ai_grounding_suggestions(extraction_run_id, suggestion_status, confidence);
                CREATE TABLE IF NOT EXISTS knowledge_builder_ai_grounding_runs (
                  ai_grounding_run_id TEXT PRIMARY KEY,
                  extraction_run_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  provider TEXT,
                  model TEXT,
                  agent_id TEXT,
                  prompt_ref TEXT NOT NULL,
                  prompt_hash TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
                  usage_json TEXT NOT NULL,
                  latency_ms INTEGER,
                  started_at TEXT NOT NULL,
                  completed_at TEXT,
                  failure_reason TEXT,
                  response_hash TEXT,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(extraction_run_id) REFERENCES knowledge_builder_extraction_runs(extraction_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_kbagr_latest
                  ON knowledge_builder_ai_grounding_runs(extraction_run_id, status, completed_at);
                CREATE TABLE IF NOT EXISTS knowledge_builder_ai_grounding_results (
                  ai_grounding_run_id TEXT NOT NULL,
                  assertion_id TEXT NOT NULL,
                  suggestion_status TEXT NOT NULL CHECK(suggestion_status IN
                    ('supported','unsupported','ambiguous','needs_edit','provider_error')),
                  confidence REAL NOT NULL,
                  corrected_assertion_json TEXT NOT NULL,
                  evidence_anchor_refs_json TEXT NOT NULL,
                  split_proposals_json TEXT NOT NULL DEFAULT '[]',
                  rationale TEXT NOT NULL,
                  issues_json TEXT NOT NULL,
                  result_hash TEXT NOT NULL,
                  PRIMARY KEY(ai_grounding_run_id, assertion_id),
                  FOREIGN KEY(ai_grounding_run_id) REFERENCES knowledge_builder_ai_grounding_runs(ai_grounding_run_id),
                  FOREIGN KEY(assertion_id) REFERENCES knowledge_builder_document_candidate_assertions(assertion_id)
                );
                CREATE TABLE IF NOT EXISTS knowledge_builder_grounding_batches (
                  batch_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  extraction_run_id TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  decision TEXT NOT NULL CHECK(decision='accept'),
                  response_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  UNIQUE(organization_id, workspace_id, idempotency_key),
                  FOREIGN KEY(extraction_run_id) REFERENCES knowledge_builder_extraction_runs(extraction_run_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_kbgb_tenant_idempotency
                  ON knowledge_builder_grounding_batches(
                    organization_id, COALESCE(workspace_id, ''), idempotency_key
                  );
                CREATE TABLE IF NOT EXISTS knowledge_builder_grounding_batch_items (
                  batch_id TEXT NOT NULL,
                  event_id TEXT NOT NULL UNIQUE,
                  assertion_id TEXT NOT NULL,
                  ordinal INTEGER NOT NULL,
                  PRIMARY KEY(batch_id, assertion_id),
                  FOREIGN KEY(batch_id) REFERENCES knowledge_builder_grounding_batches(batch_id),
                  FOREIGN KEY(event_id) REFERENCES knowledge_learning_events(event_id),
                  FOREIGN KEY(assertion_id) REFERENCES knowledge_builder_document_candidate_assertions(assertion_id)
                );
                CREATE TABLE IF NOT EXISTS knowledge_builder_assertion_occurrences (
                  assertion_id TEXT NOT NULL,
                  occurrence_id TEXT NOT NULL,
                  ordinal INTEGER NOT NULL,
                  PRIMARY KEY(assertion_id, occurrence_id),
                  FOREIGN KEY(assertion_id) REFERENCES knowledge_builder_document_candidate_assertions(assertion_id),
                  FOREIGN KEY(occurrence_id) REFERENCES knowledge_builder_semantic_occurrences(occurrence_id)
                );
                CREATE TABLE IF NOT EXISTS knowledge_builder_knowledge_scopes (
                  package_id TEXT PRIMARY KEY,
                  discovery_run_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  scope_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(discovery_run_id) REFERENCES knowledge_builder_discovery_runs(discovery_run_id)
                );
                CREATE TABLE IF NOT EXISTS knowledge_builder_evidence_obligations (
                  obligation_id TEXT PRIMARY KEY,
                  assertion_id TEXT NOT NULL,
                  obligation_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(assertion_id) REFERENCES knowledge_builder_document_candidate_assertions(assertion_id)
                );
                CREATE TABLE IF NOT EXISTS knowledge_builder_evidence_obligation_anchors (
                  obligation_id TEXT NOT NULL,
                  source_anchor_ref TEXT NOT NULL,
                  ordinal INTEGER NOT NULL,
                  PRIMARY KEY(obligation_id, source_anchor_ref),
                  FOREIGN KEY(obligation_id) REFERENCES knowledge_builder_evidence_obligations(obligation_id)
                );
                CREATE TABLE IF NOT EXISTS knowledge_builder_evidence_obligation_resolution_events (
                  resolution_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  obligation_id TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('satisfied','unresolved')),
                  supporting_anchor_refs_json TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(obligation_id) REFERENCES knowledge_builder_evidence_obligations(obligation_id)
                );
                CREATE TABLE IF NOT EXISTS contextgraph_reference_concepts (
                  concept_id TEXT PRIMARY KEY,
                  extraction_run_id TEXT NOT NULL,
                  candidate_graph_id TEXT,
                  source_document_hash TEXT NOT NULL,
                  concept_json TEXT NOT NULL,
                  grounding_state TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(extraction_run_id) REFERENCES knowledge_builder_extraction_runs(extraction_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_cgrc_run
                  ON contextgraph_reference_concepts(extraction_run_id, grounding_state);
                CREATE TABLE IF NOT EXISTS contextgraph_canonical_expressions (
                  expression_id TEXT PRIMARY KEY,
                  concept_id TEXT NOT NULL,
                  extraction_run_id TEXT NOT NULL,
                  candidate_graph_id TEXT,
                  source_document_hash TEXT NOT NULL,
                  expression_json TEXT NOT NULL,
                  expression_kind TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(concept_id) REFERENCES contextgraph_reference_concepts(concept_id),
                  FOREIGN KEY(extraction_run_id) REFERENCES knowledge_builder_extraction_runs(extraction_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_cgce_concept
                  ON contextgraph_canonical_expressions(concept_id, expression_kind);
                CREATE TABLE IF NOT EXISTS contextgraph_observed_expression_observations (
                  observation_id TEXT PRIMARY KEY,
                  concept_id TEXT NOT NULL,
                  finding_run_id TEXT NOT NULL,
                  finding_id TEXT NOT NULL,
                  reference_concept_version_ref TEXT NOT NULL,
                  accepted_graph_release_id TEXT NOT NULL,
                  human_learning_event_id TEXT NOT NULL,
                  semantic_relation_decision_hash TEXT NOT NULL,
                  observation_json TEXT NOT NULL,
                  grounding_state TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(concept_id) REFERENCES contextgraph_reference_concepts(concept_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_cgoeo_learning_event
                  ON contextgraph_observed_expression_observations(human_learning_event_id, finding_id);
                CREATE TABLE IF NOT EXISTS contextgraph_observed_expressions (
                  expression_id TEXT PRIMARY KEY,
                  concept_id TEXT NOT NULL,
                  normalized_text TEXT NOT NULL,
                  semantic_relation TEXT NOT NULL,
                  reference_concept_version_ref TEXT NOT NULL,
                  accepted_graph_release_id TEXT NOT NULL,
                  latest_observation_id TEXT NOT NULL,
                  grounding_state TEXT NOT NULL,
                  observation_count INTEGER NOT NULL,
                  expression_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(concept_id) REFERENCES contextgraph_reference_concepts(concept_id),
                  FOREIGN KEY(latest_observation_id) REFERENCES contextgraph_observed_expression_observations(observation_id),
                  UNIQUE(concept_id, normalized_text, semantic_relation, reference_concept_version_ref, accepted_graph_release_id)
                );
                CREATE TABLE IF NOT EXISTS contextgraph_artifact_origins (
                  origin_id TEXT PRIMARY KEY,
                  workspace_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  source_id TEXT NOT NULL,
                  extraction_run_id TEXT,
                  candidate_graph_id TEXT,
                  artifact_id TEXT,
                  source_identity_ref TEXT,
                  relation_kind TEXT NOT NULL,
                  origin_kind TEXT NOT NULL DEFAULT 'unresolved'
                    CHECK (origin_kind IN ('attached', 'source_tombstoned', 'imported_without_source', 'unresolved')),
                  authority_state TEXT NOT NULL DEFAULT 'candidate'
                    CHECK (authority_state IN ('candidate', 'authoritative')),
                  release_lineage_json TEXT,
                  release_lineage_hash TEXT,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cg_artifact_origins_source
                  ON contextgraph_artifact_origins (organization_id, workspace_id, source_id);
                CREATE INDEX IF NOT EXISTS idx_cg_artifact_origins_artifact
                  ON contextgraph_artifact_origins (artifact_id);
                CREATE TABLE IF NOT EXISTS source_representation_edges (
                  representation_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  source_id TEXT NOT NULL,
                  relation_kind TEXT NOT NULL CHECK (relation_kind = 'normalized_from'),
                  original_path TEXT NOT NULL,
                  original_content_hash TEXT NOT NULL,
                  artifact_ref TEXT NOT NULL,
                  artifact_path TEXT NOT NULL,
                  artifact_content_hash TEXT NOT NULL,
                  version INTEGER NOT NULL CHECK (version >= 1),
                  created_at TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(source_id) REFERENCES knowledge_builder_sources(source_id),
                  UNIQUE(organization_id, workspace_id, source_id, relation_kind, version),
                  UNIQUE(organization_id, workspace_id, source_id, relation_kind,
                         artifact_ref, artifact_content_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_source_representation_edges_source
                  ON source_representation_edges(
                    organization_id, workspace_id, source_id, relation_kind, version
                  );
                CREATE TABLE IF NOT EXISTS source_representation_lifecycle_events (
                  event_id TEXT PRIMARY KEY,
                  representation_id TEXT NOT NULL,
                  operation_id TEXT,
                  actor_id TEXT NOT NULL,
                  previous_state TEXT,
                  next_state TEXT NOT NULL
                    CHECK (next_state IN ('active', 'superseded', 'tombstoned')),
                  reason TEXT,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(representation_id)
                    REFERENCES source_representation_edges(representation_id)
                );
                CREATE INDEX IF NOT EXISTS idx_source_representation_events_latest
                  ON source_representation_lifecycle_events(
                    representation_id, created_at
                  );
                CREATE TABLE IF NOT EXISTS source_lifecycle_events (
                  event_id TEXT PRIMARY KEY,
                  source_id TEXT NOT NULL,
                  operation_id TEXT,
                  actor_id TEXT NOT NULL,
                  previous_state TEXT,
                  next_state TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  reason TEXT,
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_source_lifecycle_events_source
                  ON source_lifecycle_events (source_id, created_at);
                CREATE TABLE IF NOT EXISTS artifact_lifecycle_events (
                  event_id TEXT PRIMARY KEY,
                  artifact_id TEXT NOT NULL,
                  origin_id TEXT,
                  operation_id TEXT,
                  actor_id TEXT NOT NULL,
                  previous_state TEXT,
                  next_state TEXT NOT NULL,
                  stage TEXT NOT NULL,
                  reason TEXT,
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifact_lifecycle_events_artifact
                  ON artifact_lifecycle_events (artifact_id, created_at);
                CREATE TABLE IF NOT EXISTS source_retirement_operations (
                  operation_id TEXT PRIMARY KEY,
                  source_id TEXT NOT NULL,
                  actor_id TEXT NOT NULL,
                  state TEXT NOT NULL,
                  original_path TEXT,
                  staged_path TEXT,
                  failure_reason TEXT,
                  failure_stage TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_source_retirement_ops_source
                  ON source_retirement_operations (source_id, created_at);
                CREATE TABLE IF NOT EXISTS lineage_repair_events (
                  event_id TEXT PRIMARY KEY,
                  repair_id TEXT NOT NULL,
                  report_hash TEXT NOT NULL,
                  operation_id TEXT NOT NULL,
                  lineage_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  actor_id TEXT NOT NULL,
                  action_count INTEGER NOT NULL,
                  before_active_count INTEGER NOT NULL,
                  after_active_count INTEGER NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE (repair_id, operation_id)
                );
                CREATE INDEX IF NOT EXISTS idx_lineage_repair_events_scope
                  ON lineage_repair_events (workspace_id, organization_id, created_at);
                """
            )
            retirement_columns = {
                row["name"]
                for row in self._conn.execute(
                    "PRAGMA table_info(source_retirement_operations)"
                ).fetchall()
            }
            for name in (
                "retirement_mode",
                "domain_id",
                "inventory_hash",
                "organization_id",
                "workspace_id",
            ):
                if name not in retirement_columns:
                    self._conn.execute(
                        f"ALTER TABLE source_retirement_operations ADD COLUMN {name} TEXT"
                    )
            columns = {
                row["name"]
                for row in self._conn.execute(
                    "PRAGMA table_info(knowledge_builder_extraction_runs)"
                ).fetchall()
            }
            result_columns = {
                row["name"]
                for row in self._conn.execute(
                    "PRAGMA table_info(knowledge_builder_ai_grounding_results)"
                ).fetchall()
            }
            if "split_proposals_json" not in result_columns:
                self._conn.execute(
                    "ALTER TABLE knowledge_builder_ai_grounding_results "
                    "ADD COLUMN split_proposals_json TEXT NOT NULL DEFAULT '[]'"
                )
            for name in (
                "provider_ref", "provider_commit", "native_artifact_ref",
                "native_artifact_hash", "failure_reason",
                "source_identity_ref", "parser_trace_id",
                "created_at",
            ):
                if name not in columns:
                    self._conn.execute(
                        f"ALTER TABLE knowledge_builder_extraction_runs ADD COLUMN {name} TEXT"
                    )
            for table in (
                "knowledge_builder_sources",
                "knowledge_builder_discovery_runs",
                "knowledge_builder_extraction_runs",
                "source_representation_edges",
                "contextgraph_artifact_origins",
            ):
                table_columns = {
                    row["name"]
                    for row in self._conn.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                }
                if "lineage_id" not in table_columns:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN lineage_id TEXT"
                    )
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_lineage_id "
                    f"ON {table}(lineage_id)"
                )
            origin_columns = {
                row["name"]
                for row in self._conn.execute(
                    "PRAGMA table_info(contextgraph_artifact_origins)"
                ).fetchall()
            }
            for name in ("release_lineage_json", "release_lineage_hash"):
                if name not in origin_columns:
                    self._conn.execute(
                        f"ALTER TABLE contextgraph_artifact_origins ADD COLUMN {name} TEXT"
                    )
            for table in (
                "knowledge_builder_sources",
                "knowledge_builder_discovery_runs",
                "knowledge_builder_extraction_runs",
            ):
                table_columns = {
                    row["name"]
                    for row in self._conn.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                }
                if "retirement_state" not in table_columns:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN retirement_state "
                        "TEXT NOT NULL DEFAULT 'active'"
                    )
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS lineage_materialization_retirement_events (
                  event_id TEXT PRIMARY KEY,
                  operation_id TEXT NOT NULL,
                  lineage_id TEXT NOT NULL,
                  store_name TEXT NOT NULL,
                  table_name TEXT NOT NULL,
                  member_id TEXT NOT NULL,
                  previous_state TEXT NOT NULL,
                  next_state TEXT NOT NULL,
                  actor_id TEXT NOT NULL,
                  reason TEXT,
                  created_at TEXT NOT NULL,
                  UNIQUE(operation_id, store_name, table_name, member_id, next_state)
                );
                CREATE INDEX IF NOT EXISTS idx_lineage_materialization_events_lineage
                  ON lineage_materialization_retirement_events(lineage_id, created_at);
                """
            )
            if "boundary_validation_json" not in columns:
                self._conn.execute(
                    "ALTER TABLE knowledge_builder_extraction_runs "
                    "ADD COLUMN boundary_validation_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "boundary_review_required" not in columns:
                self._conn.execute(
                    "ALTER TABLE knowledge_builder_extraction_runs "
                    "ADD COLUMN boundary_review_required INTEGER NOT NULL DEFAULT 0"
                )
            if "source_role" not in columns:
                self._conn.execute(
                    "ALTER TABLE knowledge_builder_extraction_runs "
                    "ADD COLUMN source_role TEXT NOT NULL DEFAULT 'legacy_unknown'"
                )
            if "workflow_kind" not in columns:
                self._conn.execute(
                    "ALTER TABLE knowledge_builder_extraction_runs "
                    "ADD COLUMN workflow_kind TEXT NOT NULL DEFAULT 'legacy_candidate_assertion'"
                )
            mark_schema_current(
                self._conn,
                schema_name=_SCHEMA_NAME,
                schema_version=_SCHEMA_VERSION,
            )
            self._conn.commit()

    # ------------------------------------------------ source representations

    @staticmethod
    def _representation_id(identity: dict[str, str]) -> str:
        digest = hashlib.sha256(_json(identity).encode("utf-8")).hexdigest()
        return f"srp_{digest}"

    def record_source_representation(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        source_id: str,
        lineage_id: str,
        relation_kind: str,
        original_path: str,
        original_content_hash: str,
        artifact_ref: str,
        artifact_path: str,
        artifact_content_hash: str,
    ) -> dict[str, Any]:
        if not isinstance(lineage_id, str) or not lineage_id.strip():
            raise ValueError("lineage_id is required")
        if lineage_id != source_id:
            raise ValueError("lineage_id must equal source_id for source-backed records")
        if relation_kind != "normalized_from":
            raise ValueError("relation_kind must be normalized_from")
        identity = {
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "source_id": source_id,
            "lineage_id": lineage_id,
            "relation_kind": relation_kind,
            "artifact_ref": artifact_ref,
            "artifact_content_hash": artifact_content_hash,
        }
        representation_id = self._representation_id(identity)
        with self._lock:
            source = self._conn.execute(
                sql_query("SELECT source_id FROM knowledge_builder_sources "
                "WHERE source_id = ? AND organization_id = ? AND workspace_id = ?",'SELECT source_id FROM knowledge_builder_sources WHERE source_id = %s AND organization_id = %s AND workspace_id = %s'),
                (source_id, organization_id, workspace_id),
            ).fetchone()
            if source is None:
                raise ValueError("source is not governed by this workspace")
            existing = self._conn.execute(
                sql_query("SELECT * FROM source_representation_edges WHERE representation_id = ?",'SELECT * FROM source_representation_edges WHERE representation_id = %s'),
                (representation_id,),
            ).fetchone()
            if existing is None:
                existing = self._conn.execute(
                    sql_query("SELECT * FROM source_representation_edges "
                    "WHERE organization_id = ? AND workspace_id = ? AND source_id = ? "
                    "AND relation_kind = ? AND artifact_ref = ? "
                    "AND artifact_content_hash = ?",'SELECT * FROM source_representation_edges WHERE organization_id = %s AND workspace_id = %s AND source_id = %s AND relation_kind = %s AND artifact_ref = %s AND artifact_content_hash = %s'),
                    (
                        organization_id,
                        workspace_id,
                        source_id,
                        relation_kind,
                        artifact_ref,
                        artifact_content_hash,
                    ),
                ).fetchone()
            if existing is not None:
                latest_state = self._conn.execute(
                    sql_query("SELECT next_state FROM source_representation_lifecycle_events "
                    "WHERE representation_id = ? "
                    "ORDER BY created_at DESC, rowid DESC LIMIT 1",'SELECT next_state FROM source_representation_lifecycle_events WHERE representation_id = %s ORDER BY created_at DESC, event_sequence DESC LIMIT 1'),
                    (existing["representation_id"],),
                ).fetchone()
                if latest_state is not None and latest_state["next_state"] == "tombstoned":
                    try:
                        begin_write(self._conn)
                        self._append_representation_event(
                            existing["representation_id"],
                            previous_state="tombstoned",
                            next_state="active",
                            actor_id="system",
                            reason="representation re-registered",
                        )
                        self._conn.commit()
                    except Exception:
                        self._conn.rollback()
                        raise
                return dict(existing)

            active = self._active_representation_rows(
                source_id=source_id,
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
            version = int(
                self._conn.execute(
                    sql_query("SELECT COALESCE(MAX(version), 0) + 1 "
                    "FROM source_representation_edges WHERE organization_id = ? "
                    "AND workspace_id = ? AND source_id = ? AND relation_kind = ?",'SELECT COALESCE(MAX(version), 0) + 1 FROM source_representation_edges WHERE organization_id = %s AND workspace_id = %s AND source_id = %s AND relation_kind = %s'),
                    (organization_id, workspace_id, source_id, relation_kind),
                ).fetchone()[0]
            )
            created_at = _now()
            row = {
                **identity,
                "representation_id": representation_id,
                "original_path": original_path,
                "original_content_hash": original_content_hash,
                "artifact_path": artifact_path,
                "version": version,
                "created_at": created_at,
            }
            row["content_hash"] = content_hash(_json(row))
            try:
                begin_write(self._conn)
                for prior in active:
                    self._append_representation_event(
                        prior["representation_id"], previous_state="active",
                        next_state="superseded", actor_id="system",
                        reason="new representation version", created_at=created_at,
                    )
                self._conn.execute(
                    sql_query("INSERT INTO source_representation_edges ("
                    "representation_id, organization_id, workspace_id, source_id, lineage_id, "
                    "relation_kind, original_path, original_content_hash, artifact_ref, "
                    "artifact_path, artifact_content_hash, version, created_at, content_hash"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",'INSERT INTO source_representation_edges (representation_id, organization_id, workspace_id, source_id, lineage_id, relation_kind, original_path, original_content_hash, artifact_ref, artifact_path, artifact_content_hash, version, created_at, content_hash) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                    tuple(row[name] for name in (
                        "representation_id", "organization_id", "workspace_id", "source_id", "lineage_id",
                        "relation_kind", "original_path", "original_content_hash", "artifact_ref",
                        "artifact_path", "artifact_content_hash", "version", "created_at",
                        "content_hash",
                    )),
                )
                self._append_representation_event(
                    representation_id, previous_state=None, next_state="active",
                    actor_id="system", created_at=created_at,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return row

    def _append_representation_event(
        self,
        representation_id: str,
        *,
        previous_state: str | None,
        next_state: str,
        actor_id: str,
        operation_id: str | None = None,
        reason: str | None = None,
        created_at: str | None = None,
    ) -> None:
        self._conn.execute(
            sql_query("INSERT INTO source_representation_lifecycle_events ("
            "event_id, representation_id, operation_id, actor_id, previous_state, "
            "next_state, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",'INSERT INTO source_representation_lifecycle_events (event_id, representation_id, operation_id, actor_id, previous_state, next_state, reason, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'),
            (
                f"srle_{uuid4().hex}", representation_id, operation_id, actor_id,
                previous_state, next_state, reason, created_at or _now(),
            ),
        )

    def _active_representation_rows(
        self, *, source_id: str, organization_id: str, workspace_id: str
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            sql_query("SELECT edge.* FROM source_representation_edges edge "
            "JOIN source_representation_lifecycle_events event "
            "ON event.event_id = (SELECT latest.event_id "
            "FROM source_representation_lifecycle_events latest "
            "WHERE latest.representation_id = edge.representation_id "
            "ORDER BY latest.created_at DESC, latest.rowid DESC LIMIT 1) "
            "WHERE edge.source_id = ? AND edge.organization_id = ? "
            "AND edge.workspace_id = ? AND event.next_state = 'active' "
            "ORDER BY edge.version","SELECT edge.* FROM source_representation_edges edge JOIN source_representation_lifecycle_events event ON event.event_id = (SELECT latest.event_id FROM source_representation_lifecycle_events latest WHERE latest.representation_id = edge.representation_id ORDER BY latest.created_at DESC, latest.event_sequence DESC LIMIT 1) WHERE edge.source_id = %s AND edge.organization_id = %s AND edge.workspace_id = %s AND event.next_state = 'active' ORDER BY edge.version"),
            (source_id, organization_id, workspace_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_active_source_representations(
        self, source_id: str, *, organization_id: str, workspace_id: str
    ) -> list[dict[str, Any]]:
        with self._lock:
            return self._active_representation_rows(
                source_id=source_id,
                organization_id=organization_id,
                workspace_id=workspace_id,
            )

    def resolve_active_source_representation(
        self, *, artifact_ref: str, organization_id: str, workspace_id: str
    ) -> dict[str, Any]:
        """Resolve one active normalized artifact inside its governed scope."""

        if not isinstance(artifact_ref, str) or not artifact_ref.strip():
            raise ValueError("normalized document artifact ref is required")
        with self._lock:
            rows = self._conn.execute(
                sql_query("SELECT edge.* FROM source_representation_edges edge "
                "JOIN source_representation_lifecycle_events event "
                "ON event.event_id = (SELECT latest.event_id "
                "FROM source_representation_lifecycle_events latest "
                "WHERE latest.representation_id = edge.representation_id "
                "ORDER BY latest.created_at DESC, latest.rowid DESC LIMIT 1) "
                "WHERE edge.artifact_ref = ? AND edge.organization_id = ? "
                "AND edge.workspace_id = ? AND event.next_state = 'active'","SELECT edge.* FROM source_representation_edges edge JOIN source_representation_lifecycle_events event ON event.event_id = (SELECT latest.event_id FROM source_representation_lifecycle_events latest WHERE latest.representation_id = edge.representation_id ORDER BY latest.created_at DESC, latest.event_sequence DESC LIMIT 1) WHERE edge.artifact_ref = %s AND edge.organization_id = %s AND edge.workspace_id = %s AND event.next_state = 'active'"),
                (artifact_ref, organization_id, workspace_id),
            ).fetchall()
        if len(rows) != 1:
            raise ValueError("active governed normalized representation not found")
        row = dict(rows[0])
        payload = {key: value for key, value in row.items() if key != "content_hash"}
        if row["content_hash"] != content_hash(_json(payload)):
            raise ValueError("normalized representation integrity failure")
        identity = {
            key: row[key]
            for key in (
                "organization_id", "workspace_id", "source_id", "lineage_id",
                "relation_kind", "artifact_ref", "artifact_content_hash",
            )
        }
        if row["representation_id"] != self._representation_id(identity):
            raise ValueError("normalized representation identity failure")
        return row

    def tombstone_source_representations(
        self,
        source_id: str,
        *,
        operation_id: str,
        actor_id: str,
        organization_id: str,
        workspace_id: str,
    ) -> int:
        with self._lock:
            active = self._active_representation_rows(
                source_id=source_id,
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
            try:
                begin_write(self._conn)
                for row in active:
                    self._append_representation_event(
                        row["representation_id"], previous_state="active",
                        next_state="tombstoned", actor_id=actor_id,
                        operation_id=operation_id, reason="source retired",
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return len(active)

    # ---------------------------------------------------------- retirement

    def begin_source_retirement(self, source_id: str, *, actor_id: str) -> str:
        """Plan A M2: thin wrapper around the orchestrator's begin.

        Returns the operation_id. Use this method from routes; the orchestrator
        owns the state machine.
        """
        from semantier.knowledge_source_retirement import SourceRetirementOrchestrator
        orch = SourceRetirementOrchestrator(extraction_store=self)
        return orch.begin(source_id=source_id, actor_id=actor_id)

    def advance_source_retirement(
        self,
        operation_id: str,
        target_state,
        *,
        file_stager=None,
        metadata_stage=None,
        artifact_stage=None,
        reason=None,
    ) -> str:
        """Plan A M2: thin wrapper around the orchestrator's advance.

        Returns the new state value as a string. Use this method from routes.
        """
        from semantier.knowledge_source_retirement import (
            SourceRetirementOrchestrator,
            SourceRetirementState,
        )
        orch = SourceRetirementOrchestrator(extraction_store=self, file_stager=file_stager)
        new_state = orch.advance(
            operation_id=operation_id,
            target_state=SourceRetirementState(target_state)
            if isinstance(target_state, str) else target_state,
            file_stager=file_stager,
            metadata_stage=metadata_stage,
            artifact_stage=artifact_stage,
            reason=reason,
        )
        return new_state.value if hasattr(new_state, "value") else str(new_state)

    def list_source_retirement(self, source_id: str) -> list[dict[str, Any]]:
        """Plan A M2: return the most recent retirement operation for a source."""
        rows = self._conn.execute(
            sql_query("SELECT operation_id, source_id, actor_id, state, original_path, "
            "staged_path, failure_reason, failure_stage, created_at, updated_at "
            "FROM source_retirement_operations WHERE source_id = ? "
            "ORDER BY created_at DESC, operation_id DESC",'SELECT operation_id, source_id, actor_id, state, original_path, staged_path, failure_reason, failure_stage, created_at, updated_at FROM source_retirement_operations WHERE source_id = %s ORDER BY created_at DESC, operation_id DESC'),
            (source_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_retirement_preview(self, source_id: str) -> dict[str, Any]:
        """Plan A M4: cascade scope and severity tier for the confirm dialog."""
        # For M2, return a minimal preview; M4 will compute authoritative descendants
        # and the locked tier.
        cascade_scope = {"extraction_runs": 0, "reference_concepts": 0,
                          "canonical_expressions": 0, "graph_artifacts": 0}
        return {
            "source_id": source_id,
            "cascade_scope": cascade_scope,
            "severity_tier": "tier-1-light",
            "authoritative_descendants": 0,
        }

    def create_extraction_run(
        self, *, extraction_run_id: str, discovery_run_id: str, source_id: str,
        lineage_id: str,
        organization_id: str, workspace_id: str | None, package_id: str | None,
        document_id: str, provider: str, provider_version: str, profile_ref: str,
        provider_ref: str | None = None, provider_commit: str | None = None,
        source_identity_ref: str | None = None, parser_trace_id: str | None = None,
        source_role: str = "legacy_unknown",
        workflow_kind: str = "legacy_candidate_assertion",
    ) -> dict[str, Any]:
        if not isinstance(lineage_id, str) or not lineage_id.strip():
            raise ValueError("lineage_id is required")
        if lineage_id != source_id:
            raise ValueError("lineage_id must equal source_id for source-backed records")
        if provider not in {"semantica", "semantica_service", "legacy", "langextract"}:
            raise ValueError(
                "provider must be semantica, semantica_service, langextract, or legacy"
            )
        if source_role not in {
            "reference_sensitive_word_list",
            "runtime_tender_document",
            "legacy_unknown",
        }:
            raise ValueError("source_role is invalid")
        if workflow_kind not in {
            "reference_graph_build",
            "runtime_tender_evaluation",
            "legacy_candidate_assertion",
        }:
            raise ValueError("workflow_kind is invalid")
        effective_provider_ref = provider_ref or provider
        started_at = _now()
        payload = {
            "schema_version": "knowledge_builder_extraction_run.v1",
            "extraction_run_id": extraction_run_id,
            "discovery_run_id": discovery_run_id,
            "source_id": source_id,
            "lineage_id": lineage_id,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "package_id": package_id,
            "document_id": document_id,
            "source_role": source_role,
            "workflow_kind": workflow_kind,
            "provider": provider,
            "provider_version": provider_version,
            "provider_ref": effective_provider_ref,
            "provider_commit": provider_commit,
            "profile_ref": profile_ref,
            "source_identity_ref": source_identity_ref,
            "parser_trace_id": parser_trace_id,
            "created_at": started_at,
            "run_status": "running",
            "raw_output_ref": None,
            "raw_output_hash": None,
            "native_artifact_ref": None,
            "native_artifact_hash": None,
            "failure_reason": None,
            "warnings": [],
            "started_at": started_at,
            "completed_at": None,
        }
        digest = content_hash(payload)
        with self._lock:
            existing = self.get_extraction_run(extraction_run_id)
            if existing:
                if existing["provider"] != provider:
                    raise ValueError("provider choice is immutable for an extraction run")
                raise ValueError(f"extraction run already exists: {extraction_run_id}")
            self._conn.execute(
                sql_query("""INSERT INTO knowledge_builder_extraction_runs
                (extraction_run_id, discovery_run_id, source_id, lineage_id, organization_id, workspace_id,
                 package_id, document_id, source_role, workflow_kind,
                 provider, provider_version, provider_ref, provider_commit,
                 profile_ref, source_identity_ref, parser_trace_id, run_status,
                 raw_output_ref, raw_output_hash, warnings_json, created_at, started_at, completed_at, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, '[]', ?, ?, NULL, ?)""","INSERT INTO knowledge_builder_extraction_runs\n                (extraction_run_id, discovery_run_id, source_id, lineage_id, organization_id, workspace_id,\n                 package_id, document_id, source_role, workflow_kind,\n                 provider, provider_version, provider_ref, provider_commit,\n                 profile_ref, source_identity_ref, parser_trace_id, run_status,\n                 raw_output_ref, raw_output_hash, warnings_json, created_at, started_at, completed_at, content_hash)\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL, '[]', %s, %s, NULL, %s)"),
                (extraction_run_id, discovery_run_id, source_id, lineage_id, organization_id, workspace_id,
                 package_id, document_id, source_role, workflow_kind,
                 provider, provider_version, effective_provider_ref,
                 provider_commit, profile_ref, source_identity_ref, parser_trace_id, "running",
                 started_at, started_at, digest),
            )
            self._conn.commit()
        return payload

    def complete_extraction_run(
        self, *, extraction_run_id: str, raw_output_ref: str, raw_output_hash: str,
        warnings: tuple[str, ...], occurrences: tuple[SemanticOccurrence, ...],
        assertions: tuple[DocumentCandidateAssertion, ...], machine_events: tuple[Any, ...] = (),
        candidate_graph: SensitiveTermCandidateGraph | None = None,
        provider_ref: str | None = None, provider_commit: str | None = None,
        native_artifact_ref: str | None = None, native_artifact_hash: str | None = None,
        boundary_validation: tuple[dict[str, Any], ...] | None = None,
        run_status: str = "completed",
    ) -> dict[str, Any]:
        # W4: allow the run to surface unresolved boundary candidates without
        # downgrading the whole row to a failure.  ``completed_with_boundary_review``
        # preserves the projection that passed every gate while flagging the
        # spans that still need curator grounding before activation.
        if run_status not in {"completed", "completed_with_boundary_review"}:
            raise ValueError(
                "complete_extraction_run accepts only 'completed' or "
                "'completed_with_boundary_review' for run_status"
            )
        run = self.get_extraction_run(extraction_run_id)
        if run is None:
            raise ValueError(f"extraction run not found: {extraction_run_id}")
        if run["run_status"] != "running":
            raise ValueError("only a running extraction run can be completed")
        occurrence_ids = {item.occurrence_id for item in occurrences}
        for assertion in assertions:
            if not set(assertion.occurrence_refs).issubset(occurrence_ids):
                raise ValueError("assertion references occurrence outside extraction result")
        with self._lock:
            if candidate_graph is not None:
                graph_payload = candidate_graph.model_dump(mode="json")
                self._conn.execute(
                    sql_query("""INSERT INTO sensitive_term_candidate_graphs
                    (candidate_graph_id, source_document_hash, extraction_run_id, proposal_id,
                     graph_json, content_hash, organization_id, workspace_id, allowed_base_graph_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)""",'INSERT INTO sensitive_term_candidate_graphs\n                    (candidate_graph_id, source_document_hash, extraction_run_id, proposal_id,\n                     graph_json, content_hash, organization_id, workspace_id, allowed_base_graph_version)\n                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL)'),
                    (candidate_graph.candidate_graph_id, candidate_graph.source_document_hash,
                     extraction_run_id, candidate_graph.machine_proposal.proposal_id,
                     _json(graph_payload), candidate_graph.graph_hash(),
                     run["organization_id"], run.get("workspace_id")),
                )
            for occurrence in occurrences:
                payload = occurrence.model_dump(mode="json")
                digest = content_hash(payload)
                self._conn.execute(
                    sql_query("""INSERT INTO knowledge_builder_semantic_occurrences
                    (occurrence_id, extraction_run_id, document_id, source_anchor_json,
                     extraction_class, extraction_attributes_json, alignment_json,
                     machine_confidence, model_ref, prompt_ref, raw_output_ref,
                     occurrence_json, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",'INSERT INTO knowledge_builder_semantic_occurrences\n                    (occurrence_id, extraction_run_id, document_id, source_anchor_json,\n                     extraction_class, extraction_attributes_json, alignment_json,\n                     machine_confidence, model_ref, prompt_ref, raw_output_ref,\n                     occurrence_json, content_hash) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                    (occurrence.occurrence_id, extraction_run_id, occurrence.document_id,
                     _json(payload["source_anchor"]), occurrence.extraction_class,
                     _json(payload["extraction_attributes"]), _json(payload["alignment"]),
                     occurrence.machine_confidence, occurrence.model_ref, occurrence.prompt_ref,
                     occurrence.raw_output_ref, _json(payload), digest),
                )
            for assertion in assertions:
                if candidate_graph is not None:
                    if assertion.candidate_graph_id != candidate_graph.candidate_graph_id:
                        raise ValueError("assertion candidate graph identity mismatch")
                    assertion = assertion.model_copy(update={
                        "candidate_graph_hash": candidate_graph.graph_hash()
                    })
                payload = assertion.model_dump(mode="json")
                digest = content_hash(payload)
                self._conn.execute(
                    sql_query("""INSERT INTO knowledge_builder_document_candidate_assertions
                    (assertion_id, extraction_run_id, candidate_graph_id,
                     source_document_hash, assertion_json, content_hash) VALUES (?, ?, ?, ?, ?, ?)""",'INSERT INTO knowledge_builder_document_candidate_assertions\n                    (assertion_id, extraction_run_id, candidate_graph_id,\n                     source_document_hash, assertion_json, content_hash) VALUES (%s, %s, %s, %s, %s, %s)'),
                    (assertion.assertion_id, extraction_run_id, assertion.candidate_graph_id,
                     assertion.source_document_hash, _json(payload), digest),
                )
                for ordinal, occurrence_id in enumerate(assertion.occurrence_refs):
                    self._conn.execute(
                        sql_query("INSERT INTO knowledge_builder_assertion_occurrences VALUES (?, ?, ?)",'INSERT INTO knowledge_builder_assertion_occurrences VALUES (%s, %s, %s)'),
                        (assertion.assertion_id, occurrence_id, ordinal),
                    )
            completed_at = _now()
            boundary_review_required = 1 if run_status == "completed_with_boundary_review" else 0
            self._conn.execute(
                sql_query("""UPDATE knowledge_builder_extraction_runs SET run_status=?,
                raw_output_ref=?, raw_output_hash=?,
                provider_commit=COALESCE(?, provider_commit), native_artifact_ref=?,
                native_artifact_hash=?, failure_reason=NULL, warnings_json=?,
                boundary_validation_json=?, boundary_review_required=?, completed_at=?
                WHERE extraction_run_id=?""",'UPDATE knowledge_builder_extraction_runs SET run_status=%s,\n                raw_output_ref=%s, raw_output_hash=%s,\n                provider_commit=COALESCE(%s, provider_commit), native_artifact_ref=%s,\n                native_artifact_hash=%s, failure_reason=NULL, warnings_json=%s,\n                boundary_validation_json=%s, boundary_review_required=%s, completed_at=%s\n                WHERE extraction_run_id=%s'),
                (run_status, raw_output_ref, raw_output_hash, provider_commit,
                 native_artifact_ref or raw_output_ref, native_artifact_hash or raw_output_hash,
                 _json(warnings), _json(list(boundary_validation or ())),
                 boundary_review_required, completed_at, extraction_run_id),
            )
            try:
                for event in machine_events:
                    self._append_event_uncommitted(event)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_extraction_run(extraction_run_id) or {}

    def fail_extraction_run(
        self, extraction_run_id: str, warning: str, *, failure_reason: str = "provider_failed",
        provider_ref: str | None = None, provider_commit: str | None = None,
        native_artifact_ref: str | None = None, native_artifact_hash: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._conn.execute(
                sql_query("""UPDATE knowledge_builder_extraction_runs SET run_status='failed',
                warnings_json=?, failure_reason=?,
                provider_commit=COALESCE(?, provider_commit), native_artifact_ref=?,
                native_artifact_hash=?, raw_output_ref=?, raw_output_hash=?, completed_at=?
                WHERE extraction_run_id=? AND run_status='running'""","UPDATE knowledge_builder_extraction_runs SET run_status='failed',\n                warnings_json=%s, failure_reason=%s,\n                provider_commit=COALESCE(%s, provider_commit), native_artifact_ref=%s,\n                native_artifact_hash=%s, raw_output_ref=%s, raw_output_hash=%s, completed_at=%s\n                WHERE extraction_run_id=%s AND run_status='running'"),
                (_json([warning]), failure_reason, provider_commit,
                 native_artifact_ref, native_artifact_hash, native_artifact_ref,
                 native_artifact_hash, _now(), extraction_run_id),
            )
            self._conn.commit()
        return self.get_extraction_run(extraction_run_id) or {}

    def rollback_failed_completion(self) -> None:
        with self._lock:
            self._conn.rollback()

    def compensate_failed_completion(
        self, extraction_run_id: str, candidate_graph_id: str, warning: str
    ) -> dict[str, Any]:
        """Atomically remove an unreviewed candidate and make its run coherently failed."""
        with self._lock:
            begin_write(self._conn)
            try:
                reviewed = self._conn.execute(
                    sql_query("SELECT 1 FROM tender_graph_deltas WHERE candidate_graph_id=? LIMIT 1",'SELECT 1 FROM tender_graph_deltas WHERE candidate_graph_id=%s LIMIT 1'),
                    (candidate_graph_id,),
                ).fetchone()
                if reviewed:
                    raise ValueError("reviewed candidate graph cannot be compensated")
                self._conn.execute(
                    sql_query("DELETE FROM knowledge_learning_events WHERE event_type='machine_proposal' "
                    "AND case_id IN (SELECT assertion_id FROM "
                    "knowledge_builder_document_candidate_assertions WHERE extraction_run_id=?)","DELETE FROM knowledge_learning_events WHERE event_type='machine_proposal' AND case_id IN (SELECT assertion_id FROM knowledge_builder_document_candidate_assertions WHERE extraction_run_id=%s)"),
                    (extraction_run_id,),
                )
                self._conn.execute(
                    sql_query("DELETE FROM sensitive_term_candidate_graphs WHERE candidate_graph_id=?",'DELETE FROM sensitive_term_candidate_graphs WHERE candidate_graph_id=%s'),
                    (candidate_graph_id,),
                )
                self._conn.execute(
                    sql_query("DELETE FROM knowledge_builder_assertion_occurrences WHERE assertion_id IN "
                    "(SELECT assertion_id FROM knowledge_builder_document_candidate_assertions "
                    "WHERE extraction_run_id=?)",'DELETE FROM knowledge_builder_assertion_occurrences WHERE assertion_id IN (SELECT assertion_id FROM knowledge_builder_document_candidate_assertions WHERE extraction_run_id=%s)'), (extraction_run_id,),
                )
                self._conn.execute(
                    sql_query("DELETE FROM knowledge_builder_document_candidate_assertions WHERE extraction_run_id=?",'DELETE FROM knowledge_builder_document_candidate_assertions WHERE extraction_run_id=%s'),
                    (extraction_run_id,),
                )
                self._conn.execute(
                    sql_query("DELETE FROM knowledge_builder_semantic_occurrences WHERE extraction_run_id=?",'DELETE FROM knowledge_builder_semantic_occurrences WHERE extraction_run_id=%s'),
                    (extraction_run_id,),
                )
                self._conn.execute(
                    sql_query("""UPDATE knowledge_builder_extraction_runs SET run_status='failed',
                    warnings_json=?, failure_reason='grounding_baseline_failed', completed_at=?
                    WHERE extraction_run_id=?""","UPDATE knowledge_builder_extraction_runs SET run_status='failed',\n                    warnings_json=%s, failure_reason='grounding_baseline_failed', completed_at=%s\n                    WHERE extraction_run_id=%s"),
                    (_json([warning]), _now(), extraction_run_id),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_extraction_run(extraction_run_id) or {}

    def get_extraction_run(self, extraction_run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("SELECT * FROM knowledge_builder_extraction_runs WHERE extraction_run_id=?",'SELECT * FROM knowledge_builder_extraction_runs WHERE extraction_run_id=%s'),
            (extraction_run_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["warnings"] = json.loads(result.pop("warnings_json"))
        raw_boundary = result.pop("boundary_validation_json", None)
        try:
            result["boundary_validation"] = (
                json.loads(raw_boundary) if raw_boundary else []
            )
        except json.JSONDecodeError:
            result["boundary_validation"] = []
        result["boundary_review_required"] = bool(
            result.get("boundary_review_required")
        )
        result["candidate_graph_id"] = self._candidate_graph_id(extraction_run_id)
        return result

    def put_reference_concepts(
        self,
        *,
        extraction_run_id: str,
        source_document_hash: str,
        concepts: tuple[ReferenceConcept, ...],
        expressions: tuple[CanonicalExpression, ...] = (),
        candidate_graph_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        run = self.get_extraction_run(extraction_run_id)
        if run is None:
            raise ValueError("extraction run not found")
        concept_ids = {concept.concept_id for concept in concepts}
        if len(concept_ids) != len(concepts):
            raise ValueError("reference concept IDs must be unique")
        unknown = [item.expression_id for item in expressions if item.concept_id not in concept_ids]
        if unknown:
            raise ValueError(f"canonical expressions reference unknown concepts: {unknown}")
        now = _now()
        candidate_graph_hash = None
        if candidate_graph_id is not None:
            graph_table = self._conn.execute(
                sql_query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sensitive_term_candidate_graphs'", "SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname=current_schema() AND tablename='sensitive_term_candidate_graphs'")
            ).fetchone()
            if graph_table is not None:
                graph_row = self._conn.execute(
                    sql_query("SELECT content_hash FROM sensitive_term_candidate_graphs "
                    "WHERE candidate_graph_id=? AND extraction_run_id=?",'SELECT content_hash FROM sensitive_term_candidate_graphs WHERE candidate_graph_id=%s AND extraction_run_id=%s'),
                    (candidate_graph_id, extraction_run_id),
                ).fetchone()
                if graph_row is None:
                    raise ValueError("reference concept candidate graph is missing")
                candidate_graph_hash = str(graph_row["content_hash"])
        with self._lock:
            begin_write(self._conn)
            try:
                for concept in concepts:
                    if candidate_graph_hash is not None:
                        concept = concept.model_copy(update={
                            "candidate_graph_hash": candidate_graph_hash
                        })
                    payload = concept.model_dump(mode="json")
                    digest = content_hash(payload)
                    self._conn.execute(
                        sql_query("""INSERT INTO contextgraph_reference_concepts (
                        concept_id,extraction_run_id,candidate_graph_id,source_document_hash,
                        concept_json,grounding_state,content_hash,created_at
                        ) VALUES (?,?,?,?,?,?,?,?)
                        ON CONFLICT(concept_id) DO UPDATE SET
                        extraction_run_id=excluded.extraction_run_id,
                        candidate_graph_id=excluded.candidate_graph_id,
                        source_document_hash=excluded.source_document_hash,
                        concept_json=excluded.concept_json,
                        grounding_state=excluded.grounding_state,
                        content_hash=excluded.content_hash""",'INSERT INTO contextgraph_reference_concepts (\n                        concept_id,extraction_run_id,candidate_graph_id,source_document_hash,\n                        concept_json,grounding_state,content_hash,created_at\n                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)\n                        ON CONFLICT(concept_id) DO UPDATE SET\n                        extraction_run_id=excluded.extraction_run_id,\n                        candidate_graph_id=excluded.candidate_graph_id,\n                        source_document_hash=excluded.source_document_hash,\n                        concept_json=excluded.concept_json,\n                        grounding_state=excluded.grounding_state,\n                        content_hash=excluded.content_hash'),
                        (
                            concept.concept_id, extraction_run_id, candidate_graph_id,
                            source_document_hash, _json(payload), concept.grounding_state,
                            digest, now,
                        ),
                    )
                for expression in expressions:
                    payload = expression.model_dump(mode="json")
                    digest = content_hash(payload)
                    self._conn.execute(
                        sql_query("""INSERT INTO contextgraph_canonical_expressions (
                        expression_id,concept_id,extraction_run_id,candidate_graph_id,
                        source_document_hash,expression_json,expression_kind,content_hash,created_at
                        ) VALUES (?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(expression_id) DO UPDATE SET
                        concept_id=excluded.concept_id,
                        extraction_run_id=excluded.extraction_run_id,
                        candidate_graph_id=excluded.candidate_graph_id,
                        source_document_hash=excluded.source_document_hash,
                        expression_json=excluded.expression_json,
                        expression_kind=excluded.expression_kind,
                        content_hash=excluded.content_hash""",'INSERT INTO contextgraph_canonical_expressions (\n                        expression_id,concept_id,extraction_run_id,candidate_graph_id,\n                        source_document_hash,expression_json,expression_kind,content_hash,created_at\n                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)\n                        ON CONFLICT(expression_id) DO UPDATE SET\n                        concept_id=excluded.concept_id,\n                        extraction_run_id=excluded.extraction_run_id,\n                        candidate_graph_id=excluded.candidate_graph_id,\n                        source_document_hash=excluded.source_document_hash,\n                        expression_json=excluded.expression_json,\n                        expression_kind=excluded.expression_kind,\n                        content_hash=excluded.content_hash'),
                        (
                            expression.expression_id, expression.concept_id, extraction_run_id,
                            candidate_graph_id, source_document_hash, _json(payload),
                            expression.expression_kind, digest, now,
                        ),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.list_reference_concepts(extraction_run_id=extraction_run_id)

    def append_reference_concept_machine_events(
        self, *, events: tuple[Any, ...], concept_ids: set[str]
    ) -> None:
        if not events or {str(event.case_id) for event in events} != concept_ids:
            raise ValueError(
                "reference concept machine proposal events must cover every concept exactly"
            )
        with self._lock:
            begin_write(self._conn)
            try:
                for event in events:
                    self._append_event_uncommitted(event)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def list_reference_concepts(
        self, *, extraction_run_id: str | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        concept_sql = "SELECT * FROM contextgraph_reference_concepts"
        concept_args: list[Any] = []
        if extraction_run_id:
            concept_sql += sql_query(" WHERE extraction_run_id=?",' WHERE extraction_run_id=%s')
            concept_args.append(extraction_run_id)
        concept_sql += " ORDER BY concept_id"
        concepts = []
        for row in self._conn.execute(concept_sql, tuple(concept_args)).fetchall():
            item = dict(row)
            payload = json.loads(item.pop("concept_json"))
            concepts.append({**item, **payload, "grounding_state": item["grounding_state"]})
        # A split may produce a segment whose canonical label already exists
        # in the same run and candidate graph. Coalesce only that scope;
        # another candidate graph retains its own review identity and state.
        state_rank = {
            "candidate": 0,
            "human_edited": 1,
            "human_accepted": 2,
            "grounded": 2,
            "human_rejected": 3,
        }
        merged_by_label: dict[tuple[str, str, str], dict[str, Any]] = {}
        for concept in concepts:
            label = str(concept.get("canonical_label") or concept.get("concept_id") or "")
            key = (str(concept.get("extraction_run_id") or ""),
                   str(concept.get("candidate_graph_id") or ""), label)
            current = merged_by_label.get(key)
            if current is None:
                merged_by_label[key] = concept
                continue
            current_refs = list(current.get("source_anchor_refs") or [])
            for anchor_ref in concept.get("source_anchor_refs") or []:
                if anchor_ref not in current_refs:
                    current_refs.append(anchor_ref)
            current["source_anchor_refs"] = current_refs
            current_state = str(current.get("grounding_state") or "candidate")
            concept_state = str(concept.get("grounding_state") or "candidate")
            if state_rank.get(concept_state, 0) > state_rank.get(current_state, 0):
                current["grounding_state"] = concept_state
        concepts = list(merged_by_label.values())
        expression_sql = "SELECT * FROM contextgraph_canonical_expressions"
        expression_args: list[Any] = []
        if extraction_run_id:
            expression_sql += sql_query(" WHERE extraction_run_id=?",' WHERE extraction_run_id=%s')
            expression_args.append(extraction_run_id)
        expression_sql += " ORDER BY concept_id, expression_id"
        expressions = []
        for row in self._conn.execute(expression_sql, tuple(expression_args)).fetchall():
            item = dict(row)
            payload = json.loads(item.pop("expression_json"))
            expressions.append({**item, **payload})
        return {"concepts": concepts, "canonical_expressions": expressions}

    def get_reference_concept(self, concept_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("SELECT * FROM contextgraph_reference_concepts WHERE concept_id=?",'SELECT * FROM contextgraph_reference_concepts WHERE concept_id=%s'),
            (concept_id,),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        payload = json.loads(item.pop("concept_json"))
        return {**item, **payload, "grounding_state": item["grounding_state"]}

    def get_reference_concept_owner(self, concept_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("""SELECT r.organization_id, r.workspace_id, c.extraction_run_id,
            c.candidate_graph_id, c.source_document_hash, r.workflow_kind
            FROM contextgraph_reference_concepts c
            JOIN knowledge_builder_extraction_runs r
              ON r.extraction_run_id=c.extraction_run_id
            WHERE c.concept_id=?""",'SELECT r.organization_id, r.workspace_id, c.extraction_run_id,\n            c.candidate_graph_id, c.source_document_hash, r.workflow_kind\n            FROM contextgraph_reference_concepts c\n            JOIN knowledge_builder_extraction_runs r\n              ON r.extraction_run_id=c.extraction_run_id\n            WHERE c.concept_id=%s'),
            (concept_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_reference_concept_review_histories(
        self, concept_ids: list[str] | tuple[str, ...], *, organization_id: str,
        workspace_id: str | None,
    ) -> dict[str, list[dict[str, Any]]]:
        ordered_ids = list(dict.fromkeys(concept_ids))
        histories: dict[str, list[dict[str, Any]]] = {item: [] for item in ordered_ids}
        if not ordered_ids:
            return histories
        placeholders = ",".join(sql_query("?",'%s') for _ in ordered_ids)
        workspace_clause = "r.workspace_id IS NULL" if workspace_id is None else sql_query("r.workspace_id=?",'r.workspace_id=%s')
        args: list[Any] = [*ordered_ids, organization_id]
        if workspace_id is not None:
            args.append(workspace_id)
        rows = self._conn.execute(
            sql_query(f"""SELECT c.concept_id, e.event_json, e.event_hash
            FROM contextgraph_reference_concepts c
            JOIN knowledge_builder_extraction_runs r
              ON r.extraction_run_id=c.extraction_run_id
            LEFT JOIN knowledge_learning_events e ON e.case_id=c.concept_id
            WHERE c.concept_id IN ({placeholders})
              AND r.organization_id=? AND {workspace_clause}
            ORDER BY c.concept_id, e.occurred_at, e.event_id""",f'SELECT c.concept_id, e.event_json, e.event_hash\n            FROM contextgraph_reference_concepts c\n            JOIN knowledge_builder_extraction_runs r\n              ON r.extraction_run_id=c.extraction_run_id\n            LEFT JOIN knowledge_learning_events e ON e.case_id=c.concept_id\n            WHERE c.concept_id IN ({placeholders})\n              AND r.organization_id=%s AND {workspace_clause}\n            ORDER BY c.concept_id, e.occurred_at, e.event_id'),
            tuple(args),
        ).fetchall()
        for row in rows:
            if row["event_json"] is not None:
                histories[row["concept_id"]].append({
                    **json.loads(row["event_json"]), "event_hash": row["event_hash"],
                })
        return histories

    def update_reference_concept_grounding_state(
        self, concept_id: str, grounding_state: str
    ) -> dict[str, Any]:
        if grounding_state not in {
            "candidate", "human_accepted", "human_edited", "human_rejected"
        }:
            raise ValueError("reference concept grounding_state is invalid")
        concept = self.get_reference_concept(concept_id)
        if concept is None:
            raise ValueError("reference concept not found")
        with self._lock:
            self._conn.execute(
                sql_query("""UPDATE contextgraph_reference_concepts
                SET grounding_state=?
                WHERE concept_id=?""",'UPDATE contextgraph_reference_concepts\n                SET grounding_state=%s\n                WHERE concept_id=%s'),
                (grounding_state, concept_id),
            )
            self._conn.commit()
        return self.get_reference_concept(concept_id) or {}

    def replace_reference_concept_with_splits(
        self,
        concept_id: str,
        replacements: tuple[ReferenceConcept, ...],
    ) -> None:
        """Replace the active candidate projection while retaining event history."""

        source = self.get_reference_concept(concept_id)
        if source is None:
            raise ValueError("reference concept not found")
        if not replacements:
            raise ValueError("split replacements are required")
        if len({item.concept_id for item in replacements}) != len(replacements):
            raise ValueError("split replacement IDs must be unique")
        state_rank = {
            "candidate": 0,
            "human_edited": 1,
            "human_accepted": 2,
            "grounded": 2,
            "human_rejected": 3,
        }
        now = _now()
        with self._lock:
            begin_write(self._conn)
            try:
                existing_rows = self._conn.execute(
                    sql_query("SELECT * FROM contextgraph_reference_concepts WHERE extraction_run_id=? AND concept_id!=?",'SELECT * FROM contextgraph_reference_concepts WHERE extraction_run_id=%s AND concept_id!=%s'),
                    (source["extraction_run_id"], concept_id),
                ).fetchall()
                existing_by_label: dict[str, dict[str, Any]] = {}
                for row in existing_rows:
                    item = dict(row)
                    payload = json.loads(item.pop("concept_json"))
                    item.update(payload)
                    label = str(item.get("canonical_label") or "")
                    if label and label not in existing_by_label:
                        existing_by_label[label] = item
                self._conn.execute(
                    sql_query("DELETE FROM contextgraph_canonical_expressions WHERE concept_id=?",'DELETE FROM contextgraph_canonical_expressions WHERE concept_id=%s'),
                    (concept_id,),
                )
                self._conn.execute(
                    sql_query("DELETE FROM contextgraph_reference_concepts WHERE concept_id=?",'DELETE FROM contextgraph_reference_concepts WHERE concept_id=%s'),
                    (concept_id,),
                )
                for concept in replacements:
                    payload = concept.model_dump(mode="json")
                    label = str(payload.get("canonical_label") or "")
                    existing = existing_by_label.get(label)
                    if existing is not None:
                        merged_payload = dict(existing)
                        for key in ("concept_id", "extraction_run_id", "candidate_graph_id", "source_document_hash", "created_at", "grounding_state", "content_hash"):
                            merged_payload.pop(key, None)
                        merged_refs = list(existing.get("source_anchor_refs") or [])
                        for anchor_ref in payload.get("source_anchor_refs") or []:
                            if anchor_ref not in merged_refs:
                                merged_refs.append(anchor_ref)
                        merged_payload["source_anchor_refs"] = merged_refs
                        merged_state = existing.get("grounding_state")
                        if state_rank.get(str(payload.get("grounding_state") or "candidate"), 0) > state_rank.get(str(merged_state or "candidate"), 0):
                            merged_state = payload.get("grounding_state")
                        merged_payload["grounding_state"] = merged_state or "candidate"
                        self._conn.execute(
                            sql_query("UPDATE contextgraph_reference_concepts SET concept_json=?, grounding_state=?, content_hash=? WHERE concept_id=?",'UPDATE contextgraph_reference_concepts SET concept_json=%s, grounding_state=%s, content_hash=%s WHERE concept_id=%s'),
                            (_json(merged_payload), merged_payload["grounding_state"], content_hash(merged_payload), existing["concept_id"]),
                        )
                        continue
                    self._conn.execute(
                        sql_query("""INSERT INTO contextgraph_reference_concepts (
                        concept_id,extraction_run_id,candidate_graph_id,source_document_hash,
                        concept_json,grounding_state,content_hash,created_at
                        ) VALUES (?,?,?,?,?,?,?,?)""",'INSERT INTO contextgraph_reference_concepts (\n                        concept_id,extraction_run_id,candidate_graph_id,source_document_hash,\n                        concept_json,grounding_state,content_hash,created_at\n                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                        (
                            concept.concept_id,
                            source["extraction_run_id"],
                            source.get("candidate_graph_id"),
                            source.get("source_document_hash"),
                            _json(payload),
                            concept.grounding_state,
                            content_hash(payload),
                            now,
                        ),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def valid_reference_concept_evidence_anchor_refs(self, concept_id: str) -> set[str]:
        concept = self.get_reference_concept(concept_id)
        if concept is None:
            return set()
        return {str(ref) for ref in concept.get("source_anchor_refs", [])}

    def append_observed_expression_observation(
        self, observation: ObservedExpressionObservation, *, expression_id: str | None = None
    ) -> dict[str, Any]:
        concept = self._conn.execute(
            sql_query("SELECT 1 FROM contextgraph_reference_concepts WHERE concept_id=?",'SELECT 1 FROM contextgraph_reference_concepts WHERE concept_id=%s'),
            (observation.concept_id,),
        ).fetchone()
        if concept is None:
            raise ValueError("reference concept not found")
        normalized_text = " ".join(observation.text.split()).casefold()
        aggregate_basis = {
            "concept_id": observation.concept_id,
            "normalized_text": normalized_text,
            "semantic_relation": observation.semantic_relation,
            "reference_concept_version_ref": observation.reference_concept_version_ref,
            "accepted_graph_release_id": observation.accepted_graph_release_id,
        }
        aggregate_id = expression_id or f"observed_{content_hash(aggregate_basis)[7:31]}"
        now = _now()
        observation_payload = observation.model_dump(mode="json")
        observation_digest = content_hash(observation_payload)
        with self._lock:
            begin_write(self._conn)
            try:
                self._conn.execute(
                    sql_query("""INSERT INTO contextgraph_observed_expression_observations (
                    observation_id,concept_id,finding_run_id,finding_id,reference_concept_version_ref,
                    accepted_graph_release_id,human_learning_event_id,semantic_relation_decision_hash,
                    observation_json,grounding_state,content_hash,created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",'INSERT INTO contextgraph_observed_expression_observations (\n                    observation_id,concept_id,finding_run_id,finding_id,reference_concept_version_ref,\n                    accepted_graph_release_id,human_learning_event_id,semantic_relation_decision_hash,\n                    observation_json,grounding_state,content_hash,created_at\n                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                    (
                        observation.observation_id, observation.concept_id,
                        observation.finding_run_id, observation.finding_id,
                        observation.reference_concept_version_ref,
                        observation.accepted_graph_release_id,
                        observation.human_learning_event_id,
                        observation.semantic_relation_decision_hash,
                        _json(observation_payload), observation.grounding_state,
                        observation_digest, now,
                    ),
                )
                existing = self._conn.execute(
                    sql_query("""SELECT observation_count FROM contextgraph_observed_expressions
                    WHERE concept_id=? AND normalized_text=? AND semantic_relation=?
                      AND reference_concept_version_ref=? AND accepted_graph_release_id=?""",'SELECT observation_count FROM contextgraph_observed_expressions\n                    WHERE concept_id=%s AND normalized_text=%s AND semantic_relation=%s\n                      AND reference_concept_version_ref=%s AND accepted_graph_release_id=%s'),
                    (
                        observation.concept_id, normalized_text,
                        observation.semantic_relation,
                        observation.reference_concept_version_ref,
                        observation.accepted_graph_release_id,
                    ),
                ).fetchone()
                aggregate = ObservedExpression(
                    schema_version="contextgraph_observed_expression.v1",
                    expression_id=aggregate_id,
                    concept_id=observation.concept_id,
                    normalized_text=normalized_text,
                    semantic_relation=observation.semantic_relation,
                    reference_concept_version_ref=observation.reference_concept_version_ref,
                    accepted_graph_release_id=observation.accepted_graph_release_id,
                    latest_observation_id=observation.observation_id,
                    grounding_state="candidate",
                    observation_count=(existing["observation_count"] if existing else 0) + 1,
                )
                aggregate_payload = aggregate.model_dump(mode="json")
                self._conn.execute(
                    sql_query("""INSERT INTO contextgraph_observed_expressions (
                    expression_id,concept_id,normalized_text,semantic_relation,
                    reference_concept_version_ref,accepted_graph_release_id,
                    latest_observation_id,grounding_state,observation_count,
                    expression_json,content_hash,updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(concept_id,normalized_text,semantic_relation,reference_concept_version_ref,accepted_graph_release_id)
                    DO UPDATE SET
                    latest_observation_id=excluded.latest_observation_id,
                    grounding_state=excluded.grounding_state,
                    observation_count=excluded.observation_count,
                    expression_json=excluded.expression_json,
                    content_hash=excluded.content_hash,
                    updated_at=excluded.updated_at""",'INSERT INTO contextgraph_observed_expressions (\n                    expression_id,concept_id,normalized_text,semantic_relation,\n                    reference_concept_version_ref,accepted_graph_release_id,\n                    latest_observation_id,grounding_state,observation_count,\n                    expression_json,content_hash,updated_at\n                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)\n                    ON CONFLICT(concept_id,normalized_text,semantic_relation,reference_concept_version_ref,accepted_graph_release_id)\n                    DO UPDATE SET\n                    latest_observation_id=excluded.latest_observation_id,\n                    grounding_state=excluded.grounding_state,\n                    observation_count=excluded.observation_count,\n                    expression_json=excluded.expression_json,\n                    content_hash=excluded.content_hash,\n                    updated_at=excluded.updated_at'),
                    (
                        aggregate.expression_id, aggregate.concept_id,
                        aggregate.normalized_text, aggregate.semantic_relation,
                        aggregate.reference_concept_version_ref,
                        aggregate.accepted_graph_release_id,
                        aggregate.latest_observation_id, aggregate.grounding_state,
                        aggregate.observation_count, _json(aggregate_payload),
                        content_hash(aggregate_payload), now,
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_observed_expression(aggregate_id) or {}

    def get_observed_expression(self, expression_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("SELECT * FROM contextgraph_observed_expressions WHERE expression_id=?",'SELECT * FROM contextgraph_observed_expressions WHERE expression_id=%s'),
            (expression_id,),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        payload = json.loads(item.pop("expression_json"))
        return {**item, **payload}

    def list_human_accepted_observed_expressions(
        self,
        *,
        concept_ids: list[str] | None = None,
        accepted_graph_release_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM contextgraph_observed_expressions WHERE grounding_state='human_accepted'"
        args: list[Any] = []
        if concept_ids:
            sql += f" AND concept_id IN ({','.join(sql_query('?','%s') for _ in concept_ids)})"
            args.extend(concept_ids)
        if accepted_graph_release_ids:
            sql += f" AND accepted_graph_release_id IN ({','.join(sql_query('?','%s') for _ in accepted_graph_release_ids)})"
            args.extend(accepted_graph_release_ids)
        sql += " ORDER BY concept_id, normalized_text, expression_id"
        result = []
        for row in self._conn.execute(sql, tuple(args)).fetchall():
            item = dict(row)
            payload = json.loads(item.pop("expression_json"))
            result.append({**item, **payload})
        return result

    def list_observed_expression_observations(self, expression_id: str | None = None) -> list[dict[str, Any]]:
        sql = """SELECT o.* FROM contextgraph_observed_expression_observations o"""
        args: list[Any] = []
        if expression_id:
            sql += sql_query(""" JOIN contextgraph_observed_expressions e
              ON e.latest_observation_id=o.observation_id OR
                 (e.concept_id=o.concept_id
                  AND e.semantic_relation=json_extract(o.observation_json, '$.semantic_relation')
                  AND e.reference_concept_version_ref=o.reference_concept_version_ref
                  AND e.accepted_graph_release_id=o.accepted_graph_release_id)
              WHERE e.expression_id=?"""," JOIN contextgraph_observed_expressions e\n              ON e.latest_observation_id=o.observation_id OR\n                 (e.concept_id=o.concept_id\n                  AND e.semantic_relation=(o.observation_json::jsonb->>'semantic_relation')\n                  AND e.reference_concept_version_ref=o.reference_concept_version_ref\n                  AND e.accepted_graph_release_id=o.accepted_graph_release_id)\n              WHERE e.expression_id=%s")
            args.append(expression_id)
        sql += " ORDER BY o.created_at, o.observation_id"
        rows = self._conn.execute(sql, tuple(args)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            payload = json.loads(item.pop("observation_json"))
            result.append({**item, **payload})
        return result

    def put_ai_grounding_suggestions(
        self, extraction_run_id: str, suggestions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Persist replaceable machine suggestions without changing grounding authority."""
        run = self.get_extraction_run(extraction_run_id)
        if run is None:
            raise ValueError("extraction run not found")
        suggested_at = _now()
        with self._lock:
            begin_write(self._conn)
            try:
                for suggestion in suggestions:
                    payload = {
                        **suggestion,
                        "extraction_run_id": extraction_run_id,
                        "suggested_at": suggested_at,
                    }
                    digest = content_hash(payload)
                    self._conn.execute(
                        sql_query("""INSERT INTO knowledge_builder_ai_grounding_suggestions (
                        assertion_id,extraction_run_id,suggestion_status,confidence,
                        evidence_anchor_count,provider,provider_version,threshold,
                        rationale,suggested_at,content_hash
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(assertion_id) DO UPDATE SET
                        extraction_run_id=excluded.extraction_run_id,
                        suggestion_status=excluded.suggestion_status,
                        confidence=excluded.confidence,
                        evidence_anchor_count=excluded.evidence_anchor_count,
                        provider=excluded.provider,
                        provider_version=excluded.provider_version,
                        threshold=excluded.threshold,rationale=excluded.rationale,
                        suggested_at=excluded.suggested_at,content_hash=excluded.content_hash""",'INSERT INTO knowledge_builder_ai_grounding_suggestions (\n                        assertion_id,extraction_run_id,suggestion_status,confidence,\n                        evidence_anchor_count,provider,provider_version,threshold,\n                        rationale,suggested_at,content_hash\n                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)\n                        ON CONFLICT(assertion_id) DO UPDATE SET\n                        extraction_run_id=excluded.extraction_run_id,\n                        suggestion_status=excluded.suggestion_status,\n                        confidence=excluded.confidence,\n                        evidence_anchor_count=excluded.evidence_anchor_count,\n                        provider=excluded.provider,\n                        provider_version=excluded.provider_version,\n                        threshold=excluded.threshold,rationale=excluded.rationale,\n                        suggested_at=excluded.suggested_at,content_hash=excluded.content_hash'),
                        (
                            suggestion["assertion_id"], extraction_run_id,
                            suggestion["suggestion_status"], suggestion["confidence"],
                            suggestion["evidence_anchor_count"], suggestion["provider"],
                            suggestion["provider_version"], suggestion["threshold"],
                            suggestion["rationale"], suggested_at, digest,
                        ),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.list_ai_grounding_suggestions(extraction_run_id)

    def list_ai_grounding_suggestions(self, extraction_run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            sql_query("""SELECT * FROM knowledge_builder_ai_grounding_suggestions
            WHERE extraction_run_id=? ORDER BY assertion_id""",'SELECT * FROM knowledge_builder_ai_grounding_suggestions\n            WHERE extraction_run_id=%s ORDER BY assertion_id'),
            (extraction_run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def create_ai_grounding_run(
        self, extraction_run_id: str, *, organization_id: str, workspace_id: str | None,
        prompt_ref: str, prompt_hash: str,
    ) -> dict[str, Any]:
        run = self.get_extraction_run(extraction_run_id)
        if run is None or run["organization_id"] != organization_id or run.get("workspace_id") != workspace_id:
            raise ValueError("extraction run not found for tenant")
        run_id = f"ai_ground_{uuid4().hex}"
        started_at = _now()
        payload = {
            "ai_grounding_run_id": run_id, "extraction_run_id": extraction_run_id,
            "organization_id": organization_id, "workspace_id": workspace_id,
            "prompt_ref": prompt_ref, "prompt_hash": prompt_hash,
            "status": "running", "started_at": started_at,
        }
        with self._lock:
            self._conn.execute(
                sql_query("""INSERT INTO knowledge_builder_ai_grounding_runs (
                ai_grounding_run_id,extraction_run_id,organization_id,workspace_id,
                prompt_ref,prompt_hash,status,usage_json,started_at,content_hash
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",'INSERT INTO knowledge_builder_ai_grounding_runs (\n                ai_grounding_run_id,extraction_run_id,organization_id,workspace_id,\n                prompt_ref,prompt_hash,status,usage_json,started_at,content_hash\n                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                (run_id, extraction_run_id, organization_id, workspace_id, prompt_ref,
                 prompt_hash, "running", "{}", started_at, content_hash(payload)),
            )
            self._conn.commit()
        return self.get_ai_grounding_run(run_id) or {}

    def get_ai_grounding_run(self, ai_grounding_run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("SELECT * FROM knowledge_builder_ai_grounding_runs WHERE ai_grounding_run_id=?",'SELECT * FROM knowledge_builder_ai_grounding_runs WHERE ai_grounding_run_id=%s'),
            (ai_grounding_run_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["usage"] = json.loads(result.pop("usage_json"))
        return result

    def complete_ai_grounding_run(
        self, ai_grounding_run_id: str, *, results: list[dict[str, Any]],
        provider: str, model: str, agent_id: str, usage: dict[str, Any],
        latency_ms: int, response_hash: str,
    ) -> dict[str, Any]:
        completed_at = _now()
        with self._lock:
            begin_write(self._conn)
            try:
                for item in results:
                    corrected = {
                        "subject_text": item.get("subject_text"),
                        "predicate_text": item.get("predicate_text"),
                        "object_text": item.get("object_text"),
                    }
                    digest = content_hash({**item, "ai_grounding_run_id": ai_grounding_run_id})
                    self._conn.execute(
                        sql_query("""INSERT INTO knowledge_builder_ai_grounding_results (
                        ai_grounding_run_id,assertion_id,suggestion_status,confidence,
                        corrected_assertion_json,evidence_anchor_refs_json,
                        split_proposals_json,rationale,issues_json,result_hash)
                        VALUES (?,?,?,?,?,?,?,?,?,?)""",'INSERT INTO knowledge_builder_ai_grounding_results (\n                        ai_grounding_run_id,assertion_id,suggestion_status,confidence,\n                        corrected_assertion_json,evidence_anchor_refs_json,\n                        split_proposals_json,rationale,issues_json,result_hash)\n                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                        (ai_grounding_run_id, item["assertion_id"], item["status"],
                         item["confidence"], _json(corrected),
                         _json(item.get("evidence_anchor_refs", [])),
                         _json(item.get("split_proposals", [])), item["rationale"],
                         _json(item.get("issues", [])), digest),
                    )
                payload = {
                    "ai_grounding_run_id": ai_grounding_run_id, "provider": provider,
                    "model": model, "agent_id": agent_id, "usage": usage,
                    "latency_ms": latency_ms, "completed_at": completed_at,
                    "response_hash": response_hash, "status": "completed",
                }
                self._conn.execute(
                    sql_query("""UPDATE knowledge_builder_ai_grounding_runs SET provider=?,model=?,agent_id=?,
                    status='completed',usage_json=?,latency_ms=?,completed_at=?,response_hash=?,
                    content_hash=? WHERE ai_grounding_run_id=? AND status='running'""","UPDATE knowledge_builder_ai_grounding_runs SET provider=%s,model=%s,agent_id=%s,\n                    status='completed',usage_json=%s,latency_ms=%s,completed_at=%s,response_hash=%s,\n                    content_hash=%s WHERE ai_grounding_run_id=%s AND status='running'"),
                    (provider, model, agent_id, _json(usage), latency_ms, completed_at,
                     response_hash, content_hash(payload), ai_grounding_run_id),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_ai_grounding_run(ai_grounding_run_id) or {}

    def fail_ai_grounding_run(self, ai_grounding_run_id: str, *, failure_reason: str,
                              latency_ms: int) -> dict[str, Any]:
        completed_at = _now()
        payload = {"ai_grounding_run_id": ai_grounding_run_id, "status": "failed",
                   "failure_reason": failure_reason, "completed_at": completed_at}
        with self._lock:
            self._conn.execute(
                sql_query("""UPDATE knowledge_builder_ai_grounding_runs SET status='failed',
                failure_reason=?,latency_ms=?,completed_at=?,content_hash=?
                WHERE ai_grounding_run_id=? AND status='running'""","UPDATE knowledge_builder_ai_grounding_runs SET status='failed',\n                failure_reason=%s,latency_ms=%s,completed_at=%s,content_hash=%s\n                WHERE ai_grounding_run_id=%s AND status='running'"),
                (failure_reason, latency_ms, completed_at, content_hash(payload), ai_grounding_run_id),
            )
            self._conn.commit()
        return self.get_ai_grounding_run(ai_grounding_run_id) or {}

    def list_latest_ai_grounding_results(self, extraction_run_id: str) -> dict[str, Any]:
        run = self._conn.execute(
            sql_query("""SELECT * FROM knowledge_builder_ai_grounding_runs
            WHERE extraction_run_id=? AND status='completed'
            ORDER BY completed_at DESC, ai_grounding_run_id DESC LIMIT 1""","SELECT * FROM knowledge_builder_ai_grounding_runs\n            WHERE extraction_run_id=%s AND status='completed'\n            ORDER BY completed_at DESC, ai_grounding_run_id DESC LIMIT 1"),
            (extraction_run_id,),
        ).fetchone()
        if run is None:
            return {
                "schemaVersion": "knowledge_builder_ai_grounding_latest_response.v2",
                "assessmentSource": "legacy_threshold",
                "suggestionSchemaVersion": "knowledge_builder_ai_grounding_suggestion.v1",
                "suggestions": self.list_ai_grounding_suggestions(extraction_run_id),
            }
        rows = self._conn.execute(
            sql_query("""SELECT gr.* FROM knowledge_builder_ai_grounding_results gr
            JOIN knowledge_builder_ai_grounding_runs r
              ON r.ai_grounding_run_id=gr.ai_grounding_run_id
            WHERE r.extraction_run_id=? AND r.status='completed'
            ORDER BY gr.assertion_id, r.completed_at DESC, r.ai_grounding_run_id DESC""","SELECT gr.* FROM knowledge_builder_ai_grounding_results gr\n            JOIN knowledge_builder_ai_grounding_runs r\n              ON r.ai_grounding_run_id=gr.ai_grounding_run_id\n            WHERE r.extraction_run_id=%s AND r.status='completed'\n            ORDER BY gr.assertion_id, r.completed_at DESC, r.ai_grounding_run_id DESC"),
            (extraction_run_id,),
        ).fetchall()
        suggestions = []
        seen_assertions: set[str] = set()
        for row in rows:
            item = dict(row)
            assertion_id = item["assertion_id"]
            if assertion_id in seen_assertions:
                continue
            seen_assertions.add(assertion_id)
            item["status"] = item.pop("suggestion_status")
            item["corrected_assertion"] = json.loads(item.pop("corrected_assertion_json"))
            item["evidence_anchor_refs"] = json.loads(item.pop("evidence_anchor_refs_json"))
            item["split_proposals"] = json.loads(item.pop("split_proposals_json", "[]"))
            item["issues"] = json.loads(item.pop("issues_json"))
            suggestions.append(item)
        suggestions.sort(key=lambda item: item["assertion_id"])
        return {
            "schemaVersion": "knowledge_builder_ai_grounding_latest_response.v2",
            "assessmentSource": "llm_structured",
            "suggestionSchemaVersion": "semantier.ai_grounding_result.v1",
            "aiGroundingRun": self.get_ai_grounding_run(run["ai_grounding_run_id"]),
            "suggestions": suggestions,
        }

    def append_human_accept_batch(
        self, *, events: list[Any], batch_id: str, organization_id: str,
        workspace_id: str | None, extraction_run_id: str, idempotency_key: str,
    ) -> dict[str, Any]:
        existing = self._conn.execute(
            sql_query("""SELECT response_json FROM knowledge_builder_grounding_batches
            WHERE organization_id=? AND workspace_id IS ? AND idempotency_key=?""",'SELECT response_json FROM knowledge_builder_grounding_batches\n            WHERE organization_id=%s AND workspace_id IS NOT DISTINCT FROM %s AND idempotency_key=%s'),
            (organization_id, workspace_id, idempotency_key),
        ).fetchone()
        if existing is not None:
            return json.loads(existing["response_json"])
        with self._lock:
            begin_write(self._conn)
            try:
                persisted = [self._append_event_uncommitted(event) for event in events]
                response = {
                    "ok": True,
                    "schemaVersion": "learning_event_grounding_batch_response.v1",
                    "batchId": batch_id,
                    "decision": "accept",
                    "acceptedCount": len(persisted),
                    "learningEvents": persisted,
                    "authorityState": "human_grounded_not_released",
                }
                created_at = _now()
                payload = {**response, "organization_id": organization_id,
                           "workspace_id": workspace_id, "created_at": created_at}
                self._conn.execute(
                    sql_query("""INSERT INTO knowledge_builder_grounding_batches (
                    batch_id,organization_id,workspace_id,extraction_run_id,idempotency_key,
                    decision,response_json,created_at,content_hash) VALUES (?,?,?,?,?,?,?,?,?)""",'INSERT INTO knowledge_builder_grounding_batches (\n                    batch_id,organization_id,workspace_id,extraction_run_id,idempotency_key,\n                    decision,response_json,created_at,content_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                    (batch_id, organization_id, workspace_id, extraction_run_id,
                     idempotency_key, "accept", _json(response), created_at,
                     content_hash(payload)),
                )
                for ordinal, event in enumerate(events):
                    self._conn.execute(
                        sql_query("""INSERT INTO knowledge_builder_grounding_batch_items (
                        batch_id,event_id,assertion_id,ordinal) VALUES (?,?,?,?)""",'INSERT INTO knowledge_builder_grounding_batch_items (\n                        batch_id,event_id,assertion_id,ordinal) VALUES (%s,%s,%s,%s)'),
                        (batch_id, event.event_id, event.case_id, ordinal),
                    )
                self._conn.commit()
                return response
            except Exception:
                self._conn.rollback()
                raise

    def _candidate_graph_id(self, extraction_run_id: str) -> str | None:
        try:
            graph = self._conn.execute(
                sql_query("SELECT candidate_graph_id FROM sensitive_term_candidate_graphs WHERE extraction_run_id=? LIMIT 1",'SELECT candidate_graph_id FROM sensitive_term_candidate_graphs WHERE extraction_run_id=%s LIMIT 1'),
                (extraction_run_id,),
            ).fetchone()
        except Exception as exc:
            if "no such table" not in str(exc):
                raise
            return None
        return graph["candidate_graph_id"] if graph else None

    def list_extraction_runs(self, *, organization_id: str, workspace_id: str | None = None,
                             source_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        sql = sql_query("SELECT * FROM knowledge_builder_extraction_runs WHERE organization_id=?",'SELECT * FROM knowledge_builder_extraction_runs WHERE organization_id=%s')
        args: list[Any] = [organization_id]
        if workspace_id is not None:
            sql += sql_query(" AND workspace_id=?",' AND workspace_id=%s'); args.append(workspace_id)
        if source_id:
            sql += sql_query(" AND source_id=?",' AND source_id=%s'); args.append(source_id)
        sql += sql_query(" ORDER BY started_at DESC, extraction_run_id DESC LIMIT ?",' ORDER BY started_at DESC, extraction_run_id DESC LIMIT %s'); args.append(limit)
        result = []
        for row in self._conn.execute(sql, tuple(args)).fetchall():
            item = dict(row)
            item["warnings"] = json.loads(item.pop("warnings_json"))
            item["candidate_graph_id"] = self._candidate_graph_id(item["extraction_run_id"])
            result.append(item)
        return result

    def list_occurrences(self, extraction_run_id: str) -> list[SemanticOccurrence]:
        rows = self._conn.execute(
            sql_query("SELECT occurrence_json FROM knowledge_builder_semantic_occurrences WHERE extraction_run_id=? ORDER BY occurrence_id",'SELECT occurrence_json FROM knowledge_builder_semantic_occurrences WHERE extraction_run_id=%s ORDER BY occurrence_id'),
            (extraction_run_id,),
        ).fetchall()
        return [SemanticOccurrence.model_validate_json(row["occurrence_json"]) for row in rows]

    def get_assertion(self, assertion_id: str) -> DocumentCandidateAssertion | None:
        row = self._conn.execute(
            sql_query("SELECT assertion_json FROM knowledge_builder_document_candidate_assertions WHERE assertion_id=?",'SELECT assertion_json FROM knowledge_builder_document_candidate_assertions WHERE assertion_id=%s'),
            (assertion_id,),
        ).fetchone()
        if not row:
            return None
        assertion = DocumentCandidateAssertion.model_validate_json(row["assertion_json"])
        binding = self.get_latest_assertion_grounding_binding(assertion_id)
        if not binding:
            return assertion
        return assertion.model_copy(update={
            "grounding_state": GroundingState.GROUNDED,
            "accepted_graph_node_id": next(iter(binding["accepted_node_ids"]), None),
            "accepted_graph_edge_id": next(iter(binding["accepted_edge_ids"]), None),
        })

    def get_latest_assertion_grounding_binding(self, assertion_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("""SELECT * FROM knowledge_builder_assertion_grounding_bindings
            WHERE assertion_id=? ORDER BY binding_id DESC LIMIT 1""",'SELECT * FROM knowledge_builder_assertion_grounding_bindings\n            WHERE assertion_id=%s ORDER BY binding_id DESC LIMIT 1'), (assertion_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        for name in ("accepted_node_ids", "accepted_edge_ids", "accepted_rule_ids"):
            result[name] = json.loads(result.pop(f"{name}_json"))
        return result

    def get_assertion_topologies(self, assertion_id: str) -> dict[str, Any]:
        assertion = self.get_assertion(assertion_id)
        if assertion is None:
            raise ValueError(f"assertion candidate not found: {assertion_id}")
        normalized = assertion.normalized_assertion.model_dump(mode="json")
        subject_id = f"candidate_subject_{assertion_id}"
        object_id = f"candidate_object_{assertion_id}"
        relation_id = normalized.get("candidate_relation_id")
        source_anchor_refs = [item.anchor_id for item in assertion.source_anchors]
        candidate = {
            "schema_version": "candidate_assertion_topology.v1",
            "assertion_id": assertion_id,
            "nodes": [
                {"node_id": subject_id, **(normalized.get("subject") or {}),
                 "source_anchor_refs": source_anchor_refs},
                {"node_id": object_id, **(normalized.get("object") or {}),
                 "source_anchor_refs": source_anchor_refs},
            ] if normalized.get("subject") and normalized.get("object") else [],
            "edges": [{
                "edge_id": relation_id, "source_node_id": subject_id,
                "target_node_id": object_id, "relation_type": normalized.get("predicate"),
                "source_anchor_refs": source_anchor_refs,
            }] if relation_id and normalized.get("subject") and normalized.get("object") else [],
        }
        binding = self.get_latest_assertion_grounding_binding(assertion_id)
        if not binding:
            return {"candidate_topology": candidate, "accepted_topology": None}
        release_row = self._conn.execute(
            sql_query("""SELECT r.release_json,r.content_hash,r.organization_id,r.workspace_id,
            (SELECT status FROM tender_graph_release_materialization_events e
             WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1) status
            FROM tender_graph_releases r WHERE r.graph_version=?""",'SELECT r.release_json,r.content_hash,r.organization_id,r.workspace_id,\n            (SELECT status FROM tender_graph_release_materialization_events e\n             WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1) status\n            FROM tender_graph_releases r WHERE r.graph_version=%s'),
            (binding["graph_version"],),
        ).fetchone()
        owner = self.get_assertion_owner(assertion_id) or {}
        if (release_row is None or release_row["status"] != "materialized" or
                release_row["organization_id"] != owner.get("organization_id") or
                release_row["workspace_id"] != owner.get("workspace_id")):
            raise ValueError("accepted graph binding scope or lifecycle mismatch")
        release = json.loads(release_row["release_json"])
        if (release_row["content_hash"] != binding["graph_release_hash"] or
                release.get("release_hash") != binding["graph_release_hash"] or
                content_hash({k: v for k, v in release.items() if k != "release_hash"}) != binding["graph_release_hash"]):
            raise ValueError("accepted graph release hash mismatch")
        accepted = {
            "schema_version": "accepted_assertion_topology.v1",
            "assertion_id": assertion_id,
            "graph_version": binding["graph_version"],
            "graph_release_hash": binding["graph_release_hash"],
            "materialized_at": binding["materialized_at"],
            "nodes": [item for item in release["nodes"] if item.get("node_id") in binding["accepted_node_ids"]],
            "edges": [item for item in release["edges"] if item.get("edge_id") in binding["accepted_edge_ids"]],
            "rules": [item for item in release["rules"] if item.get("rule_id") in binding["accepted_rule_ids"]],
            "evidence_refs": sorted({ref for collection in (release["nodes"], release["edges"], release["rules"])
                                      for item in collection for ref in item.get("source_anchor_refs", [])
                                      if (item.get("node_id") in binding["accepted_node_ids"] or
                                          item.get("edge_id") in binding["accepted_edge_ids"] or
                                          item.get("rule_id") in binding["accepted_rule_ids"])}),
        }
        return {"candidate_topology": candidate, "accepted_topology": accepted}

    def get_assertion_owner(self, assertion_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("""SELECT r.organization_id, r.workspace_id, a.extraction_run_id
            FROM knowledge_builder_document_candidate_assertions a
            JOIN knowledge_builder_extraction_runs r ON r.extraction_run_id=a.extraction_run_id
            WHERE a.assertion_id=?""",'SELECT r.organization_id, r.workspace_id, a.extraction_run_id\n            FROM knowledge_builder_document_candidate_assertions a\n            JOIN knowledge_builder_extraction_runs r ON r.extraction_run_id=a.extraction_run_id\n            WHERE a.assertion_id=%s'),
            (assertion_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_candidate_graph_owner(self, candidate_graph_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("""SELECT c.organization_id, c.workspace_id,
            r.source_id, r.lineage_id, r.extraction_run_id, r.source_identity_ref,
            s.source_hash AS materialization_source_hash,
            r.workflow_kind,
            COALESCE((SELECT r.graph_version FROM tender_graph_releases r
              WHERE r.candidate_graph_id=c.candidate_graph_id AND
                COALESCE((SELECT e.status FROM tender_graph_release_materialization_events e
                  WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),
                  r.materialization_status)='materialized'
              ORDER BY r.rowid DESC LIMIT 1), c.allowed_base_graph_version) AS allowed_base_graph_version
            FROM sensitive_term_candidate_graphs c
            JOIN knowledge_builder_extraction_runs r
              ON r.extraction_run_id=c.extraction_run_id
             AND r.organization_id=c.organization_id
             AND r.workspace_id IS c.workspace_id
            JOIN knowledge_builder_sources s
              ON s.source_id=r.source_id
             AND s.organization_id=r.organization_id
             AND s.workspace_id IS r.workspace_id
            WHERE c.candidate_graph_id=?""","SELECT c.organization_id, c.workspace_id,\n            r.source_id, r.lineage_id, r.extraction_run_id, r.source_identity_ref,\n            s.source_hash AS materialization_source_hash,\n            r.workflow_kind,\n            COALESCE((SELECT r.graph_version FROM tender_graph_releases r\n              WHERE r.candidate_graph_id=c.candidate_graph_id AND\n                COALESCE((SELECT e.status FROM tender_graph_release_materialization_events e\n                  WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),\n                  r.materialization_status)='materialized'\n              ORDER BY r.release_sequence DESC LIMIT 1), c.allowed_base_graph_version) AS allowed_base_graph_version\n            FROM sensitive_term_candidate_graphs c\n            JOIN knowledge_builder_extraction_runs r\n              ON r.extraction_run_id=c.extraction_run_id\n             AND r.organization_id=c.organization_id\n             AND r.workspace_id IS NOT DISTINCT FROM c.workspace_id\n            JOIN knowledge_builder_sources s\n              ON s.source_id=r.source_id\n             AND s.organization_id=r.organization_id\n             AND s.workspace_id IS NOT DISTINCT FROM r.workspace_id\n            WHERE c.candidate_graph_id=%s"),
            (candidate_graph_id,),
        ).fetchone()
        return dict(row) if row else None

    def record_release_artifact_origin(
        self,
        *,
        graph_version: str,
        candidate_graph_id: str,
        source_id: str,
        lineage_id: str,
        extraction_run_id: str,
        organization_id: str,
        workspace_id: str,
        source_identity_ref: str | None,
        release_hash: str,
        release_lineage: AcceptedGraphReleaseLineage | dict[str, Any] | None = None,
        authority_state: str = "authoritative",
    ) -> None:
        """Idempotently bind an accepted release artifact to its governed source."""
        try:
            record_accepted_release_origin(
                self._conn,
                graph_version=graph_version,
                candidate_graph_id=candidate_graph_id,
                source_id=source_id,
                lineage_id=lineage_id,
                extraction_run_id=extraction_run_id,
                organization_id=organization_id,
                workspace_id=workspace_id,
                source_identity_ref=source_identity_ref,
                release_hash=release_hash,
                release_lineage=release_lineage,
                created_at=_now(),
                authority_state=authority_state,
            )
            self._conn.commit()
        except Exception:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise


    def get_release_artifact_origin_lineage(
        self,
        *,
        graph_version: str,
        organization_id: str,
        workspace_id: str,
        workspace_root: str | Path | None = None,
        require_materialized_checkpoint: bool = True,
    ) -> dict[str, Any]:
        """Read exact v3 release lineage, or an explicit legacy classification.

        Historical rows are never enriched with an inferred GraphIntent.
        """
        row = self._conn.execute(
            sql_query("""SELECT * FROM contextgraph_artifact_origins
            WHERE artifact_id=? AND organization_id=? AND workspace_id=?
              AND relation_kind='accepted_release'
            ORDER BY created_at, origin_id LIMIT 1""","SELECT * FROM contextgraph_artifact_origins\n            WHERE artifact_id=%s AND organization_id=%s AND workspace_id=%s\n              AND relation_kind='accepted_release'\n            ORDER BY created_at, origin_id LIMIT 1"),
            (graph_version, organization_id, workspace_id),
        ).fetchone()
        if row is None:
            raise ValueError("accepted release artifact origin not found")
        release_row = self._conn.execute(
            sql_query("""SELECT release_json,content_hash,organization_id,workspace_id,
            lineage_contract,
            COALESCE((SELECT status FROM tender_graph_release_materialization_events e
              WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),
              r.materialization_status) AS status
            FROM tender_graph_releases r WHERE graph_version=?""",'SELECT release_json,content_hash,organization_id,workspace_id,\n            lineage_contract,\n            COALESCE((SELECT status FROM tender_graph_release_materialization_events e\n              WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),\n              r.materialization_status) AS status\n            FROM tender_graph_releases r WHERE graph_version=%s'),
            (graph_version,),
        ).fetchone()
        if release_row is None:
            raise ValueError("accepted release not found during lineage replay")
        lineage_contract = str(release_row["lineage_contract"] or "legacy")
        if not row["release_lineage_json"] and not row["release_lineage_hash"]:
            if lineage_contract == "exact_v3":
                raise ValueError("exact v3 release lineage metadata is missing")
            if lineage_contract != "legacy":
                raise ValueError("accepted release lineage contract is invalid")
            if row["authority_state"] != "authoritative":
                raise ValueError("legacy accepted release origin is not authoritative")
            from contracts.knowledge_graph_phase1 import AcceptedGraphRelease

            release_payload = json.loads(release_row["release_json"])
            release_hash = str(release_payload.pop("release_hash", ""))
            release = AcceptedGraphRelease.model_validate(release_payload)
            legacy_origin_payload = {
                "artifact_id": row["artifact_id"],
                "candidate_graph_id": row["candidate_graph_id"],
                "extraction_run_id": row["extraction_run_id"],
                "lineage_id": row["lineage_id"],
                "organization_id": row["organization_id"],
                "release_hash": release_hash,
                "source_id": row["source_id"],
                "workspace_id": row["workspace_id"],
            }
            if (
                release_row["organization_id"] != organization_id
                or release_row["workspace_id"] != workspace_id
                or release_row["status"] != "materialized"
                or release_hash != release_row["content_hash"]
                or release.release_hash() != release_hash
                or row["content_hash"] != content_hash(legacy_origin_payload)
            ):
                raise ValueError("legacy accepted release origin verification failed")
            return {
                "provenance_status": "legacy",
                "lineage": None,
                "reason": (
                    "historical release has no GraphIntent lineage; provenance is not synthesized"
                ),
            }
        if not row["release_lineage_json"] or not row["release_lineage_hash"]:
            raise ValueError("release lineage metadata is incomplete")
        if lineage_contract != "exact_v3":
            raise ValueError("exact release lineage contract is not persisted")
        if require_materialized_checkpoint:
            if row["authority_state"] != "authoritative":
                raise ValueError("accepted release origin is not authoritative")
        elif row["authority_state"] not in {"candidate", "authoritative"}:
            raise ValueError("accepted release origin authority state is invalid")
        try:
            lineage = AcceptedGraphReleaseLineage.from_dict(
                json.loads(row["release_lineage_json"])
            )
        except (TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("release lineage metadata is invalid") from exc
        if lineage.lineage_hash != row["release_lineage_hash"]:
            raise ValueError("release lineage hash mismatch")
        exact_pins = {
            "accepted release ref": (
                lineage.accepted_graph_release_ref,
                row["artifact_id"],
            ),
            "candidate graph": (lineage.candidate_graph_id, row["candidate_graph_id"]),
            "organization": (lineage.organization_id, row["organization_id"]),
            "workspace": (lineage.workspace_id, row["workspace_id"]),
            "source identity": (
                lineage.source_identity_ref,
                row["source_identity_ref"],
            ),
            "candidate graph materialization extraction run": (
                lineage.candidate_graph_materialization_extract_run_ref,
                row["extraction_run_id"],
            ),
        }
        for label, (actual, expected) in exact_pins.items():
            if actual != expected:
                raise ValueError(f"{label} mismatch in persisted release lineage")
        origin_payload = {
            "artifact_id": row["artifact_id"],
            "candidate_graph_id": row["candidate_graph_id"],
            "extraction_run_id": row["extraction_run_id"],
            "lineage_id": row["lineage_id"],
            "organization_id": row["organization_id"],
            "release_hash": lineage.accepted_graph_release_hash,
            "source_id": row["source_id"],
            "workspace_id": row["workspace_id"],
            "source_identity_ref": row["source_identity_ref"],
            "release_lineage_hash": row["release_lineage_hash"],
        }
        if row["content_hash"] != content_hash(origin_payload):
            raise ValueError("release lineage origin content hash mismatch")

        if workspace_root is None:
            raise ValueError(
                "governed workspace root is required to verify exact release lineage"
            )
        requested_workspace_root = Path(workspace_root).resolve()
        root = (
            requested_workspace_root.parent.parent
            if requested_workspace_root.name == workspace_id
            and requested_workspace_root.parent.name == "workspaces"
            else requested_workspace_root
        )
        workspace_dir = root / "workspaces" / workspace_id
        extract_path = (
            workspace_dir
            / "runs"
            / lineage.schema_validation_extract_run_ref
            / "artifacts"
            / "extract_result.v1.json"
        ).resolve()
        runs_root = (workspace_dir / "runs").resolve()
        if runs_root not in extract_path.parents or not extract_path.is_file():
            raise ValueError("governed ExtractResult not found during lineage replay")

        from contracts.governed_extract import ExtractResult
        from contracts.knowledge_graph_phase1 import AcceptedGraphRelease
        from contracts.corpus_graph_schema import CorpusGraphSchemaCandidate
        from contracts.schema_decision import SchemaDecision
        from storage.corpus_graph_schema_candidate_store import (
            CorpusGraphSchemaCandidateStore,
        )
        from storage.graph_intent_brief_store import GraphIntentBriefStore
        from storage.knowledge_builder_store import KnowledgeBuilderStore
        from storage.schema_decision_store import SchemaDecisionStore
        from services.accepted_graph_release_checkpoint import (
            verify_accepted_release_checkpoint,
        )
        from semantier.semantica_context_graph.artifact_repository import (
            ContextGraphArtifactRepository,
        )
        from semantier.semantica_context_graph.native_builder import (
            native_graph_from_accepted_release,
        )

        extract_result = ExtractResult.from_dict(
            json.loads(extract_path.read_text(encoding="utf-8"))
        )
        if (
            extract_result.extract_run_id
            != lineage.schema_validation_extract_run_ref
            or extract_result.extract_result_hash
            != lineage.schema_validation_extract_result_hash
        ):
            raise ValueError("ExtractResult exact pin mismatch during lineage replay")
        candidate = CorpusGraphSchemaCandidate.from_dict(
            CorpusGraphSchemaCandidateStore(
                root, workspace_id=workspace_id
            ).get_by_hash(lineage.schema_candidate_hash)
        )
        decision = SchemaDecision.from_dict(
            SchemaDecisionStore(root, workspace_id=workspace_id).get(
                lineage.schema_decision_ref
            )
        )

        graph_row = self._conn.execute(
            sql_query("""SELECT graph_json,content_hash,organization_id,workspace_id
            FROM sensitive_term_candidate_graphs WHERE candidate_graph_id=?""",'SELECT graph_json,content_hash,organization_id,workspace_id\n            FROM sensitive_term_candidate_graphs WHERE candidate_graph_id=%s'),
            (lineage.candidate_graph_id,),
        ).fetchone()
        if graph_row is None:
            raise ValueError("candidate graph not found during lineage replay")
        candidate_graph = SensitiveTermCandidateGraph.model_validate_json(
            graph_row["graph_json"]
        )
        candidate_owner = self.get_candidate_graph_owner(lineage.candidate_graph_id)
        expected_owner = {
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "source_identity_ref": lineage.source_identity_ref,
            "extraction_run_id": (
                lineage.candidate_graph_materialization_extract_run_ref
            ),
            "materialization_source_hash": candidate_graph.source_document_hash,
        }
        if (
            candidate_owner is None
            or any(
                candidate_owner.get(field) != expected
                for field, expected in expected_owner.items()
            )
            or graph_row["organization_id"] != organization_id
            or graph_row["workspace_id"] != workspace_id
            or graph_row["content_hash"] != candidate_graph.graph_hash()
            or lineage.candidate_graph_hash != candidate_graph.graph_hash()
            or candidate_graph.extraction_run_id
            != lineage.candidate_graph_materialization_extract_run_ref
        ):
            raise ValueError("candidate graph exact pin mismatch during lineage replay")

        release_payload = json.loads(release_row["release_json"])
        release_hash = str(release_payload.pop("release_hash", ""))
        release = AcceptedGraphRelease.model_validate(release_payload)
        if (
            release_row["organization_id"] != organization_id
            or release_row["workspace_id"] != workspace_id
            or (
                require_materialized_checkpoint
                and release_row["status"] != "materialized"
            )
            or release_hash != lineage.accepted_graph_release_hash
            or release_row["content_hash"] != release_hash
            or release.release_hash() != release_hash
        ):
            raise ValueError("accepted release exact pin mismatch during lineage replay")
        accepted_release = {**release_payload, "release_hash": release_hash}
        if require_materialized_checkpoint:
            checkpoint_repository = ContextGraphArtifactRepository(
                workspace_dir,
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
            verify_accepted_release_checkpoint(
                checkpoint_repository,
                graph=native_graph_from_accepted_release(accepted_release),
                release=accepted_release,
                lineage_id=str(row["lineage_id"]),
                release_lineage=lineage,
            )

        source_registry = KnowledgeBuilderStore(self._db_path, workspace_root=root)
        try:
            intent_store = GraphIntentBriefStore(
                root,
                organization_id=organization_id,
                workspace_id=workspace_id,
                source_registry=source_registry,
            )
            graph_intent = intent_store.get(
                lineage.graph_intent_ref,
                graph_intent_hash=lineage.graph_intent_hash,
            )
            if (
                graph_intent.review.state != lineage.graph_intent_review_state
                or graph_intent.source_binding.as_dict()
                != dict(candidate.source_binding or {})
            ):
                raise ValueError("GraphIntent exact pin mismatch during lineage replay")
            source_identity = source_registry.get_source_identity(
                source_identity_ref=lineage.source_identity_ref,
                organization_id=organization_id,
                workspace_id=workspace_id,
                lineage_id=lineage.lineage_id,
            )
            if source_identity is None:
                raise ValueError("governed SourceIdentity not found during lineage replay")
            intent_store.verify_source_binding(
                graph_intent.source_binding,
                source_identity=source_identity,
            )
        finally:
            source_registry.close()

        rebuilt = AcceptedGraphReleaseLineage.create(
            candidate=candidate,
            decision=decision,
            extract_result=extract_result,
            candidate_graph=candidate_graph,
            accepted_release=accepted_release,
            candidate_graph_id=lineage.candidate_graph_id,
        )
        if rebuilt != lineage:
            raise ValueError("release lineage replay chain mismatch")
        return {
            "provenance_status": (
                "exact_v3"
                if require_materialized_checkpoint
                else "validated_v3_retry"
            ),
            "lineage": lineage.as_dict(),
        }

    def list_assertions(self, extraction_run_id: str | None = None, *, organization_id: str | None = None,
                        workspace_id: str | None = None, limit: int = 100, cursor: str | None = None) -> list[DocumentCandidateAssertion]:
        if not 1 <= limit <= 501:
            raise ValueError("limit must be between 1 and 501")
        sql = """SELECT a.assertion_json FROM knowledge_builder_document_candidate_assertions a
        JOIN knowledge_builder_extraction_runs r ON r.extraction_run_id=a.extraction_run_id WHERE 1=1"""
        args: list[Any] = []
        if extraction_run_id:
            sql += sql_query(" AND a.extraction_run_id=?",' AND a.extraction_run_id=%s'); args.append(extraction_run_id)
        if organization_id is not None:
            sql += sql_query(" AND r.organization_id=?",' AND r.organization_id=%s'); args.append(organization_id)
            if workspace_id is None:
                sql += " AND r.workspace_id IS NULL"
        if workspace_id is not None:
            sql += sql_query(" AND r.workspace_id=?",' AND r.workspace_id=%s'); args.append(workspace_id)
        if cursor:
            sql += sql_query(" AND a.assertion_id>?",' AND a.assertion_id>%s'); args.append(cursor)
        sql += sql_query(" ORDER BY a.assertion_id LIMIT ?",' ORDER BY a.assertion_id LIMIT %s'); args.append(limit)
        rows = self._conn.execute(sql, tuple(args)).fetchall()
        raw = [DocumentCandidateAssertion.model_validate_json(row["assertion_json"]) for row in rows]
        return [self.get_assertion(item.assertion_id) or item for item in raw]

    def list_assertion_review_histories(
        self, assertion_ids: list[str] | tuple[str, ...], *, organization_id: str,
        workspace_id: str | None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Load append-only histories for one already-bounded assertion page."""
        ordered_ids = list(dict.fromkeys(assertion_ids))
        histories: dict[str, list[dict[str, Any]]] = {item: [] for item in ordered_ids}
        if not ordered_ids:
            return histories
        placeholders = ",".join(sql_query("?",'%s') for _ in ordered_ids)
        workspace_clause = "r.workspace_id IS NULL" if workspace_id is None else sql_query("r.workspace_id=?",'r.workspace_id=%s')
        args: list[Any] = [*ordered_ids, organization_id]
        if workspace_id is not None:
            args.append(workspace_id)
        rows = self._conn.execute(
            sql_query(f"""SELECT a.assertion_id, e.event_json, e.event_hash
            FROM knowledge_builder_document_candidate_assertions a
            JOIN knowledge_builder_extraction_runs r
              ON r.extraction_run_id=a.extraction_run_id
            LEFT JOIN knowledge_learning_events e ON e.case_id=a.assertion_id
            WHERE a.assertion_id IN ({placeholders})
              AND r.organization_id=? AND {workspace_clause}
            ORDER BY a.assertion_id, e.occurred_at, e.event_id""",f'SELECT a.assertion_id, e.event_json, e.event_hash\n            FROM knowledge_builder_document_candidate_assertions a\n            JOIN knowledge_builder_extraction_runs r\n              ON r.extraction_run_id=a.extraction_run_id\n            LEFT JOIN knowledge_learning_events e ON e.case_id=a.assertion_id\n            WHERE a.assertion_id IN ({placeholders})\n              AND r.organization_id=%s AND {workspace_clause}\n            ORDER BY a.assertion_id, e.occurred_at, e.event_id'),
            tuple(args),
        ).fetchall()
        for row in rows:
            if row["event_json"] is not None:
                histories[row["assertion_id"]].append({
                    **json.loads(row["event_json"]), "event_hash": row["event_hash"],
                })
        return histories

    def assertion_occurrence_ids(self, assertion_id: str) -> list[str]:
        return [row["occurrence_id"] for row in self._conn.execute(
            sql_query("SELECT occurrence_id FROM knowledge_builder_assertion_occurrences WHERE assertion_id=? ORDER BY ordinal",'SELECT occurrence_id FROM knowledge_builder_assertion_occurrences WHERE assertion_id=%s ORDER BY ordinal'),
            (assertion_id,),
        ).fetchall()]

    def list_assertion_occurrences(self, assertion_id: str) -> list[SemanticOccurrence]:
        rows = self._conn.execute(
            sql_query("""SELECT o.occurrence_json FROM knowledge_builder_assertion_occurrences ao
            JOIN knowledge_builder_semantic_occurrences o ON o.occurrence_id=ao.occurrence_id
            WHERE ao.assertion_id=? ORDER BY ao.ordinal""",'SELECT o.occurrence_json FROM knowledge_builder_assertion_occurrences ao\n            JOIN knowledge_builder_semantic_occurrences o ON o.occurrence_id=ao.occurrence_id\n            WHERE ao.assertion_id=%s ORDER BY ao.ordinal'), (assertion_id,),
        ).fetchall()
        return [SemanticOccurrence.model_validate_json(row["occurrence_json"]) for row in rows]

    def put_knowledge_scope(
        self, *, discovery_run_id: str, organization_id: str,
        workspace_id: str | None, scope: KnowledgeScope,
    ) -> dict[str, Any]:
        payload = scope.model_dump(mode="json")
        digest = content_hash(payload)
        self._insert_immutable(
            table="knowledge_builder_knowledge_scopes",
            identity_where=sql_query("package_id = ?",'package_id = %s'), identity=(scope.package_id,),
            insert_sql=sql_query("""INSERT INTO knowledge_builder_knowledge_scopes
            (package_id, discovery_run_id, organization_id, workspace_id, scope_json, content_hash)
            VALUES (?, ?, ?, ?, ?, ?)""",'INSERT INTO knowledge_builder_knowledge_scopes\n            (package_id, discovery_run_id, organization_id, workspace_id, scope_json, content_hash)\n            VALUES (%s, %s, %s, %s, %s, %s)'),
            values=(scope.package_id, discovery_run_id, organization_id, workspace_id, _json(payload), digest),
            expected_hash=digest,

            postgres_select_sql='SELECT content_hash FROM knowledge_builder_knowledge_scopes WHERE package_id = %s',
            postgres_insert_sql='INSERT INTO knowledge_builder_knowledge_scopes\n            (package_id, discovery_run_id, organization_id, workspace_id, scope_json, content_hash)\n            VALUES (%s, %s, %s, %s, %s, %s)',
        )
        return {**payload, "content_hash": digest}

    def get_knowledge_scope(self, package_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("SELECT scope_json, discovery_run_id, organization_id, workspace_id, content_hash FROM knowledge_builder_knowledge_scopes WHERE package_id=?",'SELECT scope_json, discovery_run_id, organization_id, workspace_id, content_hash FROM knowledge_builder_knowledge_scopes WHERE package_id=%s'),
            (package_id,),
        ).fetchone()
        return ({**json.loads(row["scope_json"]), "discovery_run_id": row["discovery_run_id"],
                 "organization_id": row["organization_id"],
                 "workspace_id": row["workspace_id"], "content_hash": row["content_hash"]} if row else None)

    def put_evidence_obligation(self, obligation: EvidenceObligation) -> dict[str, Any]:
        if not obligation.assertion_id:
            raise ValueError("persisted EvidenceObligation requires assertion_id")
        allowed = self._obligation_allowed_anchor_refs(obligation)
        if obligation.primary_anchor_ref not in allowed:
            raise ValueError("obligation primary anchor ref is not canonical package evidence")
        payload = obligation.model_dump(mode="json")
        digest = content_hash(payload)
        with self._lock:
            existing = self._conn.execute(
                sql_query("SELECT content_hash FROM knowledge_builder_evidence_obligations WHERE obligation_id=?",'SELECT content_hash FROM knowledge_builder_evidence_obligations WHERE obligation_id=%s'),
                (obligation.obligation_id,),
            ).fetchone()
            if existing:
                if existing["content_hash"] != digest:
                    raise ValueError("evidence obligation immutable record conflict")
                return {**payload, "content_hash": digest}
            self._conn.execute(
                sql_query("""INSERT INTO knowledge_builder_evidence_obligations
                (obligation_id, assertion_id, obligation_json, status, content_hash)
                VALUES (?, ?, ?, ?, ?)""",'INSERT INTO knowledge_builder_evidence_obligations\n                (obligation_id, assertion_id, obligation_json, status, content_hash)\n                VALUES (%s, %s, %s, %s, %s)'),
                (obligation.obligation_id, obligation.assertion_id, _json(payload), obligation.status, digest),
            )
            try:
                for ordinal, anchor_ref in enumerate(obligation.supporting_anchor_refs):
                    self._conn.execute(
                        sql_query("INSERT INTO knowledge_builder_evidence_obligation_anchors VALUES (?, ?, ?)",'INSERT INTO knowledge_builder_evidence_obligation_anchors VALUES (%s, %s, %s)'),
                        (obligation.obligation_id, anchor_ref, ordinal),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {**payload, "content_hash": digest}

    def list_evidence_obligations(self, assertion_id: str | None = None, *, organization_id: str | None = None,
                                  workspace_id: str | None = None, limit: int = 100,
                                  cursor: str | None = None) -> list[dict[str, Any]]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        sql = """SELECT o.obligation_json, o.content_hash,
        (SELECT e.status FROM knowledge_builder_evidence_obligation_resolution_events e
          WHERE e.obligation_id=o.obligation_id ORDER BY e.resolution_event_id DESC LIMIT 1) AS latest_status,
        (SELECT e.supporting_anchor_refs_json FROM knowledge_builder_evidence_obligation_resolution_events e
          WHERE e.obligation_id=o.obligation_id ORDER BY e.resolution_event_id DESC LIMIT 1) AS latest_anchor_refs
        FROM knowledge_builder_evidence_obligations o
        JOIN knowledge_builder_document_candidate_assertions a ON a.assertion_id=o.assertion_id
        JOIN knowledge_builder_extraction_runs r ON r.extraction_run_id=a.extraction_run_id WHERE 1=1"""
        args: list[Any] = []
        if assertion_id:
            sql += sql_query(" AND o.assertion_id=?",' AND o.assertion_id=%s'); args.append(assertion_id)
        if organization_id is not None:
            sql += sql_query(" AND r.organization_id=?",' AND r.organization_id=%s'); args.append(organization_id)
            if workspace_id is None:
                sql += " AND r.workspace_id IS NULL"
        if workspace_id is not None:
            sql += sql_query(" AND r.workspace_id=?",' AND r.workspace_id=%s'); args.append(workspace_id)
        if cursor:
            sql += sql_query(" AND o.obligation_id>?",' AND o.obligation_id>%s'); args.append(cursor)
        sql += sql_query(" ORDER BY o.obligation_id LIMIT ?",' ORDER BY o.obligation_id LIMIT %s'); args.append(limit)
        results = []
        for row in self._conn.execute(sql, tuple(args)).fetchall():
            item = {**json.loads(row["obligation_json"]), "content_hash": row["content_hash"]}
            if row["latest_status"]:
                item["status"] = row["latest_status"]
                item["supporting_anchor_refs"] = json.loads(row["latest_anchor_refs"])
            results.append(item)
        return results

    def _obligation_allowed_anchor_refs(self, obligation: EvidenceObligation) -> set[str]:
        refs = self.valid_evidence_anchor_refs(obligation.assertion_id or "", include_obligations=False)
        owner = self.get_assertion_owner(obligation.assertion_id or "")
        if not owner:
            return refs
        row = self._conn.execute(
            sql_query("""SELECT s.scope_json FROM knowledge_builder_extraction_runs r
            LEFT JOIN knowledge_builder_knowledge_scopes s ON s.package_id=r.package_id
            WHERE r.extraction_run_id=?""",'SELECT s.scope_json FROM knowledge_builder_extraction_runs r\n            LEFT JOIN knowledge_builder_knowledge_scopes s ON s.package_id=r.package_id\n            WHERE r.extraction_run_id=%s'), (owner["extraction_run_id"],),
        ).fetchone()
        if row and row["scope_json"]:
            scope = KnowledgeScope.model_validate_json(row["scope_json"])
            allowed_roles = set(obligation.allowed_companion_roles)
            for document in scope.documents:
                if document.role in allowed_roles or document.role == "main_tender":
                    refs.update(str(e["anchor_id"]) for e in document.elements
                                if isinstance(e, dict) and e.get("anchor_id"))
        return refs

    def _obligation_companion_anchor_refs(self, obligation: EvidenceObligation) -> set[str]:
        owner = self.get_assertion_owner(obligation.assertion_id or "")
        if not owner:
            return set()
        row = self._conn.execute(
            sql_query("""SELECT s.scope_json FROM knowledge_builder_extraction_runs r
            LEFT JOIN knowledge_builder_knowledge_scopes s ON s.package_id=r.package_id
            WHERE r.extraction_run_id=?""",'SELECT s.scope_json FROM knowledge_builder_extraction_runs r\n            LEFT JOIN knowledge_builder_knowledge_scopes s ON s.package_id=r.package_id\n            WHERE r.extraction_run_id=%s'), (owner["extraction_run_id"],),
        ).fetchone()
        if not row or not row["scope_json"]:
            return set()
        scope = KnowledgeScope.model_validate_json(row["scope_json"])
        roles = set(obligation.allowed_companion_roles)
        return {
            str(element["anchor_id"])
            for document in scope.documents if document.role in roles and document.role != "main_tender"
            for element in document.elements
            if isinstance(element, dict) and element.get("anchor_id")
        }

    def valid_evidence_anchor_refs(self, assertion_id: str, *, include_obligations: bool = True) -> set[str]:
        assertion = self.get_assertion(assertion_id)
        if assertion is None:
            return set()
        refs = {anchor.anchor_id for anchor in assertion.source_anchors}
        occurrence_ids = self.assertion_occurrence_ids(assertion_id)
        if occurrence_ids:
            marks = ",".join(sql_query("?",'%s') for _ in occurrence_ids)
            rows = self._conn.execute(
                f"SELECT source_anchor_json FROM knowledge_builder_semantic_occurrences WHERE occurrence_id IN ({marks})",
                tuple(occurrence_ids),
            ).fetchall()
            refs.update(json.loads(row["source_anchor_json"])["anchor_id"] for row in rows)
        if include_obligations:
            for obligation in self.list_evidence_obligations(assertion_id, limit=500):
                if obligation["status"] == "satisfied":
                    refs.update(obligation["supporting_anchor_refs"])
        return refs

    def get_evidence_obligation_owner(self, obligation_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("""SELECT r.organization_id, r.workspace_id, o.assertion_id
            FROM knowledge_builder_evidence_obligations o
            JOIN knowledge_builder_document_candidate_assertions a ON a.assertion_id=o.assertion_id
            JOIN knowledge_builder_extraction_runs r ON r.extraction_run_id=a.extraction_run_id
            WHERE o.obligation_id=?""",'SELECT r.organization_id, r.workspace_id, o.assertion_id\n            FROM knowledge_builder_evidence_obligations o\n            JOIN knowledge_builder_document_candidate_assertions a ON a.assertion_id=o.assertion_id\n            JOIN knowledge_builder_extraction_runs r ON r.extraction_run_id=a.extraction_run_id\n            WHERE o.obligation_id=%s'),
            (obligation_id,),
        ).fetchone()
        return dict(row) if row else None

    def resolve_evidence_obligation(
        self, *, obligation_id: str, status: str, supporting_anchor_refs: tuple[str, ...]
    ) -> dict[str, Any]:
        if status not in {"satisfied", "unresolved"}:
            raise ValueError("resolved obligation status must be satisfied or unresolved")
        row = self._conn.execute(
            sql_query("SELECT obligation_json FROM knowledge_builder_evidence_obligations WHERE obligation_id=?",'SELECT obligation_json FROM knowledge_builder_evidence_obligations WHERE obligation_id=%s'),
            (obligation_id,),
        ).fetchone()
        if row is None:
            raise ValueError("evidence obligation not found")
        current = EvidenceObligation.model_validate_json(row["obligation_json"])
        known = self._obligation_companion_anchor_refs(current)
        if not set(supporting_anchor_refs).issubset(known):
            raise ValueError(
                "supporting anchor refs are not canonical evidence from an allowed companion document role"
            )
        if status == "satisfied" and not supporting_anchor_refs:
            raise ValueError("satisfied obligation requires supporting anchor refs")
        obligation = current.model_copy(update={"status": status, "supporting_anchor_refs": supporting_anchor_refs})
        payload = obligation.model_dump(mode="json")
        digest = content_hash(payload)
        with self._lock:
            self._conn.execute(sql_query("""INSERT INTO knowledge_builder_evidence_obligation_resolution_events
                (obligation_id,status,supporting_anchor_refs_json,occurred_at,content_hash)
                VALUES (?,?,?,?,?)""",'INSERT INTO knowledge_builder_evidence_obligation_resolution_events\n                (obligation_id,status,supporting_anchor_refs_json,occurred_at,content_hash)\n                VALUES (%s,%s,%s,%s,%s)'),
                (obligation_id, status, _json(supporting_anchor_refs), _now(), digest))
            self._conn.commit()
        return {**payload, "content_hash": digest}

    def list_evidence_obligation_resolution_events(self, obligation_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute(
            sql_query("SELECT * FROM knowledge_builder_evidence_obligation_resolution_events WHERE obligation_id=? ORDER BY resolution_event_id",'SELECT * FROM knowledge_builder_evidence_obligation_resolution_events WHERE obligation_id=%s ORDER BY resolution_event_id'),
            (obligation_id,),
        ).fetchall()]
