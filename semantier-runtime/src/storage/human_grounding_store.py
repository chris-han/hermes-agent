"""Human grounding, GraphDelta, and immutable graph release persistence."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from contracts.knowledge_feature_pack import (
    LearningEventEnvelope,
    LearningEventType,
    content_hash as canonical_content_hash,
)
from contracts.knowledge_graph_phase1 import (
    AcceptedGraphRelease,
    CandidateEdge,
    CandidateNode,
    CandidateRule,
    GraphDelta,
    GraphDeltaAction,
    SensitiveTermCandidateGraph,
)
from contracts.source_semantic_witness import (
    SourceSemanticWitnessSetManifestV1,
    SourceSemanticWitnessV1,
)
from contracts.relation_usage_grounding import RelationUsageGroundingDecision
from contracts.governed_extract import ExtractResult
from contracts.relation_occurrence_grounding import RelationOccurrenceGroundingV1
from eos.artifact_hashing import content_hash as artifact_content_hash
from semantier.knowledge_feature_packs.falkordb_backend import GraphBackend
from storage.accepted_graph_release_origin import (
    verify_exact_release_origin_reservation,
)
from storage.knowledge_feature_pack_store import KnowledgeFeaturePackStore
from storage.schema_guard import mark_schema_current, schema_is_current


_SCHEMA_NAME = "human_grounding_store"
_SCHEMA_VERSION = 8
_SCHEMA_SENTINELS = frozenset(
    {
        "sensitive_term_candidate_graphs",
        "tender_graph_releases",
        "tender_graph_release_materialization_events",
        "contextgraph_reference_concept_grounding_bindings",
        "contextgraph_reference_grounding_batches",
        "source_semantic_witness_reviews",
        "source_semantic_witness_manifests",
        "relation_usage_grounding_decisions",
        "relation_occurrence_groundings",
        "canonical_restriction_patterns",
        "knowledge_relation_instances",
        "occurrence_grounding_manifests",
        "canonical_pattern_manifests",
        "accepted_relation_bindings",
        "accepted_pattern_bindings",
    }
)


class HumanGroundingStore(KnowledgeFeaturePackStore):
    def __init__(
        self,
        db_path: str | Path,
        *,
        graph_backend: GraphBackend,
        clock=None,
        materialization_lease_seconds: int = 300,
        _relation_usage_grounding_authority: object | None = None,
    ) -> None:
        super().__init__(db_path=db_path)
        self._graph_backend = graph_backend
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lease_seconds = materialization_lease_seconds
        self._relation_usage_grounding_authority = (
            _relation_usage_grounding_authority
        )
        self._init_grounding_schema()

    def _init_grounding_schema(self) -> None:
        if hosted.enabled():
            asset = Path(__file__).with_name('human_grounding_postgres.sql').read_text()
            import re
            hosted.require_tables(self._conn, *re.findall(r'CREATE TABLE IF NOT EXISTS (\w+)', asset))
            self._conn.commit()
            return
        with self._lock:
            if schema_is_current(
                self._conn,
                schema_name=_SCHEMA_NAME,
                schema_version=_SCHEMA_VERSION,
                sentinels=_SCHEMA_SENTINELS,
            ) and self._foreign_keys_are_current():
                return
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sensitive_term_candidate_graphs (
                  candidate_graph_id TEXT PRIMARY KEY,
                  source_document_hash TEXT NOT NULL,
                  extraction_run_id TEXT NOT NULL,
                  proposal_id TEXT NOT NULL,
                  graph_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  organization_id TEXT,
                  workspace_id TEXT,
                  allowed_base_graph_version TEXT
                );
                CREATE TABLE IF NOT EXISTS tender_graph_deltas (
                  graph_delta_id TEXT PRIMARY KEY,
                  candidate_graph_id TEXT NOT NULL,
                  from_graph_version TEXT,
                  to_graph_version TEXT NOT NULL,
                  human_label_event_id TEXT NOT NULL,
                  delta_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(candidate_graph_id) REFERENCES sensitive_term_candidate_graphs(candidate_graph_id)
                );
                CREATE TABLE IF NOT EXISTS tender_graph_releases (
                  graph_version TEXT PRIMARY KEY,
                  parent_graph_version TEXT,
                  candidate_graph_id TEXT NOT NULL,
                  graph_delta_id TEXT,
                  release_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  organization_id TEXT,
                  workspace_id TEXT,
                  materialization_status TEXT NOT NULL DEFAULT 'materialized',
                  retirement_state TEXT NOT NULL DEFAULT 'active',
                  review_command_hash TEXT,
                  lineage_contract TEXT NOT NULL DEFAULT 'legacy'
                    CHECK(lineage_contract IN ('legacy', 'exact_v3', 'exact_v4_v3')),
                  FOREIGN KEY(candidate_graph_id) REFERENCES sensitive_term_candidate_graphs(candidate_graph_id),
                  FOREIGN KEY(graph_delta_id) REFERENCES tender_graph_deltas(graph_delta_id)
                );
                CREATE TABLE IF NOT EXISTS tender_graph_release_materialization_events (
                  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  graph_version TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('pending', 'retry_pending', 'materialized', 'failed')),
                  occurred_at TEXT NOT NULL,
                  error_message TEXT,
                  attempt_id TEXT,
                  lease_expires_at TEXT,
                  FOREIGN KEY(graph_version) REFERENCES tender_graph_releases(graph_version)
                );
                CREATE INDEX IF NOT EXISTS idx_tgrme_version
                  ON tender_graph_release_materialization_events(graph_version, event_id);
                CREATE TABLE IF NOT EXISTS tender_graph_version_allocations (
                  allocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  candidate_graph_id TEXT NOT NULL,
                  allocated_at TEXT NOT NULL
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
                CREATE TABLE IF NOT EXISTS contextgraph_reference_concept_grounding_bindings (
                  binding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  concept_id TEXT NOT NULL,
                  human_label_event_id TEXT NOT NULL,
                  graph_version TEXT NOT NULL,
                  graph_release_hash TEXT NOT NULL,
                  accepted_node_ids_json TEXT NOT NULL,
                  accepted_edge_ids_json TEXT NOT NULL,
                  accepted_rule_ids_json TEXT NOT NULL,
                  materialized_at TEXT NOT NULL,
                  FOREIGN KEY(concept_id) REFERENCES contextgraph_reference_concepts(concept_id),
                  FOREIGN KEY(graph_version) REFERENCES tender_graph_releases(graph_version),
                  FOREIGN KEY(human_label_event_id) REFERENCES knowledge_learning_events(event_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_crcgb_event_release
                  ON contextgraph_reference_concept_grounding_bindings(human_label_event_id, graph_version);
                CREATE INDEX IF NOT EXISTS idx_crcgb_concept_latest
                  ON contextgraph_reference_concept_grounding_bindings(concept_id, binding_id);
                CREATE TABLE IF NOT EXISTS contextgraph_reference_grounding_batches (
                  batch_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  idempotency_key TEXT NOT NULL,
                  command_hash TEXT NOT NULL,
                  event_ids_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_crgb_tenant_idempotency
                  ON contextgraph_reference_grounding_batches(
                    organization_id, COALESCE(workspace_id, ''), idempotency_key
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
                  created_at TEXT NOT NULL,
                  lineage_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_cg_artifact_origins_source
                  ON contextgraph_artifact_origins (organization_id, workspace_id, source_id);
                CREATE INDEX IF NOT EXISTS idx_cg_artifact_origins_artifact
                  ON contextgraph_artifact_origins (artifact_id);
                CREATE TABLE IF NOT EXISTS source_semantic_witness_reviews (
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  witness_ref TEXT NOT NULL,
                  witness_hash TEXT NOT NULL,
                  lifecycle_state TEXT NOT NULL
                    CHECK(lifecycle_state IN ('proposed','accepted','edited','rejected')),
                  predecessor_witness_ref TEXT,
                  successor_witness_ref TEXT,
                  review_ref TEXT NOT NULL,
                  review_hash TEXT NOT NULL,
                  canonical_payload_json TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  decision_evidence_refs_json TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  PRIMARY KEY (organization_id, workspace_id, witness_ref)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_sswr_global_ref
                  ON source_semantic_witness_reviews(witness_ref);
                CREATE TABLE IF NOT EXISTS source_semantic_witness_manifests (
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  manifest_ref TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  lifecycle_state TEXT NOT NULL
                    CHECK(lifecycle_state IN ('proposed','accepted','edited','rejected')),
                  predecessor_manifest_ref TEXT,
                  successor_manifest_ref TEXT,
                  canonical_payload_json TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  artifact_ref TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  PRIMARY KEY (organization_id, workspace_id, manifest_ref)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_sswm_global_ref
                  ON source_semantic_witness_manifests(manifest_ref);
                CREATE TABLE IF NOT EXISTS relation_usage_grounding_decisions (
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  grounding_decision_ref TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  source_identity_ref TEXT NOT NULL,
                  lineage_id TEXT NOT NULL,
                  extract_run_ref TEXT NOT NULL,
                  extract_result_hash TEXT NOT NULL,
                  schema_decision_ref TEXT NOT NULL,
                  schema_decision_hash TEXT NOT NULL,
                  subject_ref TEXT NOT NULL,
                  subject_hash TEXT NOT NULL,
                  finding_kind TEXT NOT NULL,
                  required_route TEXT NOT NULL,
                  schema_learning_feedback_ref TEXT NOT NULL,
                  canonical_payload_json TEXT NOT NULL,
                  reviewed_by TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  PRIMARY KEY (organization_id, workspace_id, grounding_decision_ref)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_rugd_global_ref
                  ON relation_usage_grounding_decisions(grounding_decision_ref);
                CREATE TABLE IF NOT EXISTS relation_occurrence_groundings (
                  grounding_record_id TEXT PRIMARY KEY,
                  grounding_record_hash TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  extract_result_ref TEXT NOT NULL,
                  extract_result_hash TEXT NOT NULL,
                  schema_decision_ref TEXT NOT NULL,
                  schema_decision_hash TEXT NOT NULL,
                  extracted_occurrence_ref TEXT NOT NULL,
                  extracted_occurrence_hash TEXT NOT NULL,
                  outcome TEXT NOT NULL
                    CHECK(outcome IN ('accepted','corrected','rejected')),
                  canonical_payload_json TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  decided_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_rog_terminal_occurrence
                  ON relation_occurrence_groundings(
                    organization_id, workspace_id, extract_result_ref,
                    extract_result_hash, extracted_occurrence_ref,
                    extracted_occurrence_hash
                  );
                CREATE INDEX IF NOT EXISTS idx_rog_extract_outcome
                  ON relation_occurrence_groundings(
                    organization_id, workspace_id, extract_result_ref, outcome,
                    grounding_record_id
                  );
                CREATE TABLE IF NOT EXISTS canonical_restriction_patterns (
                  pattern_id TEXT PRIMARY KEY,
                  pattern_hash TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  canonical_payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_crp_tenant_hash
                  ON canonical_restriction_patterns(
                    organization_id, workspace_id, pattern_hash, pattern_id
                  );
                CREATE TABLE IF NOT EXISTS knowledge_relation_instances (
                  relation_instance_id TEXT PRIMARY KEY,
                  relation_instance_hash TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  relation_type_id TEXT NOT NULL,
                  representation_kind TEXT NOT NULL,
                  canonical_payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_kri_tenant_type
                  ON knowledge_relation_instances(
                    organization_id, workspace_id, relation_type_id,
                    relation_instance_id
                  );
                CREATE TABLE IF NOT EXISTS occurrence_grounding_manifests (
                  manifest_ref TEXT PRIMARY KEY,
                  manifest_hash TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  schema_decision_ref TEXT NOT NULL,
                  schema_decision_hash TEXT NOT NULL,
                  extract_result_ref TEXT NOT NULL,
                  extract_result_hash TEXT NOT NULL,
                  canonical_payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ogm_tenant_extract
                  ON occurrence_grounding_manifests(
                    organization_id, workspace_id, extract_result_ref,
                    manifest_ref
                  );
                CREATE TABLE IF NOT EXISTS canonical_pattern_manifests (
                  manifest_ref TEXT PRIMARY KEY,
                  manifest_hash TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  canonical_payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cpm_tenant_hash
                  ON canonical_pattern_manifests(
                    organization_id, workspace_id, manifest_hash, manifest_ref
                  );
                CREATE TABLE IF NOT EXISTS accepted_relation_bindings (
                  binding_id TEXT PRIMARY KEY,
                  binding_hash TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  relation_instance_ref TEXT NOT NULL,
                  relation_instance_hash TEXT NOT NULL,
                  accepted_graph_release_ref TEXT NOT NULL,
                  accepted_graph_release_hash TEXT NOT NULL,
                  canonical_payload_json TEXT NOT NULL,
                  UNIQUE(accepted_graph_release_ref, relation_instance_ref),
                  FOREIGN KEY(relation_instance_ref)
                    REFERENCES knowledge_relation_instances(relation_instance_id),
                  FOREIGN KEY(accepted_graph_release_ref)
                    REFERENCES tender_graph_releases(graph_version)
                );
                CREATE INDEX IF NOT EXISTS idx_arb_tenant_release
                  ON accepted_relation_bindings(
                    organization_id, workspace_id, accepted_graph_release_ref,
                    binding_id
                  );
                CREATE TABLE IF NOT EXISTS accepted_pattern_bindings (
                  binding_id TEXT PRIMARY KEY,
                  binding_hash TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  pattern_ref TEXT NOT NULL,
                  pattern_hash TEXT NOT NULL,
                  accepted_graph_release_ref TEXT NOT NULL,
                  accepted_graph_release_hash TEXT NOT NULL,
                  canonical_payload_json TEXT NOT NULL,
                  UNIQUE(accepted_graph_release_ref, pattern_ref),
                  FOREIGN KEY(pattern_ref)
                    REFERENCES canonical_restriction_patterns(pattern_id),
                  FOREIGN KEY(accepted_graph_release_ref)
                    REFERENCES tender_graph_releases(graph_version)
                );
                CREATE INDEX IF NOT EXISTS idx_apb_tenant_release
                  ON accepted_pattern_bindings(
                    organization_id, workspace_id, accepted_graph_release_ref,
                    binding_id
                  );
                """
            )
            self._add_column("sensitive_term_candidate_graphs", "organization_id", "TEXT")
            self._add_column("sensitive_term_candidate_graphs", "workspace_id", "TEXT")
            self._add_column("sensitive_term_candidate_graphs", "allowed_base_graph_version", "TEXT")
            self._add_column("tender_graph_releases", "organization_id", "TEXT")
            self._add_column("tender_graph_releases", "workspace_id", "TEXT")
            self._add_column("tender_graph_releases", "materialization_status", "TEXT NOT NULL DEFAULT 'materialized'")
            self._add_column(
                "tender_graph_releases",
                "retirement_state",
                "TEXT NOT NULL DEFAULT 'active'",
            )
            self._add_column("tender_graph_releases", "review_command_hash", "TEXT")
            self._add_column(
                "tender_graph_releases",
                "lineage_contract",
                "TEXT NOT NULL DEFAULT 'legacy'",
            )
            self._conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS
                idx_tender_graph_releases_review_command
                ON tender_graph_releases(review_command_hash)
                WHERE review_command_hash IS NOT NULL"""
            )
            orphan = self._legacy_grounding_orphan()
            if orphan:
                raise ValueError(f"legacy grounding foreign-key orphan: {orphan[0]}:{orphan[1]}")
            self._migrate_materialization_events()
            self._add_column("tender_graph_release_materialization_events", "attempt_id", "TEXT")
            self._add_column("tender_graph_release_materialization_events", "lease_expires_at", "TEXT")
            self._add_column("contextgraph_artifact_origins", "lineage_id", "TEXT")
            self._add_column("contextgraph_artifact_origins", "release_lineage_json", "TEXT")
            self._add_column("contextgraph_artifact_origins", "release_lineage_hash", "TEXT")
            self._conn.commit()
            self._migrate_grounding_foreign_keys()
            self._migrate_release_lineage_contract_check()
            self._migrate_binding_foreign_keys()
            mark_schema_current(
                self._conn,
                schema_name=_SCHEMA_NAME,
                schema_version=_SCHEMA_VERSION,
            )
            self._conn.commit()

    def _foreign_keys_are_current(self) -> bool:
        release_sql_row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='tender_graph_releases'"
        ).fetchone()
        if (
            release_sql_row is None
            or "'exact_v4_v3'" not in str(release_sql_row["sql"] or "")
        ):
            return False
        expected_by_table = {
            "tender_graph_deltas": {
                ("candidate_graph_id", "sensitive_term_candidate_graphs", "candidate_graph_id"),
            },
            "tender_graph_releases": {
                ("candidate_graph_id", "sensitive_term_candidate_graphs", "candidate_graph_id"),
                ("graph_delta_id", "tender_graph_deltas", "graph_delta_id"),
            },
            "tender_graph_release_materialization_events": {
                ("graph_version", "tender_graph_releases", "graph_version"),
            },
            "knowledge_builder_assertion_grounding_bindings": {
                ("assertion_id", "knowledge_builder_document_candidate_assertions", "assertion_id"),
                ("graph_version", "tender_graph_releases", "graph_version"),
                ("human_label_event_id", "knowledge_learning_events", "event_id"),
            },
            "contextgraph_reference_concept_grounding_bindings": {
                ("concept_id", "contextgraph_reference_concepts", "concept_id"),
                ("graph_version", "tender_graph_releases", "graph_version"),
                ("human_label_event_id", "knowledge_learning_events", "event_id"),
            },
            "accepted_relation_bindings": {
                ("relation_instance_ref", "knowledge_relation_instances", "relation_instance_id"),
                ("accepted_graph_release_ref", "tender_graph_releases", "graph_version"),
            },
            "accepted_pattern_bindings": {
                ("pattern_ref", "canonical_restriction_patterns", "pattern_id"),
                ("accepted_graph_release_ref", "tender_graph_releases", "graph_version"),
            },
        }
        for table, expected in expected_by_table.items():
            actual = {
                (row["from"], row["table"], row["to"])
                for row in self._conn.execute(f"PRAGMA foreign_key_list({table})")
            }
            if not expected.issubset(actual):
                return False
        return True

    def _add_column(self, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _migrate_materialization_events(self) -> None:
        sql = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='tender_graph_release_materialization_events'"
        ).fetchone()["sql"]
        if "retry_pending" in sql and "failed" in sql:
            return
        self._conn.executescript("""
            ALTER TABLE tender_graph_release_materialization_events RENAME TO tender_graph_release_materialization_events_legacy;
            CREATE TABLE tender_graph_release_materialization_events (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              graph_version TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('pending','retry_pending','materialized','failed')),
              occurred_at TEXT NOT NULL,
              error_message TEXT,
              FOREIGN KEY(graph_version) REFERENCES tender_graph_releases(graph_version)
            );
            INSERT INTO tender_graph_release_materialization_events
              (event_id,graph_version,status,occurred_at)
              SELECT event_id,graph_version,status,occurred_at
              FROM tender_graph_release_materialization_events_legacy;
            DROP TABLE tender_graph_release_materialization_events_legacy;
            CREATE INDEX idx_tgrme_version ON tender_graph_release_materialization_events(graph_version,event_id);
        """)

    def _migrate_grounding_foreign_keys(self) -> None:
        if (self._conn.execute("PRAGMA foreign_key_list(tender_graph_deltas)").fetchall() and
                self._conn.execute("PRAGMA foreign_key_list(tender_graph_releases)").fetchall() and
                self._conn.execute("PRAGMA foreign_key_list(tender_graph_release_materialization_events)").fetchall()):
            return
        orphan = self._legacy_grounding_orphan()
        if orphan:
            raise ValueError(f"legacy grounding foreign-key orphan: {orphan[0]}:{orphan[1]}")
        self._conn.commit()
        self._conn.execute("PRAGMA foreign_keys=OFF")
        self._conn.execute("PRAGMA legacy_alter_table=ON")
        try:
            self._conn.executescript("""
                BEGIN IMMEDIATE;
                ALTER TABLE tender_graph_release_materialization_events RENAME TO tgrme_legacy;
                ALTER TABLE tender_graph_releases RENAME TO tgr_legacy;
                ALTER TABLE tender_graph_deltas RENAME TO tgd_legacy;
                CREATE TABLE tender_graph_deltas (
                  graph_delta_id TEXT PRIMARY KEY, candidate_graph_id TEXT NOT NULL,
                  from_graph_version TEXT, to_graph_version TEXT NOT NULL,
                  human_label_event_id TEXT NOT NULL, delta_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(candidate_graph_id) REFERENCES sensitive_term_candidate_graphs(candidate_graph_id));
                CREATE TABLE tender_graph_releases (
                  graph_version TEXT PRIMARY KEY, parent_graph_version TEXT,
                  candidate_graph_id TEXT NOT NULL, graph_delta_id TEXT NOT NULL,
                  release_json TEXT NOT NULL, content_hash TEXT NOT NULL,
                  organization_id TEXT, workspace_id TEXT,
                  materialization_status TEXT NOT NULL DEFAULT 'materialized',
                  retirement_state TEXT NOT NULL DEFAULT 'active',
                  review_command_hash TEXT,
                  lineage_contract TEXT NOT NULL DEFAULT 'legacy'
                    CHECK(lineage_contract IN ('legacy', 'exact_v3', 'exact_v4_v3')),
                  FOREIGN KEY(candidate_graph_id) REFERENCES sensitive_term_candidate_graphs(candidate_graph_id),
                  FOREIGN KEY(graph_delta_id) REFERENCES tender_graph_deltas(graph_delta_id));
                CREATE TABLE tender_graph_release_materialization_events (
                  event_id INTEGER PRIMARY KEY AUTOINCREMENT, graph_version TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('pending','retry_pending','materialized','failed')),
                  occurred_at TEXT NOT NULL, error_message TEXT, attempt_id TEXT, lease_expires_at TEXT,
                  FOREIGN KEY(graph_version) REFERENCES tender_graph_releases(graph_version));
                INSERT INTO tender_graph_deltas SELECT * FROM tgd_legacy;
                INSERT INTO tender_graph_releases SELECT * FROM tgr_legacy;
                INSERT INTO tender_graph_release_materialization_events SELECT * FROM tgrme_legacy;
                DROP TABLE tgrme_legacy;
                DROP TABLE tgr_legacy;
                DROP TABLE tgd_legacy;
                CREATE INDEX idx_tgrme_version ON tender_graph_release_materialization_events(graph_version,event_id);
                CREATE UNIQUE INDEX idx_tender_graph_releases_review_command
                  ON tender_graph_releases(review_command_hash)
                  WHERE review_command_hash IS NOT NULL;
                COMMIT;
            """)
        except Exception:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
        finally:
            self._conn.execute("PRAGMA legacy_alter_table=OFF")
            self._conn.execute("PRAGMA foreign_keys=ON")
        violations = self._conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ValueError(f"grounding foreign-key migration failed: {violations}")

    def _migrate_release_lineage_contract_check(self) -> None:
        """Add the v2 discriminator without rewriting any stored release bytes."""

        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='tender_graph_releases'"
        ).fetchone()
        if row is not None and "'exact_v4_v3'" in str(row["sql"] or ""):
            return
        self._conn.commit()
        self._conn.execute("PRAGMA foreign_keys=OFF")
        try:
            self._conn.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE tender_graph_releases_v8 (
                  graph_version TEXT PRIMARY KEY,
                  parent_graph_version TEXT,
                  candidate_graph_id TEXT NOT NULL,
                  graph_delta_id TEXT,
                  release_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  organization_id TEXT,
                  workspace_id TEXT,
                  materialization_status TEXT NOT NULL DEFAULT 'materialized',
                  retirement_state TEXT NOT NULL DEFAULT 'active',
                  review_command_hash TEXT,
                  lineage_contract TEXT NOT NULL DEFAULT 'legacy'
                    CHECK(lineage_contract IN ('legacy', 'exact_v3', 'exact_v4_v3')),
                  FOREIGN KEY(candidate_graph_id)
                    REFERENCES sensitive_term_candidate_graphs(candidate_graph_id),
                  FOREIGN KEY(graph_delta_id)
                    REFERENCES tender_graph_deltas(graph_delta_id)
                );
                INSERT INTO tender_graph_releases_v8
                  (graph_version,parent_graph_version,candidate_graph_id,
                   graph_delta_id,release_json,content_hash,organization_id,
                   workspace_id,materialization_status,retirement_state,
                   review_command_hash,lineage_contract)
                SELECT graph_version,parent_graph_version,candidate_graph_id,
                       graph_delta_id,release_json,content_hash,organization_id,
                       workspace_id,materialization_status,retirement_state,
                       review_command_hash,lineage_contract
                FROM tender_graph_releases;
                DROP TABLE tender_graph_releases;
                ALTER TABLE tender_graph_releases_v8 RENAME TO tender_graph_releases;
                CREATE UNIQUE INDEX idx_tender_graph_releases_review_command
                  ON tender_graph_releases(review_command_hash)
                  WHERE review_command_hash IS NOT NULL;
                COMMIT;
                """
            )
        except Exception:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
        finally:
            self._conn.execute("PRAGMA foreign_keys=ON")
        violations = self._conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ValueError(
                f"release lineage discriminator migration failed: {violations}"
            )

    def _legacy_grounding_orphan(self):
        return self._conn.execute("""SELECT 'delta', d.graph_delta_id FROM tender_graph_deltas d
            LEFT JOIN sensitive_term_candidate_graphs c ON c.candidate_graph_id=d.candidate_graph_id
            WHERE c.candidate_graph_id IS NULL
            UNION ALL SELECT 'release_candidate', r.graph_version FROM tender_graph_releases r
            LEFT JOIN sensitive_term_candidate_graphs c ON c.candidate_graph_id=r.candidate_graph_id
            WHERE c.candidate_graph_id IS NULL
            UNION ALL SELECT 'release_delta', r.graph_version FROM tender_graph_releases r
            LEFT JOIN tender_graph_deltas d ON d.graph_delta_id=r.graph_delta_id
            WHERE r.graph_delta_id IS NOT NULL AND d.graph_delta_id IS NULL
            UNION ALL SELECT 'lifecycle', e.event_id FROM tender_graph_release_materialization_events e
            LEFT JOIN tender_graph_releases r ON r.graph_version=e.graph_version
            WHERE r.graph_version IS NULL LIMIT 1""").fetchone()

    def _migrate_binding_foreign_keys(self) -> None:
        expected = {
            ("assertion_id", "knowledge_builder_document_candidate_assertions", "assertion_id"),
            ("graph_version", "tender_graph_releases", "graph_version"),
            ("human_label_event_id", "knowledge_learning_events", "event_id"),
        }
        actual = {(row["from"], row["table"], row["to"]) for row in
                  self._conn.execute("PRAGMA foreign_key_list(knowledge_builder_assertion_grounding_bindings)")}
        if expected.issubset(actual):
            return
        required = {"knowledge_builder_document_candidate_assertions", "knowledge_learning_events"}
        present = {row["name"] for row in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if not required.issubset(present):
            return
        orphan = self._conn.execute("""SELECT b.binding_id FROM knowledge_builder_assertion_grounding_bindings b
            LEFT JOIN knowledge_builder_document_candidate_assertions a ON a.assertion_id=b.assertion_id
            LEFT JOIN tender_graph_releases r ON r.graph_version=b.graph_version
            LEFT JOIN knowledge_learning_events e ON e.event_id=b.human_label_event_id
            WHERE a.assertion_id IS NULL OR r.graph_version IS NULL OR e.event_id IS NULL LIMIT 1""").fetchone()
        if orphan:
            raise ValueError(f"legacy assertion grounding binding orphan: {orphan['binding_id']}")
        self._conn.commit()
        self._conn.execute("PRAGMA foreign_keys=OFF")
        try:
            self._conn.executescript("""BEGIN IMMEDIATE;
                ALTER TABLE knowledge_builder_assertion_grounding_bindings RENAME TO kbagb_legacy;
                CREATE TABLE knowledge_builder_assertion_grounding_bindings (
                  binding_id INTEGER PRIMARY KEY AUTOINCREMENT, assertion_id TEXT NOT NULL,
                  human_label_event_id TEXT NOT NULL, graph_version TEXT NOT NULL,
                  graph_release_hash TEXT NOT NULL, accepted_node_ids_json TEXT NOT NULL,
                  accepted_edge_ids_json TEXT NOT NULL, accepted_rule_ids_json TEXT NOT NULL,
                  materialized_at TEXT NOT NULL,
                  FOREIGN KEY(assertion_id) REFERENCES knowledge_builder_document_candidate_assertions(assertion_id),
                  FOREIGN KEY(graph_version) REFERENCES tender_graph_releases(graph_version),
                  FOREIGN KEY(human_label_event_id) REFERENCES knowledge_learning_events(event_id));
                INSERT INTO knowledge_builder_assertion_grounding_bindings SELECT * FROM kbagb_legacy;
                DROP TABLE kbagb_legacy;
                CREATE UNIQUE INDEX idx_kbagb_event_release ON knowledge_builder_assertion_grounding_bindings(human_label_event_id,graph_version);
                CREATE INDEX idx_kbagb_assertion_latest ON knowledge_builder_assertion_grounding_bindings(assertion_id,binding_id);
                COMMIT;""")
        except Exception:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
        finally:
            self._conn.execute("PRAGMA foreign_keys=ON")
        violations = self._conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ValueError(f"assertion grounding binding foreign-key migration failed: {violations}")

    def put_candidate_graph(self, graph: SensitiveTermCandidateGraph, *, organization_id: str | None = None,
                            workspace_id: str | None = None,
                            allowed_base_graph_version: str | None = None) -> dict[str, Any]:
        payload = graph.model_dump(mode="json")
        digest = graph.graph_hash()
        self._insert_immutable(
            table="sensitive_term_candidate_graphs",
            identity_where="candidate_graph_id = ?",
            identity=(graph.candidate_graph_id,),
            insert_sql="""
                INSERT INTO sensitive_term_candidate_graphs
                  (candidate_graph_id, source_document_hash, extraction_run_id,
                   proposal_id, graph_json, content_hash, organization_id, workspace_id,
                   allowed_base_graph_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values=(
                graph.candidate_graph_id,
                graph.source_document_hash,
                graph.extraction_run_id,
                graph.machine_proposal.proposal_id,
                _json(payload),
                digest,
                organization_id,
                workspace_id,
                allowed_base_graph_version,
            ),
            expected_hash=digest,
        postgres_select_sql='SELECT content_hash FROM sensitive_term_candidate_graphs WHERE candidate_graph_id = %s', postgres_insert_sql='\n                INSERT INTO sensitive_term_candidate_graphs\n                  (candidate_graph_id, source_document_hash, extraction_run_id,\n                   proposal_id, graph_json, content_hash, organization_id, workspace_id,\n                   allowed_base_graph_version)\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n            ',)
        return {**payload, "content_hash": digest, "grounding_state": "proposal_only"}

    def append_reference_review_batch(
        self,
        *,
        events: tuple[LearningEventEnvelope, ...],
        batch_id: str,
        organization_id: str,
        workspace_id: str | None,
        candidate_graph_id: str,
        idempotency_key: str,
        command_hash: str,
    ) -> list[dict[str, Any]]:
        """Atomically reserve one whole reference-review batch and append its events."""
        if not events:
            raise ValueError("reference review batch requires events")
        with self._lock:
            hosted.begin_write(self._conn)
            try:
                existing = self._conn.execute(
                    hosted.sql_query("""SELECT command_hash,event_ids_json
                    FROM contextgraph_reference_grounding_batches
                    WHERE organization_id=? AND workspace_id IS ? AND idempotency_key=?""", 'SELECT command_hash,event_ids_json\n                    FROM contextgraph_reference_grounding_batches\n                    WHERE organization_id=%s AND workspace_id IS NOT DISTINCT FROM %s AND idempotency_key=%s'),
                    (organization_id, workspace_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    if existing["command_hash"] != command_hash:
                        raise ValueError(
                            "batchIdempotencyKey was already used for a different review command"
                        )
                    event_ids = tuple(json.loads(existing["event_ids_json"]))
                    if event_ids != tuple(event.event_id for event in events):
                        raise ValueError("persisted batch event identity mismatch")
                    persisted = []
                    for event_id in event_ids:
                        event_row = self._conn.execute(
                            hosted.sql_query("""SELECT event_json,event_hash FROM knowledge_learning_events
                            WHERE event_id=?""", 'SELECT event_json,event_hash FROM knowledge_learning_events\n                            WHERE event_id=%s'),
                            (event_id,),
                        ).fetchone()
                        if event_row is None:
                            raise ValueError("persisted batch review event is missing")
                        event = LearningEventEnvelope.model_validate_json(
                            event_row["event_json"]
                        )
                        if event.event_hash() != event_row["event_hash"]:
                            raise ValueError("persisted batch review event hash mismatch")
                        persisted.append(
                            {
                                **event.model_dump(mode="json"),
                                "event_hash": event_row["event_hash"],
                            }
                        )
                    self._conn.commit()
                    return persisted
                events = tuple(
                    self._with_governed_review_commitment(event)
                    for event in events
                )
                for event in events:
                    prior_release = self._conn.execute(
                        hosted.sql_query("""SELECT graph_version FROM tender_graph_releases
                        WHERE candidate_graph_id=? AND release_json LIKE ? LIMIT 1""", 'SELECT graph_version FROM tender_graph_releases\n                        WHERE candidate_graph_id=%s AND release_json LIKE %s LIMIT 1'),
                        (candidate_graph_id, f'%"{event.event_id}"%'),
                    ).fetchone()
                    if prior_release is not None:
                        raise ValueError(
                            "review decision already has an accepted release: "
                            f"{prior_release['graph_version']}"
                        )
                bound_events = [event for event in events if event.reviewed_candidate_graph_hash is not None]
                if bound_events:
                    owner = self.get_candidate_graph_owner(candidate_graph_id)
                    if owner is None or owner["organization_id"] != organization_id or owner["workspace_id"] != workspace_id:
                        raise ValueError("batch candidate graph scope mismatch")
                    graph = self.get_candidate_graph(candidate_graph_id)
                    stamped_events = []
                    for event in events:
                        if event.reviewed_candidate_graph_hash is None:
                            stamped_events.append(event)
                            continue
                        evidence = self._conn.execute(
                            hosted.sql_query("SELECT extraction_run_id,candidate_graph_id FROM contextgraph_reference_concepts WHERE concept_id=? "
                            "UNION ALL SELECT extraction_run_id,candidate_graph_id FROM "
                            "knowledge_builder_document_candidate_assertions WHERE assertion_id=?", 'SELECT extraction_run_id,candidate_graph_id FROM contextgraph_reference_concepts WHERE concept_id=%s UNION ALL SELECT extraction_run_id,candidate_graph_id FROM knowledge_builder_document_candidate_assertions WHERE assertion_id=%s'),
                            (event.case_id, event.case_id),
                        ).fetchall()
                        if (
                            len(evidence) != 1 or evidence[0]["extraction_run_id"] != graph.extraction_run_id
                            or evidence[0]["candidate_graph_id"] != candidate_graph_id
                            or event.source_document_hash != graph.source_document_hash
                            or event.graph_version != str(owner.get("allowed_base_graph_version") or "KG_v0")
                        ):
                            raise ValueError("batch review does not bind exact candidate source/run/version")
                        previous = self._conn.execute(
                            hosted.sql_query("SELECT event_json,event_hash FROM knowledge_learning_events WHERE case_id=? "
                            "ORDER BY occurred_at DESC,event_id DESC LIMIT 1", 'SELECT event_json,event_hash FROM knowledge_learning_events WHERE case_id=%s ORDER BY occurred_at DESC,event_id DESC LIMIT 1'), (event.case_id,),
                        ).fetchone()
                        if event.prior_event_hash != (previous["event_hash"] if previous else None):
                            raise ValueError("batch review case history changed")
                        prior = LearningEventEnvelope.model_validate_json(previous["event_json"]) if previous else None
                        if prior is not None and prior.event_hash() != previous["event_hash"]:
                            raise ValueError("batch review prior event hash mismatch")
                        exact_proposal = (
                            prior is not None and prior.event_type is LearningEventType.MACHINE_PROPOSAL
                            and prior.machine_proposal == graph.machine_proposal
                            and prior.source_document_hash == graph.source_document_hash
                            and set(prior.source_anchor_refs) == set(event.source_anchor_refs)
                            and prior.graph_version == event.graph_version
                            and prior.rule_version == event.rule_version
                            and prior.evaluation_corpus_version == event.evaluation_corpus_version
                        )
                        if not exact_proposal:
                            # Receipt of the already-stored proposal, not a new inference.
                            proposal = LearningEventEnvelope(
                                event_id=f"batch_machine_receipt_{uuid4().hex}", case_id=event.case_id,
                                event_type=LearningEventType.MACHINE_PROPOSAL,
                                occurred_at=datetime.now(timezone.utc),
                                source_document_hash=graph.source_document_hash,
                                source_anchor_refs=event.source_anchor_refs, graph_version=event.graph_version,
                                rule_version=event.rule_version, evaluation_corpus_version=event.evaluation_corpus_version,
                                machine_proposal=graph.machine_proposal, prior_event_hash=event.prior_event_hash,
                            )
                            self._append_event_uncommitted(proposal)
                            event = LearningEventEnvelope.model_validate({
                                **event.model_dump(mode="json"),
                                "occurred_at": datetime.now(timezone.utc),
                                "prior_event_hash": proposal.event_hash(),
                            })
                        stamped_events.append(event)
                    events = tuple(stamped_events)
                persisted = [
                    self._append_event_uncommitted(event) for event in events
                ]
                self._conn.execute(
                    hosted.sql_query("""INSERT INTO contextgraph_reference_grounding_batches
                    (batch_id,organization_id,workspace_id,idempotency_key,
                     command_hash,event_ids_json,created_at)
                    VALUES (?,?,?,?,?,?,?)""", 'INSERT INTO contextgraph_reference_grounding_batches\n                    (batch_id,organization_id,workspace_id,idempotency_key,\n                     command_hash,event_ids_json,created_at)\n                    VALUES (%s,%s,%s,%s,%s,%s,%s)'),
                    (
                        batch_id,
                        organization_id,
                        workspace_id,
                        idempotency_key,
                        command_hash,
                        _json([event.event_id for event in events]),
                        self._now(),
                    ),
                )
                self._conn.commit()
                return persisted
            except Exception:
                if hosted.in_transaction(self._conn):
                    self._conn.rollback()
                raise

    def get_candidate_graph_owner(self, candidate_graph_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            hosted.sql_query("""SELECT c.organization_id, c.workspace_id,
            r.source_id, r.lineage_id, r.extraction_run_id, r.source_identity_ref,
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
            WHERE c.candidate_graph_id=?""", "SELECT c.organization_id, c.workspace_id,\n            r.source_id, r.lineage_id, r.extraction_run_id, r.source_identity_ref,\n            COALESCE((SELECT r.graph_version FROM tender_graph_releases r\n              WHERE r.candidate_graph_id=c.candidate_graph_id AND\n                COALESCE((SELECT e.status FROM tender_graph_release_materialization_events e\n                  WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),\n                  r.materialization_status)='materialized'\n              ORDER BY r.release_sequence DESC LIMIT 1), c.allowed_base_graph_version) AS allowed_base_graph_version\n            FROM sensitive_term_candidate_graphs c\n            JOIN knowledge_builder_extraction_runs r\n              ON r.extraction_run_id=c.extraction_run_id\n             AND r.organization_id=c.organization_id\n             AND r.workspace_id IS NOT DISTINCT FROM c.workspace_id\n            WHERE c.candidate_graph_id=%s"),
            (candidate_graph_id,),
        ).fetchone()
        return dict(row) if row else None

    def delete_uncommitted_candidate_graph(self, candidate_graph_id: str) -> None:
        """Compensate extraction failure before the candidate enters review history."""
        with self._lock:
            referenced = self._conn.execute(
                hosted.sql_query("SELECT 1 FROM tender_graph_deltas WHERE candidate_graph_id=? LIMIT 1", 'SELECT 1 FROM tender_graph_deltas WHERE candidate_graph_id=%s LIMIT 1'),
                (candidate_graph_id,),
            ).fetchone()
            if referenced:
                raise ValueError("reviewed candidate graph cannot be removed")
            self._conn.execute(hosted.sql_query("DELETE FROM sensitive_term_candidate_graphs WHERE candidate_graph_id=?", 'DELETE FROM sensitive_term_candidate_graphs WHERE candidate_graph_id=%s'), (candidate_graph_id,))
            self._conn.commit()

    def get_candidate_graph(self, candidate_graph_id: str) -> SensitiveTermCandidateGraph:
        row = self._conn.execute(
            hosted.sql_query("SELECT graph_json, content_hash FROM sensitive_term_candidate_graphs WHERE candidate_graph_id = ?", 'SELECT graph_json, content_hash FROM sensitive_term_candidate_graphs WHERE candidate_graph_id = %s'),
            (candidate_graph_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"candidate graph not found: {candidate_graph_id}")
        graph = SensitiveTermCandidateGraph.model_validate_json(row["graph_json"])
        if graph.graph_hash() != row["content_hash"]:
            raise ValueError(f"candidate graph content hash mismatch: {candidate_graph_id}")
        return graph

    def ensure_materialized_empty_base(self, candidate_graph_id: str) -> str:
        """Create the candidate's tenant-owned empty materialized base atomically."""
        with self._lock:
            hosted.begin_write(self._conn)
            try:
                owner = self.get_candidate_graph_owner(candidate_graph_id)
                if owner is None:
                    raise ValueError("candidate graph not found")
                if owner.get("allowed_base_graph_version"):
                    self._conn.commit()
                    return str(owner["allowed_base_graph_version"])
                graph_version = self._allocate_graph_version_uncommitted(candidate_graph_id)
                delta_id = f"baseline_delta_{uuid4().hex}"
                delta_payload = {"schema_version": "empty_graph_baseline_delta.v1",
                                 "candidate_graph_id": candidate_graph_id,
                                 "to_graph_version": graph_version, "operations": []}
                release = AcceptedGraphRelease(
                    graph_version=graph_version, parent_graph_version=None,
                    candidate_graph_id=candidate_graph_id, graph_delta_id=delta_id,
                    human_label_event_ids=(), nodes=(), edges=(), rules=(),
                )
                release_payload = {**release.model_dump(mode="json"),
                                   "release_hash": release.release_hash()}
                self._conn.execute(hosted.sql_query("""INSERT INTO tender_graph_deltas
                    (graph_delta_id,candidate_graph_id,from_graph_version,to_graph_version,
                     human_label_event_id,delta_json,content_hash) VALUES (?,?,?,?,?,?,?)""", 'INSERT INTO tender_graph_deltas\n                    (graph_delta_id,candidate_graph_id,from_graph_version,to_graph_version,\n                     human_label_event_id,delta_json,content_hash) VALUES (%s,%s,%s,%s,%s,%s,%s)'),
                    (delta_id, candidate_graph_id, None, graph_version, "system_empty_baseline",
                     _json(delta_payload), canonical_content_hash(delta_payload)))
                self._conn.execute(hosted.sql_query("""INSERT INTO tender_graph_releases
                    (graph_version,parent_graph_version,candidate_graph_id,graph_delta_id,
                     release_json,content_hash,organization_id,workspace_id,materialization_status)
                     VALUES (?,NULL,?,?,?,?,?,?,'materialized')""", "INSERT INTO tender_graph_releases\n                    (graph_version,parent_graph_version,candidate_graph_id,graph_delta_id,\n                     release_json,content_hash,organization_id,workspace_id,materialization_status)\n                     VALUES (%s,NULL,%s,%s,%s,%s,%s,%s,'materialized')"),
                    (graph_version, candidate_graph_id, delta_id, _json(release_payload),
                     release.release_hash(), owner.get("organization_id"), owner.get("workspace_id")))
                self._conn.execute(hosted.sql_query("""INSERT INTO tender_graph_release_materialization_events
                    (graph_version,status,occurred_at) VALUES (?,'materialized',?)""", "INSERT INTO tender_graph_release_materialization_events\n                    (graph_version,status,occurred_at) VALUES (%s,'materialized',%s)"),
                    (graph_version, self._now()))
                self._conn.execute(hosted.sql_query("UPDATE sensitive_term_candidate_graphs SET allowed_base_graph_version=? WHERE candidate_graph_id=?", 'UPDATE sensitive_term_candidate_graphs SET allowed_base_graph_version=%s WHERE candidate_graph_id=%s'),
                                   (graph_version, candidate_graph_id))
                self._conn.commit()
                return graph_version
            except Exception:
                self._conn.rollback()
                raise

    def ground_candidate_graph(
        self,
        *,
        human_event: LearningEventEnvelope,
        delta: GraphDelta,
    ) -> dict[str, Any]:
        if human_event.event_type not in {
            LearningEventType.HUMAN_ACCEPT,
            LearningEventType.HUMAN_EDIT,
        }:
            raise ValueError("only human accept/edit events can materialize graph assertions")
        if human_event.human_label is None:
            raise ValueError("human label is required")
        if human_event.graph_delta_ref != delta.graph_delta_id:
            raise ValueError("human event GraphDelta reference mismatch")
        if delta.human_label_event_id != human_event.event_id:
            raise ValueError("GraphDelta human event reference mismatch")
        graph = self.get_candidate_graph(delta.candidate_graph_id)
        owner = self.get_candidate_graph_owner(delta.candidate_graph_id) or {}
        if delta.from_graph_version != owner.get("allowed_base_graph_version"):
            raise ValueError("GraphDelta base graph version is not allowed for candidate")
        if delta.from_graph_version:
            base = self.get_graph_release(delta.from_graph_version,
                                          organization_id=owner.get("organization_id"),
                                          workspace_id=owner.get("workspace_id"))
            nodes = {str(item["node_id"]): dict(item) for item in base["nodes"]}
            edges = {str(item["edge_id"]): dict(item) for item in base["edges"]}
            rules = {str(item["rule_id"]): dict(item) for item in base["rules"]}
        else:
            nodes = {
                node.node_id: node.model_dump(mode="json") for node in graph.nodes
            }
            edges = {
                edge.edge_id: edge.model_dump(mode="json") for edge in graph.edges
            }
            rules = {
                rule.rule_id: rule.model_dump(mode="json") for rule in graph.rules
            }
        _apply_operations(nodes=nodes, edges=edges, rules=rules, delta=delta)
        release = AcceptedGraphRelease(
            graph_version=delta.to_graph_version,
            parent_graph_version=delta.from_graph_version,
            candidate_graph_id=delta.candidate_graph_id,
            graph_delta_id=delta.graph_delta_id,
            human_label_event_ids=[human_event.event_id],
            nodes=tuple(nodes[key] for key in sorted(nodes)),
            edges=tuple(edges[key] for key in sorted(edges)),
            rules=tuple(rules[key] for key in sorted(rules)),
        )
        release_payload = {
            **release.model_dump(mode="json"),
            "release_hash": release.release_hash(),
        }
        event_payload = human_event.model_dump(mode="json")
        event_hash = human_event.event_hash()
        attempt_id = f"attempt_{uuid4().hex}"
        with self._lock:
            hosted.begin_write(self._conn)
            try:
                current_owner = self.get_candidate_graph_owner(delta.candidate_graph_id) or {}
                if delta.from_graph_version != current_owner.get("allowed_base_graph_version"):
                    raise ValueError("candidate graph base changed; rebase review before grounding")
                active = self._conn.execute(hosted.sql_query("""SELECT r.graph_version FROM tender_graph_releases r
                    WHERE r.candidate_graph_id=? AND COALESCE(
                      (SELECT e.status FROM tender_graph_release_materialization_events e
                       WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),
                      r.materialization_status) IN ('pending','retry_pending') LIMIT 1""", "SELECT r.graph_version FROM tender_graph_releases r\n                    WHERE r.candidate_graph_id=%s AND COALESCE(\n                      (SELECT e.status FROM tender_graph_release_materialization_events e\n                       WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),\n                      r.materialization_status) IN ('pending','retry_pending') LIMIT 1"),
                    (delta.candidate_graph_id,),
                ).fetchone()
                if active:
                    raise ValueError(f"candidate graph materialization already in progress: {active['graph_version']}")
                delta = delta.model_copy(update={
                    "to_graph_version": self._allocate_graph_version_uncommitted(
                        delta.candidate_graph_id
                    )
                })
                release = release.model_copy(update={"graph_version": delta.to_graph_version})
                release_payload = {
                    **release.model_dump(mode="json"),
                    "release_hash": release.release_hash(),
                }
                prior = self._conn.execute(
                    hosted.sql_query("SELECT event_hash FROM knowledge_learning_events WHERE case_id=? ORDER BY occurred_at DESC, event_id DESC LIMIT 1", 'SELECT event_hash FROM knowledge_learning_events WHERE case_id=%s ORDER BY occurred_at DESC, event_id DESC LIMIT 1'),
                    (human_event.case_id,),
                ).fetchone()
                if human_event.prior_event_hash != (prior["event_hash"] if prior else None):
                    raise ValueError("learning event prior_event_hash does not match case history")
                self._conn.execute(
                    hosted.sql_query("INSERT INTO knowledge_learning_events (event_id,case_id,event_type,occurred_at,event_json,prior_event_hash,event_hash) VALUES (?,?,?,?,?,?,?)", 'INSERT INTO knowledge_learning_events (event_id,case_id,event_type,occurred_at,event_json,prior_event_hash,event_hash) VALUES (%s,%s,%s,%s,%s,%s,%s)'),
                    (human_event.event_id, human_event.case_id, human_event.event_type.value,
                     human_event.occurred_at.isoformat(), _json(event_payload), human_event.prior_event_hash, event_hash),
                )
                self._conn.execute(hosted.sql_query("""
                INSERT INTO tender_graph_deltas
                  (graph_delta_id, candidate_graph_id, from_graph_version,
                   to_graph_version, human_label_event_id, delta_json, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO tender_graph_deltas\n                  (graph_delta_id, candidate_graph_id, from_graph_version,\n                   to_graph_version, human_label_event_id, delta_json, content_hash)\n                VALUES (%s, %s, %s, %s, %s, %s, %s)\n                '), (
                delta.graph_delta_id,
                delta.candidate_graph_id,
                delta.from_graph_version,
                delta.to_graph_version,
                delta.human_label_event_id,
                _json(delta.model_dump(mode="json")),
                delta.delta_hash(),
                ))
                self._conn.execute(hosted.sql_query("""
                INSERT INTO tender_graph_releases
                  (graph_version, parent_graph_version, candidate_graph_id,
                   graph_delta_id, release_json, content_hash, organization_id, workspace_id,
                   materialization_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """, "\n                INSERT INTO tender_graph_releases\n                  (graph_version, parent_graph_version, candidate_graph_id,\n                   graph_delta_id, release_json, content_hash, organization_id, workspace_id,\n                   materialization_status)\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending')\n                "), (
                release.graph_version,
                release.parent_graph_version,
                release.candidate_graph_id,
                release.graph_delta_id,
                _json(release_payload),
                release.release_hash(),
                owner.get("organization_id"), owner.get("workspace_id"),
                ))
                self._conn.execute(
                    hosted.sql_query("INSERT INTO tender_graph_release_materialization_events (graph_version,status,occurred_at,attempt_id,lease_expires_at) VALUES (?,'pending',?,?,?)", "INSERT INTO tender_graph_release_materialization_events (graph_version,status,occurred_at,attempt_id,lease_expires_at) VALUES (%s,'pending',%s,%s,%s)"),
                    (release.graph_version, self._now(), attempt_id, self._lease_expires_at()),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        stored_event = {**event_payload, "event_hash": event_hash}
        try:
            self._graph_backend.materialize(release_payload)
            verify = getattr(self._graph_backend, "verify_release", None)
            if verify and not verify(release_payload):
                raise ValueError("external graph release missing after materialization")
        except Exception as exc:
            self._append_terminal_attempt(release.graph_version, attempt_id, "failed", str(exc))
            raise
        self._append_terminal_attempt(release.graph_version, attempt_id, "materialized")
        return {
            "human_event": stored_event,
            "graph_delta": {**delta.model_dump(mode="json"), "content_hash": delta.delta_hash()},
            "release": release_payload,
        }

    def record_review_decision(
        self,
        *,
        human_event: LearningEventEnvelope,
        delta: GraphDelta | None = None,
    ) -> dict[str, Any]:
        """Persist review intent without creating or mutating accepted topology."""
        if delta is not None:
            graph = self.get_candidate_graph(delta.candidate_graph_id)
            _validate_phase1_review_delta(delta, graph)
            if human_event.graph_delta_ref != delta.graph_delta_id:
                raise ValueError("human event GraphDelta reference mismatch")
            if delta.human_label_event_id != human_event.event_id:
                raise ValueError("GraphDelta human event reference mismatch")
            owner = self.get_candidate_graph_owner(delta.candidate_graph_id)
            if owner is None:
                raise ValueError("candidate graph not found")
            if delta.from_graph_version != owner.get("allowed_base_graph_version"):
                raise ValueError("GraphDelta base graph version is not allowed for candidate")
        elif human_event.graph_delta_ref is not None:
            raise ValueError("human event cannot reference an absent GraphDelta")

        with self._lock:
            hosted.begin_write(self._conn)
            try:
                human_event = self._with_governed_review_commitment(
                    human_event, delta=delta
                )
                stored_event = self._append_event_uncommitted(human_event)
                if delta is not None:
                    self._conn.execute(
                        hosted.sql_query("""INSERT INTO tender_graph_deltas
                        (graph_delta_id,candidate_graph_id,from_graph_version,to_graph_version,
                         human_label_event_id,delta_json,content_hash)
                        VALUES (?,?,?,?,?,?,?)""", 'INSERT INTO tender_graph_deltas\n                        (graph_delta_id,candidate_graph_id,from_graph_version,to_graph_version,\n                         human_label_event_id,delta_json,content_hash)\n                        VALUES (%s,%s,%s,%s,%s,%s,%s)'),
                        (
                            delta.graph_delta_id, delta.candidate_graph_id,
                            delta.from_graph_version, delta.to_graph_version,
                            delta.human_label_event_id,
                            _json(delta.model_dump(mode="json")), delta.delta_hash(),
                        ),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {
            "human_event": stored_event,
            "graph_delta": (
                {**delta.model_dump(mode="json"), "content_hash": delta.delta_hash()}
                if delta else None
            ),
            "release": None,
        }

    def _with_governed_review_commitment(
        self,
        event: LearningEventEnvelope,
        *,
        delta: GraphDelta | None = None,
    ) -> LearningEventEnvelope:
        """Derive new review commitments from exact store-owned candidate state."""

        if event.event_type not in {
            LearningEventType.HUMAN_ACCEPT,
            LearningEventType.HUMAN_EDIT,
        }:
            return event
        if event.event_type is LearningEventType.HUMAN_EDIT:
            if event.graph_delta_ref is None or delta is None:
                raise ValueError("edited review requires the exact GraphDelta for commitment")
        elif event.graph_delta_ref is not None or delta is not None:
            raise ValueError("accepted review cannot carry a GraphDelta")
        assertion = self._conn.execute(
            hosted.sql_query("SELECT candidate_graph_id,assertion_json,content_hash FROM "
            "knowledge_builder_document_candidate_assertions WHERE assertion_id=?", 'SELECT candidate_graph_id,assertion_json,content_hash FROM knowledge_builder_document_candidate_assertions WHERE assertion_id=%s'),
            (event.case_id,),
        ).fetchone()
        concept = self._conn.execute(
            hosted.sql_query("SELECT candidate_graph_id,concept_json,content_hash FROM "
            "contextgraph_reference_concepts WHERE concept_id=?", 'SELECT candidate_graph_id,concept_json,content_hash FROM contextgraph_reference_concepts WHERE concept_id=%s'),
            (event.case_id,),
        ).fetchone()
        if assertion is not None and concept is not None:
            raise ValueError("review evidence identity is ambiguous")
        row = assertion or concept
        if row is None or row["candidate_graph_id"] is None:
            return event
        payload_key = "assertion_json" if assertion is not None else "concept_json"
        payload = json.loads(row[payload_key])
        graph = self.get_candidate_graph(str(row["candidate_graph_id"]))
        pinned_graph_hash = payload.get("candidate_graph_hash")
        if pinned_graph_hash is None:
            return event
        stored_hash = str(row["content_hash"])
        expected_stored_hash = (
            canonical_content_hash(payload)
            if stored_hash.startswith("sha256:")
            else artifact_content_hash(payload)
        )
        if (
            expected_stored_hash != stored_hash
            or pinned_graph_hash != graph.graph_hash()
        ):
            raise ValueError("review evidence candidate graph binding mismatch")
        prefix = "assertion" if assertion is not None else "concept"
        derived = {
            "reviewed_candidate_graph_hash": graph.graph_hash(),
            "reviewed_evidence_ref": f"{prefix}:{event.case_id}",
            "reviewed_evidence_hash": (
                stored_hash
                if stored_hash.startswith("sha256:")
                else f"sha256:{stored_hash}"
            ),
        }
        if delta is not None:
            if event.graph_delta_ref != delta.graph_delta_id:
                raise ValueError("review GraphDelta identity mismatch")
            derived["reviewed_graph_delta_hash"] = delta.delta_hash()
        supplied = {
            key: getattr(event, key) for key in derived
        }
        if any(value is not None for value in supplied.values()) and supplied != derived:
            raise ValueError("caller review commitment differs from governed state")
        return LearningEventEnvelope.model_validate({
            **event.model_dump(mode="json"), **derived,
        })

    def put_relation_usage_grounding(
        self, decision: RelationUsageGroundingDecision
    ) -> dict[str, Any]:
        """Append one content-addressed relation-usage Grounding decision."""

        from services.relation_usage_grounding_service import (
            _create_relation_usage_grounding_authority,
        )

        authority = self._relation_usage_grounding_authority
        if authority is None:
            authority = _create_relation_usage_grounding_authority(
                self._db_path.resolve().parent,
                organization_id=decision.organization_id,
                workspace_id=decision.workspace_id,
            )
        RelationUsageGroundingDecision.validate_governed_semantics(
            decision, authority=authority, for_new_write=True
        )
        payload = decision.as_dict()
        source = decision.source_binding
        extract = decision.extract_run_binding
        schema = decision.schema_decision_binding
        subject = decision.subject_binding
        with self._lock:
            existing = self._conn.execute(
                hosted.sql_query("""SELECT organization_id,workspace_id,content_hash
                FROM relation_usage_grounding_decisions
                WHERE grounding_decision_ref=?""", 'SELECT organization_id,workspace_id,content_hash\n                FROM relation_usage_grounding_decisions\n                WHERE grounding_decision_ref=%s'),
                (decision.grounding_decision_ref,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["organization_id"] != decision.organization_id
                    or existing["workspace_id"] != decision.workspace_id
                ):
                    raise ValueError("grounding tenant scope mismatch")
                if existing["content_hash"] != decision.grounding_decision_hash:
                    raise ValueError(
                        "immutable relation usage grounding identity has different content"
                    )
                return self.get_relation_usage_grounding(
                    decision.grounding_decision_ref,
                    organization_id=decision.organization_id,
                    workspace_id=decision.workspace_id,
                )
            self._conn.execute(
                hosted.sql_query("""INSERT INTO relation_usage_grounding_decisions
                (organization_id,workspace_id,grounding_decision_ref,content_hash,
                 source_identity_ref,lineage_id,extract_run_ref,extract_result_hash,
                 schema_decision_ref,schema_decision_hash,subject_ref,subject_hash,
                 finding_kind,required_route,schema_learning_feedback_ref,
                 canonical_payload_json,reviewed_by,occurred_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", 'INSERT INTO relation_usage_grounding_decisions\n                (organization_id,workspace_id,grounding_decision_ref,content_hash,\n                 source_identity_ref,lineage_id,extract_run_ref,extract_result_hash,\n                 schema_decision_ref,schema_decision_hash,subject_ref,subject_hash,\n                 finding_kind,required_route,schema_learning_feedback_ref,\n                 canonical_payload_json,reviewed_by,occurred_at)\n                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                (
                    decision.organization_id,
                    decision.workspace_id,
                    decision.grounding_decision_ref,
                    decision.grounding_decision_hash,
                    str(source["source_identity_ref"]),
                    str(source["lineage_id"]),
                    str(extract["extract_run_ref"]),
                    str(extract["extract_result_hash"]),
                    str(schema["schema_decision_ref"]),
                    str(schema["schema_decision_hash"]),
                    str(subject["subject_ref"]),
                    str(subject["subject_hash"]),
                    decision.finding_kind,
                    decision.required_route,
                    decision.schema_learning_feedback_ref,
                    _json(payload),
                    decision.reviewed_by,
                    decision.occurred_at,
                ),
            )
            self._conn.commit()
        return self.get_relation_usage_grounding(
            decision.grounding_decision_ref,
            organization_id=decision.organization_id,
            workspace_id=decision.workspace_id,
        )

    def get_relation_usage_grounding(
        self,
        grounding_decision_ref: str,
        *,
        organization_id: str,
        workspace_id: str,
    ) -> dict[str, Any]:
        """Replay exact payload and duplicated columns, failing closed on tamper."""

        row = self._conn.execute(
            hosted.sql_query("""SELECT * FROM relation_usage_grounding_decisions
            WHERE grounding_decision_ref=?""", 'SELECT * FROM relation_usage_grounding_decisions\n            WHERE grounding_decision_ref=%s'),
            (grounding_decision_ref,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"relation usage grounding not found: {grounding_decision_ref}"
            )
        if (
            row["organization_id"] != organization_id
            or row["workspace_id"] != workspace_id
        ):
            raise ValueError("grounding tenant scope mismatch")
        try:
            payload = json.loads(row["canonical_payload_json"])
            decision = RelationUsageGroundingDecision._from_dict_structural(payload)
            from services.relation_usage_grounding_service import (
                _create_relation_usage_grounding_authority,
            )

            authority = self._relation_usage_grounding_authority
            if authority is None:
                authority = _create_relation_usage_grounding_authority(
                    self._db_path.resolve().parent,
                    organization_id=organization_id,
                    workspace_id=workspace_id,
                )
            RelationUsageGroundingDecision.validate_governed_semantics(
                decision, authority=authority, for_new_write=False
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"stored relation usage grounding payload is invalid: {exc}"
            ) from exc
        source = decision.source_binding
        extract = decision.extract_run_binding
        schema = decision.schema_decision_binding
        subject = decision.subject_binding
        metadata = {
            "organization_id": decision.organization_id,
            "workspace_id": decision.workspace_id,
            "grounding_decision_ref": decision.grounding_decision_ref,
            "content_hash": decision.grounding_decision_hash,
            "source_identity_ref": source["source_identity_ref"],
            "lineage_id": source["lineage_id"],
            "extract_run_ref": extract["extract_run_ref"],
            "extract_result_hash": extract["extract_result_hash"],
            "schema_decision_ref": schema["schema_decision_ref"],
            "schema_decision_hash": schema["schema_decision_hash"],
            "subject_ref": subject["subject_ref"],
            "subject_hash": subject["subject_hash"],
            "finding_kind": decision.finding_kind,
            "required_route": decision.required_route,
            "schema_learning_feedback_ref": decision.schema_learning_feedback_ref,
            "reviewed_by": decision.reviewed_by,
            "occurred_at": decision.occurred_at,
        }
        if any(row[key] != value for key, value in metadata.items()):
            raise ValueError("stored relation usage grounding metadata mismatch")
        if row["canonical_payload_json"] != _json(decision.as_dict()):
            raise ValueError("stored relation usage grounding canonical payload mismatch")
        return decision.as_dict()

    def get_relation_usage_grounding_for_feedback(
        self,
        schema_learning_feedback_ref: str,
        *,
        organization_id: str,
        workspace_id: str,
    ) -> dict[str, Any]:
        """Resolve exactly one governed Grounding route for persisted feedback."""

        rows = self._conn.execute(
            hosted.sql_query("""SELECT grounding_decision_ref
            FROM relation_usage_grounding_decisions
            WHERE schema_learning_feedback_ref=?
              AND organization_id=? AND workspace_id=?""", 'SELECT grounding_decision_ref\n            FROM relation_usage_grounding_decisions\n            WHERE schema_learning_feedback_ref=%s\n              AND organization_id=%s AND workspace_id=%s'),
            (schema_learning_feedback_ref, organization_id, workspace_id),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError(
                "schema learning feedback must bind exactly one grounding decision"
            )
        return self.get_relation_usage_grounding(
            str(rows[0]["grounding_decision_ref"]),
            organization_id=organization_id,
            workspace_id=workspace_id,
        )

    def _relation_occurrence_authority(
        self, *, organization_id: str, workspace_id: str
    ) -> object:
        from services.relation_usage_grounding_service import (
            _create_relation_usage_grounding_authority,
        )

        authority = self._relation_usage_grounding_authority
        if authority is not None:
            return authority
        return _create_relation_usage_grounding_authority(
            self._db_path.resolve().parent,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )

    @staticmethod
    def _normalize_unverified_v2_artifact(
        kind: str,
        payload: dict[str, Any],
        *,
        component_artifacts: tuple[tuple[bytes, str], ...] = (),
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validate canonical DTO integrity only; this confers no authority."""

        if not isinstance(payload, dict):
            raise ValueError("unverified v2 artifact must be an object")
        if kind == "canonical_restriction_pattern":
            from contracts.canonical_restriction_pattern import (
                parse_canonical_restriction_pattern_unverified,
            )

            parsed = parse_canonical_restriction_pattern_unverified(
                payload, component_artifacts=component_artifacts
            )
            normalized = parsed.model_dump(mode="json", by_alias=True)
            metadata = {
                "pattern_id": parsed.pattern_id,
                "pattern_hash": parsed.pattern_hash,
                "organization_id": parsed.organization_id,
                "workspace_id": parsed.workspace_id,
            }
        elif kind == "knowledge_relation_instance":
            from contracts.knowledge_relation_instance import (
                KnowledgeRelationInstanceV1,
            )

            parsed = KnowledgeRelationInstanceV1.model_validate(payload)
            normalized = parsed.model_dump(mode="json", by_alias=True)
            metadata = {
                "relation_instance_id": parsed.relation_instance_id,
                "relation_instance_hash": parsed.relation_instance_hash,
                "organization_id": parsed.organization_id,
                "workspace_id": parsed.workspace_id,
                "relation_type_id": parsed.relation_type_id,
                "representation_kind": parsed.representation_kind,
            }
        elif kind == "occurrence_grounding_manifest":
            from contracts.accepted_graph_release_v2 import (
                OccurrenceGroundingManifestV1,
            )

            parsed = OccurrenceGroundingManifestV1.parse_unverified(payload)
            normalized = parsed.as_dict()
            metadata = {
                "manifest_ref": parsed.manifest_ref,
                "manifest_hash": parsed.manifest_hash,
                "organization_id": parsed.organization_id,
                "workspace_id": parsed.workspace_id,
                "schema_decision_ref": parsed.schema_decision_ref,
                "schema_decision_hash": parsed.schema_decision_hash,
                "extract_result_ref": parsed.extract_result_ref,
                "extract_result_hash": parsed.extract_result_hash,
            }
        elif kind == "canonical_pattern_manifest":
            from contracts.accepted_graph_release_v2 import CanonicalPatternManifestV1

            parsed = CanonicalPatternManifestV1.parse_unverified(payload)
            normalized = parsed.as_dict()
            metadata = {
                "manifest_ref": parsed.manifest_ref,
                "manifest_hash": parsed.manifest_hash,
                "organization_id": parsed.organization_id,
                "workspace_id": parsed.workspace_id,
            }
        elif kind == "accepted_relation_binding":
            from contracts.knowledge_relation_instance import (
                parse_unverified_accepted_relation_binding,
            )

            parsed = parse_unverified_accepted_relation_binding(payload)
            normalized = parsed.model_dump(mode="json", by_alias=True)
            metadata = {
                "binding_id": parsed.binding_id,
                "binding_hash": parsed.binding_hash,
                "organization_id": parsed.organization_id,
                "workspace_id": parsed.workspace_id,
                "relation_instance_ref": parsed.relation_instance_ref,
                "relation_instance_hash": parsed.relation_instance_hash,
                "accepted_graph_release_ref": parsed.accepted_graph_release_ref,
                "accepted_graph_release_hash": parsed.accepted_graph_release_hash,
            }
        elif kind == "accepted_pattern_binding":
            expected_fields = {
                "schema_version", "binding_id", "binding_hash",
                "organization_id", "workspace_id", "pattern_ref",
                "pattern_hash", "accepted_graph_release_ref",
                "accepted_graph_release_hash",
            }
            if set(payload) != expected_fields:
                raise ValueError("accepted pattern binding must contain exact fields")
            normalized = dict(payload)
            for field in (
                "organization_id", "workspace_id", "pattern_ref",
                "accepted_graph_release_ref",
            ):
                value = normalized.get(field)
                if not isinstance(value, str) or not value or not value.isascii():
                    raise ValueError(f"{field} must be an ASCII-stable reference")
            if normalized.get("schema_version") != "accepted_pattern_binding.v1":
                raise ValueError("accepted pattern binding schema_version is invalid")
            release_ref = normalized["accepted_graph_release_ref"]
            release_suffix = release_ref[4:] if release_ref.startswith("KG_v") else ""
            if (
                not release_suffix.isdigit()
                or release_suffix.startswith("0")
                or int(release_suffix) < 1
            ):
                raise ValueError(
                    "accepted_graph_release_ref must be exact KG_v<positive integer>"
                )
            for field in (
                "binding_hash", "pattern_hash", "accepted_graph_release_hash"
            ):
                value = normalized.get(field)
                if (
                    not isinstance(value, str)
                    or len(value) != 71
                    or not value.startswith("sha256:")
                    or any(character not in "0123456789abcdef" for character in value[7:])
                ):
                    raise ValueError(f"{field} must be an exact sha256 ref")
            hash_payload = dict(normalized)
            hash_payload.pop("binding_id")
            hash_payload.pop("binding_hash")
            expected_hash = canonical_content_hash(hash_payload)
            if normalized["binding_hash"] != expected_hash:
                raise ValueError("accepted pattern binding hash mismatch")
            if normalized["binding_id"] != f"apb_{expected_hash[7:31]}":
                raise ValueError("accepted pattern binding identity mismatch")
            metadata = {
                key: normalized[key]
                for key in expected_fields
                if key not in {"schema_version"}
            }
        else:
            raise ValueError(f"unsupported unverified v2 artifact kind: {kind}")
        if normalized != payload:
            raise ValueError("unverified v2 artifact is not in canonical DTO form")
        return normalized, metadata

    @classmethod
    def _unverified_v2_artifact_identity(
        cls, kind: str, payload: dict[str, Any]
    ) -> str:
        _normalized, metadata = cls._normalize_unverified_v2_artifact(kind, payload)
        key = {
            "canonical_restriction_pattern": "pattern_id",
            "knowledge_relation_instance": "relation_instance_id",
            "occurrence_grounding_manifest": "manifest_ref",
            "canonical_pattern_manifest": "manifest_ref",
            "accepted_relation_binding": "binding_id",
            "accepted_pattern_binding": "binding_id",
        }[kind]
        return str(metadata[key])

    @staticmethod
    def _unverified_v2_artifact_table(kind: str) -> tuple[str, str]:
        return {
            "canonical_restriction_pattern": (
                "canonical_restriction_patterns", "pattern_id"
            ),
            "knowledge_relation_instance": (
                "knowledge_relation_instances", "relation_instance_id"
            ),
            "occurrence_grounding_manifest": (
                "occurrence_grounding_manifests", "manifest_ref"
            ),
            "canonical_pattern_manifest": (
                "canonical_pattern_manifests", "manifest_ref"
            ),
            "accepted_relation_binding": (
                "accepted_relation_bindings", "binding_id"
            ),
            "accepted_pattern_binding": (
                "accepted_pattern_bindings", "binding_id"
            ),
        }.get(kind, ("", ""))

    def _put_unverified_v2_artifact(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        component_artifacts: tuple[tuple[bytes, str], ...] = (),
    ) -> dict[str, Any]:
        """Append a canonical DTO without claiming authority validity."""

        with self._lock:
            hosted.begin_write(self._conn)
            try:
                normalized = self._put_unverified_v2_artifact_uncommitted(
                    kind, payload, component_artifacts=component_artifacts
                )
                self._conn.commit()
            except Exception:
                if hosted.in_transaction(self._conn):
                    self._conn.rollback()
                raise
        return normalized

    def _put_unverified_v2_artifact_uncommitted(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        component_artifacts: tuple[tuple[bytes, str], ...] = (),
    ) -> dict[str, Any]:
        """Append one canonical DTO inside the caller's existing transaction."""

        if not hosted.in_transaction(self._conn):
            raise ValueError("uncommitted v2 artifact append requires a transaction")
        normalized, metadata = self._normalize_unverified_v2_artifact(
            kind, payload, component_artifacts=component_artifacts
        )
        table, identity_column = self._unverified_v2_artifact_table(kind)
        if not table:
            raise ValueError(f"unsupported unverified v2 artifact kind: {kind}")
        canonical_payload = _json(normalized)
        columns = tuple(metadata) + ("canonical_payload_json",)
        values = tuple(metadata.values()) + (canonical_payload,)
        identity = str(metadata[identity_column])
        self._validate_unverified_binding_targets_uncommitted(kind, metadata)
        existing = self._conn.execute(
            hosted.sql_query(f"SELECT * FROM {table} WHERE {identity_column}=?", f'SELECT * FROM {table} WHERE {identity_column}=%s'), (identity,)
        ).fetchone()
        if existing is not None:
            if (
                existing["organization_id"] != metadata["organization_id"]
                or existing["workspace_id"] != metadata["workspace_id"]
            ):
                raise ValueError("unverified v2 artifact tenant scope mismatch")
            if existing["canonical_payload_json"] != canonical_payload:
                raise ValueError("immutable unverified v2 artifact content conflict")
            if any(existing[key] != value for key, value in metadata.items()):
                raise ValueError("stored unverified v2 artifact metadata mismatch")
            return json.loads(canonical_payload)
        placeholders = ",".join(hosted.sql_query("?", "%s") for _ in columns)
        self._conn.execute(
            f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
            values,
        )
        return json.loads(canonical_payload)

    def _validate_unverified_binding_targets_uncommitted(
        self, kind: str, metadata: dict[str, Any]
    ) -> None:
        """Pin binding targets within the same transaction as the append."""

        if kind not in {"accepted_relation_binding", "accepted_pattern_binding"}:
            return
        organization_id = str(metadata["organization_id"])
        workspace_id = str(metadata["workspace_id"])
        if kind == "accepted_relation_binding":
            target_kind = "knowledge_relation_instance"
            target_ref = str(metadata["relation_instance_ref"])
            target_hash = str(metadata["relation_instance_hash"])
            target_hash_field = "relationInstanceHash"
        else:
            target_kind = "canonical_restriction_pattern"
            target_ref = str(metadata["pattern_ref"])
            target_hash = str(metadata["pattern_hash"])
            target_hash_field = "pattern_hash"
        try:
            target = self._get_unverified_v2_artifact(
                target_kind,
                target_ref,
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
        except ValueError as exc:
            raise ValueError(f"binding target resolution failed: {exc}") from exc
        if target[target_hash_field] != target_hash:
            raise ValueError("binding target hash mismatch")

        release_ref = str(metadata["accepted_graph_release_ref"])
        release = self._conn.execute(
            hosted.sql_query("""SELECT release_json,content_hash,organization_id,workspace_id,
                      materialization_status,retirement_state,lineage_contract
               FROM tender_graph_releases WHERE graph_version=?""", 'SELECT release_json,content_hash,organization_id,workspace_id,\n                      materialization_status,retirement_state,lineage_contract\n               FROM tender_graph_releases WHERE graph_version=%s'),
            (release_ref,),
        ).fetchone()
        if release is None:
            raise ValueError(f"binding release not found: {release_ref}")
        if (
            release["organization_id"] != organization_id
            or release["workspace_id"] != workspace_id
        ):
            raise ValueError("binding release tenant scope mismatch")
        if release["content_hash"] != metadata["accepted_graph_release_hash"]:
            raise ValueError("binding release hash mismatch")
        if release["retirement_state"] != "active":
            raise ValueError("binding release must be active")
        lifecycle = self._conn.execute(
            hosted.sql_query("""SELECT status FROM tender_graph_release_materialization_events
               WHERE graph_version=? ORDER BY event_id DESC LIMIT 1""", 'SELECT status FROM tender_graph_release_materialization_events\n               WHERE graph_version=%s ORDER BY event_id DESC LIMIT 1'),
            (release_ref,),
        ).fetchone()
        materialization_status = (
            lifecycle["status"] if lifecycle is not None
            else release["materialization_status"]
        )
        if materialization_status != "materialized":
            raise ValueError("binding release must be materialized")
        if release["lineage_contract"] != "exact_v4_v3":
            raise ValueError("binding release must use exact_v4_v3 lineage")
        try:
            release_payload = json.loads(release["release_json"])
        except json.JSONDecodeError as exc:
            raise ValueError("binding release payload is invalid") from exc
        release_dto = self._validate_stored_release_dto(
            release_payload,
            stored_hash=str(release["content_hash"]),
            lineage_contract=str(release["lineage_contract"]),
            expected_graph_version=release_ref,
        )
        if kind == "accepted_relation_binding":
            matches = [
                item for item in release_dto.relation_membership
                if (
                    item.relation_instance_ref == metadata["relation_instance_ref"]
                    and item.relation_instance_hash
                    == metadata["relation_instance_hash"]
                )
            ]
        else:
            matches = [
                item for item in release_dto.pattern_membership
                if (
                    item.pattern_ref == metadata["pattern_ref"]
                    and item.pattern_hash == metadata["pattern_hash"]
                )
            ]
        if len(matches) != 1:
            raise ValueError(
                "binding release requires exactly one exact matching membership"
            )

    def _get_unverified_v2_artifact(
        self,
        kind: str,
        identity: str,
        *,
        organization_id: str,
        workspace_id: str,
    ) -> dict[str, Any]:
        """Replay canonical DTO integrity only; callers must not treat it as authority."""

        table, identity_column = self._unverified_v2_artifact_table(kind)
        if not table:
            raise ValueError(f"unsupported unverified v2 artifact kind: {kind}")
        row = self._conn.execute(
            hosted.sql_query(f"SELECT * FROM {table} WHERE {identity_column}=?", f'SELECT * FROM {table} WHERE {identity_column}=%s'), (identity,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unverified v2 artifact not found: {identity}")
        if (
            row["organization_id"] != organization_id
            or row["workspace_id"] != workspace_id
        ):
            raise ValueError("unverified v2 artifact tenant scope mismatch")
        try:
            payload = json.loads(row["canonical_payload_json"])
            component_artifacts: tuple[tuple[bytes, str], ...] = ()
            if (
                kind == "canonical_restriction_pattern"
                and payload.get("pattern_kind") == "composed"
            ):
                components: dict[str, tuple[bytes, str]] = {}
                visiting = {identity}

                def collect(component_ref: str) -> None:
                    if component_ref in components:
                        return
                    if component_ref in visiting:
                        raise ValueError("canonical pattern composition cycle")
                    component_row = self._conn.execute(
                        hosted.sql_query("SELECT * FROM canonical_restriction_patterns "
                        "WHERE pattern_id=?", 'SELECT * FROM canonical_restriction_patterns WHERE pattern_id=%s'),
                        (component_ref,),
                    ).fetchone()
                    if component_row is None:
                        raise ValueError("canonical pattern component is missing")
                    if (
                        component_row["organization_id"] != organization_id
                        or component_row["workspace_id"] != workspace_id
                    ):
                        raise ValueError(
                            "canonical pattern component tenant scope mismatch"
                        )
                    component_payload = json.loads(
                        component_row["canonical_payload_json"]
                    )
                    if (
                        component_payload.get("pattern_id") != component_ref
                        or component_row["pattern_hash"]
                        != component_payload.get("pattern_hash")
                        or component_payload.get("organization_id")
                        != organization_id
                        or component_payload.get("workspace_id") != workspace_id
                    ):
                        raise ValueError(
                            "stored canonical pattern component metadata mismatch"
                        )
                    visiting.add(component_ref)
                    for nested_ref in component_payload.get(
                        "composition", {}
                    ).get("member_pattern_refs", []):
                        collect(str(nested_ref))
                    visiting.remove(component_ref)
                    component_hash = component_payload.get("pattern_hash")
                    hash_payload = dict(component_payload)
                    hash_payload.pop("pattern_hash", None)
                    components[component_ref] = (
                        json.dumps(
                            hash_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8"),
                        component_hash,
                    )

                for component_ref in payload.get("composition", {}).get(
                    "member_pattern_refs", []
                ):
                    collect(str(component_ref))
                component_artifacts = tuple(
                    components[ref] for ref in sorted(components)
                )
            normalized, metadata = self._normalize_unverified_v2_artifact(
                kind, payload, component_artifacts=component_artifacts
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"stored unverified v2 artifact payload is invalid: {exc}") from exc
        if any(row[key] != value for key, value in metadata.items()):
            raise ValueError("stored unverified v2 artifact metadata mismatch")
        if row["canonical_payload_json"] != _json(normalized):
            raise ValueError("stored unverified v2 artifact canonical bytes mismatch")
        return json.loads(row["canonical_payload_json"])

    def put_relation_occurrence_grounding(
        self,
        content: dict[str, Any],
        *,
        extract_result: ExtractResult,
        relation_usage_grounding_decision: (
            RelationUsageGroundingDecision | dict[str, Any] | None
        ) = None,
    ) -> dict[str, Any]:
        """Construct and append one terminal occurrence Grounding record.

        The caller supplies decision material, not an already self-hashed record.
        The exact sealed W2e authority resolves every governed pin before the
        content-addressed contract is constructed.
        """

        organization_id = str(content.get("organization_id") or "")
        workspace_id = str(content.get("workspace_id") or "")
        authority = self._relation_occurrence_authority(
            organization_id=organization_id, workspace_id=workspace_id
        )
        record = RelationOccurrenceGroundingV1.create(
            content,
            extract_result=extract_result,
            relation_usage_grounding_decision=relation_usage_grounding_decision,
            relation_usage_authority=authority,
        )
        payload = record.as_dict()
        canonical_payload = _json(payload)
        # Seal the persistence boundary by replaying the exact bytes that will
        # be written. This prevents a mutable caller dictionary or a serializer
        # discrepancy from becoming a second authority after construction.
        record = RelationOccurrenceGroundingV1.from_dict(
            json.loads(canonical_payload),
            extract_result=extract_result,
            relation_usage_grounding_decision=relation_usage_grounding_decision,
            relation_usage_authority=authority,
        )
        payload = record.as_dict()
        occurrence_identity = (
            record.organization_id,
            record.workspace_id,
            record.extract_result_ref,
            record.extract_result_hash,
            record.extracted_occurrence_ref,
            record.extracted_occurrence_hash,
        )
        with self._lock:
            hosted.begin_write(self._conn)
            try:
                existing = self._conn.execute(
                    hosted.sql_query("""SELECT *
                       FROM relation_occurrence_groundings
                       WHERE grounding_record_id=?""", 'SELECT *\n                       FROM relation_occurrence_groundings\n                       WHERE grounding_record_id=%s'),
                    (record.grounding_record_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["organization_id"] != record.organization_id
                        or existing["workspace_id"] != record.workspace_id
                    ):
                        raise ValueError("relation occurrence Grounding tenant scope mismatch")
                    if (
                        existing["grounding_record_hash"]
                        != record.grounding_record_hash
                        or existing["canonical_payload_json"] != canonical_payload
                    ):
                        raise ValueError(
                            "immutable relation occurrence Grounding identity has different content"
                        )
                    expected_metadata = {
                        "extract_result_ref": record.extract_result_ref,
                        "extract_result_hash": record.extract_result_hash,
                        "schema_decision_ref": record.schema_decision_ref,
                        "schema_decision_hash": record.schema_decision_hash,
                        "extracted_occurrence_ref": record.extracted_occurrence_ref,
                        "extracted_occurrence_hash": record.extracted_occurrence_hash,
                        "outcome": record.outcome,
                        "actor_ref": record.actor_ref,
                        "decided_at": record.decided_at,
                    }
                    if any(
                        existing[key] != value
                        for key, value in expected_metadata.items()
                    ):
                        raise ValueError(
                            "stored relation occurrence Grounding metadata mismatch"
                        )
                    self._conn.commit()
                    return payload
                terminal = self._conn.execute(
                    hosted.sql_query("""SELECT grounding_record_id FROM relation_occurrence_groundings
                       WHERE organization_id=? AND workspace_id=?
                         AND extract_result_ref=? AND extract_result_hash=?
                         AND extracted_occurrence_ref=?
                         AND extracted_occurrence_hash=?""", 'SELECT grounding_record_id FROM relation_occurrence_groundings\n                       WHERE organization_id=%s AND workspace_id=%s\n                         AND extract_result_ref=%s AND extract_result_hash=%s\n                         AND extracted_occurrence_ref=%s\n                         AND extracted_occurrence_hash=%s'),
                    occurrence_identity,
                ).fetchone()
                if terminal is not None:
                    raise ValueError(
                        "terminal Grounding outcome already exists for exact occurrence"
                    )
                self._conn.execute(
                    hosted.sql_query("""INSERT INTO relation_occurrence_groundings
                       (grounding_record_id,grounding_record_hash,
                        organization_id,workspace_id,extract_result_ref,
                        extract_result_hash,schema_decision_ref,
                        schema_decision_hash,extracted_occurrence_ref,
                        extracted_occurrence_hash,outcome,canonical_payload_json,
                        actor_ref,decided_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", 'INSERT INTO relation_occurrence_groundings\n                       (grounding_record_id,grounding_record_hash,\n                        organization_id,workspace_id,extract_result_ref,\n                        extract_result_hash,schema_decision_ref,\n                        schema_decision_hash,extracted_occurrence_ref,\n                        extracted_occurrence_hash,outcome,canonical_payload_json,\n                        actor_ref,decided_at)\n                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                    (
                        record.grounding_record_id,
                        record.grounding_record_hash,
                        record.organization_id,
                        record.workspace_id,
                        record.extract_result_ref,
                        record.extract_result_hash,
                        record.schema_decision_ref,
                        record.schema_decision_hash,
                        record.extracted_occurrence_ref,
                        record.extracted_occurrence_hash,
                        record.outcome,
                        canonical_payload,
                        record.actor_ref,
                        record.decided_at,
                    ),
                )
                self._conn.commit()
            except Exception:
                if hosted.in_transaction(self._conn):
                    self._conn.rollback()
                raise
        return payload

    def get_relation_occurrence_grounding(
        self,
        grounding_record_id: str,
        *,
        organization_id: str,
        workspace_id: str,
    ) -> dict[str, Any]:
        """Replay exact stored bytes and revalidate historical governed pins."""

        row = self._conn.execute(
            hosted.sql_query("SELECT * FROM relation_occurrence_groundings WHERE grounding_record_id=?", 'SELECT * FROM relation_occurrence_groundings WHERE grounding_record_id=%s'),
            (grounding_record_id,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"relation occurrence Grounding not found: {grounding_record_id}"
            )
        if (
            row["organization_id"] != organization_id
            or row["workspace_id"] != workspace_id
        ):
            raise ValueError("relation occurrence Grounding tenant scope mismatch")
        try:
            payload = json.loads(row["canonical_payload_json"])
            structural = RelationOccurrenceGroundingV1._from_dict_structural(payload)
            authority = self._relation_occurrence_authority(
                organization_id=organization_id, workspace_id=workspace_id
            )
            from services.relation_usage_grounding_service import (
                _resolve_relation_occurrence_grounding_inputs,
            )

            extract_result, _occurrence = _resolve_relation_occurrence_grounding_inputs(
                authority,
                organization_id=organization_id,
                workspace_id=workspace_id,
                extract_result_ref=structural.extract_result_ref,
                extract_result_hash=structural.extract_result_hash,
                schema_decision_ref=structural.schema_decision_ref,
                schema_decision_hash=structural.schema_decision_hash,
                occurrence_ref=structural.extracted_occurrence_ref,
                occurrence_hash=structural.extracted_occurrence_hash,
                for_new_write=False,
            )
            grounding_decision = None
            if structural.relation_usage_grounding_decision_ref is not None:
                grounding_decision = self.get_relation_usage_grounding(
                    structural.relation_usage_grounding_decision_ref,
                    organization_id=organization_id,
                    workspace_id=workspace_id,
                )
            record = RelationOccurrenceGroundingV1.from_dict(
                payload,
                extract_result=extract_result,
                relation_usage_grounding_decision=grounding_decision,
                relation_usage_authority=authority,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"stored relation occurrence Grounding payload is invalid: {exc}"
            ) from exc
        metadata = {
            "grounding_record_id": record.grounding_record_id,
            "grounding_record_hash": record.grounding_record_hash,
            "organization_id": record.organization_id,
            "workspace_id": record.workspace_id,
            "extract_result_ref": record.extract_result_ref,
            "extract_result_hash": record.extract_result_hash,
            "schema_decision_ref": record.schema_decision_ref,
            "schema_decision_hash": record.schema_decision_hash,
            "extracted_occurrence_ref": record.extracted_occurrence_ref,
            "extracted_occurrence_hash": record.extracted_occurrence_hash,
            "outcome": record.outcome,
            "actor_ref": record.actor_ref,
            "decided_at": record.decided_at,
        }
        if any(row[key] != value for key, value in metadata.items()):
            raise ValueError("stored relation occurrence Grounding metadata mismatch")
        if row["canonical_payload_json"] != _json(record.as_dict()):
            raise ValueError(
                "stored relation occurrence Grounding canonical payload mismatch"
            )
        return record.as_dict()

    def list_relation_occurrence_groundings(
        self,
        *,
        extract_result_ref: str,
        organization_id: str,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            hosted.sql_query("""SELECT grounding_record_id FROM relation_occurrence_groundings
               WHERE organization_id=? AND workspace_id=? AND extract_result_ref=?
               ORDER BY grounding_record_id""", 'SELECT grounding_record_id FROM relation_occurrence_groundings\n               WHERE organization_id=%s AND workspace_id=%s AND extract_result_ref=%s\n               ORDER BY grounding_record_id'),
            (organization_id, workspace_id, extract_result_ref),
        ).fetchall()
        return [
            self.get_relation_occurrence_grounding(
                str(row["grounding_record_id"]),
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
            for row in rows
        ]

    def resolve_reviewed_candidate_selection(
        self,
        *,
        candidate_graph_id: str,
        human_event_ids: tuple[str, ...],
        organization_id: str,
        workspace_id: str | None,
    ) -> dict[str, Any]:
        """Resolve accepted rule/topology content without creating a release."""

        if not human_event_ids or len(human_event_ids) != len(set(human_event_ids)):
            raise ValueError("review selection requires unique human event IDs")
        ordered_event_ids = tuple(sorted(human_event_ids, key=lambda item: item.encode("utf-8")))
        owner = self.get_candidate_graph_owner(candidate_graph_id)
        if (
            owner is None
            or owner.get("organization_id") != organization_id
            or owner.get("workspace_id") != workspace_id
        ):
            raise ValueError("candidate graph not found for workspace")

        graph = self.get_candidate_graph(candidate_graph_id)
        admitted_anchor_ids: set[str] = set()
        events: list[LearningEventEnvelope] = []
        deltas: list[GraphDelta] = []
        for event_id in ordered_event_ids:
            row = self._conn.execute(
                hosted.sql_query("SELECT event_json,event_hash FROM knowledge_learning_events WHERE event_id=?", 'SELECT event_json,event_hash FROM knowledge_learning_events WHERE event_id=%s'),
                (event_id,),
            ).fetchone()
            if row is None:
                raise ValueError("review decision not found")
            event = LearningEventEnvelope.model_validate_json(row["event_json"])
            if event.event_hash() != row["event_hash"]:
                raise ValueError("review decision content hash mismatch")
            if event.event_type not in {
                LearningEventType.HUMAN_ACCEPT,
                LearningEventType.HUMAN_EDIT,
            }:
                raise ValueError("selection requires accepted or edited review decisions")
            if event.event_type is LearningEventType.HUMAN_EDIT:
                if (
                    event.graph_delta_ref is None
                    or event.reviewed_graph_delta_hash is None
                ):
                    raise ValueError("edited review lacks an immutable GraphDelta commitment")
            elif (
                event.graph_delta_ref is not None
                or event.reviewed_graph_delta_hash is not None
            ):
                raise ValueError("accepted review cannot carry GraphDelta authority")
            if event.source_document_hash != graph.source_document_hash:
                raise ValueError("review decision source hash mismatch")
            history_rows = self._conn.execute(
                hosted.sql_query("SELECT event_json,event_hash,event_type,prior_event_hash FROM "
                "knowledge_learning_events WHERE case_id=? "
                "ORDER BY occurred_at,event_id", 'SELECT event_json,event_hash,event_type,prior_event_hash FROM knowledge_learning_events WHERE case_id=%s ORDER BY occurred_at,event_id'),
                (event.case_id,),
            ).fetchall()
            prior_hash = None
            history: list[LearningEventEnvelope] = []
            for history_row in history_rows:
                history_event = LearningEventEnvelope.model_validate_json(
                    history_row["event_json"]
                )
                if (
                    history_event.event_hash() != history_row["event_hash"]
                    or history_event.event_type.value != history_row["event_type"]
                    or history_event.prior_event_hash != history_row["prior_event_hash"]
                    or history_event.prior_event_hash != prior_hash
                ):
                    raise ValueError("review decision case history is invalid")
                prior_hash = str(history_row["event_hash"])
                history.append(history_event)
            if not history or history[-1].event_id != event.event_id or len(history) < 2:
                raise ValueError(
                    "review decision is not the terminal case event: "
                    f"case={event.case_id}, history="
                    f"{[item.event_type.value for item in history]}"
                )
            proposal = history[-2]
            if (
                proposal.event_type is not LearningEventType.MACHINE_PROPOSAL
                or proposal.machine_proposal != graph.machine_proposal
                or proposal.source_document_hash != graph.source_document_hash
                or proposal.rule_version != event.rule_version
                or proposal.evaluation_corpus_version
                != event.evaluation_corpus_version
                or event.human_label is None
            ):
                raise ValueError("review decision does not bind the exact machine proposal")
            assertion = self._conn.execute(
                hosted.sql_query("SELECT assertion_json,content_hash,extraction_run_id,source_document_hash "
                "FROM knowledge_builder_document_candidate_assertions "
                "WHERE assertion_id=? AND candidate_graph_id=?", 'SELECT assertion_json,content_hash,extraction_run_id,source_document_hash FROM knowledge_builder_document_candidate_assertions WHERE assertion_id=%s AND candidate_graph_id=%s'),
                (event.case_id, candidate_graph_id),
            ).fetchone()
            concept = self._conn.execute(
                hosted.sql_query("SELECT concept_json,content_hash,extraction_run_id,source_document_hash "
                "FROM contextgraph_reference_concepts "
                "WHERE concept_id=? AND candidate_graph_id=?", 'SELECT concept_json,content_hash,extraction_run_id,source_document_hash FROM contextgraph_reference_concepts WHERE concept_id=%s AND candidate_graph_id=%s'),
                (event.case_id, candidate_graph_id),
            ).fetchone()
            if assertion is None and concept is None:
                raise ValueError("accepted review is not bound to candidate evidence")
            if assertion is not None:
                payload = json.loads(assertion["assertion_json"])
                stored_hash = str(assertion["content_hash"])
                expected_hash = (
                    canonical_content_hash(payload)
                    if stored_hash.startswith("sha256:")
                    else artifact_content_hash(payload)
                )
                if (
                    expected_hash != stored_hash
                    or assertion["extraction_run_id"] != graph.extraction_run_id
                    or assertion["source_document_hash"] != graph.source_document_hash
                    or payload.get("candidate_graph_hash") != graph.graph_hash()
                ):
                    raise ValueError("candidate assertion exact binding mismatch")
                review_anchor_ids = tuple(
                    str(anchor["anchor_id"])
                    for anchor in payload.get("source_anchors", [])
                    if anchor.get("anchor_id")
                )
                reviewed_evidence_ref = f"assertion:{event.case_id}"
            else:
                payload = json.loads(concept["concept_json"])
                stored_hash = str(concept["content_hash"])
                expected_hash = (
                    canonical_content_hash(payload)
                    if stored_hash.startswith("sha256:")
                    else artifact_content_hash(payload)
                )
                if (
                    expected_hash != stored_hash
                    or concept["extraction_run_id"] != graph.extraction_run_id
                    or concept["source_document_hash"] != graph.source_document_hash
                    or payload.get("candidate_graph_hash") != graph.graph_hash()
                ):
                    raise ValueError("reference concept exact binding mismatch")
                review_anchor_ids = tuple(
                    str(ref) for ref in payload.get("source_anchor_refs", []) if ref
                )
                reviewed_evidence_ref = f"concept:{event.case_id}"
            reviewed_evidence_hash = (
                stored_hash
                if stored_hash.startswith("sha256:")
                else f"sha256:{stored_hash}"
            )
            if (
                event.reviewed_candidate_graph_hash != graph.graph_hash()
                or event.reviewed_evidence_ref != reviewed_evidence_ref
                or event.reviewed_evidence_hash != reviewed_evidence_hash
            ):
                raise ValueError("review decision immutable content commitment mismatch")
            if (
                not review_anchor_ids
                or len(review_anchor_ids) != len(set(review_anchor_ids))
                or set(event.source_anchor_refs) != set(review_anchor_ids)
                or set(proposal.source_anchor_refs) != set(review_anchor_ids)
            ):
                raise ValueError("review decision evidence anchors mismatch")
            expected_graph_version = owner.get("allowed_base_graph_version") or "KG_v0"
            if event.graph_version != expected_graph_version or proposal.graph_version != expected_graph_version:
                raise ValueError(
                    "review decision graph version mismatch: "
                    f"expected={expected_graph_version}, event={event.graph_version}, "
                    f"proposal={proposal.graph_version}"
                )
            admitted_anchor_ids.update(review_anchor_ids)
            if event.graph_delta_ref:
                delta_row = self._conn.execute(
                    hosted.sql_query("SELECT delta_json,content_hash FROM tender_graph_deltas "
                    "WHERE graph_delta_id=? AND candidate_graph_id=?", 'SELECT delta_json,content_hash FROM tender_graph_deltas WHERE graph_delta_id=%s AND candidate_graph_id=%s'),
                    (event.graph_delta_ref, candidate_graph_id),
                ).fetchone()
                if delta_row is None:
                    raise ValueError("review GraphDelta not found")
                delta = GraphDelta.model_validate_json(delta_row["delta_json"])
                if delta.delta_hash() != delta_row["content_hash"]:
                    raise ValueError("review GraphDelta content hash mismatch")
                if event.reviewed_graph_delta_hash != delta_row["content_hash"]:
                    raise ValueError("review GraphDelta immutable content commitment mismatch")
                deltas.append(delta)
            events.append(event)
        if len(deltas) > 1 or (len(events) > 1 and deltas):
            raise ValueError("multi-review selection cannot combine GraphDelta edits")

        base_version = owner.get("allowed_base_graph_version")
        if base_version:
            base = self.get_graph_release(
                str(base_version),
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
            if any(base.get(field) for field in ("nodes", "edges", "rules")):
                raise ValueError(
                    "reviewed pattern selection cannot recover semantics from a nonempty accepted release"
                )
        nodes = {item.node_id: item.model_dump(mode="json") for item in graph.nodes}
        edges = {item.edge_id: item.model_dump(mode="json") for item in graph.edges}
        rules = {item.rule_id: item.model_dump(mode="json") for item in graph.rules}
        if deltas:
            delta = deltas[0]
            if delta.from_graph_version != base_version:
                raise ValueError("reviewed GraphDelta is stale against accepted base")
            _apply_operations(nodes=nodes, edges=edges, rules=rules, delta=delta)

        selected_rules = {
            rule_id: rule for rule_id, rule in rules.items()
            if admitted_anchor_ids.intersection(
                str(anchor.get("anchor_id"))
                for anchor in rule.get("source_anchors", [])
                if anchor.get("anchor_id")
            )
        }
        if not selected_rules:
            raise ValueError("accepted reviews do not admit any graph rule")
        known_node_ids = set(nodes)
        for rule in selected_rules.values():
            referenced = {
                str(item) for item in (
                    *rule.get("condition_node_ids", []),
                    *rule.get("exception_node_ids", []),
                )
            }
            if not referenced.issubset(known_node_ids):
                raise ValueError("reviewed rule has dangling condition/exception nodes")
        selected_node_ids = {
            str(node_id)
            for rule in selected_rules.values()
            for node_id in (
                *rule.get("condition_node_ids", []),
                *rule.get("exception_node_ids", []),
            )
        }
        selected_node_ids.update(
            str(node["node_id"])
            for node in nodes.values()
            if admitted_anchor_ids.intersection(
                str(anchor.get("anchor_id"))
                for anchor in node.get("source_anchors", [])
                if anchor.get("anchor_id")
            )
        )
        selected_nodes = {
            key: value for key, value in nodes.items() if key in selected_node_ids
        }
        selected_edges = {
            key: value for key, value in edges.items()
            if value.get("source_node_id") in selected_node_ids
            and value.get("target_node_id") in selected_node_ids
        }
        return {
            "candidate_graph_id": candidate_graph_id,
            "candidate_graph_hash": graph.graph_hash(),
            "graph_delta_id": deltas[0].graph_delta_id if deltas else None,
            "human_label_event_ids": list(ordered_event_ids),
            "human_label_event_hashes": [
                event.event_hash() for event in sorted(events, key=lambda item: item.event_id)
            ],
            "nodes": [selected_nodes[key] for key in sorted(selected_nodes)],
            "edges": [selected_edges[key] for key in sorted(selected_edges)],
            "rules": [selected_rules[key] for key in sorted(selected_rules)],
            "owner": dict(owner),
        }

    def materialize_reviewed_candidate(
        self,
        *,
        candidate_graph_id: str,
        human_event_id: str,
        organization_id: str,
        workspace_id: str | None,
    ) -> dict[str, Any]:
        """Create one immutable accepted release from an already-persisted review."""
        owner = self.get_candidate_graph_owner(candidate_graph_id)
        if (owner is None or owner.get("organization_id") != organization_id or
                owner.get("workspace_id") != workspace_id):
            raise ValueError("candidate graph not found for workspace")
        event_row = self._conn.execute(
            hosted.sql_query("SELECT event_json,event_hash FROM knowledge_learning_events WHERE event_id=?", 'SELECT event_json,event_hash FROM knowledge_learning_events WHERE event_id=%s'),
            (human_event_id,),
        ).fetchone()
        if event_row is None:
            raise ValueError("review decision not found")
        event = LearningEventEnvelope.model_validate_json(event_row["event_json"])
        if event.event_type not in {LearningEventType.HUMAN_ACCEPT, LearningEventType.HUMAN_EDIT}:
            raise ValueError("release requires an accepted or edited review decision")
        delta = None
        if event.graph_delta_ref:
            delta_row = self._conn.execute(
                hosted.sql_query("SELECT delta_json,content_hash FROM tender_graph_deltas WHERE graph_delta_id=? AND candidate_graph_id=?", 'SELECT delta_json,content_hash FROM tender_graph_deltas WHERE graph_delta_id=%s AND candidate_graph_id=%s'),
                (event.graph_delta_ref, candidate_graph_id),
            ).fetchone()
            if delta_row is None:
                raise ValueError("review GraphDelta not found")
            delta = GraphDelta.model_validate_json(delta_row["delta_json"])
            if delta.delta_hash() != delta_row["content_hash"]:
                raise ValueError("review GraphDelta content hash mismatch")
        prior_release = self._conn.execute(
            hosted.sql_query("SELECT 1 FROM tender_graph_releases WHERE candidate_graph_id=? AND release_json LIKE ? LIMIT 1", 'SELECT 1 FROM tender_graph_releases WHERE candidate_graph_id=%s AND release_json LIKE %s LIMIT 1'),
            (candidate_graph_id, f'%"{human_event_id}"%'),
        ).fetchone()
        if prior_release:
            raise ValueError("review decision already has an accepted release")

        graph = self.get_candidate_graph(candidate_graph_id)
        assertion_table = self._conn.execute(
            hosted.sql_query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_builder_document_candidate_assertions'", "SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname=current_schema() AND tablename='knowledge_builder_document_candidate_assertions'")
        ).fetchone()
        assertion_row = (self._conn.execute(
            hosted.sql_query("SELECT assertion_json FROM knowledge_builder_document_candidate_assertions WHERE assertion_id=? AND candidate_graph_id=?", 'SELECT assertion_json FROM knowledge_builder_document_candidate_assertions WHERE assertion_id=%s AND candidate_graph_id=%s'),
            (event.case_id, candidate_graph_id),
        ).fetchone() if assertion_table else None)
        reference_concept_table = self._conn.execute(
            hosted.sql_query("""SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='contextgraph_reference_concepts'""", "SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname=current_schema() AND tablename='contextgraph_reference_concepts'")
        ).fetchone()
        reference_concept_row = (
            self._conn.execute(
                hosted.sql_query("""SELECT concept_json FROM contextgraph_reference_concepts
                WHERE concept_id=? AND candidate_graph_id=?""", 'SELECT concept_json FROM contextgraph_reference_concepts\n                WHERE concept_id=%s AND candidate_graph_id=%s'),
                (event.case_id, candidate_graph_id),
            ).fetchone()
            if reference_concept_table
            else None
        )
        # Legacy graph-only callers predate Knowledge Builder assertions. Keep
        # their release behavior compatible; production Knowledge Builder
        # materialization always takes the granular branch below.
        admitted_anchor_ids = None
        if assertion_table:
            if assertion_row is None and reference_concept_row is None:
                raise ValueError("accepted review is not bound to a candidate assertion")
            if assertion_row is not None:
                assertion = json.loads(assertion_row["assertion_json"])
                admitted_anchor_ids = {
                    str(anchor.get("anchor_id"))
                    for anchor in assertion.get("source_anchors", [])
                    if anchor.get("anchor_id")
                }
            else:
                concept = json.loads(reference_concept_row["concept_json"])
                admitted_anchor_ids = {
                    str(anchor_ref)
                    for anchor_ref in concept.get("source_anchor_refs", [])
                    if anchor_ref
                }
        base_version = owner.get("allowed_base_graph_version")
        if base_version and delta is not None:
            base = self.get_graph_release(
                base_version, organization_id=organization_id, workspace_id=workspace_id,
            )
            nodes = {str(item["node_id"]): dict(item) for item in base["nodes"]}
            edges = {str(item["edge_id"]): dict(item) for item in base["edges"]}
            rules = {str(item["rule_id"]): dict(item) for item in base["rules"]}
        else:
            nodes = {item.node_id: item.model_dump(mode="json") for item in graph.nodes}
            edges = {item.edge_id: item.model_dump(mode="json") for item in graph.edges}
            rules = {item.rule_id: item.model_dump(mode="json") for item in graph.rules}
        if delta is not None:
            if delta.from_graph_version != base_version:
                raise ValueError("reviewed GraphDelta is stale against accepted base")
            _apply_operations(nodes=nodes, edges=edges, rules=rules, delta=delta)

        # An assertion-level accept admits only the graph topology justified by
        # that assertion.  The previous no-delta behavior copied the complete
        # candidate graph, allowing one accepted assertion to approve unrelated
        # rules.  Keep release authority granular by filtering rules and their
        # supporting topology by source-anchor intersection.
        admitted_rules = rules if admitted_anchor_ids is None else {
            rule_id: rule for rule_id, rule in rules.items()
            if admitted_anchor_ids.intersection(
                str(anchor.get("anchor_id"))
                for anchor in rule.get("source_anchors", [])
                if anchor.get("anchor_id")
            )
        }
        if admitted_anchor_ids is not None and not admitted_rules:
            raise ValueError("accepted review does not admit any graph rule")
        if admitted_anchor_ids is None:
            admitted_rules = rules
        admitted_rule_ids = set(admitted_rules)
        admitted_node_ids = {
            str(node_id)
            for rule in admitted_rules.values()
            for node_id in (*rule.get("condition_node_ids", []), *rule.get("exception_node_ids", []))
        }
        if admitted_anchor_ids is None:
            admitted_node_ids = set(nodes)
        admitted_node_ids.update(
            str(node["node_id"])
            for node in nodes.values()
            if admitted_anchor_ids is not None and admitted_anchor_ids.intersection(
                str(anchor.get("anchor_id"))
                for anchor in node.get("source_anchors", [])
                if anchor.get("anchor_id")
            )
        )
        admitted_nodes = {node_id: node for node_id, node in nodes.items() if node_id in admitted_node_ids}
        admitted_edges = {
            edge_id: edge for edge_id, edge in edges.items()
            if edge.get("source_node_id") in admitted_node_ids
            and edge.get("target_node_id") in admitted_node_ids
        }
        rules = admitted_rules
        nodes = admitted_nodes
        edges = admitted_edges

        attempt_id = f"attempt_{uuid4().hex}"
        review_command_hash = canonical_content_hash(
            {
                "candidate_graph_id": candidate_graph_id,
                "human_label_event_ids": [human_event_id],
            }
        )
        with self._lock:
            hosted.begin_write(self._conn)
            try:
                existing_command = self._conn.execute(
                    hosted.sql_query("""SELECT graph_version,release_json,content_hash,
                    COALESCE(
                      (SELECT e.status FROM tender_graph_release_materialization_events e
                       WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),
                      r.materialization_status) AS status
                    FROM tender_graph_releases r WHERE review_command_hash=?""", 'SELECT graph_version,release_json,content_hash,\n                    COALESCE(\n                      (SELECT e.status FROM tender_graph_release_materialization_events e\n                       WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),\n                      r.materialization_status) AS status\n                    FROM tender_graph_releases r WHERE review_command_hash=%s'),
                    (review_command_hash,),
                ).fetchone()
                if existing_command is not None:
                    self._conn.commit()
                    existing_payload = json.loads(existing_command["release_json"])
                    if existing_payload.get("release_hash") != existing_command["content_hash"]:
                        raise ValueError("review command release hash mismatch")
                    if existing_command["status"] == "materialized":
                        return existing_payload
                    raise ValueError(
                        "review command release already exists in "
                        f"{existing_command['status']}: {existing_command['graph_version']}"
                    )
                graph_version = self._allocate_graph_version_uncommitted(candidate_graph_id)
                release = AcceptedGraphRelease(
                    graph_version=graph_version,
                    parent_graph_version=base_version,
                    candidate_graph_id=candidate_graph_id,
                    graph_delta_id=delta.graph_delta_id if delta else None,
                    human_label_event_ids=(human_event_id,),
                    nodes=tuple(nodes[key] for key in sorted(nodes)),
                    edges=tuple(edges[key] for key in sorted(edges)),
                    rules=tuple(rules[key] for key in sorted(rules)),
                )
                payload = {**release.model_dump(mode="json"), "release_hash": release.release_hash()}
                self._conn.execute(
                    hosted.sql_query("""INSERT INTO tender_graph_releases
                    (graph_version,parent_graph_version,candidate_graph_id,graph_delta_id,
                     release_json,content_hash,organization_id,workspace_id,
                     materialization_status,review_command_hash,lineage_contract)
                    VALUES (?,?,?,?,?,?,?,?,'pending',?,?)""", "INSERT INTO tender_graph_releases\n                    (graph_version,parent_graph_version,candidate_graph_id,graph_delta_id,\n                     release_json,content_hash,organization_id,workspace_id,\n                     materialization_status,review_command_hash,lineage_contract)\n                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'pending',%s,%s)"),
                    (graph_version, base_version, candidate_graph_id,
                     release.graph_delta_id, _json(payload), release.release_hash(),
                     organization_id, workspace_id, review_command_hash,
                     "legacy"),
                )
                self._conn.execute(
                    hosted.sql_query("INSERT INTO tender_graph_release_materialization_events (graph_version,status,occurred_at,attempt_id,lease_expires_at) VALUES (?,'pending',?,?,?)", "INSERT INTO tender_graph_release_materialization_events (graph_version,status,occurred_at,attempt_id,lease_expires_at) VALUES (%s,'pending',%s,%s,%s)"),
                    (graph_version, self._now(), attempt_id, self._lease_expires_at()),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        try:
            self._graph_backend.materialize(payload)
            verify = getattr(self._graph_backend, "verify_release", None)
            if verify is None or not verify(payload):
                raise ValueError("accepted release verification failed")
        except Exception as exc:
            self._append_terminal_attempt(graph_version, attempt_id, "failed", str(exc))
            raise
        self._complete_materialization_attempt(graph_version, attempt_id)
        return payload

    def materialize_reviewed_candidates(
        self,
        *,
        candidate_graph_id: str,
        human_event_ids: tuple[str, ...],
        organization_id: str,
        workspace_id: str | None,
    ) -> dict[str, Any]:
        """Create one immutable accepted release from multiple persisted reviews."""
        ordered_event_ids = tuple(dict.fromkeys(human_event_ids))
        if not ordered_event_ids:
            raise ValueError("release requires at least one review decision")
        owner = self.get_candidate_graph_owner(candidate_graph_id)
        if (owner is None or owner.get("organization_id") != organization_id or
                owner.get("workspace_id") != workspace_id):
            raise ValueError("candidate graph not found for workspace")

        admitted_anchor_ids: set[str] = set()
        for human_event_id in ordered_event_ids:
            event_row = self._conn.execute(
                hosted.sql_query("SELECT event_json,event_hash FROM knowledge_learning_events WHERE event_id=?", 'SELECT event_json,event_hash FROM knowledge_learning_events WHERE event_id=%s'),
                (human_event_id,),
            ).fetchone()
            if event_row is None:
                raise ValueError("review decision not found")
            event = LearningEventEnvelope.model_validate_json(event_row["event_json"])
            if event.event_type not in {LearningEventType.HUMAN_ACCEPT, LearningEventType.HUMAN_EDIT}:
                raise ValueError("release requires accepted or edited review decisions")
            if event.graph_delta_ref:
                raise ValueError("batch release does not support GraphDelta edits")
            prior_release = self._conn.execute(
                hosted.sql_query("SELECT 1 FROM tender_graph_releases WHERE candidate_graph_id=? AND release_json LIKE ? LIMIT 1", 'SELECT 1 FROM tender_graph_releases WHERE candidate_graph_id=%s AND release_json LIKE %s LIMIT 1'),
                (candidate_graph_id, f'%"{human_event_id}"%'),
            ).fetchone()
            if prior_release:
                raise ValueError("review decision already has an accepted release")
            reference_concept_row = self._conn.execute(
                hosted.sql_query("""SELECT concept_json FROM contextgraph_reference_concepts
                WHERE concept_id=? AND candidate_graph_id=?""", 'SELECT concept_json FROM contextgraph_reference_concepts\n                WHERE concept_id=%s AND candidate_graph_id=%s'),
                (event.case_id, candidate_graph_id),
            ).fetchone()
            if reference_concept_row is None:
                raise ValueError("accepted review is not bound to a reference concept")
            concept = json.loads(reference_concept_row["concept_json"])
            admitted_anchor_ids.update(
                str(anchor_ref)
                for anchor_ref in concept.get("source_anchor_refs", [])
                if anchor_ref
            )
        if not admitted_anchor_ids:
            raise ValueError("accepted reviews do not admit any evidence anchors")

        graph = self.get_candidate_graph(candidate_graph_id)
        nodes = {item.node_id: item.model_dump(mode="json") for item in graph.nodes}
        edges = {item.edge_id: item.model_dump(mode="json") for item in graph.edges}
        rules = {item.rule_id: item.model_dump(mode="json") for item in graph.rules}
        admitted_rules = {
            rule_id: rule for rule_id, rule in rules.items()
            if admitted_anchor_ids.intersection(
                str(anchor.get("anchor_id"))
                for anchor in rule.get("source_anchors", [])
                if anchor.get("anchor_id")
            )
        }
        if not admitted_rules:
            raise ValueError("accepted reviews do not admit any graph rule")
        admitted_node_ids = {
            str(node_id)
            for rule in admitted_rules.values()
            for node_id in (*rule.get("condition_node_ids", []), *rule.get("exception_node_ids", []))
        }
        admitted_node_ids.update(
            str(node["node_id"])
            for node in nodes.values()
            if admitted_anchor_ids.intersection(
                str(anchor.get("anchor_id"))
                for anchor in node.get("source_anchors", [])
                if anchor.get("anchor_id")
            )
        )
        admitted_edges = {
            edge_id: edge for edge_id, edge in edges.items()
            if edge.get("source_node_id") in admitted_node_ids
            and edge.get("target_node_id") in admitted_node_ids
        }
        nodes = {node_id: node for node_id, node in nodes.items() if node_id in admitted_node_ids}
        edges = admitted_edges
        rules = admitted_rules

        base_version = owner.get("allowed_base_graph_version")
        attempt_id = f"attempt_{uuid4().hex}"
        review_command_hash = canonical_content_hash(
            {
                "candidate_graph_id": candidate_graph_id,
                "human_label_event_ids": list(ordered_event_ids),
            }
        )
        with self._lock:
            hosted.begin_write(self._conn)
            try:
                existing_command = self._conn.execute(
                    hosted.sql_query("""SELECT graph_version,release_json,content_hash,
                    COALESCE(
                      (SELECT e.status FROM tender_graph_release_materialization_events e
                       WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),
                      r.materialization_status) AS status
                    FROM tender_graph_releases r WHERE review_command_hash=?""", 'SELECT graph_version,release_json,content_hash,\n                    COALESCE(\n                      (SELECT e.status FROM tender_graph_release_materialization_events e\n                       WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),\n                      r.materialization_status) AS status\n                    FROM tender_graph_releases r WHERE review_command_hash=%s'),
                    (review_command_hash,),
                ).fetchone()
                if existing_command is not None:
                    self._conn.commit()
                    existing_payload = json.loads(existing_command["release_json"])
                    if existing_payload.get("release_hash") != existing_command["content_hash"]:
                        raise ValueError("review command release hash mismatch")
                    if existing_command["status"] == "materialized":
                        return existing_payload
                    raise ValueError(
                        "review command release already exists in "
                        f"{existing_command['status']}: {existing_command['graph_version']}"
                    )
                graph_version = self._allocate_graph_version_uncommitted(candidate_graph_id)
                release = AcceptedGraphRelease(
                    graph_version=graph_version,
                    parent_graph_version=base_version,
                    candidate_graph_id=candidate_graph_id,
                    graph_delta_id=None,
                    human_label_event_ids=ordered_event_ids,
                    nodes=tuple(nodes[key] for key in sorted(nodes)),
                    edges=tuple(edges[key] for key in sorted(edges)),
                    rules=tuple(rules[key] for key in sorted(rules)),
                )
                payload = {**release.model_dump(mode="json"), "release_hash": release.release_hash()}
                self._conn.execute(
                    hosted.sql_query("""INSERT INTO tender_graph_releases
                    (graph_version,parent_graph_version,candidate_graph_id,graph_delta_id,
                     release_json,content_hash,organization_id,workspace_id,
                     materialization_status,review_command_hash,lineage_contract)
                    VALUES (?,?,?,?,?,?,?,?,'pending',?,?)""", "INSERT INTO tender_graph_releases\n                    (graph_version,parent_graph_version,candidate_graph_id,graph_delta_id,\n                     release_json,content_hash,organization_id,workspace_id,\n                     materialization_status,review_command_hash,lineage_contract)\n                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'pending',%s,%s)"),
                    (graph_version, base_version, candidate_graph_id, None,
                     _json(payload), release.release_hash(), organization_id, workspace_id,
                     review_command_hash,
                     "legacy"),
                )
                self._conn.execute(
                    hosted.sql_query("INSERT INTO tender_graph_release_materialization_events (graph_version,status,occurred_at,attempt_id,lease_expires_at) VALUES (?,'pending',?,?,?)", "INSERT INTO tender_graph_release_materialization_events (graph_version,status,occurred_at,attempt_id,lease_expires_at) VALUES (%s,'pending',%s,%s,%s)"),
                    (graph_version, self._now(), attempt_id, self._lease_expires_at()),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        try:
            self._graph_backend.materialize(payload)
            verify = getattr(self._graph_backend, "verify_release", None)
            if verify is None or not verify(payload):
                raise ValueError("accepted release verification failed")
        except Exception as exc:
            self._append_terminal_attempt(graph_version, attempt_id, "failed", str(exc))
            raise
        self._complete_materialization_attempt(graph_version, attempt_id)
        return payload

    def _materialized_releases_for_candidate(
        self,
        *,
        candidate_graph_id: str,
        organization_id: str,
        workspace_id: str | None,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            hosted.sql_query("""SELECT r.graph_version,r.release_json,r.content_hash,r.lineage_contract
            FROM tender_graph_releases r
            WHERE r.candidate_graph_id=? AND r.organization_id=?
              AND r.workspace_id IS ? AND COALESCE(
                (SELECT e.status FROM tender_graph_release_materialization_events e
                 WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),
                r.materialization_status)='materialized'
            ORDER BY r.rowid""", "SELECT r.graph_version,r.release_json,r.content_hash,r.lineage_contract\n            FROM tender_graph_releases r\n            WHERE r.candidate_graph_id=%s AND r.organization_id=%s\n              AND r.workspace_id IS NOT DISTINCT FROM %s AND COALESCE(\n                (SELECT e.status FROM tender_graph_release_materialization_events e\n                 WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),\n                r.materialization_status)='materialized'\n            ORDER BY r.release_sequence"),
            (candidate_graph_id, organization_id, workspace_id),
        ).fetchall()
        releases: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["release_json"])
            self._validate_stored_release_dto(
                payload,
                stored_hash=str(row["content_hash"]),
                lineage_contract=str(row["lineage_contract"] or "legacy"),
                expected_graph_version=str(row["graph_version"]),
            )
            releases.append(payload)
        return releases

    def get_materialized_release_for_reviews(
        self,
        *,
        candidate_graph_id: str,
        human_event_ids: tuple[str, ...],
        organization_id: str,
        workspace_id: str | None,
    ) -> dict[str, Any] | None:
        """Recover the one exact materialized release for already-written reviews."""
        expected_ids = tuple(dict.fromkeys(human_event_ids))
        matches = [
            release
            for release in self._materialized_releases_for_candidate(
                candidate_graph_id=candidate_graph_id,
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
            if tuple(release.get("human_label_event_ids") or ()) == expected_ids
        ]
        if len(matches) > 1:
            raise ValueError("review decisions have multiple materialized releases")
        return matches[0] if matches else None

    def get_release_state_for_reviews(
        self,
        *,
        candidate_graph_id: str,
        human_event_ids: tuple[str, ...],
        organization_id: str,
        workspace_id: str | None,
    ) -> dict[str, Any] | None:
        """Recover the one exact release command, including a failed attempt."""
        expected_ids = tuple(dict.fromkeys(human_event_ids))
        rows = self._conn.execute(
            hosted.sql_query("""SELECT r.graph_version,r.release_json,r.content_hash,r.materialization_status,
            r.lineage_contract,
            COALESCE(
              (SELECT e.status FROM tender_graph_release_materialization_events e
               WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),
              r.materialization_status) AS lifecycle_status
            FROM tender_graph_releases r
            WHERE r.candidate_graph_id=? AND r.organization_id=?
              AND r.workspace_id IS ? ORDER BY r.rowid""", 'SELECT r.graph_version,r.release_json,r.content_hash,r.materialization_status,\n            r.lineage_contract,\n            COALESCE(\n              (SELECT e.status FROM tender_graph_release_materialization_events e\n               WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),\n              r.materialization_status) AS lifecycle_status\n            FROM tender_graph_releases r\n            WHERE r.candidate_graph_id=%s AND r.organization_id=%s\n              AND r.workspace_id IS NOT DISTINCT FROM %s ORDER BY r.release_sequence'),
            (candidate_graph_id, organization_id, workspace_id),
        ).fetchall()
        matches: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["release_json"])
            if tuple(payload.get("human_label_event_ids") or ()) != expected_ids:
                continue
            self._validate_stored_release_dto(
                payload,
                stored_hash=str(row["content_hash"]),
                lineage_contract=str(row["lineage_contract"] or "legacy"),
                expected_graph_version=str(row["graph_version"]),
            )
            matches.append(
                {
                    "release": payload,
                    "status": str(row["lifecycle_status"]),
                    "lineage_contract": str(row["lineage_contract"] or "legacy"),
                }
            )
        if len(matches) > 1:
            raise ValueError("review decisions have multiple accepted releases")
        return matches[0] if matches else None

    @staticmethod
    def _validate_stored_release_dto(
        payload: dict[str, Any], *, stored_hash: str, lineage_contract: str,
        expected_graph_version: str,
    ) -> Any:
        """Dispatch strict DTO replay without granting materialization authority."""

        if lineage_contract == "exact_v4_v3":
            from contracts.accepted_graph_release_v2 import AcceptedGraphReleaseV2

            release = AcceptedGraphReleaseV2.parse_unverified(payload)
            actual_hash = release.release_hash
            graph_version = release.graph_version
        elif lineage_contract in {"legacy", "exact_v3"}:
            if payload.get("schema_version") is not None:
                raise ValueError("legacy lineage discriminator rejects versioned release")
            legacy_payload = dict(payload)
            supplied_hash = legacy_payload.pop("release_hash", None)
            release = AcceptedGraphRelease.model_validate(legacy_payload)
            actual_hash = release.release_hash()
            graph_version = release.graph_version
            if supplied_hash != actual_hash:
                raise ValueError(
                    f"graph release content hash mismatch: {graph_version}"
                )
        else:
            raise ValueError(
                f"unsupported release lineage contract: {lineage_contract}"
            )
        if actual_hash != stored_hash or payload.get("release_hash") != stored_hash:
            raise ValueError(f"graph release content hash mismatch: {graph_version}")
        if graph_version != expected_graph_version:
            raise ValueError(
                "release payload graph_version differs from database identity"
            )
        return release

    def get_candidate_graph_delta(self, candidate_graph_id: str, graph_delta_id: str) -> GraphDelta | None:
        """Read a candidate-scoped immutable delta and verify its stored hash."""
        row = self._conn.execute(
            hosted.sql_query("SELECT delta_json,content_hash FROM tender_graph_deltas WHERE graph_delta_id=? AND candidate_graph_id=?", 'SELECT delta_json,content_hash FROM tender_graph_deltas WHERE graph_delta_id=%s AND candidate_graph_id=%s'),
            (graph_delta_id, candidate_graph_id),
        ).fetchone()
        if row is None:
            return None
        delta = GraphDelta.model_validate_json(row["delta_json"])
        if (delta.candidate_graph_id != candidate_graph_id or delta.graph_delta_id != graph_delta_id
                or delta.delta_hash() != row["content_hash"]):
            raise ValueError("candidate GraphDelta identity or content hash mismatch")
        return delta

    def get_learning_event(self, event_id: str) -> dict[str, Any] | None:
        """Read one immutable review event and verify its persisted hash."""
        row = self._conn.execute(
            hosted.sql_query("""SELECT event_json,event_hash FROM knowledge_learning_events
            WHERE event_id=?""", 'SELECT event_json,event_hash FROM knowledge_learning_events\n            WHERE event_id=%s'),
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        event = LearningEventEnvelope.model_validate_json(row["event_json"])
        if event.event_hash() != row["event_hash"]:
            raise ValueError("learning event content hash mismatch")
        return {**event.model_dump(mode="json"), "event_hash": row["event_hash"]}

    def _allocate_graph_version_uncommitted(self, candidate_graph_id: str) -> str:
        """Allocate under the serialized write boundary; PG sequence gaps are permitted."""
        cursor = self._conn.execute(
            hosted.sql_query("INSERT INTO tender_graph_version_allocations (candidate_graph_id,allocated_at) VALUES (?,?)", 'INSERT INTO tender_graph_version_allocations (candidate_graph_id,allocated_at) VALUES (%s,%s) RETURNING allocation_id'),
            (candidate_graph_id, self._now()),
        )
        allocation_id = int(cursor.fetchone()[0] if hosted.enabled() else cursor.lastrowid)
        while self._conn.execute(
            hosted.sql_query("SELECT 1 FROM tender_graph_releases WHERE graph_version=?", 'SELECT 1 FROM tender_graph_releases WHERE graph_version=%s'), (f"KG_v{allocation_id}",)
        ).fetchone():
            cursor = self._conn.execute(
                hosted.sql_query("INSERT INTO tender_graph_version_allocations (candidate_graph_id,allocated_at) VALUES (?,?)", 'INSERT INTO tender_graph_version_allocations (candidate_graph_id,allocated_at) VALUES (%s,%s) RETURNING allocation_id'),
                (candidate_graph_id, self._now()),
            )
            allocation_id = int(cursor.fetchone()[0] if hosted.enabled() else cursor.lastrowid)
        return f"KG_v{allocation_id}"

    def retry_graph_release(self, graph_version: str, *, organization_id: str,
                            workspace_id: str | None) -> dict[str, Any]:
        row = self._conn.execute(
            hosted.sql_query("SELECT release_json, content_hash, organization_id, workspace_id, candidate_graph_id, parent_graph_version, lineage_contract FROM tender_graph_releases WHERE graph_version=? AND retirement_state='active'", "SELECT release_json, content_hash, organization_id, workspace_id, candidate_graph_id, parent_graph_version, lineage_contract FROM tender_graph_releases WHERE graph_version=%s AND retirement_state='active'"),
            (graph_version,),
        ).fetchone()
        if row is None or row["organization_id"] != organization_id or row["workspace_id"] != workspace_id:
            raise ValueError("graph release not found for workspace")
        if row["lineage_contract"] == "exact_v4_v3":
            raise ValueError(
                "exact_v4_v3 requires the unavailable v2 materialization entrypoint"
            )
        payload = json.loads(row["release_json"])
        release = AcceptedGraphRelease.model_validate({k: v for k, v in payload.items() if k != "release_hash"})
        if release.release_hash() != row["content_hash"] or payload.get("release_hash") != row["content_hash"]:
            raise ValueError(f"graph release content hash mismatch: {graph_version}")
        attempt_id = f"attempt_{uuid4().hex}"
        with self._lock:
            hosted.begin_write(self._conn)
            try:
                state = self._conn.execute(
                    hosted.sql_query("SELECT status, lease_expires_at FROM tender_graph_release_materialization_events WHERE graph_version=? ORDER BY event_id DESC LIMIT 1", 'SELECT status, lease_expires_at FROM tender_graph_release_materialization_events WHERE graph_version=%s ORDER BY event_id DESC LIMIT 1'),
                    (graph_version,),
                ).fetchone()
                latest = state["status"] if state else self._conn.execute(
                    hosted.sql_query("SELECT materialization_status FROM tender_graph_releases WHERE graph_version=?", 'SELECT materialization_status FROM tender_graph_releases WHERE graph_version=%s'),
                    (graph_version,),
                ).fetchone()["materialization_status"]
                if latest == "materialized":
                    self._conn.commit()
                    return payload
                if latest in {"pending", "retry_pending"} and state and self._lease_is_live(state["lease_expires_at"]):
                    raise ValueError(f"graph release materialization already in progress: {graph_version}")
                if latest not in {"failed", "pending", "retry_pending"}:
                    raise ValueError(f"graph release is not retryable from state {latest}: {graph_version}")
                sibling = self._conn.execute(hosted.sql_query("""SELECT r.graph_version,
                    (SELECT e.status FROM tender_graph_release_materialization_events e
                     WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1) AS status
                    FROM tender_graph_releases r WHERE r.candidate_graph_id=? AND r.graph_version<>?
                    AND COALESCE((SELECT e.status FROM tender_graph_release_materialization_events e
                      WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),
                      r.materialization_status) IN ('pending','retry_pending') LIMIT 1""", "SELECT r.graph_version,\n                    (SELECT e.status FROM tender_graph_release_materialization_events e\n                     WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1) AS status\n                    FROM tender_graph_releases r WHERE r.candidate_graph_id=%s AND r.graph_version<>%s\n                    AND COALESCE((SELECT e.status FROM tender_graph_release_materialization_events e\n                      WHERE e.graph_version=r.graph_version ORDER BY e.event_id DESC LIMIT 1),\n                      r.materialization_status) IN ('pending','retry_pending') LIMIT 1"),
                    (row["candidate_graph_id"], graph_version),
                ).fetchone()
                if sibling:
                    raise ValueError(f"candidate graph has another active materialization: {sibling['graph_version']}")
                current = self.get_candidate_graph_owner(row["candidate_graph_id"]) or {}
                if current.get("allowed_base_graph_version") != row["parent_graph_version"]:
                    raise ValueError("graph release base is stale; retry is not canonical")
                self._conn.execute(
                    hosted.sql_query("INSERT INTO tender_graph_release_materialization_events (graph_version,status,occurred_at,attempt_id,lease_expires_at) VALUES (?,'retry_pending',?,?,?)", "INSERT INTO tender_graph_release_materialization_events (graph_version,status,occurred_at,attempt_id,lease_expires_at) VALUES (%s,'retry_pending',%s,%s,%s)"),
                    (graph_version, self._now(), attempt_id, self._lease_expires_at()),
                )
                self._conn.commit()
            except Exception:
                if hosted.in_transaction(self._conn):
                    self._conn.rollback()
                raise
        try:
            verify = getattr(self._graph_backend, "verify_release", None)
            already_materialized = bool(verify(payload)) if verify else False
            if not already_materialized:
                self._graph_backend.materialize(payload)
                if verify and not verify(payload):
                    raise ValueError("external graph release missing after materialization")
        except Exception as exc:
            self._append_terminal_attempt(graph_version, attempt_id, "failed", str(exc))
            raise
        self._complete_materialization_attempt(graph_version, attempt_id)
        return payload

    def get_release_lineage_contract(
        self,
        graph_version: str,
        *,
        organization_id: str,
        workspace_id: str | None,
    ) -> str:
        """Classify retry verification without inferring lineage for legacy releases."""
        row = self._conn.execute(
            hosted.sql_query("""SELECT lineage_contract FROM tender_graph_releases
            WHERE graph_version=? AND organization_id=? AND workspace_id IS ?""", 'SELECT lineage_contract FROM tender_graph_releases\n            WHERE graph_version=%s AND organization_id=%s AND workspace_id IS NOT DISTINCT FROM %s'),
            (graph_version, organization_id, workspace_id),
        ).fetchone()
        if row is None:
            raise ValueError("graph release not found for workspace")
        contract = str(row["lineage_contract"] or "legacy")
        if contract not in {"legacy", "exact_v3", "exact_v4_v3"}:
            raise ValueError(f"unsupported graph release lineage contract: {contract}")
        return contract

    def _complete_materialization_attempt(
        self,
        graph_version: str,
        attempt_id: str,
    ) -> None:
        try:
            self._append_terminal_attempt(graph_version, attempt_id, "materialized")
        except Exception as exc:
            self._append_terminal_attempt(
                graph_version,
                attempt_id,
                "failed",
                str(exc),
            )
            raise

    def _append_terminal_attempt(self, graph_version: str, attempt_id: str, status: str,
                                 error_message: str | None = None) -> None:
        if status not in {"materialized", "failed"}:
            raise ValueError("terminal materialization status must be materialized or failed")
        with self._lock:
            hosted.begin_write(self._conn)
            try:
                latest = self._conn.execute(
                    hosted.sql_query("SELECT status, attempt_id FROM tender_graph_release_materialization_events WHERE graph_version=? ORDER BY event_id DESC LIMIT 1", 'SELECT status, attempt_id FROM tender_graph_release_materialization_events WHERE graph_version=%s ORDER BY event_id DESC LIMIT 1'),
                    (graph_version,),
                ).fetchone()
                if (latest is None or latest["status"] not in {"pending", "retry_pending"} or
                        latest["attempt_id"] != attempt_id):
                    raise ValueError(f"stale materialization attempt: {attempt_id}")
                self._conn.execute(hosted.sql_query("""INSERT INTO tender_graph_release_materialization_events
                    (graph_version,status,occurred_at,error_message,attempt_id)
                    VALUES (?,?,?,?,?)""", 'INSERT INTO tender_graph_release_materialization_events\n                    (graph_version,status,occurred_at,error_message,attempt_id)\n                    VALUES (%s,%s,%s,%s,%s)'),
                    (graph_version, status, self._now(), error_message, attempt_id),
                )
                if status == "materialized":
                    lineage_contract = self._conn.execute(
                        hosted.sql_query("""SELECT lineage_contract FROM tender_graph_releases
                        WHERE graph_version=?""", 'SELECT lineage_contract FROM tender_graph_releases\n                        WHERE graph_version=%s'),
                        (graph_version,),
                    ).fetchone()["lineage_contract"]
                    if lineage_contract == "exact_v3":
                        origin_id = verify_exact_release_origin_reservation(
                            self._conn,
                            graph_version=graph_version,
                        )
                        promoted = self._conn.execute(
                            hosted.sql_query("""UPDATE contextgraph_artifact_origins
                            SET authority_state='authoritative'
                            WHERE origin_id=? AND authority_state='candidate'""", "UPDATE contextgraph_artifact_origins\n                            SET authority_state='authoritative'\n                            WHERE origin_id=%s AND authority_state='candidate'"),
                            (origin_id,),
                        )
                        if promoted.rowcount != 1:
                            raise ValueError(
                                "exact release candidate origin promotion failed"
                            )
                    elif lineage_contract == "legacy":
                        self._conn.execute(
                            hosted.sql_query("""UPDATE contextgraph_artifact_origins
                            SET authority_state='authoritative'
                            WHERE artifact_id=? AND relation_kind='accepted_release'
                              AND authority_state='candidate'""", "UPDATE contextgraph_artifact_origins\n                            SET authority_state='authoritative'\n                            WHERE artifact_id=%s AND relation_kind='accepted_release'\n                              AND authority_state='candidate'"),
                            (graph_version,),
                        )
                    elif lineage_contract == "exact_v4_v3":
                        raise ValueError(
                            "exact_v4_v3 requires the v2 materialization entrypoint"
                        )
                    else:
                        raise ValueError(
                            f"unsupported release lineage contract: {lineage_contract}"
                        )
                    release_row = self._conn.execute(
                        hosted.sql_query("SELECT release_json,content_hash,graph_delta_id FROM tender_graph_releases WHERE graph_version=?", 'SELECT release_json,content_hash,graph_delta_id FROM tender_graph_releases WHERE graph_version=%s'),
                        (graph_version,),
                    ).fetchone()
                    delta_row = self._conn.execute(
                        hosted.sql_query("SELECT human_label_event_id,delta_json FROM tender_graph_deltas WHERE graph_delta_id=?", 'SELECT human_label_event_id,delta_json FROM tender_graph_deltas WHERE graph_delta_id=%s'),
                        (release_row["graph_delta_id"],),
                    ).fetchone() if release_row["graph_delta_id"] else None
                    release = json.loads(release_row["release_json"])
                    assertion_table = self._conn.execute(
                        hosted.sql_query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge_builder_document_candidate_assertions'", "SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname=current_schema() AND tablename='knowledge_builder_document_candidate_assertions'")
                    ).fetchone()
                    reference_concept_table = self._conn.execute(
                        hosted.sql_query("""SELECT 1 FROM sqlite_master WHERE type='table'
                        AND name='contextgraph_reference_concepts'""", "SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname=current_schema() AND tablename='contextgraph_reference_concepts'")
                    ).fetchone()
                    human_event_ids = (
                        (delta_row["human_label_event_id"],) if delta_row
                        else tuple(release["human_label_event_ids"])
                    )
                    for human_event_id in human_event_ids:
                        event_row = (self._conn.execute(
                            hosted.sql_query("""SELECT e.case_id FROM knowledge_learning_events e
                            JOIN knowledge_builder_document_candidate_assertions a ON a.assertion_id=e.case_id
                            WHERE e.event_id=?""", 'SELECT e.case_id FROM knowledge_learning_events e\n                            JOIN knowledge_builder_document_candidate_assertions a ON a.assertion_id=e.case_id\n                            WHERE e.event_id=%s'),
                            (human_event_id,),
                        ).fetchone() if assertion_table else None)
                        if event_row:
                            delta_payload = json.loads(delta_row["delta_json"]) if delta_row else None
                            targets = (
                                {ref for op in delta_payload["operations"] for ref in op["target_refs"]}
                                if delta_payload else {
                                    *(item["node_id"] for item in release["nodes"]),
                                    *(item["edge_id"] for item in release["edges"]),
                                    *(item["rule_id"] for item in release["rules"]),
                                }
                            )
                            selected_edges = [item for item in release["edges"] if item.get("edge_id") in targets]
                            edge_ids = sorted(item["edge_id"] for item in selected_edges)
                            endpoint_ids = {ref for item in selected_edges for ref in
                                            (item.get("source_node_id"), item.get("target_node_id")) if ref}
                            node_ids = sorted(item["node_id"] for item in release["nodes"]
                                              if item.get("node_id") in targets | endpoint_ids)
                            rule_ids = sorted(item["rule_id"] for item in release["rules"] if item.get("rule_id") in targets)
                            self._conn.execute(hosted.sql_query("""INSERT INTO knowledge_builder_assertion_grounding_bindings
                                (assertion_id,human_label_event_id,graph_version,graph_release_hash,
                                 accepted_node_ids_json,accepted_edge_ids_json,accepted_rule_ids_json,materialized_at)
                                VALUES (?,?,?,?,?,?,?,?)""", 'INSERT INTO knowledge_builder_assertion_grounding_bindings\n                                (assertion_id,human_label_event_id,graph_version,graph_release_hash,\n                                 accepted_node_ids_json,accepted_edge_ids_json,accepted_rule_ids_json,materialized_at)\n                                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'), (
                                event_row["case_id"], human_event_id, graph_version,
                                release_row["content_hash"], _json(node_ids), _json(edge_ids),
                                _json(rule_ids), self._now(),
                            ))
                        else:
                            concept_row = (self._conn.execute(
                                hosted.sql_query("""SELECT e.case_id FROM knowledge_learning_events e
                                JOIN contextgraph_reference_concepts c ON c.concept_id=e.case_id
                                WHERE e.event_id=?""", 'SELECT e.case_id FROM knowledge_learning_events e\n                                JOIN contextgraph_reference_concepts c ON c.concept_id=e.case_id\n                                WHERE e.event_id=%s'),
                                (human_event_id,),
                            ).fetchone() if reference_concept_table else None)
                            if concept_row:
                                node_ids = sorted(item["node_id"] for item in release["nodes"])
                                edge_ids = sorted(item["edge_id"] for item in release["edges"])
                                rule_ids = sorted(item["rule_id"] for item in release["rules"])
                                self._conn.execute(hosted.sql_query("""INSERT INTO contextgraph_reference_concept_grounding_bindings
                                    (concept_id,human_label_event_id,graph_version,graph_release_hash,
                                     accepted_node_ids_json,accepted_edge_ids_json,accepted_rule_ids_json,materialized_at)
                                    VALUES (?,?,?,?,?,?,?,?)""", 'INSERT INTO contextgraph_reference_concept_grounding_bindings\n                                    (concept_id,human_label_event_id,graph_version,graph_release_hash,\n                                     accepted_node_ids_json,accepted_edge_ids_json,accepted_rule_ids_json,materialized_at)\n                                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'), (
                                    concept_row["case_id"], human_event_id, graph_version,
                                    release_row["content_hash"], _json(node_ids), _json(edge_ids),
                                    _json(rule_ids), self._now(),
                                ))
                self._conn.commit()
            except Exception:
                if hosted.in_transaction(self._conn):
                    self._conn.rollback()
                raise

    def persist_source_semantic_witness_package(
        self,
        *,
        manifest: SourceSemanticWitnessSetManifestV1,
        witnesses: tuple[SourceSemanticWitnessV1, ...],
        artifact_ref: str,
    ) -> dict[str, Any]:
        """Atomically append witness reviews, their set manifest, and origin."""
        manifest.verify_witnesses(witnesses)
        artifact_path = Path(artifact_ref)
        if (
            artifact_path.is_absolute()
            or ".." in artifact_path.parts
            or not artifact_path.parts
            or artifact_path.parts[0] != "artifacts"
        ):
            raise ValueError("witness manifest artifact_ref must be a governed artifact path")
        states = {item.review.state for item in witnesses}
        manifest_state = next(iter(states)) if len(states) == 1 else "edited"
        actors = {item.review.actor_ref for item in witnesses}
        actor_ref = next(iter(actors)) if len(actors) == 1 else "system:mixed-review-set"
        occurred_at = self._now()
        manifest_json = manifest.canonical_json()

        with self._lock:
            hosted.begin_write(self._conn)
            try:
                if manifest.predecessor_manifest_ref is not None:
                    predecessor = self._conn.execute(
                        hosted.sql_query("""SELECT 1 FROM source_semantic_witness_manifests
                        WHERE organization_id=? AND workspace_id=? AND manifest_ref=?""", 'SELECT 1 FROM source_semantic_witness_manifests\n                        WHERE organization_id=%s AND workspace_id=%s AND manifest_ref=%s'),
                        (
                            manifest.organization_id,
                            manifest.workspace_id,
                            manifest.predecessor_manifest_ref,
                        ),
                    ).fetchone()
                    if predecessor is None:
                        raise ValueError("manifest predecessor is missing from the same scope")

                for witness in witnesses:
                    if witness.review.predecessor_witness_ref is not None:
                        predecessor = self._conn.execute(
                            hosted.sql_query("""SELECT 1 FROM source_semantic_witness_reviews
                            WHERE organization_id=? AND workspace_id=? AND witness_ref=?""", 'SELECT 1 FROM source_semantic_witness_reviews\n                            WHERE organization_id=%s AND workspace_id=%s AND witness_ref=%s'),
                            (
                                witness.organization_id,
                                witness.workspace_id,
                                witness.review.predecessor_witness_ref,
                            ),
                        ).fetchone()
                        if predecessor is None:
                            raise ValueError("witness predecessor is missing from the same scope")
                    payload_json = witness.canonical_json()
                    existing = self._conn.execute(
                        hosted.sql_query("SELECT * FROM source_semantic_witness_reviews WHERE witness_ref=?", 'SELECT * FROM source_semantic_witness_reviews WHERE witness_ref=%s'),
                        (witness.witness_id,),
                    ).fetchone()
                    expected = {
                        "organization_id": witness.organization_id,
                        "workspace_id": witness.workspace_id,
                        "witness_hash": witness.witness_hash,
                        "lifecycle_state": witness.review.state,
                        "predecessor_witness_ref": witness.review.predecessor_witness_ref,
                        "successor_witness_ref": witness.review.successor_witness_ref,
                        "review_ref": witness.review.review_ref,
                        "review_hash": witness.review.review_hash,
                        "canonical_payload_json": payload_json,
                        "actor_ref": witness.review.actor_ref,
                        "decision_evidence_refs_json": _json(
                            [
                                item.model_dump(mode="json", by_alias=True)
                                for item in witness.review.decision_evidence_refs
                            ]
                        ),
                    }
                    if existing is not None:
                        if any(existing[key] != value for key, value in expected.items()):
                            raise ValueError("witness ref/hash/payload/scope conflict")
                        continue
                    self._conn.execute(
                        hosted.sql_query("""INSERT INTO source_semantic_witness_reviews
                        (organization_id,workspace_id,witness_ref,witness_hash,
                         lifecycle_state,predecessor_witness_ref,successor_witness_ref,
                         review_ref,review_hash,canonical_payload_json,actor_ref,
                         decision_evidence_refs_json,occurred_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", 'INSERT INTO source_semantic_witness_reviews\n                        (organization_id,workspace_id,witness_ref,witness_hash,\n                         lifecycle_state,predecessor_witness_ref,successor_witness_ref,\n                         review_ref,review_hash,canonical_payload_json,actor_ref,\n                         decision_evidence_refs_json,occurred_at)\n                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                        (
                            witness.organization_id,
                            witness.workspace_id,
                            witness.witness_id,
                            witness.witness_hash,
                            witness.review.state,
                            witness.review.predecessor_witness_ref,
                            witness.review.successor_witness_ref,
                            witness.review.review_ref,
                            witness.review.review_hash,
                            payload_json,
                            witness.review.actor_ref,
                            expected["decision_evidence_refs_json"],
                            occurred_at,
                        ),
                    )

                existing_manifest = self._conn.execute(
                    hosted.sql_query("SELECT * FROM source_semantic_witness_manifests WHERE manifest_ref=?", 'SELECT * FROM source_semantic_witness_manifests WHERE manifest_ref=%s'),
                    (manifest.manifest_ref,),
                ).fetchone()
                expected_manifest = {
                    "organization_id": manifest.organization_id,
                    "workspace_id": manifest.workspace_id,
                    "content_hash": manifest.content_hash,
                    "lifecycle_state": manifest_state,
                    "predecessor_manifest_ref": manifest.predecessor_manifest_ref,
                    "successor_manifest_ref": manifest.successor_manifest_ref,
                    "canonical_payload_json": manifest_json,
                    "actor_ref": actor_ref,
                    "artifact_ref": artifact_ref,
                }
                if existing_manifest is not None:
                    if any(
                        existing_manifest[key] != value
                        for key, value in expected_manifest.items()
                    ):
                        raise ValueError("manifest ref/hash/payload/scope conflict")
                else:
                    self._conn.execute(
                        hosted.sql_query("""INSERT INTO source_semantic_witness_manifests
                        (organization_id,workspace_id,manifest_ref,content_hash,
                         lifecycle_state,predecessor_manifest_ref,successor_manifest_ref,
                         canonical_payload_json,actor_ref,artifact_ref,occurred_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)""", 'INSERT INTO source_semantic_witness_manifests\n                        (organization_id,workspace_id,manifest_ref,content_hash,\n                         lifecycle_state,predecessor_manifest_ref,successor_manifest_ref,\n                         canonical_payload_json,actor_ref,artifact_ref,occurred_at)\n                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                        (
                            manifest.organization_id,
                            manifest.workspace_id,
                            manifest.manifest_ref,
                            manifest.content_hash,
                            manifest_state,
                            manifest.predecessor_manifest_ref,
                            manifest.successor_manifest_ref,
                            manifest_json,
                            actor_ref,
                            artifact_ref,
                            occurred_at,
                        ),
                    )
                origin_payload = {
                    "artifact_id": manifest.manifest_ref,
                    "artifact_ref": artifact_ref,
                    "artifact_hash": manifest.content_hash,
                    "organization_id": manifest.organization_id,
                    "workspace_id": manifest.workspace_id,
                    "source_identity_ref": manifest.source_binding.source_identity_ref,
                    "source_hash": manifest.source_binding.source_hash,
                    "canonical_source_ir_ref": manifest.source_binding.canonical_source_ir_ref,
                    "canonical_source_ir_hash": manifest.source_binding.canonical_source_ir_hash,
                    "relation_kind": "source_semantic_witness_set",
                    "authority_state": "candidate",
                }
                origin_hash = canonical_content_hash(origin_payload)
                origin_id = f"origin_{origin_hash[7:31]}"
                existing_origin = self._conn.execute(
                    hosted.sql_query("""SELECT * FROM contextgraph_artifact_origins
                    WHERE artifact_id=? AND relation_kind='source_semantic_witness_set'""", "SELECT * FROM contextgraph_artifact_origins\n                    WHERE artifact_id=%s AND relation_kind='source_semantic_witness_set'"),
                    (manifest.manifest_ref,),
                ).fetchone()
                if existing_origin is not None:
                    if (
                        existing_origin["origin_id"] != origin_id
                        or existing_origin["content_hash"] != origin_hash
                        or existing_origin["organization_id"] != manifest.organization_id
                        or existing_origin["workspace_id"] != manifest.workspace_id
                        or existing_origin["source_identity_ref"]
                        != manifest.source_binding.source_identity_ref
                        or existing_origin["authority_state"] != "candidate"
                    ):
                        raise ValueError("witness manifest origin conflict")
                else:
                    self._conn.execute(
                        hosted.sql_query("""INSERT INTO contextgraph_artifact_origins
                        (origin_id,workspace_id,organization_id,source_id,lineage_id,
                         extraction_run_id,candidate_graph_id,artifact_id,source_identity_ref,
                         relation_kind,origin_kind,authority_state,release_lineage_json,
                         release_lineage_hash,content_hash,created_at)
                        VALUES (?,?,?,?,?,NULL,NULL,?,?,'source_semantic_witness_set',
                                'attached','candidate',NULL,NULL,?,?)""", "INSERT INTO contextgraph_artifact_origins\n                        (origin_id,workspace_id,organization_id,source_id,lineage_id,\n                         extraction_run_id,candidate_graph_id,artifact_id,source_identity_ref,\n                         relation_kind,origin_kind,authority_state,release_lineage_json,\n                         release_lineage_hash,content_hash,created_at)\n                        VALUES (%s,%s,%s,%s,%s,NULL,NULL,%s,%s,'source_semantic_witness_set',\n                                'attached','candidate',NULL,NULL,%s,%s)"),
                        (
                            origin_id,
                            manifest.workspace_id,
                            manifest.organization_id,
                            manifest.source_binding.source_identity_ref,
                            manifest.source_binding.source_identity_ref,
                            manifest.manifest_ref,
                            manifest.source_binding.source_identity_ref,
                            origin_hash,
                            occurred_at,
                        ),
                    )
                self._conn.commit()
            except Exception:
                if hosted.in_transaction(self._conn):
                    self._conn.rollback()
                raise
        return {
            "manifest_ref": manifest.manifest_ref,
            "content_hash": manifest.content_hash,
            "witness_count": len(witnesses),
            "artifact_ref": artifact_ref,
        }

    def replay_source_semantic_witness(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        witness_ref: str,
        witness_hash: str,
    ) -> SourceSemanticWitnessV1:
        row = self._conn.execute(
            hosted.sql_query("SELECT * FROM source_semantic_witness_reviews WHERE witness_ref=?", 'SELECT * FROM source_semantic_witness_reviews WHERE witness_ref=%s'),
            (witness_ref,),
        ).fetchone()
        if row is None:
            raise ValueError("witness replay record is missing")
        if row["organization_id"] != organization_id or row["workspace_id"] != workspace_id:
            raise ValueError("witness replay scope mismatch")
        if row["witness_hash"] != witness_hash:
            raise ValueError("witness replay hash mismatch")
        try:
            witness = SourceSemanticWitnessV1.model_validate_json(
                row["canonical_payload_json"]
            )
        except Exception as exc:
            raise ValueError("witness canonical payload tamper/mismatch") from exc
        if (
            witness.witness_id != witness_ref
            or witness.witness_hash != witness_hash
            or witness.review.review_hash != row["review_hash"]
            or witness.review.review_ref != row["review_ref"]
            or witness.review.actor_ref != row["actor_ref"]
            or witness.review.state != row["lifecycle_state"]
            or witness.review.predecessor_witness_ref
            != row["predecessor_witness_ref"]
            or witness.review.successor_witness_ref != row["successor_witness_ref"]
            or _json(
                [
                    item.model_dump(mode="json", by_alias=True)
                    for item in witness.review.decision_evidence_refs
                ]
            )
            != row["decision_evidence_refs_json"]
            or witness.canonical_json() != row["canonical_payload_json"]
        ):
            raise ValueError("witness canonical payload tamper/mismatch")
        return witness

    def replay_source_semantic_witness_manifest(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        manifest_ref: str,
        content_hash: str,
        artifact_ref: str,
    ) -> SourceSemanticWitnessSetManifestV1:
        row = self._conn.execute(
            hosted.sql_query("SELECT * FROM source_semantic_witness_manifests WHERE manifest_ref=?", 'SELECT * FROM source_semantic_witness_manifests WHERE manifest_ref=%s'),
            (manifest_ref,),
        ).fetchone()
        if row is None:
            raise ValueError("witness manifest replay record is missing")
        if row["organization_id"] != organization_id or row["workspace_id"] != workspace_id:
            raise ValueError("witness manifest replay scope mismatch")
        if row["content_hash"] != content_hash:
            raise ValueError("witness manifest replay hash mismatch")
        if row["artifact_ref"] != artifact_ref:
            raise ValueError("witness manifest replay artifact ref mismatch")
        try:
            manifest = SourceSemanticWitnessSetManifestV1.model_validate_json(
                row["canonical_payload_json"]
            )
        except Exception as exc:
            raise ValueError("witness manifest canonical payload tamper/mismatch") from exc
        replayed_witnesses = tuple(
            self.replay_source_semantic_witness(
                organization_id=organization_id,
                workspace_id=workspace_id,
                witness_ref=entry.witness_ref,
                witness_hash=entry.witness_hash,
            )
            for entry in manifest.witnesses
        )
        manifest.verify_witnesses(replayed_witnesses)
        states = {item.review.state for item in replayed_witnesses}
        expected_state = next(iter(states)) if len(states) == 1 else "edited"
        actors = {item.review.actor_ref for item in replayed_witnesses}
        expected_actor = (
            next(iter(actors)) if len(actors) == 1 else "system:mixed-review-set"
        )
        if (
            manifest.manifest_ref != manifest_ref
            or manifest.content_hash != content_hash
            or manifest.canonical_json() != row["canonical_payload_json"]
            or row["lifecycle_state"] != expected_state
            or row["actor_ref"] != expected_actor
            or row["artifact_ref"] != artifact_ref
            or row["predecessor_manifest_ref"] != manifest.predecessor_manifest_ref
            or row["successor_manifest_ref"] != manifest.successor_manifest_ref
        ):
            raise ValueError("witness manifest canonical payload tamper/mismatch")

        origin_payload = {
            "artifact_id": manifest.manifest_ref,
            "artifact_ref": artifact_ref,
            "artifact_hash": manifest.content_hash,
            "organization_id": manifest.organization_id,
            "workspace_id": manifest.workspace_id,
            "source_identity_ref": manifest.source_binding.source_identity_ref,
            "source_hash": manifest.source_binding.source_hash,
            "canonical_source_ir_ref": manifest.source_binding.canonical_source_ir_ref,
            "canonical_source_ir_hash": manifest.source_binding.canonical_source_ir_hash,
            "relation_kind": "source_semantic_witness_set",
            "authority_state": "candidate",
        }
        expected_origin_hash = canonical_content_hash(origin_payload)
        expected_origin_id = f"origin_{expected_origin_hash[7:31]}"
        origins = self._conn.execute(
            hosted.sql_query("""SELECT * FROM contextgraph_artifact_origins
            WHERE artifact_id=? AND relation_kind='source_semantic_witness_set'""", "SELECT * FROM contextgraph_artifact_origins\n            WHERE artifact_id=%s AND relation_kind='source_semantic_witness_set'"),
            (manifest.manifest_ref,),
        ).fetchall()
        if len(origins) != 1:
            raise ValueError("witness manifest origin cardinality mismatch")
        origin = origins[0]
        if any(
            (
                origin["origin_id"] != expected_origin_id,
                origin["workspace_id"] != manifest.workspace_id,
                origin["organization_id"] != manifest.organization_id,
                origin["source_id"] != manifest.source_binding.source_identity_ref,
                origin["lineage_id"] != manifest.source_binding.source_identity_ref,
                origin["extraction_run_id"] is not None,
                origin["candidate_graph_id"] is not None,
                origin["artifact_id"] != manifest.manifest_ref,
                origin["source_identity_ref"] != manifest.source_binding.source_identity_ref,
                origin["relation_kind"] != "source_semantic_witness_set",
                origin["origin_kind"] != "attached",
                origin["authority_state"] != "candidate",
                origin["release_lineage_json"] is not None,
                origin["release_lineage_hash"] is not None,
                origin["content_hash"] != expected_origin_hash,
                origin["created_at"] != row["occurred_at"],
            )
        ):
            raise ValueError("witness manifest origin tamper/mismatch")
        return manifest

    def replay_source_semantic_witness_package_by_ref(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        manifest_ref: str,
        content_hash: str | None = None,
    ) -> tuple[SourceSemanticWitnessSetManifestV1, tuple[SourceSemanticWitnessV1, ...]]:
        """Resolve one exact persisted witness package without caller artifact paths."""

        row = self._conn.execute(
            hosted.sql_query("SELECT content_hash,artifact_ref FROM source_semantic_witness_manifests "
            "WHERE manifest_ref=?", 'SELECT content_hash,artifact_ref FROM source_semantic_witness_manifests WHERE manifest_ref=%s'),
            (manifest_ref,),
        ).fetchone()
        if row is None:
            raise ValueError("witness manifest replay record is missing")
        if content_hash is not None and row["content_hash"] != content_hash:
            raise ValueError("witness manifest replay hash mismatch")
        manifest = self.replay_source_semantic_witness_manifest(
            organization_id=organization_id,
            workspace_id=workspace_id,
            manifest_ref=manifest_ref,
            content_hash=str(row["content_hash"]),
            artifact_ref=str(row["artifact_ref"]),
        )
        witnesses = tuple(
            self.replay_source_semantic_witness(
                organization_id=organization_id,
                workspace_id=workspace_id,
                witness_ref=item.witness_ref,
                witness_hash=item.witness_hash,
            )
            for item in manifest.witnesses
        )
        return manifest, witnesses

    def _now_dt(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("materialization clock must return timezone-aware UTC")
        return value.astimezone(timezone.utc)

    def _now(self) -> str:
        return self._now_dt().isoformat()

    def _lease_expires_at(self) -> str:
        return (self._now_dt() + timedelta(seconds=self._lease_seconds)).isoformat()

    def _lease_is_live(self, value: str | None) -> bool:
        if not value:
            return False
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return False
        return parsed.astimezone(timezone.utc) > self._now_dt()

    def get_graph_release(self, graph_version: str, *, organization_id: str | None = None,
                          workspace_id: str | None = None) -> dict[str, Any]:
        payload, status = self.get_graph_release_projection(
            graph_version,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        if status != "materialized":
            raise ValueError(f"graph release not materialized: {graph_version}")
        return payload

    def get_graph_release_projection(
        self,
        graph_version: str,
        *,
        organization_id: str | None = None,
        workspace_id: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Read an immutable release for visualization, including pending releases."""
        row = self._conn.execute(
            hosted.sql_query("SELECT release_json, content_hash, organization_id, workspace_id, materialization_status, lineage_contract FROM tender_graph_releases WHERE graph_version = ? AND retirement_state='active'", "SELECT release_json, content_hash, organization_id, workspace_id, materialization_status, lineage_contract FROM tender_graph_releases WHERE graph_version = %s AND retirement_state='active'"),
            (graph_version,),
        ).fetchone()
        if row is None:
            raise ValueError(f"graph release not found: {graph_version}")
        lifecycle = self._conn.execute(
            hosted.sql_query("SELECT status FROM tender_graph_release_materialization_events WHERE graph_version=? ORDER BY event_id DESC LIMIT 1", 'SELECT status FROM tender_graph_release_materialization_events WHERE graph_version=%s ORDER BY event_id DESC LIMIT 1'),
            (graph_version,),
        ).fetchone()
        status = lifecycle["status"] if lifecycle else row["materialization_status"]
        if organization_id is not None and row["organization_id"] != organization_id:
            raise ValueError("graph release does not belong to candidate organization")
        if workspace_id is not None and row["workspace_id"] != workspace_id:
            raise ValueError("graph release does not belong to candidate workspace")
        payload = json.loads(row["release_json"])
        self._validate_stored_release_dto(
            payload,
            stored_hash=str(row["content_hash"]),
            lineage_contract=str(row["lineage_contract"] or "legacy"),
            expected_graph_version=graph_version,
        )
        return payload, str(status)

    def read_materialized_release_record(
        self,
        graph_version: str,
        *,
        organization_id: str,
        workspace_id: str,
    ) -> dict[str, Any]:
        """Read and validate the store-owned release/lifecycle record."""
        row = self._conn.execute(
            hosted.sql_query("SELECT release_json,content_hash,organization_id,workspace_id,"
            "materialization_status,retirement_state,lineage_contract "
            "FROM tender_graph_releases WHERE graph_version=?", 'SELECT release_json,content_hash,organization_id,workspace_id,materialization_status,retirement_state,lineage_contract FROM tender_graph_releases WHERE graph_version=%s'),
            (graph_version,),
        ).fetchone()
        if row is None:
            raise ValueError(f"graph release not found: {graph_version}")
        if (
            row["organization_id"] != organization_id
            or row["workspace_id"] != workspace_id
        ):
            raise ValueError("graph release tenant scope mismatch")
        lifecycle = self._conn.execute(
            hosted.sql_query("SELECT status FROM tender_graph_release_materialization_events "
            "WHERE graph_version=? ORDER BY event_id DESC LIMIT 1", 'SELECT status FROM tender_graph_release_materialization_events WHERE graph_version=%s ORDER BY event_id DESC LIMIT 1'),
            (graph_version,),
        ).fetchone()
        payload = json.loads(row["release_json"])
        self._validate_stored_release_dto(
            payload,
            stored_hash=str(row["content_hash"]),
            lineage_contract=str(row["lineage_contract"] or "legacy"),
            expected_graph_version=graph_version,
        )
        return {
            "release": payload,
            "content_hash": str(row["content_hash"]),
            "organization_id": str(row["organization_id"]),
            "workspace_id": str(row["workspace_id"]),
            "materialization_status": str(row["materialization_status"]),
            "lifecycle_status": str(
                lifecycle["status"] if lifecycle else row["materialization_status"]
            ),
            "retirement_state": str(row["retirement_state"]),
            "lineage_contract": str(row["lineage_contract"] or "legacy"),
        }

    def read_authoritative_release_lineage(
        self,
        graph_version: str,
        *,
        organization_id: str,
        workspace_id: str,
    ) -> dict[str, Any]:
        """Read the exact authoritative origin lineage for one release."""
        record = self.read_materialized_release_record(
            graph_version,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        if record["lineage_contract"] != "exact_v4_v3":
            raise ValueError("graph release does not have v2 exact lineage")
        origin_id = verify_exact_release_origin_reservation(
            self._conn,
            graph_version=graph_version,
            authority_state="authoritative",
        )
        row = self._conn.execute(
            hosted.sql_query("SELECT release_lineage_json FROM contextgraph_artifact_origins "
            "WHERE origin_id=?", 'SELECT release_lineage_json FROM contextgraph_artifact_origins WHERE origin_id=%s'),
            (origin_id,),
        ).fetchone()
        if row is None:
            raise ValueError("authoritative graph release lineage not found")
        return json.loads(row["release_lineage_json"])

    def list_materialized_release_binding_refs(
        self,
        graph_version: str,
        *,
        binding_kind: str,
        organization_id: str,
        workspace_id: str,
    ) -> tuple[str, ...]:
        """Enumerate binding identities without exposing storage shape."""
        self.read_materialized_release_record(
            graph_version,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        table, identity_column = self._unverified_v2_artifact_table(binding_kind)
        if binding_kind not in {
            "accepted_relation_binding", "accepted_pattern_binding"
        }:
            raise ValueError("unsupported accepted release binding kind")
        rows = self._conn.execute(
            hosted.sql_query(f"SELECT {identity_column} FROM {table} "
            "WHERE accepted_graph_release_ref=? AND organization_id=? "
            "AND workspace_id=? ORDER BY ", f'SELECT {identity_column} FROM {table} WHERE accepted_graph_release_ref=%s AND organization_id=%s AND workspace_id=%s ORDER BY ') + identity_column,
            (graph_version, organization_id, workspace_id),
        ).fetchall()
        return tuple(str(row[identity_column]) for row in rows)

    def read_materialized_v2_artifact(
        self,
        kind: str,
        identity: str,
        *,
        organization_id: str,
        workspace_id: str,
    ) -> dict[str, Any]:
        """Read one canonical tenant-scoped v2 artifact through public API."""
        return self._get_unverified_v2_artifact(
            kind,
            identity,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )

    def list_reference_graph_versions(
        self,
        *,
        organization_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = """SELECT graph_version,parent_graph_version,candidate_graph_id,
        graph_delta_id,release_json,content_hash,organization_id,workspace_id,
        materialization_status,lineage_contract
        FROM tender_graph_releases WHERE retirement_state='active'"""
        args: list[Any] = []
        if organization_id is not None:
            sql += hosted.sql_query(" AND organization_id=?", ' AND organization_id=%s')
            args.append(organization_id)
        if workspace_id is not None:
            sql += hosted.sql_query(" AND workspace_id=?", ' AND workspace_id=%s')
            args.append(workspace_id)
        sql += " ORDER BY graph_version"
        versions: list[dict[str, Any]] = []
        for row in self._conn.execute(sql, tuple(args)).fetchall():
            item = dict(row)
            release_payload = json.loads(item.pop("release_json"))
            lineage_contract = str(item.pop("lineage_contract") or "legacy")
            self._validate_stored_release_dto(
                release_payload,
                stored_hash=str(item["content_hash"]),
                lineage_contract=lineage_contract,
                expected_graph_version=str(item["graph_version"]),
            )
            item["node_count"] = len(release_payload.get("nodes") or [])
            item["edge_count"] = len(release_payload.get("edges") or [])
            item["rule_count"] = len(release_payload.get("rules") or [])
            semantic_member_count = 0
            if lineage_contract == "exact_v4_v3":
                item["relation_count"] = len(
                    release_payload.get("relation_membership") or []
                )
                item["pattern_count"] = len(
                    release_payload.get("pattern_membership") or []
                )
                semantic_member_count = item["relation_count"] + item["pattern_count"]
            referenced = self._conn.execute(
                hosted.sql_query("""SELECT 1 FROM knowledge_builder_assertion_grounding_bindings
                WHERE graph_version=? UNION ALL
                SELECT 1 FROM contextgraph_reference_concept_grounding_bindings
                WHERE graph_version=? LIMIT 1""", 'SELECT 1 FROM knowledge_builder_assertion_grounding_bindings\n                WHERE graph_version=%s UNION ALL\n                SELECT 1 FROM contextgraph_reference_concept_grounding_bindings\n                WHERE graph_version=%s LIMIT 1'),
                (item["graph_version"], item["graph_version"]),
            ).fetchone()
            item["delete_eligible"] = (
                item["node_count"] == 0
                and semantic_member_count == 0
                and not bool(referenced)
            )
            item["protection_reason"] = (
                "not_empty"
                if item["node_count"] > 0 or semantic_member_count > 0
                else "referenced" if referenced else None
            )
            item["schema_version"] = (
                "contextgraph_reference_graph_version_read_model.v2"
                if lineage_contract == "exact_v4_v3"
                else "contextgraph_reference_graph_version_read_model.v1"
            )
            if lineage_contract == "exact_v4_v3":
                item["release_contract"] = "accepted_graph_release.v2"
            item["accepted_graph_release_id"] = item["graph_version"]
            item["release_hash"] = item.pop("content_hash")
            item["authority_source"] = "AcceptedGraphRelease"
            versions.append(item)
        return versions

    def batch_delete_empty_graph_releases(
        self,
        graph_versions: list[str],
        *,
        organization_id: str,
        workspace_id: str | None,
    ) -> dict[str, Any]:
        """Delete only ungrounded releases with no topology or semantic members."""
        deleted: list[dict[str, str]] = []
        skipped: list[dict[str, str]] = []
        seen: set[str] = set()
        with self._lock:
            hosted.begin_write(self._conn)
            try:
                for raw_version in graph_versions:
                    graph_version = str(raw_version).strip()
                    if not graph_version or graph_version in seen:
                        continue
                    seen.add(graph_version)
                    row = self._conn.execute(
                        hosted.sql_query("""SELECT release_json,content_hash,lineage_contract
                        FROM tender_graph_releases
                        WHERE graph_version=? AND organization_id=? AND workspace_id IS ?""", 'SELECT release_json,content_hash,lineage_contract\n                        FROM tender_graph_releases\n                        WHERE graph_version=%s AND organization_id=%s AND workspace_id IS NOT DISTINCT FROM %s'),
                        (graph_version, organization_id, workspace_id),
                    ).fetchone()
                    if row is None:
                        skipped.append({"graph_version": graph_version, "code": "not_found"})
                        continue
                    payload = json.loads(row["release_json"])
                    lineage_contract = str(row["lineage_contract"] or "legacy")
                    self._validate_stored_release_dto(
                        payload,
                        stored_hash=str(row["content_hash"]),
                        lineage_contract=lineage_contract,
                        expected_graph_version=graph_version,
                    )
                    semantic_members = (
                        payload.get("relation_membership")
                        or payload.get("pattern_membership")
                        if lineage_contract == "exact_v4_v3"
                        else False
                    )
                    if payload.get("nodes") or semantic_members:
                        skipped.append({"graph_version": graph_version, "code": "not_empty"})
                        continue
                    referenced = None
                    for table in (
                        "knowledge_builder_assertion_grounding_bindings",
                        "contextgraph_reference_concept_grounding_bindings",
                    ):
                        referenced = self._conn.execute(
                            hosted.sql_query(f"SELECT 1 FROM {table} WHERE graph_version=? LIMIT 1", f'SELECT 1 FROM {table} WHERE graph_version=%s LIMIT 1'),
                            (graph_version,),
                        ).fetchone()
                        if referenced:
                            break
                    if referenced:
                        skipped.append({"graph_version": graph_version, "code": "referenced"})
                        continue
                    self._conn.execute(
                        hosted.sql_query("DELETE FROM tender_graph_release_materialization_events WHERE graph_version=?", 'DELETE FROM tender_graph_release_materialization_events WHERE graph_version=%s'),
                        (graph_version,),
                    )
                    self._conn.execute(
                        hosted.sql_query("DELETE FROM tender_graph_releases WHERE graph_version=?", 'DELETE FROM tender_graph_releases WHERE graph_version=%s'),
                        (graph_version,),
                    )
                    deleted.append({"graph_version": graph_version})
                self._conn.commit()
            except Exception:
                if hosted.in_transaction(self._conn):
                    self._conn.rollback()
                raise
        return {
            "ok": True,
            "requested_count": len(seen),
            "deleted": deleted,
            "skipped": skipped,
            "failed": [],
        }

    def compare_graph_versions(
        self,
        old_version: str,
        new_version: str,
        *,
        organization_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        old = self.get_graph_release(
            old_version,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        new = self.get_graph_release(
            new_version,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
        result: dict[str, Any] = {
            "old_graph_version": old_version,
            "new_graph_version": new_version,
            "old_release_hash": old["release_hash"],
            "new_release_hash": new["release_hash"],
            "causal_graph_delta_id": new["graph_delta_id"],
        }
        for plural, identity in (("nodes", "node_id"), ("edges", "edge_id"), ("rules", "rule_id")):
            old_items = {str(item[identity]): item for item in old[plural]}
            new_items = {str(item[identity]): item for item in new[plural]}
            result[plural] = {
                "before": old[plural],
                "after": new[plural],
                "added": sorted(set(new_items) - set(old_items)),
                "removed": sorted(set(old_items) - set(new_items)),
                "changed": sorted(
                    key
                    for key in set(old_items) & set(new_items)
                    if old_items[key] != new_items[key]
                ),
            }
        for field, identity in (
            ("relation_membership", "relation_instance_ref"),
            ("pattern_membership", "pattern_ref"),
        ):
            if any(
                release.get("schema_version") != "accepted_graph_release.v2"
                for release in (old, new)
            ):
                result[field] = {
                    "status": "unavailable",
                    "reason": "both releases must supply canonical v2 membership",
                }
                continue
            before = {item[identity]: item for item in old[field]}
            after = {item[identity]: item for item in new[field]}
            result[field] = {
                "status": "available",
                "before": old[field],
                "after": new[field],
                "added": sorted(set(after) - set(before)),
                "removed": sorted(set(before) - set(after)),
                "changed": sorted(
                    ref for ref in set(before) & set(after)
                    if before[ref] != after[ref]
                ),
            }
        return result


def _apply_operations(
    *,
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
    rules: dict[str, dict[str, Any]],
    delta: GraphDelta,
) -> None:
    for operation in delta.operations:
        action = operation.action.value
        after = operation.after or {}
        if action == "node_add":
            nodes[str(after["node_id"])] = dict(after)
        elif action in {"node_edit", "alias_edit", "evidence_edit"}:
            _replace(nodes, operation.target_refs[0], after)
        elif action == "node_merge":
            for target in operation.target_refs:
                nodes.pop(target, None)
            nodes[str(after["node_id"])] = dict(after)
        elif action == "node_split":
            nodes.pop(operation.target_refs[0], None)
            for item in after["nodes"]:
                nodes[str(item["node_id"])] = dict(item)
        elif action == "edge_add":
            edges[str(after["edge_id"])] = dict(after)
        elif action == "edge_edit":
            _replace(edges, operation.target_refs[0], after)
        elif action == "edge_reverse":
            current = dict(edges[operation.target_refs[0]])
            current["source_node_id"], current["target_node_id"] = (
                current["target_node_id"],
                current["source_node_id"],
            )
            current.update(after)
            edges[operation.target_refs[0]] = current
        elif action == "edge_delete":
            edges.pop(operation.target_refs[0], None)
        elif action in {"rule_edit", "condition_edit", "exception_edit"}:
            _replace(rules, operation.target_refs[0], after)
        else:
            raise ValueError(f"unsupported GraphDelta action: {action}")
    known_nodes = set(nodes)
    dangling = [
        edge_id
        for edge_id, edge in edges.items()
        if edge["source_node_id"] not in known_nodes or edge["target_node_id"] not in known_nodes
    ]
    if dangling:
        raise ValueError(f"grounded graph has dangling edges: {dangling}")


_PHASE1_REVIEW_ACTIONS = {
    GraphDeltaAction.NODE_EDIT,
    GraphDeltaAction.EDGE_EDIT,
    GraphDeltaAction.EVIDENCE_EDIT,
    GraphDeltaAction.NODE_MERGE,
    GraphDeltaAction.ALIAS_EDIT,
    GraphDeltaAction.NODE_SPLIT,
}


def _validate_phase1_review_delta(
    delta: GraphDelta,
    graph: SensitiveTermCandidateGraph,
) -> None:
    nodes = {item.node_id: item.model_dump(mode="json") for item in graph.nodes}
    edges = {item.edge_id: item.model_dump(mode="json") for item in graph.edges}
    rules = {item.rule_id: item.model_dump(mode="json") for item in graph.rules}
    for operation in delta.operations:
        if operation.action not in _PHASE1_REVIEW_ACTIONS:
            raise ValueError(
                f"GraphDelta action is reserved for Phase 2: {operation.action.value}"
            )
        action = operation.action
        targets = operation.target_refs
        if len(set(targets)) != len(targets):
            raise ValueError("GraphDelta target refs must be unique")
        if action is GraphDeltaAction.NODE_MERGE:
            if len(targets) < 2 or any(target not in nodes for target in targets):
                raise ValueError("NODE_MERGE requires at least two existing node targets")
            expected_before = {"nodes": [nodes[target] for target in targets]}
            if operation.before != expected_before:
                raise ValueError("GraphDelta before state does not match candidate graph")
            merged = CandidateNode.model_validate(operation.after)
            if merged.node_id in nodes and merged.node_id not in targets:
                raise ValueError("NODE_MERGE result collides with an unrelated node")
            continue
        if action is GraphDeltaAction.NODE_SPLIT:
            if len(targets) != 1 or targets[0] not in nodes:
                raise ValueError("NODE_SPLIT requires one existing node target")
            if operation.before != nodes[targets[0]]:
                raise ValueError("GraphDelta before state does not match candidate graph")
            split_items = (operation.after or {}).get("nodes")
            if not isinstance(split_items, list) or len(split_items) < 2:
                raise ValueError("NODE_SPLIT requires at least two result nodes")
            split_nodes = [CandidateNode.model_validate(item) for item in split_items]
            split_ids = [item.node_id for item in split_nodes]
            if len(set(split_ids)) != len(split_ids):
                raise ValueError("NODE_SPLIT result node IDs must be unique")
            if any(node_id in nodes and node_id != targets[0] for node_id in split_ids):
                raise ValueError("NODE_SPLIT result collides with an unrelated node")
            continue

        collections = (
            [("node", nodes)] if action in {GraphDeltaAction.NODE_EDIT, GraphDeltaAction.ALIAS_EDIT}
            else [("edge", edges)] if action is GraphDeltaAction.EDGE_EDIT
            else [("node", nodes), ("edge", edges), ("rule", rules)]
        )
        if len(targets) != 1:
            raise ValueError(f"{action.value.upper()} requires exactly one target")
        matches = [(kind, items[targets[0]]) for kind, items in collections if targets[0] in items]
        if len(matches) != 1:
            raise ValueError(f"GraphDelta target not found for {action.value}: {targets[0]}")
        kind, current = matches[0]
        if operation.before != current:
            raise ValueError("GraphDelta before state does not match candidate graph")
        after = operation.after or {}
        model = {"node": CandidateNode, "edge": CandidateEdge, "rule": CandidateRule}[kind]
        validated_after = model.model_validate(after).model_dump(mode="json")
        identity_field = {"node": "node_id", "edge": "edge_id", "rule": "rule_id"}[kind]
        if validated_after[identity_field] != targets[0]:
            raise ValueError(f"{action.value.upper()} cannot change object identity")
        if action is GraphDeltaAction.EDGE_EDIT:
            if any(
                validated_after[field] != current[field]
                for field in ("source_node_id", "target_node_id")
            ):
                raise ValueError("EDGE_EDIT cannot replace or reverse endpoints in Phase 1")
        if action is GraphDeltaAction.EVIDENCE_EDIT:
            if not validated_after.get("source_anchors"):
                raise ValueError("EVIDENCE_EDIT requires non-empty exact source anchors")
            without_anchors = lambda value: {  # noqa: E731 - local comparison projection
                key: item for key, item in value.items() if key != "source_anchors"
            }
            if without_anchors(validated_after) != without_anchors(current):
                raise ValueError("EVIDENCE_EDIT may change only source anchors")


def _replace(items: dict[str, dict[str, Any]], target: str, after: dict[str, Any]) -> None:
    if target not in items:
        raise ValueError(f"GraphDelta target not found: {target}")
    updated = dict(items[target])
    updated.update(after)
    items[target] = updated


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
