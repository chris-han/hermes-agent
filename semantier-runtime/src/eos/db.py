import json
import os
import sqlite3
from storage import hosted_postgres as hosted
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from runtime_paths import sqlite_db_path

DB_PATH = sqlite_db_path("eos.db", ".semantier/eos.db")
_INIT_DB_LOCK = threading.RLock()
EOS_CORE_SCHEMA_VERSION = 2
RULE_GOVERNANCE_SCHEMA_VERSION = 3
EOS_CORE_SCHEMA_SENTINELS = frozenset(
    {
        "events",
        "ledger_projections",
        "t6_materialization_candidates",
        "policy_candidates",
        "tender_detection_runs",
        "governed_rule_versions",
    }
)
LEGACY_KNOWLEDGE_BUILDER_GRAPH_TABLES = (
    "knowledge_builder_source_anchors",
    "knowledge_builder_notes",
    "knowledge_builder_candidate_nodes",
    "knowledge_builder_candidate_relations",
    "knowledge_builder_clusters",
)
LEGACY_KNOWLEDGE_BUILDER_GRAPH_INDEXES = (
    "idx_kb_builder_nodes_run",
    "idx_kb_builder_relations_run",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrate_policy_candidate_categories(conn: sqlite3.Connection) -> None:
    """Rebuild the legacy CHECK constraint without changing candidate rows."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'policy_candidates'"
    ).fetchone()
    sql = str(row[0] or "") if row else ""
    if "organization_learning_contribution" in sql:
        return
    if "policy_bundle" not in sql:
        return
    new_sql = sql.replace('CREATE TABLE "policy_candidates"', "CREATE TABLE policy_candidates_radar_new", 1)
    new_sql = new_sql.replace("CREATE TABLE policy_candidates", "CREATE TABLE policy_candidates_radar_new", 1) if new_sql == sql else new_sql
    added_categories = "'coa_version', 'organization_learning_contribution'"
    if "learned_procedure" not in sql:
        added_categories += ", 'learned_procedure'"
    if "investment_radar_universe" not in sql:
        added_categories += ", 'investment_radar_universe', 'investment_radar_scoring_profile'"
    new_sql = new_sql.replace("'coa_version'", added_categories, 1)
    conn.execute("ALTER TABLE policy_candidates RENAME TO policy_candidates_radar_legacy")
    conn.execute(new_sql)
    old_columns = {row[1] for row in conn.execute("PRAGMA table_info(policy_candidates_radar_legacy)")}
    new_columns = {row[1] for row in conn.execute("PRAGMA table_info(policy_candidates_radar_new)")}
    columns = sorted(old_columns & new_columns)
    quoted = ", ".join('"' + column.replace('"', '""') + '"' for column in columns)
    conn.execute(f"INSERT INTO policy_candidates_radar_new ({quoted}) SELECT {quoted} FROM policy_candidates_radar_legacy")
    conn.execute("DROP TABLE policy_candidates_radar_legacy")
    conn.execute("ALTER TABLE policy_candidates_radar_new RENAME TO policy_candidates")
    for name, expression in {
        "idx_candidates_name": "name", "idx_candidates_status": "status",
        "idx_candidates_category": "category", "idx_candidates_hash": "content_hash",
    }.items():
        conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON policy_candidates({expression})")


def _migrate_replay_envelope_binding_types(conn: sqlite3.Connection) -> None:
    """Rebuild the legacy replay envelope CHECK constraint in place."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'replay_envelopes'"
    ).fetchone()
    sql = str(row[0] or "") if row else ""
    if not sql or "investment_radar_candidate.v1" in sql:
        return

    child_tables = (
        "sensitive_labeling_detection_bindings",
        "sensitive_labeling_report_bindings",
        "investment_radar_candidate_bindings",
    )
    child_sql: dict[str, str] = {}
    for table_name in child_tables:
        child_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if child_row and child_row[0]:
            child_sql[table_name] = str(child_row[0])

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("DROP INDEX IF EXISTS idx_replay_envelopes_subject")
        conn.execute("DROP INDEX IF EXISTS idx_replay_envelopes_scope")
        for table_name in child_sql:
            conn.execute(f"ALTER TABLE {table_name} RENAME TO {table_name}_legacy")
        conn.execute("ALTER TABLE replay_envelopes RENAME TO replay_envelopes_legacy")
        conn.execute(
            """
            CREATE TABLE replay_envelopes (
              replay_binding_id TEXT PRIMARY KEY,
              binding_type TEXT NOT NULL CHECK(binding_type IN (
                'projection_execution.v1',
                'sensitive_labeling_detection.v1',
                'sensitive_labeling_report.v1',
                'investment_radar_candidate.v1'
              )),
              binding_schema_version TEXT NOT NULL,
              subject_type TEXT NOT NULL,
              subject_id TEXT NOT NULL,
              specific_binding_content_hash TEXT NOT NULL,
              envelope_hash TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO replay_envelopes SELECT * FROM replay_envelopes_legacy"
        )
        for table_name, original_sql in child_sql.items():
            conn.execute(
                original_sql.replace(
                    f"CREATE TABLE {table_name}",
                    f"CREATE TABLE {table_name}",
                    1,
                )
            )
            columns = [
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table_name}_legacy)")
            ]
            quoted = ", ".join(f'"{column}"' for column in columns)
            conn.execute(
                f"INSERT INTO {table_name} ({quoted}) SELECT {quoted} FROM {table_name}_legacy"
            )
            conn.execute(f"DROP TABLE {table_name}_legacy")
        conn.execute("DROP TABLE replay_envelopes_legacy")
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _migrate_source_identity_refs_to_tenant_scope(
    conn: sqlite3.Connection,
) -> None:
    """Rebuild the legacy globally keyed SourceIdentity history in place."""

    table_info = conn.execute(
        "PRAGMA table_info(knowledge_builder_source_identities)"
    ).fetchall()
    if not table_info:
        return
    primary_key = [
        row[1]
        for row in sorted(table_info, key=lambda row: row[5] or 99)
        if row[5]
    ]
    tenant_key = ["organization_id", "workspace_id", "source_identity_ref"]
    if primary_key == tenant_key:
        return
    if primary_key != ["source_identity_ref"]:
        raise RuntimeError(
            "unsupported knowledge_builder_source_identities primary key"
        )

    foreign_keys_enabled = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        hosted.begin_write(conn)
        conn.execute("DROP INDEX IF EXISTS idx_kb_source_identities_source")
        conn.execute("DROP INDEX IF EXISTS idx_kb_source_identities_lineage_order")
        conn.execute(
            "ALTER TABLE knowledge_builder_source_identities "
            "RENAME TO knowledge_builder_source_identities_global_ref_legacy"
        )
        conn.execute(
            """
            CREATE TABLE knowledge_builder_source_identities (
              source_identity_ref TEXT NOT NULL,
              source_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              workspace_id TEXT NOT NULL,
              lineage_id TEXT NOT NULL,
              source_hash TEXT NOT NULL,
              source_version TEXT NOT NULL,
              supersedes_source_identity_ref TEXT,
              artifact_ref TEXT NOT NULL,
              artifact_hash TEXT NOT NULL,
              media_type TEXT NOT NULL,
              original_name TEXT NOT NULL,
              created_at TEXT NOT NULL,
              schema_version TEXT NOT NULL,
              source_identity_json TEXT NOT NULL,
              source_identity_hash TEXT NOT NULL,
              registration_evidence_json TEXT NOT NULL,
              registration_evidence_hash TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              PRIMARY KEY (organization_id, workspace_id, source_identity_ref),
              FOREIGN KEY(source_id) REFERENCES knowledge_builder_sources(source_id),
              FOREIGN KEY(
                organization_id, workspace_id, supersedes_source_identity_ref
              ) REFERENCES knowledge_builder_source_identities(
                organization_id, workspace_id, source_identity_ref
              ),
              UNIQUE(source_id, source_version),
              UNIQUE(organization_id, workspace_id, lineage_id, source_version)
            )
            """
        )
        columns = [row[1] for row in table_info]
        quoted = ", ".join(f'"{column}"' for column in columns)
        conn.execute(
            f"INSERT INTO knowledge_builder_source_identities ({quoted}) "
            f"SELECT {quoted} FROM knowledge_builder_source_identities_global_ref_legacy"
        )
        conn.execute("DROP TABLE knowledge_builder_source_identities_global_ref_legacy")
        conn.execute(
            "CREATE INDEX idx_kb_source_identities_source "
            "ON knowledge_builder_source_identities(source_id)"
        )
        conn.execute(
            """CREATE INDEX idx_kb_source_identities_lineage_order
            ON knowledge_builder_source_identities(
              organization_id, workspace_id, lineage_id, created_at,
              source_identity_ref
            )"""
        )
        violations = conn.execute(
            "PRAGMA foreign_key_check(knowledge_builder_source_identities)"
        ).fetchall()
        if violations:
            raise RuntimeError(
                "knowledge_builder_source_identities migration has foreign key violations"
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute(f"PRAGMA foreign_keys = {int(foreign_keys_enabled)}")


def get_conn(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Return a connection to the EOS database.

    The path can be overridden for testing via the SEMANTIER_EOS_DB_PATH
    environment variable.
    """
    path = (
        Path(db_path)
        if db_path is not None
        else Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(DB_PATH)))
    )
    if hosted.enabled():
        return hosted.connect_authority('eos', path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _drop_legacy_knowledge_builder_graph_schema(conn: sqlite3.Connection) -> None:
    schema_rows = {
        row["name"]: row["sql"] or ""
        for row in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'index')"
        ).fetchall()
    }
    legacy_graph_present = (
        set(LEGACY_KNOWLEDGE_BUILDER_GRAPH_TABLES)
        | set(LEGACY_KNOWLEDGE_BUILDER_GRAPH_INDEXES)
    ).intersection(schema_rows)
    legacy_semantic_relation_present = any(
        column in schema_rows.get("semantic_relation_candidates", "")
        for column in ("source_relation_id", "from_node_id", "to_node_id")
    )
    legacy_curation_event_present = (
        "knowledge_builder_curation_events" in schema_rows
        and "'cluster'" in schema_rows["knowledge_builder_curation_events"]
    )
    if not (
        legacy_graph_present
        or legacy_semantic_relation_present
        or legacy_curation_event_present
    ):
        return
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        for index_name in LEGACY_KNOWLEDGE_BUILDER_GRAPH_INDEXES:
            conn.execute(f"DROP INDEX IF EXISTS {index_name}")
        conn.execute("DROP TABLE IF EXISTS semantic_relation_candidates")
        conn.execute("DROP TABLE IF EXISTS knowledge_builder_curation_events")
        for table_name in (
            "knowledge_builder_candidate_relations",
            "knowledge_builder_candidate_nodes",
            "knowledge_builder_clusters",
            "knowledge_builder_notes",
            "knowledge_builder_source_anchors",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _add_missing_columns(
    conn: sqlite3.Connection, table_name: str, column_defs: dict[str, str]
) -> None:
    existing_columns = {
        row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if not existing_columns:
        return
    for column_name, column_type in column_defs.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def _upgrade_legacy_tables_before_schema_indexes(conn: sqlite3.Connection) -> None:
    _add_missing_columns(
        conn,
        "policy_rule_candidates",
        {
            "organization_id": "TEXT",
            "proposed_workspace_scope": "TEXT",
            "proposed_workspace_id": "TEXT",
            "authority_domain": "TEXT",
            "semantic_tier": "TEXT",
            "jurisdiction": "TEXT",
            "candidate_idempotency_key": "TEXT",
        },
    )
    _add_missing_columns(
        conn,
        "governed_rule_versions",
        {
            "organization_id": "TEXT",
            "authority_domain": "TEXT",
            "semantic_tier": "TEXT",
            "jurisdiction": "TEXT",
        },
    )
    _add_missing_columns(
        conn,
        "tender_detection_runs",
        {
            "document_text": "TEXT NOT NULL DEFAULT ''",
            "source_document_hash": "TEXT NOT NULL DEFAULT ''",
            "parent_run_id": "TEXT",
            "root_run_id": "TEXT",
            "idempotency_key": "TEXT",
            "request_hash": "TEXT",
            "run_status": "TEXT NOT NULL DEFAULT 'FINALIZED'",
            "finding_set_hash": "TEXT",
            "runtime_conflict_set_hash": "TEXT",
            "replay_status": "TEXT NOT NULL DEFAULT 'CREATED'",
        },
    )
    _add_missing_columns(
        conn,
        "knowledge_source",
        {
            "semantic_purpose": "TEXT",
        },
    )


def ensure_graph_ingestion_schema(conn: sqlite3.Connection) -> None:
    if hosted.is_postgres(conn):
        hosted.require_tables(conn, 'graph_ingestion_runs', 'graph_source_index_manifests', 'graph_index_freshness')
        return
    graph_ingestion_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(graph_ingestion_runs)").fetchall()
    }
    graph_ingestion_column_defs = {
        "workspace_id": "TEXT",
        "trigger_type": "TEXT NOT NULL DEFAULT ''",
        "requested_by": "TEXT NOT NULL DEFAULT ''",
        "idempotency_key": "TEXT NOT NULL DEFAULT ''",
        "requested_fingerprint": "TEXT NOT NULL DEFAULT ''",
        "observed_source_hash": "TEXT",
        "prior_successful_run_ref": "TEXT",
        "lease_owner": "TEXT",
        "lease_expires_at": "TEXT",
        "attempt_count": "INTEGER NOT NULL DEFAULT 0",
        "max_attempts": "INTEGER NOT NULL DEFAULT 3",
        "next_retry_at": "TEXT",
        "error_code": "TEXT",
        "error_detail_ref": "TEXT",
        "resulting_graph_snapshot_ref": "TEXT",
        "resulting_graph_snapshot_hash": "TEXT",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
        "completed_at": "TEXT",
    }
    for column_name, column_type in graph_ingestion_column_defs.items():
        if column_name not in graph_ingestion_columns:
            conn.execute(
                f"ALTER TABLE graph_ingestion_runs ADD COLUMN {column_name} {column_type}"
            )
    conn.execute(
        """
        UPDATE graph_ingestion_runs
        SET requested_fingerprint = 'legacy:' || ingestion_run_id
        WHERE requested_fingerprint = ''
        """
    )
    conn.execute(
        """
        UPDATE graph_ingestion_runs
        SET idempotency_key = 'legacy:' || ingestion_run_id
        WHERE idempotency_key = ''
        """
    )
    conn.execute(
        """
        UPDATE graph_ingestion_runs
        SET updated_at = COALESCE(NULLIF(updated_at, ''), created_at, '')
        WHERE updated_at = ''
        """
    )

    graph_manifest_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(graph_source_index_manifests)").fetchall()
    }
    graph_manifest_column_defs = {
        "workspace_id": "TEXT",
        "index_fingerprint": "TEXT NOT NULL DEFAULT ''",
        "source_hash": "TEXT NOT NULL DEFAULT ''",
        "normalized_structure_hash": "TEXT NOT NULL DEFAULT ''",
        "extractor_version": "TEXT NOT NULL DEFAULT ''",
        "structure_schema_version": "TEXT NOT NULL DEFAULT ''",
        "ontology_compatibility_version": "TEXT NOT NULL DEFAULT ''",
        "graph_projection_version": "TEXT NOT NULL DEFAULT ''",
        "last_successful_run_ref": "TEXT NOT NULL DEFAULT ''",
        "graph_snapshot_ref": "TEXT NOT NULL DEFAULT ''",
        "graph_snapshot_hash": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    }
    for column_name, column_type in graph_manifest_column_defs.items():
        if column_name not in graph_manifest_columns:
            conn.execute(
                f"ALTER TABLE graph_source_index_manifests ADD COLUMN {column_name} {column_type}"
            )

    graph_freshness_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(graph_index_freshness)").fetchall()
    }
    graph_freshness_column_defs = {
        "workspace_id": "TEXT",
        "status": "TEXT NOT NULL DEFAULT 'unknown'",
        "indexed_at": "TEXT",
        "source_fingerprint": "TEXT",
        "last_successful_index_ref": "TEXT",
        "graph_snapshot_ref": "TEXT",
        "graph_snapshot_hash": "TEXT",
        "error_code": "TEXT",
        "error_detail_ref": "TEXT",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    }
    for column_name, column_type in graph_freshness_column_defs.items():
        if column_name not in graph_freshness_columns:
            conn.execute(
                f"ALTER TABLE graph_index_freshness ADD COLUMN {column_name} {column_type}"
            )

    conn.executescript(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_ingestion_active_source_fingerprint
          ON graph_ingestion_runs(organization_id, source_ref, requested_fingerprint)
          WHERE status IN ('claimed', 'indexing');
        CREATE INDEX IF NOT EXISTS idx_graph_ingestion_status_retry
          ON graph_ingestion_runs(organization_id, status, next_retry_at);
        CREATE INDEX IF NOT EXISTS idx_graph_index_freshness_status
          ON graph_index_freshness(organization_id, status, updated_at);
        """
    )


def _init_db_unlocked(db_path: str | Path | None = None) -> None:
    with get_conn(db_path) as conn:
        _drop_legacy_knowledge_builder_graph_schema(conn)
        _upgrade_legacy_tables_before_schema_indexes(conn)
        _migrate_replay_envelope_binding_types(conn)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
              event_id TEXT PRIMARY KEY,
              event_type TEXT NOT NULL,
              occurred_at TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              correlation_id TEXT,
              causation_id TEXT,
              stream_id TEXT,
              content_hash TEXT,
              prior_event_hash TEXT
            );

            CREATE TABLE IF NOT EXISTS signals (
              signal_id TEXT PRIMARY KEY,
              signal_type TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS semantier_schema_versions (
              schema_name TEXT PRIMARY KEY,
              schema_version INTEGER NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workflows (
              workflow_id TEXT PRIMARY KEY,
              signal_id TEXT NOT NULL,
              workflow_type TEXT NOT NULL,
              workflow_json TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rea_claims (
              rea_id TEXT PRIMARY KEY,
              workflow_id TEXT NOT NULL,
              -- Version scoping is intentionally claim-json-owned for backward
              -- compatibility with existing append-only REA rows. REAL claims
              -- must carry claim_json.dataset_version_id; readers must parse
              -- that field and fail closed for unversioned REAL claims.
              claim_json TEXT NOT NULL,
              claim_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS justifications (
              justification_id TEXT PRIMARY KEY,
              workflow_id TEXT NOT NULL,
              rea_id TEXT NOT NULL,
              justification_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS validation_results (
              validation_id TEXT PRIMARY KEY,
              justification_id TEXT NOT NULL,
              status TEXT NOT NULL,
              result_json TEXT NOT NULL,
              policy_bundle_version TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS journal_entries (
              journal_id TEXT PRIMARY KEY,
              justification_id TEXT NOT NULL,
              je_json TEXT NOT NULL,
              status TEXT NOT NULL,
              posted_at TEXT
            );

            CREATE TABLE IF NOT EXISTS exceptions (
              exception_id TEXT PRIMARY KEY,
              justification_id TEXT NOT NULL,
              status TEXT NOT NULL,
              reason_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              resolved_at TEXT
            );

            CREATE TABLE IF NOT EXISTS human_resolutions (
              resolution_id TEXT PRIMARY KEY,
              exception_id TEXT NOT NULL,
              resolution_json TEXT NOT NULL,
              approved_by TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hermes_trajectories (
              trajectory_id TEXT PRIMARY KEY,
              trajectory_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS shadow_evaluations (
              id TEXT PRIMARY KEY,
              justification_id TEXT,
              active_status TEXT,
              shadow_json TEXT,
              created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS policy_assignments (
              id TEXT PRIMARY KEY,
              justification_id TEXT,
              context_json TEXT,
              active_policy TEXT,
              shadow_policies TEXT,
              created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS resolution_outcomes (
              outcome_id TEXT PRIMARY KEY,
              dimension TEXT,
              drift_type TEXT,
              action TEXT,
              success INTEGER,
              impact_delta REAL,
              created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS policy_proposals (
              proposal_id TEXT PRIMARY KEY,
              rule_change TEXT,
              source TEXT,
              confidence REAL,
              status TEXT,
              created_at TEXT
            );

            -- NEW: persistent contextual RL stats
            CREATE TABLE IF NOT EXISTS policy_stats (
              id TEXT PRIMARY KEY,
              context_key TEXT,
              policy TEXT,
              count INTEGER,
              reward REAL,
              created_at TEXT
            );

            -- Governance promotion candidates (customised Holographic pattern)
            -- Stores proposed policy_bundle and ontology_entry changes that must
            -- pass a replay gate and receive human approval before becoming ACTIVE.
            CREATE TABLE IF NOT EXISTS policy_candidates (
              candidate_id    TEXT PRIMARY KEY,
              name            TEXT NOT NULL,
              category        TEXT NOT NULL
                CHECK(category IN ('policy_bundle', 'ontology_entry', 'coa_version', 'investment_radar_universe', 'investment_radar_scoring_profile', 'learned_procedure', 'organization_learning_contribution')),
              semantic_tier   TEXT NOT NULL DEFAULT 'T5'
                CHECK(semantic_tier IN ('T1', 'T2', 'T3', 'T4', 'T5', 'T6')),
              content_yaml    TEXT NOT NULL,
              content_hash    TEXT NOT NULL,
              status          TEXT NOT NULL DEFAULT 'PENDING'
                CHECK(status IN ('PENDING', 'APPROVED', 'REJECTED', 'ACTIVE')),
              replay_result   TEXT,
              proposed_by     TEXT NOT NULL,
              approved_by     TEXT,
              scoped_owner    TEXT,
              rollback_window_hours INTEGER,
              rollback_strategy TEXT,
              created_at      TEXT NOT NULL,
              approved_at     TEXT,
              notes           TEXT,
              review_status   TEXT,
              jurisdiction_reason TEXT,
              imported_legacy INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_candidates_name
              ON policy_candidates(name);
            CREATE INDEX IF NOT EXISTS idx_candidates_status
              ON policy_candidates(status);
            CREATE INDEX IF NOT EXISTS idx_candidates_category
              ON policy_candidates(category);
            CREATE INDEX IF NOT EXISTS idx_candidates_hash
              ON policy_candidates(content_hash);

            CREATE TABLE IF NOT EXISTS policy_candidate_events (
              candidate_event_id TEXT PRIMARY KEY,
              candidate_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              event_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_candidate_events_candidate
              ON policy_candidate_events(candidate_id, created_at);

            CREATE TABLE IF NOT EXISTS policy_rule_candidates (
              rule_candidate_id TEXT PRIMARY KEY,
              organization_id TEXT,
              proposed_workspace_scope TEXT,
              proposed_workspace_id TEXT,
              authority_domain TEXT,
              semantic_tier TEXT,
              jurisdiction TEXT,
              rule_family TEXT NOT NULL,
              candidate_state TEXT NOT NULL
                CHECK(candidate_state IN ('DRAFT', 'PROPOSED', 'NEEDS_REVIEW', 'APPROVED', 'ACTIVATED', 'SUPERSEDED', 'REJECTED')),
              source_anchor_refs_json TEXT NOT NULL,
              applicability_scope_json TEXT NOT NULL,
              extracted_rationale TEXT NOT NULL,
              draft_rule_text TEXT NOT NULL,
              severity TEXT,
              confidence REAL,
              uncertainty_notes TEXT,
              human_edits_json TEXT NOT NULL,
              approval_evidence_json TEXT NOT NULL,
              test_evidence_json TEXT NOT NULL,
              supersedes_candidate_id TEXT,
              activation_refs_json TEXT NOT NULL,
              candidate_idempotency_key TEXT,
              created_by TEXT NOT NULL,
              created_by_actor_type TEXT NOT NULL
                CHECK(created_by_actor_type IN ('ai', 'plugin', 'human', 'migration', 'system')),
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(supersedes_candidate_id) REFERENCES policy_rule_candidates(rule_candidate_id)
            );

            CREATE INDEX IF NOT EXISTS idx_policy_rule_candidates_family_state
              ON policy_rule_candidates(rule_family, candidate_state, updated_at);
            CREATE INDEX IF NOT EXISTS idx_policy_rule_candidates_hash
              ON policy_rule_candidates(content_hash);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_policy_rule_candidates_idempotency
              ON policy_rule_candidates(candidate_idempotency_key)
              WHERE candidate_idempotency_key IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_policy_rule_candidates_scope
              ON policy_rule_candidates(
                organization_id,
                proposed_workspace_scope,
                proposed_workspace_id,
                authority_domain,
                semantic_tier,
                jurisdiction
              );

            CREATE TABLE IF NOT EXISTS policy_rule_candidate_events (
              candidate_event_id TEXT PRIMARY KEY,
              rule_candidate_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              actor TEXT NOT NULL,
              actor_authority TEXT,
              from_state TEXT,
              to_state TEXT NOT NULL,
              event_json TEXT NOT NULL,
              prior_event_hash TEXT,
              event_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(rule_candidate_id) REFERENCES policy_rule_candidates(rule_candidate_id)
            );

            CREATE INDEX IF NOT EXISTS idx_policy_rule_candidate_events_candidate
              ON policy_rule_candidate_events(rule_candidate_id, created_at);

            CREATE TABLE IF NOT EXISTS governed_rule_versions (
              rule_version_id TEXT PRIMARY KEY,
              rule_candidate_id TEXT NOT NULL,
              organization_id TEXT,
              authority_domain TEXT,
              semantic_tier TEXT,
              jurisdiction TEXT,
              rule_family TEXT NOT NULL,
              rule_state TEXT NOT NULL
                CHECK(rule_state IN ('APPROVED', 'ACTIVATED', 'SUPERSEDED', 'REJECTED')),
              source_graph_version TEXT,
              source_graph_release_hash TEXT,
              source_graph_rule_id TEXT,
              source_anchor_refs_json TEXT NOT NULL,
              applicability_scope_json TEXT NOT NULL,
              governed_rule_text TEXT NOT NULL,
              extracted_rationale TEXT NOT NULL,
              severity TEXT,
              confidence REAL,
              uncertainty_notes TEXT,
              reviewer_edits_json TEXT NOT NULL,
              approval_evidence_json TEXT NOT NULL,
              test_evidence_json TEXT NOT NULL,
              activation_refs_json TEXT NOT NULL,
              supersedes_rule_version_id TEXT,
              activated_at TEXT,
              activated_by TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(rule_candidate_id) REFERENCES policy_rule_candidates(rule_candidate_id),
              FOREIGN KEY(supersedes_rule_version_id) REFERENCES governed_rule_versions(rule_version_id)
            );

            CREATE INDEX IF NOT EXISTS idx_governed_rule_versions_family_state
              ON governed_rule_versions(rule_family, rule_state, created_at);
            CREATE INDEX IF NOT EXISTS idx_governed_rule_versions_candidate
              ON governed_rule_versions(rule_candidate_id);
            CREATE INDEX IF NOT EXISTS idx_governed_rule_versions_owner
              ON governed_rule_versions(
                organization_id,
                authority_domain,
                semantic_tier,
                jurisdiction
              );

            CREATE TABLE IF NOT EXISTS governed_rule_activation (
              activation_id TEXT PRIMARY KEY,
              rule_version_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              activation_scope TEXT NOT NULL
                CHECK(activation_scope IN ('organization', 'workspace')),
              workspace_id TEXT,
              activation_status TEXT NOT NULL CHECK(activation_status = 'ACTIVE'),
              effective_from TEXT NOT NULL,
              effective_to TEXT,
              activated_by TEXT NOT NULL,
              actor_authority TEXT NOT NULL,
              approval_event_ref TEXT NOT NULL,
              source_activation_ref TEXT,
              activation_scope_hash TEXT NOT NULL,
              supersedes_activation_id TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(rule_version_id) REFERENCES governed_rule_versions(rule_version_id),
              FOREIGN KEY(supersedes_activation_id) REFERENCES governed_rule_activation(activation_id)
            );

            CREATE INDEX IF NOT EXISTS idx_governed_rule_activation_scope
              ON governed_rule_activation(organization_id, activation_scope, workspace_id, effective_from);
            CREATE INDEX IF NOT EXISTS idx_governed_rule_activation_rule
              ON governed_rule_activation(rule_version_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_governed_rule_activation_idempotency
              ON governed_rule_activation(
                organization_id,
                rule_version_id,
                activation_scope,
                COALESCE(workspace_id, ''),
                effective_from,
                COALESCE(effective_to, ''),
                approval_event_ref
              );

            CREATE TRIGGER IF NOT EXISTS trg_governed_rule_activation_immutable
            BEFORE UPDATE ON governed_rule_activation
            FOR EACH ROW
            BEGIN
              SELECT RAISE(ABORT, 'governed rule activation rows are immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS trg_governed_rule_activation_append_only
            BEFORE DELETE ON governed_rule_activation
            FOR EACH ROW
            BEGIN
              SELECT RAISE(ABORT, 'governed rule activation rows are append-only');
            END;

            CREATE TABLE IF NOT EXISTS governed_rule_activation_event (
              activation_event_id TEXT PRIMARY KEY,
              activation_stream_id TEXT NOT NULL,
              activation_id TEXT NOT NULL,
              event_type TEXT NOT NULL
                CHECK(event_type IN (
                  'ACTIVATED',
                  'SUSPENDED',
                  'RESUMED',
                  'REVOKED',
                  'SUPERSESSION_RECORDED',
                  'SCOPE_REPLACEMENT_RECORDED',
                  'EXPIRATION_RECORDED'
                )),
              from_state TEXT,
              to_state TEXT,
              effective_at TEXT NOT NULL,
              recorded_at TEXT NOT NULL,
              event_sequence INTEGER NOT NULL,
              actor TEXT NOT NULL,
              actor_authority TEXT NOT NULL,
              reason TEXT,
              prior_event_hash TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(activation_id) REFERENCES governed_rule_activation(activation_id),
              UNIQUE(activation_stream_id, event_sequence)
            );

            CREATE INDEX IF NOT EXISTS idx_governed_rule_activation_event_activation
              ON governed_rule_activation_event(activation_id, effective_at, event_sequence);

            CREATE TRIGGER IF NOT EXISTS trg_governed_rule_activation_event_immutable
            BEFORE UPDATE ON governed_rule_activation_event
            FOR EACH ROW
            BEGIN
              SELECT RAISE(ABORT, 'governed rule activation events are immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS trg_governed_rule_activation_event_append_only
            BEFORE DELETE ON governed_rule_activation_event
            FOR EACH ROW
            BEGIN
              SELECT RAISE(ABORT, 'governed rule activation events are append-only');
            END;


            CREATE TRIGGER IF NOT EXISTS trg_governed_rule_versions_activated_immutable
            BEFORE UPDATE ON governed_rule_versions
            FOR EACH ROW
            BEGIN
              SELECT RAISE(ABORT, 'governed rule versions are immutable');
            END;

            CREATE TABLE IF NOT EXISTS tender_detection_runs (
              run_id TEXT PRIMARY KEY,
              tender_document_id TEXT NOT NULL,
              document_text TEXT NOT NULL DEFAULT '',
              source_document_hash TEXT NOT NULL DEFAULT '',
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              parent_run_id TEXT,
              root_run_id TEXT,
              project_metadata_json TEXT NOT NULL,
              review_purpose TEXT NOT NULL,
              caller_role TEXT NOT NULL,
              requested_rule_families_json TEXT NOT NULL,
              activation_set_snapshot_id TEXT NOT NULL,
              activation_resolver_policy_version TEXT NOT NULL,
              resolved_activation_set_hash TEXT NOT NULL,
              ignored_local_rules_json TEXT NOT NULL,
              idempotency_key TEXT,
              request_hash TEXT,
              run_status TEXT NOT NULL DEFAULT 'FINALIZED',
              finding_set_hash TEXT,
              runtime_conflict_set_hash TEXT,
              replay_status TEXT NOT NULL DEFAULT 'CREATED',
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tender_detection_findings (
              finding_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              tender_document_id TEXT NOT NULL,
              clause_id TEXT NOT NULL,
              clause_anchor TEXT NOT NULL,
              issue_type TEXT NOT NULL,
              matched_text TEXT NOT NULL,
              triggered_rule_version_id TEXT NOT NULL,
              source_anchor_refs_json TEXT NOT NULL,
              judgment_basis TEXT NOT NULL,
              severity TEXT NOT NULL,
              confidence REAL NOT NULL,
              suggested_replacement TEXT,
              escalation_flag INTEGER NOT NULL DEFAULT 0,
              resolver_evidence_ref TEXT NOT NULL,
              target_anchor_ref TEXT,
              target_element_kind TEXT,
              target_element_id TEXT,
              normalized_match TEXT,
              start_offset INTEGER,
              end_offset INTEGER,
              occurrence_index INTEGER,
              target_element_hash TEXT,
              target_evidence_ref TEXT,
              source_graph_version TEXT,
              source_graph_release_hash TEXT,
              source_graph_rule_id TEXT,
              triggered_rule_version_refs_json TEXT NOT NULL DEFAULT '[]',
              winning_rule_version_ref TEXT,
              conflicting_rule_version_refs_json TEXT NOT NULL DEFAULT '[]',
              conflict_resolution_ref TEXT,
              matched_sensitive_concept_id TEXT,
              observed_expression TEXT,
              detection_method TEXT NOT NULL DEFAULT 'exact',
              semantic_relation TEXT NOT NULL DEFAULT 'exact',
              evidence_text TEXT,
              precedent_refs_json TEXT NOT NULL DEFAULT '[]',
              decision_status TEXT NOT NULL DEFAULT 'violation',
              grounding_status TEXT NOT NULL DEFAULT 'ungrounded',
              decision_context_json TEXT NOT NULL DEFAULT '{}',
              candidate_finding_schema_version TEXT NOT NULL DEFAULT 'contextgraph_candidate_finding.v2',
              semantic_profile_id TEXT,
              derivation_shape TEXT,
              frame_derivation_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES tender_detection_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS tender_detection_runtime_conflicts (
              conflict_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              conflict_type TEXT NOT NULL,
              target_anchor_ref TEXT,
              rule_version_ids_json TEXT NOT NULL,
              start_offset INTEGER,
              end_offset INTEGER,
              conflict_payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES tender_detection_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS tender_detection_dispositions (
              disposition_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              finding_id TEXT NOT NULL,
              disposition TEXT NOT NULL
                CHECK(disposition IN ('accepted', 'rejected', 'edited', 'deferred', 'escalated')),
              edited_replacement TEXT,
              rejection_rationale TEXT,
              decided_by TEXT NOT NULL,
              decided_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES tender_detection_runs(run_id),
              FOREIGN KEY(finding_id) REFERENCES tender_detection_findings(finding_id)
            );

            CREATE TABLE IF NOT EXISTS tender_detection_reports (
              report_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              report_json TEXT NOT NULL,
              replay_event_id TEXT NOT NULL,
              replay_binding_ref TEXT NOT NULL,
              prior_labeling_report_id TEXT,
              created_by TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES tender_detection_runs(run_id),
              FOREIGN KEY(prior_labeling_report_id) REFERENCES tender_detection_reports(report_id)
            );

            CREATE TABLE IF NOT EXISTS sensitive_labeling_labeled_docx_derivatives (
              labeled_docx_derivative_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              parent_target_artifact_ref TEXT NOT NULL,
              parent_target_hash TEXT NOT NULL,
              labeled_docx_artifact_ref TEXT NOT NULL,
              labeled_docx_hash TEXT NOT NULL,
              comment_writer_version TEXT NOT NULL,
              finding_set_hash TEXT NOT NULL,
              generated_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES tender_detection_runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_labeled_docx_derivatives_run
              ON sensitive_labeling_labeled_docx_derivatives(run_id, generated_at);

            CREATE TABLE IF NOT EXISTS runtime_feedback_events (
              feedback_event_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              source_kind TEXT NOT NULL,
              source_run_id TEXT NOT NULL,
              source_finding_id TEXT,
              feedback_type TEXT NOT NULL
                CHECK(feedback_type IN ('false_positive', 'false_negative', 'ambiguity', 'conflict', 'remediation_rejected', 'user_edited_remediation', 'missing_control', 'weak_explanation')),
              user_disposition_json TEXT NOT NULL,
              escalation_outcome TEXT
                CHECK(escalation_outcome IS NULL OR escalation_outcome IN ('not_escalated', 'escalated', 'resolved', 'deferred')),
              feedback_evidence_json TEXT NOT NULL,
              affected_term TEXT,
              affected_control_ref TEXT,
              affected_relation_ref TEXT,
              source_span_json TEXT NOT NULL,
              created_by TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runtime_feedback_candidate_deltas (
              candidate_delta_id TEXT PRIMARY KEY,
              feedback_event_id TEXT NOT NULL,
              discovery_run_id TEXT NOT NULL,
              delta_kind TEXT NOT NULL
                CHECK(delta_kind IN ('not_same_as', 'allowed_context_for', 'prohibited_context_for', 'split_cluster', 'merge_cluster', 'new_control_candidate')),
              candidate_graph_ref_json TEXT NOT NULL,
              evaluation_routing_json TEXT NOT NULL,
              governance_state TEXT NOT NULL
                CHECK(governance_state IN ('DISCOVERED', 'CURATED', 'PROPOSED', 'APPROVED', 'ACTIVATED', 'REJECTED')),
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(feedback_event_id) REFERENCES runtime_feedback_events(feedback_event_id),
              FOREIGN KEY(discovery_run_id) REFERENCES knowledge_builder_discovery_runs(discovery_run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_tender_detection_runs_org_created
              ON tender_detection_runs(organization_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_tender_detection_findings_run
              ON tender_detection_findings(run_id);
            CREATE INDEX IF NOT EXISTS idx_tender_detection_runtime_conflicts_run
              ON tender_detection_runtime_conflicts(run_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_tender_detection_runs_idempotency
              ON tender_detection_runs(organization_id, COALESCE(workspace_id, ''), idempotency_key)
              WHERE idempotency_key IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_tender_detection_dispositions_run
              ON tender_detection_dispositions(run_id, decided_at);
            CREATE INDEX IF NOT EXISTS idx_tender_detection_reports_run
              ON tender_detection_reports(run_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_runtime_feedback_source
              ON runtime_feedback_events(source_kind, source_run_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_runtime_feedback_deltas_event
              ON runtime_feedback_candidate_deltas(feedback_event_id, created_at);

            CREATE TABLE IF NOT EXISTS knowledge_builder_sources (
              source_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              source_kind TEXT NOT NULL
                CHECK(source_kind IN ('folder', 'file', 'text')),
              source_ref TEXT NOT NULL,
              source_text TEXT NOT NULL DEFAULT '',
              source_hash TEXT NOT NULL,
              source_metadata_json TEXT NOT NULL DEFAULT '{}',
              governance_state TEXT NOT NULL
                CHECK(governance_state IN ('DISCOVERED', 'CURATED', 'PROPOSED', 'APPROVED', 'ACTIVATED', 'REJECTED')),
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_builder_source_identities (
              source_identity_ref TEXT NOT NULL,
              source_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              workspace_id TEXT NOT NULL,
              lineage_id TEXT NOT NULL,
              source_hash TEXT NOT NULL,
              source_version TEXT NOT NULL,
              supersedes_source_identity_ref TEXT,
              artifact_ref TEXT NOT NULL,
              artifact_hash TEXT NOT NULL,
              media_type TEXT NOT NULL,
              original_name TEXT NOT NULL,
              created_at TEXT NOT NULL,
              schema_version TEXT NOT NULL,
              source_identity_json TEXT NOT NULL,
              source_identity_hash TEXT NOT NULL,
              registration_evidence_json TEXT NOT NULL,
              registration_evidence_hash TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              PRIMARY KEY (organization_id, workspace_id, source_identity_ref),
              FOREIGN KEY(source_id) REFERENCES knowledge_builder_sources(source_id),
              FOREIGN KEY(
                organization_id, workspace_id, supersedes_source_identity_ref
              ) REFERENCES knowledge_builder_source_identities(
                organization_id, workspace_id, source_identity_ref
              ),
              UNIQUE(source_id, source_version),
              UNIQUE(organization_id, workspace_id, lineage_id, source_version)
            );
            CREATE INDEX IF NOT EXISTS idx_kb_source_identities_source
              ON knowledge_builder_source_identities(source_id);
            CREATE INDEX IF NOT EXISTS idx_kb_source_identities_lineage_order
              ON knowledge_builder_source_identities(
                organization_id, workspace_id, lineage_id, created_at,
                source_identity_ref
              );

            CREATE TABLE IF NOT EXISTS knowledge_builder_discovery_runs (
              discovery_run_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              source_id TEXT NOT NULL,
              run_status TEXT NOT NULL
                CHECK(run_status IN ('queued', 'running', 'completed', 'failed')),
              model_provenance_json TEXT NOT NULL,
              procedure_provenance_json TEXT NOT NULL,
              governance_state TEXT NOT NULL
                CHECK(governance_state IN ('DISCOVERED', 'CURATED', 'PROPOSED', 'APPROVED', 'ACTIVATED', 'REJECTED')),
              started_at TEXT NOT NULL,
              completed_at TEXT,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(source_id) REFERENCES knowledge_builder_sources(source_id)
            );

            CREATE INDEX IF NOT EXISTS idx_kb_builder_runs_org_created
              ON knowledge_builder_discovery_runs(organization_id, started_at);

            CREATE TABLE IF NOT EXISTS knowledge_builder_rule_candidate_evidence (
              evidence_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              knowledge_source_id TEXT NOT NULL,
              knowledge_source_compilation_run_id TEXT NOT NULL,
              rule_candidate_id TEXT NOT NULL,
              candidate_state_snapshot TEXT NOT NULL,
              source_anchor_refs_json TEXT NOT NULL DEFAULT '[]',
              evidence_payload_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(knowledge_source_id) REFERENCES knowledge_source(knowledge_source_id),
              FOREIGN KEY(knowledge_source_compilation_run_id)
                REFERENCES knowledge_source_compilation_runs(knowledge_source_compilation_run_id),
              FOREIGN KEY(rule_candidate_id) REFERENCES policy_rule_candidates(rule_candidate_id),
              UNIQUE(knowledge_source_compilation_run_id, rule_candidate_id)
            );
            CREATE INDEX IF NOT EXISTS idx_kb_rule_candidate_evidence_candidate
              ON knowledge_builder_rule_candidate_evidence(rule_candidate_id);

            CREATE TABLE IF NOT EXISTS knowledge_source_compilation_runs (
              knowledge_source_compilation_run_id TEXT PRIMARY KEY,
              knowledge_source_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              semantic_purpose TEXT NOT NULL,
              source_kind TEXT NOT NULL,
              source_ref TEXT NOT NULL,
              source_artifact_ref TEXT NOT NULL,
              source_hash TEXT NOT NULL,
              normalized_document_artifact_ref TEXT NOT NULL,
              normalized_document_artifact_hash TEXT NOT NULL,
              document_schema_version TEXT NOT NULL,
              parser_method TEXT NOT NULL,
              parser_version TEXT NOT NULL,
              provider_artifact_refs_json TEXT NOT NULL DEFAULT '[]',
              provider_artifact_hashes_json TEXT NOT NULL DEFAULT '[]',
              compiler_profile_version TEXT NOT NULL,
              mapped_source_anchor_refs_json TEXT NOT NULL DEFAULT '[]',
              created_by TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(knowledge_source_id) REFERENCES knowledge_source(knowledge_source_id)
            );

            CREATE TABLE IF NOT EXISTS knowledge_source_compilation_anchors (
              compilation_anchor_ref TEXT PRIMARY KEY,
              knowledge_source_compilation_run_id TEXT NOT NULL,
              knowledge_source_id TEXT NOT NULL,
              canonical_document_anchor_ref TEXT NOT NULL,
              source_hash TEXT NOT NULL,
              compiler_profile_version TEXT NOT NULL,
              anchor_payload_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(knowledge_source_compilation_run_id)
                REFERENCES knowledge_source_compilation_runs(knowledge_source_compilation_run_id),
              FOREIGN KEY(knowledge_source_id) REFERENCES knowledge_source(knowledge_source_id)
            );

            CREATE INDEX IF NOT EXISTS idx_ks_compilation_source
              ON knowledge_source_compilation_runs(knowledge_source_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_ks_compilation_anchor_run
              ON knowledge_source_compilation_anchors(knowledge_source_compilation_run_id);

            CREATE TABLE IF NOT EXISTS semantic_relation_candidates (
              semantic_relation_candidate_id TEXT PRIMARY KEY,
              source_relation_ref TEXT,
              discovery_run_id TEXT NOT NULL,
              from_node_ref TEXT NOT NULL,
              to_node_ref TEXT NOT NULL,
              relation_type TEXT NOT NULL
                CHECK(relation_type IN ('synonym_of', 'variant_of', 'projects_to', 'not_same_as', 'allowed_context_for', 'prohibited_context_for', 'exception_to', 'conflicts_with', 'candidate_semantic_neighborhood')),
              curation_state TEXT NOT NULL
                CHECK(curation_state IN ('CURATED', 'PROPOSED', 'REJECTED')),
              evidence_summary TEXT NOT NULL,
              source_anchor_refs_json TEXT NOT NULL,
              reviewer_notes TEXT,
              created_by TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(discovery_run_id) REFERENCES knowledge_builder_discovery_runs(discovery_run_id)
            );

            CREATE TABLE IF NOT EXISTS canonical_term_candidates (
              term_candidate_id TEXT PRIMARY KEY,
              discovery_run_id TEXT NOT NULL,
              domain TEXT NOT NULL,
              canonical_label TEXT NOT NULL,
              definition TEXT NOT NULL,
              aliases_json TEXT NOT NULL,
              allowed_contexts_json TEXT NOT NULL,
              prohibited_contexts_json TEXT NOT NULL,
              source_anchor_refs_json TEXT NOT NULL,
              evidence_summary TEXT NOT NULL,
              proposed_runtime_effect_json TEXT NOT NULL,
              governance_state TEXT NOT NULL
                CHECK(governance_state IN ('DISCOVERED', 'CURATED', 'PROPOSED', 'APPROVED', 'ACTIVATED', 'REJECTED')),
              created_by TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(discovery_run_id) REFERENCES knowledge_builder_discovery_runs(discovery_run_id)
            );

            CREATE TABLE IF NOT EXISTS knowledge_builder_curation_events (
              curation_event_id TEXT PRIMARY KEY,
              discovery_run_id TEXT NOT NULL,
              target_kind TEXT NOT NULL
                CHECK(target_kind IN ('relation', 'term')),
              target_id TEXT NOT NULL,
              decision TEXT NOT NULL,
              event_json TEXT NOT NULL,
              actor TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(discovery_run_id) REFERENCES knowledge_builder_discovery_runs(discovery_run_id)
            );

            CREATE TABLE IF NOT EXISTS knowledge_builder_evaluation_datasets (
              evaluation_dataset_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              discovery_run_id TEXT,
              domain TEXT NOT NULL,
              name TEXT NOT NULL,
              description TEXT NOT NULL,
              example_ids_json TEXT NOT NULL,
              governance_state TEXT NOT NULL
                CHECK(governance_state IN ('DISCOVERED', 'CURATED', 'PROPOSED', 'APPROVED', 'ACTIVATED', 'REJECTED')),
              created_by TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(discovery_run_id) REFERENCES knowledge_builder_discovery_runs(discovery_run_id)
            );

            CREATE TABLE IF NOT EXISTS knowledge_builder_evaluation_examples (
              evaluation_example_id TEXT PRIMARY KEY,
              evaluation_dataset_id TEXT NOT NULL,
              case_type TEXT NOT NULL
                CHECK(case_type IN ('positive', 'negative', 'ambiguous', 'known_false_positive', 'known_false_negative', 'escalation_required')),
              input_text TEXT NOT NULL,
              expected_outcome TEXT NOT NULL
                CHECK(expected_outcome IN ('match', 'exclude', 'ambiguous', 'escalate')),
              expected_relation_type TEXT,
              expected_runtime_effect_json TEXT NOT NULL,
              source_anchor_refs_json TEXT NOT NULL,
              reviewer_notes TEXT,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(evaluation_dataset_id) REFERENCES knowledge_builder_evaluation_datasets(evaluation_dataset_id)
            );

            CREATE TABLE IF NOT EXISTS knowledge_builder_evaluation_runs (
              evaluation_run_id TEXT PRIMARY KEY,
              evaluation_dataset_id TEXT NOT NULL,
              discovery_run_id TEXT,
              candidate_snapshot_hash TEXT NOT NULL,
              run_status TEXT NOT NULL
                CHECK(run_status IN ('queued', 'running', 'completed', 'failed')),
              metrics_json TEXT NOT NULL,
              authority_notice TEXT NOT NULL,
              created_by TEXT NOT NULL,
              started_at TEXT NOT NULL,
              completed_at TEXT,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(evaluation_dataset_id) REFERENCES knowledge_builder_evaluation_datasets(evaluation_dataset_id),
              FOREIGN KEY(discovery_run_id) REFERENCES knowledge_builder_discovery_runs(discovery_run_id)
            );

            CREATE TABLE IF NOT EXISTS knowledge_builder_evaluation_results (
              evaluation_result_id TEXT PRIMARY KEY,
              evaluation_run_id TEXT NOT NULL,
              evaluation_example_id TEXT NOT NULL,
              expected_outcome TEXT NOT NULL,
              actual_outcome TEXT NOT NULL
                CHECK(actual_outcome IN ('match', 'exclude', 'ambiguous', 'escalate')),
              matched_path_json TEXT NOT NULL,
              excluded_path_json TEXT NOT NULL,
              human_rating TEXT
                CHECK(human_rating IS NULL OR human_rating IN ('pass', 'fail', 'needs_review')),
              ai_assisted_rating TEXT NOT NULL
                CHECK(ai_assisted_rating IN ('pass', 'fail', 'needs_review')),
              explanation_acceptance TEXT NOT NULL
                CHECK(explanation_acceptance IN ('accepted', 'rejected', 'needs_review')),
              error_labels_json TEXT NOT NULL,
              reviewer_notes TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(evaluation_run_id) REFERENCES knowledge_builder_evaluation_runs(evaluation_run_id),
              FOREIGN KEY(evaluation_example_id) REFERENCES knowledge_builder_evaluation_examples(evaluation_example_id)
            );

            CREATE TABLE IF NOT EXISTS runtime_semantic_authority_versions (
              authority_version_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              term_candidate_id TEXT NOT NULL,
              canonical_label TEXT NOT NULL,
              domain TEXT NOT NULL,
              semantic_relation_candidate_ids_json TEXT NOT NULL,
              semantic_controls_json TEXT NOT NULL,
              classification_controls_json TEXT NOT NULL,
              evaluation_run_id TEXT NOT NULL,
              candidate_lineage_json TEXT NOT NULL,
              authority_state TEXT NOT NULL
                CHECK(authority_state IN ('APPROVED', 'ACTIVATED', 'SUPERSEDED', 'REJECTED')),
              approved_by TEXT NOT NULL,
              approved_at TEXT NOT NULL,
              activated_by TEXT NOT NULL,
              activated_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(term_candidate_id) REFERENCES canonical_term_candidates(term_candidate_id),
              FOREIGN KEY(evaluation_run_id) REFERENCES knowledge_builder_evaluation_runs(evaluation_run_id)
            );

            CREATE TABLE IF NOT EXISTS runtime_semantic_read_model_rebuilds (
              rebuild_id TEXT PRIMARY KEY,
              authority_version_id TEXT NOT NULL,
              rebuild_status TEXT NOT NULL
                CHECK(rebuild_status IN ('completed', 'failed')),
              term_resolution_index_hash TEXT NOT NULL,
              concept_neighborhood_index_hash TEXT NOT NULL,
              runtime_control_index_hash TEXT NOT NULL,
              audit_path_projection_hash TEXT NOT NULL,
              non_authoritative_notice TEXT NOT NULL,
              rebuilt_by TEXT NOT NULL,
              rebuilt_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(authority_version_id) REFERENCES runtime_semantic_authority_versions(authority_version_id)
            );

            CREATE TABLE IF NOT EXISTS term_resolution_index (
              index_entry_id TEXT PRIMARY KEY,
              authority_version_id TEXT NOT NULL,
              term_text TEXT NOT NULL,
              canonical_label TEXT NOT NULL,
              domain TEXT NOT NULL,
              source_authority_hash TEXT NOT NULL,
              non_authoritative INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(authority_version_id) REFERENCES runtime_semantic_authority_versions(authority_version_id)
            );

            CREATE TABLE IF NOT EXISTS concept_neighborhood_index (
              index_entry_id TEXT PRIMARY KEY,
              authority_version_id TEXT NOT NULL,
              canonical_label TEXT NOT NULL,
              relation_ids_json TEXT NOT NULL,
              source_authority_hash TEXT NOT NULL,
              non_authoritative INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(authority_version_id) REFERENCES runtime_semantic_authority_versions(authority_version_id)
            );

            CREATE TABLE IF NOT EXISTS runtime_control_index (
              index_entry_id TEXT PRIMARY KEY,
              authority_version_id TEXT NOT NULL,
              control_family TEXT NOT NULL,
              control_json TEXT NOT NULL,
              source_authority_hash TEXT NOT NULL,
              non_authoritative INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(authority_version_id) REFERENCES runtime_semantic_authority_versions(authority_version_id)
            );

            CREATE TABLE IF NOT EXISTS audit_path_projection (
              projection_id TEXT PRIMARY KEY,
              authority_version_id TEXT NOT NULL,
              lineage_json TEXT NOT NULL,
              evaluation_evidence_json TEXT NOT NULL,
              source_authority_hash TEXT NOT NULL,
              non_authoritative INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(authority_version_id) REFERENCES runtime_semantic_authority_versions(authority_version_id)
            );

            CREATE INDEX IF NOT EXISTS idx_semantic_relation_candidates_run
              ON semantic_relation_candidates(discovery_run_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_canonical_term_candidates_run
              ON canonical_term_candidates(discovery_run_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_kb_curation_events_run
              ON knowledge_builder_curation_events(discovery_run_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_kb_eval_datasets_org
              ON knowledge_builder_evaluation_datasets(organization_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_kb_eval_examples_dataset
              ON knowledge_builder_evaluation_examples(evaluation_dataset_id);
            CREATE INDEX IF NOT EXISTS idx_kb_eval_runs_dataset
              ON knowledge_builder_evaluation_runs(evaluation_dataset_id, started_at);
            CREATE INDEX IF NOT EXISTS idx_kb_eval_results_run
              ON knowledge_builder_evaluation_results(evaluation_run_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_semantic_authority_org
              ON runtime_semantic_authority_versions(organization_id, activated_at);
            CREATE INDEX IF NOT EXISTS idx_term_resolution_authority
              ON term_resolution_index(authority_version_id, term_text);
            CREATE INDEX IF NOT EXISTS idx_concept_neighborhood_authority
              ON concept_neighborhood_index(authority_version_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_control_authority
              ON runtime_control_index(authority_version_id);
            CREATE INDEX IF NOT EXISTS idx_audit_path_authority
              ON audit_path_projection(authority_version_id);

            CREATE TABLE IF NOT EXISTS t6_materialization_candidates (
              candidate_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              workspace_id TEXT,
              source_channel TEXT NOT NULL,
              source_session_id TEXT,
              source_message_id TEXT,
              candidate_json TEXT NOT NULL,
              candidate_hash TEXT NOT NULL,
              validation_status TEXT,
              fact_maturity_stage TEXT NOT NULL
                CHECK(fact_maturity_stage IN ('C0_CAPTURED', 'C1_VALIDATED', 'C2_APPROVED', 'C3_MATERIALIZED', 'REJECTED')),
              projection_status TEXT NOT NULL
                CHECK(projection_status IN ('PROJECTION_PENDING', 'PROJECTION_EXCEPTION', 'PROJECTION_MATERIALIZED')),
              policy_bundle_version TEXT,
              projection_context_ref TEXT,
              projection_context_hash TEXT,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS t6_materialization_decisions (
              decision_id TEXT PRIMARY KEY,
              candidate_id TEXT NOT NULL,
              decision_type TEXT NOT NULL
                CHECK(decision_type IN ('QUEUED_FOR_APPROVAL', 'AUTO_ALLOW', 'HUMAN_APPROVE', 'HUMAN_REJECT', 'SYSTEM_REJECT')),
              decision_scope TEXT NOT NULL
                CHECK(decision_scope IN ('FACT_ADMISSION', 'PROJECTION_TRUST', 'BOTH')),
              actor_type TEXT NOT NULL,
              actor_id TEXT,
              approver_authority_snapshot_ref TEXT,
              quorum_policy_ref TEXT,
              decision_target_refs_json TEXT,
              reason_json TEXT,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS t6_materialization_outcomes (
              outcome_id TEXT PRIMARY KEY,
              candidate_id TEXT NOT NULL,
              rea_id TEXT,
              justification_id TEXT,
              fact_commit_status TEXT,
              projection_result_id TEXT,
              projection_status TEXT NOT NULL
                CHECK(projection_status IN ('PROJECTION_PENDING', 'PROJECTION_EXCEPTION', 'PROJECTION_MATERIALIZED')),
              trust_state TEXT,
              replay_binding_ref TEXT,
              supersedes_outcome_id TEXT,
              content_hash TEXT,
              fact_maturity_stage TEXT NOT NULL
                CHECK(fact_maturity_stage IN ('C0_CAPTURED', 'C1_VALIDATED', 'C2_APPROVED', 'C3_MATERIALIZED', 'REJECTED')),
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS t6_materialization_transitions (
              transition_id TEXT PRIMARY KEY,
              candidate_id TEXT NOT NULL,
              from_state TEXT,
              to_state TEXT NOT NULL,
              transition_reason TEXT NOT NULL,
              actor_type TEXT NOT NULL,
              actor_id TEXT,
              prior_event_hash TEXT,
              event_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS t6_projection_exceptions (
              exception_id TEXT PRIMARY KEY,
              candidate_id TEXT NOT NULL,
              rea_id TEXT,
              projection_context_hash TEXT,
              exception_code TEXT NOT NULL,
              exception_details_json TEXT,
              projection_result_id TEXT,
              resolved_by_outcome_id TEXT,
              retry_after TEXT,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_t6_transitions_candidate
              ON t6_materialization_transitions(candidate_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_t6_outcomes_candidate
              ON t6_materialization_outcomes(candidate_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_t6_decisions_candidate
              ON t6_materialization_decisions(candidate_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_t6_projection_exceptions_candidate
              ON t6_projection_exceptions(candidate_id, created_at);

            CREATE TABLE IF NOT EXISTS investment_radar_operator_outcomes (
              outcome_id TEXT PRIMARY KEY,
              candidate_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              outcome_type TEXT NOT NULL CHECK(outcome_type IN (
                'qualified', 'deferred', 'rejected', 'false_positive',
                'contacted', 'diligence_opened'
              )),
              reason TEXT NOT NULL,
              evidence_refs_json TEXT NOT NULL DEFAULT '[]',
              case_ref TEXT,
              actor_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_radar_operator_outcomes_case
              ON investment_radar_operator_outcomes(organization_id, case_ref, created_at);

            CREATE TABLE IF NOT EXISTS investment_radar_notifications (
              notification_id TEXT PRIMARY KEY,
              candidate_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              notification_type TEXT NOT NULL CHECK(notification_type IN (
                'candidate_admitted', 'candidate_materially_changed'
              )),
              content_hash TEXT NOT NULL UNIQUE,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_radar_notifications_scope
              ON investment_radar_notifications(organization_id, created_at);

            """
        )

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS legal_sources (
              source_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              canonical_title TEXT NOT NULL,
              issuer TEXT NOT NULL,
              authority_tier TEXT NOT NULL,
              jurisdiction TEXT NOT NULL,
              document_number TEXT,
              expected_status TEXT NOT NULL,
              governance_state TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_legal_sources_identity
              ON legal_sources(organization_id, canonical_title, issuer, jurisdiction, COALESCE(document_number, ''));

            CREATE TABLE IF NOT EXISTS legal_source_versions (
              version_id TEXT PRIMARY KEY,
              source_id TEXT NOT NULL,
              version_identity TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              normalized_hash TEXT,
              effective_from TEXT NOT NULL,
              effective_to TEXT,
              promulgated_at TEXT,
              lifecycle_state TEXT NOT NULL,
              source_provenance_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(source_id) REFERENCES legal_sources(source_id),
              UNIQUE(source_id, version_identity),
              UNIQUE(source_id, content_hash)
            );

            CREATE TABLE IF NOT EXISTS legal_source_artifacts (
              artifact_id TEXT PRIMARY KEY,
              version_id TEXT NOT NULL,
              artifact_kind TEXT NOT NULL,
              uri TEXT NOT NULL,
              retrieved_at TEXT NOT NULL,
              http_metadata_json TEXT NOT NULL DEFAULT '{}',
              mime_type TEXT NOT NULL,
              redirect_metadata_json TEXT NOT NULL DEFAULT '[]',
              content_hash TEXT NOT NULL,
              storage_ref TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(version_id) REFERENCES legal_source_versions(version_id),
              UNIQUE(version_id, artifact_kind, content_hash)
            );

            CREATE TABLE IF NOT EXISTS legal_source_sections (
              section_id TEXT PRIMARY KEY,
              version_id TEXT NOT NULL,
              stable_locator TEXT NOT NULL,
              parent_section_id TEXT,
              section_type TEXT NOT NULL,
              ordinal TEXT,
              title TEXT,
              text_hash TEXT NOT NULL,
              local_hash TEXT NOT NULL,
              anchor_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(version_id) REFERENCES legal_source_versions(version_id),
              FOREIGN KEY(parent_section_id) REFERENCES legal_source_sections(section_id),
              UNIQUE(version_id, stable_locator)
            );

            CREATE TABLE IF NOT EXISTS legal_source_version_edges (
              edge_id TEXT PRIMARY KEY,
              from_version_id TEXT NOT NULL,
              to_version_id TEXT NOT NULL,
              edge_type TEXT NOT NULL
                CHECK(edge_type IN ('AMENDS', 'SUPERSEDES', 'REPEALS', 'RESTATES', 'DERIVES_FROM')),
              rationale_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              FOREIGN KEY(from_version_id) REFERENCES legal_source_versions(version_id),
              FOREIGN KEY(to_version_id) REFERENCES legal_source_versions(version_id),
              UNIQUE(from_version_id, to_version_id, edge_type)
            );

            CREATE TABLE IF NOT EXISTS source_ingestion_events (
              ingestion_event_id TEXT PRIMARY KEY,
              source_id TEXT,
              version_id TEXT,
              event_type TEXT NOT NULL,
              event_json TEXT NOT NULL,
              actor_ref TEXT,
              organization_id TEXT,
              correlation_id TEXT,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_refresh_checks (
              refresh_check_id TEXT PRIMARY KEY,
              source_id TEXT NOT NULL,
              checked_uri TEXT NOT NULL,
              checked_at TEXT NOT NULL,
              status TEXT NOT NULL,
              observed_content_hash TEXT,
              result_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(source_id) REFERENCES legal_sources(source_id)
            );

            CREATE TABLE IF NOT EXISTS legal_source_refresh_schedules (
              schedule_id TEXT PRIMARY KEY,
              source_id TEXT NOT NULL,
              interval_seconds INTEGER NOT NULL,
              next_check_at TEXT NOT NULL,
              operator_ref TEXT NOT NULL,
              lifecycle_state TEXT NOT NULL,
              priority INTEGER NOT NULL DEFAULT 100,
              retry_after TEXT,
              last_result_class TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(source_id) REFERENCES legal_sources(source_id),
              UNIQUE(source_id, lifecycle_state)
            );

            CREATE TABLE IF NOT EXISTS legal_monitor_policies (
              policy_version_ref TEXT PRIMARY KEY,
              source_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              source_domain TEXT NOT NULL,
              source_ref TEXT NOT NULL,
              cadence TEXT NOT NULL,
              interval_seconds INTEGER NOT NULL,
              schedule_anchor_utc TEXT NOT NULL,
              priority INTEGER NOT NULL,
              monitor_policy_version TEXT NOT NULL,
              official_domain_allowlist_json TEXT NOT NULL,
              redirect_policy TEXT NOT NULL,
              max_artifact_bytes INTEGER NOT NULL,
              normalization_profile TEXT NOT NULL,
              comparison_profile TEXT NOT NULL,
              retry_policy_json TEXT NOT NULL,
              governance_state TEXT NOT NULL,
              effective_from TEXT NOT NULL,
              effective_to TEXT,
              supersedes_policy_version_ref TEXT,
              created_by TEXT NOT NULL,
              created_at TEXT NOT NULL,
              policy_hash TEXT NOT NULL,
              policy_payload_json TEXT NOT NULL,
              FOREIGN KEY(source_id) REFERENCES legal_sources(source_id),
              FOREIGN KEY(supersedes_policy_version_ref) REFERENCES legal_monitor_policies(policy_version_ref),
              UNIQUE(organization_id, source_ref, monitor_policy_version)
            );

            CREATE TABLE IF NOT EXISTS legal_monitor_policy_events (
              event_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              policy_version_ref TEXT NOT NULL,
              event_type TEXT NOT NULL,
              actor_ref TEXT NOT NULL,
              reason TEXT,
              prior_policy_version_ref TEXT,
              event_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(policy_version_ref) REFERENCES legal_monitor_policies(policy_version_ref)
            );

            CREATE TABLE IF NOT EXISTS legal_scan_runs (
              scan_run_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              monitor_policy_version TEXT NOT NULL,
              scheduled_window TEXT NOT NULL,
              trigger TEXT NOT NULL,
              status TEXT NOT NULL,
              started_at TEXT NOT NULL,
              completed_at TEXT,
              aggregate_counts_json TEXT NOT NULL DEFAULT '{}',
              UNIQUE(organization_id, monitor_policy_version, scheduled_window, trigger)
            );

            CREATE TABLE IF NOT EXISTS legal_scan_items (
              scan_item_id TEXT PRIMARY KEY,
              scan_run_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              source_id TEXT NOT NULL,
              monitor_policy_version TEXT NOT NULL,
              scheduled_window TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              lease_owner TEXT,
              lease_expires_at TEXT,
              attempt_count INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL,
              result_class TEXT,
              error_code TEXT,
              error_detail TEXT,
              observed_version_id TEXT,
              refresh_check_id TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(scan_run_id) REFERENCES legal_scan_runs(scan_run_id),
              FOREIGN KEY(source_id) REFERENCES legal_sources(source_id),
              FOREIGN KEY(observed_version_id) REFERENCES legal_source_versions(version_id),
              FOREIGN KEY(refresh_check_id) REFERENCES source_refresh_checks(refresh_check_id),
              UNIQUE(organization_id, idempotency_key)
            );

            CREATE TABLE IF NOT EXISTS source_claim_links (
              link_id TEXT PRIMARY KEY,
              version_id TEXT NOT NULL,
              section_id TEXT,
              claim_ref TEXT NOT NULL,
              claim_kind TEXT NOT NULL,
              source_span_json TEXT NOT NULL,
              link_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(version_id) REFERENCES legal_source_versions(version_id),
              FOREIGN KEY(section_id) REFERENCES legal_source_sections(section_id)
            );

            CREATE TABLE IF NOT EXISTS legal_semantic_candidates (
              candidate_id TEXT PRIMARY KEY,
              version_id TEXT NOT NULL,
              section_id TEXT,
              candidate_type TEXT NOT NULL,
              candidate_json TEXT NOT NULL,
              source_span_json TEXT NOT NULL,
              extraction_method TEXT NOT NULL,
              review_state TEXT NOT NULL,
              candidate_hash TEXT NOT NULL,
              rejection_reason TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(version_id) REFERENCES legal_source_versions(version_id),
              FOREIGN KEY(section_id) REFERENCES legal_source_sections(section_id)
            );

            CREATE TABLE IF NOT EXISTS source_dependency_links (
              dependency_link_id TEXT PRIMARY KEY,
              source_ref TEXT NOT NULL,
              dependency_ref TEXT NOT NULL,
              dependency_type TEXT NOT NULL,
              dependency_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              UNIQUE(source_ref, dependency_ref, dependency_type)
            );

            CREATE TABLE IF NOT EXISTS legal_certification_events (
              certification_event_id TEXT PRIMARY KEY,
              object_ref TEXT NOT NULL,
              object_type TEXT NOT NULL,
              decision TEXT NOT NULL,
              reviewer_ref TEXT NOT NULL,
              reviewer_role TEXT NOT NULL,
              evidence_hash TEXT NOT NULL,
              event_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS legal_source_registry_snapshots (
              snapshot_id TEXT PRIMARY KEY,
              source_version_ids_json TEXT NOT NULL,
              snapshot_hash TEXT NOT NULL,
              certification_event_ids_json TEXT NOT NULL,
              lifecycle_state TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS legal_knowledge_context_versions (
              knowledge_context_version_id TEXT PRIMARY KEY,
              source_registry_snapshot_id TEXT NOT NULL,
              ontology_version_ref TEXT,
              claim_version_refs_json TEXT NOT NULL DEFAULT '[]',
              content_hash TEXT NOT NULL,
              lifecycle_state TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(source_registry_snapshot_id) REFERENCES legal_source_registry_snapshots(snapshot_id)
            );

            CREATE TABLE IF NOT EXISTS legal_authority_bundle_versions (
              authority_bundle_version_id TEXT PRIMARY KEY,
              source_registry_snapshot_id TEXT NOT NULL,
              knowledge_context_version_id TEXT NOT NULL,
              bundle_hash TEXT NOT NULL,
              validation_result_ref TEXT NOT NULL,
              replay_dry_run_hash TEXT NOT NULL,
              lifecycle_state TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(source_registry_snapshot_id) REFERENCES legal_source_registry_snapshots(snapshot_id),
              FOREIGN KEY(knowledge_context_version_id) REFERENCES legal_knowledge_context_versions(knowledge_context_version_id)
            );

            CREATE TABLE IF NOT EXISTS legal_issue_model_versions (
              issue_model_version_id TEXT PRIMARY KEY,
              issue_model_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              ontology_version_ref TEXT NOT NULL,
              resolution_rules_version_ref TEXT NOT NULL,
              model_json TEXT NOT NULL,
              model_hash TEXT NOT NULL,
              lifecycle_state TEXT NOT NULL,
              effective_from TEXT NOT NULL,
              effective_to TEXT,
              created_by TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(issue_model_id, organization_id, model_hash)
            );

            CREATE TABLE IF NOT EXISTS legal_issue_model_events (
              event_id TEXT PRIMARY KEY,
              issue_model_version_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              actor_ref TEXT NOT NULL,
              actor_role TEXT NOT NULL,
              from_state TEXT,
              to_state TEXT NOT NULL,
              evidence_hash TEXT NOT NULL,
              event_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(issue_model_version_id) REFERENCES legal_issue_model_versions(issue_model_version_id)
            );

            CREATE TABLE IF NOT EXISTS legal_runtime_pin_artifacts (
              artifact_ref TEXT PRIMARY KEY,
              artifact_type TEXT NOT NULL,
              artifact_hash TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              lifecycle_state TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS legal_execution_manifests (
              manifest_hash TEXT PRIMARY KEY,
              manifest_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS graph_nodes (
              node_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              node_type TEXT NOT NULL,
              canonical_key TEXT NOT NULL,
              display_name TEXT,
              properties_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(organization_id, node_type, canonical_key)
            );

            CREATE TABLE IF NOT EXISTS graph_assertions (
              assertion_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              subject_id TEXT NOT NULL,
              predicate TEXT NOT NULL,
              object_id TEXT NOT NULL,
              assertion_class TEXT NOT NULL,
              semantic_tier TEXT,
              authority_role TEXT NOT NULL,
              governance_state TEXT NOT NULL,
              valid_from TEXT,
              valid_to TEXT,
              source_ref TEXT NOT NULL,
              source_anchor TEXT,
              extraction_method TEXT NOT NULL,
              extractor_version TEXT NOT NULL,
              confidence REAL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(subject_id) REFERENCES graph_nodes(node_id),
              FOREIGN KEY(object_id) REFERENCES graph_nodes(node_id)
            );

            CREATE TABLE IF NOT EXISTS graph_assertion_events (
              event_id TEXT PRIMARY KEY,
              assertion_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              actor_ref TEXT NOT NULL,
              event_payload_json TEXT NOT NULL,
              prior_event_hash TEXT,
              event_hash TEXT NOT NULL,
              occurred_at TEXT NOT NULL,
              FOREIGN KEY(assertion_id) REFERENCES graph_assertions(assertion_id)
            );

            CREATE TABLE IF NOT EXISTS graph_snapshots (
              graph_version_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              ontology_version TEXT NOT NULL,
              resolver_version TEXT NOT NULL,
              active_assertion_set_hash TEXT NOT NULL,
              snapshot_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS graph_snapshot_assertions (
              graph_version_id TEXT NOT NULL,
              assertion_id TEXT NOT NULL,
              ordinal INTEGER NOT NULL,
              PRIMARY KEY(graph_version_id, assertion_id),
              FOREIGN KEY(graph_version_id) REFERENCES graph_snapshots(graph_version_id),
              FOREIGN KEY(assertion_id) REFERENCES graph_assertions(assertion_id)
            );
            CREATE INDEX IF NOT EXISTS idx_graph_snapshot_assertions_assertion
              ON graph_snapshot_assertions(assertion_id, graph_version_id);

            CREATE TABLE IF NOT EXISTS graph_ingestion_runs (
              ingestion_run_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              source_ref TEXT NOT NULL,
              trigger_type TEXT NOT NULL,
              requested_by TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              requested_fingerprint TEXT NOT NULL,
              observed_source_hash TEXT,
              prior_successful_run_ref TEXT,
              status TEXT NOT NULL,
              lease_owner TEXT,
              lease_expires_at TEXT,
              attempt_count INTEGER NOT NULL DEFAULT 0,
              max_attempts INTEGER NOT NULL DEFAULT 3,
              next_retry_at TEXT,
              error_code TEXT,
              error_detail_ref TEXT,
              resulting_graph_snapshot_ref TEXT,
              resulting_graph_snapshot_hash TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              completed_at TEXT,
              UNIQUE(organization_id, source_ref, idempotency_key),
              FOREIGN KEY(prior_successful_run_ref) REFERENCES graph_ingestion_runs(ingestion_run_id),
              FOREIGN KEY(resulting_graph_snapshot_ref) REFERENCES graph_snapshots(graph_version_id)
            );

            CREATE TABLE IF NOT EXISTS graph_source_index_manifests (
              organization_id TEXT NOT NULL,
              source_ref TEXT NOT NULL,
              workspace_id TEXT,
              index_fingerprint TEXT NOT NULL,
              source_hash TEXT NOT NULL,
              normalized_structure_hash TEXT NOT NULL,
              extractor_version TEXT NOT NULL,
              structure_schema_version TEXT NOT NULL,
              ontology_compatibility_version TEXT NOT NULL,
              graph_projection_version TEXT NOT NULL,
              last_successful_run_ref TEXT NOT NULL,
              graph_snapshot_ref TEXT NOT NULL,
              graph_snapshot_hash TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(organization_id, source_ref),
              FOREIGN KEY(last_successful_run_ref) REFERENCES graph_ingestion_runs(ingestion_run_id),
              FOREIGN KEY(graph_snapshot_ref) REFERENCES graph_snapshots(graph_version_id)
            );

            CREATE TABLE IF NOT EXISTS graph_index_freshness (
              organization_id TEXT NOT NULL,
              source_ref TEXT NOT NULL,
              workspace_id TEXT,
              status TEXT NOT NULL,
              indexed_at TEXT,
              source_fingerprint TEXT,
              last_successful_index_ref TEXT,
              graph_snapshot_ref TEXT,
              graph_snapshot_hash TEXT,
              error_code TEXT,
              error_detail_ref TEXT,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(organization_id, source_ref)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS graph_node_search
              USING fts5(node_id UNINDEXED, organization_id UNINDEXED, content);

            CREATE INDEX IF NOT EXISTS idx_graph_assertions_subject
              ON graph_assertions(organization_id, subject_id, predicate, governance_state);
            CREATE INDEX IF NOT EXISTS idx_graph_assertions_subject_ordered
              ON graph_assertions(organization_id, subject_id, predicate, governance_state, object_id);
            CREATE INDEX IF NOT EXISTS idx_graph_assertions_object
              ON graph_assertions(organization_id, object_id, predicate, governance_state);
            CREATE INDEX IF NOT EXISTS idx_graph_assertions_class_state
              ON graph_assertions(organization_id, assertion_class, governance_state);
            CREATE INDEX IF NOT EXISTS idx_graph_assertions_effective
              ON graph_assertions(organization_id, valid_from, valid_to);
            CREATE INDEX IF NOT EXISTS idx_graph_assertion_events_stream
              ON graph_assertion_events(assertion_id, occurred_at);

            CREATE TABLE IF NOT EXISTS legal_activation_events (
              activation_event_id TEXT PRIMARY KEY,
              source_registry_snapshot_id TEXT NOT NULL,
              knowledge_context_version_id TEXT NOT NULL,
              authority_bundle_version_id TEXT NOT NULL,
              activation_performed INTEGER NOT NULL CHECK(activation_performed IN (0, 1)),
              approver_refs_json TEXT NOT NULL,
              content_hashes_json TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(source_registry_snapshot_id) REFERENCES legal_source_registry_snapshots(snapshot_id),
              FOREIGN KEY(knowledge_context_version_id) REFERENCES legal_knowledge_context_versions(knowledge_context_version_id),
              FOREIGN KEY(authority_bundle_version_id) REFERENCES legal_authority_bundle_versions(authority_bundle_version_id),
              UNIQUE(idempotency_key)
            );

            CREATE TABLE IF NOT EXISTS legal_acceptance_evidence_exports (
              acceptance_export_id TEXT PRIMARY KEY,
              test_run_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              evidence_hash TEXT NOT NULL,
              record_hash TEXT NOT NULL,
              evidence_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(test_run_id, organization_id, evidence_hash)
            );

            CREATE TABLE IF NOT EXISTS legal_regulatory_notifications (
              alert_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              source_id TEXT,
              refresh_check_id TEXT,
              candidate_id TEXT,
              impact_report_ref TEXT,
              event_type TEXT NOT NULL,
              severity TEXT NOT NULL,
              recipient_role TEXT,
              recipient_user_id TEXT,
              audit_event_ref TEXT,
              correlation_id TEXT,
              causation_id TEXT,
              notification_json TEXT NOT NULL,
              lifecycle_state TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(source_id) REFERENCES legal_sources(source_id),
              FOREIGN KEY(refresh_check_id) REFERENCES source_refresh_checks(refresh_check_id),
              FOREIGN KEY(candidate_id) REFERENCES legal_semantic_candidates(candidate_id)
            );

            CREATE TABLE IF NOT EXISTS legal_regulatory_notification_acknowledgements (
              acknowledgement_id TEXT PRIMARY KEY,
              alert_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              acknowledged_by TEXT NOT NULL,
              acknowledgement_json TEXT NOT NULL,
              audit_event_ref TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(alert_id) REFERENCES legal_regulatory_notifications(alert_id),
              UNIQUE(alert_id, acknowledged_by)
            );

            CREATE TABLE IF NOT EXISTS legal_audit_events (
              audit_event_id TEXT PRIMARY KEY,
              event_family TEXT NOT NULL,
              object_ref TEXT NOT NULL,
              object_type TEXT NOT NULL,
              lifecycle_state TEXT,
              attempted_endpoint TEXT,
              actor_ref TEXT,
              organization_id TEXT,
              reason_code TEXT,
              route_policy_id TEXT,
              correlation_id TEXT,
              event_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_legal_sources_state
              ON legal_sources(organization_id, jurisdiction, authority_tier, governance_state);
            CREATE INDEX IF NOT EXISTS idx_legal_versions_source_state
              ON legal_source_versions(source_id, lifecycle_state, effective_from);
            CREATE INDEX IF NOT EXISTS idx_legal_sections_version_locator
              ON legal_source_sections(version_id, stable_locator);
            CREATE INDEX IF NOT EXISTS idx_legal_edges_from
              ON legal_source_version_edges(from_version_id, edge_type);
            CREATE INDEX IF NOT EXISTS idx_legal_edges_to
              ON legal_source_version_edges(to_version_id, edge_type);
            CREATE INDEX IF NOT EXISTS idx_source_ingestion_source
              ON source_ingestion_events(source_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_source_refresh_source
              ON source_refresh_checks(source_id, checked_at);
            CREATE INDEX IF NOT EXISTS idx_legal_refresh_schedules_due
              ON legal_source_refresh_schedules(lifecycle_state, next_check_at);
            CREATE INDEX IF NOT EXISTS idx_legal_monitor_policies_resolution
              ON legal_monitor_policies(organization_id, source_ref, workspace_id, governance_state, effective_from, effective_to);
            CREATE INDEX IF NOT EXISTS idx_legal_scan_runs_org_status
              ON legal_scan_runs(organization_id, status, started_at);
            CREATE INDEX IF NOT EXISTS idx_legal_scan_items_claim
              ON legal_scan_items(status, lease_expires_at, source_id);
            CREATE INDEX IF NOT EXISTS idx_source_claim_links_claim
              ON source_claim_links(claim_ref);
            CREATE INDEX IF NOT EXISTS idx_legal_semantic_candidates_version
              ON legal_semantic_candidates(version_id, review_state, candidate_type);
            CREATE INDEX IF NOT EXISTS idx_source_dependency_links_source
              ON source_dependency_links(source_ref);
            CREATE INDEX IF NOT EXISTS idx_legal_certification_object
              ON legal_certification_events(object_ref, object_type, created_at);
            CREATE INDEX IF NOT EXISTS idx_legal_activation_bundle
              ON legal_activation_events(authority_bundle_version_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_legal_acceptance_exports_org_run
              ON legal_acceptance_evidence_exports(organization_id, test_run_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_legal_regulatory_notifications_org_state
              ON legal_regulatory_notifications(organization_id, lifecycle_state, created_at);
            CREATE INDEX IF NOT EXISTS idx_legal_regulatory_notifications_source
              ON legal_regulatory_notifications(source_id, event_type, created_at);
            CREATE INDEX IF NOT EXISTS idx_legal_regulatory_acks_alert
              ON legal_regulatory_notification_acknowledgements(alert_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_legal_audit_reason
              ON legal_audit_events(reason_code, correlation_id);
            """
        )
        _migrate_source_identity_refs_to_tenant_scope(conn)
        ensure_graph_ingestion_schema(conn)

        legal_source_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(legal_sources)").fetchall()
        }
        if "organization_id" not in legal_source_columns:
            conn.execute(
                "ALTER TABLE legal_sources ADD COLUMN organization_id TEXT NOT NULL DEFAULT 'global'"
            )
        if "workspace_id" not in legal_source_columns:
            conn.execute("ALTER TABLE legal_sources ADD COLUMN workspace_id TEXT")
        conn.execute("DROP INDEX IF EXISTS idx_legal_sources_identity")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_legal_sources_identity
              ON legal_sources(organization_id, canonical_title, issuer, jurisdiction, COALESCE(document_number, ''))
            """
        )

        legal_refresh_schedule_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(legal_source_refresh_schedules)").fetchall()
        }
        if "priority" not in legal_refresh_schedule_columns:
            conn.execute(
                "ALTER TABLE legal_source_refresh_schedules ADD COLUMN priority INTEGER NOT NULL DEFAULT 100"
            )
        if "retry_after" not in legal_refresh_schedule_columns:
            conn.execute("ALTER TABLE legal_source_refresh_schedules ADD COLUMN retry_after TEXT")
        if "last_result_class" not in legal_refresh_schedule_columns:
            conn.execute("ALTER TABLE legal_source_refresh_schedules ADD COLUMN last_result_class TEXT")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_legal_refresh_schedules_priority_due
              ON legal_source_refresh_schedules(lifecycle_state, priority, next_check_at)
            """
        )

        # Backward-compatible schema evolution for policy_candidates semantic tiering.
        policy_candidate_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(policy_candidates)").fetchall()
        }
        if "semantic_tier" not in policy_candidate_columns:
            conn.execute(
                """
                ALTER TABLE policy_candidates
                ADD COLUMN semantic_tier TEXT NOT NULL DEFAULT 'T5'
                CHECK(semantic_tier IN ('T1', 'T2', 'T3', 'T4', 'T5', 'T6'))
                """
            )
            conn.execute(
                """
                UPDATE policy_candidates
                SET semantic_tier = CASE category
                    WHEN 'coa_version' THEN 'T3'
                    WHEN 'ontology_entry' THEN 'T1'
                    ELSE 'T5'
                END
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_candidates_semantic_tier ON policy_candidates(semantic_tier)"
            )
        if "scoped_owner" not in policy_candidate_columns:
            conn.execute("ALTER TABLE policy_candidates ADD COLUMN scoped_owner TEXT")
        if "rollback_window_hours" not in policy_candidate_columns:
            conn.execute("ALTER TABLE policy_candidates ADD COLUMN rollback_window_hours INTEGER")
        if "rollback_strategy" not in policy_candidate_columns:
            conn.execute("ALTER TABLE policy_candidates ADD COLUMN rollback_strategy TEXT")
        if "review_status" not in policy_candidate_columns:
            conn.execute("ALTER TABLE policy_candidates ADD COLUMN review_status TEXT")
        if "jurisdiction_reason" not in policy_candidate_columns:
            conn.execute("ALTER TABLE policy_candidates ADD COLUMN jurisdiction_reason TEXT")
        if "imported_legacy" not in policy_candidate_columns:
            conn.execute(
                "ALTER TABLE policy_candidates ADD COLUMN imported_legacy INTEGER NOT NULL DEFAULT 0"
            )
        # Older installations may predate the governance metadata columns.
        # Add them before the category-constraint rebuild so existing rows are
        # copied into the complete current shape.
        refreshed_policy_candidate_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(policy_candidates)").fetchall()
        }
        for column, definition in {
            "replay_result": "TEXT",
            "approved_by": "TEXT",
            "approved_at": "TEXT",
            "notes": "TEXT",
            "review_status": "TEXT",
            "jurisdiction_reason": "TEXT",
        }.items():
            if column not in refreshed_policy_candidate_columns:
                conn.execute(f"ALTER TABLE policy_candidates ADD COLUMN {column} {definition}")
        _migrate_policy_candidate_categories(conn)
        t6_candidate_columns = {
          row["name"]
          for row in conn.execute("PRAGMA table_info(t6_materialization_candidates)").fetchall()
        }
        if "source_message_id" not in t6_candidate_columns:
          conn.execute("ALTER TABLE t6_materialization_candidates ADD COLUMN source_message_id TEXT")
        if "fact_maturity_stage" not in t6_candidate_columns:
          conn.execute(
            "ALTER TABLE t6_materialization_candidates ADD COLUMN fact_maturity_stage TEXT NOT NULL DEFAULT 'C0_CAPTURED'"
          )
        if "projection_status" not in t6_candidate_columns:
          conn.execute(
            "ALTER TABLE t6_materialization_candidates ADD COLUMN projection_status TEXT NOT NULL DEFAULT 'PROJECTION_PENDING'"
          )
        conn.execute(
          "CREATE INDEX IF NOT EXISTS idx_t6_candidates_org_stage ON t6_materialization_candidates(org_id, fact_maturity_stage, created_at)"
        )
        conn.execute(
          "CREATE INDEX IF NOT EXISTS idx_t6_candidates_org_projection ON t6_materialization_candidates(org_id, projection_status, created_at)"
        )
        conn.execute(
          "CREATE UNIQUE INDEX IF NOT EXISTS idx_t6_candidates_submit_dedupe ON t6_materialization_candidates(org_id, source_channel, source_message_id, candidate_hash)"
        )
        event_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(events)").fetchall()
        }
        if "stream_id" not in event_columns:
            conn.execute("ALTER TABLE events ADD COLUMN stream_id TEXT")
            conn.execute("UPDATE events SET stream_id = COALESCE(stream_id, 'global')")
        if "content_hash" not in event_columns:
            conn.execute("ALTER TABLE events ADD COLUMN content_hash TEXT")
        if "prior_event_hash" not in event_columns:
            conn.execute("ALTER TABLE events ADD COLUMN prior_event_hash TEXT")
        conn.executescript(
            """
            -- Projection Store: ledger_projections (canonical) and compatibility view journal_entries
            -- ledger_projections stores projected account lines derived from committed rea_claims.
            -- journal_entries is a compatibility/export view for systems expecting debit/credit format.
            CREATE TABLE IF NOT EXISTS ledger_projections (
              ledger_id               TEXT PRIMARY KEY,
              rea_id                  TEXT NOT NULL,
              coa_code                TEXT NOT NULL,
              amount                  REAL NOT NULL,
              currency                TEXT NOT NULL,
              debit                   REAL,
              credit                  REAL,
              projection_status       TEXT NOT NULL DEFAULT 'PENDING'
                CHECK(projection_status IN ('PENDING', 'PROJECTED', 'EXCEPTION', 'RESTATED')),
              projection_context_hash TEXT,
              created_at              TEXT NOT NULL,
              projected_at            TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_projections_rea_id
              ON ledger_projections(rea_id);
            CREATE INDEX IF NOT EXISTS idx_projections_status
              ON ledger_projections(projection_status);
            CREATE INDEX IF NOT EXISTS idx_projections_coa_code
              ON ledger_projections(coa_code);

            CREATE TABLE IF NOT EXISTS coa_projection_bundles (
              bundle_version TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              bundle_id TEXT NOT NULL,
              jurisdiction TEXT NOT NULL,
              accounting_standard TEXT NOT NULL,
              status TEXT NOT NULL
                CHECK(status IN ('CANDIDATE', 'REVIEWED', 'CERTIFIED', 'ACTIVE', 'DEPRECATED')),
              content_yaml TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              source_refs TEXT NOT NULL DEFAULT '[]',
              created_at TEXT NOT NULL,
              activated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS coa_entries (
              coa_entry_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              bundle_version TEXT NOT NULL,
              account_code TEXT NOT NULL,
              account_name TEXT NOT NULL,
              account_class TEXT NOT NULL
                CHECK(account_class IN ('asset', 'liability', 'equity', 'revenue', 'expense')),
              normal_balance TEXT NOT NULL
                CHECK(normal_balance IN ('debit', 'credit')),
              parent_code TEXT,
              auxiliary_dimensions TEXT NOT NULL DEFAULT '[]',
              authority_domain TEXT NOT NULL DEFAULT 'accounting',
              tag_authority_level TEXT NOT NULL DEFAULT 'accounting',
              semantic_tier TEXT NOT NULL DEFAULT 'T3',
              source_refs TEXT NOT NULL DEFAULT '[]',
              status TEXT NOT NULL CHECK(status IN ('ACTIVE', 'DEPRECATED')),
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE (org_id, bundle_version, account_code)
            );

            CREATE TABLE IF NOT EXISTS coa_mapping_rules (
              rule_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              projection_bundle_version TEXT NOT NULL,
              coa_bundle_version TEXT NOT NULL,
              rule_yaml TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL
                CHECK(status IN ('CANDIDATE', 'REVIEWED', 'CERTIFIED', 'ACTIVE', 'DEPRECATED')),
              source_refs TEXT NOT NULL DEFAULT '[]',
              created_at TEXT NOT NULL,
              activated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS kgl_versions (
              knowledge_version_id TEXT PRIMARY KEY,
              active_artifacts_json TEXT NOT NULL DEFAULT '[]',
              source_hashes_json TEXT NOT NULL DEFAULT '[]',
              authority_source_manifest_json TEXT NOT NULL DEFAULT '[]',
              extracted_claims_json TEXT NOT NULL DEFAULT '[]',
              extracted_constraints_json TEXT NOT NULL DEFAULT '[]',
              extracted_projection_rules_json TEXT NOT NULL DEFAULT '[]',
              effective_from TEXT NOT NULL,
              effective_to TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_artifacts (
              artifact_id TEXT PRIMARY KEY,
              source_type TEXT NOT NULL
                CHECK(source_type IN ('official_regulation', 'official_taxonomy', 'field_guide', 'firm_guidance', 'internal_policy', 'runtime_observation')),
              authority_domain TEXT NOT NULL
                CHECK(authority_domain IN ('accounting', 'tax', 'management', 'compliance', 'internal_control')),
              tag_authority_level TEXT NOT NULL
                CHECK(tag_authority_level IN ('regulatory', 'accounting', 'management', 'operational', 'descriptive')),
              semantic_tier TEXT NOT NULL
                CHECK(semantic_tier IN ('T2', 'T3', 'T4', 'T5', 'T6')),
              authority_origin TEXT NOT NULL DEFAULT 'runtime_observation'
                CHECK(authority_origin IN ('official_source', 'interpretive_source', 'internal_governance', 'runtime_observation', 'ontology_governance')),
              source_ref TEXT NOT NULL,
              source_hash TEXT NOT NULL,
              source_uri TEXT,
              jurisdiction TEXT,
              effective_from TEXT,
              effective_to TEXT,
              source_version TEXT,
              source_anchors_json TEXT NOT NULL DEFAULT '[]',
              extraction_method TEXT,
              curator TEXT,
              ambiguities_json TEXT NOT NULL DEFAULT '[]',
              extracted_claims_json TEXT NOT NULL DEFAULT '[]',
              extracted_constraints_json TEXT NOT NULL DEFAULT '[]',
              extracted_projection_rules_json TEXT NOT NULL DEFAULT '[]',
              ingestion_status TEXT NOT NULL
                CHECK(ingestion_status IN ('PROPOSED', 'VALIDATED', 'APPROVED', 'ACTIVE', 'REJECTED', 'DEPRECATED')),
              content_hash TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              approved_by TEXT
            );

            CREATE TABLE IF NOT EXISTS kgl_approval_records (
              record_id TEXT PRIMARY KEY,
              artifact_id TEXT NOT NULL,
              from_status TEXT NOT NULL,
              to_status TEXT NOT NULL,
              actor TEXT NOT NULL,
              notes TEXT,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS kgl_conflict_records (
              conflict_id TEXT PRIMARY KEY,
              artifact_id TEXT NOT NULL,
              higher_tier_artifact_ids_json TEXT NOT NULL,
              same_domain_artifact_ids_json TEXT NOT NULL,
              decision TEXT NOT NULL,
              review_notes TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              created_by TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_status
              ON knowledge_artifacts(ingestion_status);
            CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_domain
              ON knowledge_artifacts(authority_domain);
            CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_source_hash
              ON knowledge_artifacts(source_hash);
            CREATE INDEX IF NOT EXISTS idx_kgl_approval_artifact
              ON kgl_approval_records(artifact_id);
            CREATE INDEX IF NOT EXISTS idx_kgl_conflict_artifact
              ON kgl_conflict_records(artifact_id, created_at);

            CREATE TABLE IF NOT EXISTS knowledge_source (
              knowledge_source_id TEXT PRIMARY KEY,
              source_kind TEXT NOT NULL
                CHECK(source_kind IN ('legal', 'wiki', 'research', 'dataset', 'controlled_document')),
              semantic_purpose TEXT,
              semantic_tier TEXT NOT NULL
                CHECK(semantic_tier IN ('T1', 'T2', 'T3', 'T4', 'T5', 'T6')),
              authority_role TEXT NOT NULL
                CHECK(authority_role IN ('binding', 'interpretive', 'empirical', 'advisory', 'descriptive')),
              source_lifecycle_status TEXT NOT NULL
                CHECK(source_lifecycle_status IN ('proposed', 'validated', 'approved', 'rejected', 'deprecated')),
              source_ref TEXT NOT NULL,
              source_version TEXT,
              source_hash TEXT NOT NULL,
              effective_from TEXT,
              effective_to TEXT,
              provenance_manifest_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_knowledge_source_kind_purpose
              ON knowledge_source(source_kind, semantic_purpose);

            CREATE TABLE IF NOT EXISTS dataset_asset_version (
              dataset_asset_version_id TEXT PRIMARY KEY,
              dataset_type TEXT NOT NULL CHECK(dataset_type IN ('DEMO', 'REAL')),
              dataset_key TEXT,
              dataset_version_id TEXT,
              dataset_usage_role TEXT NOT NULL
                CHECK(dataset_usage_role IN ('primary_analytics', 'benchmark', 'reference', 'sandbox')),
              authority_level TEXT NOT NULL
                CHECK(authority_level IN ('binding', 'governed_default', 'advisory', 'sandbox')),
              user_context_control_level TEXT NOT NULL
                CHECK(user_context_control_level IN ('none', 'query_context', 'prompt_context', 'full_optional_context')),
              schema_contract_hash TEXT NOT NULL,
              lineage_manifest_hash TEXT NOT NULL,
              readiness_status TEXT NOT NULL
                CHECK(readiness_status IN ('ready', 'stale', 'blocked', 'deprecated')),
              quality_evidence_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_source_activation (
              activation_id TEXT PRIMARY KEY,
              knowledge_source_id TEXT NOT NULL,
              dataset_asset_version_id TEXT,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              activation_scope TEXT NOT NULL CHECK(activation_scope IN ('organization', 'workspace')),
              scope_key TEXT NOT NULL,
              activation_status TEXT NOT NULL
                CHECK(activation_status IN ('active', 'inactive', 'blocked', 'superseded', 'deprecated')),
              workspace_override_policy TEXT NOT NULL
                CHECK(workspace_override_policy IN ('none', 'retrieval_only', 'prompt_context_only', 'query_context_only', 'optional_context', 'governed_replacement')),
              user_context_control_level TEXT NOT NULL
                CHECK(user_context_control_level IN ('none', 'query_context', 'prompt_context', 'full_optional_context')),
              activation_event_head_hash TEXT NOT NULL,
              authority_source_manifest_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(knowledge_source_id) REFERENCES knowledge_source(knowledge_source_id),
              FOREIGN KEY(dataset_asset_version_id) REFERENCES dataset_asset_version(dataset_asset_version_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ksa_active_source_scope
              ON knowledge_source_activation(organization_id, scope_key, knowledge_source_id)
              WHERE activation_status = 'active' AND dataset_asset_version_id IS NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ksa_active_dataset_role_scope
              ON knowledge_source_activation(organization_id, scope_key, dataset_asset_version_id)
              WHERE activation_status = 'active' AND dataset_asset_version_id IS NOT NULL;

            CREATE TRIGGER IF NOT EXISTS trg_ksa_active_dataset_role_insert
            BEFORE INSERT ON knowledge_source_activation
            WHEN NEW.activation_status = 'active'
              AND NEW.dataset_asset_version_id IS NOT NULL
              AND EXISTS (
                SELECT 1
                FROM knowledge_source_activation existing
                JOIN dataset_asset_version existing_asset
                  ON existing_asset.dataset_asset_version_id = existing.dataset_asset_version_id
                JOIN dataset_asset_version new_asset
                  ON new_asset.dataset_asset_version_id = NEW.dataset_asset_version_id
                WHERE existing.organization_id = NEW.organization_id
                  AND existing.scope_key = NEW.scope_key
                  AND existing.activation_status = 'active'
                  AND existing.dataset_asset_version_id IS NOT NULL
                  AND existing_asset.dataset_usage_role = new_asset.dataset_usage_role
              )
            BEGIN
              SELECT RAISE(ABORT, 'duplicate active dataset binding for dataset_usage_role');
            END;

            CREATE TRIGGER IF NOT EXISTS trg_ksa_active_dataset_role_update
            BEFORE UPDATE OF organization_id, scope_key, activation_status, dataset_asset_version_id
            ON knowledge_source_activation
            WHEN NEW.activation_status = 'active'
              AND NEW.dataset_asset_version_id IS NOT NULL
              AND EXISTS (
                SELECT 1
                FROM knowledge_source_activation existing
                JOIN dataset_asset_version existing_asset
                  ON existing_asset.dataset_asset_version_id = existing.dataset_asset_version_id
                JOIN dataset_asset_version new_asset
                  ON new_asset.dataset_asset_version_id = NEW.dataset_asset_version_id
                WHERE existing.activation_id != NEW.activation_id
                  AND existing.organization_id = NEW.organization_id
                  AND existing.scope_key = NEW.scope_key
                  AND existing.activation_status = 'active'
                  AND existing.dataset_asset_version_id IS NOT NULL
                  AND existing_asset.dataset_usage_role = new_asset.dataset_usage_role
              )
            BEGIN
              SELECT RAISE(ABORT, 'duplicate active dataset binding for dataset_usage_role');
            END;

            CREATE TABLE IF NOT EXISTS knowledge_context_inclusion_preference (
              preference_id TEXT PRIMARY KEY,
              activation_id TEXT NOT NULL,
              principal_id TEXT,
              role_id TEXT,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              preference_scope TEXT NOT NULL
                CHECK(preference_scope IN ('principal', 'role', 'workspace_default')),
              retrieval_enabled INTEGER NOT NULL CHECK(retrieval_enabled IN (0, 1)),
              prompt_context_enabled INTEGER NOT NULL CHECK(prompt_context_enabled IN (0, 1)),
              query_context_enabled INTEGER NOT NULL CHECK(query_context_enabled IN (0, 1)),
              preference_event_head_hash TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(activation_id) REFERENCES knowledge_source_activation(activation_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_kcip_subject_activation
              ON knowledge_context_inclusion_preference(
                activation_id,
                preference_scope,
                COALESCE(principal_id, ''),
                COALESCE(role_id, ''),
                organization_id,
                COALESCE(workspace_id, '')
              );

            CREATE TABLE IF NOT EXISTS knowledge_activation_set_snapshot (
              activation_set_snapshot_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              principal_entitlement_ref TEXT NOT NULL,
              activation_resolver_policy_version TEXT NOT NULL,
              effective_at TEXT NOT NULL,
              included_activation_ids_json TEXT NOT NULL,
              excluded_activation_evidence_json TEXT NOT NULL DEFAULT '[]',
              resolved_activation_set_hash TEXT NOT NULL,
              prior_snapshot_hash TEXT,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_activation_migration_state (
              organization_id TEXT PRIMARY KEY,
              migration_status TEXT NOT NULL
                CHECK(migration_status IN ('pending', 'activation_written', 'complete', 'failed')),
              source_legacy_fingerprint TEXT NOT NULL,
              activation_id TEXT,
              migration_event_ref TEXT,
              started_at TEXT NOT NULL,
              completed_at TEXT,
              last_error_code TEXT,
              content_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_source_activation_events (
              activation_event_id TEXT PRIMARY KEY,
              knowledge_source_id TEXT NOT NULL,
              dataset_asset_version_id TEXT,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              prior_activation_event_hash TEXT,
              decision TEXT NOT NULL
                CHECK(decision IN ('ACTIVATE_AUTHORITY', 'DEACTIVATE_AUTHORITY', 'SUPERSEDE', 'BLOCK', 'UNBLOCK', 'DEPRECATE', 'MIGRATE')),
              actor TEXT NOT NULL,
              actor_authority TEXT NOT NULL,
              reason TEXT NOT NULL,
              artifact_version_hash TEXT NOT NULL,
              occurred_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ksae_source
              ON knowledge_source_activation_events(knowledge_source_id, occurred_at);

            CREATE TABLE IF NOT EXISTS knowledge_context_inclusion_preference_events (
              preference_event_id TEXT PRIMARY KEY,
              preference_id TEXT NOT NULL,
              activation_id TEXT NOT NULL,
              principal_id TEXT,
              role_id TEXT,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              prior_preference_event_hash TEXT,
              decision TEXT NOT NULL
                CHECK(decision IN ('ENABLE_RETRIEVAL', 'DISABLE_RETRIEVAL', 'ENABLE_PROMPT_CONTEXT', 'DISABLE_PROMPT_CONTEXT', 'ENABLE_QUERY_CONTEXT', 'DISABLE_QUERY_CONTEXT', 'RESET_TO_DEFAULT')),
              actor TEXT NOT NULL,
              reason TEXT NOT NULL,
              occurred_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kcipe_preference
              ON knowledge_context_inclusion_preference_events(preference_id, occurred_at);

            CREATE TABLE IF NOT EXISTS projection_context_snapshots (
              projection_context_ref TEXT PRIMARY KEY,
              projection_context_hash TEXT NOT NULL,
              org_id TEXT NOT NULL,
              event_id TEXT,
              knowledge_version TEXT,
              tag_ontology_version TEXT,
              projection_bundle_version TEXT,
              constraint_bundle_version TEXT,
              retrieval_query_hash TEXT NOT NULL,
              provider_ref TEXT NOT NULL DEFAULT 'hermes-agent/plugins/memory/holographic',
              provider_version TEXT NOT NULL,
              adapter_version TEXT NOT NULL DEFAULT 'semantier_memory_adapter_v1',
              admissibility_policy_version TEXT,
              rejected_items_summary_json TEXT NOT NULL DEFAULT '[]',
              retrieved_source_refs_json TEXT NOT NULL DEFAULT '[]',
              selected_context_json TEXT NOT NULL DEFAULT '{}',
              excluded_context_refs_json TEXT NOT NULL DEFAULT '[]',
              selection_reason TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              created_by TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_projection_ctx_org
              ON projection_context_snapshots(org_id);
            CREATE INDEX IF NOT EXISTS idx_projection_ctx_hash
              ON projection_context_snapshots(projection_context_hash);

            CREATE TABLE IF NOT EXISTS replay_bindings (
              event_id TEXT PRIMARY KEY,
              ontology_version TEXT,
              contract_version TEXT,
              tag_ontology_version TEXT,
              knowledge_version TEXT,
              projection_bundle_version TEXT,
              constraint_bundle_version TEXT,
              precedence_graph_version TEXT,
              resolution_rules_version TEXT,
              projection_context_ref TEXT NOT NULL,
              projection_context_hash TEXT NOT NULL,
              authority_source_manifest_json TEXT NOT NULL DEFAULT '[]',
              authority_source_manifest_hash TEXT NOT NULL DEFAULT '',
              resolution_trace_hash TEXT NOT NULL,
              effect_hash TEXT NOT NULL,
              event_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_replay_bindings_ctx_ref
              ON replay_bindings(projection_context_ref);

            CREATE TABLE IF NOT EXISTS replay_envelopes (
              replay_binding_id TEXT PRIMARY KEY,
              binding_type TEXT NOT NULL
                CHECK(binding_type IN (
                  'projection_execution.v1',
                  'sensitive_labeling_detection.v1',
                  'sensitive_labeling_report.v1',
                  'investment_radar_candidate.v1'
                )),
              binding_schema_version TEXT NOT NULL,
              subject_type TEXT NOT NULL,
              subject_id TEXT NOT NULL,
              specific_binding_content_hash TEXT NOT NULL,
              envelope_hash TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_replay_envelopes_subject
              ON replay_envelopes(binding_type, subject_type, subject_id);
            CREATE INDEX IF NOT EXISTS idx_replay_envelopes_scope
              ON replay_envelopes(organization_id, workspace_id, created_at);

            CREATE TABLE IF NOT EXISTS sensitive_labeling_detection_bindings (
              replay_binding_id TEXT PRIMARY KEY,
              binding_type TEXT NOT NULL
                CHECK(binding_type = 'sensitive_labeling_detection.v1'),
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              binding_payload_json TEXT NOT NULL,
              binding_payload_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(replay_binding_id) REFERENCES replay_envelopes(replay_binding_id)
            );

            CREATE TABLE IF NOT EXISTS sensitive_labeling_report_bindings (
              replay_binding_id TEXT PRIMARY KEY,
              binding_type TEXT NOT NULL
                CHECK(binding_type = 'sensitive_labeling_report.v1'),
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              binding_payload_json TEXT NOT NULL,
              binding_payload_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(replay_binding_id) REFERENCES replay_envelopes(replay_binding_id)
            );

            CREATE TABLE IF NOT EXISTS investment_radar_candidate_bindings (
              replay_binding_id TEXT PRIMARY KEY,
              binding_type TEXT NOT NULL
                CHECK(binding_type = 'investment_radar_candidate.v1'),
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              binding_payload_json TEXT NOT NULL,
              binding_payload_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(replay_binding_id) REFERENCES replay_envelopes(replay_binding_id)
            );

            CREATE VIEW IF NOT EXISTS rea_events AS
            SELECT
              rea_id AS rea_event_id,
              workflow_id,
              claim_json AS event_json,
              claim_hash AS content_hash,
              created_at
            FROM rea_claims;

            CREATE TABLE IF NOT EXISTS coa_versions (
              coa_version_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              bundle_version TEXT,
              version_label TEXT NOT NULL,
              content_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              activated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS projection_bundles (
              projection_bundle_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              bundle_version TEXT NOT NULL,
              coa_version_id TEXT,
              content_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              activated_at TEXT
            );

            CREATE VIEW IF NOT EXISTS projection_results AS
            SELECT
              ledger_id AS projection_result_id,
              rea_id,
              coa_code,
              amount,
              currency,
              debit,
              credit,
              projection_status AS status,
              projection_context_hash,
              projection_scope,
              projection_bundle_version,
              coa_bundle_version,
              knowledge_context_version,
              constraint_bundle_version,
              tag_ontology_version,
              precedence_graph_version,
              resolution_rules_version,
              created_at,
              projected_at
            FROM ledger_projections;

            CREATE TABLE IF NOT EXISTS projection_exceptions (
              projection_exception_id TEXT PRIMARY KEY,
              rea_id TEXT NOT NULL,
              justification_id TEXT,
              projection_context_ref TEXT,
              projection_context_hash TEXT,
              coa_version TEXT,
              projection_bundle_version TEXT,
              constraint_bundle_version TEXT,
              status TEXT NOT NULL,
              exception_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              resolved_at TEXT
            );

            CREATE TABLE IF NOT EXISTS projection_exception_governance (
              governance_id TEXT PRIMARY KEY,
              projection_exception_id TEXT NOT NULL,
              governance_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              approved_at TEXT,
              activated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS coa_change_proposals (
              coa_change_proposal_id TEXT PRIMARY KEY,
              base_coa_version_id TEXT,
              proposal_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL,
              proposed_by TEXT,
              created_at TEXT NOT NULL,
              approved_at TEXT
            );

            CREATE TABLE IF NOT EXISTS projection_rule_proposals (
              projection_rule_proposal_id TEXT PRIMARY KEY,
              base_projection_bundle_id TEXT,
              proposal_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL,
              proposed_by TEXT,
              created_at TEXT NOT NULL,
              approved_at TEXT
            );

            CREATE TABLE IF NOT EXISTS governed_reprojections (
              governed_reprojection_id TEXT PRIMARY KEY,
              projection_exception_id TEXT NOT NULL,
              governance_id TEXT,
              prior_projection_result_ref TEXT,
              new_projection_result_ref TEXT,
              replay_binding_ref TEXT,
              reprojection_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            -- v7.7 Onboarding and Projection Exception Journey --

            CREATE TABLE IF NOT EXISTS coa_taxonomy_candidates (
              coa_taxonomy_candidate_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              candidate_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL
                CHECK(status IN ('CANDIDATE', 'APPROVED', 'ACTIVE', 'REJECTED')),
              proposed_by TEXT,
              approved_by TEXT,
              created_at TEXT NOT NULL,
              approved_at TEXT,
              activated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS projection_bundle_candidates (
              projection_bundle_candidate_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              candidate_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL
                CHECK(status IN ('CANDIDATE', 'APPROVED', 'ACTIVE', 'REJECTED')),
              proposed_by TEXT,
              approved_by TEXT,
              created_at TEXT NOT NULL,
              approved_at TEXT,
              activated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS constraint_bundle_candidates (
              constraint_bundle_candidate_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              candidate_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL
                CHECK(status IN ('CANDIDATE', 'APPROVED', 'ACTIVE', 'REJECTED')),
              proposed_by TEXT,
              approved_by TEXT,
              created_at TEXT NOT NULL,
              approved_at TEXT,
              activated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS projection_exception_status_events (
              projection_exception_status_event_id TEXT PRIMARY KEY,
              projection_exception_id TEXT NOT NULL,
              status TEXT NOT NULL
                CHECK(status IN (
                  'PROJECTION_EXCEPTION',
                  'RESOLVED_BY_GOVERNED_REPROJECTION',
                  'REQUEST_MORE_EVIDENCE',
                  'ESCALATED',
                  'MARK_NOT_PROJECTABLE'
                )),
              reason_code TEXT,
              event_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_projection_exception_status_events_exception
              ON projection_exception_status_events(projection_exception_id, created_at);

            CREATE TABLE IF NOT EXISTS explainability_surfaces (
              explainability_surface_id TEXT PRIMARY KEY,
              event_id TEXT NOT NULL,
              surface_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_evidence_packages (
              audit_evidence_package_id TEXT PRIMARY KEY,
              subject_ref TEXT NOT NULL,
              package_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS external_audit_exports (
              external_audit_export_id TEXT PRIMARY KEY,
              subject_ref TEXT NOT NULL,
              export_json TEXT NOT NULL,
              export_lineage_json TEXT NOT NULL DEFAULT '[]',
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS adapter_admissibility_policies (
              policy_version TEXT PRIMARY KEY,
              policy_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              activated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS external_verification_manifests (
              external_verification_manifest_id TEXT PRIMARY KEY,
              subject_ref TEXT NOT NULL,
              manifest_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS verification_contracts (
              verification_contract_id TEXT PRIMARY KEY,
              subject_ref TEXT NOT NULL,
              contract_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              activated_at TEXT
            );

            -- v7.6 Trial Balance Table Families --

            CREATE TABLE IF NOT EXISTS trial_balance_views (
              trial_balance_view_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              trust_state TEXT NOT NULL
                CHECK(trust_state IN (
                  'REA_COMMITTED', 'PROJECTION_CANDIDATE', 'PROJECTION_VALIDATED',
                  'PROJECTION_WARNING', 'PROJECTION_BLOCKED', 'PROJECTION_REQUIRES_GOVERNANCE',
                  'PROJECTION_TRUSTED', 'PROJECTION_EXPORTED'
                )),
              pins_json TEXT NOT NULL,
              source_projection_result_ids_json TEXT NOT NULL DEFAULT '[]',
              accounts_json TEXT NOT NULL DEFAULT '[]',
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              created_by TEXT,
              UNIQUE(org_id, period_id, content_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_tb_views_org_period
              ON trial_balance_views(org_id, period_id);

            CREATE TABLE IF NOT EXISTS trial_balance_validation_contracts (
              contract_version TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              contract_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL
                CHECK(status IN ('CANDIDATE', 'ACTIVE', 'DEPRECATED')),
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trial_balance_validation_results (
              validation_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              trial_balance_view_id TEXT NOT NULL,
              status TEXT NOT NULL
                CHECK(status IN ('passed', 'warning', 'failed', 'requires_governance')),
              resulting_trust_state TEXT NOT NULL,
              pins_json TEXT NOT NULL,
              checks_json TEXT NOT NULL DEFAULT '[]',
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              created_by TEXT,
              UNIQUE(org_id, period_id, content_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_tb_validation_results_view
              ON trial_balance_validation_results(trial_balance_view_id);
            CREATE INDEX IF NOT EXISTS idx_tb_validation_results_org_period
              ON trial_balance_validation_results(org_id, period_id);

            CREATE TABLE IF NOT EXISTS trial_balance_validation_exceptions (
              exception_id TEXT PRIMARY KEY,
              validation_id TEXT NOT NULL,
              check_id TEXT NOT NULL,
              check_type TEXT NOT NULL,
              severity TEXT NOT NULL
                CHECK(severity IN ('info', 'warning', 'high', 'critical')),
              status TEXT NOT NULL
                CHECK(status IN ('passed', 'failed', 'waived', 'escalated')),
              affected_accounts_json TEXT NOT NULL DEFAULT '[]',
              observed_value_json TEXT,
              expected_value_json TEXT,
              evidence_refs_json TEXT NOT NULL DEFAULT '[]',
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tb_validation_exceptions_validation
              ON trial_balance_validation_exceptions(validation_id);

            CREATE TABLE IF NOT EXISTS trial_balance_justifications (
              justification_id TEXT PRIMARY KEY,
              validation_id TEXT NOT NULL,
              check_id TEXT NOT NULL,
              claim TEXT NOT NULL,
              evidence_refs_json TEXT NOT NULL DEFAULT '[]',
              reasoning_json TEXT NOT NULL,
              decision_status TEXT NOT NULL
                CHECK(decision_status IN ('accept', 'reject', 'escalate', 'waive')),
              actor TEXT NOT NULL,
              approved_by TEXT,
              approved_at TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tb_justifications_validation
              ON trial_balance_justifications(validation_id);

            CREATE TABLE IF NOT EXISTS trial_balance_governance_decisions (
              decision_id TEXT PRIMARY KEY,
              validation_id TEXT NOT NULL,
              check_id TEXT NOT NULL,
              decision_enum TEXT NOT NULL
                CHECK(decision_enum IN (
                  'ACCEPT_VALIDATION_RESULT', 'REJECT_VALIDATION_RESULT',
                  'WAIVE_CHECK_WITH_REASON', 'ESCALATE_TO_ACCOUNTANT',
                  'ESCALATE_TO_EXTERNAL_AUDITOR', 'REQUEST_MORE_EVIDENCE',
                  'REPROJECT_UNDER_APPROVED_BUNDLE'
                )),
              actor TEXT NOT NULL,
              justification_ref TEXT,
              evidence_refs_json TEXT NOT NULL DEFAULT '[]',
              approved_by TEXT,
              approved_at TEXT,
              notes TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tb_governance_decisions_validation
              ON trial_balance_governance_decisions(validation_id);

            CREATE TABLE IF NOT EXISTS trial_balance_user_feedback_signals (
              feedback_id TEXT PRIMARY KEY,
              validation_id TEXT NOT NULL,
              check_id TEXT NOT NULL,
              org_id TEXT NOT NULL,
              audience TEXT NOT NULL
                CHECK(audience IN ('owner', 'accountant', 'auditor', 'admin')),
              severity TEXT NOT NULL
                CHECK(severity IN ('info', 'warning', 'high', 'critical')),
              feedback_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tb_feedback_signals_validation
              ON trial_balance_user_feedback_signals(validation_id);

            CREATE TABLE IF NOT EXISTS tiered_verification_contracts (
              contract_id TEXT PRIMARY KEY,
              contract_version TEXT NOT NULL,
              org_id TEXT NOT NULL,
              purpose TEXT NOT NULL,
              contract_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL
                CHECK(status IN ('ACTIVE', 'DEPRECATED')),
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS order_maintenance_results (
              result_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              verification_contract_id TEXT NOT NULL,
              status TEXT NOT NULL
                CHECK(status IN (
                  'order_created', 'order_preserved', 'order_repaired',
                  'order_escalated', 'order_failed'
                )),
              trust_state TEXT NOT NULL,
              authority_trace_json TEXT NOT NULL DEFAULT '[]',
              failed_assertions_json TEXT NOT NULL DEFAULT '[]',
              result_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_order_maintenance_results_org_period
              ON order_maintenance_results(org_id, period_id);

            CREATE TABLE IF NOT EXISTS trial_balance_replay_bindings (
              replay_binding_id TEXT PRIMARY KEY,
              validation_id TEXT NOT NULL,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              pins_json TEXT NOT NULL,
              replay_inputs_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(validation_id, content_hash)
            );

            -- v8 CQ v2: Compliance Quality, Actuarial Readiness, Insurance Risk Contracts --

            CREATE TABLE IF NOT EXISTS cq_feature_vectors (
              feature_vector_id TEXT PRIMARY KEY,
              projection_id TEXT NOT NULL,
              trial_balance_view_id TEXT,
              tenant_id TEXT NOT NULL,
              cq_contract_version TEXT NOT NULL,
              features_json TEXT NOT NULL,
              source_artifacts_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_cq_feature_vectors_projection
              ON cq_feature_vectors(projection_id);

            CREATE TABLE IF NOT EXISTS cq_scores (
              cq_score_id TEXT PRIMARY KEY,
              feature_vector_id TEXT NOT NULL,
              projection_id TEXT NOT NULL,
              tenant_id TEXT NOT NULL,
              cq_contract_version TEXT NOT NULL,
              calibration_model_version TEXT NOT NULL,
              calibration_status TEXT NOT NULL
                CHECK(calibration_status IN ('governance_prior', 'empirically_calibrated')),
              score REAL NOT NULL,
              gates_json TEXT NOT NULL,
              explanation_json TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_cq_scores_projection
              ON cq_scores(projection_id, created_at);

            CREATE TABLE IF NOT EXISTS cq_outcomes (
              outcome_id TEXT PRIMARY KEY,
              projection_id TEXT NOT NULL,
              tenant_id TEXT NOT NULL,
              outcome_type TEXT NOT NULL
                CHECK(outcome_type IN (
                  'audit_adjustment', 'tax_penalty', 'insurance_claim',
                  'manual_correction', 'customer_dispute', 'regulator_notice',
                  'external_audit_finding', 'advisor_override',
                  'reversal_after_close'
                )),
              outcome_date TEXT NOT NULL,
              claim_occurred INTEGER NOT NULL DEFAULT 0,
              claim_amount REAL NOT NULL DEFAULT 0,
              audit_adjustment_amount REAL NOT NULL DEFAULT 0,
              tax_penalty_amount REAL NOT NULL DEFAULT 0,
              correction_cost REAL NOT NULL DEFAULT 0,
              time_to_resolution_days INTEGER,
              source_type TEXT NOT NULL,
              source_ref TEXT,
              evidence_hash TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_cq_outcomes_projection
              ON cq_outcomes(projection_id);

            CREATE TABLE IF NOT EXISTS cq_calibration_models (
              model_id TEXT PRIMARY KEY,
              model_version TEXT NOT NULL,
              model_type TEXT NOT NULL
                CHECK(model_type IN ('tiered_multiplier', 'logistic', 'poisson', 'gamma', 'lognormal', 'empirical_percentile')),
              calibration_status TEXT NOT NULL
                CHECK(calibration_status IN ('governance_prior', 'empirically_calibrated')),
              training_window_start TEXT,
              training_window_end TEXT,
              feature_schema_hash TEXT NOT NULL,
              outcome_population_hash TEXT,
              parameters_json TEXT NOT NULL,
              performance_json TEXT,
              governance_state TEXT NOT NULL
                CHECK(governance_state IN ('candidate', 'reviewed', 'certified', 'active', 'deprecated')),
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS insurance_eligibility_results (
              insurance_eligibility_id TEXT PRIMARY KEY,
              projection_id TEXT NOT NULL,
              tenant_id TEXT NOT NULL,
              insurance_contract_version TEXT NOT NULL,
              cq_score_id TEXT NOT NULL,
              eligibility_state TEXT NOT NULL
                CHECK(eligibility_state IN ('ELIGIBLE', 'ELIGIBLE_WITH_EXCLUSIONS', 'REQUIRES_REVIEW', 'DECLINED')),
              exclusions_json TEXT,
              liability_basis_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_insurance_eligibility_projection
              ON insurance_eligibility_results(projection_id);

            CREATE TABLE IF NOT EXISTS cq_risk_quotes (
              quote_id TEXT PRIMARY KEY,
              projection_id TEXT NOT NULL,
              tenant_id TEXT NOT NULL,
              cq_score REAL NOT NULL,
              cq_contract_version TEXT NOT NULL,
              calibration_model_version TEXT NOT NULL,
              calibration_status TEXT NOT NULL
                CHECK(calibration_status IN ('governance_prior', 'empirically_calibrated')),
              exposure_amount REAL NOT NULL,
              expected_frequency REAL,
              expected_severity_rate REAL,
              expected_loss REAL,
              pure_premium REAL,
              gross_premium REAL,
              liability_cap REAL,
              deductible REAL,
              eligibility_state TEXT NOT NULL
                CHECK(eligibility_state IN ('ELIGIBLE', 'ELIGIBLE_WITH_EXCLUSIONS', 'REQUIRES_REVIEW', 'DECLINED')),
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_cq_risk_quotes_projection
              ON cq_risk_quotes(projection_id);

            CREATE TABLE IF NOT EXISTS reinsurance_portfolio_manifests (
              manifest_id TEXT PRIMARY KEY,
              portfolio_id TEXT NOT NULL,
              period_start TEXT NOT NULL,
              period_end TEXT NOT NULL,
              cq_contract_version TEXT NOT NULL,
              calibration_model_version TEXT NOT NULL,
              insured_projection_count INTEGER NOT NULL,
              total_exposure_amount REAL NOT NULL,
              cq_distribution_json TEXT NOT NULL,
              expected_loss_total REAL,
              gross_premium_total REAL,
              reinsurance_layers_json TEXT NOT NULL,
              external_verification_manifest_refs_json TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_reinsurance_manifests_portfolio
              ON reinsurance_portfolio_manifests(portfolio_id);

            -- v8.1: Full-Cycle Accounting Runtime Contracts

            CREATE TABLE IF NOT EXISTS source_document_reviews (
              review_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              document_type TEXT NOT NULL,
              submitter_id TEXT,
              channel TEXT,
              file_hash TEXT NOT NULL,
              authenticity_check TEXT NOT NULL,
              legality_check TEXT NOT NULL,
              completeness_check TEXT NOT NULL,
              arithmetic_check TEXT NOT NULL,
              missing_fields_json TEXT,
              extraction_result_json TEXT,
              evidence_status TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_source_document_reviews_period
              ON source_document_reviews(org_id, period_id);

            CREATE TABLE IF NOT EXISTS journal_voucher_projections (
              jvp_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              source_event_id TEXT NOT NULL,
              projection_bundle_version TEXT NOT NULL,
              coa_version TEXT NOT NULL,
              voucher_type TEXT NOT NULL,
              lines_json TEXT NOT NULL,
              justification_ref TEXT,
              projection_status TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_jvp_period
              ON journal_voucher_projections(org_id, period_id);

            CREATE TABLE IF NOT EXISTS cash_journal_views (
              view_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              source_jvp_ids_json TEXT NOT NULL,
              coa_version TEXT NOT NULL,
              projection_bundle_version TEXT NOT NULL,
              entries_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS bank_journal_views (
              view_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              source_jvp_ids_json TEXT NOT NULL,
              coa_version TEXT NOT NULL,
              projection_bundle_version TEXT NOT NULL,
              entries_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS subsidiary_ledger_views (
              view_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              account_code TEXT NOT NULL,
              source_jvp_ids_json TEXT NOT NULL,
              coa_version TEXT NOT NULL,
              projection_bundle_version TEXT NOT NULL,
              entries_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS general_ledger_views (
              view_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              source_jvp_ids_json TEXT NOT NULL,
              coa_version TEXT NOT NULL,
              projection_bundle_version TEXT NOT NULL,
              balances_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS financial_statement_packages (
              package_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              trial_balance_view_id TEXT NOT NULL,
              period_close_validation_id TEXT,
              reconciliation_validation_id TEXT,
              reporting_contract_version TEXT NOT NULL,
              balance_sheet_json TEXT NOT NULL,
              income_statement_json TEXT NOT NULL,
              cash_flow_statement_json TEXT NOT NULL,
              validation_results_json TEXT NOT NULL,
              replay_binding_ref TEXT,
              audit_evidence_package_ref TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_financial_statement_packages_period
              ON financial_statement_packages(org_id, period_id);

            CREATE TABLE IF NOT EXISTS tax_filing_packages (
              filing_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              tax_type TEXT NOT NULL,
              tax_rule_bundle_version TEXT NOT NULL,
              source_refs_json TEXT NOT NULL,
              filing_form_json TEXT,
              validation_result_json TEXT,
              approval_state TEXT NOT NULL DEFAULT 'draft',
              submission_receipt_ref TEXT,
              replay_binding_ref TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_tax_filing_packages_period
              ON tax_filing_packages(org_id, period_id);

            CREATE TABLE IF NOT EXISTS accounting_archive_packages (
              archive_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              source_document_review_ids_json TEXT NOT NULL,
              rea_event_ids_json TEXT NOT NULL,
              jvp_ids_json TEXT NOT NULL,
              ledger_view_ids_json TEXT NOT NULL,
              validation_result_ids_json TEXT NOT NULL,
              governance_decision_ids_json TEXT NOT NULL,
              financial_statement_package_ids_json TEXT NOT NULL,
              tax_filing_package_ids_json TEXT NOT NULL,
              external_audit_export_refs_json TEXT,
              retention_policy_ref TEXT,
              archive_hash TEXT NOT NULL,
              manifest_ref TEXT,
              immutable_after TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_accounting_archive_packages_period
              ON accounting_archive_packages(org_id, period_id);

            CREATE TABLE IF NOT EXISTS monthly_accounting_workflows (
              workflow_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              close_state TEXT NOT NULL DEFAULT 'open',
              blockers_json TEXT NOT NULL DEFAULT '[]',
              opened_at TEXT NOT NULL,
              closed_at TEXT,
              reported_at TEXT,
              filed_at TEXT,
              archived_at TEXT,
              content_hash TEXT NOT NULL,
              updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_monthly_workflows_period
              ON monthly_accounting_workflows(org_id, period_id);

            -- v8.2: Cross-Domain Trust and Action Admissibility

            CREATE TABLE IF NOT EXISTS cross_domain_policies (
              policy_id TEXT PRIMARY KEY,
              policy_version TEXT NOT NULL,
              domain_set_json TEXT NOT NULL,
              compatibility_rules_json TEXT NOT NULL,
              action_policies_json TEXT NOT NULL,
              disclosure_requirements_json TEXT,
              limit_policies_json TEXT,
              governance_state TEXT NOT NULL
                CHECK(governance_state IN ('candidate', 'reviewed', 'certified', 'active', 'deprecated')),
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_cross_domain_policies_version
              ON cross_domain_policies(policy_version);

            CREATE TABLE IF NOT EXISTS cross_domain_resolver_contracts (
              resolver_contract_id TEXT PRIMARY KEY,
              resolver_version TEXT NOT NULL,
              semantics_json TEXT NOT NULL,
              precedence_bindings_json TEXT NOT NULL,
              deterministic_rules_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_cross_domain_resolver_version
              ON cross_domain_resolver_contracts(resolver_version);

            CREATE TABLE IF NOT EXISTS cross_domain_order_results (
              xorder_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT,
              action_intent TEXT NOT NULL,
              projection_set_hash TEXT NOT NULL,
              policy_version TEXT NOT NULL,
              resolver_version TEXT NOT NULL,
              order_result TEXT NOT NULL
                CHECK(order_result IN (
                  'ALLOW', 'ALLOW_WITH_DISCLOSURE', 'ALLOW_WITH_LIMITS',
                  'ESCALATE', 'BLOCK_ACTION', 'BLOCK_EXPORT'
                )),
              disclosure_required INTEGER NOT NULL DEFAULT 0,
              limits_json TEXT,
              blocked_action INTEGER NOT NULL DEFAULT 0,
              blocked_export INTEGER NOT NULL DEFAULT 0,
              reason_codes_json TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_cross_domain_order_results_org_period
              ON cross_domain_order_results(org_id, period_id);

            CREATE TABLE IF NOT EXISTS cross_domain_replay_bindings (
              cross_replay_id TEXT PRIMARY KEY,
              xorder_id TEXT NOT NULL,
              xpolicy_version TEXT NOT NULL,
              r_cross_version TEXT NOT NULL,
              projection_set_hash TEXT NOT NULL,
              cross_domain_order_result_hash TEXT NOT NULL,
              cross_domain_governance_hash TEXT,
              cross_domain_view_hash TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_cross_domain_replay_xorder
              ON cross_domain_replay_bindings(xorder_id);

            CREATE TABLE IF NOT EXISTS cross_domain_governance_decisions (
              cross_decision_id TEXT PRIMARY KEY,
              xorder_id TEXT NOT NULL,
              decision TEXT NOT NULL,
              reviewer_id TEXT,
              justification TEXT,
              approved_limits_json TEXT,
              disclosure_package_ref TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_cross_domain_decisions_xorder
              ON cross_domain_governance_decisions(xorder_id);
            """
        )
        projection_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(ledger_projections)").fetchall()
        }
        projection_column_defs = {
            "projection_scope": "TEXT",
            "projection_bundle_version": "TEXT",
            "coa_bundle_version": "TEXT",
            "knowledge_context_version": "TEXT",
            "constraint_bundle_version": "TEXT",
            "tag_ontology_version": "TEXT",
            "precedence_graph_version": "TEXT",
            "resolution_rules_version": "TEXT",
        }
        for column_name, column_type in projection_column_defs.items():
            if column_name not in projection_columns:
                conn.execute(
                    f"ALTER TABLE ledger_projections ADD COLUMN {column_name} {column_type}"
                )
        active_dataset_role_conflicts = conn.execute(
            """
            SELECT ksa.organization_id, ksa.scope_key, dav.dataset_usage_role, COUNT(*) AS conflict_count
            FROM knowledge_source_activation ksa
            JOIN dataset_asset_version dav
              ON dav.dataset_asset_version_id = ksa.dataset_asset_version_id
            WHERE ksa.activation_status = 'active'
              AND ksa.dataset_asset_version_id IS NOT NULL
            GROUP BY ksa.organization_id, ksa.scope_key, dav.dataset_usage_role
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if active_dataset_role_conflicts is not None:
            raise RuntimeError(
                "historical active dataset role conflict in knowledge_source_activation"
            )
        projection_context_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(projection_context_snapshots)").fetchall()
        }
        projection_context_column_defs = {
            "provider_ref": "TEXT NOT NULL DEFAULT 'hermes-agent/plugins/memory/holographic'",
            "adapter_version": "TEXT NOT NULL DEFAULT 'semantier_memory_adapter_v1'",
            "admissibility_policy_version": "TEXT",
            "rejected_items_summary_json": "TEXT NOT NULL DEFAULT '[]'",
        }
        for column_name, column_type in projection_context_column_defs.items():
            if column_name not in projection_context_columns:
                conn.execute(
                    f"ALTER TABLE projection_context_snapshots ADD COLUMN {column_name} {column_type}"
                )
        knowledge_artifact_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(knowledge_artifacts)").fetchall()
        }
        if "content_hash" not in knowledge_artifact_columns:
            conn.execute(
                "ALTER TABLE knowledge_artifacts ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
            )
        if "authority_origin" not in knowledge_artifact_columns:
            conn.execute(
                "ALTER TABLE knowledge_artifacts ADD COLUMN authority_origin TEXT NOT NULL DEFAULT 'runtime_observation'"
            )
            conn.execute(
                """
                UPDATE knowledge_artifacts
                SET authority_origin = CASE source_type
                    WHEN 'official_regulation' THEN 'official_source'
                    WHEN 'official_taxonomy' THEN 'ontology_governance'
                    WHEN 'field_guide' THEN 'interpretive_source'
                    WHEN 'firm_guidance' THEN 'interpretive_source'
                    WHEN 'internal_policy' THEN 'internal_governance'
                    WHEN 'runtime_observation' THEN 'runtime_observation'
                    ELSE 'runtime_observation'
                END
                """
            )
        knowledge_artifact_column_defs = {
            "source_uri": "TEXT",
            "jurisdiction": "TEXT",
            "effective_from": "TEXT",
            "effective_to": "TEXT",
            "source_version": "TEXT",
            "source_anchors_json": "TEXT NOT NULL DEFAULT '[]'",
            "extraction_method": "TEXT",
            "curator": "TEXT",
            "ambiguities_json": "TEXT NOT NULL DEFAULT '[]'",
        }
        for column_name, column_type in knowledge_artifact_column_defs.items():
            if column_name not in knowledge_artifact_columns:
                conn.execute(
                    f"ALTER TABLE knowledge_artifacts ADD COLUMN {column_name} {column_type}"
                )
        knowledge_artifacts_schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'knowledge_artifacts'"
        ).fetchone()
        knowledge_artifacts_sql = (
            str(knowledge_artifacts_schema["sql"])
            if knowledge_artifacts_schema and knowledge_artifacts_schema["sql"]
            else ""
        )
        if (
            "source_type IN ('official_regulation', 'official_taxonomy', 'field_guide', 'firm_guidance', 'internal_policy')"
            in knowledge_artifacts_sql
        ):
            conn.execute("ALTER TABLE knowledge_artifacts RENAME TO knowledge_artifacts_legacy")
            conn.executescript(
                """
                CREATE TABLE knowledge_artifacts (
                  artifact_id TEXT PRIMARY KEY,
                  source_type TEXT NOT NULL
                    CHECK(source_type IN ('official_regulation', 'official_taxonomy', 'field_guide', 'firm_guidance', 'internal_policy', 'runtime_observation')),
                  authority_domain TEXT NOT NULL
                    CHECK(authority_domain IN ('accounting', 'tax', 'management', 'compliance', 'internal_control')),
                  tag_authority_level TEXT NOT NULL
                    CHECK(tag_authority_level IN ('regulatory', 'accounting', 'management', 'operational', 'descriptive')),
                  semantic_tier TEXT NOT NULL
                    CHECK(semantic_tier IN ('T2', 'T3', 'T4', 'T5', 'T6')),
                  authority_origin TEXT NOT NULL DEFAULT 'runtime_observation'
                    CHECK(authority_origin IN ('official_source', 'interpretive_source', 'internal_governance', 'runtime_observation', 'ontology_governance')),
                  source_ref TEXT NOT NULL,
                  source_hash TEXT NOT NULL,
                  source_uri TEXT,
                  jurisdiction TEXT,
                  effective_from TEXT,
                  effective_to TEXT,
                  source_version TEXT,
                  source_anchors_json TEXT NOT NULL DEFAULT '[]',
                  extraction_method TEXT,
                  curator TEXT,
                  ambiguities_json TEXT NOT NULL DEFAULT '[]',
                  extracted_claims_json TEXT NOT NULL DEFAULT '[]',
                  extracted_constraints_json TEXT NOT NULL DEFAULT '[]',
                  extracted_projection_rules_json TEXT NOT NULL DEFAULT '[]',
                  ingestion_status TEXT NOT NULL
                    CHECK(ingestion_status IN ('PROPOSED', 'VALIDATED', 'APPROVED', 'ACTIVE', 'REJECTED', 'DEPRECATED')),
                  content_hash TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL,
                  approved_by TEXT
                );
                """
            )
            conn.execute(
                """
                INSERT INTO knowledge_artifacts (
                  artifact_id, source_type, authority_domain, tag_authority_level, semantic_tier,
                  authority_origin, source_ref, source_hash, source_uri, jurisdiction, effective_from,
                  effective_to, source_version, source_anchors_json, extraction_method, curator,
                  ambiguities_json, extracted_claims_json, extracted_constraints_json,
                  extracted_projection_rules_json, ingestion_status, content_hash, created_at, approved_by
                )
                SELECT
                  artifact_id,
                  source_type,
                  authority_domain,
                  tag_authority_level,
                  semantic_tier,
                  COALESCE(
                    authority_origin,
                    CASE source_type
                      WHEN 'official_regulation' THEN 'official_source'
                      WHEN 'official_taxonomy' THEN 'ontology_governance'
                      WHEN 'field_guide' THEN 'interpretive_source'
                      WHEN 'firm_guidance' THEN 'interpretive_source'
                      WHEN 'internal_policy' THEN 'internal_governance'
                      WHEN 'runtime_observation' THEN 'runtime_observation'
                      ELSE 'runtime_observation'
                    END
                  ),
                  source_ref,
                  source_hash,
                  source_uri,
                  jurisdiction,
                  effective_from,
                  effective_to,
                  source_version,
                  COALESCE(source_anchors_json, '[]'),
                  extraction_method,
                  curator,
                  COALESCE(ambiguities_json, '[]'),
                  COALESCE(extracted_claims_json, '[]'),
                  COALESCE(extracted_constraints_json, '[]'),
                  COALESCE(extracted_projection_rules_json, '[]'),
                  ingestion_status,
                  COALESCE(content_hash, ''),
                  created_at,
                  approved_by
                FROM knowledge_artifacts_legacy
                """
            )
            conn.execute("DROP TABLE knowledge_artifacts_legacy")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_status ON knowledge_artifacts(ingestion_status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_domain ON knowledge_artifacts(authority_domain)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_source_hash ON knowledge_artifacts(source_hash)"
            )
        kgl_version_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(kgl_versions)").fetchall()
        }
        if "authority_source_manifest_json" not in kgl_version_columns:
            conn.execute(
                "ALTER TABLE kgl_versions ADD COLUMN authority_source_manifest_json TEXT NOT NULL DEFAULT '[]'"
            )
        replay_binding_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(replay_bindings)").fetchall()
        }
        if "authority_source_manifest_json" not in replay_binding_columns:
            conn.execute(
                "ALTER TABLE replay_bindings ADD COLUMN authority_source_manifest_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "authority_source_manifest_hash" not in replay_binding_columns:
            conn.execute(
                "ALTER TABLE replay_bindings ADD COLUMN authority_source_manifest_hash TEXT NOT NULL DEFAULT ''"
            )
        knowledge_source_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(knowledge_source)").fetchall()
        }
        if "semantic_purpose" not in knowledge_source_columns:
            conn.execute("ALTER TABLE knowledge_source ADD COLUMN semantic_purpose TEXT")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_knowledge_source_kind_purpose
              ON knowledge_source(source_kind, semantic_purpose)
            """
        )
        policy_rule_candidate_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(policy_rule_candidates)").fetchall()
        }
        policy_rule_candidate_column_defs = {
            "organization_id": "TEXT",
            "proposed_workspace_scope": "TEXT",
            "proposed_workspace_id": "TEXT",
            "authority_domain": "TEXT",
            "semantic_tier": "TEXT",
            "jurisdiction": "TEXT",
            "candidate_idempotency_key": "TEXT",
        }
        for column_name, column_type in policy_rule_candidate_column_defs.items():
            if column_name not in policy_rule_candidate_columns:
                conn.execute(
                    f"ALTER TABLE policy_rule_candidates ADD COLUMN {column_name} {column_type}"
                )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_policy_rule_candidates_idempotency
              ON policy_rule_candidates(candidate_idempotency_key)
              WHERE candidate_idempotency_key IS NOT NULL
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_policy_rule_candidates_scope
              ON policy_rule_candidates(
                organization_id,
                proposed_workspace_scope,
                proposed_workspace_id,
                authority_domain,
                semantic_tier,
                jurisdiction
              )
            """
        )
        governed_rule_version_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(governed_rule_versions)").fetchall()
        }
        governed_rule_version_column_defs = {
            "organization_id": "TEXT",
            "authority_domain": "TEXT",
            "semantic_tier": "TEXT",
            "jurisdiction": "TEXT",
            "source_graph_version": "TEXT",
            "source_graph_release_hash": "TEXT",
            "source_graph_rule_id": "TEXT",
        }
        for column_name, column_type in governed_rule_version_column_defs.items():
            if column_name not in governed_rule_version_columns:
                conn.execute(
                    f"ALTER TABLE governed_rule_versions ADD COLUMN {column_name} {column_type}"
                )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_governed_rule_versions_owner
              ON governed_rule_versions(
                organization_id,
                authority_domain,
                semantic_tier,
                jurisdiction
              )
            """
        )
        conn.execute(
            """
            INSERT INTO semantier_schema_versions (schema_name, schema_version, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(schema_name) DO UPDATE SET
              schema_version = excluded.schema_version,
              updated_at = excluded.updated_at
            """,
            ("rule_governance", RULE_GOVERNANCE_SCHEMA_VERSION, now_iso()),
        )
        tender_run_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(tender_detection_runs)").fetchall()
        }
        tender_run_column_defs = {
            "document_text": "TEXT NOT NULL DEFAULT ''",
            "source_document_hash": "TEXT NOT NULL DEFAULT ''",
            "parent_run_id": "TEXT",
            "root_run_id": "TEXT",
            "idempotency_key": "TEXT",
            "request_hash": "TEXT",
            "run_status": "TEXT NOT NULL DEFAULT 'FINALIZED'",
            "finding_set_hash": "TEXT",
            "runtime_conflict_set_hash": "TEXT",
            "replay_status": "TEXT NOT NULL DEFAULT 'CREATED'",
        }
        for column_name, column_type in tender_run_column_defs.items():
            if column_name not in tender_run_columns:
                conn.execute(
                    f"ALTER TABLE tender_detection_runs ADD COLUMN {column_name} {column_type}"
                )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_tender_detection_runs_idempotency
              ON tender_detection_runs(organization_id, COALESCE(workspace_id, ''), idempotency_key)
              WHERE idempotency_key IS NOT NULL
            """
        )
        tender_finding_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(tender_detection_findings)").fetchall()
        }
        tender_finding_column_defs = {
            "target_anchor_ref": "TEXT",
            "target_element_kind": "TEXT",
            "target_element_id": "TEXT",
            "normalized_match": "TEXT",
            "start_offset": "INTEGER",
            "end_offset": "INTEGER",
            "occurrence_index": "INTEGER",
            "target_element_hash": "TEXT",
            "target_evidence_ref": "TEXT",
            "source_graph_version": "TEXT",
            "source_graph_release_hash": "TEXT",
            "source_graph_rule_id": "TEXT",
            "triggered_rule_version_refs_json": "TEXT NOT NULL DEFAULT '[]'",
            "winning_rule_version_ref": "TEXT",
            "conflicting_rule_version_refs_json": "TEXT NOT NULL DEFAULT '[]'",
            "conflict_resolution_ref": "TEXT",
            "matched_sensitive_concept_id": "TEXT",
            "observed_expression": "TEXT",
            "detection_method": "TEXT NOT NULL DEFAULT 'exact'",
            "semantic_relation": "TEXT NOT NULL DEFAULT 'exact'",
            "evidence_text": "TEXT",
            "precedent_refs_json": "TEXT NOT NULL DEFAULT '[]'",
            "decision_status": "TEXT NOT NULL DEFAULT 'violation'",
            "grounding_status": "TEXT NOT NULL DEFAULT 'ungrounded'",
            "decision_context_json": "TEXT NOT NULL DEFAULT '{}'",
            "candidate_finding_schema_version": "TEXT NOT NULL DEFAULT 'contextgraph_candidate_finding.v2'",
            "semantic_profile_id": "TEXT",
            "derivation_shape": "TEXT",
            "frame_derivation_json": "TEXT NOT NULL DEFAULT '{}'",
        }
        for column_name, column_type in tender_finding_column_defs.items():
            if column_name not in tender_finding_columns:
                conn.execute(
                    f"ALTER TABLE tender_detection_findings ADD COLUMN {column_name} {column_type}"
                )
        tender_report_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(tender_detection_reports)").fetchall()
        }
        if "prior_labeling_report_id" not in tender_report_columns:
            conn.execute(
                "ALTER TABLE tender_detection_reports ADD COLUMN prior_labeling_report_id TEXT"
            )
        external_audit_export_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(external_audit_exports)").fetchall()
        }
        if "export_lineage_json" not in external_audit_export_columns:
            conn.execute(
                "ALTER TABLE external_audit_exports ADD COLUMN export_lineage_json TEXT NOT NULL DEFAULT '[]'"
            )
        conn.execute(
            """
            INSERT INTO semantier_schema_versions (schema_name, schema_version, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(schema_name) DO UPDATE SET
              schema_version = excluded.schema_version,
              updated_at = excluded.updated_at
            """,
            ("eos_core", EOS_CORE_SCHEMA_VERSION, now_iso()),
        )


def _eos_core_schema_is_current(db_path: str | Path | None = None) -> bool:
    try:
        with get_conn(db_path) as conn:
            version = conn.execute(
                "SELECT schema_version FROM semantier_schema_versions "
                "WHERE schema_name = 'eos_core'"
            ).fetchone()
            if version is None or int(version["schema_version"]) != EOS_CORE_SCHEMA_VERSION:
                return False
            placeholders = ",".join("?" for _ in EOS_CORE_SCHEMA_SENTINELS)
            present = {
                str(row["name"])
                for row in conn.execute(
                    f"SELECT name FROM sqlite_master WHERE type = 'table' "
                    f"AND name IN ({placeholders})",
                    tuple(sorted(EOS_CORE_SCHEMA_SENTINELS)),
                ).fetchall()
            }
            return present == EOS_CORE_SCHEMA_SENTINELS
    except (OSError, sqlite3.DatabaseError, ValueError):
        return False


def init_db(db_path: str | Path | None = None, *, force: bool = False) -> None:
    """Initialize the EOS schema for one database path.

    Schema setup is serialized per process because runtime request handlers may
    construct stores concurrently while sharing the same SQLite file.
    """
    if hosted.enabled():
        from eos.postgres import require_ready
        with get_conn(db_path) as conn:
            require_ready(conn)
        return
    with _INIT_DB_LOCK:
        if not force and _eos_core_schema_is_current(db_path):
            return
        _init_db_unlocked(db_path)


def verify_rule_governance_schema_ready(conn: sqlite3.Connection | None = None) -> None:
    """Fail closed when the clean cut-over rule-governance schema is unavailable."""

    owns_conn = conn is None
    active_conn = conn or get_conn()
    try:
        version = active_conn.execute(
            """
            SELECT schema_version FROM semantier_schema_versions
            WHERE schema_name = 'rule_governance'
            """
        ).fetchone()
        if version is None or int(version["schema_version"]) != RULE_GOVERNANCE_SCHEMA_VERSION:
            raise RuntimeError("RULE_GOVERNANCE_SCHEMA_VERSION=3 is required")
        required_columns = {
            "governed_rule_activation": {
                "activation_id",
                "rule_version_id",
                "organization_id",
                "activation_scope",
                "workspace_id",
                "activation_status",
                "effective_from",
                "effective_to",
                "approval_event_ref",
                "activation_scope_hash",
                "supersedes_activation_id",
                "content_hash",
            },
            "governed_rule_activation_event": {
                "activation_event_id",
                "activation_stream_id",
                "activation_id",
                "event_type",
                "from_state",
                "to_state",
                "effective_at",
                "recorded_at",
                "event_sequence",
                "prior_event_hash",
                "content_hash",
            },
            "knowledge_activation_set_snapshot": {
                "activation_set_snapshot_id",
                "included_activation_ids_json",
                "resolved_activation_set_hash",
                "effective_at",
            },
            "governed_rule_versions": {
                "source_graph_version",
                "source_graph_release_hash",
                "source_graph_rule_id",
            },
        }
        for table_name, expected_columns in required_columns.items():
            if hosted.is_postgres(active_conn):
                hosted.require_tables(active_conn, table_name)
                actual_columns = {row[0] for row in active_conn.execute(
                    'SELECT column_name FROM information_schema.columns '
                    'WHERE table_schema=current_schema() AND table_name=%s', (table_name,))}
                if expected_columns - actual_columns:
                    raise RuntimeError('required rule-governance columns missing: ' + table_name)
                continue
            table = active_conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table_name,),
            ).fetchone()
            if table is None:
                raise RuntimeError(f"required rule-governance table missing: {table_name}")
            actual_columns = {
                row["name"] for row in active_conn.execute(f"PRAGMA table_info({table_name})")
            }
            missing_columns = sorted(expected_columns - actual_columns)
            if missing_columns:
                raise RuntimeError(
                    f"required rule-governance columns missing from {table_name}: {missing_columns}"
                )
    finally:
        if owns_conn:
            active_conn.close()


def insert_json(table: str, values: dict) -> dict:
    keys = list(values.keys())
    placeholders = ", ".join([hosted.sql_query("?", '%s')] * len(keys))
    columns = ", ".join(keys)
    with get_conn() as conn:
        conn.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            [values[k] for k in keys],
        )
    return values


def append_event(event_type: str, payload: dict, correlation_id: str | None = None, causation_id: str | None = None) -> dict:
    payload_json = canonical_json(payload)
    event_hash = content_hash(payload)
    event = {
        "event_id": f"evt_{uuid4().hex}",
        "event_type": event_type,
        "occurred_at": now_iso(),
        "payload_json": payload_json,
        "correlation_id": correlation_id,
        "causation_id": causation_id,
        "stream_id": correlation_id or "global",
        "content_hash": event_hash,
        "prior_event_hash": None,
    }
    return insert_json("events", event)


def get_signal(signal_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(hosted.sql_query("SELECT * FROM signals WHERE signal_id = ?", 'SELECT * FROM signals WHERE signal_id = %s'), (signal_id,)).fetchone()
    if row is None:
        return None
    return {**dict(row), "payload": json.loads(row["payload_json"])}


def get_exception(exception_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(hosted.sql_query("SELECT * FROM exceptions WHERE exception_id = ?", 'SELECT * FROM exceptions WHERE exception_id = %s'), (exception_id,)).fetchone()
    if row is None:
        return None
    return {**dict(row), "reason": json.loads(row["reason_json"])}


def get_justification(justification_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(hosted.sql_query("SELECT * FROM justifications WHERE justification_id = ?", 'SELECT * FROM justifications WHERE justification_id = %s'), (justification_id,)).fetchone()
    if row is None:
        return None
    return {**dict(row), "justification": json.loads(row["justification_json"])}


def get_journal_entry(journal_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(hosted.sql_query("SELECT * FROM journal_entries WHERE journal_id = ?", 'SELECT * FROM journal_entries WHERE journal_id = %s'), (journal_id,)).fetchone()
    if row is None:
        return None
    return {**dict(row), "journal_entry": json.loads(row["je_json"])}


def list_journal_entries():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM journal_entries").fetchall()
    return [dict(r) for r in rows]


def get_latest_validation(justification_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            hosted.sql_query("""
            SELECT * FROM validation_results
            WHERE justification_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """, """
            SELECT * FROM validation_results
            WHERE justification_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """),
            (justification_id,),
        ).fetchone()
    if row is None:
        return None
    return {**dict(row), "result": json.loads(row["result_json"])}


def list_ledger_projections_for_rea(rea_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            hosted.sql_query("""
            SELECT * FROM ledger_projections
            WHERE rea_id = ?
            ORDER BY created_at ASC, ledger_id ASC
            """, """
            SELECT * FROM ledger_projections
            WHERE rea_id = %s
            ORDER BY created_at ASC, ledger_id ASC
            """),
            (rea_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def close_exception(exception_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            hosted.sql_query("UPDATE exceptions SET status = ?, resolved_at = ? WHERE exception_id = ?", 'UPDATE exceptions SET status = %s, resolved_at = %s WHERE exception_id = %s'),
            ("RESOLVED", now_iso(), exception_id),
        )
