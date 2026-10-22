-- Native PostgreSQL additions owned by long-lived storage APIs.
-- Runtime constructors only validate these explicit migrated contracts.
ALTER TABLE knowledge_builder_sources ADD COLUMN IF NOT EXISTS lineage_id TEXT;
ALTER TABLE knowledge_builder_sources ADD COLUMN IF NOT EXISTS retirement_state TEXT NOT NULL DEFAULT 'active';
CREATE INDEX IF NOT EXISTS idx_knowledge_builder_sources_lineage_id ON knowledge_builder_sources(lineage_id);
ALTER TABLE knowledge_builder_discovery_runs ADD COLUMN IF NOT EXISTS lineage_id TEXT;
ALTER TABLE knowledge_builder_discovery_runs ADD COLUMN IF NOT EXISTS retirement_state TEXT NOT NULL DEFAULT 'active';
CREATE INDEX IF NOT EXISTS idx_knowledge_builder_discovery_runs_lineage_id ON knowledge_builder_discovery_runs(lineage_id);

CREATE TABLE IF NOT EXISTS knowledge_derivations (
    derivation_ref TEXT PRIMARY KEY, derivation_hash TEXT NOT NULL UNIQUE,
    output_ref TEXT NOT NULL UNIQUE, payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS relation_temporal_states (
    state_ref TEXT PRIMARY KEY, state_hash TEXT NOT NULL UNIQUE,
    relation_instance_ref TEXT NOT NULL, relation_instance_hash TEXT NOT NULL,
    admitted_at TEXT NOT NULL, supersedes_state_ref TEXT, payload_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_relation_temporal_states_history
    ON relation_temporal_states(relation_instance_ref,admitted_at,state_ref);

-- Feature-pack owner extension, explicit migration only.
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

CREATE TABLE IF NOT EXISTS corpus_growth_observations (
              observation_ref TEXT PRIMARY KEY,
              observation_hash TEXT NOT NULL UNIQUE,
              organization_id TEXT NOT NULL,
              workspace_id TEXT NOT NULL,
              semantic_key TEXT NOT NULL,
              observed_at TEXT NOT NULL,
              payload_json TEXT NOT NULL
            );

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
