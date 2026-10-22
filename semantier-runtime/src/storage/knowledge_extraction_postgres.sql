-- Explicit PostgreSQL extraction owner schema; Human owns shared grounding bindings.
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
                  content_hash TEXT NOT NULL, lineage_id TEXT, retirement_state TEXT NOT NULL DEFAULT 'active',
                  FOREIGN KEY(discovery_run_id) REFERENCES knowledge_builder_discovery_runs(discovery_run_id),
                  FOREIGN KEY(source_id) REFERENCES knowledge_builder_sources(source_id)
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
                , lineage_id TEXT);

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
                  content_hash TEXT NOT NULL, lineage_id TEXT,
                  FOREIGN KEY(source_id) REFERENCES knowledge_builder_sources(source_id),
                  UNIQUE(organization_id, workspace_id, source_id, relation_kind, version),
                  UNIQUE(organization_id, workspace_id, source_id, relation_kind,
                         artifact_ref, artifact_content_hash)
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
                , retirement_mode TEXT, domain_id TEXT, inventory_hash TEXT, organization_id TEXT, workspace_id TEXT);

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

CREATE TABLE IF NOT EXISTS knowledge_builder_semantic_occurrences (
                  occurrence_id TEXT PRIMARY KEY,
                  extraction_run_id TEXT NOT NULL,
                  document_id TEXT NOT NULL,
                  source_anchor_json TEXT NOT NULL,
                  extraction_class TEXT NOT NULL,
                  extraction_attributes_json TEXT NOT NULL,
                  alignment_json TEXT NOT NULL,
                  machine_confidence DOUBLE PRECISION,
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

CREATE TABLE IF NOT EXISTS knowledge_builder_ai_grounding_suggestions (
                  assertion_id TEXT PRIMARY KEY,
                  extraction_run_id TEXT NOT NULL,
                  suggestion_status TEXT NOT NULL CHECK(suggestion_status IN
                    ('ready_for_review','low_confidence','missing_evidence')),
                  confidence DOUBLE PRECISION NOT NULL,
                  evidence_anchor_count INTEGER NOT NULL,
                  provider TEXT NOT NULL,
                  provider_version TEXT NOT NULL,
                  threshold DOUBLE PRECISION NOT NULL,
                  rationale TEXT NOT NULL,
                  suggested_at TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(assertion_id) REFERENCES knowledge_builder_document_candidate_assertions(assertion_id),
                  FOREIGN KEY(extraction_run_id) REFERENCES knowledge_builder_extraction_runs(extraction_run_id)
                );

CREATE TABLE IF NOT EXISTS knowledge_builder_ai_grounding_results (
                  ai_grounding_run_id TEXT NOT NULL,
                  assertion_id TEXT NOT NULL,
                  suggestion_status TEXT NOT NULL CHECK(suggestion_status IN
                    ('supported','unsupported','ambiguous','needs_edit','provider_error')),
                  confidence DOUBLE PRECISION NOT NULL,
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

CREATE TABLE IF NOT EXISTS knowledge_builder_evidence_obligations (
                  obligation_id TEXT PRIMARY KEY,
                  assertion_id TEXT NOT NULL,
                  obligation_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(assertion_id) REFERENCES knowledge_builder_document_candidate_assertions(assertion_id)
                );

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

CREATE TABLE IF NOT EXISTS knowledge_builder_evidence_obligation_anchors (
                  obligation_id TEXT NOT NULL,
                  source_anchor_ref TEXT NOT NULL,
                  ordinal INTEGER NOT NULL,
                  PRIMARY KEY(obligation_id, source_anchor_ref),
                  FOREIGN KEY(obligation_id) REFERENCES knowledge_builder_evidence_obligations(obligation_id)
                );

CREATE TABLE IF NOT EXISTS knowledge_builder_evidence_obligation_resolution_events (
                  resolution_event_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                  obligation_id TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('satisfied','unresolved')),
                  supporting_anchor_refs_json TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  FOREIGN KEY(obligation_id) REFERENCES knowledge_builder_evidence_obligations(obligation_id)
                );

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

CREATE INDEX IF NOT EXISTS idx_kber_org ON knowledge_builder_extraction_runs(organization_id, started_at);

CREATE INDEX IF NOT EXISTS idx_kbags_run_status
                  ON knowledge_builder_ai_grounding_suggestions(extraction_run_id, suggestion_status, confidence);

CREATE INDEX IF NOT EXISTS idx_kbagr_latest
                  ON knowledge_builder_ai_grounding_runs(extraction_run_id, status, completed_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_kbgb_tenant_idempotency
                  ON knowledge_builder_grounding_batches(
                    organization_id, COALESCE(workspace_id, ''), idempotency_key
                  );

CREATE INDEX IF NOT EXISTS idx_cgrc_run
                  ON contextgraph_reference_concepts(extraction_run_id, grounding_state);

CREATE INDEX IF NOT EXISTS idx_cgce_concept
                  ON contextgraph_canonical_expressions(concept_id, expression_kind);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cgoeo_learning_event
                  ON contextgraph_observed_expression_observations(human_learning_event_id, finding_id);

CREATE INDEX IF NOT EXISTS idx_cg_artifact_origins_source
                  ON contextgraph_artifact_origins (organization_id, workspace_id, source_id);

CREATE INDEX IF NOT EXISTS idx_cg_artifact_origins_artifact
                  ON contextgraph_artifact_origins (artifact_id);

CREATE INDEX IF NOT EXISTS idx_source_representation_edges_source
                  ON source_representation_edges(
                    organization_id, workspace_id, source_id, relation_kind, version
                  );

CREATE INDEX IF NOT EXISTS idx_source_representation_events_latest
                  ON source_representation_lifecycle_events(
                    representation_id, created_at
                  );

CREATE INDEX IF NOT EXISTS idx_source_lifecycle_events_source
                  ON source_lifecycle_events (source_id, created_at);

CREATE INDEX IF NOT EXISTS idx_artifact_lifecycle_events_artifact
                  ON artifact_lifecycle_events (artifact_id, created_at);

CREATE INDEX IF NOT EXISTS idx_source_retirement_ops_source
                  ON source_retirement_operations (source_id, created_at);

CREATE INDEX IF NOT EXISTS idx_lineage_repair_events_scope
                  ON lineage_repair_events (workspace_id, organization_id, created_at);

CREATE INDEX IF NOT EXISTS idx_knowledge_builder_extraction_runs_lineage_id ON knowledge_builder_extraction_runs(lineage_id);

CREATE INDEX IF NOT EXISTS idx_source_representation_edges_lineage_id ON source_representation_edges(lineage_id);

CREATE INDEX IF NOT EXISTS idx_contextgraph_artifact_origins_lineage_id ON contextgraph_artifact_origins(lineage_id);

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

-- Preserve SQLite's implicit insertion-order tie break without touching payload hashes.
ALTER TABLE source_representation_lifecycle_events ADD COLUMN IF NOT EXISTS
    event_sequence BIGINT GENERATED BY DEFAULT AS IDENTITY;
CREATE UNIQUE INDEX IF NOT EXISTS source_representation_event_sequence_unique
    ON source_representation_lifecycle_events(event_sequence);

-- Keep same-timestamp source lifecycle ordering independent of heap layout.
ALTER TABLE source_lifecycle_events ADD COLUMN IF NOT EXISTS
    event_sequence BIGINT GENERATED BY DEFAULT AS IDENTITY;
CREATE UNIQUE INDEX IF NOT EXISTS source_lifecycle_event_sequence_unique
    ON source_lifecycle_events(event_sequence);
