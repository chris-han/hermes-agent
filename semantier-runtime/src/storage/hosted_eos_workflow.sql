-- Explicit evaluation, benchmark projection and guidance owner migrations.
CREATE TABLE IF NOT EXISTS evaluation_control_plane_subject_manifests (
                  content_hash TEXT PRIMARY KEY,
                  subject_id TEXT NOT NULL,
                  schema_version TEXT NOT NULL,
                  record_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE INDEX IF NOT EXISTS idx_ecp_subject_id
                  ON evaluation_control_plane_subject_manifests(subject_id);

CREATE TABLE IF NOT EXISTS evaluation_control_plane_environment_manifests (
                  environment_manifest_ref TEXT PRIMARY KEY,
                  content_hash TEXT NOT NULL UNIQUE,
                  runtime_commit TEXT NOT NULL,
                  environment_fingerprint TEXT NOT NULL,
                  schema_version TEXT NOT NULL,
                  record_json TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS evaluation_control_plane_runs (
                  run_id TEXT PRIMARY KEY,
                  benchmark_family_ref TEXT NOT NULL,
                  scenario_ref TEXT NOT NULL,
                  subject_manifest_ref TEXT NOT NULL,
                  environment_manifest_ref TEXT NOT NULL,
                  evaluator_bundle_ref TEXT NOT NULL,
                  validity TEXT NOT NULL,
                  started_at TEXT NOT NULL,
                  completed_at TEXT NULL,
                  retry_of_run_ref TEXT NULL,
                  parent_orchestration_ref TEXT NULL,
                  schema_version TEXT NOT NULL,
                  content_hash TEXT NOT NULL UNIQUE,
                  record_json TEXT NOT NULL
                );

CREATE INDEX IF NOT EXISTS idx_ecp_runs_parent
                  ON evaluation_control_plane_runs(parent_orchestration_ref);

CREATE TABLE IF NOT EXISTS evaluation_control_plane_evidence (
                  evidence_id TEXT PRIMARY KEY,
                  run_ref TEXT NOT NULL UNIQUE,
                  benchmark_family TEXT NOT NULL,
                  certification_result TEXT NOT NULL,
                  signed_result_ref TEXT NOT NULL UNIQUE,
                  schema_version TEXT NOT NULL,
                  content_hash TEXT NOT NULL UNIQUE,
                  record_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS evaluation_control_plane_waivers (
                  waiver_id TEXT PRIMARY KEY,
                  scope_ref TEXT NOT NULL,
                  signed_waiver_ref TEXT NOT NULL UNIQUE,
                  expires_at TEXT NULL,
                  content_hash TEXT NOT NULL UNIQUE,
                  record_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS benchmark_read_model_schema_metadata (
                  schema_version TEXT PRIMARY KEY,
                  initialized_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS benchmark_read_model_orchestrations (
                  benchmark_orchestration_id TEXT PRIMARY KEY,
                  schema_version TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NULL,
                  benchmark_family_ref TEXT NOT NULL,
                  profile_id TEXT NOT NULL,
                  profile_version TEXT NOT NULL,
                  requested_layers_json TEXT NOT NULL,
                  execution_mode TEXT NOT NULL,
                  operational_status TEXT NOT NULL,
                  warnings_json TEXT NOT NULL,
                  child_run_refs_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  completed_at TEXT NULL,
                  stale_reason_code TEXT NULL,
                  content_hash TEXT NOT NULL
                );

CREATE INDEX IF NOT EXISTS idx_brm_orch_org_created
                  ON benchmark_read_model_orchestrations(organization_id, created_at DESC, benchmark_orchestration_id DESC);

CREATE INDEX IF NOT EXISTS idx_brm_orch_workspace_created
                  ON benchmark_read_model_orchestrations(organization_id, workspace_id, created_at DESC, benchmark_orchestration_id DESC);

CREATE INDEX IF NOT EXISTS idx_brm_orch_status
                  ON benchmark_read_model_orchestrations(organization_id, operational_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_brm_orch_profile
                  ON benchmark_read_model_orchestrations(organization_id, profile_id, profile_version, created_at DESC);

CREATE TABLE IF NOT EXISTS benchmark_read_model_layer_runs (
                  evaluation_run_id TEXT PRIMARY KEY,
                  schema_version TEXT NOT NULL,
                  benchmark_orchestration_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NULL,
                  layer TEXT NOT NULL,
                  provider_id TEXT NULL,
                  profile_id TEXT NOT NULL,
                  profile_version TEXT NOT NULL,
                  execution_mode TEXT NOT NULL,
                  operational_status TEXT NOT NULL,
                  evaluation_evidence_id TEXT NULL,
                  run_validity TEXT NULL,
                  certification_result TEXT NULL,
                  benchmark_family_ref TEXT NOT NULL,
                  scenario_ref TEXT NOT NULL,
                  subject_manifest_ref TEXT NOT NULL,
                  environment_manifest_ref TEXT NOT NULL,
                  evaluator_bundle_ref TEXT NOT NULL,
                  reference_release_id TEXT NULL,
                  reference_release_hash TEXT NULL,
                  created_at TEXT NOT NULL,
                  completed_at TEXT NULL,
                  summary_json TEXT NOT NULL,
                  warnings_json TEXT NOT NULL,
                  stale_reason_code TEXT NULL,
                  content_hash TEXT NOT NULL,
                  UNIQUE (benchmark_orchestration_id, layer, evaluation_run_id)
                );

CREATE INDEX IF NOT EXISTS idx_brm_layer_parent ON benchmark_read_model_layer_runs(benchmark_orchestration_id, layer);

CREATE INDEX IF NOT EXISTS idx_brm_layer_cert ON benchmark_read_model_layer_runs(organization_id, certification_result, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_brm_layer_provider ON benchmark_read_model_layer_runs(organization_id, layer, provider_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_brm_layer_status ON benchmark_read_model_layer_runs(organization_id, execution_mode, operational_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_brm_layer_reference ON benchmark_read_model_layer_runs(organization_id, reference_release_id, created_at DESC);

CREATE TABLE IF NOT EXISTS benchmark_read_model_case_results (
                  evaluation_run_id TEXT NOT NULL, case_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL, workspace_id TEXT NULL,
                  layer TEXT NOT NULL, provider_id TEXT NULL, status TEXT NOT NULL,
                  challenge_tags_json TEXT NOT NULL, metric_contributions_json TEXT NOT NULL,
                  source_anchor_refs_json TEXT NOT NULL, assertion_refs_json TEXT NOT NULL,
                  artifact_refs_json TEXT NOT NULL, result_json TEXT NOT NULL,
                  stale_reason_code TEXT NULL, content_hash TEXT NOT NULL,
                  PRIMARY KEY (evaluation_run_id, case_id)
                );

CREATE INDEX IF NOT EXISTS idx_brm_case_org_status ON benchmark_read_model_case_results(organization_id, layer, status, case_id);

CREATE INDEX IF NOT EXISTS idx_brm_case_provider ON benchmark_read_model_case_results(organization_id, provider_id, status, case_id);

CREATE TABLE IF NOT EXISTS benchmark_read_model_challenge_slices (
                  evaluation_run_id TEXT NOT NULL, challenge_tag TEXT NOT NULL,
                  organization_id TEXT NOT NULL, workspace_id TEXT NULL, layer TEXT NOT NULL,
                  base_case_set_hash TEXT NOT NULL, base_case_count BIGINT NOT NULL,
                  included_case_count BIGINT NOT NULL, excluded_case_count BIGINT NOT NULL,
                  excluded_case_refs_json TEXT NOT NULL, denominator_policy TEXT NOT NULL,
                  metrics_json TEXT NOT NULL, artifact_refs_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  PRIMARY KEY (evaluation_run_id, challenge_tag)
                );

CREATE TABLE IF NOT EXISTS benchmark_read_model_artifact_refs (
                  evaluation_run_id TEXT NOT NULL, artifact_ref TEXT NOT NULL,
                  organization_id TEXT NOT NULL, workspace_id TEXT NULL,
                  artifact_kind TEXT NOT NULL, media_type TEXT NOT NULL,
                  artifact_content_hash TEXT NOT NULL, availability TEXT NOT NULL,
                  stale_reason_code TEXT NULL,
                  PRIMARY KEY (evaluation_run_id, artifact_ref)
                );

CREATE TABLE IF NOT EXISTS guidance_trigger_state_events (
                sequence BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                suppression_key TEXT NOT NULL,
                material_revision TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('dismissed', 'resolved')),
                actor_or_resolution_ref TEXT NOT NULL
            );
INSERT INTO benchmark_read_model_schema_metadata(schema_version,initialized_at)
VALUES ('benchmark_read_model_schema.v1',to_char(now() AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS.US"Z"'))
ON CONFLICT DO NOTHING;
