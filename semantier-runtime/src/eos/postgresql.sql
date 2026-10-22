-- Explicit EOS PostgreSQL schema v1. Migration role only.
-- FTS5 shadow pages are rebuildable; logical graph_node_search content is preserved.

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
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

CREATE TABLE IF NOT EXISTS adapter_admissibility_policies (
              policy_version TEXT PRIMARY KEY,
              policy_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              activated_at TEXT
            );

CREATE TABLE IF NOT EXISTS architecture_assessments (
                  assessment_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  metric_snapshot_ref TEXT NOT NULL,
                  threshold_profile_ref TEXT NOT NULL,
                  assessment_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS architecture_handoff_grants (
                  grant_id TEXT PRIMARY KEY,
                  review_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  grant_type TEXT NOT NULL,
                  status TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  revoked_at TEXT
                );

CREATE TABLE IF NOT EXISTS architecture_learning_classifications (
                  classification_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  assessment_ref TEXT NOT NULL,
                  classification_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS architecture_legal_graph_pilot_evidence (
                  evidence_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  evidence_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS architecture_metric_snapshots (
                  snapshot_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  environment TEXT NOT NULL,
                  architecture_profile TEXT NOT NULL,
                  observed_from TEXT NOT NULL,
                  observed_to TEXT NOT NULL,
                  metric_definition_set_ref TEXT NOT NULL,
                  snapshot_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS architecture_monitor_scheduled_runs (
                  scheduled_run_id TEXT PRIMARY KEY,
                  scope_id TEXT NOT NULL,
                  schedule_ref TEXT NOT NULL,
                  snapshot_ref TEXT NOT NULL,
                  assessment_ref TEXT NOT NULL,
                  classification_ref TEXT NOT NULL,
                  review_ref TEXT,
                  run_json TEXT NOT NULL,
                  output_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS architecture_monitor_schedules (
                  scope_id TEXT NOT NULL,
                  schedule_ref TEXT NOT NULL,
                  schedule_json TEXT NOT NULL,
                  cron_job_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(scope_id, schedule_ref)
                );

CREATE TABLE IF NOT EXISTS architecture_review_decisions (
                  decision_id TEXT PRIMARY KEY,
                  review_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  from_state TEXT NOT NULL,
                  to_state TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  decision_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  signature TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  UNIQUE(organization_id, review_id, idempotency_key)
                );

CREATE TABLE IF NOT EXISTS architecture_review_packages (
                  review_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  state TEXT NOT NULL,
                  version BIGINT NOT NULL,
                  package_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS architecture_threshold_profiles (
                  profile_ref TEXT PRIMARY KEY,
                  profile_id TEXT NOT NULL,
                  profile_version TEXT NOT NULL,
                  scope_json TEXT NOT NULL,
                  effective_from TEXT NOT NULL,
                  effective_to TEXT,
                  governance_state TEXT NOT NULL,
                  profile_json TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS bank_journal_views (
              view_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              source_jvp_ids_json TEXT NOT NULL,
              coa_version TEXT NOT NULL,
              projection_bundle_version TEXT NOT NULL,
              entries_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

CREATE TABLE IF NOT EXISTS cash_journal_views (
              view_id TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              period_id TEXT NOT NULL,
              source_jvp_ids_json TEXT NOT NULL,
              coa_version TEXT NOT NULL,
              projection_bundle_version TEXT NOT NULL,
              entries_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
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

CREATE TABLE IF NOT EXISTS company_dataset_import_files (
              file_id TEXT PRIMARY KEY,
              import_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              file_role TEXT NOT NULL,
              original_filename TEXT,
              content_type TEXT,
              byte_size BIGINT NOT NULL,
              artifact_path TEXT NOT NULL,
              source_file_hash TEXT,
              normalized_file_hash TEXT,
              created_at TEXT NOT NULL
            );

CREATE TABLE IF NOT EXISTS company_dataset_import_validation_runs (
              validation_run_id TEXT PRIMARY KEY,
              import_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              parser_profile_schema_version TEXT NOT NULL,
              parser_profile_hash TEXT NOT NULL,
              validation_result_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE (organization_id, idempotency_key)
            );

CREATE TABLE IF NOT EXISTS company_dataset_imports (
              import_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              actor_user_id TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              status TEXT NOT NULL,
              parser_profile_schema_version TEXT NOT NULL,
              parser_profile_hash TEXT NOT NULL,
              parser_profile_json TEXT NOT NULL,
              validation_result_json TEXT,
              promotion_result_json TEXT,
              error_category TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              retention_until TEXT,
              UNIQUE (organization_id, idempotency_key)
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
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

CREATE TABLE IF NOT EXISTS cq_feature_vectors (
              feature_vector_id TEXT PRIMARY KEY,
              projection_id TEXT NOT NULL,
              trial_balance_view_id TEXT,
              tenant_id TEXT NOT NULL,
              cq_contract_version TEXT NOT NULL,
              features_json TEXT NOT NULL,
              source_artifacts_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

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
              claim_occurred BIGINT NOT NULL DEFAULT 0,
              claim_amount DOUBLE PRECISION NOT NULL DEFAULT 0,
              audit_adjustment_amount DOUBLE PRECISION NOT NULL DEFAULT 0,
              tax_penalty_amount DOUBLE PRECISION NOT NULL DEFAULT 0,
              correction_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
              time_to_resolution_days BIGINT,
              source_type TEXT NOT NULL,
              source_ref TEXT,
              evidence_hash TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

CREATE TABLE IF NOT EXISTS cq_risk_quotes (
              quote_id TEXT PRIMARY KEY,
              projection_id TEXT NOT NULL,
              tenant_id TEXT NOT NULL,
              cq_score DOUBLE PRECISION NOT NULL,
              cq_contract_version TEXT NOT NULL,
              calibration_model_version TEXT NOT NULL,
              calibration_status TEXT NOT NULL
                CHECK(calibration_status IN ('governance_prior', 'empirically_calibrated')),
              exposure_amount DOUBLE PRECISION NOT NULL,
              expected_frequency DOUBLE PRECISION,
              expected_severity_rate DOUBLE PRECISION,
              expected_loss DOUBLE PRECISION,
              pure_premium DOUBLE PRECISION,
              gross_premium DOUBLE PRECISION,
              liability_cap DOUBLE PRECISION,
              deductible DOUBLE PRECISION,
              eligibility_state TEXT NOT NULL
                CHECK(eligibility_state IN ('ELIGIBLE', 'ELIGIBLE_WITH_EXCLUSIONS', 'REQUIRES_REVIEW', 'DECLINED')),
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

CREATE TABLE IF NOT EXISTS cq_scores (
              cq_score_id TEXT PRIMARY KEY,
              feature_vector_id TEXT NOT NULL,
              projection_id TEXT NOT NULL,
              tenant_id TEXT NOT NULL,
              cq_contract_version TEXT NOT NULL,
              calibration_model_version TEXT NOT NULL,
              calibration_status TEXT NOT NULL
                CHECK(calibration_status IN ('governance_prior', 'empirically_calibrated')),
              score DOUBLE PRECISION NOT NULL,
              gates_json TEXT NOT NULL,
              explanation_json TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

CREATE TABLE IF NOT EXISTS cross_domain_governance_decisions (
              cross_decision_id TEXT PRIMARY KEY,
              xorder_id TEXT NOT NULL,
              decision TEXT NOT NULL,
              reviewer_id TEXT,
              justification TEXT,
              approved_limits_json TEXT,
              disclosure_package_ref TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

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
              disclosure_required BIGINT NOT NULL DEFAULT 0,
              limits_json TEXT,
              blocked_action BIGINT NOT NULL DEFAULT 0,
              blocked_export BIGINT NOT NULL DEFAULT 0,
              reason_codes_json TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

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
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

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
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

CREATE TABLE IF NOT EXISTS cross_domain_resolver_contracts (
              resolver_contract_id TEXT PRIMARY KEY,
              resolver_version TEXT NOT NULL,
              semantics_json TEXT NOT NULL,
              precedence_bindings_json TEXT NOT NULL,
              deterministic_rules_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

CREATE TABLE IF NOT EXISTS dataset_asset_version (
              dataset_asset_version_id TEXT PRIMARY KEY,
              dataset_type TEXT NOT NULL CHECK(dataset_type IN ('DEMO', 'DOUBLE PRECISION')),
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

CREATE TABLE IF NOT EXISTS effective_context_checkpoints (
                  checkpoint_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  boundary_message_id TEXT NOT NULL,
                  boundary_event_sequence BIGINT NOT NULL,
                  knowledge_state_ref TEXT NOT NULL,
                  active_contract_ref TEXT NOT NULL,
                  authority_pins_ref TEXT NOT NULL,
                  transcript_snapshot_ref TEXT NOT NULL,
                  trajectory_from_sequence BIGINT NOT NULL,
                  trajectory_to_sequence BIGINT NOT NULL,
                  parent_checkpoint_id TEXT,
                  lineage_root_checkpoint_id TEXT NOT NULL,
                  delta_ref TEXT,
                  checkpoint_hash TEXT NOT NULL UNIQUE,
                  verification_status TEXT NOT NULL,
                  verification_reason TEXT,
                  created_at TEXT NOT NULL,
                  checkpoint_json TEXT NOT NULL,
                  UNIQUE(organization_id, workspace_id, session_id, boundary_message_id)
                );

CREATE TABLE IF NOT EXISTS effective_context_eval_results (
                  result_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  checkpoint_id TEXT,
                  result_json TEXT NOT NULL,
                  result_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS effective_context_events (
                  event_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  sequence BIGINT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  state_effect TEXT NOT NULL,
                  input_refs_json TEXT NOT NULL,
                  output_refs_json TEXT NOT NULL,
                  cause_event_ids_json TEXT NOT NULL,
                  authority_context_ref TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  prior_event_hash TEXT,
                  event_json TEXT NOT NULL,
                  UNIQUE(organization_id, workspace_id, session_id, sequence)
                );

CREATE TABLE IF NOT EXISTS effective_context_heads (
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  latest_sequence BIGINT NOT NULL,
                  prior_hash TEXT,
                  latest_checkpoint_id TEXT,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(organization_id, workspace_id, session_id)
                );

CREATE TABLE IF NOT EXISTS effective_context_idempotency (
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  operation_type TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  request_hash TEXT NOT NULL,
                  result_ref TEXT NOT NULL,
                  expires_at TEXT,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY(organization_id, workspace_id, operation_type, idempotency_key)
                );

CREATE TABLE IF NOT EXISTS effective_context_payloads (
                  payload_ref TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  payload_type TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  byte_length BIGINT NOT NULL,
                  uri TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(organization_id, workspace_id, content_hash)
                );

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

CREATE TABLE IF NOT EXISTS exceptions (
              exception_id TEXT PRIMARY KEY,
              justification_id TEXT NOT NULL,
              status TEXT NOT NULL,
              reason_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              resolved_at TEXT
            );

CREATE TABLE IF NOT EXISTS explainability_surfaces (
              explainability_surface_id TEXT PRIMARY KEY,
              event_id TEXT NOT NULL,
              surface_json TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS external_verification_manifests (
              external_verification_manifest_id TEXT PRIMARY KEY,
              subject_ref TEXT NOT NULL,
              manifest_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
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
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
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
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
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

CREATE TABLE IF NOT EXISTS graph_governance_events (
                  event_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  target_object_ref TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  event_payload_json TEXT NOT NULL,
                  event_hash TEXT NOT NULL,
                  occurred_at TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS graph_node_search (node_id TEXT, organization_id TEXT, content TEXT);

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

CREATE TABLE IF NOT EXISTS graph_projection_contexts (
                  projection_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  source_ref TEXT NOT NULL,
                  focus_ref TEXT,
                  graph_snapshot_ref TEXT,
                  as_of TEXT,
                  lens TEXT NOT NULL,
                  context_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS hermes_trajectories (
              trajectory_id TEXT PRIMARY KEY,
              trajectory_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

CREATE TABLE IF NOT EXISTS human_resolutions (
              resolution_id TEXT PRIMARY KEY,
              exception_id TEXT NOT NULL,
              resolution_json TEXT NOT NULL,
              approved_by TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

CREATE TABLE IF NOT EXISTS institutional_drift_baseline_snapshots (
                  baseline_snapshot_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  asset_ref TEXT NOT NULL,
                  metadata_asset_version_ref TEXT NOT NULL,
                  activation_set_snapshot_id TEXT NOT NULL,
                  resolved_activation_set_hash TEXT NOT NULL,
                  ontology_version_ref TEXT NOT NULL,
                  semantic_contract_version_ref TEXT NOT NULL,
                  data_contract_version_ref TEXT NOT NULL,
                  policy_version_refs_json TEXT NOT NULL,
                  source_hashes_json TEXT NOT NULL,
                  effective_at TEXT NOT NULL,
                  baseline_snapshot_hash TEXT NOT NULL,
                  lifecycle_state TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS institutional_drift_findings (
                  finding_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  drift_family TEXT NOT NULL,
                  drift_subtype TEXT NOT NULL,
                  baseline_snapshot_id TEXT NOT NULL,
                  observation_id TEXT NOT NULL,
                  deterministic_differences_json TEXT NOT NULL,
                  semantic_hypotheses_json TEXT NOT NULL,
                  evidence_completeness TEXT NOT NULL,
                  authority_impact_json TEXT NOT NULL,
                  selected_learning_route TEXT NOT NULL,
                  reason_codes_json TEXT NOT NULL,
                  severity TEXT NOT NULL,
                  urgency TEXT NOT NULL,
                  lifecycle_state TEXT NOT NULL,
                  finding_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS institutional_drift_learning_signals (
                  signal_id TEXT PRIMARY KEY,
                  finding_id TEXT NOT NULL,
                  signal_type TEXT NOT NULL,
                  lifecycle_state TEXT NOT NULL,
                  candidate_ref TEXT,
                  approval_ref TEXT,
                  activation_ref TEXT,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS institutional_drift_observations (
                  observation_id TEXT PRIMARY KEY,
                  idempotency_key TEXT NOT NULL UNIQUE,
                  policy_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  asset_ref TEXT NOT NULL,
                  observation_window TEXT NOT NULL,
                  normalized_payload_json TEXT NOT NULL,
                  source_anchors_json TEXT NOT NULL,
                  collector_version TEXT NOT NULL,
                  observation_hash TEXT NOT NULL,
                  prior_observation_id TEXT,
                  complete BIGINT NOT NULL CHECK(complete IN (0, 1)),
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS institutional_drift_recovery_verifications (
                  verification_id TEXT PRIMARY KEY,
                  finding_id TEXT NOT NULL,
                  remediation_ref TEXT NOT NULL,
                  deployed_artifact_refs_json TEXT NOT NULL,
                  pre_observation_hash TEXT NOT NULL,
                  post_observation_hash TEXT NOT NULL,
                  replay_comparison_result TEXT NOT NULL,
                  non_interference_result TEXT NOT NULL,
                  expected_outcome TEXT NOT NULL,
                  observed_outcome TEXT NOT NULL,
                  regression_result TEXT NOT NULL,
                  verification_status TEXT NOT NULL,
                  evidence_json TEXT NOT NULL,
                  verification_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS institutional_drift_routing_events (
                  event_id TEXT PRIMARY KEY,
                  finding_id TEXT NOT NULL,
                  from_state TEXT,
                  to_state TEXT NOT NULL,
                  route TEXT NOT NULL,
                  reason_codes_json TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  event_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS institutional_drift_runs (
                  run_id TEXT PRIMARY KEY,
                  idempotency_key TEXT NOT NULL UNIQUE,
                  policy_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  baseline_snapshot_id TEXT NOT NULL,
                  observation_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  lease_owner TEXT,
                  lease_expires_at TEXT,
                  created_at TEXT NOT NULL,
                  completed_at TEXT
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
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

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

CREATE TABLE IF NOT EXISTS journal_entries (
              journal_id TEXT PRIMARY KEY,
              justification_id TEXT NOT NULL,
              je_json TEXT NOT NULL,
              status TEXT NOT NULL,
              posted_at TEXT
            );

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
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

CREATE TABLE IF NOT EXISTS justifications (
              justification_id TEXT PRIMARY KEY,
              workflow_id TEXT NOT NULL,
              rea_id TEXT NOT NULL,
              justification_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS ledger_projections (
              ledger_id               TEXT PRIMARY KEY,
              rea_id                  TEXT NOT NULL,
              coa_code                TEXT NOT NULL,
              amount                  DOUBLE PRECISION NOT NULL,
              currency                TEXT NOT NULL,
              debit                   DOUBLE PRECISION,
              credit                  DOUBLE PRECISION,
              projection_status       TEXT NOT NULL DEFAULT 'PENDING'
                CHECK(projection_status IN ('PENDING', 'PROJECTED', 'EXCEPTION', 'RESTATED')),
              projection_context_hash TEXT,
              created_at              TEXT NOT NULL,
              projected_at            TEXT
            , projection_scope TEXT, projection_bundle_version TEXT, coa_bundle_version TEXT, knowledge_context_version TEXT, constraint_bundle_version TEXT, tag_ontology_version TEXT, precedence_graph_version TEXT, resolution_rules_version TEXT);

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

CREATE TABLE IF NOT EXISTS legal_execution_manifests (
              manifest_hash TEXT PRIMARY KEY,
              manifest_json TEXT NOT NULL,
              created_at TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS legal_runtime_pin_artifacts (
              artifact_ref TEXT PRIMARY KEY,
              artifact_type TEXT NOT NULL,
              artifact_hash TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              lifecycle_state TEXT NOT NULL,
              created_at TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS legal_source_registry_snapshots (
              snapshot_id TEXT PRIMARY KEY,
              source_version_ids_json TEXT NOT NULL,
              snapshot_hash TEXT NOT NULL,
              certification_event_ids_json TEXT NOT NULL,
              lifecycle_state TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

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

CREATE TABLE IF NOT EXISTS metadata_drift_monitor_policies (
                  policy_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  asset_selector_json TEXT NOT NULL,
                  monitored_dimensions_json TEXT NOT NULL,
                  cadence TEXT NOT NULL,
                  event_subscription_mode TEXT NOT NULL,
                  baseline_resolution_mode TEXT NOT NULL,
                  severity_thresholds_json TEXT NOT NULL,
                  allowed_self_healing_level TEXT NOT NULL,
                  owner_role TEXT NOT NULL,
                  reviewer_role TEXT NOT NULL,
                  governance_state TEXT NOT NULL,
                  policy_version TEXT NOT NULL,
                  effective_from TEXT NOT NULL,
                  effective_to TEXT,
                  policy_hash TEXT NOT NULL,
                  created_by TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS metadata_drift_monitor_schedules (
                  schedule_id TEXT PRIMARY KEY,
                  policy_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  cadence TEXT NOT NULL,
                  cron_ref TEXT NOT NULL,
                  status TEXT NOT NULL,
                  schedule_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(policy_id, status)
                );

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
              updated_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
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

CREATE TABLE IF NOT EXISTS organization_dataset_versions (
              dataset_version_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              import_id TEXT NOT NULL,
              artifact_hash_manifest_json TEXT NOT NULL,
              activated_at TEXT,
              activated_by TEXT,
              superseded_by TEXT,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

CREATE TABLE IF NOT EXISTS policy_assignments (
              id TEXT PRIMARY KEY,
              justification_id TEXT,
              context_json TEXT,
              active_policy TEXT,
              shadow_policies TEXT,
              created_at TEXT
            );

CREATE TABLE IF NOT EXISTS policy_candidate_events (
              candidate_event_id TEXT PRIMARY KEY,
              candidate_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              event_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

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
              rollback_window_hours BIGINT,
              rollback_strategy TEXT,
              created_at      TEXT NOT NULL,
              approved_at     TEXT,
              notes           TEXT,
              review_status   TEXT,
              jurisdiction_reason TEXT,
              imported_legacy BIGINT NOT NULL DEFAULT 0
            );

CREATE TABLE IF NOT EXISTS policy_proposals (
              proposal_id TEXT PRIMARY KEY,
              rule_change TEXT,
              source TEXT,
              confidence DOUBLE PRECISION,
              status TEXT,
              created_at TEXT
            );

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
              confidence DOUBLE PRECISION,
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

CREATE TABLE IF NOT EXISTS policy_stats (
              id TEXT PRIMARY KEY,
              context_key TEXT,
              policy TEXT,
              count BIGINT,
              reward DOUBLE PRECISION,
              created_at TEXT
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

CREATE TABLE IF NOT EXISTS rea_claims (
              rea_id TEXT PRIMARY KEY,
              workflow_id TEXT NOT NULL,
              -- Version scoping is intentionally claim-json-owned for backward
              -- compatibility with existing append-only REA rows. DOUBLE PRECISION claims
              -- must carry claim_json.dataset_version_id; readers must parse
              -- that field and fail closed for unversioned DOUBLE PRECISION claims.
              claim_json TEXT NOT NULL,
              claim_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

CREATE TABLE IF NOT EXISTS real_company_admission_exceptions (
              exception_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              dataset_version_id TEXT NOT NULL,
              source_row_hash TEXT NOT NULL,
              admission_profile_hash TEXT NOT NULL,
              reason_code TEXT NOT NULL,
              details_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              UNIQUE (organization_id, dataset_version_id, source_row_hash, admission_profile_hash)
            );

CREATE TABLE IF NOT EXISTS real_company_setup_events (
              event_id TEXT PRIMARY KEY,
              setup_run_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              dataset_version_id TEXT NOT NULL,
              stage TEXT NOT NULL,
              event_type TEXT NOT NULL,
              event_status TEXT NOT NULL,
              details_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            );

CREATE TABLE IF NOT EXISTS real_company_setup_runs (
              setup_run_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              dataset_version_id TEXT NOT NULL,
              requested_by TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              status TEXT NOT NULL,
              current_stage TEXT,
              blocking_reason TEXT,
              lock_owner TEXT,
              lock_acquired_at TEXT,
              lock_released_at TEXT,
              started_at TEXT NOT NULL,
              completed_at TEXT,
              updated_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              UNIQUE (organization_id, dataset_version_id, idempotency_key)
            );

CREATE TABLE IF NOT EXISTS real_company_source_rows (
              source_row_id TEXT PRIMARY KEY,
              organization_id TEXT NOT NULL,
              dataset_version_id TEXT NOT NULL,
              import_id TEXT,
              file_id TEXT,
              source_row_hash TEXT NOT NULL,
              source_file_hash TEXT,
              parser_profile_hash TEXT,
              field_mapping_profile_hash TEXT,
              classification_status TEXT,
              row_type TEXT,
              row_json TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              UNIQUE (organization_id, dataset_version_id, source_row_hash)
            );

CREATE TABLE IF NOT EXISTS reinsurance_portfolio_manifests (
              manifest_id TEXT PRIMARY KEY,
              portfolio_id TEXT NOT NULL,
              period_start TEXT NOT NULL,
              period_end TEXT NOT NULL,
              cq_contract_version TEXT NOT NULL,
              calibration_model_version TEXT NOT NULL,
              insured_projection_count BIGINT NOT NULL,
              total_exposure_amount DOUBLE PRECISION NOT NULL,
              cq_distribution_json TEXT NOT NULL,
              expected_loss_total DOUBLE PRECISION,
              gross_premium_total DOUBLE PRECISION,
              reinsurance_layers_json TEXT NOT NULL,
              external_verification_manifest_refs_json TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

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

CREATE TABLE IF NOT EXISTS resolution_outcomes (
              outcome_id TEXT PRIMARY KEY,
              dimension TEXT,
              drift_type TEXT,
              action TEXT,
              success BIGINT,
              impact_delta DOUBLE PRECISION,
              created_at TEXT
            );

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

CREATE TABLE IF NOT EXISTS semantic_projection_inputs (
            org_id TEXT NOT NULL,
            input_id TEXT NOT NULL,
            version BIGINT NOT NULL,
            input_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (org_id, input_id, version)
        );

CREATE TABLE IF NOT EXISTS semantier_schema_versions (
              schema_name TEXT PRIMARY KEY,
              schema_version BIGINT NOT NULL,
              updated_at TEXT NOT NULL
            );

CREATE TABLE IF NOT EXISTS sensitive_governance_assessments (
                  assessment_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  session_id TEXT,
                  contract_version TEXT NOT NULL,
                  state TEXT NOT NULL,
                  source_kind TEXT NOT NULL,
                  purpose TEXT NOT NULL,
                  destination TEXT NOT NULL,
                  request_hash TEXT NOT NULL,
                  assessment_hash TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  prepared_payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  consumed_at TEXT,
                  consumed_by_run_id TEXT,
                  UNIQUE(organization_id, idempotency_key)
                );

CREATE TABLE IF NOT EXISTS sensitive_governance_bundle_activations (
                  activation_id TEXT PRIMARY KEY,
                  bundle_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  manifest_hash TEXT NOT NULL,
                  artifact_hashes_json TEXT NOT NULL,
                  activated_at TEXT NOT NULL,
                  UNIQUE(bundle_id, organization_id, manifest_hash)
                );

CREATE TABLE IF NOT EXISTS sensitive_governance_capability_grants (
                  grant_ref TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  principal_id TEXT NOT NULL,
                  capability TEXT NOT NULL,
                  resource_scope_json TEXT NOT NULL,
                  workflow_scope_json TEXT NOT NULL,
                  purpose_scope TEXT NOT NULL,
                  status TEXT NOT NULL,
                  granted_at TEXT NOT NULL,
                  revoked_at TEXT,
                  UNIQUE(organization_id, principal_id, capability, resource_scope_json, purpose_scope)
                );

CREATE TABLE IF NOT EXISTS sensitive_governance_dispatch_outbox (
                  dispatch_id TEXT PRIMARY KEY,
                  aggregate_id TEXT NOT NULL,
                  adapter TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  status TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  payload_hash TEXT NOT NULL,
                  claimed_by TEXT,
                  claimed_at TEXT,
                  created_at TEXT NOT NULL,
                  UNIQUE(adapter, idempotency_key)
                );

CREATE TABLE IF NOT EXISTS sensitive_governance_events (
                  event_id TEXT PRIMARY KEY,
                  aggregate_id TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  replay_phase TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  event_hash TEXT NOT NULL,
                  causation_id TEXT
                );

CREATE TABLE IF NOT EXISTS sensitive_governance_response_authority_objects (
                  object_ref TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  assessment_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  object_type TEXT NOT NULL,
                  object_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS sensitive_governance_reveal_decisions (
                  decision_id TEXT PRIMARY KEY,
                  operation TEXT NOT NULL,
                  response_id TEXT NOT NULL,
                  segment_id TEXT NOT NULL,
                  run_id TEXT,
                  organization_id TEXT NOT NULL,
                  principal_id TEXT NOT NULL,
                  allowed BIGINT NOT NULL,
                  reason_codes_json TEXT NOT NULL,
                  lease_id TEXT,
                  payload_json TEXT NOT NULL,
                  decision_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(organization_id, operation, response_id, segment_id, principal_id)
                );

CREATE TABLE IF NOT EXISTS sensitive_governance_reveal_leases (
                  lease_id TEXT PRIMARY KEY,
                  response_id TEXT NOT NULL,
                  segment_id TEXT NOT NULL,
                  run_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  principal_id TEXT NOT NULL,
                  purpose TEXT NOT NULL,
                  state TEXT NOT NULL,
                  max_uses BIGINT NOT NULL,
                  use_count BIGINT NOT NULL,
                  issued_at TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  policy_hash TEXT NOT NULL,
                  profile_hash TEXT NOT NULL,
                  clear_value_json TEXT NOT NULL,
                  decision_hash TEXT NOT NULL,
                  UNIQUE(organization_id, response_id, segment_id, principal_id, purpose)
                );

CREATE TABLE IF NOT EXISTS shadow_evaluations (
              id TEXT PRIMARY KEY,
              justification_id TEXT,
              active_status TEXT,
              shadow_json TEXT,
              created_at TEXT
            );

CREATE TABLE IF NOT EXISTS signals (
              signal_id TEXT PRIMARY KEY,
              signal_type TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
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
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
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
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

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
              created_at TEXT DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
            );

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

CREATE TABLE IF NOT EXISTS trial_balance_validation_contracts (
              contract_version TEXT PRIMARY KEY,
              org_id TEXT NOT NULL,
              contract_json TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              status TEXT NOT NULL
                CHECK(status IN ('CANDIDATE', 'ACTIVE', 'DEPRECATED')),
              created_at TEXT NOT NULL
            );

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

CREATE TABLE IF NOT EXISTS validation_results (
              validation_id TEXT PRIMARY KEY,
              justification_id TEXT NOT NULL,
              status TEXT NOT NULL,
              result_json TEXT NOT NULL,
              policy_bundle_version TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS wiki_mutation_attempts (
                  sequence BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, workspace TEXT NOT NULL,
                  operation_id TEXT NOT NULL, produced_at TEXT NOT NULL, state TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS wiki_mutation_operations (
                  workspace TEXT NOT NULL, operation_id TEXT NOT NULL, payload TEXT NOT NULL,
                  PRIMARY KEY(workspace,operation_id));

CREATE TABLE IF NOT EXISTS wiki_task_bindings (
                  workspace TEXT NOT NULL, session TEXT NOT NULL, actor TEXT NOT NULL,
                  selected_at TEXT NOT NULL, PRIMARY KEY(workspace,session));

CREATE TABLE IF NOT EXISTS workflows (
              workflow_id TEXT PRIMARY KEY,
              signal_id TEXT NOT NULL,
              workflow_type TEXT NOT NULL,
              workflow_json TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

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
              confidence DOUBLE PRECISION,
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
              confidence DOUBLE PRECISION,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(subject_id) REFERENCES graph_nodes(node_id),
              FOREIGN KEY(object_id) REFERENCES graph_nodes(node_id)
            );

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
              attempt_count BIGINT NOT NULL DEFAULT 0,
              max_attempts BIGINT NOT NULL DEFAULT 3,
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

CREATE TABLE IF NOT EXISTS legal_monitor_policies (
              policy_version_ref TEXT PRIMARY KEY,
              source_id TEXT NOT NULL,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              source_domain TEXT NOT NULL,
              source_ref TEXT NOT NULL,
              cadence TEXT NOT NULL,
              interval_seconds BIGINT NOT NULL,
              schedule_anchor_utc TEXT NOT NULL,
              priority BIGINT NOT NULL,
              monitor_policy_version TEXT NOT NULL,
              official_domain_allowlist_json TEXT NOT NULL,
              redirect_policy TEXT NOT NULL,
              max_artifact_bytes BIGINT NOT NULL,
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

CREATE TABLE IF NOT EXISTS legal_source_refresh_schedules (
              schedule_id TEXT PRIMARY KEY,
              source_id TEXT NOT NULL,
              interval_seconds BIGINT NOT NULL,
              next_check_at TEXT NOT NULL,
              operator_ref TEXT NOT NULL,
              lifecycle_state TEXT NOT NULL,
              priority BIGINT NOT NULL DEFAULT 100,
              retry_after TEXT,
              last_result_class TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(source_id) REFERENCES legal_sources(source_id),
              UNIQUE(source_id, lifecycle_state)
            );

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

CREATE TABLE IF NOT EXISTS metadata_drift_policy_events (
                  event_id TEXT PRIMARY KEY,
                  policy_id TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  actor_role TEXT NOT NULL,
                  from_state TEXT,
                  to_state TEXT NOT NULL,
                  evidence_hash TEXT NOT NULL,
                  event_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(policy_id) REFERENCES metadata_drift_monitor_policies(policy_id)
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

CREATE TABLE IF NOT EXISTS sensitive_governance_authority_objects (
                  object_ref TEXT PRIMARY KEY,
                  assessment_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  object_type TEXT NOT NULL,
                  object_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(assessment_id) REFERENCES sensitive_governance_assessments(assessment_id)
                );

CREATE TABLE IF NOT EXISTS sensitive_governance_capability_revocations (
                  revocation_ref TEXT PRIMARY KEY,
                  grant_ref TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  principal_id TEXT NOT NULL,
                  capability TEXT NOT NULL,
                  revoked_at TEXT NOT NULL,
                  reason_code TEXT NOT NULL,
                  FOREIGN KEY(grant_ref) REFERENCES sensitive_governance_capability_grants(grant_ref)
                );

CREATE TABLE IF NOT EXISTS sensitive_governance_escalation_cases (
                  escalation_id TEXT PRIMARY KEY,
                  assessment_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  assigned_role TEXT NOT NULL,
                  assigned_principal TEXT,
                  status TEXT NOT NULL,
                  reason_codes_json TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(assessment_id) REFERENCES sensitive_governance_assessments(assessment_id)
                );

CREATE TABLE IF NOT EXISTS sensitive_governance_restricted_payloads (
                  restricted_payload_ref TEXT PRIMARY KEY,
                  assessment_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  classification TEXT NOT NULL,
                  clear_value_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(assessment_id) REFERENCES sensitive_governance_assessments(assessment_id)
                );

CREATE TABLE IF NOT EXISTS sensitive_governance_runs (
                  run_id TEXT PRIMARY KEY,
                  assessment_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  contract_version TEXT NOT NULL,
                  state TEXT NOT NULL,
                  execution_idempotency_key TEXT NOT NULL,
                  governed_response_json TEXT NOT NULL,
                  governed_response_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  completed_at TEXT,
                  UNIQUE(organization_id, execution_idempotency_key),
                  FOREIGN KEY(assessment_id) REFERENCES sensitive_governance_assessments(assessment_id)
                );

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
              confidence DOUBLE PRECISION NOT NULL,
              suggested_replacement TEXT,
              escalation_flag BIGINT NOT NULL DEFAULT 0,
              resolver_evidence_ref TEXT NOT NULL,
              target_anchor_ref TEXT,
              target_element_kind TEXT,
              target_element_id TEXT,
              normalized_match TEXT,
              start_offset BIGINT,
              end_offset BIGINT,
              occurrence_index BIGINT,
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

CREATE TABLE IF NOT EXISTS tender_detection_runtime_conflicts (
              conflict_id TEXT PRIMARY KEY,
              run_id TEXT NOT NULL,
              conflict_type TEXT NOT NULL,
              target_anchor_ref TEXT,
              rule_version_ids_json TEXT NOT NULL,
              start_offset BIGINT,
              end_offset BIGINT,
              conflict_payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(run_id) REFERENCES tender_detection_runs(run_id)
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

CREATE TABLE IF NOT EXISTS graph_snapshot_assertions (
              graph_version_id TEXT NOT NULL,
              assertion_id TEXT NOT NULL,
              ordinal BIGINT NOT NULL,
              PRIMARY KEY(graph_version_id, assertion_id),
              FOREIGN KEY(graph_version_id) REFERENCES graph_snapshots(graph_version_id),
              FOREIGN KEY(assertion_id) REFERENCES graph_assertions(assertion_id)
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

CREATE TABLE IF NOT EXISTS knowledge_context_inclusion_preference (
              preference_id TEXT PRIMARY KEY,
              activation_id TEXT NOT NULL,
              principal_id TEXT,
              role_id TEXT,
              organization_id TEXT NOT NULL,
              workspace_id TEXT,
              preference_scope TEXT NOT NULL
                CHECK(preference_scope IN ('principal', 'role', 'workspace_default')),
              retrieval_enabled BIGINT NOT NULL CHECK(retrieval_enabled IN (0, 1)),
              prompt_context_enabled BIGINT NOT NULL CHECK(prompt_context_enabled IN (0, 1)),
              query_context_enabled BIGINT NOT NULL CHECK(query_context_enabled IN (0, 1)),
              preference_event_head_hash TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(activation_id) REFERENCES knowledge_source_activation(activation_id)
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
              attempt_count BIGINT NOT NULL DEFAULT 0,
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

CREATE TABLE IF NOT EXISTS sensitive_governance_escalation_reviews (
                  review_ref TEXT PRIMARY KEY,
                  escalation_id TEXT NOT NULL,
                  assessment_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  reviewer_principal TEXT NOT NULL,
                  decision TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  review_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(organization_id, idempotency_key),
                  FOREIGN KEY(escalation_id) REFERENCES sensitive_governance_escalation_cases(escalation_id),
                  FOREIGN KEY(assessment_id) REFERENCES sensitive_governance_assessments(assessment_id)
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
              event_sequence BIGINT NOT NULL,
              actor TEXT NOT NULL,
              actor_authority TEXT NOT NULL,
              reason TEXT,
              prior_event_hash TEXT,
              content_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(activation_id) REFERENCES governed_rule_activation(activation_id),
              UNIQUE(activation_stream_id, event_sequence)
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

CREATE TABLE IF NOT EXISTS legal_activation_events (
              activation_event_id TEXT PRIMARY KEY,
              source_registry_snapshot_id TEXT NOT NULL,
              knowledge_context_version_id TEXT NOT NULL,
              authority_bundle_version_id TEXT NOT NULL,
              activation_performed BIGINT NOT NULL CHECK(activation_performed IN (0, 1)),
              approver_refs_json TEXT NOT NULL,
              content_hashes_json TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(source_registry_snapshot_id) REFERENCES legal_source_registry_snapshots(snapshot_id),
              FOREIGN KEY(knowledge_context_version_id) REFERENCES legal_knowledge_context_versions(knowledge_context_version_id),
              FOREIGN KEY(authority_bundle_version_id) REFERENCES legal_authority_bundle_versions(authority_bundle_version_id),
              UNIQUE(idempotency_key)
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

CREATE TABLE IF NOT EXISTS audit_path_projection (
              projection_id TEXT PRIMARY KEY,
              authority_version_id TEXT NOT NULL,
              lineage_json TEXT NOT NULL,
              evaluation_evidence_json TEXT NOT NULL,
              source_authority_hash TEXT NOT NULL,
              non_authoritative BIGINT NOT NULL DEFAULT 1,
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
              non_authoritative BIGINT NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(authority_version_id) REFERENCES runtime_semantic_authority_versions(authority_version_id)
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

CREATE TABLE IF NOT EXISTS runtime_control_index (
              index_entry_id TEXT PRIMARY KEY,
              authority_version_id TEXT NOT NULL,
              control_family TEXT NOT NULL,
              control_json TEXT NOT NULL,
              source_authority_hash TEXT NOT NULL,
              non_authoritative BIGINT NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(authority_version_id) REFERENCES runtime_semantic_authority_versions(authority_version_id)
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
              non_authoritative BIGINT NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              FOREIGN KEY(authority_version_id) REFERENCES runtime_semantic_authority_versions(authority_version_id)
            );

CREATE INDEX IF NOT EXISTS idx_accounting_archive_packages_period
              ON accounting_archive_packages(org_id, period_id);

CREATE INDEX IF NOT EXISTS idx_audit_path_authority
              ON audit_path_projection(authority_version_id);

CREATE INDEX IF NOT EXISTS idx_candidate_events_candidate
              ON policy_candidate_events(candidate_id, created_at);

CREATE INDEX IF NOT EXISTS idx_candidates_category
              ON policy_candidates(category);

CREATE INDEX IF NOT EXISTS idx_candidates_hash
              ON policy_candidates(content_hash);

CREATE INDEX IF NOT EXISTS idx_candidates_name
              ON policy_candidates(name);

CREATE INDEX IF NOT EXISTS idx_candidates_status
              ON policy_candidates(status);

CREATE INDEX IF NOT EXISTS idx_canonical_term_candidates_run
              ON canonical_term_candidates(discovery_run_id, created_at);

CREATE INDEX IF NOT EXISTS idx_company_dataset_import_files_import
              ON company_dataset_import_files(import_id, file_role);

CREATE INDEX IF NOT EXISTS idx_company_dataset_imports_org
              ON company_dataset_imports(organization_id, updated_at);

CREATE INDEX IF NOT EXISTS idx_concept_neighborhood_authority
              ON concept_neighborhood_index(authority_version_id);

CREATE INDEX IF NOT EXISTS idx_cq_feature_vectors_projection
              ON cq_feature_vectors(projection_id);

CREATE INDEX IF NOT EXISTS idx_cq_outcomes_projection
              ON cq_outcomes(projection_id);

CREATE INDEX IF NOT EXISTS idx_cq_risk_quotes_projection
              ON cq_risk_quotes(projection_id);

CREATE INDEX IF NOT EXISTS idx_cq_scores_projection
              ON cq_scores(projection_id, created_at);

CREATE INDEX IF NOT EXISTS idx_cross_domain_decisions_xorder
              ON cross_domain_governance_decisions(xorder_id);

CREATE INDEX IF NOT EXISTS idx_cross_domain_order_results_org_period
              ON cross_domain_order_results(org_id, period_id);

CREATE INDEX IF NOT EXISTS idx_cross_domain_policies_version
              ON cross_domain_policies(policy_version);

CREATE INDEX IF NOT EXISTS idx_cross_domain_replay_xorder
              ON cross_domain_replay_bindings(xorder_id);

CREATE INDEX IF NOT EXISTS idx_cross_domain_resolver_version
              ON cross_domain_resolver_contracts(resolver_version);

CREATE INDEX IF NOT EXISTS idx_effective_context_checkpoints_boundary
                  ON effective_context_checkpoints(organization_id, workspace_id, session_id, boundary_event_sequence);

CREATE INDEX IF NOT EXISTS idx_effective_context_checkpoints_parent
                  ON effective_context_checkpoints(organization_id, workspace_id, parent_checkpoint_id);

CREATE INDEX IF NOT EXISTS idx_effective_context_eval_results_session
                  ON effective_context_eval_results(organization_id, workspace_id, session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_effective_context_events_occurred
                  ON effective_context_events(organization_id, workspace_id, occurred_at);

CREATE INDEX IF NOT EXISTS idx_effective_context_events_session_type
                  ON effective_context_events(organization_id, workspace_id, session_id, event_type);

CREATE INDEX IF NOT EXISTS idx_effective_context_idempotency_expires
                  ON effective_context_idempotency(expires_at);

CREATE INDEX IF NOT EXISTS idx_financial_statement_packages_period
              ON financial_statement_packages(org_id, period_id);

CREATE INDEX IF NOT EXISTS idx_governed_rule_activation_event_activation
              ON governed_rule_activation_event(activation_id, effective_at, event_sequence);

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

CREATE INDEX IF NOT EXISTS idx_governed_rule_activation_rule
              ON governed_rule_activation(rule_version_id);

CREATE INDEX IF NOT EXISTS idx_governed_rule_activation_scope
              ON governed_rule_activation(organization_id, activation_scope, workspace_id, effective_from);

CREATE INDEX IF NOT EXISTS idx_governed_rule_versions_candidate
              ON governed_rule_versions(rule_candidate_id);

CREATE INDEX IF NOT EXISTS idx_governed_rule_versions_family_state
              ON governed_rule_versions(rule_family, rule_state, created_at);

CREATE INDEX IF NOT EXISTS idx_governed_rule_versions_owner
              ON governed_rule_versions(
                organization_id,
                authority_domain,
                semantic_tier,
                jurisdiction
              );

CREATE INDEX IF NOT EXISTS idx_graph_assertion_events_stream
              ON graph_assertion_events(assertion_id, occurred_at);

CREATE INDEX IF NOT EXISTS idx_graph_assertions_class_state
              ON graph_assertions(organization_id, assertion_class, governance_state);

CREATE INDEX IF NOT EXISTS idx_graph_assertions_effective
              ON graph_assertions(organization_id, valid_from, valid_to);

CREATE INDEX IF NOT EXISTS idx_graph_assertions_object
              ON graph_assertions(organization_id, object_id, predicate, governance_state);

CREATE INDEX IF NOT EXISTS idx_graph_assertions_subject
              ON graph_assertions(organization_id, subject_id, predicate, governance_state);

CREATE INDEX IF NOT EXISTS idx_graph_assertions_subject_ordered
              ON graph_assertions(organization_id, subject_id, predicate, governance_state, object_id);

CREATE INDEX IF NOT EXISTS idx_graph_governance_events_target
                  ON graph_governance_events(organization_id, target_object_ref, occurred_at);

CREATE INDEX IF NOT EXISTS idx_graph_index_freshness_status
          ON graph_index_freshness(organization_id, status, updated_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_ingestion_active_source_fingerprint
          ON graph_ingestion_runs(organization_id, source_ref, requested_fingerprint)
          WHERE status IN ('claimed', 'indexing');

CREATE INDEX IF NOT EXISTS idx_graph_ingestion_status_retry
          ON graph_ingestion_runs(organization_id, status, next_retry_at);

CREATE INDEX IF NOT EXISTS idx_graph_projection_contexts_org_workspace
                  ON graph_projection_contexts(organization_id, workspace_id, updated_at);

CREATE INDEX IF NOT EXISTS idx_graph_snapshot_assertions_assertion
              ON graph_snapshot_assertions(assertion_id, graph_version_id);

CREATE INDEX IF NOT EXISTS idx_insurance_eligibility_projection
              ON insurance_eligibility_results(projection_id);

CREATE INDEX IF NOT EXISTS idx_jvp_period
              ON journal_voucher_projections(org_id, period_id);

CREATE INDEX IF NOT EXISTS idx_kb_builder_runs_org_created
              ON knowledge_builder_discovery_runs(organization_id, started_at);

CREATE INDEX IF NOT EXISTS idx_kb_curation_events_run
              ON knowledge_builder_curation_events(discovery_run_id, created_at);

CREATE INDEX IF NOT EXISTS idx_kb_eval_datasets_org
              ON knowledge_builder_evaluation_datasets(organization_id, created_at);

CREATE INDEX IF NOT EXISTS idx_kb_eval_examples_dataset
              ON knowledge_builder_evaluation_examples(evaluation_dataset_id);

CREATE INDEX IF NOT EXISTS idx_kb_eval_results_run
              ON knowledge_builder_evaluation_results(evaluation_run_id);

CREATE INDEX IF NOT EXISTS idx_kb_eval_runs_dataset
              ON knowledge_builder_evaluation_runs(evaluation_dataset_id, started_at);

CREATE INDEX IF NOT EXISTS idx_kb_rule_candidate_evidence_candidate
              ON knowledge_builder_rule_candidate_evidence(rule_candidate_id);

CREATE INDEX IF NOT EXISTS idx_kb_source_identities_lineage_order
              ON knowledge_builder_source_identities(
                organization_id, workspace_id, lineage_id, created_at,
                source_identity_ref
              );

CREATE INDEX IF NOT EXISTS idx_kb_source_identities_source
              ON knowledge_builder_source_identities(source_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_kcip_subject_activation
              ON knowledge_context_inclusion_preference(
                activation_id,
                preference_scope,
                COALESCE(principal_id, ''),
                COALESCE(role_id, ''),
                organization_id,
                COALESCE(workspace_id, '')
              );

CREATE INDEX IF NOT EXISTS idx_kcipe_preference
              ON knowledge_context_inclusion_preference_events(preference_id, occurred_at);

CREATE INDEX IF NOT EXISTS idx_kgl_approval_artifact
              ON kgl_approval_records(artifact_id);

CREATE INDEX IF NOT EXISTS idx_kgl_conflict_artifact
              ON kgl_conflict_records(artifact_id, created_at);

CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_domain
              ON knowledge_artifacts(authority_domain);

CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_source_hash
              ON knowledge_artifacts(source_hash);

CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_status
              ON knowledge_artifacts(ingestion_status);

CREATE INDEX IF NOT EXISTS idx_knowledge_source_kind_purpose
              ON knowledge_source(source_kind, semantic_purpose);

CREATE INDEX IF NOT EXISTS idx_ks_compilation_anchor_run
              ON knowledge_source_compilation_anchors(knowledge_source_compilation_run_id);

CREATE INDEX IF NOT EXISTS idx_ks_compilation_source
              ON knowledge_source_compilation_runs(knowledge_source_id, created_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ksa_active_dataset_role_scope
              ON knowledge_source_activation(organization_id, scope_key, dataset_asset_version_id)
              WHERE activation_status = 'active' AND dataset_asset_version_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_ksa_active_source_scope
              ON knowledge_source_activation(organization_id, scope_key, knowledge_source_id)
              WHERE activation_status = 'active' AND dataset_asset_version_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_ksae_source
              ON knowledge_source_activation_events(knowledge_source_id, occurred_at);

CREATE INDEX IF NOT EXISTS idx_labeled_docx_derivatives_run
              ON sensitive_labeling_labeled_docx_derivatives(run_id, generated_at);

CREATE INDEX IF NOT EXISTS idx_legal_acceptance_exports_org_run
              ON legal_acceptance_evidence_exports(organization_id, test_run_id, created_at);

CREATE INDEX IF NOT EXISTS idx_legal_activation_bundle
              ON legal_activation_events(authority_bundle_version_id, created_at);

CREATE INDEX IF NOT EXISTS idx_legal_audit_reason
              ON legal_audit_events(reason_code, correlation_id);

CREATE INDEX IF NOT EXISTS idx_legal_certification_object
              ON legal_certification_events(object_ref, object_type, created_at);

CREATE INDEX IF NOT EXISTS idx_legal_edges_from
              ON legal_source_version_edges(from_version_id, edge_type);

CREATE INDEX IF NOT EXISTS idx_legal_edges_to
              ON legal_source_version_edges(to_version_id, edge_type);

CREATE INDEX IF NOT EXISTS idx_legal_monitor_policies_resolution
              ON legal_monitor_policies(organization_id, source_ref, workspace_id, governance_state, effective_from, effective_to);

CREATE INDEX IF NOT EXISTS idx_legal_refresh_schedules_due
              ON legal_source_refresh_schedules(lifecycle_state, next_check_at);

CREATE INDEX IF NOT EXISTS idx_legal_refresh_schedules_priority_due
              ON legal_source_refresh_schedules(lifecycle_state, priority, next_check_at)
            ;

CREATE INDEX IF NOT EXISTS idx_legal_regulatory_acks_alert
              ON legal_regulatory_notification_acknowledgements(alert_id, created_at);

CREATE INDEX IF NOT EXISTS idx_legal_regulatory_notifications_org_state
              ON legal_regulatory_notifications(organization_id, lifecycle_state, created_at);

CREATE INDEX IF NOT EXISTS idx_legal_regulatory_notifications_source
              ON legal_regulatory_notifications(source_id, event_type, created_at);

CREATE INDEX IF NOT EXISTS idx_legal_scan_items_claim
              ON legal_scan_items(status, lease_expires_at, source_id);

CREATE INDEX IF NOT EXISTS idx_legal_scan_runs_org_status
              ON legal_scan_runs(organization_id, status, started_at);

CREATE INDEX IF NOT EXISTS idx_legal_sections_version_locator
              ON legal_source_sections(version_id, stable_locator);

CREATE INDEX IF NOT EXISTS idx_legal_semantic_candidates_version
              ON legal_semantic_candidates(version_id, review_state, candidate_type);

CREATE UNIQUE INDEX IF NOT EXISTS idx_legal_sources_identity
              ON legal_sources(organization_id, canonical_title, issuer, jurisdiction, COALESCE(document_number, ''))
            ;

CREATE INDEX IF NOT EXISTS idx_legal_sources_state
              ON legal_sources(organization_id, jurisdiction, authority_tier, governance_state);

CREATE INDEX IF NOT EXISTS idx_legal_versions_source_state
              ON legal_source_versions(source_id, lifecycle_state, effective_from);

CREATE UNIQUE INDEX IF NOT EXISTS idx_monthly_workflows_period
              ON monthly_accounting_workflows(org_id, period_id);

CREATE INDEX IF NOT EXISTS idx_order_maintenance_results_org_period
              ON order_maintenance_results(org_id, period_id);

CREATE INDEX IF NOT EXISTS idx_policy_rule_candidate_events_candidate
              ON policy_rule_candidate_events(rule_candidate_id, created_at);

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

CREATE INDEX IF NOT EXISTS idx_projection_ctx_hash
              ON projection_context_snapshots(projection_context_hash);

CREATE INDEX IF NOT EXISTS idx_projection_ctx_org
              ON projection_context_snapshots(org_id);

CREATE INDEX IF NOT EXISTS idx_projection_exception_status_events_exception
              ON projection_exception_status_events(projection_exception_id, created_at);

CREATE INDEX IF NOT EXISTS idx_projections_coa_code
              ON ledger_projections(coa_code);

CREATE INDEX IF NOT EXISTS idx_projections_rea_id
              ON ledger_projections(rea_id);

CREATE INDEX IF NOT EXISTS idx_projections_status
              ON ledger_projections(projection_status);

CREATE INDEX IF NOT EXISTS idx_radar_notifications_scope
              ON investment_radar_notifications(organization_id, created_at);

CREATE INDEX IF NOT EXISTS idx_radar_operator_outcomes_case
              ON investment_radar_operator_outcomes(organization_id, case_ref, created_at);

CREATE INDEX IF NOT EXISTS idx_real_company_admission_exceptions_scope
              ON real_company_admission_exceptions(organization_id, dataset_version_id);

CREATE INDEX IF NOT EXISTS idx_real_company_setup_events_scope
              ON real_company_setup_events(organization_id, dataset_version_id, stage, created_at);

CREATE INDEX IF NOT EXISTS idx_real_company_setup_runs_lock
              ON real_company_setup_runs(organization_id, dataset_version_id, status);

CREATE INDEX IF NOT EXISTS idx_real_company_source_rows_scope
              ON real_company_source_rows(organization_id, dataset_version_id, classification_status);

CREATE INDEX IF NOT EXISTS idx_reinsurance_manifests_portfolio
              ON reinsurance_portfolio_manifests(portfolio_id);

CREATE INDEX IF NOT EXISTS idx_replay_bindings_ctx_ref
              ON replay_bindings(projection_context_ref);

CREATE INDEX IF NOT EXISTS idx_replay_envelopes_scope
              ON replay_envelopes(organization_id, workspace_id, created_at);

CREATE INDEX IF NOT EXISTS idx_replay_envelopes_subject
              ON replay_envelopes(binding_type, subject_type, subject_id);

CREATE INDEX IF NOT EXISTS idx_runtime_control_authority
              ON runtime_control_index(authority_version_id);

CREATE INDEX IF NOT EXISTS idx_runtime_feedback_deltas_event
              ON runtime_feedback_candidate_deltas(feedback_event_id, created_at);

CREATE INDEX IF NOT EXISTS idx_runtime_feedback_source
              ON runtime_feedback_events(source_kind, source_run_id, created_at);

CREATE INDEX IF NOT EXISTS idx_runtime_semantic_authority_org
              ON runtime_semantic_authority_versions(organization_id, activated_at);

CREATE INDEX IF NOT EXISTS idx_semantic_relation_candidates_run
              ON semantic_relation_candidates(discovery_run_id, created_at);

CREATE INDEX IF NOT EXISTS idx_source_claim_links_claim
              ON source_claim_links(claim_ref);

CREATE INDEX IF NOT EXISTS idx_source_dependency_links_source
              ON source_dependency_links(source_ref);

CREATE INDEX IF NOT EXISTS idx_source_document_reviews_period
              ON source_document_reviews(org_id, period_id);

CREATE INDEX IF NOT EXISTS idx_source_ingestion_source
              ON source_ingestion_events(source_id, created_at);

CREATE INDEX IF NOT EXISTS idx_source_refresh_source
              ON source_refresh_checks(source_id, checked_at);

CREATE INDEX IF NOT EXISTS idx_t6_candidates_org_projection ON t6_materialization_candidates(org_id, projection_status, created_at);

CREATE INDEX IF NOT EXISTS idx_t6_candidates_org_stage ON t6_materialization_candidates(org_id, fact_maturity_stage, created_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_t6_candidates_submit_dedupe ON t6_materialization_candidates(org_id, source_channel, source_message_id, candidate_hash);

CREATE INDEX IF NOT EXISTS idx_t6_decisions_candidate
              ON t6_materialization_decisions(candidate_id, created_at);

CREATE INDEX IF NOT EXISTS idx_t6_outcomes_candidate
              ON t6_materialization_outcomes(candidate_id, created_at);

CREATE INDEX IF NOT EXISTS idx_t6_projection_exceptions_candidate
              ON t6_projection_exceptions(candidate_id, created_at);

CREATE INDEX IF NOT EXISTS idx_t6_transitions_candidate
              ON t6_materialization_transitions(candidate_id, created_at);

CREATE INDEX IF NOT EXISTS idx_tax_filing_packages_period
              ON tax_filing_packages(org_id, period_id);

CREATE INDEX IF NOT EXISTS idx_tb_feedback_signals_validation
              ON trial_balance_user_feedback_signals(validation_id);

CREATE INDEX IF NOT EXISTS idx_tb_governance_decisions_validation
              ON trial_balance_governance_decisions(validation_id);

CREATE INDEX IF NOT EXISTS idx_tb_justifications_validation
              ON trial_balance_justifications(validation_id);

CREATE INDEX IF NOT EXISTS idx_tb_validation_exceptions_validation
              ON trial_balance_validation_exceptions(validation_id);

CREATE INDEX IF NOT EXISTS idx_tb_validation_results_org_period
              ON trial_balance_validation_results(org_id, period_id);

CREATE INDEX IF NOT EXISTS idx_tb_validation_results_view
              ON trial_balance_validation_results(trial_balance_view_id);

CREATE INDEX IF NOT EXISTS idx_tb_views_org_period
              ON trial_balance_views(org_id, period_id);

CREATE INDEX IF NOT EXISTS idx_tender_detection_dispositions_run
              ON tender_detection_dispositions(run_id, decided_at);

CREATE INDEX IF NOT EXISTS idx_tender_detection_findings_run
              ON tender_detection_findings(run_id);

CREATE INDEX IF NOT EXISTS idx_tender_detection_reports_run
              ON tender_detection_reports(run_id, created_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_tender_detection_runs_idempotency
              ON tender_detection_runs(organization_id, COALESCE(workspace_id, ''), idempotency_key)
              WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tender_detection_runs_org_created
              ON tender_detection_runs(organization_id, created_at);

CREATE INDEX IF NOT EXISTS idx_tender_detection_runtime_conflicts_run
              ON tender_detection_runtime_conflicts(run_id);

CREATE INDEX IF NOT EXISTS idx_term_resolution_authority
              ON term_resolution_index(authority_version_id, term_text);

INSERT INTO semantier_schema_versions(schema_name,schema_version,updated_at) VALUES ('eos_core',2,to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS.US"Z"')), ('rule_governance',3,to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS.US"Z"')) ON CONFLICT(schema_name) DO NOTHING;

CREATE TABLE IF NOT EXISTS eos_sqlite_imports (source_store_id TEXT PRIMARY KEY, source_fingerprint TEXT NOT NULL, imported_at TEXT NOT NULL);

-- Explicit hosted migration version, separate from imported SQLite provenance.
CREATE TABLE IF NOT EXISTS eos_postgres_schema_contract (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1), asset_sha256 TEXT NOT NULL
);
