-- Explicit PostgreSQL execution authority schema, version 1.

CREATE TABLE IF NOT EXISTS semantier_execution_imports (
    source_store_id TEXT PRIMARY KEY,
    source_fingerprint TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS semantier_jobs (
            job_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            organization_id TEXT,
            state TEXT NOT NULL,
            enabled BIGINT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY(user_id, workspace_id, job_id)
        );

CREATE INDEX IF NOT EXISTS idx_semantier_jobs_owner
        ON semantier_jobs(user_id, workspace_id, updated_at);

CREATE TABLE IF NOT EXISTS semantier_cron_jobs (
            scope_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY(scope_id, job_id)
        );

CREATE TABLE IF NOT EXISTS semantier_cron_scopes (
            scope_id TEXT PRIMARY KEY,
            migrated_legacy_at TEXT,
            updated_at TEXT NOT NULL
        );

CREATE INDEX IF NOT EXISTS idx_semantier_cron_jobs_scope_updated
        ON semantier_cron_jobs(scope_id, updated_at);

CREATE TABLE IF NOT EXISTS semantier_cron_outputs (
            scope_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            run_at TEXT NOT NULL,
            output_text TEXT NOT NULL,
            PRIMARY KEY(scope_id, job_id, run_at)
        );

CREATE INDEX IF NOT EXISTS idx_semantier_cron_outputs_latest
        ON semantier_cron_outputs(scope_id, job_id, run_at DESC);

CREATE TABLE IF NOT EXISTS semantier_responsibilities (
                responsibility_id TEXT PRIMARY KEY,
                responsibility_kind TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                domain_ref TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                workflow_instance_id TEXT NOT NULL,
                lifecycle_state TEXT NOT NULL,
                continuation_disposition TEXT,
                checkpoint_version BIGINT NOT NULL DEFAULT 0,
                checkpoint_hash TEXT NOT NULL,
                current_wait_ref TEXT,
                lease_owner TEXT,
                lease_epoch BIGINT NOT NULL DEFAULT 0,
                lease_expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                schema_version BIGINT NOT NULL DEFAULT 1,
                UNIQUE(responsibility_kind, workspace_id, domain_ref, workflow_id, workflow_instance_id)
            );

CREATE INDEX IF NOT EXISTS idx_semantier_responsibilities_ready
            ON semantier_responsibilities(lifecycle_state, lease_expires_at, updated_at);

CREATE TABLE IF NOT EXISTS semantier_responsibility_checkpoints (
                responsibility_id TEXT NOT NULL,
                checkpoint_version BIGINT NOT NULL,
                lifecycle_state TEXT NOT NULL,
                continuation_disposition TEXT,
                domain_ref TEXT NOT NULL,
                workflow_ref TEXT NOT NULL,
                session_ref TEXT,
                wait_ref TEXT,
                accepted_event_cursor TEXT,
                pending_effect_refs_json TEXT NOT NULL,
                prior_checkpoint_hash TEXT,
                checkpoint_hash TEXT NOT NULL,
                produced_at TEXT NOT NULL,
                PRIMARY KEY(responsibility_id, checkpoint_version),
                FOREIGN KEY(responsibility_id) REFERENCES semantier_responsibilities(responsibility_id)
                    ON DELETE CASCADE
            );

CREATE TABLE IF NOT EXISTS semantier_responsibility_waits (
                wait_id TEXT PRIMARY KEY,
                responsibility_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                correlation_key TEXT NOT NULL,
                expected_checkpoint_version BIGINT NOT NULL,
                state TEXT NOT NULL,
                deadline_at TEXT,
                created_at TEXT NOT NULL,
                satisfied_at TEXT,
                FOREIGN KEY(responsibility_id) REFERENCES semantier_responsibilities(responsibility_id)
                    ON DELETE CASCADE,
                UNIQUE(responsibility_id, kind, source_ref, correlation_key, expected_checkpoint_version)
            );

CREATE INDEX IF NOT EXISTS idx_semantier_responsibility_waits_pending
            ON semantier_responsibility_waits(responsibility_id, state, expected_checkpoint_version);

CREATE TABLE IF NOT EXISTS semantier_responsibility_wakes (
                wake_id TEXT PRIMARY KEY,
                responsibility_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                correlation_key TEXT NOT NULL,
                dedupe_key TEXT NOT NULL UNIQUE,
                expected_checkpoint_version BIGINT NOT NULL,
                observed_at TEXT NOT NULL,
                accepted BIGINT NOT NULL,
                reason TEXT NOT NULL,
                FOREIGN KEY(responsibility_id) REFERENCES semantier_responsibilities(responsibility_id)
                    ON DELETE CASCADE
            );

CREATE TABLE IF NOT EXISTS semantier_responsibility_events (
                event_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                responsibility_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(responsibility_id) REFERENCES semantier_responsibilities(responsibility_id)
                    ON DELETE CASCADE
            );

CREATE TABLE IF NOT EXISTS semantier_durable_events (
                    workflow_type TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    accepted_sequence BIGINT NOT NULL,
                    prior_state_version BIGINT NOT NULL,
                    resulting_state_version BIGINT NOT NULL,
                    transition_kind TEXT NOT NULL,
                    schema_version BIGINT NOT NULL DEFAULT 1,
                    workflow_contract_version BIGINT NOT NULL DEFAULT 1,
                    PRIMARY KEY(workflow_type, workflow_id, event_id),
                    UNIQUE(workflow_type, workflow_id, dedupe_key),
                    UNIQUE(workflow_type, workflow_id, accepted_sequence)
                );

CREATE INDEX IF NOT EXISTS idx_semantier_durable_events_replay
                ON semantier_durable_events(
                    workflow_type, workflow_id, accepted_sequence
                );

CREATE TABLE IF NOT EXISTS semantier_effect_outbox (
                effect_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                effect_type TEXT NOT NULL,
                semantic_round BIGINT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                request_hash TEXT NOT NULL,
                canonical_payload BYTEA NOT NULL,
                state TEXT NOT NULL,
                dispatch_attempt_id TEXT,
                provider_correlation_ref TEXT,
                receipt_ref TEXT,
                failure_detail TEXT,
                schema_version BIGINT NOT NULL DEFAULT 1,
                workflow_contract_version BIGINT NOT NULL DEFAULT 1,
                effect_contract_version BIGINT NOT NULL DEFAULT 1,
                CHECK (state IN (
                    'READY','DISPATCHING','COMPLETED','FAILED_SAFE','EFFECT_UNKNOWN'
                ))
            );

CREATE INDEX IF NOT EXISTS idx_semantier_effect_outbox_ready
            ON semantier_effect_outbox(state, workflow_id, semantic_round);

CREATE TABLE IF NOT EXISTS semantier_internal_continuations (
                continuation_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                state TEXT NOT NULL,
                rejection_reason TEXT,
                schema_version BIGINT NOT NULL DEFAULT 1,
                workflow_contract_version BIGINT NOT NULL DEFAULT 1,
                continuation_payload_version BIGINT NOT NULL DEFAULT 1,
                CHECK (state IN ('PENDING','DISPATCHED','REJECTED')),
                UNIQUE(workspace_id, workflow_id, event_id, session_key, payload_hash)
            );

CREATE INDEX IF NOT EXISTS idx_semantier_internal_continuations_pending
            ON semantier_internal_continuations(state, session_key);

CREATE TABLE IF NOT EXISTS semantier_timer_occurrences (
                timer_id TEXT PRIMARY KEY,
                workflow_type TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                purpose TEXT NOT NULL,
                semantic_round BIGINT NOT NULL,
                scheduled_at TEXT NOT NULL,
                event_id TEXT NOT NULL UNIQUE,
                dedupe_key TEXT NOT NULL UNIQUE,
                state TEXT NOT NULL,
                provider_ref TEXT,
                schema_version BIGINT NOT NULL DEFAULT 1,
                workflow_contract_version BIGINT NOT NULL DEFAULT 1,
                CHECK (state IN ('SCHEDULED','ACCEPTED','CANCELLED')),
                UNIQUE(
                    workflow_type,
                    workflow_id,
                    purpose,
                    semantic_round,
                    scheduled_at
                )
            );

CREATE INDEX IF NOT EXISTS idx_semantier_timer_occurrences_pending
            ON semantier_timer_occurrences(state, scheduled_at);

CREATE TABLE IF NOT EXISTS semantier_coordination_contexts (
                coordination_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                work_source_kind TEXT NOT NULL,
                work_source_id TEXT NOT NULL,
                work_source_version TEXT,
                goal_ref TEXT NOT NULL,
                goal_revision TEXT NOT NULL,
                principal_ref TEXT NOT NULL,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                schema_version BIGINT NOT NULL DEFAULT 1,
                UNIQUE(
                    workspace_id,
                    work_source_kind,
                    work_source_id,
                    work_source_version,
                    goal_ref,
                    goal_revision
                )
            );

CREATE TABLE IF NOT EXISTS semantier_coordination_receipts (
                sequence BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                receipt_id TEXT NOT NULL UNIQUE,
                coordination_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_ref TEXT NOT NULL,
                responsibility_ref TEXT,
                agent_identity_ref TEXT,
                delegation_id TEXT,
                subagent_id TEXT,
                owner_session_ref TEXT,
                basis_refs_json TEXT NOT NULL,
                authority_ref TEXT,
                prior_receipt_hash TEXT,
                payload_json TEXT NOT NULL,
                produced_at TEXT NOT NULL,
                receipt_hash TEXT NOT NULL,
                schema_version BIGINT NOT NULL DEFAULT 1,
                UNIQUE(coordination_id, event_type, event_ref),
                FOREIGN KEY(coordination_id)
                    REFERENCES semantier_coordination_contexts(coordination_id)
                    ON DELETE RESTRICT
            );

CREATE INDEX IF NOT EXISTS idx_semantier_coordination_receipts_context
            ON semantier_coordination_receipts(coordination_id, sequence);
