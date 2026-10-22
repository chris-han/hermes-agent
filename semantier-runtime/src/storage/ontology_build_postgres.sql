-- Frozen workspace ontology-build owner schema; explicit operator migration only.

CREATE TABLE IF NOT EXISTS advisory_trial_native_publication_index (
            organization_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            publication_kind TEXT NOT NULL CHECK (publication_kind IN ('compiler_inputs', 'initial_candidate', 'initial_validation', 'initial_plan', 'trial_candidate', 'trial_plan', 'trial_derivation', 'observed_facts')),
            artifact_ref TEXT NOT NULL,
            artifact_hash TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            input_ref TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            derivation_ref TEXT,
            derivation_hash TEXT,
            subject_ref TEXT NOT NULL,
            subject_hash TEXT NOT NULL,
            PRIMARY KEY (organization_id, workspace_id, publication_kind, artifact_ref),
            CHECK (
                (publication_kind IN ('compiler_inputs', 'initial_candidate', 'initial_validation', 'initial_plan', 'observed_facts')
                 AND derivation_ref IS NULL AND derivation_hash IS NULL)
                OR (publication_kind IN ('trial_candidate', 'trial_plan', 'trial_derivation')
                    AND derivation_ref IS NOT NULL AND derivation_hash IS NOT NULL)
            )
        );

CREATE TABLE IF NOT EXISTS hr_interpretation_heads (
            organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL, actor_id TEXT NOT NULL, session_id TEXT NOT NULL,
            model_ref TEXT NOT NULL, resume_ref TEXT NOT NULL, requirement_id TEXT NOT NULL,
            artifact_ref TEXT NOT NULL, artifact_hash TEXT NOT NULL,
            PRIMARY KEY (organization_id,workspace_id,actor_id,session_id,model_ref,resume_ref,requirement_id));

CREATE TABLE IF NOT EXISTS hr_screening_operations (
            organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL, actor_id TEXT NOT NULL, session_id TEXT NOT NULL,
            request_id TEXT NOT NULL, request_hash TEXT NOT NULL, command_json TEXT NOT NULL, status TEXT NOT NULL,
            owner_pid INTEGER NOT NULL, attempt_token TEXT NOT NULL, checkpoint_json TEXT NOT NULL, checkpoint_hash TEXT NOT NULL,
            result_ref TEXT, result_hash TEXT, stop_disposition TEXT, started_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY(organization_id,workspace_id,actor_id,session_id,request_id));

CREATE TABLE IF NOT EXISTS ontology_artifact_integrity_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                subject_ref TEXT NOT NULL,
                subject_hash TEXT NOT NULL,
                integrity_ref TEXT NOT NULL,
                integrity_hash TEXT NOT NULL,
                producer_profile TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, subject_ref),
                UNIQUE (integrity_ref)
            );

CREATE TABLE IF NOT EXISTS ontology_build_artifacts (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                artifact_ref TEXT NOT NULL,
                artifact_hash TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, artifact_ref)
            );

CREATE TABLE IF NOT EXISTS ontology_seed_request_index (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                actor_ref TEXT NOT NULL, operation TEXT NOT NULL, request_id TEXT NOT NULL,
                request_hash TEXT NOT NULL, receipt_ref TEXT NOT NULL, receipt_hash TEXT NOT NULL,
                state_ref TEXT NOT NULL, state_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, actor_ref, operation, request_id)
            );

CREATE TABLE IF NOT EXISTS ontology_seed_state_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                build_ref TEXT NOT NULL,
                revision INTEGER NOT NULL,
                state_ref TEXT NOT NULL UNIQUE,
                state_hash TEXT NOT NULL,
                predecessor_state_ref TEXT,
                predecessor_state_hash TEXT,
                PRIMARY KEY (organization_id, workspace_id, build_ref, revision)
            );

CREATE TABLE IF NOT EXISTS runtime_context_authority_request_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                request_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                artifact_kind TEXT NOT NULL,
                artifact_ref TEXT NOT NULL,
                artifact_hash TEXT NOT NULL,
                PRIMARY KEY (
                    organization_id, workspace_id, actor_id, operation, request_id
                ),
                UNIQUE (artifact_ref)
            );

CREATE TABLE IF NOT EXISTS runtime_context_heads (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                stream_ref TEXT NOT NULL,
                version INTEGER NOT NULL,
                context_ref TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, stream_ref)
            );

CREATE TABLE IF NOT EXISTS runtime_context_provider_descriptor_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                descriptor_ref TEXT NOT NULL,
                descriptor_hash TEXT NOT NULL,
                implementation_id TEXT NOT NULL,
                implementation_version TEXT NOT NULL,
                implementation_hash TEXT NOT NULL,
                configuration_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, descriptor_ref),
                UNIQUE (descriptor_ref)
            );

CREATE TABLE IF NOT EXISTS runtime_context_request_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                request_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                receipt_ref TEXT NOT NULL,
                receipt_hash TEXT NOT NULL,
                PRIMARY KEY (
                    organization_id, workspace_id, actor_id, operation, request_id
                )
            );

CREATE TABLE IF NOT EXISTS runtime_context_version_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                stream_ref TEXT NOT NULL,
                version INTEGER NOT NULL,
                context_ref TEXT NOT NULL UNIQUE,
                context_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, stream_ref, version)
            );

CREATE TABLE IF NOT EXISTS runtime_evaluation_heads (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                run_ref TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision>=0),
                state_ref TEXT NOT NULL,
                state_hash TEXT NOT NULL,
                outcome_ref TEXT NOT NULL,
                outcome_hash TEXT NOT NULL,
                PRIMARY KEY(organization_id, workspace_id, run_ref)
            );

CREATE TABLE IF NOT EXISTS runtime_evaluation_state_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                run_ref TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision>=0),
                state_ref TEXT NOT NULL,
                state_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                outcome_ref TEXT NOT NULL,
                outcome_hash TEXT NOT NULL,
                PRIMARY KEY(organization_id, workspace_id, run_ref, revision),
                UNIQUE(organization_id, workspace_id, state_ref)
            );

CREATE TABLE IF NOT EXISTS tender_review_heads (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                session_ref TEXT NOT NULL, target_ref TEXT NOT NULL, target_hash TEXT NOT NULL,
                subject_hash TEXT NOT NULL, actor_ref TEXT NOT NULL,
                record_ref TEXT NOT NULL, record_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id,workspace_id,session_ref,target_ref,target_hash,subject_hash,actor_ref)
            );

CREATE TABLE IF NOT EXISTS workbench_projection_cursors (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                session_id TEXT NOT NULL, actor_ref TEXT NOT NULL, cursor_ref TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision >= 0),
                cursor_head_ref TEXT NOT NULL UNIQUE, cursor_head_hash TEXT NOT NULL,
                projection_basis_hash TEXT NOT NULL,
                selection_ref TEXT NOT NULL, selection_hash TEXT NOT NULL,
                design_binding_set_ref TEXT NOT NULL, design_binding_set_hash TEXT NOT NULL,
                runtime_binding_set_ref TEXT NOT NULL, runtime_binding_set_hash TEXT NOT NULL,
                design_snapshot_ref TEXT, design_snapshot_hash TEXT,
                PRIMARY KEY(organization_id,workspace_id,session_id,actor_ref,cursor_ref),
                CHECK((design_snapshot_ref IS NULL) = (design_snapshot_hash IS NULL))
            );

CREATE TABLE IF NOT EXISTS workspace_artifact_binding_set_heads (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                lifecycle TEXT NOT NULL, role TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision >= 0),
                head_ref TEXT NOT NULL, head_hash TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, lifecycle, role),
                CHECK((lifecycle='DESIGN' AND role='SOURCE') OR
                      (lifecycle='RUNTIME' AND role='TARGET'))
            );

CREATE TABLE IF NOT EXISTS workspace_artifact_collection_heads (
                    organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 0),
                    head_ref TEXT NOT NULL, head_hash TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (organization_id, workspace_id)
                );

CREATE TABLE IF NOT EXISTS workspace_artifact_preparation_catalog_heads (
                    organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 0),
                    head_ref TEXT NOT NULL, head_hash TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (organization_id, workspace_id)
                );

CREATE TABLE IF NOT EXISTS workspace_artifact_preparation_heads (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                artifact_ref TEXT NOT NULL,
                purpose TEXT NOT NULL CHECK(purpose IN ('DESIGN_SOURCE', 'RUNTIME_TARGET')),
                revision INTEGER NOT NULL CHECK(revision >= 0),
                head_ref TEXT NOT NULL, head_hash TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, artifact_ref, purpose)
            );

CREATE TABLE IF NOT EXISTS workspace_artifact_preparation_operation_index (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                correlation_ref TEXT NOT NULL, operation_key TEXT NOT NULL,
                request_id TEXT NOT NULL, request_hash TEXT NOT NULL,
                claim_ref TEXT NOT NULL, claim_hash TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('PENDING', 'COMPLETE')),
                adapter_result_ref TEXT, adapter_result_hash TEXT,
                preparation_ref TEXT, preparation_hash TEXT,
                receipt_ref TEXT, receipt_hash TEXT,
                PRIMARY KEY(organization_id,workspace_id,correlation_ref),
                UNIQUE(organization_id,workspace_id,operation_key),
                CHECK((adapter_result_ref IS NULL) = (adapter_result_hash IS NULL)),
                CHECK((preparation_ref IS NULL) = (preparation_hash IS NULL)),
                CHECK((receipt_ref IS NULL) = (receipt_hash IS NULL)),
                CHECK(state != 'COMPLETE' OR
                      (adapter_result_ref IS NOT NULL AND preparation_ref IS NOT NULL
                       AND receipt_ref IS NOT NULL))
            );

CREATE TABLE IF NOT EXISTS workspace_artifact_selection_heads (
                    organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 0),
                    head_ref TEXT NOT NULL, head_hash TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (organization_id, workspace_id)
                );

CREATE TABLE IF NOT EXISTS workspace_artifacts (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                content_hash TEXT NOT NULL, artifact_ref TEXT NOT NULL UNIQUE,
                artifact_hash TEXT NOT NULL, recorded_at TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, content_hash)
            );

CREATE TABLE IF NOT EXISTS workspace_operation_request_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                actor_ref TEXT NOT NULL,
                operation TEXT NOT NULL,
                request_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                receipt_ref TEXT NOT NULL,
                receipt_hash TEXT NOT NULL,
                PRIMARY KEY (
                    organization_id, workspace_id, actor_ref, operation, request_id
                )
            );

CREATE UNIQUE INDEX IF NOT EXISTS advisory_initial_input_unique
        ON advisory_trial_native_publication_index(
            organization_id, workspace_id, publication_kind, input_ref)
        WHERE publication_kind IN ('initial_candidate', 'initial_validation', 'initial_plan')
    ;

CREATE UNIQUE INDEX IF NOT EXISTS advisory_trial_derivation_subject_unique
        ON advisory_trial_native_publication_index(organization_id, workspace_id, subject_ref)
        WHERE publication_kind = 'trial_derivation'
    ;

CREATE UNIQUE INDEX IF NOT EXISTS advisory_trial_derivation_unique
        ON advisory_trial_native_publication_index(
            organization_id, workspace_id, publication_kind, derivation_ref)
        WHERE publication_kind IN ('trial_candidate', 'trial_plan', 'trial_derivation')
    ;

CREATE UNIQUE INDEX IF NOT EXISTS ontology_build_artifact_ref_unique ON ontology_build_artifacts(artifact_ref);

CREATE UNIQUE INDEX IF NOT EXISTS ontology_seed_request_receipt_unique ON ontology_seed_request_index(receipt_ref);

CREATE UNIQUE INDEX IF NOT EXISTS runtime_context_request_receipt_unique ON runtime_context_request_index(receipt_ref);

CREATE TABLE IF NOT EXISTS learning_review_command_requests (
            organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL, actor_ref TEXT NOT NULL,
            request_id TEXT NOT NULL, request_hash TEXT NOT NULL,
            PRIMARY KEY(organization_id,workspace_id,actor_ref,request_id));

CREATE TABLE IF NOT EXISTS learning_review_presentation (
            organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL, actor_ref TEXT NOT NULL,
            notice_id TEXT NOT NULL, is_read INTEGER NOT NULL DEFAULT 0, deferred INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL, PRIMARY KEY(organization_id,workspace_id,actor_ref,notice_id));

CREATE TABLE IF NOT EXISTS learning_review_presentation_requests (
            workspace_id TEXT NOT NULL, actor_ref TEXT NOT NULL, request_id TEXT NOT NULL,
            request_hash TEXT NOT NULL, PRIMARY KEY(workspace_id,actor_ref,request_id));


            CREATE TABLE IF NOT EXISTS runtime_report_heads (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                lineage_ref TEXT NOT NULL,
                revision INTEGER NOT NULL,
                report_ref TEXT NOT NULL,
                report_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, lineage_ref)
            )
            ;
