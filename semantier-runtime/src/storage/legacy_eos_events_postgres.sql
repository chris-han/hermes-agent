-- Legacy Merkle/replay owner, explicitly hosted in the governed EOS namespace.
CREATE TABLE IF NOT EXISTS semantier_events (
            event_id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            intent_id TEXT,
            signal_ref TEXT,
            workflow_ref TEXT,
            rea_ref TEXT,
            ontology_version TEXT,
            projection_version INTEGER,
            policy_json TEXT,
            justification_json TEXT,
            execution_json TEXT,
            effects_json TEXT,
            content_hash TEXT,
            prev_event_hash TEXT,
            signature TEXT,
            created_at TEXT NOT NULL
        );

CREATE TABLE IF NOT EXISTS legacy_projection_versions (
            org_id TEXT NOT NULL,
            ontology_version TEXT NOT NULL,
            projection_version INTEGER NOT NULL,
            semantic_projection_input_json TEXT,
            semantic_scope_json TEXT,
            bindings_json TEXT,
            constraint_layer_json TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (org_id, ontology_version, projection_version)
        );

CREATE TABLE IF NOT EXISTS merkle_batches (
            batch_id TEXT PRIMARY KEY,
            root_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

CREATE TABLE IF NOT EXISTS merkle_proofs (
            event_id TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            proof_json TEXT NOT NULL,
            PRIMARY KEY(event_id, batch_id)
        );
