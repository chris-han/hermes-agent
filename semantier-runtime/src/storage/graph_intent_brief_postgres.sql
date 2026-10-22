-- Frozen graph publication metadata; artifact files remain governed.
CREATE TABLE IF NOT EXISTS graph_intent_publication_manifests (
                manifest_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                publication_kind TEXT NOT NULL,
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                lineage_id TEXT NOT NULL,
                graph_intent_ref TEXT NOT NULL,
                graph_intent_hash TEXT NOT NULL,
                source_identity_ref TEXT NOT NULL,
                predecessor_graph_intent_ref TEXT,
                manifest_hash TEXT NOT NULL UNIQUE,
                manifest_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (organization_id, workspace_id, graph_intent_ref),
                UNIQUE (organization_id, workspace_id, graph_intent_hash)
            );
            CREATE TABLE IF NOT EXISTS graph_intent_publication_artifacts (
                manifest_id TEXT NOT NULL,
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                graph_intent_ref TEXT NOT NULL,
                artifact_role TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                artifact_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (manifest_id, artifact_role),
                FOREIGN KEY (manifest_id)
                    REFERENCES graph_intent_publication_manifests(manifest_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS graph_intent_publication_lineage_idx
                ON graph_intent_publication_manifests (
                    organization_id, workspace_id, lineage_id
                );
            CREATE INDEX IF NOT EXISTS graph_intent_publication_source_idx
                ON graph_intent_publication_manifests (
                    organization_id, workspace_id, source_identity_ref
                );
            CREATE INDEX IF NOT EXISTS graph_intent_publication_predecessor_idx
                ON graph_intent_publication_manifests (
                    organization_id, workspace_id, predecessor_graph_intent_ref
                );
