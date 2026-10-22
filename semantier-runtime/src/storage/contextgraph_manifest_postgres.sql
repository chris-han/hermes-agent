-- ContextGraph per-domain manifests cohabit only a governed workspace auxiliary schema.
CREATE TABLE IF NOT EXISTS graph_artifacts (
                    artifact_id TEXT NOT NULL,
                    organization_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    graph_version TEXT NOT NULL,
                    artifact_kind TEXT NOT NULL CHECK (artifact_kind IN ('candidate', 'evaluation_baseline', 'authoritative')),
                    authority_state TEXT NOT NULL CHECK (authority_state IN ('candidate', 'authoritative')),
                    run_mode TEXT,
                    parent_artifact_id TEXT,
                    lineage_root_artifact_id TEXT NOT NULL,
                    lineage_id TEXT,
                    content_hash TEXT NOT NULL,
                    semantica_commit TEXT NOT NULL,
                    source_manifest_hash TEXT NOT NULL,
                    grounding_manifest_hash TEXT,
                    evaluation_manifest_hash TEXT,
                    governance_ref TEXT,
                    created_by_user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    relative_artifact_path TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL DEFAULT 'active',
                    PRIMARY KEY (organization_id,workspace_id,domain_id,artifact_id),
                    UNIQUE (organization_id, workspace_id, domain_id, graph_version)
                );

CREATE INDEX IF NOT EXISTS idx_graph_artifacts_scope ON graph_artifacts (organization_id, workspace_id, domain_id);

CREATE INDEX IF NOT EXISTS idx_graph_artifacts_lineage ON graph_artifacts (lineage_root_artifact_id);

CREATE INDEX IF NOT EXISTS idx_graph_artifacts_lineage_id ON graph_artifacts (lineage_id);

CREATE TABLE IF NOT EXISTS runtime_feedback (
                    feedback_id TEXT NOT NULL,
                    organization_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    graph_version TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    note TEXT NOT NULL,
                    source_refs_json TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (organization_id,workspace_id,domain_id,feedback_id)
                );

CREATE INDEX IF NOT EXISTS idx_runtime_feedback_artifact ON runtime_feedback (organization_id, workspace_id, domain_id, artifact_id);

CREATE TABLE IF NOT EXISTS artifact_lifecycle_events_local (
                    event_id TEXT NOT NULL,
                    organization_id TEXT NOT NULL,workspace_id TEXT NOT NULL,domain_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    origin_id TEXT,
                    operation_id TEXT,
                    actor_id TEXT NOT NULL,
                    previous_state TEXT,
                    next_state TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (organization_id,workspace_id,domain_id,event_id)
                );

CREATE INDEX IF NOT EXISTS idx_artifact_lifecycle_events_local_artifact ON artifact_lifecycle_events_local (artifact_id, created_at);
