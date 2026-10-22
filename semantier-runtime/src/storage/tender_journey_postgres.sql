-- Explicit workspace Tender journey persistence.
CREATE TABLE IF NOT EXISTS tender_journey (
                journey_id TEXT PRIMARY KEY,
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                journey_kind TEXT NOT NULL CHECK(journey_kind='TENDER_REVIEW'),
                created_by TEXT NOT NULL,
                lifecycle_status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

CREATE TABLE IF NOT EXISTS tender_journey_transition (
                transition_id TEXT PRIMARY KEY,
                journey_id TEXT NOT NULL,
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                predecessor_revision BIGINT NOT NULL CHECK(predecessor_revision>=0),
                successor_revision BIGINT NOT NULL CHECK(successor_revision=predecessor_revision+1),
                from_phase TEXT NOT NULL,
                to_phase TEXT NOT NULL,
                cause_code TEXT NOT NULL,
                actor_ref TEXT NOT NULL,
                surface TEXT NOT NULL CHECK(surface IN ('CHAT','WORKBENCH','SYSTEM')),
                command_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(journey_id) REFERENCES tender_journey(journey_id),
                UNIQUE(journey_id, successor_revision),
                UNIQUE(journey_id, idempotency_key)
            );

CREATE TABLE IF NOT EXISTS tender_journey_head (
                journey_id TEXT PRIMARY KEY,
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                revision BIGINT NOT NULL CHECK(revision>=0),
                current_phase TEXT NOT NULL,
                lifecycle_status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                latest_transition_id TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(journey_id) REFERENCES tender_journey(journey_id)
            );

CREATE INDEX IF NOT EXISTS tender_journey_active_scope_idx ON tender_journey(organization_id,workspace_id,lifecycle_status,created_at);

CREATE TABLE IF NOT EXISTS tender_journey_session_binding (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                actor_ref TEXT NOT NULL,
                session_ref TEXT NOT NULL,
                journey_id TEXT NOT NULL REFERENCES tender_journey(journey_id),
                PRIMARY KEY (organization_id, workspace_id, actor_ref, session_ref)
            );
