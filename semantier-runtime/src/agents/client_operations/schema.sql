-- Version 1. Fresh bootstrap and explicit idempotent existing-DB migration.
CREATE TABLE IF NOT EXISTS schema_version(version integer PRIMARY KEY,applied_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS grants(user_id text NOT NULL,capability text NOT NULL,channel text NOT NULL,PRIMARY KEY(user_id,capability,channel));
CREATE TABLE IF NOT EXISTS principals(token_hash text PRIMARY KEY,principal_id text NOT NULL,kind text NOT NULL CHECK(kind IN ('ci','reporting')),user_id text,installation_ref text,channel text NOT NULL,expires_at timestamptz NOT NULL,revoked_at timestamptz);
CREATE TABLE IF NOT EXISTS releases(release_id text PRIMARY KEY,build_id text UNIQUE NOT NULL,body jsonb NOT NULL,body_hash text NOT NULL,qualification text NOT NULL CHECK(qualification IN ('verified','rejected')),received_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS channels(channel text PRIMARY KEY,revision bigint NOT NULL DEFAULT 1,paused boolean NOT NULL DEFAULT false,release_id text REFERENCES releases,record jsonb NOT NULL);
CREATE TABLE IF NOT EXISTS operations(operation_id text PRIMARY KEY,user_id text NOT NULL,channel text NOT NULL REFERENCES channels,idempotency_key text NOT NULL,request_hash text NOT NULL,body jsonb NOT NULL,status text NOT NULL,created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now(),UNIQUE(user_id,idempotency_key));
CREATE TABLE IF NOT EXISTS audit(event_id text PRIMARY KEY,actor text NOT NULL,action text NOT NULL,object_id text NOT NULL,body jsonb NOT NULL,created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS installations(installation_ref text PRIMARY KEY,user_id text NOT NULL,channel text NOT NULL,consent_version text NOT NULL,consented_at timestamptz NOT NULL DEFAULT now(),revoked_at timestamptz);
CREATE TABLE IF NOT EXISTS events(installation_ref text NOT NULL REFERENCES installations,event_id text NOT NULL,update_id text NOT NULL,release_id text NOT NULL REFERENCES releases,body_hash text NOT NULL,body jsonb NOT NULL,received_at timestamptz NOT NULL DEFAULT now(),PRIMARY KEY(installation_ref,event_id));
CREATE INDEX IF NOT EXISTS events_attempt ON events(update_id,received_at);
CREATE TABLE IF NOT EXISTS diagnostics(diagnostic_id text PRIMARY KEY,user_id text NOT NULL,submission_key text NOT NULL,body_hash text NOT NULL,release_id text REFERENCES releases,update_id text,metadata jsonb NOT NULL,payload text,classification jsonb NOT NULL,revision bigint NOT NULL DEFAULT 0,received_at timestamptz NOT NULL DEFAULT now(),expires_at timestamptz NOT NULL DEFAULT now()+interval '30 days',deleted_at timestamptz,UNIQUE(user_id,submission_key));
CREATE TABLE IF NOT EXISTS access_sources(source_id text PRIMARY KEY,source_path text NOT NULL,cursor_bytes bigint NOT NULL DEFAULT 0,gap boolean NOT NULL DEFAULT false,last_ingested_at timestamptz);
CREATE TABLE IF NOT EXISTS access_events(source_id text NOT NULL REFERENCES access_sources,offset_bytes bigint NOT NULL,artifact_key text NOT NULL,method text NOT NULL,status integer NOT NULL,bytes bigint NOT NULL,occurred_at timestamptz NOT NULL,received_at timestamptz NOT NULL DEFAULT now(),PRIMARY KEY(source_id,offset_bytes));
INSERT INTO schema_version(version) VALUES(1) ON CONFLICT DO NOTHING;
-- v2: feedback-only authorization, separate from provider/device installation.
ALTER TABLE principals DROP CONSTRAINT IF EXISTS principals_kind_check;
ALTER TABLE principals ADD CONSTRAINT principals_kind_check CHECK(kind IN ('ci','reporting','diagnostic'));
CREATE TABLE IF NOT EXISTS authorizations(intent_ref text PRIMARY KEY,proof_challenge text NOT NULL,purpose text NOT NULL CHECK(purpose IN ('diagnostic','reporting')),profile_ref text NOT NULL,status text NOT NULL DEFAULT 'pending',user_id text,ciphertext bytea,nonce bytea,token_hash text,expires_at timestamptz NOT NULL DEFAULT now()+interval '15 minutes',created_at timestamptz NOT NULL DEFAULT now());
INSERT INTO schema_version(version) VALUES(2) ON CONFLICT DO NOTHING;
-- v3: target-scoped immutable admission and durable multi-target publication.
CREATE TABLE IF NOT EXISTS delivery_grants(
 principal_id text NOT NULL, channel text NOT NULL, target_id text NOT NULL CHECK(target_id IN ('channel1','channel2','channel3')),
 action text NOT NULL CHECK(action IN ('read','publish','ingest')), PRIMARY KEY(principal_id,channel,target_id,action));
CREATE TABLE IF NOT EXISTS release_delivery_variants(
 release_id text NOT NULL REFERENCES releases, target_id text NOT NULL CHECK(target_id IN ('channel1','channel2','channel3')),
 variant_hash text NOT NULL, body jsonb NOT NULL, first_admitted_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(release_id,target_id,variant_hash));
CREATE TABLE IF NOT EXISTS delivery_heads(
 target_id text NOT NULL CHECK(target_id IN ('channel1','channel2','channel3')), channel text NOT NULL REFERENCES channels,
 revision bigint NOT NULL DEFAULT 1, release_id text REFERENCES releases, record jsonb NOT NULL,
 PRIMARY KEY(target_id,channel));
CREATE TABLE IF NOT EXISTS publications(
 publication_id text PRIMARY KEY, user_id text NOT NULL, environment text NOT NULL, channel text NOT NULL REFERENCES channels,
 release_id text NOT NULL REFERENCES releases, idempotency_key text NOT NULL, request_hash text NOT NULL, intent jsonb NOT NULL,
 status text NOT NULL DEFAULT 'preflighting', lifecycle text NOT NULL DEFAULT 'active' CHECK(lifecycle IN ('active','settled')),
 batch_revision bigint NOT NULL DEFAULT 1, maintenance_status text NOT NULL DEFAULT 'none',
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(user_id,environment,idempotency_key));
CREATE TABLE IF NOT EXISTS publication_targets(
 publication_id text NOT NULL REFERENCES publications, target_id text NOT NULL CHECK(target_id IN ('channel1','channel2','channel3')),
 environment text NOT NULL, channel text NOT NULL, attempt integer NOT NULL, variant_hash text NOT NULL,
 operation_id text NOT NULL UNIQUE, status text NOT NULL DEFAULT 'queued', stage text NOT NULL DEFAULT 'preflight',
 reserved boolean NOT NULL DEFAULT true, expected_revision bigint NOT NULL, previous_head jsonb, intended_head jsonb,
 preview_plan_digest text NOT NULL, capacity jsonb, error_code text, actual_revision bigint, readback_digest text,
 owner_instance jsonb, remote_lock jsonb, maintenance_status text NOT NULL DEFAULT 'none',
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(publication_id,target_id,attempt));
CREATE UNIQUE INDEX IF NOT EXISTS one_delivery_reservation ON publication_targets(environment,target_id,channel) WHERE reserved;
CREATE INDEX IF NOT EXISTS publication_history ON publications(environment,channel,release_id,created_at DESC,publication_id);
INSERT INTO schema_version(version) VALUES(3) ON CONFLICT DO NOTHING;
