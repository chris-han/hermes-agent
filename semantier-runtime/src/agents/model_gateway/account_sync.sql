-- Account projection v2. Only the explicit migrator installs this asset.
CREATE TABLE IF NOT EXISTS semantier_gateway.account_scopes (
 organization_id text NOT NULL,account_id text NOT NULL,currency text NOT NULL CHECK(currency IN ('CNY','USD')),
 billing_schema text NOT NULL,funding text NOT NULL CHECK(funding IN ('SEMANTIER_CREDITS','CUSTOMER_CLOUD')),
 PRIMARY KEY(organization_id,account_id,currency)
);
CREATE TABLE IF NOT EXISTS semantier_gateway.account_access (
 organization_id text NOT NULL,account_id text NOT NULL,currency text NOT NULL,user_id text NOT NULL,
 role text NOT NULL CHECK(role IN ('owner','billing_admin','reader','member')),
 status text NOT NULL CHECK(status IN ('ACTIVE','REVOKED','SUSPENDED')),revision bigint NOT NULL DEFAULT 1 CHECK(revision>0),
 created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(organization_id,account_id,currency,user_id),
 FOREIGN KEY(organization_id,account_id,currency) REFERENCES semantier_gateway.account_scopes
);
CREATE TABLE IF NOT EXISTS semantier_gateway.native_user_mappings (
 target_instance_id text NOT NULL,environment_namespace text NOT NULL,user_id text NOT NULL,native_user_id text NOT NULL,
 identity_revision bigint NOT NULL DEFAULT 1 CHECK(identity_revision>0),verified_at timestamptz,
 PRIMARY KEY(target_instance_id,environment_namespace,user_id),UNIQUE(target_instance_id,environment_namespace,native_user_id)
);
CREATE TABLE IF NOT EXISTS semantier_gateway.account_projections (
 target_instance_id text NOT NULL,environment_namespace text NOT NULL,organization_id text NOT NULL,account_id text NOT NULL,currency text NOT NULL,
 native_team_id text NOT NULL,desired_revision bigint NOT NULL CHECK(desired_revision>0),applied_revision bigint NOT NULL DEFAULT 0 CHECK(applied_revision>=0),
 desired_policy jsonb NOT NULL,desired_policy_hash text NOT NULL,observed_hash text,observed_state jsonb,
 status text NOT NULL CHECK(status IN ('PENDING','APPLYING','SYNCED','RETRY_WAIT','NEEDS_ATTENTION','OWNERSHIP_CONFLICT','BLOCKED_DEPENDENCY')),
 last_verified_at timestamptz,last_error_code text,lease_owner text,lease_token text,lease_until timestamptz,updated_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(target_instance_id,environment_namespace,organization_id,account_id,currency),
 UNIQUE(target_instance_id,environment_namespace,native_team_id),
 FOREIGN KEY(organization_id,account_id,currency) REFERENCES semantier_gateway.account_scopes
);
CREATE TABLE IF NOT EXISTS semantier_gateway.account_sync_intents (
 event_id text PRIMARY KEY,target_instance_id text NOT NULL,environment_namespace text NOT NULL,organization_id text NOT NULL,account_id text NOT NULL,currency text NOT NULL,
 revision bigint NOT NULL CHECK(revision>0),cause text NOT NULL,status text NOT NULL DEFAULT 'PENDING'
 CHECK(status IN ('PENDING','APPLYING','DONE','SUPERSEDED','RETRY_WAIT','NEEDS_ATTENTION','OWNERSHIP_CONFLICT','BLOCKED_DEPENDENCY')),
 attempts integer NOT NULL DEFAULT 0 CHECK(attempts>=0),failures integer NOT NULL DEFAULT 0 CHECK(failures>=0),available_at timestamptz NOT NULL DEFAULT now(),
 lease_owner text,lease_token text,lease_until timestamptz,last_error_code text,created_at timestamptz NOT NULL DEFAULT now(),completed_at timestamptz,
 UNIQUE(target_instance_id,environment_namespace,organization_id,account_id,currency,revision),
 FOREIGN KEY(target_instance_id,environment_namespace,organization_id,account_id,currency) REFERENCES semantier_gateway.account_projections
);
CREATE INDEX IF NOT EXISTS account_sync_due ON semantier_gateway.account_sync_intents(available_at,event_id) WHERE status IN ('PENDING','RETRY_WAIT','APPLYING');
CREATE TABLE IF NOT EXISTS semantier_gateway.native_credentials (
 target_instance_id text NOT NULL,environment_namespace text NOT NULL,organization_id text NOT NULL,account_id text NOT NULL,currency text NOT NULL,user_id text NOT NULL,
 generation bigint NOT NULL CHECK(generation>0),policy_revision bigint NOT NULL,policy_hash text NOT NULL,native_key_hash text NOT NULL,key_alias text NOT NULL,
 encryption_key_id text NOT NULL,nonce bytea NOT NULL CHECK(octet_length(nonce)=12),ciphertext bytea NOT NULL,
 status text NOT NULL CHECK(status IN ('PREPARED','ACTIVE','BLOCK_PENDING','BLOCKED','SUPERSEDED')),verified_at timestamptz,created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(target_instance_id,environment_namespace,organization_id,account_id,currency,user_id,generation),
 UNIQUE(target_instance_id,environment_namespace,native_key_hash),UNIQUE(target_instance_id,environment_namespace,key_alias),UNIQUE(encryption_key_id,nonce),
 FOREIGN KEY(target_instance_id,environment_namespace,organization_id,account_id,currency) REFERENCES semantier_gateway.account_projections,
 FOREIGN KEY(organization_id,account_id,currency,user_id) REFERENCES semantier_gateway.account_access
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_account_credential ON semantier_gateway.native_credentials(target_instance_id,environment_namespace,organization_id,account_id,currency,user_id) WHERE status='ACTIVE';
CREATE TABLE IF NOT EXISTS semantier_gateway.account_sync_audit (
 audit_id text PRIMARY KEY,event_id text NOT NULL,target_instance_id text NOT NULL,environment_namespace text NOT NULL,
 organization_id text NOT NULL,account_id text NOT NULL,currency text NOT NULL,revision bigint NOT NULL,
 attempt integer NOT NULL,reason text NOT NULL,outcome text NOT NULL,error_code text,before_revision bigint,after_revision bigint,
 started_at timestamptz NOT NULL,finished_at timestamptz NOT NULL DEFAULT now()
);
CREATE OR REPLACE FUNCTION semantier_gateway.account_sync_audit_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'Account synchronization audit is immutable'; END $$;
DROP TRIGGER IF EXISTS account_sync_audit_immutable ON semantier_gateway.account_sync_audit;
CREATE TRIGGER account_sync_audit_immutable BEFORE UPDATE OR DELETE ON semantier_gateway.account_sync_audit FOR EACH ROW EXECUTE FUNCTION semantier_gateway.account_sync_audit_immutable();
INSERT INTO semantier_gateway.schema_version(version) VALUES(2) ON CONFLICT DO NOTHING;

-- Idempotent explicit upgrade of early v2 candidate installations.
ALTER TABLE semantier_gateway.account_projections ADD COLUMN IF NOT EXISTS lease_owner text;
ALTER TABLE semantier_gateway.account_projections ADD COLUMN IF NOT EXISTS lease_token text;
ALTER TABLE semantier_gateway.account_projections ADD COLUMN IF NOT EXISTS lease_until timestamptz;
ALTER TABLE semantier_gateway.account_sync_intents ADD COLUMN IF NOT EXISTS failures integer NOT NULL DEFAULT 0;
ALTER TABLE semantier_gateway.account_sync_audit ADD COLUMN IF NOT EXISTS reason text NOT NULL DEFAULT 'legacy';

CREATE OR REPLACE FUNCTION semantier_gateway.native_credential_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF (to_jsonb(NEW)-'status'-'verified_at') IS DISTINCT FROM (to_jsonb(OLD)-'status'-'verified_at') THEN
  RAISE EXCEPTION 'Native credential generation is immutable';
 END IF;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS native_credential_immutable ON semantier_gateway.native_credentials;
CREATE TRIGGER native_credential_immutable BEFORE UPDATE ON semantier_gateway.native_credentials FOR EACH ROW EXECUTE FUNCTION semantier_gateway.native_credential_immutable();
