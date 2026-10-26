-- Explicit additive gateway migration; never applied from a request handler.
ALTER TABLE semantier_gateway.tokens ADD COLUMN IF NOT EXISTS account_id text;
ALTER TABLE semantier_gateway.tokens ADD COLUMN IF NOT EXISTS currency text;
CREATE TABLE IF NOT EXISTS semantier_gateway.device_connections (
 intent_ref text PRIMARY KEY, challenge text NOT NULL, idempotency_key text NOT NULL,
 request_hash text NOT NULL, context jsonb NOT NULL, status text NOT NULL DEFAULT 'pending'
 CHECK(status IN ('pending','approved','claimed','installed','denied','cancelled','expired','revoked')),
 created_at timestamptz NOT NULL DEFAULT now(), expires_at timestamptz NOT NULL,
 next_poll_at timestamptz, user_id text, organization_id text, workspace_id text,
 account_id text, currency text, continuation jsonb NOT NULL DEFAULT '{}',
 credential_expires_at timestamptz,
 claim_id text, token_id text REFERENCES semantier_gateway.tokens,
 encryption_key_id text, nonce bytea, ciphertext bytea, installed_generation integer,
 UNIQUE(challenge,idempotency_key)
);

ALTER TABLE semantier_gateway.device_connections ADD COLUMN IF NOT EXISTS credential_expires_at timestamptz;

INSERT INTO semantier_gateway.schema_version(version) VALUES(3) ON CONFLICT DO NOTHING;
