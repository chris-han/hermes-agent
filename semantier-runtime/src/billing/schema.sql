-- Semantier financial schema v1. Independent of LiteLLM operational migrations.
CREATE TABLE IF NOT EXISTS schema_version (version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now());
INSERT INTO schema_version(version) VALUES (1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS accounts (
 organization_id text NOT NULL, account_id text NOT NULL, currency text NOT NULL CHECK(currency IN ('USD','CNY')),
 funding text NOT NULL CHECK(funding IN ('SEMANTIER_CREDITS','CUSTOMER_CLOUD')),
 wallet numeric(38,12) NOT NULL DEFAULT 0 CHECK(wallet>=0 AND wallet<1e26),
 held numeric(38,12) NOT NULL DEFAULT 0 CHECK(held>=0 AND held<1e26),
 budget_limit numeric(38,12) NOT NULL CHECK(budget_limit>=0 AND budget_limit<1e26),
 budget_spent numeric(38,12) NOT NULL DEFAULT 0 CHECK(budget_spent>=0 AND budget_spent<1e26),
 blocked boolean NOT NULL DEFAULT false,
 PRIMARY KEY(organization_id,account_id,currency)
);
CREATE TABLE IF NOT EXISTS task_budgets (
 organization_id text NOT NULL, account_id text NOT NULL, currency text NOT NULL, task_id text NOT NULL,
 budget_limit numeric(38,12) NOT NULL CHECK(budget_limit>=0 AND budget_limit<1e26), spent numeric(38,12) NOT NULL DEFAULT 0 CHECK(spent>=0 AND spent<1e26), held numeric(38,12) NOT NULL DEFAULT 0 CHECK(held>=0 AND held<1e26),
 PRIMARY KEY(organization_id,account_id,currency,task_id), FOREIGN KEY(organization_id,account_id,currency) REFERENCES accounts
);
CREATE TABLE IF NOT EXISTS invocations (
 invocation_id text PRIMARY KEY,organization_id text NOT NULL,account_id text NOT NULL,currency text NOT NULL,
 operation text NOT NULL,idempotency_key text NOT NULL,request_hash text NOT NULL,deployment_id text NOT NULL,
 funding text NOT NULL,max_cost numeric(38,12) NOT NULL CHECK(max_cost>=0 AND max_cost<1e26),attribution jsonb NOT NULL,
 task_id text,execution_state text NOT NULL CHECK(execution_state IN ('NEW','ADMITTED','DISPATCHING','STREAMING','FINISHED','FAILED_BEFORE_SEND','POSSIBLY_EXECUTED')),settlement_state text NOT NULL CHECK(settlement_state IN ('NOT_APPLICABLE','HELD','SETTLED','PENDING_RECONCILIATION')),usage_state text NOT NULL DEFAULT 'MISSING' CHECK(usage_state IN ('MISSING','PARTIAL','PROVIDER_REPORTED','RECONCILED')),pricing_state text NOT NULL DEFAULT 'UNPRICED' CHECK(pricing_state IN ('UNPRICED','ESTIMATED','RATED','RECONCILED')),
 hold_active boolean NOT NULL DEFAULT true,customer_debit numeric(38,12),supplier_variance numeric(38,12),settlement_id text,
 created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now(),
 FOREIGN KEY(organization_id,account_id,currency) REFERENCES accounts,
 UNIQUE(organization_id,account_id,operation,idempotency_key)
);
CREATE TABLE IF NOT EXISTS attempts (
 attempt_id text PRIMARY KEY,invocation_id text NOT NULL REFERENCES invocations,ordinal integer NOT NULL DEFAULT 1,
 outbound_intent text NOT NULL DEFAULT 'PREPARED',dispatched_at timestamptz,UNIQUE(invocation_id,ordinal)
);
CREATE TABLE IF NOT EXISTS raw_observations (
 id bigserial PRIMARY KEY,attempt_id text NOT NULL REFERENCES attempts,event_id text NOT NULL,provider text NOT NULL,raw_usage jsonb NOT NULL,
 provenance jsonb NOT NULL,physical_send_count integer NOT NULL CHECK(physical_send_count>=0),completion_state text NOT NULL CHECK(completion_state IN ('FINAL','PARTIAL','MISSING')),
 observed_at timestamptz NOT NULL DEFAULT now(),UNIQUE(attempt_id,event_id)
);
CREATE TABLE IF NOT EXISTS prices (
 version_id text NOT NULL,kind text NOT NULL CHECK(kind IN ('supplier','customer')),payload jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),PRIMARY KEY(version_id,kind)
);
CREATE TABLE IF NOT EXISTS fx_quotes (quote_id text PRIMARY KEY,payload jsonb NOT NULL,created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS usage_observations (
 id bigserial PRIMARY KEY,attempt_id text NOT NULL REFERENCES attempts,event_id text NOT NULL,raw_usage jsonb NOT NULL,normalized jsonb NOT NULL,
 customer_cost numeric(38,12),supplier_cost numeric(38,12),price_version text,fx_version text,
 supplier_price_version text,supplier_native_cost numeric(38,12),supplier_currency text,
 observed_at timestamptz NOT NULL DEFAULT now(),UNIQUE(attempt_id,event_id)
);
CREATE TABLE IF NOT EXISTS ledger_transactions (
 id text PRIMARY KEY,organization_id text NOT NULL,account_id text NOT NULL,currency text NOT NULL,
 kind text NOT NULL,invocation_id text UNIQUE REFERENCES invocations,price_version text,created_at timestamptz NOT NULL DEFAULT now(),
 FOREIGN KEY(organization_id,account_id,currency) REFERENCES accounts
);
CREATE TABLE IF NOT EXISTS ledger_entries (
 id bigserial PRIMARY KEY,transaction_id text NOT NULL REFERENCES ledger_transactions,
 book_account text NOT NULL CHECK(book_account IN ('wallet','issuance','consumption')),
 amount numeric(38,12) NOT NULL CHECK(abs(amount)<1e26)
);
CREATE TABLE IF NOT EXISTS outbox (
 id text PRIMARY KEY,invocation_id text NOT NULL UNIQUE REFERENCES invocations,payload jsonb NOT NULL,
 lease_owner text,lease_until timestamptz,delivered_at timestamptz,delivery_attempts integer NOT NULL DEFAULT 0,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS invocations_recovery ON invocations(settlement_state,created_at);
CREATE INDEX IF NOT EXISTS usage_attempt ON usage_observations(attempt_id,id);
CREATE INDEX IF NOT EXISTS invocations_scoped_time ON invocations(organization_id,account_id,created_at);
CREATE OR REPLACE FUNCTION reject_financial_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'immutable financial evidence'; END $$;
CREATE OR REPLACE FUNCTION check_transaction_balance() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE selected_id text;
BEGIN
 selected_id := NEW.transaction_id;
 IF (SELECT coalesce(sum(amount),0) FROM ledger_entries WHERE transaction_id=selected_id) <> 0
 THEN RAISE EXCEPTION 'unbalanced service-credit transaction'; END IF;
 RETURN NULL;
END $$;
CREATE TABLE IF NOT EXISTS reconciliation_items (
 id bigserial PRIMARY KEY,invocation_id text NOT NULL REFERENCES invocations,usage_observation_id bigint NOT NULL REFERENCES usage_observations,
 reason text NOT NULL,created_at timestamptz NOT NULL DEFAULT now(),UNIQUE(invocation_id,usage_observation_id,reason)
);
-- Additive upgrade for early development v1 receipts; no historical rewrite.
ALTER TABLE usage_observations ADD COLUMN IF NOT EXISTS supplier_price_version text;
ALTER TABLE usage_observations ADD COLUMN IF NOT EXISTS supplier_native_cost numeric(38,12);
ALTER TABLE usage_observations ADD COLUMN IF NOT EXISTS supplier_currency text;
INSERT INTO schema_version(version) VALUES (2) ON CONFLICT DO NOTHING;
ALTER TABLE attempts ADD COLUMN IF NOT EXISTS native_send_claimed_at timestamptz;
INSERT INTO schema_version(version) VALUES (3) ON CONFLICT DO NOTHING;
-- Explicit organization financial policy administration; never a credit-grant path.
CREATE TABLE IF NOT EXISTS organization_customer_prices (
 organization_id text NOT NULL,version_id text NOT NULL,kind text NOT NULL DEFAULT 'customer' CHECK(kind='customer'),
 PRIMARY KEY(organization_id,version_id),UNIQUE(version_id),FOREIGN KEY(version_id,kind) REFERENCES prices
);
CREATE TABLE IF NOT EXISTS financial_policy_events (
 event_id text PRIMARY KEY,organization_id text NOT NULL,actor_user_id text NOT NULL,
 operation text NOT NULL,target text NOT NULL,payload jsonb NOT NULL,created_at timestamptz NOT NULL DEFAULT now()
);
INSERT INTO schema_version(version) VALUES (4) ON CONFLICT DO NOTHING;
-- Commerce is independently durable; invocation outbox remains usage-only.
CREATE TABLE IF NOT EXISTS commerce_quotes (
 quote_id text PRIMARY KEY,organization_id text NOT NULL,account_id text NOT NULL,
 product jsonb NOT NULL,expires_at timestamptz NOT NULL,created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS commerce_orders (
 order_id text PRIMARY KEY,organization_id text NOT NULL,account_id text NOT NULL,
 quote_id text NOT NULL REFERENCES commerce_quotes,idempotency_key text NOT NULL,
 product jsonb NOT NULL,status text NOT NULL DEFAULT 'PENDING_PAYMENT',
 created_at timestamptz NOT NULL DEFAULT now(),UNIQUE(organization_id,account_id,idempotency_key)
);
CREATE TABLE IF NOT EXISTS commerce_payment_events (
 event_id text PRIMARY KEY,order_id text NOT NULL REFERENCES commerce_orders,payload jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS commerce_fulfillment (
 order_id text PRIMARY KEY REFERENCES commerce_orders,fulfilled_at timestamptz
);
CREATE TABLE IF NOT EXISTS commerce_subscriptions (
 order_id text PRIMARY KEY REFERENCES commerce_orders,organization_id text NOT NULL,account_id text NOT NULL,
 product jsonb NOT NULL,starts_at timestamptz NOT NULL,ends_at timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS credit_grants (
 grant_id text PRIMARY KEY,organization_id text NOT NULL,account_id text NOT NULL,currency text NOT NULL,
 kind text NOT NULL CHECK(kind IN ('allowance','purchased')),amount numeric(38,12) NOT NULL,
 remaining numeric(38,12) NOT NULL CHECK(remaining>=0),held numeric(38,12) NOT NULL DEFAULT 0 CHECK(held>=0),
 starts_at timestamptz NOT NULL,expires_at timestamptz,expired boolean NOT NULL DEFAULT false
);
CREATE TABLE IF NOT EXISTS credit_hold_grants (
 invocation_id text NOT NULL REFERENCES invocations,grant_id text NOT NULL REFERENCES credit_grants,
 amount numeric(38,12) NOT NULL CHECK(amount>=0),PRIMARY KEY(invocation_id,grant_id)
);
INSERT INTO schema_version(version) VALUES (5) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS commerce_account_bindings (
 organization_id text NOT NULL,account_id text NOT NULL,currency text NOT NULL CHECK(currency='CNY'),
 merchant_id text NOT NULL CHECK(merchant_id='local-development'),environment text NOT NULL CHECK(environment='DEVELOPMENT'),
 PRIMARY KEY(organization_id,account_id),FOREIGN KEY(organization_id,account_id,currency) REFERENCES accounts
);
CREATE OR REPLACE FUNCTION freeze_commerce_order() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='DELETE' OR (to_jsonb(NEW)-'status') IS DISTINCT FROM (to_jsonb(OLD)-'status')
 THEN RAISE EXCEPTION 'immutable commerce order snapshot'; END IF;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS frozen_order ON commerce_orders;
CREATE TRIGGER frozen_order BEFORE UPDATE OR DELETE ON commerce_orders FOR EACH ROW EXECUTE FUNCTION freeze_commerce_order();

-- Paid commercial exceptions are retained without blocking independent usage work.
ALTER TABLE commerce_fulfillment ADD COLUMN IF NOT EXISTS reconciliation_reason text;
INSERT INTO schema_version(version) VALUES (6) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS commerce_enterprise_offers (
 offer_id text PRIMARY KEY, organization_id text NOT NULL, account_id text NOT NULL,
 product jsonb NOT NULL, expires_at timestamptz NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS commerce_payment_attempts (
 attempt_id text PRIMARY KEY, order_id text NOT NULL REFERENCES commerce_orders,
 method text NOT NULL CHECK(method IN ('wechat_test','alipay_test')),
 status text NOT NULL CHECK(status IN ('AWAITING_PAYMENT','SUCCEEDED','CANCELED','EXPIRED','DECLINED')),
 expires_at timestamptz NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS commerce_attempt_order ON commerce_payment_attempts(order_id,created_at);
CREATE OR REPLACE FUNCTION freeze_commerce_attempt() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='DELETE' OR (to_jsonb(NEW)-'status') IS DISTINCT FROM (to_jsonb(OLD)-'status')
 THEN RAISE EXCEPTION 'immutable commerce payment attempt'; END IF;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS frozen_attempt ON commerce_payment_attempts;
CREATE TRIGGER frozen_attempt BEFORE UPDATE OR DELETE ON commerce_payment_attempts FOR EACH ROW EXECUTE FUNCTION freeze_commerce_attempt();
INSERT INTO schema_version(version) VALUES (7) ON CONFLICT DO NOTHING;
