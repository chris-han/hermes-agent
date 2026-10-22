-- Canonical EOS append-only and active-dataset constraints, PostgreSQL v1.
CREATE OR REPLACE FUNCTION eos_reject_history_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '%', TG_ARGV[0] USING ERRCODE = '23514';
END;
$$;
ALTER FUNCTION eos_reject_history_mutation() SET search_path FROM CURRENT;

DROP TRIGGER IF EXISTS trg_governed_rule_activation_immutable ON governed_rule_activation;
CREATE TRIGGER trg_governed_rule_activation_immutable BEFORE UPDATE ON governed_rule_activation
FOR EACH ROW EXECUTE FUNCTION eos_reject_history_mutation('governed rule activation rows are immutable');
DROP TRIGGER IF EXISTS trg_governed_rule_activation_append_only ON governed_rule_activation;
CREATE TRIGGER trg_governed_rule_activation_append_only BEFORE DELETE ON governed_rule_activation
FOR EACH ROW EXECUTE FUNCTION eos_reject_history_mutation('governed rule activation rows are append-only');
DROP TRIGGER IF EXISTS trg_governed_rule_activation_event_immutable ON governed_rule_activation_event;
CREATE TRIGGER trg_governed_rule_activation_event_immutable BEFORE UPDATE ON governed_rule_activation_event
FOR EACH ROW EXECUTE FUNCTION eos_reject_history_mutation('governed rule activation events are immutable');
DROP TRIGGER IF EXISTS trg_governed_rule_activation_event_append_only ON governed_rule_activation_event;
CREATE TRIGGER trg_governed_rule_activation_event_append_only BEFORE DELETE ON governed_rule_activation_event
FOR EACH ROW EXECUTE FUNCTION eos_reject_history_mutation('governed rule activation events are append-only');
DROP TRIGGER IF EXISTS trg_governed_rule_versions_activated_immutable ON governed_rule_versions;
CREATE TRIGGER trg_governed_rule_versions_activated_immutable BEFORE UPDATE ON governed_rule_versions
FOR EACH ROW EXECUTE FUNCTION eos_reject_history_mutation('governed rule versions are immutable');

CREATE OR REPLACE FUNCTION eos_require_unique_dataset_role() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.activation_status = 'active' AND NEW.dataset_asset_version_id IS NOT NULL THEN
        PERFORM pg_advisory_xact_lock(hashtext(NEW.organization_id),hashtext(NEW.scope_key));
        IF EXISTS (
            SELECT 1 FROM knowledge_source_activation existing
            JOIN dataset_asset_version existing_asset
              ON existing_asset.dataset_asset_version_id=existing.dataset_asset_version_id
            JOIN dataset_asset_version new_asset
              ON new_asset.dataset_asset_version_id=NEW.dataset_asset_version_id
            WHERE existing.organization_id=NEW.organization_id
              AND existing.scope_key=NEW.scope_key AND existing.activation_status='active'
              AND existing.dataset_asset_version_id IS NOT NULL
              AND existing_asset.dataset_usage_role=new_asset.dataset_usage_role
              AND (TG_OP='INSERT' OR existing.activation_id != NEW.activation_id)
        ) THEN
            RAISE EXCEPTION 'duplicate active dataset binding for dataset_usage_role'
                USING ERRCODE = '23505';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;
ALTER FUNCTION eos_require_unique_dataset_role() SET search_path FROM CURRENT;
DROP TRIGGER IF EXISTS trg_ksa_active_dataset_role_insert ON knowledge_source_activation;
CREATE TRIGGER trg_ksa_active_dataset_role_insert BEFORE INSERT ON knowledge_source_activation
FOR EACH ROW EXECUTE FUNCTION eos_require_unique_dataset_role();
DROP TRIGGER IF EXISTS trg_ksa_active_dataset_role_update ON knowledge_source_activation;
CREATE TRIGGER trg_ksa_active_dataset_role_update
BEFORE UPDATE OF organization_id,scope_key,activation_status,dataset_asset_version_id ON knowledge_source_activation
FOR EACH ROW EXECUTE FUNCTION eos_require_unique_dataset_role();
