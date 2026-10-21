"""Pinned PostgreSQL hosted-room tables and immutable history/budget guards. Migration owner only."""
SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS hosted_rooms (
            room_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            members_json TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch BIGINT NOT NULL DEFAULT 1 CHECK (authority_epoch >= 1),
            next_seq BIGINT NOT NULL DEFAULT 1 CHECK (next_seq >= 1),
            event_bytes BIGINT NOT NULL DEFAULT 0 CHECK (event_bytes >= 0),
            revision BIGINT NOT NULL DEFAULT 1 CHECK (revision >= 1),
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL,
            disbanded_at DOUBLE PRECISION
        );

CREATE TABLE IF NOT EXISTS hosted_room_events (
            room_id TEXT NOT NULL,
            seq BIGINT NOT NULL CHECK (seq >= 1),
            event_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            actor_json TEXT NOT NULL,
            authority_epoch BIGINT CHECK (authority_epoch IS NULL OR authority_epoch >= 1),
            payload_json TEXT NOT NULL,
            created_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (room_id, seq),
            UNIQUE (room_id, event_id),
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id)
        );

CREATE TABLE IF NOT EXISTS hosted_room_retired_ids (
            room_id TEXT PRIMARY KEY,
            retired_at DOUBLE PRECISION NOT NULL
        );

CREATE TABLE IF NOT EXISTS hosted_room_links (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            target_url TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            "grant" TEXT NOT NULL,
            catalog_json TEXT NOT NULL,
            cancellation_scope_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            transport_security TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ready',
            updated_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (room_id, member_id)
        );

CREATE TABLE IF NOT EXISTS hosted_room_remote_runs (
            room_id TEXT NOT NULL,
            home_install_id TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch BIGINT NOT NULL CHECK (authority_epoch >= 1),
            member_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            execution_generation BIGINT NOT NULL CHECK (execution_generation >= 1),
            target_install_id TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (
                room_id, home_install_id, authority_gateway_id, authority_epoch,
                member_id, target_install_id, target_profile, task_id,
                execution_generation
            )
        );

CREATE TABLE IF NOT EXISTS hosted_room_revoked_grants (
            scope_key TEXT PRIMARY KEY,
            expires_at DOUBLE PRECISION NOT NULL,
            revoked_before DOUBLE PRECISION NOT NULL
        );

CREATE TABLE IF NOT EXISTS hosted_room_peer_reservations (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch BIGINT NOT NULL CHECK (authority_epoch >= 1),
            expires_at DOUBLE PRECISION NOT NULL,
            revoked_at DOUBLE PRECISION,
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (room_id, member_id, target_profile)
        );

CREATE INDEX IF NOT EXISTS idx_hosted_room_events_cursor ON hosted_room_events(room_id, seq);

CREATE TABLE IF NOT EXISTS hosted_room_legacy_imports (source TEXT PRIMARY KEY, imported_at DOUBLE PRECISION NOT NULL, rooms BIGINT NOT NULL);

CREATE TABLE IF NOT EXISTS hosted_room_driver_leases (
            room_id TEXT PRIMARY KEY, gateway_id TEXT NOT NULL,
            authority_epoch BIGINT NOT NULL CHECK (authority_epoch >= 1), process_generation TEXT NOT NULL,
            lease_generation BIGINT NOT NULL CHECK (lease_generation >= 1),
            expires_at DOUBLE PRECISION NOT NULL, acquired_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL, released_at DOUBLE PRECISION,
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id));

CREATE TABLE IF NOT EXISTS hosted_room_driver_tasks (
            room_id TEXT NOT NULL, task_id TEXT NOT NULL, thread_id TEXT NOT NULL, turn_id TEXT NOT NULL,
            source_event_seq BIGINT NOT NULL CHECK (source_event_seq >= 1),
            payload_json TEXT NOT NULL, payload_digest TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN (
                'queued', 'running', 'settled', 'failed', 'cancelled', 'indeterminate', 'deferred', 'stopping')),
            execution_generation BIGINT NOT NULL DEFAULT 0 CHECK (execution_generation >= 0),
            cancel_generation BIGINT NOT NULL DEFAULT 0 CHECK (cancel_generation >= 0),
            run_gateway_id TEXT, run_process_generation TEXT, run_lease_generation BIGINT, cancel_id TEXT,
            settlement_id TEXT, settlement_status TEXT, result_json TEXT, created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL, started_at DOUBLE PRECISION, terminal_at DOUBLE PRECISION, indeterminate_at DOUBLE PRECISION,
            PRIMARY KEY (room_id, task_id), UNIQUE (room_id, thread_id, turn_id),
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id));

CREATE INDEX IF NOT EXISTS idx_hosted_room_driver_tasks_status
           ON hosted_room_driver_tasks(room_id, status, source_event_seq, created_at, task_id);

CREATE TABLE IF NOT EXISTS hosted_room_replicas (
            room_id TEXT PRIMARY KEY, name TEXT NOT NULL, members_json TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch BIGINT NOT NULL CHECK (authority_epoch >= 1),
            last_seq BIGINT NOT NULL DEFAULT 0 CHECK (last_seq >= 0),
            latest_seq BIGINT NOT NULL DEFAULT 0, event_bytes BIGINT NOT NULL DEFAULT 0,
            created_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL
        );

CREATE TABLE IF NOT EXISTS hosted_room_replica_events (
            room_id TEXT NOT NULL, seq BIGINT NOT NULL CHECK (seq >= 1), event_id TEXT NOT NULL,
            kind TEXT NOT NULL, actor_json TEXT NOT NULL, authority_epoch BIGINT,
            payload_json TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (room_id, seq)
        );

CREATE TABLE IF NOT EXISTS hosted_room_work_records_invalid (
        evidence_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, source_table TEXT NOT NULL,
        room_id TEXT NOT NULL, revision BIGINT NOT NULL, digest TEXT NOT NULL, record_json TEXT NOT NULL,
        target_install_id TEXT, route_generation TEXT, status TEXT,
        disposition TEXT NOT NULL CHECK(disposition='invalid'));

CREATE TABLE IF NOT EXISTS hosted_room_work_records_source (
            room_id TEXT NOT NULL, revision BIGINT NOT NULL, digest TEXT NOT NULL, record_json TEXT NOT NULL,
            producer_gateway_id TEXT NOT NULL, producer_epoch BIGINT NOT NULL,
            disposition TEXT NOT NULL CHECK(disposition IN ('current','superseded_authority','historical','invalid')),
             PRIMARY KEY(room_id,producer_gateway_id,producer_epoch),
            FOREIGN KEY(room_id) REFERENCES hosted_rooms(room_id) ON DELETE CASCADE);

CREATE TABLE IF NOT EXISTS hosted_room_work_records_pending (
            room_id TEXT NOT NULL, revision BIGINT NOT NULL, digest TEXT NOT NULL, record_json TEXT NOT NULL,
            producer_gateway_id TEXT NOT NULL, producer_epoch BIGINT NOT NULL,
            disposition TEXT NOT NULL CHECK(disposition IN ('current','superseded_authority','historical','invalid')),
            target_install_id TEXT NOT NULL, route_generation TEXT NOT NULL, status TEXT NOT NULL, PRIMARY KEY(room_id,producer_gateway_id,producer_epoch,target_install_id),
            FOREIGN KEY(room_id) REFERENCES hosted_rooms(room_id) ON DELETE CASCADE);

CREATE OR REPLACE FUNCTION room_guard_767cf6ab52f11931() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN PERFORM pg_advisory_xact_lock(hashtextextended(current_schema()||':work-record-budget',0)); IF (
                    ((20+COALESCE(octet_length(convert_to(CAST(NEW.room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_epoch AS text),'UTF8')),0))>(COALESCE((SELECT 20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0) FROM hosted_room_work_records_source WHERE room_id=NEW.room_id AND producer_gateway_id=NEW.producer_gateway_id AND producer_epoch=NEW.producer_epoch),0)) AND ((SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)),0) FROM hosted_room_work_records_source) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_pending) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(source_table AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_invalid))-(COALESCE((SELECT 20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0) FROM hosted_room_work_records_source WHERE room_id=NEW.room_id AND producer_gateway_id=NEW.producer_gateway_id AND producer_epoch=NEW.producer_epoch),0))+(20+COALESCE(octet_length(convert_to(CAST(NEW.room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_epoch AS text),'UTF8')),0))>4194304)
                    OR ((SELECT COUNT(*) FROM hosted_room_work_records_source WHERE room_id=NEW.room_id AND producer_gateway_id=NEW.producer_gateway_id AND producer_epoch=NEW.producer_epoch)=0 AND ((SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)),0) FROM hosted_room_work_records_source) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_pending) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(source_table AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_invalid)>4194304 OR (SELECT COUNT(*) FROM hosted_room_work_records_source) + (SELECT COUNT(*) FROM hosted_room_work_records_pending) + (SELECT COUNT(*) FROM hosted_room_work_records_invalid)+1>512))) THEN RAISE EXCEPTION 'work record storage is full' USING ERRCODE='23514'; END IF; RETURN NEW; END $room_guard$;

DROP TRIGGER IF EXISTS trg_hosted_room_work_records_source_budget_v4_insert ON hosted_room_work_records_source;

CREATE TRIGGER trg_hosted_room_work_records_source_budget_v4_insert BEFORE INSERT ON hosted_room_work_records_source FOR EACH ROW EXECUTE FUNCTION room_guard_767cf6ab52f11931();

CREATE OR REPLACE FUNCTION room_guard_039d90d6f6318594() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN PERFORM pg_advisory_xact_lock(hashtextextended(current_schema()||':work-record-budget',0)); IF (
                    ((20+COALESCE(octet_length(convert_to(CAST(NEW.room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_epoch AS text),'UTF8')),0))>(20+COALESCE(octet_length(convert_to(CAST(OLD.room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.producer_epoch AS text),'UTF8')),0)) AND ((SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)),0) FROM hosted_room_work_records_source) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_pending) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(source_table AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_invalid))-(20+COALESCE(octet_length(convert_to(CAST(OLD.room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.producer_epoch AS text),'UTF8')),0))+(20+COALESCE(octet_length(convert_to(CAST(NEW.room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_epoch AS text),'UTF8')),0))>4194304)
                    OR (1=0 AND ((SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)),0) FROM hosted_room_work_records_source) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_pending) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(source_table AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_invalid)>4194304 OR (SELECT COUNT(*) FROM hosted_room_work_records_source) + (SELECT COUNT(*) FROM hosted_room_work_records_pending) + (SELECT COUNT(*) FROM hosted_room_work_records_invalid)+1>512))) THEN RAISE EXCEPTION 'work record storage is full' USING ERRCODE='23514'; END IF; RETURN NEW; END $room_guard$;

DROP TRIGGER IF EXISTS trg_hosted_room_work_records_source_budget_v4_update ON hosted_room_work_records_source;

CREATE TRIGGER trg_hosted_room_work_records_source_budget_v4_update BEFORE UPDATE ON hosted_room_work_records_source FOR EACH ROW EXECUTE FUNCTION room_guard_039d90d6f6318594();

CREATE OR REPLACE FUNCTION room_guard_6c55e945adcdab02() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN IF NEW.room_id!=OLD.room_id OR NEW.producer_gateway_id!=OLD.producer_gateway_id
                OR NEW.producer_epoch!=OLD.producer_epoch
                OR (OLD.disposition!='current' AND (NEW.record_json!=OLD.record_json OR NEW.revision!=OLD.revision
                    OR NEW.digest!=OLD.digest OR NEW.disposition!=OLD.disposition)) THEN RAISE EXCEPTION 'historical work record is immutable' USING ERRCODE='23514'; END IF; RETURN NEW; END $room_guard$;

DROP TRIGGER IF EXISTS trg_hosted_room_work_records_source_immutable_v2 ON hosted_room_work_records_source;

CREATE TRIGGER trg_hosted_room_work_records_source_immutable_v2 BEFORE UPDATE ON hosted_room_work_records_source FOR EACH ROW EXECUTE FUNCTION room_guard_6c55e945adcdab02();

CREATE OR REPLACE FUNCTION room_guard_4ab1a4e93084cb41() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN IF OLD.disposition!='current' AND EXISTS (SELECT 1 FROM hosted_rooms
                WHERE room_id=OLD.room_id AND disbanded_at IS NULL) THEN RAISE EXCEPTION 'historical work record is immutable' USING ERRCODE='23514'; END IF; RETURN OLD; END $room_guard$;

DROP TRIGGER IF EXISTS trg_hosted_room_work_records_source_delete_v2 ON hosted_room_work_records_source;

CREATE TRIGGER trg_hosted_room_work_records_source_delete_v2 BEFORE DELETE ON hosted_room_work_records_source FOR EACH ROW EXECUTE FUNCTION room_guard_4ab1a4e93084cb41();

CREATE OR REPLACE FUNCTION room_guard_55d5e7164a171f15() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN IF EXISTS (SELECT 1 FROM hosted_room_work_records_source p WHERE p.room_id=NEW.room_id
                AND p.producer_gateway_id=NEW.producer_gateway_id AND p.producer_epoch=NEW.producer_epoch
                 AND p.disposition!='current') THEN RAISE EXCEPTION 'historical work record is immutable' USING ERRCODE='23514'; END IF; RETURN NEW; END $room_guard$;

DROP TRIGGER IF EXISTS trg_hosted_room_work_records_source_replace_v2 ON hosted_room_work_records_source;

CREATE TRIGGER trg_hosted_room_work_records_source_replace_v2 BEFORE INSERT ON hosted_room_work_records_source FOR EACH ROW EXECUTE FUNCTION room_guard_55d5e7164a171f15();

CREATE OR REPLACE FUNCTION room_guard_17884c16e5ca6db6() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN PERFORM pg_advisory_xact_lock(hashtextextended(current_schema()||':work-record-budget',0)); IF (
                    ((20+COALESCE(octet_length(convert_to(CAST(NEW.room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.status AS text),'UTF8')),0))>(COALESCE((SELECT 20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0) FROM hosted_room_work_records_pending WHERE room_id=NEW.room_id AND producer_gateway_id=NEW.producer_gateway_id AND producer_epoch=NEW.producer_epoch AND target_install_id=NEW.target_install_id),0)) AND ((SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)),0) FROM hosted_room_work_records_source) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_pending) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(source_table AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_invalid))-(COALESCE((SELECT 20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0) FROM hosted_room_work_records_pending WHERE room_id=NEW.room_id AND producer_gateway_id=NEW.producer_gateway_id AND producer_epoch=NEW.producer_epoch AND target_install_id=NEW.target_install_id),0))+(20+COALESCE(octet_length(convert_to(CAST(NEW.room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.status AS text),'UTF8')),0))>4194304)
                    OR ((SELECT COUNT(*) FROM hosted_room_work_records_pending WHERE room_id=NEW.room_id AND producer_gateway_id=NEW.producer_gateway_id AND producer_epoch=NEW.producer_epoch AND target_install_id=NEW.target_install_id)=0 AND ((SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)),0) FROM hosted_room_work_records_source) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_pending) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(source_table AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_invalid)>4194304 OR (SELECT COUNT(*) FROM hosted_room_work_records_source) + (SELECT COUNT(*) FROM hosted_room_work_records_pending) + (SELECT COUNT(*) FROM hosted_room_work_records_invalid)+1>512))) THEN RAISE EXCEPTION 'work record storage is full' USING ERRCODE='23514'; END IF; RETURN NEW; END $room_guard$;

DROP TRIGGER IF EXISTS trg_hosted_room_work_records_pending_budget_v4_insert ON hosted_room_work_records_pending;

CREATE TRIGGER trg_hosted_room_work_records_pending_budget_v4_insert BEFORE INSERT ON hosted_room_work_records_pending FOR EACH ROW EXECUTE FUNCTION room_guard_17884c16e5ca6db6();

CREATE OR REPLACE FUNCTION room_guard_918000698c7a67ee() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN PERFORM pg_advisory_xact_lock(hashtextextended(current_schema()||':work-record-budget',0)); IF (
                    ((20+COALESCE(octet_length(convert_to(CAST(NEW.room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.status AS text),'UTF8')),0))>(20+COALESCE(octet_length(convert_to(CAST(OLD.room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.status AS text),'UTF8')),0)) AND ((SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)),0) FROM hosted_room_work_records_source) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_pending) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(source_table AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_invalid))-(20+COALESCE(octet_length(convert_to(CAST(OLD.room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(OLD.status AS text),'UTF8')),0))+(20+COALESCE(octet_length(convert_to(CAST(NEW.room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(NEW.status AS text),'UTF8')),0))>4194304)
                    OR (1=0 AND ((SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)),0) FROM hosted_room_work_records_source) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_gateway_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(producer_epoch AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_pending) + (SELECT COALESCE(SUM(20+COALESCE(octet_length(convert_to(CAST(room_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(revision AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(digest AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(record_json AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(source_table AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(target_install_id AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(route_generation AS text),'UTF8')),0)+COALESCE(octet_length(convert_to(CAST(status AS text),'UTF8')),0)),0) FROM hosted_room_work_records_invalid)>4194304 OR (SELECT COUNT(*) FROM hosted_room_work_records_source) + (SELECT COUNT(*) FROM hosted_room_work_records_pending) + (SELECT COUNT(*) FROM hosted_room_work_records_invalid)+1>512))) THEN RAISE EXCEPTION 'work record storage is full' USING ERRCODE='23514'; END IF; RETURN NEW; END $room_guard$;

DROP TRIGGER IF EXISTS trg_hosted_room_work_records_pending_budget_v4_update ON hosted_room_work_records_pending;

CREATE TRIGGER trg_hosted_room_work_records_pending_budget_v4_update BEFORE UPDATE ON hosted_room_work_records_pending FOR EACH ROW EXECUTE FUNCTION room_guard_918000698c7a67ee();

CREATE OR REPLACE FUNCTION room_guard_924b092873456f13() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN IF NEW.room_id!=OLD.room_id OR NEW.producer_gateway_id!=OLD.producer_gateway_id
                OR NEW.producer_epoch!=OLD.producer_epoch
                OR (OLD.disposition!='current' AND (NEW.record_json!=OLD.record_json OR NEW.revision!=OLD.revision
                    OR NEW.digest!=OLD.digest OR NEW.disposition!=OLD.disposition)) THEN RAISE EXCEPTION 'historical work record is immutable' USING ERRCODE='23514'; END IF; RETURN NEW; END $room_guard$;

DROP TRIGGER IF EXISTS trg_hosted_room_work_records_pending_immutable_v2 ON hosted_room_work_records_pending;

CREATE TRIGGER trg_hosted_room_work_records_pending_immutable_v2 BEFORE UPDATE ON hosted_room_work_records_pending FOR EACH ROW EXECUTE FUNCTION room_guard_924b092873456f13();

CREATE OR REPLACE FUNCTION room_guard_3d5448f7d58df910() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN IF OLD.disposition!='current' AND EXISTS (SELECT 1 FROM hosted_rooms
                WHERE room_id=OLD.room_id AND disbanded_at IS NULL) THEN RAISE EXCEPTION 'historical work record is immutable' USING ERRCODE='23514'; END IF; RETURN OLD; END $room_guard$;

DROP TRIGGER IF EXISTS trg_hosted_room_work_records_pending_delete_v2 ON hosted_room_work_records_pending;

CREATE TRIGGER trg_hosted_room_work_records_pending_delete_v2 BEFORE DELETE ON hosted_room_work_records_pending FOR EACH ROW EXECUTE FUNCTION room_guard_3d5448f7d58df910();

CREATE OR REPLACE FUNCTION room_guard_2459295aa6a1af49() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN IF EXISTS (SELECT 1 FROM hosted_room_work_records_pending p WHERE p.room_id=NEW.room_id
                AND p.producer_gateway_id=NEW.producer_gateway_id AND p.producer_epoch=NEW.producer_epoch
                 AND p.target_install_id=NEW.target_install_id AND p.disposition!='current') THEN RAISE EXCEPTION 'historical work record is immutable' USING ERRCODE='23514'; END IF; RETURN NEW; END $room_guard$;

DROP TRIGGER IF EXISTS trg_hosted_room_work_records_pending_replace_v2 ON hosted_room_work_records_pending;

CREATE TRIGGER trg_hosted_room_work_records_pending_replace_v2 BEFORE INSERT ON hosted_room_work_records_pending FOR EACH ROW EXECUTE FUNCTION room_guard_2459295aa6a1af49();

CREATE OR REPLACE FUNCTION room_guard_534d338d3bb75757() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN IF OLD.disposition!='current' AND (NEW.status!=OLD.status OR NEW.route_generation!=OLD.route_generation
            OR NEW.target_install_id!=OLD.target_install_id) THEN RAISE EXCEPTION 'historical delivery outcome is immutable' USING ERRCODE='23514'; END IF; RETURN NEW; END $room_guard$;

DROP TRIGGER IF EXISTS trg_work_pending_outcome_v2 ON hosted_room_work_records_pending;

CREATE TRIGGER trg_work_pending_outcome_v2 BEFORE UPDATE ON hosted_room_work_records_pending FOR EACH ROW EXECUTE FUNCTION room_guard_534d338d3bb75757();

CREATE OR REPLACE FUNCTION room_guard_4fa672e038eab2b0() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN UPDATE hosted_room_work_records_source SET disposition='historical' WHERE room_id=NEW.room_id AND disposition='current'
            AND (producer_gateway_id!=NEW.authority_gateway_id OR producer_epoch!=NEW.authority_epoch);
        UPDATE hosted_room_work_records_pending SET disposition=CASE WHEN status='acked' THEN 'historical' ELSE 'superseded_authority' END
            WHERE room_id=NEW.room_id AND disposition='current'
            AND (producer_gateway_id!=NEW.authority_gateway_id OR producer_epoch!=NEW.authority_epoch); RETURN NEW; END $room_guard$;

DROP TRIGGER IF EXISTS trg_work_authority_transition_v2 ON hosted_rooms;

CREATE TRIGGER trg_work_authority_transition_v2 AFTER UPDATE OF authority_gateway_id,authority_epoch ON hosted_rooms FOR EACH ROW EXECUTE FUNCTION room_guard_4fa672e038eab2b0();

CREATE OR REPLACE FUNCTION room_guard_c0166d345e779fa3() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN RAISE EXCEPTION 'invalid work evidence is immutable' USING ERRCODE='23514'; RETURN NEW; END $room_guard$;

DROP TRIGGER IF EXISTS trg_work_invalid_insert ON hosted_room_work_records_invalid;

CREATE TRIGGER trg_work_invalid_insert BEFORE INSERT ON hosted_room_work_records_invalid FOR EACH ROW EXECUTE FUNCTION room_guard_c0166d345e779fa3();

CREATE OR REPLACE FUNCTION room_guard_877d2fdd6bf2f220() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN RAISE EXCEPTION 'invalid work evidence is immutable' USING ERRCODE='23514'; RETURN NEW; END $room_guard$;

DROP TRIGGER IF EXISTS trg_work_invalid_update ON hosted_room_work_records_invalid;

CREATE TRIGGER trg_work_invalid_update BEFORE UPDATE ON hosted_room_work_records_invalid FOR EACH ROW EXECUTE FUNCTION room_guard_877d2fdd6bf2f220();

CREATE OR REPLACE FUNCTION room_guard_2bff29a7dc96b5a9() RETURNS trigger LANGUAGE plpgsql SET search_path FROM CURRENT AS $room_guard$ BEGIN RAISE EXCEPTION 'invalid work evidence is immutable' USING ERRCODE='23514'; RETURN OLD; END $room_guard$;

DROP TRIGGER IF EXISTS trg_work_invalid_delete ON hosted_room_work_records_invalid;

CREATE TRIGGER trg_work_invalid_delete BEFORE DELETE ON hosted_room_work_records_invalid FOR EACH ROW EXECUTE FUNCTION room_guard_2bff29a7dc96b5a9();

CREATE TABLE IF NOT EXISTS hosted_room_policy_cursors (
        room_id TEXT PRIMARY KEY, through_seq BIGINT NOT NULL DEFAULT 0,
        stopped_through_seq BIGINT NOT NULL DEFAULT 0, updated_at DOUBLE PRECISION NOT NULL DEFAULT 0);

CREATE TABLE IF NOT EXISTS hosted_room_policy_threads (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, discussion_event_id TEXT NOT NULL,
        latest_user_seq BIGINT NOT NULL, completed BIGINT NOT NULL DEFAULT 0,
        PRIMARY KEY(room_id, thread_id));

CREATE INDEX IF NOT EXISTS idx_hosted_room_policy_pending
        ON hosted_room_policy_threads(room_id, completed, latest_user_seq, thread_id);

CREATE TABLE IF NOT EXISTS hosted_room_policy_events (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, discussion_event_id TEXT NOT NULL,
        seq BIGINT NOT NULL, event_json TEXT NOT NULL, PRIMARY KEY(room_id, seq));

CREATE INDEX IF NOT EXISTS idx_hosted_room_policy_events_active
        ON hosted_room_policy_events(room_id, discussion_event_id, seq);

CREATE TABLE IF NOT EXISTS hosted_room_policy_watermarks (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, member_id TEXT NOT NULL,
        seen_through_seq BIGINT NOT NULL, PRIMARY KEY(room_id, thread_id, member_id));

CREATE TABLE IF NOT EXISTS hosted_room_policy_publications (
        room_id TEXT NOT NULL, task_id TEXT NOT NULL, kind TEXT NOT NULL,
        execution_generation BIGINT NOT NULL DEFAULT 0, seq BIGINT NOT NULL,
        PRIMARY KEY(room_id, task_id, kind, execution_generation));

CREATE TABLE IF NOT EXISTS hosted_room_policy_transcript (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, seq BIGINT NOT NULL,
        kind TEXT NOT NULL, settled_seq BIGINT, PRIMARY KEY(room_id, thread_id, seq));

CREATE TABLE IF NOT EXISTS hosted_room_policy_transcript_state (
        room_id TEXT PRIMARY KEY, schema_version BIGINT NOT NULL);

CREATE INDEX IF NOT EXISTS idx_hosted_room_user_thread
       ON hosted_room_events(room_id, (payload_json::jsonb->>'thread_id'), seq)
       WHERE kind='message.user';

CREATE TABLE IF NOT EXISTS hosted_room_attachment_blobs (
                blob_id TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL UNIQUE,
                size BIGINT NOT NULL CHECK (size > 0),
                ref_count BIGINT NOT NULL CHECK (ref_count > 0),
                created_at DOUBLE PRECISION NOT NULL
            );

CREATE TABLE IF NOT EXISTS hosted_room_attachments (
                attachment_id TEXT PRIMARY KEY,
                upload_id TEXT NOT NULL,
                room_id TEXT NOT NULL,
                event_id TEXT,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                size BIGINT NOT NULL CHECK (size > 0),
                mime TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                blob_id TEXT NOT NULL,
                recipient_member_ids_json TEXT NOT NULL DEFAULT '[]',
                viewer_access BIGINT NOT NULL DEFAULT 0 CHECK (viewer_access IN (0, 1)),
                state TEXT NOT NULL CHECK (state IN ('uploaded', 'committed', 'disbanded')),
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                expires_at DOUBLE PRECISION, catalog_name TEXT,
                UNIQUE (room_id, upload_id),
                FOREIGN KEY (blob_id) REFERENCES hosted_room_attachment_blobs(blob_id)
            );

CREATE INDEX IF NOT EXISTS idx_hosted_room_attachments_room_state
               ON hosted_room_attachments(room_id, state, created_at);

CREATE INDEX IF NOT EXISTS idx_hosted_room_attachments_expiry
               ON hosted_room_attachments(expires_at);

CREATE INDEX IF NOT EXISTS idx_hosted_room_attachments_event
               ON hosted_room_attachments(room_id, event_id);
"""

TABLES = ('hosted_rooms', 'hosted_room_events', 'hosted_room_retired_ids', 'hosted_room_links', 'hosted_room_remote_runs', 'hosted_room_revoked_grants', 'hosted_room_peer_reservations', 'hosted_room_legacy_imports', 'hosted_room_driver_leases', 'hosted_room_driver_tasks', 'hosted_room_replicas', 'hosted_room_replica_events', 'hosted_room_work_records_invalid', 'hosted_room_work_records_source', 'hosted_room_work_records_pending', 'hosted_room_policy_cursors', 'hosted_room_policy_threads', 'hosted_room_policy_events', 'hosted_room_policy_watermarks', 'hosted_room_policy_publications', 'hosted_room_policy_transcript', 'hosted_room_policy_transcript_state', 'hosted_room_attachment_blobs', 'hosted_room_attachments')
