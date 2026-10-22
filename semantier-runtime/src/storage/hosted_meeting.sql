-- Explicit meeting persistence schema v1; core migration role only.

CREATE TABLE IF NOT EXISTS meeting_rsvp_attendees (
                    monitor_id TEXT NOT NULL,
                    attendee_user_id TEXT NOT NULL,
                    message_user_id TEXT,
                    display_name TEXT,
                    response_status TEXT NOT NULL,
                    last_response_at TEXT,
                    last_followup_at TEXT,
                    followup_count BIGINT NOT NULL DEFAULT 0,
                    delivery_status TEXT NOT NULL,
                    escalated_at TEXT, negotiation_status TEXT NOT NULL DEFAULT 'none', negotiation_id TEXT,
                    PRIMARY KEY(monitor_id, attendee_user_id)
                );

CREATE TABLE IF NOT EXISTS meeting_rsvp_delivery_tasks (
                    delivery_task_id TEXT PRIMARY KEY,
                    monitor_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    target_user_id TEXT NOT NULL,
                    delivery_binding_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count BIGINT NOT NULL DEFAULT 0,
                    next_attempt_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    result_detail TEXT
                );

CREATE TABLE IF NOT EXISTS meeting_rsvp_escalations (
                    escalation_id TEXT PRIMARY KEY,
                    monitor_id TEXT NOT NULL,
                    attendee_user_id TEXT NOT NULL,
                    creator_user_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    delivery_task_id TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS meeting_rsvp_followups (
                    followup_id TEXT PRIMARY KEY,
                    monitor_id TEXT NOT NULL,
                    attendee_user_id TEXT NOT NULL,
                    round_number BIGINT NOT NULL,
                    status TEXT NOT NULL,
                    message_channel TEXT NOT NULL,
                    message_id TEXT,
                    error_detail TEXT,
                    created_at TEXT NOT NULL,
                    sent_at TEXT
                );

CREATE TABLE IF NOT EXISTS meeting_rsvp_monitors (
                    monitor_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    creator_user_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    event_revision_id TEXT NOT NULL,
                    calendar_id TEXT NOT NULL,
                    cron_job_id TEXT,
                    creator_delivery_binding_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_start_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    last_checked_at TEXT,
                    payload_json TEXT NOT NULL,
                    UNIQUE(workspace_id, event_id, event_revision_id)
                );

CREATE TABLE IF NOT EXISTS meeting_rsvp_workspace_state (
                    workspace_id TEXT PRIMARY KEY,
                    delivery_retry_scheduler_status TEXT NOT NULL,
                    delivery_retry_scheduler_detail TEXT,
                    max_followups BIGINT NOT NULL DEFAULT 3,
                    updated_at TEXT NOT NULL
                );

CREATE TABLE IF NOT EXISTS meeting_workflow_execution_receipts (
                    receipt_hash TEXT PRIMARY KEY,
                    negotiation_id TEXT NOT NULL,
                    event_count BIGINT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    UNIQUE(negotiation_id,event_count)
                );

CREATE TABLE IF NOT EXISTS meeting_time_negotiations (
                    negotiation_id TEXT PRIMARY KEY,
                    monitor_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    event_revision_id TEXT NOT NULL,
                    calendar_id TEXT NOT NULL,
                    creator_user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL DEFAULT '',
                    declined_attendee_user_id TEXT,
                    status TEXT NOT NULL CHECK (status IN (
                        'pending_decliner_input',
                        'collecting_votes',
                        'awaiting_requester_decision',
                        'consented',
                        'requester_decided',
                        'cancelled',
                        'expired',
                        'failed'
                    )),
                    current_round BIGINT NOT NULL CHECK (current_round >= 0),
                    max_rounds BIGINT NOT NULL CHECK (max_rounds >= 1),
                    duration_minutes BIGINT NOT NULL,
                    timezone TEXT NOT NULL,
                    original_start_time TEXT NOT NULL,
                    original_end_time TEXT NOT NULL,
                    selected_slot_json TEXT,
                    creator_delivery_binding_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    last_agent_error TEXT,
                    failure_reason TEXT CHECK (failure_reason IS NULL OR failure_reason IN (
                        'failed_config',
                        'failed_delivery',
                        'failed_calendar_update',
                        'stale_meeting_revision',
                        'unsafe_runtime_error'
                    )),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    terminal_authority TEXT,
                    terminal_at TEXT,
                    terminal_reason TEXT,
                    terminal_event_revision_id TEXT,
                    kanban_task_id TEXT,
                    followup_cron_job_id TEXT,
                    followup_cron_status TEXT NOT NULL DEFAULT 'not_created',
                    followup_cron_last_tick_at TEXT,
                    followup_cron_failure_count BIGINT NOT NULL DEFAULT 0,
                    next_followup_at TEXT,
                    expires_at_utc TEXT NOT NULL,
                    finalize_status TEXT NOT NULL DEFAULT 'not_started',
                    finalize_attempt_id TEXT,
                    trigger_attendee_user_ids_json TEXT NOT NULL, organization_id TEXT, workflow_binding_json TEXT, workflow_binding_hash TEXT, actor_origin_json TEXT,
                    UNIQUE(monitor_id, event_revision_id),
                    FOREIGN KEY(monitor_id) REFERENCES meeting_rsvp_monitors(monitor_id)
                );

CREATE TABLE IF NOT EXISTS meeting_time_candidate_slots (
                    slot_id TEXT PRIMARY KEY,
                    negotiation_id TEXT NOT NULL,
                    proposed_by_user_id TEXT NOT NULL,
                    round_number BIGINT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    timezone TEXT NOT NULL,
                    source_text TEXT,
                    status TEXT NOT NULL CHECK (status IN ('candidate', 'superseded', 'selected', 'rejected')),
                    created_at TEXT NOT NULL,
                    UNIQUE(negotiation_id, start_time, end_time, timezone),
                    FOREIGN KEY(negotiation_id) REFERENCES meeting_time_negotiations(negotiation_id)
                );

CREATE TABLE IF NOT EXISTS meeting_time_finalize_attempts (
                    finalize_attempt_id TEXT PRIMARY KEY,
                    finalize_idempotency_key TEXT NOT NULL UNIQUE,
                    negotiation_id TEXT NOT NULL,
                    event_revision_id TEXT NOT NULL,
                    selected_slot_id TEXT NOT NULL,
                    decision_source TEXT NOT NULL CHECK (
                        decision_source IN ('consent', 'requester_final_decision')
                    ),
                    requested_by_user_id TEXT NOT NULL,
                    calendar_update_payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN (
                        'pending',
                        'calendar_update_started',
                        'calendar_update_succeeded',
                        'calendar_update_failed_retryable',
                        'calendar_update_failed_permanent'
                    )),
                    next_retry_at TEXT,
                    calendar_update_result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK (
                        status != 'calendar_update_failed_retryable'
                        OR next_retry_at IS NOT NULL
                    ),
                    FOREIGN KEY(negotiation_id) REFERENCES meeting_time_negotiations(negotiation_id)
                );

CREATE TABLE IF NOT EXISTS meeting_time_negotiation_case_locks (
                    negotiation_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    lease_expires_at TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(negotiation_id) REFERENCES meeting_time_negotiations(negotiation_id)
                        ON DELETE CASCADE
                );

CREATE TABLE IF NOT EXISTS meeting_time_negotiation_events (
                    event_id TEXT PRIMARY KEY,
                    negotiation_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    prior_state TEXT,
                    next_state TEXT,
                    prior_state_version BIGINT,
                    next_state_version BIGINT,
                    kanban_task_id TEXT,
                    kanban_run_id BIGINT,
                    payload_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, event_sequence BIGINT,
                    FOREIGN KEY(negotiation_id) REFERENCES meeting_time_negotiations(negotiation_id)
                );

CREATE TABLE IF NOT EXISTS meeting_time_negotiation_followup_crons (
                    owner_profile TEXT NOT NULL,
                    owner_idempotency_key TEXT NOT NULL DEFAULT '',
                    workspace_id TEXT NOT NULL,
                    negotiation_id TEXT NOT NULL,
                    cron_name TEXT NOT NULL,
                    cron_job_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    acquired_at TEXT NOT NULL,
                    lease_expires_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (workspace_id, negotiation_id, cron_name),
                    FOREIGN KEY(negotiation_id) REFERENCES meeting_time_negotiations(negotiation_id)
                        ON DELETE CASCADE
                );

CREATE TABLE IF NOT EXISTS meeting_time_negotiation_messages (
                    message_event_id TEXT PRIMARY KEY,
                    negotiation_id TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK (direction IN ('outbound', 'inbound')),
                    participant_user_id TEXT NOT NULL,
                    message_channel TEXT NOT NULL,
                    message_id TEXT,
                    message_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    kanban_comment_id TEXT,
                    agent_trace_ref TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(message_channel, message_id),
                    FOREIGN KEY(negotiation_id) REFERENCES meeting_time_negotiations(negotiation_id)
                );

CREATE TABLE IF NOT EXISTS meeting_time_negotiation_participants (
                    negotiation_id TEXT NOT NULL,
                    attendee_user_id TEXT NOT NULL,
                    message_user_id TEXT,
                    display_name TEXT,
                    role TEXT NOT NULL CHECK (role IN ('decliner', 'attendee', 'requester')),
                    required_for_consent BIGINT NOT NULL CHECK (required_for_consent IN (0, 1)),
                    latest_response_status TEXT NOT NULL CHECK (latest_response_status IN (
                        'unknown',
                        'asked',
                        'accepted_slot',
                        'declined_slot',
                        'proposed_slot',
                        'abstained'
                    )),
                    latest_slot_id TEXT,
                    last_contacted_at TEXT,
                    last_response_at TEXT,
                    followup_count BIGINT NOT NULL DEFAULT 0,
                    last_followup_at TEXT,
                    delivery_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(negotiation_id, attendee_user_id),
                    FOREIGN KEY(negotiation_id) REFERENCES meeting_time_negotiations(negotiation_id)
                );

CREATE TABLE IF NOT EXISTS meeting_time_negotiation_votes (
                    vote_id TEXT PRIMARY KEY,
                    negotiation_id TEXT NOT NULL,
                    slot_id TEXT NOT NULL,
                    attendee_user_id TEXT NOT NULL,
                    vote TEXT NOT NULL CHECK (vote IN ('yes', 'no', 'propose_alternative')),
                    alternative_slot_id TEXT,
                    note TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(negotiation_id, slot_id, attendee_user_id),
                    FOREIGN KEY(negotiation_id) REFERENCES meeting_time_negotiations(negotiation_id),
                    FOREIGN KEY(slot_id) REFERENCES meeting_time_candidate_slots(slot_id)
                );

CREATE INDEX IF NOT EXISTS idx_meeting_time_events_kanban
                ON meeting_time_negotiation_events(kanban_task_id, kanban_run_id);

CREATE INDEX IF NOT EXISTS idx_meeting_time_events_replay
                ON meeting_time_negotiation_events(negotiation_id, created_at);

CREATE INDEX IF NOT EXISTS idx_meeting_time_events_type
                ON meeting_time_negotiation_events(negotiation_id, event_type);

CREATE INDEX IF NOT EXISTS idx_meeting_time_finalize_replay
                ON meeting_time_finalize_attempts(negotiation_id, created_at, finalize_attempt_id);

CREATE INDEX IF NOT EXISTS idx_meeting_time_messages_correlation
                ON meeting_time_negotiation_messages(
                    negotiation_id, participant_user_id, message_type, created_at
                );

CREATE INDEX IF NOT EXISTS idx_meeting_time_messages_replay
                ON meeting_time_negotiation_messages(negotiation_id, created_at, message_event_id);

CREATE INDEX IF NOT EXISTS idx_meeting_time_negotiation_followup_crons_owner
                ON meeting_time_negotiation_followup_crons(owner_profile, owner_idempotency_key);

CREATE INDEX IF NOT EXISTS idx_meeting_time_negotiation_followup_crons_status
                ON meeting_time_negotiation_followup_crons(workspace_id, status, lease_expires_at);

CREATE INDEX IF NOT EXISTS idx_meeting_time_negotiations_expiry
                ON meeting_time_negotiations(status, expires_at_utc);

CREATE INDEX IF NOT EXISTS idx_meeting_time_negotiations_followup_status
                ON meeting_time_negotiations(followup_cron_status);

CREATE INDEX IF NOT EXISTS idx_meeting_time_negotiations_kanban_task
                ON meeting_time_negotiations(kanban_task_id);

CREATE INDEX IF NOT EXISTS idx_meeting_time_negotiations_workspace_status
                ON meeting_time_negotiations(workspace_id, status, updated_at);

CREATE INDEX IF NOT EXISTS idx_meeting_time_participants_delivery
                ON meeting_time_negotiation_participants(negotiation_id, delivery_status, last_contacted_at);

CREATE INDEX IF NOT EXISTS idx_meeting_time_participants_followup
                ON meeting_time_negotiation_participants(
                    negotiation_id,
                    required_for_consent,
                    followup_count,
                    last_followup_at
                );

CREATE INDEX IF NOT EXISTS idx_meeting_time_slots_case_round
                ON meeting_time_candidate_slots(negotiation_id, round_number, status);

CREATE INDEX IF NOT EXISTS idx_meeting_time_slots_replay
                ON meeting_time_candidate_slots(negotiation_id, created_at, slot_id);

CREATE INDEX IF NOT EXISTS idx_meeting_time_votes_replay
                ON meeting_time_negotiation_votes(negotiation_id, created_at, vote_id);

CREATE OR REPLACE FUNCTION meeting_reject_history_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$ BEGIN
    RAISE EXCEPTION '%',TG_ARGV[0] USING ERRCODE='23514';
END; $$;
ALTER FUNCTION meeting_reject_history_mutation() SET search_path FROM CURRENT;
DROP TRIGGER IF EXISTS meeting_receipt_no_update ON meeting_workflow_execution_receipts;
CREATE TRIGGER meeting_receipt_no_update BEFORE UPDATE ON meeting_workflow_execution_receipts
FOR EACH ROW EXECUTE FUNCTION meeting_reject_history_mutation('WORKFLOW_RECEIPT_IMMUTABLE');
DROP TRIGGER IF EXISTS meeting_receipt_no_delete ON meeting_workflow_execution_receipts;
CREATE TRIGGER meeting_receipt_no_delete BEFORE DELETE ON meeting_workflow_execution_receipts
FOR EACH ROW EXECUTE FUNCTION meeting_reject_history_mutation('WORKFLOW_RECEIPT_IMMUTABLE');
DROP TRIGGER IF EXISTS meeting_native_event_no_update ON meeting_time_negotiation_events;
CREATE TRIGGER meeting_native_event_no_update BEFORE UPDATE ON meeting_time_negotiation_events
FOR EACH ROW EXECUTE FUNCTION meeting_reject_history_mutation('WORKFLOW_EVENT_HISTORY_IMMUTABLE');
DROP TRIGGER IF EXISTS meeting_native_event_no_delete ON meeting_time_negotiation_events;
CREATE TRIGGER meeting_native_event_no_delete BEFORE DELETE ON meeting_time_negotiation_events
FOR EACH ROW EXECUTE FUNCTION meeting_reject_history_mutation('WORKFLOW_EVENT_HISTORY_IMMUTABLE');
DROP TRIGGER IF EXISTS meeting_actor_origin_immutable ON meeting_time_negotiations;
CREATE TRIGGER meeting_actor_origin_immutable BEFORE UPDATE OF actor_origin_json ON meeting_time_negotiations
FOR EACH ROW WHEN (OLD.actor_origin_json IS DISTINCT FROM NEW.actor_origin_json)
EXECUTE FUNCTION meeting_reject_history_mutation('ACTOR_ORIGIN_IMMUTABLE');
DROP TRIGGER IF EXISTS meeting_workflow_binding_immutable ON meeting_time_negotiations;
CREATE TRIGGER meeting_workflow_binding_immutable
BEFORE UPDATE OF organization_id,workspace_id,workflow_binding_json,workflow_binding_hash ON meeting_time_negotiations
FOR EACH ROW WHEN (
    OLD.workflow_binding_json IS DISTINCT FROM NEW.workflow_binding_json
    OR OLD.workflow_binding_hash IS DISTINCT FROM NEW.workflow_binding_hash
    OR OLD.organization_id IS DISTINCT FROM NEW.organization_id
    OR (OLD.workflow_binding_json IS NOT NULL AND OLD.workspace_id IS DISTINCT FROM NEW.workspace_id)
) EXECUTE FUNCTION meeting_reject_history_mutation('WORKFLOW_INSTANCE_REBINDING_FORBIDDEN');
