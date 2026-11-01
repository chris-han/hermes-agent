"""Core policy for one explicitly admitted missing-slot clarification.

Intent/domain/inbox/outbox share PG. Wait and wake use recoverable owner handoffs;
external SDK calls are outside transactions. Transport acceptance is not an answer.
"""

from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import secrets
import uuid

from agents.durable_event_arbitration import DurableEventInbox, WorkflowVersion
from agents.durable_responsibility import (
    DurableResponsibilityStore,
    EVENT,
    WAIT_EXTERNAL,
    DETERMINISTIC_CONTINUE,
)
from agents.effect_outbox import (
    AtomicEffectOutbox,
    DISPATCHING,
    COMPLETED,
)
from agents.effect_reconciliation import (
    EffectReconciliationService,
    ProviderEvidence,
    CONFIRMED_COMPLETED,
    STILL_UNKNOWN,
)
from contracts.meeting_delivery import (
    CONTRACT,
    PURPOSE,
    canonical,
    digest,
    utc,
    instant,
    slot,
    bot_generation,
)
from contracts.workflow_execution import (
    verify_meeting_binding,
    read_active_meeting_binding,
)
from storage import hosted_postgres as pg


def selected(store, case_id):
    if not pg.enabled():
        return False  # Incumbent SQLite cases are not migrated.
    with store._connect() as c:
        if (
            c.execute(
                "SELECT to_regclass(current_schema()||'.meeting_core_delivery_bindings')"
            ).fetchone()[0]
            is None
        ):
            return False
        return (
            c.execute(
                "SELECT 1 FROM meeting_core_delivery_bindings WHERE negotiation_id=%s",
                (case_id,),
            ).fetchone()
            is not None
        )


class MeetingClarificationDeliveryService:
    def __init__(self, store):
        if not pg.enabled():
            raise RuntimeError("core clarification requires PostgreSQL")
        self.store = store
        with store._connect() as c:
            pg.require_tables(
                c, "meeting_core_delivery_bindings", "meeting_core_clarifications"
            )
        self.responsibilities = DurableResponsibilityStore(store.path)
        self.outbox = AtomicEffectOutbox(store.path)
        self.inbox = DurableEventInbox(store.path)

    def binding(self, case_id):
        with self.store._connect() as c:
            row = c.execute(
                "SELECT * FROM meeting_core_delivery_bindings WHERE negotiation_id=%s",
                (case_id,),
            ).fetchone()
        if row is None:
            raise PermissionError("new core delivery admission required")
        a = json.loads(row["association_json"])
        if digest(a) != row["association_hash"] or row["revoked_at"]:
            raise PermissionError("delivery binding revoked or corrupt")
        return a

    def _authority(self, a, *, pending=True, contract=CONTRACT, purpose=PURPOSE):
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from agents.auth_db import get_feishu_bot_config_for_workspace
        from services.runtime_errors import RuntimeContextError

        actor = RuntimeContextDTO(
            organization_id=a["organization_id"],
            workspace_id=a["workspace_id"],
            actor_id=a["principal_id"],
        )
        try:
            RuntimeAuthorityResolver.require_current_actor(actor_context=actor)
            _, org = RuntimeAuthorityResolver._current_identity_context(actor)
        except RuntimeContextError as exc:
            raise PermissionError("current meeting principal required") from exc
        if org.get("membership_status") != "active":
            raise PermissionError("revoked meeting principal")
        case = self.store.get_negotiation(a["negotiation_id"])
        if (
            a["contract"] != contract
            or a["purpose"] != purpose
            or instant(a["expires_at"]) <= datetime.now(timezone.utc)
            or case["workspace_id"] != a["workspace_id"]
            or case["organization_id"] != a["organization_id"]
            or case["creator_user_id"] != a["principal_id"]
            or case["workflow_binding_hash"] != a["workflow_binding_hash"]
            or (
                pending
                and (
                    case["status"] != "pending_decliner_input"
                    or int(case["current_round"]) + 1 != a["round"]
                )
            )
        ):
            raise PermissionError("meeting scope/state/definition changed")
        workflow = json.loads(case["workflow_binding_json"])
        verify_meeting_binding(
            workflow,
            organization_id=a["organization_id"],
            workspace_id=a["workspace_id"],
        )
        proof = digest(
            dict(
                monitor_id=case["monitor_id"],
                event_revision_id=case["event_revision_id"],
                participant_id=a["participant_id"],
            )
        )
        responsibility = self.responsibilities.get(a["responsibility_id"])
        if (
            a["participant_proof"] != proof
            or a["definition_hash"] != workflow["workflowHash"]
            or a["execution_hash"] != workflow["executionHash"]
            or a["work_ref"] != "meeting:" + case["negotiation_id"]
            or a["expires_at"] != case["expires_at_utc"]
            or responsibility is None
            or responsibility.domain_ref != case["negotiation_id"]
            or responsibility.workspace_id != a["workspace_id"]
            or (pending and responsibility.lifecycle_state == "TERMINAL")
        ):
            raise PermissionError("work/participant/definition proof mismatch")
        participant = next(
            (
                p
                for p in self.store.list_negotiation_participants(a["negotiation_id"])
                if p["attendee_user_id"] == a["participant_id"]
            ),
            None,
        )
        if (
            participant is None
            or (participant.get("message_user_id") or participant["attendee_user_id"])
            != a["recipient_id"]
        ):
            raise PermissionError("admitted participant required")
        bot = get_feishu_bot_config_for_workspace(a["workspace_id"]) or {}
        if (
            not bot.get("validated_at")
            or bot.get("last_error")
            or bot.get("owner_user_id") != a["principal_id"]
            or not all(
                bot.get(k)
                for k in (
                    "app_id",
                    "app_secret",
                    "bot_open_id",
                    "tenant_key",
                    "verification_token",
                    "encrypt_key",
                )
            )
            or bot_generation(bot) != a["bot_generation"]
        ):
            raise PermissionError("current validated bot binding required")
        if any(
            a[key] != bot[field]
            for key, field in (
                ("app_id", "app_id"),
                ("bot_id", "bot_open_id"),
                ("tenant_key", "tenant_key"),
            )
        ):
            raise PermissionError("bot/app/tenant proof mismatch")
        return case, bot

    def admit_new_case(self, context, *, monitor_id, event_revision_id, participant_id, retry_policy=None):
        # Server ingress supplies the genuine RequestContext; request/model metadata
        # cannot supply definition pins, destination, purpose or contract selection.
        from agents.core_file_scope import CoreFileScope

        CoreFileScope.issue(context, "meeting-delivery-admission")
        if retry_policy not in (None, "clarification-one-retry-v1"):
            raise ValueError("unsupported server retry policy")
        monitor = self.store.get_monitor(monitor_id)
        if (
            monitor["workspace_id"] != context.workspace_id
            or monitor["creator_user_id"] != context.user_id
            or monitor["event_revision_id"] != event_revision_id
        ):
            raise PermissionError("owned admitted meeting required")
        request_hash = digest(dict(principal_id=context.user_id, organization_id=context.organization_id,
            workspace_id=context.workspace_id, monitor_id=monitor_id, event_revision_id=event_revision_id,
            participant_id=participant_id, retry_policy=retry_policy))
        # A session advisory lock serializes reservation and recovery across processes;
        # closing the connection after a crash releases the lock, not the reservation.
        lock_conn = self.store._connect()
        try:
            lock_conn.execute('SELECT pg_advisory_lock(hashtextextended(%s,0))', ('meeting-admission:'+monitor_id+':'+event_revision_id,))
            with self.store._connect() as c:
                c.execute('SELECT monitor_id FROM meeting_rsvp_monitors WHERE monitor_id=%s FOR UPDATE', (monitor_id,))
                reservation = c.execute('SELECT request_hash FROM meeting_core_admissions WHERE monitor_id=%s AND event_revision_id=%s',
                    (monitor_id, event_revision_id)).fetchone()
                existing = c.execute('SELECT negotiation_id FROM meeting_time_negotiations WHERE monitor_id=%s', (monitor_id,)).fetchall()
                if reservation is None:
                    if existing:
                        raise PermissionError('existing case cannot be adopted')
                    c.execute('INSERT INTO meeting_core_admissions VALUES(%s,%s,%s)',
                        (monitor_id,event_revision_id,request_hash))
                elif reservation['request_hash'] != request_hash:
                    raise PermissionError('admission identity conflict')
            self._cut('after_admission_reservation')
            existing = self.store.list_negotiations_for_monitor(monitor_id)
            if existing:
                case = next((c for c in existing if c['event_revision_id'] == event_revision_id), None)
                if case is None:
                    raise PermissionError('admission case conflict')
                if selected(self.store, case['negotiation_id']):
                    a = self.binding(case['negotiation_id'])
                    self._authority(a)
                    if a['participant_id'] != participant_id or a.get('retry_policy') != retry_policy:
                        raise PermissionError('admission cannot retarget or change retry policy')
                    return a
            a = self._issue_new_case(context, monitor_id=monitor_id,
                event_revision_id=event_revision_id, participant_id=participant_id,
                allow_reserved_recovery=bool(existing), admission_request_hash=request_hash)
            if retry_policy is not None:
                a['retry_policy'] = retry_policy
            with self.store._connect() as c:
                c.execute('INSERT INTO meeting_core_delivery_bindings VALUES(%s,%s,%s,0,NULL)',
                    (a['negotiation_id'], canonical(a).decode(), digest(a)))
            return a
        finally:
            lock_conn.close()

    def _issue_new_case(
        self,
        context,
        *,
        monitor_id,
        event_revision_id,
        participant_id,
        contract=CONTRACT,
        purpose=PURPOSE,
        allow_reserved_recovery=False,
        admission_request_hash=None,
    ):
        from agents.core_file_scope import CoreFileScope
        from agents.auth_db import get_feishu_bot_config_for_workspace

        CoreFileScope.issue(context, "meeting-delivery-admission")
        monitor = self.store.get_monitor(monitor_id)
        if (
            monitor["workspace_id"] != context.workspace_id
            or monitor["creator_user_id"] != context.user_id
            or monitor["event_revision_id"] != event_revision_id
            or (not allow_reserved_recovery and self.store.list_negotiations_for_monitor(monitor_id))
        ):
            raise PermissionError("new owned meeting required")
        existing = self.store.list_negotiations_for_monitor(monitor_id) if allow_reserved_recovery else []
        workflow = json.loads(existing[0]["workflow_binding_json"]) if existing else read_active_meeting_binding(context)
        case = self.store.create_or_get_negotiation_case(
            monitor_id=monitor_id,
            event_revision_id=event_revision_id,
            trigger_attendee_user_id=participant_id,
            workflow_binding=workflow,
            runtime_context=context,
            admission_request_hash=admission_request_hash,
        )
        self._cut('after_admission_case')
        bot = get_feishu_bot_config_for_workspace(context.workspace_id) or {}
        participant = next(
            (
                p
                for p in self.store.list_negotiation_participants(
                    case["negotiation_id"]
                )
                if p["attendee_user_id"] == participant_id
            ),
            None,
        )
        if participant is None:
            raise PermissionError("admitted participant required")
        responsibility = self.responsibilities.ensure(
            responsibility_kind="feishu_meeting_negotiation",
            workspace_id=context.workspace_id,
            domain_ref=case["negotiation_id"],
            workflow_id="feishu-meeting-coordinator",
            workflow_instance_id=case["negotiation_id"],
        )
        a = dict(
            contract=contract,
            purpose=purpose,
            principal_id=context.user_id,
            organization_id=context.organization_id,
            workspace_id=context.workspace_id,
            negotiation_id=case["negotiation_id"],
            work_ref="meeting:" + case["negotiation_id"],
            responsibility_id=responsibility.responsibility_id,
            workflow_binding_hash=case["workflow_binding_hash"],
            definition_hash=workflow["workflowHash"],
            execution_hash=workflow["executionHash"],
            participant_id=participant_id,
            recipient_id=participant.get("message_user_id") or participant_id,
            participant_proof=digest(
                dict(
                    monitor_id=monitor_id,
                    event_revision_id=event_revision_id,
                    participant_id=participant_id,
                )
            ),
            app_id=bot.get("app_id"),
            bot_id=bot.get("bot_open_id"),
            tenant_key=bot.get("tenant_key"),
            bot_generation=bot_generation(bot),
            round=int(case["current_round"]) + 1,
            issued_at=utc(),
            expires_at=case["expires_at_utc"],
        )
        self._authority(a, contract=contract, purpose=purpose)
        return a

    def _row(self, case_id):
        with self.store._connect() as c:
            row = c.execute(
                "SELECT * FROM meeting_core_clarifications WHERE negotiation_id=%s",
                (case_id,),
            ).fetchone()
        return dict(row) if row else None

    def _cut(self, stage):
        pass  # Crash-test scheduling seam, no production callback.

    @staticmethod
    def _input(event):
        return {
            key: getattr(event, key)
            for key in (
                "sender",
                "app_id",
                "tenant_key",
                "message_id",
                "chat_id",
                "parent_id",
                "text",
            )
        }

    def accept_verified(self, event):
        # VerifiedIngress is minted only after SDK callback token/signature checks.
        from services.meeting_clarification_feishu import VerifiedIngress

        if not isinstance(event, VerifiedIngress) or not event.authentic():
            raise PermissionError("verified SDK ingress required")
        a = self.binding(event.case_id)
        self._authority(a, pending=False)
        if (
            event.sender != a["recipient_id"]
            or event.app_id != a["app_id"]
            or event.tenant_key != a["tenant_key"]
            or event.generation != a["bot_generation"]
            or event.chat_type != "p2p"
        ):
            raise PermissionError("foreign meeting reply")
        row = self._row(event.case_id)
        if row:
            return self._reply(a, row, event)
        self._authority(a)
        # Only the selected missing-normalized-slot ambiguity. Other reasons stay
        # explicit incumbent behavior outside this admitted new contract.
        try:
            body = json.loads(event.text)
        except ValueError:
            body = {}
        if isinstance(body, dict) and body.get("slot"):
            raise ValueError(
                "initial structured slot uses existing non-clarification path"
            )
        nonce = secrets.token_urlsafe(32)
        text = (
            (Path(__file__).parents[1] / "prompts/meetings/missing_normalized_slot.txt")
            .read_text()
            .format(correlation=nonce, round=a["round"])
            .strip()
        )
        payload = dict(
            initiating_input=self._input(event),
            initiating_input_hash=digest(self._input(event)),
            contract=CONTRACT,
            association_hash=digest(a),
            recipient_id=a["recipient_id"],
            app_id=a["app_id"],
            tenant_key=a["tenant_key"],
            bot_id=a["bot_id"],
            chat_id=event.chat_id,
            purpose=PURPOSE,
            round=a["round"],
            text=text,
            correlation=nonce,
            ingress_hash=event.proof_hash,
        )
        oid = "clarification_" + digest(dict(case=event.case_id, round=a["round"]))[:32]
        with self.store._connect() as c:
            pg.begin_write(c)
            c.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (event.case_id,)
            )

            def load(conn):
                b = conn.execute(
                    "SELECT protocol_version FROM meeting_core_delivery_bindings WHERE negotiation_id=%s",
                    (event.case_id,),
                ).fetchone()
                return WorkflowVersion(b[0])

            def transition(conn, current):
                if current.version != 0:
                    raise ValueError("clarification already pending")
                effect = self.outbox.enqueue_ready(
                    conn,
                    workflow_id=event.case_id,
                    effect_type="meeting_missing_slot",
                    semantic_round=a["round"],
                    canonical_payload=canonical(payload),
                )
                self.store._record_negotiation_event(
                    conn,
                    negotiation_id=event.case_id,
                    event_type="CORE_CLARIFICATION_REQUESTED",
                    actor_type="participant",
                    actor_id=a["participant_id"],
                    payload=dict(
                        obligation_id=oid,
                        initiating_event=event.message_id,
                        proof_hash=event.proof_hash,
                    ),
                )
                version = conn.execute(
                    "SELECT COUNT(*) FROM meeting_time_negotiation_events WHERE negotiation_id=%s",
                    (event.case_id,),
                ).fetchone()[0]
                conn.execute(
                    "INSERT INTO meeting_core_clarifications (obligation_id,negotiation_id,initiating_event,round_number,payload_json,effect_id,state,domain_version,wait_version,send_lease_until,response_event,retry_policy) VALUES(%s,%s,%s,%s,%s,%s,'WAIT_PENDING',%s,NULL,NULL,NULL,%s)",
                    (
                        oid,
                        event.case_id,
                        event.message_id,
                        a["round"],
                        canonical(payload).decode(),
                        effect.effect_id,
                        version,
                        a.get("retry_policy"),
                    ),
                )
                conn.execute(
                    "UPDATE meeting_core_delivery_bindings SET protocol_version=1 WHERE negotiation_id=%s",
                    (event.case_id,),
                )
                self._cut("before_intent_commit")
                return WorkflowVersion(1), oid

            self.inbox.accept(
                workflow_type=CONTRACT,
                workflow_id=event.case_id,
                event_id=event.message_id,
                event_type=PURPOSE,
                dedupe_key="ask:" + event.message_id,
                expected_state_version=0,
                load_state=load,
                transition=transition,
                connection=c,
            )
        self._cut("after_intent_commit")
        self._ensure_wait(a)
        return self._row(event.case_id)

    def _ensure_wait(self, a):
        # Serialize the recoverable handoff; responsibility commits remain its
        # own transactions. A crash cannot duplicate or lose the addressed wait.
        with self.store._connect() as c:
            c.execute(
                "BEGIN"
            )  # Do not hold the shared authority-write lock across owner calls.
            c.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                ("meeting-wait:" + a["negotiation_id"],),
            )
            raw = c.execute(
                "SELECT * FROM meeting_core_clarifications WHERE negotiation_id=%s",
                (a["negotiation_id"],),
            ).fetchone()
            if raw is None or raw["state"] != "WAIT_PENDING":
                return
            row = dict(raw)
            source = "meeting-clarification:" + row["obligation_id"]
            waits = [
                w
                for w in self.responsibilities.pending_waits(a["responsibility_id"])
                if w.source_ref == source
            ]
            if not waits:
                from agents.durable_responsibility import DurableResponsibilityError

                try:
                    claim = self.responsibilities.claim(
                        responsibility_id=a["responsibility_id"],
                        owner_id="meeting-clarification-wait",
                    )
                except DurableResponsibilityError:
                    return  # Existing live/expired-lease owner decides recovery.
                waits = [
                    self.responsibilities.register_wait(
                        claim,
                        kind=EVENT,
                        source_ref=source,
                        correlation_key=row["obligation_id"],
                        lifecycle_state=WAIT_EXTERNAL,
                        continuation_disposition=DETERMINISTIC_CONTINUE,
                    )
                ]
            c.execute(
                "UPDATE meeting_core_clarifications SET state='WAITING',wait_version=%s WHERE obligation_id=%s AND state='WAIT_PENDING'",
                (waits[0].expected_checkpoint_version, row["obligation_id"]),
            )

    def dispatch(
        self, case_id, transport, *, expected_checkpoint=None, lease_seconds=30
    ):
        a = self.binding(case_id)
        _, bot = self._authority(a)
        self._ensure_wait(a)
        row = self._row(case_id)
        if row is None or row["state"] != "WAITING":
            raise PermissionError("current clarification wait required")
        responsibility = self.responsibilities.get(a["responsibility_id"])
        if (
            responsibility.lifecycle_state != WAIT_EXTERNAL
            or responsibility.checkpoint_version != row["wait_version"]
            or expected_checkpoint not in (None, row["wait_version"])
        ):
            raise PermissionError("stale clarification claim")
        if transport.generation != a["bot_generation"]:
            raise PermissionError("transport bot generation mismatch")
        if not 1 <= lease_seconds <= 300:
            raise ValueError("bounded dispatch lease required")
        attempt = uuid.uuid4().hex
        with self.store._connect() as c:
            pg.begin_write(c)
            if not self.outbox.claim_dispatch_in_transaction(
                c, effect_id=row["effect_id"], attempt_id=attempt
            ):
                return self.outbox.get(row["effect_id"])
            c.execute(
                "UPDATE meeting_core_clarifications SET send_lease_until=%s WHERE obligation_id=%s",
                ((datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat(), row["obligation_id"]),
            )
        # This core-owned phase executes before any transport method can be called.
        # Its test fault is the only safe classifier; transport exceptions below
        # remain uncertain even if they describe themselves as safe.
        try:
            if row["retry_policy"] == "clarification-one-retry-v1":
                self._cut("before_create_preparation")
        except Exception as exc:
            with self.store._connect() as c:
                pg.begin_write(c)
                updated = self.outbox.mark_failed_safe_in_transaction(
                    c, effect_id=row["effect_id"], attempt_id=attempt,
                    failure_detail="pre-create:" + type(exc).__name__,
                )
                if not updated:
                    raise RuntimeError("safe failure lost dispatch fence")
                updated = c.execute(
                    "UPDATE meeting_core_clarifications SET retry_attempts=CASE WHEN retry_attempts=0 THEN 1 ELSE retry_attempts END, retry_due_at=CASE WHEN retry_attempts=0 THEN %s ELSE NULL END, retry_reason=CASE WHEN retry_attempts=0 THEN 'PRE_CREATE_FAILURE' ELSE 'EXHAUSTED' END,send_lease_until=NULL WHERE obligation_id=%s AND state='WAITING' AND retry_policy='clarification-one-retry-v1' AND retry_attempts IN (0,2)",
                    ((datetime.now(timezone.utc)+timedelta(seconds=120)).isoformat(), row["obligation_id"]),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("safe failure has no admitted retry policy")
                self._cut("before_safe_commit")
            self._cut("after_safe_commit")
            return self.outbox.get(row["effect_id"])
        try:
            self._cut("before_send")
            self._authority(self.binding(case_id))
            current = self.responsibilities.get(a["responsibility_id"])
            if (
                current.lifecycle_state != WAIT_EXTERNAL
                or current.checkpoint_version != row["wait_version"]
            ):
                raise PermissionError("wait changed before send")
            effect = self.outbox.get(row["effect_id"])
            payload = json.loads(effect.canonical_payload)
            result = transport.send(payload, effect.idempotency_key)
            mid = result.get("message_ids", {}).get(a["recipient_id"])
            if (
                not mid
                or result.get("failed")
                or result.get("delivered") != [a["recipient_id"]]
            ):
                raise RuntimeError("uncertain SDK send")
            self.outbox.pin_provider_correlation(
                effect_id=effect.effect_id, correlation_ref=mid
            )
            self._cut("after_send")
            self.outbox.mark_completed(
                effect_id=effect.effect_id, attempt_id=attempt, receipt_ref=mid
            )
        except Exception as exc:
            self.outbox.mark_effect_unknown(
                effect_id=row["effect_id"],
                attempt_id=attempt,
                failure_detail=type(exc).__name__,
            )
        return self.outbox.get(row["effect_id"])

    def recover_expired_retries(self, *, limit=32):
        """Lost claimed retries are uncertain; recovery never sends or requeues."""
        from agents.effect_outbox import InvalidEffectTransition
        if not 1 <= limit <= 100:
            raise ValueError("bounded recovery batch required")
        with self.store._connect() as c:
            rows = c.execute(
                "SELECT e.effect_id,e.dispatch_attempt_id FROM meeting_core_clarifications q JOIN semantier_effect_outbox e ON e.effect_id=q.effect_id WHERE q.retry_policy='clarification-one-retry-v1' AND e.state='DISPATCHING' AND q.send_lease_until IS NOT NULL AND q.send_lease_until::timestamptz<=%s::timestamptz ORDER BY q.send_lease_until::timestamptz,q.obligation_id LIMIT %s",
                (utc(), limit),
            ).fetchall()
        for row in rows:
            try:
                self.outbox.mark_effect_unknown(effect_id=row['effect_id'], attempt_id=row['dispatch_attempt_id'], failure_detail='expired dispatch lease; possible send')
            except InvalidEffectTransition:
                pass  # Another owner completed or fenced this attempt.
        return len(rows)

    def due_retries(self, *, limit=32, now=None):
        """Read-only bounded selection; this method never constructs a transport."""
        if not 1 <= limit <= 100:
            raise ValueError("bounded retry batch required")
        now = now or utc()
        with self.store._connect() as c:
            rows = c.execute(
                "SELECT q.negotiation_id,q.obligation_id FROM meeting_core_clarifications q JOIN meeting_core_delivery_bindings b USING(negotiation_id) JOIN semantier_effect_outbox e ON e.effect_id=q.effect_id WHERE b.revoked_at IS NULL AND q.state='WAITING' AND q.retry_policy='clarification-one-retry-v1' AND q.retry_due_at IS NOT NULL AND q.retry_due_at::timestamptz<=%s::timestamptz AND ((e.state='FAILED_SAFE' AND q.retry_attempts=1) OR (e.state='READY' AND q.retry_attempts=2)) ORDER BY q.retry_due_at::timestamptz,q.obligation_id LIMIT %s",
                (now, limit),
            ).fetchall()
        return rows

    def retry_due(self, *, limit=32, transport_factory=None):
        """Dispatch only admitted overdue safe retries, with outbox fencing."""
        now = utc()
        rows = self.due_retries(limit=limit, now=now)
        if transport_factory is None:
            from services.meeting_clarification_feishu import FeishuClarificationTransport
            endpoint = os.environ.get("SEMANTIER_MEETING_PROTOCOL_ENDPOINT")
            transport_factory = lambda bot: FeishuClarificationTransport(bot, endpoint=endpoint)
        for item in rows:
            case_id = item['negotiation_id']
            try:
                a = self.binding(case_id)
                _, bot = self._authority(a)
                row = self._row(case_id)
                responsibility = self.responsibilities.get(a['responsibility_id'])
                if responsibility.lifecycle_state != WAIT_EXTERNAL or responsibility.checkpoint_version != row['wait_version']:
                    continue
                with self.store._connect() as c:
                    pg.begin_write(c)
                    eligible = c.execute(
                        "SELECT 1 FROM meeting_core_clarifications WHERE obligation_id=%s AND state='WAITING' AND retry_attempts=1 AND retry_due_at::timestamptz<=%s::timestamptz AND retry_policy='clarification-one-retry-v1' FOR UPDATE",
                        (row['obligation_id'], now),
                    ).fetchone()
                    if eligible and self.outbox.retry_failed_safe_in_transaction(c, effect_id=row['effect_id']):
                        c.execute("UPDATE meeting_core_clarifications SET retry_attempts=2 WHERE obligation_id=%s AND retry_attempts=1", (row['obligation_id'],))
                self._cut("after_retry_ready_commit")
                self.dispatch(case_id, transport_factory(bot), expected_checkpoint=row['wait_version'])
            except PermissionError:
                continue
        return len(rows)

    def _reply(self, a, row, event):
        if event.message_id == row["initiating_event"]:
            if (
                digest(self._input(event))
                != json.loads(row["payload_json"])["initiating_input_hash"]
            ):
                raise PermissionError("duplicate event payload changed")
            return row
        if row["state"] == "APPLIED":
            if event.message_id == row["response_event"]:
                original = next(
                    json.loads(e["payload_json"])
                    for e in self.store.list_negotiation_events(event.case_id)
                    if e["event_type"] == "SLOT_PROPOSED"
                    and json.loads(e["payload_json"]).get("clarification_event")
                    == event.message_id
                )
                if original.get("reply_input_hash") != digest(self._input(event)):
                    raise PermissionError("duplicate response payload changed")
                return row
            raise PermissionError("clarification already consumed")
        self._authority(a)
        effect = self.outbox.get(row["effect_id"])
        payload = json.loads(row["payload_json"])
        if effect.state != COMPLETED:
            raise PermissionError("delivery receipt required before reply")
        try:
            body = json.loads(event.text)
        except ValueError:
            raise ValueError("unstructured reply remains unresolved") from None
        if (
            not isinstance(body, dict)
            or set(body) != {"correlation", "round", "slot"}
            or body["correlation"] != payload["correlation"]
            or body["round"] != a["round"]
            or event.chat_id != payload["chat_id"]
            or (event.parent_id and event.parent_id != effect.receipt_ref)
        ):
            raise PermissionError("reply correlation mismatch")
        normalized = slot(body["slot"])
        with self.store._connect() as c:
            pg.begin_write(c)
            c.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (event.case_id,)
            )
            self._authority(self.binding(event.case_id))

            def load(conn):
                return WorkflowVersion(
                    conn.execute(
                        "SELECT protocol_version FROM meeting_core_delivery_bindings WHERE negotiation_id=%s",
                        (event.case_id,),
                    ).fetchone()[0]
                )

            def transition(conn, current):
                version = conn.execute(
                    "SELECT COUNT(*) FROM meeting_time_negotiation_events WHERE negotiation_id=%s",
                    (event.case_id,),
                ).fetchone()[0]
                if version != row["domain_version"]:
                    raise PermissionError("meeting domain version changed")
                self.store.apply_core_clarification(
                    conn,
                    negotiation_id=event.case_id,
                    participant_id=a["participant_id"],
                    round_number=a["round"],
                    slot=normalized,
                    event_id=event.message_id,
                    input_hash=digest(self._input(event)),
                )
                conn.execute(
                    "UPDATE meeting_core_clarifications SET state='APPLIED',response_event=%s WHERE obligation_id=%s AND state='WAITING'",
                    (event.message_id, row["obligation_id"]),
                )
                conn.execute(
                    "UPDATE meeting_core_delivery_bindings SET protocol_version=2 WHERE negotiation_id=%s",
                    (event.case_id,),
                )
                return WorkflowVersion(2), None

            self.inbox.accept(
                workflow_type=CONTRACT,
                workflow_id=event.case_id,
                event_id=event.message_id,
                event_type="STRUCTURED_SLOT",
                dedupe_key="reply:" + event.message_id,
                expected_state_version=1,
                load_state=load,
                transition=transition,
                connection=c,
            )
        self._cut("after_reply_commit")
        self._wake(a, self._row(event.case_id))
        return self._row(event.case_id)

    def _wake(self, a, row):
        self.responsibilities.accept_wake(
            responsibility_id=a["responsibility_id"],
            kind=EVENT,
            source_ref="meeting-clarification:" + row["obligation_id"],
            correlation_key=row["obligation_id"],
            dedupe_key="clarified:" + row["response_event"],
            expected_checkpoint_version=row["wait_version"],
        )

    def drain_pending(self):
        # Explicit bounded startup owner call; no scheduler, model or network.
        with self.store._connect() as c:
            rows = c.execute(
                "SELECT q.* FROM meeting_core_clarifications q JOIN meeting_core_delivery_bindings b USING(negotiation_id) JOIN semantier_responsibilities r ON r.responsibility_id=(b.association_json::jsonb->>'responsibility_id') JOIN semantier_effect_outbox e ON e.effect_id=q.effect_id WHERE b.revoked_at IS NULL AND (q.state='WAIT_PENDING' OR (q.state='APPLIED' AND r.lifecycle_state='WAIT_EXTERNAL') OR (e.state='DISPATCHING' AND q.send_lease_until<=%s)) ORDER BY q.obligation_id LIMIT 100",
                (utc(),),
            ).fetchall()
        for raw in rows:
            row = dict(raw)
            try:
                a = self.binding(row["negotiation_id"])
                self._authority(a, pending=row["state"] != "APPLIED")
            except PermissionError:
                continue
            if row["state"] == "WAIT_PENDING":
                self._authority(a)
                self._ensure_wait(a)
            elif row["state"] == "APPLIED":
                self._authority(a, pending=False)
                self._wake(a, row)
            effect = self.outbox.get(row["effect_id"])
            if (
                effect.state == DISPATCHING
                and row["send_lease_until"]
                and instant(row["send_lease_until"]) <= datetime.now(timezone.utc)
            ):
                self.outbox.mark_effect_unknown(
                    effect_id=effect.effect_id,
                    attempt_id=effect.dispatch_attempt_id,
                    failure_detail="expired dispatch lease; possible send",
                )
        return len(rows)

    def reconcile(self, case_id, transport):
        a = self.binding(case_id)
        self._authority(a)
        row = self._row(case_id)
        payload = json.loads(row["payload_json"])

        class ExactEvidence:
            def inspect(self, effect):
                if (
                    transport.generation != a["bot_generation"]
                    or not effect.provider_correlation_ref
                ):
                    return ProviderEvidence(STILL_UNKNOWN)
                try:
                    lookup = transport.get_message(effect.provider_correlation_ref)
                except Exception:
                    return ProviderEvidence(STILL_UNKNOWN)
                message = lookup.get("message") or {}
                sender = message.get("sender") or {}
                content = message.get("body") or {}
                if not isinstance(content, dict):
                    content = vars(content)
                if (
                    lookup.get("found") is not True
                    or message.get("message_id") != effect.provider_correlation_ref
                    or message.get("chat_id") != payload["chat_id"]
                    or sender.get("id") != a["bot_id"]
                    or sender.get("tenant_key") != a["tenant_key"]
                    or content.get("content")
                    != json.dumps({"text": payload["text"]}, ensure_ascii=False)
                ):
                    return ProviderEvidence(STILL_UNKNOWN)
                return ProviderEvidence(
                    CONFIRMED_COMPLETED,
                    effect.provider_correlation_ref,
                    effect.provider_correlation_ref,
                )

        return EffectReconciliationService(self.outbox).reconcile(
            effect_id=row["effect_id"], provider=ExactEvidence()
        )

    def cancel(self, context, case_id):
        a = self.binding(case_id)
        if not context.authenticated or (context.user_id, context.workspace_id) != (
            a["principal_id"],
            a["workspace_id"],
        ):
            raise PermissionError("owned cancellation required")
        self._authority(a)
        with self.store._connect() as c:
            c.execute(
                "UPDATE meeting_core_delivery_bindings SET revoked_at=%s WHERE negotiation_id=%s",
                (utc(), case_id),
            )
            c.execute(
                "UPDATE meeting_core_clarifications SET state='CANCELLED' WHERE negotiation_id=%s AND state!='APPLIED'",
                (case_id,),
            )
            self.store.cancel_core_clarification(
                c, negotiation_id=case_id, actor_id=context.user_id
            )
        self.responsibilities.terminate(
            responsibility_id=a["responsibility_id"], reason="clarification cancelled"
        )


def recover_configured_meeting_delivery(execution_root):
    """Existing task API startup seam; no migration, network, host or scheduler."""
    from feishu_meeting_coordinator.store import MeetingCoordinatorStore

    store = MeetingCoordinatorStore(Path(execution_root) / "state.db")
    with store._connect() as c:
        if (
            c.execute(
                "SELECT to_regclass(current_schema()||'.meeting_core_delivery_bindings')"
            ).fetchone()[0]
            is None
        ):
            return 0
    return MeetingClarificationDeliveryService(store).drain_pending()
