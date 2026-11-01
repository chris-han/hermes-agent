"""Explicit new-case reminder authority and continuation; transport only signals."""

import json
import os
import logging

from agents.timer_occurrence import TimerOccurrenceStore
from agents.durable_responsibility import (
    DurableResponsibilityDispatcher,
    DurableResponsibilityError,
    TIMER,
    EVENT,
    WAIT_EXTERNAL,
    DETERMINISTIC_CONTINUE,
    ContinuationDecision,
    WaitSpec,
)
from agents.effect_reconciliation import (
    ProviderEvidence,
    STILL_UNKNOWN,
    CONFIRMED_COMPLETED,
)
from contracts.meeting_delivery import canonical, digest, instant, utc
from services.meeting_clarification_delivery_service import (
    MeetingClarificationDeliveryService,
)
from services.meeting_clarification_feishu import FeishuClarificationTransport
from services.core_one_shot_timer_service import (
    CoreOneShotTimerProvider,
    CoreOneShotDueDriver,
    PREFIX,
)
from storage import hosted_postgres as pg

CONTRACT = "meeting-core-followup.v1"
PURPOSE = "followup_reminder"
log = logging.getLogger(__name__)


class MeetingFollowupTimerService:
    def __init__(self, store, *, transport_factory=None, lease_seconds=30):
        self.store = store
        self.delivery = MeetingClarificationDeliveryService(store)
        self.responsibilities = self.delivery.responsibilities
        self.timers = TimerOccurrenceStore(store.path)
        self.lease_seconds = lease_seconds
        endpoint = os.environ.get("SEMANTIER_MEETING_PROTOCOL_ENDPOINT")
        self.transport_factory = transport_factory or (
            lambda bot: FeishuClarificationTransport(bot, endpoint=endpoint)
        )
        self.provider = CoreOneShotTimerProvider(self.timers, self.validate_arm)
        self.driver = CoreOneShotDueDriver(self.timers, self.accept_timer)

    def _cut(self, stage):
        pass

    def row(self, timer_id):
        with self.store._connect() as c:
            r = c.execute(
                "SELECT * FROM semantier_timer_occurrences WHERE timer_id=%s",
                (timer_id,),
            ).fetchone()
        if not r or not r["core_binding_json"]:
            raise PermissionError("explicit core timer admission required")
        return dict(r)

    def binding(self, case_id):
        with self.store._connect() as c:
            r = c.execute(
                "SELECT * FROM semantier_timer_occurrences WHERE workflow_id=%s AND core_binding_json IS NOT NULL ORDER BY scheduled_at::timestamptz DESC,timer_id DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        if not r:
            raise PermissionError("core reminder case required")
        return self._binding(dict(r))

    def _binding(self, row):
        a = json.loads(row["core_binding_json"])
        if digest(a) != row["core_binding_hash"]:
            raise PermissionError("corrupt reminder binding")
        return a

    def _authority(self, a, *, pending=False):
        case, bot = self.delivery._authority(
            a, pending=False, contract=CONTRACT, purpose=PURPOSE
        )
        from feishu_meeting_coordinator.gateway import _negotiation_is_terminal

        if (
            _negotiation_is_terminal(case)
            or int(case["current_round"]) + 1 != a["round"]
        ):
            raise PermissionError("terminal/replaced reminder case")
        return case, bot

    def _eligible(self, a):
        from feishu_meeting_coordinator.gateway import _followup_reminder_needed

        self._authority(a)
        if int(self.store.get_workspace_state(a["workspace_id"])["max_followups"]) != 1:
            raise PermissionError("one-followup policy changed")
        p = next(
            p
            for p in self.store.list_negotiation_participants(a["negotiation_id"])
            if p["attendee_user_id"] == a["participant_id"]
        )
        if not _followup_reminder_needed(p) or int(p.get("followup_count") or 0) >= 1:
            raise PermissionError("reminder no longer eligible")

    def admit_new_case(
        self,
        context,
        *,
        monitor_id,
        event_revision_id,
        participant_id,
        scheduled_at,
        interval_minutes=2,
    ):
        due = instant(scheduled_at).isoformat()  # Reject before any owner mutation.
        if type(interval_minutes) is not int or interval_minutes < 0:
            raise ValueError("nonnegative existing reminder interval required")
        a = self.delivery._issue_new_case(
            context,
            monitor_id=monitor_id,
            event_revision_id=event_revision_id,
            participant_id=participant_id,
            contract=CONTRACT,
            purpose=PURPOSE,
        )
        a["interval_minutes"] = interval_minutes
        self._eligible(a)
        return self._register(a, due)

    def _register(self, a, due):
        claim = self.responsibilities.claim(
            responsibility_id=a["responsibility_id"],
            owner_id="core-timer-registration",
            lease_seconds=self.lease_seconds,
        )
        with self.store._connect() as c:
            pg.begin_write(c)
            self.responsibilities.assert_claim(claim, conn=c)
            timer = self.timers.schedule(
                c,
                workflow_type="meeting_negotiation",
                workflow_id=a["negotiation_id"],
                purpose=PURPOSE,
                semantic_round=a["round"],
                scheduled_at=due,
            )
            if timer.provider_ref or timer.state != "SCHEDULED":
                raise PermissionError("cannot adopt prior timer")
            c.execute(
                "UPDATE semantier_timer_occurrences SET core_binding_json=%s,core_binding_hash=%s,responsibility_id=%s,expected_checkpoint_version=%s,continuation_state='WAIT_PENDING' WHERE timer_id=%s AND core_binding_json IS NULL",
                (
                    canonical(a).decode(),
                    digest(a),
                    a["responsibility_id"],
                    claim.checkpoint_version + 1,
                    timer.timer_id,
                ),
            )
        self._cut("after_occurrence_commit")
        self.ensure_wait(timer.timer_id, claim=claim)
        return self.row(timer.timer_id)

    def ensure_wait(self, timer_id, *, claim=None):
        # Narrow lock only: responsibility calls acquire their own authority lock.
        with self.store._connect() as c:
            c.execute("BEGIN")
            c.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                ("core-timer-registration:" + timer_id,),
            )
            row = dict(
                c.execute(
                    "SELECT * FROM semantier_timer_occurrences WHERE timer_id=%s",
                    (timer_id,),
                ).fetchone()
            )
            if row["continuation_state"] != "WAIT_PENDING":
                return
            a = self._binding(row)
            self._eligible(a)
            expected = row["expected_checkpoint_version"]
            waits = [
                w
                for w in self.responsibilities.pending_waits(a["responsibility_id"])
                if w.kind == TIMER
                and w.source_ref == timer_id
                and w.expected_checkpoint_version == expected
            ]
            if not waits:
                current = self.responsibilities.get(a["responsibility_id"])
                if current.checkpoint_version != expected - 1:
                    c.execute(
                        "UPDATE semantier_timer_occurrences SET state='CANCELLED',continuation_state='CANCELLED',wake_outcome='registration_replaced' WHERE timer_id=%s",
                        (timer_id,),
                    )
                    return
                if claim is None:
                    try:
                        claim = self.responsibilities.claim(
                            responsibility_id=a["responsibility_id"],
                            owner_id="core-timer-recovery",
                            lease_seconds=self.lease_seconds,
                        )
                    except DurableResponsibilityError:
                        return
                waits = self.responsibilities.register_waits(
                    claim,
                    waits=(
                        WaitSpec(
                            kind=TIMER,
                            source_ref=timer_id,
                            correlation_key=a["negotiation_id"],
                            deadline_at=row["scheduled_at"],
                        ),
                        WaitSpec(
                            kind=EVENT,
                            source_ref="feishu:negotiation:"
                            + a["negotiation_id"]
                            + ":reply",
                            correlation_key=a["negotiation_id"],
                        ),
                    ),
                    lifecycle_state=WAIT_EXTERNAL,
                    continuation_disposition=DETERMINISTIC_CONTINUE,
                )
                waits = [w for w in waits if w.kind == TIMER]
            if waits[0].expected_checkpoint_version != expected:
                raise PermissionError("original wait checkpoint changed")
            self._cut("after_wait_commit")
            c.execute(
                "UPDATE semantier_timer_occurrences SET wait_id=%s,continuation_state='WAITING' WHERE timer_id=%s",
                (waits[0].wait_id, timer_id),
            )
        self.timers.arm(timer_id=timer_id, provider=self.provider)

    def validate_arm(self, timer_id, scheduled_at):
        row = self.row(timer_id)
        a = self._binding(row)
        self._eligible(a)
        if (
            row["workflow_type"] != "meeting_negotiation"
            or row["workflow_id"] != a["negotiation_id"]
            or row["purpose"] != PURPOSE
            or row["semantic_round"] != a["round"]
            or row["responsibility_id"] != a["responsibility_id"]
            or row["scheduled_at"] != scheduled_at
        ):
            raise PermissionError("timer ownership mismatch")
        waits = self.responsibilities.pending_waits(a["responsibility_id"])
        if not any(
            w.wait_id == row["wait_id"]
            and w.source_ref == timer_id
            and w.expected_checkpoint_version == row["expected_checkpoint_version"]
            and w.kind == TIMER
            for w in waits
        ):
            raise PermissionError("original timer wait missing")

    def accept_timer(self, timer_id, *, now=None):
        row = self.row(timer_id)
        self._binding(row)
        now = now or utc()
        if instant(now) < instant(row["scheduled_at"]):
            return "early"
        if row["state"] != "SCHEDULED" or row["provider_ref"] != PREFIX + timer_id:
            return "inactive"
        try:
            self.validate_arm(timer_id, row["scheduled_at"])
        except PermissionError:
            # A durable prior acceptance may have satisfied this exact wait.
            with self.store._connect() as c:
                prior = c.execute(
                    "SELECT * FROM semantier_responsibility_wakes WHERE dedupe_key=%s",
                    (row["dedupe_key"],),
                ).fetchone()
            if (
                not prior
                or not prior["accepted"]
                or prior["source_ref"] != timer_id
                or prior["correlation_key"] != row["workflow_id"]
                or prior["kind"] != TIMER
                or prior["responsibility_id"] != row["responsibility_id"]
                or prior["expected_checkpoint_version"]
                != row["expected_checkpoint_version"]
            ):
                self.invalidate(timer_id, reason="ineligible_or_replaced")
                return "rejected"
        result = self.responsibilities.accept_wake(
            responsibility_id=row["responsibility_id"],
            kind=TIMER,
            source_ref=timer_id,
            correlation_key=row["workflow_id"],
            dedupe_key=row["dedupe_key"],
            expected_checkpoint_version=row["expected_checkpoint_version"],
            observed_at=now,
        )
        self._cut("after_wake_commit")
        with self.store._connect() as c:
            prior = c.execute(
                "SELECT * FROM semantier_responsibility_wakes WHERE dedupe_key=%s",
                (row["dedupe_key"],),
            ).fetchone()
            accepted = bool(prior and prior["accepted"])
            c.execute(
                "UPDATE semantier_timer_occurrences SET state=%s,wake_id=%s,wake_outcome=%s,continuation_state=%s WHERE timer_id=%s AND state='SCHEDULED'",
                (
                    "ACCEPTED" if accepted else "CANCELLED",
                    result.wake_id,
                    prior["reason"] if prior else result.reason,
                    "WOKEN" if accepted else "CANCELLED",
                    timer_id,
                ),
            )
        self._cut("after_wake_bookkeeping")
        return result.reason

    def invalidate(self, timer_id, *, reason):
        with self.store._connect() as c:
            c.execute(
                "UPDATE semantier_timer_occurrences SET state=CASE WHEN state='SCHEDULED' THEN 'CANCELLED' ELSE state END,continuation_state='CANCELLED',wake_outcome=%s WHERE timer_id=%s AND core_binding_json IS NOT NULL AND continuation_state NOT IN ('DONE','CANCELLED')",
                (reason, timer_id),
            )

    def cancel(self, context, timer_id):
        row = self.row(timer_id)
        a = self._binding(row)
        from agents.core_file_scope import CoreFileScope

        CoreFileScope.issue(context, "core-timer-cancel")
        if (context.user_id, context.workspace_id) != (
            a["principal_id"],
            a["workspace_id"],
        ):
            raise PermissionError("owned timer required")
        self.invalidate(timer_id, reason="owner_cancelled")
        self.responsibilities.terminate(
            responsibility_id=a["responsibility_id"], reason="owner_cancelled"
        )

    def replace(self, context, timer_id, *, scheduled_at):
        due = instant(scheduled_at).isoformat()
        row = self.row(timer_id)
        a = self._binding(row)
        self._eligible(a)
        from agents.core_file_scope import CoreFileScope

        CoreFileScope.issue(context, "core-timer-replace")
        if (context.user_id, context.workspace_id) != (
            a["principal_id"],
            a["workspace_id"],
        ):
            raise PermissionError("owned timer required")
        result = self.responsibilities.accept_wake(
            responsibility_id=a["responsibility_id"],
            kind=EVENT,
            source_ref="feishu:negotiation:" + a["negotiation_id"] + ":reply",
            correlation_key=a["negotiation_id"],
            dedupe_key="replace:" + timer_id,
            expected_checkpoint_version=row["expected_checkpoint_version"],
        )
        if not result.accepted:
            raise PermissionError("timer replacement lost original wait")
        self.invalidate(timer_id, reason="replaced")
        return self._register(a, due)

    def accept_verified(self, event):
        from services.meeting_clarification_feishu import VerifiedIngress
        from contracts.meeting_delivery import slot

        if not isinstance(event, VerifiedIngress) or not event.authentic():
            raise PermissionError("verified SDK reply required")
        a = self.binding(event.case_id)
        self._authority(a)
        if (
            event.sender != a["recipient_id"]
            or event.app_id != a["app_id"]
            or event.tenant_key != a["tenant_key"]
            or event.generation != a["bot_generation"]
            or event.chat_type != "p2p"
        ):
            raise PermissionError("foreign reminder reply")
        body = json.loads(event.text)
        row = self.row(body.get("timer_id", ""))
        if (
            row["workflow_id"] != event.case_id
            or body.get("round") != a["round"]
            or row["continuation_state"] not in ("WAITING", "WOKEN")
        ):
            raise PermissionError("reply no longer addresses original wait")
        corrected = slot(body.get("slot"))
        with self.store._connect() as c:
            pg.begin_write(c)
            self.store.apply_core_clarification(
                c,
                negotiation_id=event.case_id,
                participant_id=a["participant_id"],
                round_number=a["round"],
                slot=corrected,
                event_id=event.message_id,
                input_hash=digest(body),
            )
            c.execute(
                "UPDATE semantier_timer_occurrences SET state=CASE WHEN state='SCHEDULED' THEN 'CANCELLED' ELSE state END,continuation_state='REPLY_PENDING',wake_outcome=%s WHERE timer_id=%s",
                ("reply:" + event.message_id, row["timer_id"]),
            )
        self._finish_reply(self.row(row["timer_id"]))
        return {"accepted": True, "timer_id": row["timer_id"]}

    def _finish_reply(self, row):
        a = self._binding(row)
        self.responsibilities.accept_wake(
            responsibility_id=a["responsibility_id"],
            kind=EVENT,
            source_ref="feishu:negotiation:" + a["negotiation_id"] + ":reply",
            correlation_key=a["negotiation_id"],
            dedupe_key=row["wake_outcome"],
            expected_checkpoint_version=row["expected_checkpoint_version"],
        )
        self.invalidate(row["timer_id"], reason=row["wake_outcome"])

    def recover(self, *, limit=32):
        with self.store._connect() as c:
            rows = c.execute(
                "SELECT timer_id,continuation_state,scheduled_at FROM semantier_timer_occurrences WHERE core_binding_json IS NOT NULL AND (continuation_state IN ('WAIT_PENDING','REPLY_PENDING') OR (continuation_state='WAITING' AND provider_ref IS NULL)) ORDER BY scheduled_at::timestamptz,timer_id LIMIT %s",
                (limit,),
            ).fetchall()
        for row in rows:
            try:
                if row["continuation_state"] == "REPLY_PENDING":
                    self._finish_reply(self.row(row["timer_id"]))
                elif row["continuation_state"] == "WAIT_PENDING":
                    self.ensure_wait(row["timer_id"])
                else:
                    self.timers.arm(timer_id=row["timer_id"], provider=self.provider)
            except PermissionError:
                self.invalidate(row["timer_id"], reason="recovery_ineligible")
        return len(rows)

    def dispatch(self, timer_id):
        row = self.row(timer_id)
        a = self._binding(row)
        if row["state"] != "ACCEPTED" or row["continuation_state"] not in (
            "WOKEN",
            "FINISHED",
        ):
            return None
        try:
            self._authority(a)
        except PermissionError:
            self.invalidate(timer_id, reason="continuation_authority_revoked")
            return None
        service = self

        def handler(record, checkpoint, claim):
            # Only this accepted timer/checkpoint may continue this responsibility.
            service._authority(a)
            if claim.checkpoint_version != row["expected_checkpoint_version"]:
                raise PermissionError("stale timer continuation")

            def guard(conn=None):
                service._authority(a)
                service.responsibilities.assert_claim(claim, conn=conn)
                if service.row(timer_id)["continuation_state"] == "CANCELLED":
                    raise PermissionError("cancelled timer")

            guard()
            if service.row(timer_id)["continuation_state"] != "FINISHED":
                _, bot = service._authority(a)
                transport = service.transport_factory(bot)

                class Client:
                    def send_attendee_message(self, **kw):
                        guard()
                        service._eligible(a)
                        service._cut("before_send")
                        guard()
                        service._eligible(a)
                        from feishu_meeting_coordinator.scripts.feishu_bot_api import (
                            send_attendee_message,
                        )

                        result = send_attendee_message(**kw, client=transport.client)
                        service._cut("after_send")
                        return result

                    def get_message(self, **kw):
                        return transport.get_message(kw["message_id"])

                class ExactEvidence:
                    def inspect(self, effect):
                        if not effect.provider_correlation_ref:
                            return ProviderEvidence(STILL_UNKNOWN)
                        try:
                            lookup = transport.get_message(
                                effect.provider_correlation_ref
                            )
                            if lookup.get("found") is not True:
                                return ProviderEvidence(STILL_UNKNOWN)
                            m = lookup.get("message") or {}
                        except Exception:
                            return ProviderEvidence(STILL_UNKNOWN)
                        payload = json.loads(effect.canonical_payload)
                        sender = m.get("sender") or {}
                        if (
                            m.get("message_id") == effect.provider_correlation_ref
                            and sender.get("id") == a["bot_id"]
                            and sender.get("tenant_key") == a["tenant_key"]
                            and payload["target_id"] == a["recipient_id"]
                            and (m.get("body") or {}).get("content")
                            == json.dumps(
                                {"text": payload["message"]}, ensure_ascii=False
                            )
                        ):
                            return ProviderEvidence(
                                CONFIRMED_COMPLETED,
                                effect.provider_correlation_ref,
                                effect.provider_correlation_ref,
                            )
                        return ProviderEvidence(STILL_UNKNOWN)

                from feishu_meeting_coordinator.gateway import run_core_followup_once

                service._cut("before_continuation")
                run_core_followup_once(
                    store=service.store,
                    negotiation_id=a["negotiation_id"],
                    timer_id=timer_id,
                    participant_id=a["participant_id"],
                    client=Client(),
                    claim_guard=guard,
                    transaction_guard=lambda c: guard(conn=c),
                    evidence_provider=ExactEvidence(),
                    interval_minutes=a["interval_minutes"],
                )
                service._cut("after_business")
                with service.store._connect() as c:
                    c.execute(
                        "UPDATE semantier_timer_occurrences SET continuation_state='FINISHED' WHERE timer_id=%s AND continuation_state='WOKEN'",
                        (timer_id,),
                    )
            return ContinuationDecision(
                disposition=WAIT_EXTERNAL,
                waits=(
                    WaitSpec(
                        kind=EVENT,
                        source_ref="feishu:negotiation:"
                        + a["negotiation_id"]
                        + ":reply",
                        correlation_key=a["negotiation_id"],
                    ),
                ),
            )

        current = self.responsibilities.get(a["responsibility_id"])
        if current.checkpoint_version != row["expected_checkpoint_version"]:
            if (
                row["continuation_state"] == "FINISHED"
                and current.checkpoint_version == row["expected_checkpoint_version"] + 1
            ):
                with self.store._connect() as c:
                    c.execute(
                        "UPDATE semantier_timer_occurrences SET continuation_state='DONE' WHERE timer_id=%s",
                        (timer_id,),
                    )
            else:
                self.invalidate(timer_id, reason="stale_continuation")
            return None
        dispatcher = DurableResponsibilityDispatcher(
            self.responsibilities,
            handlers={"feishu_meeting_negotiation": handler},
            owner_id="core-reminder-continuation",
            lease_seconds=self.lease_seconds,
        )
        try:
            result = dispatcher.run_one(responsibility_id=a["responsibility_id"])
        except DurableResponsibilityError:
            return None
        if result.error:
            log.warning(
                "Core reminder continuation %s deferred: %s", timer_id, result.error
            )
        if not result.error:
            with self.store._connect() as c:
                c.execute(
                    "UPDATE semantier_timer_occurrences SET continuation_state='DONE' WHERE timer_id=%s AND continuation_state='FINISHED'",
                    (timer_id,),
                )
        return result

    def dispatch_pending(self, *, limit=32):
        with self.store._connect() as c:
            rows = c.execute(
                "SELECT timer_id FROM semantier_timer_occurrences WHERE state='ACCEPTED' AND continuation_state IN ('WOKEN','FINISHED') ORDER BY scheduled_at::timestamptz,timer_id LIMIT %s",
                (limit,),
            ).fetchall()
        for row in rows:
            self.dispatch(row["timer_id"])
        return len(rows)


def configured_service(execution_root):
    from feishu_meeting_coordinator.store import MeetingCoordinatorStore

    store = MeetingCoordinatorStore(execution_root / "state.db")
    with store._connect() as c:
        found = c.execute(
            "SELECT 1 FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='semantier_timer_occurrences' AND column_name='core_binding_json'"
        ).fetchone()
    return MeetingFollowupTimerService(store) if found else None


def selected_core_timer(store, case_id):
    if not pg.enabled():
        return False
    with store._connect() as c:
        column = c.execute(
            "SELECT 1 FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='semantier_timer_occurrences' AND column_name='core_binding_json'"
        ).fetchone()
        return bool(
            column
            and c.execute(
                "SELECT 1 FROM semantier_timer_occurrences WHERE workflow_id=%s AND core_binding_json IS NOT NULL LIMIT 1",
                (case_id,),
            ).fetchone()
        )
