"""General meeting handoffs over the existing domain, responsibility, inbox and outbox."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from agents.durable_event_arbitration import WorkflowVersion
from agents.durable_responsibility import (
    EVENT, WAIT_EXTERNAL, TERMINAL, DETERMINISTIC_CONTINUE, WaitSpec,
)
from agents.effect_outbox import READY, COMPLETED, DISPATCHING
from contracts.meeting_delivery import canonical, digest, instant, slot, utc
from services.meeting_clarification_delivery_service import MeetingClarificationDeliveryService
from storage import hosted_postgres as pg

CONTRACT = 'meeting-core-negotiation.v1'
PURPOSE = 'meeting_negotiation'


class CoreMeetingContinuationService(MeetingClarificationDeliveryService):
    """Business continuation only. No native worker, model or ambient transport."""

    def _authority(self, a, *, pending=False, **kwargs):
        return super()._authority(a, pending=pending, contract=CONTRACT, purpose=PURPOSE)

    @contextmanager
    def _locked(self, case_id):
        conn = self.store._connect()
        try:
            conn.execute('SELECT pg_advisory_lock(hashtextextended(%s,0))',
                         ('core-meeting-continuation:' + case_id,))
            yield
        finally:
            conn.close()

    def bind(self, context, case_id, marker):
        from services.core_meeting_board_service import CoreMeetingBoardService
        board = CoreMeetingBoardService(self.store)
        board.get(context, case_id)
        case = self.store.get_negotiation(case_id)
        with self._locked(case_id):
            with self.store._connect() as conn:
                old = conn.execute('SELECT association_json FROM meeting_core_delivery_bindings WHERE negotiation_id=%s',
                                   (case_id,)).fetchone()
            if old:
                a = self.binding(case_id)
                self._authority(a)
                return a
            a = self._issue_new_case(context, monitor_id=case['monitor_id'],
                event_revision_id=case['event_revision_id'],
                participant_id=case['declined_attendee_user_id'], contract=CONTRACT,
                purpose=PURPOSE, allow_reserved_recovery=True, admission_request_hash=marker)
            with self.store._connect() as conn:
                conn.execute('INSERT INTO meeting_core_delivery_bindings VALUES(%s,%s,%s,0,NULL)',
                             (case_id, canonical(a).decode(), digest(a)))
        return a

    def effects(self, case_id):
        with self.store._connect() as conn:
            rows = conn.execute("""SELECT effect_id FROM semantier_effect_outbox
                WHERE workflow_id=%s AND effect_type LIKE %s
                ORDER BY semantic_round,effect_id""", (case_id,'meeting-negotiation:%')).fetchall()
        return [self.outbox.get(row['effect_id']) for row in rows]

    def _intents(self, case, a):
        from feishu_meeting_coordinator import gateway
        participants = self.store.list_negotiation_participants(case['negotiation_id'])
        locale = gateway._default_locale_for_negotiation(case)
        if case['status'] == 'pending_decliner_input':
            p = next(p for p in participants if p['attendee_user_id'] == a['participant_id'])
            yield p, None, gateway._render_ask_decliner_message(case, p, locale)
        elif case['status'] == 'collecting_votes':
            # The existing domain planner selects material information requests.
            frontier = gateway._refresh_decision_frontier(case['negotiation_id'], self.store)
            eligible = {(v.target_ref, v.slot_id) for v in frontier.information_actions
                        if v.action == 'ASK' and v.expected_frontier_impact == 'MATERIAL'}
            for participant in participants:
                if (participant['attendee_user_id'], 'original') in eligible:
                    original = {'slot_id':'original', 'start_time':case['original_start_time'],
                                'end_time':case['original_end_time']}
                    message = (Path(__file__).parents[1] / 'prompts/meetings/original_meeting_availability.txt').read_text().format(
                        start=case['original_start_time'], end=case['original_end_time'],
                        timezone=case.get('timezone') or 'UTC')
                    yield participant, original, message.strip()
            for candidate in self.store.list_candidate_slots(case['negotiation_id']):
                for participant in participants:
                    if (participant['attendee_user_id'], candidate['slot_id']) in eligible:
                        yield participant, candidate, gateway._render_ask_vote_message(case, participant, candidate, locale)

    def run(self, case_id):
        from feishu_meeting_coordinator.gateway import _negotiation_is_terminal
        with self._locked(case_id):
            a = self.binding(case_id)
            case, _ = self._authority(a)
            record = self.responsibilities.get(a['responsibility_id'])
            if record.lifecycle_state in (TERMINAL, WAIT_EXTERNAL):
                return record
            claim = self.responsibilities.claim(responsibility_id=record.responsibility_id,
                                                owner_id='core-meeting-continuation')
            if _negotiation_is_terminal(case):
                return self.responsibilities.commit(claim, continuation_disposition=TERMINAL,
                    lifecycle_state=TERMINAL, require_unexpired=True)
            version = self._version(case_id)
            intents = list(self._intents(case, a))
            frontier = None
            if not intents and case['status'] == 'collecting_votes':
                from feishu_meeting_coordinator.decision_planning import form_decision_frontier
                frontier = form_decision_frontier(self.store.decision_planning_basis(case_id))
            waits = []
            with self.store._connect() as conn:
                pg.begin_write(conn)
                self.responsibilities.assert_claim(claim, conn=conn)
                for participant, candidate, message in intents:
                    target = participant.get('message_user_id') or participant['attendee_user_id']
                    identity = dict(case=case_id, participant=participant['attendee_user_id'],
                        round=int(case['current_round'])+1, slot_id=candidate['slot_id'] if candidate else None,
                        state=case['status'])
                    correlation = digest(identity)
                    example = dict(correlation=correlation, round=identity['round'])
                    if candidate and candidate['slot_id'] == 'original':
                        example.update(availability='available', slot_id='original')
                    elif candidate:
                        example.update(vote='yes', slot_id=candidate['slot_id'])
                    else:
                        example['slot'] = dict(start_time='ISO-8601', end_time='ISO-8601', timezone=case['timezone'])
                    instruction = (Path(__file__).parents[1] / 'prompts/meetings/structured_negotiation_reply.txt').read_text().format(example=json.dumps(example))
                    payload = dict(**identity, recipient_id=target, participant_id=participant['attendee_user_id'],
                        text=message+'\n\n'+instruction.strip(), correlation=correlation,
                        association_hash=digest(a), contract=CONTRACT)
                    effect = self.outbox.enqueue_ready(conn, workflow_id=case_id,
                        effect_type='meeting-negotiation:'+correlation, semantic_round=identity['round'],
                        canonical_payload=canonical(payload))
                    waits.append(WaitSpec(kind=EVENT, source_ref=effect.effect_id, correlation_key=correlation))
                if not waits:
                    if frontier and frontier.decision_frontier and frontier.sufficient_progress_disposition == 'PRESENT_FRONTIER':
                        self.store.present_core_decision_frontier(conn, negotiation_id=case_id,
                                                                 expected_input_hash=frontier.input_hash)
                    # Requester decisions and unresolved domain facts remain explicit waits.
                    waits.append(WaitSpec(kind=EVENT, source_ref='meeting-decision:'+case_id,
                                          correlation_key=case_id))
            self._cut('after_intents_commit')
            self.responsibilities.register_waits(claim, waits=tuple(waits),
                lifecycle_state=WAIT_EXTERNAL, continuation_disposition=DETERMINISTIC_CONTINUE)
            return self.responsibilities.get(a['responsibility_id'])

    def _version(self, case_id):
        with self.store._connect() as conn:
            return conn.execute('SELECT protocol_version FROM meeting_core_delivery_bindings WHERE negotiation_id=%s',
                                (case_id,)).fetchone()['protocol_version']

    def dispatch(self, case_id, transport):
        results = []
        with self._locked(case_id):
            a = self.binding(case_id)
            case, _ = self._authority(a)
            if transport.generation != a['bot_generation']:
                raise PermissionError('transport bot generation mismatch')
            record = self.responsibilities.get(a['responsibility_id'])
            if record.lifecycle_state != WAIT_EXTERNAL:
                raise PermissionError('current meeting wait required')
            waits = {w.source_ref: w for w in self.responsibilities.pending_waits(record.responsibility_id)}
            for effect in self.effects(case_id):
                if effect.effect_id not in waits or effect.state != READY:
                    continue
                payload = json.loads(effect.canonical_payload)
                if payload['state'] != case['status']:
                    continue
                wait = waits[effect.effect_id]
                participants = self.store.list_negotiation_participants(case_id)
                participant = next((p for p in participants if p['attendee_user_id'] == payload['participant_id']), None)
                if (payload['association_hash'] != digest(a) or payload['contract'] != CONTRACT
                    or wait.expected_checkpoint_version != record.checkpoint_version
                    or wait.correlation_key != payload['correlation'] or participant is None
                    or (participant.get('message_user_id') or participant['attendee_user_id']) != payload['recipient_id']):
                    raise PermissionError('meeting effect authority or wait changed')
                attempt = uuid4().hex
                with self.store._connect() as conn:
                    pg.begin_write(conn)
                    if not self.outbox.claim_dispatch_in_transaction(conn, effect_id=effect.effect_id, attempt_id=attempt):
                        continue
                try:
                    self._cut('before_send')
                    self._authority(self.binding(case_id))
                    result = transport.send(payload, effect.idempotency_key)
                    mid = result.get('message_ids', {}).get(payload['recipient_id'])
                    if not mid or result.get('failed') or result.get('delivered') != [payload['recipient_id']]:
                        raise RuntimeError('uncertain SDK send')
                    self.outbox.pin_provider_correlation(effect_id=effect.effect_id, correlation_ref=mid)
                    self._cut('after_send')
                    self.outbox.mark_completed(effect_id=effect.effect_id, attempt_id=attempt, receipt_ref=mid)
                except Exception as exc:
                    self.outbox.mark_effect_unknown(effect_id=effect.effect_id, attempt_id=attempt,
                                                   failure_detail=type(exc).__name__)
                results.append(self.outbox.get(effect.effect_id))
        return results

    def dispatch_pending(self, *, limit=32, transport_factory=None):
        """Dispatch bounded ready intents from the existing application poll."""
        if not 1 <= limit <= 100:
            raise ValueError('bounded delivery batch required')
        if transport_factory is None:
            import os
            from services.meeting_clarification_feishu import FeishuClarificationTransport
            endpoint = os.environ.get('SEMANTIER_MEETING_PROTOCOL_ENDPOINT')
            transport_factory = lambda bot: FeishuClarificationTransport(bot, endpoint=endpoint)
        with self.store._connect() as conn:
            rows = conn.execute("""SELECT DISTINCT e.workflow_id FROM semantier_effect_outbox e
                JOIN meeting_core_delivery_bindings b ON b.negotiation_id=e.workflow_id
                WHERE e.state='READY' AND e.effect_type LIKE %s
                  AND b.revoked_at IS NULL AND b.association_json::jsonb->>'contract'=%s
                ORDER BY e.workflow_id LIMIT %s""", ('meeting-negotiation:%', CONTRACT, limit)).fetchall()
        for row in rows:
            try:
                a = self.binding(row['workflow_id'])
                _, bot = self._authority(a)
                self.dispatch(row['workflow_id'], transport_factory(bot))
            except PermissionError:
                continue
        return len(rows)

    def recover_pending(self, *, limit=32):
        """Recover interrupted attempts without redispatching uncertain effects."""
        if not 1 <= limit <= 100:
            raise ValueError('bounded recovery batch required')
        with self.store._connect() as conn:
            rows = conn.execute("""SELECT DISTINCT e.workflow_id FROM semantier_effect_outbox e
                JOIN meeting_core_delivery_bindings b ON b.negotiation_id=e.workflow_id
                WHERE e.state='DISPATCHING' AND e.effect_type LIKE %s
                  AND b.revoked_at IS NULL AND b.association_json::jsonb->>'contract'=%s
                UNION
                SELECT n.negotiation_id AS workflow_id FROM meeting_time_negotiations n
                JOIN meeting_core_delivery_bindings b ON b.negotiation_id=n.negotiation_id
                JOIN semantier_responsibilities r ON r.responsibility_id=b.association_json::jsonb->>'responsibility_id'
                WHERE r.lifecycle_state IN ('WAIT_EXTERNAL','READY')
                  AND b.revoked_at IS NULL AND b.association_json::jsonb->>'contract'=%s
                  AND EXISTS (SELECT 1 FROM meeting_time_negotiation_events ev
                    WHERE ev.negotiation_id=n.negotiation_id
                      AND ev.event_type IN ('CORE_REPLY_APPLIED','REQUESTER_DECISION_APPLIED')
                      AND ev.payload_json::jsonb ? 'wait_version'
                      AND (r.lifecycle_state='READY' OR
                           (ev.payload_json::jsonb->>'wait_version')::bigint=r.checkpoint_version))
                ORDER BY workflow_id LIMIT %s""", ('meeting-negotiation:%', CONTRACT, CONTRACT, limit)).fetchall()
        for row in rows:
            try:
                self.recover(row['workflow_id'])
            except PermissionError:
                continue
        return len(rows)

    def accept_verified(self, event):
        from services.meeting_clarification_feishu import VerifiedIngress
        if not isinstance(event, VerifiedIngress) or not event.authentic():
            raise PermissionError('verified SDK ingress required')
        with self._locked(event.case_id):
            a = self.binding(event.case_id)
            case, _ = self._authority(a)
            if (event.app_id != a['app_id'] or event.tenant_key != a['tenant_key']
                or event.generation != a['bot_generation'] or event.chat_type != 'p2p'):
                raise PermissionError('foreign meeting reply')
            try:
                body = json.loads(event.text)
            except ValueError:
                raise ValueError('unstructured reply remains unresolved') from None
            effect = next((e for e in self.effects(event.case_id)
                if json.loads(e.canonical_payload)['correlation'] == body.get('correlation')), None) if isinstance(body,dict) else None
            if not effect or effect.state != COMPLETED:
                raise PermissionError('exact delivered meeting prompt required')
            payload = json.loads(effect.canonical_payload)
            if (event.sender != payload['recipient_id'] or event.parent_id != effect.receipt_ref
                or body.get('round') != payload['round']):
                raise PermissionError('reply correlation mismatch')
            evidence = next((json.loads(e['payload_json']) for e in self.store.list_negotiation_events(event.case_id)
                if e['event_type']=='CORE_REPLY_APPLIED' and json.loads(e['payload_json']).get('event_id')==event.message_id), None)
            input_hash = digest(self._input(event))
            if evidence:
                if evidence['input_hash'] != input_hash:
                    raise PermissionError('duplicate response payload changed')
                return dict(accepted=True, duplicate=True)
            if payload['state'] != case['status']:
                raise PermissionError('stale meeting reply')
            waits = [w for w in self.responsibilities.pending_waits(a['responsibility_id'])
                     if w.source_ref == effect.effect_id and w.correlation_key == payload['correlation']]
            if len(waits) != 1:
                raise PermissionError('exact pending meeting wait required')
            wait = waits[0]
            if payload['slot_id'] is None:
                if set(body) != {'correlation','round','slot'}:
                    raise ValueError('structured slot required')
                response = dict(slot=slot(body['slot']))
            elif payload['slot_id'] == 'original':
                if (set(body) != {'correlation','round','availability','slot_id'}
                    or body['slot_id'] != 'original' or body['availability'] not in ('available','unavailable')):
                    raise ValueError('exact original availability required')
                response = dict(slot_id='original', availability=body['availability'])
            else:
                if set(body) != {'correlation','round','vote','slot_id'} or body['vote'] not in ('yes','no') or body['slot_id'] != payload['slot_id']:
                    raise ValueError('exact structured vote required')
                response = dict(slot_id=payload['slot_id'], vote=body['vote'])
            with self.store._connect() as conn:
                pg.begin_write(conn)
                def load(c):
                    return WorkflowVersion(c.execute('SELECT protocol_version FROM meeting_core_delivery_bindings WHERE negotiation_id=%s',
                        (event.case_id,)).fetchone()['protocol_version'])
                def transition(c, current):
                    self.store.apply_core_meeting_reply(c, negotiation_id=event.case_id,
                        participant_id=payload['participant_id'], response=response,
                        event_id=event.message_id, expected_state=payload['state'])
                    self.store._record_negotiation_event(c, negotiation_id=event.case_id,
                        event_type='CORE_REPLY_APPLIED', actor_type='participant', actor_id=payload['participant_id'],
                        payload=dict(event_id=event.message_id,input_hash=input_hash,effect_id=effect.effect_id,
                            correlation=payload['correlation'],wait_version=wait.expected_checkpoint_version))
                    c.execute('UPDATE meeting_core_delivery_bindings SET protocol_version=protocol_version+1 WHERE negotiation_id=%s',(event.case_id,))
                    return WorkflowVersion(current.version+1), None
                self.inbox.accept(workflow_type=CONTRACT,workflow_id=event.case_id,event_id=event.message_id,
                    event_type='STRUCTURED_REPLY',dedupe_key=event.message_id,expected_state_version=self._version(event.case_id),
                    load_state=load,transition=transition,connection=conn)
            self._cut('after_reply_commit')
            self._wake_general(a, dict(event_id=event.message_id,effect_id=effect.effect_id,
                correlation=payload['correlation'],wait_version=wait.expected_checkpoint_version))
        self.run(event.case_id)
        return dict(accepted=True, duplicate=False)

    def _wake_general(self, a, reply):
        return self.responsibilities.accept_wake(responsibility_id=a['responsibility_id'],kind=EVENT,
            source_ref=reply['effect_id'],correlation_key=reply['correlation'],
            dedupe_key='meeting-reply:'+reply['event_id'],expected_checkpoint_version=reply['wait_version'])

    def requester_decision(self, context, case_id, action, *, slot_id=None):
        """Record the requester's domain decision before waking its exact wait."""
        from services.core_meeting_board_service import CoreMeetingBoardService
        CoreMeetingBoardService(self.store).get(context, case_id)
        if action == 'requester_cancel':
            self.cancel(context, case_id)
            return self.store.get_negotiation(case_id)
        if action not in ('requester_keep_original', 'requester_select_slot'):
            raise ValueError('invalid requester action')
        with self._locked(case_id):
            a = self.binding(case_id)
            self._authority(a)
            with self.store._connect() as conn:
                pg.begin_write(conn)
                case = conn.execute('SELECT * FROM meeting_time_negotiations WHERE negotiation_id=%s FOR UPDATE', (case_id,)).fetchone()
                record = conn.execute('SELECT * FROM semantier_responsibilities WHERE responsibility_id=%s FOR UPDATE', (a['responsibility_id'],)).fetchone()
                if not case or case['status'] != 'awaiting_requester_decision' or record['lifecycle_state'] != WAIT_EXTERNAL:
                    raise ValueError('meeting decision state changed')
                wait = conn.execute("""SELECT * FROM semantier_responsibility_waits WHERE responsibility_id=%s
                    AND kind=%s AND source_ref=%s AND correlation_key=%s AND state='PENDING'
                    AND expected_checkpoint_version=%s FOR UPDATE""", (a['responsibility_id'], EVENT,
                    'meeting-decision:'+case_id, case_id, record['checkpoint_version'])).fetchone()
                if not wait:
                    raise ValueError('current meeting decision wait required')
                if action == 'requester_keep_original':
                    if slot_id:
                        raise ValueError('slot not applicable')
                    selected = dict(decision='keep_original_time', start_time=case['original_start_time'],
                                    end_time=case['original_end_time'], timezone=case['timezone'], calendar_update=False)
                else:
                    selected = conn.execute('SELECT * FROM meeting_time_candidate_slots WHERE negotiation_id=%s AND slot_id=%s', (case_id, slot_id)).fetchone()
                    if not selected:
                        raise ValueError('meeting candidate slot required')
                    selected = {k:selected[k] for k in ('slot_id','start_time','end_time','timezone')}
                    selected['calendar_update'] = True
                now = utc()
                conn.execute("""UPDATE meeting_time_negotiations SET status='requester_decided',selected_slot_json=%s,
                    terminal_authority=%s,terminal_at=%s,terminal_reason=%s,terminal_event_revision_id=event_revision_id,
                    completed_at=%s,updated_at=%s WHERE negotiation_id=%s""",
                    (json.dumps(selected,sort_keys=True), 'user:'+context.user_id, now, action, now, now, case_id))
                self.store._record_negotiation_event(conn, negotiation_id=case_id, event_type='REQUESTER_DECISION_APPLIED',
                    actor_type='requester', actor_id=context.user_id, prior_state='awaiting_requester_decision',
                    next_state='requester_decided', payload={'action':action, 'slot_id':slot_id,
                        'wait_version':record['checkpoint_version']})
            self._cut('after_decision_commit')
            self._wake_decision(a, record['checkpoint_version'])
        self.run(case_id)
        return self.store.get_negotiation(case_id)

    def _wake_decision(self, a, version):
        return self.responsibilities.accept_wake(responsibility_id=a['responsibility_id'],kind=EVENT,
            source_ref='meeting-decision:'+a['negotiation_id'],correlation_key=a['negotiation_id'],
            dedupe_key='meeting-decision:'+a['negotiation_id']+':'+str(version),expected_checkpoint_version=version)

    def recover(self, case_id):
        with self._locked(case_id):
            a = self.binding(case_id)
            self._authority(a)
            # An interrupted dispatch is uncertain and is never sent again here.
            for effect in self.effects(case_id):
                if effect.state == DISPATCHING:
                    self.outbox.mark_effect_unknown(effect_id=effect.effect_id,
                        attempt_id=effect.dispatch_attempt_id,failure_detail='interrupted meeting dispatch')
            for event in self.store.list_negotiation_events(case_id):
                if event['event_type']=='CORE_REPLY_APPLIED':
                    self._wake_general(a,json.loads(event['payload_json']))
                elif event['event_type']=='REQUESTER_DECISION_APPLIED':
                    payload = json.loads(event['payload_json'])
                    if 'wait_version' in payload:
                        self._wake_decision(a,payload['wait_version'])
        return self.run(case_id)

    def cancel(self, context, case_id):
        from services.core_meeting_board_service import CoreMeetingBoardService
        CoreMeetingBoardService(self.store).get(context,case_id)
        with self._locked(case_id):
            a = self.binding(case_id)
            self._authority(a)
            with self.store._connect() as conn:
                pg.begin_write(conn)
                conn.execute('UPDATE meeting_core_delivery_bindings SET revoked_at=%s WHERE negotiation_id=%s',(utc(),case_id))
                self.store.cancel_core_meeting(conn,negotiation_id=case_id,actor_id=context.user_id)
            self._cut('after_cancel_commit')
            self.responsibilities.terminate(responsibility_id=a['responsibility_id'],reason='meeting cancelled')
