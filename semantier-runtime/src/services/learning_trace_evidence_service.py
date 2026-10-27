"""Owner-scoped observation joins; no inference, financial writes or new store."""
from datetime import datetime, timezone
import json
from contracts.learning_trace_evidence import LearningTraceObservationV1
from eos.artifact_hashing import content_hash
from runtime_paths import workspace_hermes_home_path, workspace_root_path


class LearningTraceEvidenceService:
    MAX_PAYLOAD_BYTES = 1024 * 1024
    MAX_SNAPSHOT_OBSERVATIONS = 1000

    def __init__(self, *, store, billing=None):
        self.store = store
        self.billing = billing

    @staticmethod
    def scope(context):
        if (not context.authenticated or not context.user_id or not context.organization_id
                or context.membership_status != 'active'
                or context.hermes_home.resolve() != workspace_hermes_home_path(context.workspace_id).resolve()
                or context.workspace_root.resolve() != workspace_root_path(context.workspace_id).resolve()):
            raise PermissionError('FORBIDDEN_TRACE_SCOPE')
        return dict(organization_id=context.organization_id, workspace_id=context.workspace_id)

    def observations(self, *, context, session_id):
        scope = self.scope(context)
        if not session_id.startswith(context.workspace_id + ':'):
            raise PermissionError('FORBIDDEN_TRACE_SESSION')
        pins = self.store.list_learning_trace_pins(**scope, owner_ref=context.user_id, session_id=session_id, limit=self.MAX_SNAPSHOT_OBSERVATIONS + 1)
        if len(pins) > self.MAX_SNAPSHOT_OBSERVATIONS:
            raise ValueError('TRACE_SCOPE_LIMIT_EXCEEDED')
        rows = [self.store.get(**scope, artifact_ref=p.artifact_ref,
                               contract_type=LearningTraceObservationV1) for p in pins]
        return [r for r in rows if r.owner_ref == context.user_id and r.session_id == session_id]

    @staticmethod
    def _safe_payload(value):
        if isinstance(value, dict):
            if any(str(k).lower() in {'headers', 'authorization', 'api_key', 'password',
                                      'access_token', 'secret'} for k in value):
                raise ValueError('SECRET_BEARING_TRACE_PAYLOAD')
            for v in value.values():
                LearningTraceEvidenceService._safe_payload(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                LearningTraceEvidenceService._safe_payload(v)

    def record(self, *, context, source_store_id, source_epoch, profile_id, backend,
               session_id, collection_method, source_locator, event_kind, phase,
               native_occurrence_id, payload, source_occurred_at, captured_at):
        scope = self.scope(context)
        prior = self.observations(context=context, session_id=session_id)
        self._safe_payload(payload)
        if len(json.dumps(payload, ensure_ascii=False).encode()) > self.MAX_PAYLOAD_BYTES:
            raise ValueError('TRACE_PAYLOAD_LIMIT_EXCEEDED')
        if isinstance(source_occurred_at, str):
            source_occurred_at = datetime.fromisoformat(source_occurred_at.replace('Z', '+00:00'))
        key = None
        if source_epoch and native_occurrence_id:
            key = 'sha256:' + content_hash([context.organization_id, context.workspace_id,
                context.user_id, backend, source_store_id, source_epoch, profile_id,
                session_id, event_kind, phase, native_occurrence_id])
        values = dict(**scope, owner_ref=context.user_id, backend=backend,
            source_store_id=source_store_id, source_epoch=source_epoch, profile_id=profile_id,
            session_id=session_id, collection_method=collection_method,
            source_locator=source_locator, event_kind=event_kind, phase=phase,
            native_occurrence_id=native_occurrence_id, occurrence_id=key,
            payload=payload, source_occurred_at=source_occurred_at)
        candidate = LearningTraceObservationV1.create(**values, captured_at=captured_at)
        # A repeated read of the same exact source version is not a new observation.
        for row in prior:
            if row.model_dump(exclude={'artifact_ref', 'artifact_hash', 'schema_version',
                    'adapter_revision', 'captured_at'}) == values:
                return row
        row = candidate
        self.store.put(row)
        return row

    def routing_feedback(self, *, context, session_id):
        from contracts.learning_trace_evidence import RoutingFeedbackObservationV1
        scope = self.scope(context)
        if not session_id.startswith(context.workspace_id + ':'):
            raise PermissionError('FORBIDDEN_TRACE_SESSION')
        pins = self.store.list_routing_feedback_pins(**scope, owner_ref=context.user_id,
            session_id=session_id, limit=self.MAX_SNAPSHOT_OBSERVATIONS + 1)
        if len(pins) > self.MAX_SNAPSHOT_OBSERVATIONS:
            raise ValueError('TRACE_SCOPE_LIMIT_EXCEEDED')
        return [self.store.get(**scope, artifact_ref=p.artifact_ref,
                contract_type=RoutingFeedbackObservationV1) for p in pins]

    async def record_routing_feedback(self, *, context, **values):
        """Internal producer seam; HTTP clients cannot claim identity or reference quality."""
        from contracts.learning_trace_evidence import RoutingFeedbackObservationV1
        scope = self.scope(context)
        self._safe_payload(values)
        if len(json.dumps(values, default=str).encode()) > self.MAX_PAYLOAD_BYTES:
            raise ValueError('TRACE_PAYLOAD_LIMIT_EXCEEDED')
        if self.billing is None:
            raise ValueError('QUALITY_ATTACHMENT_UNAVAILABLE_NO_BILLING_OWNER')
        bound = await self.billing.feedback_subject(context, values['invocation_id'], values['attempt_id'])
        for key in ('account_id', 'root_task_id', 'session_id'):
            if values[key] != bound[key]:
                raise PermissionError('FEEDBACK_SUBJECT_MISMATCH')
        prior = self.routing_feedback(context=context, session_id=values['session_id'])
        if values.get('supersedes_ref'):
            predecessor = next((r for r in prior if r.artifact_ref == values['supersedes_ref']), None)
            subject = ('account_id', 'root_task_id', 'invocation_id', 'attempt_id', 'dimension')
            if predecessor is None or any(getattr(predecessor, key) != values[key] for key in subject):
                raise ValueError('FEEDBACK_PREDECESSOR_OUTSIDE_SUBJECT')
        row = RoutingFeedbackObservationV1.create(**scope, owner_ref=context.user_id, **values)
        for old in prior:
            if old.artifact_ref == row.artifact_ref:
                return old
        if len(prior) >= self.MAX_SNAPSHOT_OBSERVATIONS:
            raise ValueError('TRACE_SCOPE_LIMIT_EXCEEDED')
        self.store.put(row)
        return row

    def snapshot(self, *, context, session_id):
        rows = self.observations(context=context, session_id=session_id)
        rows.sort(key=lambda r: r.artifact_ref)
        groups = {}
        for row in rows:
            if row.occurrence_id:
                groups.setdefault(row.occurrence_id, []).append(row)
        source_groups = {}
        for row in rows:
            key = (row.backend, row.source_store_id, row.source_epoch, row.profile_id, row.collection_method, row.source_locator)
            source_groups.setdefault(key, []).append(row)
        conflicts = ['sha256:' + content_hash(key) for key, group in source_groups.items()
                     if len({content_hash(r.payload) for r in group}) > 1]
        pins = [r.pin().model_dump(mode='json') for r in rows]
        return dict(schema_version='learning_trace_snapshot.v1',
            snapshot_hash='sha256:' + content_hash(pins), observation_count=len(rows),
            occurrence_count=len(groups), unresolved_observations=sum(r.occurrence_id is None for r in rows),
            conflicts=conflicts, collection_gaps=[r.payload['collection_report'] for r in rows if r.payload.get('collection_report', {}).get('dropped') or r.payload.get('collection_report', {}).get('errors')], observations=[r.model_dump(mode='json') for r in rows],
            coverage='partial' if conflicts or any(r.occurrence_id is None or r.source_occurred_at is None for r in rows) else 'complete_for_retained_observations',
            history_coverage='unavailable_without_explicit_collection',
            supplier_cost=None, causal_attribution='unknown', quality='unmeasured')


    def result_views(self, *, context, session_id, tool_call_ids, source_binding=None, native_rows=()):
        """Four read projections, one snapshot; only owner-proven calls are linked."""
        snapshot = self.snapshot(context=context, session_id=session_id)
        binding = source_binding or {}
        # Unknown incarnation cannot qualify an occurrence join. An exact native
        # row version can still support a material-source link, labeled separately.
        row_versions = {content_hash(row) for row in native_rows}
        linked = [r for r in snapshot['observations'] if r['event_kind'] == 'tool'
                  and r['native_occurrence_id'] is not None and r['native_occurrence_id'] in tool_call_ids
                  and all(r.get(key) == binding.get(key) for key in
                      ('backend', 'source_store_id', 'source_epoch', 'profile_id'))
                  and binding.get('source_store_id')
                  and (r['occurrence_id'] is not None or
                       (r['payload'].get('native_row') is not None and
                        content_hash(r['payload']['native_row']) in row_versions))]
        pins = [{'artifact_ref': r['artifact_ref'], 'artifact_hash':r['artifact_hash']} for r in linked]
        selected_hash = 'sha256:' + content_hash(pins)
        resources = [{'observation_ref': r['artifact_ref'], 'occurrence_id': r['occurrence_id'],
            'duration_ms': r['payload'].get('duration_ms'), 'basis': r['collection_method']}
            for r in linked if r['payload'].get('duration_ms') is not None]
        return {'schema_version': 'learning_result_trace_views.v1',
            'snapshot_hash': selected_hash,
            'relation_basis': 'exact_source_occurrence_or_native_row_version; unknown_incarnation_not_resolved',
            'coverage': snapshot['coverage'], 'conflicts': snapshot['conflicts'],
            'collection_gaps': snapshot['collection_gaps'],
            'linked_observation_refs': [r['artifact_ref'] for r in linked],
            'unlinked_observation_count': len(snapshot['observations']) - len(linked),
            'observations': linked,
            'views': {
                'quality': {'status': 'unmeasured', 'causal_attribution': 'unknown'},
                'efficiency': {'resources': resources, 'supplier_cost': None,
                    'aggregation': 'source observations shown separately; overlapping spans and repeated usage are not summed'},
                'component_use': {'context_lineage': {'working_context': 'unavailable_without_owner_pin', 'attention': 'unavailable_without_owner_pin', 'invocation_payload': 'serialized_observations_retained_separately; not_proof_of_send_or_utilization'}, 'status': 'retained_source_versions' if linked else 'unavailable',
                    'items': [{'observation_ref': r['artifact_ref'], 'tool_name': r['payload'].get('tool_name'),
                               'phase': r['phase'], 'status': r['payload'].get('status')}
                              for r in linked], 'causal_attribution': 'unknown'},
                'approval': {'status': 'unavailable', 'items': [],
                    'reason': 'no owner-proven approval relation in this selected result'},
            }}


def native_trace_source(db):
    from hermes_state_backend import session_store_key, session_store_home
    key = session_store_key(db)
    return dict(backend='postgresql' if isinstance(key, tuple) else 'sqlite',
                source_store_id='native:' + content_hash(str(key)), source_epoch=None,
                profile_id='home:' + content_hash(str(session_store_home(db))))


def read_native_trace_basis(session_id):
    from agents.webapi.session_routes import _session_runtime_home
    from hermes_state_backend import open_session_db
    db = open_session_db(db_path=_session_runtime_home() / 'state.db', read_only=True)
    try:
        return native_trace_source(db), db.get_messages(session_id, limit=100, latest=True)
    finally:
        db.close()


def collect_native_history(*, service, context, session_id, offset=0, limit=100,
                           dry_run=False, cancelled=lambda: False):
    """One bounded rescan page through the native read-only owner, never a SQL fallback.

    Offset is a caller checkpoint, published only after durable evidence. Rescan the
    same page to detect UPDATEs; this does not claim an exhaustive change feed.
    """
    service.scope(context)
    if (not session_id.startswith(context.workspace_id + ':') or type(offset) is not int
            or offset < 0 or type(limit) is not int or not 1 <= limit <= 100):
        raise ValueError('INVALID_TRACE_PAGE')
    if cancelled():
        return {'status': 'cancelled', 'next_offset': offset, 'read_count': 0}
    from agents.webapi.session_routes import _session_runtime_home
    from hermes_state_backend import open_session_db, session_store_key, postgres_enabled
    db = open_session_db(db_path=_session_runtime_home() / 'state.db', read_only=True)
    try:
        session = db.get_session(session_id)
        if session is None:
            raise LookupError('TRACE_SESSION_NOT_FOUND')
        if session.get('ended_at') is None:
            return {'status': 'live_session_skipped', 'next_offset': offset, 'read_count': 0}
        source = native_trace_source(db)
        rows = db.get_messages(session_id, limit=limit + 1, offset=offset)
    finally:
        db.close()
    selected = rows[:limit]
    # Native locator is not an incarnation receipt. Never fabricate an epoch from
    # an inode/path, current runtime epoch, or message text.
    for row in selected:
        if cancelled():
            return {'status': 'cancelled', 'next_offset': offset, 'read_count': len(selected)}
        if dry_run:
            continue
        source_time = row.get('timestamp')
        if isinstance(source_time, (int, float)):
            source_time = datetime.fromtimestamp(source_time, timezone.utc)
        elif not isinstance(source_time, str):
            source_time = None
        service.record(context=context, **source, session_id=session_id,
            collection_method='poll', source_locator='native-message:' + str(row['id']),
            event_kind='tool' if row.get('role') == 'tool' else 'message',
            phase='terminal' if row.get('role') == 'tool' else 'committed',
            native_occurrence_id=row.get('tool_call_id') or str(row['id']),
            payload={'native_row': row, 'time_normalization': 'native_epoch_seconds_UTC' if isinstance(row.get('timestamp'), (int,float)) else 'source_ISO_or_missing',
                     'provenance': 'historical_native_row', 'request_stage': 'not_established'},
            source_occurred_at=source_time, captured_at=datetime.now(timezone.utc))
    return {'status': 'dry_run' if dry_run else 'retained', 'read_count': len(selected),
            'next_offset': offset if dry_run else offset + len(selected),
            'has_more': len(rows) > limit, 'coverage': 'bounded_page_rescan; updates_outside_page_not_claimed',
            'source_epoch': 'unavailable', 'source_store_id': source['source_store_id']}


class NativeLearningTraceObserver:
    """Bounded observation-only buffer. Producer callbacks perform no store I/O.

    The conversation owner drains on its own thread after disposing native hooks;
    no new daemon, worker/store connection or mutable request middleware is added.
    """
    def __init__(self, *, service, context, session_id, source_store_id, source_epoch,
                 profile_id, backend, capacity=128):
        from queue import Queue
        if not 1 <= capacity <= 128:
            raise ValueError('INVALID_TRACE_CAPACITY')
        service.scope(context)
        self.service, self.context, self.session_id = service, context, session_id
        self.source = dict(source_store_id=source_store_id, source_epoch=source_epoch,
                           profile_id=profile_id, backend=backend)
        self.queue = Queue(maxsize=capacity)
        self.dropped = 0

    def _enqueue(self, payload, *, kind, phase, native_id, locator):
        from queue import Full
        try:
            # Copy now: native hooks expose shallow mutable lists; later mutation
            # must not change retained request identity. Never copy transport headers.
            wire = json.dumps(payload, ensure_ascii=False)
            if len(wire.encode()) > self.service.MAX_PAYLOAD_BYTES:
                self.dropped += 1
                return
            self.queue.put_nowait(dict(event_kind=kind, phase=phase,
                native_occurrence_id=native_id, source_locator=locator,
                payload=json.loads(wire), source_occurred_at=None,
                captured_at=datetime.now(timezone.utc)))
        except (Full, ValueError, TypeError):
            self.dropped += 1

    def tool(self, **payload):
        if payload.get('session_id') != self.session_id:
            return None
        call = payload.get('tool_call_id')
        self._enqueue({key: payload[key] for key in ('tool_name', 'tool_call_id', 'task_id',
            'turn_id', 'api_request_id', 'result', 'status', 'duration_ms', 'error_type') if key in payload},
            kind='tool', phase='terminal', native_id=call,
            locator='hook:post_tool_call:' + (call or 'unresolved'))
        return None

    def request(self, **payload):
        if payload.get('session_id') != self.session_id:
            return None
        request_id = payload.get('api_request_id')
        retry = payload.get('retry_count')
        call = ('request-attempt:' + content_hash([request_id, retry])
                if request_id and type(retry) is int and retry >= 0 else None)
        request = payload.get('request') or {}
        body = request.get('body') or {}
        # Captured stage is serialization. It cannot attest a network send.
        safe = {k: v for k, v in body.items() if k in {'model', 'messages', 'tools',
            'max_tokens', 'max_completion_tokens', 'temperature', 'stream'}}
        if isinstance(payload.get('request_messages'), list):
            safe['messages'] = payload['request_messages']
        self._enqueue({'request_body': safe, 'request_stage': 'serialized',
            'usage_basis': 'local_estimate', 'approx_input_tokens': payload.get('approx_input_tokens'),
            'retry_count': retry, 'api_request_id': request_id},
            kind='request', phase='serialized', native_id=call,
            locator='hook:pre_api_request:' + (call or 'unresolved'))
        return None

    def flush(self):
        from queue import Empty
        persisted, errors = 0, 0
        while True:
            try:
                item = self.queue.get_nowait()
            except Empty:
                break
            try:
                self.service.record(context=self.context, **self.source,
                    session_id=self.session_id, collection_method='hook', **item)
                persisted += 1
            except (ValueError, OSError, PermissionError):
                errors += 1
        report = dict(persisted=persisted, dropped=self.dropped, errors=errors)
        # Gap metadata is not a native occurrence, usage counter or task outcome.
        self.service.record(context=self.context, **self.source, session_id=self.session_id,
            collection_method='hook', event_kind='message', phase='collection_coverage',
            native_occurrence_id=None, source_locator='collector:bounded-session-drain',
            payload={'collection_report': report}, source_occurred_at=None,
            captured_at=datetime.now(timezone.utc))
        return report


def validate_lesson_origin(*, context, store, origin, target_ref, target_hash):
    """Validate draft metadata at each existing writer, never grant authority."""
    if not origin or not origin.get('lesson_json'):
        return
    from contracts.learning_trace_evidence import ActionableLearningLessonV1
    from contracts.ontology_building import ArtifactPinV1
    lesson = ActionableLearningLessonV1.model_validate_json(origin['lesson_json'])
    if lesson.target_ref != target_ref or lesson.target_hash != target_hash:
        raise ValueError('LESSON_TARGET_MISMATCH')
    scope = LearningTraceEvidenceService.scope(context)
    for raw_pin in (*lesson.evidence_refs, *lesson.counter_evidence_refs):
        pin = ArtifactPinV1.model_validate(raw_pin)
        raw = store.get_raw(**scope, artifact_ref=pin.artifact_ref)
        if raw is None or raw['artifact_hash'] != pin.artifact_hash:
            raise ValueError('LESSON_EVIDENCE_MISMATCH')
        if raw['payload'].get('owner_ref', context.user_id) != context.user_id:
            raise PermissionError('FORBIDDEN_LESSON_EVIDENCE')


from contextlib import contextmanager

@contextmanager
def observe_native_conversation(*, context, session_id, native_db):
    """Lease observation hooks only for an authenticated, native-store-bound run.

    Capture is optional evidence, never execution authority. Failures are returned
    to the caller as unavailable rather than changing a task result or retrying.
    """
    report = {'status': 'unavailable', 'reason': 'native_context_or_store_absent'}
    handles, store, observer = [], None, None
    if context is not None and native_db is not None and session_id:
        try:
            from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
            from hermes_state_backend import session_store_key, postgres_enabled
            from hermes_cli.plugins import PluginContext, get_plugin_manager
            from hermes_cli.plugins_manifest import PluginManifest
            LearningTraceEvidenceService.scope(context)
            if not session_id.startswith(context.workspace_id + ':'):
                raise PermissionError('FORBIDDEN_TRACE_SESSION')
            source = native_trace_source(native_db)
            store = OntologyBuildStore(ontology_build_store_path(context.workspace_root))
            observer = NativeLearningTraceObserver(service=LearningTraceEvidenceService(store=store),
                context=context, session_id=session_id, **source)
            plugin = PluginContext(PluginManifest(name='semantier_learning_observer', version='1.0.0',
                description='Scoped observation only', path=context.hermes_home), get_plugin_manager())
            handles.append(plugin.register_hook('post_tool_call', observer.tool))
            handles.append(plugin.register_hook('pre_api_request', observer.request))
            report.update(status='observing', reason=None)
        except Exception as exc:
            report.update(status='unavailable', reason=type(exc).__name__)
    try:
        yield report
    finally:
        for handle in handles:
            handle.dispose()
        try:
            if observer is not None:
                report.update(status='retained', **observer.flush())
        except Exception as exc:
            report.update(status='unavailable', reason=type(exc).__name__)
        finally:
            if store is not None:
                store.close()
