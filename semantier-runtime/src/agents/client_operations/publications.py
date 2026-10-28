"""Bounded three-target publication, durably resumed by existing maintenance."""
from contextlib import ExitStack, contextmanager
import json
import os
import socket
from uuid import uuid4
from contracts.client_operations import digest
from .publication import OperationError


ACTIVE = {'queued', 'preflighting', 'running', 'uncertain'}
CONFIRMED = {'published', 'already_current'}


def instance_identity():
    import psutil
    process = psutil.Process(os.getpid())
    return {'host': socket.gethostname(), 'pid': process.pid,
            'start_time_seconds': process.create_time(), 'instance': uuid4().hex}


class Publications:
    def __init__(self, targets):
        self.targets, self.store = targets, targets.store
        self.environment, self.channel = targets.environment, targets.channel
        self.instance = instance_identity()

    def authorize(self, actor, target_ids, action='publish'):
        capability = 'release_read' if action == 'read' else 'release_manage'
        if capability not in self.store.permissions(actor, self.channel):
            raise OperationError(403, 'explicit_platform_grant_required')
        self.targets.require(actor, target_ids, action)
        if action != 'read':
            if 'release_read' not in self.store.permissions(actor, self.channel):
                raise OperationError(403, 'explicit_platform_grant_required')
            self.targets.require(actor, target_ids, 'read')

    def promote_legacy(self, actor, request):
        """Capture the old single-target request into the same durable owner.

        The compatibility operation remembers the exact reviewed target intent;
        it never writes a head and survives loss before batch submission.
        """
        from contracts.client_operations import PublicationIntent
        target_id = self.targets.legacy_target_id
        self.authorize(actor, [target_id])
        supplied = request.model_dump()
        with self.store.connection() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (self.store.schema+':'+self.channel,))
            old = c.execute('SELECT * FROM operations WHERE user_id=%s AND idempotency_key=%s FOR UPDATE',
                            (actor, request.idempotency_key)).fetchone()
            if old:
                if old['request_hash'] != digest(supplied):
                    raise OperationError(409, 'operation_idempotency_conflict')
                if 'publication_intent' not in old['body']:
                    return old
                captured = old['body']['publication_intent']
                operation_id = old['operation_id']
            else:
                track = c.execute('SELECT revision,paused FROM channels WHERE channel=%s', (self.channel,)).fetchone()
                if not track or track['revision'] != request.expected_revision:
                    raise OperationError(409, 'channel_revision_conflict')
                if track['paused']:
                    raise OperationError(409, 'promotions_paused')
                preview = self.targets.preview(actor, request.release_id, target_id)
                captured = PublicationIntent(release_id=request.release_id, channel=self.channel,
                    delivery_target_ids=[target_id], target_bindings={target_id: preview['binding']['variant_hash']},
                    expected_revisions={target_id: preview['head']['revision']},
                    preview_plan_digests={target_id: preview['preview_plan_digest']},
                    idempotency_key=request.idempotency_key).model_dump()
                operation_id = uuid4().hex
                c.execute('INSERT INTO operations(operation_id,user_id,channel,idempotency_key,request_hash,body,status) '
                          'VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s)',
                          (operation_id, actor, self.channel, request.idempotency_key, digest(supplied),
                           json.dumps({**supplied, 'publication_intent': captured}), 'capturing'))
        publication = self.submit(actor, PublicationIntent.model_validate(captured))
        with self.store.connection() as c:
            c.execute("UPDATE operations SET status='batch_owned',updated_at=now() WHERE operation_id=%s", (operation_id,))
        return {'operation_id': operation_id, 'publication_id': publication['publication_id'],
                'body': supplied, 'status': self.legacy_status(publication)}

    @staticmethod
    def legacy_status(publication):
        if publication['lifecycle'] == 'active':
            return 'preparing'
        return 'published' if publication['status'] == 'succeeded' else 'failed'

    def submit(self, actor, request):
        if request.channel != self.channel:
            raise OperationError(403, 'channel_scope_denied')
        self.authorize(actor, request.delivery_target_ids)
        try:
            publication_id = self._capture(actor, request)
        except OperationError as error:
            error.mutation_outcome = 'not_started'
            raise
        # A post-commit permission/read failure does not prove no mutation.
        return self.snapshot(actor, publication_id)

    def _capture(self, actor, request):
        intent = request.model_dump()
        request_hash = digest(intent)
        with self.store.connection() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',
                      (self.store.schema+':'+self.channel,))
            previous = c.execute('SELECT * FROM publications WHERE user_id=%s AND environment=%s '
                                 'AND idempotency_key=%s FOR UPDATE',
                                 (actor, self.environment, request.idempotency_key)).fetchone()
            if previous:
                if previous['request_hash'] != request_hash:
                    raise OperationError(409, 'publication_idempotency_conflict')
                publication_id = previous['publication_id']
            else:
                track = c.execute('SELECT * FROM channels WHERE channel=%s FOR UPDATE', (self.channel,)).fetchone()
                if not track:
                    raise OperationError(503, 'release_track_unavailable')
                if not c.execute("SELECT 1 FROM releases WHERE release_id=%s AND qualification='verified'",
                                 (request.release_id,)).fetchone():
                    raise OperationError(422, 'verified_release_required')
                busy = c.execute('SELECT 1 FROM publication_targets WHERE environment=%s AND channel=%s '
                                 'AND target_id=ANY(%s) AND reserved LIMIT 1',
                                 (self.environment, self.channel, request.delivery_target_ids)).fetchone()
                legacy_busy = c.execute("SELECT 1 FROM operations WHERE channel=%s AND status='preparing' LIMIT 1",
                                        (self.channel,)).fetchone()
                error = 'promotions_paused' if track['paused'] else 'delivery_target_busy' if busy or legacy_busy else None
                publication_id = uuid4().hex
                c.execute('INSERT INTO publications(publication_id,user_id,environment,channel,release_id,idempotency_key,request_hash,intent,status,lifecycle) '
                          'VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)',
                          (publication_id, actor, self.environment, self.channel, request.release_id, request.idempotency_key,
                           request_hash, json.dumps(intent), 'blocked' if error else 'preflighting', 'settled' if error else 'active'))
                for target_id in request.delivery_target_ids:
                    operation_id = digest({'publication_id': publication_id, 'target_id': target_id, 'attempt': 1})[:32]
                    c.execute('INSERT INTO publication_targets(publication_id,target_id,environment,channel,attempt,variant_hash,operation_id,status,reserved,expected_revision,preview_plan_digest,error_code) '
                              'VALUES(%s,%s,%s,%s,1,%s,%s,%s,%s,%s,%s,%s)',
                              (publication_id, target_id, self.environment, self.channel, request.target_bindings[target_id],
                               operation_id, 'blocked' if error else 'queued', not bool(error),
                               request.expected_revisions[target_id], request.preview_plan_digests[target_id], error))
                self.store.audit(c, actor, 'publication_requested', publication_id,
                                 {'release_id': request.release_id, 'channel': self.channel, 'target_ids': request.delivery_target_ids})
        return publication_id

    def _load(self, publication_id):
        with self.store.connection() as c:
            parent = c.execute('SELECT * FROM publications WHERE publication_id=%s AND environment=%s AND channel=%s',
                               (publication_id, self.environment, self.channel)).fetchone()
            if not parent:
                raise OperationError(404, 'publication_unavailable')
            children = c.execute('SELECT * FROM publication_targets WHERE publication_id=%s ORDER BY target_id,attempt',
                                 (publication_id,)).fetchall()
        return parent, children

    def lookup(self, actor, key):
        with self.store.connection() as c:
            row = c.execute('SELECT publication_id FROM publications WHERE user_id=%s AND environment=%s AND channel=%s AND idempotency_key=%s',
                            (actor, self.environment, self.channel, key)).fetchone()
        if not row:
            raise OperationError(404, 'publication_unavailable')
        return self.snapshot(actor, row['publication_id'])

    def snapshot(self, actor, publication_id):
        parent, children = self._load(publication_id)
        self.authorize(actor, parent['intent']['delivery_target_ids'], 'read')
        # Process and private bucket-lock identity are maintenance-only facts.
        safe_children = [{k: v for k, v in row.items() if k not in {'owner_instance', 'remote_lock'}} for row in children]
        latest = {row['target_id']: row for row in children}
        counts = {state: sum(row['status'] == state for row in latest.values()) for state in sorted({row['status'] for row in latest.values()})}
        return {**parent, 'targets': safe_children, 'counts': counts}

    def list(self, actor, release_id=None, status='all', offset=0):
        if 'release_read' not in self.store.permissions(actor, self.channel):
            raise OperationError(403, 'explicit_platform_grant_required')
        if status not in {'active', 'all'} or not 0 <= offset <= 10000:
            raise OperationError(422, 'invalid_publication_window')
        with self.store.connection() as c:
            rows = c.execute("SELECT p.publication_id FROM publications p WHERE environment=%s AND channel=%s "
                             "AND (%s::text IS NULL OR release_id=%s) AND (%s='all' OR lifecycle='active' OR maintenance_status IN ('pending','failed')) "
                             "AND NOT EXISTS(SELECT 1 FROM publication_targets t WHERE t.publication_id=p.publication_id AND NOT EXISTS("
                             "SELECT 1 FROM delivery_grants g WHERE g.principal_id=%s AND g.channel=p.channel AND g.target_id=t.target_id AND g.action='read')) "
                             'ORDER BY created_at DESC,publication_id DESC LIMIT 101 OFFSET %s',
                             (self.environment, self.channel, release_id, release_id, status, actor, offset)).fetchall()
        return {'items': [self.snapshot(actor, row['publication_id']) for row in rows[:100]],
                'next_offset': offset+100 if len(rows) > 100 else None}

    def _target(self, operation_id, **changes):
        allowed = {'status', 'stage', 'reserved', 'previous_head', 'intended_head', 'capacity', 'error_code',
                   'actual_revision', 'readback_digest', 'owner_instance', 'remote_lock', 'maintenance_status'}
        if not changes or not set(changes) <= allowed:
            raise ValueError('invalid target transition')
        from psycopg import sql
        json_fields = {'previous_head', 'intended_head', 'capacity', 'owner_instance', 'remote_lock'}
        assignments, values = [], []
        for key, value in changes.items():
            assignments.append(sql.SQL('{}={}').format(sql.Identifier(key), sql.SQL('%s::jsonb' if key in json_fields else '%s')))
            values.append(json.dumps(value) if key in json_fields else value)
        with self.store.connection() as c:
            current = c.execute('SELECT * FROM publication_targets WHERE operation_id=%s FOR UPDATE', (operation_id,)).fetchone()
            if not current:
                raise OperationError(404, 'publication_target_unavailable')
            if all(current[key] == value for key, value in changes.items()):
                return
            parent = c.execute(sql.SQL('UPDATE publication_targets SET {},updated_at=now() WHERE operation_id=%s RETURNING publication_id').format(
                sql.SQL(',').join(assignments)), (*values, operation_id)).fetchone()
            c.execute('UPDATE publications SET batch_revision=batch_revision+1,updated_at=now() WHERE publication_id=%s',
                      (parent['publication_id'],))

    def _summarize(self, publication_id):
        with self.store.connection() as c:
            rows = c.execute('SELECT DISTINCT ON(target_id) * FROM publication_targets WHERE publication_id=%s '
                             'ORDER BY target_id,attempt DESC', (publication_id,)).fetchall()
            states = [row['status'] for row in rows]
            if 'preflighting' in states:
                status = 'preflighting'
            elif any(state in {'queued', 'running'} for state in states):
                status = 'running'
            elif 'uncertain' in states:
                status = 'uncertain'
            elif all(state in CONFIRMED for state in states):
                status = 'succeeded'
            elif any(state in CONFIRMED for state in states):
                status = 'partial'
            else:
                status = 'blocked' if all(state == 'blocked' for state in states) else 'failed'
            maintenance = 'failed' if any(row['maintenance_status'] == 'failed' for row in rows) else 'pending' if any(row['maintenance_status'] == 'pending' for row in rows) else 'none'
            c.execute('UPDATE publications SET status=%s,lifecycle=%s,maintenance_status=%s,batch_revision=batch_revision+1,updated_at=now() '
                      'WHERE publication_id=%s AND (status,maintenance_status) IS DISTINCT FROM (%s,%s)',
                      (status, 'active' if any(state in ACTIVE for state in states) else 'settled', maintenance,
                       publication_id, status, maintenance))

    def _acquire(self, parent, row, stack, status):
        owner = self.targets.owner(row['target_id'])
        lock_owner = {**self.instance, 'publication_id': parent['publication_id'], 'operation_id': row['operation_id'], 'release_id': parent['release_id']}
        lock = {'owner': lock_owner, 'id': uuid4().hex}
        self._target(row['operation_id'], status=status, stage='acquiring_lock', reserved=True,
                     remote_lock=lock, owner_instance=lock_owner)
        stack.enter_context(owner.remote_target.locked(owner=lock_owner, reservation=lock))
        return lock

    def _prepare(self, parent, children, stack):
        self.authorize(parent['user_id'], [row['target_id'] for row in children])
        previews, remote_locks = {}, {}
        for row in children:
            owner = self.targets.owner(row['target_id'])
            if owner.remote_target:
                remote_locks[row['target_id']] = self._acquire(parent, row, stack, 'preflighting')
        for row in children:
            self._target(row['operation_id'], status='preflighting', stage='preflight')
            preview = self.targets.preview(parent['user_id'], parent['release_id'], row['target_id'],
                                           variant_hash=row['variant_hash'], reservation=parent['publication_id'],
                                           remote_reservation=remote_locks.get(row['target_id']))
            if preview['head']['revision'] != row['expected_revision']:
                raise OperationError(409, 'channel_revision_conflict')
            if preview['preview_plan_digest'] != row['preview_plan_digest']:
                raise OperationError(409, 'review_updated_storage_plan')
            intended = self.targets.owner(row['target_id']).intended_record(preview['binding']['body'], preview['head']['record'])
            self._target(row['operation_id'], previous_head=preview['head']['record'], intended_head=intended,
                         capacity=preview['capacity'])
            previews[row['target_id']] = preview
        return previews, remote_locks

    @contextmanager
    def _locks(self, publication_id):
        stack = ExitStack()
        try:
            yield stack
        finally:
            try:
                stack.close()
            except (OperationError, OSError):
                pass  # Read back every durable acquisition intent below.
            from .r2_target import LOCK_KEY
            from contracts.client_operations import canonical
            _, history = self._load(publication_id)
            latest = {row['target_id']: row for row in history}
            for row in latest.values():
                owner = self.targets.publishers.get(row['target_id'])
                if not owner or not owner.remote_target:
                    continue
                try:
                    raw, _ = owner.remote_target.read(LOCK_KEY)
                    unresolved = bool(row['remote_lock'] and raw == canonical(row['remote_lock']))
                except (OperationError, OSError, owner.remote_target.transport.R2RequestError):
                    unresolved = bool(row['remote_lock'])
                if unresolved:
                    self._target(row['operation_id'], maintenance_status='failed', reserved=True)
                elif row['status'] not in ACTIVE:
                    self._target(row['operation_id'], maintenance_status='none' if row['maintenance_status']=='pending' else row['maintenance_status'], reserved=False)
            self._summarize(publication_id)

    def run(self, publication_id):
        """One maintenance worker; DB reservation survives process/HTTP loss."""
        with self.store.connection() as claim:
            acquired = claim.execute('SELECT pg_try_advisory_xact_lock(hashtext(%s)) AS acquired',
                                     (self.store.schema+':publication:'+publication_id,)).fetchone()['acquired']
            if not acquired:
                return
            parent, history = self._load(publication_id)
            latest = {row['target_id']: row for row in history}
            children = [row for row in latest.values() if row['status'] in ACTIVE]
            if not children:
                return self._maintenance(parent, list(latest.values()))
            for row in children:
                if row['status'] in {'queued', 'preflighting'} and row['remote_lock']:
                    try:
                        self.targets.owner(row['target_id']).remote_target.recover_owned_lock(row['remote_lock'], row['owner_instance'], current_instance=self.instance)
                        self._target(row['operation_id'], remote_lock=None, owner_instance=None)
                    except (OperationError, OSError):
                        return
            # A prior process may have crossed the external-head boundary.
            # Reconcile it before admitting another upload or deleting its lock.
            if any(row['status'] in {'running', 'uncertain'} for row in children):
                return self._recover(parent, children)
            with self._locks(publication_id) as stack:
                try:
                    previews, remote_locks = self._prepare(parent, children, stack)
                except (OperationError, OSError) as error:
                    code = error.code if isinstance(error, OperationError) else 'target_preflight_unavailable'
                    for row in children:
                        self._target(row['operation_id'], status='blocked', stage='preflight', reserved=bool(self.targets.publishers.get(row['target_id']) and self.targets.publishers[row['target_id']].remote_target), error_code=code)
                    self._summarize(publication_id)
                    return
                # The complete selected-target barrier passed while reservations remain held.
                for row in children:
                    preview = previews[row['target_id']]
                    owner = self.targets.owner(row['target_id'])
                    if preview['readiness'] == 'already_current':
                        self._target(row['operation_id'], status='already_current', stage='confirmed', reserved=bool(owner.remote_target),
                                     actual_revision=preview['head']['revision'], readback_digest=digest(preview['head']['record']),
                                     maintenance_status='pending' if owner.remote_target else 'none')
                        continue
                    self._target(row['operation_id'], status='running', stage='uploading')
                    try:
                        result = owner.publish_variant(preview['binding']['body'], row['operation_id'],
                            remote_reservation=remote_locks.get(row['target_id']),
                            authorize=lambda: self.authorize(parent['user_id'], [row['target_id']]))
                        self._target(row['operation_id'], status='published', stage='confirmed', reserved=bool(owner.remote_target),
                                     actual_revision=result['revision'], readback_digest=result['readback_digest'],
                                     maintenance_status='failed' if result.get('maintenance_status')=='failed' else 'pending' if owner.remote_target else 'none')
                    except (OperationError, OSError) as error:
                        known = isinstance(error, OperationError) and error.status < 500
                        self._target(row['operation_id'], status='failed' if known else 'uncertain',
                                     stage='verifying', reserved=not known,
                                     error_code=error.code if isinstance(error, OperationError) else 'publication_outcome_unknown')
                self._summarize(publication_id)

    def _maintenance(self, parent, children):
        for row in children:
            if row['status'] not in CONFIRMED | {'blocked', 'failed'} or (row['maintenance_status'] not in {'pending','failed'} and not row['reserved']):
                continue
            owner = self.targets.owner(row['target_id'])
            if not owner.remote_target:
                continue
            with self.store.connection() as c:
                if c.execute('SELECT 1 FROM publication_targets WHERE target_id=%s AND reserved AND operation_id<>%s LIMIT 1',
                             (row['target_id'], row['operation_id'])).fetchone():
                    continue
            try:
                self.authorize(parent['user_id'], [row['target_id']])
                owner.remote_target.recover_owned_lock(row['remote_lock'], row['owner_instance'], current_instance=self.instance)
                with self._locks(parent['publication_id']) as stack:
                    if row['status'] in CONFIRMED:
                        reservation = self._acquire(parent, row, stack, row['status'])
                        owner.remote_target.cleanup(parent['release_id'],
                            lambda: self.authorize(parent['user_id'], [row['target_id']]), reservation=reservation)
                    self._target(row['operation_id'], maintenance_status='pending' if row['status'] in CONFIRMED else 'none')
            except (OperationError, OSError):
                continue
        self._summarize(parent['publication_id'])

    def _recover(self, parent, children):
        for row in children:
            if row['status'] not in {'running', 'uncertain'}:
                continue
            owner = self.targets.owner(row['target_id'])
            try:
                if owner.remote_target:
                    owner.remote_target.recover_owned_lock(row['remote_lock'], row['owner_instance'], current_instance=self.instance)
                binding = self.targets.binding(parent['release_id'], row['target_id'], row['variant_hash'])
                with self._locks(parent['publication_id']) as stack:
                    reservation = self._acquire(parent, row, stack, 'uncertain') if owner.remote_target else None
                    result = owner.publish_variant(binding['body'], row['operation_id'], remote_reservation=reservation,
                        authorize=lambda: self.authorize(parent['user_id'], [row['target_id']]))
                    self._target(row['operation_id'], status='published', stage='confirmed', reserved=bool(owner.remote_target),
                                 actual_revision=result['revision'], readback_digest=result['readback_digest'],
                                 maintenance_status='failed' if result.get('maintenance_status')=='failed' else 'pending' if owner.remote_target else 'none', error_code=None)
            except (OperationError, OSError) as error:
                self._target(row['operation_id'], status='uncertain', stage='reconciling', reserved=True,
                             error_code=error.code if isinstance(error, OperationError) else 'publication_outcome_unknown')
        self._summarize(parent['publication_id'])

    def reconcile(self, actor, publication_id):
        parent, _ = self._load(publication_id)
        self.authorize(actor, parent['intent']['delivery_target_ids'])
        # A trigger only; existing maintenance owns resumed writes.
        with self.store.connection() as c:
            self.store.audit(c, actor, 'publication_reconcile_requested', publication_id, {})
        return self.snapshot(actor, publication_id)

    def retry(self, actor, publication_id, request):
        parent,_ = self._load(publication_id)
        self.authorize(actor, parent['intent']['delivery_target_ids'], 'read')
        self.authorize(actor, request.delivery_target_ids)
        if not set(request.delivery_target_ids) <= set(parent['intent']['delivery_target_ids']):
            raise OperationError(422, 'retry_target_outside_original_selection')
        try:
            captured = self._capture_retry(actor, publication_id, request)
        except OperationError as error:
            error.mutation_outcome = 'not_started'
            raise
        return self.snapshot(actor, captured)

    def _capture_retry(self, actor, publication_id, request):
        parent, history = self._load(publication_id)
        retry_hash = digest(request.model_dump())
        with self.store.connection() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (self.store.schema+':'+self.channel,))
            locked = c.execute('SELECT * FROM publications WHERE publication_id=%s FOR UPDATE', (publication_id,)).fetchone()
            previous = c.execute("SELECT body FROM audit WHERE action='publication_retry' AND actor=%s AND object_id=%s AND body->>'idempotency_key'=%s",
                                 (actor, publication_id, request.idempotency_key)).fetchone()
            if previous:
                if previous['body']['request_hash'] != retry_hash:
                    raise OperationError(409, 'publication_idempotency_conflict')
                return publication_id
            if locked['batch_revision'] != request.expected_batch_revision:
                raise OperationError(409, 'publication_revision_conflict')
            history = c.execute('SELECT * FROM publication_targets WHERE publication_id=%s ORDER BY target_id,attempt', (publication_id,)).fetchall()
            latest = {row['target_id']: row for row in history}
            for target_id in request.delivery_target_ids:
                row = latest[target_id]
                if row['status'] != 'failed':
                    raise OperationError(409, 'reconcile_before_retry')
                if c.execute('SELECT 1 FROM publication_targets WHERE environment=%s AND target_id=%s AND channel=%s AND reserved',
                             (self.environment, target_id, self.channel)).fetchone():
                    raise OperationError(409, 'delivery_target_busy')
                attempt = row['attempt']+1
                operation_id = digest({'publication_id': publication_id, 'target_id': target_id, 'attempt': attempt})[:32]
                c.execute('INSERT INTO publication_targets(publication_id,target_id,environment,channel,attempt,variant_hash,operation_id,expected_revision,preview_plan_digest) '
                          'VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                          (publication_id, target_id, self.environment, self.channel, attempt, row['variant_hash'],
                           operation_id, request.expected_revisions[target_id], request.preview_plan_digests[target_id]))
            c.execute("UPDATE publications SET status='preflighting',lifecycle='active',batch_revision=batch_revision+1,updated_at=now() WHERE publication_id=%s", (publication_id,))
            self.store.audit(c, actor, 'publication_retry', publication_id,
                             {'idempotency_key': request.idempotency_key, 'request_hash': retry_hash,
                              'target_ids': request.delivery_target_ids})
        return publication_id

    def maintain(self):
        from contracts.client_operations import Promotion
        with self.store.connection() as c:
            captures = c.execute("SELECT user_id,body FROM operations WHERE channel=%s AND status='capturing' ORDER BY created_at LIMIT 10", (self.channel,)).fetchall()
        for row in captures:
            try:
                request = {key: row['body'][key] for key in ('release_id','expected_revision','idempotency_key')}
                self.promote_legacy(row['user_id'], Promotion.model_validate(request))
            except OperationError:
                continue
        with self.store.connection() as c:
            rows = c.execute("SELECT publication_id FROM publications WHERE environment=%s AND channel=%s AND (lifecycle='active' OR maintenance_status IN ('pending','failed') OR EXISTS(SELECT 1 FROM publication_targets t WHERE t.publication_id=publications.publication_id AND t.reserved)) ORDER BY created_at LIMIT 10",
                             (self.environment, self.channel)).fetchall()
        for row in rows:
            try:
                self.run(row['publication_id'])
            except OperationError as error:
                if error.status!=503:
                    raise
                # Keep unavailable provider outcomes/reservations unchanged;
                # independent batches can still make progress.
                continue
