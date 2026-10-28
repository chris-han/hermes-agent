"""Configured delivery owners and immutable origin-bound release admission."""
import hashlib
import hmac
import json
from copy import deepcopy
from contracts.client_operations import DELIVERY_TARGET_IDS, canonical, digest, utcnow
from .artifact_io import OperationError, file_facts, safe_file


class DeliveryTargets:
    def __init__(self, store, environment, publishers, channel, errors=None, legacy_target_id=None, origins=None):
        if not set(publishers) <= set(DELIVERY_TARGET_IDS):
            raise ValueError('unsupported delivery target')
        self.store, self.environment, self.publishers, self.channel = store, environment, publishers, channel
        self.errors = errors or {}
        self.legacy_target_id = legacy_target_id
        self.origins = origins if origins is not None else {key: owner.base for key,owner in publishers.items()}

    def owner(self, target_id):
        if target_id in self.errors:
            raise OperationError(503, self.errors[target_id])
        if target_id not in self.publishers:
            raise OperationError(503, self.errors.get(target_id, 'delivery_target_not_configured'))
        return self.publishers[target_id]

    def require(self, principal_id, target_ids, action):
        grants = self.store.target_permissions(principal_id, self.channel)
        if any(action not in grants.get(target, set()) for target in target_ids):
            raise OperationError(403, 'delivery_target_scope_denied')

    def admit(self, admission, principal):
        target_id = admission.delivery_target_id
        self.require(principal['principal_id'], [target_id], 'ingest')
        publisher = self.owner(target_id)
        candidate = admission.candidate
        with self.store.connection() as c:
            original = c.execute("SELECT body FROM releases WHERE release_id=%s AND qualification='verified'",
                                 (candidate.release_id,)).fetchone()
        if not original:
            raise OperationError(422, 'verified_release_required')
        original = original['body']
        expected = deepcopy(original['manifest'])
        expected['request']['publicBase'] = publisher.base
        supplied = {**candidate.manifest, 'notes': candidate.notes}
        if (supplied != expected or candidate.native_receipt != original['native_receipt']
                or candidate.receipt_hmac != original['receipt_hmac']):
            raise OperationError(422, 'target_variant_changes_candidate')
        receipt = admission.descriptor_receipt
        expected_receipt = {
            'delivery_target_id': target_id, 'public_base': publisher.base,
            'source_commit': expected['request']['commit'],
            'build_id': expected['request']['buildId'], 'manifest_sha256': digest(expected),
            'sha256': expected['packages'][0]['artifact']['sha256'],
        }
        if (not publisher.receipt_key or any(receipt.get(k) != v for k, v in expected_receipt.items())
                or not hmac.compare_digest(admission.descriptor_receipt_hmac,
                    hmac.new(publisher.receipt_key, canonical(receipt), hashlib.sha256).hexdigest())):
            raise OperationError(422, 'trusted_target_descriptor_receipt_required')
        descriptor = safe_file(publisher.incoming, expected['packages'][0]['feed']['key'])
        if descriptor.stat().st_size > 1024 * 1024 or file_facts(descriptor)[1] != receipt.get('descriptor_sha256'):
            raise OperationError(422, 'target_descriptor_integrity_mismatch')
        from .ci import verify_workflow
        verification = (publisher.workflow_verifier or verify_workflow)(receipt)
        body = publisher.admit(candidate, principal, persist=False)
        body['descriptor_receipt'] = receipt
        body['descriptor_ci_verification'] = verification
        variant_hash = digest({'target_id': target_id, 'manifest': body['manifest'],
                               'descriptor_receipt': receipt, 'native_receipt': body['native_receipt']})
        with self.store.connection() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',
                      (self.store.schema+':admission:'+target_id+':'+candidate.release_id,))
            first = c.execute('SELECT min(first_admitted_at) AS first FROM release_delivery_variants '
                              'WHERE release_id=%s AND target_id=%s', (candidate.release_id, target_id)).fetchone()['first']
            row = c.execute('INSERT INTO release_delivery_variants(release_id,target_id,variant_hash,body,first_admitted_at) '
                            'VALUES(%s,%s,%s,%s::jsonb,COALESCE(%s,now())) ON CONFLICT DO NOTHING RETURNING *',
                            (candidate.release_id, target_id, variant_hash, json.dumps(body), first)).fetchone()
            if row:
                self.store.audit(c, principal['principal_id'], 'target_binding_admitted', candidate.release_id,
                                 {'target_id': target_id, 'variant_hash': variant_hash, 'public_base': publisher.base})
            else:
                row = c.execute('SELECT * FROM release_delivery_variants WHERE release_id=%s AND target_id=%s AND variant_hash=%s',
                                (candidate.release_id, target_id, variant_hash)).fetchone()
        return row

    def binding(self, release_id, target_id, variant_hash=None):
        with self.store.connection() as c:
            row = c.execute('SELECT * FROM release_delivery_variants WHERE release_id=%s AND target_id=%s '
                            'AND (%s::text IS NULL OR variant_hash=%s) ORDER BY first_admitted_at DESC,variant_hash LIMIT 1',
                            (release_id, target_id, variant_hash, variant_hash)).fetchone()
        if not row:
            raise OperationError(422, 'target_qualification_missing')
        if row['body']['manifest']['request']['publicBase'] != self.owner(target_id).base:
            raise OperationError(422, 'target_origin_binding_changed')
        return row

    def head(self, target_id, *, verify=True):
        publisher = self.owner(target_id)
        with self.store.connection() as c:
            head = c.execute('SELECT * FROM delivery_heads WHERE target_id=%s AND channel=%s',
                             (target_id, self.channel)).fetchone()
        if not head:
            raise OperationError(503, 'delivery_head_unavailable')
        if verify and publisher.read_record() != head['record']:
            raise OperationError(503, 'delivery_head_reconciliation_required')
        return head

    def preview(self, principal_id, release_id, target_id, *, variant_hash=None, reservation=None, remote_reservation=None):
        self.require(principal_id, [target_id], 'read')
        publisher = self.owner(target_id)
        head = self.head(target_id)
        binding = self.binding(release_id, target_id, variant_hash)
        with self.store.connection() as c:
            track = c.execute('SELECT paused,revision FROM channels WHERE channel=%s', (self.channel,)).fetchone()
            busy = c.execute('SELECT publication_id FROM publication_targets WHERE environment=%s AND target_id=%s '
                             'AND channel=%s AND reserved AND (%s::text IS NULL OR publication_id<>%s) LIMIT 1',
                             (self.environment, target_id, self.channel, reservation, reservation)).fetchone()
        if busy:
            raise OperationError(409, 'delivery_target_busy')
        if track['paused']:
            raise OperationError(409, 'promotions_paused')
        already_current = bool(head['record'].get('head') and
                               head['record']['head']['sha256'] == binding['body']['manifest_sha256'])
        release = {**binding['body'], 'retention_admission': {'release_id': release_id, 'first_admitted_at': binding['first_admitted_at'].isoformat()}}
        capacity = publisher.preflight(release, remote_reservation=remote_reservation) if not already_current else None
        facts = {'target_id': target_id, 'variant_hash': binding['variant_hash'], 'head': head['record'], 'capacity': capacity}
        return {'delivery_target_id': target_id, 'public_base': publisher.base,
                'readiness': 'already_current' if already_current else 'ready',
                'head': head, 'binding': binding, 'capacity': capacity,
                'preview_plan_digest': digest(facts), 'fetched_at': utcnow()}

    def catalog(self, principal_id, release_id):
        rows = []
        grants = self.store.target_permissions(principal_id, self.channel)
        for target_id in DELIVERY_TARGET_IDS:
            actions = sorted(grants.get(target_id, set()))
            row = {'delivery_target_id': target_id, 'allowed_actions': actions,
                   'configured': target_id in self.origins, 'public_base': None}
            try:
                preview = self.preview(principal_id, release_id, target_id)
                row.update({**preview, 'binding': {'variant_hash': preview['binding']['variant_hash']}})
                if row['capacity']:
                    row['capacity'] = {key: value for key, value in row['capacity'].items() if key != 'retention'}
            except OperationError as error:
                row.update(readiness='not_permitted' if error.status == 403 else 'unavailable', error_code=error.code)
                if 'read' in actions and target_id in self.origins:
                    row['public_base'] = self.origins[target_id]
            except OSError:
                row.update(readiness='unavailable', error_code='target_artifacts_unavailable')
            rows.append(row)
        return {'items': rows, 'fetched_at': utcnow(), 'coverage': 'explicit_three_targets'}
