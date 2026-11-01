"""Exact locally signed Channel1 manual candidate; never grants native admission."""
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from cryptography import x509

from .artifact_io import OperationError, safe_file

# Immutable facts from the actual producer and independent Setup signing receipts.
CHANNEL1_073 = {
    'build': '20bceda1fd864ca7a9b9dd67fa4879c3',
    'version': '0.0.73.0',
    'installer_filename': 'semantier-setup-channel1.exe',
    'zip_filename': 'SemantierAgent-0.0.73-win-x64.zip',
    'source': '0d651616b2c8d9809d43999249c34f46674323e9',
    'controller': 'ad87a4d05200330bad0cbb3c0249403392260fa5',
    'thumbprint': '57BF15739B4AA447842EAF388856A01B4E5029FF',
    'cert_sha256': '10a6b55e9a03bbc0dceb8d899c1776bf1fbad963c82050c16534da4983c9661d',
    'producer_sha256': 'da4fe57103720ec51243f4368f0ec804ea85be6d1b03eafac93e412c8fd598c6',
    'receipt_sha256': '2dbbafdf6afc616a646a857ff82f5b946bb988e024e4e616a64389cc8f324c41',
    'graph_sha256': '2f3eea86cc097494fb99f5e56b191ebb0b5667391e24f2e3789a973c2d409654',
    'installer_sha256': '7168c79a77c760161bb2bfbca953207d6bb83808102412e1886d5811acdfea38',
    'installer_size': 476664,
    'zip_sha256': 'cfeaec50103aa16bb4eacfecc67990f7e57c32a0dd2461c51459e93ea8ac2e7e',
    'zip_size': 2263077009,
}


def validate_channel1_073(incoming):
    """Validate pinned bytes and provenance before the existing journal can reserve."""
    facts = CHANNEL1_073
    prefix = 'manual-local/' + facts['build'] + '/'
    paths = {name: safe_file(Path(incoming), prefix + name) for name in
             ('result.json', 'receipt.json', 'native-graph.json', 'release.zip', 'installer.exe')}
    for name, field in (('result.json', 'producer'), ('receipt.json', 'receipt'),
                        ('native-graph.json', 'graph'), ('release.zip', 'zip'),
                        ('installer.exe', 'installer')):
        path = paths[name]
        if path.is_symlink() or not path.is_file():
            raise OperationError(422, 'manual_artifact_missing')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != facts[field + '_sha256']:
            raise OperationError(422, 'manual_artifact_hash_mismatch')
        if field in ('zip', 'installer') and path.stat().st_size != facts[field + '_size']:
            raise OperationError(422, 'manual_artifact_size_mismatch')
    try:
        producer = json.loads(paths['result.json'].read_text(encoding='utf-8-sig'))
        receipt = json.loads(paths['receipt.json'].read_text(encoding='utf-8-sig'))
        if (producer.get('result') != 'PASS' or producer.get('sourceCommit') != facts['source']
                or producer.get('controllerCommit') != facts['controller']
                or producer.get('certificateThumbprint') != facts['thumbprint']
                or producer.get('trustedTimestamp') is not False
                or receipt.get('sourceCommit') != facts['source']
                or receipt.get('packageVersion') != facts['version']
                or receipt.get('packageName') != 'Semantier.Agent.Acceptance'
                or receipt.get('publisher') != 'CN=Semantier Technology, O=Semantier Technology'
                or receipt.get('certificateThumbprint') != facts['thumbprint']
                or receipt.get('signing') != 'PASS'
                or not receipt.get('verification', '').startswith('PASS:')
                or receipt.get('trustedTimestamp') is not False):
            raise OperationError(422, 'manual_producer_mismatch')
        with ZipFile(paths['release.zip']) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise OperationError(422, 'manual_archive_graph_mismatch')
            graph = archive.read('native-graph.json')
            if hashlib.sha256(graph).hexdigest() != facts['graph_sha256']:
                raise OperationError(422, 'manual_archive_graph_mismatch')
            cert = archive.read('SemantierAgent-signing.cer')
            if hashlib.sha256(cert).hexdigest() != facts['cert_sha256']:
                raise OperationError(422, 'manual_certificate_mismatch')
            parsed = x509.load_der_x509_certificate(cert)
            if not parsed.not_valid_before_utc <= datetime.now(UTC) < parsed.not_valid_after_utc:
                raise OperationError(422, 'manual_certificate_expired')
    except OperationError:
        raise
    except (ValueError, KeyError, TypeError, BadZipFile) as error:
        raise OperationError(422, 'manual_artifact_invalid') from error
    return paths


def publish_channel1_073(store, remote, actor, track, incoming, *, stage_only=False):
    """Journal one exact manual release; resume an uncertain remote outcome in place."""
    from uuid import uuid4

    from .ecs_protocol import canonical, sha
    from .ecs_qualification import _authorize
    from .ecs_storage import manual_catalog_entries

    facts = CHANNEL1_073
    root = Path(incoming).resolve(strict=True)
    def authorize():
        _authorize(store, actor, track)
    authorize()
    validate_channel1_073(root)
    binding = {'endpoint': remote.endpoint, 'config_hash': remote.config_hash,
               'public_base': remote.base, 'incarnation': remote.key.incarnation,
               'auth_revision': remote.auth_revision,
               'private_tls_ca_sha256': remote.private_tls_ca_sha256,
               'incoming_root': str(root), 'release_id': facts['build']}

    def persist(operation, body):
        with store.connection() as c:
            result = c.execute(
                "UPDATE operations SET body=%s::jsonb,updated_at=now() "
                "WHERE operation_id=%s AND user_id=%s AND status='ecs_manual_pending'",
                (json.dumps(body), operation, actor),
            )
            if result.rowcount != 1:
                raise OperationError(503, 'ecs_manual_journal_changed')

    with store.connection() as claim:
        acquired = claim.execute('SELECT pg_try_advisory_xact_lock(hashtext(%s)) AS acquired',
                                 (store.schema + ':ecs-manual-channel1',)).fetchone()['acquired']
        if not acquired:
            raise OperationError(409, 'ecs_manual_publication_busy')
        with store.connection() as c:
            pending = c.execute("SELECT operation_id,user_id,channel,body FROM operations "
                                "WHERE status='ecs_manual_pending' ORDER BY created_at LIMIT 2").fetchall()
        if len(pending) > 1:
            raise OperationError(503, 'ecs_manual_journal_requires_recovery')
        if not stage_only and (not pending or pending[0]['body'].get('staged') is not True):
            raise OperationError(409, 'ecs_manual_stage_required')
        if pending:
            row = pending[0]
            body = row['body']
            if row['user_id'] != actor or row['channel'] != track or any(
                    body.get(k) != v for k, v in binding.items()):
                raise OperationError(503, 'ecs_manual_binding_changed')
            operation = row['operation_id']
            lock = body['remote_lock']
            catalog = bytes.fromhex(body['catalog_hex'])
            remote.recover_reservation(lock, lock['owner'],
                                       persist=lambda value: persist(operation, {**body, 'remote_lock': value}))
        else:
            raw, _ = remote.read('releases/manual/catalog.json')
            prior = manual_catalog_entries(raw) if raw is not None else {}
            if facts['build'] in prior or len(prior) >= 3:
                raise OperationError(409, 'ecs_manual_candidate_conflict')
            prefix = 'https://channel1.semantier.com/releases/channel-builds/' + facts['build'] + '/'
            entry = {'version': facts['version'], 'track': 'acceptance',
                     'identity': 'Semantier.Agent.Acceptance', 'provenance': 'local-signed',
                     'publishedAt': facts.get('published_at') or datetime.now(UTC).isoformat(),
                     'installer': {'url': prefix + facts['installer_filename'],
                                   'size': facts['installer_size'], 'sha256': facts['installer_sha256']},
                     'zip': {'url': prefix + facts['zip_filename'], 'size': facts['zip_size'],
                             'sha256': facts['zip_sha256']}}
            entries = sorted([*prior.values(), entry],
                             key=lambda row: tuple(map(int, row['version'].split('.'))), reverse=True)
            catalog = canonical({'schema': 1, 'releases': entries})
            manual_catalog_entries(catalog)
            operation = uuid4().hex
            owner = remote.identity({'operation_id': operation, 'kind': 'manual_download', 'actor': str(actor)})
            lock = remote.reservation(owner, track)
            body = {**binding, 'remote_lock': lock, 'catalog_hex': catalog.hex()}
            with store.connection() as c:
                c.execute("INSERT INTO operations(operation_id,user_id,channel,idempotency_key,request_hash,body,status) "
                          "VALUES(%s,%s,%s,%s,%s,%s::jsonb,'ecs_manual_pending')",
                          (operation, actor, track, operation, sha(canonical(body)), json.dumps(body)))
                store.audit(c, actor, 'ecs_manual_requested', facts['build'], {'operation_id': operation})
        with remote.locked(owner=lock['owner'], reservation=lock,
                           persist=lambda value: persist(operation, {**body, 'remote_lock': value}),
                           guard=authorize, hold_authorized=stage_only):
            if not lock.get('intent'):
                lock['intent'] = remote.manual_intent(root, lock, catalog)
                body['remote_lock'] = lock
                persist(operation, body)
            result = remote.publish_manual(root, lock, authorize=authorize, stage_only=stage_only)
        if stage_only:
            body.update(remote_lock=lock, staged=True)
            persist(operation, body)
            return {**result, 'operation_id': operation}
        if remote.status(lock)['state'] != 'RELEASED':
            raise OperationError(503, 'ecs_manual_release_unknown')
        with store.connection() as c:
            updated = c.execute("UPDATE operations SET status='ecs_manual_published',body=body || %s::jsonb,updated_at=now() "
                                "WHERE operation_id=%s AND user_id=%s AND status='ecs_manual_pending'",
                                (json.dumps({'result': result}), operation, actor))
            if updated.rowcount != 1:
                raise OperationError(503, 'ecs_manual_journal_changed')
            store.audit(c, actor, 'ecs_manual_published', facts['build'], {'operation_id': operation})
        return {**result, 'operation_id': operation}
