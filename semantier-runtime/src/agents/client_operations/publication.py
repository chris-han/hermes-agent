"""One x64 publisher. Upstream full-matrix admission remains unchanged."""
import hashlib
import hmac
import json
from pathlib import Path
from uuid import uuid4
from contracts.client_operations import canonical,digest,utcnow


from .artifact_io import OperationError, safe_file, atomic_write, immutable_write, file_facts, immutable_copy


class Publisher:
    def __init__(self,store,public_root,incoming_root,public_base,receipt_key,identity,channel,workflow_verifier=None,remote_target=None,target_id=None):
        self.store,self.root,self.incoming=store,Path(public_root),Path(incoming_root)
        self.base,self.receipt_key,self.identity,self.channel=public_base,receipt_key,identity,channel
        self.workflow_verifier=workflow_verifier
        self.remote_target=remote_target
        self.target_id = target_id or ("channel2" if remote_target else "channel1")
        if self.target_id not in {"channel1", "channel2", "channel3"}:
            raise ValueError("unsupported publication target")

    def empty_record(self):
        return {'schema':1,'name':self.channel,'repository':'chris-han/hermes-agent','policy':'stable-release','revision':1,'nextSequence':1,'identity':self.identity,'head':None,'state':'active'}

    def initialize_channel(self):
        """One target head, separate from shared track policy; no auto-promotion."""
        record = self.empty_record()
        with self.store.connection() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (self.store.schema+':'+self.channel,))
            c.execute('INSERT INTO channels(channel,record) VALUES(%s,%s::jsonb) ON CONFLICT DO NOTHING',
                      (self.channel, json.dumps(record)))
            c.execute('INSERT INTO delivery_heads(target_id,channel,record) VALUES(%s,%s,%s::jsonb) ON CONFLICT DO NOTHING',
                      (self.target_id, self.channel, json.dumps(record)))
            head = c.execute('SELECT record FROM delivery_heads WHERE target_id=%s AND channel=%s',
                             (self.target_id, self.channel)).fetchone()['record']
            if self.remote_target:
                self.remote_target.initialize(head)
            else:
                target = safe_file(self.root, 'releases/channels/'+self.channel+'.json')
                if target.exists() and json.loads(target.read_bytes()) != head:
                    raise OperationError(503, 'channel_provenance_mismatch')
                immutable_write(target, canonical(head))

    def admit(self,candidate,principal,*,persist=True):
        from agents.webapi.deps import ensure_repo_hermes_agent_on_path
        ensure_repo_hermes_agent_on_path()
        from hermes_cli.release_channels import validate_manifest,validate_record,build_prefix
        data=candidate.model_dump();manifest={**data['manifest'],'notes':data['notes']};data['manifest']=manifest;request=manifest.get('request',{})
        if data['release_id']!=request.get('buildId'):raise OperationError(422,'release_build_identity_mismatch')
        record=self.empty_record()
        try:
            validate_record(record)
            validate_manifest(manifest,record,self.base)
        except ValueError as exc:raise OperationError(422,'manifest_contract_mismatch') from exc
        packages=manifest['packages']
        if len(packages)!=1 or (packages[0]['platform'],packages[0]['arch'])!=('win32','x64'):
            raise OperationError(422,'windows_x64_scope_required')
        receipt=data['native_receipt'];package=packages[0]
        from .chunks import chunk_entries, prepare_chunks
        chunk_entries(package['artifact'], request['buildId'])
        expected={'source_commit':request['commit'],'controller_commit':request.get('controllerCommit'),
                  'build_id':request['buildId'],'repository':record['repository'],
                  'platform':'win32','arch':'x64','identity':package['identity'],
                  'publisher':package['publisher'],'application_id':self.identity['appNamePascal'],
                  'windows_version':request['windowsVersion'],'sha256':package['artifact']['sha256'],
                  'size':package['artifact']['size'],'notes_sha256':digest(data['notes']),'signature_result':'PASS','qualification':'windows-x64-native'}
        if package['artifact'].get('chunks') is not None:
            expected['chunk_descriptor_sha256'] = digest(package['artifact']['chunks'])
        archive=package.get('distributionArchive')
        if archive is not None:
            if (not isinstance(archive,dict) or set(archive)!={'key','sha256','size'}
                or not isinstance(archive['key'],str) or not archive['key'].startswith(build_prefix(request['buildId']))
                or not archive['key'].endswith('.zip') or not isinstance(archive['sha256'],str)
                or len(archive['sha256'])!=64 or any(c not in '0123456789abcdef' for c in archive['sha256'])
                or type(archive['size']) is not int or archive['size']<=0):
                raise OperationError(422,'distribution_archive_contract_mismatch')
            expected['distribution_archive']=archive
        if (not self.receipt_key or not hmac.compare_digest(hmac.new(self.receipt_key,canonical(receipt),hashlib.sha256).hexdigest(),data['receipt_hmac'])
            or any(receipt.get(k)!=v for k,v in expected.items()) or not receipt.get('workflow_run_id')
            or receipt.get('workflow_result')!='native_gates_passed'
            or not str(receipt.get('package_family','')).startswith(package['identity']+'_')):
            raise OperationError(422,'trusted_native_receipt_required')
        from .ci import verify_workflow
        ci_verification=(self.workflow_verifier or verify_workflow)(receipt)
        data['ci_verification']=ci_verification
        # Receipt-signer is CI-only. Operators cannot turn uploads into qualification.
        artifact=package['artifact'];artifacts=[artifact]+([archive] if archive else []);keys=[a['key'] for a in artifacts]+[package['feed']['key']]
        prefix=build_prefix(request['buildId']);manifest_key=prefix+'build.json'
        for key in keys:
            source = safe_file(self.incoming, key)
            destination = safe_file(self.root, key)
            bound = next((item for item in artifacts if item['key'] == key), None)
            try:
                if bound:
                    immutable_copy(source, destination, bound)
                else:
                    if source.stat().st_size > 1024 * 1024:
                        raise OperationError(422, 'native_descriptor_mismatch')
                    body = source.read_bytes()
                    self.verify_descriptor(body, package, request)
                    immutable_write(destination, body)
            except OSError as error:
                raise OperationError(422, 'staged_artifact_unavailable') from error
        if artifact.get('chunks') is not None:
            prepared = prepare_chunks(safe_file(self.root, artifact['key']), self.root, request['buildId'], artifact)
            if prepared != artifact['chunks']:
                raise OperationError(422, 'chunk_integrity_mismatch')
        body=canonical(manifest);target=safe_file(self.root,manifest_key)
        if target.exists() and target.read_bytes()!=body:raise OperationError(409,'immutable_manifest_conflict')
        immutable_write(target,body)
        data['manifest_key']=manifest_key;data['manifest_sha256']=hashlib.sha256(body).hexdigest()
        data['received_at']=utcnow();data['qualification']='verified'
        if not persist:
            return data
        with self.store.connection() as c:
            old=c.execute('SELECT body_hash,body FROM releases WHERE release_id=%s OR build_id=%s FOR UPDATE',(data['release_id'],request['buildId'])).fetchone()
            input_hash=digest(candidate.model_dump())
            if old:
                if old['body_hash']!=input_hash:raise OperationError(409,'release_idempotency_conflict')
                return old['body']
            c.execute('INSERT INTO releases(release_id,build_id,body,body_hash,qualification) VALUES(%s,%s,%s::jsonb,%s,%s)',(data['release_id'],request['buildId'],json.dumps(data),input_hash,'verified'))
            c.execute('INSERT INTO channels(channel,record) VALUES(%s,%s::jsonb) ON CONFLICT DO NOTHING',(self.channel,json.dumps(record)))
            c.execute('INSERT INTO delivery_heads(target_id,channel,record) VALUES(%s,%s,%s::jsonb) ON CONFLICT DO NOTHING', (self.target_id,self.channel,json.dumps(record)))
            self.store.audit(c,principal['principal_id'],'release_ingest',data['release_id'],{'build_id':request['buildId'],'scope':'windows-x64-native'})
        return data

    def read_record(self):
        if self.remote_target:
            raw, _ = self.remote_target.read('releases/channels/'+self.channel+'.json')
            return json.loads(raw) if raw is not None else None
        target = safe_file(self.root, 'releases/channels/'+self.channel+'.json')
        if not target.exists():
            return None
        if target.stat().st_size > 4 * 1024 * 1024:
            raise OperationError(503, 'channel_metadata_too_large')
        return json.loads(target.read_bytes())

    def verify_release(self, release):
        manifest = release['manifest']
        if digest(manifest) != release['manifest_sha256']:
            raise OperationError(422, 'manifest_integrity_mismatch')
        from .chunks import chunk_entries
        for package in manifest['packages']:
            for chunk in chunk_entries(package['artifact'], manifest['request']['buildId']):
                if file_facts(safe_file(self.root, chunk['key'])) != (chunk['size'], chunk['sha256']):
                    raise OperationError(422, 'chunk_integrity_mismatch')
            descriptor = safe_file(self.root, package['feed']['key'])
            if descriptor.stat().st_size > 1024 * 1024:
                raise OperationError(422, 'native_descriptor_mismatch')
            self.verify_descriptor(descriptor.read_bytes(), package, manifest['request'])
            for artifact in [package['artifact']]+([package['distributionArchive']] if package.get('distributionArchive') else []):
                if file_facts(safe_file(self.root, artifact['key'])) != (artifact['size'], artifact['sha256']):
                    raise OperationError(422, 'artifact_integrity_mismatch')
        return manifest

    def preflight(self, release, *, remote_reservation=None):
        self.verify_release(release)
        if self.remote_target:
            return self.remote_target.preflight(release, self.root, reservation=remote_reservation)
        return None

    def intended_record(self, release, previous):
        from hermes_cli.release_channels import validate_record
        build = release['manifest']['request']
        record = {**previous, 'revision': previous['revision']+1,
                  'nextSequence': max(previous['nextSequence'], build['sequence']+1),
                  'head': {'buildId': build['buildId'], 'sequence': build['sequence'],
                           'manifestKey': release['manifest_key'], 'sha256': release['manifest_sha256']}}
        try:
            validate_record(record)
        except ValueError as error:
            raise OperationError(422, 'channel_contract_mismatch') from error
        return record

    def publish_variant(self, release, operation_id, *, remote_reservation=None, authorize):
        """Publish one reserved target; actual head readback precedes DB projection.

        A crash between these boundaries is reconciled against the captured exact
        intended head, never by allocating another revision or another operation.
        """
        with self.store.connection() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',
                      (self.store.schema+':delivery:'+self.target_id+':'+self.channel,))
            target = c.execute('SELECT t.*,p.user_id,p.release_id,v.first_admitted_at FROM publication_targets t '
                               'JOIN publications p USING(publication_id) JOIN release_delivery_variants v ON v.release_id=p.release_id '
                               'AND v.target_id=t.target_id AND v.variant_hash=t.variant_hash WHERE operation_id=%s FOR UPDATE OF t',
                               (operation_id,)).fetchone()
            if not target or target['target_id'] != self.target_id or target['channel'] != self.channel:
                raise OperationError(404, 'publication_target_unavailable')
            if target['status'] in {'published', 'already_current'}:
                return {'revision': target['actual_revision'], 'readback_digest': target['readback_digest'],
                        'maintenance_status': target['maintenance_status']}
            if not target['reserved'] or not target['previous_head'] or not target['intended_head']:
                raise OperationError(409, 'publication_reservation_required')
            previous, intended = target['previous_head'], target['intended_head']
            if self.intended_record(release, previous) != intended:
                raise OperationError(409, 'publication_intent_changed')
            head = c.execute('SELECT * FROM delivery_heads WHERE target_id=%s AND channel=%s FOR UPDATE',
                             (self.target_id, self.channel)).fetchone()
            if not head or head['record'] not in (previous, intended):
                raise OperationError(409, 'channel_revision_conflict')
            self.verify_release(release)
            actual = self.read_record()
            if actual not in (previous, intended):
                raise OperationError(503, 'publication_reconciliation_required')
            # Authorization may have been revoked since the captured request.
            authorize()
            maintenance = 'none'
            if self.remote_target:
                bound_release = {**release, 'retention_admission': {'release_id': target['release_id'],
                                  'first_admitted_at': target['first_admitted_at'].isoformat()}}
                outcome = self.remote_target.promote(bound_release, self.root, previous, intended,
                                                     reservation=remote_reservation, authorize=authorize)
                maintenance = (outcome or {}).get('maintenance_status', 'none')
            elif actual != intended:
                atomic_write(safe_file(self.root, 'releases/channels/'+self.channel+'.json'), canonical(intended))
            if self.read_record() != intended:
                raise OperationError(503, 'publication_readback_failed')
            if head['record'] != intended:
                c.execute('UPDATE delivery_heads SET revision=%s,release_id=%s,record=%s::jsonb '
                          'WHERE target_id=%s AND channel=%s',
                          (intended['revision'], target['release_id'], json.dumps(intended), self.target_id, self.channel))
                self.store.audit(c, target['user_id'], 'target_publication', target['release_id'],
                                 {'operation_id': operation_id, 'target_id': self.target_id,
                                  'revision': intended['revision']})
            return {'revision': intended['revision'], 'readback_digest': digest(intended),
                    'maintenance_status': maintenance}

    def record_rejection(self,candidate,code,principal):
        """Keep failed candidates inspectable without granting publication eligibility."""
        from contracts.client_operations import sanitize
        supplied=candidate.model_dump();request=supplied['manifest'].get('request',{})
        text=lambda value: sanitize(str(value)[:16000])[0]
        body={'release_id':candidate.release_id,'manifest':{'request':{'buildId':candidate.release_id,'version':text(request.get('version','Unavailable')),'windowsVersion':text(request.get('windowsVersion','Unavailable')),'commit':text(request.get('commit','Unavailable')),'controllerCommit':text(request.get('controllerCommit','Unavailable'))},'packages':[]},'notes':{k:text(v) for k,v in supplied['notes'].items()},'native_receipt':{'failure_code':code},'qualification':'rejected'}
        with self.store.connection() as c:
            inserted=c.execute("INSERT INTO releases(release_id,build_id,body,body_hash,qualification) VALUES(%s,%s,%s::jsonb,%s,'rejected') ON CONFLICT DO NOTHING RETURNING release_id",(candidate.release_id,candidate.release_id,json.dumps(body),digest(supplied))).fetchone()
            if inserted:self.store.audit(c,principal,'candidate_refused',candidate.release_id,{'code':code})

    def verify_descriptor(self,body,package,request):
        import xml.etree.ElementTree as ET
        if len(body)>1024*1024 or b'<!DOCTYPE' in body.upper() or b'<!ENTITY' in body.upper():
            raise OperationError(422,'native_descriptor_mismatch')
        try:
            root=ET.fromstring(body)
            main=[e for e in root if e.tag.rsplit('}',1)[-1]=='MainPackage']
            expected={'Name':package['identity'],'Publisher':package['publisher'],
                      'Version':package['version'],'ProcessorArchitecture':'x64',
                      'Uri':self.base+'/'+package['artifact']['key']}
            if (root.tag.rsplit('}',1)[-1]!='AppInstaller' or root.get('Uri')!=self.base+'/'+package['feed']['key']
                or root.get('Version')!=package['version'] or len(main)!=1
                or any(main[0].get(k)!=v for k,v in expected.items())):
                raise ValueError('descriptor differs from admitted package')
        except (ET.ParseError,ValueError) as exc:raise OperationError(422,'native_descriptor_mismatch') from exc

    def promote(self,user_id,body):
        request=body.model_dump()
        with self.store.connection() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(self.store.schema+':'+self.channel,))
            if c.execute('SELECT 1 FROM publication_targets WHERE channel=%s AND target_id=%s AND reserved LIMIT 1',(self.channel,self.target_id)).fetchone():
                raise OperationError(409,'delivery_target_busy')
            old=c.execute('SELECT * FROM operations WHERE user_id=%s AND idempotency_key=%s',(user_id,body.idempotency_key)).fetchone()
            if old:
                if old['request_hash']!=digest(request):raise OperationError(409,'operation_idempotency_conflict')
                operation_id=old['operation_id']
            else:
                operation_id=uuid4().hex
                c.execute('INSERT INTO operations(operation_id,user_id,channel,idempotency_key,request_hash,body,status) VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s)',(operation_id,user_id,self.channel,body.idempotency_key,digest(request),json.dumps(request),'preparing'))
        return self.reconcile(operation_id)

    def reconcile(self,operation_id):
        try:return self._reconcile(operation_id)
        except OperationError as error:
            # Known refusals are final. Actual I/O/crash uncertainty stays preparing
            # so readback can reconcile without another promotion.
            if 400<=error.status<500:
                with self.store.connection() as c:
                    c.execute("UPDATE operations SET status='failed',body=body || %s::jsonb,updated_at=now() WHERE operation_id=%s AND status='preparing'",(json.dumps({'error_code':error.code}),operation_id))
            raise

    def _reconcile(self,operation_id):
        from agents.webapi.deps import ensure_repo_hermes_agent_on_path
        ensure_repo_hermes_agent_on_path()
        from hermes_cli.release_channels import build_prefix,validate_record
        with self.store.connection() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(self.store.schema+':'+self.channel,))
            op=c.execute('SELECT * FROM operations WHERE operation_id=%s FOR UPDATE',(operation_id,)).fetchone()
            if not op:raise OperationError(404,'operation_unavailable')
            if op['status']!='preparing':return op
            request=op['body'];channel=c.execute('SELECT * FROM channels WHERE channel=%s FOR UPDATE',(self.channel,)).fetchone()
            head=c.execute('SELECT * FROM delivery_heads WHERE target_id=%s AND channel=%s FOR UPDATE',(self.target_id,self.channel)).fetchone()
            if not head:raise OperationError(503,'delivery_head_unavailable')
            if channel['paused']:raise OperationError(409,'promotions_paused')
            if channel['revision']!=request['expected_revision']:raise OperationError(409,'channel_revision_conflict')
            release=c.execute("SELECT body FROM releases WHERE release_id=%s AND qualification='verified'",(request['release_id'],)).fetchone()
            if not release:raise OperationError(422,'verified_release_required')
            release=release['body'];manifest=release['manifest'];build=manifest['request']
            # Retain historical admissions; publish their exact manifest at the
            # canonical native-consumer key, including pre-repair admissions.
            manifest_key=build_prefix(build['buildId'])+'build.json'
            manifest_body=canonical(manifest)
            if hashlib.sha256(manifest_body).hexdigest()!=release['manifest_sha256']:
                raise OperationError(422,'manifest_integrity_mismatch')
            manifest_target=safe_file(self.root,manifest_key)
            immutable_write(manifest_target,manifest_body)
            if manifest_target.read_bytes()!=manifest_body:
                raise OperationError(503,'manifest_readback_failed')
            record={**head['record'],'revision':head['revision']+1,'nextSequence':max(head['record']['nextSequence'],build['sequence']+1),
                    'head':{'buildId':build['buildId'],'sequence':build['sequence'],'manifestKey':manifest_key,'sha256':release['manifest_sha256']}}
            try:validate_record(record)
            except ValueError as exc:raise OperationError(422,'channel_contract_mismatch') from exc
            # Recheck chunks, native descriptors and pinned bytes before the head.
            self.verify_release(release)
            if self.remote_target:
                self.remote_target.promote({**release,'manifest_key':manifest_key},self.root,head['record'],record)
            else:
                target=safe_file(self.root,'releases/channels/'+self.channel+'.json')
                actual=json.loads(target.read_bytes()) if target.exists() else None
                if actual not in (None,head['record'],record):raise OperationError(503,'publication_reconciliation_required')
                if actual!=record:atomic_write(target,canonical(record))
                if json.loads(target.read_bytes())!=record:raise OperationError(503,'publication_readback_failed')
            c.execute('UPDATE delivery_heads SET revision=%s,release_id=%s,record=%s::jsonb WHERE target_id=%s AND channel=%s',(record['revision'],request['release_id'],json.dumps(record),self.target_id,self.channel))
            c.execute('UPDATE channels SET revision=revision+1 WHERE channel=%s',(self.channel,))
            c.execute("UPDATE operations SET status='published',updated_at=now() WHERE operation_id=%s",(operation_id,))
            self.store.audit(c,op['user_id'],'promotion',request['release_id'],{'operation_id':operation_id,'expected_revision':request['expected_revision'],'actual_revision':record['revision']})
            return {**op,'status':'published'}
