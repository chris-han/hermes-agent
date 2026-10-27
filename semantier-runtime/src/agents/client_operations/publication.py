"""One x64 publisher. Upstream full-matrix admission remains unchanged."""
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from uuid import uuid4
from contracts.client_operations import canonical,digest,utcnow


class OperationError(ValueError):
    def __init__(self,status,code):self.status,self.code=status,code;super().__init__(code)


def safe_file(root,key):
    relative=PurePosixPath(key)
    if relative.is_absolute() or any(p in {'','..','.'} for p in relative.parts) or '\\' in key:
        raise OperationError(422,'invalid_artifact_key')
    target=root.joinpath(*relative.parts)
    if target.is_symlink() or any(p.is_symlink() for p in target.parents if p!=root.parent):
        raise OperationError(422,'artifact_symlink_refused')
    if not target.resolve().is_relative_to(root.resolve()):raise OperationError(422,'artifact_outside_root')
    return target


def atomic_write(target,body):
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.publishing-',dir=target.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
        os.replace(name,target)
        directory=os.open(target.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(name):os.unlink(name)


def immutable_write(target,body):
    """Publish once with an atomic no-replace link; concurrent writers cannot win silently."""
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.admitting-',dir=target.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
        try:os.link(name,target)
        except FileExistsError:
            if target.read_bytes()!=body:raise OperationError(409,'immutable_artifact_conflict')
        directory=os.open(target.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(name):os.unlink(name)


class Publisher:
    def __init__(self,store,public_root,incoming_root,public_base,receipt_key,identity,channel,workflow_verifier=None,remote_target=None):
        self.store,self.root,self.incoming=store,Path(public_root),Path(incoming_root)
        self.base,self.receipt_key,self.identity,self.channel=public_base,receipt_key,identity,channel
        self.workflow_verifier=workflow_verifier
        self.remote_target=remote_target

    def empty_record(self):
        return {'schema':1,'name':self.channel,'repository':'chris-han/hermes-agent','policy':'stable-release','revision':1,'nextSequence':1,'identity':self.identity,'head':None,'state':'active'}

    def initialize_channel(self):
        """Expose an explicit configured empty channel; never promote on startup."""
        record=self.empty_record()
        with self.store.connection() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(self.store.schema+':'+self.channel,))
            existing=c.execute('SELECT record FROM channels WHERE channel=%s',(self.channel,)).fetchone()
            if existing:
                if self.remote_target:self.remote_target.initialize(existing['record'])
                return
            target=safe_file(self.root,'releases/channels/'+self.channel+'.json')
            if target.exists() and json.loads(target.read_bytes())!=record:raise OperationError(503,'channel_provenance_mismatch')
            immutable_write(target,canonical(record))
            c.execute('INSERT INTO channels(channel,record) VALUES(%s,%s::jsonb)',(self.channel,json.dumps(record)))
            self.store.audit(c,'configured-owner','channel_initialized',self.channel,{'head':None,'identity':self.identity})
            if self.remote_target:self.remote_target.initialize(record)

    def admit(self,candidate,principal):
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
        expected={'source_commit':request['commit'],'controller_commit':request.get('controllerCommit'),
                  'build_id':request['buildId'],'repository':record['repository'],
                  'platform':'win32','arch':'x64','identity':package['identity'],
                  'publisher':package['publisher'],'application_id':self.identity['appNamePascal'],
                  'windows_version':request['windowsVersion'],'sha256':package['artifact']['sha256'],
                  'size':package['artifact']['size'],'notes_sha256':digest(data['notes']),'signature_result':'PASS','qualification':'windows-x64-native'}
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
            source=safe_file(self.incoming,key)
            try:body=source.read_bytes()
            except OSError as error:raise OperationError(422,'staged_artifact_unavailable') from error
            if key==package['feed']['key']:self.verify_descriptor(body,package,request)
            bound=next((a for a in artifacts if a['key']==key),None)
            if bound and (len(body)!=bound['size'] or hashlib.sha256(body).hexdigest()!=bound['sha256']):
                raise OperationError(422,'artifact_integrity_mismatch')
            destination=safe_file(self.root,key)
            if destination.exists() and destination.read_bytes()!=body:raise OperationError(409,'immutable_artifact_conflict')
            immutable_write(destination,body)
            if destination.read_bytes()!=body:raise OperationError(503,'artifact_readback_failed')
        body=canonical(manifest);target=safe_file(self.root,manifest_key)
        if target.exists() and target.read_bytes()!=body:raise OperationError(409,'immutable_manifest_conflict')
        immutable_write(target,body)
        data['manifest_key']=manifest_key;data['manifest_sha256']=hashlib.sha256(body).hexdigest()
        data['received_at']=utcnow();data['qualification']='verified'
        with self.store.connection() as c:
            old=c.execute('SELECT body_hash,body FROM releases WHERE release_id=%s OR build_id=%s FOR UPDATE',(data['release_id'],request['buildId'])).fetchone()
            input_hash=digest(candidate.model_dump())
            if old:
                if old['body_hash']!=input_hash:raise OperationError(409,'release_idempotency_conflict')
                return old['body']
            c.execute('INSERT INTO releases(release_id,build_id,body,body_hash,qualification) VALUES(%s,%s,%s::jsonb,%s,%s)',(data['release_id'],request['buildId'],json.dumps(data),input_hash,'verified'))
            c.execute('INSERT INTO channels(channel,record) VALUES(%s,%s::jsonb) ON CONFLICT DO NOTHING',(self.channel,json.dumps(record)))
            self.store.audit(c,principal['principal_id'],'release_ingest',data['release_id'],{'build_id':request['buildId'],'scope':'windows-x64-native'})
        return data

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
            record={**channel['record'],'revision':channel['revision']+1,'nextSequence':max(channel['record']['nextSequence'],build['sequence']+1),
                    'head':{'buildId':build['buildId'],'sequence':build['sequence'],'manifestKey':manifest_key,'sha256':release['manifest_sha256']}}
            try:validate_record(record)
            except ValueError as exc:raise OperationError(422,'channel_contract_mismatch') from exc
            # Verify bytes again immediately before head publication.
            for package in manifest['packages']:
                self.verify_descriptor(safe_file(self.root,package['feed']['key']).read_bytes(),package,build)
                for artifact in [package['artifact']]+([package['distributionArchive']] if package.get('distributionArchive') else []):
                    raw=safe_file(self.root,artifact['key']).read_bytes()
                    if len(raw)!=artifact['size'] or hashlib.sha256(raw).hexdigest()!=artifact['sha256']:raise OperationError(422,'artifact_integrity_mismatch')
            if self.remote_target:
                self.remote_target.promote({**release,'manifest_key':manifest_key},self.root,channel['record'],record)
            else:
                target=safe_file(self.root,'releases/channels/'+self.channel+'.json')
                actual=json.loads(target.read_bytes()) if target.exists() else None
                if actual not in (None,channel['record'],record):raise OperationError(503,'publication_reconciliation_required')
                if actual!=record:atomic_write(target,canonical(record))
                if json.loads(target.read_bytes())!=record:raise OperationError(503,'publication_readback_failed')
            c.execute('UPDATE channels SET revision=%s,release_id=%s,record=%s::jsonb WHERE channel=%s',(record['revision'],request['release_id'],json.dumps(record),self.channel))
            c.execute("UPDATE operations SET status='published',updated_at=now() WHERE operation_id=%s",(operation_id,))
            self.store.audit(c,op['user_id'],'promotion',request['release_id'],{'operation_id':operation_id,'expected_revision':request['expected_revision'],'actual_revision':record['revision']})
            return {**op,'status':'published'}
