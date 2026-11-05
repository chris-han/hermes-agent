"""Native channel active/archive transition, independent of Runtime and databases."""
from copy import deepcopy


def advance(record, new_head, *, allow_sequence_gap=False):
    """Return a new record; callers must commit it with the observed ETag."""
    from hermes_cli.release_channels import _head
    if type(allow_sequence_gap) is not bool:
        raise ValueError('Invalid sequence gap policy')
    if record.get('head') is not None:
        raise ValueError('legacy channel head requires separate retirement')
    old = _head(record.get('nativeHead'))
    _head(new_head)
    archive = _head(record.get('nativeArchive'))
    if archive is not None and (old is None or archive['buildId'] == old['buildId']
                                or archive['sequence'] == old['sequence']
                                or archive['sequence'] >= record['nextSequence']):
        raise ValueError('invalid native archive')
    if old == new_head:
        return deepcopy(record)
    if record.get('nativeRetirePending') is not None:
        raise ValueError('previous archive cleanup pending')
    if (type(new_head.get('sequence')) is not int
            or (new_head['sequence'] > record['nextSequence'] and not allow_sequence_gap)
            or (old is not None and new_head['sequence'] <= old['sequence'])
            or (archive is not None and new_head['sequence'] <= archive['sequence'])):
        raise ValueError('native release sequence is stale or unallocated')
    result = deepcopy(record)
    result['nativeRetirePending'] = deepcopy(record.get('nativeArchive'))
    result['nativeArchive'] = deepcopy(old)
    result['nativeHead'] = deepcopy(new_head)
    result['revision'] += 1
    if new_head['sequence'] >= record['nextSequence']:
        result['nextSequence'] = new_head['sequence'] + 1
    return result


def validate_solo_allocation(intent, allocation, manifest):
    """Admit only the existing source-bound Windows allocator's reserved version."""
    import hashlib
    import json
    import re
    from scripts.releases.semantier_windows import is_acceptance_request
    if not isinstance(intent, dict) or not isinstance(allocation, dict) or not isinstance(manifest, dict):
        raise ValueError('Invalid solo allocation evidence')
    source = intent.get('source')
    run_id = intent.get('runId')
    request = manifest.get('request')
    try:
        acceptance = isinstance(request, dict) and is_acceptance_request(request)
    except (TypeError, ValueError):
        acceptance = False
    if (not isinstance(source, dict) or set(source) != {'parentCommit', 'hermesCommit', 'productReceiptSha256'}
            or not isinstance(run_id, str) or not re.fullmatch('[a-f0-9]{32}', run_id)
            or any(not isinstance(source.get(key), str) or not re.fullmatch('[a-f0-9]{40}', source[key])
                   for key in ('parentCommit', 'hermesCommit'))
            or not isinstance(source.get('productReceiptSha256'), str)
            or not re.fullmatch('[a-f0-9]{64}', source['productReceiptSha256'])
            or not acceptance):
        raise ValueError('Invalid solo allocation source or request')
    binding = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    version = allocation.get('version')
    if (allocation.get('kind') != 'semantier-windows-build-version-v1'
            or allocation.get('reservationId') != run_id or allocation.get('sourceBinding') != binding
            or allocation.get('sourceCommit') != source['hermesCommit']
            or request.get('commit') != source['hermesCommit']
            or request.get('controllerCommit') != source['parentCommit']
            or request.get('publicBase') != 'https://download.semantier.com'
            or version != request.get('version') or request.get('windowsVersion') != version + '.0'
            or request.get('sequence') != int(version.split('.')[2])):
        raise ValueError('Solo allocation does not match the signed release source and version')
    return True


def verify_solo_source(intent):
    """Rebind the selected parent gitlink and committed product receipt."""
    import hashlib
    import re
    import subprocess
    from pathlib import Path
    source = intent['source']
    hermes = Path(__file__).resolve().parents[2]
    parent = hermes.parent
    entry = subprocess.check_output(['/usr/bin/git', '-C', str(parent), 'ls-tree',
        source['parentCommit'], '--', 'hermes-agent'], text=True).strip()
    if entry != '160000 commit '+source['hermesCommit']+'\thermes-agent':
        raise ValueError('Solo parent does not pin the accepted Hermes commit')
    body = subprocess.check_output(['/usr/bin/git', '-C', str(hermes), 'show',
        source['hermesCommit']+':semantier-product/source-receipt.json'])
    if hashlib.sha256(body).hexdigest() != source['productReceiptSha256']:
        raise ValueError('Solo committed product receipt digest differs')
    tree = subprocess.check_output(['/usr/bin/git', '-C', str(hermes), 'rev-parse',
        source['hermesCommit']+'^{tree}'], text=True).strip()
    if not re.fullmatch('[a-f0-9]{40}', tree) or tree != intent.get('hermesTree'):
        raise ValueError('Solo Hermes tree differs from frozen intent')


def load_solo_allocation(output, intent_path, manifest, *, work_root=None, ledger_path=None):
    """Read only the run-owned intent and the allocator receipt from its build."""
    import re
    from pathlib import Path
    from hermes_cli.release_channels import decode_json
    root = Path(work_root) if work_root is not None else Path('/mnt/d/repo/deliverables/work')
    output, intent_path = Path(output), Path(intent_path)
    if (intent_path.name != 'intent.json' or intent_path.parent.parent != root/'windows-solo-release'
            or not re.fullmatch('[a-f0-9]{32}', intent_path.parent.name)
            or output.parent != root or not re.fullmatch(
                r'semantier-solo-'+intent_path.parent.name+r'-build-[a-f0-9]{32}', output.name)):
        raise ValueError('Solo allocation paths do not belong to one release run')
    allocation_path = output/'build-version.json'
    result_path = intent_path.with_name('result.json')
    ledger_path = Path(ledger_path) if ledger_path is not None else Path('/mnt/d/repo/semantier-agent-build-version.json')
    for path in (intent_path, allocation_path, result_path, ledger_path):
        if path.is_symlink() or path.resolve() != path or not path.is_file() or path.stat().st_size > 4*1024*1024:
            raise ValueError('Unsafe solo allocation evidence path')
    intent = decode_json(intent_path.read_bytes())
    allocation = decode_json(allocation_path.read_bytes())
    receipt = decode_json(result_path.read_bytes())
    ledger = decode_json(ledger_path.read_bytes())
    if intent.get('runId') != intent_path.parent.name:
        raise ValueError('Solo intent path and run ID differ')
    validate_solo_allocation(intent, allocation, manifest)
    request = manifest['request']
    packages = manifest.get('packages')
    if (receipt.get('runId') != intent['runId'] or receipt.get('source') != intent['source']
            or receipt.get('version') != allocation['version'] or receipt.get('buildId') != request['buildId']
            or receipt.get('build') != 'SIGNED_PACKAGE_VERIFIED' or receipt.get('installedAcceptance') != 'PASS'
            or not isinstance(receipt.get('acceptance'), dict)
            or receipt['acceptance'].get('runId') != intent['runId']
            or receipt['acceptance'].get('buildId') != request['buildId']
            or receipt['acceptance'].get('artifactSha256') != receipt.get('signedArtifactSha256')
            or not isinstance(packages, list) or len(packages) != 1
            or receipt.get('signedArtifactSha256') != packages[0].get('artifact', {}).get('sha256')):
        raise ValueError('Solo result does not bind the allocated signed release')
    if ledger.get('kind') != 'semantier-windows-build-version-v1' or not re.fullmatch(
            r'(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}', str(ledger.get('version', ''))):
        raise ValueError('Current Windows version ledger is invalid')
    current = tuple(map(int, ledger['version'].split('.')))
    reserved = tuple(map(int, allocation['version'].split('.')))
    if any(part > 65535 for part in (*current, *reserved)) or current < reserved or (current == reserved and any(ledger.get(key) != allocation.get(key)
            for key in ('version', 'sourceCommit', 'reservationId', 'sourceBinding'))):
        raise ValueError('Current version ledger conflicts with the solo allocation')
    verify_solo_source(intent)
    return True


def rollback(record):
    """Calculate a slot swap; publication and object verification remain separate."""
    from hermes_cli.release_channels import _head
    if record.get('head') is not None or record.get('nativeRetirePending') is not None:
        raise ValueError('channel has unresolved legacy or retention state')
    active = _head(record.get('nativeHead'))
    archive = _head(record.get('nativeArchive'))
    if (active is None or archive is None or active['buildId'] == archive['buildId']
            or active['sequence'] == archive['sequence']
            or active['sequence'] >= record['nextSequence'] or archive['sequence'] >= record['nextSequence']):
        raise ValueError('rollback requires two distinct allocated slots')
    result = deepcopy(record)
    result['nativeHead'], result['nativeArchive'] = deepcopy(archive), deepcopy(active)
    result['revision'] += 1
    return result


def commit_slot(store, key, expected_etag, new_head, *, allow_sequence_gap=False):
    """Atomically change both slots; an ETag conflict never overwrites a newer head."""
    from hermes_cli.release_channels import canonical_json, decode_json
    found = store.get(key)
    if found is None or found[1] != expected_etag:
        raise ValueError('channel changed since verification')
    record = decode_json(found[0])
    result = advance(record, new_head, allow_sequence_gap=allow_sequence_gap)
    if result != record:
        store.put(key, canonical_json(result), expected_etag)
    return result


def commit_rollback(store, key, expected_record, expected_etag):
    """CAS a previously verified swap; an exact completed swap is replay-safe.

    The caller must hold the publication lock and verify both retained release
    objects before calling this primitive. It does not verify Windows downgrade.
    """
    from hermes_cli.release_channels import canonical_json, decode_json
    desired = rollback(expected_record)
    found = store.get(key)
    if found is None:
        raise ValueError('channel changed since verification')
    actual = decode_json(found[0])
    if actual == desired:
        return actual
    if actual != expected_record or found[1] != expected_etag:
        raise ValueError('channel changed since verification')
    store.put(key, canonical_json(desired), expected_etag)
    confirmed = store.get(key)
    if confirmed is None or decode_json(confirmed[0]) != desired:
        raise RuntimeError('rollback readback failed')
    return desired


def slot_view(channel, record, slot):
    """Describe a slot; the channel record remains the only authority."""
    from hermes_cli.release_channels import _head, channel_key
    if slot not in ('active', 'archive') or type(record.get('revision')) is not int:
        raise ValueError('invalid slot or channel revision')
    return {'schema': 1, 'channel': channel, 'channelKey': channel_key(channel),
            'revision': record['revision'], 'slot': slot,
            'head': deepcopy(_head(record.get('nativeHead' if slot == 'active' else 'nativeArchive')))}


def validate_slot_view(view, channel, record, slot):
    """Reject a view unless it exactly matches a freshly fetched channel record."""
    expected = slot_view(channel, record, slot)
    if view != expected:
        raise ValueError('stale or invalid slot view')
    return deepcopy(expected['head'])


def reconcile_slot_views(store, channel, record):
    """Repair derived views with CAS. Caller must hold and verify the publication lock.

    Each view is advisory until a reader validates it against a fresh record.
    Retry the entire operation after interruption while holding the lock.
    """
    from hermes_cli.release_channels import canonical_json
    for slot in ('active', 'archive'):
        key = slot + '/manifest.json'
        body = canonical_json(slot_view(channel, record, slot))
        current = store.get(key)
        if current is None or current[0] != body:
            store.put(key, body, None if current is None else current[1])
        confirmed = store.get(key)
        if confirmed is None or confirmed[0] != body:
            raise RuntimeError('slot view readback failed')


def publication_lock(store):
    """Share the existing storage lock; never clear another publisher's lock."""
    from contextlib import contextmanager
    from uuid import uuid4
    from hermes_cli.release_channels import canonical_json

    @contextmanager
    def held():
        key = '_control/publication.lock'
        body = canonical_json({'id': uuid4().hex, 'owner': {'kind': 'simple-native-publisher'}})
        store.put(key, body)
        current = store.get(key)
        if current is None or current[0] != body:
            raise RuntimeError('publication lock readback failed')
        try:
            yield body
        finally:
            current = store.get(key)
            if current is not None and current[0] == body:
                store.delete(key, current[1])
    return held()


class ControlStore:
    """The private R2 lock file shared with the existing publisher."""
    def __init__(self, transport, credentials, base, bucket, scope=None):
        import re
        if not isinstance(bucket, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', bucket):
            raise ValueError('invalid control bucket')
        self.transport = transport
        self.credentials = credentials
        self.base = base.rstrip('/')
        self.bucket = bucket
        self.scope = scope

    def _url(self, key):
        if key not in ('_control/publication.lock', '_control/retention.json'):
            raise ValueError('unsupported control key')
        physical_key = self.scope.key(key) if self.scope is not None else key
        return f'{self.base}/{self.bucket}/{physical_key}'

    def get(self, key):
        try:
            result = self.transport.signed_request('GET', self._url(key), creds=self.credentials,
                                                   now=self.transport.amz_timestamp(), tries=1)
        except self.transport.R2RequestError as error:
            if error.status == 404:
                return None
            raise
        if len(result._body) > 4 * 1024 * 1024 or not result.header('etag'):
            raise RuntimeError('invalid publication lock')
        return result._body, result.header('etag')

    def put(self, key, body, etag=None):
        import hashlib
        if key != '_control/publication.lock' or etag is not None:
            raise ValueError('lock must be newly acquired')
        try:
            self.transport.signed_request('PUT', self._url(key), body=body,
                body_hash=hashlib.sha256(body).hexdigest(), content_length=len(body),
                content_type='application/json', creds=self.credentials,
                now=self.transport.amz_timestamp(), extra_headers={'If-None-Match': '*'}, tries=1)
        except (OSError, self.transport.R2RequestError):
            current = self.get(key)
            if current is None or current[0] != body:
                raise

    def delete(self, key, etag):
        if key != '_control/publication.lock':
            raise ValueError('only the acquired lock may be removed')
        self.transport.signed_request('DELETE', self._url(key), creds=self.credentials,
            now=self.transport.amz_timestamp(), extra_headers={'If-Match': etag}, tries=1)


def native_manifest(request, signed, graph_bytes):
    """Build metadata from a graph whose package objects were signed."""
    import hashlib
    from hermes_cli.release_channels import decode_json, validate_native_graph
    graph = decode_json(graph_bytes)
    identity = request['identity']['msixAppIdWithOrg']
    if (signed.get('signing') != 'PASS' or not signed.get('verification', '').startswith('PASS:')
            or signed.get('sourceCommit') != request['commit']
            or signed.get('packageName') != identity or signed.get('packageVersion') != request['windowsVersion']):
        raise ValueError('signed candidate identity mismatch')
    validate_native_graph(graph, build_id=request['buildId'], identity=identity,
                          publisher=signed['publisher'], version=request['windowsVersion'], arch='x64')
    if graph != signed.get('nativeGraph'):
        raise ValueError('native graph differs from signed receipt')
    root = next(row for row in graph['objects'] if row['key'] == graph['root'])
    if root['sha256'] != signed['signedSha256']:
        raise ValueError('native root differs from signed receipt')
    graph_key = graph['root'].rsplit('/', 1)[0] + '/native-graph.json'
    descriptor = {'key': graph_key, 'size': len(graph_bytes), 'sha256': hashlib.sha256(graph_bytes).hexdigest()}
    package = {'platform': 'win32', 'arch': 'x64', 'variant': 'native-msix-assets',
        'identity': identity, 'publisher': signed['publisher'], 'version': request['windowsVersion'],
        'artifact': {k: root[k] for k in ('key', 'size', 'sha256')},
        'feed': {'key': graph_key, 'channel': 'stable'},
        'native': {'schema': 1, 'api': 'PackageManager.AddPackageByUriAsync',
                   'digestMethod': 'IAppxDigestProvider.GetDigest', 'digestEncoding': 'base64', 'graph': descriptor}}
    # This projection is head-bound but is not itself cryptographically signed.
    # Do not treat it as the signed install-closure release requirement.
    installation = {'schema': 1, 'root': graph['root'],
                    'objects': [{k: row[k] for k in ('key', 'size', 'sha256')}
                                for row in graph['objects']]}
    return {'schema': 2, 'request': request, 'packages': [package],
            'installation': installation}, graph


def _sign_installation_payload(value, sign):
    import base64
    from hermes_cli.release_channels import canonical_json
    payload = canonical_json(value)
    signature = sign(payload)
    if not isinstance(signature, bytes) or not signature or len(signature) > 4096:
        raise ValueError('invalid installation proof signature')
    return {'schema': 1, 'algorithm': 'rsa-pkcs1-sha256',
            'payload': base64.b64encode(payload).decode('ascii'),
            'signature': base64.b64encode(signature).decode('ascii')}


def add_installation_proof(manifest, sign):
    """Bind the install closure to one origin and release with the publisher key."""
    request = manifest['request']
    origin = request['publicBase']
    if origin not in ('https://download.semantier.cn', 'https://download.semantier.com'):
        raise ValueError('invalid installation origin')
    if len(manifest['packages']) != 1 or 'installationProof' in manifest:
        raise ValueError('invalid installation proof input')
    proof = _sign_installation_payload({'schema': 1, 'buildId': request['buildId'],
        'channel': request['channel'], 'sequence': request['sequence'], 'origin': origin,
        'version': request['windowsVersion'], 'identity': request['identity']['msixAppIdWithOrg'],
        'publisher': manifest['packages'][0]['publisher'], 'installation': manifest['installation']}, sign)
    manifest['installationProof'] = proof
    return proof


def legacy_installation_projection(manifest_bytes, graph_bytes, head, origin, verify_objects, sign):
    """Create a signed closure for an unchanged, head-bound older manifest.

    The caller must verify the exact object bytes and signatures and publish the
    returned immutable object through the existing fenced channel writer.
    """
    import hashlib
    from hermes_cli.release_channels import _head, decode_json, validate_native_graph
    _head(head)
    if origin not in ('https://download.semantier.cn','https://download.semantier.com'):
        raise ValueError('invalid installation origin')
    if len(manifest_bytes)>4*1024*1024 or len(graph_bytes)>4*1024*1024 or hashlib.sha256(manifest_bytes).hexdigest()!=head['sha256']:
        raise ValueError('legacy manifest does not match active head')
    manifest=decode_json(manifest_bytes);request=manifest['request'];packages=manifest['packages']
    if (manifest.get('schema')!=2 or 'installation' in manifest or 'installationProof' in manifest
            or request.get('buildId')!=head['buildId'] or request.get('sequence')!=head['sequence']
            or request.get('channel')!='semantier-acceptance' or len(packages)!=1):
        raise ValueError('invalid legacy installation metadata')
    package=packages[0];identity=request['identity']['msixAppIdWithOrg'];version=request['windowsVersion']
    publisher='CN=Semantier Technology, O=Semantier Technology'
    if (identity!='Semantier.Agent.Acceptance' or package.get('publisher')!=publisher
            or package.get('identity')!=identity or package.get('version')!=version
            or (package.get('platform'),package.get('arch'),package.get('variant'))!=('win32','x64','native-msix-assets')):
        raise ValueError('legacy package identity mismatch')
    descriptor=package['native']['graph'];prefix='releases/channel-builds/'+head['buildId']+'/'
    if (descriptor.get('key')!=prefix+'native-graph.json' or descriptor.get('size')!=len(graph_bytes)
            or descriptor.get('sha256')!=hashlib.sha256(graph_bytes).hexdigest()):
        raise ValueError('legacy native graph mismatch')
    graph=decode_json(graph_bytes)
    validate_native_graph(graph,build_id=head['buildId'],identity=identity,publisher=publisher,version=version,arch='x64')
    root=next(row for row in graph['objects'] if row['key']==graph['root'])
    if package['artifact']!={k:root[k] for k in ('key','size','sha256')}:
        raise ValueError('legacy native root mismatch')
    verify_objects(graph)
    install={'schema':1,'root':graph['root'],'objects':[{k:row[k] for k in ('key','size','sha256')} for row in graph['objects']]}
    proof=_sign_installation_payload({'schema':1,'buildId':head['buildId'],'channel':request['channel'],
        'sequence':head['sequence'],'origin':origin,'version':version,'identity':identity,'publisher':publisher,
        'manifestSha256':head['sha256'],'installation':install},sign)
    return {'schema':1,'installation':install,'installationProof':proof}


def sign_windows_installation_proof(payload, thumbprint):
    """Sign locally with the existing certificate; never change certificate stores."""
    import base64
    import os
    import subprocess
    if thumbprint != '57BF15739B4AA447842EAF388856A01B4E5029FF':
        raise ValueError('unexpected publisher certificate')
    if not isinstance(payload, bytes) or not payload or len(payload) > 4 * 1024 * 1024:
        raise ValueError('invalid installation proof payload')
    from hermes_platform.host.facts import os_family
    from hermes_platform.resolver import locate_command
    powershell = (locate_command('powershell.exe').command or [None])[0] if os_family() == 'win32' else '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    if not powershell or not os.path.isfile(powershell):
        raise RuntimeError('Windows signing host unavailable')
    script = r'''
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$cert=Get-Item -LiteralPath 'Cert:\CurrentUser\My\57BF15739B4AA447842EAF388856A01B4E5029FF'
if (!$cert.HasPrivateKey -or $cert.NotBefore -gt (Get-Date) -or $cert.NotAfter -le (Get-Date) -or
    $cert.EnhancedKeyUsageList.ObjectId -notcontains '1.3.6.1.5.5.7.3.3') {throw 'Invalid signing certificate'}
$sha=[Security.Cryptography.SHA256]::Create()
try {$digest=([BitConverter]::ToString($sha.ComputeHash($cert.RawData))).Replace('-','')}
finally {$sha.Dispose()}
if ($digest -cne '10A6B55E9A03BBC0DCEB8D899C1776BF1FBAD963C82050C16534DA4983C9661D') {throw 'Publisher certificate mismatch'}
$data=[Convert]::FromBase64String([Console]::In.ReadToEnd())
$private=[Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPrivateKey($cert)
$public=[Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPublicKey($cert)
if (!$private -or !$public) {throw 'Publisher RSA key unavailable'}
try {
  $signature=$private.SignData($data,[Security.Cryptography.HashAlgorithmName]::SHA256,[Security.Cryptography.RSASignaturePadding]::Pkcs1)
  if (!$public.VerifyData($data,$signature,[Security.Cryptography.HashAlgorithmName]::SHA256,[Security.Cryptography.RSASignaturePadding]::Pkcs1)) {throw 'Signature verification failed'}
  [Console]::Out.Write([Convert]::ToBase64String($signature))
} finally {$private.Dispose();$public.Dispose()}
'''
    env = os.environ.copy()
    removed = {name for name in env if name.startswith('CLOUDFLARE_R2_') or name == 'SEMANTIER_CLIENT_RECEIPT_KEY' or name.upper() == 'PSMODULEPATH'}
    for name in removed:
        env.pop(name, None)
    env['WSLENV'] = ':'.join(item for item in env.get('WSLENV', '').split(':')
                             if item and item.split('/')[0].upper() not in {name.upper() for name in removed})
    result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-EncodedCommand',
                             base64.b64encode(script.encode('utf-16le')).decode()],
                            input=base64.b64encode(payload), capture_output=True, check=True, env=env)
    return base64.b64decode(result.stdout, validate=True)


def prepare(output, expected_thumbprint, verify_signatures):
    """Inspect local graph and signed packages; no network or R2 changes."""
    import hashlib
    import stat
    from pathlib import Path
    from hermes_cli.release_channels import decode_json
    output = Path(output)
    def read(relative):
        path = output / relative
        if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('invalid metadata file')
        return decode_json(path.read_bytes())
    request = read('channel-request.json')
    signed = read('signed/receipt.json')
    result = read('result.json')
    if result.get('result') != 'PASS' or result.get('sourceCommit') != request['commit']:
        raise ValueError('completed exact source receipt required')
    directory = output / 'signed/distribution'
    graph_file = directory / 'native-graph.json'
    if graph_file.is_symlink() or graph_file.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('invalid graph file')
    graph_bytes = graph_file.read_bytes()
    manifest, graph = native_manifest(request, signed, graph_bytes)
    files = {}
    for row in graph['objects']:
        path = directory / row['key'].rsplit('/', 1)[1]
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size != row['size']:
            raise ValueError('native object location or size mismatch')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                raise ValueError('native object hash mismatch')
        files[row['key']] = {'path': path, 'size': row['size'], 'sha256': row['sha256']}
    if signed.get('certificateThumbprint') != expected_thumbprint:
        raise ValueError('untrusted signing certificate')
    verify_signatures([v['path'] for v in files.values()], expected_thumbprint,
                      [{'size':v['size'],'sha256':v['sha256']} for v in files.values()])
    graph_descriptor = manifest['packages'][0]['native']['graph']
    files[graph_descriptor['key']] = {'path': graph_file, 'size': graph_descriptor['size'], 'sha256': graph_descriptor['sha256']}
    return manifest, graph, files



def verify_setup_source(commit, tree):
    """Require an available exact source commit/tree; signing remains separate."""
    import subprocess
    from pathlib import Path
    from hermes_platform.resolver import locate_command
    root=Path(__file__).resolve().parents[2]
    git = locate_command('git').command
    if not git:
        raise ValueError('Native Git unavailable')
    try:
        actual=subprocess.check_output([*git,'-C',str(root),'rev-parse',commit+'^{tree}'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError) as error:
        raise ValueError('setup source commit unavailable') from error
    if actual!=tree:raise ValueError('setup source tree mismatch')


def prepare_manual_artifacts(output, manifest, graph, setup_path, setup_receipt_path, expected_thumbprint, verify_signatures, *, setup_key=None, verify_source=verify_setup_source):
    """Bind the signed reusable setup and compatibility ZIP to verified inputs."""
    import hashlib
    import re
    import stat
    from pathlib import Path
    from zipfile import ZipFile
    from hermes_cli.release_channels import build_prefix, decode_json
    output=Path(output);setup_path=Path(setup_path);setup_receipt_path=Path(setup_receipt_path)
    def regular(path):
        if any(part.is_symlink() for part in (path,*path.parents)) or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError('invalid manual artifact path')
        return path
    def digest(path):
        with regular(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    def metadata(path):
        regular(path)
        if path.stat().st_size>4*1024*1024:raise ValueError('invalid manual metadata')
        return decode_json(path.read_bytes())
    request=manifest['request'];prefix=build_prefix(request['buildId']);origin=request['publicBase']
    channels={'https://download.semantier.cn':('channel1','semantier-setup-ch1.exe'),
              'https://download.semantier.com':('channel2','semantier-setup-ch2.exe')}
    selected=channels.get(origin)
    if selected is None or setup_path.name!=selected[1]:
        raise ValueError('invalid setup channel')
    setup=metadata(setup_receipt_path)
    actual=digest(setup_path);size=setup_path.stat().st_size
    if (setup.get('channel')!=selected[0] or setup.get('signing')!='PASS' or setup.get('certificateThumbprint')!=expected_thumbprint
            or not re.fullmatch('[a-f0-9]{40}',str(setup.get('sourceCommit',''))) or not re.fullmatch('[a-f0-9]{40}',str(setup.get('sourceTree','')))
            or type(setup.get('size')) is not int or size!=setup['size'] or actual!=setup.get('sha256')):
        raise ValueError('setup receipt mismatch')
    verify_source(setup['sourceCommit'],setup['sourceTree'])
    verify_signatures([setup_path],expected_thumbprint,[{'size':size,'sha256':actual}])
    signed=metadata(output/'signed/receipt.json');archive_name=signed.get('archiveName')
    if not isinstance(archive_name,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+\.zip',archive_name):raise ValueError('invalid archive name')
    archive=regular(output/'signed'/archive_name);sidecar=regular(archive.with_name(archive_name+'.sha256'))
    if sidecar.stat().st_size>4096:raise ValueError('invalid archive checksum')
    tokens=sidecar.read_text(encoding='utf-8-sig').split()
    archive_hash=digest(archive)
    if not tokens or tokens[0].lower()!=archive_hash:raise ValueError('archive checksum mismatch')
    graph_path=regular(output/'signed/distribution/native-graph.json')
    if graph_path.stat().st_size>4*1024*1024:raise ValueError('graph metadata too large')
    graph_bytes=graph_path.read_bytes()
    if decode_json(graph_bytes)!=graph:raise ValueError('archive graph source mismatch')
    expected={row['key'].rsplit('/',1)[1]:row for row in graph['objects']}
    if len(expected)!=len(graph['objects']):raise ValueError('duplicate archive member')
    with ZipFile(archive) as z:
        entries=z.infolist();names=[entry.filename for entry in entries]
        if len(names)!=len(set(names)) or any('/' in name or '\\' in name or name in ('.','..') for name in names):
            raise ValueError('unsafe archive members')
        if any(((entry.external_attr>>16)&0o170000)==0o120000 for entry in entries):raise ValueError('unsafe archive link')
        for name,row in expected.items():
            try: entry=z.getinfo(name)
            except KeyError as error:raise ValueError('archive package missing') from error
            if entry.file_size!=row['size']:raise ValueError('archive package size mismatch')
            with z.open(entry) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=row['sha256']:raise ValueError('archive package digest mismatch')
        try:
            if z.getinfo('native-graph.json').file_size!=len(graph_bytes) or z.read('native-graph.json')!=graph_bytes:
                raise ValueError('archive graph mismatch')
            if z.getinfo('SemantierAgent-signing.cer').file_size>65536 or hashlib.sha256(z.read('SemantierAgent-signing.cer')).hexdigest()!='10a6b55e9a03bbc0dceb8d899c1776bf1fbad963c82050c16534da4983c9661d':
                raise ValueError('archive certificate mismatch')
        except KeyError as error:raise ValueError('archive metadata missing') from error
    selected_key=setup_key or prefix+setup_path.name
    if not re.fullmatch(r'releases/channel-builds/[a-f0-9]{32}/'+re.escape(selected[1]),selected_key):raise ValueError('invalid setup key')
    zip_key=prefix+archive_name
    files={selected_key:{'path':setup_path,'size':size,'sha256':actual},
           zip_key:{'path':archive,'size':archive.stat().st_size,'sha256':archive_hash}}
    def descriptor(key):return {'url':origin+'/'+key,'size':files[key]['size'],'sha256':files[key]['sha256']}
    return files,descriptor(selected_key),descriptor(zip_key)


def publish_prepared(driver, manifest, files):
    """Upload and verify all immutable objects before conditionally moving slots."""
    import hashlib
    from hermes_cli.release_channels import build_prefix, canonical_json
    request = manifest['request']
    record, etag = driver.read_channel(request['channel'])
    driver.validate(record, manifest)
    driver.ensure_capacity(manifest, files)
    for key, path in sorted(files.items()):
        driver.upload(key, path)
    manifest_key = build_prefix(request['buildId']) + 'build.json'
    manifest_bytes = canonical_json(manifest)
    driver.upload_metadata(manifest_key, manifest_bytes)
    head = {'buildId': request['buildId'], 'sequence': request['sequence'],
            'manifestKey': manifest_key, 'sha256': hashlib.sha256(manifest_bytes).hexdigest()}
    return driver.commit(request['channel'], etag, head)


def publish_with_manual(driver, manifest, files, manual_files, installer, archive, published_at, *, expected_head):
    """Stage both verified manual artifacts before promoting one native head."""
    import hashlib
    from hermes_cli.release_channels import canonical_json
    request=manifest['request'];channel=request['channel'];body=canonical_json(manifest)
    candidate={'buildId':request['buildId'],'sequence':request['sequence'],
               'manifestKey':'releases/channel-builds/'+request['buildId']+'/build.json',
               'sha256':hashlib.sha256(body).hexdigest()}
    driver.verify_lock();record,etag=driver.read_channel(channel);active=record.get('nativeHead')
    already=active==candidate
    if not already and (active!=expected_head or record.get('nativeRetirePending') is not None):
        raise ValueError('native head changed or cleanup pending')
    if already:
        published=driver.store.get(candidate['manifestKey'])
        if published is None or published[0]!=body:raise ValueError('published build differs from local candidate')
    else:
        driver.validate(record,manifest)
    if set(files)&set(manual_files):raise ValueError('manual and native objects overlap')
    combined={**files,**manual_files};driver.ensure_capacity(manifest,combined)
    prior=driver.store.get('releases/manual/catalog.json')
    project_manual_catalog(None if prior is None else prior[0],candidate,body,driver.public_base,installer,archive,published_at)
    driver.stage_manual_artifacts(channel,record,etag,candidate,manual_files,installer,archive)
    driver.verify_lock();latest,latest_etag=driver.read_channel(channel)
    if latest!=record or latest_etag!=etag:raise ValueError('native head changed while staging manual artifacts')
    result=record if already else publish_prepared(driver,manifest,files)
    if result.get('nativeHead')!=candidate:raise ValueError('native promotion did not select candidate')
    driver.reconcile_manual_catalog(channel,candidate,body,installer,archive,published_at)
    return result



def project_manual_catalog(raw, head, manifest_bytes, origin, installer, archive, published_at):
    """Derive a manual compatibility row from an exact native head.

    The caller must verify the actual artifact bytes/signatures, hold the shared
    writer lock, and fence/re-read the channel around the conditional write.
    This function never selects or changes the native head.
    """
    import hashlib
    import re
    from datetime import datetime, timedelta
    from urllib.parse import urlsplit
    from hermes_cli.release_channels import _head, canonical_json, decode_json

    if origin not in ('https://download.semantier.cn','https://download.semantier.com'):
        raise ValueError('invalid setup origin')
    active=_head(head)
    if active is None or hashlib.sha256(manifest_bytes).hexdigest()!=active['sha256']:
        raise ValueError('native head manifest mismatch')
    manifest=decode_json(manifest_bytes);request=manifest.get('request',{});packages=manifest.get('packages')
    identity='Semantier.Agent.Acceptance';publisher='CN=Semantier Technology, O=Semantier Technology'
    if (manifest.get('schema')!=2 or request.get('buildId')!=active['buildId'] or request.get('sequence')!=active['sequence']
            or request.get('channel')!='semantier-acceptance' or request.get('identity',{}).get('msixAppIdWithOrg')!=identity
            or not isinstance(packages,list) or len(packages)!=1 or packages[0].get('identity')!=identity
            or packages[0].get('publisher')!=publisher or packages[0].get('version')!=request.get('windowsVersion')):
        raise ValueError('native head identity mismatch')
    closure=manifest.get('installation');root=packages[0].get('artifact')
    if (not isinstance(closure,dict) or closure.get('schema')!=1 or not isinstance(root,dict)
            or closure.get('root')!=root.get('key') or not isinstance(closure.get('objects'),list)
            or not closure['objects'] or len(closure['objects'])>64):
        raise ValueError('missing installation closure')
    seen=set();prefix='releases/channel-builds/'+active['buildId']+'/'
    for obj in closure['objects']:
        if (not isinstance(obj,dict) or set(obj)!={'key','size','sha256'} or not isinstance(obj['key'],str)
                or not obj['key'].startswith(prefix) or '/' in obj['key'][len(prefix):]
                or not re.fullmatch(r'[A-Za-z0-9_.-]+\.(?:msix|msixbundle)',obj['key'][len(prefix):])
                or obj['key'] in seen or type(obj['size']) is not int or not 0<obj['size']<=480_000_000
                or not re.fullmatch('[a-f0-9]{64}',str(obj['sha256']))):
            raise ValueError('invalid installation closure')
        seen.add(obj['key'])
    if not any(obj==root for obj in closure['objects']):raise ValueError('installation root mismatch')
    version=request.get('windowsVersion')
    if not isinstance(version,str) or not re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){3}',version) or any(int(x)>65535 for x in version.split('.')):
        raise ValueError('invalid release version')
    try:
        when=datetime.fromisoformat(published_at)
        if when.utcoffset()!=timedelta(0):raise ValueError('UTC required')
    except (ValueError,TypeError,AttributeError) as error:
        raise ValueError('invalid publication time') from error
    def artifact(value, extension):
        if not isinstance(value,dict) or set(value)!={'url','size','sha256'} or type(value['size']) is not int or value['size']<=0 or not re.fullmatch('[a-f0-9]{64}',str(value['sha256'])):
            raise ValueError('invalid setup artifact')
        url=urlsplit(value['url'])
        match=re.fullmatch(r'/releases/channel-builds/([a-f0-9]{32})/[A-Za-z0-9_.-]+\.'+extension,url.path)
        if url.scheme!='https' or url.netloc!=urlsplit(origin).netloc or url.query or url.fragment or not match:
            raise ValueError('invalid setup origin or path')
        return match.group(1),url.path
    setup_build,setup_path=artifact(installer,'exe');archive_build,_=artifact(archive,'zip')
    expected_name='semantier-setup-ch1.exe' if origin.endswith('.cn') else 'semantier-setup-ch2.exe'
    if not setup_path.endswith('/'+expected_name) or archive_build!=active['buildId']:
        raise ValueError('setup or package build mismatch')
    catalog={'schema':1,'releases':[]} if raw is None else decode_json(raw)
    if set(catalog)!={'schema','releases'} or catalog['schema']!=1 or not isinstance(catalog['releases'],list) or len(catalog['releases'])>3:
        raise ValueError('invalid manual catalog')
    rows=catalog['releases']
    old_versions=[];old_builds=set()
    allowed_origins=({'channel1.semantier.com','download.semantier.cn'} if origin.endswith('.cn')
                     else {'update2.semantier.com','download.semantier.com'})
    for old in rows:
        required={'version','track','identity','provenance','publishedAt','installer','zip'}
        if (not isinstance(old,dict) or set(old) not in (required,required|{'nativeBuildId'})
                or old['track']!='acceptance' or old['identity']!=identity
                or old['provenance'] not in ('local-signed','ci-qualified') or not isinstance(old['version'],str)
                or not re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){3}',old['version'])
                or any(int(x)>65535 for x in old['version'].split('.'))):
            raise ValueError('invalid existing catalog')
        try:
            stamp=datetime.fromisoformat(old['publishedAt'])
            if stamp.utcoffset()!=timedelta(0):raise ValueError('UTC required')
        except (ValueError,TypeError,AttributeError) as error:raise ValueError('invalid existing catalog time') from error
        ids={};hosts=set()
        for label,ext in (('installer','exe'),('zip','zip')):
            item=old[label]
            if (not isinstance(item,dict) or set(item)!={'url','size','sha256'} or type(item['size']) is not int
                    or item['size']<=0 or not re.fullmatch('[a-f0-9]{64}',str(item['sha256']))):
                raise ValueError('invalid existing catalog artifact')
            parsed=urlsplit(item['url'])
            match=re.fullmatch(r'/releases/channel-builds/([a-f0-9]{32})/[A-Za-z0-9_.-]+\.'+ext,parsed.path)
            if parsed.scheme!='https' or parsed.netloc not in allowed_origins or parsed.query or parsed.fragment or not match:
                raise ValueError('invalid existing catalog origin')
            ids[label]=match.group(1);hosts.add(parsed.netloc)
        old_build=old.get('nativeBuildId',ids['zip'])
        if (len(hosts)!=1 or old_build!=ids['zip'] or ('nativeBuildId' not in old and ids['installer']!=old_build)
                or old_build in old_builds):raise ValueError('invalid existing catalog binding')
        old_builds.add(old_build);old_versions.append(tuple(map(int,old['version'].split('.'))))
    if len(old_versions)!=len(set(old_versions)) or old_versions!=sorted(old_versions,reverse=True):
        raise ValueError('invalid existing catalog order')
    if setup_build!=active['buildId']:
        if not any(isinstance(row,dict) and isinstance(row.get('installer'),dict)
                   and urlsplit(row['installer'].get('url','')).path==setup_path
                   and row['installer'].get('size')==installer['size'] and row['installer'].get('sha256')==installer['sha256'] for row in rows):
            raise ValueError('reused setup is not protected by prior catalog')
    row={'version':version,'track':'acceptance','identity':identity,'provenance':'local-signed','publishedAt':published_at,
         'nativeBuildId':active['buildId'],'installer':installer,'zip':archive}
    for old in rows:
        if not isinstance(old,dict) or not isinstance(old.get('version'),str) or not re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){3}',old['version']):
            raise ValueError('invalid existing catalog')
        if old.get('nativeBuildId')==active['buildId'] or old['version']==version:
            if old==row:return canonical_json(catalog)
            raise ValueError('manual catalog candidate conflict')
    if len(rows)>=3:raise ValueError('manual catalog capacity requires protected-reference reconciliation')
    return canonical_json({'schema':1,'releases':sorted([*rows,row],key=lambda item:tuple(map(int,item['version'].split('.'))),reverse=True)})


def check_capacity(existing, incoming, allowance=16_777_216):
    """Conservatively preserve the existing exclusive 10 GB storage limit."""
    if any(type(value) is not int or value < 0 for value in existing.values()):
        raise ValueError('invalid storage inventory')
    total = sum(existing.values()) + allowance
    for key, row in incoming.items():
        if type(row.get('size')) is not int or row['size'] < 0:
            raise ValueError('invalid incoming object size')
        if key not in existing:
            total += row['size']
        elif existing[key] != row['size']:
            raise ValueError('immutable object size conflict')
    if total >= 10_000_000_000:
        raise ValueError('R2 capacity exceeded')
    return total


def storage_inventory(transport, credentials, base, bucket, scope):
    """List every scoped object with its size; incomplete pagination fails closed."""
    import xml.etree.ElementTree as ET
    objects = {}
    token = None
    seen = set()
    while True:
        query = {'list-type': '2', 'max-keys': '1000', 'prefix': scope.listing_prefix('')}
        if token is not None:
            query['continuation-token'] = token
        url = scope.bucket_url(base, bucket) + '?' + transport.canonical_query(query)
        response = transport.signed_request('GET', url, creds=credentials, now=transport.amz_timestamp())
        root = ET.fromstring(response._body)
        for item in root.findall('{*}Contents'):
            key = scope.logical_key(item.findtext('{*}Key'))
            size = int(item.findtext('{*}Size'))
            if key in objects or size < 0:
                raise ValueError('invalid storage inventory')
            objects[key] = size
        truncated = root.findtext('{*}IsTruncated')
        if truncated == 'false':
            return objects
        token = root.findtext('{*}NextContinuationToken')
        if truncated != 'true' or not token or token in seen or len(objects) > 100_000:
            raise ValueError('incomplete storage inventory')
        seen.add(token)


class R2Driver:
    """Use the existing R2 transport directly; no Runtime service or database."""
    def __init__(self, store, public_base, control=None, lock_body=None, *, allow_sequence_gap=False):
        self.store = store
        self.public_base = public_base.rstrip('/')
        self.control = control
        self.lock_body = lock_body
        self.allow_sequence_gap = allow_sequence_gap
        self.public_verified = set()
        self.retired_builds = []
        self.retained_shared = []

    def verify_lock(self):
        current = self.control.get('_control/publication.lock') if self.control else None
        if current is None or current[0] != self.lock_body:
            raise RuntimeError('publication lock lost')

    def read_channel(self, channel):
        from hermes_cli.release_channels import channel_key, decode_json, validate_record
        value = self.store.get(channel_key(channel))
        if value is None:
            raise ValueError('channel does not exist')
        record = validate_record(decode_json(value[0]), name=channel)
        return record, value[1]

    def reconcile_views(self, channel):
        """Rebuild root views only while the same channel record and lock are held."""
        from hermes_cli.release_channels import decode_json
        self.verify_lock()
        record, etag = self.read_channel(channel)
        for key in ('active/manifest.json', 'archive/manifest.json'):
            current = self.store.get(key)
            if current is not None and decode_json(current[0]).get('channel') != channel:
                raise ValueError('slot view belongs to a different channel')
        driver = self
        class FencedStore:
            def get(self, key):
                return driver.store.get(key)
            def put(self, key, body, expected_etag=None):
                driver.verify_lock()
                current_record, current_etag = driver.read_channel(channel)
                if current_etag != etag or current_record != record:
                    raise ValueError('channel changed during view reconciliation')
                driver.store.put(key, body, expected_etag)
        reconcile_slot_views(FencedStore(), channel, record)
        self.verify_lock()
        confirmed, confirmed_etag = self.read_channel(channel)
        if confirmed_etag != etag or confirmed != record:
            raise ValueError('channel changed during view reconciliation')
        return record

    def validate(self, record, manifest):
        from copy import deepcopy
        from hermes_cli.release_channels import validate_manifest
        from scripts.releases.semver import compare
        request = manifest['request']
        if record['state'] != 'active' or request['publicBase'].rstrip('/') != self.public_base:
            raise ValueError('channel or public origin mismatch')
        candidate = deepcopy(record)
        candidate['head'] = None
        validate_manifest(manifest, candidate, self.public_base, allow_native=True)
        current = record.get('nativeHead')
        if current:
            old = self.store.get(current['manifestKey'])
            import hashlib
            from hermes_cli.release_channels import decode_json
            if old is None or hashlib.sha256(old[0]).hexdigest() != current['sha256']:
                raise ValueError('current manifest unavailable or changed')
            if compare(request['version'], decode_json(old[0])['request']['version']) <= 0:
                raise ValueError('release version must increase')
        advance(record, {'buildId': request['buildId'], 'sequence': request['sequence'],
                         'manifestKey': 'releases/channel-builds/'+request['buildId']+'/build.json',
                         'sha256': '0'*64}, allow_sequence_gap=self.allow_sequence_gap)

    def ensure_no_uploads(self):
        import xml.etree.ElementTree as ET
        from hermes_cli.release_channels import decode_json
        retained = self.control.get('_control/retention.json') if self.control else None
        if retained is not None:
            state = decode_json(retained[0])
            if not isinstance(state, dict) or not isinstance(state.get('inflight'), list):
                raise ValueError('invalid retained upload state')
            if state['inflight']:
                raise RuntimeError('in-flight release requires recovery')
        from scripts.releases import r2
        url = self.store.scope.bucket_url(self.store.base, self.store.bucket) + '?uploads='
        listing = r2.signed_request('GET', url, creds=self.store.creds, now=r2.amz_timestamp())
        root = ET.fromstring(listing._body)
        if root.findtext('{*}IsTruncated') != 'false' or root.findall('{*}Upload'):
            raise RuntimeError('incomplete multipart upload requires recovery')

    def ensure_capacity(self, manifest, files):
        from hermes_cli.release_channels import build_prefix, canonical_json
        from scripts.releases import r2
        self.verify_lock()
        self.ensure_no_uploads()
        public = storage_inventory(r2, self.store.creds, self.store.base, self.store.bucket, self.store.scope)
        private = storage_inventory(r2, self.control.credentials, self.control.base, self.control.bucket, self.store.scope)
        combined = {'public/'+key: size for key, size in public.items()}
        combined.update({'private/'+key: size for key, size in private.items()})
        incoming = {'public/'+key: row for key, row in files.items()}
        incoming['public/'+build_prefix(manifest['request']['buildId'])+'build.json'] = {'size': len(canonical_json(manifest))}
        return check_capacity(combined, incoming)

    def publish_legacy_projection(self, channel, expected_head, directory, thumbprint, verify_signatures, sign):
        """Publish an immutable closure for the currently selected old head."""
        import hashlib
        import stat
        from pathlib import Path
        from hermes_cli.release_channels import canonical_json, decode_json
        from scripts.releases import r2
        self.verify_lock()
        record,etag=self.read_channel(channel)
        if record.get('nativeHead')!=expected_head or record.get('nativeRetirePending') is not None:
            raise ValueError('legacy head changed or cleanup pending')
        self.ensure_no_uploads()
        current=self.store.get(expected_head['manifestKey'])
        if current is None or hashlib.sha256(current[0]).hexdigest()!=expected_head['sha256']:
            raise ValueError('legacy immutable manifest mismatch')
        manifest=decode_json(current[0]);descriptor=manifest['packages'][0]['native']['graph']
        prefix='releases/channel-builds/'+expected_head['buildId']+'/'
        if descriptor.get('key')!=prefix+'native-graph.json':
            raise ValueError('legacy graph key mismatch')
        remote_graph=self.store.get(descriptor['key'])
        if remote_graph is None:raise ValueError('legacy graph missing')
        directory=Path(directory)
        if directory.is_symlink() or not directory.is_dir():raise ValueError('legacy package directory invalid')
        def verify_objects(graph):
            paths=[]
            for row in graph['objects']:
                path=directory/row['key'].rsplit('/',1)[1]
                if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size!=row['size']:
                    raise ValueError('legacy local object location or size mismatch')
                with path.open('rb') as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest()!=row['sha256']:
                        raise ValueError('legacy local object digest mismatch')
                paths.append(path)
            verify_signatures(paths,thumbprint,[{'size':row['size'],'sha256':row['sha256']} for row in graph['objects']])
            for row in graph['objects']:
                self.verify_lock()
                url=self.store.scope.object_url(self.store.base,self.store.bucket,row['key'])
                r2.verify_remote_artifact(url,self.store.creds,r2.amz_timestamp(),row['size'],row['sha256'],algorithm='sha256',encoding='hex')
        projected=legacy_installation_projection(current[0],remote_graph[0],expected_head,self.public_base,verify_objects,sign)
        body=canonical_json(projected);key=prefix+'installation.json'
        self.verify_lock();self.ensure_no_uploads()
        public=storage_inventory(r2,self.store.creds,self.store.base,self.store.bucket,self.store.scope)
        private=storage_inventory(r2,self.control.credentials,self.control.base,self.control.bucket,self.store.scope)
        combined={'public/'+k:v for k,v in public.items()};combined.update({'private/'+k:v for k,v in private.items()})
        check_capacity(combined,{'public/'+key:{'size':len(body)}})
        current_record,current_etag=self.read_channel(channel)
        if current_record!=record or current_etag!=etag:raise ValueError('legacy head changed before projection upload')
        self.upload_metadata(key,body)
        self.verify_lock();final,final_etag=self.read_channel(channel)
        if final!=record or final_etag!=etag:raise ValueError('legacy head changed during projection upload')
        stored=self.store.get(key)
        if stored is None or stored[0]!=body:raise ValueError('legacy projection readback mismatch')
        return key,body

    def verify_manual_object(self, descriptor):
        """Read back the exact already-uploaded immutable manual artifact."""
        from urllib.parse import urlsplit
        from scripts.releases import r2
        key=urlsplit(descriptor['url']).path.lstrip('/')
        self.verify_lock()
        url=self.store.scope.object_url(self.store.base,self.store.bucket,key)
        r2.verify_remote_artifact(url,self.store.creds,r2.amz_timestamp(),descriptor['size'],descriptor['sha256'],algorithm='sha256',encoding='hex')
        self.verify_public_object({'key': key, 'size': descriptor['size'], 'sha256': descriptor['sha256']})

    def verify_public_object(self, descriptor):
        from scripts.releases.public_readback import verify_public_object
        facts = (descriptor['key'], descriptor['size'], descriptor['sha256'])
        if facts not in self.public_verified:
            verify_public_object(self.public_base, descriptor, self.verify_lock)
            self.public_verified.add(facts)

    def stage_manual_artifacts(self, channel, record, etag, candidate, files, installer, archive):
        """Create absent same-build manual bytes; never overwrite or recreate reused setup."""
        from urllib.parse import urlsplit
        from scripts.releases import r2
        entries={urlsplit(row['url']).path.lstrip('/'):row for row in (installer,archive)}
        if len(entries)!=2 or set(entries)!=set(files):raise ValueError('manual artifact binding mismatch')
        for key,row in sorted(entries.items()):
            local=files[key]
            if local['size']!=row['size'] or local['sha256']!=row['sha256']:
                raise ValueError('manual artifact facts mismatch')
            self.verify_lock();current,current_etag=self.read_channel(channel)
            if current!=record or current_etag!=etag:raise ValueError('native head changed while staging manual artifacts')
            url=self.store.scope.object_url(self.store.base,self.store.bucket,key)
            try:
                r2.signed_request('HEAD',url,creds=self.store.creds,now=r2.amz_timestamp(),tries=1)
                present=True
            except r2.R2RequestError as error:
                if error.status!=404:raise
                present=False
            self.verify_lock();current,current_etag=self.read_channel(channel)
            if current!=record or current_etag!=etag:raise ValueError('native head changed while staging manual artifacts')
            if not present:
                if not key.startswith('releases/channel-builds/'+candidate['buildId']+'/'):
                    raise ValueError('reused setup bytes are unavailable')
                self.upload(key,local)
            self.verify_manual_object(row)

    def reconcile_manual_catalog(self, channel, expected_head, manifest_bytes, installer, archive, published_at):
        """Conditionally publish a projection while the native head is fenced."""
        self.verify_lock()
        record,etag=self.read_channel(channel)
        if record.get('nativeHead')!=expected_head:
            raise ValueError('native head changed before catalog projection')
        published=self.store.get(expected_head['manifestKey'])
        if published is None or published[0]!=manifest_bytes:
            raise ValueError('active immutable manifest readback mismatch')
        current=self.store.get('releases/manual/catalog.json')
        old,old_etag=(None,None) if current is None else current
        body=project_manual_catalog(old,expected_head,manifest_bytes,self.public_base,installer,archive,published_at)
        for row in (installer,archive): self.verify_manual_object(row)
        self.verify_lock()
        latest,latest_etag=self.read_channel(channel)
        if latest!=record or latest_etag!=etag:raise ValueError('native head changed during catalog projection')
        if old!=body:self.store.put('releases/manual/catalog.json',body,old_etag)
        self.verify_lock()
        final,final_etag=self.read_channel(channel)
        if final!=record or final_etag!=etag:raise ValueError('native head changed after catalog projection')
        result=self.store.get('releases/manual/catalog.json')
        if result is None or result[0]!=body:raise ValueError('manual catalog readback mismatch')
        return body

    def upload(self, key, descriptor):
        self.verify_lock()
        import os
        from scripts.releases import r2
        path = str(descriptor['path'])
        digest = descriptor['sha256']
        if os.path.getsize(path) != descriptor['size']:
            raise ValueError('local artifact size changed')
        r2.put_object(self.store.creds, self.store.base, self.store.bucket, key, path,
                      r2.amz_timestamp(), r2.content_type_for(key), {'If-None-Match': '*'},
                      multipart_part_size=480_000_000, expected_sha256=digest)
        url = self.store.scope.object_url(self.store.base, self.store.bucket, key)
        r2.verify_remote_artifact(url, self.store.creds, r2.amz_timestamp(),
                                  descriptor['size'], digest, algorithm='sha256', encoding='hex')

    def upload_metadata(self, key, body):
        self.verify_lock()
        current = self.store.get(key)
        if current is not None:
            if current[0] != body:
                raise ValueError('immutable manifest conflict')
        else:
            self.store.put(key, body)

    def commit(self, channel, etag, head):
        self.verify_lock()
        import hashlib
        from hermes_cli.release_channels import decode_json
        stored = self.store.get(head['manifestKey'])
        if stored is None or hashlib.sha256(stored[0]).hexdigest() != head['sha256']:
            raise ValueError('Immutable candidate manifest differs before public readback')
        manifest = decode_json(stored[0])
        for row in manifest['installation']['objects']:
            self.verify_public_object(row)
        self.verify_public_object(manifest['packages'][0]['native']['graph'])
        self.verify_public_object({'key': head['manifestKey'], 'size': len(stored[0]), 'sha256': head['sha256']})
        self.verify_lock()
        from hermes_cli.release_channels import channel_key
        return commit_slot(self.store, channel_key(channel), etag, head, allow_sequence_gap=self.allow_sequence_gap)


    def can_retire(self, channel, build_id):
        import json
        import re
        from hermes_cli.release_channels import decode_json
        if not re.fullmatch(r'[a-f0-9]{32}', build_id):
            raise ValueError('invalid build identity')
        self.verify_lock()
        retained = self.control.get('_control/retention.json')
        if retained is not None:
            data = decode_json(retained[0])
            if not isinstance(data, dict) or not isinstance(data.get('inflight'), list):
                raise ValueError('invalid retained upload state')
            if build_id in data['inflight']:
                self.retained_shared.append(build_id)
                return False
        self.ensure_no_uploads()
        for key in self.store.keys('releases/'):
            if key.startswith('releases/channel-builds/'+build_id+'/'):
                continue
            if not key.endswith(('.json', '.appinstaller')):
                continue
            value = self.store.get(key)
            if value is None:
                raise RuntimeError('release reference changed during inspection')
            body = value[0]
            if key == 'releases/channels/'+channel+'.json':
                record = decode_json(body)
                record.pop('nativeRetirePending', None)
                body = json.dumps(record).encode()
            if build_id.encode() in body:
                self.retained_shared.append(build_id)
                return False
        return True

    def delete_build(self, build_id):
        from scripts.releases import r2
        prefix = 'releases/channel-builds/'+build_id+'/'
        keys = self.store.keys(prefix)
        for key in sorted(keys, key=lambda value: (value.endswith('/build.json'), value)):
            if not key.startswith(prefix):
                raise RuntimeError('retirement escaped build namespace')
            self.verify_lock()
            url = self.store.scope.object_url(self.store.base, self.store.bucket, key)
            try:
                response = r2.signed_request('HEAD', url, creds=self.store.creds,
                                             now=r2.amz_timestamp(), tries=1)
            except r2.R2RequestError as error:
                if error.status == 404:
                    continue
                raise
            etag = response.header('etag')
            if not etag:
                raise RuntimeError('retirement object lacks ETag')
            r2.signed_request('DELETE', url, creds=self.store.creds, now=r2.amz_timestamp(),
                              extra_headers={'If-Match': etag}, tries=1)
            try:
                r2.signed_request('HEAD', url, creds=self.store.creds, now=r2.amz_timestamp(), tries=1)
            except r2.R2RequestError as error:
                if error.status == 404:
                    continue
                raise
            raise RuntimeError('retired object remains visible')
        self.retired_builds.append(build_id)

    def finish_cleanup(self, channel, etag, build_id):
        from hermes_cli.release_channels import canonical_json, channel_key, decode_json
        self.verify_lock()
        key = channel_key(channel)
        current = self.store.get(key)
        if current is None or current[1] != etag:
            raise ValueError('channel changed during cleanup')
        record = decode_json(current[0])
        if record.get('nativeRetirePending', {}).get('buildId') != build_id:
            raise ValueError('pending archive changed')
        record.pop('nativeRetirePending')
        record['revision'] += 1
        self.store.put(key, canonical_json(record), etag)
        return record


    def sweep_old(self, channel):
        """Remove earlier unreferenced manifests after an active/archive pair exists."""
        import json
        import re
        from copy import deepcopy
        from hermes_cli.release_channels import decode_json, validate_manifest
        record, _ = self.read_channel(channel)
        archive = record.get('nativeArchive')
        if archive is None:
            return []
        active = record.get('nativeHead')
        retired = []
        for key in self.store.keys('releases/channel-builds/'):
            match = re.fullmatch(r'releases/channel-builds/([a-f0-9]{32})/build\.json', key)
            if not match or match.group(1) in (archive['buildId'], active['buildId']):
                continue
            raw = self.store.get(key)
            if raw is None:
                raise RuntimeError('release manifest changed during inventory')
            manifest = decode_json(raw[0])
            request = manifest.get('request', {})
            if request.get('channel') != channel:
                continue
            if request.get('buildId') != match.group(1) or type(request.get('sequence')) is not int:
                raise ValueError('invalid release inventory')
            if request['sequence'] >= archive['sequence']:
                raise ValueError('unexpected newer release outside active/archive slots')
            candidate = deepcopy(record)
            candidate['head'] = None
            validate_manifest(manifest, candidate, self.public_base, allow_native=True)
            if self.can_retire(channel, match.group(1)):
                self.delete_build(match.group(1))
                retired.append(match.group(1))
        return retired


def cleanup_pending(driver, channel):
    """Resume displaced-archive cleanup without dropping uncertain work."""
    record, etag = driver.read_channel(channel)
    pending = record.get('nativeRetirePending')
    if pending is None:
        return record
    build_id = pending['buildId']
    if driver.can_retire(channel, build_id):
        driver.delete_build(build_id)
    return driver.finish_cleanup(channel, etag, build_id)


def purge_explicit_versions(driver, channel, expected_head, expected_catalog_sha256, build_ids):
    """Retire explicitly authorized versions, including the current native head.

    The caller holds the shared publication lock.  The empty catalog is a
    fail-closed compatibility view; nativeRetirePending journals the head.
    """
    import hashlib
    import re
    from copy import deepcopy
    from hermes_cli.release_channels import canonical_json, channel_key, decode_json
    ids = set(build_ids)
    if (not ids or len(ids) != len(build_ids) or any(not re.fullmatch('[a-f0-9]{32}', x) for x in ids)
            or not re.fullmatch('[a-f0-9]{64}', expected_catalog_sha256)
            or expected_head['buildId'] not in ids):
        raise ValueError('invalid explicit retirement inventory')
    driver.verify_lock()
    driver.ensure_no_uploads()
    record, etag = driver.read_channel(channel)
    pending_retry = record.get('nativeHead') is None and record.get('nativeRetirePending') == expected_head
    if (record.get('head') is not None or record.get('nativeArchive') is not None
            or not (pending_retry or (record.get('nativeHead') == expected_head and record.get('nativeRetirePending') is None))):
        raise ValueError('channel is not the expected settled head')
    key = 'releases/manual/catalog.json'
    catalog = driver.store.get(key)
    empty_catalog = canonical_json({'schema': 1, 'releases': []})
    if catalog is None or (catalog[0] != empty_catalog and hashlib.sha256(catalog[0]).hexdigest() != expected_catalog_sha256):
        raise ValueError('catalog changed since retirement inventory')
    parsed = decode_json(catalog[0])
    if set(parsed) != {'schema', 'releases'} or parsed['schema'] != 1 or not isinstance(parsed['releases'], list):
        raise ValueError('invalid manual catalog')
    # Every other release reference remains protected, including other channels.
    own_prefixes = tuple('releases/channel-builds/'+x+'/' for x in ids)
    for name in driver.store.keys('releases/'):
        if name in (channel_key(channel), key) or name.startswith(own_prefixes) or not name.endswith(('.json', '.appinstaller')):
            continue
        data = driver.store.get(name)
        if data is None or any(x.encode() in data[0] for x in ids):
            raise ValueError('old version has an external or changing reference')
    for row in parsed['releases']:
        for field in ('installer', 'zip'):
            url = row.get(field, {}).get('url', '')
            if not any('/releases/channel-builds/'+x+'/' in url for x in ids):
                raise ValueError('catalog contains an unrelated release')
    # Eliminate catalog references before deleting the versions they name.
    driver.verify_lock()
    if catalog[0] != empty_catalog:
        driver.store.put(key, empty_catalog, catalog[1])
    for build in sorted(ids - {expected_head['buildId']}):
        if not driver.can_retire(channel, build):
            raise RuntimeError('old version remains referenced')
        driver.delete_build(build)
    driver.verify_lock()
    current, current_etag = driver.read_channel(channel)
    if current != record or current_etag != etag:
        raise ValueError('channel changed during retirement')
    if not pending_retry:
        pending = deepcopy(record)
        pending['nativeHead'] = None
        pending['nativeRetirePending'] = deepcopy(expected_head)
        pending['revision'] += 1
        driver.store.put(channel_key(channel), canonical_json(pending), etag)
    if not driver.can_retire(channel, expected_head['buildId']):
        raise RuntimeError('previous active version remains referenced')
    return cleanup_pending(driver, channel)


def purge_explicit_canary(driver, expected):
    """Remove a pinned obsolete canary only after checking other references."""
    import hashlib
    import re
    from scripts.releases import r2
    keys = ('releases/win32/canary/canary.appinstaller',
            'releases/win32/canary/SemantierAgent.msixbundle')
    if set(expected) != set(keys) or any(not re.fullmatch('[a-f0-9]{64}', x) for x in expected.values()):
        raise ValueError('invalid canary inventory')
    driver.verify_lock()
    driver.ensure_no_uploads()
    objects = {key: driver.store.get(key) for key in keys}
    for key, obj in objects.items():
        if obj is not None and hashlib.sha256(obj[0]).hexdigest() != expected[key]:
            raise ValueError('canary object changed since inventory')
    for key in driver.store.keys('releases/'):
        if key in keys or not key.endswith(('.json', '.appinstaller')):
            continue
        obj = driver.store.get(key)
        if obj is None or any(name.encode() in obj[0] for name in keys):
            raise ValueError('canary has an external or changing reference')
    for key in keys:
        obj = objects[key]
        if obj is None:
            continue
        driver.verify_lock()
        url = driver.store.scope.object_url(driver.store.base, driver.store.bucket, key)
        r2.signed_request('DELETE', url, creds=driver.store.creds, now=r2.amz_timestamp(),
                          extra_headers={'If-Match': obj[1]}, tries=1)
        if driver.store.get(key) is not None:
            raise RuntimeError('retired canary object remains visible')


def verify_windows_signatures(files, thumbprint, expected=None):
    """Use the Windows SDK signer verifier without forwarding R2 credentials."""
    import base64
    import json
    import os
    import re
    import subprocess
    if not re.fullmatch('[A-F0-9]{40}', thumbprint):
        raise ValueError('expected certificate thumbprint required')
    from hermes_platform.host.facts import os_family
    from hermes_platform.resolver import locate_command
    if os_family() == 'win32':
        powershell = (locate_command('powershell.exe').command or [None])[0]
        paths = [str(path.resolve(strict=True)) for path in files]
    else:
        powershell = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
        paths = [subprocess.check_output(['wslpath', '-w', str(path.resolve(strict=True))], text=True).strip() for path in files]
    if not powershell or not os.path.isfile(powershell):
        raise RuntimeError('Windows signing host unavailable')
    encoded = base64.b64encode(json.dumps(paths).encode()).decode()
    if expected is not None and (len(expected)!=len(paths) or any(type(row.get('size')) is not int or not re.fullmatch('[a-f0-9]{64}',row.get('sha256','')) for row in expected)):
        raise ValueError('invalid expected signed object facts')
    encoded_facts=base64.b64encode(json.dumps(expected or []).encode()).decode()
    script = r'''
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$decodedFiles=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('PATHS')) | ConvertFrom-Json
$decodedFacts=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('FACTS')) | ConvertFrom-Json
$files=@();foreach($item in $decodedFiles){$files+=,[string]$item}
$expected=@();foreach($item in $decodedFacts){$expected+=,$item}
$tool=@(Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" | Sort-Object FullName -Descending | Select-Object -First 1)
if($tool.Count -ne 1){throw 'SignTool unavailable'}
$held=@()
try {
  foreach($file in $files){$held+=,[IO.File]::Open($file,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)}
  for($i=0;$i -lt $files.Count;$i++){
    & $tool[0].FullName verify /pa $files[$i] *> $null
    if($LASTEXITCODE -ne 0){throw 'SignTool verification failed'}
    $signature=Get-AuthenticodeSignature -LiteralPath $files[$i]
    if($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Thumbprint -cne 'THUMB'){throw 'Signature or signer mismatch'}
    if(CHECKFACTS){
      $sha=[Security.Cryptography.SHA256]::Create()
      try {$digest=([BitConverter]::ToString($sha.ComputeHash($held[$i]))).Replace('-','').ToLowerInvariant()}
      finally {$sha.Dispose()}
      if($held[$i].Length -ne $expected[$i].size -or $digest -cne $expected[$i].sha256){throw 'Signed object digest mismatch'}
    }
  }
} finally {foreach($stream in $held){$stream.Dispose()}}
'''.replace('PATHS', encoded).replace('CHECKFACTS', '$true' if expected is not None else '$false').replace('FACTS', encoded_facts).replace('THUMB', thumbprint)
    env = os.environ.copy()
    removed = {name for name in env if name.startswith('CLOUDFLARE_R2_') or name == 'SEMANTIER_CLIENT_RECEIPT_KEY' or name.upper() == 'PSMODULEPATH'}
    for name in removed:
        env.pop(name, None)
    env['WSLENV'] = ':'.join(item for item in env.get('WSLENV', '').split(':')
                             if item and item.split('/')[0].upper() not in {name.upper() for name in removed})
    command = base64.b64encode(script.encode('utf-16le')).decode()
    subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-EncodedCommand', command], check=True, env=env)


def validate_notes(notes):
    import re
    if not isinstance(notes, dict) or len(notes) > 9:
        raise ValueError('invalid release notes')
    for language, text in notes.items():
        if (not isinstance(language, str) or not re.fullmatch(r'[a-z]{2}(?:-[a-z]+)?', language)
                or not isinstance(text, str) or not text or len(text.encode('utf-16-le')) // 2 > 16000
                or any(ord(char) < 32 and char not in '\n\r\t' for char in text)):
            raise ValueError('invalid release notes')
    return notes


def verify_public(base, channel, expected_head):
    """Require the updater-facing channel and manifest to match the commit."""
    from hermes_cli.release_channels import ChannelReader, channel_key, decode_json, validate_record
    reader = ChannelReader(base)
    original_open = reader.opener
    def publisher_open(request, timeout):
        request.add_header('User-Agent', 'SemantierPublisher/1.0')
        return original_open(request, timeout=timeout)
    reader.opener = publisher_open
    record = validate_record(decode_json(reader.read_bytes(channel_key(channel))), name=channel)
    if record.get('nativeHead') != expected_head:
        raise RuntimeError('public channel head has not converged')
    manifest = decode_json(reader.read_bytes(expected_head['manifestKey'], expected_head['sha256']))
    request = manifest['request']
    if request['buildId'] != expected_head['buildId'] or request['sequence'] != expected_head['sequence']:
        raise RuntimeError('public manifest identity does not match the channel head')
    package = manifest['packages'][0]
    descriptor = package['native']['graph']
    graph_bytes = reader.read_bytes(descriptor['key'], descriptor['sha256'])
    if len(graph_bytes) != descriptor['size']:
        raise RuntimeError('public native graph size mismatch')
    from hermes_cli.release_channels import validate_native_graph
    validate_native_graph(decode_json(graph_bytes), build_id=request['buildId'], identity=package['identity'],
                          publisher=package['publisher'], version=package['version'], arch=package['arch'])
    return record


def main(argv=None):
    import argparse
    import json
    import os
    from pathlib import Path
    from hermes_cli.release_channels import canonical_json
    from scripts.releases import r2
    from scripts.releases.channels import R2ChannelStore
    from scripts.releases.r2_scope import R2Scope

    parser = argparse.ArgumentParser(description='Publish one signed native release directly to R2')
    parser.add_argument('--output', type=Path, help='completed signed Windows build directory')
    parser.add_argument('--thumbprint', help='expected uppercase code-signing certificate thumbprint')
    parser.add_argument('--notes', type=Path, help='optional JSON map of localized release notes')
    parser.add_argument('--resume-cleanup', metavar='CHANNEL', help='resume a pending archive retirement')
    parser.add_argument('--project-legacy-channel', metavar='CHANNEL', help='sign the current legacy native installation closure')
    parser.add_argument('--legacy-directory', type=Path, help='retained signed native objects under deliverables work or release')
    parser.add_argument('--expected-build', help='exact active build ID for the legacy projection')
    parser.add_argument('--expected-manifest-sha256', help='exact active manifest SHA-256 for the legacy projection')
    parser.add_argument('--setup-exe', type=Path, help='signed reusable Channel 2 setup EXE')
    parser.add_argument('--setup-receipt', type=Path, help='source-bound setup signing receipt')
    parser.add_argument('--setup-key', help='previously catalog-protected reusable setup key')
    parser.add_argument('--manual-published-at', help='stable UTC catalog timestamp for retry')
    parser.add_argument('--expected-head', type=Path, help='exact currently active native head JSON')
    parser.add_argument('--expect-empty-head', action='store_true', help='require no currently active native head')
    parser.add_argument('--solo-intent', type=Path, help='source-bound solo allocator intent for a skipped build sequence')
    parser.add_argument('--windows-allocation', type=Path, help='existing native Windows version reservation')
    parser.add_argument('--windows-ledger', type=Path, help='current native Windows version ledger')
    parser.add_argument('--windows-acceptance', type=Path, help='installed Windows package acceptance receipt')
    parser.add_argument('--windows-ui', type=Path, help='real Windows application and backend acceptance receipt')
    parser.add_argument('--purge-explicit-channel', help='explicitly authorized cleanup-only channel')
    parser.add_argument('--purge-head', type=Path, help='exact active head JSON for cleanup')
    parser.add_argument('--purge-catalog-sha256', help='SHA-256 of the catalog inventoried before cleanup')
    parser.add_argument('--purge-build', action='append', help='exact build ID authorized for deletion; repeat for each')
    parser.add_argument('--purge-canary', action='store_true', help='delete the explicitly inventoried obsolete canary')
    parser.add_argument('--purge-canary-installer-sha256', help='exact appinstaller SHA-256')
    parser.add_argument('--purge-canary-bundle-sha256', help='exact bundle SHA-256')
    args = parser.parse_args(argv)
    windows_evidence = (args.windows_allocation, args.windows_ledger, args.windows_acceptance, args.windows_ui)
    if any(windows_evidence) and (not all(windows_evidence) or not args.output or args.solo_intent):
        parser.error('Windows recovery requires all four evidence paths and a signed output')
    legacy=args.project_legacy_channel
    purge=args.purge_explicit_channel
    manual=bool(args.setup_exe or args.setup_receipt or args.setup_key or args.manual_published_at or args.expected_head or args.expect_empty_head)
    if (sum(bool(v) for v in (args.resume_cleanup,args.output,legacy,purge,args.purge_canary))!=1
            or (args.output and (not args.thumbprint or args.legacy_directory or args.expected_build or args.expected_manifest_sha256))
            or (args.resume_cleanup and (args.thumbprint or args.notes or args.legacy_directory or args.expected_build or args.expected_manifest_sha256 or manual))
            or (legacy and (not args.thumbprint or not args.legacy_directory or not args.expected_build or not args.expected_manifest_sha256 or args.notes or manual))
            or (purge and (not args.purge_head or not args.purge_catalog_sha256 or not args.purge_build or args.thumbprint or args.notes or manual))
            or (not purge and (args.purge_head or args.purge_catalog_sha256 or args.purge_build))
            or (args.purge_canary and (not args.purge_canary_installer_sha256 or not args.purge_canary_bundle_sha256 or args.thumbprint or args.notes or manual))
            or (not args.purge_canary and (args.purge_canary_installer_sha256 or args.purge_canary_bundle_sha256))
            or (args.solo_intent and (not args.output or not manual))
            or (manual and (not args.output or not args.setup_exe or not args.setup_receipt or not args.manual_published_at
                            or bool(args.expected_head)==bool(args.expect_empty_head)))):
        parser.error('provide one publish, cleanup, or explicitly pinned legacy projection operation')
    if legacy:
        import re
        if not re.fullmatch('[a-f0-9]{32}',args.expected_build) or not re.fullmatch('[a-f0-9]{64}',args.expected_manifest_sha256):
            parser.error('legacy projection requires an exact build ID and manifest SHA-256')
        directory=args.legacy_directory.resolve(strict=True)
        roots=[Path('/mnt/d/repo/deliverables/work').resolve(strict=True),Path('/mnt/d/repo/deliverables/release').resolve(strict=True)]
        if not any(directory.is_relative_to(root) for root in roots):
            parser.error('legacy signed objects must be under the Windows deliverables work or release root')
    manifest = files = manual_files = None
    allow_sequence_gap = False
    if args.output:
        manifest, graph, files = prepare(args.output, args.thumbprint, verify_windows_signatures)
        if args.notes:
            if args.notes.is_symlink() or args.notes.stat().st_size > 160_000:
                raise ValueError('invalid release notes file')
            from hermes_cli.release_channels import decode_json
            manifest['notes'] = validate_notes(decode_json(args.notes.read_bytes()))
        if manifest['request']['publicBase']=='https://download.semantier.com' and not manual:
            raise ValueError('Channel 2 publication requires its reusable setup and catalog projection')
        if manual:
            if manifest['request']['publicBase']!='https://download.semantier.com':
                raise ValueError('manual publication requires Channel 2 origin')
            from hermes_cli.release_channels import _head,decode_json
            if args.expected_head:
                if args.expected_head.is_symlink() or args.expected_head.stat().st_size>65536:
                    raise ValueError('invalid expected native head file')
                expected_head=_head(decode_json(args.expected_head.read_bytes()))
                if expected_head is None:raise ValueError('expected native head missing')
            else:expected_head=None
            manual_files,installer,archive=prepare_manual_artifacts(args.output,manifest,graph,args.setup_exe,
                args.setup_receipt,args.thumbprint,verify_windows_signatures,setup_key=args.setup_key)
        if args.solo_intent:
            allow_sequence_gap = load_solo_allocation(args.output, args.solo_intent, manifest)
        if args.windows_allocation:
            from scripts.releases.windows_allocation import admit_windows_allocation
            allow_sequence_gap = admit_windows_allocation(args.output, *windows_evidence, manifest)
        add_installation_proof(manifest, lambda payload: sign_windows_installation_proof(payload, args.thumbprint))
    creds, base, bucket = r2.credentials()
    scope = R2Scope.configured()
    configured_public = os.environ.get('CLOUDFLARE_R2_PUBLIC_URL', '').rstrip('/')
    if manifest and (not configured_public or scope.public_base(configured_public) != manifest['request']['publicBase'].rstrip('/')):
        raise ValueError('configured public R2 origin does not match the signed request')
    if legacy and (not configured_public or scope.public_base(configured_public)!='https://download.semantier.com'):
        raise ValueError('legacy projection requires the configured Channel 2 origin')
    required = ('CLOUDFLARE_R2_CONTROL_BUCKET', 'CLOUDFLARE_R2_CONTROL_ACCESS_KEY_ID', 'CLOUDFLARE_R2_CONTROL_SECRET_ACCESS_KEY')
    if any(not os.environ.get(name) for name in required):
        raise RuntimeError('private R2 control bucket credentials required')
    control_bucket = os.environ[required[0]]
    if control_bucket == bucket:
        raise ValueError('control bucket must be separate from the public bucket')
    control = ControlStore(r2, {'access_key_id': os.environ[required[1]], 'secret_key': os.environ[required[2]]},
                           base, control_bucket, scope=scope)
    store = R2ChannelStore(creds, base, bucket, scope=scope)
    if args.purge_canary:
        if configured_public != 'https://download.semantier.com':
            raise ValueError('explicit purge requires the Channel 2 origin')
        expected = {'releases/win32/canary/canary.appinstaller': args.purge_canary_installer_sha256,
                    'releases/win32/canary/SemantierAgent.msixbundle': args.purge_canary_bundle_sha256}
        with publication_lock(control) as token:
            purge_explicit_canary(R2Driver(store, configured_public, control, token), expected)
        print(json.dumps({'retiredCanary': sorted(expected)}))
        return
    if purge:
        from hermes_cli.release_channels import _head, decode_json
        if args.purge_head.is_symlink() or args.purge_head.stat().st_size > 65536:
            raise ValueError('invalid expected purge head file')
        expected = _head(decode_json(args.purge_head.read_bytes()))
        if expected is None or configured_public != 'https://download.semantier.com':
            raise ValueError('explicit purge requires a pinned Channel 2 head and origin')
        with publication_lock(control) as token:
            driver = R2Driver(store, configured_public, control, token)
            result = purge_explicit_versions(driver, purge, expected, args.purge_catalog_sha256, args.purge_build)
        print(json.dumps({'channel': purge, 'record': result, 'retiredBuilds': driver.retired_builds}))
        return
    channel = args.resume_cleanup or legacy or manifest['request']['channel']
    projection_key=projection_body=catalog_body=None
    with publication_lock(control) as token:
        driver = R2Driver(store, manifest['request']['publicBase'] if manifest else (configured_public if legacy else ''), control, token,
                          allow_sequence_gap=allow_sequence_gap)
        if not legacy and not manual:cleanup_pending(driver, channel)
        if manual:
            result=publish_with_manual(driver,manifest,files,manual_files,installer,archive,args.manual_published_at,expected_head=expected_head)
            result=cleanup_pending(driver,channel)
            stored=store.get('releases/manual/catalog.json')
            if stored is None:raise RuntimeError('manual catalog missing after publication')
            catalog_body=stored[0]
        elif manifest:
            current, _ = driver.read_channel(channel)
            active = current.get('nativeHead')
            if active and active['buildId'] == manifest['request']['buildId']:
                existing = store.get(active['manifestKey'])
                if existing is None or existing[0] != canonical_json(manifest):
                    raise ValueError('published build differs from local candidate')
                result = current
            else:
                result = publish_prepared(driver, manifest, files)
            result = cleanup_pending(driver, channel)
            driver.sweep_old(channel)
        elif legacy:
            result,_=driver.read_channel(channel);active=result.get('nativeHead')
            if not active or active['buildId']!=args.expected_build or active['sha256']!=args.expected_manifest_sha256:
                raise ValueError('legacy active head differs from expected identity')
            projection_key,projection_body=driver.publish_legacy_projection(channel,active,directory,args.thumbprint,
                verify_windows_signatures,lambda payload:sign_windows_installation_proof(payload,args.thumbprint))
        else:
            result, _ = driver.read_channel(channel)
    if manifest:
        verify_public(manifest['request']['publicBase'], channel, result['nativeHead'])
    if manual:
        import hashlib
        from hermes_cli.release_channels import ChannelReader
        reader=ChannelReader(configured_public);opener=reader.opener
        def catalog_open(request,timeout):
            request.add_header('User-Agent','SemantierPublisher/1.0')
            return opener(request,timeout=timeout)
        reader.opener=catalog_open
        public_catalog=reader.read_bytes('releases/manual/catalog.json',hashlib.sha256(catalog_body).hexdigest())
        if public_catalog!=catalog_body or project_manual_catalog(public_catalog,result['nativeHead'],canonical_json(manifest),
                configured_public,installer,archive,args.manual_published_at)!=catalog_body:
            raise RuntimeError('public manual catalog does not agree with active head')
        verify_public(configured_public,channel,result['nativeHead'])
    if legacy:
        import hashlib
        from hermes_cli.release_channels import ChannelReader
        verify_public(configured_public,channel,result['nativeHead'])
        reader=ChannelReader(configured_public);opener=reader.opener
        def public_open(request,timeout):
            request.add_header('User-Agent','SemantierPublisher/1.0')
            return opener(request,timeout=timeout)
        reader.opener=public_open
        if reader.read_bytes(projection_key,hashlib.sha256(projection_body).hexdigest())!=projection_body:
            raise RuntimeError('public legacy projection mismatch')
        verify_public(configured_public,channel,result['nativeHead'])
    print(json.dumps({'channel': channel, 'revision': result['revision'],
                      'active': result.get('nativeHead'), 'archive': result.get('nativeArchive'),
                      'retired': driver.retired_builds, 'retained_shared': driver.retained_shared,
                      **({'legacy_projection':projection_key} if legacy else {})}))


if __name__ == '__main__':
    main()
