"""Native channel active/archive transition, independent of Runtime and databases."""
from copy import deepcopy


def advance(record, new_head):
    """Return a new record; callers must commit it with the observed ETag."""
    from hermes_cli.release_channels import _head
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
            or new_head['sequence'] > record['nextSequence']
            or (old is not None and new_head['sequence'] <= old['sequence'])
            or (archive is not None and new_head['sequence'] <= archive['sequence'])):
        raise ValueError('native release sequence is stale or unallocated')
    result = deepcopy(record)
    result['nativeRetirePending'] = deepcopy(record.get('nativeArchive'))
    result['nativeArchive'] = deepcopy(old)
    result['nativeHead'] = deepcopy(new_head)
    result['revision'] += 1
    if new_head['sequence'] == record['nextSequence']:
        result['nextSequence'] += 1
    return result


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


def commit_slot(store, key, expected_etag, new_head):
    """Atomically change both slots; an ETag conflict never overwrites a newer head."""
    from hermes_cli.release_channels import canonical_json, decode_json
    found = store.get(key)
    if found is None or found[1] != expected_etag:
        raise ValueError('channel changed since verification')
    record = decode_json(found[0])
    result = advance(record, new_head)
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
    """Build updater-compatible metadata from the signed native graph."""
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
    return {'schema': 2, 'request': request, 'packages': [package]}, graph


def prepare(output, expected_thumbprint, verify_signatures):
    """Inspect the locally signed graph; no network or R2 changes."""
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
    verify_signatures([v['path'] for v in files.values()], expected_thumbprint)
    graph_descriptor = manifest['packages'][0]['native']['graph']
    files[graph_descriptor['key']] = {'path': graph_file, 'size': graph_descriptor['size'], 'sha256': graph_descriptor['sha256']}
    return manifest, graph, files


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
    def __init__(self, store, public_base, control=None, lock_body=None):
        self.store = store
        self.public_base = public_base.rstrip('/')
        self.control = control
        self.lock_body = lock_body
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
                         'sha256': '0'*64})

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
        from hermes_cli.release_channels import channel_key
        return commit_slot(self.store, channel_key(channel), etag, head)


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


def verify_windows_signatures(files, thumbprint):
    """Use the Windows SDK signer verifier without forwarding R2 credentials."""
    import base64
    import json
    import os
    import re
    import subprocess
    if not re.fullmatch('[A-F0-9]{40}', thumbprint):
        raise ValueError('expected certificate thumbprint required')
    powershell = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    if not os.path.isfile(powershell):
        raise RuntimeError('Windows SignTool verification requires WSL with Windows SDK')
    paths = [subprocess.check_output(['wslpath', '-w', str(path.resolve(strict=True))], text=True).strip() for path in files]
    encoded = base64.b64encode(json.dumps(paths).encode()).decode()
    script = r'''
$ErrorActionPreference='Stop'
$files=([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('PATHS')) | ConvertFrom-Json)
$tool=@(Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" | Sort-Object FullName -Descending | Select-Object -First 1)
if($tool.Count -ne 1){throw 'SignTool unavailable'}
foreach($file in $files){
  & $tool[0].FullName verify /pa $file *> $null
  if($LASTEXITCODE -ne 0){throw 'SignTool verification failed'}
  $signature=Get-AuthenticodeSignature -LiteralPath $file
  if($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Thumbprint -cne 'THUMB'){throw 'Signature or signer mismatch'}
}
'''.replace('PATHS', encoded).replace('THUMB', thumbprint)
    env = os.environ.copy()
    removed = {name for name in env if name.startswith('CLOUDFLARE_R2_') or name == 'SEMANTIER_CLIENT_RECEIPT_KEY'}
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
    args = parser.parse_args(argv)
    if (bool(args.resume_cleanup) == bool(args.output) or (args.output and not args.thumbprint)
            or (args.resume_cleanup and (args.thumbprint or args.notes))):
        parser.error('provide --output and --thumbprint, or --resume-cleanup CHANNEL')
    manifest = files = None
    if args.output:
        manifest, _, files = prepare(args.output, args.thumbprint, verify_windows_signatures)
        if args.notes:
            if args.notes.is_symlink() or args.notes.stat().st_size > 160_000:
                raise ValueError('invalid release notes file')
            from hermes_cli.release_channels import decode_json
            manifest['notes'] = validate_notes(decode_json(args.notes.read_bytes()))
    creds, base, bucket = r2.credentials()
    scope = R2Scope.configured()
    configured_public = os.environ.get('CLOUDFLARE_R2_PUBLIC_URL', '').rstrip('/')
    if manifest and (not configured_public or scope.public_base(configured_public) != manifest['request']['publicBase'].rstrip('/')):
        raise ValueError('configured public R2 origin does not match the signed request')
    required = ('CLOUDFLARE_R2_CONTROL_BUCKET', 'CLOUDFLARE_R2_CONTROL_ACCESS_KEY_ID', 'CLOUDFLARE_R2_CONTROL_SECRET_ACCESS_KEY')
    if any(not os.environ.get(name) for name in required):
        raise RuntimeError('private R2 control bucket credentials required')
    control_bucket = os.environ[required[0]]
    if control_bucket == bucket:
        raise ValueError('control bucket must be separate from the public bucket')
    control = ControlStore(r2, {'access_key_id': os.environ[required[1]], 'secret_key': os.environ[required[2]]},
                           base, control_bucket, scope=scope)
    store = R2ChannelStore(creds, base, bucket, scope=scope)
    channel = args.resume_cleanup or manifest['request']['channel']
    with publication_lock(control) as token:
        driver = R2Driver(store, manifest['request']['publicBase'] if manifest else '', control, token)
        cleanup_pending(driver, channel)
        if manifest:
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
        else:
            result, _ = driver.read_channel(channel)
    if manifest:
        verify_public(manifest['request']['publicBase'], channel, result['nativeHead'])
    print(json.dumps({'channel': channel, 'revision': result['revision'],
                      'active': result.get('nativeHead'), 'archive': result.get('nativeArchive'),
                      'retired': driver.retired_builds, 'retained_shared': driver.retained_shared}))


if __name__ == '__main__':
    main()
