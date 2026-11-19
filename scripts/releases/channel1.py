"""Metadata companion to deploy-ecs.ps1's single signed MSIX publication.

The canonical channel pointer is the final commit. Package signing remains the
native operator's prerequisite; this module never builds or signs packages.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

BASE = 'https://cloud.semantier.cn'
CHANNEL = 'semantier-acceptance'
POINTER = 'releases/channels/' + CHANNEL + '.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _safe(root, key):
    path = root / key
    if not path.is_relative_to(root) or any(part in ('..', '') for part in Path(key).parts):
        raise ValueError('Invalid publication path')
    for parent in (path, *path.parents):
        if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
            raise ValueError('Linked publication path')
        if parent == root:
            break
    return path


def prepare(package, output, *, expected_sha256, source_commit, windows_version, build_id, native_digest):
    """Bind metadata to the operator-verified signed package's embedded facts."""
    from hermes_cli import release_channels as protocol
    package, output = Path(package), Path(output)
    decoded = base64.b64decode(native_digest, validate=True)
    if len(decoded) != 32 or base64.b64encode(decoded).decode() != native_digest:
        raise ValueError('Invalid native digest')
    if package.is_symlink() or sha(package) != expected_sha256:
        raise ValueError('Package SHA256 integrity mismatch')
    with zipfile.ZipFile(package) as archive:
        native = ET.fromstring(archive.read('AppxManifest.xml')).find('{*}Identity').attrib
        inventory = hashlib.sha256(archive.read('AppxBlockMap.xml')).hexdigest()
        stamp = json.loads(archive.read('app/resources/install-stamp.json'))
    request = protocol.validate_request(stamp['channelBuild'], repository='chris-han/hermes-agent', base_url=BASE)
    if (stamp.get('dirty') is not False or stamp['commit'] != source_commit or request['commit'] != source_commit
            or request['buildId'] != build_id or request['windowsVersion'] != windows_version
            or request['channel'] != CHANNEL or native['Version'] != windows_version
            or native['Name'] != 'Semantier.Agent.Acceptance' or native['Name'] != request['identity']['msixAppIdWithOrg']
            or native['Publisher'] != 'CN=Semantier Technology, O=Semantier Technology'
            or native['ProcessorArchitecture'] != 'x64'):
        raise ValueError('Package identity/version/source binding mismatch')
    prefix = protocol.build_prefix(build_id)
    artifact = {'key': prefix+'SemantierAgent.msix', 'sha256': expected_sha256, 'size': package.stat().st_size}
    graph = {'schema': 1, 'transport': 'windows-package-uri', 'root': artifact['key'],
             'payloadInventorySha256': inventory,
             'objects': [{**artifact, 'kind': 'main-package', 'identity': native['Name'],
                          'publisher': native['Publisher'], 'version': windows_version, 'arch': 'x64',
                          'nativeDigest': native_digest, 'references': []}]}
    graph_body = protocol.canonical_json(graph)
    descriptor = {'key': prefix+'native-graph.json', 'sha256': hashlib.sha256(graph_body).hexdigest(), 'size': len(graph_body)}
    row = {'platform': 'win32', 'arch': 'x64', 'variant': 'bundled', 'version': windows_version,
           'identity': native['Name'], 'publisher': native['Publisher'], 'artifact': artifact,
           'feed': {'key': descriptor['key'], 'channel': 'stable'},
           'native': {'schema': 2, 'packageBase': 'https://assets.semantier.cn', 'api': 'PackageManager.AddPackageByUriAsync',
                      'digestMethod': 'IAppxDigestProvider.GetDigest', 'digestEncoding': 'base64', 'graph': descriptor}}
    manifest = {'schema': 2, 'request': request, 'packages': [row]}
    record = _record(manifest, None, protocol)
    protocol.validate_manifest(manifest, record, BASE, allow_native=True)
    protocol.validate_native_graph(graph, build_id=build_id, identity=native['Name'], publisher=native['Publisher'], version=windows_version, arch='x64')
    result = {'manifest': manifest, 'graph': graph}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(protocol.canonical_json(result))
    return result


def _record(manifest, previous, protocol):
    request = manifest['request']
    digest = hashlib.sha256(protocol.canonical_json(manifest)).hexdigest()
    same = bool(previous and (previous.get('nativeHead') or previous.get('head') or {}).get('sha256') == digest)
    if previous:
        protocol.validate_record(previous, name=CHANNEL, repository=request['repository'])
        old = previous.get('nativeHead') or previous.get('head')
        if previous['identity'] != request['identity'] or previous['state'] != 'active':
            raise ValueError('Existing channel identity/state mismatch')
        if old and (request['sequence'] < old['sequence'] or
                    (request['sequence'] == old['sequence'] and not same)):
            raise ValueError('Older sequence or changed immutable release')
    record = {'schema': 1, 'name': CHANNEL, 'repository': request['repository'], 'state': 'active',
              'policy': previous['policy'] if previous else 'stable-release', 'identity': request['identity'],
              'revision': previous['revision'] + (not same) if previous else 1,
              'nextSequence': max(previous['nextSequence'] if previous else 1, request['sequence']+1),
              'head': None, 'nativeHead': {'buildId': request['buildId'], 'sequence': request['sequence'],
                       'manifestKey': protocol.build_prefix(request['buildId'])+'build.json', 'sha256': digest}}
    return protocol.validate_record(record)


@contextmanager
def _lock(root):
    with (root.parent/'.channel1-publication.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b'\0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _sync_directory(path):
    if os.name != 'nt':
        descriptor = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.channel1-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        os.chmod(name, 0o644)
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def _uploaded_metadata(stage, metadata_sha256, package_sha256, protocol):
    metadata = stage/'channel1-metadata.json'
    package = stage/'SemantierAgent.msix'
    if sha(metadata) != metadata_sha256 or sha(package) != package_sha256:
        raise ValueError('Uploaded SHA256 integrity mismatch')
    data = protocol.decode_json(metadata.read_bytes())
    manifest = data['manifest']; request = manifest['request']; row = manifest['packages'][0]
    prefix = protocol.build_prefix(request['buildId'])
    if (request['publicBase'] != BASE or request['channel'] != CHANNEL or len(manifest['packages']) != 1
            or row['artifact'] != {'key': prefix+'SemantierAgent.msix', 'sha256': package_sha256, 'size': package.stat().st_size}
            or manifest['schema'] != 2 or row['variant'] != 'bundled'
            or row['feed'] != {'key': prefix+'native-graph.json', 'channel': 'stable'}):
        raise ValueError('Uploaded artifact namespace/binding mismatch')
    protocol.validate_manifest(manifest, _record(manifest, None, protocol), BASE, allow_native=True)
    graph = data['graph']
    protocol.validate_native_graph(graph, build_id=request['buildId'], identity=row['identity'], publisher=row['publisher'], version=row['version'], arch='x64')
    graph_body = protocol.canonical_json(graph)
    if (len(graph['objects']) != 1 or graph['root'] != row['artifact']['key']
            or any(graph['objects'][0][key] != row['artifact'][key] for key in ('key', 'sha256', 'size'))
            or row['native']['graph'] != {'key': row['feed']['key'], 'sha256': hashlib.sha256(graph_body).hexdigest(), 'size': len(graph_body)}):
        raise ValueError('Native graph binding mismatch')
    with zipfile.ZipFile(package) as archive:
        native = ET.fromstring(archive.read('AppxManifest.xml')).find('{*}Identity').attrib
        stamp = protocol.decode_json(archive.read('app/resources/install-stamp.json'))
        if (native['Name'] != row['identity'] or native['Publisher'] != row['publisher']
                or native['Publisher'] != 'CN=Semantier Technology, O=Semantier Technology'
                or native['Name'] != 'Semantier.Agent.Acceptance' or native['Version'] != row['version']
                or native['ProcessorArchitecture'] != 'x64' or stamp['channelBuild'] != request
                or stamp.get('dirty') is not False or stamp['commit'] != request['commit']
                or graph['payloadInventorySha256'] != hashlib.sha256(archive.read('AppxBlockMap.xml')).hexdigest()):
            raise ValueError('Uploaded signed package metadata binding mismatch')
    return manifest, graph_body


def _obsolete_metadata(root, prefix, request, protocol):
    obsolete = []
    for path in (root/'releases/channel-builds').glob('*/build.json'):
        path = _safe(root, str(path.relative_to(root)))
        old = protocol.decode_json(path.read_bytes())
        old_request = old.get('request', {})
        if path == root/(prefix+'build.json') or old_request.get('channel') != CHANNEL:
            continue
        if old_request.get('identity') != request['identity'] or old_request.get('publicBase') != BASE:
            raise ValueError('Unrecognized historical manifest; refusing cleanup')
        # Legacy consumers may still hold the former pointer; preserve its closure.
        if old.get('schema') == 1:
            continue
        for relative in ('build.json', 'native-graph.json'):
            candidate = _safe(root, str((path.parent/relative).relative_to(root)))
            if candidate.exists(): obsolete.append(candidate)
    return obsolete


def _rollback(snapshots, pointer, stable, created_dirs):
    # Restore the former closure before switching back; remove new paths last.
    existing = [p for p, saved in snapshots.items() if saved is not None and p != pointer]
    missing = [p for p, saved in snapshots.items() if saved is None and p != pointer]
    for target in [*existing, pointer, *missing]:
        saved = snapshots[target]
        if saved is None: target.unlink(missing_ok=True)
        elif target.exists() and target.samefile(saved): continue
        else: os.replace(saved, target)
    temporary = stable.with_name('.channel1-package.tmp'); temporary.unlink(missing_ok=True)
    for path in sorted(created_dirs, key=lambda p: len(p.parts), reverse=True):
        if path.exists(): path.rmdir()


def _activate(root, stage, record, row, bodies, obsolete, backup, protocol):
    pointer = _safe(root, POINTER)
    artifact = _safe(root, row['artifact']['key'])
    package = stage/'SemantierAgent.msix'
    package_sha256 = row['artifact']['sha256']
    stable = _safe(root, 'releases/SemantierAgent.msix')
    checksum = _safe(root, 'releases/SemantierAgent.msix.sha256')
    targets = [stable, checksum, artifact, *(_safe(root, k) for k in bodies), *obsolete, pointer]
    snapshots = {}
    for index, target in enumerate(targets):
        if target.exists():
            saved = backup/str(index); os.link(target, saved)
            snapshots[target] = saved
        else: snapshots[target] = None
    created_dirs = set()
    for path in targets:
        parent = path.parent
        while not parent.exists(): created_dirs.add(parent); parent = parent.parent
    try:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if not artifact.exists():
            os.link(package, artifact)
            os.chmod(artifact, 0o644)
        for key, body in bodies.items(): _atomic(root/key, body)
        stable.parent.mkdir(parents=True, exist_ok=True)
        temporary = stable.with_name('.channel1-package.tmp')
        if temporary.exists(): raise ValueError('Unreconciled package activation')
        # POSIX rename is a no-op when both names already share an inode.
        # Avoid leaving the temporary link behind on an idempotent publish.
        if not stable.exists() or not artifact.samefile(stable):
            os.link(artifact, temporary); os.replace(temporary, stable)
        _atomic(checksum, (package_sha256+'  SemantierAgent.msix\n').encode())
        pointer.parent.mkdir(parents=True, exist_ok=True)
        for path in sorted(created_dirs, key=lambda p: len(p.parts), reverse=True):
            _sync_directory(path); _sync_directory(path.parent)
        # Readers of the old pointer retain its complete immutable closure.
        _atomic(pointer, protocol.canonical_json(record))
        for path in obsolete: path.unlink()
    except BaseException:
        _rollback(snapshots, pointer, stable, created_dirs)
        raise


def _save_recovery(root, recovery, record, package_sha256, protocol):
    recovery = Path(recovery).absolute()
    if recovery.parent != root.parent or not recovery.name.startswith('.channel1-recovery-'):
        raise ValueError('Invalid recovery boundary')
    _safe(root.parent, recovery.name)
    pointer = root/POINTER
    if (recovery/'recovery.json').exists():
        saved = protocol.decode_json((recovery/'recovery.json').read_bytes())
        if (saved['root'] != str(root) or saved['afterPackageSha256'] != package_sha256
                or saved['afterPointerSha256'] != hashlib.sha256(protocol.canonical_json(record)).hexdigest()):
            raise ValueError('Recovery binding changed')
        for row in saved['files']:
            if sha(_safe(recovery, row['saved'])) != row['sha256']:
                raise ValueError('Recovery SHA256 mismatch')
        return
    recovery.mkdir(mode=0o700, exist_ok=False)
    files = []
    for index, key in enumerate(('releases/SemantierAgent.msix', 'releases/SemantierAgent.msix.sha256', POINTER)):
        target = _safe(root, key)
        saved = recovery/str(index)
        os.link(target, saved)
        files.append({'key': key, 'saved': str(index), 'sha256': sha(saved)})
    _atomic(recovery/'recovery.json', protocol.canonical_json({
        'root': str(root), 'beforePointerSha256': sha(pointer),
        'afterPointerSha256': hashlib.sha256(protocol.canonical_json(record)).hexdigest(),
        'afterPackageSha256': package_sha256, 'files': files}))
    _sync_directory(recovery)


def restore(root, recovery):
    """Restore future-client availability; never perform a Windows OS downgrade."""
    root, recovery = Path(root).absolute(), Path(recovery).absolute()
    if recovery.parent != root.parent or not recovery.name.startswith('.channel1-recovery-'):
        raise ValueError('Invalid recovery boundary')
    _safe(root.parent, recovery.name)
    saved = json.loads((recovery/'recovery.json').read_bytes())
    if saved['root'] != str(root):
        raise ValueError('Recovery root mismatch')
    with _lock(root):
        if sha(root/POINTER) not in (saved['beforePointerSha256'], saved['afterPointerSha256']):
            raise ValueError('Concurrent pointer; refusing rollback')
        package = root/'releases/SemantierAgent.msix'
        if sha(package) not in (saved['files'][0]['sha256'], saved['afterPackageSha256']):
            raise ValueError('Concurrent package; refusing rollback')
        expected = ['releases/SemantierAgent.msix', 'releases/SemantierAgent.msix.sha256', POINTER]
        if [row['key'] for row in saved['files']] != expected:
            raise ValueError('Recovery target mismatch')
        for row in saved['files']:
            source = _safe(recovery, row['saved'])
            if sha(source) != row['sha256']:
                raise ValueError('Recovery SHA256 mismatch')
        old = json.loads((recovery/saved['files'][2]['saved']).read_bytes())
        head = old.get('nativeHead') or old.get('head')
        body = _safe(root, head['manifestKey']).read_bytes()
        if hashlib.sha256(body).hexdigest() != head['sha256']:
            raise ValueError('Previous immutable manifest changed')
        row = json.loads(body)['packages'][0]
        if sha(_safe(root, row['artifact']['key'])) != saved['files'][0]['sha256']:
            raise ValueError('Previous immutable package changed')
        if row.get('native') and sha(_safe(root, row['native']['graph']['key'])) != row['native']['graph']['sha256']:
            raise ValueError('Previous immutable graph changed')
        route = saved.get('route')
        if route:
            live = Path(route['path'])
            if sha(live) not in (route['before'], route['after']):
                raise ValueError('Concurrent Caddy configuration; refusing rollback')
            if sha(recovery/'Caddyfile') != route['before']:
                raise ValueError('Caddy recovery SHA256 mismatch')
        for row in saved['files']:
            target = _safe(root, row['key'])
            source = recovery/row['saved']
            if row['key'].endswith('.msix'):
                temporary = target.with_name('.channel1-restore.tmp')
                if temporary.exists():
                    raise ValueError('Unreconciled rollback')
                os.link(source, temporary); os.replace(temporary, target)
                _sync_directory(target.parent)
            else:
                _atomic(target, source.read_bytes())
        if sha(root/POINTER) != saved['beforePointerSha256']:
            raise ValueError('Rollback readback failed')
        if route:
            _apply_route(live, (recovery/'Caddyfile').read_bytes())
    return {'result': 'POINTER_RESTORED', 'osDowngrade': 'NOT_ATTEMPTED', 'recovery': str(recovery)}


def _apply_route(path, body):
    inspected = json.loads(subprocess.check_output(['docker', 'inspect', 'semantier-public-edge'], timeout=20))[0]
    mounts = [m for m in inspected['Mounts'] if m['Destination'] == '/etc/caddy/Caddyfile']
    if len(mounts) != 1 or mounts[0]['Source'] != str(path) or not inspected['State']['Running']:
        raise ValueError('Wrong live Caddy deployment identity')
    edge = inspected['Id']
    subprocess.run(['docker', 'exec', '-i', edge, 'caddy', 'validate',
                    '--config', '-', '--adapter', 'caddyfile'], input=body, check=True,
                   capture_output=True, timeout=45)
    # Keep the existing bind-mounted inode.
    with path.open('wb') as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    # The maintained production config deliberately disables the admin API.
    # Select its supported apply operation before execution; never enable admin
    # or treat an arbitrary reload failure as permission to restart.
    if re.search(rb'(?m)^\s*admin\s+off\s*$', body):
        command = ['docker', 'restart', edge]
    else:
        command = ['docker', 'exec', edge, 'caddy', 'reload',
                   '--config', '/etc/caddy/Caddyfile', '--adapter', 'caddyfile']
    applied = subprocess.run(command, check=False, capture_output=True, timeout=45)
    if applied.returncode:
        raise ValueError('Caddy apply failed: '+applied.stderr.decode(errors='replace')[-2000:])


def _native_route_body(before, old_id, new_id):
    if not re.fullmatch('[a-f0-9]{32}', old_id) or old_id == new_id:
        raise ValueError('Invalid route baseline')
    text = before.decode()
    for name in ('SemantierAgent.msix', 'build.json', 'native-graph.json'):
        old = '/releases/channel-builds/'+old_id+'/'+name
        new = '/releases/channel-builds/'+new_id+'/'+name
        if old not in text:
            raise ValueError('Missing exact native route seam')
        text = text.replace(old, old+' '+new)
    return text.encode()


def stage_native(root, stage, *, metadata_sha256, package_sha256, protocol,
                 expected_previous_sha256, recovery, caddy_sha256):
    """Expose exact immutable candidate URLs for probing; leave pointer unchanged."""
    root, stage, recovery = Path(root).absolute(), Path(stage).absolute(), Path(recovery).absolute()
    if root == stage or root in stage.parents or stage.is_symlink():
        raise ValueError('Invalid upload boundary')
    manifest, graph = _uploaded_metadata(stage, metadata_sha256, package_sha256, protocol)
    row = manifest['packages'][0]; request = manifest['request']
    with _lock(root):
        pointer = _safe(root, POINTER)
        if sha(pointer) != expected_previous_sha256:
            raise ValueError('Concurrent channel publication')
        previous = protocol.decode_json(pointer.read_bytes())
        record = _record(manifest, previous, protocol)
        _save_recovery(root, recovery, record, package_sha256, protocol)
        # Preserve the exact deployed rollback implementation and old closure.
        shutil.copyfile(Path(__file__), recovery/'channel1.py')
        bodies = {protocol.build_prefix(request['buildId'])+'build.json': protocol.canonical_json(manifest),
                  row['feed']['key']: graph}
        artifact = _safe(root, row['artifact']['key'])
        if artifact.exists() and sha(artifact) != package_sha256:
            raise ValueError('Changed immutable package')
        for key, body in bodies.items():
            target = _safe(root, key)
            if target.exists() and target.read_bytes() != body:
                raise ValueError('Changed immutable metadata')
        live = Path('/opt/semantier-cloud-live-v1/Caddyfile')
        saved = protocol.decode_json((recovery/'recovery.json').read_bytes())
        if saved.get('route'):
            current = sha(live)
            if current not in (saved['route']['before'], saved['route']['after']):
                raise ValueError('Concurrent Caddy configuration')
            if current == saved['route']['before']:
                before = (recovery/'Caddyfile').read_bytes()
                after = _native_route_body(before, previous['nativeHead']['buildId'], request['buildId'])
                if hashlib.sha256(after).hexdigest() != saved['route']['after']:
                    raise ValueError('Recovery route binding changed')
                _apply_route(live, after)
        else:
            if sha(live) != caddy_sha256:
                raise ValueError('Concurrent Caddy configuration')
            before = live.read_bytes()
            after = _native_route_body(before, previous['nativeHead']['buildId'], request['buildId'])
            _atomic(recovery/'Caddyfile', before)
            saved['route'] = {'path': str(live), 'before': caddy_sha256,
                              'after': hashlib.sha256(after).hexdigest()}
            _atomic(recovery/'recovery.json', protocol.canonical_json(saved))
            _apply_route(live, after)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if not artifact.exists():
            os.link(stage/'SemantierAgent.msix', artifact); os.chmod(artifact, 0o644)
        for key, body in bodies.items():
            _atomic(_safe(root, key), body)
        if sha(pointer) != expected_previous_sha256:
            raise ValueError('Pointer changed during staging')
    return {'result': 'IMMUTABLE_STAGED_POINTER_UNCHANGED', 'recovery': str(recovery)}


def publish(root, stage, *, metadata_sha256, package_sha256, protocol,
            expected_previous_sha256=None, recovery=None):
    """Activate an uploaded, validated closure; restore the previous view on error."""
    root, stage = Path(root).absolute(), Path(stage).absolute()
    _safe(root, POINTER)
    if root == stage or root in stage.parents or stage.is_symlink():
        raise ValueError('Upload stage must be outside the public root')
    manifest, feed = _uploaded_metadata(stage, metadata_sha256, package_sha256, protocol)
    request = manifest['request']; row = manifest['packages'][0]
    prefix = protocol.build_prefix(request['buildId'])
    pointer = _safe(root, POINTER)
    with _lock(root), tempfile.TemporaryDirectory(prefix='.channel1-rollback-', dir=root.parent) as directory:
        previous = protocol.decode_json(pointer.read_bytes()) if pointer.exists() else None
        record = _record(manifest, previous, protocol)
        if expected_previous_sha256 is not None and sha(pointer) != expected_previous_sha256:
            # A successful, exact rerun is allowed; a different writer is not.
            if previous != record:
                raise ValueError('Concurrent channel publication')
        if previous:
            old_head = previous.get('nativeHead') or previous.get('head')
            if old_head:
                old_body = _safe(root, old_head['manifestKey']).read_bytes()
                if hashlib.sha256(old_body).hexdigest() != old_head['sha256']:
                    raise ValueError('Existing metadata integrity mismatch')
                old_manifest = protocol.decode_json(old_body)
                old_version = tuple(map(int, old_manifest['request']['windowsVersion'].split('.')))
                new_version = tuple(map(int, request['windowsVersion'].split('.')))
                if new_version < old_version or (new_version == old_version and old_head['sha256'] != record['nativeHead']['sha256']):
                    raise ValueError('Version downgrade or changed same-version release')
        protocol.validate_manifest(manifest, record, BASE, allow_native=True)
        bodies = {prefix+'build.json': protocol.canonical_json(manifest), row['feed']['key']: feed}
        artifact = _safe(root, row['artifact']['key'])
        for key, body in bodies.items():
            p = _safe(root, key)
            if p.exists() and p.read_bytes() != body:
                raise ValueError('Changed immutable metadata')
        if artifact.exists() and sha(artifact) != package_sha256:
            raise ValueError('Changed immutable package')
        obsolete = [] if recovery else _obsolete_metadata(root, prefix, request, protocol)
        if recovery:
            _save_recovery(root, recovery, record, package_sha256, protocol)
        _activate(root, stage, record, row, bodies, obsolete, Path(directory), protocol)
    return {'buildId': request['buildId'], 'version': request['windowsVersion'],
            'channelSha256': hashlib.sha256(protocol.canonical_json(record)).hexdigest(),
            'manifestSha256': record['nativeHead']['sha256'], 'alreadyPublished': record == previous}


def recover_space(root, *, expected_previous_sha256, required_bytes, free_bytes=None,
                  retire_images=None, now=None, protected_stage=None, reference_roots=()):
    """Bounded prepublication recovery under the existing channel lock.

    Unknown metadata, links, live uploads and recovery references are retained.
    Free bytes are measured, never inferred from logical sizes or Docker totals.
    """
    import stat
    import time
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or required_bytes < 0:
        raise ValueError('Invalid capacity root/budget')
    free_bytes = free_bytes or (lambda: shutil.disk_usage(root).free)
    now = time.time() if now is None else now
    report = {'result': 'BLOCKED_INSUFFICIENT_CAPACITY', 'requiredBytes': required_bytes,
              'removed': [], 'retained': [], 'docker': {'result': 'NOT_NEEDED'}}
    with _lock(root):
        pointer = _safe(root, POINTER)
        if sha(pointer) != expected_previous_sha256:
            raise ValueError('Concurrent channel change before capacity recovery')
        before = free_bytes()
        report['freeBeforeBytes'] = before
        def finish():
            after = free_bytes()
            report.update(freeAfterBytes=after, reclaimedBytes=max(0, after-before),
                          deficitBytes=max(0, required_bytes-after))
            if after >= required_bytes:
                report['result'] = 'CAPACITY_READY'
            return report
        if before >= required_bytes:
            report['inventory'] = 'SKIPPED_CAPACITY_SUFFICIENT'
            return finish()
        report['inventory'] = 'BOUNDED_RECOVERY'
        # Any unreadable/oversized protection evidence stops recovery, before deletion.
        protected = set()
        sequences = []
        def references(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in ('key', 'manifestKey') and isinstance(item, str):
                        protected.add(item)
                    references(item)
            elif isinstance(value, list):
                for item in value: references(item)
        record = json.loads(pointer.read_bytes())
        references(record)
        for head in (record.get('head'), record.get('nativeHead')):
            if head:
                manifest = _safe(root, head['manifestKey'])
                if sha(manifest) != head['sha256']:
                    raise ValueError('Active manifest hash mismatch')
                value = json.loads(manifest.read_bytes()); references(value)
                sequences.append(value['request']['sequence'])
        evidence = sorted(root.parent.glob('.channel1-recovery-*/**/*.json'))
        evidence += sorted(root.parent.glob('.msix-rollback-*/**/*.json'))
        if len(evidence) > 4096:
            raise ValueError('Recovery inventory bound exceeded')
        for path in evidence:
            if path.is_symlink() or path.stat().st_size > 4*1024*1024:
                raise ValueError('Ambiguous recovery evidence')
            value = json.loads(path.read_bytes()); references(value)
            if path.name == 'recovery.json':
                for row in value['files']:
                    saved = _safe(path.parent, row['saved'])
                    if sha(saved) != row['sha256']:
                        raise ValueError('Recovery byte binding mismatch')
                    if row['key'] == POINTER:
                        references(json.loads(saved.read_bytes()))
        # Protect both heads AND their artifact closures, including rollback pointers.
        for key in list(protected):
            if key.endswith('/build.json'):
                path = _safe(root, key)
                if path.exists():
                    value = json.loads(path.read_bytes()); references(value)
                    sequences.append(value['request']['sequence'])
        report['protectedReferences'] = sorted(protected)
        manifests = sorted((root/'releases/channel-builds').glob('*/build.json'))
        if not sequences or len(manifests) > 4096:
            raise ValueError('Ambiguous/beyond-bound release inventory')
        # Legacy Setup/appinstaller users are independent public references.
        descriptors = sorted(root.rglob('*.appinstaller'))
        if len(descriptors) > 4096:
            raise ValueError('Descriptor inventory bound exceeded')
        for directory in reference_roots:
            directory = Path(directory)
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError('Public reference owner unavailable')
            paths = sorted(directory.rglob('*'))
            if len(paths) > 8192: raise ValueError('Public reference inventory bound exceeded')
            descriptors += [p for p in paths if p.is_file() and p.suffix in ('.html','.json','.js','.css')]
        descriptor_text = []
        for path in descriptors:
            if path.is_symlink() or path.stat().st_size > 1024*1024:
                raise ValueError('Ambiguous descriptor')
            descriptor_text.append(path.read_text())
        def retain(path, reason):
            report['retained'].append({'path': str(path), 'reason': reason})
        # A closed, old, owner-private upload is the only disposable stage shape.
        stages = sorted(root.parent.glob('.semantier-stage-*'))
        if len(stages) > 256:
            raise ValueError('Upload inventory bound exceeded')
        for stage in stages:
            if free_bytes() >= required_bytes: break
            if protected_stage is not None and stage == Path(protected_stage):
                retain(stage, 'current-upload'); continue
            info = stage.lstat()
            if (not re.fullmatch(r'\.semantier-stage-[0-9a-f]{32}', stage.name)
                    or stage.is_symlink() or not stat.S_ISDIR(info.st_mode)
                    or now-info.st_mtime < 86400 or stat.S_IMODE(info.st_mode) != 0o700
                    or (hasattr(os, 'getuid') and info.st_uid != os.getuid())):
                retain(stage, 'unknown-or-active-stage'); continue
            files = list(stage.iterdir())
            allowed = {'SemantierAgent.msix', 'SemantierAgent.msix.sha256'}
            if not files or not {p.name for p in files}.issubset(allowed):
                retain(stage, 'unknown-stage-contents'); continue
            # /proc is required for live-writer exclusion; non-Linux fixtures retain.
            if not Path('/proc').is_dir():
                retain(stage, 'writer-evidence-unavailable'); continue
            stats = {p: p.lstat() for p in files}
            if any(not stat.S_ISREG(i.st_mode) or i.st_nlink != 1 or now-i.st_mtime < 86400
                   or i.st_uid != info.st_uid for i in stats.values()):
                retain(stage, 'hardlinked-or-active-stage'); continue
            open_paths = set()
            for proc in Path('/proc').glob('[0-9]*'):
                try:
                    if str(stage).encode() in (proc/'cmdline').read_bytes():
                        open_paths.add(stage)
                    for fd in (proc/'fd').iterdir():
                        try: open_paths.add(Path(os.readlink(fd)))
                        except FileNotFoundError: pass
                except FileNotFoundError: pass
                except PermissionError:
                    raise ValueError('Cannot exclude active upload owner')
            if stage in open_paths or any(p in open_paths for p in files):
                retain(stage, 'live-upload'); continue
            if any(p.lstat() != i for p, i in stats.items()):
                raise ValueError('Upload changed during recovery')
            for path in files: path.unlink()
            stage.rmdir()
            report['removed'].append({'path': str(stage), 'kind': 'expired-upload',
                                      'logicalBytes': sum(i.st_size for i in stats.values())})
        for manifest in manifests:
            if free_bytes() >= required_bytes: break
            try:
                value = json.loads(manifest.read_bytes())
                request = value['request']
                if request['sequence'] >= min(sequences):
                    retain(manifest, 'active-rollback-or-newer-release'); continue
                for row in value['packages']:
                    key = row['artifact']['key']; path = _safe(root, key)
                    if not path.exists(): continue
                    if key in protected or any(key.startswith(p.removesuffix('build.json')) for p in protected if p.endswith('/build.json')) or any(key in text for text in descriptor_text):
                        retain(path, 'release-reference'); continue
                    info = path.lstat()
                    if info.st_nlink != 1:
                        retain(path, 'hardlinked'); continue
                    if (not stat.S_ISREG(info.st_mode) or manifest.is_symlink()
                            or not key.endswith('/SemantierAgent.msix')
                            or key != manifest.relative_to(root).as_posix().removesuffix('build.json')+'SemantierAgent.msix'
                            or row['identity'] != 'Semantier.Agent.Acceptance'
                            or row['publisher'] != 'CN=Semantier Technology, O=Semantier Technology'
                            or (hasattr(os, 'getuid') and info.st_uid != os.getuid())
                            or row['artifact']['size'] != info.st_size
                            or sha(path) != row['artifact']['sha256']):
                        retain(path, 'unknown-provenance'); continue
                    with zipfile.ZipFile(path) as archive:
                        identity = ET.fromstring(archive.read('AppxManifest.xml')).find('{*}Identity').attrib
                        if any(identity[k] != row[v] for k,v in [('Name','identity'),('Publisher','publisher'),('Version','version')]):
                            raise ValueError('Package identity mismatch')
                    if path.lstat() != info: raise ValueError('Package changed during recovery')
                    path.unlink()
                    report['removed'].append({'path': str(path), 'kind': 'obsolete-msix', 'logicalBytes': info.st_size})
            except (KeyError, ValueError, zipfile.BadZipFile, ET.ParseError):
                retain(manifest, 'unknown-provenance')
        if free_bytes() < required_bytes and retire_images is not None:
            report['docker'] = retire_images()
        return finish()


def _assert_package_inactive(paths):
    """Require Linux process evidence before unlinking a retired package inode."""
    proc = Path('/proc')
    if not proc.is_dir():
        raise ValueError('Process ownership evidence unavailable')
    identities = {(p.stat().st_dev, p.stat().st_ino) for p in paths}
    owners = {str(p.parent) for p in paths if p.parent.name.startswith(('.semantier-stage-', '.channel1-recovery-', '.msix-rollback-'))}
    for process in proc.glob('[0-9]*'):
        if process.name == str(os.getpid()):
            continue
        try:
            command = (process/'cmdline').read_bytes()
            if any(owner.encode() in command for owner in owners):
                raise ValueError('Live original publication owner')
            for fd in (process/'fd').iterdir():
                try:
                    info = fd.stat()
                    if (info.st_dev, info.st_ino) in identities:
                        raise ValueError('Live package reader/writer')
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
        except PermissionError as error:
            raise ValueError('Cannot exclude live package ownership') from error


def retire_package(root, *, build_id, package_sha256, expected_previous_sha256,
                   apply=False, assert_inactive=None):
    """Explicit retirement of one obsolete signed package and its known aliases.

    Completed rollback controls are retained as historical evidence, with their
    manifest renamed recovery.retired so they cannot masquerade as usable recovery.
    Active pointers, unknown links and live owners block before any mutation.
    """
    from contextlib import nullcontext
    root = Path(root).absolute()
    if not re.fullmatch('[0-9a-f]{32}', build_id) or not re.fullmatch('[0-9a-f]{64}', package_sha256):
        raise ValueError('Explicit immutable package identity required')
    _safe(root.parent, root.name)
    with _lock(root) if apply else nullcontext():
        pointer = _safe(root, POINTER)
        if sha(pointer) != expected_previous_sha256:
            raise ValueError('Concurrent pointer change before retirement')
        record = json.loads(pointer.read_bytes())
        active_keys = set()
        active_sequences = []
        for channel in (root/'releases/channels').glob('*.json'):
            value = json.loads(_safe(root, channel.relative_to(root)).read_bytes())
            for head in (value.get('head'), value.get('nativeHead')):
                if not head:
                    continue
                if head['buildId'] == build_id:
                    raise ValueError('Active release cannot be retired')
                manifest = _safe(root, head['manifestKey'])
                if sha(manifest) != head['sha256']:
                    raise ValueError('Active manifest binding mismatch')
                body = json.loads(manifest.read_bytes())
                active_sequences.append(body['request']['sequence'])
                for row in body['packages']:
                    active_keys.add(row['artifact']['key'])
                    if row['artifact']['sha256'] == package_sha256:
                        raise ValueError('Shared active package cannot be retired')
        manifest = _safe(root, 'releases/channel-builds/'+build_id+'/build.json')
        body = json.loads(manifest.read_bytes())
        if body['request']['buildId'] != build_id or not active_sequences or body['request']['sequence'] >= min(active_sequences):
            raise ValueError('Package is not an obsolete release')
        rows = body['packages']
        key = 'releases/channel-builds/'+build_id+'/SemantierAgent.msix'
        if len(rows) != 1 or rows[0]['artifact']['key'] != key or rows[0]['artifact']['sha256'] != package_sha256 or key in active_keys:
            raise ValueError('Retirement artifact binding mismatch')
        package = _safe(root, key)
        stable = _safe(root, 'releases/SemantierAgent.msix')
        active_sha = sha(stable)
        head = record.get('nativeHead') or record.get('head')
        active = json.loads(_safe(root, head['manifestKey']).read_bytes())['packages']
        if (record.get('state') != 'active' or len(active) != 1
                or active[0]['artifact']['sha256'] != active_sha
                or not _safe(root, active[0]['artifact']['key']).samefile(stable)):
            raise ValueError('Active stable package binding mismatch')
        if not package.exists():
            return {'result': 'ALREADY_RETIRED', 'buildId': build_id, 'sha256': package_sha256, 'reclaimedBytes': 0}
        info = package.stat()
        if package.samefile(stable) or info.st_size != rows[0]['artifact']['size'] or sha(package) != package_sha256:
            raise ValueError('Retired/current package integrity mismatch')
        actor = os.geteuid() if hasattr(os, 'geteuid') else None
        root_owner = root.stat()
        active_owner = stable.stat().st_uid
        if actor is not None:
            own_files = info.st_uid == actor and root_owner.st_uid == actor
            root_maintenance = actor == 0 and info.st_uid == active_owner and not (root_owner.st_mode & 0o022)
            if not (own_files or root_maintenance):
                raise ValueError('Retirement actor does not own package/root')
        with zipfile.ZipFile(package) as archive:
            identity = ET.fromstring(archive.read('AppxManifest.xml')).find('{*}Identity').attrib
            if (identity['Name'] != 'Semantier.Agent.Acceptance'
                    or identity['Publisher'] != 'CN=Semantier Technology, O=Semantier Technology'
                    or identity['Version'] != body['request']['windowsVersion']):
                raise ValueError('Retired package identity mismatch')
        entries = list(root.parent.rglob('*'))
        if len(entries) > 16384:
            raise ValueError('Retirement inventory bound exceeded')
        aliases = []
        for path in entries:
            if path.is_symlink() or not path.is_file():
                continue
            value = path.stat()
            if (value.st_dev, value.st_ino) != (info.st_dev, info.st_ino):
                continue
            relative = path.relative_to(root.parent)
            _safe(root.parent, relative)
            owner = relative.parts[0]
            known = (path == package or path == root/'releases/.channel1-restore.tmp'
                     or re.fullmatch(r'\.channel1-recovery-[0-9a-f]{32}', owner)
                     or re.fullmatch(r'\.semantier-stage-[0-9a-f]{32}', owner)
                     or re.fullmatch(r'\.msix-rollback-\d+-[A-Za-z0-9TZ]+', owner))
            if not known:
                raise ValueError('Unknown package alias owner')
            aliases.append(path)
        if len(aliases) != info.st_nlink:
            raise ValueError('Unknown external hardlink; retirement stopped')
        recoveries = []
        for recovery in root.parent.glob('.channel1-recovery-*/recovery.json'):
            _safe(root.parent, recovery.relative_to(root.parent))
            saved = json.loads(recovery.read_bytes())
            if not any(row['sha256'] == package_sha256 for row in saved['files']):
                continue
            if (saved['root'] != str(root) or saved['afterPointerSha256'] != expected_previous_sha256
                    or saved['afterPackageSha256'] != active_sha):
                raise ValueError('Active or ambiguous recovery transaction')
            for row in saved['files']:
                if sha(_safe(recovery.parent, row['saved'])) != row['sha256']:
                    raise ValueError('Rollback receipt byte binding mismatch')
            retired = recovery.with_name('recovery.retired')
            if retired.exists():
                raise ValueError('Conflicting retired recovery evidence')
            recoveries.append(recovery)
        (assert_inactive or _assert_package_inactive)(aliases)
        report = {'result': 'RETIREMENT_PLAN', 'buildId': build_id, 'sha256': package_sha256,
                  'actor': {'uid': actor, 'directoryUid': root_owner.st_uid, 'packageUid': info.st_uid, 'activePackageUid': active_owner},
                  'version': body['request']['windowsVersion'], 'aliases': [str(p) for p in aliases],
                  'retiredRecoveryManifests': [str(p) for p in recoveries], 'reclaimedBytes': 0}
        if not apply:
            return report
        before = shutil.disk_usage(root).free
        for recovery in recoveries:
            recovery.rename(recovery.with_name('recovery.retired'))
        # Remove the immutable alias last: an interrupted retirement can resume.
        aliases.sort(key=lambda p: p == package)
        for remaining, path in enumerate(aliases):
            current = path.stat()
            if ((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
                    != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                    or current.st_nlink != len(aliases)-remaining):
                raise ValueError('Package links changed during retirement')
            path.unlink()
        if sha(pointer) != expected_previous_sha256 or sha(stable) != active_sha:
            raise ValueError('Active release changed during retirement')
        report.update(result='PACKAGE_RETIRED', reclaimedBytes=max(0, shutil.disk_usage(root).free-before),
                      logicalPackageBytes=info.st_size, activePackageSha256=active_sha)
        return report


def stage_oss(package, metadata, output, signature_package=None):
    """WSL staging mode of the existing publisher; never writes an active pointer.

    Metadata must come from the existing prepare operation. Native verification
    is intentionally read-only; no signing, version allocation or installation.
    """
    from agents.client_operations.oss_release import candidate_from_manifest, stage, Objects, assume_role
    package = Path(package).resolve()
    # Fail before cloud writes if the required scoped identity is unavailable.
    credentials = assume_role('upload')
    m = json.loads(Path(metadata).read_text())
    candidate_from_manifest(m['manifest'])  # Validate ASCII pins before constructing the native command.
    windows_path = subprocess.check_output(['wslpath', '-w', str(Path(signature_package).resolve() if signature_package else package)], text=True).strip()
    escaped = windows_path.replace("'", "''")
    script = ("$ErrorActionPreference='Stop';$p='"+escaped+"';"
        "$s=Get-AuthenticodeSignature -LiteralPath $p;"
        "if((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() -cne '"+m['manifest']['packages'][0]['artifact']['sha256']+"'){exit 3};"
        "if($s.Status -ne 'Valid' -or $s.SignerCertificate.Thumbprint -cne '57BF15739B4AA447842EAF388856A01B4E5029FF'){exit 2};"
        r"& 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe' verify /pa $p | Out-Null;exit $LASTEXITCODE")
    result = subprocess.run(['/mnt/c/Program Files/PowerShell/7/pwsh.exe', '-NoProfile', '-NonInteractive', '-Command', script], capture_output=True, timeout=300, check=False)
    if result.returncode:
        raise ValueError('Native MSIX signature verification failed')
    # Keep the exact metadata snapshot whose SHA was checked by native verification.
    # Reuse prepare, including embedded source/version/build and native digest checks.
    request = m['manifest']['request']; row = m['manifest']['packages'][0]
    with tempfile.TemporaryDirectory(prefix='semantier-oss-prepare-') as directory:
        bound = prepare(package, Path(directory)/'bound.json', expected_sha256=row['artifact']['sha256'],
            source_commit=request['commit'], windows_version=request['windowsVersion'],
            build_id=request['buildId'], native_digest=m['graph']['objects'][0]['nativeDigest'])
    if bound != m:
        raise ValueError('Prepared metadata changed')
    candidate = candidate_from_manifest(m['manifest'])
    return stage(package, candidate, output, Objects(credentials))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    local = commands.add_parser('prepare')
    for name in ('package', 'output', 'expected-sha256', 'source-commit', 'windows-version', 'build-id', 'native-digest'):
        local.add_argument('--'+name, required=True)
    remote = commands.add_parser('publish')
    for name in ('root', 'stage', 'metadata-sha256', 'package-sha256', 'protocol', 'protocol-sha256'):
        remote.add_argument('--'+name, required=True)
    remote.add_argument('--expected-previous-sha256')
    remote.add_argument('--recovery')
    rollback = commands.add_parser('rollback')
    rollback.add_argument('--root', required=True)
    rollback.add_argument('--recovery', required=True)
    staging = commands.add_parser('stage-native')
    for name in ('root', 'stage', 'metadata-sha256', 'package-sha256', 'protocol', 'protocol-sha256',
                 'expected-previous-sha256', 'recovery', 'caddy-sha256'):
        staging.add_argument('--'+name, required=True)
    oss = commands.add_parser('stage-oss', help='Private WSL staging only; no ECS/pointer writes')
    for name in ('package', 'metadata', 'output'):
        oss.add_argument('--'+name, required=True)
    oss.add_argument('--signature-package', help='Original matching signed package on the Windows release host (read-only verification)')
    args = vars(parser.parse_args()); command = args.pop('command')
    if command == 'stage-oss':
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        try:
            result = stage_oss(**args)
        except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError):
            print(json.dumps({'state':'STAGING_FAILED','error':'staging_prerequisite_or_integrity_failed'}))
            raise SystemExit(1) from None
        print(json.dumps(result))
        return
    if command == 'prepare':
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        prepare(**args)
        print(json.dumps({'metadataSha256': sha(args['output'])}))
    elif command == 'rollback':
        print(json.dumps(restore(**args)))
    else:
        path = args.pop('protocol'); digest = args.pop('protocol_sha256')
        if sha(path) != digest: raise ValueError('Protocol source SHA256 mismatch')
        spec = importlib.util.spec_from_file_location('channel1_protocol', path)
        protocol = importlib.util.module_from_spec(spec);sys.modules[spec.name] = protocol;spec.loader.exec_module(protocol)
        if command == 'stage-native':
            print(json.dumps(stage_native(**args, protocol=protocol)))
            return
        result = publish(**args, protocol=protocol)
        stage = Path(args['stage'])
        owned = {'SemantierAgent.msix', 'channel1-metadata.json', 'channel1.py', 'channel1-protocol.py'}
        if {p.name for p in stage.iterdir()} == owned and Path(__file__).resolve() == (stage/'channel1.py').resolve():
            for name in owned: (stage/name).unlink()
            stage.rmdir()
            result['stageCleanup'] = 'removed'
        else:
            result['stageCleanup'] = 'retained-unrecognized-stage'
        print(json.dumps(result))


if __name__ == '__main__':
    main()
