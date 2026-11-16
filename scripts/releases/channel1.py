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
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

BASE = 'https://download.semantier.cn'
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


def prepare(package, output, *, expected_sha256, source_commit, windows_version, build_id):
    """Bind metadata to the operator-verified signed package's embedded facts."""
    from hermes_cli import release_channels as protocol
    from scripts.bundles.release_artifacts import write_appinstaller
    package, output = Path(package), Path(output)
    if package.is_symlink() or sha(package) != expected_sha256:
        raise ValueError('Package SHA256 integrity mismatch')
    with zipfile.ZipFile(package) as archive:
        native = ET.fromstring(archive.read('AppxManifest.xml')).find('{*}Identity').attrib
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
    feed_key = prefix+'win32/stable.appinstaller'
    row = {'platform': 'win32', 'arch': 'x64', 'variant': 'bundled', 'version': windows_version,
           'identity': native['Name'], 'publisher': native['Publisher'], 'artifact': artifact,
           'feed': {'key': feed_key, 'channel': 'stable'}}
    manifest = {'schema': 1, 'request': request, 'packages': [row]}
    record = _record(manifest, None, protocol)
    protocol.validate_manifest(manifest, record, BASE)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='channel1-feed-', dir=output.parent) as directory:
        feed = Path(directory)/'stable.appinstaller'
        write_appinstaller(feed, identity=native['Name'], publisher=native['Publisher'], version=windows_version,
                           self_uri=BASE+'/'+feed_key, artifact_uri=BASE+'/'+artifact['key'],
                           update_policy='pinned', package_arch='x64')
        result = {'manifest': manifest, 'feed': base64.b64encode(feed.read_bytes()).decode()}
    output.write_bytes(protocol.canonical_json(result))
    return result


def _record(manifest, previous, protocol):
    request = manifest['request']
    digest = hashlib.sha256(protocol.canonical_json(manifest)).hexdigest()
    same = bool(previous and (previous.get('head') or {}).get('sha256') == digest)
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
              'head': {'buildId': request['buildId'], 'sequence': request['sequence'],
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
            or row['feed'] != {'key': prefix+'win32/stable.appinstaller', 'channel': 'stable'}):
        raise ValueError('Uploaded artifact namespace/binding mismatch')
    feed = base64.b64decode(data['feed'], validate=True)
    xml = ET.fromstring(feed); native = xml.find('{*}MainPackage')
    if (native is None or native.attrib != {'Name': row['identity'], 'Publisher': row['publisher'], 'Version': row['version'],
            'Uri': BASE+'/'+row['artifact']['key'], 'ProcessorArchitecture': 'x64'}
            or xml.attrib != {'Uri': BASE+'/'+row['feed']['key'], 'Version': row['version']}
            or xml.find('{*}UpdateSettings') is not None):
        raise ValueError('App Installer descriptor binding mismatch')
    return manifest, feed


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
        for relative in ('build.json', 'native-graph.json', 'win32/stable.appinstaller'):
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


def publish(root, stage, *, metadata_sha256, package_sha256, protocol):
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
        protocol.validate_manifest(manifest, record, BASE)
        bodies = {prefix+'build.json': protocol.canonical_json(manifest), row['feed']['key']: feed}
        artifact = _safe(root, row['artifact']['key'])
        for key, body in bodies.items():
            p = _safe(root, key)
            if p.exists() and p.read_bytes() != body:
                raise ValueError('Changed immutable metadata')
        if artifact.exists() and sha(artifact) != package_sha256:
            raise ValueError('Changed immutable package')
        obsolete = _obsolete_metadata(root, prefix, request, protocol)
        _activate(root, stage, record, row, bodies, obsolete, Path(directory), protocol)
    return {'buildId': request['buildId'], 'version': request['windowsVersion'],
            'channelSha256': hashlib.sha256(protocol.canonical_json(record)).hexdigest(),
            'manifestSha256': record['head']['sha256'], 'alreadyPublished': record == previous}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    local = commands.add_parser('prepare')
    for name in ('package', 'output', 'expected-sha256', 'source-commit', 'windows-version', 'build-id'):
        local.add_argument('--'+name, required=True)
    remote = commands.add_parser('publish')
    for name in ('root', 'stage', 'metadata-sha256', 'package-sha256', 'protocol', 'protocol-sha256'):
        remote.add_argument('--'+name, required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    if command == 'prepare':
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        prepare(**args)
        print(json.dumps({'metadataSha256': sha(args['output'])}))
    else:
        path = args.pop('protocol'); digest = args.pop('protocol_sha256')
        if sha(path) != digest: raise ValueError('Protocol source SHA256 mismatch')
        spec = importlib.util.spec_from_file_location('channel1_protocol', path)
        protocol = importlib.util.module_from_spec(spec);sys.modules[spec.name] = protocol;spec.loader.exec_module(protocol)
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
