"""Sealed native preparation cache; final composition remains source-commit bound."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import io


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def diagnostic(root: Path, **values) -> None:
    path = root / 'diagnostics/runtime.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'producedAt': datetime.now(timezone.utc).isoformat(), **values}) + '\n')


def runtime_key(source: Path, revision: str, toolchain: str, target: str) -> str:
    from scripts.bundles.payload import INERT_SNAPSHOT_DIRS
    tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', revision], cwd=source)
    selected = []
    for row in tree.split(b'\0'):
        if not row:
            continue
        metadata, name = row.split(b'\t', 1)
        path = name.decode('utf-8')
        if (path.split('/')[0] not in INERT_SNAPSHOT_DIRS
                or path.startswith(('scripts/bundles/', 'scripts/build/'))):
            selected.append([metadata.decode('ascii'), path])
    value = {'schema': 1, 'tree': selected, 'toolchain': toolchain, 'target': target}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def cache_lock(entry: Path):
    from hermes_cli.runtime_state import _lock
    if entry.absolute() != entry.resolve():
        raise ValueError('runtime cache must not be symlinked')
    entry.mkdir(parents=True, exist_ok=True)
    with (entry / '.lock').open('a+b') as stream:
        if not _lock(stream.fileno(), wait=False):
            raise ValueError('runtime cache is active; leave it untouched')
        yield


def publish_cache(entry: Path, key: str, out: Path, prepared: Path) -> None:
    """Called under output/cache locks, immediately after real preparation admission."""
    from hermes_cli.runtime_state import _atomic_bytes
    entry.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    archive = entry / 'runtime.tar'
    temporary = entry / 'runtime.tar.tmp'
    if any(path.is_symlink() for path in (archive, temporary, entry / 'receipt.json')):
        raise ValueError('linked runtime cache artifact')
    data = json.loads(prepared.read_text(encoding='utf-8-sig'))
    if data['out'] != str(out) or data['schema'] != 1:
        raise ValueError('runtime preparation binding changed')
    try:
        with tarfile.open(temporary, 'w', format=tarfile.PAX_FORMAT) as bundle:
            roots = sorted({name.split('/')[0] for name in data['digests']})
            for name in roots:
                bundle.add(out / name, arcname=name)
            metadata = json.dumps({'key': key, 'prepared': data}).encode()
            member = tarfile.TarInfo('.runtime-prepared.json'); member.size = len(metadata)
            bundle.addfile(member, io.BytesIO(metadata))
        digest = file_hash(temporary)
        os.replace(temporary, archive)
        _atomic_bytes(entry / 'receipt.json', json.dumps({
            'schema': 1, 'key': key, 'archiveSha256': digest,
        }, sort_keys=True).encode())
        diagnostic(entry.parents[2], result='published', key=key, elapsedSeconds=round(time.monotonic()-started, 3))
    finally:
        temporary.unlink(missing_ok=True)


def restore_cache(entry: Path, key: str, out: Path, source: Path, revision: str) -> Path | None:
    from hermes_cli.runtime_state import _atomic_bytes
    from scripts.bundles.native_prepared import prepared_path
    started = time.monotonic()
    reason = 'missing-receipt'
    if any(p.is_symlink() for p in (entry, entry / 'receipt.json', entry / 'runtime.tar')):
        raise ValueError('linked runtime cache artifact')
    try:
        if out.exists() and any(out.iterdir()):
            reason = 'output-not-empty'
            return None
        data = json.loads((entry / 'receipt.json').read_text(encoding='utf-8-sig'))
        reason = 'identity-or-archive-changed'
        if data['schema'] != 1 or data['key'] != key:
            return None
        with (entry / 'runtime.tar').open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != data['archiveSha256']:
                return None
            stream.seek(0)
            with tarfile.open(fileobj=stream) as bundle:
                member = bundle.getmember('.runtime-prepared.json')
                if not member.isfile() or member.size > 1024 * 1024:
                    raise ValueError('invalid sealed runtime metadata')
                sealed = json.load(bundle.extractfile(member))
                if sealed['key'] != key:
                    raise ValueError('sealed runtime key changed')
                prepared = sealed['prepared']
                previous = prepared['out']
                if prepared['schema'] != 1 or not isinstance(previous, str):
                    raise ValueError('invalid sealed preparation')
                def relocate(value):
                    if isinstance(value, dict):
                        return {k: relocate(v) for k, v in value.items()}
                    if isinstance(value, list):
                        return [relocate(v) for v in value]
                    if isinstance(value, str) and (value == previous or value.startswith((previous+'/', previous+'\\'))):
                        return str(out) + value[len(previous):]
                    return value
                prepared = relocate(prepared)
                prepared.update(out=str(out), source=str(source), revision=revision)
                prepared['inputs']['ref'] = revision
                out.mkdir(parents=True, exist_ok=True)
                # The data filter rejects escaped links/paths and devices.
                bundle.extractall(out, members=[m for m in bundle.getmembers() if m.name != member.name], filter='data')
        result = prepared_path(out)
        _atomic_bytes(result, json.dumps(prepared).encode())
        reason = 'verified-archive'
        return result
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return None
    finally:
        diagnostic(entry.parents[2], result='hit' if reason == 'verified-archive' else 'miss',
                   reason=reason, key=key, elapsedSeconds=round(time.monotonic()-started, 3))
