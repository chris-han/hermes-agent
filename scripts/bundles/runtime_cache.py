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


@contextmanager
def measure(root: Path | None, stage: str):
    started = time.monotonic()
    result = "PASS"
    try:
        yield
    except BaseException:
        result = "FAIL"
        raise
    finally:
        if root is not None:
            diagnostic(root, stage=stage, result=result, elapsedSeconds=round(time.monotonic()-started, 3))


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
    value = {'schema': 1, 'tree': selected, 'toolchain': toolchain, 'target': target, 'sourceLayout': str(source.resolve())}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# These components stay at the admitted absolute layout across source-only builds.
DEPENDENCY_COMPONENTS = frozenset({'tools', 'venv', 'pm-runtime', 'uv-cache', 'enabled-features.json'})


def dependency_key(source: Path, revision: str, toolchain: str, target: str) -> str:
    """Install inputs, independently of the application's Python source commit."""
    tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', revision], cwd=source)
    recipes = {'pyproject.toml', 'uv.lock', 'setup.py', 'setup.cfg', 'MANIFEST.in', 'scripts/bundles/native.py',
               'scripts/bundles/native_prepared.py', 'scripts/bundles/runtime_cache.py', 'scripts/bundles/bytecode.py',
               'scripts/build/inputs.py', 'scripts/bundles/payload.py'}
    selected = []
    for row in tree.split(b'\0'):
        if row:
            metadata, name = row.split(b'\t', 1)
            path = name.decode('utf-8')
            if path in recipes or path.startswith('pm/'):
                selected.append([metadata.decode('ascii'), path])
    value = {'schema': 1, 'tree': selected, 'toolchain': toolchain, 'target': target,
             'sourceLayout': str(source.resolve())}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def dependencies_path(out: Path) -> Path:
    return out.with_name(out.name + '.dependencies.json')


def publish_dependencies(prepared: Path, key: str) -> None:
    """Reuse the fresh preparation digests; no second copy or environment relocation."""
    from hermes_cli.runtime_state import _atomic_bytes
    data = json.loads(prepared.read_text(encoding='utf-8-sig'))
    components = {name: data['digests'][name] for name in sorted(DEPENDENCY_COMPONENTS)}
    record = {'schema': 1, 'key': key, 'out': data['out'], 'source': data['source'],
              'producedAt': datetime.now(timezone.utc).isoformat(),
              'preparedRevision': data['revision'], 'components': components}
    _atomic_bytes(dependencies_path(Path(data['out'])), json.dumps(record).encode())


def admit_dependencies(out: Path, source: Path, key: str) -> bool:
    """Fresh current-byte/link admission, never a timestamp or receipt-only hit."""
    from concurrent.futures import ThreadPoolExecutor
    from scripts.bundles.native_prepared import _owned, _check_links, _digest
    try:
        receipt = dependencies_path(out)
        if receipt.is_symlink() or out.absolute() != out.resolve():
            return False
        data = json.loads(receipt.read_text(encoding='utf-8-sig'))
        if (data['schema'] != 1 or data['key'] != key or data['out'] != str(out)
                or data['source'] != str(source) or set(data['components']) != DEPENDENCY_COMPONENTS):
            return False
        def matches(item):
            name, expected = item
            path = out / name
            _owned(path, out)
            _check_links(path, out)
            return _digest(path) == expected
        with ThreadPoolExecutor(max_workers=4) as readers:
            return all(readers.map(matches, data['components'].items()))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False


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


def retention_path(out: Path) -> Path:
    # Build ownership is not a shipped runtime resource.
    return out.with_name(out.name + '.retention.json')


def retained_runtime(out: Path, source: Path | None = None, key: str | None = None) -> bool:
    """Ownership only: never a substitute for sealed metadata/byte admission."""
    try:
        marker = retention_path(out)
        if marker.is_symlink() or out.absolute() != out.resolve():
            return False
        data = json.loads(marker.read_text(encoding='utf-8'))
        return (data['schema'] == 1 and data.get('kind') == 'hermes-native-runtime-retention-v1'
                and data['out'] == str(out)
                and isinstance(data['source'], str) and Path(data['source']).is_absolute()
                and isinstance(data['key'], str) and bool(data['key'])
                and (source is None or data['source'] == str(source))
                and (key is None or data['key'] == key))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False


def retain_runtime(out: Path, source: Path, key: str) -> None:
    from hermes_cli.runtime_state import _atomic_bytes
    marker = retention_path(out)
    if (marker.exists() or marker.is_symlink()) and not retained_runtime(out, source):
        raise ValueError('unowned runtime retention marker; preserve and inspect it')
    _atomic_bytes(marker, json.dumps({'schema': 1, 'kind': 'hermes-native-runtime-retention-v1', 'out': str(out),
                                               'source': str(source), 'key': key}).encode())


def revoke_retention(out: Path) -> None:
    if retained_runtime(out):
        retention_path(out).unlink()


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
        with tarfile.open(temporary, 'w', format=tarfile.PAX_FORMAT, copybufsize=1024 * 1024) as bundle:
            # Keep sealed metadata first: retained candidates need not seek
            # through every member after verifying the complete archive hash.
            metadata = json.dumps({'key': key, 'prepared': data}).encode()
            member = tarfile.TarInfo('.runtime-prepared.json'); member.size = len(metadata)
            bundle.addfile(member, io.BytesIO(metadata))
            roots = sorted({name.split('/')[0] for name in data['digests']})
            for name in roots:
                bundle.add(out / name, arcname=name)
        digest = file_hash(temporary)
        os.replace(temporary, archive)
        _atomic_bytes(entry / 'receipt.json', json.dumps({
            'schema': 1, 'key': key, 'archiveSha256': digest,
        }, sort_keys=True).encode())
        retain_runtime(out, Path(data['source']), key)
        diagnostic(entry.parents[2], result='published', key=key, elapsedSeconds=round(time.monotonic()-started, 3))
    finally:
        temporary.unlink(missing_ok=True)


def restore_cache(entry: Path, key: str, out: Path, source: Path, revision: str) -> Path | None:
    from hermes_cli.runtime_state import _atomic_bytes
    from scripts.bundles.native_prepared import prepared_path
    started = time.monotonic()
    if out.absolute() != out.resolve():
        raise ValueError('symlinked native output')
    reason = 'missing-receipt'
    extracting = False
    if any(p.is_symlink() for p in (entry, entry / 'receipt.json', entry / 'runtime.tar')):
        raise ValueError('linked runtime cache artifact')
    try:
        existing = out.exists() and any(out.iterdir())
        if existing and not retained_runtime(out, source, key):
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
            with tarfile.open(fileobj=stream, copybufsize=1024 * 1024) as bundle:
                member = bundle.next()
                if member is None or member.name != '.runtime-prepared.json':
                    member = bundle.getmember('.runtime-prepared.json')
                if not member.isfile() or member.size > 1024 * 1024:
                    raise ValueError('invalid sealed runtime metadata')
                sealed = json.load(bundle.extractfile(member))
                if sealed['key'] != key:
                    raise ValueError('sealed runtime key changed')
                prepared = sealed['prepared']
                if (prepared['schema'] != 1 or Path(prepared['out']) != out
                        or Path(prepared['source']) != source):
                    reason = 'layout-changed'
                    return None
                # Native venvs may carry absolute editable/launcher metadata.
                # Reuse only at the admitted layout; rebind the final commit,
                # never guess how to rewrite arbitrary dependency files.
                prepared['revision'] = revision
                prepared['inputs']['ref'] = revision
                if not existing:
                    revoke_retention(out)
                out.mkdir(parents=True, exist_ok=True)
                # Retained bytes remain candidates until load_prepared checks
                # all current bytes/links before assembly. No deep check is skipped.
                if not existing:
                    extracting = True
                    bundle.extractall(out, members=[m for m in bundle.getmembers() if m.name != member.name], filter='data')
        result = prepared_path(out)
        _atomic_bytes(result, json.dumps(prepared).encode())
        retain_runtime(out, source, key)
        reason = 'retained-candidate' if existing else 'verified-archive'
        return result
    except FileNotFoundError:
        if extracting:
            raise ValueError("runtime extraction failed; partial output is not admitted")
        return None
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        if extracting:
            raise ValueError("runtime extraction failed; partial output is not admitted") from exc
        reason = "invalid-cache-metadata"
        return None
    finally:
        diagnostic(entry.parents[2], result='candidate' if reason == 'retained-candidate' else ('hit' if reason == 'verified-archive' else ('failure' if extracting else 'miss')),
                   reason=reason, key=key, elapsedSeconds=round(time.monotonic()-started, 3))
