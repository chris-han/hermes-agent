"""One resumable, offline promotion of the former Desktop home into default.

The source remains intact until every staged entry is installed. Conflicting
root entries remain in previous/; no SQLite row or opaque credential is merged.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile

from hermes_cli import profiles
from hermes_cli.backup_sqlite import _safe_copy_db
from utils import atomic_write_bytes

_LEGACY = 'semantier-local'
_SKIP = {'gateway.pid', 'gateway.lock', '.cloud-device.lock'}
_PROTECTED = {'.cloud-device.bin', '.semantier-feedback.bin'}


def _plain_tree(path: Path) -> None:
    if path.is_symlink() or profiles._junction_target(str(path)):
        raise ValueError('Desktop migration cannot relocate linked user data')
    mode = path.stat().st_mode
    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        raise ValueError('Desktop migration cannot relocate a special file')
    if path.is_dir():
        children = list(path.iterdir())
        ignored = profiles._non_exportable_entries(str(path), [child.name for child in children])
        for child in children:
            if child.name not in ignored:
                _plain_tree(child)


def _require_no_profile_writer(home: Path) -> None:
    """An interactive CLI also writes sessions but has no backend PID record."""
    import psutil
    current = psutil.Process()
    own_chain = {current.pid, *(parent.pid for parent in current.parents())}
    username = current.username().casefold()
    profile = _LEGACY if home.name == _LEGACY else 'default'
    for process in psutil.process_iter(['pid', 'name', 'username', 'cmdline']):
        try:
            if process.pid in own_chain:
                continue
            argv = process.info.get('cmdline') or []
            direct = len(argv) > 1 and Path(argv[1]).name in {'cli.py', 'run_agent.py'}
            if not argv or not (direct or profiles._is_hermes_argv(argv)):
                continue
            if process.info.get('username') is None:
                raise RuntimeError('Cannot establish profile writer ownership')
            if process.info['username'].casefold() != username:
                continue
            environment = process.environ()
            bound = next((value for key, value in environment.items() if key.upper() == 'HERMES_HOME'), '')
            selectors = list(profiles._argv_profile_selectors(argv))
            if ((bound and Path(bound).resolve() == home.resolve()) or profile in selectors
                    or (profile == 'default' and not bound and not selectors)):
                raise RuntimeError('Desktop profile is owned by a running writer')
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied as error:
            raise RuntimeError('Cannot establish profile writer ownership') from error


def _require_idle(home: Path) -> None:
    """No active gateway/backend may write either database during promotion."""
    from gateway.status import _is_gateway_runtime_lock_active_strict
    _require_no_profile_writer(home)
    if (_is_gateway_runtime_lock_active_strict(home / 'gateway.lock')
            or profiles._check_gateway_running(home)
            or profiles._profile_bound_backend_pids(_LEGACY if home.name == _LEGACY else 'default', home)):
        raise RuntimeError('Desktop profile migration requires its running gateway to stop')


def _snapshot(source: Path, target: Path) -> None:
    if source.is_dir():
        target.mkdir(mode=0o700)
        children = list(source.iterdir())
        ignored = profiles._non_exportable_entries(str(source), [child.name for child in children])
        for child in children:
            if child.name in ignored or child.name in _SKIP or child.name.endswith(('-wal', '-shm', '-journal')):
                continue
            _snapshot(child, target / child.name)
    elif source.suffix in {'.db', '.sqlite', '.sqlite3'}:
        if not _safe_copy_db(source, target):
            raise RuntimeError('Desktop database snapshot failed; original profiles retained')
    else:
        shutil.copy2(source, target)
        target.chmod(0o600)


def _rebind_protected(stage: Path, source: Path, owner: Path) -> None:
    from hermes_cli.cloud_device_client import _protect, _unprotect
    for name in _PROTECTED:
        path = stage / name
        if not path.exists():
            continue
        if path.stat().st_size > 1024 * 1024 + 65536:
            raise ValueError('Protected profile store exceeds its size limit')
        value = json.loads(_unprotect(path.read_bytes()))
        if not isinstance(value, dict) or value.get('home') != str(source.resolve()):
            raise ValueError('Protected profile binding requires recovery before migration')
        # Server-issued proof, profile_ref, token and account bindings stay intact.
        value['home'] = str(owner.resolve())
        atomic_write_bytes(path, _protect(json.dumps(value).encode()), mode=0o600, fsync_dir=True)


def _prepare(source: Path, owner: Path, work: Path) -> dict:
    _plain_tree(source)
    from hermes_cli.config import require_readable_config_before_write
    if (source / 'config.yaml').exists():
        _plain_tree(source / 'config.yaml')
    require_readable_config_before_write(source / 'config.yaml')
    if any((source / name).exists() for name in ('profiles', '.desktop-default-migration', 'active_profile')):
        raise ValueError('Legacy Desktop home contains a reserved owner namespace')
    stage = Path(tempfile.mkdtemp(prefix='staging-', dir=work))
    try:
        children = list(source.iterdir())
        ignored = profiles._non_exportable_entries(str(source), [child.name for child in children])
        for child in children:
            if child.name not in ignored and child.name not in _SKIP and not child.name.endswith(('-wal', '-shm', '-journal')):
                _snapshot(child, stage / child.name)
        _rebind_protected(stage, source, owner)
        names = sorted(child.name for child in stage.iterdir())
        # Sidecars from the old root must never be replayed against the incoming DB.
        sidecars = [name + suffix for name in names if (stage / name).is_file()
                    and Path(name).suffix in {'.db', '.sqlite', '.sqlite3'}
                    for suffix in ('-wal', '-shm', '-journal')]
        for name in [*names, *sidecars]:
            target = owner / name
            if target.exists() or target.is_symlink():
                _plain_tree(target)
        return {'schema': 1, 'status': 'applying', 'stage': stage.name, 'names': names, 'sidecars': sidecars,
                'digests': {name: _entry_digest(stage / name) for name in names}}
    except BaseException:
        shutil.rmtree(stage)
        raise


def _entry_digest(path: Path) -> str:
    if path.is_dir():
        entries = [(child.name, _entry_digest(child)) for child in sorted(path.iterdir())]
        return hashlib.sha256(json.dumps(entries, ensure_ascii=True).encode()).hexdigest()
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _apply_entry(owner: Path, stage: Path, previous: Path, name: str, digest: str) -> None:
    incoming, target, backup = stage / name, owner / name, previous / name
    if target.exists() or target.is_symlink():
        _plain_tree(target)
    if not incoming.exists():
        if not target.exists() or _entry_digest(target) != digest:
            raise RuntimeError('Desktop migration target changed; recover the retained snapshot')
        return
    if _entry_digest(incoming) != digest:
        raise RuntimeError('Desktop migration staged snapshot changed')
    if target.exists():
        if backup.exists():
            raise RuntimeError('Desktop data changed during migration; retained backup is intact')
        target.rename(backup)
    incoming.rename(target)


def migrate_local_to_default(owner: Path) -> bool:
    """Called under the existing profile lifecycle lock, before gateway startup."""
    source = owner / 'profiles' / _LEGACY
    work = owner / '.desktop-default-migration'
    journal = work / 'migration.json'
    if not source.exists() and not journal.exists():
        return False
    if work.is_symlink() or profiles._junction_target(str(work)):
        raise ValueError('Desktop migration journal escapes its owner')
    if journal.exists():
        _plain_tree(journal)
        record = json.loads(journal.read_text(encoding='utf-8-sig'))
        if not isinstance(record, dict):
            raise ValueError('Desktop migration journal requires recovery')
        if record.get('schema') == 1 and record.get('status') == 'complete':
            return False
        if (record.get('schema') != 1 or record.get('status') != 'applying'
                or not isinstance(record.get('stage'), str) or not record['stage'].startswith('staging-')):
            raise ValueError('Desktop migration journal requires recovery')
        if not all(isinstance(record.get(key), list) for key in ('names', 'sidecars')):
            raise ValueError('Desktop migration journal requires recovery')
        for name in [record['stage'], *record['names'], *record['sidecars']]:
            if not isinstance(name, str) or Path(name).name != name or name in {'.', '..', 'profiles', 'active_profile', '.desktop-default-migration'}:
                raise ValueError('Desktop migration journal escapes its owner')
        if (not isinstance(record.get('digests'), dict) or set(record['digests']) != set(record['names'])
                or any(not isinstance(value, str) or len(value) != 64 for value in record['digests'].values())):
            raise ValueError('Desktop migration snapshot manifest requires recovery')
    else:
        if not profiles.named_profile_is_live(source):
            return False
        record = None
    _require_idle(owner)
    from gateway.runtime_ownership import exclusive_maintenance
    with exclusive_maintenance([owner]):
        return _promote_pending(owner, source, work, journal, record)


def _promote_pending(owner: Path, source: Path, work: Path, journal: Path, record: dict | None) -> bool:
    if source.exists() or source.is_symlink():
        _plain_tree(source)
    _require_idle(source)
    if record is None:
        work.mkdir(mode=0o700, exist_ok=True)
        record = _prepare(source, owner, work)
        atomic_write_bytes(journal, json.dumps(record).encode(), mode=0o600, fsync_dir=True)
    _plain_tree(work)
    _require_idle(source)
    stage = work / record['stage']
    previous = work / 'previous'
    previous.mkdir(mode=0o700, exist_ok=True)
    for name in record['sidecars']:
        path = owner / name
        if path.exists():
            if (previous / name).exists():
                raise RuntimeError('Desktop database sidecar changed during migration')
            path.rename(previous / name)
    for name in record['names']:
        _apply_entry(owner, stage, previous, name, record['digests'][name])
    retained = work / 'legacy-home'
    if source.exists():
        if retained.exists():
            raise RuntimeError('Legacy Desktop backup already exists; original data retained')
        source.rename(retained)
    elif not retained.is_dir():
        raise RuntimeError('Legacy Desktop backup is missing')
    record['status'] = 'complete'
    atomic_write_bytes(journal, json.dumps(record).encode(), mode=0o600, fsync_dir=True)
    return True
