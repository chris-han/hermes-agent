"""Exact pending custody, independent of admission authority owned by Semantier."""
from __future__ import annotations
from contextvars import ContextVar
from contextlib import contextmanager
import base64
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from hermes_constants import get_hermes_home


managed_workspace = ContextVar('managed_review_workspace', default=False)
stage_actor = ContextVar('managed_skill_stage_actor', default=None)

@contextmanager
def bind_stage_actor(user_id, workspace_id):
    token = stage_actor.set({'user_id':user_id,'workspace_id':workspace_id})
    try:
        yield
    finally:
        stage_actor.reset(token)

def managed_home():
    # These are denial signals only. They never supply identity or authority.
    home = get_hermes_home()
    if managed_workspace.get() or os.environ.get('SEMANTIER_WORKSPACE_ID') or (home/'.learned-procedures').exists() or (home/'.semantier-managed-skills.json').exists():
        return True
    pending = home/'pending'/'skills'
    if pending.is_symlink():
        return True
    for path in pending.glob('*.json'):
        if path.is_symlink():
            continue
        try:
            record = json.loads(path.read_text(encoding='utf-8-sig'))
            if isinstance(record, dict) and record.get('custody') and record.get('exact_capture'):
                return True
        except (OSError,ValueError):
            continue
    return False


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def name_check(name):
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):
        raise ValueError('INVALID_MANAGED_SKILL_ID')
    return name


def path_check(path):
    if not isinstance(path,str) or '\\' in path or '\x00' in path or str(PurePosixPath(path)) != path or PurePosixPath(path).is_absolute() or any(x in ('','.','..') for x in path.split('/')):
        raise ValueError('INVALID_MANAGED_SKILL_FILE')
    if path == '.semantier-publication.json':
        raise ValueError('RESERVED_MANAGED_SKILL_FILE')
    return path


def read_package(directory):
    directory = Path(directory)
    if not directory.exists():
        return {}
    if directory.is_symlink():
        # A core-published package points only into its retained generation root.
        root = get_hermes_home()/'.learned-procedures'
        if root.resolve() not in directory.resolve().parents:
            raise ValueError('UNMANAGED_SKILL_SYMLINK')
    result = {}
    for file in sorted(directory.rglob('*')):
        if file.is_symlink():
            raise ValueError('UNMANAGED_SKILL_SYMLINK')
        if file.is_file():
            relative = file.relative_to(directory).as_posix()
            if relative != '.semantier-publication.json':
                result[path_check(relative)] = base64.b64encode(file.read_bytes()).decode('ascii')
    return result


def capture(payload):
    from tools.skill_manager_tool import _skill_mutation_locks, _find_skill, _resolve_skill_dir
    from tools.write_approval import _fold_patch
    operations = payload.get('operations') if payload.get('action') == 'batch' else [payload]
    if not isinstance(operations,list) or not operations:
        raise ValueError('INVALID_SKILL_OPERATIONS')
    names = [name_check(op.get('name',payload.get('name'))) for op in operations]
    home = get_hermes_home()
    root = home/'skills'
    if root.is_symlink():
        raise ValueError('UNMANAGED_SKILL_ROOT')
    with _skill_mutation_locks(names):
        packages = {}
        for name in sorted(set(names)):
            target = root/name
            found = _find_skill(name)
            if found and Path(found['path']).resolve() != target.resolve():
                raise ValueError('UNMANAGED_SKILL_TARGET')
            known_entry = generation().get(name)
            managed = None if known_entry and known_entry.get('state') == 'absent' else resolve_managed(name)
            base = {} if known_entry and known_entry.get('state') == 'absent' else read_package(managed or target)
            packages[name] = {'skill_id':name,'base':base,'candidate':dict(base),'base_version':{'artifact_ref':known_entry['version_ref'],'artifact_hash':known_entry['version_hash']} if known_entry else None}
        for op,name in zip(operations,names):
            files = packages[name]['candidate']
            action = op.get('action')
            if action == 'create':
                if Path(_resolve_skill_dir(name,op.get('category'))).resolve() != (root/name).resolve():
                    raise ValueError('UNMANAGED_SKILL_CREATE_TARGET')
                if files or op.get('category'):
                    raise ValueError('MANAGED_SKILL_CREATE_CONFLICT')
                files['SKILL.md'] = base64.b64encode((op.get('content') or '').encode()).decode()
            elif action == 'delete':
                if not files: raise ValueError('SKILL_NOT_FOUND')
                files.clear()
            elif action in ('edit','patch','write_file','remove_file'):
                if not files: raise ValueError('SKILL_NOT_FOUND')
                relative = path_check(op.get('file_path') or 'SKILL.md') if action != 'edit' else 'SKILL.md'
                if action == 'remove_file':
                    if relative == 'SKILL.md' or relative not in files: raise ValueError('INVALID_SKILL_REMOVE')
                    del files[relative]
                elif action == 'patch' and not op.get('content'):
                    text = base64.b64decode(files.get(relative,'')).decode('utf-8-sig')
                    text,error = _fold_patch(text,op.get('old_string') or '',op.get('new_string') or '',op.get('replace_all',False))
                    if error: raise ValueError(error)
                    files[relative] = base64.b64encode(text.encode()).decode()
                else:
                    text = op.get('file_content') if action == 'write_file' else op.get('content')
                    if not isinstance(text,str): raise ValueError('INVALID_SKILL_CONTENT')
                    files[relative] = base64.b64encode(text.encode()).decode()
            else:
                raise ValueError('INVALID_SKILL_OPERATION')
        exact = {'schema_version':'hermes.skill_stage.v1','payload_hash':digest(payload),'packages':list(packages.values())}
        return {**exact,'capture_hash':digest(exact)}


_generation_snapshot = ContextVar('managed_skill_generation_snapshot', default=None)
_generation_verifier = ContextVar('managed_skill_generation_verifier', default=None)


@contextmanager
def bind_generation_verifier(callback, scope_key):
    token = _generation_verifier.set((scope_key, callback))
    try:
        yield
    finally:
        _generation_verifier.reset(token)


def _verify_generation(packages):
    verifier = _generation_verifier.get()
    root = get_hermes_home()/'.learned-procedures'
    if root.exists() and not (root/'active-generation.json').is_file():
        raise ValueError('MANAGED_GENERATION_UNAVAILABLE')
    requires_owner = verifier is not None or bool(packages) or (get_hermes_home()/'.learned-procedures').exists() or (get_hermes_home()/'.semantier-managed-skills.json').exists()
    if requires_owner and verifier is None:
        raise ValueError('MANAGED_CANONICAL_VERIFIER_UNAVAILABLE')
    if requires_owner:
        return frozenset(verifier[1](packages))
    return frozenset()



@contextmanager
def pin_generation():
    home = get_hermes_home().resolve()
    existing = _generation_snapshot.get()
    verifier = _generation_verifier.get()
    key = verifier[0] if verifier else None
    if existing is not None and existing[0] == home and existing[2] == key:
        yield
        return
    packages = generation(live=True)
    allowed = _verify_generation(packages)
    token = _generation_snapshot.set((home, packages, key, allowed))
    try:
        yield
    finally:
        _generation_snapshot.reset(token)


def generation(*, live=False):
    snapshot = _generation_snapshot.get()
    verifier = _generation_verifier.get()
    key = verifier[0] if verifier else None
    if not live and snapshot is not None and snapshot[0] == get_hermes_home().resolve() and snapshot[2] == key:
        return dict(snapshot[1])
    root = get_hermes_home()/'.learned-procedures'
    if root.is_symlink(): raise ValueError('MANAGED_GENERATION_UNAVAILABLE')
    path = root/'active-generation.json'
    if not path.exists(): return {}
    if path.is_symlink(): raise ValueError('MANAGED_GENERATION_UNAVAILABLE')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('schema_version') != 'hermes.managed_generation.v1' or data.get('generation_hash') != digest(data.get('packages')):
        raise ValueError('MANAGED_GENERATION_TAMPERED')
    return data['packages']


def resolve_managed(name):
    """None for unmanaged; explicit error for missing/tampered managed bytes."""
    packages = generation()
    snapshot = _generation_snapshot.get()
    verifier = _generation_verifier.get()
    key = verifier[0] if verifier else None
    if snapshot is None or snapshot[0] != get_hermes_home().resolve() or snapshot[2] != key:
        allowed = _verify_generation(packages)
    else:
        allowed = snapshot[3]
    bare = Path(name).name
    entry = packages.get(name) or packages.get(bare)
    if entry is None: return None
    if bare not in allowed: raise ValueError('FORBIDDEN_MANAGED_SKILL')
    if entry.get('state') == 'absent': raise ValueError('MANAGED_SKILL_REVOKED')
    root = get_hermes_home()/'.learned-procedures'
    if root.is_symlink(): raise ValueError('MANAGED_GENERATION_UNAVAILABLE')
    expected = name_check(bare)+'/'+entry['version_hash'].removeprefix('sha256:')
    if path_check(entry['relative_path']) != expected or entry['version_ref'] != 'skill_artifact_version:'+entry['version_hash'].removeprefix('sha256:'):
        raise ValueError('MANAGED_GENERATION_IDENTITY_MISMATCH')
    directory = root/entry['relative_path']
    if directory.is_symlink() or root.resolve() not in directory.resolve().parents:
        raise ValueError('MANAGED_GENERATION_UNAVAILABLE')
    actual = read_package(directory)
    hashes = {p:'sha256:'+hashlib.sha256(base64.b64decode(b)).hexdigest() for p,b in actual.items()}
    if hashes != entry['files']:
        raise ValueError('MANAGED_GENERATION_BYTES_CHANGED')
    return directory


def guard_managed_name(name):
    if Path(name).name in generation():
        raise ValueError('MANAGED_SKILL_NAME_CONFLICT: review changes in Learning')


@contextmanager
def managed_turn():
    """Standalone is unchanged; managed turns pin a projection, never infer an actor."""
    if not managed_home():
        yield
        return
    from gateway.execution_boundary import current_execution_boundary
    boundary = current_execution_boundary()
    actor = stage_actor.get()
    if actor is None and boundary is not None and boundary.user_id and boundary.workspace_id:
        actor = {'user_id':boundary.user_id,'workspace_id':boundary.workspace_id}
    token = stage_actor.set(actor)
    try:
        with pin_generation():
            yield
    finally:
        stage_actor.reset(token)


def managed_skill_index():
    packages = generation()
    snapshot = _generation_snapshot.get()
    verifier = _generation_verifier.get()
    key = verifier[0] if verifier else None
    if snapshot is None or snapshot[0] != get_hermes_home().resolve() or snapshot[2] != key:
        allowed = _verify_generation(packages)
    else:
        allowed = snapshot[3]
    return {name: None if name not in allowed or entry.get('state') == 'absent' else resolve_managed(name) for name, entry in packages.items()}
