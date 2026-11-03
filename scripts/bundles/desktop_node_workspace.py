"""Stage committed npm workspace inputs without writing the source checkout."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import tarfile


def _git(source: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    if os.name == 'nt':
        if not os.environ.get('SEMANTIER_WSL_GIT_DISTRO'):
            raise ValueError('external Node staging requires the controller WSL Git binding')
        from scripts.bundles.git_command import git_command
        command = git_command(source, *args)
    else:
        command = ['/usr/bin/git', '-C', str(source), *args]
    return subprocess.run(command, cwd=source, capture_output=True, check=check)


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    return bool(name) and not path.is_absolute() and '\\' not in name and all(
        part not in ('', '.', '..') and ':' not in part and not part.endswith((' ', '.'))
        and part.split('.')[0].upper() not in reserved for part in path.parts
    )


def stage_node_workspace(source: Path, commit: str, destination: Path) -> None:
    """Copy tracked manifests and local workspace package inputs at an exact commit.

    The caller owns a fresh job destination and must separately bind the staged
    contents and npm output into its preparation identity before using them.
    """
    from pm.filesystem import is_junction, is_reparse_point

    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('exact source commit required')
    source = Path(source)
    destination = Path(destination)
    if not source.is_absolute() or not destination.is_absolute() or destination.exists() or destination.is_symlink() or is_junction(destination):
        raise ValueError('Node staging destination already exists or is not absolute')
    for location in (source, destination.parent):
        for part in (location, *location.parents):
            if part.is_symlink() or is_junction(part) or is_reparse_point(part):
                raise ValueError('Node staging path has a redirected ancestor')
    source = source.resolve(strict=True)
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Node staging destination must not overlap source')
    def read_json(name: str) -> dict:
        if not _safe_name(name):
            raise ValueError('invalid Node workspace path')
        value = json.loads(_git(source, 'show', f'{commit}:{name}').stdout)
        if not isinstance(value, dict):
            raise ValueError('invalid Node workspace manifest')
        return value

    lock = read_json('package-lock.json')
    entries = lock.get('packages')
    if not isinstance(entries, dict):
        raise ValueError('Node lock has no package inventory')
    workspaces = {name for name in entries if name and 'node_modules' not in name.split('/')}
    if any(not _safe_name(name) for name in workspaces):
        raise ValueError('invalid Node workspace path')
    manifests = {'': read_json('package.json')}
    manifests.update({name: read_json(f'{name}/package.json') for name in sorted(workspaces)})
    local_inputs = set()
    for workspace, manifest in manifests.items():
        for group in ('dependencies', 'devDependencies', 'optionalDependencies'):
            values = manifest.get(group, {})
            if not isinstance(values, dict):
                raise ValueError('invalid Node workspace dependencies')
            for value in values.values():
                if not isinstance(value, str):
                    raise ValueError('invalid Node dependency specification')
                if value.startswith(('link:', 'workspace:')):
                    raise ValueError('unsupported local Node dependency')
                if value.startswith('file:'):
                    target = posixpath.normpath(posixpath.join(workspace, value[5:]))
                    if target not in workspaces:
                        raise ValueError('local Node dependency is not a locked workspace')
                    local_inputs.add(target)
    paths = ['package.json', 'package-lock.json', *(f'{name}/package.json' for name in sorted(workspaces)), *sorted(local_inputs)]
    for workspace in ('', *sorted(workspaces)):
        config = f'{workspace}/tsconfig.json' if workspace else 'tsconfig.json'
        if _git(source, 'cat-file', '-e', f'{commit}:{config}', check=False).returncode == 0:
            paths.append(config)
    if _git(source, 'cat-file', '-e', f'{commit}:.npmrc', check=False).returncode == 0:
        paths.append('.npmrc')
    archive = _git(source, 'archive', '--format=tar', commit, '--', *paths).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as stream:
        members = stream.getmembers()
        seen = set()
        for member in members:
            name = member.name.rstrip('/')
            if not _safe_name(name) or not (member.isfile() or member.isdir()):
                raise ValueError('Node workspace archive contains an unsafe link or path')
            key = name.casefold()
            if key in seen:
                raise ValueError('Node workspace archive contains a duplicate path')
            seen.add(key)
        destination.mkdir()
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                content = stream.extractfile(member)
                if content is None:
                    raise ValueError('Node workspace archive file unavailable')
                with target.open('xb') as output:
                    while chunk := content.read(1024 * 1024):
                        output.write(chunk)
