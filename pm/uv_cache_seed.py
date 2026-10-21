"""Stream a packaged Windows UV seed into the writable cache on demand."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from pm.package import InstallError


def seed_archive(archive: Path, destination: Path) -> None:
    """Validate the inventory first; publish each CRC-checked file atomically.

    No completion marker is owned here. The caller records it only after the
    complete seed succeeds. Retries replace complete files without exposing
    partial files to other profiles sharing the writable machine cache.
    """
    try:
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            seen = set()
            targets = []
            for member in members:
                name = member.filename
                parts = PurePosixPath(name).parts
                mode = stat.S_IFMT(member.external_attr >> 16)
                reserved = {'CON', 'PRN', 'AUX', 'NUL',
                            *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
                if (not parts or name.startswith('/') or any(c in name for c in '<>:"\\|?*')
                        or PurePosixPath(name).as_posix() != name.rstrip('/')
                        or any(p in {'.', '..'} or p.endswith((' ', '.'))
                               or p.split('.')[0].upper() in reserved for p in parts)
                        or parts[0].casefold() == '.seeded' or '\x00' in name
                        or mode not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError(f'unsafe cache archive member: {name!r}')
                key = name.rstrip('/').casefold()
                if key in seen:
                    raise ValueError(f'duplicate cache archive member: {name!r}')
                seen.add(key)
                target = destination.joinpath(*parts)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError(f'cache archive path escapes destination: {name!r}')
                targets.append((member, target))
            for member, target in targets:
                # Recheck immediately before writing: existing cache entries
                # can contain links and must never redirect writes outside it.
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError('cache archive destination changed')
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix='.seed-', dir=target.parent)
                try:
                    with os.fdopen(fd, 'wb') as output, source.open(member) as stream:
                        shutil.copyfileobj(stream, output, length=1024 * 1024)
                    os.replace(temporary, target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
    except (zipfile.BadZipFile, ValueError, NotImplementedError, RuntimeError, EOFError) as exc:
        raise InstallError('uv-cache', f'invalid packaged cache archive: {exc}') from exc
